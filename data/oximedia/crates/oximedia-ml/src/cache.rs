//! Bounded LRU cache for loaded ONNX models.
//!
//! [`ModelCache`] wraps a capacity-bounded `HashMap` keyed on the
//! canonicalised model path. Each entry is an `Arc<OnnxModel>` so that
//! callers get cheap cloning and the underlying session is shared across
//! concurrent pipelines.
//!
//! The LRU policy is tracked with a simple `Vec<PathBuf>` that records
//! insertion / access order. When capacity is exceeded, the front of the
//! vector (least-recently-used) is evicted.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # fn demo() -> oximedia_ml::MlResult<()> {
//! use oximedia_ml::{DeviceType, ModelCache};
//!
//! let cache = ModelCache::with_capacity(4)?;
//!
//! // First call loads from disk; second call reuses the cached Arc.
//! let a = cache.get_or_load("scene.onnx", DeviceType::auto())?;
//! let b = cache.get_or_load("scene.onnx", DeviceType::auto())?;
//! assert!(std::sync::Arc::ptr_eq(&a, &b));
//! # Ok(())
//! # }
//! ```
//!
//! The cache is thread-safe (internal `Mutex`), so a single cache can be
//! wrapped in `Arc<ModelCache>` and shared across pipelines / async
//! tasks without additional synchronisation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::{canonical_path, OnnxModel};

/// Default cache capacity.
pub const DEFAULT_CAPACITY: usize = 8;

/// LRU cache of loaded [`OnnxModel`] handles.
///
/// See the [module-level docs][self] for an end-to-end example. Keys are
/// canonicalised via [`canonical_path`] so equivalent relative/absolute
/// paths resolve to the same slot.
pub struct ModelCache {
    inner: Mutex<Inner>,
    capacity: usize,
}

struct Inner {
    map: HashMap<PathBuf, Arc<OnnxModel>>,
    order: Vec<PathBuf>,
}

impl ModelCache {
    /// Create a new cache with [`DEFAULT_CAPACITY`] slots.
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: DEFAULT_CAPACITY is a non-zero const, so with_capacity cannot fail.
        match Self::with_capacity(DEFAULT_CAPACITY) {
            Ok(c) => c,
            Err(_) => unreachable!("DEFAULT_CAPACITY is non-zero"),
        }
    }

    /// Create a new cache with the given positive capacity.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::CacheCapacityZero`] if `capacity == 0`.
    pub fn with_capacity(capacity: usize) -> MlResult<Self> {
        if capacity == 0 {
            return Err(MlError::CacheCapacityZero);
        }
        Ok(Self {
            inner: Mutex::new(Inner {
                map: HashMap::with_capacity(capacity),
                order: Vec::with_capacity(capacity),
            }),
            capacity,
        })
    }

    /// Capacity reported by this cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of cached models.
    pub fn len(&self) -> MlResult<usize> {
        let g = self
            .inner
            .lock()
            .map_err(|_| MlError::pipeline("cache", "cache mutex poisoned"))?;
        Ok(g.map.len())
    }

    /// Return whether the cache is empty.
    pub fn is_empty(&self) -> MlResult<bool> {
        self.len().map(|n| n == 0)
    }

    /// Load `path` or return the cached handle.
    ///
    /// Under the hood the entire operation is serialised by the internal
    /// mutex, which avoids the classic double-checked-locking race where
    /// two threads both miss and both load the same model.
    ///
    /// # Errors
    ///
    /// Returns [`MlError::Pipeline`] with stage `"cache"` if the internal
    /// mutex is poisoned, or any error produced by
    /// [`OnnxModel::load`][crate::OnnxModel::load] on a cache miss.
    pub fn get_or_load(
        &self,
        path: impl AsRef<Path>,
        device: DeviceType,
    ) -> MlResult<Arc<OnnxModel>> {
        let key = canonical_path(path.as_ref());
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| MlError::pipeline("cache", "cache mutex poisoned"))?;

        if let Some(existing) = guard.map.get(&key) {
            let arc = Arc::clone(existing);
            Self::touch(&mut guard.order, &key);
            return Ok(arc);
        }

        let model = Arc::new(OnnxModel::load(&key, device)?);
        guard.map.insert(key.clone(), Arc::clone(&model));
        guard.order.push(key);
        Self::evict_if_needed(&mut guard, self.capacity);
        Ok(model)
    }

    /// Remove the cached entry for `path` if present and return it.
    pub fn remove(&self, path: impl AsRef<Path>) -> MlResult<Option<Arc<OnnxModel>>> {
        let key = canonical_path(path.as_ref());
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| MlError::pipeline("cache", "cache mutex poisoned"))?;
        let removed = guard.map.remove(&key);
        guard.order.retain(|p| p != &key);
        Ok(removed)
    }

    /// Drop all cached entries.
    pub fn clear(&self) -> MlResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| MlError::pipeline("cache", "cache mutex poisoned"))?;
        guard.map.clear();
        guard.order.clear();
        Ok(())
    }

    fn touch(order: &mut Vec<PathBuf>, key: &Path) {
        if let Some(pos) = order.iter().position(|p| p == key) {
            let entry = order.remove(pos);
            order.push(entry);
        }
    }

    fn evict_if_needed(inner: &mut Inner, capacity: usize) {
        while inner.map.len() > capacity {
            if inner.order.is_empty() {
                break;
            }
            let oldest = inner.order.remove(0);
            inner.map.remove(&oldest);
        }
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ModelCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.len().unwrap_or(0);
        f.debug_struct("ModelCache")
            .field("capacity", &self.capacity)
            .field("len", &len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_capacity_is_rejected() {
        let err = ModelCache::with_capacity(0).expect_err("expected error");
        assert!(matches!(err, MlError::CacheCapacityZero));
    }

    #[test]
    fn default_capacity_is_non_empty() {
        let cache = ModelCache::new();
        assert_eq!(cache.capacity(), DEFAULT_CAPACITY);
        assert!(cache.is_empty().unwrap_or(false));
    }
}
