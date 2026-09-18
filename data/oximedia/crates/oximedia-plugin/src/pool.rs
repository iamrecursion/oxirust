//! Plugin instance pooling for expensive-to-initialise codecs.
//!
//! Some codec implementations are costly to set up (e.g. hardware encoder
//! session allocation, neural network weight loading).  Creating a fresh
//! instance per request would be prohibitively slow.
//!
//! [`PluginPool`] maintains a bounded set of pre-warmed plugin instances.
//! Callers *borrow* an instance via [`PluginPool::acquire`] and return it via
//! [`PoolGuard::release`] (or automatically on drop if `auto_release` is true).
//!
//! # Design
//!
//! - The pool is backed by a `Vec` protected by a `Mutex`.
//! - Instances are of type `Arc<dyn CodecPlugin>` so they can be shared
//!   between threads during checkout.
//! - A `PoolGuard` implements `Deref<Target = Arc<dyn CodecPlugin>>` so
//!   callers can use the plugin transparently.
//! - When the pool is empty, [`PluginPool::acquire`] returns
//!   [`PluginError::InitFailed`] with a descriptive message; callers can
//!   either spin-wait or propagate the back-pressure signal upstream.

use crate::error::{PluginError, PluginResult};
use crate::traits::CodecPlugin;
use std::sync::{Arc, Mutex};

// ── PoolStats ─────────────────────────────────────────────────────────────────

/// Snapshot statistics about a [`PluginPool`].
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Total capacity (maximum number of instances in the pool).
    pub capacity: usize,
    /// Instances currently available for checkout.
    pub available: usize,
    /// Instances currently checked out.
    pub checked_out: usize,
    /// Total number of successful checkouts since pool creation.
    pub total_checkouts: u64,
}

// ── PluginPool ────────────────────────────────────────────────────────────────

/// A bounded pool of pre-warmed [`CodecPlugin`] instances.
///
/// # Example
///
/// ```rust
/// use oximedia_plugin::pool::PluginPool;
/// use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PLUGIN_API_VERSION};
/// use std::sync::Arc;
///
/// let info = CodecPluginInfo {
///     name: "pool-plugin".to_string(),
///     version: "1.0.0".to_string(),
///     author: "Test".to_string(),
///     description: "Pooled plugin".to_string(),
///     api_version: PLUGIN_API_VERSION,
///     license: "MIT".to_string(),
///     patent_encumbered: false,
/// };
///
/// let mut pool = PluginPool::new(2);
/// pool.add_instance(Arc::new(StaticPlugin::new(info.clone())));
/// pool.add_instance(Arc::new(StaticPlugin::new(info)));
///
/// let guard = pool.acquire().expect("acquire");
/// assert_eq!(guard.info().name, "pool-plugin");
/// // guard auto-releases when dropped
/// ```
pub struct PluginPool {
    /// Maximum number of instances.
    capacity: usize,
    /// Available (idle) instances.
    available: Arc<Mutex<Vec<Arc<dyn CodecPlugin>>>>,
    /// How many instances are currently checked out.
    checked_out: Arc<Mutex<usize>>,
    /// Cumulative checkout counter.
    total_checkouts: Arc<Mutex<u64>>,
}

impl PluginPool {
    /// Create an empty pool with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            available: Arc::new(Mutex::new(Vec::with_capacity(capacity))),
            checked_out: Arc::new(Mutex::new(0)),
            total_checkouts: Arc::new(Mutex::new(0)),
        }
    }

    /// Add a pre-warmed instance to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] if the pool is already at capacity
    /// or if the lock is poisoned.
    pub fn add_instance(&mut self, instance: Arc<dyn CodecPlugin>) -> PluginResult<()> {
        let mut avail = self
            .available
            .lock()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        if avail.len() >= self.capacity {
            return Err(PluginError::InitFailed(format!(
                "Pool is at capacity ({}); cannot add more instances",
                self.capacity
            )));
        }

        avail.push(instance);
        Ok(())
    }

    /// Attempt to acquire a plugin instance from the pool.
    ///
    /// Returns a [`PoolGuard`] that holds the instance.  When the guard is
    /// dropped it automatically returns the instance to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] if the pool is empty (all instances
    /// are checked out) or if a lock is poisoned.
    pub fn acquire(&self) -> PluginResult<PoolGuard> {
        let mut avail = self
            .available
            .lock()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        let instance = avail.pop().ok_or_else(|| {
            PluginError::InitFailed("Plugin pool is exhausted; no instances available".to_string())
        })?;

        // Increment counters.
        drop(avail);
        {
            let mut out = self
                .checked_out
                .lock()
                .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
            *out += 1;
        }
        {
            let mut total = self
                .total_checkouts
                .lock()
                .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;
            *total += 1;
        }

        Ok(PoolGuard {
            instance: Some(instance),
            pool_available: Arc::clone(&self.available),
            pool_checked_out: Arc::clone(&self.checked_out),
        })
    }

    /// Return current pool statistics.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InitFailed`] on lock poisoning.
    pub fn stats(&self) -> PluginResult<PoolStats> {
        let available = self
            .available
            .lock()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?
            .len();

        let checked_out = *self
            .checked_out
            .lock()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        let total_checkouts = *self
            .total_checkouts
            .lock()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        Ok(PoolStats {
            capacity: self.capacity,
            available,
            checked_out,
            total_checkouts,
        })
    }
}

// ── PoolGuard ─────────────────────────────────────────────────────────────────

/// A RAII guard holding a checked-out plugin instance.
///
/// Implements `Deref` so the inner plugin methods can be called directly.
/// On drop the instance is automatically returned to the pool.
pub struct PoolGuard {
    instance: Option<Arc<dyn CodecPlugin>>,
    pool_available: Arc<Mutex<Vec<Arc<dyn CodecPlugin>>>>,
    pool_checked_out: Arc<Mutex<usize>>,
}

impl PoolGuard {
    /// Explicitly return the instance to the pool before the guard is dropped.
    pub fn release(mut self) {
        self.do_release();
    }

    fn do_release(&mut self) {
        if let Some(inst) = self.instance.take() {
            if let Ok(mut avail) = self.pool_available.lock() {
                avail.push(inst);
            }
            if let Ok(mut out) = self.pool_checked_out.lock() {
                *out = out.saturating_sub(1);
            }
        }
    }
}

impl std::ops::Deref for PoolGuard {
    type Target = Arc<dyn CodecPlugin>;

    fn deref(&self) -> &Self::Target {
        // instance is always Some while the guard is alive.
        self.instance
            .as_ref()
            .expect("instance must be present while guard is live")
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        self.do_release();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use crate::traits::{CodecPluginInfo, PLUGIN_API_VERSION};

    fn make_instance(name: &str) -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Pooled test plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        Arc::new(StaticPlugin::new(info))
    }

    // 1. Empty pool returns error.
    #[test]
    fn test_empty_pool_returns_error() {
        let pool = PluginPool::new(2);
        assert!(pool.acquire().is_err());
    }

    // 2. acquire returns the instance.
    #[test]
    fn test_acquire_returns_instance() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("p")).expect("add");
        let guard = pool.acquire().expect("acquire");
        assert_eq!(guard.info().name, "p");
    }

    // 3. Acquired instance is not available until released.
    #[test]
    fn test_acquired_instance_not_reacquirable() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("p")).expect("add");
        let _guard = pool.acquire().expect("first acquire");
        assert!(pool.acquire().is_err()); // pool is empty
    }

    // 4. After guard drop, instance is returned to pool.
    #[test]
    fn test_auto_release_on_drop() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("p")).expect("add");
        {
            let _guard = pool.acquire().expect("acquire");
        } // dropped here
        let guard2 = pool.acquire().expect("re-acquire after drop");
        assert_eq!(guard2.info().name, "p");
    }

    // 5. Explicit release works.
    #[test]
    fn test_explicit_release() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("p")).expect("add");
        let guard = pool.acquire().expect("acquire");
        guard.release();
        // Should be re-acquirable.
        let _guard2 = pool.acquire().expect("re-acquire");
    }

    // 6. Pool at capacity rejects add_instance.
    #[test]
    fn test_add_beyond_capacity_fails() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("a")).expect("first add");
        assert!(pool.add_instance(make_instance("b")).is_err());
    }

    // 7. stats: capacity, available, checked_out.
    #[test]
    fn test_stats() {
        let mut pool = PluginPool::new(3);
        pool.add_instance(make_instance("a")).expect("a");
        pool.add_instance(make_instance("b")).expect("b");

        let s = pool.stats().expect("stats");
        assert_eq!(s.capacity, 3);
        assert_eq!(s.available, 2);
        assert_eq!(s.checked_out, 0);
        assert_eq!(s.total_checkouts, 0);

        let _g = pool.acquire().expect("acquire");
        let s2 = pool.stats().expect("stats2");
        assert_eq!(s2.available, 1);
        assert_eq!(s2.checked_out, 1);
        assert_eq!(s2.total_checkouts, 1);
    }

    // 8. total_checkouts accumulates.
    #[test]
    fn test_total_checkouts_accumulates() {
        let mut pool = PluginPool::new(2);
        pool.add_instance(make_instance("a")).expect("a");
        pool.add_instance(make_instance("b")).expect("b");

        {
            let _g1 = pool.acquire().expect("1");
            let _g2 = pool.acquire().expect("2");
        }
        let s = pool.stats().expect("stats");
        assert_eq!(s.total_checkouts, 2);
    }

    // 9. Multiple concurrent guards.
    #[test]
    fn test_multiple_concurrent_guards() {
        let mut pool = PluginPool::new(3);
        pool.add_instance(make_instance("a")).expect("a");
        pool.add_instance(make_instance("b")).expect("b");
        pool.add_instance(make_instance("c")).expect("c");

        let g1 = pool.acquire().expect("g1");
        let g2 = pool.acquire().expect("g2");
        let g3 = pool.acquire().expect("g3");
        assert!(pool.acquire().is_err()); // pool exhausted

        drop(g1);
        // One slot freed.
        let _g4 = pool.acquire().expect("g4");
        drop(g2);
        drop(g3);
    }

    // 10. PoolGuard deref gives access to plugin methods.
    #[test]
    fn test_guard_deref() {
        let mut pool = PluginPool::new(1);
        pool.add_instance(make_instance("my-plugin")).expect("add");
        let guard = pool.acquire().expect("acquire");
        // Use deref to call info().
        let info = guard.info();
        assert_eq!(info.name, "my-plugin");
    }
}
