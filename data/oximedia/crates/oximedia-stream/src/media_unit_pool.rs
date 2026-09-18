//! Bounded recycle pool for [`MediaUnit`] allocations.
//!
//! Live streaming workloads cycle through hundreds of [`MediaUnit`] values per
//! second; each unit owns a `Vec<u8>` payload buffer. Constantly dropping and
//! re-allocating those buffers puts measurable pressure on the global
//! allocator and produces frequent fragmentation in long-lived processes.
//!
//! [`MediaUnitPool`] keeps a fixed-capacity free list of "spent" units whose
//! payload `Vec` has been drained but whose underlying allocation has been
//! preserved (via [`Vec::clear`]).  Callers acquire a [`PooledMediaUnit`]
//! RAII guard; when the guard is dropped the unit is reset and returned to
//! the pool if there is room.  If the pool is already at capacity the unit
//! is allowed to drop normally (the pool never grows past `capacity`).
//!
//! # Thread-safety
//!
//! The pool is internally `Mutex`-guarded so it can be shared across threads
//! via `Arc<MediaUnitPool>`.  Acquire/release are O(1) under the lock.
//!
//! # Example
//!
//! ```
//! use oximedia_stream::media_unit_pool::MediaUnitPool;
//! use oximedia_stream::stream_packager::{MediaUnit, StreamType};
//!
//! let pool = MediaUnitPool::new(16);
//! {
//!     let mut g = pool.acquire();
//!     let u = g.unit_mut();
//!     u.pts_ms = 100;
//!     u.dts_ms = 100;
//!     u.is_keyframe = true;
//!     u.stream_type = StreamType::Video;
//!     u.data.extend_from_slice(&[1, 2, 3]);
//!     // ... pass `g.unit()` to packager / segmenter ...
//! }
//! // `g` dropped here — payload allocation is recycled back into the pool.
//! ```

use std::sync::Mutex;

use crate::stream_packager::{MediaUnit, StreamType};

// ─────────────────────────────────────────────────────────────────────────────
// MediaUnitPool
// ─────────────────────────────────────────────────────────────────────────────

/// A bounded recycle pool for [`MediaUnit`] payload allocations.
#[derive(Debug)]
pub struct MediaUnitPool {
    /// Free list of spent units waiting to be reused.
    free: Mutex<Vec<MediaUnit>>,
    /// Maximum number of units the pool will retain.
    capacity: usize,
}

impl MediaUnitPool {
    /// Create a new pool that will retain at most `capacity` recycled units.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            free: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }

    /// Maximum number of units the pool will retain.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of units currently held in the pool's free list.
    #[must_use]
    pub fn len(&self) -> usize {
        match self.free.lock() {
            Ok(g) => g.len(),
            Err(p) => p.into_inner().len(),
        }
    }

    /// `true` if the pool's free list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Acquire a [`PooledMediaUnit`] from the pool.
    ///
    /// Returns a recycled unit if one is available; otherwise allocates a
    /// fresh empty unit. The returned guard restores the unit to the pool
    /// when dropped (subject to capacity).
    pub fn acquire(&self) -> PooledMediaUnit<'_> {
        let unit = self.take_or_new();
        PooledMediaUnit {
            pool: self,
            unit: Some(unit),
        }
    }

    /// Pop a unit off the free list, or construct a fresh default unit.
    fn take_or_new(&self) -> MediaUnit {
        let popped = match self.free.lock() {
            Ok(mut g) => g.pop(),
            Err(p) => p.into_inner().pop(),
        };
        popped.unwrap_or_else(empty_media_unit)
    }

    /// Return a spent unit to the pool's free list.
    ///
    /// The unit's payload buffer is `clear()`ed (preserving capacity) before
    /// being reinserted. If the pool is already at capacity the unit is
    /// dropped here.
    fn release(&self, mut unit: MediaUnit) {
        // Reset payload but keep the underlying allocation.
        unit.data.clear();
        // Reset scalar fields to defaults so a stale read after `acquire`
        // never leaks state from a prior owner.
        unit.pts_ms = 0;
        unit.dts_ms = 0;
        unit.is_keyframe = false;
        unit.stream_type = StreamType::Video;

        let mut g = match self.free.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if g.len() < self.capacity {
            g.push(unit);
        }
        // else: drop — pool is at capacity.
    }
}

/// Construct an empty `MediaUnit` with default fields.
fn empty_media_unit() -> MediaUnit {
    MediaUnit {
        pts_ms: 0,
        dts_ms: 0,
        data: Vec::new(),
        is_keyframe: false,
        stream_type: StreamType::Video,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PooledMediaUnit
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a [`MediaUnit`] borrowed from a [`MediaUnitPool`].
///
/// Dropping the guard returns the inner unit to the pool's free list.
/// Call [`PooledMediaUnit::take`] to consume the unit (suppressing the
/// recycle behaviour) when ownership must be transferred elsewhere.
#[derive(Debug)]
pub struct PooledMediaUnit<'p> {
    pool: &'p MediaUnitPool,
    unit: Option<MediaUnit>,
}

impl PooledMediaUnit<'_> {
    /// Immutable view of the underlying unit.
    ///
    /// # Panics
    ///
    /// Never — `unit` is always `Some` for the lifetime of the guard until
    /// `take` is called, after which the guard is consumed.
    #[must_use]
    pub fn unit(&self) -> &MediaUnit {
        match self.unit.as_ref() {
            Some(u) => u,
            // SAFETY: `unit` is only `None` after `take` consumed self; the
            // borrow checker prevents calling `unit()` after that.
            None => unreachable!("PooledMediaUnit::unit() called after take"),
        }
    }

    /// Mutable view of the underlying unit.
    pub fn unit_mut(&mut self) -> &mut MediaUnit {
        match self.unit.as_mut() {
            Some(u) => u,
            None => unreachable!("PooledMediaUnit::unit_mut() called after take"),
        }
    }

    /// Consume the guard and transfer ownership of the underlying unit
    /// out of the pool. The unit will not be recycled.
    #[must_use]
    pub fn take(mut self) -> MediaUnit {
        match self.unit.take() {
            Some(u) => u,
            None => unreachable!("PooledMediaUnit::take called twice"),
        }
    }
}

impl Drop for PooledMediaUnit<'_> {
    fn drop(&mut self) {
        if let Some(unit) = self.unit.take() {
            self.pool.release(unit);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_pool_is_empty() {
        let pool = MediaUnitPool::new(8);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.capacity(), 8);
    }

    #[test]
    fn test_acquire_from_empty_pool_allocates() {
        let pool = MediaUnitPool::new(4);
        let g = pool.acquire();
        assert!(g.unit().data.is_empty());
    }

    #[test]
    fn test_drop_returns_unit_to_pool() {
        let pool = MediaUnitPool::new(2);
        {
            let mut g = pool.acquire();
            g.unit_mut().data.extend_from_slice(&[1, 2, 3]);
        }
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_respects_capacity() {
        let pool = MediaUnitPool::new(2);
        for _ in 0..10 {
            let _g = pool.acquire();
        }
        // Each release-on-drop refills the pool, but capacity caps at 2.
        // Sequential drops each push and then pop again, so at any point
        // the pool size is at most `capacity`.
        assert!(pool.len() <= 2);
    }

    #[test]
    fn test_take_does_not_return_to_pool() {
        let pool = MediaUnitPool::new(4);
        {
            let g = pool.acquire();
            let _u = g.take();
            // Guard consumed — no Drop fires.
        }
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_recycled_unit_data_capacity_preserved() {
        let pool = MediaUnitPool::new(2);
        let cap = {
            let mut g = pool.acquire();
            g.unit_mut().data.extend(std::iter::repeat_n(0u8, 1024));
            g.unit().data.capacity()
        };
        // Reacquire — should be the same buffer.
        let g = pool.acquire();
        assert!(g.unit().data.is_empty());
        assert!(g.unit().data.capacity() >= cap);
    }

    #[test]
    fn test_release_resets_scalar_fields() {
        let pool = MediaUnitPool::new(2);
        {
            let mut g = pool.acquire();
            let u = g.unit_mut();
            u.pts_ms = 12345;
            u.dts_ms = 12000;
            u.is_keyframe = true;
            u.stream_type = StreamType::Audio;
        }
        let g = pool.acquire();
        let u = g.unit();
        assert_eq!(u.pts_ms, 0);
        assert_eq!(u.dts_ms, 0);
        assert!(!u.is_keyframe);
        assert_eq!(u.stream_type, StreamType::Video);
    }

    #[test]
    fn test_thread_shared_pool() {
        use std::sync::Arc;
        use std::thread;

        let pool = Arc::new(MediaUnitPool::new(8));
        let mut handles = Vec::new();
        for i in 0..4 {
            let p = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let mut g = p.acquire();
                g.unit_mut().pts_ms = i;
                g.unit_mut().data.extend_from_slice(&[i as u8; 16]);
                // dropped → recycled
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }
        // Up to 4 units should have been pushed back.
        assert!(pool.len() <= 4);
    }
}
