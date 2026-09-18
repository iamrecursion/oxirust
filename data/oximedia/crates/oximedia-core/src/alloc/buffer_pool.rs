//! Buffer pool for zero-copy media pipeline operations.
//!
//! [`BufferPool`] manages a fixed-size set of reusable byte buffers, avoiding
//! repeated heap allocation/deallocation in high-throughput video and audio
//! processing paths where allocation cost would dominate.
//!
//! # Usage patterns
//!
//! ## Pattern 1 — basic acquire / release
//!
//! The simplest pattern: acquire a buffer, process data into it, then return
//! it to the pool.  The pool zeroes the buffer on every release so stale data
//! never leaks between uses.
//!
//! ```
//! use oximedia_core::alloc::BufferPool;
//!
//! let pool = BufferPool::new(4, 1920 * 1080 * 3); // 4 × 1080p RGB buffers
//! let buf = pool.acquire().expect("pool has free buffers");
//! {
//!     let mut guard = buf.write().expect("exclusive access");
//!     guard[0] = 0xFF; // write pixel data
//! }
//! pool.release(buf); // buffer is zeroed and returned; available() increases
//! ```
//!
//! ## Pattern 2 — acquire_or_alloc for unbounded pipelines
//!
//! When bursts can temporarily exhaust the pool, [`BufferPool::acquire_or_alloc`]
//! falls back to a fresh allocation rather than returning `None`.  Released
//! buffers are returned to the pool if there is room; excess buffers are
//! dropped automatically.
//!
//! ```
//! use std::sync::Arc;
//! use oximedia_core::alloc::BufferPool;
//!
//! let pool = Arc::new(BufferPool::new(8, 4096));
//!
//! // Decode loop: never blocks, never returns None
//! let buf = pool.acquire_or_alloc();
//! // ... decode frame data into buf ...
//! pool.release(buf); // return to pool or drop if at capacity
//! ```
//!
//! ## Pattern 3 — memory-pressure management
//!
//! For long-running services, attach a [`PressureConfig`] to automatically
//! shrink the free list when it grows beyond a high-watermark.  An optional
//! callback fires before each shrink — useful for logging or metrics.
//!
//! ```
//! use oximedia_core::alloc::{BufferPool, PressureConfig};
//!
//! let mut pool = BufferPool::new(32, 4096);
//! pool.set_pressure_config(PressureConfig {
//!     high_watermark_free: 16,  // shrink when > 16 free buffers
//!     shrink_to_target: 8,      // retain 8 after shrink
//! });
//! pool.on_pressure(|| {
//!     // e.g. emit a metric or log entry
//!     eprintln!("buffer pool pressure: shrinking free list");
//! });
//! // After every release(), watermark_check() fires automatically.
//! ```
//!
//! ## Pattern 4 — sharing across threads
//!
//! Wrap the pool in `Arc` to share it safely between producer and consumer
//! threads.  Each `Arc<RwLock<Vec<u8>>>` buffer carries its own lock so
//! a decoder and a renderer can hold different buffers simultaneously.
//!
//! ```
//! use std::sync::Arc;
//! use oximedia_core::alloc::BufferPool;
//!
//! let pool = Arc::new(BufferPool::new(8, 1920 * 1080 * 3));
//!
//! // Producer thread
//! let pool_prod = Arc::clone(&pool);
//! let produce = std::thread::spawn(move || {
//!     let buf = pool_prod.acquire_or_alloc();
//!     {
//!         let mut guard = buf.write().expect("write");
//!         guard.fill(0x80); // fill with grey
//!     }
//!     pool_prod.release(buf);
//! });
//! produce.join().expect("producer thread");
//! ```
//!
//! ## Important invariants
//!
//! - **Never reclaim in-use buffers**: [`shrink_to`](BufferPool::shrink_to) and
//!   the pressure auto-shrink only remove buffers from the *free* list.
//!   Buffers that are currently checked out are tracked by `in_use_count()` and
//!   are always safe to write to.
//! - **Zeroed on release**: every buffer is `fill(0)`-ed before it re-enters the
//!   free list, preventing data leakage between processing steps.
//! - **Thread-safe**: all mutations go through `RwLock` / `Mutex`; it is safe
//!   to share one `Arc<BufferPool>` across arbitrarily many threads.

use std::sync::{Arc, Mutex, RwLock};

// ---------------------------------------------------------------------------
// Memory pressure configuration
// ---------------------------------------------------------------------------

/// Configuration for automatic memory-pressure management in a [`BufferPool`].
///
/// When the number of free (idle) buffers in the pool exceeds
/// `high_watermark_free`, the pool will automatically shrink to
/// `shrink_to_target` free buffers by dropping the excess.  In-use buffers
/// are **never** reclaimed.
///
/// # Examples
///
/// ```
/// use oximedia_core::alloc::buffer_pool::PressureConfig;
///
/// let cfg = PressureConfig {
///     high_watermark_free: 8,
///     shrink_to_target: 4,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PressureConfig {
    /// Free-buffer count above which auto-shrink fires.
    pub high_watermark_free: usize,
    /// Number of free buffers to retain after auto-shrink.
    pub shrink_to_target: usize,
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

/// A pool of reusable buffers for zero-copy operations.
///
/// `BufferPool` manages a collection of fixed-size buffers that can be
/// acquired and released. This helps reduce allocation overhead in
/// hot paths like frame decoding.
///
/// ## Memory-pressure management
///
/// Attach a [`PressureConfig`] via [`set_pressure_config`](Self::set_pressure_config)
/// to enable automatic shrinking: when `release()` causes the free count to
/// exceed `high_watermark_free`, the pool drops idle buffers down to
/// `shrink_to_target`.  An optional callback (set via
/// [`on_pressure`](Self::on_pressure)) fires just before each shrink.
///
/// ## Thread Safety
///
/// `BufferPool` is thread-safe and can be shared across threads.
/// Acquired buffers are wrapped in `Arc<RwLock<_>>` for safe concurrent access.
///
/// # Examples
///
/// ```
/// use oximedia_core::alloc::BufferPool;
///
/// // Create a pool with 4 buffers of 1MB each
/// let pool = BufferPool::new(4, 1024 * 1024);
///
/// // Acquire a buffer
/// let buffer = pool.acquire();
/// assert!(buffer.is_some());
///
/// // Write to the buffer
/// {
///     let mut guard = buffer.as_ref().expect("buffer present").write().expect("lock ok");
///     guard[0] = 42;
/// }
///
/// // Release it back to the pool
/// pool.release(buffer.expect("buffer present"));
/// ```
pub struct BufferPool {
    /// Free buffers available for acquisition.
    free_buffers: RwLock<Vec<Arc<RwLock<Vec<u8>>>>>,
    /// Size of each buffer in bytes.
    buffer_size: usize,
    /// Maximum number of free buffers the pool will hold.
    max_buffers: usize,
    /// Count of buffers currently checked out (in use).
    in_use_count: Mutex<usize>,
    /// Optional memory-pressure configuration.
    pressure_config: Mutex<Option<PressureConfig>>,
    /// Optional callback invoked just before a pressure-triggered shrink.
    pressure_callback: Mutex<Option<Box<dyn Fn() + Send + Sync + 'static>>>,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let free = self.free_buffers.read().map(|v| v.len()).unwrap_or(0);
        let in_use = self.in_use_count.lock().map(|g| *g).unwrap_or(0);
        f.debug_struct("BufferPool")
            .field("buffer_size", &self.buffer_size)
            .field("max_buffers", &self.max_buffers)
            .field("free_count", &free)
            .field("in_use_count", &in_use)
            .finish()
    }
}

impl BufferPool {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Creates a new buffer pool.
    ///
    /// # Arguments
    ///
    /// * `count` - Initial number of buffers to allocate
    /// * `buffer_size` - Size of each buffer in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(8, 4096);
    /// ```
    #[must_use]
    pub fn new(count: usize, buffer_size: usize) -> Self {
        let buffers: Vec<_> = (0..count)
            .map(|_| Arc::new(RwLock::new(vec![0u8; buffer_size])))
            .collect();

        Self {
            free_buffers: RwLock::new(buffers),
            buffer_size,
            max_buffers: count,
            in_use_count: Mutex::new(0),
            pressure_config: Mutex::new(None),
            pressure_callback: Mutex::new(None),
        }
    }

    /// Creates a new buffer pool with a specified maximum capacity.
    ///
    /// The pool starts empty and allocates buffers on demand up to `max_buffers`.
    ///
    /// # Arguments
    ///
    /// * `max_buffers` - Maximum number of buffers the pool can hold
    /// * `buffer_size` - Size of each buffer in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::with_capacity(16, 8192);
    /// ```
    #[must_use]
    pub fn with_capacity(max_buffers: usize, buffer_size: usize) -> Self {
        Self {
            free_buffers: RwLock::new(Vec::with_capacity(max_buffers)),
            buffer_size,
            max_buffers,
            in_use_count: Mutex::new(0),
            pressure_config: Mutex::new(None),
            pressure_callback: Mutex::new(None),
        }
    }

    // -----------------------------------------------------------------------
    // Pressure configuration
    // -----------------------------------------------------------------------

    /// Attaches a memory-pressure policy to the pool.
    ///
    /// When `release()` causes the free count to exceed
    /// `config.high_watermark_free`, the pool automatically calls
    /// [`shrink_to`](Self::shrink_to) with `config.shrink_to_target`.
    ///
    /// Returns `&mut Self` for builder-style chaining.
    pub fn set_pressure_config(&mut self, config: PressureConfig) -> &mut Self {
        if let Ok(mut guard) = self.pressure_config.lock() {
            *guard = Some(config);
        }
        self
    }

    /// Registers a callback that fires **before** every pressure-triggered
    /// [`shrink_to`](Self::shrink_to) call.
    ///
    /// Returns `&mut Self` for builder-style chaining.
    pub fn on_pressure<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        if let Ok(mut guard) = self.pressure_callback.lock() {
            *guard = Some(Box::new(callback));
        }
        self
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Acquires a buffer from the pool.
    ///
    /// Returns `None` if no buffers are available. Use
    /// [`acquire_or_alloc`](Self::acquire_or_alloc) if you want to allocate a
    /// new buffer when the pool is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 1024);
    /// let buf1 = pool.acquire();
    /// let buf2 = pool.acquire();
    /// let buf3 = pool.acquire(); // Returns None, pool exhausted
    /// assert!(buf1.is_some());
    /// assert!(buf2.is_some());
    /// assert!(buf3.is_none());
    /// ```
    #[must_use]
    pub fn acquire(&self) -> Option<Arc<RwLock<Vec<u8>>>> {
        let buffer = self.free_buffers.write().ok()?.pop()?;
        // Increment in-use counter
        if let Ok(mut guard) = self.in_use_count.lock() {
            *guard = guard.saturating_add(1);
        }
        Some(buffer)
    }

    /// Acquires a buffer from the pool, allocating a new one if necessary.
    ///
    /// If the pool is empty, allocates a new buffer. This is useful when
    /// you need a buffer regardless of pool state.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(0, 1024); // Empty pool
    /// let buffer = pool.acquire_or_alloc();
    /// assert_eq!(buffer.read().expect("lock ok").len(), 1024);
    /// ```
    #[must_use]
    pub fn acquire_or_alloc(&self) -> Arc<RwLock<Vec<u8>>> {
        self.acquire().unwrap_or_else(|| {
            // Freshly allocated buffer also counts as in-use
            if let Ok(mut guard) = self.in_use_count.lock() {
                *guard = guard.saturating_add(1);
            }
            Arc::new(RwLock::new(vec![0u8; self.buffer_size]))
        })
    }

    /// Releases a buffer back to the pool.
    ///
    /// The buffer should have been previously acquired from this pool.
    /// If the pool is at capacity, the buffer is dropped.
    ///
    /// After the buffer is returned, [`watermark_check`](Self::watermark_check)
    /// fires automatically when a pressure config is active.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to return to the pool
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 1024);
    /// let buffer = pool.acquire().expect("buffer available");
    /// // Use the buffer...
    /// pool.release(buffer);
    /// ```
    pub fn release(&self, buffer: Arc<RwLock<Vec<u8>>>) {
        // Decrement in-use counter
        if let Ok(mut guard) = self.in_use_count.lock() {
            *guard = guard.saturating_sub(1);
        }

        let returned = if let Ok(mut buffers) = self.free_buffers.write() {
            if buffers.len() < self.max_buffers {
                // Clear the buffer for security and consistency
                if let Ok(mut guard) = buffer.write() {
                    guard.fill(0);
                }
                buffers.push(buffer);
                true
            } else {
                // At capacity — buffer is dropped
                false
            }
        } else {
            false
        };

        // Auto watermark check only when a buffer was actually returned
        if returned {
            self.watermark_check();
        }
    }

    // -----------------------------------------------------------------------
    // Pressure management
    // -----------------------------------------------------------------------

    /// Drops free (idle) buffers from the pool until `free_count ≤ target`.
    ///
    /// **In-use buffers are never reclaimed.**  This method operates only on
    /// the free list.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(10, 64);
    /// pool.shrink_to(3);
    /// // The pool now holds at most 3 free buffers
    /// ```
    pub fn shrink_to(&self, target: usize) {
        if let Ok(mut buffers) = self.free_buffers.write() {
            while buffers.len() > target {
                buffers.pop(); // drops the Arc → memory freed
            }
        }
    }

    /// Checks whether the free count exceeds the configured high-watermark,
    /// and if so, invokes the pressure callback (if any) and then
    /// [`shrink_to`](Self::shrink_to) with `shrink_to_target`.
    ///
    /// This is called automatically by [`release`](Self::release) when a
    /// [`PressureConfig`] is active.  It can also be called manually.
    pub fn watermark_check(&self) {
        let cfg = match self.pressure_config.lock().ok().and_then(|g| *g) {
            Some(c) => c,
            None => return,
        };

        let free_count = self.free_buffers.read().map(|v| v.len()).unwrap_or(0);
        if free_count <= cfg.high_watermark_free {
            return;
        }

        // Fire optional callback before shrinking
        if let Ok(guard) = self.pressure_callback.lock() {
            if let Some(cb) = guard.as_ref() {
                cb();
            }
        }

        self.shrink_to(cfg.shrink_to_target);
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Returns the number of buffers currently available (free) in the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(4, 1024);
    /// assert_eq!(pool.available(), 4);
    /// let _buf = pool.acquire();
    /// assert_eq!(pool.available(), 3);
    /// ```
    #[must_use]
    pub fn available(&self) -> usize {
        self.free_buffers.read().map(|b| b.len()).unwrap_or(0)
    }

    /// Returns the number of buffers currently checked out (in use).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(4, 1024);
    /// assert_eq!(pool.in_use_count(), 0);
    /// let _buf = pool.acquire();
    /// assert_eq!(pool.in_use_count(), 1);
    /// ```
    #[must_use]
    pub fn in_use_count(&self) -> usize {
        self.in_use_count.lock().map(|g| *g).unwrap_or(0)
    }

    /// Returns the size of each buffer in the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 4096);
    /// assert_eq!(pool.buffer_size(), 4096);
    /// ```
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Returns the maximum number of free buffers the pool can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(8, 1024);
    /// assert_eq!(pool.max_buffers(), 8);
    /// ```
    #[must_use]
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(4, 4096)
    }
}

// ---------------------------------------------------------------------------
// Tests — original suite
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.max_buffers(), 4);
    }

    #[test]
    fn test_with_capacity() {
        let pool = BufferPool::with_capacity(8, 2048);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.buffer_size(), 2048);
        assert_eq!(pool.max_buffers(), 8);
    }

    #[test]
    fn test_acquire_release() {
        let pool = BufferPool::new(2, 1024);
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available(), 1);

        let buf2 = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available(), 0);

        assert!(pool.acquire().is_none());

        pool.release(buf1);
        assert_eq!(pool.available(), 1);

        pool.release(buf2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_acquire_or_alloc() {
        let pool = BufferPool::new(0, 1024);
        assert_eq!(pool.available(), 0);

        let buffer = pool.acquire_or_alloc();
        assert_eq!(buffer.read().expect("read lock should succeed").len(), 1024);
    }

    #[test]
    fn test_buffer_contents() {
        let pool = BufferPool::new(1, 64);
        let buffer = pool.acquire().expect("acquire should succeed");

        // Write to buffer
        {
            let mut guard = buffer.write().expect("write lock should succeed");
            guard[0] = 42;
            guard[63] = 255;
        }

        // Read from buffer
        {
            let guard = buffer.read().expect("read lock should succeed");
            assert_eq!(guard[0], 42);
            assert_eq!(guard[63], 255);
        }

        // Release and reacquire - buffer should be zeroed
        pool.release(buffer);
        let buffer = pool.acquire().expect("acquire should succeed");
        {
            let guard = buffer.read().expect("read lock should succeed");
            assert_eq!(guard[0], 0);
            assert_eq!(guard[63], 0);
        }
    }

    #[test]
    fn test_default() {
        let pool = BufferPool::default();
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.buffer_size(), 4096);
    }

    #[test]
    fn test_release_at_capacity() {
        let pool = BufferPool::new(2, 1024);
        let extra_buffer = Arc::new(RwLock::new(vec![0u8; 1024]));

        // Pool is full, releasing should not add more buffers
        pool.release(extra_buffer);
        assert_eq!(pool.available(), 2); // Still 2, not 3
    }
}

// ---------------------------------------------------------------------------
// Tests — memory pressure suite
// ---------------------------------------------------------------------------

#[cfg(test)]
mod buffer_pool_pressure_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Helper: create a pool with N pre-allocated free buffers and a pressure
    /// config already set, without needing `&mut self` in the test body.
    fn pool_with_pressure(
        count: usize,
        buf_size: usize,
        watermark: usize,
        target: usize,
    ) -> BufferPool {
        let mut pool = BufferPool::new(count, buf_size);
        pool.set_pressure_config(PressureConfig {
            high_watermark_free: watermark,
            shrink_to_target: target,
        });
        pool
    }

    // -----------------------------------------------------------------------
    // Test 1: pool shrinks to target on manual watermark_check
    // -----------------------------------------------------------------------
    #[test]
    fn test_pool_shrinks_to_target_on_pressure() {
        // 10 free buffers, watermark=5, target=3
        let pool = pool_with_pressure(10, 64, 5, 3);
        assert_eq!(pool.available(), 10);

        // Trigger the watermark check explicitly (simulates what release does)
        pool.watermark_check();

        assert_eq!(
            pool.available(),
            3,
            "pool should shrink to target=3 when free_count=10 > watermark=5"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: in-use buffers are never reclaimed by shrink_to
    // -----------------------------------------------------------------------
    #[test]
    fn test_pool_retains_in_use_buffers() {
        // Create a large-capacity pool so we can add extra free buffers later
        let pool = BufferPool::with_capacity(20, 64);

        // Acquire 5 buffers — they are now in-use
        let handles: Vec<_> = (0..5).map(|_| pool.acquire_or_alloc()).collect();
        assert_eq!(pool.in_use_count(), 5);

        // Manually inject 5 fresh free buffers into the pool
        for _ in 0..5 {
            let buf = Arc::new(RwLock::new(vec![0u8; 64]));
            if let Ok(mut v) = pool.free_buffers.write() {
                v.push(buf);
            }
        }
        assert_eq!(pool.available(), 5);

        // Shrink free list all the way to zero
        pool.shrink_to(0);

        // Free list is empty but in-use count is unchanged
        assert_eq!(pool.available(), 0, "all free buffers should be dropped");
        assert_eq!(
            pool.in_use_count(),
            5,
            "in-use buffers must not be reclaimed by shrink_to"
        );

        // Drop the handles to verify we can still release them afterwards
        for h in handles {
            pool.release(h);
        }
        assert_eq!(pool.in_use_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 3: watermark auto-shrink fires automatically on release
    // -----------------------------------------------------------------------
    #[test]
    fn test_watermark_auto_shrink_fires_above_threshold() {
        // Pool capacity 20, watermark=5, target=3
        let mut pool = BufferPool::with_capacity(20, 64);
        pool.set_pressure_config(PressureConfig {
            high_watermark_free: 5,
            shrink_to_target: 3,
        });

        // Acquire 8 buffers via acquire_or_alloc (they are freshly allocated)
        let handles: Vec<_> = (0..8).map(|_| pool.acquire_or_alloc()).collect();
        assert_eq!(pool.in_use_count(), 8);
        assert_eq!(pool.available(), 0);

        // Release all 8 — each release calls watermark_check internally
        for h in handles {
            pool.release(h);
        }

        // After releasing 8 buffers with watermark=5, target=3 the auto-shrink
        // fires the first time available() would exceed 5 (i.e. on the 6th
        // release) and again on subsequent releases.  The final state depends
        // on the exact interleaving, but the free count MUST be ≤ 3 or equal
        // to 5 (if a single shrink brought it exactly to 3 and then 2 more
        // were added without triggering again).  The invariant we verify is
        // that the free count never grew unbounded past the watermark.
        let final_free = pool.available();
        assert!(
            final_free <= 5,
            "auto-shrink must keep free count ≤ watermark after all releases; got {final_free}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: no shrink below threshold (count stays at 3 with watermark=5)
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_shrink_below_threshold() {
        // Capacity 10, watermark=5, target=3
        let mut pool = BufferPool::with_capacity(10, 64);
        pool.set_pressure_config(PressureConfig {
            high_watermark_free: 5,
            shrink_to_target: 3,
        });

        // Acquire and immediately release only 3 buffers
        let handles: Vec<_> = (0..3).map(|_| pool.acquire_or_alloc()).collect();
        for h in handles {
            pool.release(h);
        }

        // 3 < watermark(5) → no auto-shrink should have fired
        assert_eq!(
            pool.available(),
            3,
            "pool must not shrink when free count is below watermark"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: pressure callback fires before shrink
    // -----------------------------------------------------------------------
    #[test]
    fn test_pressure_callback_fires_before_shrink() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let mut pool = pool_with_pressure(10, 64, 5, 3);
        pool.on_pressure(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Trigger manually (free_count=10 > watermark=5)
        pool.watermark_check();

        assert!(
            counter.load(Ordering::SeqCst) >= 1,
            "pressure callback must fire at least once"
        );
        assert_eq!(pool.available(), 3);
    }
}
