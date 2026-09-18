//! Pre-allocated buffer pool for quality-metric frame processing.
//!
//! `FramePool` maintains a pool of reusable `Vec<f64>` buffers, eliminating
//! per-frame heap allocations in tight quality-assessment loops.  Buffers are
//! returned to the pool when released and resized (not reallocated) on the
//! next acquisition if the requested size differs.
//!
//! # Example
//!
//! ```rust
//! use oximedia_quality::frame_pool::FramePool;
//!
//! let mut pool = FramePool::new(4);
//!
//! let buf = pool.acquire(1920 * 1080);
//! assert_eq!(buf.len(), 1920 * 1080);
//!
//! // ... use buf for frame processing ...
//!
//! pool.release(buf);
//! assert_eq!(pool.pool_size(), 1);
//! ```

/// A simple object pool for `Vec<f64>` frame buffers.
///
/// Avoids repeated heap allocations when processing many frames in a loop.
/// Internally backed by a `Vec<Vec<f64>>` stack; `acquire` pops from the top
/// and `release` pushes back.
#[derive(Debug, Default)]
pub struct FramePool {
    pool: Vec<Vec<f64>>,
}

impl FramePool {
    /// Creates a new pool pre-populated with `initial_capacity` empty buffers.
    ///
    /// No memory is allocated for the individual buffers at construction time;
    /// each buffer grows lazily on first `acquire`.
    #[must_use]
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            pool: Vec::with_capacity(initial_capacity),
        }
    }

    /// Acquires a buffer of exactly `size` elements.
    ///
    /// If the pool is non-empty the top buffer is popped and resized to `size`
    /// (zeroing any newly added elements).  Otherwise a fresh `vec![0.0; size]`
    /// is allocated.
    pub fn acquire(&mut self, size: usize) -> Vec<f64> {
        if let Some(mut v) = self.pool.pop() {
            v.resize(size, 0.0);
            // Zero out existing elements so callers always get a clean buffer.
            for elem in &mut v {
                *elem = 0.0;
            }
            v
        } else {
            vec![0.0; size]
        }
    }

    /// Returns a buffer to the pool for later reuse.
    ///
    /// The buffer is pushed onto the internal stack; its capacity is
    /// preserved so subsequent acquisitions of the same size are O(1).
    pub fn release(&mut self, v: Vec<f64>) {
        self.pool.push(v);
    }

    /// Returns the number of buffers currently held in the pool.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Clears all pooled buffers, freeing their memory.
    pub fn clear(&mut self) {
        self.pool.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An acquired buffer has the requested length.
    #[test]
    fn test_acquire_length() {
        let mut pool = FramePool::new(2);
        let buf = pool.acquire(1024);
        assert_eq!(buf.len(), 1024);
    }

    /// All elements in an acquired buffer are zeroed.
    #[test]
    fn test_acquire_zeroed() {
        let mut pool = FramePool::new(0);
        let buf = pool.acquire(64);
        assert!(buf.iter().all(|&v| v == 0.0), "buffer should be zeroed");
    }

    /// After release the pool size increases by one.
    #[test]
    fn test_release_increases_pool_size() {
        let mut pool = FramePool::new(0);
        assert_eq!(pool.pool_size(), 0);
        let buf = pool.acquire(256);
        pool.release(buf);
        assert_eq!(pool.pool_size(), 1);
    }

    /// A released buffer is reused on the next acquire; its capacity is
    /// guaranteed to be at least the requested size (no reallocation needed).
    #[test]
    fn test_frame_pool_reuse() {
        let mut pool = FramePool::new(2);

        let buf1 = pool.acquire(512);
        let cap_before = buf1.capacity();
        pool.release(buf1);

        // Acquire the same size — should reuse the same backing allocation
        let buf2 = pool.acquire(512);
        assert_eq!(buf2.len(), 512);
        assert!(
            buf2.capacity() >= cap_before,
            "reused buffer capacity should be >= original"
        );
        pool.release(buf2);
    }

    /// A buffer released with one size and re-acquired with a larger size is
    /// correctly extended.
    #[test]
    fn test_acquire_resize_larger() {
        let mut pool = FramePool::new(0);
        let buf = pool.acquire(16);
        pool.release(buf);
        let buf2 = pool.acquire(64);
        assert_eq!(buf2.len(), 64);
        assert!(buf2.iter().all(|&v| v == 0.0), "resized elements must be 0");
    }

    /// A buffer re-acquired with a smaller size is correctly truncated.
    #[test]
    fn test_acquire_resize_smaller() {
        let mut pool = FramePool::new(0);
        let buf = pool.acquire(64);
        pool.release(buf);
        let buf2 = pool.acquire(16);
        assert_eq!(buf2.len(), 16);
    }

    /// `clear()` empties the pool.
    #[test]
    fn test_clear() {
        let mut pool = FramePool::new(0);
        pool.release(vec![1.0, 2.0, 3.0]);
        pool.release(vec![4.0, 5.0]);
        assert_eq!(pool.pool_size(), 2);
        pool.clear();
        assert_eq!(pool.pool_size(), 0);
    }

    /// Pool starts empty when capacity = 0.
    #[test]
    fn test_empty_pool_fresh_alloc() {
        let mut pool = FramePool::new(0);
        let buf = pool.acquire(32);
        assert_eq!(buf.len(), 32);
        assert_eq!(pool.pool_size(), 0);
    }

    /// Multiple buffers can be managed independently.
    #[test]
    fn test_multiple_buffers() {
        let mut pool = FramePool::new(4);
        let b1 = pool.acquire(100);
        let b2 = pool.acquire(200);
        let b3 = pool.acquire(300);
        pool.release(b1);
        pool.release(b2);
        pool.release(b3);
        assert_eq!(pool.pool_size(), 3);
        let _b4 = pool.acquire(150);
        assert_eq!(pool.pool_size(), 2);
    }
}
