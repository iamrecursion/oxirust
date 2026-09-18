//! Pre-allocated frame buffer pool to avoid per-frame heap allocation.
//!
//! At production frame rates (e.g. 60 fps × 4K UYVY422 ≈ 4 GB/s), creating a
//! new heap allocation for every video frame is prohibitively expensive.  This
//! module provides a simple, **single-threaded** frame pool that holds a fixed
//! number of pre-allocated [`NdiVideoFrame`] payloads and hands them out on
//! demand, accepting them back when the caller is done.
//!
//! # Design
//!
//! * The pool owns a `Vec` of pre-allocated frames in the *free list*.
//! * [`FramePool::acquire`] pops one frame off the free list; callers may
//!   overwrite its `data` field in-place rather than allocating fresh storage.
//! * [`FramePool::release`] pushes a frame back onto the free list.
//! * [`FramePool::utilization`] returns a `f32` in `0.0 ..= 1.0` representing
//!   the fraction of frames currently checked out.
//!
//! For multi-threaded use, wrap the pool in a `Mutex`.
//!
//! # Example
//!
//! ```
//! use oximedia_ndi::frame_pool::FramePool;
//!
//! let mut pool = FramePool::new(4, 1920 * 1080 * 2); // 4 UYVY422 1080p frames
//! let frame = pool.acquire().expect("pool is not empty");
//! assert_eq!(pool.utilization(), 0.25); // 1 of 4 checked out
//! pool.release(frame);
//! assert_eq!(pool.utilization(), 0.0);
//! ```

use crate::protocol::{NdiFrameHeader, NdiFrameType, NdiVideoFrame};
use crate::VideoFormat;
use bytes::Bytes;

// ─────────────────────────────────────────────────────────────────────────────
// FramePool
// ─────────────────────────────────────────────────────────────────────────────

/// A pool of pre-allocated [`NdiVideoFrame`] buffers.
///
/// Frames are pre-allocated at construction time with `frame_size` bytes of
/// zeroed payload storage.  [`acquire`](FramePool::acquire) pops a frame from
/// the free list, allowing the caller to fill it without allocating; and
/// [`release`](FramePool::release) returns the frame to the pool.
///
/// The `capacity` field holds the pool's maximum size (never shrinks once
/// set), and `in_use` tracks how many frames are currently checked out.
pub struct FramePool {
    /// Free-list of available frames.
    free: Vec<NdiVideoFrame>,
    /// Number of frames currently checked out.
    in_use: usize,
    /// Maximum number of frames the pool can hold.
    capacity: usize,
}

impl FramePool {
    /// Creates a new frame pool with `capacity` pre-allocated frames.
    ///
    /// Each frame's `data` buffer is pre-allocated to `frame_size` bytes of
    /// zeroed storage, and the [`VideoFormat`] is set to a 1920×1080 30p
    /// placeholder (callers should overwrite it before sending).
    ///
    /// # Arguments
    ///
    /// * `capacity` — total number of frames in the pool.
    /// * `frame_size` — byte size pre-allocated for each frame's pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize, frame_size: usize) -> Self {
        assert!(capacity > 0, "FramePool: capacity must be > 0");

        let placeholder_format = VideoFormat::full_hd_30p();
        let free = (0..capacity)
            .map(|_| {
                let header = NdiFrameHeader::new(NdiFrameType::Video, 0, 0, frame_size as u32);
                NdiVideoFrame {
                    header,
                    format: placeholder_format,
                    data: Bytes::from(vec![0u8; frame_size]),
                    stride: 1920 * 2, // UYVY422 at 1920 px
                }
            })
            .collect();

        Self {
            free,
            in_use: 0,
            capacity,
        }
    }

    /// Acquire a frame from the pool.
    ///
    /// Returns `None` if all frames are currently checked out.  The caller
    /// may mutate the returned frame in-place (e.g. overwrite `data`,
    /// `format`, and `stride`) before sending it.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_ndi::frame_pool::FramePool;
    ///
    /// let mut pool = FramePool::new(2, 1024);
    /// let f1 = pool.acquire().expect("first frame");
    /// let f2 = pool.acquire().expect("second frame");
    /// assert!(pool.acquire().is_none(), "pool exhausted");
    /// pool.release(f1);
    /// assert!(pool.acquire().is_some());
    /// ```
    pub fn acquire(&mut self) -> Option<NdiVideoFrame> {
        let frame = self.free.pop()?;
        self.in_use += 1;
        Some(frame)
    }

    /// Return a frame to the pool.
    ///
    /// The frame's contents are **not** zeroed; callers that need clean
    /// buffers should overwrite the data before the next use.
    ///
    /// # Notes
    ///
    /// Releasing a frame that was never acquired (or releasing more frames
    /// than were acquired) is **not** detected and will grow the free list
    /// beyond capacity.  Callers are expected to use the pool correctly.
    pub fn release(&mut self, frame: NdiVideoFrame) {
        if self.in_use > 0 {
            self.in_use -= 1;
        }
        self.free.push(frame);
    }

    /// Return the pool utilization as a fraction in `0.0 ..= 1.0`.
    ///
    /// `0.0` means no frames are checked out; `1.0` means all frames are in
    /// use.  Returns `0.0` for a zero-capacity pool (which the constructor
    /// rejects, but guards against anyway).
    #[must_use]
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.in_use as f32 / self.capacity as f32
    }

    /// Number of frames currently checked out.
    #[must_use]
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// Number of frames available in the free list.
    #[must_use]
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Maximum pool capacity (set at construction time).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl std::fmt::Debug for FramePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePool")
            .field("capacity", &self.capacity)
            .field("in_use", &self.in_use)
            .field("available", &self.available())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_all_frames_available() {
        let pool = FramePool::new(4, 1024);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.in_use(), 0);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn acquire_all_then_none() {
        let mut pool = FramePool::new(3, 64);
        let f1 = pool.acquire().expect("frame 1");
        let f2 = pool.acquire().expect("frame 2");
        let f3 = pool.acquire().expect("frame 3");

        // Pool is now empty.
        assert!(
            pool.acquire().is_none(),
            "expected None when pool exhausted"
        );
        assert_eq!(pool.in_use(), 3);
        assert_eq!(pool.available(), 0);
        assert!((pool.utilization() - 1.0).abs() < f32::EPSILON);

        pool.release(f1);
        pool.release(f2);
        pool.release(f3);
    }

    #[test]
    fn release_one_allows_acquire() {
        let mut pool = FramePool::new(2, 64);
        let f1 = pool.acquire().expect("frame 1");
        let _f2 = pool.acquire().expect("frame 2");

        // Pool empty.
        assert!(pool.acquire().is_none());

        // Release one.
        pool.release(f1);

        // Now one is available.
        let f3 = pool.acquire().expect("frame 3 after release");
        pool.release(f3);
        pool.release(_f2);
    }

    #[test]
    fn utilization_proportional() {
        let mut pool = FramePool::new(4, 32);
        let f = pool.acquire().expect("frame");
        // 1/4 = 0.25
        assert!((pool.utilization() - 0.25_f32).abs() < 1e-5);
        pool.release(f);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn pre_allocated_frame_data_correct_size() {
        let frame_size = 1920 * 1080 * 2; // UYVY422 1080p
        let mut pool = FramePool::new(2, frame_size);
        let frame = pool.acquire().expect("frame");
        assert_eq!(
            frame.data.len(),
            frame_size,
            "pre-allocated data size mismatch"
        );
        pool.release(frame);
    }

    #[test]
    fn debug_format_does_not_panic() {
        let pool = FramePool::new(2, 128);
        let s = format!("{pool:?}");
        assert!(s.contains("FramePool"));
    }
}
