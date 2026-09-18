//! Pool of reusable [`Frame`] allocations to avoid repeated large heap allocations
//! in effect chains.
//!
//! A [`FramePool`] is fixed to a single frame resolution; frames of a different
//! size are not eligible for re-use and are simply dropped on [`release`].
//!
//! [`release`]: FramePool::release

use crate::{Frame, VfxError, VfxResult};

/// Pool of reusable [`Frame`] buffers for a single (width, height) resolution.
///
/// Call [`acquire`](FramePool::acquire) to get a fresh frame (from the pool or
/// newly allocated) and [`release`](FramePool::release) to return it.  The pool
/// will not grow beyond `max_size`; excess frames returned via `release` are
/// simply dropped.
///
/// # Example
/// ```
/// use oximedia_vfx::frame_pool::FramePool;
///
/// let mut pool = FramePool::new(1920, 1080, 4).expect("valid dims");
/// let frame = pool.acquire().expect("allocate");
/// pool.release(frame); // return to pool
/// assert_eq!(pool.len(), 1);
/// ```
#[derive(Debug)]
pub struct FramePool {
    pool: Vec<Frame>,
    width: u32,
    height: u32,
    max_size: usize,
}

impl FramePool {
    /// Create a new pool for frames of `width × height` pixels.
    ///
    /// `max_size` caps how many frames may be stored at once; any frame
    /// returned via [`release`](FramePool::release) when the pool is already
    /// at capacity is simply dropped.
    ///
    /// # Errors
    ///
    /// Returns [`VfxError::InvalidDimensions`] if `width` or `height` is zero.
    pub fn new(width: u32, height: u32, max_size: usize) -> VfxResult<Self> {
        if width == 0 || height == 0 {
            return Err(VfxError::InvalidDimensions { width, height });
        }
        Ok(Self {
            pool: Vec::with_capacity(max_size.min(64)),
            width,
            height,
            max_size,
        })
    }

    /// Acquire a frame from the pool or allocate a fresh one.
    ///
    /// The returned frame is cleared to all-zero (transparent black) before
    /// being handed to the caller.
    ///
    /// # Errors
    ///
    /// Propagates any allocation error from [`Frame::new`].
    pub fn acquire(&mut self) -> VfxResult<Frame> {
        match self.pool.pop() {
            Some(mut frame) => {
                frame.clear([0, 0, 0, 0]);
                Ok(frame)
            }
            None => Frame::new(self.width, self.height),
        }
    }

    /// Return a frame to the pool.
    ///
    /// If the frame dimensions do not match the pool, or if the pool is already
    /// at `max_size`, the frame is dropped silently.
    pub fn release(&mut self, frame: Frame) {
        if frame.width != self.width || frame.height != self.height {
            return; // wrong size — drop
        }
        if self.pool.len() < self.max_size {
            self.pool.push(frame);
        }
        // else: pool full — drop frame
    }

    /// Number of frames currently held in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Returns `true` if no frames are currently in the pool.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Maximum number of frames the pool will retain.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.max_size
    }

    /// Frame width this pool is configured for.
    #[must_use]
    pub fn frame_width(&self) -> u32 {
        self.width
    }

    /// Frame height this pool is configured for.
    #[must_use]
    pub fn frame_height(&self) -> u32 {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_new_valid() {
        let pool = FramePool::new(320, 240, 4).expect("valid");
        assert!(pool.is_empty());
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.frame_width(), 320);
        assert_eq!(pool.frame_height(), 240);
    }

    #[test]
    fn test_pool_new_zero_width_fails() {
        assert!(FramePool::new(0, 240, 4).is_err());
    }

    #[test]
    fn test_pool_new_zero_height_fails() {
        assert!(FramePool::new(320, 0, 4).is_err());
    }

    #[test]
    fn test_pool_acquire_from_empty_allocates() {
        let mut pool = FramePool::new(8, 8, 2).expect("valid");
        let frame = pool.acquire().expect("acquire");
        assert_eq!(frame.width, 8);
        assert_eq!(frame.height, 8);
        assert_eq!(frame.byte_size(), 8 * 8 * 4);
    }

    #[test]
    fn test_pool_acquire_clears_frame() {
        let mut pool = FramePool::new(4, 4, 2).expect("valid");
        // Fill a frame and return it
        let mut frame = Frame::new(4, 4).expect("frame");
        frame.clear([255, 255, 255, 255]);
        pool.release(frame);
        // Re-acquire — should be zeroed
        let frame2 = pool.acquire().expect("acquire");
        let p = frame2.get_pixel(2, 2).expect("pixel");
        assert_eq!(p, [0, 0, 0, 0]);
    }

    #[test]
    fn test_pool_release_and_reacquire() {
        let mut pool = FramePool::new(16, 16, 4).expect("valid");
        let frame = pool.acquire().expect("acquire");
        pool.release(frame);
        assert_eq!(pool.len(), 1);
        let _frame2 = pool.acquire().expect("re-acquire");
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_pool_respects_max_size() {
        let mut pool = FramePool::new(4, 4, 2).expect("valid");
        for _ in 0..5 {
            let f = Frame::new(4, 4).expect("frame");
            pool.release(f);
        }
        assert_eq!(pool.len(), 2, "pool should cap at max_size");
    }

    #[test]
    fn test_pool_ignores_wrong_size() {
        let mut pool = FramePool::new(8, 8, 4).expect("valid");
        let wrong = Frame::new(16, 16).expect("wrong size");
        pool.release(wrong);
        assert_eq!(pool.len(), 0, "wrong-size frame should be dropped");
    }

    #[test]
    fn test_pool_zero_max_size_never_stores() {
        let mut pool = FramePool::new(4, 4, 0).expect("valid");
        let frame = pool.acquire().expect("acquire");
        pool.release(frame);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_pool_multiple_acquire_release_cycles() {
        let mut pool = FramePool::new(32, 32, 3).expect("valid");
        // Fill pool: acquire all three, then release them so they accumulate.
        let f1 = pool.acquire().expect("acquire 1");
        let f2 = pool.acquire().expect("acquire 2");
        let f3 = pool.acquire().expect("acquire 3");
        pool.release(f1);
        pool.release(f2);
        pool.release(f3);
        assert_eq!(pool.len(), 3);
        // Drain
        let _a = pool.acquire().expect("a");
        let _b = pool.acquire().expect("b");
        let _c = pool.acquire().expect("c");
        assert_eq!(pool.len(), 0);
        // One more from fresh alloc
        let _d = pool.acquire().expect("fresh");
    }
}
