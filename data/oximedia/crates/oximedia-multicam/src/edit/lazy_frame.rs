//! Lazy frame loading for multi-camera timeline editing.
//!
//! `LazyFrameRef` holds the metadata for a frame (angle index, frame number,
//! pixel dimensions) but defers the actual pixel-data decode until the caller
//! provides a resolver function via [`LazyFrameRef::resolve_with`].  The result
//! is cached inside the struct via [`std::cell::OnceCell`], so the resolver is
//! called at most once per instance.

use std::cell::OnceCell;

/// A lazy reference to a single decoded frame.
///
/// The pixel data is not loaded until [`resolve_with`] is called for the first
/// time.  Subsequent calls return the cached bytes without invoking the resolver
/// again.
///
/// [`resolve_with`]: LazyFrameRef::resolve_with
#[derive(Debug)]
pub struct LazyFrameRef {
    /// Index of the camera angle this frame belongs to.
    pub angle_idx: usize,
    /// Frame number within the timeline.
    pub frame_number: u64,
    /// Pixel width of the decoded frame.
    pub width: u32,
    /// Pixel height of the decoded frame.
    pub height: u32,
    /// Lazily populated pixel data (row-major, packed bytes).
    cached: OnceCell<Vec<u8>>,
}

impl LazyFrameRef {
    /// Create a new unloaded frame reference.
    #[must_use]
    pub fn new(angle_idx: usize, frame_number: u64, width: u32, height: u32) -> Self {
        Self {
            angle_idx,
            frame_number,
            width,
            height,
            cached: OnceCell::new(),
        }
    }

    /// Return the decoded pixel bytes, loading them on the first call.
    ///
    /// `resolver` receives `(angle_idx, frame_number)` and must return the raw
    /// pixel bytes for that frame.  It is guaranteed to be called **at most
    /// once** per `LazyFrameRef` instance.
    pub fn resolve_with(&self, resolver: impl Fn(usize, u64) -> Vec<u8>) -> &[u8] {
        self.cached
            .get_or_init(|| resolver(self.angle_idx, self.frame_number))
    }

    /// Returns `true` if the pixel data has already been loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.cached.get().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_lazy_frame_not_loaded_initially() {
        let frame = LazyFrameRef::new(0, 42, 1920, 1080);
        assert!(
            !frame.is_loaded(),
            "frame should not be loaded before resolve_with"
        );
    }

    #[test]
    fn test_lazy_frame_loads_on_access() {
        let frame = LazyFrameRef::new(1, 10, 320, 240);
        let _ = frame.resolve_with(|_angle, _num| vec![0u8; 4]);
        assert!(
            frame.is_loaded(),
            "frame should be loaded after resolve_with"
        );
    }

    #[test]
    fn test_lazy_frame_caches_result() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);

        let frame = LazyFrameRef::new(0, 5, 64, 64);

        // Call resolve_with twice; the resolver closure should only execute once.
        for _ in 0..2 {
            let cc_inner = Arc::clone(&cc);
            let bytes = frame.resolve_with(move |_a, _f| {
                cc_inner.fetch_add(1, Ordering::SeqCst);
                vec![255u8; 64 * 64 * 3]
            });
            assert_eq!(bytes.len(), 64 * 64 * 3);
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolver must be called exactly once regardless of how many times resolve_with is invoked"
        );
    }

    #[test]
    fn test_lazy_frame_metadata() {
        let frame = LazyFrameRef::new(3, 99, 4096, 2160);
        assert_eq!(frame.angle_idx, 3);
        assert_eq!(frame.frame_number, 99);
        assert_eq!(frame.width, 4096);
        assert_eq!(frame.height, 2160);
    }
}
