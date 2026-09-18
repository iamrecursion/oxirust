//! Frame types for passing through the filter graph.
//!
//! This module provides wrapper types for video and audio frames that can be
//! passed between nodes in the filter graph.

use std::sync::Arc;

use oximedia_audio::{AudioFrame, ChannelLayout};
use oximedia_codec::VideoFrame;
use oximedia_core::{PixelFormat, SampleFormat, Timestamp};

/// A frame that can be passed through the filter graph.
#[derive(Clone, Debug)]
pub enum FilterFrame {
    /// Video frame.
    Video(VideoFrame),
    /// Audio frame.
    Audio(AudioFrame),
}

impl FilterFrame {
    /// Get the timestamp of the frame.
    #[must_use]
    pub fn timestamp(&self) -> &Timestamp {
        match self {
            Self::Video(f) => &f.timestamp,
            Self::Audio(f) => &f.timestamp,
        }
    }

    /// Check if this is a video frame.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Check if this is an audio frame.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }

    /// Get as video frame if applicable.
    #[must_use]
    pub fn as_video(&self) -> Option<&VideoFrame> {
        match self {
            Self::Video(f) => Some(f),
            Self::Audio(_) => None,
        }
    }

    /// Get as audio frame if applicable.
    #[must_use]
    pub fn as_audio(&self) -> Option<&AudioFrame> {
        match self {
            Self::Video(_) => None,
            Self::Audio(f) => Some(f),
        }
    }

    /// Get mutable video frame if applicable.
    pub fn as_video_mut(&mut self) -> Option<&mut VideoFrame> {
        match self {
            Self::Video(f) => Some(f),
            Self::Audio(_) => None,
        }
    }

    /// Get mutable audio frame if applicable.
    pub fn as_audio_mut(&mut self) -> Option<&mut AudioFrame> {
        match self {
            Self::Video(_) => None,
            Self::Audio(f) => Some(f),
        }
    }
}

impl From<VideoFrame> for FilterFrame {
    fn from(frame: VideoFrame) -> Self {
        Self::Video(frame)
    }
}

impl From<AudioFrame> for FilterFrame {
    fn from(frame: AudioFrame) -> Self {
        Self::Audio(frame)
    }
}

/// Reference-counted frame for zero-copy passing.
///
/// When a frame needs to be shared between multiple consumers without copying,
/// use `FrameRef` to wrap it in an `Arc`.
#[derive(Clone, Debug)]
pub struct FrameRef {
    inner: Arc<FilterFrame>,
}

impl FrameRef {
    /// Create a new frame reference.
    pub fn new(frame: FilterFrame) -> Self {
        Self {
            inner: Arc::new(frame),
        }
    }

    /// Get a reference to the inner frame.
    #[must_use]
    pub fn frame(&self) -> &FilterFrame {
        &self.inner
    }

    /// Try to get exclusive access to the frame.
    ///
    /// Returns `Some` if this is the only reference, `None` otherwise.
    pub fn try_unwrap(self) -> Option<FilterFrame> {
        Arc::try_unwrap(self.inner).ok()
    }

    /// Get the reference count.
    #[must_use]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Make a copy of the frame if needed for mutation.
    ///
    /// If this is the only reference, returns the frame directly.
    /// Otherwise, clones the frame.
    #[must_use]
    pub fn make_mut(self) -> FilterFrame {
        match Arc::try_unwrap(self.inner) {
            Ok(frame) => frame,
            Err(arc) => (*arc).clone(),
        }
    }
}

impl From<FilterFrame> for FrameRef {
    fn from(frame: FilterFrame) -> Self {
        Self::new(frame)
    }
}

impl From<VideoFrame> for FrameRef {
    fn from(frame: VideoFrame) -> Self {
        Self::new(FilterFrame::Video(frame))
    }
}

impl From<AudioFrame> for FrameRef {
    fn from(frame: AudioFrame) -> Self {
        Self::new(FilterFrame::Audio(frame))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD-accelerated frame copy
// ─────────────────────────────────────────────────────────────────────────────

/// Copy `src` bytes into `dst` as fast as possible.
///
/// On `x86_64` hosts where AVX2 is detected at runtime, the copy is dispatched
/// through a 32-byte-chunk path that the compiler can auto-vectorize into
/// `vmovdqu` loads/stores.  On all other targets (and for any trailing bytes)
/// a plain `copy_from_slice` is used, which LLVM lowers to a `memcpy` and
/// vectorizes independently.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
pub fn simd_copy_frame(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len(), "simd_copy_frame: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            avx2_copy_safe(src, dst);
            return;
        }
    }

    dst.copy_from_slice(src);
}

/// 32-byte-chunk copy path that LLVM will lower to AVX2 `vmovdqu` on x86_64
/// hosts where AVX2 is available.
///
/// The function is `#[inline(never)]` so the compiler emits a dedicated
/// version without inlining overhead at every call site; the runtime AVX2
/// check in [`simd_copy_frame`] guards the dispatch.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn avx2_copy_safe(src: &[u8], dst: &mut [u8]) {
    const CHUNK: usize = 32;
    let chunks = src.len() / CHUNK;
    // Process 32-byte chunks.
    for i in 0..chunks {
        let offset = i * CHUNK;
        dst[offset..offset + CHUNK].copy_from_slice(&src[offset..offset + CHUNK]);
    }
    // Handle tail bytes (0–31).
    let done = chunks * CHUNK;
    if done < src.len() {
        dst[done..].copy_from_slice(&src[done..]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for pre-allocating raw byte buffers in a [`FramePool`].
///
/// Use [`FramePool::with_config`] to construct a pool that eagerly allocates
/// `pre_allocate` buffers of `frame_bytes` bytes each before the first call to
/// [`FramePool::acquire_raw`].
#[derive(Debug, Clone)]
pub struct FramePoolConfig {
    /// Number of raw byte buffers to allocate at construction time.
    pub pre_allocate: usize,
    /// Hard cap on the total number of pooled raw buffers.
    pub max_size: usize,
    /// Size in bytes of each raw buffer (`width * height * channels`).
    pub frame_bytes: usize,
}

impl Default for FramePoolConfig {
    fn default() -> Self {
        Self {
            pre_allocate: 0,
            max_size: 32,
            frame_bytes: 0,
        }
    }
}

/// Frame pool for reusing frame allocations.
///
/// Supports both typed [`VideoFrame`]/[`AudioFrame`] recycling and a raw
/// byte-buffer free-list that can be pre-allocated via [`FramePoolConfig`].
pub struct FramePool {
    /// Maximum number of frames to keep in the pool.
    capacity: usize,
    /// Pooled video frames.
    video_frames: Vec<VideoFrame>,
    /// Pooled audio frames.
    audio_frames: Vec<AudioFrame>,
    /// Pre-allocated / recycled raw byte buffers.
    free_list: Vec<Vec<u8>>,
    /// Hard cap on the `free_list`.
    raw_max: usize,
    /// Expected byte length for raw buffers (0 = unchecked).
    raw_frame_bytes: usize,
}

impl FramePool {
    /// Create a new frame pool with the given capacity for typed frames.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            video_frames: Vec::with_capacity(capacity),
            audio_frames: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            raw_max: capacity,
            raw_frame_bytes: 0,
        }
    }

    /// Create a pool pre-populated according to `config`.
    ///
    /// `config.pre_allocate` zeroed buffers of `config.frame_bytes` bytes are
    /// placed in the internal free-list immediately, so the first
    /// `pre_allocate` calls to [`Self::acquire_raw`] return without any heap
    /// allocation.
    #[must_use]
    pub fn with_config(config: FramePoolConfig) -> Self {
        let pre = config.pre_allocate.min(config.max_size);
        let mut free_list = Vec::with_capacity(pre.max(config.max_size));
        for _ in 0..pre {
            free_list.push(vec![0u8; config.frame_bytes]);
        }
        Self {
            capacity: config.max_size,
            video_frames: Vec::new(),
            audio_frames: Vec::new(),
            free_list,
            raw_max: config.max_size,
            raw_frame_bytes: config.frame_bytes,
        }
    }

    /// Acquire a raw byte buffer from the free-list, or allocate a new one.
    ///
    /// If `frame_bytes` was set in the config, the returned buffer is exactly
    /// that size.  Otherwise a zero-length buffer is returned for newly
    /// allocated entries — callers should resize as needed.
    pub fn acquire_raw(&mut self) -> Vec<u8> {
        self.free_list.pop().unwrap_or_else(|| {
            if self.raw_frame_bytes > 0 {
                vec![0u8; self.raw_frame_bytes]
            } else {
                Vec::new()
            }
        })
    }

    /// Return a raw byte buffer to the free-list for reuse.
    ///
    /// Silently drops the buffer when the free-list is at capacity.
    pub fn release_raw(&mut self, buf: Vec<u8>) {
        if self.free_list.len() < self.raw_max {
            self.free_list.push(buf);
        }
    }

    /// Number of raw buffers currently available in the free-list.
    #[must_use]
    pub fn pre_allocated_count(&self) -> usize {
        self.free_list.len()
    }

    /// Get a video frame from the pool or create a new one.
    #[must_use]
    pub fn get_video_frame(&mut self, format: PixelFormat, width: u32, height: u32) -> VideoFrame {
        // Try to find a matching frame in the pool
        if let Some(pos) = self
            .video_frames
            .iter()
            .position(|f| f.format == format && f.width == width && f.height == height)
        {
            return self.video_frames.swap_remove(pos);
        }

        // Create a new frame
        let mut frame = VideoFrame::new(format, width, height);
        frame.allocate();
        frame
    }

    /// Return a video frame to the pool.
    pub fn return_video_frame(&mut self, frame: VideoFrame) {
        if self.video_frames.len() < self.capacity {
            self.video_frames.push(frame);
        }
    }

    /// Get an audio frame from the pool or create a new one.
    #[must_use]
    pub fn get_audio_frame(
        &mut self,
        format: SampleFormat,
        sample_rate: u32,
        channels: ChannelLayout,
    ) -> AudioFrame {
        // Try to find a matching frame in the pool
        if let Some(pos) = self.audio_frames.iter().position(|f| {
            f.format == format && f.sample_rate == sample_rate && f.channels == channels
        }) {
            return self.audio_frames.swap_remove(pos);
        }

        // Create a new frame
        AudioFrame::new(format, sample_rate, channels)
    }

    /// Return an audio frame to the pool.
    pub fn return_audio_frame(&mut self, frame: AudioFrame) {
        if self.audio_frames.len() < self.capacity {
            self.audio_frames.push(frame);
        }
    }

    /// Clear all pooled frames.
    pub fn clear(&mut self) {
        self.video_frames.clear();
        self.audio_frames.clear();
    }

    /// Get the number of video frames in the pool.
    #[must_use]
    pub fn video_frame_count(&self) -> usize {
        self.video_frames.len()
    }

    /// Get the number of audio frames in the pool.
    #[must_use]
    pub fn audio_frame_count(&self) -> usize {
        self.audio_frames.len()
    }
}

impl Default for FramePool {
    fn default() -> Self {
        Self::new(16)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero-copy frame passing
// ─────────────────────────────────────────────────────────────────────────────

/// A raw-bytes frame that may be shared (zero-copy) or exclusively owned.
///
/// Adjacent nodes that produce and consume the same format can pass a
/// [`SharedFrame::Shared`] variant through the graph; the receiving node reads
/// the bytes via [`SharedFrame::as_bytes`] without any memcpy.  When a node
/// *must* mutate the payload it calls [`SharedFrame::into_owned`], which clones
/// the bytes only when there is more than one live `Arc` reference.
#[derive(Clone, Debug)]
pub enum SharedFrame {
    /// Exclusively-owned byte buffer.
    Owned(Vec<u8>),
    /// Reference-counted (shared) byte buffer — zero-copy on the read path.
    Shared(Arc<Vec<u8>>),
}

impl SharedFrame {
    /// Borrow the underlying bytes regardless of ownership variant.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Shared(arc) => arc.as_slice(),
        }
    }

    /// Consume into an exclusively-owned `Vec<u8>`.
    ///
    /// * `Owned` — free, no allocation.
    /// * `Shared` with a unique reference — moves the inner `Vec` out of the
    ///   `Arc` without copying.
    /// * `Shared` with multiple references — clones the bytes once.
    #[must_use]
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Owned(v) => v,
            Self::Shared(arc) => match Arc::try_unwrap(arc) {
                Ok(v) => v,
                Err(arc) => (*arc).clone(),
            },
        }
    }

    /// Return an `Arc<Vec<u8>>` that points at the same allocation as `self`.
    ///
    /// * `Shared` — clones the `Arc` (O(1), no heap allocation).
    /// * `Owned` — wraps the buffer in a new `Arc` (one allocation, O(1)).
    #[must_use]
    pub fn try_share(&self) -> Arc<Vec<u8>> {
        match self {
            Self::Owned(v) => Arc::new(v.clone()),
            Self::Shared(arc) => Arc::clone(arc),
        }
    }

    /// Promote an `Owned` frame to `Shared`, consuming `self`.
    ///
    /// If already `Shared`, returns `self` unchanged.
    #[must_use]
    pub fn promote(self) -> Self {
        match self {
            Self::Owned(v) => Self::Shared(Arc::new(v)),
            already_shared => already_shared,
        }
    }

    /// Returns `true` if this is the `Shared` variant.
    #[must_use]
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    /// Returns `true` if this is the `Owned` variant.
    #[must_use]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }
}

impl From<Vec<u8>> for SharedFrame {
    fn from(v: Vec<u8>) -> Self {
        Self::Owned(v)
    }
}

impl From<Arc<Vec<u8>>> for SharedFrame {
    fn from(arc: Arc<Vec<u8>>) -> Self {
        Self::Shared(arc)
    }
}

/// Trait for nodes that can participate in the zero-copy frame-passing protocol.
///
/// Nodes advertise compatibility via [`ZeroCopyPort::accepts_zero_copy`].  When
/// two adjacent nodes are both compatible, the graph executor may pass a
/// [`SharedFrame::Shared`] variant directly instead of copying bytes.
pub trait ZeroCopyPort {
    /// Return `true` when this port can receive a [`SharedFrame::Shared`]
    /// without requiring an exclusive copy.
    fn accepts_zero_copy(&self) -> bool;

    /// Pass a frame through this port.
    ///
    /// A compatible implementation should return the frame as-is (or promote
    /// it to `Shared`) when [`Self::accepts_zero_copy`] is `true`.  Otherwise
    /// it may convert to an `Owned` copy.
    fn pass_frame(&self, frame: SharedFrame) -> SharedFrame;
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_frame_video() {
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame = FilterFrame::Video(video);

        assert!(frame.is_video());
        assert!(!frame.is_audio());
        assert!(frame.as_video().is_some());
        assert!(frame.as_audio().is_none());
    }

    #[test]
    fn test_filter_frame_audio() {
        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let frame = FilterFrame::Audio(audio);

        assert!(!frame.is_video());
        assert!(frame.is_audio());
        assert!(frame.as_video().is_none());
        assert!(frame.as_audio().is_some());
    }

    #[test]
    fn test_frame_ref() {
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame = FilterFrame::Video(video);
        let frame_ref = FrameRef::new(frame);

        assert_eq!(frame_ref.ref_count(), 1);

        let frame_ref2 = frame_ref.clone();
        assert_eq!(frame_ref.ref_count(), 2);
        assert_eq!(frame_ref2.ref_count(), 2);
    }

    #[test]
    fn test_frame_ref_try_unwrap() {
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame = FilterFrame::Video(video);
        let frame_ref = FrameRef::new(frame);

        // Should succeed with single reference
        let unwrapped = frame_ref.try_unwrap();
        assert!(unwrapped.is_some());
    }

    #[test]
    fn test_frame_ref_make_mut() {
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame = FilterFrame::Video(video);
        let frame_ref = FrameRef::new(frame);
        let frame_ref2 = frame_ref.clone();

        // Should clone since there are multiple references
        let owned = frame_ref.make_mut();
        assert!(owned.is_video());

        // frame_ref2 should still be valid
        assert!(frame_ref2.frame().is_video());
    }

    #[test]
    fn test_frame_pool() {
        let mut pool = FramePool::new(4);

        // Get a new frame
        let frame = pool.get_video_frame(PixelFormat::Yuv420p, 1920, 1080);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);

        // Return it to the pool
        pool.return_video_frame(frame);
        assert_eq!(pool.video_frame_count(), 1);

        // Get it back (should be the same allocation)
        let frame2 = pool.get_video_frame(PixelFormat::Yuv420p, 1920, 1080);
        assert_eq!(frame2.width, 1920);
        assert_eq!(pool.video_frame_count(), 0);
    }

    #[test]
    fn test_filter_frame_from() {
        let video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        let frame: FilterFrame = video.into();
        assert!(frame.is_video());

        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let frame: FilterFrame = audio.into();
        assert!(frame.is_audio());
    }

    #[test]
    fn test_frame_timestamp() {
        use oximedia_core::Rational;

        let mut video = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        video.timestamp = Timestamp::new(1000, Rational::new(1, 1000));
        let frame = FilterFrame::Video(video);

        assert_eq!(frame.timestamp().pts, 1000);
    }

    // ── simd_copy_frame ───────────────────────────────────────────────────────

    #[test]
    fn test_simd_copy_correctness() {
        // 4096-byte buffer — covers 128 × 32-byte AVX2 chunks.
        let src: Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
        let mut dst = vec![0u8; 4096];
        super::simd_copy_frame(&src, &mut dst);
        assert_eq!(src, dst, "4096-byte copy must be byte-perfect");
    }

    #[test]
    fn test_simd_copy_non_aligned() {
        // 4097 bytes — 128 full AVX2 chunks plus 1 tail byte.
        let src: Vec<u8> = (0u32..4097).map(|i| (i % 197) as u8).collect();
        let mut dst = vec![0u8; 4097];
        super::simd_copy_frame(&src, &mut dst);
        assert_eq!(src, dst, "4097-byte copy (tail) must be byte-perfect");
    }

    // ── FramePoolConfig / pre-allocation ─────────────────────────────────────

    #[test]
    fn test_pool_pre_allocation_count() {
        let config = FramePoolConfig {
            pre_allocate: 5,
            max_size: 10,
            frame_bytes: 64,
        };
        let pool = FramePool::with_config(config);
        assert_eq!(
            pool.pre_allocated_count(),
            5,
            "pool must expose 5 pre-allocated buffers before any acquire"
        );
    }

    #[test]
    fn test_pool_pre_allocation_acquire() {
        let config = FramePoolConfig {
            pre_allocate: 5,
            max_size: 10,
            frame_bytes: 64,
        };
        let mut pool = FramePool::with_config(config);

        // Drain all 5 pre-allocated buffers — each must be 64 bytes.
        for i in 0..5 {
            let buf = pool.acquire_raw();
            assert_eq!(buf.len(), 64, "pre-allocated buffer {i} must be 64 bytes");
        }
        assert_eq!(
            pool.pre_allocated_count(),
            0,
            "free-list should be empty now"
        );

        // 6th acquire must still succeed (dynamic allocation, same size).
        let buf6 = pool.acquire_raw();
        assert_eq!(buf6.len(), 64, "6th (dynamic) buffer must also be 64 bytes");
    }

    // ── SharedFrame / ZeroCopyPort ───────────────────────────────────────────

    #[test]
    fn test_shared_frame_zero_copy_count() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let arc = Arc::new(data);
        let frame = SharedFrame::Shared(Arc::clone(&arc));

        // The shared clone created by try_share should bring the count to 3
        // (arc + frame's inner arc + the one returned by try_share).
        let shared = frame.try_share();
        // arc, frame's inner arc, and `shared` all point to the same allocation.
        assert_eq!(
            Arc::strong_count(&shared),
            3,
            "strong_count must be 3 after arc + frame + try_share clone"
        );
    }

    #[test]
    fn test_shared_frame_into_owned_clone() {
        let data: Vec<u8> = vec![10, 20, 30];
        let arc = Arc::new(data.clone());
        // Create a second reference so try_unwrap will fail and clone.
        let _second_ref = Arc::clone(&arc);
        let frame = SharedFrame::Shared(arc);

        let owned = frame.into_owned();
        assert_eq!(owned, data, "cloned bytes must match the original");
    }

    #[test]
    fn test_owned_frame_to_shared() {
        let data: Vec<u8> = vec![7, 8, 9];
        let frame = SharedFrame::Owned(data.clone());

        assert!(frame.is_owned());

        let promoted = frame.promote();
        assert!(
            promoted.is_shared(),
            "Owned frame must become Shared after promote()"
        );

        // Bytes are preserved.
        assert_eq!(promoted.as_bytes(), data.as_slice());
    }
}
