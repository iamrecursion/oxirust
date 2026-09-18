//! Frame buffer management for broadcast playout.
//!
//! Provides a fixed-capacity ring buffer of video frames with pre-roll
//! support, frame-accurate indexing, and under-run / overflow detection.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Pixel format used in a frame buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit YUV 4:2:2.
    Yuv422P8,
    /// 10-bit YUV 4:2:2.
    Yuv422P10,
    /// 8-bit RGBA.
    Rgba8,
    /// 10-bit RGBA.
    Rgba10,
}

impl PixelFormat {
    /// Bytes per pixel (approximate for planar formats).
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Yuv422P8 => 2,
            Self::Yuv422P10 => 3,
            Self::Rgba8 => 4,
            Self::Rgba10 => 5,
        }
    }
}

/// Metadata attached to each buffered frame.
#[derive(Debug, Clone)]
pub struct FrameMeta {
    /// Monotonic frame index since playout start.
    pub frame_index: u64,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
    /// Whether this frame is a keyframe.
    pub is_key: bool,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
}

/// A buffered video frame.
#[derive(Debug, Clone)]
pub struct BufferedFrame {
    /// Frame metadata.
    pub meta: FrameMeta,
    /// Raw pixel data (opaque byte vector).
    pub data: Vec<u8>,
}

/// Statistics reported by the frame buffer.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Total frames pushed into the buffer since creation.
    pub total_pushed: u64,
    /// Total frames popped (consumed).
    pub total_popped: u64,
    /// Number of times a push was rejected because the buffer was full.
    pub overflow_count: u64,
    /// Number of times a pop was attempted on an empty buffer.
    pub underrun_count: u64,
}

/// Strategy to use when a frame pop encounters an empty buffer (underrun).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDropRecovery {
    /// Return `None` — the caller must handle the gap.
    None,
    /// Repeat the last successfully delivered frame.
    RepeatLast,
    /// Generate a black (all-zero) frame with the same dimensions as the last frame.
    Black,
    /// Generate a mid-grey slate frame (Y=128 for 8-bit) as a visual "signal lost" indicator.
    Slate,
}

impl Default for FrameDropRecovery {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for the frame ring buffer.
#[derive(Debug, Clone)]
pub struct FrameBufferConfig {
    /// Maximum number of frames the buffer can hold.
    pub capacity: usize,
    /// Number of frames that must be buffered before playout may begin.
    pub pre_roll: usize,
    /// Strategy when a pop encounters an empty buffer.
    pub drop_recovery: FrameDropRecovery,
}

impl Default for FrameBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 30,
            pre_roll: 5,
            drop_recovery: FrameDropRecovery::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame Ring Buffer
// ---------------------------------------------------------------------------

/// A ring buffer of video frames with pre-roll gating and statistics.
#[derive(Debug)]
pub struct FrameBuffer {
    config: FrameBufferConfig,
    ring: VecDeque<BufferedFrame>,
    stats: BufferStats,
    pre_roll_met: bool,
    /// Last successfully popped frame, kept for recovery strategies.
    last_frame: Option<BufferedFrame>,
}

impl FrameBuffer {
    /// Create a new frame buffer with the given configuration.
    pub fn new(config: FrameBufferConfig) -> Self {
        let cap = config.capacity;
        Self {
            config,
            ring: VecDeque::with_capacity(cap),
            stats: BufferStats::default(),
            pre_roll_met: false,
            last_frame: None,
        }
    }

    /// Return the current fill level (number of buffered frames).
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Check whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check whether the buffer is full.
    pub fn is_full(&self) -> bool {
        self.ring.len() >= self.config.capacity
    }

    /// Check whether the pre-roll requirement has been met.
    pub fn pre_roll_ready(&self) -> bool {
        self.pre_roll_met
    }

    /// Return a snapshot of the current buffer statistics.
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &FrameBufferConfig {
        &self.config
    }

    /// Push a frame into the buffer.
    ///
    /// Returns `true` if the frame was accepted, `false` if the buffer was
    /// full (overflow).
    pub fn push(&mut self, frame: BufferedFrame) -> bool {
        if self.is_full() {
            self.stats.overflow_count += 1;
            return false;
        }
        self.ring.push_back(frame);
        self.stats.total_pushed += 1;

        if !self.pre_roll_met && self.ring.len() >= self.config.pre_roll {
            self.pre_roll_met = true;
        }
        true
    }

    /// Pop the oldest frame from the buffer.
    ///
    /// When the buffer is empty the configured [`FrameDropRecovery`] strategy
    /// is applied:
    /// - `None` → returns `None` (caller handles the gap).
    /// - `RepeatLast` → clones the last successfully delivered frame.
    /// - `Black` → synthesises an all-zero frame matching the last frame's dimensions.
    /// - `Slate` → synthesises a mid-grey (0x80) frame as a "signal lost" indicator.
    ///
    /// In all underrun cases the `underrun_count` statistic is incremented.
    pub fn pop(&mut self) -> Option<BufferedFrame> {
        if let Some(f) = self.ring.pop_front() {
            self.stats.total_popped += 1;
            self.last_frame = Some(f.clone());
            Some(f)
        } else {
            self.stats.underrun_count += 1;
            self.apply_recovery()
        }
    }

    /// Apply the configured frame-drop recovery strategy.
    fn apply_recovery(&self) -> Option<BufferedFrame> {
        match self.config.drop_recovery {
            FrameDropRecovery::None => None,
            FrameDropRecovery::RepeatLast => self.last_frame.clone(),
            FrameDropRecovery::Black => self.synthesize_recovery_frame(0x00),
            FrameDropRecovery::Slate => self.synthesize_recovery_frame(0x80),
        }
    }

    /// Synthesize a solid-fill recovery frame based on the last known frame
    /// dimensions. Returns `None` if no frame has ever been delivered.
    fn synthesize_recovery_frame(&self, fill_byte: u8) -> Option<BufferedFrame> {
        let last = self.last_frame.as_ref()?;
        let mut frame = last.clone();
        // Fill pixel data with the requested value
        for byte in &mut frame.data {
            *byte = fill_byte;
        }
        // Advance the frame index by one so downstream can detect the synthetic frame
        frame.meta.frame_index = last.meta.frame_index.saturating_add(1);
        frame.meta.pts_us = last.meta.pts_us.saturating_add(
            // Estimate one frame period from the previous two indices (fallback 40 ms)
            40_000,
        );
        frame.meta.is_key = false;
        Some(frame)
    }

    /// Peek at the next frame without removing it.
    pub fn peek(&self) -> Option<&BufferedFrame> {
        self.ring.front()
    }

    /// Flush all buffered frames and reset pre-roll state.
    pub fn flush(&mut self) {
        self.ring.clear();
        self.pre_roll_met = false;
        self.last_frame = None;
    }

    /// Fill percentage as a value in [0.0, 1.0].
    ///
    /// Frame-buffer capacities are always small (≪ 2^53), so the usize→f64
    /// cast is exact on any platform with a 64-bit usize or smaller.
    pub fn fill_ratio(&self) -> f64 {
        if self.config.capacity == 0 {
            return 0.0;
        }
        // SAFETY: broadcast playout capacities (< 10 000 frames) fit exactly in f64.
        let len = self.ring.len() as f64;
        let cap = self.config.capacity as f64;
        len / cap
    }
}

// ---------------------------------------------------------------------------
// PlayoutFrameBuffer — simple capacity-limited frame queue
// ---------------------------------------------------------------------------

/// A thread-safe, capacity-limited FIFO queue of raw video frame bytes.
///
/// Designed for simple playout pipelines where a single producer encodes
/// frames and a single consumer delivers them. Uses a `Mutex<VecDeque>` for
/// simplicity; for ultra-low-latency SPSC use [`crate::lockfree_frame_ring`].
///
/// # Example
///
/// ```
/// use oximedia_playout::frame_buffer::PlayoutFrameBuffer;
/// let buf = PlayoutFrameBuffer::new(10);
/// assert!(buf.push(vec![0u8; 1920 * 1080 * 3]));
/// let frame = buf.pop();
/// assert!(frame.is_some());
/// ```
pub struct PlayoutFrameBuffer {
    capacity: usize,
    inner: std::sync::Mutex<std::collections::VecDeque<Vec<u8>>>,
}

impl PlayoutFrameBuffer {
    /// Create a new buffer that holds at most `capacity` frames.
    ///
    /// A capacity of 0 means no frames can ever be pushed (every push returns
    /// `false`), which is useful as a "disabled" sentinel.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(capacity)),
        }
    }

    /// Push a frame into the buffer.
    ///
    /// Returns `true` on success, or `false` if the buffer is already at
    /// capacity (the frame is silently dropped).
    pub fn push(&self, frame: Vec<u8>) -> bool {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.len() >= self.capacity {
            return false;
        }
        guard.push_back(frame);
        true
    }

    /// Pop the oldest frame from the buffer.
    ///
    /// Returns `None` if the buffer is empty.
    pub fn pop(&self) -> Option<Vec<u8>> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.pop_front()
    }

    /// Current number of frames queued in the buffer.
    pub fn len(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.len()
    }

    /// Returns `true` if the buffer contains no frames.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the buffer is at capacity (next push will fail).
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// The maximum number of frames this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(idx: u64) -> BufferedFrame {
        BufferedFrame {
            meta: FrameMeta {
                frame_index: idx,
                pts_us: (idx as i64) * 40_000,
                is_key: idx.is_multiple_of(10),
                width: 1920,
                height: 1080,
                format: PixelFormat::Yuv422P8,
            },
            data: vec![0u8; 64],
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = FrameBufferConfig::default();
        assert_eq!(cfg.capacity, 30);
        assert_eq!(cfg.pre_roll, 5);
    }

    #[test]
    fn test_push_pop_single() {
        let mut buf = FrameBuffer::new(FrameBufferConfig::default());
        assert!(buf.is_empty());
        assert!(buf.push(make_frame(0)));
        assert_eq!(buf.len(), 1);
        let f = buf.pop().expect("should succeed in test");
        assert_eq!(f.meta.frame_index, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_overflow() {
        let cfg = FrameBufferConfig {
            capacity: 3,
            pre_roll: 1,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        assert!(buf.push(make_frame(0)));
        assert!(buf.push(make_frame(1)));
        assert!(buf.push(make_frame(2)));
        assert!(!buf.push(make_frame(3))); // overflow
        assert_eq!(buf.stats().overflow_count, 1);
    }

    #[test]
    fn test_underrun() {
        let mut buf = FrameBuffer::new(FrameBufferConfig::default());
        assert!(buf.pop().is_none());
        assert_eq!(buf.stats().underrun_count, 1);
    }

    #[test]
    fn test_pre_roll() {
        let cfg = FrameBufferConfig {
            capacity: 10,
            pre_roll: 3,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        assert!(!buf.pre_roll_ready());
        buf.push(make_frame(0));
        buf.push(make_frame(1));
        assert!(!buf.pre_roll_ready());
        buf.push(make_frame(2));
        assert!(buf.pre_roll_ready());
    }

    #[test]
    fn test_flush_resets_pre_roll() {
        let cfg = FrameBufferConfig {
            capacity: 10,
            pre_roll: 2,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        buf.push(make_frame(1));
        assert!(buf.pre_roll_ready());
        buf.flush();
        assert!(!buf.pre_roll_ready());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_fifo_order() {
        let mut buf = FrameBuffer::new(FrameBufferConfig {
            capacity: 10,
            pre_roll: 1,
            ..Default::default()
        });
        buf.push(make_frame(10));
        buf.push(make_frame(20));
        buf.push(make_frame(30));
        assert_eq!(
            buf.pop().expect("should succeed in test").meta.frame_index,
            10
        );
        assert_eq!(
            buf.pop().expect("should succeed in test").meta.frame_index,
            20
        );
        assert_eq!(
            buf.pop().expect("should succeed in test").meta.frame_index,
            30
        );
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut buf = FrameBuffer::new(FrameBufferConfig::default());
        buf.push(make_frame(5));
        assert_eq!(
            buf.peek().expect("should succeed in test").meta.frame_index,
            5
        );
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_fill_ratio() {
        let cfg = FrameBufferConfig {
            capacity: 4,
            pre_roll: 1,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        assert!((buf.fill_ratio() - 0.0).abs() < f64::EPSILON);
        buf.push(make_frame(0));
        buf.push(make_frame(1));
        assert!((buf.fill_ratio() - 0.5).abs() < f64::EPSILON);
        buf.push(make_frame(2));
        buf.push(make_frame(3));
        assert!((buf.fill_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_full() {
        let cfg = FrameBufferConfig {
            capacity: 2,
            pre_roll: 1,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        assert!(!buf.is_full());
        buf.push(make_frame(0));
        buf.push(make_frame(1));
        assert!(buf.is_full());
    }

    #[test]
    fn test_stats_counters() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            ..Default::default()
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        buf.push(make_frame(1));
        buf.pop();
        assert_eq!(buf.stats().total_pushed, 2);
        assert_eq!(buf.stats().total_popped, 1);
    }

    #[test]
    fn test_pixel_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Yuv422P8.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgba10.bytes_per_pixel(), 5);
    }

    #[test]
    fn test_fill_ratio_zero_capacity() {
        let cfg = FrameBufferConfig {
            capacity: 0,
            pre_roll: 0,
            ..Default::default()
        };
        let buf = FrameBuffer::new(cfg);
        assert!((buf.fill_ratio() - 0.0).abs() < f64::EPSILON);
    }

    // --- Frame drop recovery tests ---

    #[test]
    fn test_recovery_none_returns_none() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::None,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        let _ = buf.pop(); // consume the only frame
        assert!(buf.pop().is_none());
        assert_eq!(buf.stats().underrun_count, 1);
    }

    #[test]
    fn test_recovery_repeat_last() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::RepeatLast,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(42));
        let _ = buf.pop(); // consume frame 42
        let recovered = buf.pop();
        assert!(recovered.is_some());
        let f = recovered.expect("recovery should produce a frame");
        assert_eq!(f.meta.frame_index, 42); // identical clone
        assert_eq!(buf.stats().underrun_count, 1);
    }

    #[test]
    fn test_recovery_repeat_last_no_prior_frame() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::RepeatLast,
        };
        let mut buf = FrameBuffer::new(cfg);
        // No frames ever delivered
        assert!(buf.pop().is_none());
        assert_eq!(buf.stats().underrun_count, 1);
    }

    #[test]
    fn test_recovery_black_frame() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::Black,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(10));
        let _ = buf.pop(); // deliver frame 10
        let recovered = buf.pop();
        assert!(recovered.is_some());
        let f = recovered.expect("recovery should produce a frame");
        assert_eq!(f.meta.frame_index, 11); // advanced by 1
        assert!(
            f.data.iter().all(|&b| b == 0x00),
            "black frame should be all zeros"
        );
        assert!(!f.meta.is_key);
    }

    #[test]
    fn test_recovery_slate_frame() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::Slate,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(20));
        let _ = buf.pop();
        let recovered = buf.pop();
        assert!(recovered.is_some());
        let f = recovered.expect("recovery should produce a frame");
        assert_eq!(f.meta.frame_index, 21);
        assert!(
            f.data.iter().all(|&b| b == 0x80),
            "slate frame should be 0x80"
        );
    }

    #[test]
    fn test_recovery_preserves_dimensions() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::Black,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        let original = buf.pop().expect("should deliver frame");
        let recovered = buf.pop().expect("should recover frame");
        assert_eq!(recovered.meta.width, original.meta.width);
        assert_eq!(recovered.meta.height, original.meta.height);
        assert_eq!(recovered.data.len(), original.data.len());
    }

    #[test]
    fn test_flush_clears_last_frame_for_recovery() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::RepeatLast,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        let _ = buf.pop();
        buf.flush();
        // After flush, recovery should return None since last_frame is cleared
        assert!(buf.pop().is_none());
    }

    #[test]
    fn test_recovery_multiple_underruns() {
        let cfg = FrameBufferConfig {
            capacity: 5,
            pre_roll: 1,
            drop_recovery: FrameDropRecovery::Slate,
        };
        let mut buf = FrameBuffer::new(cfg);
        buf.push(make_frame(0));
        let _ = buf.pop();
        // Multiple underruns in a row
        for i in 1..=5 {
            let r = buf.pop();
            assert!(r.is_some());
            assert_eq!(buf.stats().underrun_count, i);
        }
    }

    // --- PlayoutFrameBuffer tests ---

    #[test]
    fn test_frame_buffer_push_pop() {
        let buf = PlayoutFrameBuffer::new(10);
        // Push 3 distinct frames and verify FIFO pop order.
        assert!(buf.push(vec![1u8; 64]));
        assert!(buf.push(vec![2u8; 64]));
        assert!(buf.push(vec![3u8; 64]));
        assert_eq!(buf.len(), 3);

        let f1 = buf.pop().expect("should pop first frame");
        assert_eq!(f1[0], 1, "first pop should return the first pushed frame");
        let f2 = buf.pop().expect("should pop second frame");
        assert_eq!(f2[0], 2, "second pop should return the second pushed frame");
        let f3 = buf.pop().expect("should pop third frame");
        assert_eq!(f3[0], 3, "third pop should return the third pushed frame");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_frame_buffer_capacity() {
        let buf = PlayoutFrameBuffer::new(2);
        assert!(buf.push(vec![0u8; 32]));
        assert!(buf.push(vec![0u8; 32]));
        // Buffer is now full — next push must return false.
        assert!(
            !buf.push(vec![0u8; 32]),
            "push beyond capacity should return false"
        );
        assert!(buf.is_full());
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_frame_buffer_pop_empty() {
        let buf = PlayoutFrameBuffer::new(5);
        assert!(
            buf.pop().is_none(),
            "pop on empty buffer should return None"
        );
    }

    #[test]
    fn test_frame_buffer_zero_capacity() {
        let buf = PlayoutFrameBuffer::new(0);
        assert!(
            !buf.push(vec![1u8]),
            "capacity-0 buffer should reject all pushes"
        );
        assert!(buf.is_empty());
    }
}
