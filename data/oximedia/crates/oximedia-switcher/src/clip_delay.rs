//! Clip delay buffer for the video switcher — holds frames for deliberate output delay.
#![allow(dead_code)]

use std::collections::VecDeque;

/// Unit used to express a delay amount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayUnit {
    /// Delay expressed as an integer number of frames.
    Frames,
    /// Delay expressed in whole milliseconds (converted to frames on configuration).
    Milliseconds,
    /// Delay expressed in whole fields (interlaced: 2 fields per frame).
    Fields,
}

impl DelayUnit {
    /// Convert a delay value in this unit to an equivalent frame count.
    /// `frame_rate` is frames per second (e.g. 25, 30, 50, 60).
    pub fn to_frames(self, value: u32, frame_rate: u32) -> u32 {
        match self {
            Self::Frames => value,
            Self::Fields => value / 2,
            Self::Milliseconds => {
                if frame_rate == 0 {
                    return 0;
                }
                (value * frame_rate) / 1000
            }
        }
    }
}

/// Configuration for a clip delay instance.
#[derive(Debug, Clone)]
pub struct ClipDelayConfig {
    /// Requested delay value.
    pub delay_value: u32,
    /// Unit of the delay value.
    pub delay_unit: DelayUnit,
    /// Frame rate of the video stream.
    pub frame_rate: u32,
    /// Hard upper bound on the delay buffer in frames.
    pub max_delay_frames: u32,
}

impl ClipDelayConfig {
    /// Create a new config.
    pub fn new(delay_value: u32, delay_unit: DelayUnit, frame_rate: u32) -> Self {
        Self {
            delay_value,
            delay_unit,
            frame_rate,
            max_delay_frames: 300, // 10 seconds at 30 fps
        }
    }

    /// Return the delay expressed in frames.
    pub fn delay_frames(&self) -> u32 {
        self.delay_unit
            .to_frames(self.delay_value, self.frame_rate)
            .min(self.max_delay_frames)
    }

    /// Return the configured hard maximum delay in frames.
    pub fn max_delay_frames(&self) -> u32 {
        self.max_delay_frames
    }

    /// Set a custom maximum delay.
    pub fn with_max_delay_frames(mut self, max: u32) -> Self {
        self.max_delay_frames = max;
        self
    }
}

/// A lightweight frame token stored in the delay buffer.
#[derive(Debug, Clone)]
pub struct FrameToken {
    /// Frame sequence number.
    pub seq: u64,
    /// Arbitrary payload (e.g. a handle into a frame pool).
    pub payload: u64,
}

impl FrameToken {
    /// Create a new frame token.
    pub fn new(seq: u64, payload: u64) -> Self {
        Self { seq, payload }
    }
}

/// A `ClipDelay` wraps a `ClipDelayConfig` and exposes semantic accessors.
#[derive(Debug, Clone)]
pub struct ClipDelay {
    config: ClipDelayConfig,
}

impl ClipDelay {
    /// Create a new `ClipDelay` from a config.
    pub fn new(config: ClipDelayConfig) -> Self {
        Self { config }
    }

    /// Return the delay in frames.
    pub fn delay_frames(&self) -> u32 {
        self.config.delay_frames()
    }

    /// Return the maximum allowed delay in frames.
    pub fn max_delay_frames(&self) -> u32 {
        self.config.max_delay_frames()
    }
}

/// Ring-buffer style delay buffer that accepts incoming frames and
/// releases them after the configured number of frames have elapsed.
#[derive(Debug)]
pub struct DelayBuffer {
    delay_frames: u32,
    buffer: VecDeque<FrameToken>,
}

impl DelayBuffer {
    /// Create a new buffer with the specified delay in frames.
    pub fn new(delay_frames: u32) -> Self {
        Self {
            delay_frames,
            buffer: VecDeque::new(),
        }
    }

    /// Write a new frame into the buffer.
    pub fn write_frame(&mut self, token: FrameToken) {
        self.buffer.push_back(token);
    }

    /// Read the delayed frame if enough frames have accumulated.
    /// Returns `Some(FrameToken)` when `buffer.len() > delay_frames`, else `None`.
    pub fn read_delayed(&mut self) -> Option<FrameToken> {
        if self.buffer.len() as u32 > self.delay_frames {
            self.buffer.pop_front()
        } else {
            None
        }
    }

    /// Return the number of frames currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return the configured delay in frames.
    pub fn delay_frames(&self) -> u32 {
        self.delay_frames
    }

    /// Drain all frames from the buffer.
    pub fn flush(&mut self) {
        self.buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_unit_frames_identity() {
        assert_eq!(DelayUnit::Frames.to_frames(10, 25), 10);
    }

    #[test]
    fn test_delay_unit_fields_to_frames() {
        assert_eq!(DelayUnit::Fields.to_frames(50, 25), 25);
    }

    #[test]
    fn test_delay_unit_milliseconds_to_frames_25fps() {
        // 1000 ms at 25 fps = 25 frames
        assert_eq!(DelayUnit::Milliseconds.to_frames(1000, 25), 25);
    }

    #[test]
    fn test_delay_unit_milliseconds_zero_framerate() {
        assert_eq!(DelayUnit::Milliseconds.to_frames(1000, 0), 0);
    }

    #[test]
    fn test_config_delay_frames_basic() {
        let cfg = ClipDelayConfig::new(5, DelayUnit::Frames, 25);
        assert_eq!(cfg.delay_frames(), 5);
    }

    #[test]
    fn test_config_delay_clamped_to_max() {
        let cfg = ClipDelayConfig::new(500, DelayUnit::Frames, 25).with_max_delay_frames(100);
        assert_eq!(cfg.delay_frames(), 100);
    }

    #[test]
    fn test_config_max_delay_frames() {
        let cfg = ClipDelayConfig::new(10, DelayUnit::Frames, 30).with_max_delay_frames(600);
        assert_eq!(cfg.max_delay_frames(), 600);
    }

    #[test]
    fn test_clip_delay_delay_frames() {
        let cfg = ClipDelayConfig::new(8, DelayUnit::Frames, 25);
        let delay = ClipDelay::new(cfg);
        assert_eq!(delay.delay_frames(), 8);
    }

    #[test]
    fn test_clip_delay_max_delay_frames() {
        let cfg = ClipDelayConfig::new(8, DelayUnit::Frames, 25).with_max_delay_frames(50);
        let delay = ClipDelay::new(cfg);
        assert_eq!(delay.max_delay_frames(), 50);
    }

    #[test]
    fn test_buffer_no_output_before_delay() {
        let mut buf = DelayBuffer::new(3);
        buf.write_frame(FrameToken::new(0, 0));
        buf.write_frame(FrameToken::new(1, 1));
        buf.write_frame(FrameToken::new(2, 2));
        // 3 frames written, delay = 3 -> buffer.len() == delay, not > -> no output
        assert!(buf.read_delayed().is_none());
    }

    #[test]
    fn test_buffer_outputs_after_delay_exceeded() {
        let mut buf = DelayBuffer::new(3);
        for i in 0..4u64 {
            buf.write_frame(FrameToken::new(i, i));
        }
        // 4 frames written, delay = 3 -> first frame should be released
        let token = buf.read_delayed().expect("should succeed in test");
        assert_eq!(token.seq, 0);
    }

    #[test]
    fn test_buffer_buffered_count() {
        let mut buf = DelayBuffer::new(5);
        buf.write_frame(FrameToken::new(0, 0));
        buf.write_frame(FrameToken::new(1, 1));
        assert_eq!(buf.buffered_count(), 2);
    }

    #[test]
    fn test_buffer_flush() {
        let mut buf = DelayBuffer::new(2);
        buf.write_frame(FrameToken::new(0, 0));
        buf.write_frame(FrameToken::new(1, 1));
        buf.flush();
        assert_eq!(buf.buffered_count(), 0);
    }

    #[test]
    fn test_buffer_zero_delay_outputs_immediately() {
        let mut buf = DelayBuffer::new(0);
        buf.write_frame(FrameToken::new(10, 42));
        // len=1 > 0 -> immediate output
        let token = buf.read_delayed().expect("should succeed in test");
        assert_eq!(token.payload, 42);
    }
}
