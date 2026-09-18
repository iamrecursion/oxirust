//! Streaming / pipeline-mode denoiser for integration with `oximedia-graph`.
//!
//! `StreamingDenoiser` processes one frame at a time with a configurable
//! temporal look-ahead latency.  It keeps an internal [`VecDeque`]-based
//! frame ring of size `window`, outputs the denoised centre frame as soon
//! as the window is full, and then advances by one frame each call.
//!
//! # Latency
//!
//! When `window = N` (must be odd, minimum 3), the denoiser introduces a
//! latency of `(N - 1) / 2` frames before the first output is produced.
//! Once warmed up, every `process_frame` call produces exactly one output.
//!
//! # Integration with oximedia-graph
//!
//! ```
//! use oximedia_denoise::streaming::{StreamingDenoiser, StreamingConfig};
//! use oximedia_denoise::DenoiseMode;
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let cfg = StreamingConfig {
//!     window: 5,
//!     mode: DenoiseMode::Balanced,
//!     strength: 0.5,
//! };
//! let mut sd = StreamingDenoiser::new(cfg).expect("valid config");
//!
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
//! frame.allocate();
//!
//! // First (window-1)/2 calls buffer frames and return None.
//! assert!(sd.process_frame(frame.clone()).is_none());
//! // ... until the window is filled.
//! ```

use std::collections::VecDeque;

use crate::hybrid::spatiotemporal;
use crate::spatial::bilateral;
use crate::{DenoiseError, DenoiseMode, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Configuration for [`StreamingDenoiser`].
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Number of frames in the temporal window (must be odd, in `[3, 15]`).
    pub window: usize,
    /// Denoising mode applied to the centre frame of the window.
    pub mode: DenoiseMode,
    /// Denoising strength (0.0 – 1.0).
    pub strength: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window: 5,
            mode: DenoiseMode::Balanced,
            strength: 0.5,
        }
    }
}

impl StreamingConfig {
    /// Validate configuration fields.
    pub fn validate(&self) -> DenoiseResult<()> {
        if self.window < 3 || self.window > 15 {
            return Err(DenoiseError::InvalidConfig(
                "StreamingConfig window must be in [3, 15]".to_string(),
            ));
        }
        if self.window % 2 == 0 {
            return Err(DenoiseError::InvalidConfig(
                "StreamingConfig window must be odd".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(DenoiseError::InvalidConfig(
                "StreamingConfig strength must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

/// A low-latency streaming denoiser that processes one frame at a time.
///
/// Uses a [`VecDeque`] internally so push/pop operations are O(1).
pub struct StreamingDenoiser {
    config: StreamingConfig,
    /// Ring buffer of buffered frames.  Capacity = `config.window`.
    ring: VecDeque<VideoFrame>,
    /// Total number of frames pushed so far (for diagnostics).
    frames_pushed: u64,
    /// Total number of frames output so far.
    frames_output: u64,
}

impl StreamingDenoiser {
    /// Construct a `StreamingDenoiser`.
    ///
    /// # Errors
    ///
    /// Returns [`DenoiseError::InvalidConfig`] if the configuration is invalid.
    pub fn new(config: StreamingConfig) -> DenoiseResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            ring: VecDeque::new(),
            frames_pushed: 0,
            frames_output: 0,
        })
    }

    /// Push a new frame into the pipeline.
    ///
    /// Returns `Some(denoised_frame)` once the window is full (i.e., after
    /// `(window - 1) / 2` frames have been buffered).  Returns `None` while
    /// filling the initial look-ahead buffer.
    ///
    /// After the pipeline is warm, every call returns `Some`.
    pub fn process_frame(&mut self, frame: VideoFrame) -> Option<VideoFrame> {
        self.ring.push_back(frame);
        self.frames_pushed += 1;

        // Trim excess (should not normally happen but be defensive).
        while self.ring.len() > self.config.window {
            self.ring.pop_front();
        }

        if self.ring.len() < self.config.window {
            // Window not yet filled — no output yet.
            return None;
        }

        // Window is full: denoise the centre frame.
        let centre_idx = self.config.window / 2;
        let centre = self.ring[centre_idx].clone();
        let frames: Vec<VideoFrame> = self.ring.iter().cloned().collect();

        let result = self.apply_denoise(&centre, &frames);
        // Advance: drop oldest frame.
        self.ring.pop_front();
        self.frames_output += 1;

        match result {
            Ok(denoised) => Some(denoised),
            Err(_) => Some(centre),
        }
    }

    /// Drain any remaining frames from the pipeline (flush the look-ahead).
    ///
    /// Calling this after all input frames have been pushed yields the
    /// denoised versions of the trailing `(window - 1) / 2` frames.
    /// Each call returns the next pending frame, or `None` when empty.
    pub fn flush_next(&mut self) -> Option<VideoFrame> {
        if self.ring.is_empty() {
            return None;
        }

        // Pad with last frame to maintain window.
        let last = self.ring.back()?.clone();
        self.ring.push_back(last);

        if self.ring.len() < self.config.window {
            return None;
        }

        let centre_idx = self.config.window / 2;
        let centre = self.ring[centre_idx].clone();
        let frames: Vec<VideoFrame> = self.ring.iter().cloned().collect();

        let result = self.apply_denoise(&centre, &frames);
        self.ring.pop_front();
        self.frames_output += 1;

        match result {
            Ok(denoised) => Some(denoised),
            Err(_) => Some(centre),
        }
    }

    /// Number of frames pushed so far.
    #[must_use]
    pub fn frames_pushed(&self) -> u64 {
        self.frames_pushed
    }

    /// Number of frames output so far.
    #[must_use]
    pub fn frames_output(&self) -> u64 {
        self.frames_output
    }

    /// Current ring buffer depth.
    #[must_use]
    pub fn buffer_depth(&self) -> usize {
        self.ring.len()
    }

    /// Returns `true` if the pipeline has been fully warmed up.
    #[must_use]
    pub fn is_warmed_up(&self) -> bool {
        self.ring.len() >= self.config.window
    }

    /// Reset internal state without changing configuration.
    pub fn reset(&mut self) {
        self.ring.clear();
        self.frames_pushed = 0;
        self.frames_output = 0;
    }

    /// Get a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Apply denoising to `centre` given the full frame window `frames`.
    fn apply_denoise(
        &self,
        centre: &VideoFrame,
        frames: &[VideoFrame],
    ) -> DenoiseResult<VideoFrame> {
        match self.config.mode {
            DenoiseMode::Fast | DenoiseMode::GrainAware => {
                bilateral::bilateral_filter(centre, self.config.strength)
            }
            DenoiseMode::Balanced | DenoiseMode::Quality | DenoiseMode::Custom => {
                if frames.len() >= 3 {
                    spatiotemporal::spatio_temporal_denoise(
                        centre,
                        frames,
                        self.config.strength,
                        true,
                    )
                } else {
                    bilateral::bilateral_filter(centre, self.config.strength)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_frame(w: u32, h: u32) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Yuv420p, w, h);
        f.allocate();
        f
    }

    #[test]
    fn test_streaming_config_valid() {
        let cfg = StreamingConfig {
            window: 5,
            mode: DenoiseMode::Fast,
            strength: 0.5,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_streaming_config_window_too_small() {
        let cfg = StreamingConfig {
            window: 1,
            mode: DenoiseMode::Fast,
            strength: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_streaming_config_window_even() {
        let cfg = StreamingConfig {
            window: 4,
            mode: DenoiseMode::Fast,
            strength: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_streaming_config_window_too_large() {
        let cfg = StreamingConfig {
            window: 17,
            mode: DenoiseMode::Fast,
            strength: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_streaming_config_invalid_strength() {
        let cfg = StreamingConfig {
            window: 5,
            mode: DenoiseMode::Fast,
            strength: 1.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_streaming_latency_warmup() {
        // With window=5, latency = (5-1)/2 = 2 frames.
        let cfg = StreamingConfig {
            window: 5,
            mode: DenoiseMode::Fast,
            strength: 0.3,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");

        // First 4 calls should return None (window needs 5 frames total).
        for i in 0..4 {
            let result = sd.process_frame(make_frame(32, 32));
            assert!(result.is_none(), "frame {i}: expected None during warmup");
        }
        // 5th call fills the window -> output.
        let result = sd.process_frame(make_frame(32, 32));
        assert!(result.is_some(), "frame 5: should produce output");
    }

    #[test]
    fn test_streaming_steady_state() {
        // Once warmed, every call should produce output.
        let cfg = StreamingConfig {
            window: 3,
            mode: DenoiseMode::Fast,
            strength: 0.3,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");

        // Fill window (window=3, need 2 frames to fill before 3rd produces output).
        sd.process_frame(make_frame(16, 16));
        sd.process_frame(make_frame(16, 16));
        // From here every frame produces output.
        for _ in 0..10 {
            let result = sd.process_frame(make_frame(16, 16));
            assert!(result.is_some(), "should produce output in steady state");
        }
    }

    #[test]
    fn test_streaming_output_dimensions() {
        let cfg = StreamingConfig {
            window: 3,
            mode: DenoiseMode::Fast,
            strength: 0.5,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");
        sd.process_frame(make_frame(48, 48));
        sd.process_frame(make_frame(48, 48));
        let result = sd.process_frame(make_frame(48, 48)).expect("output");
        assert_eq!(result.width, 48);
        assert_eq!(result.height, 48);
    }

    #[test]
    fn test_streaming_frame_counter() {
        let cfg = StreamingConfig {
            window: 3,
            mode: DenoiseMode::Fast,
            strength: 0.3,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");
        for _ in 0..5 {
            sd.process_frame(make_frame(16, 16));
        }
        assert_eq!(sd.frames_pushed(), 5);
        // window=3: outputs start on frame 3, so 3 outputs total from 5 pushes.
        assert_eq!(sd.frames_output(), 3);
    }

    #[test]
    fn test_streaming_reset() {
        let cfg = StreamingConfig {
            window: 3,
            mode: DenoiseMode::Fast,
            strength: 0.3,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");
        sd.process_frame(make_frame(16, 16));
        sd.process_frame(make_frame(16, 16));
        sd.reset();
        assert_eq!(sd.frames_pushed(), 0);
        assert_eq!(sd.frames_output(), 0);
        assert_eq!(sd.buffer_depth(), 0);
    }

    #[test]
    fn test_streaming_flush() {
        let cfg = StreamingConfig {
            window: 5,
            mode: DenoiseMode::Fast,
            strength: 0.3,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");
        // Push enough to fill window.
        for _ in 0..5 {
            sd.process_frame(make_frame(16, 16));
        }
        // Flush trailing frames.
        let flushed = sd.flush_next();
        assert!(flushed.is_some());
    }

    #[test]
    fn test_streaming_balanced_mode() {
        let cfg = StreamingConfig {
            window: 5,
            mode: DenoiseMode::Balanced,
            strength: 0.5,
        };
        let mut sd = StreamingDenoiser::new(cfg).expect("valid");
        for _ in 0..8 {
            sd.process_frame(make_frame(32, 32));
        }
        assert!(sd.frames_output() > 0, "should have produced output");
    }
}
