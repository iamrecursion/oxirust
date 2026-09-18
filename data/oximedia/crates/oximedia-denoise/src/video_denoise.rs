//! Video denoising pipeline — per-frame spatial/temporal configuration and processing.
#![allow(dead_code)]

// ── Enums ─────────────────────────────────────────────────────────────────────

/// High-level video denoising mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoDenoiseMode {
    /// Fastest path: spatial-only, single-pass bilateral.
    Realtime,
    /// Balanced spatial + light temporal averaging.
    Balanced,
    /// High-quality NLM + motion-compensated temporal filter.
    Quality,
    /// Preserve intentional film grain while reducing sensor noise.
    GrainPreserve,
    /// No denoising — passthrough for reference.
    Passthrough,
}

impl VideoDenoiseMode {
    /// Whether this mode requires a temporal frame buffer.
    #[must_use]
    pub fn needs_temporal_buffer(self) -> bool {
        matches!(self, VideoDenoiseMode::Balanced | VideoDenoiseMode::Quality)
    }

    /// Display-friendly name.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            VideoDenoiseMode::Realtime => "Realtime",
            VideoDenoiseMode::Balanced => "Balanced",
            VideoDenoiseMode::Quality => "Quality",
            VideoDenoiseMode::GrainPreserve => "GrainPreserve",
            VideoDenoiseMode::Passthrough => "Passthrough",
        }
    }
}

// ── Spatial config ────────────────────────────────────────────────────────────

/// Configuration for the spatial (per-frame) denoising stage.
#[derive(Debug, Clone)]
pub struct SpatialDenoiseConfig {
    /// Sigma for the spatial Gaussian kernel (pixels).
    pub sigma_spatial: f32,
    /// Sigma for the range/intensity Gaussian kernel (intensity units 0–255).
    pub sigma_range: f32,
    /// Kernel radius in pixels.
    pub kernel_radius: usize,
    /// Enable edge-preserving bilateral filter (vs. simple Gaussian).
    pub use_bilateral: bool,
}

impl SpatialDenoiseConfig {
    /// Return `true` when the spatial strength is high enough to be
    /// considered "strong" filtering.
    #[must_use]
    pub fn is_strong(&self) -> bool {
        self.sigma_spatial > 3.0 || self.kernel_radius >= 5
    }
}

impl Default for SpatialDenoiseConfig {
    fn default() -> Self {
        Self {
            sigma_spatial: 2.0,
            sigma_range: 20.0,
            kernel_radius: 3,
            use_bilateral: true,
        }
    }
}

// ── Temporal config ───────────────────────────────────────────────────────────

/// Configuration for the temporal (cross-frame) denoising stage.
#[derive(Debug, Clone)]
pub struct TemporalDenoiseConfig {
    /// Number of reference frames to look back.
    pub window_size: usize,
    /// Blend weight for the current frame vs. the temporal average (0.0–1.0).
    /// `1.0` = only current frame, `0.0` = full temporal average.
    pub current_weight: f32,
    /// Motion threshold — pixels with motion above this are excluded from averaging.
    pub motion_threshold: f32,
    /// Use motion-compensated temporal filtering when `true`.
    pub motion_compensated: bool,
}

impl Default for TemporalDenoiseConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            current_weight: 0.5,
            motion_threshold: 8.0,
            motion_compensated: false,
        }
    }
}

// ── Frame buffer ──────────────────────────────────────────────────────────────

/// A simplified luma-plane frame representation for the pipeline.
#[derive(Debug, Clone)]
pub struct LumaFrame {
    /// Luma samples in raster order.
    pub data: Vec<f32>,
    /// Frame width.
    pub width: usize,
    /// Frame height.
    pub height: usize,
}

impl LumaFrame {
    /// Create a frame filled with zeros.
    #[must_use]
    pub fn blank(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0f32; width * height],
            width,
            height,
        }
    }

    /// Pixel value at `(x, y)`, clamped to frame bounds.
    #[must_use]
    pub fn pixel(&self, x: usize, y: usize) -> f32 {
        let idx = y.min(self.height - 1) * self.width + x.min(self.width - 1);
        self.data[idx]
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Video denoising pipeline combining spatial and temporal stages.
#[derive(Debug)]
pub struct VideoDenoisePipeline {
    /// Selected mode.
    pub mode: VideoDenoiseMode,
    /// Spatial stage configuration.
    pub spatial: SpatialDenoiseConfig,
    /// Temporal stage configuration.
    pub temporal: TemporalDenoiseConfig,
    /// Circular buffer of recent frames.
    frame_buffer: Vec<LumaFrame>,
}

impl VideoDenoisePipeline {
    /// Create a pipeline with explicit settings.
    #[must_use]
    pub fn new(
        mode: VideoDenoiseMode,
        spatial: SpatialDenoiseConfig,
        temporal: TemporalDenoiseConfig,
    ) -> Self {
        Self {
            mode,
            spatial,
            temporal,
            frame_buffer: Vec::new(),
        }
    }

    /// Create a pipeline using defaults for the given mode.
    #[must_use]
    pub fn from_mode(mode: VideoDenoiseMode) -> Self {
        let temporal = match mode {
            VideoDenoiseMode::Quality => TemporalDenoiseConfig {
                window_size: 7,
                current_weight: 0.3,
                motion_compensated: true,
                ..Default::default()
            },
            _ => TemporalDenoiseConfig::default(),
        };
        Self::new(mode, SpatialDenoiseConfig::default(), temporal)
    }

    /// Process one luma frame and return the denoised result.
    ///
    /// The frame is added to the internal temporal buffer automatically.
    pub fn process_frame(&mut self, frame: &LumaFrame) -> LumaFrame {
        self.frame_buffer.push(frame.clone());
        // Trim buffer to window size
        while self.frame_buffer.len() > self.temporal.window_size + 1 {
            self.frame_buffer.remove(0);
        }

        match self.mode {
            VideoDenoiseMode::Passthrough => frame.clone(),
            VideoDenoiseMode::Realtime => self.apply_spatial(frame),
            VideoDenoiseMode::Balanced | VideoDenoiseMode::GrainPreserve => {
                let sp = self.apply_spatial(frame);
                if self.frame_buffer.len() > 1 {
                    self.apply_temporal(&sp)
                } else {
                    sp
                }
            }
            VideoDenoiseMode::Quality => {
                let sp = self.apply_spatial(frame);
                self.apply_temporal(&sp)
            }
        }
    }

    /// Number of frames currently held in the buffer.
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Reset the frame buffer.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Apply a simple box-blur as a stand-in for bilateral filtering.
    fn apply_spatial(&self, frame: &LumaFrame) -> LumaFrame {
        let r = self.spatial.kernel_radius as isize;
        let w = frame.width;
        let h = frame.height;
        let mut out = LumaFrame::blank(w, h);

        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let nx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        let ny = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        sum += frame.data[ny * w + nx];
                        count += 1;
                    }
                }
                out.data[y * w + x] = sum / count as f32;
            }
        }
        out
    }

    /// Temporal average of the buffer frames blended with the current frame.
    fn apply_temporal(&self, current: &LumaFrame) -> LumaFrame {
        if self.frame_buffer.is_empty() {
            return current.clone();
        }
        let w = current.width;
        let h = current.height;
        let n = self.frame_buffer.len() as f32;
        let alpha = self.temporal.current_weight;
        let mut out = LumaFrame::blank(w, h);

        // Compute temporal average from buffer
        let mut avg = vec![0.0f32; w * h];
        for frame in &self.frame_buffer {
            for (a, &p) in avg.iter_mut().zip(frame.data.iter()) {
                *a += p;
            }
        }
        for a in &mut avg {
            *a /= n;
        }

        // Blend current with temporal average
        for i in 0..w * h {
            out.data[i] = alpha * current.data[i] + (1.0 - alpha) * avg[i];
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, value: f32) -> LumaFrame {
        LumaFrame {
            data: vec![value; w * h],
            width: w,
            height: h,
        }
    }

    #[test]
    fn test_video_denoise_mode_needs_temporal_buffer() {
        assert!(!VideoDenoiseMode::Realtime.needs_temporal_buffer());
        assert!(VideoDenoiseMode::Balanced.needs_temporal_buffer());
        assert!(VideoDenoiseMode::Quality.needs_temporal_buffer());
        assert!(!VideoDenoiseMode::GrainPreserve.needs_temporal_buffer());
        assert!(!VideoDenoiseMode::Passthrough.needs_temporal_buffer());
    }

    #[test]
    fn test_video_denoise_mode_label() {
        assert_eq!(VideoDenoiseMode::Realtime.label(), "Realtime");
        assert_eq!(VideoDenoiseMode::Quality.label(), "Quality");
        assert_eq!(VideoDenoiseMode::Passthrough.label(), "Passthrough");
    }

    #[test]
    fn test_spatial_config_is_strong_true() {
        let cfg = SpatialDenoiseConfig {
            sigma_spatial: 5.0,
            kernel_radius: 3,
            ..Default::default()
        };
        assert!(cfg.is_strong());
    }

    #[test]
    fn test_spatial_config_is_strong_large_radius() {
        let cfg = SpatialDenoiseConfig {
            sigma_spatial: 1.0,
            kernel_radius: 5,
            ..Default::default()
        };
        assert!(cfg.is_strong());
    }

    #[test]
    fn test_spatial_config_is_strong_false() {
        let cfg = SpatialDenoiseConfig::default();
        assert!(!cfg.is_strong());
    }

    #[test]
    fn test_temporal_config_default() {
        let cfg = TemporalDenoiseConfig::default();
        assert_eq!(cfg.window_size, 5);
        assert!(!cfg.motion_compensated);
    }

    #[test]
    fn test_pipeline_passthrough() {
        let mut pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Passthrough);
        let frame = make_frame(8, 8, 128.0);
        let out = pipeline.process_frame(&frame);
        assert!((out.data[0] - 128.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pipeline_realtime_spatial_blur() {
        let mut pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Realtime);
        // Frame with a hot pixel in the center
        let w = 9;
        let h = 9;
        let mut data = vec![0.0f32; w * h];
        data[4 * w + 4] = 255.0;
        let frame = LumaFrame {
            data,
            width: w,
            height: h,
        };
        let out = pipeline.process_frame(&frame);
        // Hot pixel should be attenuated after blur
        assert!(out.data[4 * w + 4] < 255.0);
    }

    #[test]
    fn test_pipeline_buffer_grows() {
        let mut pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Balanced);
        let frame = make_frame(4, 4, 50.0);
        pipeline.process_frame(&frame);
        assert_eq!(pipeline.buffer_len(), 1);
        pipeline.process_frame(&frame);
        assert_eq!(pipeline.buffer_len(), 2);
    }

    #[test]
    fn test_pipeline_buffer_capped_at_window_plus_one() {
        let temporal = TemporalDenoiseConfig {
            window_size: 3,
            ..Default::default()
        };
        let mut pipeline = VideoDenoisePipeline::new(
            VideoDenoiseMode::Balanced,
            SpatialDenoiseConfig::default(),
            temporal,
        );
        let frame = make_frame(4, 4, 10.0);
        for _ in 0..10 {
            pipeline.process_frame(&frame);
        }
        // Buffer is capped at window_size + 1 = 4
        assert!(pipeline.buffer_len() <= 4);
    }

    #[test]
    fn test_pipeline_reset_clears_buffer() {
        let mut pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Balanced);
        let frame = make_frame(4, 4, 0.0);
        pipeline.process_frame(&frame);
        pipeline.reset();
        assert_eq!(pipeline.buffer_len(), 0);
    }

    #[test]
    fn test_pipeline_temporal_blend() {
        let mut pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Balanced);
        let f1 = make_frame(4, 4, 100.0);
        let f2 = make_frame(4, 4, 0.0);
        pipeline.process_frame(&f1);
        let out = pipeline.process_frame(&f2);
        // Output should be between 0 and 100
        assert!(out.data[0] > 0.0 && out.data[0] < 100.0);
    }

    #[test]
    fn test_pipeline_quality_mode_uses_motion_compensation() {
        let pipeline = VideoDenoisePipeline::from_mode(VideoDenoiseMode::Quality);
        assert!(pipeline.temporal.motion_compensated);
        assert_eq!(pipeline.temporal.window_size, 7);
    }
}
