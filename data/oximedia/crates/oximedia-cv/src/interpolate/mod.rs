//! Video frame interpolation using optical flow.
//!
//! This module provides advanced frame interpolation capabilities using optical flow
//! algorithms for motion-compensated interpolation. It supports:
//!
//! - **Optical flow methods**: Dense (Farneback), Sparse (Lucas-Kanade), Block matching
//! - **Bidirectional flow**: Forward and backward motion estimation
//! - **Occlusion handling**: Detection and compensation for occluded regions
//! - **Advanced blending**: Multiple strategies including motion-based adaptive blending
//! - **Multi-scale processing**: Pyramid-based approach for better accuracy
//! - **Quality modes**: Fast, balanced, and high-quality interpolation
//!
//! # Example
//!
//! ```
//! use oximedia_cv::interpolate::{FrameInterpolator, InterpolationQuality};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let mut interpolator = FrameInterpolator::new(InterpolationQuality::Balanced);
//!
//! let frame1 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! let frame2 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//!
//! // Interpolate at t=0.5 (midpoint between frames)
//! // let result = interpolator.interpolate_between(&frame1, &frame2, 0.5);
//! ```

pub mod artifact;
pub mod blend;
pub mod occlusion;
pub mod optical_flow;
pub mod quality;
pub mod warp;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

pub use artifact::{ArtifactReducer, ArtifactReductionConfig, ArtifactType};
pub use blend::{BlendMode, Blender};
pub use occlusion::{OcclusionDetector, OcclusionMap};
pub use optical_flow::{FlowEstimator, FlowMethod, FlowPyramid};
pub use quality::{AdaptiveParameterTuner, InterpolationQualityMetrics, QualityAssessor};
pub use warp::{WarpMode, Warper};

/// Interpolation quality modes.
///
/// Different quality modes trade off between speed and accuracy:
/// - **Fast**: Block matching, suitable for real-time applications
/// - **Balanced**: Sparse optical flow with occlusion handling
/// - **HighQuality**: Dense optical flow with multi-scale pyramid and advanced blending
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationQuality {
    /// Fast mode using block matching.
    Fast,
    /// Balanced mode using sparse optical flow.
    #[default]
    Balanced,
    /// High quality mode using dense optical flow.
    HighQuality,
}

/// Frame rate conversion configuration.
///
/// Specifies the source and target frame rates for conversion.
#[derive(Debug, Clone, Copy)]
pub struct FrameRateConfig {
    /// Source frame rate (e.g., 24.0).
    pub source_fps: f64,
    /// Target frame rate (e.g., 60.0).
    pub target_fps: f64,
}

impl FrameRateConfig {
    /// Create a new frame rate configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::interpolate::FrameRateConfig;
    ///
    /// // Convert from 24fps to 60fps
    /// let config = FrameRateConfig::new(24.0, 60.0);
    /// ```
    #[must_use]
    pub const fn new(source_fps: f64, target_fps: f64) -> Self {
        Self {
            source_fps,
            target_fps,
        }
    }

    /// Calculate the interpolation ratio.
    ///
    /// This returns how many target frames are needed per source frame.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.target_fps / self.source_fps
    }
}

/// Configuration for the frame interpolator.
#[derive(Debug, Clone)]
pub struct InterpolatorConfig {
    /// Quality mode.
    pub quality: InterpolationQuality,
    /// Enable occlusion detection.
    pub detect_occlusions: bool,
    /// Enable artifact reduction.
    pub reduce_artifacts: bool,
    /// Use multi-scale pyramid.
    pub use_pyramid: bool,
    /// Maximum pyramid levels.
    pub pyramid_levels: u32,
    /// Block size for block matching (Fast mode only).
    pub block_size: u32,
    /// Search range for block matching.
    pub search_range: i32,
    /// Blending mode.
    pub blend_mode: BlendMode,
}

impl Default for InterpolatorConfig {
    fn default() -> Self {
        Self {
            quality: InterpolationQuality::Balanced,
            detect_occlusions: true,
            reduce_artifacts: true,
            use_pyramid: true,
            pyramid_levels: 3,
            block_size: 16,
            search_range: 16,
            blend_mode: BlendMode::Adaptive,
        }
    }
}

impl InterpolatorConfig {
    /// Create a configuration for the given quality mode.
    #[must_use]
    pub fn for_quality(quality: InterpolationQuality) -> Self {
        match quality {
            InterpolationQuality::Fast => Self {
                quality,
                detect_occlusions: false,
                reduce_artifacts: false,
                use_pyramid: false,
                blend_mode: BlendMode::Linear,
                ..Self::default()
            },
            InterpolationQuality::Balanced => Self {
                quality,
                detect_occlusions: true,
                reduce_artifacts: true,
                use_pyramid: true,
                pyramid_levels: 2,
                blend_mode: BlendMode::Adaptive,
                ..Self::default()
            },
            InterpolationQuality::HighQuality => Self {
                quality,
                detect_occlusions: true,
                reduce_artifacts: true,
                use_pyramid: true,
                pyramid_levels: 3,
                blend_mode: BlendMode::MotionWeighted,
                ..Self::default()
            },
        }
    }
}

/// Video frame interpolator using optical flow.
///
/// This interpolator uses optical flow to estimate motion between frames and
/// generates intermediate frames at arbitrary time positions.
///
/// # Examples
///
/// ```
/// use oximedia_cv::interpolate::{FrameInterpolator, InterpolationQuality};
///
/// let interpolator = FrameInterpolator::new(InterpolationQuality::HighQuality);
/// ```
pub struct FrameInterpolator {
    /// Configuration.
    config: InterpolatorConfig,
    /// Flow estimator.
    flow_estimator: FlowEstimator,
    /// Warper for forward/backward warping.
    warper: Warper,
    /// Blender for combining warped frames.
    blender: Blender,
    /// Occlusion detector.
    occlusion_detector: OcclusionDetector,
}

impl FrameInterpolator {
    /// Create a new frame interpolator with the specified quality.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::interpolate::{FrameInterpolator, InterpolationQuality};
    ///
    /// let interpolator = FrameInterpolator::new(InterpolationQuality::Balanced);
    /// ```
    #[must_use]
    pub fn new(quality: InterpolationQuality) -> Self {
        let config = InterpolatorConfig::for_quality(quality);
        Self::with_config(config)
    }

    /// Create a new frame interpolator with custom configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::interpolate::{FrameInterpolator, InterpolatorConfig};
    ///
    /// let config = InterpolatorConfig::default();
    /// let interpolator = FrameInterpolator::with_config(config);
    /// ```
    #[must_use]
    pub fn with_config(config: InterpolatorConfig) -> Self {
        let flow_method = match config.quality {
            InterpolationQuality::Fast => FlowMethod::BlockMatching,
            InterpolationQuality::Balanced => FlowMethod::LucasKanade,
            InterpolationQuality::HighQuality => FlowMethod::Farneback,
        };

        let mut flow_estimator = FlowEstimator::new(flow_method);
        if config.use_pyramid {
            flow_estimator = flow_estimator.with_pyramid_levels(config.pyramid_levels);
        }
        if config.quality == InterpolationQuality::Fast {
            flow_estimator = flow_estimator
                .with_block_size(config.block_size)
                .with_search_range(config.search_range);
        }

        let warper = Warper::new(WarpMode::Bilinear);
        let blender = Blender::new(config.blend_mode);
        let occlusion_detector = OcclusionDetector::new();

        Self {
            config,
            flow_estimator,
            warper,
            blender,
            occlusion_detector,
        }
    }

    /// Interpolate a frame between two input frames at time position t.
    ///
    /// # Arguments
    ///
    /// * `frame1` - First frame (at t=0.0)
    /// * `frame2` - Second frame (at t=1.0)
    /// * `t` - Interpolation position (0.0 to 1.0, where 0.5 is the midpoint)
    ///
    /// # Returns
    ///
    /// Interpolated frame at the specified time position.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frames have different dimensions or formats
    /// - `t` is outside the range [0.0, 1.0]
    /// - Optical flow computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::interpolate::{FrameInterpolator, InterpolationQuality};
    /// use oximedia_codec::VideoFrame;
    /// use oximedia_core::PixelFormat;
    ///
    /// let mut interpolator = FrameInterpolator::new(InterpolationQuality::Balanced);
    ///
    /// let frame1 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
    /// let frame2 = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
    ///
    /// // Create midpoint frame
    /// // let result = interpolator.interpolate_between(&frame1, &frame2, 0.5);
    /// ```
    pub fn interpolate_between(
        &mut self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
        t: f32,
    ) -> CvResult<VideoFrame> {
        // Validate inputs
        self.validate_frames(frame1, frame2)?;
        self.validate_time_parameter(t)?;

        // Special cases: return exact frame at boundaries
        if t <= 0.0 {
            return Ok(frame1.clone());
        }
        if t >= 1.0 {
            return Ok(frame2.clone());
        }

        // Estimate bidirectional optical flow
        let (flow_forward, flow_backward) =
            self.flow_estimator.estimate_bidirectional(frame1, frame2)?;

        // Detect occlusions if enabled
        let occlusion_map = if self.config.detect_occlusions {
            Some(self.occlusion_detector.detect(
                &flow_forward,
                &flow_backward,
                frame1.width,
                frame1.height,
            )?)
        } else {
            None
        };

        // Warp both frames to the target time position
        let warped_from_1 = self.warper.warp_forward(frame1, &flow_forward, t)?;
        let warped_from_2 = self.warper.warp_backward(frame2, &flow_backward, 1.0 - t)?;

        // Blend the warped frames
        let result = self.blender.blend(
            &warped_from_1,
            &warped_from_2,
            t,
            &flow_forward,
            &flow_backward,
            occlusion_map.as_ref(),
        )?;

        // Apply artifact reduction if enabled
        if self.config.reduce_artifacts {
            self.reduce_artifacts(&result)
        } else {
            Ok(result)
        }
    }

    /// Interpolate multiple frames for frame rate conversion.
    ///
    /// # Arguments
    ///
    /// * `frame1` - First frame
    /// * `frame2` - Second frame
    /// * `config` - Frame rate conversion configuration
    ///
    /// # Returns
    ///
    /// Vector of interpolated frames between the two input frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frame interpolation fails.
    pub fn interpolate_for_rate_conversion(
        &mut self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
        config: &FrameRateConfig,
    ) -> CvResult<Vec<VideoFrame>> {
        let ratio = config.ratio();
        let num_frames = ratio.floor() as usize;

        if num_frames <= 1 {
            return Ok(vec![frame1.clone()]);
        }

        let mut result = Vec::with_capacity(num_frames);
        result.push(frame1.clone());

        for i in 1..num_frames {
            let t = i as f32 / num_frames as f32;
            let interpolated = self.interpolate_between(frame1, frame2, t)?;
            result.push(interpolated);
        }

        Ok(result)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &InterpolatorConfig {
        &self.config
    }

    /// Set the quality mode.
    pub fn set_quality(&mut self, quality: InterpolationQuality) {
        self.config.quality = quality;
        let new_config = InterpolatorConfig::for_quality(quality);
        *self = Self::with_config(new_config);
    }

    /// Enable or disable occlusion detection.
    pub fn set_occlusion_detection(&mut self, enabled: bool) {
        self.config.detect_occlusions = enabled;
    }

    /// Enable or disable artifact reduction.
    pub fn set_artifact_reduction(&mut self, enabled: bool) {
        self.config.reduce_artifacts = enabled;
    }

    /// Set the blending mode.
    pub fn set_blend_mode(&mut self, mode: BlendMode) {
        self.config.blend_mode = mode;
        self.blender = Blender::new(mode);
    }

    /// Validate that two frames are compatible for interpolation.
    fn validate_frames(&self, frame1: &VideoFrame, frame2: &VideoFrame) -> CvResult<()> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(CvError::invalid_dimensions(frame1.width, frame1.height));
        }

        if frame1.format != frame2.format {
            return Err(CvError::unsupported_format(format!(
                "Frame format mismatch: {:?} vs {:?}",
                frame1.format, frame2.format
            )));
        }

        if frame1.width == 0 || frame1.height == 0 {
            return Err(CvError::invalid_dimensions(frame1.width, frame1.height));
        }

        Ok(())
    }

    /// Validate the time parameter.
    fn validate_time_parameter(&self, t: f32) -> CvResult<()> {
        if !(0.0..=1.0).contains(&t) {
            return Err(CvError::invalid_parameter(
                "t",
                format!("{t} (must be in range [0.0, 1.0])"),
            ));
        }
        Ok(())
    }

    /// Apply artifact reduction to the interpolated frame.
    ///
    /// This reduces common interpolation artifacts such as halos and ghosting.
    fn reduce_artifacts(&self, frame: &VideoFrame) -> CvResult<VideoFrame> {
        let mut config = ArtifactReductionConfig::default();

        match self.config.quality {
            InterpolationQuality::Fast => {
                config.reduce_halo = false;
                config.reduce_ghosting = false;
                config.sharpen = false;
            }
            InterpolationQuality::Balanced => {
                config.halo_strength = 0.3;
                config.ghosting_strength = 0.2;
                config.sharpen_strength = 0.1;
            }
            InterpolationQuality::HighQuality => {
                config.halo_strength = 0.5;
                config.ghosting_strength = 0.3;
                config.sharpen_strength = 0.2;
            }
        }

        let reducer = ArtifactReducer::with_config(config);
        reducer.reduce_artifacts(frame, None, None)
    }
}

impl Default for FrameInterpolator {
    fn default() -> Self {
        Self::new(InterpolationQuality::Balanced)
    }
}
