//! Video stabilization with motion smoothing.
//!
//! This module provides comprehensive video stabilization algorithms including:
//!
//! - [`motion`]: Motion estimation (feature tracking, homography, transformations)
//! - [`smooth`]: Motion smoothing filters (Gaussian, low-pass, adaptive)
//! - [`transform`]: Frame warping and transformation
//! - [`rolling_shutter`]: Rolling shutter correction
//!
//! # Example
//!
//! ```
//! use oximedia_cv::stabilize::{VideoStabilizer, StabilizationConfig};
//!
//! let config = StabilizationConfig::default();
//! // let stabilized = VideoStabilizer::stabilize(&frames, config);
//! ```

pub mod motion;
pub mod rolling_shutter;
pub mod smooth;
pub mod transform;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

pub use motion::{
    GyroscopeData, GyroscopeFusionConfig, HomographyEstimator, HybridMotionEstimator,
    MotionEstimator, TransformMatrix,
};
pub use rolling_shutter::RollingShutterCorrector;
pub use smooth::{AdaptiveSmoother, GaussianSmoother, LowPassFilter, MotionSmoother};
pub use transform::{BorderMode, FrameWarper};

/// Stabilization configuration.
///
/// Controls the behavior of video stabilization algorithms.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::StabilizationConfig;
///
/// let config = StabilizationConfig {
///     smoothing_strength: 0.8,
///     crop_ratio: 0.9,
///     enable_rolling_shutter: false,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StabilizationConfig {
    /// Smoothing strength (0.0 = no smoothing, 1.0 = maximum smoothing).
    pub smoothing_strength: f64,
    /// Crop ratio to avoid black borders (0.0-1.0).
    pub crop_ratio: f64,
    /// Enable rolling shutter correction.
    pub enable_rolling_shutter: bool,
    /// Enable 3D stabilization (rotation + translation).
    pub enable_3d_stabilization: bool,
    /// Border handling mode.
    pub border_mode: BorderMode,
    /// Maximum motion magnitude for adaptive smoothing.
    pub max_motion_magnitude: f64,
    /// Window size for motion smoothing.
    pub smoothing_window: usize,
    /// Enable zoom/crop stabilization mode.
    pub enable_zoom_crop: bool,
}

impl Default for StabilizationConfig {
    fn default() -> Self {
        Self {
            smoothing_strength: 0.7,
            crop_ratio: 0.95,
            enable_rolling_shutter: false,
            enable_3d_stabilization: true,
            border_mode: BorderMode::Replicate,
            max_motion_magnitude: 50.0,
            smoothing_window: 15,
            enable_zoom_crop: false,
        }
    }
}

impl StabilizationConfig {
    /// Create a new stabilization configuration with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new();
    /// assert_eq!(config.smoothing_strength, 0.7);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set smoothing strength.
    ///
    /// # Arguments
    ///
    /// * `strength` - Smoothing strength (0.0-1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_smoothing_strength(0.9);
    /// assert_eq!(config.smoothing_strength, 0.9);
    /// ```
    #[must_use]
    pub fn with_smoothing_strength(mut self, strength: f64) -> Self {
        self.smoothing_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set crop ratio.
    ///
    /// # Arguments
    ///
    /// * `ratio` - Crop ratio (0.0-1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_crop_ratio(0.85);
    /// assert_eq!(config.crop_ratio, 0.85);
    /// ```
    #[must_use]
    pub fn with_crop_ratio(mut self, ratio: f64) -> Self {
        self.crop_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable rolling shutter correction.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_rolling_shutter(true);
    /// assert!(config.enable_rolling_shutter);
    /// ```
    #[must_use]
    pub const fn with_rolling_shutter(mut self, enable: bool) -> Self {
        self.enable_rolling_shutter = enable;
        self
    }

    /// Enable or disable 3D stabilization.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_3d_stabilization(false);
    /// assert!(!config.enable_3d_stabilization);
    /// ```
    #[must_use]
    pub const fn with_3d_stabilization(mut self, enable: bool) -> Self {
        self.enable_3d_stabilization = enable;
        self
    }

    /// Set border mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::{StabilizationConfig, BorderMode};
    ///
    /// let config = StabilizationConfig::new().with_border_mode(BorderMode::Reflect);
    /// ```
    #[must_use]
    pub const fn with_border_mode(mut self, mode: BorderMode) -> Self {
        self.border_mode = mode;
        self
    }

    /// Set maximum motion magnitude for adaptive smoothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_max_motion_magnitude(100.0);
    /// assert_eq!(config.max_motion_magnitude, 100.0);
    /// ```
    #[must_use]
    pub fn with_max_motion_magnitude(mut self, magnitude: f64) -> Self {
        self.max_motion_magnitude = magnitude.max(0.0);
        self
    }

    /// Set smoothing window size.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_smoothing_window(21);
    /// assert_eq!(config.smoothing_window, 21);
    /// ```
    #[must_use]
    pub const fn with_smoothing_window(mut self, window: usize) -> Self {
        self.smoothing_window = window;
        self
    }

    /// Enable or disable zoom/crop stabilization mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::StabilizationConfig;
    ///
    /// let config = StabilizationConfig::new().with_zoom_crop(true);
    /// assert!(config.enable_zoom_crop);
    /// ```
    #[must_use]
    pub const fn with_zoom_crop(mut self, enable: bool) -> Self {
        self.enable_zoom_crop = enable;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> CvResult<()> {
        if !(0.0..=1.0).contains(&self.smoothing_strength) {
            return Err(CvError::invalid_parameter(
                "smoothing_strength",
                format!("{}", self.smoothing_strength),
            ));
        }
        if !(0.0..=1.0).contains(&self.crop_ratio) {
            return Err(CvError::invalid_parameter(
                "crop_ratio",
                format!("{}", self.crop_ratio),
            ));
        }
        if self.max_motion_magnitude <= 0.0 {
            return Err(CvError::invalid_parameter(
                "max_motion_magnitude",
                format!("{}", self.max_motion_magnitude),
            ));
        }
        if self.smoothing_window < 3 {
            return Err(CvError::invalid_parameter(
                "smoothing_window",
                format!("{}", self.smoothing_window),
            ));
        }
        Ok(())
    }
}

/// Video stabilizer.
///
/// Main API for stabilizing video sequences with motion smoothing.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::{VideoStabilizer, StabilizationConfig};
///
/// let stabilizer = VideoStabilizer::new();
/// let config = StabilizationConfig::default();
/// // let stabilized = stabilizer.stabilize(&frames, config);
/// ```
#[derive(Debug, Clone)]
pub struct VideoStabilizer {
    motion_estimator: MotionEstimator,
    gaussian_smoother: GaussianSmoother,
    low_pass_filter: LowPassFilter,
    adaptive_smoother: AdaptiveSmoother,
    frame_warper: FrameWarper,
    rolling_shutter_corrector: RollingShutterCorrector,
}

impl VideoStabilizer {
    /// Create a new video stabilizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::VideoStabilizer;
    ///
    /// let stabilizer = VideoStabilizer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            motion_estimator: MotionEstimator::new(),
            gaussian_smoother: GaussianSmoother::new(15, 3.0),
            low_pass_filter: LowPassFilter::new(0.3),
            adaptive_smoother: AdaptiveSmoother::new(15, 50.0),
            frame_warper: FrameWarper::new(),
            rolling_shutter_corrector: RollingShutterCorrector::new(),
        }
    }

    /// Stabilize a sequence of video frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Input video frames
    /// * `config` - Stabilization configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration is invalid
    /// - Frames are empty
    /// - Motion estimation fails
    /// - Frame warping fails
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::{VideoStabilizer, StabilizationConfig};
    ///
    /// let stabilizer = VideoStabilizer::new();
    /// let config = StabilizationConfig::default();
    /// // let stabilized = stabilizer.stabilize(&frames, config)?;
    /// ```
    pub fn stabilize(
        &mut self,
        frames: &[VideoFrame],
        config: StabilizationConfig,
    ) -> CvResult<Vec<VideoFrame>> {
        // Validate configuration
        config.validate()?;

        // Check if frames are empty
        if frames.is_empty() {
            return Err(CvError::invalid_parameter(
                "frames",
                "empty frame sequence".to_string(),
            ));
        }

        // Step 1: Estimate inter-frame transformations
        let transformations = self.estimate_motion(frames)?;

        // Step 2: Apply rolling shutter correction if enabled
        let transformations = if config.enable_rolling_shutter {
            self.rolling_shutter_corrector
                .correct_transformations(&transformations)?
        } else {
            transformations
        };

        // Step 3: Smooth the motion trajectories
        let smoothed_transformations = self.smooth_motion(&transformations, &config)?;

        // Step 4: Compute stabilizing transformations
        let stabilizing_transforms =
            self.compute_stabilizing_transforms(&transformations, &smoothed_transformations)?;

        // Step 5: Warp frames with stabilizing transformations
        let stabilized_frames = self.warp_frames(frames, &stabilizing_transforms, &config)?;

        Ok(stabilized_frames)
    }

    /// Estimate motion between consecutive frames.
    fn estimate_motion(&mut self, frames: &[VideoFrame]) -> CvResult<Vec<TransformMatrix>> {
        let mut transformations = Vec::with_capacity(frames.len());

        // First frame has identity transformation
        transformations.push(TransformMatrix::identity());

        // Estimate transformation for each consecutive frame pair
        for i in 1..frames.len() {
            let prev_frame = &frames[i - 1];
            let curr_frame = &frames[i];

            let transform = self
                .motion_estimator
                .estimate_transform(prev_frame, curr_frame)?;
            transformations.push(transform);
        }

        Ok(transformations)
    }

    /// Smooth motion trajectories.
    fn smooth_motion(
        &mut self,
        transformations: &[TransformMatrix],
        config: &StabilizationConfig,
    ) -> CvResult<Vec<TransformMatrix>> {
        // Extract motion parameters from transformations
        let motion_params = self.extract_motion_parameters(transformations);

        // Apply smoothing based on configuration
        let smoothed_params = if config.enable_3d_stabilization {
            // Use adaptive smoothing for 3D stabilization
            self.adaptive_smoother
                .smooth(&motion_params, config.max_motion_magnitude)?
        } else {
            // Use Gaussian smoothing for 2D stabilization
            let gaussian_smoothed = self.gaussian_smoother.smooth(&motion_params)?;

            // Apply low-pass filter
            self.low_pass_filter.smooth(&gaussian_smoothed)?
        };

        // Reconstruct transformation matrices from smoothed parameters
        self.reconstruct_transformations(&smoothed_params)
    }

    /// Extract motion parameters from transformation matrices.
    fn extract_motion_parameters(&self, transformations: &[TransformMatrix]) -> MotionParameters {
        let mut dx = Vec::with_capacity(transformations.len());
        let mut dy = Vec::with_capacity(transformations.len());
        let mut da = Vec::with_capacity(transformations.len());
        let mut ds = Vec::with_capacity(transformations.len());

        for transform in transformations {
            dx.push(transform.tx);
            dy.push(transform.ty);
            da.push(transform.angle);
            ds.push(transform.scale);
        }

        MotionParameters { dx, dy, da, ds }
    }

    /// Reconstruct transformation matrices from motion parameters.
    fn reconstruct_transformations(
        &self,
        params: &MotionParameters,
    ) -> CvResult<Vec<TransformMatrix>> {
        let len = params.dx.len();
        let mut transformations = Vec::with_capacity(len);

        for i in 0..len {
            let transform = TransformMatrix {
                tx: params.dx[i],
                ty: params.dy[i],
                angle: params.da[i],
                scale: params.ds[i],
            };
            transformations.push(transform);
        }

        Ok(transformations)
    }

    /// Compute stabilizing transformations.
    fn compute_stabilizing_transforms(
        &self,
        original: &[TransformMatrix],
        smoothed: &[TransformMatrix],
    ) -> CvResult<Vec<TransformMatrix>> {
        if original.len() != smoothed.len() {
            return Err(CvError::matrix_error(
                "Original and smoothed transformation counts do not match",
            ));
        }

        let mut stabilizing = Vec::with_capacity(original.len());

        for (orig, smooth) in original.iter().zip(smoothed.iter()) {
            // Compute the difference transformation
            let stabilize = TransformMatrix {
                tx: smooth.tx - orig.tx,
                ty: smooth.ty - orig.ty,
                angle: smooth.angle - orig.angle,
                scale: smooth.scale / orig.scale,
            };
            stabilizing.push(stabilize);
        }

        Ok(stabilizing)
    }

    /// Warp frames with stabilizing transformations.
    fn warp_frames(
        &mut self,
        frames: &[VideoFrame],
        transforms: &[TransformMatrix],
        config: &StabilizationConfig,
    ) -> CvResult<Vec<VideoFrame>> {
        if frames.len() != transforms.len() {
            return Err(CvError::matrix_error(
                "Frame and transformation counts do not match",
            ));
        }

        let mut stabilized = Vec::with_capacity(frames.len());

        for (frame, transform) in frames.iter().zip(transforms.iter()) {
            let warped =
                self.frame_warper
                    .warp(frame, transform, config.border_mode, config.crop_ratio)?;
            stabilized.push(warped);
        }

        Ok(stabilized)
    }
}

impl Default for VideoStabilizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Motion parameters for smoothing.
#[derive(Debug, Clone)]
pub struct MotionParameters {
    /// Translation in X direction.
    pub(crate) dx: Vec<f64>,
    /// Translation in Y direction.
    pub(crate) dy: Vec<f64>,
    /// Rotation angle.
    pub(crate) da: Vec<f64>,
    /// Scale factor.
    pub(crate) ds: Vec<f64>,
}

/// Stabilization statistics.
///
/// Provides information about the stabilization process.
#[derive(Debug, Clone, Copy, Default)]
pub struct StabilizationStats {
    /// Average motion magnitude before stabilization.
    pub avg_motion_before: f64,
    /// Average motion magnitude after stabilization.
    pub avg_motion_after: f64,
    /// Maximum motion magnitude before stabilization.
    pub max_motion_before: f64,
    /// Maximum motion magnitude after stabilization.
    pub max_motion_after: f64,
    /// Stabilization improvement ratio.
    pub improvement_ratio: f64,
    /// Number of frames processed.
    pub frame_count: usize,
}

impl StabilizationStats {
    /// Create new stabilization statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            avg_motion_before: 0.0,
            avg_motion_after: 0.0,
            max_motion_before: 0.0,
            max_motion_after: 0.0,
            improvement_ratio: 0.0,
            frame_count: 0,
        }
    }

    /// Compute statistics from motion data.
    #[must_use]
    pub fn compute(before: &[TransformMatrix], after: &[TransformMatrix]) -> Self {
        if before.is_empty() || after.is_empty() {
            return Self::new();
        }

        let avg_before = Self::compute_average_magnitude(before);
        let avg_after = Self::compute_average_magnitude(after);
        let max_before = Self::compute_max_magnitude(before);
        let max_after = Self::compute_max_magnitude(after);

        let improvement = if avg_before > 0.0 {
            (avg_before - avg_after) / avg_before
        } else {
            0.0
        };

        Self {
            avg_motion_before: avg_before,
            avg_motion_after: avg_after,
            max_motion_before: max_before,
            max_motion_after: max_after,
            improvement_ratio: improvement,
            frame_count: before.len(),
        }
    }

    /// Compute average motion magnitude.
    fn compute_average_magnitude(transforms: &[TransformMatrix]) -> f64 {
        if transforms.is_empty() {
            return 0.0;
        }

        let sum: f64 = transforms
            .iter()
            .map(motion::TransformMatrix::magnitude)
            .sum();
        sum / transforms.len() as f64
    }

    /// Compute maximum motion magnitude.
    fn compute_max_magnitude(transforms: &[TransformMatrix]) -> f64 {
        transforms
            .iter()
            .map(motion::TransformMatrix::magnitude)
            .fold(0.0, f64::max)
    }
}
