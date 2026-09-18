//! Rolling shutter correction for video stabilization.
//!
//! This module provides algorithms for correcting rolling shutter artifacts:
//!
//! - Row-wise transformation estimation
//! - Scanline-based motion model
//! - Temporal interpolation for correction

use crate::error::{CvError, CvResult};
use crate::stabilize::motion::TransformMatrix;
use std::f64::consts::PI;

/// Rolling shutter correction parameters.
#[derive(Debug, Clone, Copy)]
pub struct RollingShutterParams {
    /// Readout time (time to read the entire frame).
    pub readout_time: f64,
    /// Scan direction (true = top-to-bottom, false = bottom-to-top).
    pub top_to_bottom: bool,
    /// Number of scanline segments.
    pub num_segments: usize,
}

impl Default for RollingShutterParams {
    fn default() -> Self {
        Self {
            readout_time: 0.033, // ~30fps
            top_to_bottom: true,
            num_segments: 10,
        }
    }
}

impl RollingShutterParams {
    /// Create new rolling shutter parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::rolling_shutter::RollingShutterParams;
    ///
    /// let params = RollingShutterParams::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set readout time.
    #[must_use]
    pub const fn with_readout_time(mut self, readout_time: f64) -> Self {
        self.readout_time = readout_time;
        self
    }

    /// Set scan direction.
    #[must_use]
    pub const fn with_scan_direction(mut self, top_to_bottom: bool) -> Self {
        self.top_to_bottom = top_to_bottom;
        self
    }

    /// Set number of segments.
    #[must_use]
    pub const fn with_num_segments(mut self, num_segments: usize) -> Self {
        self.num_segments = num_segments;
        self
    }
}

/// Rolling shutter corrector.
///
/// Corrects rolling shutter artifacts in video frames.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::RollingShutterCorrector;
///
/// let corrector = RollingShutterCorrector::new();
/// ```
#[derive(Debug, Clone)]
pub struct RollingShutterCorrector {
    /// Rolling shutter parameters.
    params: RollingShutterParams,
    /// Motion model for scanline transformations.
    motion_model: MotionModel,
}

impl RollingShutterCorrector {
    /// Create a new rolling shutter corrector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::RollingShutterCorrector;
    ///
    /// let corrector = RollingShutterCorrector::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: RollingShutterParams::default(),
            motion_model: MotionModel::Linear,
        }
    }

    /// Create with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Rolling shutter parameters
    #[must_use]
    pub const fn with_params(mut self, params: RollingShutterParams) -> Self {
        self.params = params;
        self
    }

    /// Set motion model.
    #[must_use]
    pub const fn with_motion_model(mut self, model: MotionModel) -> Self {
        self.motion_model = model;
        self
    }

    /// Correct rolling shutter in transformations.
    ///
    /// # Arguments
    ///
    /// * `transformations` - Input frame-to-frame transformations
    ///
    /// # Errors
    ///
    /// Returns an error if correction fails.
    pub fn correct_transformations(
        &self,
        transformations: &[TransformMatrix],
    ) -> CvResult<Vec<TransformMatrix>> {
        if transformations.is_empty() {
            return Ok(Vec::new());
        }

        let mut corrected = Vec::with_capacity(transformations.len());

        for i in 0..transformations.len() {
            let transform = &transformations[i];

            // Compute scanline transformations
            let scanline_transforms =
                self.compute_scanline_transforms(transform, i, transformations)?;

            // Average the scanline transformations to get a global correction
            let averaged = self.average_scanline_transforms(&scanline_transforms);
            corrected.push(averaged);
        }

        Ok(corrected)
    }

    /// Compute per-scanline transformations.
    fn compute_scanline_transforms(
        &self,
        base_transform: &TransformMatrix,
        frame_idx: usize,
        all_transforms: &[TransformMatrix],
    ) -> CvResult<Vec<TransformMatrix>> {
        let num_segments = self.params.num_segments;
        let mut scanline_transforms = Vec::with_capacity(num_segments);

        for segment_idx in 0..num_segments {
            // Compute time offset for this scanline segment
            let time_offset = if self.params.top_to_bottom {
                segment_idx as f64 / num_segments as f64
            } else {
                1.0 - segment_idx as f64 / num_segments as f64
            };

            // Interpolate transformation based on motion model
            let transform = match self.motion_model {
                MotionModel::Linear => self.linear_interpolation(
                    base_transform,
                    frame_idx,
                    all_transforms,
                    time_offset,
                ),
                MotionModel::Quadratic => self.quadratic_interpolation(
                    base_transform,
                    frame_idx,
                    all_transforms,
                    time_offset,
                ),
                MotionModel::Cubic => {
                    self.cubic_interpolation(base_transform, frame_idx, all_transforms, time_offset)
                }
            };

            scanline_transforms.push(transform);
        }

        Ok(scanline_transforms)
    }

    /// Linear interpolation between frames.
    fn linear_interpolation(
        &self,
        base_transform: &TransformMatrix,
        frame_idx: usize,
        all_transforms: &[TransformMatrix],
        t: f64,
    ) -> TransformMatrix {
        if frame_idx + 1 < all_transforms.len() {
            let next_transform = &all_transforms[frame_idx + 1];
            base_transform.interpolate(next_transform, t)
        } else {
            *base_transform
        }
    }

    /// Quadratic interpolation using three frames.
    fn quadratic_interpolation(
        &self,
        base_transform: &TransformMatrix,
        frame_idx: usize,
        all_transforms: &[TransformMatrix],
        t: f64,
    ) -> TransformMatrix {
        if frame_idx + 2 < all_transforms.len() {
            let t0 = base_transform;
            let t1 = &all_transforms[frame_idx + 1];
            let t2 = &all_transforms[frame_idx + 2];

            // Quadratic Bezier interpolation
            let one_minus_t = 1.0 - t;
            let a = one_minus_t * one_minus_t;
            let b = 2.0 * one_minus_t * t;
            let c = t * t;

            TransformMatrix {
                tx: a * t0.tx + b * t1.tx + c * t2.tx,
                ty: a * t0.ty + b * t1.ty + c * t2.ty,
                angle: a * t0.angle + b * t1.angle + c * t2.angle,
                scale: a * t0.scale + b * t1.scale + c * t2.scale,
            }
        } else {
            self.linear_interpolation(base_transform, frame_idx, all_transforms, t)
        }
    }

    /// Cubic interpolation using four frames.
    fn cubic_interpolation(
        &self,
        base_transform: &TransformMatrix,
        frame_idx: usize,
        all_transforms: &[TransformMatrix],
        t: f64,
    ) -> TransformMatrix {
        if frame_idx + 3 < all_transforms.len() {
            let t0 = base_transform;
            let t1 = &all_transforms[frame_idx + 1];
            let t2 = &all_transforms[frame_idx + 2];
            let t3 = &all_transforms[frame_idx + 3];

            // Cubic Bezier interpolation
            let one_minus_t = 1.0 - t;
            let a = one_minus_t * one_minus_t * one_minus_t;
            let b = 3.0 * one_minus_t * one_minus_t * t;
            let c = 3.0 * one_minus_t * t * t;
            let d = t * t * t;

            TransformMatrix {
                tx: a * t0.tx + b * t1.tx + c * t2.tx + d * t3.tx,
                ty: a * t0.ty + b * t1.ty + c * t2.ty + d * t3.ty,
                angle: a * t0.angle + b * t1.angle + c * t2.angle + d * t3.angle,
                scale: a * t0.scale + b * t1.scale + c * t2.scale + d * t3.scale,
            }
        } else {
            self.quadratic_interpolation(base_transform, frame_idx, all_transforms, t)
        }
    }

    /// Average scanline transformations to get global correction.
    fn average_scanline_transforms(&self, transforms: &[TransformMatrix]) -> TransformMatrix {
        if transforms.is_empty() {
            return TransformMatrix::identity();
        }

        let mut sum_tx = 0.0;
        let mut sum_ty = 0.0;
        let mut sum_angle = 0.0;
        let mut sum_scale = 0.0;

        for transform in transforms {
            sum_tx += transform.tx;
            sum_ty += transform.ty;
            sum_angle += transform.angle;
            sum_scale += transform.scale;
        }

        let n = transforms.len() as f64;
        TransformMatrix {
            tx: sum_tx / n,
            ty: sum_ty / n,
            angle: sum_angle / n,
            scale: sum_scale / n,
        }
    }

    /// Estimate rolling shutter parameters from motion data.
    pub fn estimate_params(&mut self, transformations: &[TransformMatrix]) -> CvResult<()> {
        if transformations.is_empty() {
            return Err(CvError::invalid_parameter(
                "transformations",
                "empty sequence".to_string(),
            ));
        }

        // Analyze motion patterns to estimate rolling shutter parameters
        let motion_variance = self.compute_motion_variance(transformations);
        let dominant_direction = self.compute_dominant_direction(transformations);

        // Adjust parameters based on analysis
        if motion_variance > 0.5 {
            self.params.num_segments = 15; // More segments for high motion
        } else {
            self.params.num_segments = 10; // Fewer segments for low motion
        }

        // Determine scan direction based on motion patterns
        self.params.top_to_bottom = dominant_direction >= 0.0;

        Ok(())
    }

    /// Compute motion variance across frames.
    fn compute_motion_variance(&self, transformations: &[TransformMatrix]) -> f64 {
        if transformations.is_empty() {
            return 0.0;
        }

        let mean_tx: f64 =
            transformations.iter().map(|t| t.tx).sum::<f64>() / transformations.len() as f64;
        let mean_ty: f64 =
            transformations.iter().map(|t| t.ty).sum::<f64>() / transformations.len() as f64;

        let variance: f64 = transformations
            .iter()
            .map(|t| {
                let dx = t.tx - mean_tx;
                let dy = t.ty - mean_ty;
                dx * dx + dy * dy
            })
            .sum::<f64>()
            / transformations.len() as f64;

        variance.sqrt()
    }

    /// Compute dominant motion direction.
    fn compute_dominant_direction(&self, transformations: &[TransformMatrix]) -> f64 {
        if transformations.is_empty() {
            return 0.0;
        }

        let sum_ty: f64 = transformations.iter().map(|t| t.ty).sum();
        sum_ty / transformations.len() as f64
    }
}

impl Default for RollingShutterCorrector {
    fn default() -> Self {
        Self::new()
    }
}

/// Motion model for scanline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionModel {
    /// Linear interpolation between frames.
    Linear,
    /// Quadratic interpolation (smoother).
    Quadratic,
    /// Cubic interpolation (smoothest).
    Cubic,
}

/// Rolling shutter estimator.
///
/// Estimates rolling shutter parameters from video frames.
#[derive(Debug, Clone)]
pub struct RollingShutterEstimator {
    /// Number of frames to analyze.
    analysis_window: usize,
    /// Minimum motion threshold for detection.
    motion_threshold: f64,
}

impl RollingShutterEstimator {
    /// Create a new rolling shutter estimator.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::rolling_shutter::RollingShutterEstimator;
    ///
    /// let estimator = RollingShutterEstimator::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            analysis_window: 30,
            motion_threshold: 5.0,
        }
    }

    /// Set analysis window size.
    #[must_use]
    pub const fn with_analysis_window(mut self, window: usize) -> Self {
        self.analysis_window = window;
        self
    }

    /// Set motion threshold.
    #[must_use]
    pub const fn with_motion_threshold(mut self, threshold: f64) -> Self {
        self.motion_threshold = threshold;
        self
    }

    /// Estimate rolling shutter parameters from transformations.
    ///
    /// # Arguments
    ///
    /// * `transformations` - Frame-to-frame transformations
    ///
    /// # Errors
    ///
    /// Returns an error if estimation fails.
    pub fn estimate(&self, transformations: &[TransformMatrix]) -> CvResult<RollingShutterParams> {
        if transformations.len() < self.analysis_window {
            return Err(CvError::invalid_parameter(
                "transformations",
                format!("need at least {} frames", self.analysis_window),
            ));
        }

        // Analyze motion patterns
        let motion_stats = self.analyze_motion_patterns(transformations);

        // Detect rolling shutter artifacts
        let has_rolling_shutter = self.detect_rolling_shutter(&motion_stats);

        // Estimate parameters
        let readout_time = if has_rolling_shutter {
            self.estimate_readout_time(&motion_stats)
        } else {
            0.033 // Default ~30fps
        };

        let top_to_bottom = motion_stats.vertical_bias >= 0.0;
        let num_segments = if has_rolling_shutter { 15 } else { 10 };

        Ok(RollingShutterParams {
            readout_time,
            top_to_bottom,
            num_segments,
        })
    }

    /// Analyze motion patterns in transformations.
    fn analyze_motion_patterns(&self, transformations: &[TransformMatrix]) -> MotionStatistics {
        let mut vertical_motion = Vec::new();
        let mut horizontal_motion = Vec::new();
        let mut rotation = Vec::new();

        for transform in transformations.iter().take(self.analysis_window) {
            vertical_motion.push(transform.ty);
            horizontal_motion.push(transform.tx);
            rotation.push(transform.angle);
        }

        let vertical_variance = compute_variance(&vertical_motion);
        let horizontal_variance = compute_variance(&horizontal_motion);
        let rotation_variance = compute_variance(&rotation);

        let vertical_bias = vertical_motion.iter().sum::<f64>() / vertical_motion.len() as f64;

        MotionStatistics {
            vertical_variance,
            horizontal_variance,
            rotation_variance,
            vertical_bias,
        }
    }

    /// Detect presence of rolling shutter artifacts.
    fn detect_rolling_shutter(&self, stats: &MotionStatistics) -> bool {
        // Rolling shutter typically shows higher vertical variance
        // and specific patterns in rotation
        stats.vertical_variance > self.motion_threshold
            && stats.vertical_variance > stats.horizontal_variance * 1.5
    }

    /// Estimate readout time from motion statistics.
    fn estimate_readout_time(&self, stats: &MotionStatistics) -> f64 {
        // Simplified estimation based on motion variance
        // Real implementation would use more sophisticated analysis
        let base_time = 0.033; // ~30fps
        let variance_factor = (stats.vertical_variance / 100.0).clamp(0.5, 2.0);
        base_time * variance_factor
    }
}

impl Default for RollingShutterEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Motion statistics for rolling shutter detection.
#[derive(Debug, Clone, Copy)]
struct MotionStatistics {
    /// Variance in vertical motion.
    vertical_variance: f64,
    /// Variance in horizontal motion.
    horizontal_variance: f64,
    /// Variance in rotation.
    rotation_variance: f64,
    /// Bias in vertical direction.
    vertical_bias: f64,
}

/// Compute variance of a signal.
fn compute_variance(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / signal.len() as f64;

    variance
}

/// Scanline transformation.
///
/// Represents transformation for a single scanline in rolling shutter correction.
#[derive(Debug, Clone, Copy)]
pub struct ScanlineTransform {
    /// Scanline index (row number).
    pub scanline_idx: u32,
    /// Transformation for this scanline.
    pub transform: TransformMatrix,
    /// Confidence score.
    pub confidence: f64,
}

impl ScanlineTransform {
    /// Create a new scanline transformation.
    #[must_use]
    pub const fn new(scanline_idx: u32, transform: TransformMatrix, confidence: f64) -> Self {
        Self {
            scanline_idx,
            transform,
            confidence,
        }
    }
}

/// Rolling shutter model.
///
/// Models the rolling shutter effect for correction.
#[derive(Debug, Clone)]
pub struct RollingShutterModel {
    /// Scanline transformations.
    scanline_transforms: Vec<ScanlineTransform>,
    /// Frame height.
    frame_height: u32,
}

impl RollingShutterModel {
    /// Create a new rolling shutter model.
    ///
    /// # Arguments
    ///
    /// * `frame_height` - Height of the video frame
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::rolling_shutter::RollingShutterModel;
    ///
    /// let model = RollingShutterModel::new(1080);
    /// ```
    #[must_use]
    pub fn new(frame_height: u32) -> Self {
        Self {
            scanline_transforms: Vec::new(),
            frame_height,
        }
    }

    /// Add a scanline transformation.
    pub fn add_scanline(&mut self, transform: ScanlineTransform) {
        self.scanline_transforms.push(transform);
    }

    /// Get transformation for a specific scanline.
    #[must_use]
    pub fn get_scanline_transform(&self, scanline_idx: u32) -> Option<TransformMatrix> {
        // Find the closest scanline transformation
        self.scanline_transforms
            .iter()
            .min_by_key(|st| (st.scanline_idx as i32 - scanline_idx as i32).abs())
            .map(|st| st.transform)
    }

    /// Interpolate transformation for a scanline.
    #[must_use]
    pub fn interpolate_scanline_transform(&self, scanline_idx: u32) -> TransformMatrix {
        if self.scanline_transforms.is_empty() {
            return TransformMatrix::identity();
        }

        // Find surrounding scanlines
        let mut prev: Option<&ScanlineTransform> = None;
        let mut next: Option<&ScanlineTransform> = None;

        for st in &self.scanline_transforms {
            if st.scanline_idx <= scanline_idx {
                prev = Some(st);
            }
            if st.scanline_idx >= scanline_idx && next.is_none() {
                next = Some(st);
            }
        }

        match (prev, next) {
            (Some(p), Some(n)) if p.scanline_idx != n.scanline_idx => {
                // Interpolate between prev and next
                let t = (scanline_idx - p.scanline_idx) as f64
                    / (n.scanline_idx - p.scanline_idx) as f64;
                p.transform.interpolate(&n.transform, t)
            }
            (Some(p), _) => p.transform,
            (_, Some(n)) => n.transform,
            _ => TransformMatrix::identity(),
        }
    }

    /// Get all scanline transformations.
    #[must_use]
    pub fn scanline_transforms(&self) -> &[ScanlineTransform] {
        &self.scanline_transforms
    }
}

/// Jello effect corrector.
///
/// Corrects the "jello" or "wobble" effect caused by rolling shutter.
#[derive(Debug, Clone)]
pub struct JelloCorrector {
    /// Frequency of wobble detection.
    wobble_frequency: f64,
    /// Amplitude threshold.
    amplitude_threshold: f64,
}

impl JelloCorrector {
    /// Create a new jello corrector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::rolling_shutter::JelloCorrector;
    ///
    /// let corrector = JelloCorrector::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            wobble_frequency: 0.0,
            amplitude_threshold: 2.0,
        }
    }

    /// Detect wobble frequency from motion data.
    pub fn detect_wobble(&mut self, transformations: &[TransformMatrix]) -> f64 {
        if transformations.len() < 10 {
            return 0.0;
        }

        // Extract vertical motion
        let vertical_motion: Vec<f64> = transformations.iter().map(|t| t.ty).collect();

        // Simplified frequency detection using zero-crossings
        let frequency = self.detect_frequency_zero_crossing(&vertical_motion);
        self.wobble_frequency = frequency;
        frequency
    }

    /// Detect frequency using zero-crossing method.
    fn detect_frequency_zero_crossing(&self, signal: &[f64]) -> f64 {
        if signal.len() < 3 {
            return 0.0;
        }

        // Remove mean
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

        // Count zero crossings
        let mut crossings = 0;
        for i in 1..centered.len() {
            if (centered[i - 1] >= 0.0 && centered[i] < 0.0)
                || (centered[i - 1] < 0.0 && centered[i] >= 0.0)
            {
                crossings += 1;
            }
        }

        // Frequency is half the number of zero crossings per sample
        crossings as f64 / (2.0 * signal.len() as f64)
    }

    /// Correct jello effect in transformations.
    pub fn correct(&self, transformations: &[TransformMatrix]) -> CvResult<Vec<TransformMatrix>> {
        if self.wobble_frequency < 0.01 {
            // No significant wobble detected
            return Ok(transformations.to_vec());
        }

        let mut corrected = Vec::with_capacity(transformations.len());

        for (i, transform) in transformations.iter().enumerate() {
            // Estimate and remove wobble component
            let wobble_amplitude = self.estimate_wobble_amplitude(transformations, i);
            let phase = 2.0 * PI * self.wobble_frequency * i as f64;
            let wobble_offset = wobble_amplitude * phase.sin();

            corrected.push(TransformMatrix {
                tx: transform.tx,
                ty: transform.ty - wobble_offset,
                angle: transform.angle,
                scale: transform.scale,
            });
        }

        Ok(corrected)
    }

    /// Estimate wobble amplitude at a specific frame.
    fn estimate_wobble_amplitude(
        &self,
        transformations: &[TransformMatrix],
        frame_idx: usize,
    ) -> f64 {
        // Use local window to estimate amplitude
        let window_size = 10;
        let start = frame_idx.saturating_sub(window_size / 2);
        let end = (frame_idx + window_size / 2).min(transformations.len());

        let mut max_ty = f64::MIN;
        let mut min_ty = f64::MAX;

        for transform in &transformations[start..end] {
            max_ty = max_ty.max(transform.ty);
            min_ty = min_ty.min(transform.ty);
        }

        (max_ty - min_ty) / 2.0
    }
}

impl Default for JelloCorrector {
    fn default() -> Self {
        Self::new()
    }
}
