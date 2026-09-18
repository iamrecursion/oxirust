//! Motion blur synthesis and removal.
//!
//! This module provides comprehensive motion blur processing capabilities:
//!
//! - **Synthesis**: Create realistic motion blur effects
//!   - Directional blur based on motion vectors
//!   - Accumulation buffer approach
//!   - Camera shake simulation
//!   - Configurable shutter angle (0-360°)
//!   - Per-object blur
//!
//! - **Removal**: Reduce or eliminate motion blur
//!   - Blind deconvolution
//!   - Wiener deconvolution
//!   - Richardson-Lucy deconvolution
//!   - Motion PSF estimation
//!   - Multi-frame deblurring
//!
//! - **Motion estimation**
//!   - Optical flow (reuse from tracking module)
//!   - Block matching
//!   - Motion vector analysis
//!   - Camera motion vs object motion separation
//!
//! - **Advanced features**
//!   - Depth-aware blur
//!   - Rolling shutter compensation
//!   - Motion trail generation
//!   - Speed ramping effects
//!
//! # Quality Modes
//!
//! - **Fast**: Simple directional blur
//! - **Balanced**: Optical flow + blur
//! - **High Quality**: Per-pixel motion blur with accumulation
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::motion_blur::{MotionBlurSynthesizer, QualityMode};
//!
//! let synthesizer = MotionBlurSynthesizer::new()
//!     .with_shutter_angle(180.0)
//!     .with_samples(8)
//!     .with_quality(QualityMode::Balanced);
//! ```

mod deconvolve;
mod psf;
mod removal;
mod synthesis;

pub use deconvolve::{
    DeconvolutionMethod, Deconvolver, PsfPadStrategy, RichardsonLucyParams, WienerFftParams,
    WienerParams,
};
pub use psf::{MotionPSF, PSFEstimator, PSFShape};
pub use removal::{DeblurMethod, DeblurQuality, MotionBlurRemover, MultiFrameDeblur};
pub use synthesis::{
    AccumulationBuffer, CameraShake, MotionBlurSynthesizer, MotionTrail, PerObjectBlur, SpeedRamp,
};

use crate::error::{CvError, CvResult};
use crate::tracking::FlowField;

/// Quality mode for motion blur processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityMode {
    /// Fast processing with simple directional blur.
    Fast,
    /// Balanced quality and speed with optical flow.
    #[default]
    Balanced,
    /// High quality per-pixel motion blur with accumulation.
    HighQuality,
}

/// Motion vector for a pixel or region.
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    /// Horizontal component (pixels per frame).
    pub dx: f32,
    /// Vertical component (pixels per frame).
    pub dy: f32,
    /// Confidence/reliability (0.0 to 1.0).
    pub confidence: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(dx: f32, dy: f32) -> Self {
        Self {
            dx,
            dy,
            confidence: 1.0,
        }
    }

    /// Create a new motion vector with confidence.
    #[must_use]
    pub const fn with_confidence(dx: f32, dy: f32, confidence: f32) -> Self {
        Self { dx, dy, confidence }
    }

    /// Get the magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Get the direction of the motion vector in radians.
    #[must_use]
    pub fn direction(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Scale the motion vector by a factor.
    #[must_use]
    pub const fn scale(&self, factor: f32) -> Self {
        Self {
            dx: self.dx * factor,
            dy: self.dy * factor,
            confidence: self.confidence,
        }
    }

    /// Interpolate between two motion vectors.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            dx: self.dx + (other.dx - self.dx) * t,
            dy: self.dy + (other.dy - self.dy) * t,
            confidence: self.confidence + (other.confidence - self.confidence) * t,
        }
    }

    /// Check if the motion vector is zero or negligible.
    #[must_use]
    pub fn is_negligible(&self) -> bool {
        self.magnitude() < 0.5
    }
}

impl Default for MotionVector {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

/// Motion vector field for an entire image.
#[derive(Debug, Clone)]
pub struct MotionVectorField {
    /// Motion vectors (one per pixel).
    pub vectors: Vec<MotionVector>,
    /// Field width.
    pub width: u32,
    /// Field height.
    pub height: u32,
}

impl MotionVectorField {
    /// Create a new motion vector field.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            vectors: vec![MotionVector::default(); size],
            width,
            height,
        }
    }

    /// Create from a flow field.
    #[must_use]
    pub fn from_flow_field(flow: &FlowField) -> Self {
        let mut field = Self::new(flow.width, flow.height);
        for i in 0..flow.flow_x.len() {
            field.vectors[i] = MotionVector::new(flow.flow_x[i], flow.flow_y[i]);
        }
        field
    }

    /// Get motion vector at a specific position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> MotionVector {
        if x >= self.width || y >= self.height {
            return MotionVector::default();
        }
        let idx = (y * self.width + x) as usize;
        if idx < self.vectors.len() {
            self.vectors[idx]
        } else {
            MotionVector::default()
        }
    }

    /// Set motion vector at a specific position.
    pub fn set(&mut self, x: u32, y: u32, vector: MotionVector) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        if idx < self.vectors.len() {
            self.vectors[idx] = vector;
        }
    }

    /// Get bilinear interpolated motion vector at a fractional position.
    #[must_use]
    pub fn get_interpolated(&self, x: f32, y: f32) -> MotionVector {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let v00 = self.get(x0, y0);
        let v10 = self.get(x1, y0);
        let v01 = self.get(x0, y1);
        let v11 = self.get(x1, y1);

        let v0 = v00.lerp(&v10, fx);
        let v1 = v01.lerp(&v11, fx);
        v0.lerp(&v1, fy)
    }

    /// Get average motion magnitude across the field.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(MotionVector::magnitude).sum();
        sum / self.vectors.len() as f32
    }

    /// Get maximum motion magnitude in the field.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        self.vectors
            .iter()
            .map(MotionVector::magnitude)
            .fold(0.0f32, f32::max)
    }

    /// Scale all motion vectors by a factor.
    pub fn scale(&mut self, factor: f32) {
        for vector in &mut self.vectors {
            *vector = vector.scale(factor);
        }
    }

    /// Apply median filter to smooth the motion field.
    pub fn median_filter(&mut self, radius: u32) {
        let mut filtered = self.vectors.clone();
        let rad = radius as i32;

        for y in 0..self.height {
            for x in 0..self.width {
                let mut dx_vals = Vec::new();
                let mut dy_vals = Vec::new();

                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        let nx = (x as i32 + dx).max(0).min(self.width as i32 - 1) as u32;
                        let ny = (y as i32 + dy).max(0).min(self.height as i32 - 1) as u32;
                        let v = self.get(nx, ny);
                        dx_vals.push(v.dx);
                        dy_vals.push(v.dy);
                    }
                }

                dx_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                dy_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median_idx = dx_vals.len() / 2;
                let idx = (y * self.width + x) as usize;
                if idx < filtered.len() {
                    filtered[idx].dx = dx_vals[median_idx];
                    filtered[idx].dy = dy_vals[median_idx];
                }
            }
        }

        self.vectors = filtered;
    }

    /// Separate camera motion from object motion.
    ///
    /// Returns (camera_motion, object_motion_field).
    #[must_use]
    pub fn separate_camera_motion(&self) -> (MotionVector, Self) {
        // Estimate global camera motion using median
        let mut all_dx: Vec<f32> = self.vectors.iter().map(|v| v.dx).collect();
        let mut all_dy: Vec<f32> = self.vectors.iter().map(|v| v.dy).collect();

        all_dx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_dy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_idx = all_dx.len() / 2;
        let camera_motion = if median_idx < all_dx.len() && median_idx < all_dy.len() {
            MotionVector::new(all_dx[median_idx], all_dy[median_idx])
        } else {
            MotionVector::default()
        };

        // Subtract camera motion from all vectors to get object motion
        let mut object_field = self.clone();
        for vector in &mut object_field.vectors {
            vector.dx -= camera_motion.dx;
            vector.dy -= camera_motion.dy;
        }

        (camera_motion, object_field)
    }
}

/// Motion blur configuration.
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    /// Shutter angle in degrees (0-360).
    pub shutter_angle: f32,
    /// Number of samples for accumulation.
    pub samples: usize,
    /// Quality mode.
    pub quality: QualityMode,
    /// Enable depth-aware blur.
    pub depth_aware: bool,
    /// Enable rolling shutter compensation.
    pub rolling_shutter: bool,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            shutter_angle: 180.0,
            samples: 8,
            quality: QualityMode::Balanced,
            depth_aware: false,
            rolling_shutter: false,
        }
    }
}

impl MotionBlurConfig {
    /// Create a new motion blur configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set shutter angle.
    #[must_use]
    pub const fn with_shutter_angle(mut self, angle: f32) -> Self {
        self.shutter_angle = angle;
        self
    }

    /// Set number of samples.
    #[must_use]
    pub const fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Set quality mode.
    #[must_use]
    pub const fn with_quality(mut self, quality: QualityMode) -> Self {
        self.quality = quality;
        self
    }

    /// Enable depth-aware blur.
    #[must_use]
    pub const fn with_depth_aware(mut self, enabled: bool) -> Self {
        self.depth_aware = enabled;
        self
    }

    /// Enable rolling shutter compensation.
    #[must_use]
    pub const fn with_rolling_shutter(mut self, enabled: bool) -> Self {
        self.rolling_shutter = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> CvResult<()> {
        if self.shutter_angle < 0.0 || self.shutter_angle > 360.0 {
            return Err(CvError::invalid_parameter(
                "shutter_angle",
                format!("{} (must be 0-360)", self.shutter_angle),
            ));
        }
        if self.samples == 0 {
            return Err(CvError::invalid_parameter("samples", "0 (must be > 0)"));
        }
        if self.samples > 64 {
            return Err(CvError::invalid_parameter(
                "samples",
                format!("{} (must be <= 64)", self.samples),
            ));
        }
        Ok(())
    }

    /// Get the motion scaling factor based on shutter angle.
    #[must_use]
    pub fn motion_scale(&self) -> f32 {
        self.shutter_angle / 360.0
    }
}

/// Depth map for depth-aware motion blur.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Depth values (normalized 0.0 to 1.0, where 0 is near and 1 is far).
    pub depths: Vec<f32>,
    /// Map width.
    pub width: u32,
    /// Map height.
    pub height: u32,
}

impl DepthMap {
    /// Create a new depth map.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            depths: vec![0.5; size],
            width,
            height,
        }
    }

    /// Get depth at a specific position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.5;
        }
        let idx = (y * self.width + x) as usize;
        if idx < self.depths.len() {
            self.depths[idx]
        } else {
            0.5
        }
    }

    /// Set depth at a specific position.
    pub fn set(&mut self, x: u32, y: u32, depth: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        if idx < self.depths.len() {
            self.depths[idx] = depth.clamp(0.0, 1.0);
        }
    }

    /// Get interpolated depth at a fractional position.
    #[must_use]
    pub fn get_interpolated(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let d00 = self.get(x0, y0);
        let d10 = self.get(x1, y0);
        let d01 = self.get(x0, y1);
        let d11 = self.get(x1, y1);

        let d0 = d00 + (d10 - d00) * fx;
        let d1 = d01 + (d11 - d01) * fx;
        d0 + (d1 - d0) * fy
    }
}

/// Rolling shutter parameters.
#[derive(Debug, Clone)]
pub struct RollingShutterParams {
    /// Scan direction (true = top-to-bottom, false = bottom-to-top).
    pub top_to_bottom: bool,
    /// Rolling shutter time as fraction of frame time (typically 0.5-1.0).
    pub scan_time: f32,
    /// Readout offset (0.0 to 1.0).
    pub readout_offset: f32,
}

impl Default for RollingShutterParams {
    fn default() -> Self {
        Self {
            top_to_bottom: true,
            scan_time: 1.0,
            readout_offset: 0.0,
        }
    }
}

impl RollingShutterParams {
    /// Create new rolling shutter parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the time offset for a given scanline.
    #[must_use]
    pub fn time_offset(&self, y: u32, height: u32) -> f32 {
        let normalized_y = if self.top_to_bottom {
            y as f32 / height as f32
        } else {
            1.0 - (y as f32 / height as f32)
        };
        normalized_y * self.scan_time + self.readout_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_new() {
        let v = MotionVector::new(10.0, 20.0);
        assert_eq!(v.dx, 10.0);
        assert_eq!(v.dy, 20.0);
        assert_eq!(v.confidence, 1.0);
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let v = MotionVector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_vector_scale() {
        let v = MotionVector::new(10.0, 20.0);
        let scaled = v.scale(0.5);
        assert_eq!(scaled.dx, 5.0);
        assert_eq!(scaled.dy, 10.0);
    }

    #[test]
    fn test_motion_vector_lerp() {
        let v1 = MotionVector::new(0.0, 0.0);
        let v2 = MotionVector::new(10.0, 20.0);
        let lerped = v1.lerp(&v2, 0.5);
        assert_eq!(lerped.dx, 5.0);
        assert_eq!(lerped.dy, 10.0);
    }

    #[test]
    fn test_motion_vector_field_new() {
        let field = MotionVectorField::new(100, 100);
        assert_eq!(field.width, 100);
        assert_eq!(field.height, 100);
        assert_eq!(field.vectors.len(), 10000);
    }

    #[test]
    fn test_motion_vector_field_get_set() {
        let mut field = MotionVectorField::new(10, 10);
        let v = MotionVector::new(5.0, 10.0);
        field.set(5, 5, v);
        let retrieved = field.get(5, 5);
        assert_eq!(retrieved.dx, 5.0);
        assert_eq!(retrieved.dy, 10.0);
    }

    #[test]
    fn test_motion_blur_config_default() {
        let config = MotionBlurConfig::default();
        assert_eq!(config.shutter_angle, 180.0);
        assert_eq!(config.samples, 8);
    }

    #[test]
    fn test_motion_blur_config_validate() {
        let config = MotionBlurConfig::new()
            .with_shutter_angle(180.0)
            .with_samples(16);
        assert!(config.validate().is_ok());

        let bad_config = MotionBlurConfig::new().with_shutter_angle(400.0);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_depth_map_new() {
        let depth = DepthMap::new(100, 100);
        assert_eq!(depth.width, 100);
        assert_eq!(depth.height, 100);
        assert_eq!(depth.depths.len(), 10000);
    }

    #[test]
    fn test_depth_map_get_set() {
        let mut depth = DepthMap::new(10, 10);
        depth.set(5, 5, 0.8);
        assert!((depth.get(5, 5) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_rolling_shutter_params() {
        let params = RollingShutterParams::default();
        assert!(params.top_to_bottom);

        let offset = params.time_offset(50, 100);
        assert!((offset - 0.5).abs() < 0.001);
    }
}
