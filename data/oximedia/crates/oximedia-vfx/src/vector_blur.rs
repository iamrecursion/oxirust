//! Vector-based motion blur using per-pixel motion vectors.
//!
//! Implements motion blur by reconstructing the blur trail from 2-D motion
//! vectors and a configurable shutter angle.

#![allow(dead_code)]

use std::f32::consts::PI;

/// A 2-D motion vector with sub-pixel precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Euclidean magnitude of the vector in pixels.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Angle of the vector in radians (0 = rightward).
    #[must_use]
    pub fn angle_radians(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Scale the vector by a scalar.
    #[must_use]
    pub fn scale(&self, s: f32) -> Self {
        Self::new(self.dx * s, self.dy * s)
    }

    /// Zero vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Default for MotionVector {
    fn default() -> Self {
        Self::zero()
    }
}

/// Configuration for a motion-blur render pass.
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    /// Camera shutter angle in degrees (0–360). 180° ≈ film standard.
    pub shutter_angle_deg: f32,
    /// Number of accumulation samples along the blur trail.
    pub samples: u32,
    /// Frames per second of the source material.
    pub fps: f32,
    /// Maximum blur length in pixels (safety cap).
    pub max_blur_pixels: f32,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            shutter_angle_deg: 180.0,
            samples: 16,
            fps: 24.0,
            max_blur_pixels: 64.0,
        }
    }
}

impl MotionBlurConfig {
    /// Create a config with a specific shutter angle.
    #[must_use]
    pub fn with_shutter_angle(shutter_angle_deg: f32) -> Self {
        Self {
            shutter_angle_deg: shutter_angle_deg.clamp(0.0, 360.0),
            ..Default::default()
        }
    }

    /// Return the shutter angle in radians.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn shutter_angle_radians(&self) -> f32 {
        self.shutter_angle_deg * PI / 180.0
    }

    /// Exposure fraction derived from shutter angle (`shutter_angle` / 360).
    #[must_use]
    pub fn exposure_fraction(&self) -> f32 {
        self.shutter_angle_deg / 360.0
    }

    /// Returns `true` if settings are in a sane range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.samples > 0 && self.fps > 0.0 && self.shutter_angle_deg >= 0.0
    }
}

/// Accumulation motion-blur pass operating on RGBA byte buffers.
pub struct MotionBlurPass {
    config: MotionBlurConfig,
}

impl MotionBlurPass {
    /// Create a new motion-blur pass.
    #[must_use]
    pub fn new(config: MotionBlurConfig) -> Self {
        Self { config }
    }

    /// Number of accumulation samples.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.config.samples
    }

    /// Apply motion blur to `src` writing into `dst`.
    ///
    /// `vectors` must have one `MotionVector` per pixel (row-major order).
    /// Both `src` and `dst` must be `width * height * 4` bytes (RGBA).
    ///
    /// # Panics
    ///
    /// Panics if buffer sizes are inconsistent.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(
        &self,
        src: &[u8],
        dst: &mut [u8],
        vectors: &[MotionVector],
        width: u32,
        height: u32,
    ) {
        let n = (width * height) as usize;
        assert_eq!(src.len(), n * 4, "src buffer size mismatch");
        assert_eq!(dst.len(), n * 4, "dst buffer size mismatch");
        assert_eq!(vectors.len(), n, "vector buffer size mismatch");

        let scale = self.config.exposure_fraction();
        let w = width as usize;
        let h = height as usize;
        let samples = self.config.samples.max(1) as usize;
        let max_blur = self.config.max_blur_pixels;

        for py in 0..h {
            for px in 0..w {
                let idx = py * w + px;
                let mv = vectors[idx];
                let blur_len = (mv.magnitude() * scale).min(max_blur);

                let mut r = 0.0_f32;
                let mut g = 0.0_f32;
                let mut b = 0.0_f32;
                let mut a = 0.0_f32;

                for s in 0..samples {
                    let t = if samples == 1 {
                        0.0
                    } else {
                        (s as f32 / (samples - 1) as f32) - 0.5
                    };
                    let sx = (px as f32 + mv.dx * scale * t * blur_len / (mv.magnitude().max(1e-6)))
                        .clamp(0.0, (w - 1) as f32) as usize;
                    let sy = (py as f32 + mv.dy * scale * t * blur_len / (mv.magnitude().max(1e-6)))
                        .clamp(0.0, (h - 1) as f32) as usize;
                    let si = (sy * w + sx) * 4;
                    r += src[si] as f32;
                    g += src[si + 1] as f32;
                    b += src[si + 2] as f32;
                    a += src[si + 3] as f32;
                }

                let inv = 1.0 / samples as f32;
                let di = idx * 4;
                dst[di] = (r * inv).round() as u8;
                dst[di + 1] = (g * inv).round() as u8;
                dst[di + 2] = (b * inv).round() as u8;
                dst[di + 3] = (a * inv).round() as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_magnitude_zero() {
        let mv = MotionVector::zero();
        assert_eq!(mv.magnitude(), 0.0);
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_motion_vector_angle() {
        let mv = MotionVector::new(1.0, 0.0);
        assert!((mv.angle_radians()).abs() < 1e-5);
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector::new(2.0, 3.0);
        let scaled = mv.scale(2.0);
        assert!((scaled.dx - 4.0).abs() < 1e-5);
        assert!((scaled.dy - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_shutter_angle_radians() {
        let cfg = MotionBlurConfig::with_shutter_angle(180.0);
        let rad = cfg.shutter_angle_radians();
        assert!((rad - PI).abs() < 1e-5);
    }

    #[test]
    fn test_config_exposure_fraction() {
        let cfg = MotionBlurConfig::with_shutter_angle(180.0);
        assert!((cfg.exposure_fraction() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_config_is_valid() {
        let cfg = MotionBlurConfig::default();
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_config_invalid_samples() {
        let mut cfg = MotionBlurConfig::default();
        cfg.samples = 0;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_pass_sample_count() {
        let cfg = MotionBlurConfig {
            samples: 32,
            ..Default::default()
        };
        let pass = MotionBlurPass::new(cfg);
        assert_eq!(pass.sample_count(), 32);
    }

    #[test]
    fn test_pass_apply_no_motion() {
        let w = 4u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let src: Vec<u8> = (0..n * 4).map(|i| (i % 256) as u8).collect();
        let mut dst = vec![0u8; n * 4];
        let vectors = vec![MotionVector::zero(); n];
        let pass = MotionBlurPass::new(MotionBlurConfig::default());
        pass.apply(&src, &mut dst, &vectors, w, h);
        // With no motion every pixel should equal src
        assert_eq!(src, dst);
    }

    #[test]
    fn test_pass_apply_with_motion_preserves_size() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h) as usize;
        let src: Vec<u8> = vec![200u8; n * 4];
        let mut dst = vec![0u8; n * 4];
        let vectors = vec![MotionVector::new(2.0, 0.0); n];
        let pass = MotionBlurPass::new(MotionBlurConfig::default());
        pass.apply(&src, &mut dst, &vectors, w, h);
        assert_eq!(dst.len(), n * 4);
    }

    #[test]
    fn test_config_shutter_clamped() {
        let cfg = MotionBlurConfig::with_shutter_angle(720.0);
        assert!(cfg.shutter_angle_deg <= 360.0);
    }

    #[test]
    fn test_motion_vector_default_is_zero() {
        let mv = MotionVector::default();
        assert_eq!(mv.dx, 0.0);
        assert_eq!(mv.dy, 0.0);
    }
}
