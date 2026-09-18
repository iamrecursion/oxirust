//! Motion blur configuration and vector-based blur application.
//!
//! Complements the kernel-based [`motion_blur`](super::motion_blur) module
//! with a shutter-angle based configuration model and a simpler linear-vector
//! accumulation path.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── MotionBlurConfig ──────────────────────────────────────────────────────────

/// High-level configuration for motion blur rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBlurConfig {
    /// Number of accumulation samples along the motion vector.
    pub samples: u32,
    /// Shutter angle in degrees (0 – 360).
    pub shutter_angle_degrees: f32,
    /// Maximum blur extent in pixels (clamps long vectors).
    pub max_blur_pixels: f32,
}

impl MotionBlurConfig {
    /// Film-style preset (180° shutter, 16 samples).
    #[must_use]
    pub fn film() -> Self {
        Self {
            samples: 16,
            shutter_angle_degrees: 180.0,
            max_blur_pixels: 32.0,
        }
    }

    /// Broadcast video preset (90° shutter, 8 samples, tight blur).
    #[must_use]
    pub fn video() -> Self {
        Self {
            samples: 8,
            shutter_angle_degrees: 90.0,
            max_blur_pixels: 16.0,
        }
    }

    /// No motion blur (effectively disabled).
    #[must_use]
    pub fn off() -> Self {
        Self {
            samples: 1,
            shutter_angle_degrees: 0.0,
            max_blur_pixels: 0.0,
        }
    }

    /// Fraction of a full revolution the shutter is open (angle / 360).
    #[must_use]
    pub fn shutter_fraction(&self) -> f32 {
        self.shutter_angle_degrees / 360.0
    }
}

// ── MotionVector2d ────────────────────────────────────────────────────────────

/// A 2-D per-pixel motion vector (displacement in pixels between frames).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector2d {
    /// Horizontal component (pixels).
    pub vx: f32,
    /// Vertical component (pixels).
    pub vy: f32,
}

impl MotionVector2d {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(vx: f32, vy: f32) -> Self {
        Self { vx, vy }
    }

    /// Euclidean magnitude of the vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }

    /// Return a new vector scaled by `s`.
    #[must_use]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            vx: self.vx * s,
            vy: self.vy * s,
        }
    }
}

// ── BlurStrength ──────────────────────────────────────────────────────────────

/// Derived blur strength helper.
pub struct BlurStrength;

impl BlurStrength {
    /// Compute the effective blur magnitude from a motion magnitude and shutter
    /// fraction: `magnitude * shutter_fraction`.
    #[must_use]
    pub fn from_motion_magnitude(mag: f32, shutter_fraction: f32) -> f32 {
        mag * shutter_fraction
    }
}

// ── apply_linear_motion_blur ──────────────────────────────────────────────────

/// Accumulate `samples` equally-spaced taps along `mv` on an interleaved
/// `channels`-bytes-per-pixel image and write results into `dst`.
///
/// `src` and `dst` must both be `width * height * channels` bytes long.
/// Out-of-bounds sample positions are clamped to the image edge.
pub fn apply_linear_motion_blur(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    mv: &MotionVector2d,
    samples: u32,
    channels: usize,
) {
    assert_eq!(src.len(), width * height * channels);
    assert_eq!(dst.len(), width * height * channels);

    let n = samples.max(1) as usize;
    let weight = 1.0_f32 / n as f32;

    for y in 0..height {
        for x in 0..width {
            let mut acc = vec![0.0_f32; channels];
            for k in 0..n {
                let t = if n == 1 {
                    0.0_f32
                } else {
                    k as f32 / (n - 1) as f32 - 0.5
                };
                let sx = (x as f32 + mv.vx * t).round() as i64;
                let sy = (y as f32 + mv.vy * t).round() as i64;
                let sx = sx.clamp(0, (width as i64) - 1) as usize;
                let sy = sy.clamp(0, (height as i64) - 1) as usize;
                let src_idx = (sy * width + sx) * channels;
                for c in 0..channels {
                    acc[c] += src[src_idx + c] as f32 * weight;
                }
            }
            let dst_idx = (y * width + x) * channels;
            for c in 0..channels {
                dst[dst_idx + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // MotionBlurConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_film_preset_shutter_angle() {
        assert_eq!(MotionBlurConfig::film().shutter_angle_degrees, 180.0);
    }

    #[test]
    fn test_film_shutter_fraction() {
        let cfg = MotionBlurConfig::film();
        assert!((cfg.shutter_fraction() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_video_preset_samples() {
        assert_eq!(MotionBlurConfig::video().samples, 8);
    }

    #[test]
    fn test_video_shutter_fraction() {
        let cfg = MotionBlurConfig::video();
        assert!((cfg.shutter_fraction() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_off_preset_zero_angle() {
        let cfg = MotionBlurConfig::off();
        assert_eq!(cfg.shutter_angle_degrees, 0.0);
    }

    #[test]
    fn test_off_shutter_fraction_is_zero() {
        let cfg = MotionBlurConfig::off();
        assert_eq!(cfg.shutter_fraction(), 0.0);
    }

    // MotionVector2d ───────────────────────────────────────────────────────────

    #[test]
    fn test_motion_vector_magnitude_zero() {
        let mv = MotionVector2d::new(0.0, 0.0);
        assert_eq!(mv.magnitude(), 0.0);
    }

    #[test]
    fn test_motion_vector_magnitude_345() {
        let mv = MotionVector2d::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector2d::new(2.0, 3.0).scale(2.0);
        assert!((mv.vx - 4.0).abs() < 1e-6);
        assert!((mv.vy - 6.0).abs() < 1e-6);
    }

    // BlurStrength ─────────────────────────────────────────────────────────────

    #[test]
    fn test_blur_strength_film() {
        let cfg = MotionBlurConfig::film();
        let strength = BlurStrength::from_motion_magnitude(10.0, cfg.shutter_fraction());
        assert!((strength - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_blur_strength_zero_shutter() {
        let strength = BlurStrength::from_motion_magnitude(10.0, 0.0);
        assert_eq!(strength, 0.0);
    }

    // apply_linear_motion_blur ─────────────────────────────────────────────────

    #[test]
    fn test_apply_blur_identity_single_sample() {
        let src: Vec<u8> = (0..48).map(|i| i as u8).collect(); // 4x4 RGB
        let mut dst = vec![0u8; 48];
        let mv = MotionVector2d::new(0.0, 0.0);
        apply_linear_motion_blur(&src, &mut dst, 4, 4, &mv, 1, 3);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_apply_blur_uniform_image_unchanged() {
        let src = vec![128u8; 3 * 8 * 8];
        let mut dst = vec![0u8; 3 * 8 * 8];
        let mv = MotionVector2d::new(5.0, 3.0);
        apply_linear_motion_blur(&src, &mut dst, 8, 8, &mv, 8, 3);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_apply_blur_output_length() {
        let src = vec![50u8; 4 * 6 * 6];
        let mut dst = vec![0u8; 4 * 6 * 6];
        let mv = MotionVector2d::new(2.0, 1.0);
        apply_linear_motion_blur(&src, &mut dst, 6, 6, &mv, 4, 4);
        assert_eq!(dst.len(), src.len());
    }
}
