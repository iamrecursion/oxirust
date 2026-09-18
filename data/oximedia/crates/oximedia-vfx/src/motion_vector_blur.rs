//! Motion-vector-based motion blur.
//!
//! Unlike the uniform-kernel approach in [`crate::motion_blur`], this module
//! reads a per-pixel motion vector field (a displacement map) that describes
//! where each pixel *came from* in the previous frame.  The blur for each
//! output pixel is computed by accumulating samples along its individual
//! displacement trajectory, producing physically plausible motion blur that
//! respects object boundaries and varying speeds.
//!
//! # Displacement Map Layout
//!
//! The displacement map is a `width × height` slice of `MotionVector`s.
//! Each entry `mv[y * width + x]` stores the **signed pixel displacement**
//! `(dx, dy)` such that the pixel at `(x, y)` moved *from* `(x − dx, y − dy)`
//! to `(x, y)` between the previous and current frame.
//!
//! Vectors of zero magnitude produce no blur (sharp pixel).

use crate::{Frame, VfxError, VfxResult};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// MotionVector
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-component per-pixel motion vector (horizontal/vertical displacement in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct MotionVector {
    /// Horizontal displacement in pixels (positive = moved right).
    pub dx: f32,
    /// Vertical displacement in pixels (positive = moved down).
    pub dy: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Zero-displacement (stationary) vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Magnitude of the displacement.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Returns `true` if this vector has no meaningful displacement
    /// (magnitude < `threshold`).
    #[must_use]
    pub fn is_stationary(&self, threshold: f32) -> bool {
        self.magnitude() < threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionVectorField
// ─────────────────────────────────────────────────────────────────────────────

/// A dense per-pixel motion vector field for a single frame.
#[derive(Debug, Clone)]
pub struct MotionVectorField {
    /// Width of the field in pixels.
    pub width: u32,
    /// Height of the field in pixels.
    pub height: u32,
    /// Row-major flat array of motion vectors (length = width × height).
    pub vectors: Vec<MotionVector>,
}

impl MotionVectorField {
    /// Create a zero-displacement field (no motion anywhere).
    ///
    /// # Errors
    ///
    /// Returns [`VfxError::InvalidDimensions`] if either dimension is zero.
    pub fn new_zero(width: u32, height: u32) -> VfxResult<Self> {
        if width == 0 || height == 0 {
            return Err(VfxError::InvalidDimensions { width, height });
        }
        let count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(VfxError::InvalidDimensions { width, height })?;
        Ok(Self {
            width,
            height,
            vectors: vec![MotionVector::zero(); count],
        })
    }

    /// Create a field from an existing vector slice.
    ///
    /// # Errors
    ///
    /// Returns [`VfxError::BufferSizeMismatch`] if `vectors.len() != width * height`.
    pub fn from_vectors(width: u32, height: u32, vectors: Vec<MotionVector>) -> VfxResult<Self> {
        let expected = (width as usize) * (height as usize);
        if vectors.len() != expected {
            return Err(VfxError::BufferSizeMismatch {
                expected,
                actual: vectors.len(),
            });
        }
        Ok(Self {
            width,
            height,
            vectors,
        })
    }

    /// Get the motion vector at pixel `(x, y)`.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<MotionVector> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.vectors
            .get((y as usize) * (self.width as usize) + (x as usize))
            .copied()
    }

    /// Set the motion vector at pixel `(x, y)`.
    pub fn set(&mut self, x: u32, y: u32, mv: MotionVector) {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            if let Some(slot) = self.vectors.get_mut(idx) {
                *slot = mv;
            }
        }
    }

    /// Fill the entire field with a single uniform vector (e.g. panning blur).
    pub fn fill_uniform(&mut self, mv: MotionVector) {
        for slot in &mut self.vectors {
            *slot = mv;
        }
    }

    /// Create a synthetic radial field centred at `(cx, cy)` scaled by `scale`.
    ///
    /// Each pixel's vector points outward from the centre proportional to its
    /// distance multiplied by `scale`.  Useful for zoom-motion blur testing.
    pub fn from_radial(width: u32, height: u32, cx: f32, cy: f32, scale: f32) -> VfxResult<Self> {
        let mut field = Self::new_zero(width, height)?;
        for y in 0..height {
            for x in 0..width {
                let dx = (x as f32 - cx) * scale;
                let dy = (y as f32 - cy) * scale;
                field.set(x, y, MotionVector::new(dx, dy));
            }
        }
        Ok(field)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionVectorBlur
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the motion-vector-based blur pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvBlurConfig {
    /// Number of samples taken along each pixel's motion trail [1, 64].
    /// More samples = smoother blur at higher cost.
    pub num_samples: u32,
    /// Global strength multiplier: 0.0 = no blur, 1.0 = full displacement.
    pub strength: f32,
    /// Minimum motion magnitude (pixels) below which the pixel is not blurred.
    pub min_magnitude: f32,
    /// Maximum clamped displacement in pixels (prevents artefacts from noisy MVs).
    pub max_displacement: f32,
}

impl Default for MvBlurConfig {
    fn default() -> Self {
        Self {
            num_samples: 8,
            strength: 1.0,
            min_magnitude: 0.5,
            max_displacement: 32.0,
        }
    }
}

impl MvBlurConfig {
    /// Clamp configuration values to valid ranges.
    #[must_use]
    pub fn validated(mut self) -> Self {
        self.num_samples = self.num_samples.clamp(1, 64);
        self.strength = self.strength.clamp(0.0, 4.0);
        self.min_magnitude = self.min_magnitude.max(0.0);
        self.max_displacement = self.max_displacement.clamp(1.0, 256.0);
        self
    }
}

/// Applies motion-vector-based motion blur.
///
/// Each output pixel accumulates `num_samples` bilinearly-interpolated source
/// pixels taken at evenly-spaced positions along the vector `[0 .. displacement]`.
/// Pixels whose motion vector magnitude is below `min_magnitude` are copied
/// unchanged.
///
/// # Errors
///
/// Returns [`VfxError::InvalidDimensions`] if `input` and `field` dimensions differ,
/// or if `output` dimensions differ from `input`.
pub fn apply_mv_blur(
    input: &Frame,
    output: &mut Frame,
    field: &MotionVectorField,
    config: &MvBlurConfig,
) -> VfxResult<()> {
    if input.width != field.width || input.height != field.height {
        return Err(VfxError::InvalidDimensions {
            width: field.width,
            height: field.height,
        });
    }
    if input.width != output.width || input.height != output.height {
        return Err(VfxError::InvalidDimensions {
            width: output.width,
            height: output.height,
        });
    }

    let w = input.width;
    let h = input.height;
    let n = config.num_samples.max(1);
    let inv_n = 1.0 / n as f32;

    for y in 0..h {
        for x in 0..w {
            let mv = field.get(x, y).unwrap_or(MotionVector::zero());

            // Clamp displacement magnitude
            let raw_mag = mv.magnitude();
            if raw_mag < config.min_magnitude {
                // Below threshold — copy source pixel unchanged
                let src = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                output.set_pixel(x, y, src);
                continue;
            }

            let scale = (raw_mag.min(config.max_displacement) / raw_mag) * config.strength;
            let dx = mv.dx * scale;
            let dy = mv.dy * scale;

            // Accumulate along the displacement trail from (x, y) to (x - dx, y - dy)
            let mut acc_r = 0.0f32;
            let mut acc_g = 0.0f32;
            let mut acc_b = 0.0f32;
            let mut acc_a = 0.0f32;

            for s in 0..n {
                let t = s as f32 * inv_n; // [0, 1)
                let sx = x as f32 - dx * t;
                let sy = y as f32 - dy * t;

                let [r, g, b, a] = sample_bilinear(input, sx, sy);
                acc_r += r as f32;
                acc_g += g as f32;
                acc_b += b as f32;
                acc_a += a as f32;
            }

            let pixel = [
                (acc_r * inv_n).clamp(0.0, 255.0) as u8,
                (acc_g * inv_n).clamp(0.0, 255.0) as u8,
                (acc_b * inv_n).clamp(0.0, 255.0) as u8,
                (acc_a * inv_n).clamp(0.0, 255.0) as u8,
            ];
            output.set_pixel(x, y, pixel);
        }
    }

    Ok(())
}

/// Bilinearly sample `frame` at floating-point pixel coordinates.
///
/// Clamps to frame bounds.
fn sample_bilinear(frame: &Frame, fx: f32, fy: f32) -> [u8; 4] {
    let w = frame.width as i32;
    let h = frame.height as i32;

    let x0 = (fx.floor() as i32).clamp(0, w - 1) as u32;
    let y0 = (fy.floor() as i32).clamp(0, h - 1) as u32;
    let x1 = (x0 + 1).min(frame.width - 1);
    let y1 = (y0 + 1).min(frame.height - 1);

    let tx = fx - fx.floor();
    let ty = fy - fy.floor();

    let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
    let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
    let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
    let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

    let mut out = [0u8; 4];
    for ch in 0..4 {
        let top = p00[ch] as f32 * (1.0 - tx) + p10[ch] as f32 * tx;
        let bot = p01[ch] as f32 * (1.0 - tx) + p11[ch] as f32 * tx;
        out[ch] = (top * (1.0 - ty) + bot * ty).clamp(0.0, 255.0) as u8;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear(rgba);
        f
    }

    // ── MotionVector ───────────────────────────────────────────────────────

    #[test]
    fn test_mv_zero_magnitude() {
        let mv = MotionVector::zero();
        assert_eq!(mv.magnitude(), 0.0);
        assert!(mv.is_stationary(0.5));
    }

    #[test]
    fn test_mv_magnitude_345() {
        let mv = MotionVector::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mv_not_stationary_above_threshold() {
        let mv = MotionVector::new(2.0, 0.0);
        assert!(!mv.is_stationary(0.5));
    }

    // ── MotionVectorField ──────────────────────────────────────────────────

    #[test]
    fn test_field_new_zero() {
        let f = MotionVectorField::new_zero(4, 4).expect("ok");
        assert_eq!(f.vectors.len(), 16);
        assert!(f.get(2, 2).expect("get").magnitude() == 0.0);
    }

    #[test]
    fn test_field_new_zero_fails_on_zero_dim() {
        assert!(MotionVectorField::new_zero(0, 4).is_err());
        assert!(MotionVectorField::new_zero(4, 0).is_err());
    }

    #[test]
    fn test_field_from_vectors_wrong_size() {
        let bad = vec![MotionVector::zero(); 10]; // wrong for 4×4 = 16
        assert!(MotionVectorField::from_vectors(4, 4, bad).is_err());
    }

    #[test]
    fn test_field_set_get() {
        let mut f = MotionVectorField::new_zero(8, 8).expect("ok");
        f.set(3, 5, MotionVector::new(1.5, -2.0));
        let mv = f.get(3, 5).expect("get");
        assert!((mv.dx - 1.5).abs() < 1e-5);
        assert!((mv.dy + 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_field_fill_uniform() {
        let mut f = MotionVectorField::new_zero(4, 4).expect("ok");
        f.fill_uniform(MotionVector::new(5.0, 5.0));
        assert!(f.vectors.iter().all(|mv| (mv.dx - 5.0).abs() < 1e-5));
    }

    #[test]
    fn test_field_out_of_bounds_get_none() {
        let f = MotionVectorField::new_zero(4, 4).expect("ok");
        assert!(f.get(10, 10).is_none());
    }

    // ── apply_mv_blur ──────────────────────────────────────────────────────

    #[test]
    fn test_mv_blur_zero_field_copies_input() {
        let input = solid_frame(8, 8, [200, 100, 50, 255]);
        let mut output = Frame::new(8, 8).expect("output");
        let field = MotionVectorField::new_zero(8, 8).expect("field");
        let config = MvBlurConfig::default();
        apply_mv_blur(&input, &mut output, &field, &config).expect("apply");

        // All pixels should equal input (no motion → no blur)
        for y in 0..8 {
            for x in 0..8 {
                let p = output.get_pixel(x, y).expect("pixel");
                assert_eq!(p, [200, 100, 50, 255], "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn test_mv_blur_uniform_field_uniform_image_unchanged() {
        // On a uniform colour image any blur should produce the same colour
        let input = solid_frame(16, 16, [128, 64, 32, 255]);
        let mut output = Frame::new(16, 16).expect("output");
        let mut field = MotionVectorField::new_zero(16, 16).expect("field");
        field.fill_uniform(MotionVector::new(3.0, 2.0));
        let config = MvBlurConfig {
            num_samples: 6,
            ..Default::default()
        };
        apply_mv_blur(&input, &mut output, &field, &config).expect("apply");

        // On a uniform source the blurred result must also be uniform
        let center = output.get_pixel(8, 8).expect("center");
        assert!((center[0] as i32 - 128).abs() <= 2);
        assert!((center[1] as i32 - 64).abs() <= 2);
    }

    #[test]
    fn test_mv_blur_dimension_mismatch_input_vs_field() {
        let input = solid_frame(8, 8, [0, 0, 0, 255]);
        let mut output = Frame::new(8, 8).expect("output");
        let field = MotionVectorField::new_zero(16, 16).expect("field");
        assert!(apply_mv_blur(&input, &mut output, &field, &MvBlurConfig::default()).is_err());
    }

    #[test]
    fn test_mv_blur_dimension_mismatch_input_vs_output() {
        let input = solid_frame(8, 8, [0, 0, 0, 255]);
        let mut output = Frame::new(4, 4).expect("output");
        let field = MotionVectorField::new_zero(8, 8).expect("field");
        assert!(apply_mv_blur(&input, &mut output, &field, &MvBlurConfig::default()).is_err());
    }

    #[test]
    fn test_mv_blur_config_validated() {
        let cfg = MvBlurConfig {
            num_samples: 200,
            strength: -1.0,
            min_magnitude: -5.0,
            max_displacement: 0.0,
        }
        .validated();
        assert_eq!(cfg.num_samples, 64);
        assert_eq!(cfg.strength, 0.0);
        assert_eq!(cfg.min_magnitude, 0.0);
        assert_eq!(cfg.max_displacement, 1.0);
    }

    #[test]
    fn test_mv_blur_radial_field_output_same_size() {
        let input = solid_frame(32, 32, [100, 150, 200, 255]);
        let mut output = Frame::new(32, 32).expect("output");
        let field = MotionVectorField::from_radial(32, 32, 16.0, 16.0, 0.1).expect("radial");
        let config = MvBlurConfig::default();
        apply_mv_blur(&input, &mut output, &field, &config).expect("apply");
        assert_eq!(output.width, 32);
        assert_eq!(output.height, 32);
    }
}
