#![allow(dead_code)]
//! LUT blending – crossfade between two LUTs with a mix factor.
//!
//! Provides:
//! * Per-pixel blending of 1-D and 3-D LUT outputs.
//! * Baking blended LUTs into new standalone LUTs.
//! * Spatially varying blend masks for creative transitions.
//! * Multiple blend modes (linear, perceptual, luminance-preserving).

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ---------------------------------------------------------------------------
// Blend modes
// ---------------------------------------------------------------------------

/// Blend mode that controls how two LUT outputs are combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// Simple linear interpolation: `A * (1 - t) + B * t`.
    Linear,
    /// Perceptual blend in gamma space (applies a gamma 2.2 round-trip).
    Perceptual,
    /// Luminance-preserving blend: blends chrominance at the mix factor
    /// but preserves the luminance ratio between the two LUTs.
    LuminancePreserving,
    /// Hue-preserving blend: interpolates in a polar-like manner so
    /// that hue angles are not skewed by direct RGB mixing.
    HuePreserving,
}

// ---------------------------------------------------------------------------
// Blend functions for single pixels
// ---------------------------------------------------------------------------

/// Blend two RGB values using the specified mode and mix factor `t` in `[0, 1]`.
///
/// `t = 0.0` returns `a`; `t = 1.0` returns `b`.
#[must_use]
pub fn blend_rgb(a: &Rgb, b: &Rgb, t: f64, mode: BlendMode) -> Rgb {
    let t_clamped = t.clamp(0.0, 1.0);
    match mode {
        BlendMode::Linear => blend_linear(a, b, t_clamped),
        BlendMode::Perceptual => blend_perceptual(a, b, t_clamped),
        BlendMode::LuminancePreserving => blend_luminance_preserving(a, b, t_clamped),
        BlendMode::HuePreserving => blend_hue_preserving(a, b, t_clamped),
    }
}

/// Linear blend.
fn blend_linear(a: &Rgb, b: &Rgb, t: f64) -> Rgb {
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    ]
}

/// Perceptual blend: convert to gamma space, blend, convert back.
fn blend_perceptual(a: &Rgb, b: &Rgb, t: f64) -> Rgb {
    const GAMMA: f64 = 2.2;
    const INV_GAMMA: f64 = 1.0 / 2.2;

    let a_gamma = [
        a[0].max(0.0).powf(INV_GAMMA),
        a[1].max(0.0).powf(INV_GAMMA),
        a[2].max(0.0).powf(INV_GAMMA),
    ];
    let b_gamma = [
        b[0].max(0.0).powf(INV_GAMMA),
        b[1].max(0.0).powf(INV_GAMMA),
        b[2].max(0.0).powf(INV_GAMMA),
    ];

    let blended_gamma = blend_linear(&a_gamma, &b_gamma, t);

    [
        blended_gamma[0].max(0.0).powf(GAMMA),
        blended_gamma[1].max(0.0).powf(GAMMA),
        blended_gamma[2].max(0.0).powf(GAMMA),
    ]
}

/// Luminance-preserving blend.
fn blend_luminance_preserving(a: &Rgb, b: &Rgb, t: f64) -> Rgb {
    // Rec.709 luminance coefficients
    const LR: f64 = 0.2126;
    const LG: f64 = 0.7152;
    const LB: f64 = 0.0722;

    let lum_a = LR * a[0] + LG * a[1] + LB * a[2];
    let lum_b = LR * b[0] + LG * b[1] + LB * b[2];
    let target_lum = lum_a * (1.0 - t) + lum_b * t;

    // Blend chrominance linearly
    let blended = blend_linear(a, b, t);
    let lum_blended = LR * blended[0] + LG * blended[1] + LB * blended[2];

    // Scale to match target luminance
    if lum_blended < 1e-10 {
        return blended;
    }
    let scale = target_lum / lum_blended;
    [
        (blended[0] * scale).clamp(0.0, 1.0),
        (blended[1] * scale).clamp(0.0, 1.0),
        (blended[2] * scale).clamp(0.0, 1.0),
    ]
}

/// Hue-preserving blend using a simple polar decomposition in RGB.
fn blend_hue_preserving(a: &Rgb, b: &Rgb, t: f64) -> Rgb {
    // Decompose into intensity + chroma direction
    let intensity_a = (a[0] + a[1] + a[2]) / 3.0;
    let intensity_b = (b[0] + b[1] + b[2]) / 3.0;
    let intensity_blend = intensity_a * (1.0 - t) + intensity_b * t;

    let chroma_a = [a[0] - intensity_a, a[1] - intensity_a, a[2] - intensity_a];
    let chroma_b = [b[0] - intensity_b, b[1] - intensity_b, b[2] - intensity_b];

    let mag_a =
        (chroma_a[0] * chroma_a[0] + chroma_a[1] * chroma_a[1] + chroma_a[2] * chroma_a[2]).sqrt();
    let mag_b =
        (chroma_b[0] * chroma_b[0] + chroma_b[1] * chroma_b[1] + chroma_b[2] * chroma_b[2]).sqrt();
    let mag_blend = mag_a * (1.0 - t) + mag_b * t;

    // Blend chroma direction via SLERP-like approach (simplified as NLERP)
    let dir_blend = blend_linear(&chroma_a, &chroma_b, t);
    let dir_mag =
        (dir_blend[0] * dir_blend[0] + dir_blend[1] * dir_blend[1] + dir_blend[2] * dir_blend[2])
            .sqrt();

    if dir_mag < 1e-10 {
        return [intensity_blend; 3];
    }

    let scale = mag_blend / dir_mag;
    [
        (intensity_blend + dir_blend[0] * scale).clamp(0.0, 1.0),
        (intensity_blend + dir_blend[1] * scale).clamp(0.0, 1.0),
        (intensity_blend + dir_blend[2] * scale).clamp(0.0, 1.0),
    ]
}

// ---------------------------------------------------------------------------
// 3-D LUT blending
// ---------------------------------------------------------------------------

/// Blend two 3-D LUTs (flat `[r][g][b]` layout) into a new LUT.
///
/// Both LUTs must have the same `size`.
///
/// # Errors
///
/// Returns `LutError::InvalidData` if the LUTs have mismatched sizes.
pub fn blend_lut3d(
    lut_a: &[Rgb],
    lut_b: &[Rgb],
    size: usize,
    mix: f64,
    mode: BlendMode,
) -> LutResult<Vec<Rgb>> {
    let expected = size * size * size;
    if lut_a.len() != expected || lut_b.len() != expected {
        return Err(LutError::InvalidData(format!(
            "Expected {} entries for size {}, got a={} b={}",
            expected,
            size,
            lut_a.len(),
            lut_b.len(),
        )));
    }

    let result: Vec<Rgb> = lut_a
        .iter()
        .zip(lut_b.iter())
        .map(|(a, b)| blend_rgb(a, b, mix, mode))
        .collect();

    Ok(result)
}

/// Blend two 1-D curves (interleaved `[r0, g0, b0, r1, g1, b1, ...]`) into a new curve.
///
/// Both curves must have the same `size`.
///
/// # Errors
///
/// Returns `LutError::InvalidData` if the curves have mismatched sizes.
pub fn blend_curve(
    curve_a: &[[f64; 3]],
    curve_b: &[[f64; 3]],
    mix: f64,
    mode: BlendMode,
) -> LutResult<Vec<[f64; 3]>> {
    if curve_a.len() != curve_b.len() {
        return Err(LutError::InvalidData(format!(
            "Curve sizes differ: a={} b={}",
            curve_a.len(),
            curve_b.len(),
        )));
    }

    let result: Vec<[f64; 3]> = curve_a
        .iter()
        .zip(curve_b.iter())
        .map(|(a, b)| blend_rgb(a, b, mix, mode))
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Spatially varying blend
// ---------------------------------------------------------------------------

/// A blend mask that provides per-pixel mix factors.
#[derive(Debug, Clone)]
pub struct BlendMask {
    /// Width of the mask.
    pub width: usize,
    /// Height of the mask.
    pub height: usize,
    /// Mix factors in row-major order, each in `[0, 1]`.
    pub data: Vec<f64>,
}

impl BlendMask {
    /// Create a uniform mask (same mix factor everywhere).
    #[must_use]
    pub fn uniform(width: usize, height: usize, mix: f64) -> Self {
        Self {
            width,
            height,
            data: vec![mix.clamp(0.0, 1.0); width * height],
        }
    }

    /// Create a horizontal gradient mask (left = 0.0, right = 1.0).
    #[must_use]
    pub fn horizontal_gradient(width: usize, height: usize) -> Self {
        let mut data = Vec::with_capacity(width * height);
        let scale = if width > 1 {
            1.0 / (width - 1) as f64
        } else {
            0.0
        };
        for _y in 0..height {
            for x in 0..width {
                data.push(x as f64 * scale);
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a vertical gradient mask (top = 0.0, bottom = 1.0).
    #[must_use]
    pub fn vertical_gradient(width: usize, height: usize) -> Self {
        let mut data = Vec::with_capacity(width * height);
        let scale = if height > 1 {
            1.0 / (height - 1) as f64
        } else {
            0.0
        };
        for y in 0..height {
            let val = y as f64 * scale;
            for _x in 0..width {
                data.push(val);
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a radial gradient mask centred in the frame.
    #[must_use]
    pub fn radial_gradient(width: usize, height: usize) -> Self {
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();
        let mut data = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                data.push(if max_r > 0.0 {
                    (r / max_r).clamp(0.0, 1.0)
                } else {
                    0.0
                });
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Get the mix factor at position `(x, y)`, returning 0.0 for out-of-bounds.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-LUT crossfade
// ---------------------------------------------------------------------------

/// Blend N LUTs with corresponding weights (weights are normalised internally).
///
/// All LUTs must have the same `size`.
///
/// # Errors
///
/// Returns `LutError::InvalidData` if LUTs/weights are empty or sizes differ.
pub fn blend_multi_lut3d(luts: &[&[Rgb]], weights: &[f64], size: usize) -> LutResult<Vec<Rgb>> {
    if luts.is_empty() || weights.is_empty() || luts.len() != weights.len() {
        return Err(LutError::InvalidData(
            "LUTs and weights must be non-empty and same length".to_string(),
        ));
    }

    let expected = size * size * size;
    for (i, lut) in luts.iter().enumerate() {
        if lut.len() != expected {
            return Err(LutError::InvalidData(format!(
                "LUT {i} has {} entries, expected {expected}",
                lut.len(),
            )));
        }
    }

    let weight_sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
    if weight_sum < 1e-15 {
        return Err(LutError::InvalidData("All weights are zero".to_string()));
    }

    let norm_weights: Vec<f64> = weights.iter().map(|w| w.max(0.0) / weight_sum).collect();

    let mut result = vec![[0.0, 0.0, 0.0]; expected];
    for (lut, &w) in luts.iter().zip(norm_weights.iter()) {
        for (i, entry) in lut.iter().enumerate() {
            result[i][0] += entry[0] * w;
            result[i][1] += entry[1] * w;
            result[i][2] += entry[2] * w;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut3d(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    fn constant_lut3d(size: usize, val: f64) -> Vec<Rgb> {
        vec![[val, val, val]; size * size * size]
    }

    fn rgb_close(a: &Rgb, b: &Rgb, tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    #[test]
    fn test_blend_linear_endpoints() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];

        let r0 = blend_rgb(&a, &b, 0.0, BlendMode::Linear);
        assert!(rgb_close(&r0, &a, 1e-10));

        let r1 = blend_rgb(&a, &b, 1.0, BlendMode::Linear);
        assert!(rgb_close(&r1, &b, 1e-10));
    }

    #[test]
    fn test_blend_linear_midpoint() {
        let a = [0.0, 0.2, 0.4];
        let b = [1.0, 0.8, 0.6];
        let mid = blend_rgb(&a, &b, 0.5, BlendMode::Linear);
        assert!(rgb_close(&mid, &[0.5, 0.5, 0.5], 1e-10));
    }

    #[test]
    fn test_blend_perceptual_endpoints() {
        let a = [0.5, 0.5, 0.5];
        let b = [0.5, 0.5, 0.5];
        let r = blend_rgb(&a, &b, 0.5, BlendMode::Perceptual);
        assert!(rgb_close(&r, &a, 1e-6));
    }

    #[test]
    fn test_blend_luminance_preserving() {
        let a = [0.5, 0.5, 0.5];
        let b = [0.5, 0.5, 0.5];
        let r = blend_rgb(&a, &b, 0.5, BlendMode::LuminancePreserving);
        assert!(rgb_close(&r, &a, 1e-6));
    }

    #[test]
    fn test_blend_hue_preserving_achromatic() {
        let a = [0.3, 0.3, 0.3];
        let b = [0.7, 0.7, 0.7];
        let r = blend_rgb(&a, &b, 0.5, BlendMode::HuePreserving);
        // Should be roughly 0.5 for all channels (achromatic)
        assert!(rgb_close(&r, &[0.5, 0.5, 0.5], 0.01));
    }

    #[test]
    fn test_blend_clamps_mix_factor() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];

        let r_neg = blend_rgb(&a, &b, -1.0, BlendMode::Linear);
        assert!(rgb_close(&r_neg, &a, 1e-10));

        let r_over = blend_rgb(&a, &b, 2.0, BlendMode::Linear);
        assert!(rgb_close(&r_over, &b, 1e-10));
    }

    #[test]
    fn test_blend_lut3d_linear() {
        let size = 3;
        let lut_a = identity_lut3d(size);
        let lut_b = constant_lut3d(size, 0.5);

        let blended = blend_lut3d(&lut_a, &lut_b, size, 0.0, BlendMode::Linear)
            .expect("blend should succeed");
        // At t=0 should equal lut_a
        assert!(rgb_close(&blended[0], &lut_a[0], 1e-10));

        let blended1 = blend_lut3d(&lut_a, &lut_b, size, 1.0, BlendMode::Linear)
            .expect("blend should succeed");
        assert!(rgb_close(&blended1[0], &lut_b[0], 1e-10));
    }

    #[test]
    fn test_blend_lut3d_size_mismatch() {
        let lut_a = identity_lut3d(3);
        let lut_b = identity_lut3d(5);
        let result = blend_lut3d(&lut_a, &lut_b, 3, 0.5, BlendMode::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_curve() {
        let curve_a: Vec<[f64; 3]> = (0..10)
            .map(|i| {
                let v = i as f64 / 9.0;
                [v, v, v]
            })
            .collect();
        let curve_b: Vec<[f64; 3]> = vec![[0.5, 0.5, 0.5]; 10];

        let blended =
            blend_curve(&curve_a, &curve_b, 0.5, BlendMode::Linear).expect("blend should succeed");
        assert_eq!(blended.len(), 10);
    }

    #[test]
    fn test_blend_curve_mismatch() {
        let curve_a: Vec<[f64; 3]> = vec![[0.0; 3]; 10];
        let curve_b: Vec<[f64; 3]> = vec![[0.0; 3]; 20];
        let result = blend_curve(&curve_a, &curve_b, 0.5, BlendMode::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_mask_uniform() {
        let mask = BlendMask::uniform(10, 10, 0.7);
        assert!((mask.get(0, 0) - 0.7).abs() < 1e-10);
        assert!((mask.get(9, 9) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_blend_mask_horizontal_gradient() {
        let mask = BlendMask::horizontal_gradient(11, 1);
        assert!((mask.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((mask.get(5, 0) - 0.5).abs() < 1e-10);
        assert!((mask.get(10, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blend_mask_vertical_gradient() {
        let mask = BlendMask::vertical_gradient(1, 11);
        assert!((mask.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((mask.get(0, 10) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blend_mask_radial() {
        let mask = BlendMask::radial_gradient(100, 100);
        // Center should be near 0
        assert!(mask.get(50, 50) < 0.1);
        // Corner should be near 1
        assert!(mask.get(0, 0) > 0.9);
    }

    #[test]
    fn test_blend_mask_out_of_bounds() {
        let mask = BlendMask::uniform(5, 5, 0.5);
        assert!((mask.get(10, 10) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blend_multi_lut3d_single() {
        let size = 3;
        let lut = identity_lut3d(size);
        let result = blend_multi_lut3d(&[&lut], &[1.0], size).expect("blend should succeed");
        assert!(rgb_close(&result[0], &lut[0], 1e-10));
    }

    #[test]
    fn test_blend_multi_lut3d_equal_weights() {
        let size = 3;
        let lut_a = constant_lut3d(size, 0.0);
        let lut_b = constant_lut3d(size, 1.0);
        let result =
            blend_multi_lut3d(&[&lut_a, &lut_b], &[1.0, 1.0], size).expect("blend should succeed");
        // Should be 0.5
        assert!(rgb_close(&result[0], &[0.5, 0.5, 0.5], 1e-10));
    }

    #[test]
    fn test_blend_multi_empty() {
        let result = blend_multi_lut3d(&[], &[], 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_multi_zero_weights() {
        let size = 3;
        let lut = identity_lut3d(size);
        let result = blend_multi_lut3d(&[&lut], &[0.0], size);
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_multi_three_luts() {
        let size = 3;
        let a = constant_lut3d(size, 0.0);
        let b = constant_lut3d(size, 0.5);
        let c = constant_lut3d(size, 1.0);
        let result =
            blend_multi_lut3d(&[&a, &b, &c], &[1.0, 2.0, 1.0], size).expect("blend should succeed");
        // Weighted average: (0*1 + 0.5*2 + 1*1) / 4 = 2.0/4 = 0.5
        assert!(rgb_close(&result[0], &[0.5, 0.5, 0.5], 1e-10));
    }
}
