//! Tone mapping operators for HDR/SDR conversion.
//!
//! This module provides various tone mapping algorithms for converting between
//! High Dynamic Range (HDR) and Standard Dynamic Range (SDR) content.
//!
//! # Operators
//!
//! - **Reinhard**: Simple and effective tone mapping
//! - **Extended Reinhard**: Reinhard with white point control
//! - **ACES**: Academy Color Encoding System tone curve
//! - **Hable (Uncharted 2)**: Film-like tone curve
//! - **Exposure**: Simple exposure adjustment
//! - **Linear**: Linear scaling

use crate::Rgb;

/// Tone mapping operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneMapOp {
    /// Reinhard tone mapping.
    Reinhard,
    /// Extended Reinhard with white point.
    ExtendedReinhard,
    /// ACES filmic tone curve.
    Aces,
    /// Hable (Uncharted 2) tone curve.
    Hable,
    /// Simple exposure adjustment.
    Exposure,
    /// Linear scaling.
    Linear,
}

/// Tone mapping parameters.
#[derive(Clone, Copy, Debug)]
pub struct ToneMapParams {
    /// Exposure adjustment (stops).
    pub exposure: f64,
    /// White point for extended Reinhard (0.0 = auto).
    pub white_point: f64,
    /// Contrast adjustment (1.0 = neutral).
    pub contrast: f64,
    /// Shoulder strength for Hable.
    pub shoulder_strength: f64,
    /// Linear strength for Hable.
    pub linear_strength: f64,
}

impl Default for ToneMapParams {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            white_point: 0.0,
            contrast: 1.0,
            shoulder_strength: 0.22,
            linear_strength: 0.3,
        }
    }
}

/// Apply tone mapping to an RGB color.
#[must_use]
pub fn tonemap(rgb: &Rgb, op: ToneMapOp, params: &ToneMapParams) -> Rgb {
    // Apply exposure first
    let exposed = apply_exposure(rgb, params.exposure);

    match op {
        ToneMapOp::Reinhard => reinhard(&exposed),
        ToneMapOp::ExtendedReinhard => extended_reinhard(&exposed, params.white_point),
        ToneMapOp::Aces => aces_filmic(&exposed),
        ToneMapOp::Hable => hable(&exposed, params),
        ToneMapOp::Exposure => exposed,
        ToneMapOp::Linear => linear_tonemap(&exposed),
    }
}

/// Apply exposure adjustment.
#[must_use]
fn apply_exposure(rgb: &Rgb, exposure_stops: f64) -> Rgb {
    let scale = 2.0_f64.powf(exposure_stops);
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Reinhard tone mapping.
///
/// Formula: `L_out = L_in / (1 + L_in)`
#[must_use]
pub fn reinhard(rgb: &Rgb) -> Rgb {
    [
        rgb[0] / (1.0 + rgb[0]),
        rgb[1] / (1.0 + rgb[1]),
        rgb[2] / (1.0 + rgb[2]),
    ]
}

/// Extended Reinhard tone mapping with white point control.
///
/// Formula: `L_out = L_in * (1 + L_in / L_white^2) / (1 + L_in)`
#[must_use]
pub fn extended_reinhard(rgb: &Rgb, white_point: f64) -> Rgb {
    let white = if white_point > 0.0 {
        white_point
    } else {
        // Auto white point: max luminance
        rgb[0].max(rgb[1]).max(rgb[2]).max(1.0) // Minimum white point of 1.0
    };

    let white_sq = white * white;

    [
        reinhard_channel(rgb[0], white_sq),
        reinhard_channel(rgb[1], white_sq),
        reinhard_channel(rgb[2], white_sq),
    ]
}

#[must_use]
fn reinhard_channel(value: f64, white_sq: f64) -> f64 {
    value * (1.0 + value / white_sq) / (1.0 + value)
}

/// ACES filmic tone curve.
///
/// This is a simplified version of the ACES RRT+ODT.
#[must_use]
pub fn aces_filmic(rgb: &Rgb) -> Rgb {
    [
        aces_channel(rgb[0]),
        aces_channel(rgb[1]),
        aces_channel(rgb[2]),
    ]
}

#[must_use]
fn aces_channel(x: f64) -> f64 {
    const A: f64 = 2.51;
    const B: f64 = 0.03;
    const C: f64 = 2.43;
    const D: f64 = 0.59;
    const E: f64 = 0.14;

    ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
}

/// Hable (Uncharted 2) tone curve.
///
/// This produces a film-like response with a smooth shoulder.
#[must_use]
pub fn hable(rgb: &Rgb, params: &ToneMapParams) -> Rgb {
    let exposure_bias = 2.0;
    let white_scale = 11.2;

    let curr = [
        hable_partial(rgb[0] * exposure_bias, params),
        hable_partial(rgb[1] * exposure_bias, params),
        hable_partial(rgb[2] * exposure_bias, params),
    ];

    let white_point = hable_partial(white_scale, params);

    [
        curr[0] / white_point,
        curr[1] / white_point,
        curr[2] / white_point,
    ]
}

#[must_use]
#[allow(clippy::many_single_char_names)]
fn hable_partial(x: f64, params: &ToneMapParams) -> f64 {
    let a = params.shoulder_strength; // Shoulder strength
    let b = params.linear_strength; // Linear strength
    let c = 0.10; // Linear angle
    let d = 0.20; // Toe strength
    let e = 0.02; // Toe numerator
    let f = 0.30; // Toe denominator

    ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
}

/// Linear tone mapping (simple clamp).
#[must_use]
pub fn linear_tonemap(rgb: &Rgb) -> Rgb {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

/// Inverse tone mapping (SDR to HDR).
///
/// This is an approximate inverse that expands dynamic range.
#[must_use]
pub fn inverse_tonemap(rgb: &Rgb, peak_luminance: f64) -> Rgb {
    let scale = peak_luminance;
    [
        inverse_tonemap_channel(rgb[0], scale),
        inverse_tonemap_channel(rgb[1], scale),
        inverse_tonemap_channel(rgb[2], scale),
    ]
}

#[must_use]
fn inverse_tonemap_channel(x: f64, scale: f64) -> f64 {
    // Inverse of Reinhard
    let expanded = x / (1.0 - x.clamp(0.0, 0.999));
    expanded * scale
}

/// Calculate luminance from RGB (Rec.709).
#[must_use]
pub fn luminance(rgb: &Rgb) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Luminance-preserving tone mapping.
///
/// Maps the luminance using the tone curve while preserving color ratios.
#[must_use]
pub fn luminance_preserving_tonemap(rgb: &Rgb, op: ToneMapOp, params: &ToneMapParams) -> Rgb {
    let luma = luminance(rgb);
    if luma < 1e-6 {
        return [0.0, 0.0, 0.0];
    }

    // Tonemap the luminance
    let luma_tonemapped = luminance(&tonemap(&[luma, luma, luma], op, params));

    // Scale RGB to preserve ratios
    let scale = luma_tonemapped / luma;
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinhard() {
        let rgb = [2.0, 1.5, 0.5];
        let result = reinhard(&rgb);
        assert!(result[0] > 0.0 && result[0] <= 1.0);
        assert!(result[1] > 0.0 && result[1] <= 1.0);
        assert!(result[2] > 0.0 && result[2] <= 1.0);
    }

    #[test]
    fn test_aces_filmic() {
        let rgb = [2.0, 1.5, 0.5];
        let result = aces_filmic(&rgb);
        assert!(result[0] >= 0.0 && result[0] <= 1.0);
        assert!(result[1] >= 0.0 && result[1] <= 1.0);
        assert!(result[2] >= 0.0 && result[2] <= 1.0);
    }

    #[test]
    fn test_exposure() {
        let rgb = [0.5, 0.5, 0.5];
        let result = apply_exposure(&rgb, 1.0); // +1 stop
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_luminance() {
        let rgb = [1.0, 1.0, 1.0];
        let luma = luminance(&rgb);
        assert!((luma - 1.0).abs() < 1e-10);

        let rgb = [0.0, 0.0, 0.0];
        let luma = luminance(&rgb);
        assert!(luma.abs() < 1e-10);
    }

    #[test]
    fn test_tonemap_preserves_black() {
        let rgb = [0.0, 0.0, 0.0];
        let params = ToneMapParams::default();

        // Test operators that preserve zero
        for op in &[ToneMapOp::Reinhard, ToneMapOp::Aces] {
            let result = tonemap(&rgb, *op, &params);
            assert!(
                result[0].abs() < 1e-6,
                "Tonemap {op:?} should preserve black"
            );
            assert!(
                result[1].abs() < 1e-6,
                "Tonemap {op:?} should preserve black"
            );
            assert!(
                result[2].abs() < 1e-6,
                "Tonemap {op:?} should preserve black"
            );
        }

        // Test operators that may have a toe (small offset at zero)
        for op in &[ToneMapOp::ExtendedReinhard, ToneMapOp::Hable] {
            let result = tonemap(&rgb, *op, &params);
            // These may not preserve perfect zero due to toe/offset
            assert!(result[0] < 0.2, "Tonemap {op:?} should preserve near-black");
            assert!(result[1] < 0.2, "Tonemap {op:?} should preserve near-black");
            assert!(result[2] < 0.2, "Tonemap {op:?} should preserve near-black");
        }
    }

    #[test]
    fn test_hable() {
        let rgb = [2.0, 1.5, 0.5];
        let params = ToneMapParams::default();
        let result = hable(&rgb, &params);
        assert!(result[0] >= 0.0 && result[0] <= 1.0);
        assert!(result[1] >= 0.0 && result[1] <= 1.0);
        assert!(result[2] >= 0.0 && result[2] <= 1.0);
    }
}
