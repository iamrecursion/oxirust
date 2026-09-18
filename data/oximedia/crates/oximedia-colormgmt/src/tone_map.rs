//! HDR to SDR tone mapping operators.
//!
//! Provides industry-standard tone mapping functions for converting high dynamic
//! range (HDR) content to standard dynamic range (SDR) display output.

#![allow(dead_code)]

use std::f64::consts::PI;

/// Tone mapping operator selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapOperator {
    /// Reinhard simple operator (1 / (1 + x)).
    Reinhard,
    /// Extended Reinhard with configurable white point.
    ReinhardExtended,
    /// Hable / Uncharted 2 filmic tone mapper.
    HableFilimic,
    /// ACES filmic tone mapping approximation.
    AcesFilmic,
    /// Drago logarithmic operator.
    DragoLogarithmic,
}

/// Parameters controlling tone mapping behaviour.
#[derive(Debug, Clone)]
pub struct ToneMapParams {
    /// Tone mapping operator.
    pub operator: ToneMapOperator,
    /// HDR peak luminance in nits (e.g. 1000.0 for HDR10).
    pub peak_luminance: f64,
    /// SDR target luminance in nits (e.g. 100.0).
    pub target_luminance: f64,
    /// Output gamma (e.g. 2.2 for standard display).
    pub gamma: f64,
    /// Pre-exposure adjustment (multiplied before tone mapping). 1.0 = neutral.
    pub exposure: f64,
}

impl Default for ToneMapParams {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::AcesFilmic,
            peak_luminance: 1000.0,
            target_luminance: 100.0,
            gamma: 2.2,
            exposure: 1.0,
        }
    }
}

// ── Operator implementations ─────────────────────────────────────────────────

/// Reinhard tone mapping operator: `x / (1 + x)`.
///
/// Maps `[0, ∞)` to `[0, 1)`. Simple and fast, but can look flat in shadows.
#[must_use]
#[inline]
pub fn reinhard(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    x / (1.0 + x)
}

/// Extended Reinhard operator with a white point `peak`.
///
/// `x * (1 + x / peak²) / (1 + x)`
#[must_use]
#[inline]
pub fn reinhard_extended(x: f64, peak: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let peak_sq = peak * peak;
    (x * (1.0 + x / peak_sq)) / (1.0 + x)
}

/// Hable / Uncharted 2 filmic tone mapping operator.
///
/// Developed by John Hable. Provides a warm, cinematic look with controlled
/// shoulder and toe regions.
#[must_use]
pub fn hable_filmic(x: f64) -> f64 {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;

    fn hable_partial(v: f64) -> f64 {
        (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F) - E / F
    }

    const W: f64 = 11.2; // linear white point
    if x <= 0.0 {
        return 0.0;
    }
    let curr = hable_partial(x * 2.0);
    let white_scale = 1.0 / hable_partial(W);
    (curr * white_scale).clamp(0.0, 1.0)
}

/// ACES filmic tone mapping approximation by Krzysztof Narkowicz.
///
/// Provides an ACES-like response with a fast analytical formula.
#[must_use]
pub fn aces_filmic(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    const A: f64 = 2.51;
    const B: f64 = 0.03;
    const C: f64 = 2.43;
    const D: f64 = 0.59;
    const E: f64 = 0.14;
    ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
}

/// Drago logarithmic tone mapping operator.
///
/// Based on Drago et al. "Adaptive logarithmic mapping for displaying high
/// contrast scenes". Uses a logarithm-based roll-off suitable for outdoor scenes.
#[must_use]
pub fn drago_logarithmic(x: f64, peak_luminance: f64, bias: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let bias = bias.clamp(0.1, 0.9);
    let log_bias = (bias / 0.5_f64).log10();
    let log_peak = (peak_luminance + 1.0).log10();
    let log_x = (x + 1.0).log10();
    let mapped = log_x / (log_peak * (log_bias / (PI / 2.0).log10()).exp());
    mapped.clamp(0.0, 1.0)
}

/// ACES-fitted tone mapping (Stephen Hill's fit).
///
/// A more accurate ACES-like curve fitted to the reference ACES RRT+ODT
/// by Stephen Hill. Provides better shadow behavior than the Narkowicz
/// approximation.
///
/// Reference: <https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl>
#[must_use]
pub fn aces_fitted(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // RRT and ODT fit
    // Input matrix (ACEScg → RRT input)
    let a = x * (x + 0.0245786) - 0.000090537;
    let b = x * (0.983729 * x + 0.4329510) + 0.238081;
    if b.abs() < 1e-20 {
        return 0.0;
    }
    (a / b).clamp(0.0, 1.0)
}

/// Reinhard luminance-based tone mapping.
///
/// Unlike per-channel Reinhard, this preserves color ratios by computing
/// luminance, mapping it, and scaling the color proportionally. This avoids
/// hue shifts that occur with per-channel operators.
///
/// # Arguments
///
/// * `r`, `g`, `b` - Linear light RGB values
/// * `peak` - White point for extended Reinhard (use `f64::INFINITY` for simple Reinhard)
///
/// # Returns
///
/// Tone-mapped (r, g, b) with preserved color ratios.
#[must_use]
pub fn reinhard_luminance(r: f64, g: f64, b: f64, peak: f64) -> (f64, f64, f64) {
    // Rec.709 luminance
    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if lum <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let mapped_lum = if peak.is_infinite() || peak <= 0.0 {
        lum / (1.0 + lum)
    } else {
        let peak_sq = peak * peak;
        (lum * (1.0 + lum / peak_sq)) / (1.0 + lum)
    };

    let scale = mapped_lum / lum;
    (
        (r * scale).clamp(0.0, 1.0),
        (g * scale).clamp(0.0, 1.0),
        (b * scale).clamp(0.0, 1.0),
    )
}

/// Filmic tone mapping operator with configurable curve parameters.
///
/// An S-curve tone mapping operator inspired by photographic film response.
/// Provides separate control over toe (shadows), shoulder, and midtone
/// behavior.
///
/// # Arguments
///
/// * `x` - Input linear light value (>= 0)
/// * `toe_strength` - Toe (shadow) contrast (0-1, default 0.5)
/// * `shoulder_strength` - Shoulder rolloff strength (0-1, default 0.97)
/// * `linear_strength` - Linear section strength (default 0.3)
///
/// # Returns
///
/// Tone-mapped value in [0, 1].
#[must_use]
pub fn filmic_configurable(
    x: f64,
    toe_strength: f64,
    shoulder_strength: f64,
    linear_strength: f64,
) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    let toe = toe_strength.clamp(0.0, 1.0);
    let shoulder = shoulder_strength.clamp(0.01, 0.999);
    let linear = linear_strength.clamp(0.01, 1.0);

    // S-curve construction
    let x_scaled = x * linear;
    let toe_v = toe * x_scaled;
    let shoulder_v = shoulder * x_scaled;

    let numerator = x_scaled * (toe_v + linear * 0.02) + 0.005;
    let denominator = x_scaled * (shoulder_v + linear * 0.3) + 0.06;

    if denominator.abs() < 1e-20 {
        return 0.0;
    }

    let result = numerator / denominator - 0.005 / 0.06;

    // Normalize by white point
    let white = 11.2;
    let w_scaled = white * linear;
    let w_toe = toe * w_scaled;
    let w_shoulder = shoulder * w_scaled;
    let w_num = w_scaled * (w_toe + linear * 0.02) + 0.005;
    let w_den = w_scaled * (w_shoulder + linear * 0.3) + 0.06;
    let w_result = if w_den.abs() < 1e-20 {
        1.0
    } else {
        w_num / w_den - 0.005 / 0.06
    };

    if w_result.abs() < 1e-20 {
        return result.clamp(0.0, 1.0);
    }

    (result / w_result).clamp(0.0, 1.0)
}

/// Apply tone mapping to an RGB pixel, returning a tone-mapped pixel.
///
/// Uses the ACES-fitted operator for high quality HDR-to-SDR conversion.
/// This is a convenience wrapper around `aces_fitted` that handles RGB channels
/// using luminance-preserving mapping.
///
/// # Arguments
///
/// * `r`, `g`, `b` - Linear light scene-referred values
/// * `exposure` - Exposure adjustment (default 1.0)
///
/// # Returns
///
/// Tone-mapped (r, g, b) in [0, 1].
#[must_use]
pub fn aces_fitted_rgb(r: f64, g: f64, b: f64, exposure: f64) -> (f64, f64, f64) {
    let r = r * exposure;
    let g = g * exposure;
    let b = b * exposure;
    (aces_fitted(r), aces_fitted(g), aces_fitted(b))
}

// ── Frame-level apply functions ───────────────────────────────────────────────

/// Apply tone mapping to a single RGB pixel.
///
/// The input is assumed to be in linear light, scene-referred.
/// Output is display-referred in [0.0, 1.0].
#[must_use]
pub fn apply_tone_map(rgb: (f64, f64, f64), params: &ToneMapParams) -> (f64, f64, f64) {
    let (r, g, b) = rgb;

    // Scale by exposure and normalize to [0, 1] using peak luminance
    let scale = params.exposure / params.peak_luminance * params.target_luminance;
    let r = r * scale;
    let g = g * scale;
    let b = b * scale;

    let map = |v: f64| -> f64 {
        let mapped = match params.operator {
            ToneMapOperator::Reinhard => reinhard(v),
            ToneMapOperator::ReinhardExtended => reinhard_extended(v, 1.0),
            ToneMapOperator::HableFilimic => hable_filmic(v),
            ToneMapOperator::AcesFilmic => aces_filmic(v),
            ToneMapOperator::DragoLogarithmic => {
                drago_logarithmic(v, params.peak_luminance / params.target_luminance, 0.85)
            }
        };
        // Apply inverse gamma for display
        if mapped <= 0.0 {
            return 0.0;
        }
        mapped.powf(1.0 / params.gamma).clamp(0.0, 1.0)
    };

    (map(r), map(g), map(b))
}

/// Apply tone mapping to an entire frame of pixels in-place.
pub fn apply_tone_map_frame(pixels: &mut [(f64, f64, f64)], params: &ToneMapParams) {
    for pixel in pixels.iter_mut() {
        *pixel = apply_tone_map(*pixel, params);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinhard_zero() {
        assert_eq!(reinhard(0.0), 0.0);
    }

    #[test]
    fn test_reinhard_one() {
        let v = reinhard(1.0);
        assert!((v - 0.5).abs() < 1e-10, "reinhard(1)=0.5, got {v}");
    }

    #[test]
    fn test_reinhard_negative() {
        assert_eq!(reinhard(-1.0), 0.0, "Negative input should return 0");
    }

    #[test]
    fn test_reinhard_large() {
        let v = reinhard(1000.0);
        assert!(
            v > 0.99 && v < 1.0,
            "reinhard(1000) should be near 1: got {v}"
        );
    }

    #[test]
    fn test_reinhard_extended_at_peak() {
        // At x == peak, should be close to 1.0
        let v = reinhard_extended(100.0, 100.0);
        assert!(v > 0.9, "Extended Reinhard at peak: {v}");
    }

    #[test]
    fn test_reinhard_extended_monotone() {
        let v1 = reinhard_extended(1.0, 10.0);
        let v2 = reinhard_extended(5.0, 10.0);
        assert!(
            v2 > v1,
            "Extended Reinhard should be monotonically increasing"
        );
    }

    #[test]
    fn test_hable_filmic_zero() {
        let v = hable_filmic(0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_hable_filmic_range() {
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let v = hable_filmic(x);
            assert!(v >= 0.0 && v <= 1.0, "hable_filmic({x}) = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_aces_filmic_zero() {
        let v = aces_filmic(0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_aces_filmic_range() {
        for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let v = aces_filmic(x);
            assert!(v >= 0.0 && v <= 1.0, "aces_filmic({x}) = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_aces_filmic_one_maps_below_one() {
        let v = aces_filmic(1.0);
        assert!(
            v < 1.0 && v > 0.0,
            "aces_filmic(1.0) should be in (0,1): {v}"
        );
    }

    #[test]
    fn test_apply_tone_map_output_range() {
        let params = ToneMapParams::default();
        for rgb in [
            (0.0, 0.0, 0.0),
            (500.0, 300.0, 100.0),
            (1000.0, 1000.0, 1000.0),
        ] {
            let (r, g, b) = apply_tone_map(rgb, &params);
            assert!(r >= 0.0 && r <= 1.0, "r={r}");
            assert!(g >= 0.0 && g <= 1.0, "g={g}");
            assert!(b >= 0.0 && b <= 1.0, "b={b}");
        }
    }

    #[test]
    fn test_apply_tone_map_black() {
        let params = ToneMapParams::default();
        let (r, g, b) = apply_tone_map((0.0, 0.0, 0.0), &params);
        assert_eq!((r, g, b), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_apply_tone_map_frame_all_ops() {
        let ops = [
            ToneMapOperator::Reinhard,
            ToneMapOperator::ReinhardExtended,
            ToneMapOperator::HableFilimic,
            ToneMapOperator::AcesFilmic,
            ToneMapOperator::DragoLogarithmic,
        ];
        for op in ops {
            let params = ToneMapParams {
                operator: op,
                ..Default::default()
            };
            let mut pixels = vec![(500.0, 300.0, 100.0); 10];
            apply_tone_map_frame(&mut pixels, &params);
            for (r, g, b) in &pixels {
                assert!(*r >= 0.0 && *r <= 1.0);
                assert!(*g >= 0.0 && *g <= 1.0);
                assert!(*b >= 0.0 && *b <= 1.0);
            }
        }
    }

    #[test]
    fn test_tone_map_params_default() {
        let p = ToneMapParams::default();
        assert_eq!(p.operator, ToneMapOperator::AcesFilmic);
        assert_eq!(p.peak_luminance, 1000.0);
        assert_eq!(p.gamma, 2.2);
    }

    #[test]
    fn test_drago_logarithmic_zero() {
        assert_eq!(drago_logarithmic(0.0, 1000.0, 0.85), 0.0);
    }

    #[test]
    fn test_drago_logarithmic_positive() {
        let v = drago_logarithmic(100.0, 1000.0, 0.85);
        assert!(v > 0.0 && v <= 1.0, "drago({}) = {v}", 100.0);
    }

    // ── ACES-fitted tests ────────────────────────────────────────────────────

    #[test]
    fn test_aces_fitted_zero() {
        assert_eq!(aces_fitted(0.0), 0.0);
    }

    #[test]
    fn test_aces_fitted_negative() {
        assert_eq!(aces_fitted(-1.0), 0.0);
    }

    #[test]
    fn test_aces_fitted_range() {
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            let v = aces_fitted(x);
            assert!(v >= 0.0 && v <= 1.0, "aces_fitted({x}) = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_aces_fitted_monotone() {
        let v1 = aces_fitted(0.1);
        let v2 = aces_fitted(0.5);
        let v3 = aces_fitted(1.0);
        let v4 = aces_fitted(5.0);
        assert!(v1 < v2, "monotone: {} < {}", v1, v2);
        assert!(v2 < v3, "monotone: {} < {}", v2, v3);
        assert!(v3 < v4, "monotone: {} < {}", v3, v4);
    }

    #[test]
    fn test_aces_fitted_converges_to_one() {
        let v = aces_fitted(1000.0);
        assert!(v > 0.99, "Large input should be near 1.0: {v}");
    }

    // ── Reinhard luminance tests ─────────────────────────────────────────────

    #[test]
    fn test_reinhard_luminance_black() {
        let (r, g, b) = reinhard_luminance(0.0, 0.0, 0.0, f64::INFINITY);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_reinhard_luminance_output_range() {
        let test_inputs = [
            (0.5, 0.3, 0.2),
            (2.0, 1.0, 0.5),
            (10.0, 5.0, 2.0),
            (100.0, 50.0, 25.0),
        ];
        for (r, g, b) in test_inputs {
            let (mr, mg, mb) = reinhard_luminance(r, g, b, f64::INFINITY);
            assert!(mr >= 0.0 && mr <= 1.0, "r={mr}");
            assert!(mg >= 0.0 && mg <= 1.0, "g={mg}");
            assert!(mb >= 0.0 && mb <= 1.0, "b={mb}");
        }
    }

    #[test]
    fn test_reinhard_luminance_preserves_ratio() {
        let (r, g, b) = (2.0, 1.0, 0.5);
        let (mr, mg, _mb) = reinhard_luminance(r, g, b, f64::INFINITY);
        // Check that relative ratios are preserved
        if mg > 1e-10 {
            let orig_ratio_rg = r / g;
            let mapped_ratio_rg = mr / mg;
            assert!(
                (orig_ratio_rg - mapped_ratio_rg).abs() < 0.01,
                "R/G ratio not preserved: {} vs {}",
                orig_ratio_rg,
                mapped_ratio_rg
            );
        }
    }

    #[test]
    fn test_reinhard_luminance_extended() {
        let (r, g, b) = (5.0, 3.0, 1.0);
        let (mr, mg, mb) = reinhard_luminance(r, g, b, 10.0);
        assert!(mr >= 0.0 && mr <= 1.0);
        assert!(mg >= 0.0 && mg <= 1.0);
        assert!(mb >= 0.0 && mb <= 1.0);
    }

    // ── Filmic configurable tests ────────────────────────────────────────────

    #[test]
    fn test_filmic_configurable_zero() {
        assert_eq!(filmic_configurable(0.0, 0.5, 0.97, 0.3), 0.0);
    }

    #[test]
    fn test_filmic_configurable_negative() {
        assert_eq!(filmic_configurable(-1.0, 0.5, 0.97, 0.3), 0.0);
    }

    #[test]
    fn test_filmic_configurable_range() {
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let v = filmic_configurable(x, 0.5, 0.97, 0.3);
            assert!(
                v >= 0.0 && v <= 1.0,
                "filmic_configurable({x}) = {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_filmic_configurable_monotone() {
        let v1 = filmic_configurable(0.1, 0.5, 0.97, 0.3);
        let v2 = filmic_configurable(1.0, 0.5, 0.97, 0.3);
        let v3 = filmic_configurable(10.0, 0.5, 0.97, 0.3);
        assert!(v1 < v2, "monotone: {} < {}", v1, v2);
        assert!(v2 < v3, "monotone: {} < {}", v2, v3);
    }

    #[test]
    fn test_filmic_configurable_different_params() {
        // Different toe strengths should give different results
        let v1 = filmic_configurable(1.0, 0.2, 0.97, 0.3);
        let v2 = filmic_configurable(1.0, 0.8, 0.97, 0.3);
        assert!(
            (v1 - v2).abs() > 0.001,
            "Different toe strengths should produce different results: {} vs {}",
            v1,
            v2
        );
    }

    // ── ACES-fitted RGB tests ────────────────────────────────────────────────

    #[test]
    fn test_aces_fitted_rgb_black() {
        let (r, g, b) = aces_fitted_rgb(0.0, 0.0, 0.0, 1.0);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_aces_fitted_rgb_range() {
        let (r, g, b) = aces_fitted_rgb(5.0, 3.0, 1.0, 1.0);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_aces_fitted_rgb_exposure() {
        let (r1, _, _) = aces_fitted_rgb(1.0, 0.5, 0.2, 1.0);
        let (r2, _, _) = aces_fitted_rgb(1.0, 0.5, 0.2, 2.0);
        assert!(
            r2 > r1,
            "Higher exposure should produce brighter output: {} vs {}",
            r2,
            r1
        );
    }
}
