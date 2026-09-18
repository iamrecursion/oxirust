//! GPU tone mapping operations
//!
//! Pure-Rust (CPU-fallback) implementations of common HDR → SDR tone-mapping
//! operators. The API mirrors what a real GPU shader dispatch would look like
//! so that integration into the GPU pipeline is straightforward.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Supported tone-mapping algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TonemapAlgorithm {
    /// Simple Reinhard global operator
    Reinhard,
    /// Hable / Uncharted-2 filmic operator
    HableFilimic,
    /// Academy Colour Encoding System (ACES) approximation
    Aces,
    /// Drago logarithmic operator
    DragoLog,
}

/// Parameters for a tone-mapping pass
#[derive(Debug, Clone)]
pub struct TonemapParams {
    /// Tone-mapping operator to use
    pub algorithm: TonemapAlgorithm,
    /// Exposure multiplier applied before mapping (positive, typically 1.0)
    pub exposure: f32,
    /// Output gamma exponent (typically 2.2)
    pub gamma: f32,
    /// Scene peak luminance in nits (used by some operators)
    pub peak_luminance: f32,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            algorithm: TonemapAlgorithm::Reinhard,
            exposure: 1.0,
            gamma: 2.2,
            peak_luminance: 1000.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-pixel tone-mapping functions
// ──────────────────────────────────────────────────────────────────────────────

/// Reinhard global tone-mapping operator
///
/// Maps [0, ∞) → [0, 1) via `x / (1 + x)`.
#[must_use]
pub fn reinhard_tonemap(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (r / (1.0 + r), g / (1.0 + g), b / (1.0 + b))
}

/// Hable / Uncharted-2 filmic tone-mapping operator
///
/// The `exposure` parameter is applied before the curve.
#[must_use]
pub fn hable_tonemap(r: f32, g: f32, b: f32, exposure: f32) -> (f32, f32, f32) {
    let scale = exposure;
    let hable = |x: f32| -> f32 {
        // Standard Uncharted-2 coefficients
        let (a, b_c, c, d, e, f) = (0.15_f32, 0.50, 0.10, 0.20, 0.02, 0.30);
        (x * (a * x + c * b_c) + d * e) / (x * (a * x + b_c) + d * f) - e / f
    };
    let white = hable(11.2);
    let map = |v: f32| hable(v * scale) / white;
    (map(r), map(g), map(b))
}

/// ACES filmic tone-mapping approximation (Narkowicz 2015)
#[must_use]
pub fn aces_tonemap(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let aces = |x: f32| -> f32 {
        let (a, b_c, c, d, e) = (2.51_f32, 0.03, 2.43, 0.59, 0.14);
        ((x * (a * x + b_c)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    };
    (aces(r), aces(g), aces(b))
}

/// Drago logarithmic tone-mapping operator
///
/// Approximates log-encoded HDR using `peak_luminance` to anchor the white
/// point.
#[must_use]
pub fn drago_log_tonemap(r: f32, g: f32, b: f32, peak_luminance: f32) -> (f32, f32, f32) {
    let map = |v: f32| -> f32 {
        if v <= 0.0 || peak_luminance <= 0.0 {
            return 0.0;
        }
        let l = v / peak_luminance;
        (1.0 + (l * std::f32::consts::E).ln()) / (1.0 + (std::f32::consts::E).ln())
    };
    (map(r), map(g), map(b))
}

// ──────────────────────────────────────────────────────────────────────────────
// Frame-level helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Apply gamma encoding: `value^(1/gamma)`, clamped to [0, 1]
#[must_use]
pub fn apply_gamma(value: f32, gamma: f32) -> f32 {
    if gamma <= 0.0 {
        return value.clamp(0.0, 1.0);
    }
    value.clamp(0.0, 1.0).powf(1.0 / gamma)
}

/// Apply tone-mapping and gamma correction to an interleaved RGB(A) `f32` frame
///
/// `pixels` must contain `width * height * 3` (RGB) or `width * height * 4`
/// (RGBA) values. If the stride is 4, the alpha channel is passed through
/// unchanged.
///
/// # Panics
///
/// Panics if `pixels.len()` is not `width * height * 3` or `width * height * 4`.
pub fn apply_tonemap_frame(pixels: &mut [f32], width: u32, height: u32, params: &TonemapParams) {
    let n = (width * height) as usize;
    let stride = if pixels.len() == n * 4 { 4 } else { 3 };
    assert_eq!(
        pixels.len(),
        n * stride,
        "pixels.len() must equal width*height*stride"
    );

    for i in 0..n {
        let base = i * stride;
        let r = pixels[base] * params.exposure;
        let g = pixels[base + 1] * params.exposure;
        let b = pixels[base + 2] * params.exposure;

        let (tr, tg, tb) = match params.algorithm {
            TonemapAlgorithm::Reinhard => reinhard_tonemap(r, g, b),
            TonemapAlgorithm::HableFilimic => hable_tonemap(r, g, b, 1.0),
            TonemapAlgorithm::Aces => aces_tonemap(r, g, b),
            TonemapAlgorithm::DragoLog => drago_log_tonemap(r, g, b, params.peak_luminance),
        };

        pixels[base] = apply_gamma(tr, params.gamma);
        pixels[base + 1] = apply_gamma(tg, params.gamma);
        pixels[base + 2] = apply_gamma(tb, params.gamma);
        // alpha unchanged if stride == 4
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn test_reinhard_zero() {
        let (r, g, b) = reinhard_tonemap(0.0, 0.0, 0.0);
        assert!(r.abs() < EPS && g.abs() < EPS && b.abs() < EPS);
    }

    #[test]
    fn test_reinhard_large_input_approaches_one() {
        let (r, g, b) = reinhard_tonemap(1e6, 1e6, 1e6);
        assert!((1.0 - r).abs() < 1e-4);
        assert!((1.0 - g).abs() < 1e-4);
        assert!((1.0 - b).abs() < 1e-4);
    }

    #[test]
    fn test_reinhard_half_input() {
        // 0.5 / 1.5 ≈ 0.333…
        let (r, _, _) = reinhard_tonemap(0.5, 0.0, 0.0);
        assert!((r - 1.0 / 3.0).abs() < EPS);
    }

    #[test]
    fn test_hable_unity_exposure() {
        let (r, g, b) = hable_tonemap(1.0, 1.0, 1.0, 1.0);
        // Output must be in [0, 1]
        assert!((0.0..=1.0).contains(&r));
        assert!((0.0..=1.0).contains(&g));
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn test_aces_zero() {
        let (r, g, b) = aces_tonemap(0.0, 0.0, 0.0);
        // ACES at 0 → ≈ 0
        assert!(r < EPS && g < EPS && b < EPS);
    }

    #[test]
    fn test_aces_output_clamped() {
        let (r, g, b) = aces_tonemap(1e6, 1e6, 1e6);
        assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
    }

    #[test]
    fn test_drago_zero_input() {
        let (r, g, b) = drago_log_tonemap(0.0, 0.0, 0.0, 1000.0);
        assert!(r.abs() < EPS && g.abs() < EPS && b.abs() < EPS);
    }

    #[test]
    fn test_drago_output_range() {
        let (r, g, b) = drago_log_tonemap(500.0, 500.0, 500.0, 1000.0);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_apply_gamma_identity_at_one() {
        // gamma=1 → value^1 = value
        let v = apply_gamma(0.5, 1.0);
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn test_apply_gamma_clamps_above_one() {
        let v = apply_gamma(2.0, 2.2);
        assert!((v - 1.0).abs() < EPS);
    }

    #[test]
    fn test_apply_gamma_clamps_below_zero() {
        let v = apply_gamma(-1.0, 2.2);
        assert!(v.abs() < EPS);
    }

    #[test]
    fn test_apply_tonemap_frame_reinhard_rgb() {
        let mut pixels = vec![1.0_f32; 4 * 3]; // 4 pixels, RGB stride=3
        let params = TonemapParams {
            algorithm: TonemapAlgorithm::Reinhard,
            exposure: 1.0,
            gamma: 1.0,
            peak_luminance: 1000.0,
        };
        apply_tonemap_frame(&mut pixels, 2, 2, &params);
        // Reinhard(1.0) = 0.5; gamma=1 → 0.5
        for i in 0..4 {
            let base = i * 3;
            assert!((pixels[base] - 0.5).abs() < EPS, "pixel {i} r");
        }
    }

    #[test]
    fn test_apply_tonemap_frame_rgba_alpha_preserved() {
        // 1 pixel, RGBA
        let mut pixels = vec![1.0_f32, 1.0, 1.0, 0.75];
        let params = TonemapParams::default();
        apply_tonemap_frame(&mut pixels, 1, 1, &params);
        // Alpha (index 3) must be unchanged
        assert!((pixels[3] - 0.75).abs() < EPS);
    }
}
