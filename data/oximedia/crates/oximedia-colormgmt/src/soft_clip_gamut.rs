//! Soft-clip gamut mapping in CIELAB L\*a\*b\* space.
//!
//! Instead of hard-clamping out-of-gamut colours, this module progressively
//! desaturates pixels towards the achromatic axis while preserving lightness
//! (L\*) exactly.  The compression curve is a smooth quadratic shoulder that
//! begins at a configurable knee point and asymptotically approaches a
//! maximum saturation ceiling.
//!
//! # Algorithm
//!
//! For a pixel (L\*, a\*, b\*):
//!
//! 1. Compute `C* = hypot(a*, b*)`.
//! 2. Estimate the sRGB gamut boundary in Lab space:
//!    `C_max(L) = (1 − L/100) × 128 + 20`
//!    (a conservative approximation; real boundary varies with hue angle).
//! 3. Compute the knee threshold: `knee = config.knee_point × C_max`.
//! 4. If `C* ≤ knee` the pixel is considered in-gamut — return unchanged.
//! 5. Otherwise apply a quadratic de-Casteljau shoulder:
//!    - `t = (C* − knee) / (C_max − knee)`, clamped to \[0, 1\].
//!    - `smooth(t) = t × (2 − t)`   (maps 0→0, 1→1, slope 2 at 0, 0 at 1).
//!    - `compressed_C = knee + (config.max_saturation × C_max − knee) × smooth(t)`
//! 6. Scale `a*` and `b*` by `compressed_C / C*` to preserve hue angle.
//!    `L*` is always returned unchanged.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::soft_clip_gamut::{SoftClipConfig, SoftClipGamutMapper};
//!
//! let mapper = SoftClipGamutMapper::with_defaults();
//!
//! // Neutral grey: untouched
//! let grey = mapper.map_pixel([50.0, 0.0, 0.0]);
//! assert!((grey[0] - 50.0).abs() < 1e-6);
//!
//! // Out-of-gamut red: chroma reduced
//! let vivid_red = mapper.map_pixel([50.0, 150.0, 80.0]);
//! let c_in  = f32::hypot(150.0, 80.0);
//! let c_out = f32::hypot(vivid_red[1], vivid_red[2]);
//! assert!(c_out < c_in);
//! ```

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the soft-clip gamut mapper.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftClipConfig {
    /// Normalised chroma threshold at which compression begins.
    ///
    /// Expressed as a fraction of the estimated gamut boundary `C_max(L)`.
    /// Range: 0.0–1.0.  Default: `0.8`.
    ///
    /// - `0.0` → all colours are compressed (no in-gamut region).
    /// - `1.0` → only colours that exceed `C_max` are compressed.
    pub knee_point: f32,

    /// Hard ceiling on output chroma as a fraction of `C_max(L)`.
    ///
    /// The compressed chroma will never exceed `max_saturation × C_max`.
    /// Default: `1.0`.
    pub max_saturation: f32,
}

impl Default for SoftClipConfig {
    fn default() -> Self {
        Self {
            knee_point: 0.8,
            max_saturation: 1.0,
        }
    }
}

impl SoftClipConfig {
    /// Create a configuration with explicit knee point and saturation ceiling.
    ///
    /// Both values are clamped to \[0.0, 2.0\] to prevent degenerate cases.
    #[must_use]
    pub fn new(knee_point: f32, max_saturation: f32) -> Self {
        Self {
            knee_point: knee_point.clamp(0.0, 2.0),
            max_saturation: max_saturation.clamp(0.0, 2.0),
        }
    }
}

// ── Gamut boundary estimate ───────────────────────────────────────────────────

/// Estimate the approximate sRGB gamut boundary chroma `C_max` for a given
/// lightness `L*` (range 0–100).
///
/// This is a linear approximation: darker tones have a smaller gamut than
/// lighter ones, and very bright / very dark tones converge towards zero.
/// The formula `(1 − L/100) × 128 + 20` is intentionally conservative — it
/// under-estimates the real boundary at many hue angles so that the mapper
/// always applies some compression to clearly out-of-gamut signals.
#[inline]
fn c_max_for_lightness(l: f32) -> f32 {
    let l_norm = l.clamp(0.0, 100.0) / 100.0;
    (1.0 - l_norm) * 128.0 + 20.0
}

/// Smooth quadratic shoulder curve: `t × (2 − t)`.
///
/// Maps `t ∈ [0, 1]` → `[0, 1]` with zero derivative at `t = 1` (no overshoot)
/// and derivative 2 at `t = 0` (gradual onset).
#[inline]
fn smooth_shoulder(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * (2.0 - t)
}

// ── Mapper ────────────────────────────────────────────────────────────────────

/// Soft-clip gamut mapper operating in CIELAB L\*a\*b\* space.
///
/// See the [module-level documentation](self) for a full description of the
/// algorithm.
#[derive(Debug, Clone)]
pub struct SoftClipGamutMapper {
    config: SoftClipConfig,
}

impl SoftClipGamutMapper {
    /// Create a mapper with the given configuration.
    #[must_use]
    pub fn new(config: SoftClipConfig) -> Self {
        Self { config }
    }

    /// Create a mapper with [`SoftClipConfig::default`] settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SoftClipConfig::default())
    }

    /// Return the active configuration.
    #[must_use]
    pub fn config(&self) -> &SoftClipConfig {
        &self.config
    }

    /// Map a single Lab pixel `[L*, a*, b*]`.
    ///
    /// `L*` is expected in 0–100; `a*` and `b*` in approximately −128–128.
    /// The returned `L*` is identical to the input; only `a*` / `b*` may change.
    #[must_use]
    pub fn map_pixel(&self, lab: [f32; 3]) -> [f32; 3] {
        let l = lab[0];
        let a = lab[1];
        let b = lab[2];

        let c = f32::hypot(a, b);

        // No chroma → neutral grey, always in-gamut
        if c < f32::EPSILON {
            return lab;
        }

        let c_max = c_max_for_lightness(l);
        let knee = self.config.knee_point * c_max;

        // In-gamut: return unchanged
        if c <= knee {
            return lab;
        }

        // Compute smooth compressed chroma
        let range = (c_max - knee).max(f32::EPSILON);
        let t = (c - knee) / range;
        let ceiling = self.config.max_saturation * c_max;
        let compressed_c = knee + (ceiling - knee) * smooth_shoulder(t);

        // Clamp to ceiling to be safe against extreme inputs
        let compressed_c = compressed_c.clamp(0.0, ceiling.max(0.0));

        // Scale a*, b* preserving hue angle; L* unchanged
        let scale = compressed_c / c;
        [l, a * scale, b * scale]
    }

    /// Map a slice of Lab pixels, returning a new `Vec`.
    ///
    /// Equivalent to calling [`map_pixel`](Self::map_pixel) on every element.
    #[must_use]
    pub fn map_pixels(&self, pixels: &[[f32; 3]]) -> Vec<[f32; 3]> {
        pixels.iter().map(|&px| self.map_pixel(px)).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Neutral greys (a*=0, b*=0) must always be unchanged ──────────────────

    #[test]
    fn test_neutral_grey_l0_unchanged() {
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixel([0.0, 0.0, 0.0]);
        assert!((out[0] - 0.0).abs() < 1e-6, "L* changed: {}", out[0]);
        assert!(out[1].abs() < 1e-6, "a* changed: {}", out[1]);
        assert!(out[2].abs() < 1e-6, "b* changed: {}", out[2]);
    }

    #[test]
    fn test_neutral_grey_l50_unchanged() {
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixel([50.0, 0.0, 0.0]);
        assert!((out[0] - 50.0).abs() < 1e-6, "L* changed");
        assert!(out[1].abs() < 1e-6, "a* changed");
        assert!(out[2].abs() < 1e-6, "b* changed");
    }

    #[test]
    fn test_neutral_grey_l100_unchanged() {
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixel([100.0, 0.0, 0.0]);
        assert!((out[0] - 100.0).abs() < 1e-6, "L* changed");
        assert!(out[1].abs() < 1e-6, "a* changed");
        assert!(out[2].abs() < 1e-6, "b* changed");
    }

    // ── In-gamut saturated colours must be unchanged ──────────────────────────

    #[test]
    fn test_in_gamut_small_chroma_unchanged() {
        // At L*=50, C_max ≈ (0.5)*128+20 = 84; knee = 0.8*84 ≈ 67.2
        // C* = 10 (well below knee)
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixel([50.0, 7.07, 7.07]); // C* ≈ 10
        let c_in = f32::hypot(7.07, 7.07);
        let c_out = f32::hypot(out[1], out[2]);
        assert!(
            (c_out - c_in).abs() < 1e-4,
            "In-gamut pixel changed: {c_in} → {c_out}"
        );
    }

    #[test]
    fn test_in_gamut_at_knee_boundary_unchanged() {
        let config = SoftClipConfig::new(0.8, 1.0);
        let m = SoftClipGamutMapper::new(config);
        // L*=50 → C_max≈84; knee=0.8*84=67.2; set C* slightly below knee
        let c_below_knee = 67.0_f32;
        let a = c_below_knee / f32::sqrt(2.0);
        let b = a;
        let out = m.map_pixel([50.0, a, b]);
        let c_out = f32::hypot(out[1], out[2]);
        assert!(
            (c_out - c_below_knee).abs() < 0.5,
            "Should be unchanged near knee: {c_out}"
        );
    }

    // ── Out-of-gamut colours must be desaturated ──────────────────────────────

    #[test]
    fn test_out_of_gamut_chroma_reduced() {
        let m = SoftClipGamutMapper::with_defaults();
        // Very large chroma — clearly out of gamut at L*=50
        let out = m.map_pixel([50.0, 150.0, 0.0]);
        let c_out = f32::hypot(out[1], out[2]);
        assert!(
            c_out < 150.0,
            "Out-of-gamut chroma should be reduced: {c_out}"
        );
    }

    #[test]
    fn test_out_of_gamut_lightness_preserved() {
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixel([70.0, 200.0, 100.0]);
        assert!(
            (out[0] - 70.0).abs() < 1e-5,
            "L* must not change: {}",
            out[0]
        );
    }

    #[test]
    fn test_extreme_out_of_gamut_capped_at_max_saturation() {
        let m = SoftClipGamutMapper::with_defaults(); // max_saturation=1.0
        let out = m.map_pixel([50.0, 5000.0, 0.0]);
        let c_out = f32::hypot(out[1], out[2]);
        let c_max = c_max_for_lightness(50.0);
        assert!(
            c_out <= c_max * 1.001,
            "Extreme out-of-gamut not capped: c_out={c_out} c_max={c_max}"
        );
    }

    // ── knee_point extremes ───────────────────────────────────────────────────

    #[test]
    fn test_knee_zero_all_colours_enter_compression_branch() {
        // knee=0 → every pixel with C* > 0 enters the compression branch
        // (no pixel is "in-gamut" and returned unchanged).
        // With max_saturation=0.5 the ceiling is 0.5*C_max, so large chroma
        // values that exceed C_max/2 must be reduced.
        let m = SoftClipGamutMapper::new(SoftClipConfig::new(0.0, 0.5));
        // At L*=50, C_max ≈ 84; ceiling = 0.5*84 = 42.
        // C*=80 should be compressed below 42.
        let out = m.map_pixel([50.0_f32, 80.0, 0.0]);
        let c_out = f32::hypot(out[1], out[2]);
        let c_max = c_max_for_lightness(50.0);
        let ceiling = 0.5 * c_max;
        assert!(
            c_out <= ceiling + 1e-3,
            "knee=0,max_sat=0.5 should cap chroma at ceiling={ceiling:.2}: got {c_out:.2}"
        );
    }

    #[test]
    fn test_knee_one_only_compresses_beyond_c_max() {
        // knee=1.0 → colours with C* < C_max must not be touched
        let m = SoftClipGamutMapper::new(SoftClipConfig::new(1.0, 1.0));
        let c_max_l50 = c_max_for_lightness(50.0); // ≈ 84
                                                   // C* = 0.95 * C_max — clearly inside
        let c = 0.95 * c_max_l50;
        let out = m.map_pixel([50.0, c, 0.0]);
        let c_out = f32::hypot(out[1], out[2]);
        assert!(
            (c_out - c).abs() < 0.5,
            "knee=1.0 should not touch C*<C_max: {c}→{c_out}"
        );
    }

    // ── map_pixels slice ──────────────────────────────────────────────────────

    #[test]
    fn test_map_pixels_matches_map_pixel() {
        let m = SoftClipGamutMapper::with_defaults();
        let pixels: Vec<[f32; 3]> = vec![
            [50.0, 0.0, 0.0],
            [70.0, 100.0, 50.0],
            [30.0, -80.0, 60.0],
            [90.0, 20.0, -10.0],
        ];
        let batch = m.map_pixels(&pixels);
        for (i, &px) in pixels.iter().enumerate() {
            let expected = m.map_pixel(px);
            assert!(
                (batch[i][0] - expected[0]).abs() < 1e-6,
                "L* mismatch at {i}"
            );
            assert!(
                (batch[i][1] - expected[1]).abs() < 1e-6,
                "a* mismatch at {i}"
            );
            assert!(
                (batch[i][2] - expected[2]).abs() < 1e-6,
                "b* mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_map_pixels_empty_returns_empty() {
        let m = SoftClipGamutMapper::with_defaults();
        let out = m.map_pixels(&[]);
        assert!(out.is_empty(), "Empty slice should return empty Vec");
    }

    // ── Hue angle preservation ────────────────────────────────────────────────

    #[test]
    fn test_hue_angle_preserved_for_out_of_gamut() {
        let m = SoftClipGamutMapper::with_defaults();
        // out-of-gamut vivid colour at L*=50
        let a_in = 120.0_f32;
        let b_in = 80.0_f32;
        let hue_in = f32::atan2(b_in, a_in);
        let out = m.map_pixel([50.0, a_in, b_in]);
        let hue_out = f32::atan2(out[2], out[1]);
        assert!(
            (hue_out - hue_in).abs() < 1e-4,
            "Hue angle changed: {hue_in:.6} → {hue_out:.6}"
        );
    }

    #[test]
    fn test_hue_angle_preserved_negative_quadrant() {
        let m = SoftClipGamutMapper::with_defaults();
        let a_in = -130.0_f32;
        let b_in = -90.0_f32;
        let hue_in = f32::atan2(b_in, a_in);
        let out = m.map_pixel([40.0, a_in, b_in]);
        let hue_out = f32::atan2(out[2], out[1]);
        assert!(
            (hue_out - hue_in).abs() < 1e-4,
            "Hue angle changed in negative quadrant: {hue_in:.6} → {hue_out:.6}"
        );
    }

    // ── Monotonicity of compression ───────────────────────────────────────────

    #[test]
    fn test_monotonic_compression_with_increasing_chroma() {
        let m = SoftClipGamutMapper::with_defaults();
        // Progressively more out-of-gamut → output C* must be monotonically non-decreasing
        let chroma_values = [20.0_f32, 60.0, 100.0, 150.0, 250.0, 500.0];
        let mut prev_c_out = -1.0_f32;
        for &c_in in &chroma_values {
            let out = m.map_pixel([50.0, c_in, 0.0]);
            let c_out = f32::hypot(out[1], out[2]);
            assert!(
                c_out >= prev_c_out - 1e-4,
                "Output chroma should be monotonically non-decreasing: prev={prev_c_out} current={c_out} (c_in={c_in})"
            );
            prev_c_out = c_out;
        }
    }

    // ── Config accessors ──────────────────────────────────────────────────────

    #[test]
    fn test_config_new_clamps_values() {
        let cfg = SoftClipConfig::new(-0.5, 5.0);
        assert!(cfg.knee_point >= 0.0, "knee_point should be clamped");
        assert!(
            cfg.max_saturation <= 2.0,
            "max_saturation should be clamped"
        );
    }

    #[test]
    fn test_default_config_values() {
        let cfg = SoftClipConfig::default();
        assert!((cfg.knee_point - 0.8).abs() < 1e-6);
        assert!((cfg.max_saturation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mapper_config_accessor() {
        let cfg = SoftClipConfig::new(0.7, 0.9);
        let m = SoftClipGamutMapper::new(cfg.clone());
        assert!((m.config().knee_point - 0.7).abs() < 1e-6);
        assert!((m.config().max_saturation - 0.9).abs() < 1e-6);
    }
}
