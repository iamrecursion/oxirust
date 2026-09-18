//! Tone mapping operators for HDR to SDR conversion.

/// Tone mapping operator selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneMappingOperator {
    /// Reinhard tone mapping (simple global operator)
    Reinhard,
    /// Modified Reinhard with white point
    ReinhardExtended,
    /// Hable (Uncharted 2) filmic tone mapping
    Hable,
    /// ACES filmic tone mapping
    Aces,
    /// Simple linear clamp
    Linear,
    /// Custom curve
    Custom,
}

/// Tone mapper for converting HDR to SDR.
#[derive(Clone, Debug)]
pub struct ToneMapper {
    operator: ToneMappingOperator,
    peak_input_nits: f64,
    peak_output_nits: f64,
    white_point: f64,
}

impl ToneMapper {
    /// Creates a new tone mapper.
    ///
    /// # Arguments
    ///
    /// * `operator` - Tone mapping operator to use
    /// * `peak_input_nits` - Peak luminance of input HDR content
    /// * `peak_output_nits` - Peak luminance of output display (typically 100 for SDR)
    #[must_use]
    pub const fn new(
        operator: ToneMappingOperator,
        peak_input_nits: f64,
        peak_output_nits: f64,
    ) -> Self {
        Self {
            operator,
            peak_input_nits,
            peak_output_nits,
            white_point: 1.0,
        }
    }

    /// Sets the white point for extended Reinhard.
    pub fn set_white_point(&mut self, white_point: f64) {
        self.white_point = white_point;
    }

    /// Applies tone mapping to linear HDR RGB.
    #[must_use]
    pub fn apply(&self, hdr_rgb: [f64; 3]) -> [f64; 3] {
        // Normalize to [0, 1] range where 1.0 = peak_output_nits
        let scale = self.peak_output_nits / self.peak_input_nits;
        let normalized = [hdr_rgb[0] * scale, hdr_rgb[1] * scale, hdr_rgb[2] * scale];

        match self.operator {
            ToneMappingOperator::Reinhard => apply_reinhard(normalized),
            ToneMappingOperator::ReinhardExtended => {
                apply_reinhard_extended(normalized, self.white_point)
            }
            ToneMappingOperator::Hable => apply_hable(normalized),
            ToneMappingOperator::Aces => apply_aces(normalized),
            ToneMappingOperator::Linear => apply_linear(normalized),
            ToneMappingOperator::Custom => normalized, // No-op, user provides custom
        }
    }
}

impl Default for ToneMapper {
    fn default() -> Self {
        Self::new(ToneMappingOperator::Aces, 1000.0, 100.0)
    }
}

/// Reinhard tone mapping operator.
///
/// Formula: `L_out` = `L_in` / (1 + `L_in`)
///
/// Simple and efficient, but can make highlights look flat.
#[must_use]
fn apply_reinhard(rgb: [f64; 3]) -> [f64; 3] {
    [
        rgb[0] / (1.0 + rgb[0]),
        rgb[1] / (1.0 + rgb[1]),
        rgb[2] / (1.0 + rgb[2]),
    ]
}

/// Extended Reinhard tone mapping with white point.
///
/// Formula: `L_out` = `L_in` * (1 + `L_in/L_white^2`) / (1 + `L_in`)
#[must_use]
fn apply_reinhard_extended(rgb: [f64; 3], white_point: f64) -> [f64; 3] {
    let white_sq = white_point * white_point;
    let map = |v: f64| (v * (1.0 + v / white_sq)) / (1.0 + v);

    [map(rgb[0]), map(rgb[1]), map(rgb[2])]
}

/// Hable (Uncharted 2) filmic tone mapping.
///
/// Popular filmic curve with good highlight handling.
#[must_use]
fn apply_hable(rgb: [f64; 3]) -> [f64; 3] {
    const EXPOSURE_BIAS: f64 = 2.0;
    let curr = hable_partial(
        rgb[0] * EXPOSURE_BIAS,
        rgb[1] * EXPOSURE_BIAS,
        rgb[2] * EXPOSURE_BIAS,
    );
    let white = hable_partial(11.2, 11.2, 11.2);

    [curr[0] / white[0], curr[1] / white[1], curr[2] / white[2]]
}

#[must_use]
fn hable_partial(r: f64, g: f64, b: f64) -> [f64; 3] {
    const A: f64 = 0.15;
    const B: f64 = 0.50;
    const C: f64 = 0.10;
    const D: f64 = 0.20;
    const E: f64 = 0.02;
    const F: f64 = 0.30;

    let map = |x: f64| ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

    [map(r), map(g), map(b)]
}

/// ACES filmic tone mapping.
///
/// Industry-standard filmic curve from ACES.
#[must_use]
fn apply_aces(rgb: [f64; 3]) -> [f64; 3] {
    let map = |x: f64| {
        const A: f64 = 2.51;
        const B: f64 = 0.03;
        const C: f64 = 2.43;
        const D: f64 = 0.59;
        const E: f64 = 0.14;

        ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
    };

    [map(rgb[0]), map(rgb[1]), map(rgb[2])]
}

/// Linear tone mapping (simple clamp).
#[must_use]
fn apply_linear(rgb: [f64; 3]) -> [f64; 3] {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

// ── f32 lightweight tone curve API ────────────────────────────────────────────

/// Lightweight f32-based tone-mapping curve selection.
///
/// Unlike [`ToneMapper`] (which wraps nit-based normalization and f64 math),
/// `ToneCurve` operates directly on linear-light f32 scene values in [0, ∞)
/// and returns display-referred values in [0, 1].  It is the preferred choice
/// when you already have a normalised [0, ∞) input and want a compact, branch-
/// free per-pixel operator.
///
/// # Variants
///
/// * `ReinhardSimple` — classic per-channel `x / (1 + x)`.
/// * `ReinhardExtended { l_white }` — luminance-preserving extended Reinhard
///   that maps `l_white` exactly to 1.0:
///   `x * (1 + x / l_white²) / (1 + x)`.
/// * `FilmicHable` — Hable/Uncharted-2 filmic curve (A=0.15, B=0.50, C=0.10,
///   D=0.20, E=0.02, F=0.30, W=11.2) **without** exposure bias.
/// * `AcesFitted` — Narkowicz ACES fitted rational (a=2.51, b=0.03, c=2.43,
///   d=0.59, e=0.14), output clamped to [0, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToneCurve {
    /// Simple Reinhard: `x / (1 + x)` per channel.
    ReinhardSimple,
    /// Extended Reinhard with explicit white point.
    ///
    /// At `x = l_white` the operator outputs exactly 1.0.
    ReinhardExtended {
        /// White point luminance (must be > 0).
        l_white: f32,
    },
    /// Hable/Uncharted-2 filmic curve (no exposure bias, W=11.2).
    FilmicHable,
    /// Narkowicz ACES-fitted rational approximation.
    AcesFitted,
}

impl ToneCurve {
    /// Maps a linear-light RGB triple through the selected tone curve.
    ///
    /// Input values are per-channel scene-linear in [0, ∞).  Output values
    /// are display-referred in [0, 1].
    ///
    /// # Arguments
    ///
    /// * `rgb` — linear-light scene values `[r, g, b]`.
    ///
    /// # Returns
    ///
    /// Display-referred `[r, g, b]` in [0, 1].
    #[must_use]
    pub fn map(&self, rgb: [f32; 3]) -> [f32; 3] {
        match self {
            Self::ReinhardSimple => rgb.map(|c| c / (1.0 + c)),
            Self::ReinhardExtended { l_white } => {
                let l_white = *l_white;
                let white_sq = l_white * l_white;
                rgb.map(|c| c * (1.0 + c / white_sq) / (1.0 + c))
            }
            Self::FilmicHable => {
                const W: f32 = 11.2;
                let denom = tone_curve_hable_partial(W);
                rgb.map(|c| tone_curve_hable_partial(c) / denom)
            }
            Self::AcesFitted => rgb.map(tone_curve_aces_fitted),
        }
    }
}

/// Hable/Uncharted-2 partial function (internal, no exposure bias).
///
/// Parameters: A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30.
#[must_use]
#[inline]
fn tone_curve_hable_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// Narkowicz ACES-fitted rational approximation.
///
/// Parameters: a=2.51, b=0.03, c=2.43, d=0.59, e=0.14.  Output clamped to
/// [0, 1] to handle negative values near zero.
#[must_use]
#[inline]
fn tone_curve_aces_fitted(x: f32) -> f32 {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    let v = (x * (A * x + B)) / (x * (C * x + D) + E);
    v.clamp(0.0, 1.0)
}

// ── soft-knee utility ──────────────────────────────────────────────────────────

/// Applies a smooth knee function for soft clipping.
///
/// This is useful for highlight protection and smooth rolloff.
#[must_use]
#[allow(dead_code)]
pub fn apply_soft_knee(rgb: [f64; 3], knee_start: f64, knee_end: f64) -> [f64; 3] {
    let apply_channel = |v: f64| -> f64 {
        if v < knee_start {
            v
        } else if v < knee_end {
            // Smooth knee using cosine
            let t = (v - knee_start) / (knee_end - knee_start);
            let smooth_t = (1.0 - (t * std::f64::consts::PI).cos()) / 2.0;
            knee_start + (knee_end - knee_start) * smooth_t
        } else {
            knee_end
        }
    };

    [
        apply_channel(rgb[0]),
        apply_channel(rgb[1]),
        apply_channel(rgb[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinhard_black_and_white() {
        let black = apply_reinhard([0.0, 0.0, 0.0]);
        assert_eq!(black, [0.0, 0.0, 0.0]);

        // High values should approach 1.0
        let bright = apply_reinhard([100.0, 100.0, 100.0]);
        assert!(bright[0] > 0.99);
    }

    #[test]
    fn test_tone_mapper() {
        let mapper = ToneMapper::new(ToneMappingOperator::Aces, 1000.0, 100.0);
        let hdr = [10.0, 5.0, 2.0]; // Bright HDR values
        let sdr = mapper.apply(hdr);

        // Should be mapped to [0, 1]
        assert!(sdr[0] >= 0.0 && sdr[0] <= 1.0);
        assert!(sdr[1] >= 0.0 && sdr[1] <= 1.0);
        assert!(sdr[2] >= 0.0 && sdr[2] <= 1.0);

        // Brighter values should map to brighter outputs
        assert!(sdr[0] > sdr[1]);
        assert!(sdr[1] > sdr[2]);
    }

    #[test]
    fn test_hable_tone_mapping() {
        let rgb = [0.5, 1.0, 2.0];
        let result = apply_hable(rgb);

        // All values should be in [0, 1]
        assert!(result[0] >= 0.0 && result[0] <= 1.0);
        assert!(result[1] >= 0.0 && result[1] <= 1.0);
        assert!(result[2] >= 0.0 && result[2] <= 1.0);
    }

    #[test]
    fn test_aces_tone_mapping() {
        let rgb = [0.5, 1.0, 2.0];
        let result = apply_aces(rgb);

        // All values should be in [0, 1]
        assert!(result[0] >= 0.0 && result[0] <= 1.0);
        assert!(result[1] >= 0.0 && result[1] <= 1.0);
        assert!(result[2] >= 0.0 && result[2] <= 1.0);

        // Should preserve order
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_linear_tone_mapping() {
        let rgb = [0.5, 1.5, -0.5];
        let result = apply_linear(rgb);

        assert_eq!(result, [0.5, 1.0, 0.0]);
    }

    // ── ToneCurve tests ────────────────────────────────────────────────────────

    #[test]
    fn tone_curve_reinhard_simple_identity_points() {
        let tc = ToneCurve::ReinhardSimple;
        let out = tc.map([1.0, 0.0, 10.0]);
        assert!(
            (out[0] - 0.5).abs() < 1e-5,
            "input 1.0 → 0.5, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.0).abs() < 1e-5,
            "input 0.0 → 0.0, got {}",
            out[1]
        );
        assert!(
            out[2] > 0.9 && out[2] < 1.0,
            "large input → near 1, got {}",
            out[2]
        );
    }

    #[test]
    fn tone_curve_reinhard_extended_white_point() {
        // At x = l_white: x * (1 + x / l_white²) / (1 + x)
        // = 4 * (1 + 4/16) / (1 + 4) = 4 * 1.25 / 5 = 1.0
        let tc = ToneCurve::ReinhardExtended { l_white: 4.0 };
        let out = tc.map([4.0, 4.0, 4.0]);
        for c in out {
            assert!((c - 1.0).abs() < 1e-4, "input l_white → 1.0, got {}", c);
        }
    }

    #[test]
    fn tone_curve_filmic_hable_monotone() {
        let tc = ToneCurve::FilmicHable;
        let vals: Vec<f32> = (0..100).map(|i| i as f32 * 0.12).collect();
        for window in vals.windows(2) {
            let a = tc.map([window[0]; 3])[0];
            let b = tc.map([window[1]; 3])[0];
            assert!(
                b >= a - 1e-5,
                "FilmicHable not monotone at {}: {} > {}",
                window[0],
                a,
                b
            );
        }
    }

    #[test]
    fn tone_curve_aces_fitted_known_point() {
        let tc = ToneCurve::AcesFitted;
        let out = tc.map([1.0, 1.0, 1.0]);
        // Narkowicz ACES(1.0) = (2.51 + 0.03) / (2.43 + 0.59 + 0.14) ≈ 0.8038
        for c in out {
            assert!((c - 0.803).abs() < 0.01, "ACES(1.0) ≈ 0.803, got {}", c);
        }
    }

    #[test]
    fn tone_curve_all_operators_monotone_property() {
        let operators = [
            ToneCurve::ReinhardSimple,
            ToneCurve::ReinhardExtended { l_white: 10.0 },
            ToneCurve::FilmicHable,
            ToneCurve::AcesFitted,
        ];
        let inputs: Vec<f32> = (0..1600).map(|i| i as f32 * 0.01).collect();
        for op in &operators {
            for w in inputs.windows(2) {
                let a = op.map([w[0]; 3])[0];
                let b = op.map([w[1]; 3])[0];
                assert!(
                    b >= a - 1e-5,
                    "{op:?} not monotone at {}: {} > {}",
                    w[0],
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn tone_curve_reinhard_simple_zero_preserving() {
        let tc = ToneCurve::ReinhardSimple;
        let out = tc.map([0.0, 0.0, 0.0]);
        for c in out {
            assert!((c - 0.0).abs() < 1e-7, "0 → 0, got {}", c);
        }
    }

    #[test]
    fn tone_curve_aces_output_range() {
        let tc = ToneCurve::AcesFitted;
        let test_vals = [0.0f32, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &v in &test_vals {
            let out = tc.map([v; 3]);
            for c in out {
                assert!(
                    c >= 0.0 && c <= 1.0,
                    "ACES output {c} out of [0,1] for input {v}"
                );
            }
        }
    }
}
