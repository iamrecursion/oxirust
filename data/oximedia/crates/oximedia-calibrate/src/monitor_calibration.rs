//! Display / monitor calibration support.
//!
//! Models gamma curves, white-point targets, luminance ranges and evaluates
//! whether a measured display meets a calibration specification.

#![allow(dead_code)]

// ── GammaTarget ──────────────────────────────────────────────────────────────

/// Target gamma / EOTF for a display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GammaTarget {
    /// Power-law gamma 2.2 (common consumer display).
    Gamma22,
    /// Power-law gamma 2.4 (studio / broadcast).
    Gamma24,
    /// Power-law gamma 2.6 (cinema).
    Gamma26,
    /// IEC 61966-2-1 sRGB transfer function.
    Srgb,
    /// Linear (1.0) rec.709 output.
    LinRec709,
}

impl GammaTarget {
    /// Encode a normalised linear light value through this EOTF.
    ///
    /// Returns the corresponding display signal (0.0–1.0).
    #[must_use]
    pub fn encode(&self, linear: f32) -> f32 {
        let v = linear.clamp(0.0, 1.0);
        match self {
            Self::Gamma22 => v.powf(1.0 / 2.2),
            Self::Gamma24 => v.powf(1.0 / 2.4),
            Self::Gamma26 => v.powf(1.0 / 2.6),
            Self::Srgb => {
                if v <= 0.003_130_8 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Self::LinRec709 => v,
        }
    }

    /// Nominal exponent (for display/reporting purposes).
    #[must_use]
    pub fn nominal_gamma(&self) -> f32 {
        match self {
            Self::Gamma22 => 2.2,
            Self::Gamma24 => 2.4,
            Self::Gamma26 => 2.6,
            Self::Srgb => 2.2, // approximate
            Self::LinRec709 => 1.0,
        }
    }
}

// ── WhitePointTarget ─────────────────────────────────────────────────────────

/// Chromaticity target for the display white point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhitePointTarget {
    /// CIE Standard Illuminant D50 (5000 K).
    D50,
    /// CIE Standard Illuminant D55 (5500 K).
    D55,
    /// CIE Standard Illuminant D65 (6500 K).
    D65,
    /// CIE Standard Illuminant D75 (7500 K).
    D75,
    /// Native display white (no target).
    Native,
}

impl WhitePointTarget {
    /// Returns the CIE (x, y) chromaticity of this white point.
    #[must_use]
    pub fn xy(&self) -> (f32, f32) {
        match self {
            Self::D50 => (0.3457, 0.3585),
            Self::D55 => (0.3324, 0.3474),
            Self::D65 => (0.3127, 0.3290),
            Self::D75 => (0.2990, 0.3149),
            Self::Native => (0.3127, 0.3290), // treat as D65 for computation
        }
    }
}

// ── LuminanceTarget ───────────────────────────────────────────────────────────

/// Target luminance range for a display in cd/m² (nits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuminanceTarget {
    /// Target minimum luminance (black level).
    pub min_nits: f32,
    /// Target peak luminance (white level).
    pub max_nits: f32,
}

impl LuminanceTarget {
    /// Standard SDR target (0.1 – 100 nits).
    #[must_use]
    pub fn sdr() -> Self {
        Self {
            min_nits: 0.1,
            max_nits: 100.0,
        }
    }

    /// HDR400 target (0.02 – 400 nits).
    #[must_use]
    pub fn hdr400() -> Self {
        Self {
            min_nits: 0.02,
            max_nits: 400.0,
        }
    }

    /// HDR1000 target (0.005 – 1000 nits).
    #[must_use]
    pub fn hdr1000() -> Self {
        Self {
            min_nits: 0.005,
            max_nits: 1000.0,
        }
    }
}

// ── DisplayCalibrationSpec ────────────────────────────────────────────────────

/// Full specification for a display calibration target.
#[derive(Debug, Clone)]
pub struct DisplayCalibrationSpec {
    /// Target gamma / EOTF.
    pub gamma: GammaTarget,
    /// Target white point chromaticity.
    pub white_point: WhitePointTarget,
    /// Target luminance range.
    pub luminance: LuminanceTarget,
    /// Nominal colour space name (e.g. "sRGB", "DCI-P3", "Rec.2020").
    pub color_space: String,
}

impl DisplayCalibrationSpec {
    /// Create a standard sRGB / SDR spec.
    #[must_use]
    pub fn srgb_sdr() -> Self {
        Self {
            gamma: GammaTarget::Srgb,
            white_point: WhitePointTarget::D65,
            luminance: LuminanceTarget::sdr(),
            color_space: "sRGB".to_string(),
        }
    }
}

// ── MeasuredDisplayParams ────────────────────────────────────────────────────

/// Measured parameters from a colorimeter / spectrophotometer.
#[derive(Debug, Clone, Copy)]
pub struct MeasuredDisplayParams {
    /// Measured effective gamma exponent.
    pub actual_gamma: f32,
    /// Measured white point (x, y).
    pub actual_white_xy: (f32, f32),
    /// Measured peak luminance in nits.
    pub actual_max_nits: f32,
    /// Measured black-level luminance in nits.
    pub actual_min_nits: f32,
}

// ── CalibrationResult ────────────────────────────────────────────────────────

/// Evaluated calibration quality.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationResult {
    /// Absolute gamma error (|target − actual|).
    pub error_gamma: f32,
    /// ΔE between target and actual white point (using CIE 1976 uv).
    pub error_white_de: f32,
    /// Percentage error in peak luminance.
    pub error_max_luminance_pct: f32,
    /// Whether all metrics are within tolerances.
    pub passes: bool,
}

/// Tolerances for pass/fail decisions.
const TOLERANCE_GAMMA: f32 = 0.1;
const TOLERANCE_WHITE_DE: f32 = 3.0;
const TOLERANCE_LUMINANCE_PCT: f32 = 10.0;

impl CalibrationResult {
    /// Evaluate whether the measured parameters meet the specification.
    #[must_use]
    pub fn evaluate(spec: &DisplayCalibrationSpec, measured: &MeasuredDisplayParams) -> Self {
        let target_gamma = spec.gamma.nominal_gamma();
        let error_gamma = (target_gamma - measured.actual_gamma).abs();

        let (tx, ty) = spec.white_point.xy();
        let (mx, my) = measured.actual_white_xy;
        let error_white_de = cie_uv_delta_e(tx, ty, mx, my);

        let error_max_luminance_pct = if spec.luminance.max_nits > 0.0 {
            ((measured.actual_max_nits - spec.luminance.max_nits) / spec.luminance.max_nits * 100.0)
                .abs()
        } else {
            0.0
        };

        let passes = error_gamma <= TOLERANCE_GAMMA
            && error_white_de <= TOLERANCE_WHITE_DE
            && error_max_luminance_pct <= TOLERANCE_LUMINANCE_PCT;

        Self {
            error_gamma,
            error_white_de,
            error_max_luminance_pct,
            passes,
        }
    }
}

/// CIE 1976 (u'v') ΔE from two (x, y) chromaticity pairs.
fn cie_uv_delta_e(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let u1 = 4.0 * x1 / (-2.0 * x1 + 12.0 * y1 + 3.0);
    let v1 = 9.0 * y1 / (-2.0 * x1 + 12.0 * y1 + 3.0);
    let u2 = 4.0 * x2 / (-2.0 * x2 + 12.0 * y2 + 3.0);
    let v2 = 9.0 * y2 / (-2.0 * x2 + 12.0 * y2 + 3.0);
    let du = u1 - u2;
    let dv = v1 - v2;
    // Scale to approximate ΔE (not the same as CIELAB ΔE but a useful metric)
    ((du * du + dv * dv).sqrt()) * 100.0
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma22_encode_midtone() {
        let encoded = GammaTarget::Gamma22.encode(0.5);
        let expected = 0.5f32.powf(1.0 / 2.2);
        assert!((encoded - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gamma24_encode_zero_one() {
        assert!((GammaTarget::Gamma24.encode(0.0) - 0.0).abs() < 1e-6);
        assert!((GammaTarget::Gamma24.encode(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_encode_low() {
        // Values ≤ 0.003_130_8 should be linear
        let v = 0.001f32;
        let encoded = GammaTarget::Srgb.encode(v);
        assert!((encoded - v * 12.92).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_encode_high() {
        let v = 0.5f32;
        let encoded = GammaTarget::Srgb.encode(v);
        let expected = 1.055 * v.powf(1.0 / 2.4) - 0.055;
        assert!((encoded - expected).abs() < 1e-6);
    }

    #[test]
    fn test_linear_rec709_is_identity() {
        assert!((GammaTarget::LinRec709.encode(0.75) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_white_point_d65_xy() {
        let (x, y) = WhitePointTarget::D65.xy();
        assert!((x - 0.3127).abs() < 1e-4);
        assert!((y - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn test_luminance_target_sdr() {
        let t = LuminanceTarget::sdr();
        assert!((t.max_nits - 100.0).abs() < 1e-6);
        assert!(t.min_nits < 1.0);
    }

    #[test]
    fn test_luminance_target_hdr1000() {
        let t = LuminanceTarget::hdr1000();
        assert!((t.max_nits - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_result_pass() {
        let spec = DisplayCalibrationSpec::srgb_sdr();
        let measured = MeasuredDisplayParams {
            actual_gamma: 2.2, // sRGB nominal
            actual_white_xy: (0.3127, 0.3290),
            actual_max_nits: 100.0,
            actual_min_nits: 0.1,
        };
        let result = CalibrationResult::evaluate(&spec, &measured);
        assert!(
            result.passes,
            "Expected pass but gamma_err={} white_de={} lum_err={}",
            result.error_gamma, result.error_white_de, result.error_max_luminance_pct
        );
    }

    #[test]
    fn test_calibration_result_fail_gamma() {
        let spec = DisplayCalibrationSpec::srgb_sdr();
        let measured = MeasuredDisplayParams {
            actual_gamma: 1.8, // too low
            actual_white_xy: (0.3127, 0.3290),
            actual_max_nits: 100.0,
            actual_min_nits: 0.1,
        };
        let result = CalibrationResult::evaluate(&spec, &measured);
        assert!(!result.passes);
        assert!(result.error_gamma > TOLERANCE_GAMMA);
    }

    #[test]
    fn test_calibration_result_fail_luminance() {
        let spec = DisplayCalibrationSpec::srgb_sdr();
        let measured = MeasuredDisplayParams {
            actual_gamma: 2.2,
            actual_white_xy: (0.3127, 0.3290),
            actual_max_nits: 200.0, // 100 % too bright
            actual_min_nits: 0.1,
        };
        let result = CalibrationResult::evaluate(&spec, &measured);
        assert!(!result.passes);
    }

    #[test]
    fn test_gamma_nominal_values() {
        assert!((GammaTarget::Gamma22.nominal_gamma() - 2.2).abs() < 1e-6);
        assert!((GammaTarget::Gamma26.nominal_gamma() - 2.6).abs() < 1e-6);
        assert!((GammaTarget::LinRec709.nominal_gamma() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hdr400_preset() {
        let t = LuminanceTarget::hdr400();
        assert!((t.max_nits - 400.0).abs() < 1e-6);
    }
}
