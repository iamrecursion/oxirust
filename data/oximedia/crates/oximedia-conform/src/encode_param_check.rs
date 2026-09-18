//! Encoder parameter checking for media conforming.
//!
//! Validates that encoder parameters (bitrate, resolution, codec settings) fall
//! within the legal ranges prescribed by a delivery specification before encoding
//! or re-wrapping a conformed timeline.

#![allow(dead_code)]

/// An individual encoder parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum EncoderParam {
    /// Target bitrate in kilobits per second.
    Bitrate(u32),
    /// Video width in pixels.
    Width(u32),
    /// Video height in pixels.
    Height(u32),
    /// GOP (group of pictures) size in frames.
    GopSize(u32),
    /// Quantisation parameter (H.264/HEVC CRF / QP).
    Qp(u8),
    /// Number of B-frames.
    BFrames(u8),
    /// Pixel aspect ratio as (num, den).
    PixelAspect(u32, u32),
    /// Custom named parameter with a float value.
    Custom(String, f64),
}

impl EncoderParam {
    /// Return the canonical parameter name used in validation reports.
    #[must_use]
    pub fn param_name(&self) -> &str {
        match self {
            Self::Bitrate(_) => "bitrate_kbps",
            Self::Width(_) => "width_px",
            Self::Height(_) => "height_px",
            Self::GopSize(_) => "gop_size",
            Self::Qp(_) => "qp",
            Self::BFrames(_) => "b_frames",
            Self::PixelAspect(_, _) => "pixel_aspect",
            Self::Custom(name, _) => name.as_str(),
        }
    }
}

/// A numeric range check with optional min/max bounds.
#[derive(Clone, Debug)]
pub struct ParamCheck {
    /// Human-readable name for reporting.
    pub name: String,
    /// Minimum legal value (inclusive).  `None` means no lower bound.
    pub min: Option<f64>,
    /// Maximum legal value (inclusive).  `None` means no upper bound.
    pub max: Option<f64>,
    /// The actual measured value.
    pub value: f64,
}

impl ParamCheck {
    /// Create a new `ParamCheck`.
    pub fn new(name: impl Into<String>, value: f64, min: Option<f64>, max: Option<f64>) -> Self {
        Self {
            name: name.into(),
            min,
            max,
            value,
        }
    }

    /// Returns `true` when `value` is within the [min, max] range (both inclusive).
    #[must_use]
    pub fn is_in_range(&self) -> bool {
        if let Some(lo) = self.min {
            if self.value < lo {
                return false;
            }
        }
        if let Some(hi) = self.max {
            if self.value > hi {
                return false;
            }
        }
        true
    }

    /// Amount by which `value` violates the range (0.0 when in-range).
    #[must_use]
    pub fn violation_amount(&self) -> f64 {
        if let Some(lo) = self.min {
            if self.value < lo {
                return lo - self.value;
            }
        }
        if let Some(hi) = self.max {
            if self.value > hi {
                return self.value - hi;
            }
        }
        0.0
    }
}

/// Validates a collection of [`EncoderParam`]s against configured ranges.
#[derive(Clone, Debug, Default)]
pub struct EncodeParamValidator {
    checks: Vec<ParamCheck>,
}

impl EncodeParamValidator {
    /// Create an empty validator.
    #[must_use]
    pub fn new() -> Self {
        Self { checks: Vec::new() }
    }

    /// Add a single parameter check.
    pub fn add_check(&mut self, check: ParamCheck) {
        self.checks.push(check);
    }

    /// Add a check derived from an [`EncoderParam`] with explicit bounds.
    #[allow(clippy::cast_precision_loss)]
    pub fn add_param(&mut self, param: &EncoderParam, min: Option<f64>, max: Option<f64>) {
        let value = match param {
            EncoderParam::Bitrate(v) => f64::from(*v),
            EncoderParam::Width(v) | EncoderParam::Height(v) | EncoderParam::GopSize(v) => {
                f64::from(*v)
            }
            EncoderParam::Qp(v) | EncoderParam::BFrames(v) => f64::from(*v),
            EncoderParam::PixelAspect(n, d) => {
                if *d == 0 {
                    0.0
                } else {
                    f64::from(*n) / f64::from(*d)
                }
            }
            EncoderParam::Custom(_, v) => *v,
        };
        self.checks
            .push(ParamCheck::new(param.param_name(), value, min, max));
    }

    /// Run all checks.  Returns references to out-of-range checks.
    #[must_use]
    pub fn check_all(&self) -> Vec<&ParamCheck> {
        self.checks.iter().filter(|c| !c.is_in_range()).collect()
    }

    /// Returns `true` when all parameter checks pass.
    #[must_use]
    pub fn all_pass(&self) -> bool {
        self.checks.iter().all(ParamCheck::is_in_range)
    }
}

/// A report produced by running [`EncodeParamValidator::check_all`].
#[derive(Clone, Debug)]
pub struct EncoderReport {
    /// All parameter checks (pass and fail).
    pub checks: Vec<ParamCheck>,
}

impl EncoderReport {
    /// Create a report from a complete set of checks.
    #[must_use]
    pub fn new(checks: Vec<ParamCheck>) -> Self {
        Self { checks }
    }

    /// Returns only the checks that are out-of-specification.
    #[must_use]
    pub fn out_of_spec_params(&self) -> Vec<&ParamCheck> {
        self.checks.iter().filter(|c| !c.is_in_range()).collect()
    }

    /// Returns `true` when every check passes.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.checks.iter().all(ParamCheck::is_in_range)
    }

    /// Number of out-of-spec parameters.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.is_in_range()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_param_bitrate_name() {
        let p = EncoderParam::Bitrate(5000);
        assert_eq!(p.param_name(), "bitrate_kbps");
    }

    #[test]
    fn test_encoder_param_width_name() {
        assert_eq!(EncoderParam::Width(1920).param_name(), "width_px");
    }

    #[test]
    fn test_encoder_param_height_name() {
        assert_eq!(EncoderParam::Height(1080).param_name(), "height_px");
    }

    #[test]
    fn test_encoder_param_gop_name() {
        assert_eq!(EncoderParam::GopSize(25).param_name(), "gop_size");
    }

    #[test]
    fn test_encoder_param_qp_name() {
        assert_eq!(EncoderParam::Qp(23).param_name(), "qp");
    }

    #[test]
    fn test_encoder_param_custom_name() {
        let p = EncoderParam::Custom("crf".to_string(), 20.0);
        assert_eq!(p.param_name(), "crf");
    }

    #[test]
    fn test_param_check_in_range() {
        let c = ParamCheck::new("bitrate", 5000.0, Some(1000.0), Some(8000.0));
        assert!(c.is_in_range());
    }

    #[test]
    fn test_param_check_below_min() {
        let c = ParamCheck::new("bitrate", 500.0, Some(1000.0), Some(8000.0));
        assert!(!c.is_in_range());
    }

    #[test]
    fn test_param_check_above_max() {
        let c = ParamCheck::new("bitrate", 10000.0, Some(1000.0), Some(8000.0));
        assert!(!c.is_in_range());
    }

    #[test]
    fn test_param_check_no_bounds() {
        let c = ParamCheck::new("free", 999_999.0, None, None);
        assert!(c.is_in_range());
    }

    #[test]
    fn test_param_check_violation_amount() {
        let c = ParamCheck::new("qp", 5.0, Some(10.0), None);
        assert!((c.violation_amount() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_validator_all_pass() {
        let mut v = EncodeParamValidator::new();
        v.add_param(&EncoderParam::Bitrate(4000), Some(1000.0), Some(8000.0));
        v.add_param(&EncoderParam::Qp(23), Some(18.0), Some(28.0));
        assert!(v.all_pass());
        assert!(v.check_all().is_empty());
    }

    #[test]
    fn test_validator_one_failure() {
        let mut v = EncodeParamValidator::new();
        v.add_param(&EncoderParam::Bitrate(500), Some(1000.0), Some(8000.0));
        v.add_param(&EncoderParam::Width(1920), Some(1280.0), Some(4096.0));
        assert!(!v.all_pass());
        assert_eq!(v.check_all().len(), 1);
    }

    #[test]
    fn test_encoder_report_compliant() {
        let checks = vec![
            ParamCheck::new("bitrate", 4000.0, Some(1000.0), Some(8000.0)),
            ParamCheck::new("width", 1920.0, Some(1280.0), Some(4096.0)),
        ];
        let report = EncoderReport::new(checks);
        assert!(report.is_compliant());
        assert_eq!(report.failure_count(), 0);
        assert!(report.out_of_spec_params().is_empty());
    }

    #[test]
    fn test_encoder_report_failure() {
        let checks = vec![
            ParamCheck::new("bitrate", 500.0, Some(1000.0), Some(8000.0)),
            ParamCheck::new("width", 1920.0, Some(1280.0), Some(4096.0)),
        ];
        let report = EncoderReport::new(checks);
        assert!(!report.is_compliant());
        assert_eq!(report.failure_count(), 1);
        let oos = report.out_of_spec_params();
        assert_eq!(oos[0].name, "bitrate");
    }
}
