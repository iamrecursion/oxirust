//! Test card generation and validation for broadcast conforming.
//!
//! Provides types for describing, generating, and validating test cards
//! (colour bars, PLUGE, grey ramps, checkfields) against broadcast specs.

#![allow(dead_code)]

/// Type of test card / test pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestCardType {
    /// SMPTE / EBU colour bars.
    ColorBars,
    /// Flat grey field (50 % or 40 IRE).
    Grey,
    /// Picture Line Up Generation Equipment pattern for black-level setup.
    Pluge,
    /// Checkfield (alternating black/white pixel checkerboard).
    Checkfield,
}

impl TestCardType {
    /// Human-readable label for this test-card type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::ColorBars => "Colour Bars",
            Self::Grey => "Grey Field",
            Self::Pluge => "PLUGE",
            Self::Checkfield => "Checkfield",
        }
    }
}

/// Specification of a test card (resolution, frame-rate, duration).
#[derive(Clone, Debug)]
pub struct TestCardSpec {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Pattern type.
    pub card_type: TestCardType,
}

impl TestCardSpec {
    /// Create a new test card spec.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        frame_rate_num: u32,
        frame_rate_den: u32,
        duration_seconds: f64,
        card_type: TestCardType,
    ) -> Self {
        Self {
            width,
            height,
            frame_rate_num,
            frame_rate_den,
            duration_seconds,
            card_type,
        }
    }

    /// Returns `true` if the spec is HD (width >= 1280).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.width >= 1280
    }

    /// Returns `true` if the spec is UHD (width >= 3840).
    #[must_use]
    pub fn is_uhd(&self) -> bool {
        self.width >= 3840
    }

    /// Frame rate as a floating-point value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        f64::from(self.frame_rate_num) / f64::from(self.frame_rate_den)
    }

    /// Total number of frames.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        (self.duration_seconds * self.frame_rate()) as u64
    }
}

/// Result of validating test-card signal levels.
#[derive(Clone, Debug)]
pub struct ValidationIssue {
    /// Human-readable description of the issue.
    pub description: String,
    /// Whether this issue is fatal (fails the card) or merely a warning.
    pub is_fatal: bool,
}

/// Validator for test-card signal levels.
#[derive(Clone, Debug, Default)]
pub struct TestCardValidator {
    issues: Vec<ValidationIssue>,
}

impl TestCardValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Validate white and black levels (values in IRE, 0–100).
    pub fn validate_levels(&mut self, white_ire: f32, black_ire: f32) {
        if white_ire > 100.0 {
            self.issues.push(ValidationIssue {
                description: format!("White level {white_ire:.1} IRE exceeds 100 IRE limit"),
                is_fatal: true,
            });
        }
        if black_ire < 0.0 {
            self.issues.push(ValidationIssue {
                description: format!(
                    "Black level {black_ire:.1} IRE is below 0 IRE (crushed blacks)"
                ),
                is_fatal: false,
            });
        }
        if (white_ire - black_ire) < 70.0 {
            self.issues.push(ValidationIssue {
                description: format!(
                    "Contrast range {:.1} IRE is less than the recommended 70 IRE minimum",
                    white_ire - black_ire
                ),
                is_fatal: false,
            });
        }
    }

    /// Validate chroma saturation (0.0–1.0 scale).
    pub fn validate_saturation(&mut self, max_saturation: f32) {
        if max_saturation > 1.0 {
            self.issues.push(ValidationIssue {
                description: format!("Saturation {max_saturation:.3} exceeds legal limit of 1.0"),
                is_fatal: true,
            });
        }
    }

    /// Returns all collected issues.
    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    /// Returns `true` if any fatal issues were found.
    #[must_use]
    pub fn has_fatal_issues(&self) -> bool {
        self.issues.iter().any(|i| i.is_fatal)
    }

    /// Clear all issues (reset for re-use).
    pub fn reset(&mut self) {
        self.issues.clear();
    }
}

/// Summary report produced after validating a test card.
#[derive(Clone, Debug)]
pub struct TestCardReport {
    /// The spec that was validated.
    pub spec: TestCardSpec,
    /// Issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl TestCardReport {
    /// Create a report from a spec and a list of validation issues.
    #[must_use]
    pub fn new(spec: TestCardSpec, issues: Vec<ValidationIssue>) -> Self {
        Self { spec, issues }
    }

    /// Returns `true` when there are no fatal issues.
    #[must_use]
    pub fn passes(&self) -> bool {
        !self.issues.iter().any(|i| i.is_fatal)
    }

    /// Number of warnings (non-fatal issues).
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.is_fatal).count()
    }

    /// Number of errors (fatal issues).
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.is_fatal).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_type_label_color_bars() {
        assert_eq!(TestCardType::ColorBars.label(), "Colour Bars");
    }

    #[test]
    fn test_card_type_label_grey() {
        assert_eq!(TestCardType::Grey.label(), "Grey Field");
    }

    #[test]
    fn test_card_type_label_pluge() {
        assert_eq!(TestCardType::Pluge.label(), "PLUGE");
    }

    #[test]
    fn test_card_type_label_checkfield() {
        assert_eq!(TestCardType::Checkfield.label(), "Checkfield");
    }

    #[test]
    fn test_spec_is_hd_true() {
        let spec = TestCardSpec::new(1920, 1080, 25, 1, 10.0, TestCardType::ColorBars);
        assert!(spec.is_hd());
    }

    #[test]
    fn test_spec_is_hd_false_for_sd() {
        let spec = TestCardSpec::new(720, 576, 25, 1, 10.0, TestCardType::ColorBars);
        assert!(!spec.is_hd());
    }

    #[test]
    fn test_spec_is_uhd() {
        let spec = TestCardSpec::new(3840, 2160, 50, 1, 5.0, TestCardType::Grey);
        assert!(spec.is_uhd());
        assert!(spec.is_hd());
    }

    #[test]
    fn test_spec_frame_rate() {
        let spec = TestCardSpec::new(1920, 1080, 30000, 1001, 1.0, TestCardType::Grey);
        let fps = spec.frame_rate();
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_spec_total_frames() {
        let spec = TestCardSpec::new(1920, 1080, 25, 1, 4.0, TestCardType::Pluge);
        assert_eq!(spec.total_frames(), 100);
    }

    #[test]
    fn test_validator_no_issues_for_good_levels() {
        let mut v = TestCardValidator::new();
        v.validate_levels(100.0, 7.5);
        assert!(!v.has_fatal_issues());
        assert!(v.issues().is_empty());
    }

    #[test]
    fn test_validator_white_too_high() {
        let mut v = TestCardValidator::new();
        v.validate_levels(105.0, 7.5);
        assert!(v.has_fatal_issues());
    }

    #[test]
    fn test_validator_black_below_zero() {
        let mut v = TestCardValidator::new();
        v.validate_levels(100.0, -5.0);
        let non_fatal: Vec<_> = v.issues().iter().filter(|i| !i.is_fatal).collect();
        assert!(!non_fatal.is_empty());
    }

    #[test]
    fn test_validator_low_contrast_warning() {
        let mut v = TestCardValidator::new();
        v.validate_levels(60.0, 10.0); // range 50 IRE < 70 minimum
        let warnings: Vec<_> = v.issues().iter().filter(|i| !i.is_fatal).collect();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_validator_saturation_legal() {
        let mut v = TestCardValidator::new();
        v.validate_saturation(0.75);
        assert!(!v.has_fatal_issues());
    }

    #[test]
    fn test_validator_saturation_illegal() {
        let mut v = TestCardValidator::new();
        v.validate_saturation(1.05);
        assert!(v.has_fatal_issues());
    }

    #[test]
    fn test_validator_reset() {
        let mut v = TestCardValidator::new();
        v.validate_levels(110.0, -1.0);
        assert!(!v.issues().is_empty());
        v.reset();
        assert!(v.issues().is_empty());
    }

    #[test]
    fn test_report_passes_no_fatal() {
        let spec = TestCardSpec::new(1920, 1080, 25, 1, 10.0, TestCardType::ColorBars);
        let report = TestCardReport::new(spec, vec![]);
        assert!(report.passes());
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn test_report_fails_with_fatal() {
        let spec = TestCardSpec::new(1920, 1080, 25, 1, 10.0, TestCardType::Pluge);
        let issues = vec![ValidationIssue {
            description: "Over limit".to_string(),
            is_fatal: true,
        }];
        let report = TestCardReport::new(spec, issues);
        assert!(!report.passes());
        assert_eq!(report.error_count(), 1);
    }

    #[test]
    fn test_report_warning_count() {
        let spec = TestCardSpec::new(1920, 1080, 25, 1, 10.0, TestCardType::Grey);
        let issues = vec![
            ValidationIssue {
                description: "warn1".to_string(),
                is_fatal: false,
            },
            ValidationIssue {
                description: "warn2".to_string(),
                is_fatal: false,
            },
            ValidationIssue {
                description: "err1".to_string(),
                is_fatal: true,
            },
        ];
        let report = TestCardReport::new(spec, issues);
        assert_eq!(report.warning_count(), 2);
        assert_eq!(report.error_count(), 1);
        assert!(!report.passes());
    }
}
