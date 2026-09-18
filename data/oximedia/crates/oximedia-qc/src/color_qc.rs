//! Color-space quality control checks.
//!
//! Provides `ColorStandard`, `ColorQcCheck`, and `ColorQcReport` for
//! validating that a media file's colour metadata conforms to the expected
//! primaries, transfer function, and matrix.

#![allow(dead_code)]

/// The colour standard / colour space expected for a stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ColorStandard {
    /// ITU-R BT.709 – HD broadcast standard.
    Bt709,
    /// ITU-R BT.2020 – UHD / HDR wide-colour-gamut.
    Bt2020,
    /// SMPTE ST 2084 (PQ) – Dolby Vision / HDR10 transfer.
    St2084,
    /// ITU-R BT.601 – SD broadcast.
    Bt601,
    /// Display P3 – digital cinema / consumer HDR displays.
    DisplayP3,
    /// Custom / unknown standard.
    Custom(String),
}

impl ColorStandard {
    /// Return the canonical name of this colour standard.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
            Self::St2084 => "ST 2084 (PQ)",
            Self::Bt601 => "BT.601",
            Self::DisplayP3 => "Display P3",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Return `true` for standards that imply HDR content.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        matches!(self, Self::Bt2020 | Self::St2084)
    }
}

/// Severity level of a colour QC finding.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational – no action required.
    Info,
    /// Warning – may cause issues on some displays.
    Warning,
    /// Error – content will likely be mis-displayed.
    Error,
}

/// A single colour quality-control check result.
#[derive(Debug, Clone)]
pub struct ColorQcCheck {
    /// Human-readable name of the check.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Severity when the check fails.
    pub severity: Severity,
    /// Optional diagnostic detail.
    pub detail: Option<String>,
}

impl ColorQcCheck {
    /// Create a passing check.
    #[must_use]
    pub fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            severity: Severity::Info,
            detail: None,
        }
    }

    /// Create a failing check.
    #[must_use]
    pub fn fail(name: impl Into<String>, severity: Severity, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            severity,
            detail: Some(detail.into()),
        }
    }
}

/// Aggregated colour QC report for a single stream.
#[derive(Debug, Default)]
pub struct ColorQcReport {
    /// Expected colour standard.
    pub expected: Option<ColorStandard>,
    /// Detected colour standard (from stream metadata).
    pub detected: Option<ColorStandard>,
    /// Individual check results.
    pub checks: Vec<ColorQcCheck>,
}

impl ColorQcReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a check result.
    pub fn push(&mut self, check: ColorQcCheck) {
        self.checks.push(check);
    }

    /// Validate that `detected` matches `expected`, adding a check result.
    pub fn validate_standard(&mut self, expected: ColorStandard, detected: ColorStandard) {
        let passed = expected == detected;
        let check = if passed {
            ColorQcCheck::pass(format!("color_standard:{}", expected.name()))
        } else {
            ColorQcCheck::fail(
                format!("color_standard:{}", expected.name()),
                Severity::Error,
                format!("expected {} but found {}", expected.name(), detected.name()),
            )
        };
        self.expected = Some(expected);
        self.detected = Some(detected);
        self.push(check);
    }

    /// Return the fraction of checks that passed in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        if self.checks.is_empty() {
            return 1.0;
        }
        let passed = self.checks.iter().filter(|c| c.passed).count();
        passed as f64 / self.checks.len() as f64
    }

    /// Return `true` if all checks passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Return all failing checks.
    #[must_use]
    pub fn failures(&self) -> Vec<&ColorQcCheck> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }

    /// Return the highest severity among failing checks, or `None` if all pass.
    #[must_use]
    pub fn max_severity(&self) -> Option<Severity> {
        self.failures().iter().map(|c| c.severity.clone()).max()
    }

    /// Return the number of checks at or above `min_severity` that failed.
    #[must_use]
    pub fn error_count(&self, min_severity: &Severity) -> usize {
        self.failures()
            .iter()
            .filter(|c| &c.severity >= min_severity)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_standard_name() {
        assert_eq!(ColorStandard::Bt709.name(), "BT.709");
        assert_eq!(ColorStandard::Bt2020.name(), "BT.2020");
        assert_eq!(ColorStandard::St2084.name(), "ST 2084 (PQ)");
        assert_eq!(ColorStandard::Bt601.name(), "BT.601");
        assert_eq!(ColorStandard::DisplayP3.name(), "Display P3");
        assert_eq!(ColorStandard::Custom("ACES".into()).name(), "ACES");
    }

    #[test]
    fn test_is_hdr() {
        assert!(ColorStandard::Bt2020.is_hdr());
        assert!(ColorStandard::St2084.is_hdr());
        assert!(!ColorStandard::Bt709.is_hdr());
        assert!(!ColorStandard::Bt601.is_hdr());
    }

    #[test]
    fn test_check_pass() {
        let c = ColorQcCheck::pass("primaries");
        assert!(c.passed);
        assert!(c.detail.is_none());
    }

    #[test]
    fn test_check_fail() {
        let c = ColorQcCheck::fail("primaries", Severity::Error, "mismatch");
        assert!(!c.passed);
        assert_eq!(c.severity, Severity::Error);
        assert!(c.detail.is_some());
    }

    #[test]
    fn test_report_pass_rate_empty() {
        let report = ColorQcReport::new();
        assert!((report.pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_pass_rate_all_pass() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::pass("a"));
        report.push(ColorQcCheck::pass("b"));
        assert!((report.pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_pass_rate_partial() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::pass("a"));
        report.push(ColorQcCheck::fail("b", Severity::Warning, "detail"));
        let rate = report.pass_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_passed_true() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::pass("a"));
        assert!(report.all_passed());
    }

    #[test]
    fn test_all_passed_false() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::pass("a"));
        report.push(ColorQcCheck::fail("b", Severity::Error, "bad"));
        assert!(!report.all_passed());
    }

    #[test]
    fn test_validate_standard_match() {
        let mut report = ColorQcReport::new();
        report.validate_standard(ColorStandard::Bt709, ColorStandard::Bt709);
        assert!(report.all_passed());
        assert_eq!(report.checks.len(), 1);
    }

    #[test]
    fn test_validate_standard_mismatch() {
        let mut report = ColorQcReport::new();
        report.validate_standard(ColorStandard::Bt709, ColorStandard::Bt601);
        assert!(!report.all_passed());
        assert_eq!(report.failures().len(), 1);
    }

    #[test]
    fn test_max_severity() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::fail("a", Severity::Warning, "w"));
        report.push(ColorQcCheck::fail("b", Severity::Error, "e"));
        assert_eq!(report.max_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_max_severity_none_when_all_pass() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::pass("a"));
        assert!(report.max_severity().is_none());
    }

    #[test]
    fn test_error_count() {
        let mut report = ColorQcReport::new();
        report.push(ColorQcCheck::fail("a", Severity::Warning, "w"));
        report.push(ColorQcCheck::fail("b", Severity::Error, "e"));
        assert_eq!(report.error_count(&Severity::Error), 1);
        assert_eq!(report.error_count(&Severity::Warning), 2);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }
}
