//! QC report structures for aggregating check results.
//!
//! Provides `QcFinding`, `QcCheckResult`, and `QcReport` for collecting
//! and summarising the outcome of quality-control checks.

#![allow(dead_code)]

/// Severity level of a QC finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    any(feature = "json", feature = "xml"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum FindingSeverity {
    /// Informational note; not a problem.
    Info,
    /// Minor issue that should be reviewed.
    Warning,
    /// Serious error that must be corrected.
    Error,
    /// Show-stopping defect; delivery must not proceed.
    Fatal,
}

/// A single finding produced by a QC check.
#[derive(Debug, Clone)]
#[cfg_attr(
    any(feature = "json", feature = "xml"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct QcFinding {
    /// Name of the check that produced this finding.
    pub check_name: String,
    /// Severity of the finding.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
    /// Optional timecode or position (seconds) where the issue occurs.
    #[cfg_attr(
        any(feature = "json", feature = "xml"),
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub position_secs: Option<f64>,
}

impl QcFinding {
    /// Creates a new finding.
    pub fn new(
        check_name: impl Into<String>,
        severity: FindingSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            check_name: check_name.into(),
            severity,
            message: message.into(),
            position_secs: None,
        }
    }

    /// Attaches a position (in seconds) to this finding.
    pub fn at_position(mut self, secs: f64) -> Self {
        self.position_secs = Some(secs);
        self
    }

    /// Returns `true` if this finding is fatal (delivery-blocking).
    pub fn is_fatal(&self) -> bool {
        self.severity == FindingSeverity::Fatal
    }

    /// Returns `true` if this finding represents an error or worse.
    pub fn is_error_or_worse(&self) -> bool {
        self.severity >= FindingSeverity::Error
    }
}

/// Result of running a single named QC check, potentially with multiple findings.
#[derive(Debug, Clone)]
#[cfg_attr(
    any(feature = "json", feature = "xml"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct QcCheckResult {
    /// Name of the check.
    pub check_name: String,
    /// Whether the check overall passed.
    pub passed: bool,
    /// Findings produced during the check.
    pub findings: Vec<QcFinding>,
}

impl QcCheckResult {
    /// Creates a passing result with no findings.
    pub fn pass(check_name: impl Into<String>) -> Self {
        Self {
            check_name: check_name.into(),
            passed: true,
            findings: Vec::new(),
        }
    }

    /// Creates a failing result with the provided findings.
    pub fn fail(check_name: impl Into<String>, findings: Vec<QcFinding>) -> Self {
        Self {
            check_name: check_name.into(),
            passed: false,
            findings,
        }
    }

    /// Adds a finding to this result.
    pub fn add_finding(&mut self, finding: QcFinding) {
        if finding.is_error_or_worse() {
            self.passed = false;
        }
        self.findings.push(finding);
    }

    /// Returns the number of findings that passed (no findings = 1 implicit pass).
    pub fn pass_count(&self) -> usize {
        if self.passed && self.findings.is_empty() {
            1
        } else {
            self.findings
                .iter()
                .filter(|f| f.severity == FindingSeverity::Info)
                .count()
        }
    }

    /// Returns the number of failing findings (Error + Fatal).
    pub fn fail_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.is_error_or_worse())
            .count()
    }

    /// Returns `true` if any finding is fatal.
    pub fn has_fatal(&self) -> bool {
        self.findings.iter().any(QcFinding::is_fatal)
    }
}

/// Aggregated QC report collecting results from multiple checks.
#[derive(Debug, Default, Clone)]
#[cfg_attr(
    any(feature = "json", feature = "xml"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct QcReport {
    /// All individual check results.
    results: Vec<QcCheckResult>,
    /// Optional label for this report (e.g. file name).
    #[cfg_attr(
        any(feature = "json", feature = "xml"),
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub label: Option<String>,
}

impl QcReport {
    /// Creates an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a report with a label.
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            results: Vec::new(),
            label: Some(label.into()),
        }
    }

    /// Adds a check result to this report.
    pub fn add_result(&mut self, result: QcCheckResult) {
        self.results.push(result);
    }

    /// Returns `true` if all checks passed and there are no fatal findings.
    pub fn overall_pass(&self) -> bool {
        self.results.iter().all(|r| r.passed) && !self.has_fatal_findings()
    }

    /// Returns all fatal findings across all results.
    pub fn fatal_findings(&self) -> Vec<&QcFinding> {
        self.results
            .iter()
            .flat_map(|r| r.findings.iter())
            .filter(|f| f.is_fatal())
            .collect()
    }

    /// Returns all findings at Error severity or worse.
    pub fn error_findings(&self) -> Vec<&QcFinding> {
        self.results
            .iter()
            .flat_map(|r| r.findings.iter())
            .filter(|f| f.is_error_or_worse())
            .collect()
    }

    /// Returns the total number of check results.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Returns the number of passing check results.
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Returns the number of failing check results.
    pub fn fail_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Returns `true` if any result contains a fatal finding.
    pub fn has_fatal_findings(&self) -> bool {
        self.results.iter().any(QcCheckResult::has_fatal)
    }

    /// Returns all check results in this report.
    pub fn all_results(&self) -> &[QcCheckResult] {
        &self.results
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}/{} checks passed{}",
            self.pass_count(),
            self.result_count(),
            if self.has_fatal_findings() {
                " [FATAL]"
            } else {
                ""
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finding_is_fatal() {
        let f = QcFinding::new("loudness", FindingSeverity::Fatal, "Too loud");
        assert!(f.is_fatal());
    }

    #[test]
    fn test_finding_not_fatal() {
        let f = QcFinding::new("loudness", FindingSeverity::Warning, "Slightly loud");
        assert!(!f.is_fatal());
    }

    #[test]
    fn test_finding_is_error_or_worse() {
        assert!(QcFinding::new("c", FindingSeverity::Error, "").is_error_or_worse());
        assert!(QcFinding::new("c", FindingSeverity::Fatal, "").is_error_or_worse());
        assert!(!QcFinding::new("c", FindingSeverity::Warning, "").is_error_or_worse());
        assert!(!QcFinding::new("c", FindingSeverity::Info, "").is_error_or_worse());
    }

    #[test]
    fn test_finding_at_position() {
        let f = QcFinding::new("black", FindingSeverity::Warning, "Black frame").at_position(12.5);
        assert_eq!(f.position_secs, Some(12.5));
    }

    #[test]
    fn test_check_result_pass() {
        let r = QcCheckResult::pass("video_codec");
        assert!(r.passed);
        assert_eq!(r.fail_count(), 0);
    }

    #[test]
    fn test_check_result_fail() {
        let findings = vec![QcFinding::new("c", FindingSeverity::Error, "Bad codec")];
        let r = QcCheckResult::fail("video_codec", findings);
        assert!(!r.passed);
        assert_eq!(r.fail_count(), 1);
    }

    #[test]
    fn test_check_result_pass_count_for_clean_pass() {
        let r = QcCheckResult::pass("audio");
        assert_eq!(r.pass_count(), 1);
    }

    #[test]
    fn test_check_result_has_fatal() {
        let findings = vec![QcFinding::new("c", FindingSeverity::Fatal, "Fatal")];
        let r = QcCheckResult::fail("check", findings);
        assert!(r.has_fatal());
    }

    #[test]
    fn test_report_overall_pass_empty() {
        let report = QcReport::new();
        assert!(report.overall_pass());
    }

    #[test]
    fn test_report_overall_pass_all_good() {
        let mut report = QcReport::new();
        report.add_result(QcCheckResult::pass("video"));
        report.add_result(QcCheckResult::pass("audio"));
        assert!(report.overall_pass());
    }

    #[test]
    fn test_report_overall_fail_on_error() {
        let mut report = QcReport::new();
        report.add_result(QcCheckResult::fail(
            "video",
            vec![QcFinding::new("v", FindingSeverity::Error, "Bad")],
        ));
        assert!(!report.overall_pass());
    }

    #[test]
    fn test_report_fatal_findings() {
        let mut report = QcReport::new();
        report.add_result(QcCheckResult::fail(
            "loudness",
            vec![QcFinding::new("l", FindingSeverity::Fatal, "Over limit")],
        ));
        assert_eq!(report.fatal_findings().len(), 1);
        assert!(report.has_fatal_findings());
    }

    #[test]
    fn test_report_pass_fail_counts() {
        let mut report = QcReport::new();
        report.add_result(QcCheckResult::pass("a"));
        report.add_result(QcCheckResult::pass("b"));
        report.add_result(QcCheckResult::fail(
            "c",
            vec![QcFinding::new("x", FindingSeverity::Error, "Err")],
        ));
        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.fail_count(), 1);
        assert_eq!(report.result_count(), 3);
    }

    #[test]
    fn test_report_summary_string() {
        let mut report = QcReport::with_label("test.mp4");
        report.add_result(QcCheckResult::pass("video"));
        let s = report.summary();
        assert!(s.contains("1/1"));
    }

    #[test]
    fn test_report_summary_fatal_suffix() {
        let mut report = QcReport::new();
        report.add_result(QcCheckResult::fail(
            "audio",
            vec![QcFinding::new("a", FindingSeverity::Fatal, "Fatal")],
        ));
        assert!(report.summary().contains("[FATAL]"));
    }

    // ── Round-trip serialization tests ───────────────────────────────────────

    /// Build a fixture QcReport with known contents.
    fn make_fixture_report() -> QcReport {
        let mut report = QcReport::with_label("fixture.mp4");

        let mut result1 = QcCheckResult::pass("video_codec");
        result1.add_finding(QcFinding::new(
            "video_codec",
            FindingSeverity::Info,
            "Codec is AV1",
        ));
        report.add_result(result1);

        let mut finding_err = QcFinding::new("loudness", FindingSeverity::Error, "Too loud");
        finding_err = finding_err.at_position(42.5);
        report.add_result(QcCheckResult::fail("loudness", vec![finding_err]));

        report
    }

    #[cfg(feature = "json")]
    #[test]
    fn test_qc_report_json_roundtrip() {
        let original = make_fixture_report();
        let json =
            serde_json::to_string_pretty(&original).expect("QcReport should serialize to JSON");
        let restored: QcReport =
            serde_json::from_str(&json).expect("QcReport should deserialize from JSON");

        // Structural equivalence checks.
        assert_eq!(
            original.result_count(),
            restored.result_count(),
            "result_count should be preserved through JSON round-trip"
        );
        assert_eq!(
            original.label, restored.label,
            "label should be preserved through JSON round-trip"
        );
        assert_eq!(
            original.pass_count(),
            restored.pass_count(),
            "pass_count should be preserved through JSON round-trip"
        );
        assert_eq!(
            original.fail_count(),
            restored.fail_count(),
            "fail_count should be preserved through JSON round-trip"
        );

        // Verify finding content round-trips.
        let orig_findings: Vec<_> = original
            .all_results()
            .iter()
            .flat_map(|r| r.findings.iter())
            .collect();
        let rest_findings: Vec<_> = restored
            .all_results()
            .iter()
            .flat_map(|r| r.findings.iter())
            .collect();
        assert_eq!(
            orig_findings.len(),
            rest_findings.len(),
            "Finding count should match"
        );
        for (o, r) in orig_findings.iter().zip(rest_findings.iter()) {
            assert_eq!(o.check_name, r.check_name);
            assert_eq!(o.severity, r.severity);
            assert_eq!(o.message, r.message);
            assert_eq!(o.position_secs, r.position_secs);
        }
    }

    #[cfg(feature = "xml")]
    #[test]
    fn test_qc_report_xml_roundtrip() {
        let original = make_fixture_report();
        let xml = quick_xml::se::to_string(&original).expect("QcReport should serialize to XML");
        let restored: QcReport =
            quick_xml::de::from_str(&xml).expect("QcReport should deserialize from XML");

        assert_eq!(
            original.result_count(),
            restored.result_count(),
            "result_count should be preserved through XML round-trip"
        );
        assert_eq!(
            original.label, restored.label,
            "label should be preserved through XML round-trip"
        );
    }
}
