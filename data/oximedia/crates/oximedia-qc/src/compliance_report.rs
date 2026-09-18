//! Compliance reporting: issues, levels, and delivery gate decisions.
//!
//! After QC checks run, `ComplianceReport` aggregates issues and decides
//! whether a file is cleared for delivery or blocked.

#![allow(dead_code)]

/// Overall compliance level of a media file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplianceLevel {
    /// Fully compliant; ready for delivery.
    Pass,
    /// Minor issues present; delivery may proceed with warnings noted.
    PassWithWarnings,
    /// Significant issues; requires review before delivery.
    ConditionalPass,
    /// Critical issues; delivery is blocked until resolved.
    Fail,
}

impl ComplianceLevel {
    /// Returns a numeric value for comparison (higher = more severe).
    pub fn numeric_value(&self) -> u8 {
        match self {
            ComplianceLevel::Pass => 0,
            ComplianceLevel::PassWithWarnings => 1,
            ComplianceLevel::ConditionalPass => 2,
            ComplianceLevel::Fail => 3,
        }
    }

    /// Returns `true` if the level indicates delivery is not blocked.
    pub fn is_deliverable(&self) -> bool {
        matches!(
            self,
            ComplianceLevel::Pass | ComplianceLevel::PassWithWarnings
        )
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ComplianceLevel::Pass => "PASS",
            ComplianceLevel::PassWithWarnings => "PASS WITH WARNINGS",
            ComplianceLevel::ConditionalPass => "CONDITIONAL PASS",
            ComplianceLevel::Fail => "FAIL",
        }
    }
}

/// A single compliance issue found during QC.
#[derive(Debug, Clone)]
pub struct ComplianceIssue {
    /// Short code identifying the check that found the issue (e.g. `"VID_LEVELS"`).
    pub check_code: String,
    /// Human-readable description of the issue.
    pub message: String,
    /// Severity: `"error"`, `"warning"`, or `"info"`.
    pub severity: String,
    /// Optional timecode or position where the issue was found.
    pub location: Option<String>,
    /// Optional remediation suggestion.
    pub suggestion: Option<String>,
}

impl ComplianceIssue {
    /// Create a new issue.
    pub fn new(
        check_code: impl Into<String>,
        message: impl Into<String>,
        severity: impl Into<String>,
    ) -> Self {
        Self {
            check_code: check_code.into(),
            message: message.into(),
            severity: severity.into(),
            location: None,
            suggestion: None,
        }
    }

    /// Create an error-severity issue.
    pub fn error(check_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(check_code, message, "error")
    }

    /// Create a warning-severity issue.
    pub fn warning(check_code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(check_code, message, "warning")
    }

    /// Returns `true` if this issue is an error that blocks delivery.
    pub fn blocks_delivery(&self) -> bool {
        self.severity == "error"
    }

    /// Builder: attach a location string.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Builder: attach a suggestion.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Aggregated compliance report for a single media asset.
#[derive(Debug, Default, Clone)]
pub struct ComplianceReport {
    /// Name or path of the asset being reported on.
    pub asset_name: String,
    /// List of issues found.
    issues: Vec<ComplianceIssue>,
}

impl ComplianceReport {
    /// Create an empty report for `asset_name`.
    pub fn new(asset_name: impl Into<String>) -> Self {
        Self {
            asset_name: asset_name.into(),
            issues: Vec::new(),
        }
    }

    /// Add a compliance issue to the report.
    pub fn add_issue(&mut self, issue: ComplianceIssue) {
        self.issues.push(issue);
    }

    /// Total number of issues.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    /// Number of error-severity issues.
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.blocks_delivery()).count()
    }

    /// Number of warning-severity issues.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == "warning")
            .count()
    }

    /// Returns `true` if any issue blocks delivery.
    pub fn delivery_blocked(&self) -> bool {
        self.issues.iter().any(ComplianceIssue::blocks_delivery)
    }

    /// Compute the overall `ComplianceLevel` based on accumulated issues.
    pub fn overall_level(&self) -> ComplianceLevel {
        if self.error_count() > 0 {
            ComplianceLevel::Fail
        } else if self.warning_count() > 5 {
            ComplianceLevel::ConditionalPass
        } else if self.warning_count() > 0 {
            ComplianceLevel::PassWithWarnings
        } else {
            ComplianceLevel::Pass
        }
    }

    /// Iterate over all issues.
    pub fn issues(&self) -> impl Iterator<Item = &ComplianceIssue> {
        self.issues.iter()
    }

    /// Return only error issues.
    pub fn errors(&self) -> Vec<&ComplianceIssue> {
        self.issues.iter().filter(|i| i.blocks_delivery()).collect()
    }

    /// Return only warning issues.
    pub fn warnings(&self) -> Vec<&ComplianceIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == "warning")
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_level_numeric_pass() {
        assert_eq!(ComplianceLevel::Pass.numeric_value(), 0);
    }

    #[test]
    fn test_compliance_level_numeric_fail() {
        assert_eq!(ComplianceLevel::Fail.numeric_value(), 3);
    }

    #[test]
    fn test_compliance_level_ordering() {
        assert!(ComplianceLevel::Fail > ComplianceLevel::Pass);
    }

    #[test]
    fn test_compliance_level_is_deliverable_pass() {
        assert!(ComplianceLevel::Pass.is_deliverable());
    }

    #[test]
    fn test_compliance_level_is_deliverable_pass_with_warnings() {
        assert!(ComplianceLevel::PassWithWarnings.is_deliverable());
    }

    #[test]
    fn test_compliance_level_not_deliverable_fail() {
        assert!(!ComplianceLevel::Fail.is_deliverable());
    }

    #[test]
    fn test_compliance_level_not_deliverable_conditional() {
        assert!(!ComplianceLevel::ConditionalPass.is_deliverable());
    }

    #[test]
    fn test_compliance_level_label() {
        assert_eq!(ComplianceLevel::Pass.label(), "PASS");
        assert_eq!(ComplianceLevel::Fail.label(), "FAIL");
    }

    #[test]
    fn test_compliance_issue_blocks_delivery_error() {
        let issue = ComplianceIssue::error("VID", "Luma clipping");
        assert!(issue.blocks_delivery());
    }

    #[test]
    fn test_compliance_issue_does_not_block_warning() {
        let issue = ComplianceIssue::warning("AUD", "Slightly high level");
        assert!(!issue.blocks_delivery());
    }

    #[test]
    fn test_compliance_report_empty_is_pass() {
        let report = ComplianceReport::new("file.mxf");
        assert_eq!(report.overall_level(), ComplianceLevel::Pass);
        assert!(!report.delivery_blocked());
    }

    #[test]
    fn test_compliance_report_with_error_is_fail() {
        let mut report = ComplianceReport::new("file.mxf");
        report.add_issue(ComplianceIssue::error("VID", "Clipping"));
        assert_eq!(report.overall_level(), ComplianceLevel::Fail);
        assert!(report.delivery_blocked());
    }

    #[test]
    fn test_compliance_report_with_warning_only() {
        let mut report = ComplianceReport::new("file.mxf");
        report.add_issue(ComplianceIssue::warning("AUD", "High level"));
        assert_eq!(report.overall_level(), ComplianceLevel::PassWithWarnings);
        assert!(!report.delivery_blocked());
    }

    #[test]
    fn test_compliance_report_issue_count() {
        let mut report = ComplianceReport::new("f.mp4");
        report.add_issue(ComplianceIssue::error("A", "err"));
        report.add_issue(ComplianceIssue::warning("B", "warn"));
        assert_eq!(report.issue_count(), 2);
    }

    #[test]
    fn test_compliance_report_error_count() {
        let mut report = ComplianceReport::new("f.mp4");
        report.add_issue(ComplianceIssue::error("A", "err1"));
        report.add_issue(ComplianceIssue::error("B", "err2"));
        report.add_issue(ComplianceIssue::warning("C", "warn"));
        assert_eq!(report.error_count(), 2);
    }

    #[test]
    fn test_compliance_report_warning_count() {
        let mut report = ComplianceReport::new("f.mp4");
        for i in 0..6 {
            report.add_issue(ComplianceIssue::warning("W", format!("warn {i}")));
        }
        assert_eq!(report.overall_level(), ComplianceLevel::ConditionalPass);
    }

    #[test]
    fn test_compliance_issue_with_location_and_suggestion() {
        let issue = ComplianceIssue::error("VID", "Clipping")
            .with_location("00:01:23:00")
            .with_suggestion("Reduce gain by 3 dB");
        assert_eq!(issue.location.as_deref(), Some("00:01:23:00"));
        assert!(issue.suggestion.is_some());
    }
}
