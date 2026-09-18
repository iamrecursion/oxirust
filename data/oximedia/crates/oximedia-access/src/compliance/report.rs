//! Compliance report generation.

use serde::{Deserialize, Serialize};

/// Severity of a compliance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational.
    Info,
    /// Low severity.
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
    /// Critical issue.
    Critical,
}

/// A compliance issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue identifier (e.g., "WCAG-1.2.2").
    pub id: String,
    /// Issue title.
    pub title: String,
    /// Issue description.
    pub description: String,
    /// Severity level.
    pub severity: IssueSeverity,
    /// Optional location in media.
    pub location: Option<String>,
}

impl ComplianceIssue {
    /// Create a new compliance issue.
    #[must_use]
    pub fn new(id: String, title: String, description: String, severity: IssueSeverity) -> Self {
        Self {
            id,
            title,
            description,
            severity,
            location: None,
        }
    }

    /// Set location.
    #[must_use]
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }
}

/// Compliance report containing all issues found.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// All issues found.
    issues: Vec<ComplianceIssue>,
    /// Summary statistics.
    summary: ReportSummary,
}

/// Summary statistics for a compliance report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of issues.
    pub total_issues: usize,
    /// Number of critical issues.
    pub critical_issues: usize,
    /// Number of high severity issues.
    pub high_issues: usize,
    /// Number of medium severity issues.
    pub medium_issues: usize,
    /// Number of low severity issues.
    pub low_issues: usize,
    /// Number of info issues.
    pub info_issues: usize,
}

impl ComplianceReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an issue to the report.
    pub fn add_issue(&mut self, issue: ComplianceIssue) {
        match issue.severity {
            IssueSeverity::Critical => self.summary.critical_issues += 1,
            IssueSeverity::High => self.summary.high_issues += 1,
            IssueSeverity::Medium => self.summary.medium_issues += 1,
            IssueSeverity::Low => self.summary.low_issues += 1,
            IssueSeverity::Info => self.summary.info_issues += 1,
        }
        self.summary.total_issues += 1;
        self.issues.push(issue);
    }

    /// Add multiple issues.
    pub fn add_issues(&mut self, issues: Vec<ComplianceIssue>) {
        for issue in issues {
            self.add_issue(issue);
        }
    }

    /// Get all issues.
    #[must_use]
    pub fn issues(&self) -> &[ComplianceIssue] {
        &self.issues
    }

    /// Get summary.
    #[must_use]
    pub const fn summary(&self) -> &ReportSummary {
        &self.summary
    }

    /// Check if content is compliant (no critical or high severity issues).
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.summary.critical_issues == 0 && self.summary.high_issues == 0
    }

    /// Get issues by severity.
    #[must_use]
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&ComplianceIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Export report as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export report as plain text.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut output = String::from("Accessibility Compliance Report\n");
        output.push_str("================================\n\n");

        output.push_str(&format!("Total Issues: {}\n", self.summary.total_issues));
        output.push_str(&format!("  Critical: {}\n", self.summary.critical_issues));
        output.push_str(&format!("  High:     {}\n", self.summary.high_issues));
        output.push_str(&format!("  Medium:   {}\n", self.summary.medium_issues));
        output.push_str(&format!("  Low:      {}\n", self.summary.low_issues));
        output.push_str(&format!("  Info:     {}\n\n", self.summary.info_issues));

        if self.is_compliant() {
            output.push_str("Status: COMPLIANT\n\n");
        } else {
            output.push_str("Status: NON-COMPLIANT\n\n");
        }

        if !self.issues.is_empty() {
            output.push_str("Issues:\n");
            output.push_str("-------\n\n");

            for (i, issue) in self.issues.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{:?}] {} ({})\n",
                    i + 1,
                    issue.severity,
                    issue.title,
                    issue.id
                ));
                output.push_str(&format!("   {}\n", issue.description));
                if let Some(location) = &issue.location {
                    output.push_str(&format!("   Location: {location}\n"));
                }
                output.push('\n');
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_creation() {
        let issue = ComplianceIssue::new(
            "TEST-001".to_string(),
            "Test Issue".to_string(),
            "Description".to_string(),
            IssueSeverity::High,
        );

        assert_eq!(issue.id, "TEST-001");
        assert_eq!(issue.severity, IssueSeverity::High);
    }

    #[test]
    fn test_report_creation() {
        let report = ComplianceReport::new();
        assert_eq!(report.summary().total_issues, 0);
        assert!(report.is_compliant());
    }

    #[test]
    fn test_add_issue() {
        let mut report = ComplianceReport::new();
        let issue = ComplianceIssue::new(
            "TEST-001".to_string(),
            "Test".to_string(),
            "Desc".to_string(),
            IssueSeverity::Critical,
        );

        report.add_issue(issue);

        assert_eq!(report.summary().total_issues, 1);
        assert_eq!(report.summary().critical_issues, 1);
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_issues_by_severity() {
        let mut report = ComplianceReport::new();
        report.add_issue(ComplianceIssue::new(
            "1".to_string(),
            "High".to_string(),
            "Desc".to_string(),
            IssueSeverity::High,
        ));
        report.add_issue(ComplianceIssue::new(
            "2".to_string(),
            "Low".to_string(),
            "Desc".to_string(),
            IssueSeverity::Low,
        ));

        let high_issues = report.issues_by_severity(IssueSeverity::High);
        assert_eq!(high_issues.len(), 1);
    }

    #[test]
    fn test_to_text() {
        let mut report = ComplianceReport::new();
        report.add_issue(ComplianceIssue::new(
            "TEST-001".to_string(),
            "Test Issue".to_string(),
            "Description".to_string(),
            IssueSeverity::Medium,
        ));

        let text = report.to_text();
        assert!(text.contains("Accessibility Compliance Report"));
        assert!(text.contains("Test Issue"));
    }
}
