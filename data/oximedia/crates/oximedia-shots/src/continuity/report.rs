//! Continuity report generation.

use super::check::{ContinuityIssue, Severity};

/// Continuity report generator.
pub struct ContinuityReporter;

impl ContinuityReporter {
    /// Create a new continuity reporter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate a continuity report from issues.
    #[must_use]
    pub fn generate_report(&self, issues: &[ContinuityIssue]) -> ContinuityReport {
        let total_issues = issues.len();
        let high_severity = issues
            .iter()
            .filter(|i| i.severity == Severity::High)
            .count();
        let medium_severity = issues
            .iter()
            .filter(|i| i.severity == Severity::Medium)
            .count();
        let low_severity = issues
            .iter()
            .filter(|i| i.severity == Severity::Low)
            .count();

        ContinuityReport {
            total_issues,
            high_severity,
            medium_severity,
            low_severity,
            issues: issues.to_vec(),
        }
    }
}

impl Default for ContinuityReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuity report.
#[derive(Debug, Clone)]
pub struct ContinuityReport {
    /// Total number of issues.
    pub total_issues: usize,
    /// Number of high severity issues.
    pub high_severity: usize,
    /// Number of medium severity issues.
    pub medium_severity: usize,
    /// Number of low severity issues.
    pub low_severity: usize,
    /// All issues.
    pub issues: Vec<ContinuityIssue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuity_reporter_creation() {
        let _reporter = ContinuityReporter::new();
    }

    #[test]
    fn test_generate_empty_report() {
        let reporter = ContinuityReporter::new();
        let report = reporter.generate_report(&[]);
        assert_eq!(report.total_issues, 0);
    }
}
