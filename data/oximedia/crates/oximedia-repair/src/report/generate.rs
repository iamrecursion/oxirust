//! Report generation functionality.
//!
//! This module provides functions to generate detailed repair reports.

use crate::Issue;

/// Generate repair report.
pub fn generate_report(
    all_issues: &[Issue],
    fixed_issues: &[Issue],
    unfixed_issues: &[Issue],
) -> String {
    let mut report = String::new();

    report.push_str("=== OxiMedia Repair Report ===\n\n");

    report.push_str(&format!("Total issues detected: {}\n", all_issues.len()));
    report.push_str(&format!("Issues fixed: {}\n", fixed_issues.len()));
    report.push_str(&format!("Issues unfixed: {}\n", unfixed_issues.len()));

    if !fixed_issues.is_empty() {
        report.push_str("\n--- Fixed Issues ---\n");
        for (i, issue) in fixed_issues.iter().enumerate() {
            report.push_str(&format!(
                "{}. {:?} - {}\n",
                i + 1,
                issue.issue_type,
                issue.description
            ));
        }
    }

    if !unfixed_issues.is_empty() {
        report.push_str("\n--- Unfixed Issues ---\n");
        for (i, issue) in unfixed_issues.iter().enumerate() {
            report.push_str(&format!(
                "{}. {:?} - {}\n",
                i + 1,
                issue.issue_type,
                issue.description
            ));
        }
    }

    report.push_str("\n=== End of Report ===\n");

    report
}

/// Generate summary report.
pub fn generate_summary(fixed: usize, _unfixed: usize, total: usize) -> String {
    let success_rate = if total > 0 {
        (fixed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    format!(
        "Repair Summary: {}/{} issues fixed ({:.1}% success rate)",
        fixed, total, success_rate
    )
}

/// Generate detailed issue report.
pub fn generate_issue_report(issue: &Issue) -> String {
    let mut report = String::new();

    report.push_str(&format!("Issue Type: {:?}\n", issue.issue_type));
    report.push_str(&format!("Severity: {:?}\n", issue.severity));
    report.push_str(&format!("Description: {}\n", issue.description));

    if let Some(location) = issue.location {
        report.push_str(&format!("Location: byte offset {}\n", location));
    }

    report.push_str(&format!("Fixable: {}\n", issue.fixable));

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IssueType, Severity};

    #[test]
    fn test_generate_summary() {
        let summary = generate_summary(8, 2, 10);
        assert!(summary.contains("8/10"));
        assert!(summary.contains("80.0%"));
    }

    #[test]
    fn test_generate_issue_report() {
        let issue = Issue {
            issue_type: IssueType::CorruptedHeader,
            severity: Severity::High,
            description: "Test issue".to_string(),
            location: Some(100),
            fixable: true,
            confidence: 0.8,
        };

        let report = generate_issue_report(&issue);
        assert!(report.contains("CorruptedHeader"));
        assert!(report.contains("High"));
        assert!(report.contains("byte offset 100"));
    }

    #[test]
    fn test_generate_report() {
        let all_issues = vec![
            Issue {
                issue_type: IssueType::CorruptedHeader,
                severity: Severity::High,
                description: "Header corrupt".to_string(),
                location: None,
                fixable: true,
                confidence: 0.8,
            },
            Issue {
                issue_type: IssueType::MissingIndex,
                severity: Severity::Medium,
                description: "Index missing".to_string(),
                location: None,
                fixable: false,
                confidence: 0.8,
            },
        ];

        let fixed = vec![all_issues[0].clone()];
        let unfixed = vec![all_issues[1].clone()];

        let report = generate_report(&all_issues, &fixed, &unfixed);

        assert!(report.contains("Total issues detected: 2"));
        assert!(report.contains("Issues fixed: 1"));
        assert!(report.contains("Issues unfixed: 1"));
    }
}
