//! Compliance report generation.

use super::ComplianceReport;

/// Compliance report generator.
pub struct ComplianceReportGenerator;

impl ComplianceReportGenerator {
    /// Generate a formatted report.
    #[must_use]
    pub fn generate(report: &ComplianceReport) -> String {
        let mut output = String::new();
        output.push_str("Compliance Report\n");
        output.push_str("=================\n\n");

        if let Some(standard) = &report.standard {
            output.push_str(&format!("Standard: {:?}\n", standard));
        }

        output.push_str(&format!("Compliant: {}\n", report.is_compliant));
        output.push_str(&format!("Audio Compliant: {}\n", report.audio_compliant));
        output.push_str(&format!("Video Compliant: {}\n", report.video_compliant));

        if !report.violations.is_empty() {
            output.push_str("\nViolations:\n");
            for violation in &report.violations {
                output.push_str(&format!(
                    "  - [{:?}] {}: {}\n",
                    violation.severity, violation.violation_type, violation.description
                ));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_generation() {
        let report = ComplianceReport::default();
        let output = ComplianceReportGenerator::generate(&report);
        assert!(output.contains("Compliance Report"));
    }
}
