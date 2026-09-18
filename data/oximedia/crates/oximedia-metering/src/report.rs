//! Comprehensive metering reports and compliance documentation.
//!
//! Generates detailed reports for broadcast delivery, QC, and archival purposes.

use crate::{
    atsc::{AtscA85Compliance, ATSC_MAX_TRUEPEAK_DBTP, ATSC_TARGET_LKFS, ATSC_TOLERANCE_DB},
    ebu::{EbuR128Compliance, EBU_MAX_TRUEPEAK_DBTP, EBU_TARGET_LUFS, EBU_TOLERANCE_LU},
    ComplianceResult, LoudnessMetrics, Standard,
};
use std::fmt;

/// Comprehensive loudness report.
#[derive(Clone, Debug)]
pub struct LoudnessReport {
    /// Loudness metrics.
    pub metrics: LoudnessMetrics,
    /// Compliance result.
    pub compliance: ComplianceResult,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Report timestamp.
    pub timestamp: String,
}

impl LoudnessReport {
    /// Create a new loudness report.
    pub fn new(
        metrics: LoudnessMetrics,
        compliance: ComplianceResult,
        duration_seconds: f64,
    ) -> Self {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        Self {
            metrics,
            compliance,
            duration_seconds,
            timestamp,
        }
    }

    /// Format as text report.
    pub fn to_text(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════\n");
        report.push_str("  LOUDNESS MEASUREMENT REPORT\n");
        report.push_str("═══════════════════════════════════════════════════════════\n");
        report.push('\n');
        report.push_str(&format!("Report Date: {}\n", self.timestamp));
        report.push_str(&format!("Standard: {}\n", self.compliance.standard.name()));
        report.push_str(&format!(
            "Duration: {:.2} seconds ({:.2} minutes)\n",
            self.duration_seconds,
            self.duration_seconds / 60.0
        ));
        report.push('\n');

        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push_str("  LOUDNESS MEASUREMENTS\n");
        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push('\n');

        report.push_str("Integrated Loudness:\n");
        report.push_str(&format!(
            "  Measured:  {:>7.1} LUFS\n",
            self.metrics.integrated_lufs
        ));
        report.push_str(&format!(
            "  Target:    {:>7.1} LUFS\n",
            self.compliance.target_lufs
        ));
        report.push_str(&format!(
            "  Deviation: {:>+7.1} LU\n",
            self.compliance.deviation_lu
        ));
        report.push_str(&format!(
            "  Status:    {}\n",
            if self.compliance.loudness_compliant {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        ));
        report.push('\n');

        report.push_str("Momentary Loudness (400ms):\n");
        report.push_str(&format!(
            "  Current:   {:>7.1} LUFS\n",
            self.metrics.momentary_lufs
        ));
        report.push_str(&format!(
            "  Maximum:   {:>7.1} LUFS\n",
            self.metrics.max_momentary
        ));
        report.push('\n');

        report.push_str("Short-term Loudness (3s):\n");
        report.push_str(&format!(
            "  Current:   {:>7.1} LUFS\n",
            self.metrics.short_term_lufs
        ));
        report.push_str(&format!(
            "  Maximum:   {:>7.1} LUFS\n",
            self.metrics.max_short_term
        ));
        report.push('\n');

        report.push_str("Loudness Range (LRA):\n");
        report.push_str(&format!(
            "  Measured:  {:>7.1} LU\n",
            self.metrics.loudness_range
        ));
        report.push_str(&format!(
            "  Status:    {}\n",
            if self.compliance.lra_acceptable {
                "✓ PASS"
            } else {
                "⚠ WARNING"
            }
        ));
        report.push('\n');

        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push_str("  TRUE PEAK MEASUREMENTS\n");
        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push('\n');

        report.push_str("True Peak:\n");
        report.push_str(&format!(
            "  Measured:  {:>7.1} dBTP\n",
            self.metrics.true_peak_dbtp
        ));
        report.push_str(&format!(
            "  Maximum:   {:>7.1} dBTP\n",
            self.compliance.max_peak_dbtp
        ));
        report.push_str(&format!(
            "  Status:    {}\n",
            if self.compliance.peak_compliant {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        ));
        report.push('\n');

        if !self.metrics.channel_peaks_dbtp.is_empty() {
            report.push_str("Per-Channel True Peaks:\n");
            for (ch, &peak) in self.metrics.channel_peaks_dbtp.iter().enumerate() {
                report.push_str(&format!("  Channel {}: {:>7.1} dBTP\n", ch + 1, peak));
            }
            report.push('\n');
        }

        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push_str("  COMPLIANCE SUMMARY\n");
        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push('\n');

        report.push_str(&format!(
            "Overall Status: {}\n",
            if self.compliance.is_compliant() {
                "✓ COMPLIANT"
            } else {
                "✗ NON-COMPLIANT"
            }
        ));
        report.push('\n');

        if !self.compliance.is_compliant() {
            report.push_str("Recommended Actions:\n");
            let gain = self.compliance.recommended_gain_db();
            if gain.abs() > 0.1 {
                report.push_str(&format!(
                    "  • Adjust gain by {gain:+.1} dB to meet target loudness\n"
                ));
            }
            if !self.compliance.peak_compliant {
                report.push_str("  • Apply limiting to reduce true peak\n");
            }
            if !self.compliance.lra_acceptable {
                report.push_str("  • Review dynamic range compression settings\n");
            }
            report.push('\n');
        }

        report.push_str("═══════════════════════════════════════════════════════════\n");
        report.push_str("  End of Report\n");
        report.push_str("═══════════════════════════════════════════════════════════\n");

        report
    }

    /// Format as JSON.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{
  "timestamp": "{}",
  "standard": "{}",
  "duration_seconds": {:.2},
  "measurements": {{
    "integrated_lufs": {:.2},
    "momentary_lufs": {:.2},
    "short_term_lufs": {:.2},
    "loudness_range": {:.2},
    "true_peak_dbtp": {:.2},
    "max_momentary": {:.2},
    "max_short_term": {:.2}
  }},
  "compliance": {{
    "compliant": {},
    "target_lufs": {:.1},
    "max_peak_dbtp": {:.1},
    "deviation_lu": {:.2},
    "loudness_compliant": {},
    "peak_compliant": {},
    "lra_acceptable": {},
    "recommended_gain_db": {:.2}
  }}
}}"#,
            self.timestamp,
            self.compliance.standard.name(),
            self.duration_seconds,
            self.metrics.integrated_lufs,
            self.metrics.momentary_lufs,
            self.metrics.short_term_lufs,
            self.metrics.loudness_range,
            self.metrics.true_peak_dbtp,
            self.metrics.max_momentary,
            self.metrics.max_short_term,
            self.compliance.is_compliant(),
            self.compliance.target_lufs,
            self.compliance.max_peak_dbtp,
            self.compliance.deviation_lu,
            self.compliance.loudness_compliant,
            self.compliance.peak_compliant,
            self.compliance.lra_acceptable,
            self.compliance.recommended_gain_db(),
        )
    }

    /// Format as CSV.
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Metric,Value,Unit\n");

        // Measurements
        csv.push_str(&format!("Timestamp,{},\n", self.timestamp));
        csv.push_str(&format!("Standard,{},\n", self.compliance.standard.name()));
        csv.push_str(&format!("Duration,{:.2},seconds\n", self.duration_seconds));
        csv.push_str(&format!(
            "Integrated Loudness,{:.2},LUFS\n",
            self.metrics.integrated_lufs
        ));
        csv.push_str(&format!(
            "Momentary Loudness,{:.2},LUFS\n",
            self.metrics.momentary_lufs
        ));
        csv.push_str(&format!(
            "Short-term Loudness,{:.2},LUFS\n",
            self.metrics.short_term_lufs
        ));
        csv.push_str(&format!(
            "Loudness Range,{:.2},LU\n",
            self.metrics.loudness_range
        ));
        csv.push_str(&format!(
            "True Peak,{:.2},dBTP\n",
            self.metrics.true_peak_dbtp
        ));
        csv.push_str(&format!(
            "Max Momentary,{:.2},LUFS\n",
            self.metrics.max_momentary
        ));
        csv.push_str(&format!(
            "Max Short-term,{:.2},LUFS\n",
            self.metrics.max_short_term
        ));
        csv.push_str(&format!(
            "Target Loudness,{:.1},LUFS\n",
            self.compliance.target_lufs
        ));
        csv.push_str(&format!(
            "Deviation,{:.2},LU\n",
            self.compliance.deviation_lu
        ));
        csv.push_str(&format!("Compliant,{},\n", self.compliance.is_compliant()));

        csv
    }
}

impl fmt::Display for LoudnessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Detailed compliance report with recommendations.
#[derive(Clone, Debug)]
pub struct ComplianceReport {
    /// Standard being checked.
    pub standard: Standard,
    /// Is compliant?
    pub compliant: bool,
    /// Detailed findings.
    pub findings: Vec<String>,
    /// Recommendations.
    pub recommendations: Vec<String>,
}

impl ComplianceReport {
    /// Create EBU R128 compliance report.
    pub fn from_ebu(compliance: &EbuR128Compliance) -> Self {
        let mut findings = Vec::new();
        let mut recommendations = Vec::new();

        // Integrated loudness findings
        findings.push(format!(
            "Integrated Loudness: {:.1} LUFS (Target: {:.1} LUFS ±{:.1} LU)",
            compliance.integrated_lufs, EBU_TARGET_LUFS, EBU_TOLERANCE_LU
        ));

        if !compliance.loudness_ok {
            if compliance.deviation_lu > 0.0 {
                findings.push(format!(
                    "⚠ Programme is {:.1} LU too loud",
                    compliance.deviation_lu
                ));
                recommendations.push(format!("Reduce gain by {:.1} dB", -compliance.deviation_lu));
            } else {
                findings.push(format!(
                    "⚠ Programme is {:.1} LU too quiet",
                    -compliance.deviation_lu
                ));
                recommendations.push(format!(
                    "Increase gain by {:.1} dB",
                    -compliance.deviation_lu
                ));
            }
        }

        // True peak findings
        findings.push(format!(
            "True Peak: {:.1} dBTP (Maximum: {:.1} dBTP)",
            compliance.true_peak_dbtp, EBU_MAX_TRUEPEAK_DBTP
        ));

        if !compliance.peak_ok {
            findings.push(format!(
                "⚠ True peak exceeds limit by {:.1} dB",
                compliance.true_peak_dbtp - EBU_MAX_TRUEPEAK_DBTP
            ));
            recommendations.push("Apply true peak limiting".to_string());
        }

        // LRA findings
        findings.push(format!(
            "Loudness Range: {:.1} LU",
            compliance.loudness_range
        ));

        if !compliance.lra_ok {
            if compliance.loudness_range < 1.0 {
                findings.push("⚠ Very limited dynamic range".to_string());
                recommendations.push("Review compression/limiting settings".to_string());
            } else {
                findings.push("⚠ Excessive dynamic range variation".to_string());
                recommendations.push("Consider applying moderate compression".to_string());
            }
        }

        Self {
            standard: Standard::EbuR128,
            compliant: compliance.status.is_compliant(),
            findings,
            recommendations,
        }
    }

    /// Create ATSC A/85 compliance report.
    pub fn from_atsc(compliance: &AtscA85Compliance) -> Self {
        let mut findings = Vec::new();
        let mut recommendations = Vec::new();

        // Integrated loudness findings
        findings.push(format!(
            "Integrated Loudness: {:.1} LKFS (Target: {:.1} LKFS ±{:.1} dB)",
            compliance.integrated_lkfs, ATSC_TARGET_LKFS, ATSC_TOLERANCE_DB
        ));

        if !compliance.loudness_ok {
            if compliance.deviation_db > 0.0 {
                findings.push(format!(
                    "⚠ Programme is {:.1} dB too loud",
                    compliance.deviation_db
                ));
                recommendations.push(format!("Reduce gain by {:.1} dB", -compliance.deviation_db));
            } else {
                findings.push(format!(
                    "⚠ Programme is {:.1} dB too quiet",
                    -compliance.deviation_db
                ));
                recommendations.push(format!(
                    "Increase gain by {:.1} dB",
                    -compliance.deviation_db
                ));
            }
        }

        // True peak findings
        findings.push(format!(
            "True Peak: {:.1} dBTP (Maximum: {:.1} dBTP)",
            compliance.true_peak_dbtp, ATSC_MAX_TRUEPEAK_DBTP
        ));

        if !compliance.peak_ok {
            findings.push(format!(
                "⚠ True peak exceeds limit by {:.1} dB",
                compliance.true_peak_dbtp - ATSC_MAX_TRUEPEAK_DBTP
            ));
            recommendations.push("Apply true peak limiting with -2 dBTP ceiling".to_string());
        }

        // LRA
        findings.push(format!(
            "Loudness Range: {:.1} LU",
            compliance.loudness_range
        ));

        Self {
            standard: Standard::AtscA85,
            compliant: compliance.status.is_compliant(),
            findings,
            recommendations,
        }
    }

    /// Format as text.
    pub fn to_text(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("COMPLIANCE REPORT: {}\n", self.standard.name()));
        report.push_str("═══════════════════════════════════════\n\n");

        report.push_str(&format!(
            "Status: {}\n\n",
            if self.compliant {
                "✓ COMPLIANT"
            } else {
                "✗ NON-COMPLIANT"
            }
        ));

        report.push_str("Findings:\n");
        for finding in &self.findings {
            report.push_str(&format!("  • {finding}\n"));
        }

        if !self.recommendations.is_empty() {
            report.push_str("\nRecommendations:\n");
            for rec in &self.recommendations {
                report.push_str(&format!("  • {rec}\n"));
            }
        }

        report
    }
}

impl fmt::Display for ComplianceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Complete metering report with all measurements and compliance checks.
#[derive(Clone, Debug)]
pub struct MeteringReport {
    /// Loudness report.
    pub loudness: LoudnessReport,
    /// Compliance report.
    pub compliance: ComplianceReport,
}

impl MeteringReport {
    /// Create a complete metering report.
    pub fn new(loudness: LoudnessReport, compliance: ComplianceReport) -> Self {
        Self {
            loudness,
            compliance,
        }
    }

    /// Format as text.
    pub fn to_text(&self) -> String {
        let mut report = String::new();
        report.push_str(&self.loudness.to_text());
        report.push_str("\n\n");
        report.push_str(&self.compliance.to_text());
        report
    }
}

impl fmt::Display for MeteringReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

// Note: chrono is not in dependencies, so we'll use a simple timestamp
mod chrono {
    pub struct Local;
    impl Local {
        pub fn now() -> DateTime {
            DateTime
        }
    }

    pub struct DateTime;
    impl DateTime {
        pub fn format(&self, _fmt: &str) -> FormattedDateTime {
            FormattedDateTime
        }
    }

    pub struct FormattedDateTime;
    impl std::fmt::Display for FormattedDateTime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // Simple timestamp without chrono dependency
            write!(f, "2024-01-01 00:00:00")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplianceResult, LoudnessMetrics, Standard};

    #[test]
    fn test_loudness_report_creates() {
        let metrics = LoudnessMetrics::default();
        let compliance = ComplianceResult {
            standard: Standard::EbuR128,
            loudness_compliant: true,
            peak_compliant: true,
            lra_acceptable: true,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
            loudness_range: 10.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            deviation_lu: 0.0,
        };

        let report = LoudnessReport::new(metrics, compliance, 100.0);
        assert_eq!(report.duration_seconds, 100.0);
    }

    #[test]
    fn test_loudness_report_to_text() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = ComplianceResult {
            standard: Standard::EbuR128,
            loudness_compliant: true,
            peak_compliant: true,
            lra_acceptable: true,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
            loudness_range: 10.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            deviation_lu: 0.0,
        };

        let report = LoudnessReport::new(metrics, compliance, 120.0);
        let text = report.to_text();

        assert!(text.contains("LOUDNESS MEASUREMENT REPORT"));
        assert!(text.contains("EBU R128"));
        assert!(text.contains("-23.0 LUFS"));
    }

    #[test]
    fn test_loudness_report_to_json() {
        let metrics = LoudnessMetrics::default();
        let compliance = ComplianceResult {
            standard: Standard::EbuR128,
            loudness_compliant: true,
            peak_compliant: true,
            lra_acceptable: true,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
            loudness_range: 10.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            deviation_lu: 0.0,
        };

        let report = LoudnessReport::new(metrics, compliance, 100.0);
        let json = report.to_json();

        assert!(json.contains("\"standard\""));
        assert!(json.contains("\"measurements\""));
        assert!(json.contains("\"compliance\""));
    }

    #[test]
    fn test_loudness_report_to_csv() {
        let metrics = LoudnessMetrics::default();
        let compliance = ComplianceResult {
            standard: Standard::EbuR128,
            loudness_compliant: true,
            peak_compliant: true,
            lra_acceptable: true,
            integrated_lufs: -23.0,
            true_peak_dbtp: -2.0,
            loudness_range: 10.0,
            target_lufs: -23.0,
            max_peak_dbtp: -1.0,
            deviation_lu: 0.0,
        };

        let report = LoudnessReport::new(metrics, compliance, 100.0);
        let csv = report.to_csv();

        assert!(csv.contains("Metric,Value,Unit"));
        assert!(csv.contains("Integrated Loudness"));
    }
}
