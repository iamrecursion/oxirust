#![allow(dead_code)]
//! Broadcast compliance checking engine for loudness normalization.
//!
//! This module provides a comprehensive compliance checker that validates audio
//! against multiple broadcast loudness standards simultaneously. It tracks
//! violations, generates reports, and provides pass/fail verdicts.

use std::fmt;

/// Broadcast standard identifier for compliance checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplianceStandard {
    /// EBU R128 (-23 LUFS, -1 dBTP).
    EbuR128,
    /// ATSC A/85 (-24 LKFS, -2 dBTP).
    AtscA85,
    /// ITU-R BS.1770-4 measurement standard.
    ItuBs1770,
    /// ARIB TR-B32 (Japan, -24 LKFS).
    AribTrB32,
    /// OP-59 (Australia, -24 LKFS, -2 dBTP).
    Op59,
    /// Spotify normalization (-14 LUFS).
    Spotify,
    /// YouTube normalization (-14 LUFS).
    YouTube,
    /// Apple Music / iTunes Sound Check (-16 LUFS).
    AppleMusic,
    /// Amazon Music (-14 LUFS).
    AmazonMusic,
    /// Tidal (-14 LUFS).
    Tidal,
}

impl fmt::Display for ComplianceStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EbuR128 => write!(f, "EBU R128"),
            Self::AtscA85 => write!(f, "ATSC A/85"),
            Self::ItuBs1770 => write!(f, "ITU-R BS.1770-4"),
            Self::AribTrB32 => write!(f, "ARIB TR-B32"),
            Self::Op59 => write!(f, "OP-59"),
            Self::Spotify => write!(f, "Spotify"),
            Self::YouTube => write!(f, "YouTube"),
            Self::AppleMusic => write!(f, "Apple Music"),
            Self::AmazonMusic => write!(f, "Amazon Music"),
            Self::Tidal => write!(f, "Tidal"),
        }
    }
}

/// Loudness limits for a compliance standard.
#[derive(Debug, Clone)]
pub struct LoudnessLimits {
    /// Target integrated loudness in LUFS.
    pub target_lufs: f64,
    /// Tolerance above the target in LU.
    pub tolerance_above: f64,
    /// Tolerance below the target in LU.
    pub tolerance_below: f64,
    /// Maximum true peak in dBTP.
    pub max_true_peak_dbtp: f64,
    /// Maximum loudness range (LRA) in LU, or None if unlimited.
    pub max_lra: Option<f64>,
    /// Minimum program length in seconds for valid measurement, or None.
    pub min_duration_secs: Option<f64>,
}

impl LoudnessLimits {
    /// Get the limits for a given standard.
    pub fn for_standard(standard: ComplianceStandard) -> Self {
        match standard {
            ComplianceStandard::EbuR128 => Self {
                target_lufs: -23.0,
                tolerance_above: 1.0,
                tolerance_below: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra: Some(20.0),
                min_duration_secs: None,
            },
            ComplianceStandard::AtscA85 => Self {
                target_lufs: -24.0,
                tolerance_above: 2.0,
                tolerance_below: 2.0,
                max_true_peak_dbtp: -2.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::ItuBs1770 => Self {
                target_lufs: -24.0,
                tolerance_above: 1.0,
                tolerance_below: 1.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::AribTrB32 => Self {
                target_lufs: -24.0,
                tolerance_above: 2.0,
                tolerance_below: 2.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::Op59 => Self {
                target_lufs: -24.0,
                tolerance_above: 1.0,
                tolerance_below: 1.0,
                max_true_peak_dbtp: -2.0,
                max_lra: Some(20.0),
                min_duration_secs: Some(30.0),
            },
            ComplianceStandard::Spotify => Self {
                target_lufs: -14.0,
                tolerance_above: 1.0,
                tolerance_below: 3.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::YouTube => Self {
                target_lufs: -14.0,
                tolerance_above: 1.0,
                tolerance_below: 3.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::AppleMusic => Self {
                target_lufs: -16.0,
                tolerance_above: 1.0,
                tolerance_below: 3.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::AmazonMusic => Self {
                target_lufs: -14.0,
                tolerance_above: 1.0,
                tolerance_below: 3.0,
                max_true_peak_dbtp: -2.0,
                max_lra: None,
                min_duration_secs: None,
            },
            ComplianceStandard::Tidal => Self {
                target_lufs: -14.0,
                tolerance_above: 1.0,
                tolerance_below: 3.0,
                max_true_peak_dbtp: -1.0,
                max_lra: None,
                min_duration_secs: None,
            },
        }
    }
}

/// Severity level of a compliance violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Informational (within tolerance but noteworthy).
    Info,
    /// Warning (approaching limits).
    Warning,
    /// Error (exceeds limits).
    Error,
    /// Critical (far exceeds limits).
    Critical,
}

impl fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single compliance violation.
#[derive(Debug, Clone)]
pub struct Violation {
    /// The standard that was violated.
    pub standard: ComplianceStandard,
    /// Severity of the violation.
    pub severity: ViolationSeverity,
    /// Description of the violation.
    pub description: String,
    /// Measured value.
    pub measured: f64,
    /// Limit value.
    pub limit: f64,
    /// How far the measurement exceeds the limit.
    pub excess: f64,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} - {}: measured={:.1}, limit={:.1}, excess={:.1}",
            self.severity, self.standard, self.description, self.measured, self.limit, self.excess,
        )
    }
}

/// Measured loudness parameters for compliance checking.
#[derive(Debug, Clone)]
pub struct LoudnessMeasurement {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Maximum true peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Loudness range in LU.
    pub lra_lu: f64,
    /// Short-term maximum loudness in LUFS.
    pub short_term_max_lufs: f64,
    /// Momentary maximum loudness in LUFS.
    pub momentary_max_lufs: f64,
    /// Program duration in seconds.
    pub duration_secs: f64,
}

impl LoudnessMeasurement {
    /// Create a new measurement with the given values.
    pub fn new(integrated_lufs: f64, true_peak_dbtp: f64, lra_lu: f64, duration_secs: f64) -> Self {
        Self {
            integrated_lufs,
            true_peak_dbtp,
            lra_lu,
            short_term_max_lufs: integrated_lufs + 5.0,
            momentary_max_lufs: integrated_lufs + 8.0,
            duration_secs,
        }
    }
}

/// Result of a compliance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceVerdict {
    /// Fully compliant.
    Pass,
    /// Compliant with warnings.
    PassWithWarnings,
    /// Non-compliant.
    Fail,
}

impl fmt::Display for ComplianceVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithWarnings => write!(f, "PASS (with warnings)"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Full compliance report for one or more standards.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Overall verdict.
    pub verdict: ComplianceVerdict,
    /// List of violations.
    pub violations: Vec<Violation>,
    /// Standards that were checked.
    pub standards_checked: Vec<ComplianceStandard>,
    /// The measurement that was checked.
    pub measurement: LoudnessMeasurement,
}

impl ComplianceReport {
    /// Check if the report has any errors or critical violations.
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity >= ViolationSeverity::Error)
    }

    /// Check if the report has any warnings.
    pub fn has_warnings(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Warning)
    }

    /// Count violations by severity.
    pub fn count_by_severity(&self, severity: ViolationSeverity) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == severity)
            .count()
    }
}

/// Compliance checker for validating audio loudness against broadcast standards.
#[derive(Debug)]
pub struct ComplianceChecker {
    /// Standards to check against.
    standards: Vec<ComplianceStandard>,
}

impl ComplianceChecker {
    /// Create a new compliance checker for the given standards.
    pub fn new(standards: Vec<ComplianceStandard>) -> Self {
        Self { standards }
    }

    /// Create a checker for all broadcast standards.
    pub fn all_broadcast() -> Self {
        Self::new(vec![
            ComplianceStandard::EbuR128,
            ComplianceStandard::AtscA85,
            ComplianceStandard::AribTrB32,
            ComplianceStandard::Op59,
        ])
    }

    /// Create a checker for all streaming platforms.
    pub fn all_streaming() -> Self {
        Self::new(vec![
            ComplianceStandard::Spotify,
            ComplianceStandard::YouTube,
            ComplianceStandard::AppleMusic,
            ComplianceStandard::AmazonMusic,
            ComplianceStandard::Tidal,
        ])
    }

    /// Check compliance of a measurement against all configured standards.
    pub fn check(&self, measurement: &LoudnessMeasurement) -> ComplianceReport {
        let mut violations = Vec::new();

        for &standard in &self.standards {
            let limits = LoudnessLimits::for_standard(standard);
            self.check_standard(measurement, standard, &limits, &mut violations);
        }

        let verdict = if violations
            .iter()
            .any(|v| v.severity >= ViolationSeverity::Error)
        {
            ComplianceVerdict::Fail
        } else if violations
            .iter()
            .any(|v| v.severity >= ViolationSeverity::Warning)
        {
            ComplianceVerdict::PassWithWarnings
        } else {
            ComplianceVerdict::Pass
        };

        ComplianceReport {
            verdict,
            violations,
            standards_checked: self.standards.clone(),
            measurement: measurement.clone(),
        }
    }

    /// Check a single standard.
    fn check_standard(
        &self,
        measurement: &LoudnessMeasurement,
        standard: ComplianceStandard,
        limits: &LoudnessLimits,
        violations: &mut Vec<Violation>,
    ) {
        // Check integrated loudness (above target)
        let upper = limits.target_lufs + limits.tolerance_above;
        if measurement.integrated_lufs > upper {
            let excess = measurement.integrated_lufs - upper;
            let severity = if excess > 3.0 {
                ViolationSeverity::Critical
            } else if excess > 1.0 {
                ViolationSeverity::Error
            } else {
                ViolationSeverity::Warning
            };
            violations.push(Violation {
                standard,
                severity,
                description: "Integrated loudness above target".to_string(),
                measured: measurement.integrated_lufs,
                limit: upper,
                excess,
            });
        }

        // Check integrated loudness (below target)
        let lower = limits.target_lufs - limits.tolerance_below;
        if measurement.integrated_lufs < lower {
            let excess = lower - measurement.integrated_lufs;
            let severity = if excess > 3.0 {
                ViolationSeverity::Critical
            } else if excess > 1.0 {
                ViolationSeverity::Error
            } else {
                ViolationSeverity::Warning
            };
            violations.push(Violation {
                standard,
                severity,
                description: "Integrated loudness below target".to_string(),
                measured: measurement.integrated_lufs,
                limit: lower,
                excess,
            });
        }

        // Check true peak
        if measurement.true_peak_dbtp > limits.max_true_peak_dbtp {
            let excess = measurement.true_peak_dbtp - limits.max_true_peak_dbtp;
            let severity = if excess > 2.0 {
                ViolationSeverity::Critical
            } else if excess > 0.5 {
                ViolationSeverity::Error
            } else {
                ViolationSeverity::Warning
            };
            violations.push(Violation {
                standard,
                severity,
                description: "True peak exceeds limit".to_string(),
                measured: measurement.true_peak_dbtp,
                limit: limits.max_true_peak_dbtp,
                excess,
            });
        }

        // Check LRA
        if let Some(max_lra) = limits.max_lra {
            if measurement.lra_lu > max_lra {
                let excess = measurement.lra_lu - max_lra;
                violations.push(Violation {
                    standard,
                    severity: if excess > 5.0 {
                        ViolationSeverity::Error
                    } else {
                        ViolationSeverity::Warning
                    },
                    description: "Loudness range exceeds limit".to_string(),
                    measured: measurement.lra_lu,
                    limit: max_lra,
                    excess,
                });
            }
        }

        // Check minimum duration
        if let Some(min_dur) = limits.min_duration_secs {
            if measurement.duration_secs < min_dur {
                violations.push(Violation {
                    standard,
                    severity: ViolationSeverity::Info,
                    description: "Program shorter than minimum recommended duration".to_string(),
                    measured: measurement.duration_secs,
                    limit: min_dur,
                    excess: min_dur - measurement.duration_secs,
                });
            }
        }
    }

    /// Get the configured standards.
    pub fn standards(&self) -> &[ComplianceStandard] {
        &self.standards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_standard_display() {
        assert_eq!(format!("{}", ComplianceStandard::EbuR128), "EBU R128");
        assert_eq!(format!("{}", ComplianceStandard::AtscA85), "ATSC A/85");
        assert_eq!(format!("{}", ComplianceStandard::Spotify), "Spotify");
    }

    #[test]
    fn test_loudness_limits_ebu_r128() {
        let limits = LoudnessLimits::for_standard(ComplianceStandard::EbuR128);
        assert!((limits.target_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!((limits.max_true_peak_dbtp - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(limits.max_lra, Some(20.0));
    }

    #[test]
    fn test_loudness_limits_atsc_a85() {
        let limits = LoudnessLimits::for_standard(ComplianceStandard::AtscA85);
        assert!((limits.target_lufs - (-24.0)).abs() < f64::EPSILON);
        assert!((limits.max_true_peak_dbtp - (-2.0)).abs() < f64::EPSILON);
        assert_eq!(limits.max_lra, None);
    }

    #[test]
    fn test_loudness_limits_spotify() {
        let limits = LoudnessLimits::for_standard(ComplianceStandard::Spotify);
        assert!((limits.target_lufs - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_violation_display() {
        let v = Violation {
            standard: ComplianceStandard::EbuR128,
            severity: ViolationSeverity::Error,
            description: "Too loud".to_string(),
            measured: -20.0,
            limit: -22.0,
            excess: 2.0,
        };
        let s = format!("{v}");
        assert!(s.contains("ERROR"));
        assert!(s.contains("EBU R128"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ViolationSeverity::Info < ViolationSeverity::Warning);
        assert!(ViolationSeverity::Warning < ViolationSeverity::Error);
        assert!(ViolationSeverity::Error < ViolationSeverity::Critical);
    }

    #[test]
    fn test_compliance_pass() {
        let checker = ComplianceChecker::new(vec![ComplianceStandard::EbuR128]);
        let measurement = LoudnessMeasurement::new(-23.0, -2.0, 10.0, 120.0);
        let report = checker.check(&measurement);
        assert_eq!(report.verdict, ComplianceVerdict::Pass);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_compliance_fail_loud() {
        let checker = ComplianceChecker::new(vec![ComplianceStandard::EbuR128]);
        // Way too loud: -18 LUFS when target is -23 +/- 1
        let measurement = LoudnessMeasurement::new(-18.0, -2.0, 10.0, 120.0);
        let report = checker.check(&measurement);
        assert_eq!(report.verdict, ComplianceVerdict::Fail);
        assert!(report.has_errors());
    }

    #[test]
    fn test_compliance_fail_peak() {
        let checker = ComplianceChecker::new(vec![ComplianceStandard::EbuR128]);
        // True peak exceeds limit
        let measurement = LoudnessMeasurement::new(-23.0, 0.5, 10.0, 120.0);
        let report = checker.check(&measurement);
        assert_eq!(report.verdict, ComplianceVerdict::Fail);
    }

    #[test]
    fn test_compliance_fail_lra() {
        let checker = ComplianceChecker::new(vec![ComplianceStandard::EbuR128]);
        // LRA exceeds limit of 20 LU (by more than 5 -> Error)
        let measurement = LoudnessMeasurement::new(-23.0, -2.0, 30.0, 120.0);
        let report = checker.check(&measurement);
        assert!(report
            .violations
            .iter()
            .any(|v| v.description.contains("range")));
    }

    #[test]
    fn test_compliance_all_broadcast() {
        let checker = ComplianceChecker::all_broadcast();
        assert_eq!(checker.standards().len(), 4);
        let measurement = LoudnessMeasurement::new(-24.0, -3.0, 10.0, 120.0);
        let report = checker.check(&measurement);
        assert_eq!(report.standards_checked.len(), 4);
    }

    #[test]
    fn test_compliance_all_streaming() {
        let checker = ComplianceChecker::all_streaming();
        assert_eq!(checker.standards().len(), 5);
        // Use -15.5 LUFS which is within tolerance of all streaming platforms:
        // Spotify/YouTube/Amazon/Tidal: -14 LUFS +1/-3 => range [-17, -13]
        // Apple Music: -16 LUFS +1/-3 => range [-19, -15]
        let measurement = LoudnessMeasurement::new(-15.5, -2.0, 8.0, 200.0);
        let report = checker.check(&measurement);
        assert_eq!(report.verdict, ComplianceVerdict::Pass);
    }

    #[test]
    fn test_compliance_report_count_severity() {
        let checker = ComplianceChecker::new(vec![ComplianceStandard::EbuR128]);
        // Far too loud and peak too high
        let measurement = LoudnessMeasurement::new(-15.0, 1.0, 25.0, 120.0);
        let report = checker.check(&measurement);
        assert!(
            report.count_by_severity(ViolationSeverity::Error) > 0
                || report.count_by_severity(ViolationSeverity::Critical) > 0
        );
    }

    #[test]
    fn test_measurement_new() {
        let m = LoudnessMeasurement::new(-23.0, -1.5, 12.0, 60.0);
        assert!((m.integrated_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!((m.true_peak_dbtp - (-1.5)).abs() < f64::EPSILON);
        assert!((m.lra_lu - 12.0).abs() < f64::EPSILON);
        assert!((m.duration_secs - 60.0).abs() < f64::EPSILON);
        // short_term_max defaults to integrated + 5
        assert!((m.short_term_max_lufs - (-18.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compliance_verdict_display() {
        assert_eq!(format!("{}", ComplianceVerdict::Pass), "PASS");
        assert_eq!(format!("{}", ComplianceVerdict::Fail), "FAIL");
        assert!(format!("{}", ComplianceVerdict::PassWithWarnings).contains("warning"));
    }
}
