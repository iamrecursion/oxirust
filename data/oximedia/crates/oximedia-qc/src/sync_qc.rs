#![allow(dead_code)]
//! Audio/video synchronization quality control checks.
//!
//! Detects lip-sync errors, A/V drift over time, stream discontinuities,
//! and timestamp monotonicity violations in media files.

/// Direction of A/V sync offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// Audio leads video (audio plays before corresponding video frame).
    AudioLeads,
    /// Video leads audio (video appears before corresponding audio).
    VideoLeads,
    /// Perfectly synchronized.
    InSync,
}

impl std::fmt::Display for SyncDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioLeads => write!(f, "audio leads video"),
            Self::VideoLeads => write!(f, "video leads audio"),
            Self::InSync => write!(f, "in sync"),
        }
    }
}

/// A single sync measurement at a point in time.
#[derive(Debug, Clone)]
pub struct SyncMeasurement {
    /// Timestamp in seconds from the start of the file.
    pub timestamp_secs: f64,
    /// Measured A/V offset in milliseconds (positive = audio leads).
    pub offset_ms: f64,
}

impl SyncMeasurement {
    /// Creates a new sync measurement.
    #[must_use]
    pub fn new(timestamp_secs: f64, offset_ms: f64) -> Self {
        Self {
            timestamp_secs,
            offset_ms,
        }
    }

    /// Returns the direction of the sync offset.
    #[must_use]
    pub fn direction(&self) -> SyncDirection {
        if self.offset_ms > 0.5 {
            SyncDirection::AudioLeads
        } else if self.offset_ms < -0.5 {
            SyncDirection::VideoLeads
        } else {
            SyncDirection::InSync
        }
    }

    /// Returns the absolute offset in milliseconds.
    #[must_use]
    pub fn abs_offset_ms(&self) -> f64 {
        self.offset_ms.abs()
    }
}

/// Severity level for sync findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSeverity {
    /// Informational — within tolerance.
    Info,
    /// Warning — perceptible but minor.
    Warning,
    /// Error — clearly perceptible sync issue.
    Error,
    /// Critical — severe sync failure.
    Critical,
}

impl std::fmt::Display for SyncSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single sync QC finding.
#[derive(Debug, Clone)]
pub struct SyncFinding {
    /// Severity of the finding.
    pub severity: SyncSeverity,
    /// Short code for the check.
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Timestamp in the file where the issue was detected.
    pub timestamp_secs: Option<f64>,
}

impl SyncFinding {
    /// Creates a new sync finding.
    #[must_use]
    pub fn new(severity: SyncSeverity, code: &str, message: &str) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message: message.to_string(),
            timestamp_secs: None,
        }
    }

    /// Attaches a timestamp to this finding.
    #[must_use]
    pub fn at_timestamp(mut self, ts: f64) -> Self {
        self.timestamp_secs = Some(ts);
        self
    }

    /// Returns whether this finding indicates a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self.severity, SyncSeverity::Error | SyncSeverity::Critical)
    }
}

/// Result of a sync QC analysis.
#[derive(Debug, Clone)]
pub struct SyncQcReport {
    /// Whether the overall check passed.
    pub passed: bool,
    /// All findings.
    pub findings: Vec<SyncFinding>,
    /// Number of measurement points analyzed.
    pub measurement_count: usize,
    /// Maximum observed absolute offset in ms.
    pub max_offset_ms: f64,
    /// Mean absolute offset in ms.
    pub mean_offset_ms: f64,
}

impl SyncQcReport {
    /// Creates a new empty sync QC report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            findings: Vec::new(),
            measurement_count: 0,
            max_offset_ms: 0.0,
            mean_offset_ms: 0.0,
        }
    }

    /// Adds a finding and updates pass/fail status.
    pub fn add_finding(&mut self, finding: SyncFinding) {
        if finding.is_failure() {
            self.passed = false;
        }
        self.findings.push(finding);
    }

    /// Returns only the error and critical findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&SyncFinding> {
        self.findings.iter().filter(|f| f.is_failure()).collect()
    }
}

impl Default for SyncQcReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for sync QC checks.
#[derive(Debug, Clone)]
pub struct SyncQcConfig {
    /// Warning threshold for A/V offset in ms (default 20).
    pub warning_threshold_ms: f64,
    /// Error threshold for A/V offset in ms (default 45).
    pub error_threshold_ms: f64,
    /// Critical threshold for A/V offset in ms (default 100).
    pub critical_threshold_ms: f64,
    /// Maximum allowed drift rate in ms/minute (default 5.0).
    pub max_drift_rate_ms_per_min: f64,
    /// Whether to check for timestamp monotonicity.
    pub check_monotonicity: bool,
    /// Maximum allowed timestamp gap in seconds before flagging discontinuity.
    pub max_gap_secs: f64,
}

impl SyncQcConfig {
    /// Creates a new sync QC configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            warning_threshold_ms: 20.0,
            error_threshold_ms: 45.0,
            critical_threshold_ms: 100.0,
            max_drift_rate_ms_per_min: 5.0,
            check_monotonicity: true,
            max_gap_secs: 2.0,
        }
    }

    /// Sets the warning threshold.
    #[must_use]
    pub fn with_warning_threshold(mut self, ms: f64) -> Self {
        self.warning_threshold_ms = ms;
        self
    }

    /// Sets the error threshold.
    #[must_use]
    pub fn with_error_threshold(mut self, ms: f64) -> Self {
        self.error_threshold_ms = ms;
        self
    }

    /// Sets the critical threshold.
    #[must_use]
    pub fn with_critical_threshold(mut self, ms: f64) -> Self {
        self.critical_threshold_ms = ms;
        self
    }

    /// Sets the maximum drift rate.
    #[must_use]
    pub fn with_max_drift_rate(mut self, ms_per_min: f64) -> Self {
        self.max_drift_rate_ms_per_min = ms_per_min;
        self
    }
}

impl Default for SyncQcConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio/video synchronization quality checker.
///
/// Analyzes a series of sync measurements to detect offset issues, drift,
/// and timestamp discontinuities.
#[derive(Debug, Clone)]
pub struct SyncQcChecker {
    /// Configuration for the checker.
    config: SyncQcConfig,
}

impl SyncQcChecker {
    /// Creates a new sync QC checker with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SyncQcConfig::new(),
        }
    }

    /// Creates a new sync QC checker with the given configuration.
    #[must_use]
    pub fn with_config(config: SyncQcConfig) -> Self {
        Self { config }
    }

    /// Analyzes a sequence of sync measurements.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, measurements: &[SyncMeasurement]) -> SyncQcReport {
        let mut report = SyncQcReport::new();

        if measurements.is_empty() {
            report.add_finding(SyncFinding::new(
                SyncSeverity::Info,
                "SYNC-001",
                "No sync measurements available",
            ));
            return report;
        }

        report.measurement_count = measurements.len();

        // Compute stats
        let mut max_abs = 0.0_f64;
        let mut sum_abs = 0.0_f64;

        for m in measurements {
            let abs_off = m.abs_offset_ms();
            if abs_off > max_abs {
                max_abs = abs_off;
            }
            sum_abs += abs_off;
        }

        report.max_offset_ms = max_abs;
        report.mean_offset_ms = sum_abs / measurements.len() as f64;

        // Per-measurement threshold checks
        for m in measurements {
            let abs_off = m.abs_offset_ms();
            if abs_off >= self.config.critical_threshold_ms {
                report.add_finding(
                    SyncFinding::new(
                        SyncSeverity::Critical,
                        "SYNC-010",
                        &format!(
                            "Critical A/V offset: {:.1} ms ({}) at {:.2}s",
                            m.offset_ms,
                            m.direction(),
                            m.timestamp_secs
                        ),
                    )
                    .at_timestamp(m.timestamp_secs),
                );
            } else if abs_off >= self.config.error_threshold_ms {
                report.add_finding(
                    SyncFinding::new(
                        SyncSeverity::Error,
                        "SYNC-011",
                        &format!(
                            "A/V offset error: {:.1} ms ({}) at {:.2}s",
                            m.offset_ms,
                            m.direction(),
                            m.timestamp_secs
                        ),
                    )
                    .at_timestamp(m.timestamp_secs),
                );
            } else if abs_off >= self.config.warning_threshold_ms {
                report.add_finding(
                    SyncFinding::new(
                        SyncSeverity::Warning,
                        "SYNC-012",
                        &format!(
                            "A/V offset warning: {:.1} ms ({}) at {:.2}s",
                            m.offset_ms,
                            m.direction(),
                            m.timestamp_secs
                        ),
                    )
                    .at_timestamp(m.timestamp_secs),
                );
            }
        }

        // Drift analysis
        self.check_drift(measurements, &mut report);

        // Monotonicity check
        if self.config.check_monotonicity {
            self.check_monotonicity(measurements, &mut report);
        }

        report
    }

    /// Checks for systematic drift over time.
    fn check_drift(&self, measurements: &[SyncMeasurement], report: &mut SyncQcReport) {
        if measurements.len() < 2 {
            return;
        }

        let first = &measurements[0];
        let last = &measurements[measurements.len() - 1];

        let time_span_min = (last.timestamp_secs - first.timestamp_secs) / 60.0;
        if time_span_min < 0.5 {
            return; // Too short to detect meaningful drift
        }

        let offset_change = last.offset_ms - first.offset_ms;
        let drift_rate = offset_change.abs() / time_span_min;

        if drift_rate > self.config.max_drift_rate_ms_per_min {
            let direction = if offset_change > 0.0 {
                "audio drifting ahead"
            } else {
                "video drifting ahead"
            };
            report.add_finding(SyncFinding::new(
                SyncSeverity::Error,
                "SYNC-020",
                &format!(
                    "Systematic drift detected: {:.2} ms/min ({direction}), total change {:.1} ms over {:.1} min",
                    drift_rate, offset_change.abs(), time_span_min
                ),
            ));
        }
    }

    /// Checks for non-monotonic timestamps.
    fn check_monotonicity(&self, measurements: &[SyncMeasurement], report: &mut SyncQcReport) {
        for window in measurements.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            // Non-monotonic timestamp
            if curr.timestamp_secs < prev.timestamp_secs {
                report.add_finding(
                    SyncFinding::new(
                        SyncSeverity::Error,
                        "SYNC-030",
                        &format!(
                            "Non-monotonic timestamp: {:.3}s -> {:.3}s",
                            prev.timestamp_secs, curr.timestamp_secs
                        ),
                    )
                    .at_timestamp(curr.timestamp_secs),
                );
            }

            // Timestamp gap / discontinuity
            let gap = curr.timestamp_secs - prev.timestamp_secs;
            if gap > self.config.max_gap_secs {
                report.add_finding(
                    SyncFinding::new(
                        SyncSeverity::Warning,
                        "SYNC-031",
                        &format!(
                            "Timestamp gap of {:.3}s between {:.3}s and {:.3}s (max {:.1}s)",
                            gap, prev.timestamp_secs, curr.timestamp_secs, self.config.max_gap_secs
                        ),
                    )
                    .at_timestamp(curr.timestamp_secs),
                );
            }
        }
    }
}

impl Default for SyncQcChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_measurement_direction_in_sync() {
        let m = SyncMeasurement::new(0.0, 0.3);
        assert_eq!(m.direction(), SyncDirection::InSync);
    }

    #[test]
    fn test_sync_measurement_direction_audio_leads() {
        let m = SyncMeasurement::new(1.0, 25.0);
        assert_eq!(m.direction(), SyncDirection::AudioLeads);
    }

    #[test]
    fn test_sync_measurement_direction_video_leads() {
        let m = SyncMeasurement::new(1.0, -30.0);
        assert_eq!(m.direction(), SyncDirection::VideoLeads);
    }

    #[test]
    fn test_abs_offset() {
        let m = SyncMeasurement::new(0.0, -42.5);
        assert!((m.abs_offset_ms() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_measurements() {
        let checker = SyncQcChecker::new();
        let report = checker.analyze(&[]);
        assert!(report.passed);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-001"));
    }

    #[test]
    fn test_all_in_sync() {
        let checker = SyncQcChecker::new();
        let measurements = vec![
            SyncMeasurement::new(0.0, 0.5),
            SyncMeasurement::new(1.0, -0.3),
            SyncMeasurement::new(2.0, 0.2),
        ];
        let report = checker.analyze(&measurements);
        assert!(report.passed);
        assert_eq!(report.measurement_count, 3);
    }

    #[test]
    fn test_warning_threshold() {
        let checker = SyncQcChecker::new();
        let measurements = vec![SyncMeasurement::new(5.0, 25.0)];
        let report = checker.analyze(&measurements);
        assert!(report.passed); // Warning doesn't fail
        assert!(report.findings.iter().any(|f| f.code == "SYNC-012"));
    }

    #[test]
    fn test_error_threshold() {
        let checker = SyncQcChecker::new();
        let measurements = vec![SyncMeasurement::new(5.0, 60.0)];
        let report = checker.analyze(&measurements);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-011"));
    }

    #[test]
    fn test_critical_threshold() {
        let checker = SyncQcChecker::new();
        let measurements = vec![SyncMeasurement::new(5.0, -150.0)];
        let report = checker.analyze(&measurements);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-010"));
    }

    #[test]
    fn test_drift_detection() {
        let checker = SyncQcChecker::new();
        // Simulate drift: from 0ms to 60ms over 2 minutes = 30 ms/min
        let measurements = vec![
            SyncMeasurement::new(0.0, 0.0),
            SyncMeasurement::new(60.0, 30.0),
            SyncMeasurement::new(120.0, 60.0),
        ];
        let report = checker.analyze(&measurements);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-020"));
    }

    #[test]
    fn test_no_drift_for_stable_offset() {
        let checker = SyncQcChecker::new();
        let measurements = vec![
            SyncMeasurement::new(0.0, 10.0),
            SyncMeasurement::new(60.0, 10.5),
            SyncMeasurement::new(120.0, 10.2),
        ];
        let report = checker.analyze(&measurements);
        assert!(!report.findings.iter().any(|f| f.code == "SYNC-020"));
    }

    #[test]
    fn test_non_monotonic_timestamp() {
        let checker = SyncQcChecker::new();
        let measurements = vec![
            SyncMeasurement::new(5.0, 0.0),
            SyncMeasurement::new(3.0, 0.0), // backwards
        ];
        let report = checker.analyze(&measurements);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-030"));
    }

    #[test]
    fn test_timestamp_gap() {
        let checker = SyncQcChecker::new();
        let measurements = vec![
            SyncMeasurement::new(0.0, 0.0),
            SyncMeasurement::new(5.0, 0.0), // 5s gap, threshold is 2s
        ];
        let report = checker.analyze(&measurements);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-031"));
    }

    #[test]
    fn test_custom_config() {
        let config = SyncQcConfig::new()
            .with_warning_threshold(10.0)
            .with_error_threshold(30.0)
            .with_critical_threshold(50.0)
            .with_max_drift_rate(2.0);
        let checker = SyncQcChecker::with_config(config);

        // 25ms would be error with these thresholds
        let measurements = vec![SyncMeasurement::new(1.0, 25.0)];
        let report = checker.analyze(&measurements);
        assert!(report.findings.iter().any(|f| f.code == "SYNC-012"));
    }

    #[test]
    fn test_report_stats() {
        let checker = SyncQcChecker::new();
        let measurements = vec![
            SyncMeasurement::new(0.0, 10.0),
            SyncMeasurement::new(1.0, -5.0),
            SyncMeasurement::new(2.0, 15.0),
        ];
        let report = checker.analyze(&measurements);
        assert!((report.max_offset_ms - 15.0).abs() < f64::EPSILON);
        assert!((report.mean_offset_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_finding_at_timestamp() {
        let finding = SyncFinding::new(SyncSeverity::Error, "T01", "test").at_timestamp(42.5);
        assert_eq!(finding.timestamp_secs, Some(42.5));
        assert!(finding.is_failure());
    }

    #[test]
    fn test_sync_direction_display() {
        assert_eq!(SyncDirection::AudioLeads.to_string(), "audio leads video");
        assert_eq!(SyncDirection::VideoLeads.to_string(), "video leads audio");
        assert_eq!(SyncDirection::InSync.to_string(), "in sync");
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(SyncSeverity::Info.to_string(), "INFO");
        assert_eq!(SyncSeverity::Warning.to_string(), "WARNING");
        assert_eq!(SyncSeverity::Error.to_string(), "ERROR");
        assert_eq!(SyncSeverity::Critical.to_string(), "CRITICAL");
    }
}
