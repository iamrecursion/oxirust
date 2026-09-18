#![allow(dead_code)]
//! Normalization processing report generation.
//!
//! Generates detailed reports of normalization operations including
//! input/output loudness measurements, gain applied, limiter activity,
//! compliance status, and per-channel statistics.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, SystemTime};

/// Severity level for report entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational message.
    Info,
    /// Warning: may need attention.
    Warning,
    /// Error: compliance failure or processing problem.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// Overall result status of the normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationStatus {
    /// Normalization completed successfully, all checks passed.
    Success,
    /// Normalization completed with warnings.
    SuccessWithWarnings,
    /// Normalization completed but compliance check failed.
    ComplianceFailed,
    /// Normalization aborted due to error.
    Aborted,
    /// Not yet processed.
    Pending,
}

impl fmt::Display for NormalizationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::SuccessWithWarnings => write!(f, "SUCCESS (with warnings)"),
            Self::ComplianceFailed => write!(f, "COMPLIANCE FAILED"),
            Self::Aborted => write!(f, "ABORTED"),
            Self::Pending => write!(f, "PENDING"),
        }
    }
}

/// Loudness measurement snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessMeasurement {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Loudness range in LU.
    pub lra_lu: f64,
    /// Maximum true peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Maximum momentary loudness in LUFS.
    pub momentary_max_lufs: f64,
    /// Maximum short-term loudness in LUFS.
    pub short_term_max_lufs: f64,
}

impl LoudnessMeasurement {
    /// Create a new loudness measurement.
    pub fn new(
        integrated_lufs: f64,
        lra_lu: f64,
        true_peak_dbtp: f64,
        momentary_max_lufs: f64,
        short_term_max_lufs: f64,
    ) -> Self {
        Self {
            integrated_lufs,
            lra_lu,
            true_peak_dbtp,
            momentary_max_lufs,
            short_term_max_lufs,
        }
    }

    /// Create a zeroed measurement.
    pub fn zero() -> Self {
        Self {
            integrated_lufs: -70.0,
            lra_lu: 0.0,
            true_peak_dbtp: -70.0,
            momentary_max_lufs: -70.0,
            short_term_max_lufs: -70.0,
        }
    }
}

impl fmt::Display for LoudnessMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "I={:.1} LUFS, LRA={:.1} LU, TP={:.1} dBTP, M={:.1} LUFS, S={:.1} LUFS",
            self.integrated_lufs,
            self.lra_lu,
            self.true_peak_dbtp,
            self.momentary_max_lufs,
            self.short_term_max_lufs
        )
    }
}

/// Per-channel statistics in the report.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelStats {
    /// Channel index.
    pub channel: usize,
    /// Channel label (e.g., "Left", "Right", "Center").
    pub label: String,
    /// Peak sample value (linear).
    pub peak_sample: f64,
    /// RMS level in dB.
    pub rms_db: f64,
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// DC offset detected.
    pub dc_offset: f64,
}

impl ChannelStats {
    /// Create new channel stats.
    pub fn new(channel: usize, label: &str) -> Self {
        Self {
            channel,
            label: label.to_string(),
            peak_sample: 0.0,
            rms_db: -70.0,
            true_peak_dbtp: -70.0,
            dc_offset: 0.0,
        }
    }
}

/// A single entry (message) in the report log.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportEntry {
    /// Severity level.
    pub severity: Severity,
    /// Timestamp offset from processing start.
    pub offset: Duration,
    /// Message text.
    pub message: String,
    /// Optional associated timecode (e.g., where an event occurred).
    pub timecode: Option<String>,
}

impl ReportEntry {
    /// Create a new report entry.
    pub fn new(severity: Severity, offset: Duration, message: &str) -> Self {
        Self {
            severity,
            offset,
            message: message.to_string(),
            timecode: None,
        }
    }

    /// Set the timecode.
    pub fn with_timecode(mut self, tc: &str) -> Self {
        self.timecode = Some(tc.to_string());
        self
    }
}

impl fmt::Display for ReportEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tc = self.timecode.as_deref().unwrap_or("--:--:--:--");
        write!(
            f,
            "[{:>5}] +{:.3}s @ {} {}",
            self.severity,
            self.offset.as_secs_f64(),
            tc,
            self.message
        )
    }
}

/// Gain adjustment details.
#[derive(Debug, Clone, PartialEq)]
pub struct GainInfo {
    /// Gain applied in dB.
    pub gain_db: f64,
    /// Whether gain was clamped to max.
    pub was_clamped: bool,
    /// Maximum gain allowed.
    pub max_gain_db: f64,
    /// Limiter gain reduction peak in dB.
    pub limiter_peak_gr_db: f64,
}

impl GainInfo {
    /// Create new gain info.
    pub fn new(gain_db: f64, max_gain_db: f64) -> Self {
        let was_clamped = gain_db.abs() > max_gain_db;
        Self {
            gain_db: gain_db.clamp(-max_gain_db, max_gain_db),
            was_clamped,
            max_gain_db,
            limiter_peak_gr_db: 0.0,
        }
    }

    /// Set limiter peak gain reduction.
    pub fn with_limiter_gr(mut self, gr_db: f64) -> Self {
        self.limiter_peak_gr_db = gr_db;
        self
    }
}

/// Complete normalization processing report.
#[derive(Debug, Clone)]
pub struct NormalizeReport {
    /// Report title / filename.
    pub title: String,
    /// Normalization status.
    pub status: NormalizationStatus,
    /// Target standard name.
    pub target_standard: String,
    /// Target loudness in LUFS.
    pub target_lufs: f64,
    /// Input measurement (before normalization).
    pub input_measurement: LoudnessMeasurement,
    /// Output measurement (after normalization).
    pub output_measurement: LoudnessMeasurement,
    /// Gain adjustment info.
    pub gain_info: GainInfo,
    /// Per-channel stats.
    pub channel_stats: Vec<ChannelStats>,
    /// Log entries.
    pub entries: Vec<ReportEntry>,
    /// Processing duration.
    pub processing_duration: Duration,
    /// Audio duration.
    pub audio_duration: Duration,
    /// Sample rate.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Timestamp when report was generated.
    pub generated_at: SystemTime,
    /// Custom key-value metadata.
    pub metadata: BTreeMap<String, String>,
}

impl NormalizeReport {
    /// Create a new empty report.
    pub fn new(title: &str, target_standard: &str, target_lufs: f64) -> Self {
        Self {
            title: title.to_string(),
            status: NormalizationStatus::Pending,
            target_standard: target_standard.to_string(),
            target_lufs,
            input_measurement: LoudnessMeasurement::zero(),
            output_measurement: LoudnessMeasurement::zero(),
            gain_info: GainInfo::new(0.0, 20.0),
            channel_stats: Vec::new(),
            entries: Vec::new(),
            processing_duration: Duration::ZERO,
            audio_duration: Duration::ZERO,
            sample_rate: 48000.0,
            channels: 2,
            generated_at: SystemTime::now(),
            metadata: BTreeMap::new(),
        }
    }

    /// Set input measurement.
    pub fn with_input(mut self, measurement: LoudnessMeasurement) -> Self {
        self.input_measurement = measurement;
        self
    }

    /// Set output measurement.
    pub fn with_output(mut self, measurement: LoudnessMeasurement) -> Self {
        self.output_measurement = measurement;
        self
    }

    /// Set gain info.
    pub fn with_gain(mut self, gain: GainInfo) -> Self {
        self.gain_info = gain;
        self
    }

    /// Set status.
    pub fn with_status(mut self, status: NormalizationStatus) -> Self {
        self.status = status;
        self
    }

    /// Add a channel stats entry.
    pub fn add_channel_stats(&mut self, stats: ChannelStats) {
        self.channel_stats.push(stats);
    }

    /// Add a log entry.
    pub fn add_entry(&mut self, entry: ReportEntry) {
        self.entries.push(entry);
    }

    /// Add an info entry.
    pub fn info(&mut self, offset: Duration, message: &str) {
        self.entries
            .push(ReportEntry::new(Severity::Info, offset, message));
    }

    /// Add a warning entry.
    pub fn warn(&mut self, offset: Duration, message: &str) {
        self.entries
            .push(ReportEntry::new(Severity::Warning, offset, message));
    }

    /// Add an error entry.
    pub fn error(&mut self, offset: Duration, message: &str) {
        self.entries
            .push(ReportEntry::new(Severity::Error, offset, message));
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Number of warnings in the log.
    pub fn warning_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == Severity::Warning)
            .count()
    }

    /// Number of errors in the log.
    pub fn error_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .count()
    }

    /// Loudness deviation from target (output - target) in LU.
    pub fn loudness_deviation(&self) -> f64 {
        self.output_measurement.integrated_lufs - self.target_lufs
    }

    /// Whether the output is within a given tolerance of the target.
    pub fn is_within_tolerance(&self, tolerance_lu: f64) -> bool {
        self.loudness_deviation().abs() <= tolerance_lu
    }

    /// Processing speed ratio (audio duration / processing duration).
    pub fn speed_ratio(&self) -> f64 {
        let proc_secs = self.processing_duration.as_secs_f64();
        if proc_secs < 1e-9 {
            return 0.0;
        }
        self.audio_duration.as_secs_f64() / proc_secs
    }

    /// Generate a plain text summary of the report.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Normalization Report: {} ===", self.title));
        lines.push(format!("Status: {}", self.status));
        lines.push(format!(
            "Target: {} ({:.1} LUFS)",
            self.target_standard, self.target_lufs
        ));
        lines.push(format!("Input:  {}", self.input_measurement));
        lines.push(format!("Output: {}", self.output_measurement));
        lines.push(format!(
            "Gain: {:+.2} dB (clamped: {})",
            self.gain_info.gain_db, self.gain_info.was_clamped
        ));
        lines.push(format!("Deviation: {:+.2} LU", self.loudness_deviation()));
        if !self.channel_stats.is_empty() {
            lines.push("Channel Stats:".to_string());
            for ch in &self.channel_stats {
                lines.push(format!(
                    "  Ch{} ({}): peak={:.4}, rms={:.1} dB, tp={:.1} dBTP",
                    ch.channel, ch.label, ch.peak_sample, ch.rms_db, ch.true_peak_dbtp
                ));
            }
        }
        if !self.entries.is_empty() {
            lines.push(format!(
                "Log: {} entries ({} warnings, {} errors)",
                self.entries.len(),
                self.warning_count(),
                self.error_count()
            ));
        }
        lines.push(format!("Speed: {:.1}x realtime", self.speed_ratio()));
        lines.join("\n")
    }

    /// Finalize the report, determining status from entries.
    pub fn finalize(&mut self) {
        if self.error_count() > 0 {
            self.status = NormalizationStatus::ComplianceFailed;
        } else if self.warning_count() > 0 {
            self.status = NormalizationStatus::SuccessWithWarnings;
        } else {
            self.status = NormalizationStatus::Success;
        }
    }
}

impl fmt::Display for NormalizeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
        assert_eq!(format!("{}", Severity::Warning), "WARN");
        assert_eq!(format!("{}", Severity::Error), "ERROR");
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", NormalizationStatus::Success), "SUCCESS");
        assert_eq!(
            format!("{}", NormalizationStatus::ComplianceFailed),
            "COMPLIANCE FAILED"
        );
    }

    #[test]
    fn test_loudness_measurement_display() {
        let m = LoudnessMeasurement::new(-23.0, 12.0, -1.5, -18.0, -20.0);
        let s = format!("{m}");
        assert!(s.contains("-23.0 LUFS"));
        assert!(s.contains("12.0 LU"));
    }

    #[test]
    fn test_loudness_measurement_zero() {
        let m = LoudnessMeasurement::zero();
        assert!((m.integrated_lufs - (-70.0)).abs() < f64::EPSILON);
        assert!((m.lra_lu - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_creation() {
        let report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        assert_eq!(report.title, "test.wav");
        assert_eq!(report.status, NormalizationStatus::Pending);
        assert!((report.target_lufs - (-23.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_add_entries() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.info(Duration::from_millis(100), "Analysis started");
        report.warn(Duration::from_millis(500), "High peak detected");
        report.error(Duration::from_secs(1), "Compliance failure");
        assert_eq!(report.entries.len(), 3);
        assert_eq!(report.warning_count(), 1);
        assert_eq!(report.error_count(), 1);
    }

    #[test]
    fn test_report_finalize_success() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.info(Duration::ZERO, "OK");
        report.finalize();
        assert_eq!(report.status, NormalizationStatus::Success);
    }

    #[test]
    fn test_report_finalize_with_warnings() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.warn(Duration::ZERO, "Something");
        report.finalize();
        assert_eq!(report.status, NormalizationStatus::SuccessWithWarnings);
    }

    #[test]
    fn test_report_finalize_with_errors() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.error(Duration::ZERO, "Bad");
        report.finalize();
        assert_eq!(report.status, NormalizationStatus::ComplianceFailed);
    }

    #[test]
    fn test_loudness_deviation() {
        let report = NormalizeReport::new("test.wav", "EBU R128", -23.0)
            .with_output(LoudnessMeasurement::new(-22.5, 10.0, -2.0, -18.0, -20.0));
        assert!((report.loudness_deviation() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_within_tolerance() {
        let report = NormalizeReport::new("test.wav", "EBU R128", -23.0)
            .with_output(LoudnessMeasurement::new(-22.5, 10.0, -2.0, -18.0, -20.0));
        assert!(report.is_within_tolerance(1.0));
        assert!(!report.is_within_tolerance(0.3));
    }

    #[test]
    fn test_speed_ratio() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.audio_duration = Duration::from_secs(60);
        report.processing_duration = Duration::from_secs(10);
        assert!((report.speed_ratio() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_ratio_zero_processing() {
        let report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        assert!((report.speed_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gain_info_clamped() {
        let gain = GainInfo::new(25.0, 20.0);
        assert!(gain.was_clamped);
        assert!((gain.gain_db - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gain_info_not_clamped() {
        let gain = GainInfo::new(5.0, 20.0);
        assert!(!gain.was_clamped);
        assert!((gain.gain_db - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_channel_stats() {
        let mut ch = ChannelStats::new(0, "Left");
        ch.peak_sample = 0.95;
        ch.rms_db = -18.0;
        assert_eq!(ch.channel, 0);
        assert_eq!(ch.label, "Left");
    }

    #[test]
    fn test_report_entry_display() {
        let entry = ReportEntry::new(
            Severity::Warning,
            Duration::from_millis(1500),
            "Peak exceeded",
        )
        .with_timecode("01:00:15:12");
        let s = format!("{entry}");
        assert!(s.contains("WARN"));
        assert!(s.contains("01:00:15:12"));
        assert!(s.contains("Peak exceeded"));
    }

    #[test]
    fn test_report_metadata() {
        let mut report = NormalizeReport::new("test.wav", "EBU R128", -23.0);
        report.set_metadata("encoder", "oximedia");
        assert_eq!(
            report
                .metadata
                .get("encoder")
                .expect("should succeed in test"),
            "oximedia"
        );
    }

    #[test]
    fn test_summary_output() {
        let report = NormalizeReport::new("test.wav", "EBU R128", -23.0)
            .with_status(NormalizationStatus::Success)
            .with_input(LoudnessMeasurement::new(-28.0, 15.0, -3.0, -22.0, -24.0))
            .with_output(LoudnessMeasurement::new(-23.0, 12.0, -1.5, -18.0, -20.0));
        let summary = report.summary();
        assert!(summary.contains("test.wav"));
        assert!(summary.contains("SUCCESS"));
        assert!(summary.contains("EBU R128"));
    }
}
