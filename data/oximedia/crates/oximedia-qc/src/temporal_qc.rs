//! Temporal quality control: timestamp continuity, frame-rate stability, and A/V sync.
//!
//! Provides `TemporalCheck`, `TemporalQcAnalyzer`, and `TemporalQcReport` for
//! detecting timing anomalies in media streams.

#![allow(dead_code)]

/// The type of temporal anomaly detected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemporalCheck {
    /// Presentation timestamp discontinuity detected.
    PtsDrop,
    /// Decode timestamp out of order.
    DtsOutOfOrder,
    /// Frame rate deviated beyond tolerance.
    FrameRateDeviation,
    /// Audio / video sync offset exceeds threshold.
    AvSyncError,
    /// Frozen or duplicate frames detected.
    FrozenFrame,
    /// Audio gap or silence longer than threshold.
    AudioGap,
    /// Custom / user-defined check type.
    Custom(String),
}

impl TemporalCheck {
    /// Return a stable string key.
    #[must_use]
    pub fn key(&self) -> &str {
        match self {
            Self::PtsDrop => "pts_drop",
            Self::DtsOutOfOrder => "dts_out_of_order",
            Self::FrameRateDeviation => "frame_rate_deviation",
            Self::AvSyncError => "av_sync_error",
            Self::FrozenFrame => "frozen_frame",
            Self::AudioGap => "audio_gap",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// A temporal anomaly finding at a specific position in the file.
#[derive(Debug, Clone)]
pub struct TemporalFinding {
    /// Kind of anomaly.
    pub check: TemporalCheck,
    /// Presentation timestamp of the occurrence (seconds).
    pub pts_seconds: f64,
    /// Measured value (e.g. deviation in ms, or offset in ms).
    pub measured_value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// Human-readable description.
    pub description: String,
}

impl TemporalFinding {
    /// Create a new finding.
    #[must_use]
    pub fn new(
        check: TemporalCheck,
        pts_seconds: f64,
        measured_value: f64,
        threshold: f64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            check,
            pts_seconds,
            measured_value,
            threshold,
            description: description.into(),
        }
    }

    /// Return `true` if the measured value exceeded the threshold.
    #[must_use]
    pub fn is_violation(&self) -> bool {
        self.measured_value > self.threshold
    }
}

/// Configuration for the temporal QC analyzer.
#[derive(Debug, Clone)]
pub struct TemporalQcConfig {
    /// Maximum allowed A/V sync offset in milliseconds.
    pub max_av_sync_ms: f64,
    /// Maximum allowed frame-rate deviation in percent.
    pub max_frame_rate_deviation_pct: f64,
    /// Maximum allowed PTS gap in milliseconds before flagging a drop.
    pub max_pts_gap_ms: f64,
    /// Minimum duration of audio gap to flag (milliseconds).
    pub min_audio_gap_ms: f64,
}

impl Default for TemporalQcConfig {
    fn default() -> Self {
        Self {
            max_av_sync_ms: 40.0,
            max_frame_rate_deviation_pct: 1.0,
            max_pts_gap_ms: 100.0,
            min_audio_gap_ms: 200.0,
        }
    }
}

/// Analyses a sequence of timing samples and produces a [`TemporalQcReport`].
#[derive(Debug)]
pub struct TemporalQcAnalyzer {
    config: TemporalQcConfig,
}

impl TemporalQcAnalyzer {
    /// Create an analyzer with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TemporalQcConfig::default(),
        }
    }

    /// Create an analyzer with custom config.
    #[must_use]
    pub fn with_config(config: TemporalQcConfig) -> Self {
        Self { config }
    }

    /// Analyse a series of video PTS values (in seconds, ascending expected).
    ///
    /// Returns findings for any gaps or reversals detected.
    #[must_use]
    pub fn analyze_pts(&self, pts_values: &[f64]) -> Vec<TemporalFinding> {
        let mut findings = Vec::new();
        for w in pts_values.windows(2) {
            let delta_ms = (w[1] - w[0]) * 1000.0;
            if delta_ms < 0.0 {
                findings.push(TemporalFinding::new(
                    TemporalCheck::DtsOutOfOrder,
                    w[1],
                    delta_ms.abs(),
                    0.0,
                    format!("PTS decreased by {:.2} ms at {:.3} s", delta_ms.abs(), w[1]),
                ));
            } else if delta_ms > self.config.max_pts_gap_ms {
                findings.push(TemporalFinding::new(
                    TemporalCheck::PtsDrop,
                    w[1],
                    delta_ms,
                    self.config.max_pts_gap_ms,
                    format!("PTS gap of {:.2} ms at {:.3} s", delta_ms, w[1]),
                ));
            }
        }
        findings
    }

    /// Analyse A/V sync offsets (milliseconds; positive = video ahead of audio).
    #[must_use]
    pub fn analyze_av_sync(&self, offsets_ms: &[f64]) -> Vec<TemporalFinding> {
        offsets_ms
            .iter()
            .enumerate()
            .filter(|(_, &off)| off.abs() > self.config.max_av_sync_ms)
            .map(|(i, &off)| {
                TemporalFinding::new(
                    TemporalCheck::AvSyncError,
                    i as f64,
                    off.abs(),
                    self.config.max_av_sync_ms,
                    format!("A/V sync offset {off:.2} ms at index {i}"),
                )
            })
            .collect()
    }

    /// Analyse measured frame rate samples against a nominal rate.
    ///
    /// `frame_rates` – measured FPS values; `nominal` – target FPS.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_frame_rate(&self, frame_rates: &[f64], nominal: f64) -> Vec<TemporalFinding> {
        frame_rates
            .iter()
            .enumerate()
            .filter_map(|(i, &fps)| {
                if nominal == 0.0 {
                    return None;
                }
                let deviation_pct = ((fps - nominal) / nominal * 100.0).abs();
                if deviation_pct > self.config.max_frame_rate_deviation_pct {
                    Some(TemporalFinding::new(
                        TemporalCheck::FrameRateDeviation,
                        i as f64,
                        deviation_pct,
                        self.config.max_frame_rate_deviation_pct,
                        format!(
                            "frame rate {fps:.3} fps deviates {deviation_pct:.2}% from {nominal}"
                        ),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for TemporalQcAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated temporal QC report.
#[derive(Debug, Default)]
pub struct TemporalQcReport {
    /// All temporal findings discovered.
    pub findings: Vec<TemporalFinding>,
    /// Total number of frames / samples analysed.
    pub samples_analyzed: usize,
}

impl TemporalQcReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add findings from the analyzer.
    pub fn extend(&mut self, findings: Vec<TemporalFinding>) {
        self.findings.extend(findings);
    }

    /// Return `true` if no findings were recorded.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// Return findings of the given check type.
    #[must_use]
    pub fn by_check(&self, check: &TemporalCheck) -> Vec<&TemporalFinding> {
        self.findings.iter().filter(|f| &f.check == check).collect()
    }

    /// Return the finding at the earliest PTS.
    #[must_use]
    pub fn earliest_finding(&self) -> Option<&TemporalFinding> {
        self.findings.iter().min_by(|a, b| {
            a.pts_seconds
                .partial_cmp(&b.pts_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the count of distinct check types that have findings.
    #[must_use]
    pub fn distinct_issue_types(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for f in &self.findings {
            seen.insert(f.check.key());
        }
        seen.len()
    }

    /// Return the total number of findings.
    #[must_use]
    pub fn finding_count(&self) -> usize {
        self.findings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_check_keys() {
        assert_eq!(TemporalCheck::PtsDrop.key(), "pts_drop");
        assert_eq!(TemporalCheck::DtsOutOfOrder.key(), "dts_out_of_order");
        assert_eq!(
            TemporalCheck::FrameRateDeviation.key(),
            "frame_rate_deviation"
        );
        assert_eq!(TemporalCheck::AvSyncError.key(), "av_sync_error");
        assert_eq!(TemporalCheck::FrozenFrame.key(), "frozen_frame");
        assert_eq!(TemporalCheck::AudioGap.key(), "audio_gap");
        assert_eq!(TemporalCheck::Custom("test".into()).key(), "test");
    }

    #[test]
    fn test_finding_is_violation_true() {
        let f = TemporalFinding::new(TemporalCheck::PtsDrop, 1.0, 200.0, 100.0, "big gap");
        assert!(f.is_violation());
    }

    #[test]
    fn test_finding_is_violation_false() {
        let f = TemporalFinding::new(TemporalCheck::PtsDrop, 1.0, 50.0, 100.0, "ok");
        assert!(!f.is_violation());
    }

    #[test]
    fn test_analyze_pts_clean() {
        let analyzer = TemporalQcAnalyzer::new();
        let pts = vec![0.0, 0.033, 0.066, 0.100];
        let findings = analyzer.analyze_pts(&pts);
        assert!(findings.is_empty());
    }

    #[test]
    fn test_analyze_pts_drop() {
        let analyzer = TemporalQcAnalyzer::new();
        // Large gap of 500 ms between index 1 and 2
        let pts = vec![0.0, 0.033, 0.533, 0.566];
        let findings = analyzer.analyze_pts(&pts);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].check, TemporalCheck::PtsDrop);
    }

    #[test]
    fn test_analyze_pts_out_of_order() {
        let analyzer = TemporalQcAnalyzer::new();
        let pts = vec![0.0, 0.033, 0.020, 0.066];
        let findings = analyzer.analyze_pts(&pts);
        assert!(findings
            .iter()
            .any(|f| f.check == TemporalCheck::DtsOutOfOrder));
    }

    #[test]
    fn test_analyze_av_sync_clean() {
        let analyzer = TemporalQcAnalyzer::new();
        let offsets = vec![10.0, -5.0, 20.0];
        let findings = analyzer.analyze_av_sync(&offsets);
        assert!(findings.is_empty());
    }

    #[test]
    fn test_analyze_av_sync_violation() {
        let analyzer = TemporalQcAnalyzer::new();
        let offsets = vec![10.0, 80.0, -90.0];
        let findings = analyzer.analyze_av_sync(&offsets);
        assert_eq!(findings.len(), 2);
        assert!(findings
            .iter()
            .all(|f| f.check == TemporalCheck::AvSyncError));
    }

    #[test]
    fn test_analyze_frame_rate_clean() {
        let analyzer = TemporalQcAnalyzer::new();
        let fps = vec![29.97, 29.97, 29.97];
        let findings = analyzer.analyze_frame_rate(&fps, 29.97);
        assert!(findings.is_empty());
    }

    #[test]
    fn test_analyze_frame_rate_deviation() {
        let analyzer = TemporalQcAnalyzer::new();
        let fps = vec![29.97, 25.0]; // 25 deviates heavily from 29.97
        let findings = analyzer.analyze_frame_rate(&fps, 29.97);
        assert!(!findings.is_empty());
        assert_eq!(findings[0].check, TemporalCheck::FrameRateDeviation);
    }

    #[test]
    fn test_report_is_clean_empty() {
        let report = TemporalQcReport::new();
        assert!(report.is_clean());
    }

    #[test]
    fn test_report_extend_and_count() {
        let analyzer = TemporalQcAnalyzer::new();
        let mut report = TemporalQcReport::new();
        let findings = analyzer.analyze_av_sync(&[100.0, 200.0]);
        report.extend(findings);
        assert_eq!(report.finding_count(), 2);
    }

    #[test]
    fn test_report_by_check() {
        let mut report = TemporalQcReport::new();
        report.findings.push(TemporalFinding::new(
            TemporalCheck::PtsDrop,
            1.0,
            200.0,
            100.0,
            "gap",
        ));
        report.findings.push(TemporalFinding::new(
            TemporalCheck::AvSyncError,
            2.0,
            80.0,
            40.0,
            "sync",
        ));
        assert_eq!(report.by_check(&TemporalCheck::PtsDrop).len(), 1);
        assert_eq!(report.by_check(&TemporalCheck::FrozenFrame).len(), 0);
    }

    #[test]
    fn test_distinct_issue_types() {
        let mut report = TemporalQcReport::new();
        report.findings.push(TemporalFinding::new(
            TemporalCheck::PtsDrop,
            1.0,
            200.0,
            100.0,
            "a",
        ));
        report.findings.push(TemporalFinding::new(
            TemporalCheck::PtsDrop,
            2.0,
            150.0,
            100.0,
            "b",
        ));
        report.findings.push(TemporalFinding::new(
            TemporalCheck::AvSyncError,
            3.0,
            80.0,
            40.0,
            "c",
        ));
        assert_eq!(report.distinct_issue_types(), 2);
    }

    #[test]
    fn test_earliest_finding() {
        let mut report = TemporalQcReport::new();
        report.findings.push(TemporalFinding::new(
            TemporalCheck::PtsDrop,
            5.0,
            200.0,
            100.0,
            "late",
        ));
        report.findings.push(TemporalFinding::new(
            TemporalCheck::AvSyncError,
            1.0,
            80.0,
            40.0,
            "early",
        ));
        let earliest = report.earliest_finding().expect("should succeed in test");
        assert!((earliest.pts_seconds - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = TemporalQcConfig::default();
        assert!((cfg.max_av_sync_ms - 40.0).abs() < f64::EPSILON);
        assert!((cfg.max_frame_rate_deviation_pct - 1.0).abs() < f64::EPSILON);
    }
}
