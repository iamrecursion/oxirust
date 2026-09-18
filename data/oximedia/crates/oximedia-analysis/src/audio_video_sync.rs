//! Audio/video synchronisation analysis.
//!
//! Measures A/V sync drift over time and classifies sync health,
//! providing sample-by-sample tracking and reporting.

#![allow(dead_code)]

/// A single A/V sync measurement at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct SyncMeasure {
    /// Presentation timestamp in milliseconds.
    pub pts_ms: f64,
    /// Measured audio-video offset in milliseconds.
    /// Positive = audio leads video; negative = audio lags video.
    pub offset_ms: f64,
}

impl SyncMeasure {
    /// Create a new `SyncMeasure`.
    #[must_use]
    pub fn new(pts_ms: f64, offset_ms: f64) -> Self {
        Self { pts_ms, offset_ms }
    }

    /// Return the absolute drift magnitude in milliseconds.
    #[must_use]
    pub fn drift_ms(&self) -> f64 {
        self.offset_ms.abs()
    }

    /// Return `true` if audio leads video (positive offset).
    #[must_use]
    pub fn audio_leads(&self) -> bool {
        self.offset_ms > 0.0
    }
}

/// Classification of A/V sync health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// Sync drift is within acceptable tolerance (≤ 40 ms).
    Ok,
    /// Sync drift is noticeable but not severe (40–80 ms).
    Warning,
    /// Sync drift exceeds acceptable limits (> 80 ms).
    Critical,
}

impl SyncStatus {
    /// Returns `true` if the status is `Ok`.
    #[must_use]
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Classify a drift value (in ms) into a `SyncStatus`.
    #[must_use]
    pub fn from_drift(drift_ms: f64) -> Self {
        if drift_ms <= 40.0 {
            Self::Ok
        } else if drift_ms <= 80.0 {
            Self::Warning
        } else {
            Self::Critical
        }
    }

    /// Return a short label for this status.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Warning => "Warning",
            Self::Critical => "Critical",
        }
    }
}

/// Accumulates A/V sync measurements and tracks drift.
#[derive(Debug, Clone, Default)]
pub struct AvSyncAnalyzer {
    samples: Vec<SyncMeasure>,
}

impl AvSyncAnalyzer {
    /// Create a new, empty `AvSyncAnalyzer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sync measurement sample.
    pub fn add_sample(&mut self, sample: SyncMeasure) {
        self.samples.push(sample);
    }

    /// Return the most recent drift in milliseconds.
    ///
    /// Returns `0.0` if no samples have been recorded.
    #[must_use]
    pub fn current_drift_ms(&self) -> f64 {
        self.samples.last().map_or(0.0, SyncMeasure::drift_ms)
    }

    /// Return the current sync status based on the latest sample.
    #[must_use]
    pub fn current_status(&self) -> SyncStatus {
        SyncStatus::from_drift(self.current_drift_ms())
    }

    /// Return the average drift across all samples, or `0.0` if empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_drift_ms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(SyncMeasure::drift_ms).sum();
        sum / self.samples.len() as f64
    }

    /// Build a `SyncReport` from the current state.
    #[must_use]
    pub fn build_report(&self) -> SyncReport {
        let max_drift = self
            .samples
            .iter()
            .map(SyncMeasure::drift_ms)
            .fold(0.0_f64, f64::max);
        SyncReport {
            sample_count: self.samples.len(),
            avg_drift_ms: self.avg_drift_ms(),
            max_drift_ms: max_drift,
            status: SyncStatus::from_drift(max_drift),
        }
    }

    /// Return all recorded samples as a slice.
    #[must_use]
    pub fn samples(&self) -> &[SyncMeasure] {
        &self.samples
    }
}

/// Summary report of A/V sync analysis results.
#[derive(Debug, Clone, Copy)]
pub struct SyncReport {
    /// Number of samples collected.
    pub sample_count: usize,
    /// Average drift in milliseconds.
    pub avg_drift_ms: f64,
    /// Maximum drift observed in milliseconds.
    pub max_drift_ms: f64,
    /// Overall sync health classification.
    pub status: SyncStatus,
}

impl SyncReport {
    /// Return the maximum drift in milliseconds.
    #[must_use]
    pub fn max_drift_ms(&self) -> f64 {
        self.max_drift_ms
    }

    /// Return `true` if the overall sync status is `Ok`.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.status.is_ok()
    }
}

// ---------------------------------------------------------------------------
// Drift Trend Analysis
// ---------------------------------------------------------------------------

/// Drift trend direction over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftTrend {
    /// Drift is stable / not changing significantly.
    Stable,
    /// Drift is increasing over time (getting worse).
    Increasing,
    /// Drift is decreasing over time (improving).
    Decreasing,
    /// Drift is oscillating without clear trend.
    Oscillating,
}

impl DriftTrend {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Stable => "Stable",
            Self::Increasing => "Increasing",
            Self::Decreasing => "Decreasing",
            Self::Oscillating => "Oscillating",
        }
    }
}

/// A time window in which drift exceeded the warning or critical threshold.
#[derive(Debug, Clone, Copy)]
pub struct DriftEvent {
    /// Start time in milliseconds.
    pub start_pts_ms: f64,
    /// End time in milliseconds.
    pub end_pts_ms: f64,
    /// Peak drift magnitude during this event (ms).
    pub peak_drift_ms: f64,
    /// Severity of this event.
    pub status: SyncStatus,
}

/// Comprehensive drift analysis result.
#[derive(Debug, Clone)]
pub struct DriftAnalysis {
    /// Total number of samples analysed.
    pub sample_count: usize,
    /// Average drift (signed, ms) — positive = audio leads.
    pub mean_offset_ms: f64,
    /// Average absolute drift (ms).
    pub mean_drift_ms: f64,
    /// Maximum absolute drift (ms).
    pub max_drift_ms: f64,
    /// Standard deviation of drift (ms).
    pub drift_stddev_ms: f64,
    /// Detected trend.
    pub trend: DriftTrend,
    /// Linear regression slope (ms drift per second of playback).
    /// Positive = drift growing; negative = drift shrinking.
    pub slope_ms_per_sec: f64,
    /// Drift events (time windows where drift exceeded warning threshold).
    pub events: Vec<DriftEvent>,
    /// Overall sync status.
    pub overall_status: SyncStatus,
}

impl AvSyncAnalyzer {
    /// Perform comprehensive drift analysis over time.
    ///
    /// This analyses the recorded samples to detect:
    /// - Linear drift trend (slope via least-squares regression)
    /// - Drift events (windows where sync exceeds threshold)
    /// - Statistical summary (mean, stddev, max)
    #[must_use]
    pub fn analyze_drift(&self) -> DriftAnalysis {
        let n = self.samples.len();
        if n == 0 {
            return DriftAnalysis {
                sample_count: 0,
                mean_offset_ms: 0.0,
                mean_drift_ms: 0.0,
                max_drift_ms: 0.0,
                drift_stddev_ms: 0.0,
                trend: DriftTrend::Stable,
                slope_ms_per_sec: 0.0,
                events: Vec::new(),
                overall_status: SyncStatus::Ok,
            };
        }

        let nf = n as f64;

        // Basic statistics
        let mean_offset = self.samples.iter().map(|s| s.offset_ms).sum::<f64>() / nf;
        let mean_drift = self.samples.iter().map(|s| s.drift_ms()).sum::<f64>() / nf;
        let max_drift = self
            .samples
            .iter()
            .map(|s| s.drift_ms())
            .fold(0.0_f64, f64::max);

        let variance = self
            .samples
            .iter()
            .map(|s| {
                let d = s.offset_ms - mean_offset;
                d * d
            })
            .sum::<f64>()
            / nf;
        let stddev = variance.sqrt();

        // Linear regression: offset_ms = slope * pts_sec + intercept
        // Convert pts to seconds for slope units
        let slope = if n >= 2 {
            compute_linear_slope(&self.samples)
        } else {
            0.0
        };

        // Determine trend
        let trend = classify_trend(slope, stddev, &self.samples);

        // Find drift events (contiguous windows where drift exceeds warning)
        let events = find_drift_events(&self.samples);

        let overall_status = SyncStatus::from_drift(max_drift);

        DriftAnalysis {
            sample_count: n,
            mean_offset_ms: mean_offset,
            mean_drift_ms: mean_drift,
            max_drift_ms: max_drift,
            drift_stddev_ms: stddev,
            trend,
            slope_ms_per_sec: slope,
            events,
            overall_status,
        }
    }
}

/// Compute the linear regression slope of offset_ms vs pts_seconds.
fn compute_linear_slope(samples: &[SyncMeasure]) -> f64 {
    let n = samples.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // x = pts in seconds, y = offset_ms
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_xx = 0.0f64;

    for s in samples {
        let x = s.pts_ms / 1000.0; // convert to seconds
        let y = s.offset_ms;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

/// Classify the drift trend from the slope and variability.
fn classify_trend(slope: f64, stddev: f64, samples: &[SyncMeasure]) -> DriftTrend {
    if samples.len() < 3 {
        return DriftTrend::Stable;
    }

    // If slope is significant relative to noise
    let slope_abs = slope.abs();
    if stddev > 0.0 && slope_abs < stddev * 0.1 {
        // Slope is small relative to noise
        // Check for oscillation: count sign changes in the offset
        let sign_changes = count_sign_changes(samples);
        let change_ratio = sign_changes as f64 / (samples.len() - 1) as f64;
        if change_ratio > 0.3 {
            return DriftTrend::Oscillating;
        }
        return DriftTrend::Stable;
    }

    // Significant slope
    if slope > 0.5 {
        DriftTrend::Increasing
    } else if slope < -0.5 {
        DriftTrend::Decreasing
    } else {
        let sign_changes = count_sign_changes(samples);
        let change_ratio = sign_changes as f64 / (samples.len() - 1) as f64;
        if change_ratio > 0.3 {
            DriftTrend::Oscillating
        } else {
            DriftTrend::Stable
        }
    }
}

/// Count the number of times the offset sign changes.
fn count_sign_changes(samples: &[SyncMeasure]) -> usize {
    if samples.len() < 2 {
        return 0;
    }
    let mut changes = 0usize;
    for pair in samples.windows(2) {
        let a = pair[0].offset_ms;
        let b = pair[1].offset_ms;
        if (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0) {
            changes += 1;
        }
    }
    changes
}

/// Find contiguous drift events where drift exceeds the warning threshold.
fn find_drift_events(samples: &[SyncMeasure]) -> Vec<DriftEvent> {
    const WARNING_THRESHOLD: f64 = 40.0;
    let mut events = Vec::new();
    let mut event_start: Option<usize> = None;

    for (i, s) in samples.iter().enumerate() {
        let drift = s.drift_ms();
        if drift > WARNING_THRESHOLD {
            if event_start.is_none() {
                event_start = Some(i);
            }
        } else if let Some(start) = event_start.take() {
            events.push(build_drift_event(samples, start, i - 1));
        }
    }

    // Close any open event
    if let Some(start) = event_start {
        events.push(build_drift_event(samples, start, samples.len() - 1));
    }

    events
}

/// Build a DriftEvent from a range of sample indices.
fn build_drift_event(samples: &[SyncMeasure], start: usize, end: usize) -> DriftEvent {
    let peak_drift = samples[start..=end]
        .iter()
        .map(|s| s.drift_ms())
        .fold(0.0_f64, f64::max);

    let status = SyncStatus::from_drift(peak_drift);

    DriftEvent {
        start_pts_ms: samples[start].pts_ms,
        end_pts_ms: samples[end].pts_ms,
        peak_drift_ms: peak_drift,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_measure_drift_ms_positive() {
        let m = SyncMeasure::new(0.0, 30.0);
        assert!((m.drift_ms() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_measure_drift_ms_negative() {
        let m = SyncMeasure::new(0.0, -50.0);
        assert!((m.drift_ms() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_measure_audio_leads() {
        let m = SyncMeasure::new(0.0, 10.0);
        assert!(m.audio_leads());
    }

    #[test]
    fn test_sync_measure_audio_lags() {
        let m = SyncMeasure::new(0.0, -10.0);
        assert!(!m.audio_leads());
    }

    #[test]
    fn test_sync_status_from_drift_ok() {
        assert_eq!(SyncStatus::from_drift(20.0), SyncStatus::Ok);
        assert_eq!(SyncStatus::from_drift(40.0), SyncStatus::Ok);
    }

    #[test]
    fn test_sync_status_from_drift_warning() {
        assert_eq!(SyncStatus::from_drift(41.0), SyncStatus::Warning);
        assert_eq!(SyncStatus::from_drift(80.0), SyncStatus::Warning);
    }

    #[test]
    fn test_sync_status_from_drift_critical() {
        assert_eq!(SyncStatus::from_drift(81.0), SyncStatus::Critical);
    }

    #[test]
    fn test_sync_status_is_ok() {
        assert!(SyncStatus::Ok.is_ok());
        assert!(!SyncStatus::Warning.is_ok());
        assert!(!SyncStatus::Critical.is_ok());
    }

    #[test]
    fn test_sync_status_label() {
        assert_eq!(SyncStatus::Ok.label(), "OK");
        assert_eq!(SyncStatus::Warning.label(), "Warning");
        assert_eq!(SyncStatus::Critical.label(), "Critical");
    }

    #[test]
    fn test_av_sync_analyzer_empty() {
        let analyzer = AvSyncAnalyzer::new();
        assert!((analyzer.current_drift_ms()).abs() < f64::EPSILON);
        assert_eq!(analyzer.current_status(), SyncStatus::Ok);
    }

    #[test]
    fn test_av_sync_analyzer_add_sample() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(100.0, 90.0));
        assert!((analyzer.current_drift_ms() - 90.0).abs() < f64::EPSILON);
        assert_eq!(analyzer.current_status(), SyncStatus::Critical);
    }

    #[test]
    fn test_av_sync_analyzer_avg_drift() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 20.0));
        analyzer.add_sample(SyncMeasure::new(1000.0, 60.0));
        assert!((analyzer.avg_drift_ms() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_sync_report_max_drift() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 10.0));
        analyzer.add_sample(SyncMeasure::new(1000.0, 85.0));
        analyzer.add_sample(SyncMeasure::new(2000.0, 30.0));
        let report = analyzer.build_report();
        assert!((report.max_drift_ms() - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sync_report_is_healthy() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 5.0));
        let report = analyzer.build_report();
        assert!(report.is_healthy());
    }

    #[test]
    fn test_sync_report_not_healthy_when_critical() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 200.0));
        let report = analyzer.build_report();
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_av_sync_samples_slice() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 5.0));
        analyzer.add_sample(SyncMeasure::new(100.0, 10.0));
        assert_eq!(analyzer.samples().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Drift trend analysis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drift_analysis_empty() {
        let analyzer = AvSyncAnalyzer::new();
        let analysis = analyzer.analyze_drift();
        assert_eq!(analysis.sample_count, 0);
        assert_eq!(analysis.trend, DriftTrend::Stable);
        assert_eq!(analysis.overall_status, SyncStatus::Ok);
        assert!(analysis.events.is_empty());
    }

    #[test]
    fn test_drift_analysis_stable() {
        let mut analyzer = AvSyncAnalyzer::new();
        // Stable small drift over 10 seconds
        for i in 0..100 {
            let pts_ms = i as f64 * 100.0;
            analyzer.add_sample(SyncMeasure::new(pts_ms, 5.0));
        }
        let analysis = analyzer.analyze_drift();
        assert_eq!(analysis.sample_count, 100);
        assert!(analysis.mean_drift_ms < 10.0);
        assert_eq!(analysis.trend, DriftTrend::Stable);
        assert_eq!(analysis.overall_status, SyncStatus::Ok);
        assert!(analysis.events.is_empty());
    }

    #[test]
    fn test_drift_analysis_increasing() {
        let mut analyzer = AvSyncAnalyzer::new();
        // Linearly increasing drift: 0ms at t=0, 100ms at t=10s
        for i in 0..100 {
            let pts_ms = i as f64 * 100.0;
            let offset = i as f64 * 1.0; // grows from 0 to 99
            analyzer.add_sample(SyncMeasure::new(pts_ms, offset));
        }
        let analysis = analyzer.analyze_drift();
        assert!(analysis.slope_ms_per_sec > 0.5);
        assert_eq!(analysis.trend, DriftTrend::Increasing);
    }

    #[test]
    fn test_drift_analysis_decreasing() {
        let mut analyzer = AvSyncAnalyzer::new();
        // Linearly decreasing drift
        for i in 0..100 {
            let pts_ms = i as f64 * 100.0;
            let offset = 100.0 - i as f64 * 1.0;
            analyzer.add_sample(SyncMeasure::new(pts_ms, offset));
        }
        let analysis = analyzer.analyze_drift();
        assert!(analysis.slope_ms_per_sec < -0.5);
        assert_eq!(analysis.trend, DriftTrend::Decreasing);
    }

    #[test]
    fn test_drift_analysis_events() {
        let mut analyzer = AvSyncAnalyzer::new();
        // OK for first 5 samples, then spike to critical, then back
        for i in 0..5 {
            analyzer.add_sample(SyncMeasure::new(i as f64 * 1000.0, 10.0));
        }
        for i in 5..10 {
            analyzer.add_sample(SyncMeasure::new(i as f64 * 1000.0, 90.0));
        }
        for i in 10..15 {
            analyzer.add_sample(SyncMeasure::new(i as f64 * 1000.0, 15.0));
        }
        let analysis = analyzer.analyze_drift();
        assert!(!analysis.events.is_empty());
        assert_eq!(analysis.events.len(), 1);
        assert_eq!(analysis.events[0].status, SyncStatus::Critical);
        assert!((analysis.events[0].peak_drift_ms - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_drift_analysis_max_drift() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 10.0));
        analyzer.add_sample(SyncMeasure::new(1000.0, -120.0));
        analyzer.add_sample(SyncMeasure::new(2000.0, 5.0));
        let analysis = analyzer.analyze_drift();
        assert!((analysis.max_drift_ms - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_drift_analysis_stddev() {
        let mut analyzer = AvSyncAnalyzer::new();
        // All same offset => stddev = 0
        for i in 0..20 {
            analyzer.add_sample(SyncMeasure::new(i as f64 * 100.0, 25.0));
        }
        let analysis = analyzer.analyze_drift();
        assert!(analysis.drift_stddev_ms < 0.01);
    }

    #[test]
    fn test_drift_trend_labels() {
        assert_eq!(DriftTrend::Stable.label(), "Stable");
        assert_eq!(DriftTrend::Increasing.label(), "Increasing");
        assert_eq!(DriftTrend::Decreasing.label(), "Decreasing");
        assert_eq!(DriftTrend::Oscillating.label(), "Oscillating");
    }

    #[test]
    fn test_drift_analysis_oscillating() {
        let mut analyzer = AvSyncAnalyzer::new();
        // Oscillating: alternating positive and negative offset
        for i in 0..50 {
            let pts_ms = i as f64 * 100.0;
            let offset = if i % 2 == 0 { 20.0 } else { -20.0 };
            analyzer.add_sample(SyncMeasure::new(pts_ms, offset));
        }
        let analysis = analyzer.analyze_drift();
        assert_eq!(analysis.trend, DriftTrend::Oscillating);
    }

    #[test]
    fn test_drift_event_timing() {
        let mut analyzer = AvSyncAnalyzer::new();
        analyzer.add_sample(SyncMeasure::new(0.0, 5.0));
        analyzer.add_sample(SyncMeasure::new(1000.0, 50.0));
        analyzer.add_sample(SyncMeasure::new(2000.0, 70.0));
        analyzer.add_sample(SyncMeasure::new(3000.0, 10.0));
        let analysis = analyzer.analyze_drift();
        assert_eq!(analysis.events.len(), 1);
        assert!((analysis.events[0].start_pts_ms - 1000.0).abs() < 0.01);
        assert!((analysis.events[0].end_pts_ms - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_slope_flat() {
        let samples: Vec<SyncMeasure> = (0..10)
            .map(|i| SyncMeasure::new(i as f64 * 1000.0, 25.0))
            .collect();
        let slope = compute_linear_slope(&samples);
        assert!(slope.abs() < 0.01);
    }

    #[test]
    fn test_linear_slope_rising() {
        // offset = 10 * pts_sec (10 ms per second of playback)
        let samples: Vec<SyncMeasure> = (0..10)
            .map(|i| {
                let pts_sec = i as f64;
                SyncMeasure::new(pts_sec * 1000.0, 10.0 * pts_sec)
            })
            .collect();
        let slope = compute_linear_slope(&samples);
        assert!((slope - 10.0).abs() < 0.5);
    }
}
