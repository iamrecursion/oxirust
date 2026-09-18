//! Input-to-display latency measurement and analysis for game streaming.
//!
//! Tracks the end-to-end pipeline latency from input capture through encode,
//! transmit, decode, and render.  Provides rolling statistics and threshold
//! alerting so that streamers can keep their glass-to-glass latency within
//! acceptable bounds.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Pipeline stage
// ---------------------------------------------------------------------------

/// Individual stages of the streaming pipeline whose latency can be measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Input device polling (keyboard / mouse / controller).
    InputCapture,
    /// Frame capture (screen grab or game hook).
    FrameCapture,
    /// Video encoding.
    Encode,
    /// Network transmission.
    Transmit,
    /// Video decoding on the viewer side.
    Decode,
    /// Display rendering.
    Render,
}

impl PipelineStage {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::InputCapture => "Input Capture",
            Self::FrameCapture => "Frame Capture",
            Self::Encode => "Encode",
            Self::Transmit => "Transmit",
            Self::Decode => "Decode",
            Self::Render => "Render",
        }
    }
}

// ---------------------------------------------------------------------------
// Latency record
// ---------------------------------------------------------------------------

/// A single end-to-end latency measurement broken down by stage.
#[derive(Debug, Clone)]
pub struct LatencyRecord {
    /// Per-stage durations.
    pub stages: Vec<(PipelineStage, Duration)>,
}

impl LatencyRecord {
    /// Create a new empty record.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage measurement.
    pub fn add_stage(&mut self, stage: PipelineStage, duration: Duration) {
        self.stages.push((stage, duration));
    }

    /// Total end-to-end latency.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.stages.iter().map(|(_, d)| *d).sum()
    }

    /// Latency contributed by a specific stage, or zero if not present.
    #[must_use]
    pub fn stage_latency(&self, stage: PipelineStage) -> Duration {
        self.stages
            .iter()
            .filter(|(s, _)| *s == stage)
            .map(|(_, d)| *d)
            .sum()
    }

    /// The stage that contributes the most latency.
    #[must_use]
    pub fn bottleneck(&self) -> Option<PipelineStage> {
        self.stages.iter().max_by_key(|(_, d)| *d).map(|(s, _)| *s)
    }
}

impl Default for LatencyRecord {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Threshold severity
// ---------------------------------------------------------------------------

/// Severity level when a latency threshold is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThresholdSeverity {
    /// Everything is fine.
    Ok,
    /// Approaching the limit.
    Warning,
    /// Exceeds acceptable latency.
    Critical,
}

// ---------------------------------------------------------------------------
// LatencyTracker
// ---------------------------------------------------------------------------

/// Accumulates [`LatencyRecord`]s and computes rolling statistics.
pub struct LatencyTracker {
    records: VecDeque<LatencyRecord>,
    capacity: usize,
    /// Warning threshold for total latency.
    pub warning_threshold: Duration,
    /// Critical threshold for total latency.
    pub critical_threshold: Duration,
}

impl LatencyTracker {
    /// Create a tracker that retains at most `capacity` records.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            warning_threshold: Duration::from_millis(100),
            critical_threshold: Duration::from_millis(200),
        }
    }

    /// Push a new latency record.
    pub fn push(&mut self, record: LatencyRecord) {
        if self.records.len() == self.capacity {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Number of records stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Average total latency across stored records.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_total_ms(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .records
            .iter()
            .map(|r| r.total().as_secs_f64() * 1000.0)
            .sum();
        sum / self.records.len() as f64
    }

    /// Maximum total latency seen in the stored window.
    #[must_use]
    pub fn max_total(&self) -> Duration {
        self.records
            .iter()
            .map(LatencyRecord::total)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Minimum total latency seen in the stored window.
    #[must_use]
    pub fn min_total(&self) -> Duration {
        self.records
            .iter()
            .map(LatencyRecord::total)
            .min()
            .unwrap_or(Duration::ZERO)
    }

    /// Evaluate the current average against thresholds.
    #[must_use]
    pub fn severity(&self) -> ThresholdSeverity {
        let avg = Duration::from_secs_f64(self.avg_total_ms() / 1000.0);
        if avg >= self.critical_threshold {
            ThresholdSeverity::Critical
        } else if avg >= self.warning_threshold {
            ThresholdSeverity::Warning
        } else {
            ThresholdSeverity::Ok
        }
    }

    /// Average latency of a specific pipeline stage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_stage_ms(&self, stage: PipelineStage) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .records
            .iter()
            .map(|r| r.stage_latency(stage).as_secs_f64() * 1000.0)
            .sum();
        sum / self.records.len() as f64
    }

    /// Jitter: standard deviation of total latency in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn jitter_ms(&self) -> f64 {
        if self.records.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_total_ms();
        let n = self.records.len() as f64;
        let var: f64 = self
            .records
            .iter()
            .map(|r| {
                let ms = r.total().as_secs_f64() * 1000.0;
                (ms - mean) * (ms - mean)
            })
            .sum::<f64>()
            / n;
        var.sqrt()
    }

    /// Clear all stored records.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(ms_vals: &[(PipelineStage, u64)]) -> LatencyRecord {
        let mut r = LatencyRecord::new();
        for &(stage, ms) in ms_vals {
            r.add_stage(stage, Duration::from_millis(ms));
        }
        r
    }

    #[test]
    fn test_pipeline_stage_labels() {
        assert_eq!(PipelineStage::InputCapture.label(), "Input Capture");
        assert_eq!(PipelineStage::Encode.label(), "Encode");
    }

    #[test]
    fn test_record_total() {
        let r = make_record(&[
            (PipelineStage::InputCapture, 1),
            (PipelineStage::Encode, 5),
            (PipelineStage::Transmit, 20),
        ]);
        assert_eq!(r.total(), Duration::from_millis(26));
    }

    #[test]
    fn test_record_stage_latency() {
        let r = make_record(&[(PipelineStage::Encode, 5), (PipelineStage::Transmit, 20)]);
        assert_eq!(
            r.stage_latency(PipelineStage::Encode),
            Duration::from_millis(5)
        );
        assert_eq!(r.stage_latency(PipelineStage::Decode), Duration::ZERO);
    }

    #[test]
    fn test_record_bottleneck() {
        let r = make_record(&[(PipelineStage::Encode, 5), (PipelineStage::Transmit, 50)]);
        assert_eq!(r.bottleneck(), Some(PipelineStage::Transmit));
    }

    #[test]
    fn test_record_empty_bottleneck() {
        let r = LatencyRecord::new();
        assert!(r.bottleneck().is_none());
    }

    #[test]
    fn test_tracker_push_and_len() {
        let mut t = LatencyTracker::new(10);
        assert!(t.is_empty());
        t.push(make_record(&[(PipelineStage::Encode, 5)]));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_tracker_capacity_eviction() {
        let mut t = LatencyTracker::new(3);
        for _ in 0..5 {
            t.push(make_record(&[(PipelineStage::Encode, 5)]));
        }
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_avg_total_ms() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        t.push(make_record(&[(PipelineStage::Encode, 20)]));
        assert!((t.avg_total_ms() - 15.0).abs() < 1e-3);
    }

    #[test]
    fn test_avg_total_ms_empty() {
        let t = LatencyTracker::new(10);
        assert!((t.avg_total_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_min_total() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        t.push(make_record(&[(PipelineStage::Encode, 30)]));
        assert_eq!(t.max_total(), Duration::from_millis(30));
        assert_eq!(t.min_total(), Duration::from_millis(10));
    }

    #[test]
    fn test_severity_ok() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        assert_eq!(t.severity(), ThresholdSeverity::Ok);
    }

    #[test]
    fn test_severity_critical() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Transmit, 250)]));
        assert_eq!(t.severity(), ThresholdSeverity::Critical);
    }

    #[test]
    fn test_avg_stage_ms() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[
            (PipelineStage::Encode, 10),
            (PipelineStage::Transmit, 20),
        ]));
        assert!((t.avg_stage_ms(PipelineStage::Encode) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_jitter_constant() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        assert!((t.jitter_ms() - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_jitter_variable() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        t.push(make_record(&[(PipelineStage::Encode, 30)]));
        assert!(t.jitter_ms() > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut t = LatencyTracker::new(10);
        t.push(make_record(&[(PipelineStage::Encode, 10)]));
        t.clear();
        assert!(t.is_empty());
    }
}
