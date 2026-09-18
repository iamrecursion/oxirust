#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
//! Farm-wide metrics collection and summarization.
//!
//! Provides lightweight telemetry for render farm operations: frame
//! throughput, node utilization, queue depth, and error counts.

/// A named metric tracked by the farm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FarmMetric {
    /// Frames rendered per second across the entire farm.
    FramesPerSecond,
    /// Fraction of worker slots currently occupied (0–100 percent).
    UtilizationPct,
    /// Number of jobs waiting in the priority queue.
    QueueDepth,
    /// Number of render errors in the current window.
    ErrorCount,
    /// Average round-trip latency from submission to first frame (ms).
    SubmitLatencyMs,
    /// Bytes of intermediate data written to scratch storage per second.
    ScratchIoBytesPerSec,
    /// GPU memory pressure across all CUDA/ROCm nodes (percent).
    GpuMemoryPressurePct,
    /// Number of active worker nodes.
    ActiveNodeCount,
}

impl FarmMetric {
    /// Returns the unit string for display purposes.
    pub fn unit(self) -> &'static str {
        match self {
            Self::FramesPerSecond => "fps",
            Self::UtilizationPct => "%",
            Self::QueueDepth => "jobs",
            Self::ErrorCount => "errors",
            Self::SubmitLatencyMs => "ms",
            Self::ScratchIoBytesPerSec => "B/s",
            Self::GpuMemoryPressurePct => "%",
            Self::ActiveNodeCount => "nodes",
        }
    }

    /// Returns a short human-readable name for the metric.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::FramesPerSecond => "Frames/s",
            Self::UtilizationPct => "Utilization",
            Self::QueueDepth => "Queue Depth",
            Self::ErrorCount => "Errors",
            Self::SubmitLatencyMs => "Submit Latency",
            Self::ScratchIoBytesPerSec => "Scratch I/O",
            Self::GpuMemoryPressurePct => "GPU Mem Pressure",
            Self::ActiveNodeCount => "Active Nodes",
        }
    }

    /// Returns `true` if higher values of this metric are undesirable.
    pub fn higher_is_worse(self) -> bool {
        matches!(
            self,
            Self::QueueDepth
                | Self::ErrorCount
                | Self::SubmitLatencyMs
                | Self::GpuMemoryPressurePct
        )
    }
}

// ── FarmMetricSample ──────────────────────────────────────────────────────

/// A single timestamped measurement of a `FarmMetric`.
#[derive(Debug, Clone)]
pub struct FarmMetricSample {
    /// The metric being recorded.
    pub metric: FarmMetric,
    /// Recorded value.
    pub value: f64,
    /// Wall-clock timestamp (Unix seconds).
    pub timestamp_secs: u64,
}

impl FarmMetricSample {
    /// Creates a new sample.
    pub fn new(metric: FarmMetric, value: f64, timestamp_secs: u64) -> Self {
        Self {
            metric,
            value,
            timestamp_secs,
        }
    }

    /// Returns how many seconds old this sample is relative to `now`.
    pub fn age_secs(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_secs)
    }

    /// Returns `true` if the sample is older than `max_age_secs`.
    pub fn is_stale(&self, now: u64, max_age_secs: u64) -> bool {
        self.age_secs(now) > max_age_secs
    }
}

// ── FarmMetrics ───────────────────────────────────────────────────────────

/// Rolling time-series store for farm metrics.
///
/// Retains at most `capacity` samples per metric to bound memory usage.
#[derive(Debug)]
pub struct FarmMetrics {
    samples: Vec<FarmMetricSample>,
    capacity: usize,
}

impl Default for FarmMetrics {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl FarmMetrics {
    /// Creates a new store with the given per-metric rolling window capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::new(),
            capacity,
        }
    }

    /// Records a new sample.
    ///
    /// If the total sample count for this metric exceeds `capacity`,
    /// the oldest sample for that metric is discarded.
    pub fn record(&mut self, sample: FarmMetricSample) {
        // Evict oldest sample for this metric if over capacity
        let count = self
            .samples
            .iter()
            .filter(|s| s.metric == sample.metric)
            .count();
        if count >= self.capacity {
            // Find and remove the oldest sample for this metric
            if let Some(idx) = self.samples.iter().position(|s| s.metric == sample.metric) {
                self.samples.remove(idx);
            }
        }
        self.samples.push(sample);
    }

    /// Returns the arithmetic mean of all samples for `metric`.
    ///
    /// Returns `None` if no samples exist for the metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg(&self, metric: FarmMetric) -> Option<f64> {
        let vals: Vec<f64> = self
            .samples
            .iter()
            .filter(|s| s.metric == metric)
            .map(|s| s.value)
            .collect();
        if vals.is_empty() {
            return None;
        }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Returns the peak (maximum) value seen for `metric`.
    ///
    /// Returns `None` if no samples exist.
    pub fn peak(&self, metric: FarmMetric) -> Option<f64> {
        self.samples
            .iter()
            .filter(|s| s.metric == metric)
            .map(|s| s.value)
            .reduce(f64::max)
    }

    /// Returns the most recent sample for `metric`.
    pub fn latest(&self, metric: FarmMetric) -> Option<&FarmMetricSample> {
        self.samples
            .iter()
            .filter(|s| s.metric == metric)
            .max_by_key(|s| s.timestamp_secs)
    }

    /// Returns the total number of stored samples across all metrics.
    pub fn total_sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Removes all samples older than `max_age_secs` relative to `now`.
    pub fn evict_stale(&mut self, now: u64, max_age_secs: u64) {
        self.samples.retain(|s| !s.is_stale(now, max_age_secs));
    }
}

// ── FarmMetricDashboard ───────────────────────────────────────────────────

/// A snapshot summary of key farm health metrics.
#[derive(Debug, Clone)]
pub struct FarmMetricDashboard {
    /// Average worker utilization across the observation window (percent).
    pub utilization_pct: Option<f64>,
    /// Current number of jobs waiting in the priority queue.
    pub queue_depth: Option<f64>,
    /// Average frames-per-second throughput across the observation window.
    pub frames_per_second: Option<f64>,
    /// Total accumulated error count in the observation window.
    pub error_count: Option<f64>,
    /// Latest reported count of active worker nodes.
    pub active_nodes: Option<f64>,
}

impl FarmMetricDashboard {
    /// Builds a dashboard summary from a `FarmMetrics` store.
    pub fn summary(metrics: &FarmMetrics) -> Self {
        Self {
            utilization_pct: metrics.avg(FarmMetric::UtilizationPct),
            queue_depth: metrics.latest(FarmMetric::QueueDepth).map(|s| s.value),
            frames_per_second: metrics.avg(FarmMetric::FramesPerSecond),
            error_count: metrics
                .samples
                .iter()
                .filter(|s| s.metric == FarmMetric::ErrorCount)
                .map(|s| s.value)
                .sum::<f64>()
                .into(),
            active_nodes: metrics.latest(FarmMetric::ActiveNodeCount).map(|s| s.value),
        }
    }

    /// Returns `true` if any health indicator is in a warning state.
    pub fn has_warnings(&self) -> bool {
        self.utilization_pct.is_some_and(|u| u > 90.0)
            || self.queue_depth.is_some_and(|q| q > 500.0)
            || self.error_count.is_some_and(|e| e > 0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn sample(metric: FarmMetric, value: f64, ts: u64) -> FarmMetricSample {
        FarmMetricSample::new(metric, value, ts)
    }

    #[test]
    fn test_metric_units_not_empty() {
        let metrics = [
            FarmMetric::FramesPerSecond,
            FarmMetric::UtilizationPct,
            FarmMetric::QueueDepth,
            FarmMetric::ErrorCount,
        ];
        for m in metrics {
            assert!(!m.unit().is_empty());
        }
    }

    #[test]
    fn test_metric_display_names() {
        assert_eq!(FarmMetric::FramesPerSecond.display_name(), "Frames/s");
        assert_eq!(FarmMetric::ActiveNodeCount.display_name(), "Active Nodes");
    }

    #[test]
    fn test_higher_is_worse_flags() {
        assert!(FarmMetric::ErrorCount.higher_is_worse());
        assert!(FarmMetric::QueueDepth.higher_is_worse());
        assert!(!FarmMetric::FramesPerSecond.higher_is_worse());
        assert!(!FarmMetric::UtilizationPct.higher_is_worse());
    }

    #[test]
    fn test_sample_age_secs() {
        let s = sample(FarmMetric::FramesPerSecond, 24.0, 1000);
        assert_eq!(s.age_secs(1010), 10);
        assert_eq!(s.age_secs(999), 0); // saturating
    }

    #[test]
    fn test_sample_is_stale() {
        let s = sample(FarmMetric::QueueDepth, 5.0, 1000);
        assert!(s.is_stale(1100, 50));
        assert!(!s.is_stale(1010, 50));
    }

    #[test]
    fn test_metrics_avg_empty() {
        let m = FarmMetrics::new(10);
        assert!(m.avg(FarmMetric::FramesPerSecond).is_none());
    }

    #[test]
    fn test_metrics_avg_single() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::UtilizationPct, 75.0, 0));
        assert_eq!(m.avg(FarmMetric::UtilizationPct), Some(75.0));
    }

    #[test]
    fn test_metrics_avg_multiple() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::FramesPerSecond, 20.0, 1));
        m.record(sample(FarmMetric::FramesPerSecond, 30.0, 2));
        assert_eq!(m.avg(FarmMetric::FramesPerSecond), Some(25.0));
    }

    #[test]
    fn test_metrics_peak() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::FramesPerSecond, 10.0, 1));
        m.record(sample(FarmMetric::FramesPerSecond, 50.0, 2));
        m.record(sample(FarmMetric::FramesPerSecond, 30.0, 3));
        assert_eq!(m.peak(FarmMetric::FramesPerSecond), Some(50.0));
    }

    #[test]
    fn test_metrics_latest() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::QueueDepth, 10.0, 5));
        m.record(sample(FarmMetric::QueueDepth, 20.0, 10));
        assert_eq!(
            m.latest(FarmMetric::QueueDepth)
                .expect("should succeed in test")
                .value,
            20.0
        );
    }

    #[test]
    fn test_metrics_evict_stale() {
        let mut m = FarmMetrics::new(100);
        m.record(sample(FarmMetric::ErrorCount, 1.0, 0));
        m.record(sample(FarmMetric::ErrorCount, 1.0, 100));
        m.evict_stale(200, 150);
        // Only the sample at ts=100 should survive (age=100 <= 150)
        assert_eq!(m.total_sample_count(), 1);
    }

    #[test]
    fn test_metrics_capacity_eviction() {
        let mut m = FarmMetrics::new(3);
        for i in 0..5u64 {
            m.record(sample(FarmMetric::ActiveNodeCount, i as f64, i));
        }
        // Only 3 samples should remain for that metric
        let count = m
            .samples
            .iter()
            .filter(|s| s.metric == FarmMetric::ActiveNodeCount)
            .count();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_dashboard_summary_no_warnings() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::UtilizationPct, 60.0, 0));
        m.record(sample(FarmMetric::QueueDepth, 5.0, 0));
        let dash = FarmMetricDashboard::summary(&m);
        assert!(!dash.has_warnings());
    }

    #[test]
    fn test_dashboard_summary_has_warnings_on_high_utilization() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::UtilizationPct, 95.0, 0));
        let dash = FarmMetricDashboard::summary(&m);
        assert!(dash.has_warnings());
    }

    #[test]
    fn test_dashboard_error_count_accumulated() {
        let mut m = FarmMetrics::new(10);
        m.record(sample(FarmMetric::ErrorCount, 2.0, 1));
        m.record(sample(FarmMetric::ErrorCount, 3.0, 2));
        let dash = FarmMetricDashboard::summary(&m);
        assert_eq!(dash.error_count, Some(5.0));
        assert!(dash.has_warnings());
    }
}
