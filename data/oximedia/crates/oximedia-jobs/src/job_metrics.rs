#![allow(dead_code)]
//! Per-job metrics collection — outcomes, efficiency, and aggregate statistics.

use std::collections::HashMap;
use std::time::Duration;

/// The outcome of a completed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobOutcome {
    /// Job completed successfully.
    Success,
    /// Job failed with an error.
    Failure,
    /// Job was cancelled before completion.
    Cancelled,
    /// Job timed out.
    TimedOut,
    /// Job was skipped (e.g. dependency failed).
    Skipped,
}

impl JobOutcome {
    /// Returns true for outcomes that represent successful completion.
    pub fn is_success(&self) -> bool {
        matches!(self, JobOutcome::Success)
    }

    /// Returns true for outcomes that represent a terminal failure.
    pub fn is_terminal_failure(&self) -> bool {
        matches!(self, JobOutcome::Failure | JobOutcome::TimedOut)
    }

    /// Short string label.
    pub fn label(&self) -> &'static str {
        match self {
            JobOutcome::Success => "success",
            JobOutcome::Failure => "failure",
            JobOutcome::Cancelled => "cancelled",
            JobOutcome::TimedOut => "timed_out",
            JobOutcome::Skipped => "skipped",
        }
    }
}

/// Metrics captured for a single job execution.
#[derive(Debug, Clone)]
pub struct JobMetric {
    /// Identifier for the job.
    pub job_id: String,
    /// Outcome of the job.
    pub outcome: JobOutcome,
    /// Wall-clock duration from start to finish.
    pub duration: Duration,
    /// CPU time consumed (approximate).
    pub cpu_time: Duration,
    /// Number of items processed (e.g. frames transcoded).
    pub items_processed: u64,
    /// Number of items in the total workload.
    pub items_total: u64,
    /// Worker that executed the job.
    pub worker_id: String,
    /// Number of retry attempts (0 = first try succeeded or failed).
    pub retry_count: u32,
}

impl JobMetric {
    /// Create a new metric record.
    pub fn new(
        job_id: impl Into<String>,
        outcome: JobOutcome,
        duration: Duration,
        cpu_time: Duration,
        items_processed: u64,
        items_total: u64,
        worker_id: impl Into<String>,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            outcome,
            duration,
            cpu_time,
            items_processed,
            items_total,
            worker_id: worker_id.into(),
            retry_count: 0,
        }
    }

    /// Set retry count.
    pub fn with_retries(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    /// Efficiency as a percentage: (items_processed / items_total) * 100.
    /// Returns 0.0 if items_total is 0.
    #[allow(clippy::cast_precision_loss)]
    pub fn efficiency_pct(&self) -> f64 {
        if self.items_total == 0 {
            return 0.0;
        }
        (self.items_processed as f64 / self.items_total as f64) * 100.0
    }

    /// CPU utilisation as a fraction of wall-clock time [0.0, ∞).
    /// Values > 1.0 indicate multi-core utilisation.
    pub fn cpu_utilization(&self) -> f64 {
        let wall = self.duration.as_secs_f64();
        if wall <= 0.0 {
            return 0.0;
        }
        self.cpu_time.as_secs_f64() / wall
    }

    /// Processing rate in items per second. 0.0 if duration is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn items_per_second(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.items_processed as f64 / secs
    }
}

/// Collector for multiple job metrics with aggregate query methods.
#[derive(Debug, Default, Clone)]
pub struct JobMetricsCollector {
    records: Vec<JobMetric>,
    /// Total number of successful outcomes counted.
    success_total: u64,
    /// Total number of failure outcomes counted.
    failure_total: u64,
}

impl JobMetricsCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a job metric.
    pub fn record(&mut self, metric: JobMetric) {
        if metric.outcome.is_success() {
            self.success_total += 1;
        } else if metric.outcome.is_terminal_failure() {
            self.failure_total += 1;
        }
        self.records.push(metric);
    }

    /// Total number of recorded jobs.
    pub fn total_recorded(&self) -> usize {
        self.records.len()
    }

    /// Number of successful jobs.
    pub fn success_count(&self) -> usize {
        self.records
            .iter()
            .filter(|m| m.outcome.is_success())
            .count()
    }

    /// Success rate as a fraction [0.0, 1.0]. Returns 0.0 if no records.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        self.success_count() as f64 / self.records.len() as f64
    }

    /// Average duration in milliseconds across all records. 0.0 if empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_duration_ms(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total_ms: f64 = self
            .records
            .iter()
            .map(|m| m.duration.as_secs_f64() * 1000.0)
            .sum();
        total_ms / self.records.len() as f64
    }

    /// Maximum duration across all records.
    pub fn max_duration(&self) -> Option<Duration> {
        self.records.iter().map(|m| m.duration).max()
    }

    /// Minimum duration across all records.
    pub fn min_duration(&self) -> Option<Duration> {
        self.records.iter().map(|m| m.duration).min()
    }

    /// Average efficiency percentage across all records.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_efficiency_pct(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total: f64 = self.records.iter().map(|m| m.efficiency_pct()).sum();
        total / self.records.len() as f64
    }

    /// Count records by outcome.
    pub fn outcome_counts(&self) -> HashMap<JobOutcome, usize> {
        let mut map = HashMap::new();
        for m in &self.records {
            *map.entry(m.outcome).or_insert(0) += 1;
        }
        map
    }

    /// Metrics for jobs that ran on a specific worker.
    pub fn records_for_worker(&self, worker_id: &str) -> Vec<&JobMetric> {
        self.records
            .iter()
            .filter(|m| m.worker_id == worker_id)
            .collect()
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
        self.success_total = 0;
        self.failure_total = 0;
    }
}

// ===========================================================================
// EWMA-based ETA estimation
// ===========================================================================

/// EWMA smoothing factor for duration statistics.
const EWMA_ALPHA: f64 = 0.1;

/// Rolling EWMA statistics for a single job kind.
///
/// Both mean and variance are tracked in nanoseconds to avoid floating-point
/// precision loss at sub-millisecond granularity.
#[derive(Debug, Clone, Default)]
pub struct DurationStats {
    /// EWMA of the mean duration in nanoseconds.
    pub ewma_mean_ns: f64,
    /// EWMA of the variance (M2 approximation) in nanoseconds squared.
    pub ewma_var_ns: f64,
    /// Number of samples incorporated so far.
    pub count: u64,
}

impl DurationStats {
    /// Create zero-initialised statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Incorporate a new duration sample using EWMA update rules.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, duration: std::time::Duration) {
        let sample_ns = duration.as_nanos() as f64;
        if self.count == 0 {
            self.ewma_mean_ns = sample_ns;
            self.ewma_var_ns = 0.0;
        } else {
            let delta = sample_ns - self.ewma_mean_ns;
            self.ewma_mean_ns += EWMA_ALPHA * delta;
            self.ewma_var_ns = (1.0 - EWMA_ALPHA) * (self.ewma_var_ns + EWMA_ALPHA * delta * delta);
        }
        self.count += 1;
    }
}

/// Per-kind duration statistics used for ETA estimation.
///
/// This is a thin wrapper that maps a job-kind string to its [`DurationStats`]
/// and exposes a high-level [`estimate_remaining`] method.
///
/// [`estimate_remaining`]: JobEtaEstimator::estimate_remaining
#[derive(Debug, Default, Clone)]
pub struct JobEtaEstimator {
    stats: std::collections::HashMap<String, DurationStats>,
}

impl JobEtaEstimator {
    /// Create an empty estimator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed duration for the given `kind`.
    pub fn record_duration(&mut self, kind: &str, duration: std::time::Duration) {
        self.stats
            .entry(kind.to_string())
            .or_default()
            .update(duration);
    }

    /// Estimate how much time remains for a job of the given `kind`.
    ///
    /// Returns `None` when:
    /// - No history exists for this `kind` (`count == 0`), or
    /// - `progress_frac` is effectively zero (would cause division by zero).
    ///
    /// Otherwise:
    /// ```text
    /// remaining ≈ ewma_mean × (1 - progress_frac) / max(progress_frac, 0.001)
    /// ```
    /// The result is clamped to non-negative.
    pub fn estimate_remaining(
        &self,
        kind: &str,
        _elapsed: std::time::Duration,
        progress_frac: f64,
    ) -> Option<std::time::Duration> {
        let stats = self.stats.get(kind)?;
        if stats.count == 0 {
            return None;
        }
        if progress_frac < f64::EPSILON {
            return None;
        }
        let denom = progress_frac.max(0.001);
        let remaining_ns = stats.ewma_mean_ns * (1.0 - progress_frac) / denom;
        if remaining_ns <= 0.0 {
            return Some(std::time::Duration::ZERO);
        }
        Some(std::time::Duration::from_nanos(remaining_ns as u64))
    }

    /// Return stats for a specific kind, if any.
    #[must_use]
    pub fn stats_for(&self, kind: &str) -> Option<&DurationStats> {
        self.stats.get(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn metric(id: &str, outcome: JobOutcome, dur_ms: u64, processed: u64, total: u64) -> JobMetric {
        JobMetric::new(
            id,
            outcome,
            Duration::from_millis(dur_ms),
            Duration::from_millis(dur_ms / 2),
            processed,
            total,
            "worker-1",
        )
    }

    #[test]
    fn test_job_outcome_is_success() {
        assert!(JobOutcome::Success.is_success());
        assert!(!JobOutcome::Failure.is_success());
        assert!(!JobOutcome::Cancelled.is_success());
    }

    #[test]
    fn test_job_outcome_is_terminal_failure() {
        assert!(JobOutcome::Failure.is_terminal_failure());
        assert!(JobOutcome::TimedOut.is_terminal_failure());
        assert!(!JobOutcome::Success.is_terminal_failure());
        assert!(!JobOutcome::Cancelled.is_terminal_failure());
    }

    #[test]
    fn test_job_outcome_label() {
        assert_eq!(JobOutcome::Success.label(), "success");
        assert_eq!(JobOutcome::TimedOut.label(), "timed_out");
    }

    #[test]
    fn test_efficiency_pct_full() {
        let m = metric("j1", JobOutcome::Success, 1000, 100, 100);
        assert!((m.efficiency_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_efficiency_pct_partial() {
        let m = metric("j2", JobOutcome::Failure, 500, 50, 100);
        assert!((m.efficiency_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_efficiency_pct_zero_total() {
        let m = metric("j3", JobOutcome::Success, 100, 0, 0);
        assert_eq!(m.efficiency_pct(), 0.0);
    }

    #[test]
    fn test_items_per_second() {
        // 1000 items in 1 second
        let m = metric("j4", JobOutcome::Success, 1000, 1000, 1000);
        let ips = m.items_per_second();
        assert!((ips - 1000.0).abs() < 1.0, "expected ~1000 ips, got {ips}");
    }

    #[test]
    fn test_collector_empty() {
        let c = JobMetricsCollector::new();
        assert_eq!(c.total_recorded(), 0);
        assert_eq!(c.success_rate(), 0.0);
        assert_eq!(c.avg_duration_ms(), 0.0);
    }

    #[test]
    fn test_collector_record_and_success_rate() {
        let mut c = JobMetricsCollector::new();
        c.record(metric("a", JobOutcome::Success, 100, 10, 10));
        c.record(metric("b", JobOutcome::Success, 200, 10, 10));
        c.record(metric("c", JobOutcome::Failure, 150, 5, 10));
        assert_eq!(c.total_recorded(), 3);
        assert_eq!(c.success_count(), 2);
        let rate = c.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_collector_avg_duration_ms() {
        let mut c = JobMetricsCollector::new();
        c.record(metric("a", JobOutcome::Success, 100, 1, 1));
        c.record(metric("b", JobOutcome::Success, 300, 1, 1));
        let avg = c.avg_duration_ms();
        assert!((avg - 200.0).abs() < 1e-6, "expected 200ms avg, got {avg}");
    }

    #[test]
    fn test_collector_max_min_duration() {
        let mut c = JobMetricsCollector::new();
        c.record(metric("a", JobOutcome::Success, 100, 1, 1));
        c.record(metric("b", JobOutcome::Success, 500, 1, 1));
        assert_eq!(c.max_duration(), Some(Duration::from_millis(500)));
        assert_eq!(c.min_duration(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn test_collector_outcome_counts() {
        let mut c = JobMetricsCollector::new();
        c.record(metric("a", JobOutcome::Success, 100, 1, 1));
        c.record(metric("b", JobOutcome::Failure, 100, 0, 1));
        c.record(metric("c", JobOutcome::Cancelled, 10, 0, 1));
        let counts = c.outcome_counts();
        assert_eq!(counts[&JobOutcome::Success], 1);
        assert_eq!(counts[&JobOutcome::Failure], 1);
        assert_eq!(counts[&JobOutcome::Cancelled], 1);
    }

    #[test]
    fn test_records_for_worker() {
        let mut c = JobMetricsCollector::new();
        let mut m2 = metric("b", JobOutcome::Success, 200, 10, 10);
        m2.worker_id = "worker-2".to_string();
        c.record(metric("a", JobOutcome::Success, 100, 10, 10)); // worker-1
        c.record(m2); // worker-2
        assert_eq!(c.records_for_worker("worker-1").len(), 1);
        assert_eq!(c.records_for_worker("worker-2").len(), 1);
        assert_eq!(c.records_for_worker("worker-3").len(), 0);
    }

    #[test]
    fn test_collector_clear() {
        let mut c = JobMetricsCollector::new();
        c.record(metric("a", JobOutcome::Success, 100, 1, 1));
        c.clear();
        assert_eq!(c.total_recorded(), 0);
        assert_eq!(c.success_rate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // ETA estimation tests
    // -----------------------------------------------------------------------

    /// After several warm-up samples the estimated remaining time should
    /// converge close to `ewma_mean * (1 - progress) / progress` (within 20 %
    /// tolerance).
    ///
    /// For jobs whose typical duration is 1 000 ms, at 25 % progress the
    /// formula predicts: 1000 * (1 - 0.25) / 0.25 = 3000 ms.
    #[test]
    fn test_eta_converges_within_tolerance_after_warm_up() {
        let mut est = JobEtaEstimator::new();
        // Simulate 20 completed jobs of 1 s each to warm up the EWMA mean.
        for _ in 0..20 {
            est.record_duration("transcode", Duration::from_secs(1));
        }

        // At 25 % progress: remaining = 1000 * 0.75 / 0.25 = 3000 ms.
        let remaining = est
            .estimate_remaining("transcode", Duration::from_millis(250), 0.25)
            .expect("estimate should exist after warm-up");

        // After 20 EWMA samples the mean should be very close to 1000 ms, so
        // the estimate should be close to 3000 ms.
        let expected_ms = 3000_u128;
        let actual_ms = remaining.as_millis();
        let tolerance = expected_ms / 5; // 20 %
        assert!(
            actual_ms.abs_diff(expected_ms) <= tolerance,
            "ETA expected ~{expected_ms} ms, got {actual_ms} ms"
        );
    }

    /// `estimate_remaining` must return `None` when no history exists.
    #[test]
    fn test_eta_returns_none_with_zero_history() {
        let est = JobEtaEstimator::new();
        let result = est.estimate_remaining("unknown-kind", Duration::from_millis(100), 0.5);
        assert!(
            result.is_none(),
            "expected None for unknown kind, got {result:?}"
        );
    }

    /// The estimated remaining duration must decrease (or stay constant) as
    /// `progress_frac` increases from a small value towards 1.0.
    #[test]
    fn test_eta_decreases_monotonically_as_progress_increases() {
        let mut est = JobEtaEstimator::new();
        for _ in 0..10 {
            est.record_duration("encode", Duration::from_secs(2));
        }

        let frac_points: &[f64] = &[0.1, 0.25, 0.5, 0.75, 0.9];
        let mut previous: Option<u128> = None;
        for &frac in frac_points {
            let rem = est
                .estimate_remaining("encode", Duration::from_millis(100), frac)
                .expect("estimate should exist")
                .as_millis();
            if let Some(prev) = previous {
                assert!(
                    rem <= prev,
                    "at progress {frac} remaining={rem} ms should be <= previous {prev} ms"
                );
            }
            previous = Some(rem);
        }
    }
}
