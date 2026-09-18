//! Workflow runtime metrics collection for `oximedia-workflow`.
//!
//! [`WorkflowMetricsCollector`] accumulates [`MetricSample`]s tagged with a
//! [`WorkflowMetric`] variant and exposes summary statistics (count, sum,
//! average, min, max) without requiring an external dependency.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Metric kinds ──────────────────────────────────────────────────────────────

/// Discriminates the type of measurement stored in a [`MetricSample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowMetric {
    /// Wall-clock time a task spent in the queue before execution (seconds).
    QueueWaitSeconds,
    /// Wall-clock execution time of a single task (seconds).
    TaskDurationSeconds,
    /// Peak resident memory used by a task (bytes).
    TaskMemoryBytes,
    /// CPU utilisation percentage sampled during task execution (0–100).
    CpuPercent,
    /// Number of retry attempts consumed by a task before success or failure.
    RetryCount,
    /// Total workflow wall-clock time from submission to completion (seconds).
    WorkflowDurationSeconds,
}

impl WorkflowMetric {
    /// Returns the SI unit label for this metric.
    #[must_use]
    pub fn unit(self) -> &'static str {
        match self {
            Self::QueueWaitSeconds | Self::TaskDurationSeconds | Self::WorkflowDurationSeconds => {
                "s"
            }
            Self::TaskMemoryBytes => "bytes",
            Self::CpuPercent => "%",
            Self::RetryCount => "count",
        }
    }

    /// Returns `true` when lower values indicate better performance.
    #[must_use]
    pub fn lower_is_better(self) -> bool {
        !matches!(self, Self::CpuPercent)
    }
}

impl std::fmt::Display for WorkflowMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::QueueWaitSeconds => "queue_wait_seconds",
            Self::TaskDurationSeconds => "task_duration_seconds",
            Self::TaskMemoryBytes => "task_memory_bytes",
            Self::CpuPercent => "cpu_percent",
            Self::RetryCount => "retry_count",
            Self::WorkflowDurationSeconds => "workflow_duration_seconds",
        };
        write!(f, "{s}")
    }
}

// ── Sample ────────────────────────────────────────────────────────────────────

/// A single metric observation tied to a workflow or task identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    /// The kind of metric being recorded.
    pub metric: WorkflowMetric,
    /// Identifier of the workflow or task that generated this sample.
    pub source_id: String,
    /// The measured value.
    pub value: f64,
}

impl MetricSample {
    /// Creates a new metric sample.
    #[must_use]
    pub fn new(metric: WorkflowMetric, source_id: impl Into<String>, value: f64) -> Self {
        Self {
            metric,
            source_id: source_id.into(),
            value,
        }
    }
}

// ── Summary ───────────────────────────────────────────────────────────────────

/// Aggregated statistics for a single [`WorkflowMetric`] type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    /// The metric these statistics describe.
    pub metric: WorkflowMetric,
    /// Number of samples included in the summary.
    pub count: usize,
    /// Sum of all sample values.
    pub sum: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
}

impl MetricSummary {
    /// Computes the arithmetic mean, or `0.0` when `count == 0`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

// ── Collector ─────────────────────────────────────────────────────────────────

/// Accumulates [`MetricSample`]s and produces per-metric [`MetricSummary`]s.
#[derive(Debug, Default, Clone)]
pub struct WorkflowMetricsCollector {
    samples: Vec<MetricSample>,
}

impl WorkflowMetricsCollector {
    /// Creates a new, empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a single metric observation.
    pub fn record(&mut self, sample: MetricSample) {
        self.samples.push(sample);
    }

    /// Convenience method to record a value directly.
    pub fn record_value(
        &mut self,
        metric: WorkflowMetric,
        source_id: impl Into<String>,
        value: f64,
    ) {
        self.record(MetricSample::new(metric, source_id, value));
    }

    /// Total number of samples collected so far.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns all samples for the given metric kind.
    #[must_use]
    pub fn samples_for(&self, metric: WorkflowMetric) -> Vec<&MetricSample> {
        self.samples.iter().filter(|s| s.metric == metric).collect()
    }

    /// Builds a [`MetricSummary`] for the given metric, or `None` if no
    /// samples of that kind have been collected.
    #[must_use]
    pub fn summarize(&self, metric: WorkflowMetric) -> Option<MetricSummary> {
        let relevant: Vec<f64> = self
            .samples
            .iter()
            .filter(|s| s.metric == metric)
            .map(|s| s.value)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        let sum: f64 = relevant.iter().sum();
        let min = relevant.iter().copied().fold(f64::INFINITY, f64::min);
        let max = relevant.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Some(MetricSummary {
            metric,
            count: relevant.len(),
            sum,
            min,
            max,
        })
    }

    /// Returns summaries for every metric kind that has at least one sample.
    #[must_use]
    pub fn all_summaries(&self) -> HashMap<WorkflowMetric, MetricSummary> {
        let mut map = HashMap::new();
        for metric in [
            WorkflowMetric::QueueWaitSeconds,
            WorkflowMetric::TaskDurationSeconds,
            WorkflowMetric::TaskMemoryBytes,
            WorkflowMetric::CpuPercent,
            WorkflowMetric::RetryCount,
            WorkflowMetric::WorkflowDurationSeconds,
        ] {
            if let Some(summary) = self.summarize(metric) {
                map.insert(metric, summary);
            }
        }
        map
    }

    /// Clears all collected samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn collector_with_samples() -> WorkflowMetricsCollector {
        let mut c = WorkflowMetricsCollector::new();
        c.record_value(WorkflowMetric::TaskDurationSeconds, "task-1", 10.0);
        c.record_value(WorkflowMetric::TaskDurationSeconds, "task-2", 20.0);
        c.record_value(WorkflowMetric::TaskDurationSeconds, "task-3", 30.0);
        c.record_value(WorkflowMetric::CpuPercent, "task-1", 55.0);
        c
    }

    #[test]
    fn test_new_collector_empty() {
        let c = WorkflowMetricsCollector::new();
        assert_eq!(c.sample_count(), 0);
    }

    #[test]
    fn test_record_value_increments_count() {
        let mut c = WorkflowMetricsCollector::new();
        c.record_value(WorkflowMetric::RetryCount, "wf-1", 2.0);
        assert_eq!(c.sample_count(), 1);
    }

    #[test]
    fn test_samples_for_filters_correctly() {
        let c = collector_with_samples();
        let dur_samples = c.samples_for(WorkflowMetric::TaskDurationSeconds);
        assert_eq!(dur_samples.len(), 3);
    }

    #[test]
    fn test_summarize_mean() {
        let c = collector_with_samples();
        let summary = c
            .summarize(WorkflowMetric::TaskDurationSeconds)
            .expect("should succeed in test");
        // (10 + 20 + 30) / 3 = 20.0
        assert!((summary.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_summarize_min_max() {
        let c = collector_with_samples();
        let summary = c
            .summarize(WorkflowMetric::TaskDurationSeconds)
            .expect("should succeed in test");
        assert!((summary.min - 10.0).abs() < 1e-9);
        assert!((summary.max - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_summarize_count() {
        let c = collector_with_samples();
        let summary = c
            .summarize(WorkflowMetric::TaskDurationSeconds)
            .expect("should succeed in test");
        assert_eq!(summary.count, 3);
    }

    #[test]
    fn test_summarize_none_for_missing_metric() {
        let c = collector_with_samples();
        assert!(c.summarize(WorkflowMetric::QueueWaitSeconds).is_none());
    }

    #[test]
    fn test_all_summaries_keys() {
        let c = collector_with_samples();
        let summaries = c.all_summaries();
        assert!(summaries.contains_key(&WorkflowMetric::TaskDurationSeconds));
        assert!(summaries.contains_key(&WorkflowMetric::CpuPercent));
        assert!(!summaries.contains_key(&WorkflowMetric::QueueWaitSeconds));
    }

    #[test]
    fn test_reset_clears_samples() {
        let mut c = collector_with_samples();
        c.reset();
        assert_eq!(c.sample_count(), 0);
        assert!(c.summarize(WorkflowMetric::TaskDurationSeconds).is_none());
    }

    #[test]
    fn test_metric_unit() {
        assert_eq!(WorkflowMetric::TaskDurationSeconds.unit(), "s");
        assert_eq!(WorkflowMetric::TaskMemoryBytes.unit(), "bytes");
        assert_eq!(WorkflowMetric::CpuPercent.unit(), "%");
        assert_eq!(WorkflowMetric::RetryCount.unit(), "count");
    }

    #[test]
    fn test_metric_lower_is_better() {
        assert!(WorkflowMetric::TaskDurationSeconds.lower_is_better());
        assert!(WorkflowMetric::QueueWaitSeconds.lower_is_better());
        assert!(!WorkflowMetric::CpuPercent.lower_is_better());
    }

    #[test]
    fn test_metric_display() {
        assert_eq!(
            format!("{}", WorkflowMetric::TaskDurationSeconds),
            "task_duration_seconds"
        );
    }

    #[test]
    fn test_metric_summary_mean_empty() {
        let s = MetricSummary {
            metric: WorkflowMetric::RetryCount,
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
        };
        assert!((s.mean() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_sample_summary() {
        let mut c = WorkflowMetricsCollector::new();
        c.record_value(WorkflowMetric::WorkflowDurationSeconds, "wf-a", 42.0);
        let s = c
            .summarize(WorkflowMetric::WorkflowDurationSeconds)
            .expect("should succeed in test");
        assert_eq!(s.count, 1);
        assert!((s.min - 42.0).abs() < 1e-9);
        assert!((s.max - 42.0).abs() < 1e-9);
        assert!((s.mean() - 42.0).abs() < 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workflow-level execution metrics and aggregator
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-step timing and resource metrics captured during a workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetric {
    /// Step identifier.
    pub step_id: String,
    /// Wall-clock time at which the step started.
    pub started_at: std::time::SystemTime,
    /// How long the step took to complete (seconds).
    pub duration_secs: f64,
    /// Whether the step completed successfully.
    pub success: bool,
    /// How many retry attempts were consumed.
    pub retries: u32,
    /// Optional output size in bytes (e.g. transcoded file size).
    pub output_size_bytes: Option<u64>,
    /// Optional CPU time consumed by the step (seconds).
    pub cpu_seconds: Option<f64>,
}

impl StepMetric {
    /// Construct a minimal step metric with only the required fields.
    #[must_use]
    pub fn new(step_id: impl Into<String>, duration_secs: f64, success: bool) -> Self {
        Self {
            step_id: step_id.into(),
            started_at: std::time::SystemTime::now(),
            duration_secs,
            success,
            retries: 0,
            output_size_bytes: None,
            cpu_seconds: None,
        }
    }
}

/// Execution metrics for a single workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRunMetrics {
    /// Workflow identifier.
    pub workflow_id: String,
    /// Wall-clock time at which the workflow started.
    pub started_at: std::time::SystemTime,
    /// Wall-clock time at which the workflow completed (if finished).
    pub completed_at: Option<std::time::SystemTime>,
    /// Per-step metrics collected during this run.
    pub step_metrics: Vec<StepMetric>,
    /// Total wall-clock duration of the workflow (seconds), if complete.
    pub total_duration_secs: Option<f64>,
    /// Whether the workflow completed successfully.
    pub success: Option<bool>,
}

impl WorkflowRunMetrics {
    /// Create a new metrics record for a workflow that has just started.
    #[must_use]
    pub fn new(workflow_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            started_at: std::time::SystemTime::now(),
            completed_at: None,
            step_metrics: Vec::new(),
            total_duration_secs: None,
            success: None,
        }
    }

    /// Mark the workflow as finished and compute the total duration.
    pub fn finish(&mut self, success: bool) {
        let now = std::time::SystemTime::now();
        self.completed_at = Some(now);
        self.success = Some(success);
        self.total_duration_secs = now
            .duration_since(self.started_at)
            .map(|d| d.as_secs_f64())
            .ok();
    }
}

/// Accumulates [`WorkflowRunMetrics`] across many runs and computes aggregate
/// statistics.
pub struct WorkflowMetricsAggregator {
    history: Vec<WorkflowRunMetrics>,
    max_history: usize,
}

impl WorkflowMetricsAggregator {
    /// Create a new aggregator that retains at most `max_history` workflow runs.
    ///
    /// When `max_history` is exceeded the oldest entry is evicted.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
        }
    }

    /// Add a completed workflow run to the history.
    ///
    /// If the history is at capacity the oldest entry is removed first.
    pub fn record(&mut self, metrics: WorkflowRunMetrics) {
        if self.max_history > 0 && self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(metrics);
    }

    /// Number of workflow runs currently in the history.
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Return `true` when no runs have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Mean total duration across all completed runs, or `None` when no
    /// completed run is present.
    #[must_use]
    pub fn avg_duration_secs(&self) -> Option<f64> {
        let durations: Vec<f64> = self
            .history
            .iter()
            .filter_map(|m| m.total_duration_secs)
            .collect();
        if durations.is_empty() {
            None
        } else {
            #[allow(clippy::cast_precision_loss)]
            Some(durations.iter().sum::<f64>() / durations.len() as f64)
        }
    }

    /// Fraction of runs (0.0–1.0) that completed successfully.
    ///
    /// Runs without a recorded `success` value are excluded from the
    /// denominator.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let finished: Vec<bool> = self.history.iter().filter_map(|m| m.success).collect();
        if finished.is_empty() {
            return 0.0;
        }
        let successes = finished.iter().filter(|&&s| s).count();
        successes as f64 / finished.len() as f64
    }

    /// 95th-percentile total duration across all completed runs, or `None`
    /// when fewer than two completed runs are present.
    #[must_use]
    pub fn p95_duration_secs(&self) -> Option<f64> {
        let mut durations: Vec<f64> = self
            .history
            .iter()
            .filter_map(|m| m.total_duration_secs)
            .collect();
        if durations.is_empty() {
            return None;
        }
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        #[allow(clippy::cast_precision_loss)]
        let idx = ((durations.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let idx = idx.min(durations.len() - 1);
        Some(durations[idx])
    }

    /// Return the `n` slowest steps by average duration across all runs.
    ///
    /// Returns `(step_id, avg_duration_secs)` pairs sorted by descending
    /// average duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn slowest_steps(&self, n: usize) -> Vec<(String, f64)> {
        let mut totals: HashMap<String, (f64, usize)> = HashMap::new();
        for run in &self.history {
            for step in &run.step_metrics {
                let entry = totals.entry(step.step_id.clone()).or_insert((0.0, 0));
                entry.0 += step.duration_secs;
                entry.1 += 1;
            }
        }
        let mut avgs: Vec<(String, f64)> = totals
            .into_iter()
            .map(|(id, (total, count))| (id, total / count as f64))
            .collect();
        avgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        avgs.truncate(n);
        avgs
    }

    /// Return failure rates per step, sorted by descending failure rate.
    ///
    /// Returns `(step_id, failure_rate)` where `failure_rate` is in `0.0..=1.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn failure_rate_by_step(&self) -> Vec<(String, f64)> {
        let mut counts: HashMap<String, (usize, usize)> = HashMap::new(); // (total, failures)
        for run in &self.history {
            for step in &run.step_metrics {
                let entry = counts.entry(step.step_id.clone()).or_insert((0, 0));
                entry.0 += 1;
                if !step.success {
                    entry.1 += 1;
                }
            }
        }
        let mut rates: Vec<(String, f64)> = counts
            .into_iter()
            .map(|(id, (total, failures))| (id, failures as f64 / total as f64))
            .collect();
        rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rates
    }

    /// Read-only view of all recorded runs.
    #[must_use]
    pub fn history(&self) -> &[WorkflowRunMetrics] {
        &self.history
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod aggregator_tests {
    use super::*;

    fn make_run(
        id: &str,
        duration: f64,
        success: bool,
        steps: Vec<StepMetric>,
    ) -> WorkflowRunMetrics {
        WorkflowRunMetrics {
            workflow_id: id.to_string(),
            started_at: std::time::SystemTime::now(),
            completed_at: Some(std::time::SystemTime::now()),
            step_metrics: steps,
            total_duration_secs: Some(duration),
            success: Some(success),
        }
    }

    fn make_step(id: &str, duration: f64, success: bool) -> StepMetric {
        StepMetric::new(id, duration, success)
    }

    #[test]
    fn test_aggregator_new_is_empty() {
        let agg = WorkflowMetricsAggregator::new(100);
        assert!(agg.is_empty());
        assert_eq!(agg.len(), 0);
    }

    #[test]
    fn test_record_increments_len() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        agg.record(make_run("wf-1", 10.0, true, vec![]));
        assert_eq!(agg.len(), 1);
    }

    #[test]
    fn test_max_history_evicts_oldest() {
        let mut agg = WorkflowMetricsAggregator::new(2);
        agg.record(make_run("wf-1", 10.0, true, vec![]));
        agg.record(make_run("wf-2", 20.0, true, vec![]));
        agg.record(make_run("wf-3", 30.0, true, vec![]));
        assert_eq!(agg.len(), 2);
        // Oldest (wf-1) should have been evicted.
        assert_eq!(agg.history()[0].workflow_id, "wf-2");
    }

    #[test]
    fn test_avg_duration_secs_correct() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        agg.record(make_run("wf-1", 10.0, true, vec![]));
        agg.record(make_run("wf-2", 30.0, true, vec![]));
        let avg = agg.avg_duration_secs().expect("should have avg");
        assert!((avg - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_duration_none_when_empty() {
        let agg = WorkflowMetricsAggregator::new(100);
        assert!(agg.avg_duration_secs().is_none());
    }

    #[test]
    fn test_success_rate_all_success() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        agg.record(make_run("wf-1", 10.0, true, vec![]));
        agg.record(make_run("wf-2", 10.0, true, vec![]));
        assert!((agg.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_half_success() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        agg.record(make_run("wf-1", 10.0, true, vec![]));
        agg.record(make_run("wf-2", 10.0, false, vec![]));
        assert!((agg.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_zero_when_empty() {
        let agg = WorkflowMetricsAggregator::new(100);
        assert!((agg.success_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_p95_duration_secs() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        // 20 values: 1..=20; p95 index = ceil(20 * 0.95) - 1 = 18 → value 19
        for i in 1u32..=20 {
            agg.record(make_run(&format!("wf-{i}"), f64::from(i), true, vec![]));
        }
        let p95 = agg.p95_duration_secs().expect("should have p95");
        assert!((p95 - 19.0).abs() < 1e-9);
    }

    #[test]
    fn test_p95_none_when_empty() {
        let agg = WorkflowMetricsAggregator::new(100);
        assert!(agg.p95_duration_secs().is_none());
    }

    #[test]
    fn test_slowest_steps_returns_top_n() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        agg.record(make_run(
            "wf-1",
            100.0,
            true,
            vec![
                make_step("transcode", 80.0, true),
                make_step("ingest", 10.0, true),
                make_step("deliver", 5.0, true),
            ],
        ));
        let slowest = agg.slowest_steps(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].0, "transcode");
        assert_eq!(slowest[1].0, "ingest");
    }

    #[test]
    fn test_failure_rate_by_step_sorted_desc() {
        let mut agg = WorkflowMetricsAggregator::new(100);
        // transcode fails 2/2, ingest fails 0/2
        for _ in 0..2 {
            agg.record(make_run(
                "wf",
                10.0,
                false,
                vec![
                    make_step("transcode", 5.0, false),
                    make_step("ingest", 2.0, true),
                ],
            ));
        }
        let rates = agg.failure_rate_by_step();
        assert!(!rates.is_empty());
        assert_eq!(rates[0].0, "transcode");
        assert!((rates[0].1 - 1.0).abs() < 1e-9);
        let ingest_rate = rates
            .iter()
            .find(|(id, _)| id == "ingest")
            .map(|(_, r)| *r)
            .unwrap_or(0.0);
        assert!((ingest_rate - 0.0).abs() < 1e-9);
    }
}
