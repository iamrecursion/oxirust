//! Pipeline profiler — per-node timing measurement and reporting.
//!
//! [`PipelineProfiler`] collects [`NodeTimingSample`]s from pipeline execution
//! and produces a [`ProfilingReport`] summarising throughput, latency, and
//! bottleneck analysis across all nodes.
//!
//! # Design goals
//!
//! * **Zero-unwrap**: all operations return `Result` or `Option`.
//! * **Pure Rust**: no external timing library; uses `std::time::Instant`.
//! * **Immutable analysis**: the profiler accumulates samples; the report is
//!   a snapshot computed on demand.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::profiler::{PipelineProfiler, NodeTimingSample};
//! use oximedia_pipeline::node::NodeId;
//! use std::time::Duration;
//!
//! let mut profiler = PipelineProfiler::new();
//!
//! let node_id = NodeId::new();
//! profiler.record(NodeTimingSample::new(node_id, "scale".into(), Duration::from_millis(10)));
//! profiler.record(NodeTimingSample::new(node_id, "scale".into(), Duration::from_millis(12)));
//! profiler.record(NodeTimingSample::new(node_id, "scale".into(), Duration::from_millis(8)));
//!
//! let report = profiler.report();
//! let summary = report.summary_for(node_id).expect("should have summary");
//! assert_eq!(summary.sample_count, 3);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::node::NodeId;
use crate::PipelineError;

// ── NodeTimingSample ──────────────────────────────────────────────────────────

/// A single timing observation for one node execution.
#[derive(Debug, Clone)]
pub struct NodeTimingSample {
    /// Which node produced this sample.
    pub node_id: NodeId,
    /// Human-readable node name (captured at record time for stable reports).
    pub node_name: String,
    /// Wall-clock duration of this execution.
    pub duration: Duration,
    /// When this sample was collected (relative epoch is arbitrary).
    pub timestamp: Instant,
    /// Number of frames (or audio packets) processed in this execution.
    pub frames_processed: u64,
}

impl NodeTimingSample {
    /// Create a new sample with `frames_processed` defaulting to 1.
    pub fn new(node_id: NodeId, node_name: String, duration: Duration) -> Self {
        Self {
            node_id,
            node_name,
            duration,
            timestamp: Instant::now(),
            frames_processed: 1,
        }
    }

    /// Create a sample specifying the number of frames processed.
    pub fn with_frames(
        node_id: NodeId,
        node_name: String,
        duration: Duration,
        frames: u64,
    ) -> Self {
        Self {
            node_id,
            node_name,
            duration,
            timestamp: Instant::now(),
            frames_processed: frames,
        }
    }

    /// Throughput in frames per second for this single sample.
    ///
    /// Returns `None` if the duration is zero.
    pub fn throughput_fps(&self) -> Option<f64> {
        let secs = self.duration.as_secs_f64();
        if secs < f64::EPSILON {
            None
        } else {
            Some(self.frames_processed as f64 / secs)
        }
    }
}

// ── NodeProfilingSummary ──────────────────────────────────────────────────────

/// Per-node statistical summary computed from collected samples.
#[derive(Debug, Clone)]
pub struct NodeProfilingSummary {
    /// Node identifier.
    pub node_id: NodeId,
    /// Human-readable name (from the first sample recorded for this node).
    pub node_name: String,
    /// Total number of samples collected.
    pub sample_count: usize,
    /// Minimum observed duration.
    pub min_duration: Duration,
    /// Maximum observed duration.
    pub max_duration: Duration,
    /// Mean (arithmetic average) duration.
    pub mean_duration: Duration,
    /// Median duration (p50).
    pub median_duration: Duration,
    /// 95th-percentile duration.
    pub p95_duration: Duration,
    /// 99th-percentile duration.
    pub p99_duration: Duration,
    /// Total wall-clock time accumulated across all samples.
    pub total_duration: Duration,
    /// Total frames processed across all samples.
    pub total_frames: u64,
    /// Mean throughput in frames per second (`total_frames / total_duration`).
    pub mean_throughput_fps: f64,
}

impl NodeProfilingSummary {
    fn compute(node_id: NodeId, samples: &[NodeTimingSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let node_name = samples[0].node_name.clone();
        let mut durations: Vec<Duration> = samples.iter().map(|s| s.duration).collect();
        durations.sort();

        let total_duration: Duration = durations.iter().sum();
        let total_frames: u64 = samples.iter().map(|s| s.frames_processed).sum();
        let count = durations.len();

        let min_duration = durations[0];
        let max_duration = durations[count - 1];

        let mean_nanos = total_duration.as_nanos() / count as u128;
        let mean_duration = Duration::from_nanos(mean_nanos as u64);

        let median_duration = percentile(&durations, 50);
        let p95_duration = percentile(&durations, 95);
        let p99_duration = percentile(&durations, 99);

        let mean_throughput_fps = {
            let secs = total_duration.as_secs_f64();
            if secs < f64::EPSILON {
                0.0
            } else {
                total_frames as f64 / secs
            }
        };

        Some(Self {
            node_id,
            node_name,
            sample_count: count,
            min_duration,
            max_duration,
            mean_duration,
            median_duration,
            p95_duration,
            p99_duration,
            total_duration,
            total_frames,
            mean_throughput_fps,
        })
    }
}

/// Compute the Nth percentile of a **sorted** duration slice.
fn percentile(sorted: &[Duration], p: u8) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((p as usize * (sorted.len() - 1)) + 50) / 100;
    let clamped = idx.min(sorted.len() - 1);
    sorted[clamped]
}

// ── ProfilingReport ────────────────────────────────────────────────────────────

/// A snapshot of all profiling data, computed on demand from collected samples.
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    summaries: HashMap<NodeId, NodeProfilingSummary>,
    /// Total samples recorded across all nodes.
    pub total_samples: usize,
    /// Overall wall-clock span from first to last sample (if >= 2 samples).
    pub total_elapsed: Option<Duration>,
}

impl ProfilingReport {
    /// Retrieve the profiling summary for a specific node.
    pub fn summary_for(&self, node_id: NodeId) -> Option<&NodeProfilingSummary> {
        self.summaries.get(&node_id)
    }

    /// All per-node summaries, sorted by mean duration descending (slowest
    /// first — the bottleneck nodes appear at the top).
    pub fn sorted_by_mean_desc(&self) -> Vec<&NodeProfilingSummary> {
        let mut v: Vec<&NodeProfilingSummary> = self.summaries.values().collect();
        v.sort_by(|a, b| b.mean_duration.cmp(&a.mean_duration));
        v
    }

    /// Return the bottleneck node — the node with the highest mean duration.
    pub fn bottleneck(&self) -> Option<&NodeProfilingSummary> {
        self.summaries.values().max_by_key(|s| s.mean_duration)
    }

    /// Return the node with the lowest mean throughput (fps).
    pub fn lowest_throughput(&self) -> Option<&NodeProfilingSummary> {
        self.summaries
            .values()
            .filter(|s| s.mean_throughput_fps > 0.0)
            .min_by(|a, b| {
                a.mean_throughput_fps
                    .partial_cmp(&b.mean_throughput_fps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Number of unique nodes profiled.
    pub fn node_count(&self) -> usize {
        self.summaries.len()
    }

    /// Format the report as a human-readable multi-line string suitable for
    /// logging or display in a terminal.
    pub fn format_text(&self) -> String {
        let mut out = String::new();
        out.push_str("┌─ Pipeline Profiling Report ──────────────────────────────────────\n");
        out.push_str(&format!(
            "│  Total nodes: {}  |  Total samples: {}",
            self.node_count(),
            self.total_samples
        ));
        if let Some(elapsed) = self.total_elapsed {
            out.push_str(&format!("  |  Elapsed: {:.2}s", elapsed.as_secs_f64()));
        }
        out.push('\n');
        out.push_str("├───────────────────────────────────────────────────────────────────\n");
        out.push_str("│  Node Name            │ Count │  Mean   │   P95   │   P99   │  FPS  \n");
        out.push_str("├───────────────────────────────────────────────────────────────────\n");

        for summary in self.sorted_by_mean_desc() {
            out.push_str(&format!(
                "│  {:<22} │ {:>5} │ {:>7} │ {:>7} │ {:>7} │ {:>6.1}\n",
                truncate_name(&summary.node_name, 22),
                summary.sample_count,
                format_duration(summary.mean_duration),
                format_duration(summary.p95_duration),
                format_duration(summary.p99_duration),
                summary.mean_throughput_fps,
            ));
        }
        out.push_str("└───────────────────────────────────────────────────────────────────\n");
        out
    }
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{us}µs")
    } else {
        format!("{:.1}ms", d.as_secs_f64() * 1000.0)
    }
}

fn truncate_name(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ── PipelineProfiler ──────────────────────────────────────────────────────────

/// Collects timing samples for pipeline nodes and generates profiling reports.
///
/// # Thread safety
///
/// `PipelineProfiler` is not `Sync`; for multi-threaded collection, create one
/// profiler per thread and merge them via [`PipelineProfiler::merge`].
#[derive(Debug, Default)]
pub struct PipelineProfiler {
    samples: Vec<NodeTimingSample>,
}

impl PipelineProfiler {
    /// Create a new empty profiler.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Create a profiler with a pre-allocated capacity hint.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
        }
    }

    /// Record a single timing sample.
    pub fn record(&mut self, sample: NodeTimingSample) {
        self.samples.push(sample);
    }

    /// Record a timing event using a closure: measure how long `f` takes, then
    /// store the result as a sample for `node_id`.
    ///
    /// Returns the value produced by `f`.
    pub fn time<T, F>(&mut self, node_id: NodeId, node_name: String, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.samples
            .push(NodeTimingSample::new(node_id, node_name, elapsed));
        result
    }

    /// Record a timing event specifying how many frames were processed.
    pub fn time_frames<T, F>(&mut self, node_id: NodeId, node_name: String, frames: u64, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.samples.push(NodeTimingSample::with_frames(
            node_id, node_name, elapsed, frames,
        ));
        result
    }

    /// Total number of samples collected so far.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Whether any samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Clear all collected samples, resetting the profiler.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Merge samples from another profiler into this one.
    pub fn merge(&mut self, other: PipelineProfiler) {
        self.samples.extend(other.samples);
    }

    /// Compute and return a [`ProfilingReport`] from all collected samples.
    ///
    /// This is `O(n log n)` where `n` is the total number of samples; reports
    /// are computed on demand and not cached.
    pub fn report(&self) -> ProfilingReport {
        // Group samples by NodeId
        let mut by_node: HashMap<NodeId, Vec<&NodeTimingSample>> = HashMap::new();
        for sample in &self.samples {
            by_node.entry(sample.node_id).or_default().push(sample);
        }

        let mut summaries: HashMap<NodeId, NodeProfilingSummary> = HashMap::new();
        for (id, node_samples) in &by_node {
            let owned: Vec<NodeTimingSample> = node_samples.iter().map(|s| (*s).clone()).collect();
            if let Some(summary) = NodeProfilingSummary::compute(*id, &owned) {
                summaries.insert(*id, summary);
            }
        }

        let total_elapsed = if self.samples.len() >= 2 {
            let first = self.samples.iter().map(|s| s.timestamp).min();
            let last = self.samples.iter().map(|s| s.timestamp).max();
            match (first, last) {
                (Some(f), Some(l)) if l > f => Some(l.duration_since(f)),
                _ => None,
            }
        } else {
            None
        };

        ProfilingReport {
            summaries,
            total_samples: self.samples.len(),
            total_elapsed,
        }
    }

    /// Return a `Result` wrapping the report, or a `ProfilerError` if no
    /// samples have been collected.
    pub fn report_or_err(&self) -> Result<ProfilingReport, PipelineError> {
        if self.is_empty() {
            return Err(PipelineError::ProfilerError(
                "no samples collected".to_string(),
            ));
        }
        Ok(self.report())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(id: NodeId, name: &str, millis: u64) -> NodeTimingSample {
        NodeTimingSample::new(id, name.to_string(), Duration::from_millis(millis))
    }

    #[test]
    fn profiler_empty_report_err() {
        let p = PipelineProfiler::new();
        assert!(p.report_or_err().is_err());
    }

    #[test]
    fn profiler_record_single_sample() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        p.record(make_sample(id, "scale", 10));
        assert_eq!(p.sample_count(), 1);
    }

    #[test]
    fn report_summary_mean() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        p.record(make_sample(id, "scale", 10));
        p.record(make_sample(id, "scale", 20));
        p.record(make_sample(id, "scale", 30));
        let report = p.report();
        let s = report.summary_for(id).expect("summary");
        assert_eq!(s.sample_count, 3);
        assert_eq!(s.mean_duration, Duration::from_millis(20));
        assert_eq!(s.min_duration, Duration::from_millis(10));
        assert_eq!(s.max_duration, Duration::from_millis(30));
    }

    #[test]
    fn report_bottleneck_detection() {
        let mut p = PipelineProfiler::new();
        let fast = NodeId::new();
        let slow = NodeId::new();
        p.record(make_sample(fast, "hflip", 1));
        p.record(make_sample(fast, "hflip", 2));
        p.record(make_sample(slow, "scale", 50));
        p.record(make_sample(slow, "scale", 60));
        let report = p.report();
        let bottleneck = report.bottleneck().expect("should have bottleneck");
        assert_eq!(bottleneck.node_name, "scale");
    }

    #[test]
    fn report_sorted_by_mean_desc() {
        let mut p = PipelineProfiler::new();
        let a = NodeId::new();
        let b = NodeId::new();
        let c = NodeId::new();
        p.record(make_sample(a, "a", 5));
        p.record(make_sample(b, "b", 20));
        p.record(make_sample(c, "c", 10));
        let report = p.report();
        let sorted = report.sorted_by_mean_desc();
        assert_eq!(sorted[0].node_name, "b");
        assert_eq!(sorted[1].node_name, "c");
        assert_eq!(sorted[2].node_name, "a");
    }

    #[test]
    fn profiler_time_closure() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        let result = p.time(id, "work".to_string(), || {
            // simulate work
            42u32
        });
        assert_eq!(result, 42);
        assert_eq!(p.sample_count(), 1);
    }

    #[test]
    fn profiler_time_frames() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        let _ = p.time_frames(id, "decode".to_string(), 30, || "done");
        let report = p.report();
        let s = report.summary_for(id).expect("has summary");
        assert_eq!(s.total_frames, 30);
    }

    #[test]
    fn profiler_clear_resets() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        p.record(make_sample(id, "n", 5));
        assert_eq!(p.sample_count(), 1);
        p.clear();
        assert_eq!(p.sample_count(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn profiler_merge() {
        let id = NodeId::new();
        let mut p1 = PipelineProfiler::new();
        p1.record(make_sample(id, "n", 5));

        let mut p2 = PipelineProfiler::new();
        p2.record(make_sample(id, "n", 10));
        p2.record(make_sample(id, "n", 15));

        p1.merge(p2);
        assert_eq!(p1.sample_count(), 3);
        let report = p1.report();
        let s = report.summary_for(id).expect("summary");
        assert_eq!(s.sample_count, 3);
    }

    #[test]
    fn report_percentiles() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        // Add 100 samples: 1ms … 100ms
        for i in 1u64..=100 {
            p.record(make_sample(id, "n", i));
        }
        let report = p.report();
        let s = report.summary_for(id).expect("summary");
        // p95 should be around 95ms
        assert!(s.p95_duration >= Duration::from_millis(94));
        assert!(s.p95_duration <= Duration::from_millis(97));
        // p99 should be around 99ms
        assert!(s.p99_duration >= Duration::from_millis(97));
    }

    #[test]
    fn report_format_text_contains_headers() {
        let mut p = PipelineProfiler::new();
        let id = NodeId::new();
        p.record(make_sample(id, "my_node", 10));
        let report = p.report();
        let text = report.format_text();
        assert!(
            text.contains("Pipeline Profiling Report"),
            "should have title"
        );
        assert!(text.contains("my_node"), "should have node name");
    }

    #[test]
    fn throughput_fps_zero_duration() {
        let id = NodeId::new();
        let s = NodeTimingSample::new(id, "n".to_string(), Duration::ZERO);
        assert!(s.throughput_fps().is_none());
    }

    #[test]
    fn throughput_fps_nonzero() {
        let id = NodeId::new();
        let s = NodeTimingSample::with_frames(id, "n".to_string(), Duration::from_secs(1), 60);
        let fps = s.throughput_fps().expect("should have throughput");
        assert!((fps - 60.0).abs() < 0.001);
    }

    #[test]
    fn report_node_count() {
        let mut p = PipelineProfiler::new();
        for _ in 0..5 {
            p.record(make_sample(NodeId::new(), "n", 5));
        }
        let report = p.report();
        assert_eq!(report.node_count(), 5);
        assert_eq!(report.total_samples, 5);
    }

    #[test]
    fn with_capacity_profiler() {
        let p = PipelineProfiler::with_capacity(128);
        assert!(p.is_empty());
    }

    #[test]
    fn lowest_throughput_detection() {
        let mut p = PipelineProfiler::new();
        // fast node: processes 60fps
        let fast = NodeId::new();
        p.record(NodeTimingSample::with_frames(
            fast,
            "fast".to_string(),
            Duration::from_secs(1),
            60,
        ));
        // slow node: processes 10fps
        let slow = NodeId::new();
        p.record(NodeTimingSample::with_frames(
            slow,
            "slow".to_string(),
            Duration::from_secs(1),
            10,
        ));
        let report = p.report();
        let low = report.lowest_throughput().expect("should find");
        assert_eq!(low.node_name, "slow");
    }
}
