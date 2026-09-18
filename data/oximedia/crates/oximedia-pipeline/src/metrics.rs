//! Pipeline metrics — per-node throughput, latency, and buffer statistics.
//!
//! [`PipelineMetrics`] collects runtime statistics from pipeline execution and
//! provides aggregated views per node.  Each node's data is stored in a
//! [`NodeMetrics`] entry containing [`ThroughputStats`], [`LatencyStats`], and
//! [`BufferStats`].
//!
//! # Design goals
//!
//! * **Zero-unwrap**: all operations return `Result` or `Option`.
//! * **Pure Rust**: no external crates beyond `std`.
//! * **Atomic-friendly**: individual counters use `u64` arithmetic; thread
//!   safety is left to the caller (wrap in a `Mutex` for multi-threaded use).
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::metrics::{PipelineMetrics, NodeMetrics};
//! use oximedia_pipeline::node::NodeId;
//! use std::time::Duration;
//!
//! let mut metrics = PipelineMetrics::new();
//! let id = NodeId::new();
//!
//! metrics.record_frames(id, "scale", 10);
//! metrics.record_bytes(id, "scale", 1_024_000);
//! metrics.record_latency(id, "scale", Duration::from_micros(800));
//!
//! let node = metrics.node(id).expect("should exist");
//! assert_eq!(node.throughput.total_frames, 10);
//! ```

use std::collections::HashMap;
use std::time::Duration;

use crate::node::NodeId;
use crate::PipelineError;

// ── ThroughputStats ───────────────────────────────────────────────────────────

/// Throughput statistics for a single pipeline node.
#[derive(Debug, Clone, Default)]
pub struct ThroughputStats {
    /// Total number of frames (or audio packets) processed.
    pub total_frames: u64,
    /// Total number of bytes processed.
    pub total_bytes: u64,
    /// Number of times this node's throughput was sampled.
    pub sample_count: u64,
    /// Cumulative wall-clock processing time reported for throughput samples.
    pub cumulative_duration: Duration,
    /// Peak frames-per-second observed across all throughput samples.
    pub peak_fps: f64,
    /// Number of frames that were dropped (skipped / backpressure evictions).
    pub dropped_frames: u64,
}

impl ThroughputStats {
    /// Mean frames-per-second averaged over `cumulative_duration`.
    ///
    /// Returns `0.0` when no time has elapsed.
    pub fn mean_fps(&self) -> f64 {
        let secs = self.cumulative_duration.as_secs_f64();
        if secs < f64::EPSILON {
            0.0
        } else {
            self.total_frames as f64 / secs
        }
    }

    /// Mean throughput in bytes per second.
    pub fn mean_bytes_per_sec(&self) -> f64 {
        let secs = self.cumulative_duration.as_secs_f64();
        if secs < f64::EPSILON {
            0.0
        } else {
            self.total_bytes as f64 / secs
        }
    }

    /// Drop rate as a fraction of total frames (0.0–1.0).
    pub fn drop_rate(&self) -> f64 {
        let total = self.total_frames + self.dropped_frames;
        if total == 0 {
            0.0
        } else {
            self.dropped_frames as f64 / total as f64
        }
    }
}

// ── LatencyStats ──────────────────────────────────────────────────────────────

/// Latency statistics for a single pipeline node.
///
/// Latency here means the wall-clock time spent *inside* the node per unit of
/// work (e.g. per frame), as measured by the calling executor.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// All observed latency samples, kept in insertion order.
    samples: Vec<Duration>,
    /// Sum of all samples (used for efficient mean computation).
    total: Duration,
    /// Minimum observed latency.
    pub min: Duration,
    /// Maximum observed latency.
    pub max: Duration,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            total: Duration::ZERO,
            min: Duration::MAX,
            max: Duration::ZERO,
        }
    }
}

impl LatencyStats {
    /// Record a new latency observation.
    pub fn record(&mut self, d: Duration) {
        self.total = self.total.saturating_add(d);
        if d < self.min {
            self.min = d;
        }
        if d > self.max {
            self.max = d;
        }
        self.samples.push(d);
    }

    /// Number of latency samples collected.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Arithmetic mean latency, or `None` if no samples have been recorded.
    pub fn mean(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let nanos = self.total.as_nanos() / self.samples.len() as u128;
        Some(Duration::from_nanos(nanos as u64))
    }

    /// Median latency (p50), computed by sorting a copy of all samples.
    ///
    /// Returns `None` when no samples have been recorded.
    pub fn median(&self) -> Option<Duration> {
        percentile_duration(&self.samples, 50)
    }

    /// 95th-percentile latency.
    pub fn p95(&self) -> Option<Duration> {
        percentile_duration(&self.samples, 95)
    }

    /// 99th-percentile latency.
    pub fn p99(&self) -> Option<Duration> {
        percentile_duration(&self.samples, 99)
    }

    /// Whether any latency samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Compute the Nth percentile of an unsorted duration slice by cloning and sorting.
fn percentile_duration(samples: &[Duration], p: u8) -> Option<Duration> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = ((p as usize) * (sorted.len() - 1)) / 100;
    Some(sorted[idx.min(sorted.len() - 1)])
}

// ── BufferStats ───────────────────────────────────────────────────────────────

/// Buffer / queue occupancy statistics for a single pipeline node.
///
/// These reflect the state of any intermediate buffer (e.g. a frame queue or
/// ring-buffer) between this node and its downstream consumer.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Current occupancy (number of frames or bytes, depending on context).
    pub current_occupancy: u64,
    /// Peak occupancy observed since the last reset.
    pub peak_occupancy: u64,
    /// Configured buffer capacity (0 = unknown / unbounded).
    pub capacity: u64,
    /// Total number of enqueue operations recorded.
    pub enqueue_count: u64,
    /// Total number of dequeue operations recorded.
    pub dequeue_count: u64,
    /// Number of times the buffer was completely full (back-pressure events).
    pub full_events: u64,
    /// Number of times the buffer was completely empty (starvation events).
    pub empty_events: u64,
}

impl BufferStats {
    /// Buffer utilisation as a fraction (0.0–1.0).
    ///
    /// Returns `None` when `capacity` is 0 (unknown).
    pub fn utilisation(&self) -> Option<f64> {
        if self.capacity == 0 {
            return None;
        }
        Some(self.current_occupancy as f64 / self.capacity as f64)
    }

    /// Update occupancy and maintain the peak.
    pub fn set_occupancy(&mut self, occupancy: u64) {
        self.current_occupancy = occupancy;
        if occupancy > self.peak_occupancy {
            self.peak_occupancy = occupancy;
        }
    }

    /// Record an enqueue event, incrementing occupancy by one.
    pub fn on_enqueue(&mut self) {
        self.enqueue_count += 1;
        self.set_occupancy(self.current_occupancy.saturating_add(1));
        if self.capacity > 0 && self.current_occupancy >= self.capacity {
            self.full_events += 1;
        }
    }

    /// Record a dequeue event, decrementing occupancy by one.
    pub fn on_dequeue(&mut self) {
        self.dequeue_count += 1;
        if self.current_occupancy == 0 {
            self.empty_events += 1;
        }
        self.current_occupancy = self.current_occupancy.saturating_sub(1);
    }

    /// Reset occupancy counters (peak is preserved).
    pub fn reset_occupancy(&mut self) {
        self.current_occupancy = 0;
    }
}

// ── NodeMetrics ───────────────────────────────────────────────────────────────

/// All metrics associated with a single pipeline node.
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    /// Node identifier.
    pub node_id: NodeId,
    /// Human-readable node name.
    pub node_name: String,
    /// Throughput statistics.
    pub throughput: ThroughputStats,
    /// Latency statistics.
    pub latency: LatencyStats,
    /// Buffer / queue occupancy statistics.
    pub buffer: BufferStats,
    /// Number of processing errors encountered by this node.
    pub error_count: u64,
    /// Whether this node is currently considered healthy.
    pub healthy: bool,
}

impl NodeMetrics {
    /// Create a new `NodeMetrics` entry for the given node.
    pub fn new(node_id: NodeId, node_name: impl Into<String>) -> Self {
        Self {
            node_id,
            node_name: node_name.into(),
            throughput: ThroughputStats::default(),
            latency: LatencyStats::default(),
            buffer: BufferStats::default(),
            error_count: 0,
            healthy: true,
        }
    }

    /// Record that `frames` frames were processed in the given `duration`.
    pub fn record_throughput(&mut self, frames: u64, duration: Duration) {
        self.throughput.total_frames += frames;
        self.throughput.sample_count += 1;
        self.throughput.cumulative_duration =
            self.throughput.cumulative_duration.saturating_add(duration);

        // Update peak fps
        let secs = duration.as_secs_f64();
        if secs > f64::EPSILON {
            let fps = frames as f64 / secs;
            if fps > self.throughput.peak_fps {
                self.throughput.peak_fps = fps;
            }
        }
    }

    /// Record a latency observation.
    pub fn record_latency_sample(&mut self, d: Duration) {
        self.latency.record(d);
    }

    /// Record a processing error.
    pub fn record_error(&mut self) {
        self.error_count += 1;
        if self.error_count >= 3 {
            self.healthy = false;
        }
    }
}

// ── PipelineMetrics ────────────────────────────────────────────────────────────

/// Collects and aggregates runtime metrics across all nodes in a pipeline.
///
/// # Thread safety
///
/// `PipelineMetrics` is `Send` but not `Sync`.  For concurrent access, wrap it
/// in a `Mutex<PipelineMetrics>`.
#[derive(Debug, Default)]
pub struct PipelineMetrics {
    /// Per-node metrics, keyed by `NodeId`.
    nodes: HashMap<NodeId, NodeMetrics>,
    /// Wall-clock instant when metrics collection started (approximate).
    start_instant: Option<std::time::Instant>,
    /// Global frame counter across all nodes.
    global_frames: u64,
    /// Global byte counter across all nodes.
    global_bytes: u64,
    /// Total number of errors recorded across all nodes.
    global_errors: u64,
}

impl PipelineMetrics {
    /// Create a new empty `PipelineMetrics` collector.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            start_instant: Some(std::time::Instant::now()),
            global_frames: 0,
            global_bytes: 0,
            global_errors: 0,
        }
    }

    /// Return a mutable reference to the `NodeMetrics` for `node_id`, creating
    /// a new entry if it does not yet exist.
    fn node_or_insert(&mut self, node_id: NodeId, node_name: &str) -> &mut NodeMetrics {
        self.nodes
            .entry(node_id)
            .or_insert_with(|| NodeMetrics::new(node_id, node_name))
    }

    /// Record that `frames` frames were processed by `node_id`.
    ///
    /// `node_name` is only used the first time a node is seen; subsequent calls
    /// for the same `node_id` may pass an empty string.
    pub fn record_frames(&mut self, node_id: NodeId, node_name: &str, frames: u64) {
        self.global_frames += frames;
        let node = self.node_or_insert(node_id, node_name);
        node.throughput.total_frames += frames;
        node.throughput.sample_count += 1;
    }

    /// Record that `bytes` bytes of media data were processed by `node_id`.
    pub fn record_bytes(&mut self, node_id: NodeId, node_name: &str, bytes: u64) {
        self.global_bytes += bytes;
        let node = self.node_or_insert(node_id, node_name);
        node.throughput.total_bytes += bytes;
    }

    /// Record a per-node latency observation.
    pub fn record_latency(&mut self, node_id: NodeId, node_name: &str, d: Duration) {
        let node = self.node_or_insert(node_id, node_name);
        node.latency.record(d);
    }

    /// Record a throughput sample (frames processed in a given duration).
    pub fn record_throughput_sample(
        &mut self,
        node_id: NodeId,
        node_name: &str,
        frames: u64,
        duration: Duration,
    ) {
        self.global_frames += frames;
        let node = self.node_or_insert(node_id, node_name);
        node.record_throughput(frames, duration);
    }

    /// Record a processing error for `node_id`.
    pub fn record_error(&mut self, node_id: NodeId, node_name: &str) {
        self.global_errors += 1;
        let node = self.node_or_insert(node_id, node_name);
        node.record_error();
    }

    /// Update the buffer occupancy for `node_id`.
    pub fn set_buffer_occupancy(&mut self, node_id: NodeId, node_name: &str, occupancy: u64) {
        let node = self.node_or_insert(node_id, node_name);
        node.buffer.set_occupancy(occupancy);
    }

    /// Configure the buffer capacity for `node_id`.
    pub fn set_buffer_capacity(&mut self, node_id: NodeId, node_name: &str, capacity: u64) {
        let node = self.node_or_insert(node_id, node_name);
        node.buffer.capacity = capacity;
    }

    /// Record a buffer enqueue event for `node_id`.
    pub fn on_buffer_enqueue(&mut self, node_id: NodeId, node_name: &str) {
        let node = self.node_or_insert(node_id, node_name);
        node.buffer.on_enqueue();
    }

    /// Record a buffer dequeue event for `node_id`.
    pub fn on_buffer_dequeue(&mut self, node_id: NodeId, node_name: &str) {
        let node = self.node_or_insert(node_id, node_name);
        node.buffer.on_dequeue();
    }

    /// Record that `count` frames were dropped (back-pressure / evictions) by `node_id`.
    pub fn record_dropped_frames(&mut self, node_id: NodeId, node_name: &str, count: u64) {
        let node = self.node_or_insert(node_id, node_name);
        node.throughput.dropped_frames += count;
    }

    /// Retrieve immutable `NodeMetrics` for the given `node_id`.
    ///
    /// Returns `None` if no data has been recorded for this node.
    pub fn node(&self, node_id: NodeId) -> Option<&NodeMetrics> {
        self.nodes.get(&node_id)
    }

    /// Retrieve mutable `NodeMetrics` for the given `node_id`.
    pub fn node_mut(&mut self, node_id: NodeId) -> Option<&mut NodeMetrics> {
        self.nodes.get_mut(&node_id)
    }

    /// Number of unique nodes tracked.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total frames recorded across all nodes.
    pub fn global_frames(&self) -> u64 {
        self.global_frames
    }

    /// Total bytes recorded across all nodes.
    pub fn global_bytes(&self) -> u64 {
        self.global_bytes
    }

    /// Total errors recorded across all nodes.
    pub fn global_errors(&self) -> u64 {
        self.global_errors
    }

    /// Wall-clock elapsed time since metrics collection started.
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_instant.map(|t| t.elapsed())
    }

    /// Return the node with the highest latency mean.
    ///
    /// Returns `None` if no latency samples have been collected for any node.
    pub fn slowest_node(&self) -> Option<&NodeMetrics> {
        self.nodes
            .values()
            .filter(|n| !n.latency.is_empty())
            .max_by_key(|n| n.latency.mean().unwrap_or(Duration::ZERO))
    }

    /// Return all nodes that have been marked unhealthy.
    pub fn unhealthy_nodes(&self) -> Vec<&NodeMetrics> {
        self.nodes.values().filter(|n| !n.healthy).collect()
    }

    /// Return an iterator over all tracked `NodeMetrics`, sorted by node name.
    pub fn nodes_sorted_by_name(&self) -> Vec<&NodeMetrics> {
        let mut v: Vec<&NodeMetrics> = self.nodes.values().collect();
        v.sort_by(|a, b| a.node_name.cmp(&b.node_name));
        v
    }

    /// Merge another `PipelineMetrics` into this one, combining counters.
    ///
    /// For each node seen in `other`:
    /// - Throughput counters and latency samples are accumulated.
    /// - Buffer peak occupancy is taken as the maximum.
    /// - Error counts are summed.
    pub fn merge(&mut self, other: PipelineMetrics) {
        self.global_frames += other.global_frames;
        self.global_bytes += other.global_bytes;
        self.global_errors += other.global_errors;

        for (id, other_node) in other.nodes {
            let entry = self
                .nodes
                .entry(id)
                .or_insert_with(|| NodeMetrics::new(id, &other_node.node_name));

            // Merge throughput
            entry.throughput.total_frames += other_node.throughput.total_frames;
            entry.throughput.total_bytes += other_node.throughput.total_bytes;
            entry.throughput.sample_count += other_node.throughput.sample_count;
            entry.throughput.cumulative_duration = entry
                .throughput
                .cumulative_duration
                .saturating_add(other_node.throughput.cumulative_duration);
            entry.throughput.dropped_frames += other_node.throughput.dropped_frames;
            if other_node.throughput.peak_fps > entry.throughput.peak_fps {
                entry.throughput.peak_fps = other_node.throughput.peak_fps;
            }

            // Merge latency samples
            for s in other_node.latency.samples {
                entry.latency.record(s);
            }

            // Merge buffer stats (take peak maximum)
            entry.buffer.enqueue_count += other_node.buffer.enqueue_count;
            entry.buffer.dequeue_count += other_node.buffer.dequeue_count;
            entry.buffer.full_events += other_node.buffer.full_events;
            entry.buffer.empty_events += other_node.buffer.empty_events;
            if other_node.buffer.peak_occupancy > entry.buffer.peak_occupancy {
                entry.buffer.peak_occupancy = other_node.buffer.peak_occupancy;
            }

            // Merge errors
            entry.error_count += other_node.error_count;
            if !other_node.healthy {
                entry.healthy = false;
            }
        }
    }

    /// Reset all metrics, clearing all samples and counters.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.global_frames = 0;
        self.global_bytes = 0;
        self.global_errors = 0;
        self.start_instant = Some(std::time::Instant::now());
    }

    /// Produce a human-readable summary suitable for logging.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("┌─ Pipeline Metrics Summary ────────────────────────────────────────\n");
        out.push_str(&format!(
            "│  Nodes: {}  |  Global frames: {}  |  Global bytes: {}  |  Errors: {}\n",
            self.node_count(),
            self.global_frames,
            self.global_bytes,
            self.global_errors
        ));
        if let Some(elapsed) = self.elapsed() {
            out.push_str(&format!("│  Elapsed: {:.2}s\n", elapsed.as_secs_f64()));
        }
        out.push_str("├───────────────────────────────────────────────────────────────────\n");
        out.push_str("│  Node Name            │ Frames │   Mean Lat  │  Drop%  │ Healthy\n");
        out.push_str("├───────────────────────────────────────────────────────────────────\n");

        for node in self.nodes_sorted_by_name() {
            let lat_str = node
                .latency
                .mean()
                .map(|d| format!("{:.1}ms", d.as_secs_f64() * 1000.0))
                .unwrap_or_else(|| "     —".to_string());
            let drop_pct = node.throughput.drop_rate() * 100.0;
            out.push_str(&format!(
                "│  {:<22} │ {:>6} │ {:>11} │ {:>6.1}% │ {}\n",
                truncate(&node.node_name, 22),
                node.throughput.total_frames,
                lat_str,
                drop_pct,
                if node.healthy { "✓" } else { "✗" },
            ));
        }
        out.push_str("└───────────────────────────────────────────────────────────────────\n");
        out
    }

    /// Return a `Result<&NodeMetrics, PipelineError>` for the given node.
    pub fn require_node(&self, node_id: NodeId) -> Result<&NodeMetrics, PipelineError> {
        self.nodes
            .get(&node_id)
            .ok_or_else(|| PipelineError::MetricsError(format!("node {node_id} not found")))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id() -> NodeId {
        NodeId::new()
    }

    // ── ThroughputStats ──────────────────────────────────────────────────────

    #[test]
    fn throughput_mean_fps_zero_duration() {
        let s = ThroughputStats::default();
        assert_eq!(s.mean_fps(), 0.0);
    }

    #[test]
    fn throughput_mean_fps_nonzero() {
        let mut s = ThroughputStats::default();
        s.total_frames = 60;
        s.cumulative_duration = Duration::from_secs(2);
        assert!((s.mean_fps() - 30.0).abs() < 0.001);
    }

    #[test]
    fn throughput_drop_rate() {
        let mut s = ThroughputStats::default();
        s.total_frames = 90;
        s.dropped_frames = 10;
        assert!((s.drop_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn throughput_drop_rate_zero_total() {
        let s = ThroughputStats::default();
        assert_eq!(s.drop_rate(), 0.0);
    }

    #[test]
    fn throughput_bytes_per_sec() {
        let mut s = ThroughputStats::default();
        s.total_bytes = 1_000_000;
        s.cumulative_duration = Duration::from_secs(1);
        assert!((s.mean_bytes_per_sec() - 1_000_000.0).abs() < 1.0);
    }

    // ── LatencyStats ─────────────────────────────────────────────────────────

    #[test]
    fn latency_record_and_mean() {
        let mut l = LatencyStats::default();
        l.record(Duration::from_millis(10));
        l.record(Duration::from_millis(20));
        l.record(Duration::from_millis(30));
        let mean = l.mean().expect("should have mean");
        assert_eq!(mean, Duration::from_millis(20));
    }

    #[test]
    fn latency_min_max() {
        let mut l = LatencyStats::default();
        l.record(Duration::from_millis(5));
        l.record(Duration::from_millis(50));
        assert_eq!(l.min, Duration::from_millis(5));
        assert_eq!(l.max, Duration::from_millis(50));
    }

    #[test]
    fn latency_median() {
        let mut l = LatencyStats::default();
        for i in 1u64..=9 {
            l.record(Duration::from_millis(i * 10));
        }
        let median = l.median().expect("should have median");
        // median of 10..90ms (9 elements) is 50ms
        assert_eq!(median, Duration::from_millis(50));
    }

    #[test]
    fn latency_percentiles() {
        let mut l = LatencyStats::default();
        for i in 1u64..=100 {
            l.record(Duration::from_millis(i));
        }
        let p95 = l.p95().expect("p95");
        let p99 = l.p99().expect("p99");
        assert!(p95 >= Duration::from_millis(94));
        assert!(p99 >= Duration::from_millis(98));
    }

    #[test]
    fn latency_empty_returns_none() {
        let l = LatencyStats::default();
        assert!(l.mean().is_none());
        assert!(l.median().is_none());
        assert!(l.p95().is_none());
        assert!(l.p99().is_none());
        assert!(l.is_empty());
    }

    // ── BufferStats ───────────────────────────────────────────────────────────

    #[test]
    fn buffer_utilisation() {
        let mut b = BufferStats::default();
        b.capacity = 100;
        b.set_occupancy(40);
        let util = b.utilisation().expect("should compute");
        assert!((util - 0.4).abs() < 0.001);
    }

    #[test]
    fn buffer_utilisation_unknown_capacity() {
        let b = BufferStats::default();
        assert!(b.utilisation().is_none());
    }

    #[test]
    fn buffer_peak_tracking() {
        let mut b = BufferStats::default();
        b.set_occupancy(10);
        b.set_occupancy(50);
        b.set_occupancy(30);
        assert_eq!(b.peak_occupancy, 50);
        assert_eq!(b.current_occupancy, 30);
    }

    #[test]
    fn buffer_enqueue_dequeue() {
        let mut b = BufferStats::default();
        b.capacity = 10;
        for _ in 0..5 {
            b.on_enqueue();
        }
        assert_eq!(b.current_occupancy, 5);
        assert_eq!(b.enqueue_count, 5);
        b.on_dequeue();
        assert_eq!(b.current_occupancy, 4);
        assert_eq!(b.dequeue_count, 1);
    }

    #[test]
    fn buffer_full_events_tracked() {
        let mut b = BufferStats::default();
        b.capacity = 2;
        b.on_enqueue(); // 1
        b.on_enqueue(); // 2 — full
        assert_eq!(b.full_events, 1);
    }

    #[test]
    fn buffer_empty_events_tracked() {
        let mut b = BufferStats::default();
        b.on_dequeue(); // was already 0 → empty event
        assert_eq!(b.empty_events, 1);
    }

    // ── NodeMetrics ───────────────────────────────────────────────────────────

    #[test]
    fn node_metrics_new_healthy() {
        let id = make_id();
        let n = NodeMetrics::new(id, "scale");
        assert!(n.healthy);
        assert_eq!(n.error_count, 0);
    }

    #[test]
    fn node_metrics_record_error_marks_unhealthy() {
        let id = make_id();
        let mut n = NodeMetrics::new(id, "scale");
        n.record_error();
        n.record_error();
        n.record_error();
        assert!(!n.healthy);
    }

    #[test]
    fn node_metrics_record_throughput() {
        let id = make_id();
        let mut n = NodeMetrics::new(id, "scale");
        n.record_throughput(30, Duration::from_secs(1));
        n.record_throughput(30, Duration::from_secs(1));
        assert_eq!(n.throughput.total_frames, 60);
        assert_eq!(n.throughput.sample_count, 2);
        assert!(n.throughput.peak_fps >= 30.0);
    }

    // ── PipelineMetrics ────────────────────────────────────────────────────────

    #[test]
    fn pipeline_metrics_record_frames() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_frames(id, "src", 10);
        m.record_frames(id, "src", 5);
        assert_eq!(m.global_frames(), 15);
        let node = m.node(id).expect("exists");
        assert_eq!(node.throughput.total_frames, 15);
    }

    #[test]
    fn pipeline_metrics_record_bytes() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_bytes(id, "encoder", 1_000_000);
        assert_eq!(m.global_bytes(), 1_000_000);
    }

    #[test]
    fn pipeline_metrics_record_latency() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_latency(id, "scale", Duration::from_millis(10));
        m.record_latency(id, "scale", Duration::from_millis(20));
        let node = m.node(id).expect("exists");
        assert_eq!(node.latency.count(), 2);
    }

    #[test]
    fn pipeline_metrics_record_error() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_error(id, "broken");
        m.record_error(id, "broken");
        m.record_error(id, "broken");
        assert_eq!(m.global_errors(), 3);
        let node = m.node(id).expect("exists");
        assert!(!node.healthy);
    }

    #[test]
    fn pipeline_metrics_buffer_operations() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.set_buffer_capacity(id, "queue", 100);
        m.set_buffer_occupancy(id, "queue", 40);
        let node = m.node(id).expect("exists");
        assert_eq!(node.buffer.capacity, 100);
        assert_eq!(node.buffer.current_occupancy, 40);
        let util = node.buffer.utilisation().expect("has util");
        assert!((util - 0.4).abs() < 0.001);
    }

    #[test]
    fn pipeline_metrics_node_not_found_returns_err() {
        let m = PipelineMetrics::new();
        let result = m.require_node(make_id());
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_metrics_slowest_node() {
        let mut m = PipelineMetrics::new();
        let fast = make_id();
        let slow = make_id();
        m.record_latency(fast, "hflip", Duration::from_millis(1));
        m.record_latency(slow, "scale", Duration::from_millis(50));
        let slowest = m.slowest_node().expect("should find");
        assert_eq!(slowest.node_name, "scale");
    }

    #[test]
    fn pipeline_metrics_unhealthy_nodes() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_error(id, "bad");
        m.record_error(id, "bad");
        m.record_error(id, "bad");
        let unhealthy = m.unhealthy_nodes();
        assert_eq!(unhealthy.len(), 1);
    }

    #[test]
    fn pipeline_metrics_nodes_sorted_by_name() {
        let mut m = PipelineMetrics::new();
        m.record_frames(make_id(), "zebra", 1);
        m.record_frames(make_id(), "apple", 1);
        m.record_frames(make_id(), "mango", 1);
        let names: Vec<&str> = m
            .nodes_sorted_by_name()
            .iter()
            .map(|n| n.node_name.as_str())
            .collect();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn pipeline_metrics_merge() {
        let mut m1 = PipelineMetrics::new();
        let id = make_id();
        m1.record_frames(id, "scale", 10);

        let mut m2 = PipelineMetrics::new();
        m2.record_frames(id, "scale", 20);
        m2.record_latency(id, "scale", Duration::from_millis(5));

        m1.merge(m2);
        assert_eq!(m1.global_frames(), 30);
        let node = m1.node(id).expect("exists");
        assert_eq!(node.throughput.total_frames, 30);
        assert_eq!(node.latency.count(), 1);
    }

    #[test]
    fn pipeline_metrics_reset() {
        let mut m = PipelineMetrics::new();
        m.record_frames(make_id(), "n", 100);
        m.reset();
        assert_eq!(m.global_frames(), 0);
        assert_eq!(m.node_count(), 0);
    }

    #[test]
    fn pipeline_metrics_format_summary() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_frames(id, "my_node", 100);
        m.record_latency(id, "my_node", Duration::from_millis(10));
        let summary = m.format_summary();
        assert!(summary.contains("Pipeline Metrics Summary"));
        assert!(summary.contains("my_node"));
    }

    #[test]
    fn pipeline_metrics_throughput_sample() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_throughput_sample(id, "enc", 30, Duration::from_secs(1));
        assert_eq!(m.global_frames(), 30);
        let node = m.node(id).expect("exists");
        assert!(node.throughput.peak_fps >= 30.0);
    }

    #[test]
    fn pipeline_metrics_dropped_frames() {
        let mut m = PipelineMetrics::new();
        let id = make_id();
        m.record_frames(id, "src", 100);
        m.record_dropped_frames(id, "src", 5);
        let node = m.node(id).expect("exists");
        assert_eq!(node.throughput.dropped_frames, 5);
        assert!((node.throughput.drop_rate() - 5.0 / 105.0).abs() < 0.001);
    }

    #[test]
    fn pipeline_metrics_elapsed_is_some() {
        let m = PipelineMetrics::new();
        assert!(m.elapsed().is_some());
    }
}
