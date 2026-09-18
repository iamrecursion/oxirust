//! Graph statistics and complexity analysis for the filter graph pipeline.
//!
//! Provides structural metrics for a processing graph: node count, edge count,
//! average degree, maximum depth, and complexity classification.
//!
//! Additionally provides [`LatencyHistogram`] and [`NodeLatencyStats`] for
//! per-node processing-time tracking using 32 logarithmic (power-of-two)
//! nanosecond buckets.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// A qualitative classification of graph complexity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphComplexity {
    /// Very few nodes and edges; trivial to schedule.
    Trivial,
    /// Moderate number of nodes; standard linear pipeline.
    Simple,
    /// Many nodes with branching; requires careful scheduling.
    Moderate,
    /// Dense graph with many cross-edges; potentially expensive.
    Complex,
}

impl GraphComplexity {
    /// Returns a human-readable description of the complexity level.
    pub fn description(&self) -> &'static str {
        match self {
            GraphComplexity::Trivial => "trivial (<=2 nodes)",
            GraphComplexity::Simple => "simple (3-8 nodes, linear)",
            GraphComplexity::Moderate => "moderate (9-24 nodes, branching)",
            GraphComplexity::Complex => "complex (>=25 nodes or dense edges)",
        }
    }

    /// Returns `true` if the graph may require non-trivial scheduling.
    pub fn requires_advanced_scheduling(&self) -> bool {
        matches!(self, GraphComplexity::Moderate | GraphComplexity::Complex)
    }
}

/// Structural statistics about a directed graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Number of nodes in the graph.
    node_count: usize,
    /// Number of directed edges.
    edge_count: usize,
    /// Maximum depth (longest path from any source node).
    max_depth: usize,
    /// Sum of all node in-degrees (equals `edge_count` for simple graphs).
    total_degree: usize,
}

impl GraphStats {
    /// Creates a `GraphStats` with the given raw values.
    pub fn new(node_count: usize, edge_count: usize, max_depth: usize) -> Self {
        Self {
            node_count,
            edge_count,
            max_depth,
            total_degree: edge_count,
        }
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Returns the average out-degree across all nodes.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_degree(&self) -> f64 {
        if self.node_count == 0 {
            return 0.0;
        }
        self.edge_count as f64 / self.node_count as f64
    }

    /// Returns the maximum depth (length of the longest source-to-sink path).
    pub fn depth(&self) -> usize {
        self.max_depth
    }

    /// Classifies the graph by complexity.
    pub fn complexity(&self) -> GraphComplexity {
        if self.node_count <= 2 {
            GraphComplexity::Trivial
        } else if self.node_count <= 8 && self.avg_degree() <= 1.5 {
            GraphComplexity::Simple
        } else if self.node_count < 25 {
            GraphComplexity::Moderate
        } else {
            GraphComplexity::Complex
        }
    }

    /// Returns `true` if the graph has no edges.
    pub fn is_disconnected(&self) -> bool {
        self.edge_count == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LatencyHistogram
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node latency histogram with 32 power-of-two nanosecond buckets.
///
/// Bucket `k` covers the range `[2^(k-1) ns, 2^k ns)` for `k >= 1`.
/// Bucket `0` covers `0 ns` exactly.
/// Bucket `31` is a catch-all for values >= 2^30 ns (~1.07 s).
///
/// All operations are O(1) for record and O(32) for percentile queries.
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// `buckets[k]` counts samples whose latency is in `[2^(k-1), 2^k)` ns.
    buckets: [u64; 32],
    /// Total number of recorded samples.
    total_samples: u64,
    /// Sum of all recorded latency values in nanoseconds.
    sum_ns: u64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    /// Creates a new, empty histogram.
    pub fn new() -> Self {
        Self {
            buckets: [0u64; 32],
            total_samples: 0,
            sum_ns: 0,
        }
    }

    /// Records a latency sample in nanoseconds.
    pub fn record(&mut self, latency_ns: u64) {
        self.total_samples += 1;
        self.sum_ns = self.sum_ns.saturating_add(latency_ns);
        let bucket = Self::bucket_for(latency_ns);
        self.buckets[bucket] += 1;
    }

    /// Returns the mean latency in nanoseconds, or `0.0` if no samples recorded.
    pub fn mean_ns(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.sum_ns as f64 / self.total_samples as f64
    }

    /// Returns the approximate latency at the `pct`-th percentile (0.0-100.0).
    ///
    /// Returns the lower bound of the bucket containing the `pct`-th sample.
    pub fn percentile_ns(&self, pct: f64) -> u64 {
        if self.total_samples == 0 {
            return 0;
        }
        let target = ((pct / 100.0) * self.total_samples as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (k, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Self::bucket_lower_bound(k);
            }
        }
        Self::bucket_lower_bound(31)
    }

    /// Returns the total number of recorded samples.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Maps a latency value to a bucket index 0-31.
    fn bucket_for(latency_ns: u64) -> usize {
        if latency_ns == 0 {
            return 0;
        }
        let log2 = (u64::BITS - latency_ns.leading_zeros()) as usize;
        log2.min(31)
    }

    /// Returns the lower bound (in ns) of bucket `k`.
    fn bucket_lower_bound(k: usize) -> u64 {
        if k == 0 {
            0
        } else {
            1u64 << (k - 1)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeLatencyStats
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node latency statistics keyed by node identifier string.
///
/// Wraps a `HashMap` of [`LatencyHistogram`]s so callers can record and
/// query the processing latency of individual graph nodes.
#[derive(Debug, Clone, Default)]
pub struct NodeLatencyStats {
    /// Maps node identifier to latency histogram for that node.
    pub histograms: HashMap<String, LatencyHistogram>,
}

impl NodeLatencyStats {
    /// Creates an empty `NodeLatencyStats`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a latency sample for the node identified by `node_id`.
    ///
    /// The histogram is created on the first call for each unique `node_id`.
    pub fn record_node(&mut self, node_id: &str, latency_ns: u64) {
        self.histograms
            .entry(node_id.to_string())
            .or_default()
            .record(latency_ns);
    }

    /// Returns `(mean_ns, p50_ns, p99_ns)` for `node_id`, or `None` if no
    /// samples have been recorded for that node.
    pub fn summary(&self, node_id: &str) -> Option<(f64, u64, u64)> {
        let hist = self.histograms.get(node_id)?;
        if hist.total_samples() == 0 {
            return None;
        }
        Some((
            hist.mean_ns(),
            hist.percentile_ns(50.0),
            hist.percentile_ns(99.0),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Analyzes a directed acyclic graph represented as an adjacency list.
///
/// Nodes are represented by `u64` IDs; edges as `(from, to)` pairs.
#[derive(Debug, Clone, Default)]
pub struct GraphAnalyzer {
    /// Adjacency list: node -> list of successor nodes.
    adjacency: HashMap<u64, Vec<u64>>,
}

impl GraphAnalyzer {
    /// Creates an empty `GraphAnalyzer`.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Adds a node with no edges (idempotent).
    pub fn add_node(&mut self, id: u64) {
        self.adjacency.entry(id).or_default();
    }

    /// Adds a directed edge from `from` to `to`, implicitly adding both nodes.
    pub fn add_edge(&mut self, from: u64, to: u64) {
        self.adjacency.entry(from).or_default().push(to);
        self.adjacency.entry(to).or_default();
    }

    /// Computes the maximum depth via BFS from all source nodes (in-degree 0).
    fn max_depth(&self) -> usize {
        let mut in_degree: HashMap<u64, usize> = self.adjacency.keys().map(|&k| (k, 0)).collect();
        for successors in self.adjacency.values() {
            for &succ in successors {
                *in_degree.entry(succ).or_insert(0) += 1;
            }
        }

        let mut depth_map: HashMap<u64, usize> = HashMap::new();
        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter_map(|(&n, &d)| if d == 0 { Some(n) } else { None })
            .collect();

        for &n in &queue {
            depth_map.insert(n, 0);
        }

        let mut max_d = 0;
        while let Some(node) = queue.pop_front() {
            let cur_depth = *depth_map.get(&node).unwrap_or(&0);
            if let Some(succs) = self.adjacency.get(&node) {
                for &succ in succs {
                    let new_depth = cur_depth + 1;
                    let entry = depth_map.entry(succ).or_insert(0);
                    if new_depth > *entry {
                        *entry = new_depth;
                        if new_depth > max_d {
                            max_d = new_depth;
                        }
                        queue.push_back(succ);
                    }
                }
            }
        }
        max_d
    }

    /// Returns the set of all unique node IDs.
    pub fn nodes(&self) -> HashSet<u64> {
        let mut nodes: HashSet<u64> = self.adjacency.keys().copied().collect();
        for succs in self.adjacency.values() {
            for &s in succs {
                nodes.insert(s);
            }
        }
        nodes
    }

    /// Returns the total number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }

    /// Computes and returns a `GraphStats` snapshot.
    pub fn analyze(&self) -> GraphStats {
        let node_count = self.nodes().len();
        let edge_count = self.edge_count();
        let max_depth = self.max_depth();
        GraphStats::new(node_count, edge_count, max_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_description_trivial() {
        assert!(GraphComplexity::Trivial.description().contains("trivial"));
    }

    #[test]
    fn test_complexity_requires_advanced_trivial() {
        assert!(!GraphComplexity::Trivial.requires_advanced_scheduling());
    }

    #[test]
    fn test_complexity_requires_advanced_complex() {
        assert!(GraphComplexity::Complex.requires_advanced_scheduling());
    }

    #[test]
    fn test_graph_stats_avg_degree_zero_nodes() {
        let s = GraphStats::new(0, 0, 0);
        assert_eq!(s.avg_degree(), 0.0);
    }

    #[test]
    fn test_graph_stats_avg_degree() {
        let s = GraphStats::new(4, 4, 3);
        assert!((s.avg_degree() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_graph_stats_complexity_trivial() {
        let s = GraphStats::new(2, 1, 1);
        assert_eq!(s.complexity(), GraphComplexity::Trivial);
    }

    #[test]
    fn test_graph_stats_complexity_simple() {
        let s = GraphStats::new(5, 4, 4);
        assert_eq!(s.complexity(), GraphComplexity::Simple);
    }

    #[test]
    fn test_graph_stats_complexity_moderate() {
        let s = GraphStats::new(10, 12, 5);
        assert_eq!(s.complexity(), GraphComplexity::Moderate);
    }

    #[test]
    fn test_graph_stats_complexity_complex() {
        let s = GraphStats::new(30, 60, 10);
        assert_eq!(s.complexity(), GraphComplexity::Complex);
    }

    #[test]
    fn test_graph_stats_is_disconnected() {
        let s = GraphStats::new(3, 0, 0);
        assert!(s.is_disconnected());
    }

    #[test]
    fn test_analyzer_empty_graph() {
        let a = GraphAnalyzer::new();
        let stats = a.analyze();
        assert_eq!(stats.node_count(), 0);
        assert_eq!(stats.edge_count(), 0);
        assert_eq!(stats.depth(), 0);
    }

    #[test]
    fn test_analyzer_linear_chain() {
        let mut a = GraphAnalyzer::new();
        a.add_edge(0, 1);
        a.add_edge(1, 2);
        a.add_edge(2, 3);
        let stats = a.analyze();
        assert_eq!(stats.node_count(), 4);
        assert_eq!(stats.edge_count(), 3);
        assert_eq!(stats.depth(), 3);
    }

    #[test]
    fn test_analyzer_branching_graph() {
        let mut a = GraphAnalyzer::new();
        a.add_edge(0, 1);
        a.add_edge(0, 2);
        a.add_edge(1, 3);
        a.add_edge(2, 3);
        let stats = a.analyze();
        assert_eq!(stats.node_count(), 4);
        assert_eq!(stats.edge_count(), 4);
        assert_eq!(stats.depth(), 2);
    }

    #[test]
    fn test_analyzer_isolated_nodes() {
        let mut a = GraphAnalyzer::new();
        a.add_node(0);
        a.add_node(1);
        let stats = a.analyze();
        assert_eq!(stats.node_count(), 2);
        assert_eq!(stats.edge_count(), 0);
        assert_eq!(stats.depth(), 0);
    }

    #[test]
    fn test_complexity_moderate_requires_advanced() {
        assert!(GraphComplexity::Moderate.requires_advanced_scheduling());
    }

    // ── LatencyHistogram ──────────────────────────────────────────────────────

    #[test]
    fn test_histogram_record() {
        let mut h = LatencyHistogram::new();
        for i in 1u64..=100 {
            h.record(i * 1_000);
        }
        assert_eq!(h.total_samples(), 100);
        let mean = h.mean_ns();
        assert!(
            mean > 40_000.0 && mean < 60_000.0,
            "mean {mean:.0} ns outside expected range"
        );
    }

    #[test]
    fn test_histogram_percentile() {
        let mut h = LatencyHistogram::new();
        for i in 0u64..100 {
            h.record(i * 1_000);
        }
        let p50 = h.percentile_ns(50.0);
        let p99 = h.percentile_ns(99.0);
        assert!(
            p50 >= 16_384 && p50 <= 65_536,
            "p50={p50} outside expected bucket range"
        );
        assert!(p99 >= 32_768, "p99={p99} should be in high bucket");
    }

    #[test]
    fn test_node_latency_stats() {
        let mut stats = NodeLatencyStats::new();
        for i in 0u64..50 {
            stats.record_node("decoder", (i + 1) * 1_000);
            stats.record_node("scaler", (i + 1) * 500);
            stats.record_node("encoder", (i + 1) * 10_000);
        }
        let dec = stats
            .summary("decoder")
            .expect("decoder summary must be present");
        let scl = stats
            .summary("scaler")
            .expect("scaler summary must be present");
        let enc = stats
            .summary("encoder")
            .expect("encoder summary must be present");
        assert!(dec.0 > 0.0, "decoder mean must be positive");
        assert!(scl.0 < dec.0, "scaler should be faster than decoder");
        assert!(enc.0 > dec.0, "encoder should be slower than decoder");
        let (_, p50, p99) = dec;
        assert!(p99 >= p50, "p99 must be >= p50");
        assert!(stats.summary("nonexistent").is_none());
    }
}
