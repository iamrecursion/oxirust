//! Critical path analysis for inference computation graphs.
//!
//! Computes the longest dependency chain in a DAG (the critical path) using:
//! 1. Kahn's algorithm for topological sort.
//! 2. DP over the topological order: `dist[v] = max_over_predecessors(dist[u] + cost(v))`.
//!
//! The result includes the path as a sequence of [`NodeId`]s, the total
//! accumulated latency in nanoseconds, and the single bottleneck node
//! (the one that sits at the end of the longest path).
//!
//! Nodes whose `latency_ns` is `None` are treated as 1 ns and a
//! [`MissingCostWarning`] is emitted for each one.
//!
//! Cycle detection: if Kahn's algorithm cannot drain all nodes a
//! [`CriticalPathError::CycleDetected`] is returned instead of panicking.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_infer::critical_path::{
//!     InferenceGraph, NodeLatency, critical_path,
//! };
//!
//! let mut g = InferenceGraph::default();
//! let a = g.add_node(NodeLatency { latency_ns: Some(10) });
//! let b = g.add_node(NodeLatency { latency_ns: Some(20) });
//! let c = g.add_node(NodeLatency { latency_ns: Some(5) });
//! g.add_edge(a, b).unwrap();
//! g.add_edge(b, c).unwrap();
//!
//! let result = critical_path(&g).unwrap();
//! assert_eq!(result.report.nodes, vec![a, b, c]);
//! assert_eq!(result.report.total_latency_ns, 35);
//! ```

use std::collections::VecDeque;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a node in an [`InferenceGraph`].
pub type NodeId = usize;

/// Per-node latency annotation.
///
/// When `latency_ns` is `None` the analysis falls back to 1 ns and emits a
/// [`MissingCostWarning`].
#[derive(Debug, Clone, Default)]
pub struct NodeLatency {
    /// Estimated execution latency in nanoseconds for this node.
    pub latency_ns: Option<u64>,
}

impl NodeLatency {
    /// Convenience constructor.
    pub fn new(latency_ns: u64) -> Self {
        Self {
            latency_ns: Some(latency_ns),
        }
    }
}

/// A lightweight Directed Acyclic Graph (DAG) of inference nodes.
///
/// Nodes are added in order and receive consecutive [`NodeId`]s starting from
/// zero.  Edges encode data-flow dependencies: edge `(from, to)` means "node
/// `from` must execute before node `to`".
#[derive(Debug, Clone, Default)]
pub struct InferenceGraph {
    /// Per-node latency annotations; index == [`NodeId`].
    pub nodes: Vec<NodeLatency>,
    /// Directed edges `(from, to)` — i.e. `from` → `to`.
    pub edges: Vec<(NodeId, NodeId)>,
}

impl InferenceGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node with the given latency annotation and return its [`NodeId`].
    pub fn add_node(&mut self, latency: NodeLatency) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(latency);
        id
    }

    /// Add a directed edge `from → to`.
    ///
    /// Returns [`CriticalPathError::InvalidNode`] if either node index is out
    /// of range.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<(), CriticalPathError> {
        let n = self.nodes.len();
        if from >= n {
            return Err(CriticalPathError::InvalidNode(from));
        }
        if to >= n {
            return Err(CriticalPathError::InvalidNode(to));
        }
        self.edges.push((from, to));
        Ok(())
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// The critical-path analysis report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalPathReport {
    /// Ordered sequence of [`NodeId`]s on the critical path, from source to
    /// sink (inclusive).  Empty when the graph has no nodes.
    pub nodes: Vec<NodeId>,
    /// Sum of node latencies along the critical path, in nanoseconds.
    pub total_latency_ns: u64,
    /// The node with the highest individual latency on the critical path.
    /// Zero when `nodes` is empty.
    pub bottleneck: NodeId,
}

/// Warning emitted when a node has no latency annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingCostWarning {
    /// The node whose `latency_ns` was `None`.
    pub node_id: NodeId,
}

/// Combined result: analysis report plus any latency-annotation warnings.
#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    pub report: CriticalPathReport,
    pub warnings: Vec<MissingCostWarning>,
}

/// Errors returned by [`critical_path`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CriticalPathError {
    /// The graph contains a directed cycle; critical-path analysis requires a
    /// DAG.  The enclosed string names the nodes that could not be processed.
    #[error("Cycle detected; these nodes were not reachable via topological sort: {0}")]
    CycleDetected(String),

    /// An edge references a node index that does not exist in the graph.
    #[error("Edge references out-of-range node id {0}")]
    InvalidNode(NodeId),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the critical path of `graph`.
///
/// Returns `Ok(CriticalPathResult)` for valid DAGs, or
/// `Err(CriticalPathError::CycleDetected)` if the graph is cyclic.
///
/// # Algorithm
///
/// 1. Build forward adjacency list and reverse adjacency list (predecessor map)
///    together with in-degree counts.
/// 2. Run Kahn's BFS topological sort.  If any nodes are left unprocessed,
///    a cycle exists.
/// 3. DP over topo order: `dist[v] = cost(v) + max(dist[u] for u in pred(v))`.
///    Track the predecessor that achieved the maximum for path reconstruction.
/// 4. The node with the maximum `dist` value is the end of the critical path.
/// 5. Walk predecessors back to reconstruct the path, then reverse it.
pub fn critical_path(graph: &InferenceGraph) -> Result<CriticalPathResult, CriticalPathError> {
    let n = graph.num_nodes();

    // Empty graph — trivial result.
    if n == 0 {
        return Ok(CriticalPathResult {
            report: CriticalPathReport {
                nodes: vec![],
                total_latency_ns: 0,
                bottleneck: 0,
            },
            warnings: vec![],
        });
    }

    // ── 1. Build adjacency structures ──────────────────────────────────────
    //
    // `succ[u]` = list of nodes that depend on u (successors).
    // `pred[v]` = list of nodes that v depends on (predecessors).
    // `in_degree[v]` = number of predecessors.

    let mut succ: Vec<Vec<NodeId>> = vec![vec![]; n];
    let mut pred: Vec<Vec<NodeId>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for &(from, to) in &graph.edges {
        // Edge validity was checked at add_edge time, but edges could have
        // been added to the raw field directly; guard anyway.
        if from >= n || to >= n {
            return Err(CriticalPathError::InvalidNode(if from >= n {
                from
            } else {
                to
            }));
        }
        succ[from].push(to);
        pred[to].push(from);
        in_degree[to] += 1;
    }

    // ── 2. Collect per-node costs; emit warnings for missing annotations ───

    let mut warnings: Vec<MissingCostWarning> = vec![];
    let costs: Vec<u64> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(id, nl)| {
            nl.latency_ns.unwrap_or_else(|| {
                warnings.push(MissingCostWarning { node_id: id });
                1
            })
        })
        .collect();

    // ── 3. Kahn's BFS topological sort ─────────────────────────────────────

    let mut queue: VecDeque<NodeId> = VecDeque::new();
    for v in 0..n {
        if in_degree[v] == 0 {
            queue.push_back(v);
        }
    }

    let mut topo_order: Vec<NodeId> = Vec::with_capacity(n);
    // Work on a mutable copy of in-degrees so we can decrement during BFS.
    let mut remaining_in: Vec<usize> = in_degree.clone();

    while let Some(u) = queue.pop_front() {
        topo_order.push(u);
        for &v in &succ[u] {
            remaining_in[v] -= 1;
            if remaining_in[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if topo_order.len() != n {
        // Some nodes were not processed — there is a cycle.
        let cyclic: Vec<String> = (0..n)
            .filter(|&v| !topo_order.contains(&v))
            .map(|v| v.to_string())
            .collect();
        return Err(CriticalPathError::CycleDetected(cyclic.join(", ")));
    }

    // ── 4. DP: longest path in topo order ──────────────────────────────────
    //
    // `dist[v]` = maximum accumulated latency of any path ending at `v`
    //             (including v's own cost).
    // `best_pred[v]` = the predecessor that achieved `dist[v]`, or `None` for
    //                  source nodes.

    let mut dist: Vec<u64> = vec![0; n];
    let mut best_pred: Vec<Option<NodeId>> = vec![None; n];

    for &v in &topo_order {
        // Start with just this node's own cost.
        dist[v] = costs[v];
        best_pred[v] = None;

        // Extend the longest predecessor path.
        for &u in &pred[v] {
            let candidate = dist[u].saturating_add(costs[v]);
            if candidate > dist[v] {
                dist[v] = candidate;
                best_pred[v] = Some(u);
            }
        }
    }

    // ── 5. Find the sink with the maximum distance ─────────────────────────

    let (end_node, &max_dist) = dist
        .iter()
        .enumerate()
        .max_by_key(|&(_, d)| d)
        .unwrap_or((0, &0)); // Safety: n > 0 so the iterator is non-empty.

    // ── 6. Reconstruct the path by walking back through best_pred ──────────

    let mut path: Vec<NodeId> = vec![];
    let mut current = end_node;
    loop {
        path.push(current);
        match best_pred[current] {
            Some(prev) => current = prev,
            None => break,
        }
    }
    path.reverse();

    // ── 7. Identify bottleneck: node on path with highest individual cost ──

    let bottleneck = path
        .iter()
        .copied()
        .max_by_key(|&v| costs[v])
        .unwrap_or(end_node);

    Ok(CriticalPathResult {
        report: CriticalPathReport {
            nodes: path,
            total_latency_ns: max_dist,
            bottleneck,
        },
        warnings,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a graph from a node-latency list and an edge list.
    fn build_graph(latencies: &[Option<u64>], edges: &[(usize, usize)]) -> InferenceGraph {
        let mut g = InferenceGraph::new();
        for &lat in latencies {
            g.add_node(NodeLatency { latency_ns: lat });
        }
        for &(from, to) in edges {
            g.add_edge(from, to).expect("valid edge");
        }
        g
    }

    // ── Test 1: linear chain A→B→C ────────────────────────────────────────

    #[test]
    fn test_linear_chain() {
        // A(10) → B(20) → C(5)  — only one path, total = 35
        let g = build_graph(&[Some(10), Some(20), Some(5)], &[(0, 1), (1, 2)]);
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.nodes, vec![0, 1, 2]);
        assert_eq!(res.report.total_latency_ns, 35);
        assert_eq!(res.report.bottleneck, 1); // node 1 has cost 20
        assert!(res.warnings.is_empty());
    }

    // ── Test 2: diamond — longer branch wins ─────────────────────────────

    #[test]
    fn test_diamond_longer_branch_wins() {
        // A(1) → B(100) → D(1)
        //      → C(1)   → D
        // Critical path: A→B→D, total = 102
        let g = build_graph(
            &[Some(1), Some(100), Some(1), Some(1)],
            &[(0, 1), (0, 2), (1, 3), (2, 3)],
        );
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.nodes, vec![0, 1, 3]);
        assert_eq!(res.report.total_latency_ns, 102);
        assert_eq!(res.report.bottleneck, 1); // node 1 has cost 100
        assert!(res.warnings.is_empty());
    }

    // ── Test 3: single node ───────────────────────────────────────────────

    #[test]
    fn test_single_node() {
        let g = build_graph(&[Some(42)], &[]);
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.nodes, vec![0]);
        assert_eq!(res.report.total_latency_ns, 42);
        assert_eq!(res.report.bottleneck, 0);
        assert!(res.warnings.is_empty());
    }

    // ── Test 4: missing latency annotations emit warnings ─────────────────

    #[test]
    fn test_missing_latency_warning() {
        // A(None) → B(None) → C(None)
        // Each falls back to 1 ns → total = 3, warnings for all three.
        let g = build_graph(&[None, None, None], &[(0, 1), (1, 2)]);
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.total_latency_ns, 3);
        assert_eq!(res.warnings.len(), 3);
        let warned_ids: Vec<NodeId> = res.warnings.iter().map(|w| w.node_id).collect();
        assert!(warned_ids.contains(&0));
        assert!(warned_ids.contains(&1));
        assert!(warned_ids.contains(&2));
    }

    // ── Test 5: empty graph ───────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = InferenceGraph::new();
        let res = critical_path(&g).expect("no cycle");

        assert!(res.report.nodes.is_empty());
        assert_eq!(res.report.total_latency_ns, 0);
        assert_eq!(res.report.bottleneck, 0);
        assert!(res.warnings.is_empty());
    }

    // ── Test 6: cycle detection returns an error ──────────────────────────

    #[test]
    fn test_cycle_detected() {
        // A → B → C → A  (cycle)
        let g = build_graph(&[Some(1), Some(1), Some(1)], &[(0, 1), (1, 2), (2, 0)]);
        let err = critical_path(&g).expect_err("should detect cycle");
        matches!(err, CriticalPathError::CycleDetected(_));
    }

    // ── Test 7: parallel branches without shared sink ─────────────────────

    #[test]
    fn test_parallel_branches() {
        // Two independent chains: A(5)→B(10) and C(1)→D(3)
        // Longest path ends at B with dist 15.
        let g = build_graph(&[Some(5), Some(10), Some(1), Some(3)], &[(0, 1), (2, 3)]);
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.total_latency_ns, 15);
        assert_eq!(*res.report.nodes.last().expect("non-empty"), 1);
    }

    // ── Test 8: wide graph — fan-out then fan-in ──────────────────────────

    #[test]
    fn test_fan_out_fan_in() {
        // root(1) → mid0(2) → sink(1)
        //         → mid1(5) → sink
        //         → mid2(3) → sink
        // Longest: root→mid1→sink = 1+5+1 = 7
        let g = build_graph(
            &[Some(1), Some(2), Some(5), Some(3), Some(1)],
            &[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
        );
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.total_latency_ns, 7);
        assert_eq!(res.report.nodes, vec![0, 2, 4]);
        assert_eq!(res.report.bottleneck, 2); // cost 5
    }

    // ── Test 9: invalid edge returns error ────────────────────────────────

    #[test]
    fn test_invalid_edge() {
        let mut g = InferenceGraph::new();
        g.add_node(NodeLatency::new(10));
        let err = g.add_edge(0, 5).expect_err("node 5 does not exist");
        matches!(err, CriticalPathError::InvalidNode(5));
    }

    // ── Test 10: mixed latencies, partially annotated ─────────────────────

    #[test]
    fn test_mixed_latencies() {
        // A(10) → B(None=1 fallback) → C(50)
        // total = 10+1+50 = 61, 1 warning for B
        let g = build_graph(&[Some(10), None, Some(50)], &[(0, 1), (1, 2)]);
        let res = critical_path(&g).expect("no cycle");

        assert_eq!(res.report.total_latency_ns, 61);
        assert_eq!(res.report.bottleneck, 2); // C has cost 50
        assert_eq!(res.warnings.len(), 1);
        assert_eq!(res.warnings[0].node_id, 1);
    }
}
