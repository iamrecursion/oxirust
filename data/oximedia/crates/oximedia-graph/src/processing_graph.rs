//! Media processing graph with nodes, edges, topological execution ordering,
//! and retry-with-exponential-backoff for transient node failures.
//!
//! This module models a directed acyclic graph (DAG) of media processing nodes.
//! Nodes represent processing stages (source, filter, encoder, etc.), and edges
//! represent data flow between them.

/// Classification of a processing node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    /// Produces media data (e.g. file reader, camera capture).
    Source,
    /// Decodes compressed media into raw frames.
    Decoder,
    /// Transforms media data (e.g. scaler, colour converter).
    Filter,
    /// Encodes raw frames into a compressed format.
    Encoder,
    /// Consumes media data (e.g. file writer, display).
    Sink,
    /// Combines multiple input streams into one.
    Mixer,
    /// Distributes one input stream to multiple outputs.
    Splitter,
}

impl NodeType {
    /// Maximum number of input connections accepted by this node type.
    pub fn max_inputs(&self) -> usize {
        match self {
            Self::Source => 0,
            Self::Decoder => 1,
            Self::Filter => 1,
            Self::Encoder => 1,
            Self::Sink => 1,
            Self::Mixer => 8,
            Self::Splitter => 1,
        }
    }

    /// Maximum number of output connections this node type can produce.
    pub fn max_outputs(&self) -> usize {
        match self {
            Self::Source => 1,
            Self::Decoder => 1,
            Self::Filter => 1,
            Self::Encoder => 1,
            Self::Sink => 0,
            Self::Mixer => 1,
            Self::Splitter => 8,
        }
    }
}

/// A single node in a media [`ProcessingGraph`].
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node within the graph.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Functional type of this node.
    pub node_type: NodeType,
    /// Whether this node should participate in processing.
    pub enabled: bool,
    /// Arbitrary key-value configuration parameters.
    pub params: Vec<(String, String)>,
}

impl GraphNode {
    /// Creates a new, enabled node with no parameters.
    pub fn new(id: u64, name: &str, node_type: NodeType) -> Self {
        Self {
            id,
            name: name.to_string(),
            node_type,
            enabled: true,
            params: Vec::new(),
        }
    }

    /// Returns the value for `key`, or `None` if not set.
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Sets (or updates) `key` to `value`.
    pub fn set_param(&mut self, key: &str, value: &str) {
        if let Some(entry) = self.params.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value.to_string();
        } else {
            self.params.push((key.to_string(), value.to_string()));
        }
    }
}

/// A directed connection between two ports on two nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphEdge {
    /// Source node identifier.
    pub from_node: u64,
    /// Output port index on the source node.
    pub from_port: u32,
    /// Destination node identifier.
    pub to_node: u64,
    /// Input port index on the destination node.
    pub to_port: u32,
}

impl GraphEdge {
    /// Returns `true` if this edge goes from `from` to `to`.
    pub fn connects(&self, from: u64, to: u64) -> bool {
        self.from_node == from && self.to_node == to
    }
}

/// A directed acyclic graph of media processing nodes.
#[derive(Debug, Default)]
pub struct ProcessingGraph {
    /// All nodes in the graph.
    pub nodes: Vec<GraphNode>,
    /// All edges in the graph.
    pub edges: Vec<GraphEdge>,
    /// When `true` the graph is considered executing and hot-swap is refused.
    pub is_locked: bool,
}

impl ProcessingGraph {
    /// Creates an empty processing graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `node` to the graph.  Duplicate IDs are allowed but discouraged.
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Removes the node with `id` and all edges referencing it.
    ///
    /// Returns `true` if a node was removed.
    pub fn remove_node(&mut self, id: u64) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|e| e.from_node != id && e.to_node != id);
        self.nodes.len() < before
    }

    /// Adds an edge from `(from, from_port)` to `(to, to_port)`.
    ///
    /// Returns `false` if either node does not exist; `true` on success.
    pub fn connect(&mut self, from: u64, from_port: u32, to: u64, to_port: u32) -> bool {
        let has_from = self.nodes.iter().any(|n| n.id == from);
        let has_to = self.nodes.iter().any(|n| n.id == to);
        if !has_from || !has_to {
            return false;
        }
        self.edges.push(GraphEdge {
            from_node: from,
            from_port,
            to_node: to,
            to_port,
        });
        true
    }

    /// Removes all edges from node `from` to node `to`.
    ///
    /// Returns `true` if at least one edge was removed.
    pub fn disconnect(&mut self, from: u64, to: u64) -> bool {
        let before = self.edges.len();
        self.edges.retain(|e| !e.connects(from, to));
        self.edges.len() < before
    }

    /// Returns references to all nodes whose type has zero inputs (source nodes).
    pub fn source_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .iter()
            .filter(|n| n.node_type.max_inputs() == 0)
            .collect()
    }

    /// Returns references to all nodes whose type has zero outputs (sink nodes).
    pub fn sink_nodes(&self) -> Vec<&GraphNode> {
        self.nodes
            .iter()
            .filter(|n| n.node_type.max_outputs() == 0)
            .collect()
    }

    /// Returns node IDs in topological execution order (Kahn's algorithm).
    ///
    /// Nodes not reachable from any source, or that form cycles, are appended
    /// in arbitrary order at the end.
    pub fn execution_order(&self) -> Vec<u64> {
        use std::collections::{HashMap, VecDeque};

        // Count incoming edges per node (enabled nodes only).
        let mut in_degree: HashMap<u64, usize> = self
            .nodes
            .iter()
            .filter(|n| n.enabled)
            .map(|n| (n.id, 0))
            .collect();

        for edge in &self.edges {
            if in_degree.contains_key(&edge.from_node) && in_degree.contains_key(&edge.to_node) {
                *in_degree.entry(edge.to_node).or_insert(0) += 1;
            }
        }

        // Seed the queue with zero-in-degree nodes.
        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort for determinism.
        let mut queue_vec: Vec<u64> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            order.push(id);
            // Find successors and decrement their in-degree.
            let mut new_ready: Vec<u64> = self
                .edges
                .iter()
                .filter(|e| e.from_node == id)
                .filter_map(|e| {
                    let deg = in_degree.get_mut(&e.to_node)?;
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        Some(e.to_node)
                    } else {
                        None
                    }
                })
                .collect();
            new_ready.sort_unstable();
            queue.extend(new_ready);
        }

        // Append any remaining nodes (disabled or cycle members) in id order.
        let mut remaining: Vec<u64> = self
            .nodes
            .iter()
            .map(|n| n.id)
            .filter(|id| !order.contains(id))
            .collect();
        remaining.sort_unstable();
        order.extend(remaining);

        order
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry policy for transient node failures
// ─────────────────────────────────────────────────────────────────────────────

/// Marker trait for errors that are considered *transient* (i.e. worth
/// retrying, such as a temporary resource contention or a momentary decode
/// stall).
///
/// Implement this on your custom error type and return it from a node's
/// process function to enable automatic retry via [`RetryPolicy`].
pub trait TransientError {
    /// Returns `true` when the error represents a transient condition.
    fn is_transient(&self) -> bool;
}

/// Policy controlling how many times a failing node operation should be
/// retried and how long to wait between attempts.
///
/// The inter-attempt sleep uses *exponential backoff*: attempt `n` (0-indexed)
/// sleeps for `backoff_ms * 2^n` milliseconds before being made.
///
/// # Example
/// ```
/// use oximedia_graph::processing_graph::RetryPolicy;
///
/// // Three attempts total, starting with a 10 ms back-off.
/// let policy = RetryPolicy { max_attempts: 3, backoff_ms: 10 };
/// ```
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).  A value of `1`
    /// means "try once; do not retry".
    pub max_attempts: u32,
    /// Base back-off in milliseconds.  The sleep before attempt `n` is
    /// `backoff_ms * 2^(n-1)` ms (so the first retry sleeps `backoff_ms` ms,
    /// the second sleeps `2 * backoff_ms` ms, and so on).
    pub backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_ms: 1,
        }
    }
}

impl RetryPolicy {
    /// Executes `f` up to `max_attempts` times, returning the first `Ok(T)`.
    ///
    /// If `f` returns an `Err(e)` where `e.is_transient()` returns `true` and
    /// there are remaining attempts, the function sleeps for an exponentially
    /// increasing duration and then calls `f` again.
    ///
    /// If `f` returns an `Err(e)` where `e.is_transient()` returns `false`,
    /// the error is returned immediately (no further retries are made).
    ///
    /// If all `max_attempts` are exhausted the last error is returned.
    pub fn execute<T, E, F>(&self, mut f: F) -> Result<T, E>
    where
        E: TransientError,
        F: FnMut() -> Result<T, E>,
    {
        let mut last_err: Option<E> = None;
        for attempt in 0..self.max_attempts {
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if !e.is_transient() {
                        return Err(e);
                    }
                    // Sleep with exponential back-off before the next attempt.
                    if attempt + 1 < self.max_attempts {
                        let sleep_ms = self.backoff_ms.saturating_mul(1u64 << attempt);
                        std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
                    }
                    last_err = Some(e);
                }
            }
        }
        // `max_attempts >= 1` and we enter at least one iteration, so
        // `last_err` is always `Some` here.
        Err(last_err.expect("at least one attempt must have set last_err"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: u64) -> GraphNode {
        GraphNode::new(id, &format!("source_{id}"), NodeType::Source)
    }
    fn filter(id: u64) -> GraphNode {
        GraphNode::new(id, &format!("filter_{id}"), NodeType::Filter)
    }
    fn sink(id: u64) -> GraphNode {
        GraphNode::new(id, &format!("sink_{id}"), NodeType::Sink)
    }

    // ── NodeType ─────────────────────────────────────────────────────────────

    #[test]
    fn source_has_zero_inputs() {
        assert_eq!(NodeType::Source.max_inputs(), 0);
    }

    #[test]
    fn sink_has_zero_outputs() {
        assert_eq!(NodeType::Sink.max_outputs(), 0);
    }

    #[test]
    fn mixer_accepts_multiple_inputs() {
        assert!(NodeType::Mixer.max_inputs() > 1);
    }

    #[test]
    fn splitter_produces_multiple_outputs() {
        assert!(NodeType::Splitter.max_outputs() > 1);
    }

    // ── GraphNode ────────────────────────────────────────────────────────────

    #[test]
    fn node_set_and_get_param() {
        let mut n = filter(1);
        n.set_param("width", "1920");
        assert_eq!(n.get_param("width"), Some("1920"));
    }

    #[test]
    fn node_update_existing_param() {
        let mut n = filter(2);
        n.set_param("fps", "24");
        n.set_param("fps", "60");
        assert_eq!(n.get_param("fps"), Some("60"));
        // Only one entry for the key.
        assert_eq!(n.params.iter().filter(|(k, _)| k == "fps").count(), 1);
    }

    #[test]
    fn node_missing_param_returns_none() {
        let n = source(3);
        assert!(n.get_param("nonexistent").is_none());
    }

    // ── GraphEdge ────────────────────────────────────────────────────────────

    #[test]
    fn edge_connects_returns_true_for_matching_pair() {
        let edge = GraphEdge {
            from_node: 1,
            from_port: 0,
            to_node: 2,
            to_port: 0,
        };
        assert!(edge.connects(1, 2));
    }

    #[test]
    fn edge_connects_returns_false_for_reversed_pair() {
        let edge = GraphEdge {
            from_node: 1,
            from_port: 0,
            to_node: 2,
            to_port: 0,
        };
        assert!(!edge.connects(2, 1));
    }

    // ── ProcessingGraph ───────────────────────────────────────────────────────

    #[test]
    fn add_and_remove_node() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(10));
        assert_eq!(g.nodes.len(), 1);
        assert!(g.remove_node(10));
        assert!(g.nodes.is_empty());
    }

    #[test]
    fn remove_node_also_removes_edges() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(sink(2));
        g.connect(1, 0, 2, 0);
        g.remove_node(1);
        assert!(g.edges.is_empty());
    }

    #[test]
    fn connect_fails_for_missing_node() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        assert!(!g.connect(1, 0, 99, 0)); // node 99 missing
    }

    #[test]
    fn disconnect_removes_all_matching_edges() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(sink(2));
        g.connect(1, 0, 2, 0);
        g.connect(1, 0, 2, 1);
        assert!(g.disconnect(1, 2));
        assert!(g.edges.is_empty());
    }

    #[test]
    fn source_nodes_returns_only_sources() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(filter(2));
        g.add_node(sink(3));
        let srcs: Vec<u64> = g.source_nodes().into_iter().map(|n| n.id).collect();
        assert_eq!(srcs, vec![1]);
    }

    #[test]
    fn sink_nodes_returns_only_sinks() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(sink(2));
        let sinks: Vec<u64> = g.sink_nodes().into_iter().map(|n| n.id).collect();
        assert_eq!(sinks, vec![2]);
    }

    #[test]
    fn execution_order_linear_pipeline() {
        // source(1) -> filter(2) -> sink(3)
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(filter(2));
        g.add_node(sink(3));
        g.connect(1, 0, 2, 0);
        g.connect(2, 0, 3, 0);
        let order = g.execution_order();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn execution_order_independent_nodes_are_included() {
        let mut g = ProcessingGraph::new();
        g.add_node(source(1));
        g.add_node(source(2));
        let order = g.execution_order();
        assert_eq!(order.len(), 2);
    }

    // ── RetryPolicy ───────────────────────────────────────────────────────────

    /// A simple error type used in retry tests.
    #[derive(Debug, PartialEq, Eq)]
    enum TestError {
        Transient(String),
        Fatal(String),
    }

    impl TransientError for TestError {
        fn is_transient(&self) -> bool {
            matches!(self, Self::Transient(_))
        }
    }

    #[test]
    fn test_retry_succeeds_on_second_attempt() {
        let call_count = std::cell::Cell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 3,
            backoff_ms: 0,
        };

        let result = policy.execute(|| {
            let n = call_count.get();
            call_count.set(n + 1);
            if n == 0 {
                // First call fails with a transient error.
                Err(TestError::Transient("temp".to_string()))
            } else {
                // Second call succeeds.
                Ok(42u32)
            }
        });

        assert_eq!(result, Ok(42));
        assert_eq!(call_count.get(), 2, "should have been called exactly twice");
    }

    #[test]
    fn test_retry_exhausted() {
        let call_count = std::cell::Cell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 3,
            backoff_ms: 0,
        };

        let result: Result<u32, TestError> = policy.execute(|| {
            call_count.set(call_count.get() + 1);
            Err(TestError::Transient("always fails".to_string()))
        });

        assert!(result.is_err(), "all attempts exhausted; must return Err");
        assert_eq!(
            call_count.get(),
            3,
            "execute must invoke f exactly max_attempts times"
        );
    }

    #[test]
    fn test_retry_fatal_error_no_retry() {
        let call_count = std::cell::Cell::new(0u32);
        let policy = RetryPolicy {
            max_attempts: 5,
            backoff_ms: 0,
        };

        let result: Result<u32, TestError> = policy.execute(|| {
            call_count.set(call_count.get() + 1);
            Err(TestError::Fatal("unrecoverable".to_string()))
        });

        assert!(result.is_err());
        // Fatal error on the very first call must prevent further retries.
        assert_eq!(
            call_count.get(),
            1,
            "fatal error must halt retries immediately"
        );
    }
}
