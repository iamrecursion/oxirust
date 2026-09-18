//! Dynamic graph reconfiguration: hot-swap nodes without disrupting connections.
//!
//! Hot-swapping allows replacing a processing node with a compatible alternative
//! in O(1) time while preserving all upstream/downstream connections. Two nodes
//! are compatible for hot-swap if they share the same port signature — identical
//! port counts and compatible format families (`video/*`, `audio/*`, etc.).
//!
//! # Example
//!
//! ```
//! use oximedia_graph::processing_graph::{ProcessingGraph, GraphNode, NodeType};
//! use oximedia_graph::hot_swap::{HotSwappable, HotSwapResult, PortSignature};
//!
//! let mut graph = ProcessingGraph::new();
//! graph.add_node(GraphNode::new(1, "filter_a", NodeType::Filter));
//! graph.add_node(GraphNode::new(2, "source_1", NodeType::Source));
//! graph.add_node(GraphNode::new(3, "sink_1", NodeType::Sink));
//! graph.connect(2, 0, 1, 0);
//! graph.connect(1, 0, 3, 0);
//!
//! // Replace filter_a with a compatible filter_b.
//! let replacement = GraphNode::new(1, "filter_b", NodeType::Filter);
//! let result = graph.hot_swap_node(1, replacement);
//! assert_eq!(result, HotSwapResult::Success);
//! ```

use crate::processing_graph::{GraphNode, NodeType, ProcessingGraph};

// ── Port signature ────────────────────────────────────────────────────────────

/// Describes the port surface of a node for hot-swap compatibility checking.
///
/// Two nodes are swap-compatible when their input counts, output counts, and
/// format *families* all match. Format families are the coarse type prefix,
/// e.g. `"video"`, `"audio"`, or `"data"`.
#[derive(Debug, Clone, PartialEq)]
pub struct PortSignature {
    /// Number of logical input ports.
    pub num_inputs: usize,
    /// Number of logical output ports.
    pub num_outputs: usize,
    /// Format family tags for each input port (e.g. `"video"`, `"audio"`).
    pub input_formats: Vec<String>,
    /// Format family tags for each output port.
    pub output_formats: Vec<String>,
}

impl PortSignature {
    /// Returns `true` when `self` and `other` have the same port counts and
    /// compatible format families on every corresponding port.
    #[must_use]
    pub fn is_compatible_with(&self, other: &PortSignature) -> bool {
        if self.num_inputs != other.num_inputs || self.num_outputs != other.num_outputs {
            return false;
        }
        let inputs_ok = self
            .input_formats
            .iter()
            .zip(&other.input_formats)
            .all(|(a, b)| format_family(a) == format_family(b));
        let outputs_ok = self
            .output_formats
            .iter()
            .zip(&other.output_formats)
            .all(|(a, b)| format_family(a) == format_family(b));
        inputs_ok && outputs_ok
    }
}

/// Extracts the format *family* from a slash-delimited MIME-style tag.
///
/// `"video/yuv420"` → `"video"`, `"audio/f32"` → `"audio"`, `"data"` → `"data"`.
#[must_use]
pub fn format_family(fmt: &str) -> &str {
    // Split at the first `/`; if none, the whole string is the family.
    fmt.split('/').next().unwrap_or(fmt)
}

// ── HotSwappable trait ────────────────────────────────────────────────────────

/// Nodes that can participate in hot-swap must be able to report their port
/// signature.
pub trait HotSwappable {
    /// Returns the [`PortSignature`] describing this node's connection surface.
    fn port_signature(&self) -> PortSignature;
}

/// Derive a format-family tag from a [`NodeType`].
///
/// All node types in the current implementation carry video-family ports by
/// default (a media-processing graph). Mixer/Splitter nodes are also
/// video-family. This can be extended in the future when typed ports are
/// annotated on `GraphNode` itself.
fn default_format_tag(_node_type: &NodeType) -> &'static str {
    "video"
}

impl HotSwappable for GraphNode {
    fn port_signature(&self) -> PortSignature {
        let max_in = self.node_type.max_inputs();
        let max_out = self.node_type.max_outputs();
        let tag = default_format_tag(&self.node_type);

        PortSignature {
            num_inputs: max_in,
            num_outputs: max_out,
            input_formats: vec![tag.to_string(); max_in],
            output_formats: vec![tag.to_string(); max_out],
        }
    }
}

// ── HotSwapResult ─────────────────────────────────────────────────────────────

/// Outcome of a [`ProcessingGraph::hot_swap_node`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotSwapResult {
    /// The node was replaced successfully; all existing connections are intact.
    Success,
    /// The replacement node's port signature is incompatible with the current
    /// node's signature. The `reason` field contains a human-readable
    /// explanation.
    IncompatiblePorts {
        /// Explanation of why the signatures do not match.
        reason: String,
    },
    /// No node with the given ID exists in the graph.
    NodeNotFound,
    /// The graph is currently executing; the swap cannot be performed safely.
    GraphLocked,
}

// ── ProcessingGraph hot-swap extension ───────────────────────────────────────

impl ProcessingGraph {
    /// Replace the node identified by `node_id` with `replacement`.
    ///
    /// All edges connected to `node_id` are preserved — only the node's
    /// internal data (name, params, type) is swapped. The swap is refused if:
    ///
    /// * `node_id` is not present in the graph → [`HotSwapResult::NodeNotFound`]
    /// * the graph is locked (executing) → [`HotSwapResult::GraphLocked`]
    /// * the port signatures are incompatible → [`HotSwapResult::IncompatiblePorts`]
    ///
    /// When [`HotSwapResult::Success`] is returned the replacement node's `id`
    /// field is forced to `node_id` so that all edge references remain valid.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the number of nodes (linear scan to locate the node
    /// slot). Edge preservation is trivially O(1) because edges reference node
    /// IDs, not positions; no edge data is modified.
    pub fn hot_swap_node(&mut self, node_id: u64, mut replacement: GraphNode) -> HotSwapResult {
        // ── 1. Graph-locked guard ──────────────────────────────────────────
        if self.is_locked {
            return HotSwapResult::GraphLocked;
        }

        // ── 2. Look up existing node ───────────────────────────────────────
        let existing_pos = match self.nodes.iter().position(|n| n.id == node_id) {
            Some(pos) => pos,
            None => return HotSwapResult::NodeNotFound,
        };

        let existing = &self.nodes[existing_pos];

        // ── 3. Port-signature compatibility check ──────────────────────────
        let old_sig = existing.port_signature();
        let new_sig = replacement.port_signature();

        if !old_sig.is_compatible_with(&new_sig) {
            let reason = format!(
                "port mismatch: existing node has {} input(s) / {} output(s), \
                 replacement has {} input(s) / {} output(s)",
                old_sig.num_inputs, old_sig.num_outputs, new_sig.num_inputs, new_sig.num_outputs,
            );
            return HotSwapResult::IncompatiblePorts { reason };
        }

        // ── 4. Perform the swap ────────────────────────────────────────────
        // Force the replacement's ID to match `node_id` so all edge
        // references (which store the numeric node ID) remain valid.
        replacement.id = node_id;
        self.nodes[existing_pos] = replacement;

        HotSwapResult::Success
    }

    /// Lock the graph to simulate an executing state (prevents hot-swap).
    ///
    /// Call [`ProcessingGraph::unlock`] when execution completes.
    pub fn lock(&mut self) {
        self.is_locked = true;
    }

    /// Unlock the graph after execution completes.
    pub fn unlock(&mut self) {
        self.is_locked = false;
    }

    /// Returns `true` if the graph is currently locked (executing).
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing_graph::{GraphEdge, GraphNode, NodeType, ProcessingGraph};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn build_linear_graph() -> (ProcessingGraph, u64, u64, u64) {
        // source(1) → filter(2) → sink(3)
        let mut g = ProcessingGraph::new();
        g.add_node(GraphNode::new(1, "source", NodeType::Source));
        g.add_node(GraphNode::new(2, "filter", NodeType::Filter));
        g.add_node(GraphNode::new(3, "sink", NodeType::Sink));
        g.connect(1, 0, 2, 0);
        g.connect(2, 0, 3, 0);
        (g, 1, 2, 3)
    }

    // ── test_hot_swap_compatible_nodes ────────────────────────────────────────

    /// Swap a Filter node with another Filter node (same port signature).
    /// After the swap the graph structure must remain intact and execution
    /// order must be preserved.
    #[test]
    fn test_hot_swap_compatible_nodes() {
        let (mut graph, source_id, filter_id, sink_id) = build_linear_graph();

        let replacement = GraphNode::new(filter_id, "filter_v2", NodeType::Filter);
        let result = graph.hot_swap_node(filter_id, replacement);

        assert_eq!(result, HotSwapResult::Success);

        // The swapped node must carry the replacement's name.
        let swapped = graph.nodes.iter().find(|n| n.id == filter_id);
        assert!(swapped.is_some(), "node must still exist after swap");
        assert_eq!(swapped.expect("checked above").name, "filter_v2");

        // Edges must be intact.
        assert_eq!(graph.edges.len(), 2);
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from_node == source_id && e.to_node == filter_id));
        assert!(graph
            .edges
            .iter()
            .any(|e| e.from_node == filter_id && e.to_node == sink_id));

        // Topological execution order must still be [1, 2, 3].
        let order = graph.execution_order();
        assert_eq!(order, vec![source_id, filter_id, sink_id]);
    }

    // ── test_hot_swap_incompatible_ports ──────────────────────────────────────

    /// Attempting to swap a 1-input Filter with a 0-input Source must fail with
    /// IncompatiblePorts.
    #[test]
    fn test_hot_swap_incompatible_ports() {
        let (mut graph, _source_id, filter_id, _sink_id) = build_linear_graph();

        // Source has 0 inputs / 1 output; Filter has 1 input / 1 output.
        let bad_replacement = GraphNode::new(filter_id, "source_impostor", NodeType::Source);
        let result = graph.hot_swap_node(filter_id, bad_replacement);

        match result {
            HotSwapResult::IncompatiblePorts { reason } => {
                // Reason should mention the mismatch.
                assert!(
                    reason.contains("input"),
                    "reason should mention input mismatch, got: {reason}"
                );
            }
            other => panic!("expected IncompatiblePorts, got {other:?}"),
        }
    }

    // ── test_hot_swap_node_not_found ──────────────────────────────────────────

    /// Swapping a non-existent node ID must return NodeNotFound.
    #[test]
    fn test_hot_swap_node_not_found() {
        let (mut graph, _s, _f, _k) = build_linear_graph();

        let ghost = GraphNode::new(99, "ghost", NodeType::Filter);
        let result = graph.hot_swap_node(99, ghost);

        assert_eq!(result, HotSwapResult::NodeNotFound);
    }

    // ── test_hot_swap_preserves_connections ───────────────────────────────────

    /// After a successful swap the edges referencing the swapped node must
    /// still be valid and the graph must execute in the correct order.
    #[test]
    fn test_hot_swap_preserves_connections() {
        let (mut graph, source_id, filter_id, sink_id) = build_linear_graph();

        // Swap filter(2) with another filter; the edge set must not change.
        let edges_before: Vec<GraphEdge> = graph.edges.clone();
        let replacement = GraphNode::new(filter_id, "optimised_filter", NodeType::Filter);
        let result = graph.hot_swap_node(filter_id, replacement);
        assert_eq!(result, HotSwapResult::Success);

        // Edge set must be identical in both count and content.
        assert_eq!(graph.edges.len(), edges_before.len());
        for edge in &edges_before {
            assert!(
                graph.edges.contains(edge),
                "edge {edge:?} must still be present after hot-swap"
            );
        }

        // Execution order must still route source → filter → sink.
        let order = graph.execution_order();
        let source_pos = order
            .iter()
            .position(|&id| id == source_id)
            .expect("source in order");
        let filter_pos = order
            .iter()
            .position(|&id| id == filter_id)
            .expect("filter in order");
        let sink_pos = order
            .iter()
            .position(|&id| id == sink_id)
            .expect("sink in order");
        assert!(source_pos < filter_pos, "source must precede filter");
        assert!(filter_pos < sink_pos, "filter must precede sink");
    }

    // ── test_hot_swap_graph_locked ────────────────────────────────────────────

    /// While the graph is locked (executing), hot-swap must be refused.
    #[test]
    fn test_hot_swap_graph_locked() {
        let (mut graph, _source_id, filter_id, _sink_id) = build_linear_graph();

        graph.lock();
        let replacement = GraphNode::new(filter_id, "filter_during_exec", NodeType::Filter);
        let result = graph.hot_swap_node(filter_id, replacement);
        assert_eq!(result, HotSwapResult::GraphLocked);

        // After unlocking the swap must succeed.
        graph.unlock();
        let replacement2 = GraphNode::new(filter_id, "filter_after_unlock", NodeType::Filter);
        let result2 = graph.hot_swap_node(filter_id, replacement2);
        assert_eq!(result2, HotSwapResult::Success);
    }

    // ── Port-signature unit tests ─────────────────────────────────────────────

    #[test]
    fn test_port_signature_compatible_same_type() {
        let filter_a = GraphNode::new(1, "a", NodeType::Filter);
        let filter_b = GraphNode::new(2, "b", NodeType::Filter);
        assert!(filter_a
            .port_signature()
            .is_compatible_with(&filter_b.port_signature()));
    }

    #[test]
    fn test_port_signature_incompatible_different_types() {
        let source = GraphNode::new(1, "src", NodeType::Source);
        let filter = GraphNode::new(2, "flt", NodeType::Filter);
        // Source: 0 inputs / 1 output; Filter: 1 input / 1 output → incompatible.
        assert!(!source
            .port_signature()
            .is_compatible_with(&filter.port_signature()));
    }

    #[test]
    fn test_format_family_extracts_prefix() {
        assert_eq!(format_family("video/yuv420"), "video");
        assert_eq!(format_family("audio/f32"), "audio");
        assert_eq!(format_family("data"), "data");
        assert_eq!(format_family("video"), "video");
    }

    #[test]
    fn test_hot_swap_id_is_normalised() {
        // Even if the replacement carries a different numeric ID, after a
        // successful swap the node must have the target ID.
        let (mut graph, _s, filter_id, _k) = build_linear_graph();

        // Deliberately give a different ID to the replacement.
        let replacement = GraphNode::new(999, "filter_new_id", NodeType::Filter);
        let result = graph.hot_swap_node(filter_id, replacement);
        assert_eq!(result, HotSwapResult::Success);

        let node = graph.nodes.iter().find(|n| n.id == filter_id);
        assert!(node.is_some(), "node must be found by the original ID");
        assert_eq!(node.expect("checked above").name, "filter_new_id");

        // ID 999 must NOT appear anywhere in the node list.
        assert!(!graph.nodes.iter().any(|n| n.id == 999));
    }
}
