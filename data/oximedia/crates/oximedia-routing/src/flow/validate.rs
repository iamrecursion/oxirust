//! Flow validation and loop detection for signal flow graphs.

use super::graph::{NodeType, SignalFlowGraph};
use petgraph::algo::is_cyclic_directed;
use petgraph::graph::NodeIndex;
use petgraph::visit::DfsPostOrder;
use std::collections::HashSet;

/// Result of flow validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the flow is valid
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<ValidationError>,
    /// List of validation warnings
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a new validation result
    #[must_use]
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Check if there are any issues
    #[must_use]
    pub fn has_issues(&self) -> bool {
        !self.errors.is_empty() || !self.warnings.is_empty()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Feedback loop detected
    FeedbackLoop { nodes: Vec<NodeIndex> },
    /// Disconnected input
    DisconnectedInput { node: NodeIndex },
    /// Disconnected output
    DisconnectedOutput { node: NodeIndex },
    /// Channel count mismatch
    ChannelMismatch {
        from: NodeIndex,
        to: NodeIndex,
        from_channels: u8,
        to_channels: u8,
    },
}

/// Validation warning types
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// Inactive connection
    InactiveConnection { from: NodeIndex, to: NodeIndex },
    /// Excessive gain
    ExcessiveGain {
        from: NodeIndex,
        to: NodeIndex,
        gain_db: f32,
    },
    /// Unused processor
    UnusedProcessor { node: NodeIndex },
}

impl SignalFlowGraph {
    /// Validate the signal flow graph
    #[must_use]
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check for feedback loops
        if is_cyclic_directed(self.graph()) {
            result.add_error(ValidationError::FeedbackLoop {
                nodes: self.find_cycle_nodes(),
            });
        }

        // Check for disconnected inputs/outputs
        for node in self.graph().node_indices() {
            if let Some(node_type) = self.get_node_type(node) {
                match node_type {
                    NodeType::Input { .. } => {
                        if self.get_outputs(node).is_empty() {
                            result.add_error(ValidationError::DisconnectedInput { node });
                        }
                    }
                    NodeType::Output { .. } => {
                        if self.get_inputs(node).is_empty() {
                            result.add_error(ValidationError::DisconnectedOutput { node });
                        }
                    }
                    NodeType::Processor { .. } => {
                        if self.get_inputs(node).is_empty() && self.get_outputs(node).is_empty() {
                            result.add_warning(ValidationWarning::UnusedProcessor { node });
                        }
                    }
                    NodeType::Bus { .. } => {}
                }
            }
        }

        // Check for channel mismatches and other edge-related issues
        for node in self.graph().node_indices() {
            for (target, edge) in self.get_outputs(node) {
                // Check channel compatibility
                if let (Some(from_type), Some(to_type)) =
                    (self.get_node_type(node), self.get_node_type(target))
                {
                    let from_channels = get_node_channels(from_type);
                    let to_channels = get_node_channels(to_type);

                    if let (Some(from_ch), Some(to_ch)) = (from_channels, to_channels) {
                        if from_ch > to_ch {
                            result.add_error(ValidationError::ChannelMismatch {
                                from: node,
                                to: target,
                                from_channels: from_ch,
                                to_channels: to_ch,
                            });
                        }
                    }
                }

                // Check for inactive connections
                if !edge.active {
                    result.add_warning(ValidationWarning::InactiveConnection {
                        from: node,
                        to: target,
                    });
                }

                // Check for excessive gain
                if edge.gain_db.abs() > 12.0 {
                    result.add_warning(ValidationWarning::ExcessiveGain {
                        from: node,
                        to: target,
                        gain_db: edge.gain_db,
                    });
                }
            }
        }

        result
    }

    /// Find nodes involved in cycles
    fn find_cycle_nodes(&self) -> Vec<NodeIndex> {
        let mut cycle_nodes = HashSet::new();

        for node in self.graph().node_indices() {
            // Check if there's a path back to this node
            for (target, _) in self.get_outputs(node) {
                if self.has_path(target, node) {
                    cycle_nodes.insert(node);
                    cycle_nodes.insert(target);
                }
            }
        }

        cycle_nodes.into_iter().collect()
    }

    /// Detect all feedback loops
    #[must_use]
    pub fn detect_feedback_loops(&self) -> Vec<Vec<NodeIndex>> {
        let mut loops = Vec::new();

        if !is_cyclic_directed(self.graph()) {
            return loops;
        }

        // Find strongly connected components
        let sccs = petgraph::algo::kosaraju_scc(self.graph());

        for scc in sccs {
            if scc.len() > 1 {
                loops.push(scc);
            }
        }

        loops
    }

    /// Check if adding an edge would create a loop
    #[must_use]
    pub fn would_create_loop(&self, from: NodeIndex, to: NodeIndex) -> bool {
        // If there's already a path from 'to' to 'from', adding edge from->to creates a loop
        self.has_path(to, from)
    }

    /// Get all unreachable nodes (not connected to any input)
    #[must_use]
    pub fn get_unreachable_nodes(&self) -> Vec<NodeIndex> {
        let mut reachable = HashSet::new();
        let inputs = self.get_all_inputs();

        // Mark all nodes reachable from inputs
        for &input in &inputs {
            let mut dfs = DfsPostOrder::new(self.graph(), input);
            while let Some(node) = dfs.next(self.graph()) {
                reachable.insert(node);
            }
        }

        // Find unreachable nodes
        self.graph()
            .node_indices()
            .filter(|node| !reachable.contains(node) && !inputs.contains(node))
            .collect()
    }

    /// Get all dead-end nodes (not connected to any output)
    #[must_use]
    pub fn get_dead_end_nodes(&self) -> Vec<NodeIndex> {
        let outputs = self.get_all_outputs();

        self.graph()
            .node_indices()
            .filter(|&node| {
                !outputs.contains(&node)
                    && !outputs.iter().any(|&output| self.has_path(node, output))
            })
            .collect()
    }
}

/// Helper function to get channel count from a node type
fn get_node_channels(node_type: &NodeType) -> Option<u8> {
    match node_type {
        NodeType::Input { channels, .. }
        | NodeType::Output { channels, .. }
        | NodeType::Bus { channels, .. } => Some(*channels),
        NodeType::Processor { .. } => None, // Processors may be flexible
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::graph::SignalFlowGraph;
    use crate::flow::FlowEdge;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid);
        assert!(!result.has_issues());

        result.add_error(ValidationError::FeedbackLoop { nodes: vec![] });
        assert!(!result.is_valid);
        assert!(result.has_issues());
    }

    #[test]
    fn test_valid_graph() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("In".to_string(), 2);
        let output = graph.add_output("Out".to_string(), 2);

        graph
            .connect(input, output, FlowEdge::default())
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(result.is_valid);
    }

    #[test]
    fn test_disconnected_input() {
        let mut graph = SignalFlowGraph::new();
        graph.add_input("Disconnected".to_string(), 2);

        let result = graph.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_disconnected_output() {
        let mut graph = SignalFlowGraph::new();
        graph.add_output("Disconnected".to_string(), 2);

        let result = graph.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_feedback_loop_detection() {
        let mut graph = SignalFlowGraph::new();

        let node1 = graph.add_bus("Bus1".to_string(), 2);
        let node2 = graph.add_bus("Bus2".to_string(), 2);

        graph
            .connect(node1, node2, FlowEdge::default())
            .expect("should succeed in test");
        graph
            .connect(node2, node1, FlowEdge::default())
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(!result.is_valid);

        let loops = graph.detect_feedback_loops();
        assert!(!loops.is_empty());
    }

    #[test]
    fn test_would_create_loop() {
        let mut graph = SignalFlowGraph::new();

        let node1 = graph.add_bus("Bus1".to_string(), 2);
        let node2 = graph.add_bus("Bus2".to_string(), 2);
        let node3 = graph.add_bus("Bus3".to_string(), 2);

        graph
            .connect(node1, node2, FlowEdge::default())
            .expect("should succeed in test");
        graph
            .connect(node2, node3, FlowEdge::default())
            .expect("should succeed in test");

        // Connecting node3 to node1 would create a loop
        assert!(graph.would_create_loop(node3, node1));

        // Connecting node3 to a new node would not
        let node4 = graph.add_bus("Bus4".to_string(), 2);
        assert!(!graph.would_create_loop(node3, node4));
    }

    #[test]
    fn test_channel_mismatch() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Stereo".to_string(), 2);
        let output = graph.add_output("Mono".to_string(), 1);

        graph
            .connect(input, output, FlowEdge::default())
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::ChannelMismatch { .. })));
    }

    #[test]
    fn test_inactive_warning() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("In".to_string(), 2);
        let output = graph.add_output("Out".to_string(), 2);

        let edge = FlowEdge {
            active: false,
            ..Default::default()
        };

        graph
            .connect(input, output, edge)
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_excessive_gain_warning() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("In".to_string(), 2);
        let output = graph.add_output("Out".to_string(), 2);

        let edge = FlowEdge {
            gain_db: 20.0, // Very high gain
            ..Default::default()
        };

        graph
            .connect(input, output, edge)
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_unreachable_nodes() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("In".to_string(), 2);
        let output1 = graph.add_output("Out1".to_string(), 2);
        let _output2 = graph.add_output("Out2".to_string(), 2);

        graph
            .connect(input, output1, FlowEdge::default())
            .expect("should succeed in test");
        // output2 is unreachable

        let unreachable = graph.get_unreachable_nodes();
        assert_eq!(unreachable.len(), 1);
    }
}
