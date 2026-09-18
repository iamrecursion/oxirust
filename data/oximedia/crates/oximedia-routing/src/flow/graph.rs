//! Signal flow graph for visualizing and managing audio routing.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node type in the signal flow graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    /// Input source
    Input { label: String, channels: u8 },
    /// Output destination
    Output { label: String, channels: u8 },
    /// Processing node (mixer, effect, etc.)
    Processor {
        label: String,
        processor_type: String,
    },
    /// Bus/submix
    Bus { label: String, channels: u8 },
}

/// Edge representing signal flow connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEdge {
    /// Gain applied to this connection (in dB)
    pub gain_db: f32,
    /// Number of channels in this connection
    pub channels: u8,
    /// Whether this connection is active
    pub active: bool,
}

impl Default for FlowEdge {
    fn default() -> Self {
        Self {
            gain_db: 0.0,
            channels: 2,
            active: true,
        }
    }
}

/// Signal flow graph
#[derive(Debug, Clone)]
pub struct SignalFlowGraph {
    /// The underlying directed graph
    graph: DiGraph<NodeType, FlowEdge>,
    /// Map of node labels to indices for quick lookup
    node_map: HashMap<String, NodeIndex>,
}

impl Default for SignalFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalFlowGraph {
    /// Create a new signal flow graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add an input node
    pub fn add_input(&mut self, label: String, channels: u8) -> NodeIndex {
        let node = NodeType::Input {
            label: label.clone(),
            channels,
        };
        let index = self.graph.add_node(node);
        self.node_map.insert(label, index);
        index
    }

    /// Add an output node
    pub fn add_output(&mut self, label: String, channels: u8) -> NodeIndex {
        let node = NodeType::Output {
            label: label.clone(),
            channels,
        };
        let index = self.graph.add_node(node);
        self.node_map.insert(label, index);
        index
    }

    /// Add a processor node
    pub fn add_processor(&mut self, label: String, processor_type: String) -> NodeIndex {
        let node = NodeType::Processor {
            label: label.clone(),
            processor_type,
        };
        let index = self.graph.add_node(node);
        self.node_map.insert(label, index);
        index
    }

    /// Add a bus node
    pub fn add_bus(&mut self, label: String, channels: u8) -> NodeIndex {
        let node = NodeType::Bus {
            label: label.clone(),
            channels,
        };
        let index = self.graph.add_node(node);
        self.node_map.insert(label, index);
        index
    }

    /// Connect two nodes
    pub fn connect(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        edge: FlowEdge,
    ) -> Result<(), FlowError> {
        // Check if nodes exist
        if self.graph.node_weight(from).is_none() || self.graph.node_weight(to).is_none() {
            return Err(FlowError::NodeNotFound);
        }

        self.graph.add_edge(from, to, edge);
        Ok(())
    }

    /// Disconnect two nodes
    pub fn disconnect(&mut self, from: NodeIndex, to: NodeIndex) -> Result<(), FlowError> {
        if let Some(edge) = self.graph.find_edge(from, to) {
            self.graph.remove_edge(edge);
            Ok(())
        } else {
            Err(FlowError::EdgeNotFound)
        }
    }

    /// Get node by label
    #[must_use]
    pub fn get_node_by_label(&self, label: &str) -> Option<NodeIndex> {
        self.node_map.get(label).copied()
    }

    /// Get node type
    #[must_use]
    pub fn get_node_type(&self, node: NodeIndex) -> Option<&NodeType> {
        self.graph.node_weight(node)
    }

    /// Get all inputs to a node
    #[must_use]
    pub fn get_inputs(&self, node: NodeIndex) -> Vec<(NodeIndex, &FlowEdge)> {
        self.graph
            .edges_directed(node, Direction::Incoming)
            .map(|edge| (edge.source(), edge.weight()))
            .collect()
    }

    /// Get all outputs from a node
    #[must_use]
    pub fn get_outputs(&self, node: NodeIndex) -> Vec<(NodeIndex, &FlowEdge)> {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| (edge.target(), edge.weight()))
            .collect()
    }

    /// Get all input nodes
    #[must_use]
    pub fn get_all_inputs(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| matches!(self.graph[idx], NodeType::Input { .. }))
            .collect()
    }

    /// Get all output nodes
    #[must_use]
    pub fn get_all_outputs(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| matches!(self.graph[idx], NodeType::Output { .. }))
            .collect()
    }

    /// Get total node count
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get total edge count
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if there's a path from source to destination
    #[must_use]
    pub fn has_path(&self, from: NodeIndex, to: NodeIndex) -> bool {
        petgraph::algo::has_path_connecting(&self.graph, from, to, None)
    }

    /// Get the underlying graph reference
    #[must_use]
    pub const fn graph(&self) -> &DiGraph<NodeType, FlowEdge> {
        &self.graph
    }

    /// Remove a node and all its connections
    pub fn remove_node(&mut self, node: NodeIndex) -> Result<NodeType, FlowError> {
        if let Some(node_type) = self.graph.remove_node(node) {
            // Remove from node map
            self.node_map.retain(|_, &mut idx| idx != node);
            Ok(node_type)
        } else {
            Err(FlowError::NodeNotFound)
        }
    }

    /// Clear the entire graph
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_map.clear();
    }

    /// Get all nodes
    #[must_use]
    pub fn get_all_nodes(&self) -> Vec<(NodeIndex, &NodeType)> {
        self.graph
            .node_indices()
            .filter_map(|idx| self.graph.node_weight(idx).map(|n| (idx, n)))
            .collect()
    }
}

/// Errors that can occur in signal flow operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum FlowError {
    /// Node not found
    #[error("Node not found in graph")]
    NodeNotFound,
    /// Edge not found
    #[error("Edge not found in graph")]
    EdgeNotFound,
    /// Cycle detected
    #[error("Cycle detected in signal flow")]
    CycleDetected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = SignalFlowGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Mic 1".to_string(), 1);
        let output = graph.add_output("Monitor".to_string(), 2);

        assert_eq!(graph.node_count(), 2);
        assert!(graph.get_node_type(input).is_some());
        assert!(graph.get_node_type(output).is_some());
    }

    #[test]
    fn test_connect_nodes() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Input".to_string(), 2);
        let output = graph.add_output("Output".to_string(), 2);

        let edge = FlowEdge::default();
        graph
            .connect(input, output, edge)
            .expect("should succeed in test");

        assert_eq!(graph.edge_count(), 1);
        assert!(graph.has_path(input, output));
    }

    #[test]
    fn test_disconnect_nodes() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Input".to_string(), 2);
        let output = graph.add_output("Output".to_string(), 2);

        graph
            .connect(input, output, FlowEdge::default())
            .expect("should succeed in test");
        assert_eq!(graph.edge_count(), 1);

        graph
            .disconnect(input, output)
            .expect("should succeed in test");
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_get_by_label() {
        let mut graph = SignalFlowGraph::new();
        graph.add_input("Test Input".to_string(), 2);

        let node = graph.get_node_by_label("Test Input");
        assert!(node.is_some());

        let not_found = graph.get_node_by_label("Not Exist");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_get_inputs_outputs() {
        let mut graph = SignalFlowGraph::new();

        let in1 = graph.add_input("In1".to_string(), 1);
        let in2 = graph.add_input("In2".to_string(), 1);
        let bus = graph.add_bus("Mix".to_string(), 2);
        let out1 = graph.add_output("Out".to_string(), 2);

        graph
            .connect(in1, bus, FlowEdge::default())
            .expect("should succeed in test");
        graph
            .connect(in2, bus, FlowEdge::default())
            .expect("should succeed in test");
        graph
            .connect(bus, out1, FlowEdge::default())
            .expect("should succeed in test");

        let inputs = graph.get_inputs(bus);
        assert_eq!(inputs.len(), 2);

        let outputs = graph.get_outputs(bus);
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = SignalFlowGraph::new();
        let input = graph.add_input("Input".to_string(), 2);

        assert_eq!(graph.node_count(), 1);

        graph.remove_node(input).expect("should succeed in test");
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_clear_graph() {
        let mut graph = SignalFlowGraph::new();

        graph.add_input("In1".to_string(), 1);
        graph.add_output("Out1".to_string(), 2);

        assert!(graph.node_count() > 0);

        graph.clear();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_processor_node() {
        let mut graph = SignalFlowGraph::new();
        let proc = graph.add_processor("EQ".to_string(), "Equalizer".to_string());

        if let Some(NodeType::Processor {
            label,
            processor_type,
        }) = graph.get_node_type(proc)
        {
            assert_eq!(label, "EQ");
            assert_eq!(processor_type, "Equalizer");
        } else {
            panic!("Expected processor node");
        }
    }
}
