//! Node types for the filter graph.
//!
//! Nodes are the processing units in a filter graph. Each node has input and output
//! ports and implements the [`Node`] trait to process frames.

use std::collections::HashMap;
use std::fmt;

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::port::{InputPort, OutputPort, PortId, PortType};

/// Unique identifier for a node in the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Type of node in the filter graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    /// Source node that produces frames (e.g., decoder).
    Source,
    /// Filter node that transforms frames.
    Filter,
    /// Sink node that consumes frames (e.g., encoder, display).
    Sink,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source => write!(f, "Source"),
            Self::Filter => write!(f, "Filter"),
            Self::Sink => write!(f, "Sink"),
        }
    }
}

/// State of a node during graph execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NodeState {
    /// Node is idle and ready to process.
    #[default]
    Idle,
    /// Node is currently processing.
    Processing,
    /// Node has finished processing (end of stream).
    Done,
    /// Node encountered an error.
    Error,
}

impl fmt::Display for NodeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Processing => write!(f, "Processing"),
            Self::Done => write!(f, "Done"),
            Self::Error => write!(f, "Error"),
        }
    }
}

impl NodeState {
    /// Check if the node can transition to the given state.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub fn can_transition_to(&self, new_state: Self) -> bool {
        match (self, new_state) {
            // From Idle - can transition to any state
            (Self::Idle, Self::Processing | Self::Done | Self::Error) => true,
            // From Processing - can transition to any state
            (Self::Processing, Self::Idle | Self::Done | Self::Error) => true,
            // From Done or Error - can only reset to Idle
            (Self::Done | Self::Error, Self::Idle) => true,
            // Same state is always ok
            (a, b) if *a == b => true,
            _ => false,
        }
    }
}

/// Configuration for a node.
#[derive(Clone, Debug, Default)]
pub struct NodeConfig {
    /// Human-readable name for the node.
    pub name: String,
    /// Type of the node.
    pub node_type: Option<NodeType>,
    /// Custom configuration options.
    pub options: HashMap<String, ConfigValue>,
}

impl NodeConfig {
    /// Create a new node configuration with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_type: None,
            options: HashMap::new(),
        }
    }

    /// Set the node type.
    #[must_use]
    pub fn with_type(mut self, node_type: NodeType) -> Self {
        self.node_type = Some(node_type);
        self
    }

    /// Add a configuration option.
    #[must_use]
    pub fn with_option(mut self, key: impl Into<String>, value: ConfigValue) -> Self {
        self.options.insert(key.into(), value);
        self
    }

    /// Get a configuration option.
    #[must_use]
    pub fn get_option(&self, key: &str) -> Option<&ConfigValue> {
        self.options.get(key)
    }

    /// Get an integer option.
    #[must_use]
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.options.get(key).and_then(ConfigValue::as_int)
    }

    /// Get a float option.
    #[must_use]
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.options.get(key).and_then(ConfigValue::as_float)
    }

    /// Get a string option.
    #[must_use]
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.options.get(key).and_then(ConfigValue::as_string)
    }

    /// Get a boolean option.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.options.get(key).and_then(ConfigValue::as_bool)
    }
}

/// Configuration value type.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    /// Integer value.
    Int(i64),
    /// Floating point value.
    Float(f64),
    /// String value.
    String(String),
    /// Boolean value.
    Bool(bool),
}

impl ConfigValue {
    /// Get value as integer if possible.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get value as float if possible.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get value as string if possible.
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get value as boolean if possible.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl From<i64> for ConfigValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f64> for ConfigValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for ConfigValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for ConfigValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<bool> for ConfigValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// Trait for filter graph nodes.
///
/// Nodes implement this trait to participate in the filter graph.
/// The graph will call [`Node::process`] to push frames through the pipeline.
pub trait Node: Send + Sync {
    /// Get the node's unique identifier.
    fn id(&self) -> NodeId;

    /// Get the node's name.
    fn name(&self) -> &str;

    /// Get the node type.
    fn node_type(&self) -> NodeType;

    /// Get the current state of the node.
    fn state(&self) -> NodeState;

    /// Set the node state.
    fn set_state(&mut self, state: NodeState) -> GraphResult<()>;

    /// Get the node's input ports.
    fn inputs(&self) -> &[InputPort];

    /// Get the node's output ports.
    fn outputs(&self) -> &[OutputPort];

    /// Initialize the node before processing starts.
    fn initialize(&mut self) -> GraphResult<()> {
        Ok(())
    }

    /// Process available input and produce output.
    ///
    /// Returns `Ok(Some(frame))` if a frame was produced, `Ok(None)` if no
    /// frame is ready (need more input), or `Err` on error.
    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>>;

    /// Flush any buffered data.
    fn flush(&mut self) -> GraphResult<Vec<FilterFrame>> {
        Ok(Vec::new())
    }

    /// Reset the node to initial state.
    fn reset(&mut self) -> GraphResult<()> {
        self.set_state(NodeState::Idle)
    }

    /// Get input port by ID.
    fn input_port(&self, id: PortId) -> Option<&InputPort> {
        self.inputs().iter().find(|p| p.id == id)
    }

    /// Get output port by ID.
    fn output_port(&self, id: PortId) -> Option<&OutputPort> {
        self.outputs().iter().find(|p| p.id == id)
    }

    /// Check if node accepts the given port type as input.
    fn accepts_input(&self, port_type: PortType) -> bool {
        self.inputs().iter().any(|p| p.port_type == port_type)
    }

    /// Check if node produces the given port type as output.
    fn produces_output(&self, port_type: PortType) -> bool {
        self.outputs().iter().any(|p| p.port_type == port_type)
    }
}

/// Runtime state for a node in the graph.
#[allow(dead_code)]
pub struct NodeRuntime {
    /// The node implementation.
    node: Box<dyn Node>,
    /// Input buffers indexed by port ID.
    input_buffers: HashMap<PortId, Vec<FilterFrame>>,
    /// Output buffers indexed by port ID.
    output_buffers: HashMap<PortId, Vec<FilterFrame>>,
    /// Frames processed count.
    frames_processed: u64,
}

impl NodeRuntime {
    /// Create a new node runtime.
    pub fn new(node: Box<dyn Node>) -> Self {
        let input_buffers = node.inputs().iter().map(|p| (p.id, Vec::new())).collect();
        let output_buffers = node.outputs().iter().map(|p| (p.id, Vec::new())).collect();

        Self {
            node,
            input_buffers,
            output_buffers,
            frames_processed: 0,
        }
    }

    /// Get the underlying node.
    #[must_use]
    pub fn node(&self) -> &dyn Node {
        self.node.as_ref()
    }

    /// Get mutable reference to the underlying node.
    pub fn node_mut(&mut self) -> &mut dyn Node {
        self.node.as_mut()
    }

    /// Push a frame to an input port.
    pub fn push_input(&mut self, port: PortId, frame: FilterFrame) -> GraphResult<()> {
        self.input_buffers
            .get_mut(&port)
            .ok_or(GraphError::PortNotFound {
                node: self.node.id(),
                port,
            })?
            .push(frame);
        Ok(())
    }

    /// Pop a frame from an output port.
    pub fn pop_output(&mut self, port: PortId) -> GraphResult<Option<FilterFrame>> {
        let buffer = self
            .output_buffers
            .get_mut(&port)
            .ok_or(GraphError::PortNotFound {
                node: self.node.id(),
                port,
            })?;

        Ok(if buffer.is_empty() {
            None
        } else {
            Some(buffer.remove(0))
        })
    }

    /// Process the node.
    pub fn process(&mut self) -> GraphResult<()> {
        // Get input frame from first input port if available
        let input = self.input_buffers.values_mut().find_map(|buf| {
            if buf.is_empty() {
                None
            } else {
                Some(buf.remove(0))
            }
        });

        // Process
        if let Some(output) = self.node.process(input)? {
            // Push to first output port
            if let Some(buf) = self.output_buffers.values_mut().next() {
                buf.push(output);
            }
            self.frames_processed += 1;
        }

        Ok(())
    }

    /// Get the number of frames processed.
    #[must_use]
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_display() {
        let id = NodeId(42);
        assert_eq!(format!("{id}"), "Node(42)");
    }

    #[test]
    fn test_node_type_display() {
        assert_eq!(format!("{}", NodeType::Source), "Source");
        assert_eq!(format!("{}", NodeType::Filter), "Filter");
        assert_eq!(format!("{}", NodeType::Sink), "Sink");
    }

    #[test]
    fn test_node_state_transitions() {
        let state = NodeState::Idle;
        assert!(state.can_transition_to(NodeState::Processing));
        assert!(state.can_transition_to(NodeState::Done));
        assert!(state.can_transition_to(NodeState::Error));

        let state = NodeState::Processing;
        assert!(state.can_transition_to(NodeState::Idle));
        assert!(state.can_transition_to(NodeState::Done));

        let state = NodeState::Done;
        assert!(state.can_transition_to(NodeState::Idle));
        assert!(!state.can_transition_to(NodeState::Processing));
    }

    #[test]
    fn test_node_config() {
        let config = NodeConfig::new("test")
            .with_type(NodeType::Filter)
            .with_option("quality", ConfigValue::Int(80))
            .with_option("name", ConfigValue::String("test".into()));

        assert_eq!(config.name, "test");
        assert_eq!(config.node_type, Some(NodeType::Filter));
        assert_eq!(config.get_int("quality"), Some(80));
        assert_eq!(config.get_string("name"), Some("test"));
    }

    #[test]
    fn test_config_value_conversions() {
        let int_val = ConfigValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(int_val.as_string(), None);

        let float_val = ConfigValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_int(), None);

        let str_val = ConfigValue::String("hello".into());
        assert_eq!(str_val.as_string(), Some("hello"));

        let bool_val = ConfigValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));
    }

    #[test]
    fn test_config_value_from() {
        let _: ConfigValue = 42i64.into();
        let _: ConfigValue = 3.14f64.into();
        let _: ConfigValue = "test".into();
        let _: ConfigValue = String::from("test").into();
        let _: ConfigValue = true.into();
    }
}
