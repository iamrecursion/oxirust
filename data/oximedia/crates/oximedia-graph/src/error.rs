//! Error types for the filter graph.

use thiserror::Error;

use crate::node::NodeId;
use crate::port::PortId;

/// Result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

/// Errors that can occur in filter graph operations.
#[derive(Error, Debug)]
pub enum GraphError {
    /// Node not found in the graph.
    #[error("node not found: {0:?}")]
    NodeNotFound(NodeId),

    /// Port not found on node.
    #[error("port not found: node {node:?}, port {port:?}")]
    PortNotFound {
        /// The node ID.
        node: NodeId,
        /// The port ID.
        port: PortId,
    },

    /// Connection already exists.
    #[error(
        "connection already exists from {from_node:?}:{from_port:?} to {to_node:?}:{to_port:?}"
    )]
    ConnectionExists {
        /// Source node ID.
        from_node: NodeId,
        /// Source port ID.
        from_port: PortId,
        /// Destination node ID.
        to_node: NodeId,
        /// Destination port ID.
        to_port: PortId,
    },

    /// Incompatible port formats.
    #[error("incompatible formats: source {source_format}, destination {dest_format}")]
    IncompatibleFormats {
        /// Source format description.
        source_format: String,
        /// Destination format description.
        dest_format: String,
    },

    /// Cycle detected in the graph.
    #[error("cycle detected in graph involving node {0:?}")]
    CycleDetected(NodeId),

    /// Graph is not configured properly.
    #[error("graph configuration error: {0}")]
    ConfigurationError(String),

    /// Node processing error.
    #[error("processing error in node {node:?}: {message}")]
    ProcessingError {
        /// The node where processing failed.
        node: NodeId,
        /// Error message.
        message: String,
    },

    /// End of stream reached.
    #[error("end of stream")]
    EndOfStream,

    /// No data available (non-blocking).
    #[error("no data available, try again")]
    WouldBlock,

    /// Invalid node state transition.
    #[error("invalid state transition for node {node:?}: from {from} to {to}")]
    InvalidStateTransition {
        /// The node ID.
        node: NodeId,
        /// Current state name.
        from: String,
        /// Attempted state name.
        to: String,
    },

    /// Graph is empty.
    #[error("graph has no nodes")]
    EmptyGraph,

    /// No source nodes found.
    #[error("graph has no source nodes")]
    NoSourceNodes,

    /// No sink nodes found.
    #[error("graph has no sink nodes")]
    NoSinkNodes,

    /// Port type mismatch.
    #[error("port type mismatch: expected {expected}, got {actual}")]
    PortTypeMismatch {
        /// Expected port type.
        expected: String,
        /// Actual port type.
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GraphError::NodeNotFound(NodeId(42));
        assert!(err.to_string().contains("42"));

        let err = GraphError::CycleDetected(NodeId(1));
        assert!(err.to_string().contains("cycle"));

        let err = GraphError::EmptyGraph;
        assert!(err.to_string().contains("no nodes"));
    }

    #[test]
    fn test_port_not_found_error() {
        let err = GraphError::PortNotFound {
            node: NodeId(1),
            port: PortId(2),
        };
        let msg = err.to_string();
        assert!(msg.contains("1"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_incompatible_formats_error() {
        let err = GraphError::IncompatibleFormats {
            source_format: "video/yuv420p".to_string(),
            dest_format: "audio/pcm".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("video"));
        assert!(msg.contains("audio"));
    }
}
