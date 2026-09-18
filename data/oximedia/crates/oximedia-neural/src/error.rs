//! Error types for oximedia-neural.

use thiserror::Error;

/// Errors produced by neural network operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NeuralError {
    /// The provided shape is invalid (e.g., zero-size dimension or shape/data length mismatch).
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    /// Two tensors have incompatible shapes for the requested operation.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// An index falls outside the valid range.
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// An operation received an empty input where non-empty data was required.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// An I/O or parsing error (e.g., reading an ONNX file or decoding protobuf).
    #[error("I/O error: {0}")]
    Io(String),
}
