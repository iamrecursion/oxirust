//! Error types for the autograd module
//!
//! This module defines the error types used throughout the autograd module.

use crate::ndarray;
use thiserror::Error;

/// Error type for autograd operations
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OpError {
    /// Error related to ndarray operations
    #[error("{0}: {1}")]
    NdArrayError(String, scirs2_core::ndarray::ShapeError),

    /// Shape incompatibility error
    #[error("Incompatible shape: {0}")]
    IncompatibleShape(String),

    /// Unsupported type error
    #[error("Type unsupported: {0}")]
    TypeUnsupported(String),

    /// Invalid dimensions error
    #[error("Invalid dimensions: {0}")]
    InvalidDims(String),

    /// Index out of bounds error
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),

    /// Type conversion error
    #[error("Type conversion failed: {context} (from {from_type} to {to_type})")]
    ConversionError {
        context: String,
        from_type: String,
        to_type: String,
    },

    /// Numeric value out of range error
    #[error("Numeric value out of range: {context} (value: {value}, target: {target_type})")]
    NumericRangeError {
        context: String,
        value: String,
        target_type: String,
    },

    /// Lock poisoned error
    #[error("Lock poisoned: {context} - {details}")]
    LockPoisonError { context: String, details: String },

    /// Thread join failed error
    #[error("Thread join failed: {context}")]
    ThreadJoinError { context: String },

    /// Invalid slice or index error
    #[error("Invalid slice or index: {context}")]
    SliceError { context: String },

    /// Value error for invalid input values
    #[error("Value error: {0}")]
    ValueError(String),
}

/// Error during tensor's evaluation.
#[derive(Debug, PartialEq, Clone, Error)]
pub enum EvalError {
    /// Error during `Op`'s computation.
    #[error("{0}")]
    OpError(#[from] OpError),

    /// Error related to variable access
    #[error("Variable error: {0}")]
    VariableError(String),

    /// Other evaluation error
    #[error("Evaluation error: {0}")]
    Other(String),
}

/// Generic error type for autograd operations
#[derive(Debug, Error)]
pub enum AutogradError {
    /// Operation error
    #[error("{0}")]
    OperationError(String),

    /// Shape mismatch error
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Variable error
    #[error("Variable error: {0}")]
    VariableError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Integration error with other SciRS2 modules
    #[error("Integration error: {0}")]
    IntegrationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    /// OpError
    #[error("{0}")]
    OpError(#[from] OpError),

    /// EvalError
    #[error("{0}")]
    EvalError(#[from] EvalError),
}

impl AutogradError {
    /// Create a shape error
    pub fn shape_error(msg: String) -> Self {
        Self::ShapeMismatch(msg)
    }

    /// Create an invalid argument error
    pub fn invalid_argument(msg: String) -> Self {
        Self::OperationError(format!("Invalid argument: {}", msg))
    }

    /// Create a compute error
    pub fn compute_error(msg: String) -> Self {
        Self::OperationError(format!("Computation error: {}", msg))
    }

    /// Create an internal error
    pub fn internal_error(msg: &str) -> Self {
        Self::OperationError(format!("Internal error: {}", msg))
    }

    /// Create a not implemented error
    pub fn not_implemented(msg: String) -> Self {
        Self::OperationError(format!("Not implemented: {}", msg))
    }

    /// Create a GPU error
    pub fn gpu_error(msg: String) -> Self {
        Self::OperationError(format!("GPU error: {}", msg))
    }

    /// Create a memory error
    pub fn memory_error(msg: String) -> Self {
        Self::OperationError(format!("Memory error: {}", msg))
    }
}

/// Result type for autograd operations
pub type Result<T> = std::result::Result<T, AutogradError>;

/// Result type for evaluation operations
pub type EvalResult<T> = std::result::Result<T, EvalError>;

/// Result type for op operations
pub type OpResult<T> = std::result::Result<T, OpError>;
