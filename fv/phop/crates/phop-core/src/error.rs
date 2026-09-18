//! Error types for the phop engine.

use thiserror::Error;

/// Errors that can arise during data loading, forest evaluation, or discovery.
#[derive(Debug, Error)]
pub enum PhopError {
    /// Underlying I/O failure (e.g. reading a CSV file).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CSV parsing failure.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// A value could not be parsed as a floating-point number.
    #[error("parse error: {0}")]
    Parse(String),

    /// Tensor / array shapes were incompatible.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A forward/backward pass produced non-finite values.
    #[error("numerical instability: {0}")]
    NumericalInstability(String),

    /// The optimizer did not reach the configured tolerance.
    #[error("not converged: {0}")]
    NotConverged(String),

    /// Failure while evaluating an autograd graph.
    #[error("evaluation error: {0}")]
    Eval(String),

    /// Failure in the symbolic distillation / canonicalization stage.
    #[error("symbolic error: {0}")]
    Symbolic(String),

    /// Failure in a compute backend (e.g. the CUDA GPU path).
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias for results returned by phop.
pub type Result<T> = std::result::Result<T, PhopError>;
