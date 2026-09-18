//! Error types for the tensorlogic-oxicuda-sparse crate.

use thiserror::Error;

/// Errors that can arise from sparse matrix operations.
#[derive(Debug, Error)]
pub enum SparseError {
    /// Shape or dimension mismatch between operands.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// An index exceeded the valid bounds for the matrix.
    #[error("index out of bounds: {0}")]
    IndexError(String),

    /// A GPU-side error (driver, memory allocation, or kernel launch failure).
    #[error("GPU error: {0}")]
    GpuError(String),

    /// An unexpected internal error (e.g. invariant violation).
    #[error("internal error: {0}")]
    Internal(String),
}
