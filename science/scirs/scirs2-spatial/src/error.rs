//! Error types for the SciRS2 spatial module

use thiserror::Error;

/// Spatial error type
#[derive(Error, Debug)]
pub enum SpatialError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for spatial operations
pub type SpatialResult<T> = Result<T, SpatialError>;
