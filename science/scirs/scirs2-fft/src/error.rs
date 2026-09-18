//! Error types for the `SciRS2` FFT module

use thiserror::Error;

/// FFT error type
#[derive(Error, Debug)]
pub enum FFTError {
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

    /// I/O error
    #[error("I/O error: {0}")]
    IOError(String),

    /// Backend error
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Plan creation error
    #[error("Plan creation error: {0}")]
    PlanError(String),

    /// Communication error (for distributed FFT)
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Memory error (e.g., allocation failed)
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for FFT operations
pub type FFTResult<T> = Result<T, FFTError>;
