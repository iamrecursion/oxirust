//! Error types for the SciRS2 optimization module

use scirs2_core::error::{CoreError, CoreResult};
use thiserror::Error;

// Type aliases for compatibility
pub type ScirsError = CoreError;
pub type ScirsResult<T> = CoreResult<T>;

/// Optimization error type
#[derive(Error, Debug)]
pub enum OptimizeError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Initialization error (failed to initialize optimizer)
    #[error("Initialization error: {0}")]
    InitializationError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IOError(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Maximum evaluations reached
    #[error("Maximum evaluations reached")]
    MaxEvaluationsReached,
}

/// Result type for optimization operations
pub type OptimizeResult<T> = Result<T, OptimizeError>;

// Implement conversion from SparseError to OptimizeError
impl From<scirs2_sparse::error::SparseError> for OptimizeError {
    fn from(error: scirs2_sparse::error::SparseError) -> Self {
        match error {
            scirs2_sparse::SparseError::ComputationError(msg) => {
                OptimizeError::ComputationError(msg)
            }
            scirs2_sparse::SparseError::DimensionMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                ))
            }
            scirs2_sparse::SparseError::IndexOutOfBounds { index, shape } => {
                OptimizeError::ValueError(format!(
                    "Index {:?} out of bounds for array with shape {:?}",
                    index, shape
                ))
            }
            scirs2_sparse::SparseError::InvalidAxis => {
                OptimizeError::ValueError("Invalid axis specified".to_string())
            }
            scirs2_sparse::SparseError::InvalidSliceRange => {
                OptimizeError::ValueError("Invalid slice range specified".to_string())
            }
            scirs2_sparse::SparseError::InconsistentData { reason } => {
                OptimizeError::ValueError(format!("Inconsistent data: {}", reason))
            }
            scirs2_sparse::SparseError::NotImplemented(msg) => {
                OptimizeError::NotImplementedError(msg)
            }
            scirs2_sparse::SparseError::SingularMatrix(msg) => {
                OptimizeError::ComputationError(format!("Singular matrix error: {}", msg))
            }
            scirs2_sparse::SparseError::ValueError(msg) => OptimizeError::ValueError(msg),
            scirs2_sparse::SparseError::ConversionError(msg) => {
                OptimizeError::ValueError(format!("Conversion error: {}", msg))
            }
            scirs2_sparse::SparseError::OperationNotSupported(msg) => {
                OptimizeError::NotImplementedError(format!("Operation not supported: {}", msg))
            }
            scirs2_sparse::SparseError::ShapeMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Shape mismatch: expected {:?}, found {:?}",
                    expected, found
                ))
            }
            scirs2_sparse::SparseError::IterativeSolverFailure(msg) => {
                OptimizeError::ConvergenceError(format!("Iterative solver failure: {}", msg))
            }
            scirs2_sparse::SparseError::IndexCastOverflow { value, target_type } => {
                OptimizeError::ValueError(format!(
                    "Index value {} cannot be represented in the target type {}",
                    value, target_type
                ))
            }
            scirs2_sparse::SparseError::ConvergenceError(msg) => {
                OptimizeError::ConvergenceError(format!("Convergence error: {}", msg))
            }
            scirs2_sparse::SparseError::InvalidFormat(msg) => {
                OptimizeError::ValueError(format!("Invalid format: {}", msg))
            }
            scirs2_sparse::SparseError::IoError(err) => {
                OptimizeError::IOError(format!("I/O error: {}", err))
            }
            scirs2_sparse::SparseError::CompressionError(msg) => {
                OptimizeError::ComputationError(format!("Compression error: {}", msg))
            }
            scirs2_sparse::SparseError::Io(msg) => {
                OptimizeError::IOError(format!("I/O error: {}", msg))
            }
            scirs2_sparse::SparseError::BlockNotFound(msg) => {
                OptimizeError::ValueError(format!("Block not found: {}", msg))
            }
            scirs2_sparse::SparseError::GpuError(err) => {
                OptimizeError::ComputationError(format!("GPU error: {}", err))
            }
        }
    }
}

// Implement conversion from GpuError to OptimizeError
impl From<scirs2_core::GpuError> for OptimizeError {
    fn from(error: scirs2_core::GpuError) -> Self {
        OptimizeError::ComputationError(error.to_string())
    }
}

// Implement conversion from OptimizeError to CoreError
impl From<OptimizeError> for CoreError {
    fn from(error: OptimizeError) -> Self {
        match error {
            OptimizeError::ComputationError(msg) => CoreError::ComputationError(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::ConvergenceError(msg) => CoreError::ConvergenceError(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::ValueError(msg) => CoreError::ValueError(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::NotImplementedError(msg) => CoreError::NotImplementedError(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InitializationError(msg) => CoreError::ComputationError(
                scirs2_core::error::ErrorContext::new(format!("Initialization error: {}", msg))
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::IOError(msg) => CoreError::IoError(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InvalidInput(msg) => CoreError::InvalidInput(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::InvalidParameter(msg) => CoreError::InvalidArgument(
                scirs2_core::error::ErrorContext::new(msg)
                    .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
            OptimizeError::MaxEvaluationsReached => CoreError::ComputationError(
                scirs2_core::error::ErrorContext::new(
                    "Maximum number of function evaluations reached".to_string(),
                )
                .with_location(scirs2_core::error::ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}
