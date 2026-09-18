//! Error handling for SciRS2
//!
//! This module provides error types and utility functions for error handling.

use thiserror::Error;
use std::io;

/// The main error type for SciRS2 operations
#[derive(Error, Debug)]
pub enum SciRS2Error {
    /// Generic computational error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionError(String),

    /// Out of bounds error
    #[error("Index out of bounds: {0}")]
    IndexError(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// Singular matrix error
    #[error("Singular matrix error: {0}")]
    SingularMatrixError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizeError(String),

    /// Integration error
    #[error("Integration error: {0}")]
    IntegrateError(String),

    /// Interpolation error
    #[error("Interpolation error: {0}")]
    InterpolateError(String),

    /// Statistics error
    #[error("Statistics error: {0}")]
    StatsError(String),

    /// Sparse matrix error
    #[error("Sparse matrix error: {0}")]
    SparseError(String),

    /// FFT error
    #[error("FFT error: {0}")]
    FFTError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Type error (incompatible types)
    #[error("Type error: {0}")]
    TypeError(String),

    /// Input/Output error
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    
    /// Tolerance error
    #[error("Tolerance error: {0}")]
    ToleranceError(String),
}

/// Result type for SciRS2 operations
pub type SciRS2Result<T> = Result<T, SciRS2Error>;

/// Checks if a condition is true, otherwise returns a domain error
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(SciRS2Error::DomainError)` if the condition is false
#[allow(dead_code)]
pub fn check_domain<S: AsRef<str>>(condition: bool, message: S) -> SciRS2Result<()> {
    if condition {
        Ok(())
    } else {
        Err(SciRS2Error::DomainError(message.as_ref().to_string()))
    }
}

/// Checks matrix dimensions
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(SciRS2Error::DimensionError)` if the condition is false
#[allow(dead_code)]
pub fn check_dimensions<S: AsRef<str>>(condition: bool, message: S) -> SciRS2Result<()> {
    if condition {
        Ok(())
    } else {
        Err(SciRS2Error::DimensionError(message.as_ref().to_string()))
    }
}

/// Checks if a value is valid
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(SciRS2Error::ValueError)` if the condition is false
#[allow(dead_code)]
pub fn check_value<S: AsRef<str>>(condition: bool, message: S) -> SciRS2Result<()> {
    if condition {
        Ok(())
    } else {
        Err(SciRS2Error::ValueError(message.as_ref().to_string()))
    }
}

/// Checks if an index is within bounds
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(SciRS2Error::IndexError)` if the condition is false
#[allow(dead_code)]
pub fn check_index<S: AsRef<str>>(condition: bool, message: S) -> SciRS2Result<()> {
    if condition {
        Ok(())
    } else {
        Err(SciRS2Error::IndexError(message.as_ref().to_string()))
    }
}

// Error conversion implementations
#[cfg(feature = "linalg")]
impl From<scirs2_linalg::LinalgError> for SciRS2Error {
    fn from(err: scirs2_linalg::LinalgError) -> Self {
        match err {
            scirs2_linalg::LinalgError::ComputationError(msg) => SciRS2Error::ComputationError(msg),
            scirs2_linalg::LinalgError::DomainError(msg) => SciRS2Error::DomainError(msg),
            scirs2_linalg::LinalgError::ConvergenceError(msg) => SciRS2Error::ConvergenceError(msg),
            scirs2_linalg::LinalgError::DimensionError(msg) => SciRS2Error::DimensionError(msg),
            scirs2_linalg::LinalgError::IndexError(msg) => SciRS2Error::IndexError(msg),
            scirs2_linalg::LinalgError::SingularMatrixError(msg) => SciRS2Error::SingularMatrixError(msg),
            scirs2_linalg::LinalgError::NonPositiveDefiniteError(msg) => SciRS2Error::LinalgError(msg),
            scirs2_linalg::LinalgError::NotImplementedError(msg) => SciRS2Error::NotImplementedError(msg),
            scirs2_linalg::LinalgError::ValueError(msg) => SciRS2Error::ValueError(msg),
        }
    }
}

// Add other error conversion implementations as needed when modules are implemented
