//! Domain validation utilities for mathematical operations.
//!
//! This module provides validation functions to check preconditions for mathematical
//! operations, ensuring numerical stability and preventing undefined behavior.

use crate::error::{OpError, OpResult};
use scirs2_core::ndarray::ArrayBase;
use scirs2_core::ndarray::Data;
use scirs2_core::numeric::{Float, Zero};

/// Validate tensor for sqrt (all values must be non-negative)
pub fn validate_sqrt_domain<F, S, D>(tensor: &ArrayBase<S, D>, name: &str) -> OpResult<()>
where
    F: Float + std::fmt::Display,
    S: Data<Elem = F>,
    D: scirs2_core::ndarray::Dimension,
{
    for &val in tensor.iter() {
        if !val.is_finite() {
            return Err(OpError::ValueError(format!(
                "{}: contains non-finite value for sqrt",
                name
            )));
        }
        if val < F::zero() {
            return Err(OpError::ValueError(format!(
                "{}: requires non-negative values for sqrt, found {}",
                name, val
            )));
        }
    }
    Ok(())
}

/// Validate tensor for log (all values must be positive)
pub fn validate_log_domain<F, S, D>(tensor: &ArrayBase<S, D>, name: &str) -> OpResult<()>
where
    F: Float + std::fmt::Display,
    S: Data<Elem = F>,
    D: scirs2_core::ndarray::Dimension,
{
    for &val in tensor.iter() {
        if !val.is_finite() {
            return Err(OpError::ValueError(format!(
                "{}: contains non-finite value for log",
                name
            )));
        }
        if val <= F::zero() {
            return Err(OpError::ValueError(format!(
                "{}: requires positive values for log, found {}",
                name, val
            )));
        }
    }
    Ok(())
}

/// Validate tensor for acos/asin (all values must be in [-1, 1])
pub fn validate_arcfunc_domain<F, S, D>(
    tensor: &ArrayBase<S, D>,
    name: &str,
    func: &str,
) -> OpResult<()>
where
    F: Float + std::fmt::Display,
    S: Data<Elem = F>,
    D: scirs2_core::ndarray::Dimension,
{
    let one = F::one();
    let neg_one = -one;
    for &val in tensor.iter() {
        if !val.is_finite() {
            return Err(OpError::ValueError(format!(
                "{}: contains non-finite value for {}",
                name, func
            )));
        }
        if val < neg_one || val > one {
            return Err(OpError::ValueError(format!(
                "{}: {} requires values in [-1, 1], found {}",
                name, func, val
            )));
        }
    }
    Ok(())
}

/// Validate tensor contains only finite values (no NaN or Inf)
pub fn validate_finite<F, S, D>(tensor: &ArrayBase<S, D>, name: &str) -> OpResult<()>
where
    F: Float,
    S: Data<Elem = F>,
    D: scirs2_core::ndarray::Dimension,
{
    for &val in tensor.iter() {
        if !val.is_finite() {
            return Err(OpError::ValueError(format!(
                "{}: contains non-finite value (NaN or Inf)",
                name
            )));
        }
    }
    Ok(())
}

/// Validate tensor contains no NaN values
pub fn validate_non_nan<F, S, D>(tensor: &ArrayBase<S, D>, name: &str) -> OpResult<()>
where
    F: Float,
    S: Data<Elem = F>,
    D: scirs2_core::ndarray::Dimension,
{
    for &val in tensor.iter() {
        if val.is_nan() {
            return Err(OpError::ValueError(format!("{}: contains NaN value", name)));
        }
    }
    Ok(())
}
