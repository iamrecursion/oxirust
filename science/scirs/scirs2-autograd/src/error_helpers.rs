//! Helper functions to eliminate repetitive expect() patterns throughout the codebase.
//!
//! This module provides safe wrappers for common operations that previously used expect(),
//! converting them to proper Result-based error handling.

use crate::error::{OpError, OpResult};
use scirs2_core::numeric::{Float, NumCast, ToPrimitive};
use std::sync::{Mutex, MutexGuard};

/// Convert f64 to generic Float type with error handling
pub fn try_from_f64<F: Float>(value: f64, context: &str) -> OpResult<F> {
    F::from(value).ok_or_else(|| OpError::ConversionError {
        context: context.to_string(),
        from_type: "f64".to_string(),
        to_type: std::any::type_name::<F>().to_string(),
    })
}

/// Convert any numeric type to another with error handling
pub fn try_from_numeric<T: NumCast>(
    value: impl ToPrimitive + std::fmt::Debug + Copy,
    context: &str,
) -> OpResult<T> {
    T::from(value).ok_or_else(|| OpError::NumericRangeError {
        context: context.to_string(),
        value: format!("{:?}", value),
        target_type: std::any::type_name::<T>().to_string(),
    })
}

/// Convert to f64 with error handling
pub fn try_to_f64<T: ToPrimitive + std::fmt::Debug + Copy>(
    value: T,
    context: &str,
) -> OpResult<f64> {
    value.to_f64().ok_or_else(|| OpError::NumericRangeError {
        context: context.to_string(),
        value: format!("{:?}", value),
        target_type: "f64".to_string(),
    })
}

/// Lock mutex with error handling
pub fn try_lock<'a, T>(mutex: &'a Mutex<T>, context: &str) -> OpResult<MutexGuard<'a, T>> {
    mutex.lock().map_err(|e| OpError::LockPoisonError {
        context: context.to_string(),
        details: e.to_string(),
    })
}

/// Common mathematical constants with error handling
pub mod float_constants {
    use super::*;

    pub fn pi<F: Float>() -> OpResult<F> {
        try_from_f64(std::f64::consts::PI, "pi constant")
    }

    pub fn two<F: Float>() -> OpResult<F> {
        try_from_f64(2.0, "constant 2.0")
    }

    pub fn epsilon_scaled<F: Float>(scale: f64) -> OpResult<F> {
        try_from_f64(
            F::epsilon().to_f64().unwrap_or(1e-10) * scale,
            "epsilon scaled",
        )
    }
}
