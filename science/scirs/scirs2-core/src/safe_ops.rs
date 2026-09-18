//! Safe mathematical operations that handle edge cases and validate results
//!
//! This module provides safe wrappers around common mathematical operations
//! that can produce NaN, Infinity, or other invalid results.

use crate::error::{CoreError, ErrorContext};
use crate::validation::check_finite;
use num_traits::Float;
use std::fmt::{Debug, Display};

/// Safely divide two numbers, checking for division by zero and validating the result
#[inline]
#[allow(dead_code)]
pub fn safe_divide<T>(numerator: T, denominator: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    // Check for exact zero
    if denominator == T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Division by zero: {numerator} / 0"
        ))));
    }

    // Check for near-zero values that could cause overflow
    let epsilon = T::epsilon();
    if denominator.abs() < epsilon {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Division by near-zero value: {numerator} / {denominator} (threshold: {epsilon})"
        ))));
    }

    let result = numerator / denominator;

    // Validate the result
    check_finite(result, "division result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Division produced non-finite result: {numerator} / {denominator} = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely compute square root, checking for negative values
#[inline]
#[allow(dead_code)]
pub fn safe_sqrt<T>(value: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    if value < T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Cannot compute sqrt of negative value: {value}"
        ))));
    }

    let result = value.sqrt();

    // Even for valid inputs, check the result
    check_finite(result, "sqrt result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Square root produced non-finite result: sqrt({value}) = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely compute natural logarithm, checking for non-positive values
#[inline]
#[allow(dead_code)]
pub fn safelog<T>(value: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    if value <= T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Cannot compute log of non-positive value: {value}"
        ))));
    }

    let result = value.ln();

    check_finite(result, "log result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Logarithm produced non-finite result: ln({value}) = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely compute base-10 logarithm
#[inline]
#[allow(dead_code)]
pub fn safelog10<T>(value: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    if value <= T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Cannot compute log10 of non-positive value: {value}"
        ))));
    }

    let result = value.log10();

    check_finite(result, "log10 result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Base-10 logarithm produced non-finite result: log10({value}) = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely compute power, checking for domain errors and overflow
#[inline]
#[allow(dead_code)]
pub fn safe_pow<T>(base: T, exponent: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    // Special cases that could produce NaN or Inf
    if base < T::zero() && exponent.fract() != T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Cannot compute fractional power of negative number: {base}^{exponent}"
        ))));
    }

    if base == T::zero() && exponent < T::zero() {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "Cannot compute negative power of zero: 0^{exponent}"
        ))));
    }

    let result = base.powf(exponent);

    check_finite(result, "power result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Power operation produced non-finite result: {base}^{exponent} = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely compute exponential, checking for overflow
#[inline]
#[allow(dead_code)]
pub fn safe_exp<T>(value: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    // Check for values that would cause overflow
    // For f64, exp(x) overflows when x > ~709.78
    let max_exp = T::from(700.0).unwrap_or(T::infinity());
    if value > max_exp {
        return Err(CoreError::ComputationError(ErrorContext::new(format!(
            "Exponential would overflow: exp({value}) > exp({max_exp})"
        ))));
    }

    let result = value.exp();

    check_finite(result, "exp result").map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Exponential produced non-finite result: exp({value}) = {result:?}"
        )))
    })?;

    Ok(result)
}

/// Safely normalize a value by dividing by a norm/magnitude
#[inline]
#[allow(dead_code)]
pub fn safe_normalize<T>(value: T, norm: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug,
{
    // Special case: if both are zero, return zero
    if value == T::zero() && norm == T::zero() {
        return Ok(T::zero());
    }

    safe_divide(value, norm)
}

/// Safely compute the mean of a slice, handling empty slices
#[allow(dead_code)]
pub fn safe_mean<T>(values: &[T]) -> Result<T, CoreError>
where
    T: Float + Display + Debug + std::iter::Sum,
{
    if values.is_empty() {
        return Err(CoreError::InvalidArgument(ErrorContext::new(
            "Cannot compute mean of empty array",
        )));
    }

    let sum: T = values.iter().copied().sum();
    let len = values.len();
    let count = T::from(len).ok_or_else(|| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Failed to convert array length {len} to numeric type"
        )))
    })?;

    safe_divide(sum, count)
}

/// Safely compute variance, handling numerical issues
#[allow(dead_code)]
pub fn safe_variance<T>(values: &[T], mean: T) -> Result<T, CoreError>
where
    T: Float + Display + Debug + std::iter::Sum,
{
    let len = values.len();
    if len < 2 {
        return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
            "Cannot compute variance with {len} values (need at least 2)"
        ))));
    }

    let sum_sq_diff: T = values
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum();

    let count = values.len() - 1;
    let n_minus_1 = T::from(count).ok_or_else(|| {
        CoreError::ComputationError(ErrorContext::new(format!(
            "Failed to convert count {count} to numeric type"
        )))
    })?;

    safe_divide(sum_sq_diff, n_minus_1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_divide() {
        // Normal cases
        assert_eq!(safe_divide(10.0, 2.0).expect("Operation failed"), 5.0);
        assert_eq!(safe_divide(-10.0, 2.0).expect("Operation failed"), -5.0);

        // Division by zero
        assert!(safe_divide(10.0, 0.0).is_err());
        assert!(safe_divide(10.0, 1e-100).is_err()); // Near zero

        // Overflow case
        assert!(safe_divide(f64::MAX, f64::MIN_POSITIVE).is_err());
    }

    #[test]
    fn test_safe_sqrt() {
        // Normal cases
        assert_eq!(safe_sqrt(4.0).expect("Operation failed"), 2.0);
        assert_eq!(safe_sqrt(0.0).expect("Operation failed"), 0.0);

        // Negative input
        assert!(safe_sqrt(-1.0).is_err());
        assert!(safe_sqrt(-1e-10).is_err());
    }

    #[test]
    fn test_safelog() {
        // Normal cases
        assert!((safelog(std::f64::consts::E).expect("Operation failed") - 1.0).abs() < 1e-10);
        assert_eq!(safelog(1.0).expect("Operation failed"), 0.0);

        // Invalid inputs
        assert!(safelog(0.0).is_err());
        assert!(safelog(-1.0).is_err());
    }

    #[test]
    fn test_safe_pow() {
        // Normal cases
        assert_eq!(safe_pow(2.0, 3.0).expect("Operation failed"), 8.0);
        assert_eq!(safe_pow(4.0, 0.5).expect("Operation failed"), 2.0);

        // Invalid cases
        assert!(safe_pow(-2.0, 0.5).is_err()); // Fractional power of negative
        assert!(safe_pow(0.0, -1.0).is_err()); // Negative power of zero

        // Overflow
        assert!(safe_pow(10.0, 1000.0).is_err());
    }

    #[test]
    fn test_safe_exp() {
        // Normal cases
        assert!((safe_exp(1.0).expect("Operation failed") - std::f64::consts::E).abs() < 1e-10);
        assert_eq!(safe_exp(0.0).expect("Operation failed"), 1.0);

        // Overflow
        assert!(safe_exp(1000.0).is_err());
    }

    #[test]
    fn test_safe_mean() {
        // Normal case
        assert_eq!(safe_mean(&[1.0, 2.0, 3.0]).expect("Operation failed"), 2.0);

        // Empty array
        assert!(safe_mean::<f64>(&[]).is_err());

        // Single value
        assert_eq!(safe_mean(&[5.0]).expect("Operation failed"), 5.0);
    }

    #[test]
    fn test_safe_variance() {
        // Normal case
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        assert!((safe_variance(&values, mean).expect("Operation failed") - 2.5).abs() < 1e-10);

        // Too few values
        assert!(safe_variance(&[1.0], 1.0).is_err());
        assert!(safe_variance::<f64>(&[], 0.0).is_err());
    }

    #[test]
    fn test_safe_normalize() {
        // Normal case
        assert_eq!(safe_normalize(3.0, 4.0).expect("Operation failed"), 0.75);

        // Zero norm
        assert!(safe_normalize(1.0, 0.0).is_err());

        // Both zero
        assert_eq!(safe_normalize(0.0, 0.0).expect("Operation failed"), 0.0);
    }
}
