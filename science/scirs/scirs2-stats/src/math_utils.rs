//! Mathematical utility functions with SIMD acceleration
//!
//! This module provides common mathematical operations optimized with SIMD
//! when available, with automatic fallback to scalar implementations.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Compute absolute values of array elements (f64, SIMD-accelerated)
///
/// Uses scirs2-core's SIMD implementation with AVX2/NEON acceleration
/// and automatic scalar fallback for unsupported platforms.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Ok(Array1<f64>)` - Array of absolute values
/// * `Err(StatsError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_stats::math_utils::abs_f64;
///
/// let data = vec![-3.0, -1.5, 0.0, 2.5, 5.0];
/// let result = abs_f64(&data)?;
/// assert_eq!(result.to_vec(), vec![3.0, 1.5, 0.0, 2.5, 5.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - **AVX2 (x86_64)**: Processes 4 f64 elements per cycle
/// - **NEON (ARM)**: Processes 2 f64 elements per cycle
/// - **Scalar fallback**: Available on all platforms
/// - **Speedup**: 1.5-2x for arrays with 1000+ elements
pub fn abs_f64(x: &[f64]) -> StatsResult<Array1<f64>> {
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check for finite values
    for (i, &val) in x.iter().enumerate() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "Input contains non-finite value {} at index {}",
                val, i
            )));
        }
    }

    // Use scirs2-core SIMD implementation
    let x_view = ArrayView1::from(x);
    let result = f64::simd_abs(&x_view);

    Ok(result)
}

/// Compute absolute values of array elements (f32, SIMD-accelerated)
///
/// f32 variant provides better SIMD performance (8 elements/cycle on AVX2).
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Ok(Array1<f32>)` - Array of absolute values
/// * `Err(StatsError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_stats::math_utils::abs_f32;
///
/// let data = vec![-3.0f32, -1.5, 0.0, 2.5, 5.0];
/// let result = abs_f32(&data)?;
/// assert_eq!(result.to_vec(), vec![3.0f32, 1.5, 0.0, 2.5, 5.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - **AVX2 (x86_64)**: Processes 8 f32 elements per cycle
/// - **NEON (ARM)**: Processes 4 f32 elements per cycle
/// - **Scalar fallback**: Available on all platforms
/// - **Speedup**: 2-3x for arrays with 1000+ elements (better than f64)
pub fn abs_f32(x: &[f32]) -> StatsResult<Array1<f32>> {
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check for finite values
    for (i, &val) in x.iter().enumerate() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "Input contains non-finite value {} at index {}",
                val, i
            )));
        }
    }

    // Use scirs2-core SIMD implementation
    let x_view = ArrayView1::from(x);
    let result = f32::simd_abs(&x_view);

    Ok(result)
}

/// Compute sign of array elements (f64, SIMD-accelerated)
///
/// Returns -1 for negative values, 0 for zero, +1 for positive values.
/// Uses scirs2-core's SIMD implementation with AVX2/NEON acceleration.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Ok(Array1<f64>)` - Array of signs (-1.0, 0.0, or 1.0)
/// * `Err(StatsError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_stats::math_utils::sign_f64;
///
/// let data = vec![-3.0, -0.5, 0.0, 1.5, 5.0];
/// let result = sign_f64(&data)?;
/// assert_eq!(result.to_vec(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - **AVX2 (x86_64)**: Processes 4 f64 elements per cycle
/// - **NEON (ARM)**: Processes 2 f64 elements per cycle
/// - **Scalar fallback**: Available on all platforms
/// - **Speedup**: 1.5-2x for arrays with 1000+ elements
pub fn sign_f64(x: &[f64]) -> StatsResult<Array1<f64>> {
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check for finite values
    for (i, &val) in x.iter().enumerate() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "Input contains non-finite value {} at index {}",
                val, i
            )));
        }
    }

    // Use scirs2-core SIMD implementation
    let x_view = ArrayView1::from(x);
    let result = f64::simd_sign(&x_view);

    Ok(result)
}

/// Compute sign of array elements (f32, SIMD-accelerated)
///
/// Returns -1 for negative values, 0 for zero, +1 for positive values.
/// f32 variant provides better SIMD performance (8 elements/cycle on AVX2).
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Ok(Array1<f32>)` - Array of signs (-1.0, 0.0, or 1.0)
/// * `Err(StatsError)` if input validation fails
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use scirs2_stats::math_utils::sign_f32;
///
/// let data = vec![-3.0f32, -0.5, 0.0, 1.5, 5.0];
/// let result = sign_f32(&data)?;
/// assert_eq!(result.to_vec(), vec![-1.0f32, -1.0, 0.0, 1.0, 1.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - **AVX2 (x86_64)**: Processes 8 f32 elements per cycle
/// - **NEON (ARM)**: Processes 4 f32 elements per cycle
/// - **Scalar fallback**: Available on all platforms
/// - **Speedup**: 2-3x for arrays with 1000+ elements (better than f64)
pub fn sign_f32(x: &[f32]) -> StatsResult<Array1<f32>> {
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Check for finite values
    for (i, &val) in x.iter().enumerate() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "Input contains non-finite value {} at index {}",
                val, i
            )));
        }
    }

    // Use scirs2-core SIMD implementation
    let x_view = ArrayView1::from(x);
    let result = f32::simd_sign(&x_view);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_abs_f64_basic() {
        let data = vec![-3.0, -1.5, 0.0, 2.5, 5.0];
        let result = abs_f64(&data).expect("Operation failed");
        let expected = vec![3.0, 1.5, 0.0, 2.5, 5.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_abs_f32_basic() {
        let data = vec![-3.0f32, -1.5, 0.0, 2.5, 5.0];
        let result = abs_f32(&data).expect("Operation failed");
        let expected = vec![3.0f32, 1.5, 0.0, 2.5, 5.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_abs_f64_large() {
        // Test with large array to ensure SIMD path is used
        let data: Vec<f64> = (0..10000).map(|i| i as f64 - 5000.0).collect();
        let result = abs_f64(&data).expect("Operation failed");
        for (i, &val) in result.iter().enumerate() {
            let expected = (data[i]).abs();
            assert_abs_diff_eq!(val, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sign_f64_basic() {
        let data = vec![-3.0, -0.5, 0.0, 1.5, 5.0];
        let result = sign_f64(&data).expect("Operation failed");
        let expected = vec![-1.0, -1.0, 0.0, 1.0, 1.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sign_f32_basic() {
        let data = vec![-3.0f32, -0.5, 0.0, 1.5, 5.0];
        let result = sign_f32(&data).expect("Operation failed");
        let expected = vec![-1.0f32, -1.0, 0.0, 1.0, 1.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sign_f64_large() {
        // Test with large array to ensure SIMD path is used
        let data: Vec<f64> = (0..10000).map(|i| i as f64 - 5000.0).collect();
        let result = sign_f64(&data).expect("Operation failed");
        for (i, &val) in result.iter().enumerate() {
            let expected = if data[i] > 0.0 {
                1.0
            } else if data[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
            assert_abs_diff_eq!(val, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_abs_empty() {
        let data: Vec<f64> = vec![];
        let result = abs_f64(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_empty() {
        let data: Vec<f64> = vec![];
        let result = sign_f64(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_abs_nonfinite() {
        let data = vec![1.0, f64::NAN, 3.0];
        let result = abs_f64(&data);
        assert!(result.is_err());

        let data = vec![1.0, f64::INFINITY, 3.0];
        let result = abs_f64(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_nonfinite() {
        let data = vec![1.0, f64::NAN, 3.0];
        let result = sign_f64(&data);
        assert!(result.is_err());

        let data = vec![1.0, f64::INFINITY, 3.0];
        let result = sign_f64(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_abs_all_positive() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = abs_f64(&data).expect("Operation failed");
        assert_eq!(result.to_vec(), data);
    }

    #[test]
    fn test_sign_all_positive() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sign_f64(&data).expect("Operation failed");
        let expected = vec![1.0; 5];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_sign_all_negative() {
        let data = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let result = sign_f64(&data).expect("Operation failed");
        let expected = vec![-1.0; 5];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_sign_all_zero() {
        let data = vec![0.0; 100];
        let result = sign_f64(&data).expect("Operation failed");
        let expected = vec![0.0; 100];
        assert_eq!(result.to_vec(), expected);
    }
}
