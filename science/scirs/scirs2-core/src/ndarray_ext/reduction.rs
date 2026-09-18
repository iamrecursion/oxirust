//! Array reduction operations with SIMD acceleration
//!
//! This module provides SIMD-accelerated reduction functions for finding
//! indices of minimum and maximum elements in arrays. These are fundamental
//! operations for optimization, statistics, and data analysis.

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Find the index of the minimum element in a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * `Some(index)` - Index of the minimum element
/// * `None` - If the array is empty
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Automatically selects SIMD or scalar based on array size and platform capabilities
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::argmin_simd;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let idx = argmin_simd(&x.view()).expect("Operation failed");
/// assert_eq!(idx, 1); // First occurrence of minimum value (1.0)
/// ```
#[allow(dead_code)]
pub fn argmin_simd<F>(x: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return None;
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        return F::simd_argmin(x);
    }

    // Scalar fallback for small arrays or unsupported platforms
    let mut min_idx = 0;
    let mut min_val = x[0];

    for (i, &val) in x.iter().enumerate().skip(1) {
        if val < min_val {
            min_idx = i;
            min_val = val;
        }
    }

    Some(min_idx)
}

/// Find the index of the maximum element in a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * `Some(index)` - Index of the maximum element
/// * `None` - If the array is empty
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Automatically selects SIMD or scalar based on array size and platform capabilities
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::argmax_simd;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let idx = argmax_simd(&x.view()).expect("Operation failed");
/// assert_eq!(idx, 5); // Index of maximum value (9.0)
/// ```
#[allow(dead_code)]
pub fn argmax_simd<F>(x: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return None;
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        return F::simd_argmax(x);
    }

    // Scalar fallback for small arrays or unsupported platforms
    let mut max_idx = 0;
    let mut max_val = x[0];

    for (i, &val) in x.iter().enumerate().skip(1) {
        if val > max_val {
            max_idx = i;
            max_val = val;
        }
    }

    Some(max_idx)
}

/// Find indices of top-k minimum elements in a 1D array.
///
/// Returns the indices of the k smallest elements in ascending order of their values.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `k` - Number of minimum elements to find
///
/// # Returns
///
/// * Array of indices of the k minimum elements, sorted by value (ascending)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::argmin_k;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let indices = argmin_k(&x.view(), 3).expect("Operation failed");
/// // Returns indices of 3 smallest values: [1, 3, 6] (values 1.0, 1.0, 2.0)
/// assert_eq!(indices.len(), 3);
/// ```
#[allow(dead_code)]
pub fn argmin_k<F>(x: &ArrayView1<F>, k: usize) -> Option<Array1<usize>>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || k == 0 {
        return None;
    }

    let k = k.min(x.len()); // Cap k at array length

    // Create (index, value) pairs
    let mut indexed: Vec<(usize, F)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    // Partial sort to find k smallest elements
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract and sort the k smallest indices by value
    let mut result: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
    result.sort_unstable_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));

    Some(Array1::from_vec(result))
}

/// Find indices of top-k maximum elements in a 1D array.
///
/// Returns the indices of the k largest elements in descending order of their values.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `k` - Number of maximum elements to find
///
/// # Returns
///
/// * Array of indices of the k maximum elements, sorted by value (descending)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::argmax_k;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let indices = argmax_k(&x.view(), 3).expect("Operation failed");
/// // Returns indices of 3 largest values: [5, 4, 2] (values 9.0, 5.0, 4.0)
/// assert_eq!(indices.len(), 3);
/// ```
#[allow(dead_code)]
pub fn argmax_k<F>(x: &ArrayView1<F>, k: usize) -> Option<Array1<usize>>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || k == 0 {
        return None;
    }

    let k = k.min(x.len()); // Cap k at array length

    // Create (index, value) pairs
    let mut indexed: Vec<(usize, F)> = x.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    // Partial sort to find k largest elements
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) // Reverse comparison for max
    });

    // Extract and sort the k largest indices by value (descending)
    let mut result: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
    result.sort_unstable_by(|&a, &b| {
        x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal) // Reverse for descending
    });

    Some(Array1::from_vec(result))
}

/// Compute cumulative sum of array elements with SIMD acceleration.
///
/// Returns a new array where each element is the sum of all previous elements
/// (including itself) in the input array. This is also known as a prefix sum
/// or running total.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * Array of cumulative sums with the same length as input
///
/// # Performance
///
/// - Uses SIMD acceleration for large arrays (> threshold)
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::cumsum_simd;
///
/// let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
/// let result = cumsum_simd(&x.view());
/// // result = [1.0, 3.0, 6.0, 10.0, 15.0]
/// assert_eq!(result[0], 1.0);
/// assert_eq!(result[4], 15.0);
/// ```
#[allow(dead_code)]
pub fn cumsum_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cumsum(x);
    }

    // Scalar fallback for small arrays
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let mut cumsum = x[0];
    let mut result = Array1::zeros(x.len());
    result[0] = cumsum;

    for i in 1..x.len() {
        cumsum = cumsum + x[i];
        result[i] = cumsum;
    }

    result
}

/// Compute cumulative product of array elements with SIMD acceleration.
///
/// Returns a new array where each element is the product of all previous elements
/// (including itself) in the input array. This is also known as a prefix product
/// or running product.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * Array of cumulative products with the same length as input
///
/// # Performance
///
/// - Uses SIMD acceleration for large arrays (> threshold)
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::cumprod_simd;
///
/// let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
/// let result = cumprod_simd(&x.view());
/// // result = [1.0, 2.0, 6.0, 24.0, 120.0]
/// assert_eq!(result[0], 1.0);
/// assert_eq!(result[4], 120.0);
/// ```
#[allow(dead_code)]
pub fn cumprod_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cumprod(x);
    }

    // Scalar fallback for small arrays
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let mut cumprod = x[0];
    let mut result = Array1::zeros(x.len());
    result[0] = cumprod;

    for i in 1..x.len() {
        cumprod = cumprod * x[i];
        result[i] = cumprod;
    }

    result
}

// ============================================================================
// Scalar Reduction Operations (Phase 29 - min/max values)
// ============================================================================

/// Find the minimum value in a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * `Some(value)` - The minimum value in the array
/// * `None` - If the array is empty
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Automatically selects SIMD or scalar based on array size and platform capabilities
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::min_simd;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let min_val = min_simd(&x.view()).expect("Operation failed");
/// assert_eq!(min_val, 1.0);
/// ```
#[allow(dead_code)]
pub fn min_simd<F>(x: &ArrayView1<F>) -> Option<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return None;
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        Some(F::simd_min_element(x))
    } else {
        // Scalar fallback for small arrays
        let mut min_val = x[0];
        for &val in x.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }
        Some(min_val)
    }
}

/// Find the maximum value in a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * `Some(value)` - The maximum value in the array
/// * `None` - If the array is empty
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Automatically selects SIMD or scalar based on array size and platform capabilities
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::max_simd;
///
/// let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let max_val = max_simd(&x.view()).expect("Operation failed");
/// assert_eq!(max_val, 9.0);
/// ```
#[allow(dead_code)]
pub fn max_simd<F>(x: &ArrayView1<F>) -> Option<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return None;
    }

    let optimizer = AutoOptimizer::new();

    // Use SIMD fast path for large arrays
    if optimizer.should_use_simd(x.len()) {
        Some(F::simd_max_element(x))
    } else {
        // Scalar fallback for small arrays
        let mut max_val = x[0];
        for &val in x.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }
        Some(max_val)
    }
}

// ============================================================================
// Statistical Reduction Operations (Phase 30 - mean/variance/std)
// ============================================================================

/// Compute the arithmetic mean of a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * `Some(mean)` - The arithmetic mean of the array
/// * `None` - If the array is empty
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Uses SIMD sum followed by division
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::mean_simd;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let mean = mean_simd(&x.view()).expect("Operation failed");
/// assert!((mean - 3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn mean_simd<F>(x: &ArrayView1<F>) -> Option<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return None;
    }

    Some(F::simd_mean(x))
}

/// Compute the variance of a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations for computing the variance with Welford's
/// algorithm for numerical stability. The variance is calculated as the average
/// of squared deviations from the mean.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
///
/// # Returns
///
/// * `Some(variance)` - The variance of the array
/// * `None` - If the array is empty or has insufficient data (length <= ddof)
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Uses SIMD operations for mean and sum of squared deviations
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::variance_simd;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let var = variance_simd(&x.view(), 1).expect("Operation failed"); // Sample variance (ddof=1)
/// assert!((var - 2.5).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn variance_simd<F>(x: &ArrayView1<F>, ddof: usize) -> Option<F>
where
    F: Float + SimdUnifiedOps,
{
    let n = x.len();
    if n == 0 || n <= ddof {
        return None;
    }

    // Note: F::simd_variance computes SAMPLE variance (ddof=1, divides by n-1)
    // We need to adjust for different ddof values

    if ddof == 1 {
        // Use the built-in simd_variance which already computes sample variance
        return Some(F::simd_variance(x));
    }

    // For other ddof values, we need to adjust
    // simd_variance computes: sum_sq_dev / (n - 1)
    // We want: sum_sq_dev / (n - ddof)
    let sample_var = F::simd_variance(x); // This is sum_sq_dev / (n-1)
    let n_f = F::from(n).expect("Failed to convert to float");
    let ddof_f = F::from(ddof).expect("Failed to convert to float");

    // Convert: var(ddof) = var(ddof=1) * (n-1) / (n-ddof)
    Some(sample_var * (n_f - F::one()) / (n_f - ddof_f))
}

/// Compute the standard deviation of a 1D array with SIMD acceleration.
///
/// This function computes the square root of the variance using SIMD-accelerated
/// variance computation.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
///
/// # Returns
///
/// * `Some(std)` - The standard deviation of the array
/// * `None` - If the array is empty or has insufficient data (length <= ddof)
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
/// - Uses SIMD variance followed by sqrt
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::std_simd;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let std = std_simd(&x.view(), 1).expect("Operation failed"); // Sample std (ddof=1)
/// assert!((std - 1.5811388300841898).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn std_simd<F>(x: &ArrayView1<F>, ddof: usize) -> Option<F>
where
    F: Float + SimdUnifiedOps,
{
    variance_simd(x, ddof).map(|var| var.sqrt())
}

/// Compute sum of a 1D array with SIMD acceleration.
///
/// This function uses SIMD operations when beneficial for performance, providing
/// significant speedups for large arrays on supported platforms (AVX2, NEON).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// * The sum of all elements in the array (0 for empty array)
///
/// # Performance
///
/// - f32: ~2-3x faster than scalar for arrays > 1000 elements
/// - f64: ~2-3x faster than scalar for arrays > 1000 elements
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::reduction::sum_simd;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let sum = sum_simd(&x.view());
/// assert_eq!(sum, 15.0);
/// ```
#[allow(dead_code)]
pub fn sum_simd<F>(x: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return F::zero();
    }

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(x.len()) {
        F::simd_sum(x)
    } else {
        x.iter().fold(F::zero(), |acc, &val| acc + val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    #[test]
    fn test_argmin_simd_f64_basic() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let idx = argmin_simd(&x.view()).expect("Operation failed");
        assert_eq!(idx, 1, "Should find first occurrence of minimum");
    }

    #[test]
    fn test_argmin_simd_f32_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let idx = argmin_simd(&x.view()).expect("Operation failed");
        assert_eq!(idx, 1, "Should find first occurrence of minimum");
    }

    #[test]
    fn test_argmax_simd_f64_basic() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let idx = argmax_simd(&x.view()).expect("Operation failed");
        assert_eq!(idx, 5, "Should find maximum element");
    }

    #[test]
    fn test_argmax_simd_f32_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let idx = argmax_simd(&x.view()).expect("Operation failed");
        assert_eq!(idx, 5, "Should find maximum element");
    }

    #[test]
    fn test_argmin_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(argmin_simd(&x.view()), None);
    }

    #[test]
    fn test_argmax_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(argmax_simd(&x.view()), None);
    }

    #[test]
    fn test_argmin_simd_single() {
        let x = array![42.0f64];
        assert_eq!(argmin_simd(&x.view()), Some(0));
    }

    #[test]
    fn test_argmax_simd_single() {
        let x = array![42.0f64];
        assert_eq!(argmax_simd(&x.view()), Some(0));
    }

    #[test]
    fn test_argmin_simd_negative() {
        let x = array![1.0f64, -5.0, 3.0, -2.0];
        assert_eq!(argmin_simd(&x.view()), Some(1));
    }

    #[test]
    fn test_argmax_simd_negative() {
        let x = array![-10.0f64, -5.0, -20.0, -2.0];
        assert_eq!(argmax_simd(&x.view()), Some(3));
    }

    #[test]
    fn test_argmin_simd_large_f32() {
        let x: Array1<f32> = Array1::from_vec((0..10000).map(|i| (i as f32) % 100.0).collect());
        assert_eq!(argmin_simd(&x.view()), Some(0));
    }

    #[test]
    fn test_argmax_simd_large_f64() {
        let x: Array1<f64> = Array1::from_vec((0..10000).map(|i| (i as f64) % 100.0).collect());
        assert_eq!(argmax_simd(&x.view()), Some(99));
    }

    #[test]
    fn test_argmin_k_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let indices = argmin_k(&x.view(), 3).expect("Operation failed");
        assert_eq!(indices.len(), 3);
        // Should be indices of the 3 smallest values (1.0, 1.0, 2.0)
        // Values at these indices should be <= 2.0
        for &idx in indices.iter() {
            assert!(x[idx] <= 2.0);
        }
    }

    #[test]
    fn test_argmax_k_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let indices = argmax_k(&x.view(), 3).expect("Operation failed");
        assert_eq!(indices.len(), 3);
        // Should be indices of the 3 largest values (9.0, 5.0, 4.0)
        // Values at these indices should be >= 4.0
        for &idx in indices.iter() {
            assert!(x[idx] >= 4.0);
        }
    }

    #[test]
    fn test_argmin_k_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(argmin_k(&x.view(), 3), None);
    }

    #[test]
    fn test_argmax_k_zero() {
        let x = array![1.0f64, 2.0, 3.0];
        assert_eq!(argmax_k(&x.view(), 0), None);
    }

    #[test]
    fn test_argmin_k_exceeds_length() {
        let x = array![1.0f64, 2.0, 3.0];
        let indices = argmin_k(&x.view(), 10).expect("Operation failed");
        assert_eq!(indices.len(), 3); // Should cap at array length
    }

    // ========================================================================
    // Tests for Phase 29: min_simd and max_simd
    // ========================================================================

    #[test]
    fn test_min_simd_f64_basic() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let min_val = min_simd(&x.view()).expect("Operation failed");
        assert_eq!(min_val, 1.0);
    }

    #[test]
    fn test_min_simd_f32_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let min_val = min_simd(&x.view()).expect("Operation failed");
        assert_eq!(min_val, 1.0);
    }

    #[test]
    fn test_max_simd_f64_basic() {
        let x = array![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let max_val = max_simd(&x.view()).expect("Operation failed");
        assert_eq!(max_val, 9.0);
    }

    #[test]
    fn test_max_simd_f32_basic() {
        let x = array![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let max_val = max_simd(&x.view()).expect("Operation failed");
        assert_eq!(max_val, 9.0);
    }

    #[test]
    fn test_min_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(min_simd(&x.view()), None);
    }

    #[test]
    fn test_max_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(max_simd(&x.view()), None);
    }

    #[test]
    fn test_min_simd_single() {
        let x = array![42.0f64];
        assert_eq!(min_simd(&x.view()), Some(42.0));
    }

    #[test]
    fn test_max_simd_single() {
        let x = array![42.0f64];
        assert_eq!(max_simd(&x.view()), Some(42.0));
    }

    #[test]
    fn test_min_simd_negative() {
        let x = array![1.0f64, -5.0, 3.0, -2.0];
        assert_eq!(min_simd(&x.view()), Some(-5.0));
    }

    #[test]
    fn test_max_simd_negative() {
        let x = array![-10.0f64, -5.0, -20.0, -2.0];
        assert_eq!(max_simd(&x.view()), Some(-2.0));
    }

    #[test]
    fn test_min_simd_large_f32() {
        let x: Array1<f32> = Array1::from_vec((0..10000).map(|i| (i as f32) % 100.0).collect());
        assert_eq!(min_simd(&x.view()), Some(0.0));
    }

    #[test]
    fn test_max_simd_large_f64() {
        let x: Array1<f64> = Array1::from_vec((0..10000).map(|i| (i as f64) % 100.0).collect());
        assert_eq!(max_simd(&x.view()), Some(99.0));
    }

    // ========================================================================
    // Tests for Phase 30: mean_simd, variance_simd, std_simd
    // ========================================================================

    #[test]
    fn test_mean_simd_f64_basic() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mean = mean_simd(&x.view()).expect("Operation failed");
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_simd_f32_basic() {
        let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mean = mean_simd(&x.view()).expect("Operation failed");
        assert!((mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(mean_simd(&x.view()), None);
    }

    #[test]
    fn test_mean_simd_single() {
        let x = array![42.0f64];
        assert_eq!(mean_simd(&x.view()), Some(42.0));
    }

    #[test]
    fn test_mean_simd_negative() {
        let x = array![-2.0f64, -4.0, -6.0, -8.0, -10.0];
        let mean = mean_simd(&x.view()).expect("Operation failed");
        assert!((mean - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mean_simd_large() {
        let x: Array1<f64> = Array1::from_vec((1..=10000).map(|i| i as f64).collect());
        let mean = mean_simd(&x.view()).expect("Operation failed");
        let expected = 5000.5; // Mean of 1..=10000
        assert!((mean - expected).abs() < 1e-6);
    }

    #[test]
    fn test_variance_simd_f64_population() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let var = variance_simd(&x.view(), 0).expect("Operation failed"); // Population variance
        assert!((var - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_simd_f64_sample() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let var = variance_simd(&x.view(), 1).expect("Operation failed"); // Sample variance
        assert!((var - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_variance_simd_f32_population() {
        let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let var = variance_simd(&x.view(), 0).expect("Operation failed");
        assert!((var - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_variance_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(variance_simd(&x.view(), 0), None);
    }

    #[test]
    fn test_variance_simd_insufficient_data() {
        let x = array![42.0f64];
        assert_eq!(variance_simd(&x.view(), 1), None); // n=1, ddof=1 -> insufficient
        assert!(variance_simd(&x.view(), 0).is_some()); // n=1, ddof=0 -> OK (but var=0)
    }

    #[test]
    fn test_variance_simd_constant() {
        let x = array![5.0f64, 5.0, 5.0, 5.0, 5.0];
        let var = variance_simd(&x.view(), 0).expect("Operation failed");
        assert!(var.abs() < 1e-10); // Variance of constant array is 0
    }

    #[test]
    fn test_std_simd_f64_sample() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let std = std_simd(&x.view(), 1).expect("Operation failed");
        let expected = 2.5_f64.sqrt(); // sqrt(2.5) ≈ 1.5811388300841898
        assert!((std - expected).abs() < 1e-10);
    }

    #[test]
    fn test_std_simd_f32_sample() {
        let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let std = std_simd(&x.view(), 1).expect("Operation failed");
        let expected = 2.5_f32.sqrt();
        assert!((std - expected).abs() < 1e-6);
    }

    #[test]
    fn test_std_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(std_simd(&x.view(), 0), None);
    }

    #[test]
    fn test_std_simd_constant() {
        let x = array![7.0f64, 7.0, 7.0, 7.0, 7.0];
        let std = std_simd(&x.view(), 0).expect("Operation failed");
        assert!(std.abs() < 1e-10); // Std of constant array is 0
    }

    #[test]
    fn test_sum_simd_f64_basic() {
        let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let sum = sum_simd(&x.view());
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_sum_simd_f32_basic() {
        let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let sum = sum_simd(&x.view());
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_sum_simd_empty() {
        let x: Array1<f64> = array![];
        assert_eq!(sum_simd(&x.view()), 0.0);
    }

    #[test]
    fn test_sum_simd_negative() {
        let x = array![-1.0f64, -2.0, -3.0, -4.0, -5.0];
        let sum = sum_simd(&x.view());
        assert_eq!(sum, -15.0);
    }

    #[test]
    fn test_sum_simd_large() {
        let x: Array1<f64> = Array1::from_vec((1..=10000).map(|i| i as f64).collect());
        let sum = sum_simd(&x.view());
        let expected = 50005000.0; // Sum of 1..=10000
        assert!((sum - expected).abs() < 1e-6);
    }
}
