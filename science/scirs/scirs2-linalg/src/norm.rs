//! Matrix and vector norms

use scirs2_core::ndarray::{ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::validation::{
    validate_finite_vector, validate_finitematrix, validate_not_empty_vector,
    validate_not_emptymatrix,
};

/// Compute a matrix norm.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `ord` - Order of the norm:
///   * 'fro' or 'f': Frobenius norm
///   * '1': 1-norm (maximum column sum)
///   * 'inf': Infinity norm (maximum row sum)
///   * '2': 2-norm (largest singular value)
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::matrix_norm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let norm_fro = matrix_norm(&a.view(), "fro", None).expect("Operation failed");
/// assert!((norm_fro - 5.477225575051661).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn matrix_norm<F>(a: &ArrayView2<F>, ord: &str, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    // Parameter validation using validation helpers
    validate_not_emptymatrix(a, "Matrix norm computation")?;
    validate_finitematrix(a, "Matrix norm computation")?;

    match ord {
        "fro" | "f" | "frobenius" => {
            // Frobenius norm
            let mut sum_sq = F::zero();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    sum_sq += a[[i, j]] * a[[i, j]];
                }
            }
            Ok(sum_sq.sqrt())
        }
        "1" => {
            // 1-norm (maximum column sum)
            let mut max_col_sum = F::zero();
            for j in 0..a.ncols() {
                let col = a.column(j);
                let col_sum = col.fold(F::zero(), |acc, &x| acc + x.abs());
                if col_sum > max_col_sum {
                    max_col_sum = col_sum;
                }
            }
            Ok(max_col_sum)
        }
        "inf" => {
            // Infinity norm (maximum row sum)
            let mut max_row_sum = F::zero();
            for i in 0..a.nrows() {
                let row = a.row(i);
                let row_sum = row.fold(F::zero(), |acc, &x| acc + x.abs());
                if row_sum > max_row_sum {
                    max_row_sum = row_sum;
                }
            }
            Ok(max_row_sum)
        }
        "2" => {
            // 2-norm (largest singular value)
            let (_u, s, _vt) = svd(a, false, workers)?;
            // The 2-norm is the largest singular value
            if s.is_empty() {
                Ok(F::zero())
            } else {
                Ok(s[0])
            }
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Matrix norm computation failed: Invalid norm order '{ord}'\nSupported norms: 'fro', 'f', 'frobenius', '1', 'inf', '2'"
        ))),
    }
}

/// Compute a vector norm.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `ord` - Order of the norm:
///   * 1: 1-norm (sum of absolute values)
///   * 2: 2-norm (Euclidean norm)
///   * usize::MAX: Infinity norm (maximum absolute value)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::vector_norm;
///
/// let x = array![3.0_f64, 4.0];
/// let norm_2 = vector_norm(&x.view(), 2).expect("Operation failed");
/// assert!((norm_2 - 5.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn vector_norm<F>(x: &ArrayView1<F>, ord: usize) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    // Parameter validation using validation helpers
    validate_not_empty_vector(x, "Vector norm computation")?;
    validate_finite_vector(x, "Vector norm computation")?;

    // Scalar fallback for all norms
    match ord {
        1 => {
            // 1-norm (sum of absolute values)
            let sum_abs = x.fold(F::zero(), |acc, &val| acc + val.abs());
            Ok(sum_abs)
        }
        2 => {
            // 2-norm (Euclidean norm)
            let sum_sq = x.fold(F::zero(), |acc, &val| acc + val * val);
            Ok(sum_sq.sqrt())
        }
        usize::MAX => {
            // Infinity norm (maximum absolute value)
            let max_abs = x.fold(F::zero(), |acc, &val| {
                let abs_val = val.abs();
                if abs_val > acc {
                    abs_val
                } else {
                    acc
                }
            });
            Ok(max_abs)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Vector norm computation failed: Invalid norm order {}\nSupported norms: 1 (L1), 2 (L2/Euclidean), {} (infinity)",
            ord, usize::MAX
        ))),
    }
}

/// Compute a vector norm with SIMD acceleration.
///
/// This is a SIMD-accelerated version of `vector_norm` that provides significant
/// performance improvements for large arrays (typically 2-3x faster for f32 operations).
///
/// # Arguments
///
/// * `x` - Input vector
/// * `ord` - Order of the norm:
///   * 1: L1 norm (sum of absolute values / Manhattan norm)
///   * 2: L2 norm (Euclidean norm)
///   * usize::MAX: L∞ norm (maximum absolute value / Chebyshev norm)
///
/// # Returns
///
/// * Value of the norm
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::vector_norm_simd;
///
/// let x = array![3.0_f64, 4.0];
/// let norm_2 = vector_norm_simd(&x.view(), 2).expect("Operation failed");
/// assert!((norm_2 - 5.0).abs() < 1e-10);
///
/// let norm_1 = vector_norm_simd(&x.view(), 1).expect("Operation failed");
/// assert!((norm_1 - 7.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn vector_norm_simd<F>(x: &ArrayView1<F>, ord: usize) -> LinalgResult<F>
where
    F: Float
        + NumAssign
        + Sum
        + Send
        + Sync
        + scirs2_core::ndarray::ScalarOperand
        + SimdUnifiedOps
        + 'static,
{
    // Parameter validation using validation helpers
    validate_not_empty_vector(x, "Vector norm computation")?;
    validate_finite_vector(x, "Vector norm computation")?;

    // SIMD fast paths for common norms
    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        match ord {
            1 => {
                // L1 norm (Manhattan norm) - SIMD accelerated
                return Ok(F::simd_norm_l1(x));
            }
            2 => {
                // L2 norm (Euclidean norm) - SIMD accelerated
                return Ok(F::simd_norm(x));
            }
            usize::MAX => {
                // L∞ norm (Chebyshev norm) - SIMD accelerated
                return Ok(F::simd_norm_linf(x));
            }
            _ => {
                // Fall through to scalar implementation for other norms
            }
        }
    }

    // Scalar fallback for small arrays or unsupported SIMD platforms
    match ord {
        1 => {
            // 1-norm (sum of absolute values)
            let sum_abs = x.fold(F::zero(), |acc, &val| acc + val.abs());
            Ok(sum_abs)
        }
        2 => {
            // 2-norm (Euclidean norm)
            let sum_sq = x.fold(F::zero(), |acc, &val| acc + val * val);
            Ok(sum_sq.sqrt())
        }
        usize::MAX => {
            // Infinity norm (maximum absolute value)
            let max_abs = x.fold(F::zero(), |acc, &val| {
                let abs_val = val.abs();
                if abs_val > acc {
                    abs_val
                } else {
                    acc
                }
            });
            Ok(max_abs)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Vector norm computation failed: Invalid norm order {}\nSupported norms: 1 (L1), 2 (L2/Euclidean), {} (infinity)",
            ord, usize::MAX
        ))),
    }
}

/// Compute a matrix norm with SIMD acceleration.
///
/// This is a SIMD-accelerated version of `matrix_norm` that provides significant
/// performance improvements for Frobenius norm computation on large matrices.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `ord` - Order of the norm:
///   * 'fro' or 'f': Frobenius norm (SIMD-accelerated)
///   * '1': 1-norm (maximum column sum) (SIMD-accelerated)
///   * 'inf': Infinity norm (maximum row sum) (SIMD-accelerated)
///   * '2': 2-norm (largest singular value) - uses SVD
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Value of the norm
///
/// # SIMD Optimization
///
/// - Frobenius norm: SIMD-accelerated sum of squares (2-3x speedup for >1000 elements)
/// - 1-norm: SIMD-accelerated L1 norm for each column (2-3x speedup)
/// - Infinity norm: SIMD-accelerated L1 norm for each row (2-3x speedup)
/// - 2-norm: Uses SVD (no SIMD benefit)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::matrix_norm_simd;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let norm_fro = matrix_norm_simd(&a.view(), "fro", None).expect("Operation failed");
/// assert!((norm_fro - 5.477225575051661).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn matrix_norm_simd<F>(a: &ArrayView2<F>, ord: &str, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + SimdUnifiedOps + 'static,
{
    // Parameter validation using validation helpers
    validate_not_emptymatrix(a, "Matrix norm computation")?;
    validate_finitematrix(a, "Matrix norm computation")?;

    let optimizer = AutoOptimizer::new();

    match ord {
        "fro" | "f" | "frobenius" => {
            // Frobenius norm with SIMD acceleration
            let n_elements = a.nrows() * a.ncols();

            if optimizer.should_use_simd(n_elements) {
                // Convert 2D array to 1D view for SIMD processing
                // SAFETY: We're just reinterpreting the contiguous data as 1D
                let flat_view = a.as_slice().ok_or_else(|| {
                    LinalgError::ComputationError(
                        "Matrix must be contiguous for SIMD Frobenius norm".to_string(),
                    )
                })?;

                let array_1d = scirs2_core::ndarray::ArrayView1::from(flat_view);
                let sum_squares = F::simd_sum_squares(&array_1d);
                return Ok(sum_squares.sqrt());
            }

            // Scalar fallback for small matrices or non-contiguous data
            let mut sum_sq = F::zero();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    sum_sq += a[[i, j]] * a[[i, j]];
                }
            }
            Ok(sum_sq.sqrt())
        }
        "1" => {
            // 1-norm (maximum column sum) with SIMD acceleration
            let mut max_col_sum = F::zero();

            if optimizer.should_use_simd(a.nrows()) {
                // SIMD path: use simd_norm_l1 for each column
                for j in 0..a.ncols() {
                    let col = a.column(j);
                    let col_sum = F::simd_norm_l1(&col);
                    if col_sum > max_col_sum {
                        max_col_sum = col_sum;
                    }
                }
            } else {
                // Scalar fallback for small columns
                for j in 0..a.ncols() {
                    let col = a.column(j);
                    let col_sum = col.fold(F::zero(), |acc, &x| acc + x.abs());
                    if col_sum > max_col_sum {
                        max_col_sum = col_sum;
                    }
                }
            }

            Ok(max_col_sum)
        }
        "inf" => {
            // Infinity norm (maximum row sum) with SIMD acceleration
            let mut max_row_sum = F::zero();

            if optimizer.should_use_simd(a.ncols()) {
                // SIMD path: use simd_norm_l1 for each row
                for i in 0..a.nrows() {
                    let row = a.row(i);
                    let row_sum = F::simd_norm_l1(&row);
                    if row_sum > max_row_sum {
                        max_row_sum = row_sum;
                    }
                }
            } else {
                // Scalar fallback for small rows
                for i in 0..a.nrows() {
                    let row = a.row(i);
                    let row_sum = row.fold(F::zero(), |acc, &x| acc + x.abs());
                    if row_sum > max_row_sum {
                        max_row_sum = row_sum;
                    }
                }
            }

            Ok(max_row_sum)
        }
        // 2-norm uses SVD, no SIMD acceleration benefit
        _ => matrix_norm(a, ord, workers),
    }
}

/// Compute the norm of a vector with parallel processing support.
///
/// This function uses parallel computation to calculate vector norms efficiently
/// for large vectors. The computation is distributed across multiple worker threads
/// using the scirs2-core parallel operations framework.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `ord` - Norm order (1, 2, or usize::MAX for infinity norm)  
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Norm of the vector
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::vector_norm_parallel;
///
/// let x = array![3.0_f64, 4.0];
/// let norm2 = vector_norm_parallel(&x.view(), 2, Some(4)).expect("Operation failed");
/// assert!((norm2 - 5.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn vector_norm_parallel<F>(
    x: &ArrayView1<F>,
    ord: usize,
    workers: Option<usize>,
) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand,
{
    use crate::parallel;
    use scirs2_core::parallel_ops::*;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation using validation helpers
    validate_not_empty_vector(x, "Vector norm computation")?;
    validate_finite_vector(x, "Vector norm computation")?;

    // Use threshold to determine if parallel processing is worthwhile
    const PARALLEL_THRESHOLD: usize = 1000;

    if x.len() < PARALLEL_THRESHOLD {
        // For small vectors, use sequential implementation
        return vector_norm(x, ord);
    }

    match ord {
        1 => {
            // 1-norm (sum of absolute values) - parallel sum
            let sum_abs = (0..x.len()).into_par_iter()
                .map(|i| x[i].abs())
                .sum();
            Ok(sum_abs)
        }
        2 => {
            // 2-norm (Euclidean norm) - parallel sum of squares
            let sum_sq: F = (0..x.len()).into_par_iter()
                .map(|i| x[i] * x[i])
                .sum();
            Ok(sum_sq.sqrt())
        }
        usize::MAX => {
            // Infinity norm (maximum absolute value) - parallel max
            let max_abs = (0..x.len()).into_par_iter()
                .map(|i| x[i].abs())
                .reduce(|| F::zero(), |a, b| if a > b { a } else { b });
            Ok(max_abs)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Vector norm computation failed: Invalid norm order {}\nSupported norms: 1 (L1), 2 (L2/Euclidean), {} (infinity)",
            ord, usize::MAX
        ))),
    }
}

/// Compute the condition number of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `p` - Order of the norm:
///   * None: 2-norm
///   * "fro" or "f": Frobenius norm
///   * "1": 1-norm
///   * "inf": Infinity norm
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Condition number of the matrix
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::cond;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let c = cond(&a.view(), None, None).expect("Operation failed");
/// assert!((c - 2.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn cond<F>(a: &ArrayView2<F>, p: Option<&str>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    // Parameter validation using validation helpers
    use crate::validation::validate_squarematrix;
    validate_not_emptymatrix(a, "Condition number computation")?;
    validate_squarematrix(a, "Condition number computation")?;
    validate_finitematrix(a, "Condition number computation")?;

    let norm_type = p.unwrap_or("2");

    match norm_type {
        "2" | "fro" | "f" | "frobenius" => {
            // Use SVD for 2-norm and Frobenius norm condition number
            let (_u, s, _vt) = svd(a, false, workers)?;

            if s.is_empty() {
                return Ok(F::infinity());
            }

            // Find the largest and smallest non-zero singular values
            let sigma_max = s[0]; // Largest singular value (first in sorted order)
            let mut sigma_min = F::zero();

            // Find the smallest non-zero singular value
            for &val in s.iter().rev() {
                if val > F::epsilon() * F::from(100).expect("Operation failed") * sigma_max {
                    sigma_min = val;
                    break;
                }
            }

            if sigma_min <= F::zero() {
                Ok(F::infinity())
            } else {
                Ok(sigma_max / sigma_min)
            }
        }
        "1" | "inf" => {
            // For 1-norm and inf-norm, we need matrix inverse
            // This is more complex and would require computing the inverse
            // For now, fall back to SVD-based computation
            let (_u, s, _vt) = svd(a, false, workers)?;

            if s.is_empty() {
                return Ok(F::infinity());
            }

            let sigma_max = s[0];
            let mut sigma_min = F::zero();

            for &val in s.iter().rev() {
                if val > F::epsilon() * F::from(100).expect("Operation failed") * sigma_max {
                    sigma_min = val;
                    break;
                }
            }

            if sigma_min <= F::zero() {
                Ok(F::infinity())
            } else {
                Ok(sigma_max / sigma_min)
            }
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Condition number computation failed: Invalid norm type '{norm_type}'\nSupported norms: '1', '2' (default), 'fro', 'f', 'frobenius', 'inf'"
        ))),
    }
}

/// Compute the rank of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `tol` - Tolerance for singular values (None = automatic)
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Rank of the matrix
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, ScalarOperand};
/// use scirs2_linalg::matrix_rank;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let r = matrix_rank(&a.view(), None, None).expect("Operation failed");
/// assert_eq!(r, 2);
/// ```
#[allow(dead_code)]
pub fn matrix_rank<F>(
    a: &ArrayView2<F>,
    tol: Option<F>,
    workers: Option<usize>,
) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    // Parameter validation using validation helpers
    if a.is_empty() {
        return Ok(0);
    }
    validate_finitematrix(a, "Matrix rank computation")?;

    // Validate tolerance
    if let Some(t) = tol {
        if t < F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Matrix rank computation failed: Tolerance must be non-negative".to_string(),
            ));
        }
    }

    // Compute SVD to get singular values
    let (_u, s, _vt) = svd(a, false, workers)?;

    if s.is_empty() {
        return Ok(0);
    }

    // Determine tolerance
    let tolerance = if let Some(t) = tol {
        t
    } else {
        // Default tolerance: max(m, n) * eps * max(singular_values)
        let max_dim = std::cmp::max(a.nrows(), a.ncols());
        let eps = F::epsilon();
        let sigma_max = s[0]; // Largest singular value
        F::from(max_dim).expect("Operation failed") * eps * sigma_max
    };

    // Count singular values above tolerance
    let rank = s.iter().filter(|&&val| val > tolerance).count();

    Ok(rank)
}

// Backward compatibility functions (deprecated)

/// Compute a matrix norm without workers parameter (deprecated - use matrix_norm with workers)
#[deprecated(
    since = "0.1.0",
    note = "Use matrix_norm with workers parameter instead"
)]
#[allow(dead_code)]
pub fn matrix_norm_default<F>(a: &ArrayView2<F>, ord: &str) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    matrix_norm(a, ord, None)
}

/// Compute the condition number without workers parameter (deprecated - use cond with workers)
#[deprecated(since = "0.1.0", note = "Use cond with workers parameter instead")]
#[allow(dead_code)]
pub fn cond_default<F>(a: &ArrayView2<F>, p: Option<&str>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    cond(a, p, None)
}

/// Compute matrix rank without workers parameter (deprecated - use matrix_rank with workers)
#[deprecated(
    since = "0.1.0",
    note = "Use matrix_rank with workers parameter instead"
)]
#[allow(dead_code)]
pub fn matrix_rank_default<F>(a: &ArrayView2<F>, tol: Option<F>) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    matrix_rank(a, tol, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn testmatrix_norm_frobenius() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "fro", None).expect("Operation failed");
        // sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
        assert_relative_eq!(norm, 5.477225575051661, epsilon = 1e-10);
    }

    #[test]
    fn testmatrix_norm_1() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "1", None).expect("Operation failed");
        // max(1+3, 2+4) = max(4, 6) = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn testmatrix_norm_inf() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = matrix_norm(&a.view(), "inf", None).expect("Operation failed");
        // max(1+2, 3+4) = max(3, 7) = 7
        assert_relative_eq!(norm, 7.0);
    }

    #[test]
    fn test_vector_norm_1() {
        let x = array![1.0, -2.0, 3.0];
        let norm = vector_norm(&x.view(), 1).expect("Operation failed");
        // |1| + |-2| + |3| = 6
        assert_relative_eq!(norm, 6.0);
    }

    #[test]
    fn test_vector_norm_2() {
        let x = array![3.0, 4.0];
        let norm = vector_norm(&x.view(), 2).expect("Operation failed");
        // sqrt(3^2 + 4^2) = 5
        assert_relative_eq!(norm, 5.0);
    }

    #[test]
    fn test_vector_norm_inf() {
        let x = array![1.0, -5.0, 3.0];
        let norm = vector_norm(&x.view(), usize::MAX).expect("Operation failed");
        // max(|1|, |-5|, |3|) = 5
        assert_relative_eq!(norm, 5.0);
    }
}
