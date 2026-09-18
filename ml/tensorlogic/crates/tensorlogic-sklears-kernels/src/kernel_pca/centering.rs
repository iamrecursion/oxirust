//! Double-centering utilities for Kernel PCA.
//!
//! Kernel PCA operates on a *centered* Gram matrix
//!
//! ```text
//! K_c = K - 1_n K - K 1_n + 1_n K 1_n
//! ```
//!
//! where `1_n` is the `n x n` matrix of `1/n`. This module exposes both
//! the training-time centering (which also captures per-row means and
//! the grand mean as [`KernelCenteringStats`]) and the test-time
//! centering used during out-of-sample projection.
//!
//! All routines are dense and allocate the result — KPCA is an
//! inherently `O(n^2)` problem, so specialising for sparsity would not
//! change the asymptotic cost.

use scirs2_core::ndarray::{Array1, Array2};

use crate::kernel_pca::error::{KernelPcaError, KernelPcaResult};

/// Training-time centering statistics kept around so that we can center
/// the test-time kernel evaluations against the same reference frame.
///
/// Given a training kernel matrix `K` of shape `(n, n)`:
///
/// * `row_means[j] = (1/n) sum_i K[i, j]` — the mean of column `j`
///   (equivalently, row `j`; `K` is symmetric). These are the means
///   *of the training kernels with every training point*.
/// * `grand_mean = (1/n^2) sum_{i,j} K[i, j]` — the double sum.
///
/// The two statistics are exactly what appears in the transformation
/// applied to test-time kernel evaluations:
///
/// ```text
/// k_c(x, X_j) = k(x, X_j) - row_means[j] - (1/n) sum_i k(x, X_i) + grand_mean
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCenteringStats {
    /// Per-column means of the training Gram matrix, length `n`.
    pub row_means: Array1<f64>,
    /// Scalar grand mean of the training Gram matrix.
    pub grand_mean: f64,
}

impl KernelCenteringStats {
    /// Number of training points `n`.
    pub fn n(&self) -> usize {
        self.row_means.len()
    }
}

/// Double-center a symmetric training Gram matrix in place-free form.
///
/// Returns both the centered matrix and the centering statistics needed
/// later by [`center_test_kernel`].
///
/// # Errors
///
/// * [`KernelPcaError::InvalidInput`] if `k` is empty or non-square.
pub fn double_center(k: &Array2<f64>) -> KernelPcaResult<(Array2<f64>, KernelCenteringStats)> {
    let (rows, cols) = (k.nrows(), k.ncols());
    if rows == 0 || cols == 0 {
        return Err(KernelPcaError::InvalidInput(
            "double_center: Gram matrix must be non-empty".to_string(),
        ));
    }
    if rows != cols {
        return Err(KernelPcaError::InvalidInput(format!(
            "double_center: Gram matrix must be square, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;
    let n_f = n as f64;

    // Per-column means.
    let mut row_means = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += k[(i, j)];
        }
        row_means[j] = s / n_f;
    }

    // Grand mean = mean of the column means (cheaper than re-summing).
    let grand_mean = row_means.iter().copied().sum::<f64>() / n_f;

    // Centered matrix.
    let mut centered = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            centered[(i, j)] = k[(i, j)] - row_means[i] - row_means[j] + grand_mean;
        }
    }

    // Re-symmetrise to absorb f64 rounding before feeding eigh (which
    // insists on exact symmetry).
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (centered[(i, j)] + centered[(j, i)]);
            centered[(i, j)] = avg;
            centered[(j, i)] = avg;
        }
    }

    Ok((
        centered,
        KernelCenteringStats {
            row_means,
            grand_mean,
        },
    ))
}

/// Center a row of test-time kernel evaluations
/// `k_test[j] = k(x_new, X_j)` against the training-time statistics.
///
/// The returned `Array1<f64>` is the centered vector used by
/// [`crate::kernel_pca::FittedKernelPCA::transform`] to compute the projection
/// of `x_new` onto the top-k eigenvectors of `K_c`.
///
/// # Errors
///
/// * [`KernelPcaError::DimensionMismatch`] if `k_test.len() != stats.n()`.
pub fn center_test_kernel(
    k_test: &[f64],
    stats: &KernelCenteringStats,
) -> KernelPcaResult<Array1<f64>> {
    let n = stats.n();
    if k_test.len() != n {
        return Err(KernelPcaError::DimensionMismatch {
            expected: n,
            got: k_test.len(),
            context: "center_test_kernel: test kernel row length".to_string(),
        });
    }

    let n_f = n as f64;
    let row_mean_test = k_test.iter().copied().sum::<f64>() / n_f;

    let mut out = Array1::<f64>::zeros(n);
    for j in 0..n {
        out[j] = k_test[j] - stats.row_means[j] - row_mean_test + stats.grand_mean;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_matrix(n: usize, value: f64) -> Array2<f64> {
        Array2::<f64>::from_shape_fn((n, n), |_| value)
    }

    #[test]
    fn double_center_rejects_empty_matrix() {
        let k = Array2::<f64>::zeros((0, 0));
        assert!(double_center(&k).is_err());
    }

    #[test]
    fn double_center_rejects_non_square() {
        let k = Array2::<f64>::zeros((3, 4));
        assert!(double_center(&k).is_err());
    }

    #[test]
    fn double_center_of_constant_matrix_is_zero() {
        // A constant matrix has per-column means equal to the constant,
        // grand mean equal to the constant, and hence every entry of the
        // centered matrix is 0.
        let k = constant_matrix(5, 3.7);
        let (centered, stats) = double_center(&k).expect("double_center");
        for v in centered.iter() {
            assert!(v.abs() < 1e-12, "expected zero, got {}", v);
        }
        assert_eq!(stats.row_means.len(), 5);
        for &rm in stats.row_means.iter() {
            assert!((rm - 3.7).abs() < 1e-12);
        }
        assert!((stats.grand_mean - 3.7).abs() < 1e-12);
    }

    #[test]
    fn double_center_row_column_sums_are_zero() {
        // The canonical post-centering invariant: each row and column
        // must sum to zero up to numerical roundoff.
        let k = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| ((i + 1) as f64) * ((j + 1) as f64));
        let (centered, _) = double_center(&k).expect("double_center");
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| centered[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-10, "row {} sum = {}", i, row_sum);
        }
        for j in 0..4 {
            let col_sum: f64 = (0..4).map(|i| centered[(i, j)]).sum();
            assert!(col_sum.abs() < 1e-10, "col {} sum = {}", j, col_sum);
        }
    }

    #[test]
    fn double_center_is_symmetric() {
        // The centered matrix must be exactly symmetric (same type is
        // required by scirs2_linalg::eigh downstream).
        let k = Array2::<f64>::from_shape_fn((6, 6), |(i, j)| {
            // Make a symmetric but varied K.
            let a = (i as f64).sin();
            let b = (j as f64).sin();
            1.0 + a + b + 0.5 * (a * b)
        });
        let (centered, _) = double_center(&k).expect("double_center");
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (centered[(i, j)] - centered[(j, i)]).abs() < 1e-14,
                    "asymmetry at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn center_test_kernel_matches_pulling_row_from_double_center() {
        // Sanity: centering a training row using `center_test_kernel`
        // must yield the same result as taking that row from the
        // double-centered training Gram matrix.
        let k = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| {
            // symmetric positive function
            (-((i as f64 - j as f64).powi(2)) / 4.0).exp()
        });
        let (centered, stats) = double_center(&k).expect("double_center");
        for i in 0..4 {
            let test_row: Vec<f64> = (0..4).map(|j| k[(i, j)]).collect();
            let out = center_test_kernel(&test_row, &stats).expect("center_test_kernel");
            for j in 0..4 {
                assert!(
                    (out[j] - centered[(i, j)]).abs() < 1e-12,
                    "row {} col {} mismatch: test={}, expected={}",
                    i,
                    j,
                    out[j],
                    centered[(i, j)]
                );
            }
        }
    }

    #[test]
    fn center_test_kernel_rejects_wrong_length() {
        let stats = KernelCenteringStats {
            row_means: Array1::<f64>::zeros(3),
            grand_mean: 0.0,
        };
        let err = center_test_kernel(&[1.0, 2.0], &stats).expect_err("must reject");
        match err {
            KernelPcaError::DimensionMismatch { expected, got, .. } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }
}
