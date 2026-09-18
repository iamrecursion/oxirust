//! Wrappers around [`scirs2_linalg::eigh`] tailored to Kernel PCA.
//!
//! `eigh` returns eigenvalues in an unspecified order (see its rustdoc
//! example), so this module is responsible for:
//!
//! 1. Calling `eigh` on the centered Gram matrix,
//! 2. Sorting the resulting `(eigenvalue, eigenvector)` pairs in
//!    *descending* order of eigenvalue,
//! 3. Flooring tiny-negative eigenvalues back to `0` — they come from
//!    numerical error, not from the model, and downstream code divides
//!    by `sqrt(eigenvalue)` so we must never let it see a negative
//!    number,
//! 4. Selecting the top `n_components` components and making sure we
//!    have enough strictly-positive eigenvalues to satisfy the request.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::eigh;

use crate::kernel_pca::error::{KernelPcaError, KernelPcaResult};

/// Eigenvalues smaller than this in absolute value are treated as zero
/// — they come from floating-point noise around degenerate or
/// low-rank kernels.
const POSITIVITY_FLOOR: f64 = 1e-10;

/// Eigen-pair bundle returned by [`symmetric_eigendecomp`] — top-k
/// eigenvalues sorted in descending order and the matching eigenvector
/// columns.
#[derive(Debug, Clone)]
pub struct TopKEigen {
    /// Eigenvalues, length `n_components`, strictly positive and
    /// sorted in descending order.
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors — columns `j` is the eigenvector associated with
    /// `eigenvalues[j]`. Shape is `(n, n_components)`.
    pub eigenvectors: Array2<f64>,
}

/// Compute the top-`n_components` eigenvalue/eigenvector pairs of a
/// symmetric matrix, sorted in descending order of eigenvalue.
///
/// # Errors
///
/// * [`KernelPcaError::InvalidInput`] if `n_components == 0` or
///   `n_components > matrix.nrows()`.
/// * [`KernelPcaError::EigendecompositionFailed`] if the underlying
///   `eigh` call fails.
/// * [`KernelPcaError::InsufficientComponents`] if fewer than
///   `n_components` eigenvalues survive the positivity floor.
pub fn symmetric_eigendecomp(
    matrix: &Array2<f64>,
    n_components: usize,
) -> KernelPcaResult<TopKEigen> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());
    if rows != cols {
        return Err(KernelPcaError::InvalidInput(format!(
            "symmetric_eigendecomp: matrix must be square, got {}x{}",
            rows, cols
        )));
    }
    if n_components == 0 {
        return Err(KernelPcaError::InvalidInput(
            "symmetric_eigendecomp: n_components must be >= 1".to_string(),
        ));
    }
    if n_components > rows {
        return Err(KernelPcaError::InvalidInput(format!(
            "symmetric_eigendecomp: n_components ({}) cannot exceed matrix size ({})",
            n_components, rows
        )));
    }

    let (raw_eigenvalues, raw_eigenvectors) = eigh(&matrix.view(), None)
        .map_err(|e| KernelPcaError::EigendecompositionFailed(e.to_string()))?;

    let n = raw_eigenvalues.len();
    if n != rows {
        return Err(KernelPcaError::EigendecompositionFailed(format!(
            "eigh returned {} eigenvalues for a {}x{} matrix",
            n, rows, cols
        )));
    }
    if raw_eigenvectors.nrows() != rows || raw_eigenvectors.ncols() != n {
        return Err(KernelPcaError::EigendecompositionFailed(format!(
            "eigh returned eigenvector matrix with shape {}x{} (expected {}x{})",
            raw_eigenvectors.nrows(),
            raw_eigenvectors.ncols(),
            rows,
            n
        )));
    }

    // Pair (eigenvalue, original_column_index) and sort by eigenvalue
    // in DESCENDING order. NaNs are treated as less-than so they sink
    // to the bottom and never end up among the chosen components.
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|idx| (raw_eigenvalues[idx], idx)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // How many survive the positivity floor?
    let available = indexed
        .iter()
        .filter(|&&(v, _)| v > POSITIVITY_FLOOR)
        .count();
    if available < n_components {
        return Err(KernelPcaError::InsufficientComponents {
            requested: n_components,
            available,
        });
    }

    // Assemble the top-k bundle.
    let mut eigenvalues = Array1::<f64>::zeros(n_components);
    let mut eigenvectors = Array2::<f64>::zeros((rows, n_components));
    for (k, &(lambda, col)) in indexed.iter().take(n_components).enumerate() {
        // Floor any borderline-positive roundoff to exactly zero is
        // unnecessary here — we already checked `> POSITIVITY_FLOOR`.
        eigenvalues[k] = lambda;
        for i in 0..rows {
            eigenvectors[(i, k)] = raw_eigenvectors[(i, col)];
        }
    }

    Ok(TopKEigen {
        eigenvalues,
        eigenvectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix_eigenvalues_are_sorted_descending() {
        // Diag(2, 5, 1, 8) — the top-2 descending should be 8, 5.
        let m = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| {
            if i == j {
                [2.0, 5.0, 1.0, 8.0][i]
            } else {
                0.0
            }
        });
        let res = symmetric_eigendecomp(&m, 2).expect("eigendecomp");
        assert!((res.eigenvalues[0] - 8.0).abs() < 1e-10);
        assert!((res.eigenvalues[1] - 5.0).abs() < 1e-10);
        // Eigenvalues must be in strictly decreasing order.
        assert!(res.eigenvalues[0] >= res.eigenvalues[1]);
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        // Build a PSD symmetric matrix as A^T A for a random-ish A.
        let a = Array2::<f64>::from_shape_fn((5, 5), |(i, j)| {
            (i as f64 * 0.3 + j as f64 * 0.7 + 1.0).sin()
        });
        // M = A^T A
        let mut m = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                let mut s = 0.0;
                for k in 0..5 {
                    s += a[(k, i)] * a[(k, j)];
                }
                m[(i, j)] = s;
            }
        }
        // Symmetrize to absorb float noise.
        for i in 0..5 {
            for j in (i + 1)..5 {
                let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
                m[(i, j)] = avg;
                m[(j, i)] = avg;
            }
        }
        let k = 2;
        let res = symmetric_eigendecomp(&m, k).expect("eigendecomp");
        // Each column must have unit norm.
        for c in 0..k {
            let norm_sq: f64 = (0..5).map(|r| res.eigenvectors[(r, c)].powi(2)).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-6,
                "col {} norm_sq = {}",
                c,
                norm_sq
            );
        }
        // Different columns must be (nearly) orthogonal.
        for c1 in 0..k {
            for c2 in (c1 + 1)..k {
                let dot: f64 = (0..5)
                    .map(|r| res.eigenvectors[(r, c1)] * res.eigenvectors[(r, c2)])
                    .sum();
                assert!(dot.abs() < 1e-6, "cols {}/{} dot = {}", c1, c2, dot);
            }
        }
    }

    #[test]
    fn rejects_n_components_zero_or_too_large() {
        let m = Array2::<f64>::eye(3);
        assert!(symmetric_eigendecomp(&m, 0).is_err());
        assert!(symmetric_eigendecomp(&m, 4).is_err());
    }

    #[test]
    fn insufficient_components_on_low_rank_matrix() {
        // Rank-1 PSD matrix: e.g. outer product [1,1,1][1,1,1]^T.
        let m = Array2::<f64>::from_shape_fn((3, 3), |_| 1.0);
        // Rank 1 -> only 1 positive eigenvalue, the other two ~ 0.
        let err = symmetric_eigendecomp(&m, 2).expect_err("must fail");
        match err {
            KernelPcaError::InsufficientComponents {
                requested,
                available,
            } => {
                assert_eq!(requested, 2);
                assert!(available <= 1, "available = {}", available);
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }
}
