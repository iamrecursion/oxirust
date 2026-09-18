//! Mixed CPU/GPU preconditioning for sparse linear systems.
//!
//! ILU(0) factorization runs on CPU; SpMV applies the preconditioner.
//! The mixed strategy exploits CPU's suitability for sequential triangular solves
//! and GPU's throughput for parallel sparse matrix-vector products.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. §10.3.
//! - Benzi, M. (2002). Preconditioning techniques for large linear systems. J. Comput. Phys.

use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray::Array1;

// ---------------------------------------------------------------------------
// Lightweight f64 CSR used internally by this module.
// We deliberately keep this separate from the generic `csr::CsrMatrix<T>` to
// avoid the heavyweight trait bounds and stay self-contained.
// ---------------------------------------------------------------------------

/// Sparse matrix in CSR format (row-major compressed), f64 values only.
///
/// This is a lightweight internal type used by the GPU preconditioner.
/// For the general sparse matrix API use [`crate::csr::CsrMatrix`].
#[derive(Debug, Clone)]
pub struct GpuPrecondCsr {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointers: `row_ptr[i]..row_ptr[i+1]` indexes row `i`.
    pub row_ptr: Vec<usize>,
    /// Column indices of non-zeros.
    pub col_idx: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f64>,
}

impl GpuPrecondCsr {
    /// Create from raw CSR data.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Create from a dense `ndarray::Array2<f64>` (useful for testing).
    pub fn from_dense(dense: &scirs2_core::ndarray::Array2<f64>) -> Self {
        let (m, n) = (dense.nrows(), dense.ncols());
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let v = dense[[i, j]];
                if v.abs() > 1e-14 {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Self {
            nrows: m,
            ncols: n,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Sparse matrix-vector product `y = A * x`.
    pub fn spmv(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(self.nrows);
        for i in 0..self.nrows {
            let mut sum = 0.0_f64;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = sum;
        }
        y
    }

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// ILU(0) factorization
// ---------------------------------------------------------------------------

/// ILU(0) preconditioner — zero fill-in incomplete LU factorization.
///
/// Computes factors L and U such that LU ≈ A on the same sparsity pattern
/// as A.  The factorization is performed row-by-row (Crout variant).
pub struct Ilu0Preconditioner {
    n: usize,
    /// L factor values stored in the combined `row_ptr`/`col_idx` layout.
    /// Entries with `col_idx[k] < i` are L; `col_idx[k] == i` is L diagonal
    /// (always 1.0); entries with `col_idx[k] > i` are 0.0 here.
    l_values: Vec<f64>,
    /// U factor values stored in the same layout.
    /// Entries with `col_idx[k] >= i` are U; `col_idx[k] < i` is 0.0 here.
    u_values: Vec<f64>,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl Ilu0Preconditioner {
    /// Compute ILU(0) factorization of `a` (must be square with non-zero diagonal).
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ValueError`] if the matrix is non-square or if a
    /// zero pivot is encountered during factorization.
    pub fn compute(a: &GpuPrecondCsr) -> SparseResult<Self> {
        if a.nrows != a.ncols {
            return Err(SparseError::ValueError(
                "ILU(0) requires a square matrix".to_string(),
            ));
        }
        let n = a.nrows;
        if n == 0 {
            return Ok(Self {
                n,
                l_values: Vec::new(),
                u_values: Vec::new(),
                row_ptr: vec![0],
                col_idx: Vec::new(),
            });
        }

        // Working copy of values; we overwrite in-place during elimination.
        let mut values = a.values.clone();
        let row_ptr = a.row_ptr.clone();
        let col_idx = a.col_idx.clone();

        // Locate diagonal entries for each row (needed for pivot access).
        let mut diag_idx = vec![usize::MAX; n];
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i {
                    diag_idx[i] = k;
                    break;
                }
            }
            if diag_idx[i] == usize::MAX {
                return Err(SparseError::ValueError(format!(
                    "ILU(0): missing diagonal entry in row {i}"
                )));
            }
        }

        // Row-by-row ILU(0) Gaussian elimination (keeps original sparsity).
        for i in 1..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                let j = col_idx[k];
                if j >= i {
                    break; // columns are sorted; strictly lower part is done
                }
                let u_jj = values[diag_idx[j]];
                if u_jj.abs() < 1e-300 {
                    return Err(SparseError::ValueError(format!(
                        "ILU(0): zero pivot at diagonal ({j},{j})"
                    )));
                }
                // L[i,j] = A[i,j] / U[j,j]
                values[k] /= u_jj;
                let l_ij = values[k];

                // Eliminate: A[i, col] -= L[i,j] * U[j, col] for col > j
                // walking row i and row j in lock-step (both sorted by column).
                let mut ki = k + 1; // position in row i, col > j
                let mut kj = diag_idx[j] + 1; // position in row j, col > j
                while ki < row_ptr[i + 1] && kj < row_ptr[j + 1] {
                    let ci = col_idx[ki];
                    let cj = col_idx[kj];
                    match ci.cmp(&cj) {
                        std::cmp::Ordering::Equal => {
                            // Pattern match: both rows have column ci=cj; subtract.
                            values[ki] -= l_ij * values[kj];
                            ki += 1;
                            kj += 1;
                        }
                        std::cmp::Ordering::Less => ki += 1,
                        std::cmp::Ordering::Greater => kj += 1,
                    }
                }
            }
        }

        // Split into L (lower, unit diagonal) and U (upper + diagonal).
        let nnz = a.nnz();
        let mut l_values = vec![0.0_f64; nnz];
        let mut u_values = vec![0.0_f64; nnz];

        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                let j = col_idx[k];
                if j < i {
                    l_values[k] = values[k]; // L factor
                } else if j == i {
                    l_values[k] = 1.0; // unit diagonal for L
                    u_values[k] = values[k]; // diagonal of U
                } else {
                    u_values[k] = values[k]; // strictly upper part of U
                }
            }
        }

        Ok(Self {
            n,
            l_values,
            u_values,
            row_ptr,
            col_idx,
        })
    }

    /// Apply preconditioner: solve `(LU) z = r`.
    ///
    /// Performs forward solve `Ly = r` then backward solve `Uz = y`.
    pub fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        // Forward solve: L y = r  (unit lower triangular)
        let mut y = r.clone();
        for i in 0..self.n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                if j < i {
                    let ly = self.l_values[k] * y[j];
                    y[i] -= ly;
                }
            }
        }

        // Backward solve: U z = y  (upper triangular with explicit diagonal)
        let mut z = y;
        for ii in 0..self.n {
            let i = self.n - 1 - ii;
            let mut diag_val = 1.0_f64;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                if j > i {
                    let uz = self.u_values[k] * z[j];
                    z[i] -= uz;
                } else if j == i {
                    diag_val = self.u_values[k];
                }
            }
            if diag_val.abs() > 1e-300 {
                z[i] /= diag_val;
            }
        }
        z
    }
}

// ---------------------------------------------------------------------------
// Mixed preconditioned CG
// ---------------------------------------------------------------------------

/// Mixed CPU/GPU preconditioned conjugate gradient solver.
///
/// The preconditioner (ILU(0)) runs entirely on CPU; SpMV is the hot path
/// that would execute on GPU in a real deployment.  In this pure-Rust
/// implementation SpMV uses [`GpuPrecondCsr::spmv`], which simulates the GPU
/// compute path without an actual GPU dependency.
pub struct MixedPreconditionedCg {
    preconditioner: Ilu0Preconditioner,
    max_iter: usize,
    tol: f64,
}

impl MixedPreconditionedCg {
    /// Create a new solver, pre-computing ILU(0) from matrix `a`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Ilu0Preconditioner::compute`].
    pub fn new(a: &GpuPrecondCsr, max_iter: usize, tol: f64) -> SparseResult<Self> {
        let preconditioner = Ilu0Preconditioner::compute(a)?;
        Ok(Self {
            preconditioner,
            max_iter,
            tol,
        })
    }

    /// Solve `Ax = b` using preconditioned CG (Fletcher-Reeves with ILU(0)).
    ///
    /// Returns `(x, iterations_used)`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ValueError`] if `b.len() != a.nrows`.
    pub fn solve(&self, a: &GpuPrecondCsr, b: &Array1<f64>) -> SparseResult<(Array1<f64>, usize)> {
        let n = b.len();
        if n != a.nrows {
            return Err(SparseError::ValueError(format!(
                "solve: b length {n} does not match matrix nrows {}",
                a.nrows
            )));
        }

        // Initial residual r = b - A*x  (start from zero)
        let mut x = Array1::zeros(n);
        let ax0 = a.spmv(&x);
        let mut r: Array1<f64> = b - &ax0;
        let mut z = self.preconditioner.apply(&r);
        let mut p = z.clone();
        let mut rz = r.dot(&z);

        for iter in 0..self.max_iter {
            let ap = a.spmv(&p);
            let pap = p.dot(&ap);
            if pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let res_norm = r.dot(&r).sqrt();
            if res_norm < self.tol {
                return Ok((x, iter + 1));
            }

            z = self.preconditioner.apply(&r);
            let rz_new = r.dot(&z);
            if rz.abs() < 1e-300 {
                break;
            }
            let beta = rz_new / rz;
            p = &z + &(&p * beta);
            rz = rz_new;
        }

        Ok((x, self.max_iter))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a tridiagonal matrix [-1, 2, -1] of size n×n.
    fn tridiag(n: usize) -> GpuPrecondCsr {
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(-1.0_f64);
            }
            col_idx.push(i);
            values.push(2.0_f64);
            if i + 1 < n {
                col_idx.push(i + 1);
                values.push(-1.0_f64);
            }
            row_ptr.push(col_idx.len());
        }
        GpuPrecondCsr::new(n, n, row_ptr, col_idx, values)
    }

    /// Build a diagonally dominant SPD matrix of size n×n for CG testing.
    fn spd_matrix(n: usize) -> GpuPrecondCsr {
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(-1.0_f64);
            }
            col_idx.push(i);
            // Diagonal is (n + 2.0) to ensure strict diagonal dominance.
            values.push((n as f64) + 2.0);
            if i + 1 < n {
                col_idx.push(i + 1);
                values.push(-1.0_f64);
            }
            row_ptr.push(col_idx.len());
        }
        GpuPrecondCsr::new(n, n, row_ptr, col_idx, values)
    }

    #[test]
    fn test_spmv() {
        let n = 4;
        let a = tridiag(n);
        // x = [1, 1, 1, 1]  => A*x = [1, 0, 0, 1] for tridiag(2,-1)
        let x = Array1::ones(n);
        let y = a.spmv(&x);
        // row 0: 2*1 + (-1)*1 = 1
        // row 1: (-1)*1 + 2*1 + (-1)*1 = 0
        // row 2: (-1)*1 + 2*1 + (-1)*1 = 0
        // row 3: (-1)*1 + 2*1 = 1
        assert!((y[0] - 1.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1]).abs() < 1e-12, "y[1]={}", y[1]);
        assert!((y[2]).abs() < 1e-12, "y[2]={}", y[2]);
        assert!((y[3] - 1.0).abs() < 1e-12, "y[3]={}", y[3]);
    }

    #[test]
    fn test_ilu0_compute() {
        let a = tridiag(4);
        let ilu = Ilu0Preconditioner::compute(&a).expect("ILU(0) on tridiagonal must succeed");
        // Verify we got a proper object (n = 4).
        assert_eq!(ilu.n, 4);
        assert_eq!(ilu.row_ptr.len(), 5);
    }

    #[test]
    fn test_ilu0_apply() {
        let a = tridiag(6);
        let ilu = Ilu0Preconditioner::compute(&a).expect("ILU(0) should succeed");
        let r = Array1::from(vec![1.0_f64; 6]);
        let z = ilu.apply(&r);
        // Shape check
        assert_eq!(z.len(), 6);
        // All-ones r should produce a finite, non-NaN result.
        for (i, &v) in z.iter().enumerate() {
            assert!(v.is_finite(), "z[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_from_dense_roundtrip() {
        let mut dense = Array2::zeros((3, 3));
        dense[[0, 0]] = 4.0;
        dense[[0, 1]] = -1.0;
        dense[[1, 0]] = -1.0;
        dense[[1, 1]] = 4.0;
        dense[[1, 2]] = -1.0;
        dense[[2, 1]] = -1.0;
        dense[[2, 2]] = 4.0;

        let a = GpuPrecondCsr::from_dense(&dense);
        assert_eq!(a.nrows, 3);
        assert_eq!(a.ncols, 3);
        assert_eq!(a.nnz(), 7);

        let x = Array1::from(vec![1.0, 1.0, 1.0]);
        let y = a.spmv(&x);
        // Row 0: 4 - 1 = 3
        assert!((y[0] - 3.0).abs() < 1e-12);
        // Row 1: -1 + 4 - 1 = 2
        assert!((y[1] - 2.0).abs() < 1e-12);
        // Row 2: -1 + 4 = 3
        assert!((y[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_preconditioned_cg_solve() {
        let n = 10;
        let a = spd_matrix(n);
        // b = A * x_exact, x_exact = [1, 2, ..., n]
        let x_exact = Array1::from_iter((1..=n).map(|i| i as f64));
        let b = a.spmv(&x_exact);

        let solver = MixedPreconditionedCg::new(&a, 200, 1e-10)
            .expect("MixedPreconditionedCg construction must succeed");
        let (x_sol, iters) = solver.solve(&a, &b).expect("PCG solve must succeed");

        assert!(iters <= 200, "solver used more iterations than max");
        // Verify ||x_sol - x_exact||_inf < 1e-6
        for i in 0..n {
            assert!(
                (x_sol[i] - x_exact[i]).abs() < 1e-6,
                "x_sol[{i}]={} vs x_exact[{i}]={}",
                x_sol[i],
                x_exact[i]
            );
        }
    }
}
