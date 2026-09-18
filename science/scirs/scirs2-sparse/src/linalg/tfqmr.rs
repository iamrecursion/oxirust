//! Transpose-Free Quasi-Minimal Residual (TFQMR) method for sparse linear systems
//!
//! TFQMR is a Krylov subspace method that can solve non-symmetric linear systems
//! without requiring the transpose of the coefficient matrix.
//!
//! Implementation follows SciPy's tfqmr based on R.W. Freund (1993).
//!
//! Reference:
//! R. W. Freund, "A Transpose-Free Quasi-Minimal Residual Algorithm for
//! Non-Hermitian Linear Systems", SIAM J. Sci. Comput., 14(2), 470-482, 1993.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;

/// Options for the TFQMR solver
#[derive(Debug, Clone)]
pub struct TFQMROptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to use left preconditioning
    pub use_left_preconditioner: bool,
    /// Whether to use right preconditioning
    pub use_right_preconditioner: bool,
}

impl Default for TFQMROptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            use_left_preconditioner: false,
            use_right_preconditioner: false,
        }
    }
}

/// Result from TFQMR solver
#[derive(Debug, Clone)]
pub struct TFQMRResult<T> {
    /// Solution vector
    pub x: Array1<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: T,
    /// Whether the solver converged
    pub converged: bool,
    /// Residual history (if requested)
    pub residual_history: Option<Vec<T>>,
}

/// Transpose-Free Quasi-Minimal Residual method
///
/// Solves the linear system A * x = b using the TFQMR method.
/// This method is suitable for non-symmetric matrices and does not
/// require computing A^T explicitly.
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A
/// * `b` - The right-hand side vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options
///
/// # Returns
///
/// A `TFQMRResult` containing the solution and convergence information
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::{tfqmr, TFQMROptions};
/// use scirs2_core::ndarray::Array1;
///
/// // Create a simple matrix
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 1, 2];
/// let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");
///
/// // Right-hand side
/// let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
///
/// // Solve using TFQMR
/// let result = tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn tfqmr<T, S>(
    matrix: &S,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    options: TFQMROptions,
) -> SparseResult<TFQMRResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = b.len();
    let (rows, cols) = matrix.shape();

    if rows != cols || rows != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: rows,
        });
    }

    let one = T::sparse_one();
    let zero = T::sparse_zero();

    // Initialize solution vector
    let mut x = match x0 {
        Some(x0_val) => x0_val.to_owned(),
        None => Array1::zeros(n),
    };

    // Compute initial residual: r = b - A * x
    let ax = matrix_vector_multiply(matrix, &x.view())?;
    let r = b - &ax;

    let r0norm = l2_norm(&r.view());
    let b_norm = l2_norm(b);
    let tolerance = T::from(options.tol).expect("Operation failed")
        * b_norm.max(T::from(1e-10).expect("Operation failed"));

    if r0norm <= tolerance || r0norm == zero {
        return Ok(TFQMRResult {
            x,
            iterations: 0,
            residual_norm: r0norm,
            converged: true,
            residual_history: Some(vec![r0norm]),
        });
    }

    // Initialize vectors following SciPy's notation
    let mut u = r.clone();
    let mut w = r.clone();
    let rstar = r.clone(); // Shadow residual

    // v = A * r (no preconditioner in this version)
    let ar = matrix_vector_multiply(matrix, &r.view())?;
    let mut v = ar;
    let mut uhat = v.clone();

    let mut d: Array1<T> = Array1::zeros(n);
    let mut theta = zero;
    let mut eta = zero;

    // rho = <rstar, r>
    let mut rho = dot_product(&rstar.view(), &r.view());
    let mut rho_last = rho;
    let mut tau = r0norm;

    let mut residual_history = Vec::new();
    residual_history.push(r0norm);

    let mut converged = false;
    let mut iter = 0;

    for it in 0..options.max_iter {
        iter = it + 1;
        let even = it % 2 == 0;

        // On even iterations, compute alpha and uNext
        let mut alpha = zero;
        let mut u_next: Array1<T> = Array1::zeros(n);

        if even {
            let vtrstar = dot_product(&rstar.view(), &v.view());
            if vtrstar.abs() < T::from(1e-300).expect("Operation failed") {
                return Err(SparseError::ConvergenceError(
                    "TFQMR breakdown: v'*rstar = 0".to_string(),
                ));
            }
            alpha = rho / vtrstar;

            // uNext = u - alpha * v
            for i in 0..n {
                u_next[i] = u[i] - alpha * v[i];
            }
        }

        // w = w - alpha * uhat (every iteration)
        let alpha_used = if even {
            alpha
        } else {
            rho / dot_product(&rstar.view(), &v.view())
        };

        for i in 0..n {
            w[i] = w[i] - alpha_used * uhat[i];
        }

        // d = u + (theta^2 / alpha) * eta * d
        let theta_sq_over_alpha = if alpha_used.abs() > T::from(1e-300).expect("Operation failed") {
            theta * theta / alpha_used
        } else {
            zero
        };
        for i in 0..n {
            d[i] = u[i] + theta_sq_over_alpha * eta * d[i];
        }

        // theta = ||w|| / tau
        theta = l2_norm(&w.view()) / tau;

        // c = 1 / sqrt(1 + theta^2)
        let c = one / (one + theta * theta).sqrt();

        // tau = tau * theta * c
        tau = tau * theta * c;

        // eta = c^2 * alpha
        eta = c * c * alpha_used;

        // x = x + eta * d (no preconditioner)
        for i in 0..n {
            x[i] = x[i] + eta * d[i];
        }

        residual_history.push(tau);

        // Convergence criterion: tau * sqrt(iter+1) < tolerance
        let iter_f = T::from(iter).expect("Operation failed");
        if tau * iter_f.sqrt() < tolerance {
            converged = true;
            break;
        }

        if !even {
            // Odd iteration updates
            rho = dot_product(&rstar.view(), &w.view());

            if rho.abs() < T::from(1e-300).expect("Operation failed") {
                return Err(SparseError::ConvergenceError(
                    "TFQMR breakdown: rho = 0".to_string(),
                ));
            }

            let beta = rho / rho_last;

            // u = w + beta * u
            for i in 0..n {
                u[i] = w[i] + beta * u[i];
            }

            // v = beta * uhat + beta^2 * v
            for i in 0..n {
                v[i] = beta * uhat[i] + beta * beta * v[i];
            }

            // uhat = A * u
            let au = matrix_vector_multiply(matrix, &u.view())?;
            uhat = au;

            // v = v + uhat
            for i in 0..n {
                v[i] = v[i] + uhat[i];
            }
        } else {
            // Even iteration updates
            // uhat = A * uNext
            let au_next = matrix_vector_multiply(matrix, &u_next.view())?;
            uhat = au_next;

            // u = uNext
            u = u_next;

            // rho_last = rho
            rho_last = rho;
        }
    }

    // Compute final residual norm by explicit calculation
    let ax_final = matrix_vector_multiply(matrix, &x.view())?;
    let final_residual = b - &ax_final;
    let final_residual_norm = l2_norm(&final_residual.view());

    Ok(TFQMRResult {
        x,
        iterations: iter,
        residual_norm: final_residual_norm,
        converged,
        residual_history: Some(residual_history),
    })
}

/// Helper function for matrix-vector multiplication
#[allow(dead_code)]
fn matrix_vector_multiply<T, S>(matrix: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    if x.len() != cols {
        return Err(SparseError::DimensionMismatch {
            expected: cols,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(rows);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[i] = result[i] + values[k] * x[j];
    }

    Ok(result)
}

/// Compute L2 norm of a vector
#[allow(dead_code)]
fn l2_norm<T>(x: &ArrayView1<T>) -> T
where
    T: Float + SparseElement + Debug + Copy,
{
    (x.iter()
        .map(|&val| val * val)
        .fold(T::sparse_zero(), |a, b| a + b))
    .sqrt()
}

/// Compute dot product of two vectors
#[allow(dead_code)]
fn dot_product<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float + SparseElement + Debug + Copy,
{
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi * yi)
        .fold(T::sparse_zero(), |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_tfqmr_simple_system() {
        // Create a simple 3x3 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let result =
            tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).expect("Operation failed");

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).expect("Operation failed");
        let residual = &b - &ax;
        let residual_norm = l2_norm(&residual.view());

        assert!(residual_norm < 1e-6);
    }

    #[test]
    fn test_tfqmr_diagonal_system() {
        // Create a diagonal system
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![4.0, 9.0, 16.0]);
        let result =
            tfqmr(&matrix, &b.view(), None, TFQMROptions::default()).expect("Operation failed");

        assert!(result.converged);

        // For diagonal system, solution should be [2, 3, 4]
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[2], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tfqmr_with_initial_guess() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![5.0, 6.0, 7.0]);
        let x0 = Array1::from_vec(vec![4.0, 5.0, 6.0]); // Close to solution

        let result = tfqmr(
            &matrix,
            &b.view(),
            Some(&x0.view()),
            TFQMROptions::default(),
        )
        .expect("Operation failed");

        assert!(result.converged);
        assert!(result.iterations <= 5); // Should converge quickly with good initial guess
    }
}
