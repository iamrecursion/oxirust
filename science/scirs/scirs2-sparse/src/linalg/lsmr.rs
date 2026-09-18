//! Least Squares Minimal Residual (LSMR) method for sparse linear systems
//!
//! LSMR is an iterative algorithm for solving large sparse least squares problems
//! and sparse systems of linear equations. It's closely related to LSQR but
//! can be more stable for ill-conditioned problems.
//!
//! Implementation follows SciPy reference based on:
//! D. C.-L. Fong and M. A. Saunders (2011), "LSMR: An iterative algorithm
//! for sparse least-squares problems", SIAM J. Sci. Comput., 33(5), 2950-2971.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, One, SparseElement};
use std::fmt::Debug;

/// Stable implementation of Givens rotation (sym_ortho)
///
/// Computes (c, s, r) such that [ c  s] [a] = [r]
///                               [-s  c] [b]   [0]
///
/// Uses the stable formulation to avoid overflow/underflow.
fn sym_ortho<T: Float + SparseElement>(a: T, b: T) -> (T, T, T) {
    let zero = T::sparse_zero();
    let one = <T as One>::one();

    if b == zero {
        return (if a >= zero { one } else { -one }, zero, a.abs());
    } else if a == zero {
        return (zero, if b >= zero { one } else { -one }, b.abs());
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s_sign = if b >= zero { one } else { -one };
        let s = s_sign / (one + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c_sign = if a >= zero { one } else { -one };
        let c = c_sign / (one + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

/// Options for the LSMR solver
#[derive(Debug, Clone)]
pub struct LSMROptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance for the residual
    pub atol: f64,
    /// Convergence tolerance for the solution
    pub btol: f64,
    /// Condition number limit
    pub conlim: f64,
    /// Whether to compute standard errors
    pub calc_var: bool,
    /// Whether to store residual history
    pub store_residual_history: bool,
    /// Local reorthogonalization parameter
    pub local_size: usize,
}

impl Default for LSMROptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            conlim: 1e8,
            calc_var: false,
            store_residual_history: true,
            local_size: 0,
        }
    }
}

/// Result from LSMR solver
#[derive(Debug, Clone)]
pub struct LSMRResult<T> {
    /// Solution vector
    pub x: Array1<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ||Ax - b||
    pub residualnorm: T,
    /// Final solution norm ||x||
    pub solution_norm: T,
    /// Condition number estimate
    pub condition_number: T,
    /// Whether the solver converged
    pub converged: bool,
    /// Standard errors (if requested)
    pub standard_errors: Option<Array1<T>>,
    /// Residual history (if requested)
    pub residual_history: Option<Vec<T>>,
    /// Convergence reason
    pub convergence_reason: String,
}

/// LSMR algorithm for sparse least squares problems
///
/// Solves the least squares problem min ||Ax - b||_2 or the linear system Ax = b.
/// The method is based on the Golub-Kahan bidiagonalization process.
///
/// # Arguments
///
/// * `matrix` - The coefficient matrix A (m x n)
/// * `b` - The right-hand side vector (length m)
/// * `x0` - Initial guess (optional, length n)
/// * `options` - Solver options
///
/// # Returns
///
/// An `LSMRResult` containing the solution and convergence information
///
/// # Example
///
/// ```rust
/// use scirs2_sparse::csr_array::CsrArray;
/// use scirs2_sparse::linalg::{lsmr, LSMROptions};
/// use scirs2_core::ndarray::Array1;
///
/// // Create an overdetermined system
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![0, 1, 0, 1, 0, 1];
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).expect("Operation failed");
///
/// // Right-hand side
/// let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
///
/// // Solve using LSMR
/// let result = lsmr(&matrix, &b.view(), None, LSMROptions::default()).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn lsmr<T, S>(
    matrix: &S,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    options: LSMROptions,
) -> SparseResult<LSMRResult<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (m, n) = matrix.shape();

    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    // Initialize solution vector
    let mut x = match x0 {
        Some(x0_val) => {
            if x0_val.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0_val.len(),
                });
            }
            x0_val.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Compute initial residual
    let ax = matrix_vector_multiply(matrix, &x.view())?;
    let mut u = b - &ax;
    let mut beta = l2_norm(&u.view());

    // Tolerances
    let atol = T::from(options.atol).expect("Operation failed");
    let btol = T::from(options.btol).expect("Operation failed");
    let conlim = T::from(options.conlim).expect("Operation failed");

    let mut residual_history = if options.store_residual_history {
        Some(vec![beta])
    } else {
        None
    };

    // Check for immediate convergence
    if beta <= atol {
        let solution_norm = l2_norm(&x.view());
        return Ok(LSMRResult {
            x,
            iterations: 0,
            residualnorm: beta,
            solution_norm,
            condition_number: T::sparse_one(),
            converged: true,
            standard_errors: None,
            residual_history,
            convergence_reason: "Already converged".to_string(),
        });
    }

    // Normalize u
    if beta > T::sparse_zero() {
        for i in 0..m {
            u[i] = u[i] / beta;
        }
    }

    // Initialize bidiagonalization
    let mut v = matrix_transpose_vector_multiply(matrix, &u.view())?;
    let mut alpha = l2_norm(&v.view());

    if alpha > T::sparse_zero() {
        for i in 0..n {
            v[i] = v[i] / alpha;
        }
    }

    // Initialize LSMR-specific variables (following SciPy reference)
    let one = T::sparse_one();
    let zero = T::sparse_zero();

    let mut alphabar = alpha;
    let mut zetabar = alpha * beta;
    let mut rho = one;
    let mut rhobar = one;
    let mut cbar = one;
    let mut sbar = zero;

    let mut h = v.clone();
    let mut hbar: Array1<T> = Array1::zeros(n);

    // For norm estimation
    let mut anorm = zero;
    let mut acond = zero;
    let mut rnorm = beta;
    let mut xnorm = zero;

    let bnorm = beta;
    let mut norm_a2 = alpha * alpha;
    let mut maxrbar = zero;
    let mut minrbar = T::from(1e100).expect("Operation failed");

    let mut converged = false;
    let mut convergence_reason = String::new();
    let mut iter = 0;

    for itn in 0..options.max_iter {
        iter = itn + 1;

        // Perform the next step of the bidiagonalization.
        // Golub-Kahan bidiagonalization: u = A*v - alpha*u
        let av = matrix_vector_multiply(matrix, &v.view())?;
        for i in 0..m {
            u[i] = av[i] - alpha * u[i];
        }
        beta = l2_norm(&u.view());

        if beta > zero {
            for i in 0..m {
                u[i] = u[i] / beta;
            }

            // v = A'*u - beta*v
            let atu = matrix_transpose_vector_multiply(matrix, &u.view())?;
            for i in 0..n {
                v[i] = atu[i] - beta * v[i];
            }
            alpha = l2_norm(&v.view());

            if alpha > zero {
                for i in 0..n {
                    v[i] = v[i] / alpha;
                }
            }
        }

        // Construct rotation Q_{i,2i+1} (plane rotation to eliminate beta)
        let rhoold = rho;
        let (c, s, rho_new) = sym_ortho(alphabar, beta);
        rho = rho_new;
        let thetanew = s * alpha;
        alphabar = c * alpha;

        // Construct rotation Qbar_{i,2i+1} (plane rotation for LSMR)
        let rhobarold = rhobar;
        let zetaold = zetabar;
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        let (cbar_new, sbar_new, rhobar_new) = sym_ortho(rhotemp, thetanew);
        cbar = cbar_new;
        sbar = sbar_new;
        rhobar = rhobar_new;
        let zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;

        // Update h, hbar, x
        for i in 0..n {
            let hbar_old = hbar[i];
            hbar[i] = h[i] - (thetabar * rho / (rhoold * rhobarold)) * hbar_old;
        }
        for i in 0..n {
            x[i] = x[i] + (zeta / (rho * rhobar)) * hbar[i];
        }
        for i in 0..n {
            h[i] = v[i] - (thetanew / rho) * h[i];
        }

        // Estimate norms
        norm_a2 = norm_a2 + beta * beta;
        anorm = norm_a2.sqrt();
        norm_a2 = norm_a2 + alpha * alpha;

        // Update estimates
        if c.abs() > zero {
            maxrbar = maxrbar.max(rhobarold);
            if itn > 1 {
                minrbar = minrbar.min(rhobarold);
            }
        }
        acond = maxrbar / minrbar;

        // Compute norm estimates
        let betadd = c * zetaold;
        let betad = -(sbar * betadd);
        let rhodold = rho;

        // Use the recurrence for ||r_k||
        let thetahat = sbar * rho;
        let rhohat = cbar * rho;
        let chat = rhohat / rhodold;
        let shat = thetahat / rhodold;

        rnorm = (rnorm * rnorm * shat * shat + betad * betad).sqrt();
        xnorm = (xnorm * xnorm + (zeta / (rho * rhobar)) * (zeta / (rho * rhobar))).sqrt();

        let arnorm = alpha * beta.abs() * c.abs() * s.abs();

        if let Some(ref mut history) = residual_history {
            history.push(rnorm);
        }

        // Check stopping criteria
        // Condition 1: ||Ax - b|| / ||b|| small enough
        let test1 = rnorm / (bnorm + anorm * xnorm + one);
        // Condition 2: ||A'r|| / (||A|| ||r||) small enough
        let test2 = if rnorm > zero {
            arnorm / (anorm * rnorm + one)
        } else {
            zero
        };

        if test1 <= atol || rnorm <= atol * bnorm {
            converged = true;
            convergence_reason = "Residual tolerance satisfied".to_string();
            break;
        }

        if test2 <= btol {
            converged = true;
            convergence_reason = "Solution tolerance satisfied".to_string();
            break;
        }

        if acond >= conlim {
            converged = true;
            convergence_reason = "Condition number limit reached".to_string();
            break;
        }
    }

    if !converged {
        convergence_reason = "Maximum iterations reached".to_string();
    }

    // Compute final metrics
    let ax_final = matrix_vector_multiply(matrix, &x.view())?;
    let final_residual = b - &ax_final;
    let final_residualnorm = l2_norm(&final_residual.view());
    let final_solution_norm = l2_norm(&x.view());

    // Condition number estimate
    let condition_number = acond;

    // Compute standard errors if requested
    let standard_errors = if options.calc_var {
        Some(compute_standard_errors(matrix, final_residualnorm, n)?)
    } else {
        None
    };

    Ok(LSMRResult {
        x,
        iterations: iter,
        residualnorm: final_residualnorm,
        solution_norm: final_solution_norm,
        condition_number,
        converged,
        standard_errors,
        residual_history,
        convergence_reason,
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

/// Helper function for matrix transpose-vector multiplication
#[allow(dead_code)]
fn matrix_transpose_vector_multiply<T, S>(matrix: &S, x: &ArrayView1<T>) -> SparseResult<Array1<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();
    if x.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: x.len(),
        });
    }

    let mut result = Array1::zeros(cols);
    let (row_indices, col_indices, values) = matrix.find();

    for (k, (&i, &j)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        result[j] = result[j] + values[k] * x[i];
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

/// Compute standard errors (simplified implementation)
#[allow(dead_code)]
fn compute_standard_errors<T, S>(matrix: &S, residualnorm: T, n: usize) -> SparseResult<Array1<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (m, _) = matrix.shape();

    // Simplified standard error computation
    let variance = if m > n {
        residualnorm * residualnorm / T::from(m - n).expect("Operation failed")
    } else {
        residualnorm * residualnorm
    };

    let std_err = variance.sqrt();
    Ok(Array1::from_elem(n, std_err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    #[test]
    fn test_lsmr_square_system() {
        // Create a simple 3x3 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, 2.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let result =
            lsmr(&matrix, &b.view(), None, LSMROptions::default()).expect("Operation failed");

        assert!(result.converged);

        // Verify solution by computing residual
        let ax = matrix_vector_multiply(&matrix, &result.x.view()).expect("Operation failed");
        let residual = &b - &ax;
        let residualnorm = l2_norm(&residual.view());

        assert!(residualnorm < 1e-6);
    }

    #[test]
    fn test_lsmr_overdetermined_system() {
        // Create an overdetermined 3x2 system
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 2), false).expect("Operation failed");

        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result =
            lsmr(&matrix, &b.view(), None, LSMROptions::default()).expect("Operation failed");

        assert!(result.converged);
        assert_eq!(result.x.len(), 2);

        // For overdetermined systems, check that we get a reasonable least squares solution
        assert!(result.residualnorm < 2.0);
    }

    #[test]
    fn test_lsmr_diagonal_system() {
        // Create a diagonal system
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![4.0, 9.0, 16.0]);
        let result =
            lsmr(&matrix, &b.view(), None, LSMROptions::default()).expect("Operation failed");

        assert!(result.converged);

        // For diagonal system, solution should be [2, 3, 4]
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(result.x[2], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lsmr_with_initial_guess() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![5.0, 6.0, 7.0]);
        let x0 = Array1::from_vec(vec![4.0, 5.0, 6.0]); // Close to solution

        let result = lsmr(&matrix, &b.view(), Some(&x0.view()), LSMROptions::default())
            .expect("Operation failed");

        assert!(result.converged);
        assert!(result.iterations <= 10); // Should converge reasonably quickly
    }

    #[test]
    fn test_lsmr_standard_errors() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let matrix =
            CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).expect("Operation failed");

        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let options = LSMROptions {
            calc_var: true,
            ..Default::default()
        };

        let result = lsmr(&matrix, &b.view(), None, options).expect("Operation failed");

        assert!(result.converged);
        assert!(result.standard_errors.is_some());

        let std_errs = result.standard_errors.expect("Operation failed");
        assert_eq!(std_errs.len(), 3);
    }
}
