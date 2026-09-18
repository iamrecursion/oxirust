//! Mixed CPU/GPU linear system solver.
//!
//! Performs matrix factorization at low precision (simulating GPU dispatch)
//! then applies iterative residual refinement on the CPU at higher precision.
//!
//! # Algorithm
//!
//! 1. Determine precision using the auto-precision policy.
//! 2. Factorize the system `Ax = b` at the selected precision (f32 or f64).
//! 3. Compute the residual `r = b - A x` in f64.
//! 4. Solve a correction system `A delta = r` in f64 and apply `x += delta`.
//! 5. Repeat until residual is smaller than `tol` or `refinement_steps` is
//!    exhausted.
//!
//! # References
//!
//! - Higham (2002). "Accuracy and Stability of Numerical Algorithms." §12.4.
//! - Demmel et al. (2006). "Error bounds from extra-precise iterative
//!   refinement."

use scirs2_core::ndarray::{Array1, Array2};

use crate::auto_precision::{solve_f32, solve_f64, Precision, PrecisionPolicy};
use crate::error::LinalgError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Statistics returned by [`MixedSolver::solve`].
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Which precision was used for the initial solve.
    pub precision_used: Precision,
    /// Number of iterative refinement steps actually applied.
    pub refinement_steps_done: usize,
    /// 2-norm of the final residual `||b - A x||`.
    pub final_residual: f64,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Mixed CPU/GPU linear system solver with iterative residual refinement.
///
/// The factorization step is dispatched at the precision recommended by
/// [`PrecisionPolicy`].  Refinement steps always run in f64 on the CPU.
pub struct MixedSolver {
    /// Maximum number of iterative refinement steps.
    refinement_steps: usize,
    /// Convergence tolerance for the residual 2-norm.
    tol: f64,
    /// Precision policy for the initial factorization.
    policy: PrecisionPolicy,
}

impl MixedSolver {
    /// Create a new solver.
    ///
    /// # Arguments
    ///
    /// * `refinement_steps` — Maximum number of residual refinement iterations.
    /// * `tol` — Stop refining when `||b - Ax|| < tol`.
    pub fn new(refinement_steps: usize, tol: f64) -> Self {
        Self {
            refinement_steps,
            tol,
            policy: PrecisionPolicy::default(),
        }
    }

    /// Create a new solver with an explicit precision policy.
    pub fn with_policy(refinement_steps: usize, tol: f64, policy: PrecisionPolicy) -> Self {
        Self {
            refinement_steps,
            tol,
            policy,
        }
    }

    /// Solve `Ax = b`.
    ///
    /// Returns the solution and solver statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is singular, dimensions are mismatched,
    /// or an internal numerical failure occurs.
    pub fn solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<(Array1<f64>, SolverStats), LinalgError> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::DimensionError(format!(
                "MixedSolver requires a square matrix, got {}x{}",
                n,
                a.ncols()
            )));
        }
        if b.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "rhs length {} does not match matrix dimension {}",
                b.len(),
                n
            )));
        }

        // Step 1: select precision and solve initial system
        let precision = crate::auto_precision::select_precision(a, &self.policy);
        let mut x = match precision {
            Precision::Single => solve_f32(a, b)?,
            Precision::Double | Precision::Mixed => solve_f64(a, b)?,
        };

        // Step 2: iterative residual refinement
        let mut steps_done = 0;
        let mut final_res = residual_norm(a, b, &x);

        for _ in 0..self.refinement_steps {
            if final_res < self.tol {
                break;
            }
            // Compute residual r = b - Ax  (in f64)
            let r = compute_residual(a, b, &x);
            // Solve A delta = r  (in f64)
            let delta = solve_f64(a, &r)?;
            // Apply correction
            for i in 0..n {
                x[i] += delta[i];
            }
            steps_done += 1;
            final_res = residual_norm(a, b, &x);
        }

        Ok((
            x,
            SolverStats {
                precision_used: precision,
                refinement_steps_done: steps_done,
                final_residual: final_res,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the residual vector `r = b - A x`.
fn compute_residual(a: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut r = b.to_owned();
    for i in 0..n {
        let mut ax_i = 0.0;
        for j in 0..n {
            ax_i += a[[i, j]] * x[j];
        }
        r[i] -= ax_i;
    }
    r
}

/// Compute the 2-norm of the residual `||b - Ax||`.
fn residual_norm(a: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>) -> f64 {
    let r = compute_residual(a, b, x);
    r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_precision::Precision;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mixed_solver_well_conditioned() {
        // Solve a simple 3x3 system
        let a = array![[2.0_f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let b = array![8.0_f64, -11.0, -3.0];
        let solver = MixedSolver::new(3, 1e-12);
        let (x, stats) = solver.solve(&a, &b).expect("should succeed");

        assert!((x[0] - 2.0).abs() < 1e-8, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-8, "x[1]={}", x[1]);
        assert!((x[2] - (-1.0)).abs() < 1e-8, "x[2]={}", x[2]);
        assert!(
            stats.final_residual < 1e-10,
            "residual={}",
            stats.final_residual
        );
    }

    #[test]
    fn test_mixed_solver_force_single_refines() {
        // Force single precision so refinement is needed
        let policy = PrecisionPolicy {
            force: Some(Precision::Single),
            ..Default::default()
        };
        let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
        let b = array![1.0_f64, 2.0];
        let solver = MixedSolver::with_policy(5, 1e-12, policy);
        let (x, stats) = solver.solve(&a, &b).expect("should succeed");

        // Exact solution: x = [1/11, 7/11]
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6, "x[0]={}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-6, "x[1]={}", x[1]);
        assert_eq!(stats.precision_used, Precision::Single);
    }

    #[test]
    fn test_mixed_solver_dimension_mismatch() {
        let a = Array2::<f64>::eye(3);
        let b = Array1::<f64>::zeros(2);
        let solver = MixedSolver::new(3, 1e-10);
        assert!(solver.solve(&a, &b).is_err());
    }

    #[test]
    fn test_mixed_solver_non_square() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array1::<f64>::zeros(2);
        let solver = MixedSolver::new(3, 1e-10);
        assert!(solver.solve(&a, &b).is_err());
    }

    #[test]
    fn test_mixed_solver_stats_precision() {
        // A well-conditioned matrix should use Single
        let a = array![[2.0_f64, 0.5], [0.5, 2.0]];
        let b = array![1.0_f64, 1.0];
        let solver = MixedSolver::new(3, 1e-10);
        let (_x, stats) = solver.solve(&a, &b).expect("should succeed");
        // The default policy threshold is 1e4; this matrix is well-conditioned
        assert_eq!(stats.precision_used, Precision::Single);
    }
}
