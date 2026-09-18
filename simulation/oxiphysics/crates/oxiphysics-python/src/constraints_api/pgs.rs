// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Projected Gauss-Seidel (PGS) solvers of the constraint API.
//!
//! A velocity-level constraint solve is the box-constrained linear
//! complementarity problem: find impulses `lambda` with `lo <= lambda <= hi`
//! for the Delassus matrix `A` and the right-hand side `rhs`, such that every
//! impulse strictly inside its bounds satisfies `(A lambda)_i = rhs_i`, one at
//! its lower bound satisfies `(A lambda)_i >= rhs_i`, and one at its upper bound
//! satisfies `(A lambda)_i <= rhs_i`.
//!
//! One Gauss-Seidel *sweep* visits the rows in order and moves every impulse to
//!
//! ```text
//! lambda_i <- clamp(lambda_i + omega * (rhs_i - sum_j A_ij lambda_j) / A_ii, lo_i, hi_i)
//! ```
//!
//! using the impulses already updated in the same sweep. [`solve_pgs_sweep`]
//! performs one sweep on a dense matrix, [`solve_pgs_iteration`] one sweep on
//! the diagonal of `A` alone (no coupling), and
//! [`PyConstraintSolver::try_solve`] repeats sweeps as its configuration says.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Smallest `|A_ii|` a sweep divides by; a row with a smaller diagonal has no
/// effective mass (for example a constraint between two static bodies) and is
/// left unchanged.
const MIN_DIAGONAL: f64 = 1e-15;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Input rejected by the PGS solvers.
///
/// Every variant surfaces in Python as a `ValueError`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PgsInputError {
    /// Two inputs that must describe the same constraints differ in length.
    #[error("`{name}` has {got} entries, expected {expected}")]
    LengthMismatch {
        /// Name of the offending argument.
        name: &'static str,
        /// The number of entries the problem size requires.
        expected: usize,
        /// The number of entries given.
        got: usize,
    },
    /// A value that has to be finite is NaN or infinite.
    #[error("`{name}[{index}]` is {value}, but it must be finite")]
    NotFinite {
        /// Name of the offending argument.
        name: &'static str,
        /// Index of the offending entry.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A bound is NaN (an infinite bound is allowed: it means unbounded).
    #[error("`{name}[{index}]` is NaN")]
    NanBound {
        /// Name of the offending argument.
        name: &'static str,
        /// Index of the offending entry.
        index: usize,
    },
    /// A lower bound lies above the upper bound of the same impulse.
    #[error("lower bound lo[{index}] = {lo} is above the upper bound hi[{index}] = {hi}")]
    InvertedBounds {
        /// Index of the constraint.
        index: usize,
        /// The lower bound.
        lo: f64,
        /// The upper bound.
        hi: f64,
    },
    /// A diagonal entry of the Delassus matrix is negative.
    #[error(
        "the diagonal entry {name}[{index}] = {value} is negative, but an effective mass is not"
    )]
    NegativeDiagonal {
        /// Name of the argument holding the diagonal.
        name: &'static str,
        /// Index of the offending entry.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The over-relaxation factor is outside the open interval `(0, 2)`.
    #[error("sor_factor must lie strictly between 0 and 2, got {0}")]
    InvalidSorFactor(f64),
}

impl From<PgsInputError> for PyErr {
    fn from(err: PgsInputError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// `Ok` if `got` equals `expected`.
fn check_len(name: &'static str, expected: usize, got: usize) -> Result<(), PgsInputError> {
    if expected == got {
        Ok(())
    } else {
        Err(PgsInputError::LengthMismatch {
            name,
            expected,
            got,
        })
    }
}

/// `Ok` if every value is finite.
fn check_finite(name: &'static str, values: &[f64]) -> Result<(), PgsInputError> {
    match values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        Some((index, &value)) => Err(PgsInputError::NotFinite { name, index, value }),
        None => Ok(()),
    }
}

/// `Ok` if no bound is NaN and every lower bound is at most its upper bound.
fn check_bounds(lo: &[f64], hi: &[f64]) -> Result<(), PgsInputError> {
    if let Some(index) = lo.iter().position(|v| v.is_nan()) {
        return Err(PgsInputError::NanBound { name: "lo", index });
    }
    if let Some(index) = hi.iter().position(|v| v.is_nan()) {
        return Err(PgsInputError::NanBound { name: "hi", index });
    }
    match lo.iter().zip(hi).position(|(l, h)| l > h) {
        Some(index) => Err(PgsInputError::InvertedBounds {
            index,
            lo: lo[index],
            hi: hi[index],
        }),
        None => Ok(()),
    }
}

/// `Ok` if no entry of the diagonal is negative.
fn check_diagonal<'a>(
    name: &'static str,
    diagonal: impl IntoIterator<Item = &'a f64>,
) -> Result<(), PgsInputError> {
    match diagonal.into_iter().enumerate().find(|(_, v)| **v < 0.0) {
        Some((index, &value)) => Err(PgsInputError::NegativeDiagonal { name, index, value }),
        None => Ok(()),
    }
}

/// Check the arguments of [`solve_pgs_iteration`] the way the Python wrapper does.
///
/// All five slices must have the same length, `rhs`, `diag` and `lambda` must
/// be finite, `diag` must not be negative, the bounds must not be NaN (they
/// may be infinite) and `lo[i] <= hi[i]` must hold for every constraint.
pub fn check_pgs_iteration_inputs(
    lambda: &[f64],
    rhs: &[f64],
    diag: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> Result<(), PgsInputError> {
    let n = lambda.len();
    check_len("rhs", n, rhs.len())?;
    check_len("diag", n, diag.len())?;
    check_len("lo", n, lo.len())?;
    check_len("hi", n, hi.len())?;
    check_finite("lambda", lambda)?;
    check_finite("rhs", rhs)?;
    check_finite("diag", diag)?;
    check_diagonal("diag", diag)?;
    check_bounds(lo, hi)
}

// ---------------------------------------------------------------------------
// Sweeps
// ---------------------------------------------------------------------------

/// Perform one PGS (Projected Gauss-Seidel) sweep over a set of constraints
/// that are not coupled: the Delassus matrix is reduced to its diagonal.
///
/// `lambda` — accumulated impulse per constraint (modified in place).
/// `rhs`    — constraint violation / desired velocity change per constraint.
/// `diag`   — diagonal of the effective-mass matrix per constraint.
/// `lo`     — lower bound per constraint (clamping).
/// `hi`     — upper bound per constraint (clamping).
///
/// Returns the Euclidean norm of the impulse changes of the sweep.
///
/// The function never panics: it processes as many constraints as the shortest
/// of the five slices holds, skips a constraint whose diagonal is (almost)
/// zero, and clamps with `max(lo).min(hi)`, so a NaN bound is ignored and an
/// inverted pair (`lo > hi`) acts as the upper bound. The Python wrapper
/// ([`py_solve_pgs_iteration`]) rejects such input instead.
pub fn solve_pgs_iteration(
    lambda: &mut [f64],
    rhs: &[f64],
    diag: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> f64 {
    let mut residual = 0.0;
    for ((((impulse, &b), &d), &lower), &upper) in
        lambda.iter_mut().zip(rhs).zip(diag).zip(lo).zip(hi)
    {
        if d.abs() < MIN_DIAGONAL {
            continue;
        }
        let delta = (b - d * *impulse) / d;
        let new_lambda = (*impulse + delta).max(lower).min(upper);
        let actual_delta = new_lambda - *impulse;
        *impulse = new_lambda;
        residual += actual_delta * actual_delta;
    }
    residual.sqrt()
}

/// Perform one projected Gauss-Seidel sweep with over-relaxation `omega` on the
/// dense system `a lambda = rhs`.
///
/// `a` is the `n x n` Delassus matrix in row-major order, `n = lambda.len()`.
/// The other arguments are as in [`solve_pgs_iteration`]. With `omega = 1` this
/// is plain Gauss-Seidel, with `1 < omega < 2` successive over-relaxation.
///
/// Returns the Euclidean norm of the impulse changes of the sweep.
///
/// The function never panics: rows beyond the end of `a`, `rhs`, `lo` or `hi`
/// are left unchanged, as are rows whose diagonal is (almost) zero.
pub fn solve_pgs_sweep(
    lambda: &mut [f64],
    a: &[f64],
    rhs: &[f64],
    lo: &[f64],
    hi: &[f64],
    omega: f64,
) -> f64 {
    let dim = lambda.len();
    let mut residual = 0.0;
    for (i, ((&b, &lower), &upper)) in rhs.iter().zip(lo).zip(hi).enumerate().take(dim) {
        let Some(row) = i
            .checked_mul(dim)
            .and_then(|start| a.get(start..start.checked_add(dim)?))
        else {
            break;
        };
        let Some(&diagonal) = row.get(i) else {
            continue;
        };
        if diagonal.abs() < MIN_DIAGONAL {
            continue;
        }
        let coupled: f64 = row
            .iter()
            .zip(lambda.iter())
            .map(|(a_ij, l)| a_ij * l)
            .sum();
        let current = lambda[i];
        let new_lambda = (current + omega * (b - coupled) / diagonal)
            .max(lower)
            .min(upper);
        let actual_delta = new_lambda - current;
        lambda[i] = new_lambda;
        residual += actual_delta * actual_delta;
    }
    residual.sqrt()
}

/// Python-exposed wrapper for PGS iteration.
///
/// Takes lambda as a Vec (returns modified vec) and all other slices as Vec.
/// Returns `(updated_lambda, residual)`.
///
/// Raises `ValueError` unless all five lists have the same length, `lambda`,
/// `rhs` and `diag` are finite, `diag` is not negative, no bound is NaN and
/// `lo[i] <= hi[i]` holds for every constraint.
#[pyfunction]
pub fn py_solve_pgs_iteration(
    mut lambda: Vec<f64>,
    rhs: Vec<f64>,
    diag: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
) -> PyResult<(Vec<f64>, f64)> {
    check_pgs_iteration_inputs(&lambda, &rhs, &diag, &lo, &hi)?;
    let residual = solve_pgs_iteration(&mut lambda, &rhs, &diag, &lo, &hi);
    Ok((lambda, residual))
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Constraint solver algorithm selection.
///
/// All three iterate Gauss-Seidel sweeps over the assembled system given to
/// [`PyConstraintSolver::solve`]; they differ in the relaxation and in the
/// number of sweeps.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SolverType {
    /// Projected Gauss-Seidel with the over-relaxation factor `sor_factor`:
    /// `velocity_iterations` sweeps.
    Pgs,
    /// Gauss-Seidel with `sor_factor` and a second phase: `velocity_iterations`
    /// velocity-level sweeps followed by `position_iterations` further sweeps
    /// over the same system (the position-level pass). It does not substep
    /// time, which would need the geometry that `solve` does not see.
    Tgs,
    /// Sequential impulses: plain Gauss-Seidel on the accumulated impulses,
    /// `velocity_iterations` sweeps without over-relaxation. On a fixed
    /// assembled system this is the `Pgs` iteration with `sor_factor = 1`.
    Si,
}

/// Main constraint solver configuration.
///
/// [`PyConstraintSolver::solve`] reads `solver_type`, `velocity_iterations`,
/// `position_iterations`, `warm_start`, `sor_factor` and `tolerance`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyConstraintSolver {
    /// Solver algorithm.
    pub solver_type: SolverType,
    /// Number of velocity-level sweeps.
    pub velocity_iterations: u32,
    /// Number of sweeps a `Tgs` solve runs after the velocity sweeps (ignored
    /// by `Pgs` and `Si`).
    pub position_iterations: u32,
    /// Whether `solve` starts from the `lambda0` it is given (otherwise it
    /// starts from zero), the way a solver is warm started from the previous
    /// frame.
    pub warm_start: bool,
    /// Successive over-relaxation factor in `(0, 2)` (1.0 = plain
    /// Gauss-Seidel), applied by `Pgs` and `Tgs`, not by `Si`.
    pub sor_factor: f64,
    /// A solve stops early once a sweep changes the impulses by less than
    /// `tolerance` (Euclidean norm of the change).
    pub tolerance: f64,
    /// Penetration slop in metres. `solve` works on an assembled system and
    /// does not read it: it is meant for the code that assembles `rhs` from
    /// penetration depths, where a Baumgarte term acts on the part of the depth
    /// beyond `slop`.
    pub slop: f64,
}

#[pymethods]
impl PyConstraintSolver {
    /// Create a default PGS solver.
    #[staticmethod]
    pub fn default_pgs() -> Self {
        Self {
            solver_type: SolverType::Pgs,
            velocity_iterations: 10,
            position_iterations: 5,
            warm_start: true,
            sor_factor: 1.0,
            tolerance: 1e-4,
            slop: 0.005,
        }
    }

    /// Create a TGS solver (fewer iterations needed).
    #[staticmethod]
    pub fn default_tgs() -> Self {
        Self {
            solver_type: SolverType::Tgs,
            velocity_iterations: 4,
            position_iterations: 2,
            warm_start: true,
            sor_factor: 1.3,
            tolerance: 1e-4,
            slop: 0.005,
        }
    }

    /// Solve the box-constrained complementarity problem `a lambda = rhs`,
    /// `lo <= lambda <= hi` by Gauss-Seidel sweeps.
    ///
    /// `a` is the `n x n` Delassus (effective-mass) matrix in row-major order
    /// with `n = len(rhs)`; `rhs`, `lo` and `hi` hold one entry per constraint
    /// (an infinite bound means unbounded). Every sweep sets
    /// `lambda_i = clamp(lambda_i + omega * (rhs_i - sum_j a_ij lambda_j) / a_ii,
    /// lo_i, hi_i)` with `omega = sor_factor` (`1` for `Si`), and the solver
    /// runs `velocity_iterations` sweeps (plus `position_iterations` for `Tgs`),
    /// stopping early once a sweep changes the impulses by less than
    /// `tolerance`.
    ///
    /// The solve starts from `lambda0` when `warm_start` is set and `lambda0`
    /// is given, otherwise from zero; the start is projected into the bounds.
    /// A constraint whose diagonal entry is (almost) zero is not updated.
    ///
    /// Returns the accumulated impulses.
    ///
    /// Raises `ValueError` if `a` does not hold `len(rhs)**2` values, `lo`,
    /// `hi` or `lambda0` do not hold `len(rhs)` values, a value is NaN or
    /// (apart from the bounds) infinite, a diagonal entry is negative,
    /// `lo[i] > hi[i]`, or `sor_factor` is outside `(0, 2)` (for `Pgs` and
    /// `Tgs`).
    #[pyo3(signature = (a, rhs, lo, hi, lambda0 = None))]
    pub fn solve(
        &self,
        a: Vec<f64>,
        rhs: Vec<f64>,
        lo: Vec<f64>,
        hi: Vec<f64>,
        lambda0: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.try_solve(&a, &rhs, &lo, &hi, lambda0.as_deref())?)
    }
}

impl PyConstraintSolver {
    /// Over-relaxation factor the configured algorithm applies.
    fn relaxation(&self) -> Result<f64, PgsInputError> {
        match self.solver_type {
            SolverType::Si => Ok(1.0),
            SolverType::Pgs | SolverType::Tgs => {
                if self.sor_factor > 0.0 && self.sor_factor < 2.0 {
                    Ok(self.sor_factor)
                } else {
                    Err(PgsInputError::InvalidSorFactor(self.sor_factor))
                }
            }
        }
    }

    /// Number of sweeps the configured algorithm runs.
    fn sweep_count(&self) -> u64 {
        let velocity = u64::from(self.velocity_iterations);
        match self.solver_type {
            SolverType::Tgs => velocity + u64::from(self.position_iterations),
            SolverType::Pgs | SolverType::Si => velocity,
        }
    }

    /// Rust-level core of [`Self::solve`] with a typed error.
    pub fn try_solve(
        &self,
        a: &[f64],
        rhs: &[f64],
        lo: &[f64],
        hi: &[f64],
        lambda0: Option<&[f64]>,
    ) -> Result<Vec<f64>, PgsInputError> {
        let n = rhs.len();
        check_len("a", n.saturating_mul(n), a.len())?;
        check_len("lo", n, lo.len())?;
        check_len("hi", n, hi.len())?;
        check_finite("a", a)?;
        check_finite("rhs", rhs)?;
        check_diagonal("a", a.iter().step_by(n + 1))?;
        check_bounds(lo, hi)?;
        let omega = self.relaxation()?;
        let start = match lambda0 {
            Some(start) if self.warm_start => {
                check_len("lambda0", n, start.len())?;
                check_finite("lambda0", start)?;
                start
            }
            _ => &[],
        };
        let mut lambda: Vec<f64> = (0..n)
            .map(|i| {
                let initial = start.get(i).copied().unwrap_or(0.0);
                initial.max(lo[i]).min(hi[i])
            })
            .collect();
        for _ in 0..self.sweep_count() {
            if solve_pgs_sweep(&mut lambda, a, rhs, lo, hi, omega) < self.tolerance {
                break;
            }
        }
        Ok(lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;

    /// Run `f` with the interpreter of the test process attached.
    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    /// The coupled two-contact system used by the analytic tests.
    const A: [f64; 4] = [2.0, 1.0, 1.0, 2.0];

    fn solver(solver_type: SolverType, sweeps: u32) -> PyConstraintSolver {
        PyConstraintSolver {
            solver_type,
            velocity_iterations: sweeps,
            position_iterations: 0,
            warm_start: false,
            sor_factor: 1.0,
            tolerance: 0.0,
            slop: 0.005,
        }
    }

    fn assert_close(got: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected) {
            assert!(
                (g - e).abs() <= tolerance,
                "got {got:?}, expected {expected:?}"
            );
        }
    }

    // ---- diagonal sweep ----------------------------------------------------

    #[test]
    fn test_solve_pgs_basic() {
        let mut lam = vec![0.0f64];
        let rhs = [1.0];
        let diag = [2.0];
        let lo = [0.0];
        let hi = [10.0];
        let res = solve_pgs_iteration(&mut lam, &rhs, &diag, &lo, &hi);
        assert!((lam[0] - 0.5).abs() < 1e-15);
        assert!((res - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_solve_pgs_clamped() {
        let mut lam = vec![0.0f64];
        let res = solve_pgs_iteration(&mut lam, &[100.0], &[1.0], &[0.0], &[5.0]);
        assert_eq!(lam[0], 5.0);
        assert_eq!(res, 5.0);
    }

    #[test]
    fn test_solve_pgs_iteration_survives_inverted_and_nan_bounds() {
        // lo > hi acts as the upper bound instead of panicking in f64::clamp.
        let mut lam = vec![0.0];
        solve_pgs_iteration(&mut lam, &[1.0], &[1.0], &[2.0], &[1.0]);
        assert_eq!(lam, [1.0]);
        // NaN bounds are ignored.
        let mut lam = vec![0.0];
        solve_pgs_iteration(&mut lam, &[3.0], &[1.0], &[f64::NAN], &[f64::NAN]);
        assert_eq!(lam, [3.0]);
        // Infinite bounds mean unbounded.
        let mut lam = vec![0.0];
        solve_pgs_iteration(
            &mut lam,
            &[-7.0],
            &[1.0],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
        );
        assert_eq!(lam, [-7.0]);
    }

    #[test]
    fn test_solve_pgs_iteration_stops_at_the_shortest_slice() {
        let mut lam = vec![0.0, 0.0, 9.0];
        let res = solve_pgs_iteration(
            &mut lam,
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0],
            &[0.0; 3],
            &[5.0; 3],
        );
        assert_eq!(lam, [1.0, 1.0, 9.0], "the third constraint has no diagonal");
        assert!((res - 2.0f64.sqrt()).abs() < 1e-15);
        // Nothing at all to process: no panic and no residual.
        let mut lam = vec![4.0];
        assert_eq!(solve_pgs_iteration(&mut lam, &[], &[], &[], &[]), 0.0);
        assert_eq!(lam, [4.0]);
    }

    #[test]
    fn test_solve_pgs_iteration_skips_a_massless_constraint() {
        let mut lam = vec![0.5, 0.0];
        solve_pgs_iteration(&mut lam, &[3.0, 2.0], &[0.0, 2.0], &[0.0; 2], &[9.0; 2]);
        assert_eq!(lam, [0.5, 1.0]);
    }

    #[test]
    fn test_pgs_iteration_inputs_are_validated() {
        let ok = check_pgs_iteration_inputs(&[0.0], &[1.0], &[1.0], &[0.0], &[1.0]);
        assert_eq!(ok, Ok(()));
        assert_eq!(
            check_pgs_iteration_inputs(&[0.0; 2], &[1.0; 2], &[1.0], &[0.0; 2], &[1.0; 2]),
            Err(PgsInputError::LengthMismatch {
                name: "diag",
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            check_pgs_iteration_inputs(&[0.0], &[1.0], &[1.0], &[2.0], &[1.0]),
            Err(PgsInputError::InvertedBounds {
                index: 0,
                lo: 2.0,
                hi: 1.0
            })
        );
        assert_eq!(
            check_pgs_iteration_inputs(&[0.0], &[1.0], &[1.0], &[f64::NAN], &[1.0]),
            Err(PgsInputError::NanBound {
                name: "lo",
                index: 0
            })
        );
        assert_eq!(
            check_pgs_iteration_inputs(&[0.0], &[1.0], &[1.0], &[0.0], &[f64::NAN]),
            Err(PgsInputError::NanBound {
                name: "hi",
                index: 0
            })
        );
        assert!(matches!(
            check_pgs_iteration_inputs(&[0.0], &[f64::NAN], &[1.0], &[0.0], &[1.0]),
            Err(PgsInputError::NotFinite { name: "rhs", .. })
        ));
        assert!(matches!(
            check_pgs_iteration_inputs(&[0.0], &[1.0], &[f64::INFINITY], &[0.0], &[1.0]),
            Err(PgsInputError::NotFinite { name: "diag", .. })
        ));
        assert!(matches!(
            check_pgs_iteration_inputs(&[0.0], &[1.0], &[-1.0], &[0.0], &[1.0]),
            Err(PgsInputError::NegativeDiagonal { name: "diag", .. })
        ));
        // Infinite bounds are legitimate.
        assert_eq!(
            check_pgs_iteration_inputs(
                &[0.0],
                &[1.0],
                &[1.0],
                &[f64::NEG_INFINITY],
                &[f64::INFINITY]
            ),
            Ok(())
        );
    }

    #[test]
    fn test_py_solve_pgs_iteration_raises_value_error_instead_of_panicking() {
        with_python(|py| {
            let inverted =
                py_solve_pgs_iteration(vec![0.0], vec![1.0], vec![1.0], vec![2.0], vec![1.0]);
            let err = inverted.expect_err("lo > hi must be rejected");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(err.to_string().contains("lo[0]"), "{err}");
            let nan =
                py_solve_pgs_iteration(vec![0.0], vec![1.0], vec![1.0], vec![f64::NAN], vec![1.0]);
            assert!(
                nan.expect_err("NaN bound")
                    .is_instance_of::<PyValueError>(py)
            );
            let short = py_solve_pgs_iteration(
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
            );
            let err = short.expect_err("a short diag list must be rejected");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let (lambda, residual) =
                py_solve_pgs_iteration(vec![0.0], vec![1.0], vec![2.0], vec![0.0], vec![10.0])
                    .expect("valid input");
            assert_eq!(lambda, [0.5]);
            assert_eq!(residual, 0.5);
        });
    }

    // ---- coupled solve -----------------------------------------------------

    #[test]
    fn test_solve_matches_the_analytic_solution_of_a_coupled_lcp() {
        // [2 1; 1 2] lambda = [1 1] has the interior solution [1/3, 1/3].
        let lambda = solver(SolverType::Pgs, 200)
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[f64::INFINITY; 2], None)
            .expect("valid system");
        assert_close(&lambda, &[1.0 / 3.0, 1.0 / 3.0], 1e-12);
        // The same matrix with rhs [1, -1]: the free solution [1, -1] leaves the
        // box, so lambda_2 sits at its lower bound 0 and lambda_1 = rhs_1 / 2.
        let lambda = solver(SolverType::Pgs, 200)
            .try_solve(&A, &[1.0, -1.0], &[0.0; 2], &[f64::INFINITY; 2], None)
            .expect("valid system");
        assert_close(&lambda, &[0.5, 0.0], 1e-12);
        // An active upper bound: lambda_1 = 0.5, lambda_2 = (3 - 0.5) / 2.
        let lambda = solver(SolverType::Pgs, 200)
            .try_solve(&A, &[3.0, 3.0], &[0.0; 2], &[0.5, 10.0], None)
            .expect("valid system");
        assert_close(&lambda, &[0.5, 1.25], 1e-12);
    }

    #[test]
    fn test_solve_couples_the_constraints() {
        // A diagonal-only solve would answer rhs / diag = [0.5, 0.5].
        let lambda = solver(SolverType::Pgs, 200)
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert!((lambda[0] - 0.5).abs() > 0.1, "{lambda:?}");
    }

    #[test]
    fn test_solve_applies_sor_factor_per_sweep() {
        let mut config = solver(SolverType::Pgs, 1);
        config.sor_factor = 1.3;
        let lambda = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        // lambda_1 = 1.3 * 1 / 2, lambda_2 = 1.3 * (1 - 0.65) / 2.
        assert_close(&lambda, &[0.65, 0.2275], 1e-15);
        // Over-relaxation still converges to the same fixed point.
        config.velocity_iterations = 200;
        let lambda = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert_close(&lambda, &[1.0 / 3.0, 1.0 / 3.0], 1e-12);
    }

    #[test]
    fn test_si_does_not_over_relax() {
        let mut config = solver(SolverType::Si, 1);
        config.sor_factor = 1.3;
        let lambda = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert_close(&lambda, &[0.5, 0.25], 1e-15);
        // The factor is not read for Si, so even a nonsensical one is accepted.
        config.sor_factor = 7.0;
        assert!(
            config
                .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
                .is_ok()
        );
    }

    #[test]
    fn test_tgs_runs_position_iterations_after_the_velocity_sweeps() {
        let mut config = solver(SolverType::Tgs, 1);
        config.position_iterations = 60;
        let tgs = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert_close(&tgs, &[1.0 / 3.0, 1.0 / 3.0], 1e-9);
        // Pgs and Si ignore position_iterations.
        for kind in [SolverType::Pgs, SolverType::Si] {
            let mut config = solver(kind, 1);
            config.position_iterations = 60;
            let one_sweep = config
                .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
                .expect("valid system");
            assert_close(&one_sweep, &[0.5, 0.25], 1e-15);
        }
    }

    #[test]
    fn test_warm_start_uses_lambda0_only_when_enabled() {
        let exact = [1.0 / 3.0, 1.0 / 3.0];
        let mut config = solver(SolverType::Pgs, 1);
        config.warm_start = true;
        let warm = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], Some(&exact))
            .expect("valid system");
        assert_close(&warm, &exact, 1e-15);
        // Without warm_start the same single sweep starts from zero.
        config.warm_start = false;
        let cold = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], Some(&exact))
            .expect("valid system");
        assert_close(&cold, &[0.5, 0.25], 1e-15);
        // warm_start without a lambda0 starts from zero too.
        config.warm_start = true;
        let none = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert_close(&none, &[0.5, 0.25], 1e-15);
    }

    #[test]
    fn test_start_is_projected_into_the_bounds() {
        let mut config = solver(SolverType::Pgs, 0);
        config.warm_start = true;
        let lambda = config
            .try_solve(
                &A,
                &[1.0, 1.0],
                &[0.0, 0.0],
                &[1.0, 1.0],
                Some(&[5.0, -5.0]),
            )
            .expect("valid system");
        assert_eq!(lambda, [1.0, 0.0], "zero sweeps return the projected start");
        // A cold start from zero is projected as well.
        let lambda = solver(SolverType::Pgs, 0)
            .try_solve(&A, &[1.0, 1.0], &[2.0, -3.0], &[4.0, -1.0], None)
            .expect("valid system");
        assert_eq!(lambda, [2.0, -1.0]);
    }

    #[test]
    fn test_solve_stops_once_the_sweep_residual_is_below_tolerance() {
        let mut config = solver(SolverType::Pgs, 1_000_000);
        config.tolerance = 1e-6;
        let lambda = config
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], None)
            .expect("valid system");
        // Far fewer than a million sweeps were needed, and the result is close
        // but not exact: the loop ended on the tolerance.
        assert_close(&lambda, &[1.0 / 3.0, 1.0 / 3.0], 1e-5);
        assert!((lambda[0] - 1.0 / 3.0).abs() > 1e-16);
    }

    #[test]
    fn test_solve_leaves_a_massless_constraint_alone() {
        // The first constraint has no effective mass: it stays at its start.
        let a = [0.0, 0.0, 0.0, 2.0];
        let lambda = solver(SolverType::Pgs, 5)
            .try_solve(&a, &[4.0, 4.0], &[-1.0; 2], &[10.0; 2], None)
            .expect("valid system");
        assert_eq!(lambda, [0.0, 2.0]);
    }

    #[test]
    fn test_solve_rejects_invalid_input_with_typed_errors() {
        let config = solver(SolverType::Pgs, 5);
        let solve =
            |a: &[f64], rhs: &[f64], lo: &[f64], hi: &[f64]| config.try_solve(a, rhs, lo, hi, None);
        // A diagonal passed where the full matrix is expected (the old call shape).
        assert_eq!(
            solve(&[2.0, 2.0], &[1.0, 1.0], &[0.0; 2], &[1.0; 2]),
            Err(PgsInputError::LengthMismatch {
                name: "a",
                expected: 4,
                got: 2
            })
        );
        assert_eq!(
            solve(&A, &[1.0, 1.0], &[0.0], &[1.0; 2]),
            Err(PgsInputError::LengthMismatch {
                name: "lo",
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            solve(&A, &[1.0, 1.0], &[0.0; 2], &[1.0; 3]),
            Err(PgsInputError::LengthMismatch {
                name: "hi",
                expected: 2,
                got: 3
            })
        );
        assert_eq!(
            solve(&A, &[1.0, 1.0], &[0.0, 2.0], &[1.0, 1.0]),
            Err(PgsInputError::InvertedBounds {
                index: 1,
                lo: 2.0,
                hi: 1.0
            })
        );
        assert!(matches!(
            solve(&A, &[1.0, f64::NAN], &[0.0; 2], &[1.0; 2]),
            Err(PgsInputError::NotFinite {
                name: "rhs",
                index: 1,
                ..
            })
        ));
        let nan_matrix = [2.0, f64::NAN, 1.0, 2.0];
        assert!(matches!(
            solve(&nan_matrix, &[1.0, 1.0], &[0.0; 2], &[1.0; 2]),
            Err(PgsInputError::NotFinite {
                name: "a",
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            solve(&[2.0, 1.0, 1.0, -2.0], &[1.0, 1.0], &[0.0; 2], &[1.0; 2]),
            Err(PgsInputError::NegativeDiagonal {
                name: "a",
                index: 1,
                ..
            })
        ));
        assert!(matches!(
            solve(&A, &[1.0, 1.0], &[f64::NAN, 0.0], &[1.0; 2]),
            Err(PgsInputError::NanBound {
                name: "lo",
                index: 0
            })
        ));
        let mut warm = solver(SolverType::Pgs, 5);
        warm.warm_start = true;
        assert_eq!(
            warm.try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[1.0; 2], Some(&[0.0])),
            Err(PgsInputError::LengthMismatch {
                name: "lambda0",
                expected: 2,
                got: 1
            })
        );
        // An empty system has nothing to solve.
        assert_eq!(solve(&[], &[], &[], &[]), Ok(vec![]));
    }

    #[test]
    fn test_sor_factor_must_lie_strictly_between_zero_and_two() {
        for bad in [0.0, -0.5, 2.0, 3.0, f64::NAN, f64::INFINITY] {
            for kind in [SolverType::Pgs, SolverType::Tgs] {
                let mut config = solver(kind, 5);
                config.sor_factor = bad;
                let result = config.try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[1.0; 2], None);
                assert!(
                    matches!(result, Err(PgsInputError::InvalidSorFactor(_))),
                    "sor_factor {bad}: {result:?}"
                );
            }
        }
        let mut config = solver(SolverType::Pgs, 5);
        for good in [0.1, 1.0, 1.9] {
            config.sor_factor = good;
            assert!(
                config
                    .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[1.0; 2], None)
                    .is_ok()
            );
        }
    }

    #[test]
    fn test_python_solve_raises_value_error() {
        with_python(|py| {
            let config = PyConstraintSolver::default_pgs();
            let err = config
                .solve(
                    vec![2.0, 2.0],
                    vec![1.0, 1.0],
                    vec![0.0; 2],
                    vec![1.0; 2],
                    None,
                )
                .expect_err("a diagonal is not a matrix");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = config
                .solve(
                    A.to_vec(),
                    vec![1.0, 1.0],
                    vec![0.0, 3.0],
                    vec![1.0, 1.0],
                    None,
                )
                .expect_err("lo > hi");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let lambda = config
                .solve(
                    A.to_vec(),
                    vec![1.0, 1.0],
                    vec![0.0; 2],
                    vec![10.0; 2],
                    None,
                )
                .expect("valid system");
            assert_eq!(lambda.len(), 2);
            assert!(lambda.iter().all(|l| *l > 0.0));
        });
    }

    #[test]
    fn test_default_presets_honor_their_configuration() {
        let pgs = PyConstraintSolver::default_pgs();
        assert_eq!(pgs.solver_type, SolverType::Pgs);
        assert_eq!(pgs.sor_factor, 1.0);
        let tgs = PyConstraintSolver::default_tgs();
        assert_eq!(tgs.solver_type, SolverType::Tgs);
        assert_eq!(tgs.sweep_count(), 6);
        assert_eq!(pgs.sweep_count(), 10);
        // The default TGS (1.3 over-relaxation, 6 sweeps, warm start from the
        // exact solution) keeps the exact solution.
        let exact = [1.0 / 3.0, 1.0 / 3.0];
        let lambda = tgs
            .try_solve(&A, &[1.0, 1.0], &[0.0; 2], &[10.0; 2], Some(&exact))
            .expect("valid system");
        assert_close(&lambda, &exact, 1e-9);
    }

    #[test]
    fn test_sweep_never_panics_on_short_inputs() {
        let mut lambda = vec![0.0, 0.0];
        // A matrix with only the first row: the second row is left alone.
        let residual = solve_pgs_sweep(
            &mut lambda,
            &[2.0, 1.0],
            &[1.0, 1.0],
            &[0.0; 2],
            &[9.0; 2],
            1.0,
        );
        assert_eq!(lambda, [0.5, 0.0]);
        assert_eq!(residual, 0.5);
        // No matrix at all.
        let mut lambda = vec![1.0];
        assert_eq!(
            solve_pgs_sweep(&mut lambda, &[], &[1.0], &[0.0], &[1.0], 1.0),
            0.0
        );
        assert_eq!(lambda, [1.0]);
        // Inverted bounds clamp to the upper bound without panicking.
        let mut lambda = vec![0.0];
        solve_pgs_sweep(&mut lambda, &[1.0], &[1.0], &[2.0], &[1.0], 1.0);
        assert_eq!(lambda, [1.0]);
    }
}
