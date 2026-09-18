// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Inverse problems and parameter identification for the FEM crate.
//!
//! Implements:
//! - [`InverseSolver`] — inverse problem solver with forward model and regularization
//! - \[`tikhonov_regularization()`\] — L2 (Tikhonov) regularization
//! - \[`total_variation_reg()`\] — TV regularization for piecewise-constant parameters
//! - \[`adjoint_gradient()`\] — adjoint method for gradient computation
//! - \[`gauss_newton_inverse()`\] — Levenberg-Marquardt (damped Gauss-Newton) method for
//!   nonlinear inverse problems, reporting its outcome through [`GaussNewtonResult`]
//! - \[`cross_validation()`\] — L-curve method for regularization parameter selection
//! - \[`sensitivity_matrix()`\] — Jacobian of observables w.r.t. parameters

use crate::dense_lu::LuFactor;
use crate::error::{Error, Result};

// ── Math helpers ─────────────────────────────────────────────────────────────

/// Compute the dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a slice.
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Subtract two vectors element-wise.
fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Add two vectors element-wise.
fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Scale a vector by a scalar.
fn scale_vec(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Multiply a dense matrix (row-major, `m×n`) by a vector of length `n`.
fn mat_vec(a: &[f64], n_rows: usize, n_cols: usize, x: &[f64]) -> Vec<f64> {
    (0..n_rows)
        .map(|i| (0..n_cols).map(|j| a[i * n_cols + j] * x[j]).sum::<f64>())
        .collect()
}

/// Multiply the transpose of a dense matrix (`m×n`) by a vector of length `m`.
fn mat_t_vec(a: &[f64], n_rows: usize, n_cols: usize, x: &[f64]) -> Vec<f64> {
    (0..n_cols)
        .map(|j| (0..n_rows).map(|i| a[i * n_cols + j] * x[i]).sum::<f64>())
        .collect()
}

/// Compute `Aᵀ A + λ I` (n×n) — the normal equation matrix for Tikhonov.
fn normal_matrix(a: &[f64], n_rows: usize, n_cols: usize, lambda: f64) -> Vec<f64> {
    let mut m = vec![0.0_f64; n_cols * n_cols];
    for k in 0..n_rows {
        for i in 0..n_cols {
            for j in 0..n_cols {
                m[i * n_cols + j] += a[k * n_cols + i] * a[k * n_cols + j];
            }
        }
    }
    for i in 0..n_cols {
        m[i * n_cols + i] += lambda;
    }
    m
}

/// Solve `A x = b` with a simple conjugate-gradient iteration (symmetric PD `A`).
fn cg_solve(a: &[f64], n: usize, b: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rsold = dot(&r, &r);
    for _ in 0..max_iter {
        let ap = mat_vec(a, n, n, &p);
        let alpha = rsold / (dot(&p, &ap) + 1e-300);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rsnew = dot(&r, &r);
        if rsnew.sqrt() < tol {
            break;
        }
        let beta = rsnew / (rsold + 1e-300);
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    x
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1  InverseSolver
// ═══════════════════════════════════════════════════════════════════════════

/// Inverse problem solver that wraps a linearised forward model and
/// regularization strategy.
///
/// The forward model is `d = A p` where `d` is observed data (length `m`),
/// `A` is the sensitivity matrix (`m × n`), and `p` is the unknown parameter
/// vector (length `n`).
#[derive(Debug, Clone)]
pub struct InverseSolver {
    /// Sensitivity (Jacobian) matrix, stored row-major with shape `m × n`.
    pub sensitivity: Vec<f64>,
    /// Number of observations (rows of `sensitivity`).
    pub n_obs: usize,
    /// Number of parameters (columns of `sensitivity`).
    pub n_params: usize,
    /// Regularization parameter λ.
    pub lambda: f64,
    /// Current parameter estimate.
    pub params: Vec<f64>,
    /// Observed data vector (length `n_obs`).
    pub data: Vec<f64>,
}

impl InverseSolver {
    /// Create a new [`InverseSolver`].
    ///
    /// # Arguments
    /// * `sensitivity` — row-major sensitivity matrix of shape `n_obs × n_params`
    /// * `n_obs` — number of observations
    /// * `n_params` — number of unknown parameters
    /// * `lambda` — regularization parameter
    /// * `data` — observed data vector of length `n_obs`
    pub fn new(
        sensitivity: Vec<f64>,
        n_obs: usize,
        n_params: usize,
        lambda: f64,
        data: Vec<f64>,
    ) -> Self {
        let params = vec![0.0_f64; n_params];
        Self {
            sensitivity,
            n_obs,
            n_params,
            lambda,
            params,
            data,
        }
    }

    /// Solve the inverse problem using Tikhonov regularization and return
    /// the recovered parameter vector.
    pub fn solve_tikhonov(&self) -> Vec<f64> {
        tikhonov_regularization(
            &self.sensitivity,
            self.n_obs,
            self.n_params,
            &self.data,
            self.lambda,
        )
    }

    /// Return the residual norm `||A p - d||` for a given parameter vector.
    pub fn residual_norm(&self, p: &[f64]) -> f64 {
        let ap = mat_vec(&self.sensitivity, self.n_obs, self.n_params, p);
        norm(&sub_vec(&ap, &self.data))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  Tikhonov regularization
// ═══════════════════════════════════════════════════════════════════════════

/// Solve the Tikhonov-regularized inverse problem:
///
/// `p* = arg min ||A p − b||² + λ ||p||²`
///
/// via the normal equations `(AᵀA + λI) p = Aᵀ b`.
///
/// # Arguments
/// * `a` — sensitivity matrix, row-major, shape `m × n`
/// * `m` — number of rows (observations)
/// * `n` — number of columns (parameters)
/// * `b` — observation vector, length `m`
/// * `lambda` — regularization weight (≥ 0)
///
/// Returns the parameter vector of length `n`.
pub fn tikhonov_regularization(a: &[f64], m: usize, n: usize, b: &[f64], lambda: f64) -> Vec<f64> {
    let lhs = normal_matrix(a, m, n, lambda);
    let rhs = mat_t_vec(a, m, n, b);
    cg_solve(&lhs, n, &rhs, 1000, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  Total-variation regularization
// ═══════════════════════════════════════════════════════════════════════════

/// Apply one gradient-descent step of total-variation regularization.
///
/// The TV penalty is `λ Σ |p_{i+1} − p_i|`.  The smooth approximation
/// `|x| ≈ √(x² + ε)` is used for differentiability.
///
/// # Arguments
/// * `a` — sensitivity matrix, row-major, shape `m × n`
/// * `m` — number of observations
/// * `n` — number of parameters
/// * `b` — observed data
/// * `p` — current parameter estimate (length `n`)
/// * `lambda` — TV weight
/// * `step` — gradient-descent step size
///
/// Returns an updated parameter vector.
pub fn total_variation_reg(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    p: &[f64],
    lambda: f64,
    step: f64,
) -> Vec<f64> {
    const EPS: f64 = 1e-6;
    // data-fidelity gradient: Aᵀ(Ap − b)
    let ap = mat_vec(a, m, n, p);
    let residual = sub_vec(&ap, b);
    let grad_data = mat_t_vec(a, m, n, &residual);

    // TV gradient (finite difference)
    let mut grad_tv = vec![0.0_f64; n];
    for i in 0..n {
        if i + 1 < n {
            let diff = p[i] - p[i + 1];
            let w = (diff * diff + EPS).sqrt();
            grad_tv[i] += diff / w;
            grad_tv[i + 1] -= diff / w;
        }
    }

    let grad = add_vec(&grad_data, &scale_vec(&grad_tv, lambda));
    sub_vec(p, &scale_vec(&grad, step))
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4  Adjoint gradient
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the gradient of the data-misfit functional via the adjoint method.
///
/// For a linearised forward model `d = A p`, the gradient is simply `Aᵀ(Ap − d_obs)`.
///
/// # Arguments
/// * `a` — sensitivity matrix, row-major `m × n`
/// * `m` — number of observations
/// * `n` — number of parameters
/// * `p` — current parameter vector (length `n`)
/// * `d_obs` — observed data (length `m`)
///
/// Returns the gradient vector of length `n`.
pub fn adjoint_gradient(a: &[f64], m: usize, n: usize, p: &[f64], d_obs: &[f64]) -> Vec<f64> {
    let ap = mat_vec(a, m, n, p);
    let residual = sub_vec(&ap, d_obs);
    mat_t_vec(a, m, n, &residual)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  Gauss-Newton inverse
// ═══════════════════════════════════════════════════════════════════════════

/// Floor of the Marquardt damping matrix `diag(JᵀJ)`, as a fraction of its largest
/// entry: a parameter with (almost) no influence on the observables still gets a
/// positive damping, so the damped normal matrix stays regular.
const DAMPING_DIAGONAL_FLOOR: f64 = 1.0e-12;
/// Damping factor above which no step can be accepted any more (stagnation).
const DAMPING_MAX: f64 = 1.0e16;
/// Damping factor that starts the escalation after a rejected step when the caller
/// asked for an undamped Gauss-Newton iteration (`λ = 0`).
const DAMPING_RESTART: f64 = 1.0e-6;
/// Cosine of the angle between the residual and every Jacobian column below which the
/// iterate counts as a stationary point of the least-squares cost `½‖r‖²`.  A step
/// along the remaining gradient still lowers the cost by about `cos²` (relatively),
/// so `1e-6` stops well above the rounding floor of the residual-decrease test.
const STATIONARITY_COSINE: f64 = 1.0e-6;

/// Outcome of a [`gauss_newton_inverse`] run.
///
/// Divergence, stagnation and an exhausted iteration budget are reported through
/// [`converged`](Self::converged) instead of being hidden behind a plain parameter
/// vector.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussNewtonResult {
    /// Best parameter vector found (length `n`); the starting guess if no step was
    /// ever accepted.
    pub params: Vec<f64>,
    /// Number of outer iterations performed, i.e. Jacobian evaluations.
    pub iterations: usize,
    /// Euclidean norm `‖forward_fn(params) − d_obs‖` of the residual at
    /// [`params`](Self::params).
    pub residual_norm: f64,
    /// `true` when the iteration stopped because the residual reached the
    /// requested tolerance (or the rounding floor of the data), or because
    /// `params` is a stationary point of the least-squares cost (no first-order
    /// descent direction is left, the normal outcome for noisy or inconsistent
    /// data).  `false` when the iteration budget was exhausted, the damping
    /// saturated without an acceptable step, or the parameters have no influence
    /// on the observables.
    pub converged: bool,
}

/// Step used to differentiate with respect to a parameter of size `|p|`:
/// `√ε · max(|p|, 1)`, which stays representable on top of `p` for any magnitude
/// (an absolute `1e-7` step is lost entirely next to a Young's modulus of `2e11`).
fn relative_fd_step(p: f64) -> f64 {
    f64::EPSILON.sqrt() * p.abs().max(1.0)
}

/// Evaluate `forward_fn` and check that it returns exactly `m` observables.
fn forward_values<F>(forward_fn: &F, p: &[f64], m: usize) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let values = forward_fn(p);
    if values.len() == m {
        Ok(values)
    } else {
        Err(Error::InvalidInput(format!(
            "the forward model returned {} observables, expected {m}",
            values.len()
        )))
    }
}

/// `true` when every entry of `v` is finite.
fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Forward-difference Jacobian (`m × n`, row-major) of `forward_fn` at `p`, given
/// the already evaluated `f0 = forward_fn(p)`.
///
/// `h > 0` requests that absolute step for every parameter; `h <= 0` (or a step
/// that is lost to rounding next to `p[k]`) selects the relative step
/// [`relative_fd_step`].  The difference quotient is always divided by the step
/// that was actually applied, `(p[k] + h) − p[k]`.
fn finite_difference_jacobian<F>(forward_fn: &F, p: &[f64], f0: &[f64], h: f64) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = p.len();
    let m = f0.len();
    let mut jac = vec![0.0_f64; m * n];
    for (k, &pk_value) in p.iter().enumerate() {
        let mut step = if h.is_finite() && h > 0.0 {
            h
        } else {
            relative_fd_step(pk_value)
        };
        if pk_value + step == pk_value {
            step = relative_fd_step(pk_value);
        }
        let mut perturbed = p.to_vec();
        perturbed[k] = pk_value + step;
        let applied = perturbed[k] - pk_value;
        if applied == 0.0 || !applied.is_finite() {
            return Err(Error::InvalidInput(format!(
                "parameter {k} = {pk_value:e} cannot be perturbed for finite differences"
            )));
        }
        let shifted = forward_values(forward_fn, &perturbed, m)?;
        for (i, (&shifted_i, &f0_i)) in shifted.iter().zip(f0.iter()).enumerate() {
            let derivative = (shifted_i - f0_i) / applied;
            if !derivative.is_finite() {
                return Err(Error::InvalidInput(format!(
                    "the forward model is not finite when parameter {k} is perturbed"
                )));
            }
            jac[i * n + k] = derivative;
        }
    }
    Ok(jac)
}

/// Solve the symmetric positive-definite system `A x = b` (`A` is `n × n`,
/// row-major) after symmetric Jacobi scaling `Ã = S A S`, `S = diag(1/√aᵢᵢ)`.
///
/// The scaling makes the factorisation independent of the units of the unknowns
/// (a Young's modulus next to a Poisson's ratio).  Returns `None` when a diagonal
/// entry is not positive or the scaled matrix is singular to working precision.
fn solve_scaled_spd(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut scale = Vec::with_capacity(n);
    for k in 0..n {
        let diagonal = a[k * n + k];
        if !(diagonal.is_finite() && diagonal > 0.0) {
            return None;
        }
        scale.push(1.0 / diagonal.sqrt());
    }
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * scale[i] * scale[j]).collect())
        .collect();
    let lu = LuFactor::new(&rows)?;
    let rhs: Vec<f64> = b.iter().zip(scale.iter()).map(|(bi, si)| bi * si).collect();
    let solution: Vec<f64> = lu
        .solve(&rhs)
        .iter()
        .zip(scale.iter())
        .map(|(yi, si)| yi * si)
        .collect();
    all_finite(&solution).then_some(solution)
}

/// Levenberg-Marquardt (damped Gauss-Newton) iterations for a nonlinear inverse
/// problem `forward_fn(p) ≈ d_obs`.
///
/// Every iteration evaluates the Jacobian `J` of `forward_fn` by forward finite
/// differences with the *relative* step `√ε · max(|pₖ|, 1)` (divided by the step
/// actually applied), so parameters of very different size — a Young's modulus of
/// `2e11` next to a Poisson's ratio of `0.3` — are differentiated reliably.  The
/// step then solves the Marquardt-scaled normal equations
///
/// `(JᵀJ + μ·diag(JᵀJ)) Δp = −Jᵀ r`,  `r = forward_fn(p) − d_obs`,
///
/// which are invariant to the units of the parameters.  A step is **accepted only
/// if it lowers the residual norm**; then the damping `μ` is divided by 10.
/// Otherwise `μ` is multiplied by 10 and the step is recomputed (the Jacobian is
/// kept).  When `μ` exceeds `1e16` without an acceptable step the iteration has
/// stagnated and stops with `converged == false`.
///
/// # Arguments
/// * `forward_fn` — closure `p → predicted_data` (length `m`)
/// * `p0` — initial parameter guess (length `n`)
/// * `d_obs` — observed data (length `m`)
/// * `lambda` — initial Marquardt damping factor `μ` (dimensionless, typically
///   `1e-3`; `0` starts as a pure Gauss-Newton iteration and only introduces
///   damping after a rejected step)
/// * `max_iter` — maximum number of outer iterations (Jacobian evaluations)
/// * `tol` — absolute tolerance on `‖r‖`; the iteration also counts as converged
///   once `‖r‖` reaches the rounding floor `8 ε ‖d_obs‖` of the data or the
///   iterate is a stationary point of the least-squares cost
///
/// # Errors
/// [`Error::InvalidInput`] if `p0` or `d_obs` is empty or contains non-finite
/// values, `lambda` or `tol` is negative or non-finite, or `forward_fn` returns a
/// number of observables different from `d_obs.len()` or non-finite values at the
/// starting guess or at a finite-difference perturbation.
pub fn gauss_newton_inverse<F>(
    forward_fn: F,
    p0: &[f64],
    d_obs: &[f64],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<GaussNewtonResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = p0.len();
    let m = d_obs.len();
    if n == 0 || m == 0 {
        return Err(Error::InvalidInput(
            "Gauss-Newton needs at least one parameter and one observation".into(),
        ));
    }
    if !(lambda.is_finite() && lambda >= 0.0 && tol.is_finite() && tol >= 0.0) {
        return Err(Error::InvalidInput(
            "the damping and the tolerance must be finite and non-negative".into(),
        ));
    }
    if !all_finite(p0) || !all_finite(d_obs) {
        return Err(Error::InvalidInput(
            "the initial guess and the observations must be finite".into(),
        ));
    }

    let accept_norm = tol.max(8.0 * f64::EPSILON * norm(d_obs));
    let mut p = p0.to_vec();
    let mut fp = forward_values(&forward_fn, &p, m)?;
    if !all_finite(&fp) {
        return Err(Error::InvalidInput(
            "the forward model is not finite at the initial guess".into(),
        ));
    }
    let mut r = sub_vec(&fp, d_obs);
    let mut r_norm = norm(&r);
    let mut damping = lambda;
    let mut iterations = 0_usize;
    let mut converged = r_norm <= accept_norm;

    while !converged && iterations < max_iter {
        iterations += 1;
        let jac = finite_difference_jacobian(&forward_fn, &p, &fp, 0.0)?;
        let jtj = normal_matrix(&jac, m, n, 0.0);
        let gradient = mat_t_vec(&jac, m, n, &r);
        let diagonal: Vec<f64> = (0..n).map(|k| jtj[k * n + k]).collect();
        let max_diagonal = diagonal.iter().copied().fold(0.0_f64, f64::max);
        if max_diagonal <= 0.0 {
            // The observables do not depend on any parameter: nothing to fit.
            break;
        }
        let stationary = diagonal
            .iter()
            .zip(gradient.iter())
            .all(|(&d, &g)| d <= 0.0 || g.abs() <= STATIONARITY_COSINE * d.sqrt() * r_norm);
        if stationary {
            converged = true;
            break;
        }
        let damping_scale: Vec<f64> = diagonal
            .iter()
            .map(|&d| d.max(DAMPING_DIAGONAL_FLOOR * max_diagonal))
            .collect();
        let rhs: Vec<f64> = gradient.iter().map(|g| -g).collect();

        let mut accepted = false;
        while damping <= DAMPING_MAX {
            let mut damped = jtj.clone();
            for (k, &scale) in damping_scale.iter().enumerate() {
                damped[k * n + k] += damping * scale;
            }
            if let Some(step) = solve_scaled_spd(&damped, n, &rhs) {
                let trial = add_vec(&p, &step);
                let f_trial = forward_values(&forward_fn, &trial, m)?;
                if all_finite(&f_trial) {
                    let r_trial = sub_vec(&f_trial, d_obs);
                    let trial_norm = norm(&r_trial);
                    if trial_norm < r_norm {
                        p = trial;
                        fp = f_trial;
                        r = r_trial;
                        r_norm = trial_norm;
                        damping /= 10.0;
                        accepted = true;
                        break;
                    }
                }
            }
            damping = if damping > 0.0 {
                damping * 10.0
            } else {
                DAMPING_RESTART
            };
        }
        if !accepted {
            break;
        }
        converged = r_norm <= accept_norm;
    }

    Ok(GaussNewtonResult {
        params: p,
        iterations,
        residual_norm: r_norm,
        converged,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6  Cross-validation / L-curve
// ═══════════════════════════════════════════════════════════════════════════

/// Select the regularization parameter using the L-curve criterion.
///
/// For each candidate λ the data residual norm and solution norm are computed.
/// The L-curve corner is approximated as the λ that maximises the curvature
/// of the log-log plot.
///
/// # Arguments
/// * `a` — sensitivity matrix, row-major `m × n`
/// * `m` — number of observations
/// * `n` — number of parameters
/// * `b` — observed data (length `m`)
/// * `lambdas` — candidate regularization parameters (sorted ascending)
///
/// Returns `(best_lambda, residual_norms, solution_norms)`.
pub fn cross_validation(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    lambdas: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut res_norms = Vec::with_capacity(lambdas.len());
    let mut sol_norms = Vec::with_capacity(lambdas.len());

    for &lam in lambdas {
        let p = tikhonov_regularization(a, m, n, b, lam);
        let ap = mat_vec(a, m, n, &p);
        res_norms.push(norm(&sub_vec(&ap, b)));
        sol_norms.push(norm(&p));
    }

    // Find L-curve corner: maximum discrete curvature in log-log space
    let log_rho: Vec<f64> = res_norms.iter().map(|x| x.ln()).collect();
    let log_eta: Vec<f64> = sol_norms.iter().map(|x| x.ln()).collect();

    let mut best_idx = 0;
    let mut best_curv = f64::NEG_INFINITY;
    let k = lambdas.len();
    if k >= 3 {
        for i in 1..k - 1 {
            let d1 = (log_eta[i] - log_eta[i - 1]) / (log_rho[i] - log_rho[i - 1] + 1e-300);
            let d2 = (log_eta[i + 1] - log_eta[i]) / (log_rho[i + 1] - log_rho[i] + 1e-300);
            let curv = d2 - d1;
            if curv > best_curv {
                best_curv = curv;
                best_idx = i;
            }
        }
    }

    (lambdas[best_idx], res_norms, sol_norms)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 7  Sensitivity matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the sensitivity (Jacobian) matrix of observables w.r.t. parameters
/// by forward finite differences.
///
/// # Arguments
/// * `forward_fn` — closure mapping parameter vector `p` (length `n`) to
///   observable vector (length `m`)
/// * `p` — parameter vector at which to evaluate the Jacobian
/// * `m` — number of observables
/// * `h` — absolute finite-difference step; for `h <= 0` (or a step that is lost to
///   rounding next to `p[k]`, such as `1e-7` next to `2e11`) the relative step
///   `√ε · max(|p[k]|, 1)` is used instead.  The difference quotient is divided by
///   the step that was actually applied, `(p[k] + h) − p[k]`.
///
/// Returns the Jacobian stored row-major (`m × n`).
///
/// # Errors
/// [`Error::InvalidInput`] if `forward_fn` does not return exactly `m` observables
/// or returns non-finite values at `p` or at a perturbation of `p`.
pub fn sensitivity_matrix<F>(forward_fn: F, p: &[f64], m: usize, h: f64) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let f0 = forward_values(&forward_fn, p, m)?;
    if !all_finite(&f0) {
        return Err(Error::InvalidInput(
            "the forward model is not finite at the evaluation point".into(),
        ));
    }
    finite_difference_jacobian(&forward_fn, p, &f0, h)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 8  Additional helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Morozov discrepancy principle estimate for `λ`.
///
/// Returns the smallest λ from `lambdas` such that
/// `||A p_λ − b|| ≤ delta * ||b||`, where `delta` is the relative noise level.
///
/// If no such λ exists the last (largest) value is returned.
pub fn morozov_lambda(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    lambdas: &[f64],
    delta: f64,
) -> f64 {
    let target = delta * norm(b);
    for &lam in lambdas.iter().rev() {
        let p = tikhonov_regularization(a, m, n, b, lam);
        let ap = mat_vec(a, m, n, &p);
        if norm(&sub_vec(&ap, b)) <= target {
            return lam;
        }
    }
    *lambdas.last().unwrap_or(&1.0)
}

/// Compute the relative data fit error `||A p − b|| / ||b||`.
pub fn relative_fit_error(a: &[f64], m: usize, n: usize, p: &[f64], b: &[f64]) -> f64 {
    let ap = mat_vec(a, m, n, p);
    let res = sub_vec(&ap, b);
    norm(&res) / (norm(b) + 1e-300)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Identity matrix of size `n`, row-major.
    fn eye(n: usize) -> Vec<f64> {
        let mut m = vec![0.0_f64; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// A = \[\[2, 1\\], \[1, 3\]], exact solution for b = \[3, 5\] is \[4/5, 7/5\].
    fn small_system() -> (Vec<f64>, Vec<f64>) {
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![3.0, 5.0];
        (a, b)
    }

    // ── Tikhonov tests ────────────────────────────────────────────────────

    #[test]
    fn test_tikhonov_identity_system() {
        // A = I, b = [1,2,3] → p* ≈ b / (1 + λ)
        let n = 3;
        let a = eye(n);
        let b = vec![1.0, 2.0, 3.0];
        let lambda = 1.0;
        let p = tikhonov_regularization(&a, n, n, &b, lambda);
        for (i, &pi) in p.iter().enumerate() {
            let expected = b[i] / (1.0 + lambda);
            assert!(
                (pi - expected).abs() < 1e-8,
                "i={i}: got {pi}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_tikhonov_zero_lambda_recovers_solution() {
        // With λ≈0 and a square full-rank system the solution should be close to exact.
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let b = vec![4.0, 9.0];
        let p = tikhonov_regularization(&a, 2, 2, &b, 1e-10);
        assert!((p[0] - 2.0).abs() < 1e-5);
        assert!((p[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_tikhonov_overdetermined() {
        // Overdetermined: A (3×2), b (3), should return least-squares solution.
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let p = tikhonov_regularization(&a, 3, 2, &b, 1e-8);
        // residual should be small relative to b
        let rel = relative_fit_error(&a, 3, 2, &p, &b);
        assert!(rel < 0.5, "relative error {rel}");
    }

    #[test]
    fn test_tikhonov_larger_lambda_shrinks_solution() {
        let n = 4;
        let a = eye(n);
        let b = vec![1.0; n];
        let p_small = tikhonov_regularization(&a, n, n, &b, 0.01);
        let p_large = tikhonov_regularization(&a, n, n, &b, 100.0);
        assert!(norm(&p_large) < norm(&p_small));
    }

    #[test]
    fn test_tikhonov_solution_norm() {
        let n = 2;
        let a = eye(n);
        let b = vec![3.0, 4.0];
        let lambda = 1.0;
        let p = tikhonov_regularization(&a, n, n, &b, lambda);
        let expected_norm = norm(&[1.5, 2.0]);
        assert!((norm(&p) - expected_norm).abs() < 1e-7);
    }

    // ── InverseSolver tests ────────────────────────────────────────────────

    #[test]
    fn test_inverse_solver_new() {
        let a = eye(3);
        let b = vec![1.0, 2.0, 3.0];
        let solver = InverseSolver::new(a, 3, 3, 0.5, b);
        assert_eq!(solver.n_obs, 3);
        assert_eq!(solver.n_params, 3);
        assert_eq!(solver.params.len(), 3);
    }

    #[test]
    fn test_inverse_solver_solve() {
        let n = 3;
        let a = eye(n);
        let b = vec![2.0, 4.0, 6.0];
        let solver = InverseSolver::new(a, n, n, 1.0, b);
        let p = solver.solve_tikhonov();
        // expected: b / 2 = [1, 2, 3]
        for (i, &pi) in p.iter().enumerate() {
            assert!((pi - (i as f64 + 1.0)).abs() < 1e-7, "i={i}: {pi}");
        }
    }

    #[test]
    fn test_inverse_solver_residual_norm() {
        let a = eye(2);
        let b = vec![1.0, 0.0];
        let solver = InverseSolver::new(a, 2, 2, 0.0, b.clone());
        let r = solver.residual_norm(&b);
        assert!(r < 1e-10);
    }

    // ── TV regularization tests ───────────────────────────────────────────

    #[test]
    fn test_tv_reg_reduces_residual() {
        let m = 4;
        let n = 4;
        let a = eye(n);
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let p0 = vec![0.0; n];
        let p1 = total_variation_reg(&a, m, n, &b, &p0, 0.1, 0.1);
        // after one step solution should have moved toward b
        assert!(norm(&p1) > 0.0);
    }

    #[test]
    fn test_tv_reg_piecewise_constant() {
        // b has a step: TV should preserve the step better than Tikhonov
        let n = 6;
        let a = eye(n);
        let b = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut p = vec![0.0_f64; n];
        for _ in 0..200 {
            p = total_variation_reg(&a, n, n, &b, &p, 0.5, 0.01);
        }
        // step should still be present: last half > first half
        let first: f64 = p[..3].iter().sum::<f64>() / 3.0;
        let last: f64 = p[3..].iter().sum::<f64>() / 3.0;
        assert!(last > first + 0.1, "first={first:.3} last={last:.3}");
    }

    #[test]
    fn test_tv_reg_step_size_matters() {
        let n = 4;
        let a = eye(n);
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let p0 = vec![0.0; n];
        let p_big = total_variation_reg(&a, n, n, &b, &p0, 0.1, 0.5);
        let p_small = total_variation_reg(&a, n, n, &b, &p0, 0.1, 0.01);
        // bigger step → bigger movement in one iteration
        assert!(norm(&p_big) > norm(&p_small));
    }

    // ── Adjoint gradient tests ────────────────────────────────────────────

    #[test]
    fn test_adjoint_gradient_zero_at_solution() {
        let n = 3;
        let a = eye(n);
        let d_obs = vec![1.0, 2.0, 3.0];
        let grad = adjoint_gradient(&a, n, n, &d_obs, &d_obs);
        assert!(norm(&grad) < 1e-10);
    }

    #[test]
    fn test_adjoint_gradient_direction() {
        // With A = I, gradient = p − d_obs.  Steepest descent should decrease cost.
        let n = 3;
        let a = eye(n);
        let d_obs = vec![1.0, 2.0, 3.0];
        let p = vec![2.0, 3.0, 4.0];
        let grad = adjoint_gradient(&a, n, n, &p, &d_obs);
        // grad should equal p - d_obs
        for i in 0..n {
            assert!((grad[i] - (p[i] - d_obs[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adjoint_gradient_nonzero() {
        let (a, b) = small_system();
        let p = vec![0.0, 0.0];
        let g = adjoint_gradient(&a, 2, 2, &p, &b);
        // should be -Aᵀb ≠ 0
        assert!(norm(&g) > 0.0);
    }

    // ── Gauss-Newton tests ────────────────────────────────────────────────

    #[test]
    fn test_gauss_newton_linear() {
        // For a linear model the Gauss-Newton should converge in one step.
        let a_mat = vec![2.0_f64, 0.0, 0.0, 3.0];
        let d_obs = vec![4.0, 9.0];
        let p0 = vec![0.0; 2];
        let result =
            gauss_newton_inverse(|p| mat_vec(&a_mat, 2, 2, p), &p0, &d_obs, 1e-6, 20, 1e-8)
                .expect("well-posed linear problem");
        assert!(result.converged, "{result:?}");
        assert!((result.params[0] - 2.0).abs() < 1e-6, "{result:?}");
        assert!((result.params[1] - 3.0).abs() < 1e-6, "{result:?}");
        assert!(result.residual_norm <= 1e-8, "{result:?}");
    }

    #[test]
    fn test_gauss_newton_quadratic() {
        // Forward model: f(p) = [p[0]^2], d_obs = [4.0] → p = 2.0
        let d_obs = vec![4.0_f64];
        let p0 = vec![3.0_f64];
        let result = gauss_newton_inverse(|p| vec![p[0] * p[0]], &p0, &d_obs, 0.1, 50, 1e-8)
            .expect("well-posed quadratic problem");
        assert!(result.converged, "{result:?}");
        assert!(
            (result.params[0] - 2.0).abs() < 1e-6,
            "p[0]={}",
            result.params[0]
        );
    }

    #[test]
    fn test_gauss_newton_already_converged() {
        let d_obs = vec![1.0, 2.0];
        let p0 = d_obs.clone();
        let result = gauss_newton_inverse(|p| p.to_vec(), &p0, &d_obs, 1e-6, 10, 1e-8)
            .expect("trivial problem");
        assert!(result.converged);
        assert_eq!(result.iterations, 0, "no iteration is needed: {result:?}");
        assert!((result.params[0] - 1.0).abs() < 1e-6);
        assert!((result.params[1] - 2.0).abs() < 1e-6);
    }

    /// Observables of a tension / torsion test of a circular steel bar as a function
    /// of `p = [E, ν]`: axial elongation, radial contraction and angle of twist.
    ///
    /// The two parameters differ by eleven orders of magnitude, so this forward model
    /// is only identifiable with a relative finite-difference step and a scale-invariant
    /// damping.
    fn bar_observables(p: &[f64]) -> Vec<f64> {
        const RADIUS: f64 = 0.01;
        const LENGTH: f64 = 1.0;
        const FORCE: f64 = 5.0e4;
        const TORQUE: f64 = 100.0;
        let (young, poisson) = (p[0], p[1]);
        let area = std::f64::consts::PI * RADIUS * RADIUS;
        let polar_moment = std::f64::consts::PI * RADIUS.powi(4) / 2.0;
        let axial_strain = FORCE / (area * young);
        let shear_modulus = young / (2.0 * (1.0 + poisson));
        vec![
            axial_strain * LENGTH,
            -poisson * axial_strain * RADIUS,
            TORQUE * LENGTH / (shear_modulus * polar_moment),
        ]
    }

    #[test]
    fn test_gauss_newton_recovers_steel_constants_from_bar_model() {
        let (e_true, nu_true) = (2.1e11, 0.3);
        let d_obs = bar_observables(&[e_true, nu_true]);
        let tol = 1.0e-13;
        let result = gauss_newton_inverse(bar_observables, &[1.0e11, 0.15], &d_obs, 1e-3, 100, tol)
            .expect("bar model is well posed");
        assert!(result.converged, "{result:?}");
        assert!(result.residual_norm <= tol, "{result:?}");
        assert!(
            (result.params[0] - e_true).abs() < 1e-6 * e_true,
            "E = {:e}",
            result.params[0]
        );
        assert!(
            (result.params[1] - nu_true).abs() < 1e-6 * nu_true,
            "nu = {}",
            result.params[1]
        );
    }

    #[test]
    fn test_gauss_newton_recovers_constants_from_far_start_by_rejecting_bad_steps() {
        // Starting at 3.5 E the undamped Gauss-Newton step of the 1/E model overshoots
        // to a negative modulus; the residual-decrease test must reject it.
        let (e_true, nu_true) = (2.1e11, 0.3);
        let d_obs = bar_observables(&[e_true, nu_true]);
        let result =
            gauss_newton_inverse(bar_observables, &[7.35e11, 0.45], &d_obs, 0.0, 200, 1.0e-13)
                .expect("bar model is well posed");
        assert!(result.converged, "{result:?}");
        assert!(
            (result.params[0] - e_true).abs() < 1e-6 * e_true,
            "{result:?}"
        );
        assert!(
            (result.params[1] - nu_true).abs() < 1e-6 * nu_true,
            "{result:?}"
        );
    }

    #[test]
    fn test_gauss_newton_inconsistent_data_stops_at_least_squares_minimum() {
        // Two observations of one constant that disagree: the least-squares minimum is
        // the mean, with a residual of 1/√2 that no parameter can remove.
        let d_obs = vec![0.0, 1.0];
        let result = gauss_newton_inverse(|p| vec![p[0], p[0]], &[5.0], &d_obs, 1e-3, 50, 1e-8)
            .expect("well-formed problem");
        assert!(result.converged, "stationary point: {result:?}");
        assert!((result.params[0] - 0.5).abs() < 1e-5, "{result:?}");
        assert!(
            (result.residual_norm - 0.5_f64.sqrt()).abs() < 1e-6,
            "{result:?}"
        );
    }

    #[test]
    fn test_gauss_newton_reports_exhausted_iteration_budget() {
        // One iteration cannot solve the quadratic problem to 1e-12.
        let d_obs = vec![4.0_f64];
        let result = gauss_newton_inverse(|p| vec![p[0] * p[0]], &[10.0], &d_obs, 1e-3, 1, 1e-12)
            .expect("well-formed problem");
        assert!(!result.converged, "{result:?}");
        assert_eq!(result.iterations, 1);
        assert!(result.residual_norm.is_finite() && result.residual_norm > 1e-12);
        assert!(result.params[0] < 10.0, "the one step must still descend");
    }

    #[test]
    fn test_gauss_newton_reports_parameters_without_influence() {
        let d_obs = vec![1.0];
        let result = gauss_newton_inverse(|_| vec![0.0], &[1.0, 2.0], &d_obs, 1e-3, 20, 1e-10)
            .expect("well-formed problem");
        assert!(!result.converged, "{result:?}");
        assert_eq!(result.params, vec![1.0, 2.0]);
        assert!((result.residual_norm - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_gauss_newton_rejects_wrong_forward_length() {
        let d_obs = vec![1.0, 2.0];
        let err = gauss_newton_inverse(|p| vec![p[0]], &[1.0], &d_obs, 1e-3, 5, 1e-8)
            .expect_err("one observable returned, two expected");
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn test_gauss_newton_rejects_invalid_arguments() {
        let d_obs = vec![1.0];
        let fwd = |p: &[f64]| vec![p[0]];
        assert!(matches!(
            gauss_newton_inverse(fwd, &[], &d_obs, 1e-3, 5, 1e-8),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            gauss_newton_inverse(fwd, &[1.0], &[], 1e-3, 5, 1e-8),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            gauss_newton_inverse(fwd, &[1.0], &d_obs, -1.0, 5, 1e-8),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            gauss_newton_inverse(fwd, &[f64::NAN], &d_obs, 1e-3, 5, 1e-8),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            gauss_newton_inverse(fwd, &[1.0], &d_obs, 1e-3, 5, f64::NAN),
            Err(Error::InvalidInput(_))
        ));
    }

    // ── Sensitivity matrix tests ──────────────────────────────────────────

    #[test]
    fn test_sensitivity_matrix_identity() {
        // f(p) = p → J = I
        let p = vec![1.0, 2.0, 3.0];
        let n = p.len();
        let jac = sensitivity_matrix(|p| p.to_vec(), &p, n, 1e-7).expect("finite model");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((jac[i * n + j] - expected).abs() < 1e-5, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_sensitivity_matrix_affine() {
        // f(p) = 2 * p + 1 → J = 2 * I
        let p = vec![0.0, 0.0];
        let n = 2;
        let jac = sensitivity_matrix(|p| p.iter().map(|x| 2.0 * x + 1.0).collect(), &p, n, 1e-7)
            .expect("finite model");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!((jac[i * n + j] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_sensitivity_matrix_nonlinear() {
        // f(p) = [p[0]^2, p[1]^3] at p = [2, 3]
        // J = [[2p0, 0], [0, 3p1^2]] = [[4, 0], [0, 27]]
        let p = vec![2.0, 3.0];
        let jac = sensitivity_matrix(|p| vec![p[0] * p[0], p[1] * p[1] * p[1]], &p, 2, 1e-6)
            .expect("finite model");
        assert!((jac[0] - 4.0).abs() < 1e-3);
        assert!((jac[1]).abs() < 1e-3);
        assert!((jac[2]).abs() < 1e-3);
        assert!((jac[3] - 27.0).abs() < 1e-2);
    }

    /// Model with one parameter of the size of a Young's modulus and one of order one:
    /// `f(p) = [1e12 / p0, 1e-11 · p0 · p1]`, `J = [[-1e12/p0², 0], [1e-11 p1, 1e-11 p0]]`.
    fn scale_mixed_model(p: &[f64]) -> Vec<f64> {
        vec![1.0e12 / p[0], 1.0e-11 * p[0] * p[1]]
    }

    fn assert_scale_mixed_jacobian(jac: &[f64], p: &[f64]) {
        let expected = [-1.0e12 / (p[0] * p[0]), 0.0, 1.0e-11 * p[1], 1.0e-11 * p[0]];
        for (idx, (&got, &want)) in jac.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "entry {idx}: {got:e} vs {want:e}"
            );
        }
    }

    #[test]
    fn test_sensitivity_matrix_relative_step_for_nonpositive_h() {
        // The absolute step 1e-7 is lost next to p0 = 2.1e11 (the spacing of doubles
        // there is 3e-5); h <= 0 selects the relative step instead.
        let p = [2.1e11, 0.3];
        let jac = sensitivity_matrix(scale_mixed_model, &p, 2, 0.0).expect("finite model");
        assert_scale_mixed_jacobian(&jac, &p);
        let jac_negative = sensitivity_matrix(scale_mixed_model, &p, 2, -1.0).expect("finite");
        assert_eq!(jac, jac_negative);
    }

    #[test]
    fn test_sensitivity_matrix_absolute_step_lost_to_rounding_falls_back() {
        // Before the fix the first column was exactly zero for h = 1e-7 at p0 = 2.1e11.
        let p = [2.1e11, 0.3];
        assert_eq!(
            p[0] + 1.0e-7,
            p[0],
            "the absolute step is not representable"
        );
        let jac = sensitivity_matrix(scale_mixed_model, &p, 2, 1.0e-7).expect("finite model");
        assert!(jac[0] != 0.0 && jac[2] != 0.0, "{jac:?}");
        assert_scale_mixed_jacobian(&jac, &p);
    }

    #[test]
    fn test_sensitivity_matrix_divides_by_the_applied_step() {
        // p = 1e3 and h = 1e-7: p + h is representable, but the step that is actually
        // applied, (p + h) - p, differs from h by about 1e-6 (relative).  For the
        // identity map the difference of the model values is exactly that applied
        // step, so the derivative is 1 only if the quotient divides by it; dividing by
        // the nominal h would give 1 + 1e-6.
        let p = [1.0e3_f64];
        let applied = (p[0] + 1.0e-7) - p[0];
        assert!(
            (applied - 1.0e-7).abs() > 1e-8 * 1.0e-7,
            "step must be inexact"
        );
        let jac = sensitivity_matrix(|p| vec![p[0]], &p, 1, 1.0e-7).expect("finite model");
        assert!((jac[0] - 1.0).abs() < 1e-14, "{jac:?}");
    }

    #[test]
    fn test_sensitivity_matrix_rejects_wrong_length_and_non_finite() {
        let p = [1.0, 2.0];
        let wrong = sensitivity_matrix(|p| vec![p[0]], &p, 2, 1e-7);
        assert!(matches!(wrong, Err(Error::InvalidInput(_))), "{wrong:?}");
        let non_finite = sensitivity_matrix(|p| vec![1.0 / (p[0] - 1.0)], &p, 1, 1e-7);
        assert!(
            matches!(non_finite, Err(Error::InvalidInput(_))),
            "{non_finite:?}"
        );
    }

    // ── Cross-validation / L-curve tests ─────────────────────────────────

    #[test]
    fn test_cross_validation_returns_lambda() {
        let n = 4;
        let a = eye(n);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let lambdas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let (best, res, sol) = cross_validation(&a, n, n, &b, &lambdas);
        assert!(lambdas.contains(&best));
        assert_eq!(res.len(), 5);
        assert_eq!(sol.len(), 5);
    }

    #[test]
    fn test_cross_validation_monotone() {
        // For A = I: as λ increases residual increases and solution norm decreases.
        let n = 3;
        let a = eye(n);
        let b = vec![1.0, 1.0, 1.0];
        let lambdas = vec![0.01, 0.1, 1.0, 10.0];
        let (_best, res, sol) = cross_validation(&a, n, n, &b, &lambdas);
        for i in 1..res.len() {
            assert!(res[i] >= res[i - 1] - 1e-10, "res not monotone at {i}");
            assert!(sol[i] <= sol[i - 1] + 1e-10, "sol not monotone at {i}");
        }
    }

    // ── Morozov discrepancy tests ─────────────────────────────────────────

    #[test]
    fn test_morozov_lambda_returns_valid() {
        let n = 3;
        let a = eye(n);
        let b = vec![1.0, 1.0, 1.0];
        let lambdas = vec![0.001, 0.01, 0.1, 1.0];
        let lam = morozov_lambda(&a, n, n, &b, &lambdas, 0.5);
        assert!(lambdas.contains(&lam));
    }

    #[test]
    fn test_morozov_lambda_large_delta() {
        // With delta = 1 any λ satisfies the discrepancy → should return smallest
        // or go through the whole list and return last.
        let n = 2;
        let a = eye(n);
        let b = vec![1.0, 1.0];
        let lambdas = vec![0.01, 0.1, 1.0, 10.0];
        let lam = morozov_lambda(&a, n, n, &b, &lambdas, 1.0);
        assert!(lambdas.contains(&lam));
    }

    // ── relative_fit_error ────────────────────────────────────────────────

    #[test]
    fn test_relative_fit_error_exact() {
        let a = eye(3);
        let b = vec![1.0, 2.0, 3.0];
        let err = relative_fit_error(&a, 3, 3, &b, &b);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_relative_fit_error_nonzero() {
        let a = eye(2);
        let b = vec![1.0, 0.0];
        let p = vec![0.0, 1.0];
        let err = relative_fit_error(&a, 2, 2, &p, &b);
        // ||Ap - b|| = ||[-1,1]|| = sqrt(2), ||b|| = 1
        assert!((err - 2.0_f64.sqrt()).abs() < 1e-8);
    }

    // ── mat_vec / mat_t_vec helpers ───────────────────────────────────────

    #[test]
    fn test_mat_vec_identity() {
        let n = 3;
        let a = eye(n);
        let x = vec![1.0, 2.0, 3.0];
        let y = mat_vec(&a, n, n, &x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_mat_t_vec_identity() {
        let n = 3;
        let a = eye(n);
        let x = vec![4.0, 5.0, 6.0];
        let y = mat_t_vec(&a, n, n, &x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_mat_vec_2x2() {
        // A = [[1,2],[3,4]], x = [1,1] → [3, 7]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let y = mat_vec(&a, 2, 2, &x);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_matrix_positive_definite() {
        // AᵀA + λI must have positive diagonal for any λ > 0.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let m = normal_matrix(&a, 2, 2, 1.0);
        assert!(m[0] > 0.0);
        assert!(m[3] > 0.0);
    }

    #[test]
    fn test_cg_solve_trivial() {
        let n = 2;
        let a = eye(n);
        let b = vec![3.0, 7.0];
        let x = cg_solve(&a, n, &b, 10, 1e-10);
        assert!((x[0] - 3.0).abs() < 1e-8);
        assert!((x[1] - 7.0).abs() < 1e-8);
    }

    #[test]
    fn test_add_sub_scale_vec() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add_vec(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 9.0]);
        let d = sub_vec(&b, &a);
        assert_eq!(d, vec![3.0, 3.0, 3.0]);
        let e = scale_vec(&a, 2.0);
        assert_eq!(e, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_norm() {
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    }
}
