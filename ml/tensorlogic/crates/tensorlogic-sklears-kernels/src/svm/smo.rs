//! Sequential Minimal Optimization (SMO) solver for SVM kernels.
//!
//! Implements Platt's SMO algorithm (1998) with Keerthi et al. (2001) modifications
//! for efficient solving of the C-SVM dual optimization problem.
//!
//! ## Mathematical Background
//!
//! The C-SVM dual problem is:
//!
//! ```text
//! maximize  W(α) = Σ_i α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
//! subject to  0 ≤ α_i ≤ C  ∀i
//!             Σ_i α_i y_i = 0
//! ```
//!
//! SMO decomposes this into minimal 2-variable sub-problems that have analytic solutions.
//! At each step, two variables (α_i1, α_i2) are selected, the others are fixed, and
//! the sub-problem is solved exactly using the KKT conditions.
//!
//! ## References
//!
//! - Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training
//!   Support Vector Machines. MSR-TR-98-14.
//! - Keerthi, S.S. et al. (2001). Improvements to Platt's SMO algorithm for SVM
//!   classifier design. Neural Computation 13(3).

use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Configuration for the SMO solver.
#[derive(Debug, Clone)]
pub struct SmoConfig {
    /// Regularization parameter C (box constraint upper bound).
    /// Larger C → harder margin (less regularization).
    pub c: f64,
    /// KKT violation tolerance. Points with |E_i * y_i| > tol are candidates for
    /// optimization. Typical values: 1e-3.
    pub tol: f64,
    /// ε-insensitive tube half-width for SVR (unused in SVC). Default 0.1.
    pub epsilon: f64,
    /// Maximum number of passes through the full training set without progress.
    /// A "pass" occurs in the outer loop; total kernel evaluations can be much larger.
    pub max_iter: usize,
}

impl Default for SmoConfig {
    fn default() -> Self {
        SmoConfig {
            c: 1.0,
            tol: 1e-3,
            epsilon: 0.1,
            max_iter: 10_000,
        }
    }
}

/// Internal mutable SMO state, separated so we can pass it cleanly to `take_step`.
struct SmoState {
    alpha: Vec<f64>,
    b: f64,
    error_cache: Vec<f64>,
    n: usize,
    c: f64,
    tol: f64,
    y: Vec<f64>,
    /// Pre-computed full kernel matrix K[i][j] = kernel(x_i, x_j).
    kernel_matrix: Vec<Vec<f64>>,
}

impl SmoState {
    /// Compute the SVM decision function value at training index `i`
    /// using the full kernel matrix:
    ///   f(x_i) = Σ_j α_j y_j K(x_j, x_i) - b
    fn decision_function(&self, i: usize) -> f64 {
        let mut sum = 0.0;
        for j in 0..self.n {
            let aj = self.alpha[j];
            if aj.abs() > 1e-12 {
                sum += aj * self.y[j] * self.kernel_matrix[j][i];
            }
        }
        sum - self.b
    }

    /// Refresh the error cache entry for index `i`:
    ///   E_i = f(x_i) - y_i
    fn refresh_error(&mut self, i: usize) {
        self.error_cache[i] = self.decision_function(i) - self.y[i];
    }

    /// Returns true if index `i` is a "non-bound" support vector:
    ///   0 < α_i < C  (i.e., strictly inside the feasible box)
    fn is_non_bound(&self, i: usize) -> bool {
        self.alpha[i] > 0.0 && self.alpha[i] < self.c
    }

    /// Clip value to [lo, hi].
    fn clip(val: f64, lo: f64, hi: f64) -> f64 {
        val.max(lo).min(hi)
    }

    /// Execute the SMO take_step sub-routine for indices (i1, i2).
    ///
    /// Returns `true` if a meaningful update was made (α changed by more than
    /// the numerical epsilon threshold).
    fn take_step(&mut self, i1: usize, i2: usize) -> Result<bool> {
        if i1 == i2 {
            return Ok(false);
        }

        let a1 = self.alpha[i1];
        let a2 = self.alpha[i2];
        let y1 = self.y[i1];
        let y2 = self.y[i2];
        let e1 = self.error_cache[i1];
        let e2 = self.error_cache[i2];
        let s = y1 * y2;

        // Compute box constraints (L, H) for the new α_2.
        // These arise from the linear equality constraint Σ α_i y_i = 0
        // and the box constraints 0 ≤ α_i ≤ C.
        let (lo, hi) = if (s - 1.0).abs() > 1e-10 {
            // y1 ≠ y2  (opposite signs)
            (f64::max(0.0, a2 - a1), f64::min(self.c, self.c + a2 - a1))
        } else {
            // y1 == y2 (same signs)
            (f64::max(0.0, a1 + a2 - self.c), f64::min(self.c, a1 + a2))
        };

        if lo >= hi {
            return Ok(false);
        }

        let k11 = self.kernel_matrix[i1][i1];
        let k12 = self.kernel_matrix[i1][i2];
        let k22 = self.kernel_matrix[i2][i2];

        // Second derivative of the objective along the α_2 direction.
        // η = K(x1,x1) + K(x2,x2) - 2*K(x1,x2) ≥ 0 for PSD kernels.
        let eta = k11 + k22 - 2.0 * k12;

        let a2_new = if eta > 1e-12 {
            // Normal case: unique maximum along the constrained line.
            // Unconstrained update: α_2^new = α_2 + y_2*(E_1 - E_2)/η
            let a2_unc = a2 + y2 * (e1 - e2) / eta;
            Self::clip(a2_unc, lo, hi)
        } else {
            // Degenerate case: η ≤ 0 (kernel is not strictly PD at these points,
            // or numerical issues). Evaluate objective at endpoints.
            //
            // Objective: W = α_1 + α_2 - (1/2)[K11*α1² + K22*α2² + 2*s*K12*α1*α2
            //                                    + 2*y1*α1*(Σ_{k≠1,2} α_k y_k K_{k1})
            //                                    + 2*y2*α2*(Σ_{k≠1,2} α_k y_k K_{k2})]
            // Since we only vary along the line α_1 = a1_orig + s*(a2_orig - α_2):
            // W(α_2) is quadratic (or linear if η=0). We evaluate at the endpoints.
            //
            // Let γ = a1 + s*a2 (conserved by the update: a1_new + s*a2_new = γ).
            let gamma = a1 + s * a2;
            // At α_2 = lo: α_1 = γ - s*lo
            let a1_at_lo = gamma - s * lo;
            // At α_2 = hi: α_1 = γ - s*hi
            let a1_at_hi = gamma - s * hi;

            // The objective terms that depend on α_1 and α_2 (dropping constants):
            // W_clip = α_1 + α_2 - (1/2)(K11*α1² + 2*s*K12*α1*α2 + K22*α2²)
            //        - v1*y1*α1 - v2*y2*α2
            // where v_i = Σ_{k≠1,2} α_k y_k K_{k,i} = (e_i + b - α_i*y_i*K_{i,i} - α_j*y_j*K_{j,i})*y_i
            // However the simplest way is to note that objective *difference* between
            // the two endpoints can be computed without v terms if we substitute
            // the equality constraint. Using the compact form (Platt 1998, eqs 19-21):

            // f1 = f(x_1) = E_1 + y_1  (E_i = f(x_i) - y_i => f(x_i) = E_i + y_i)
            let f1 = e1 + y1;
            let f2 = e2 + y2;

            // Lobj = W at α_2 = lo
            let a1_lo = a1_at_lo;
            let a2_lo = lo;
            let lobj = -0.5 * k11 * a1_lo * a1_lo
                - 0.5 * k22 * a2_lo * a2_lo
                - s * k12 * a1_lo * a2_lo
                - y1 * a1_lo * f1
                - y2 * a2_lo * f2
                + a1_lo
                + a2_lo;

            // Hobj = W at α_2 = hi
            let a1_hi = a1_at_hi;
            let a2_hi = hi;
            let hobj = -0.5 * k11 * a1_hi * a1_hi
                - 0.5 * k22 * a2_hi * a2_hi
                - s * k12 * a1_hi * a2_hi
                - y1 * a1_hi * f1
                - y2 * a2_hi * f2
                + a1_hi
                + a2_hi;

            if lobj > hobj + 1e-12 {
                lo
            } else if hobj > lobj + 1e-12 {
                hi
            } else {
                // Objectives are equal; no improvement possible.
                a2
            }
        };

        // If the change in α_2 is negligibly small, skip the update.
        // This avoids an infinite loop when numerical precision prevents progress.
        if (a2_new - a2).abs() < 1e-5 * (a2_new + a2 + 1e-10) {
            return Ok(false);
        }

        // Update α_1 using the linear equality constraint:
        //   a1_new = a1 + s*(a2 - a2_new)
        let a1_new = a1 + s * (a2 - a2_new);

        // --- Update bias b ---
        //
        // The bias is chosen so that the KKT condition f(x_i) = y_i holds
        // for each non-bound updated variable.
        //
        // Derivation (at x_1, if a1_new is non-bound):
        //   f_new(x_1) = Σ_j α_j_new y_j K(x_j, x_1) - b_new = y_1
        //   => b_new = Σ_j α_j_new y_j K(x_j, x_1) - y_1
        //            = [old sum] + y1*(a1_new-a1)*K11 + y2*(a2_new-a2)*K12 - y_1
        //            = (E_1 + y_1 + b_old) + y1*(a1_new-a1)*K11 + y2*(a2_new-a2)*K12 - y_1
        //            = b_old + E_1 + y1*(a1_new-a1)*K11 + y2*(a2_new-a2)*K12
        let b_old = self.b;

        // Candidate b from KKT conditions at x_1 (if a1_new is non-bound).
        let b1 = b_old + e1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12;
        // Candidate b from KKT conditions at x_2 (if a2_new is non-bound).
        let b2 = b_old + e2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22;

        let b_new = if a1_new > 1e-8 * self.c && a1_new < self.c * (1.0 - 1e-8) {
            // α_1 is non-bound: use b1 (KKT at x_1 is tight)
            b1
        } else if a2_new > 1e-8 * self.c && a2_new < self.c * (1.0 - 1e-8) {
            // α_2 is non-bound: use b2 (KKT at x_2 is tight)
            b2
        } else {
            // Both α are at bounds; average the two candidates.
            (b1 + b2) * 0.5
        };

        // Commit the updates.
        self.alpha[i1] = a1_new;
        self.alpha[i2] = a2_new;
        self.b = b_new;

        // Update error cache for all non-bound examples.
        // For non-bound j:
        //   E_j^new = E_j^old + y1*(a1_new-a1)*K_{j,1} + y2*(a2_new-a2)*K_{j,2}
        //             - (b_new - b_old)
        let delta_alpha1 = a1_new - a1;
        let delta_alpha2 = a2_new - a2;
        let delta_b = b_new - b_old;
        for j in 0..self.n {
            if self.is_non_bound(j) {
                self.error_cache[j] += y1 * delta_alpha1 * self.kernel_matrix[j][i1]
                    + y2 * delta_alpha2 * self.kernel_matrix[j][i2]
                    - delta_b;
            }
        }
        // Recompute exact errors for i1 and i2.
        // (Their non-bound status may have changed.)
        self.refresh_error(i1);
        self.refresh_error(i2);

        Ok(true)
    }

    /// Choose the second index i1 given i2 using the SMO heuristics:
    ///
    /// 1. Heuristic: pick non-bound j ≠ i2 that maximises |E_j - E_{i2}|.
    /// 2. Fallback: scan all non-bound examples (starting from random offset).
    /// 3. Last resort: scan all examples (starting from random offset).
    ///
    /// Returns `true` if a successful `take_step` was found.
    fn examine_example(&mut self, i2: usize, random_offset: usize) -> Result<bool> {
        // For bound variables (α=0 or C) the error cache may be stale because
        // incremental updates only apply to non-bound variables.
        // Always use the exact decision function for the KKT check at i2.
        if !self.is_non_bound(i2) {
            self.refresh_error(i2);
        }
        let e2 = self.error_cache[i2];
        let r2 = e2 * self.y[i2];

        // KKT check: only examine if the KKT condition is violated.
        let kkt_violated =
            (r2 < -self.tol && self.alpha[i2] < self.c) || (r2 > self.tol && self.alpha[i2] > 0.0);
        if !kkt_violated {
            return Ok(false);
        }

        // Collect non-bound indices.
        let non_bound: Vec<usize> = (0..self.n).filter(|&j| self.is_non_bound(j)).collect();

        // Heuristic step 1: find non-bound j ≠ i2 maximising |E_j - E_{i2}|.
        if non_bound.len() > 1 {
            let mut best_i1 = None;
            let mut best_diff = 0.0;
            for &j in &non_bound {
                if j == i2 {
                    continue;
                }
                let diff = (self.error_cache[j] - e2).abs();
                if diff > best_diff {
                    best_diff = diff;
                    best_i1 = Some(j);
                }
            }
            if let Some(i1) = best_i1 {
                if self.take_step(i1, i2)? {
                    return Ok(true);
                }
            }
        }

        // Heuristic step 2: scan all non-bound examples, starting at random offset.
        if !non_bound.is_empty() {
            let start = random_offset % non_bound.len();
            for k in 0..non_bound.len() {
                let i1 = non_bound[(start + k) % non_bound.len()];
                if i1 == i2 {
                    continue;
                }
                if self.take_step(i1, i2)? {
                    return Ok(true);
                }
            }
        }

        // Step 3: scan all training examples, starting at random offset.
        let start = random_offset % self.n;
        for k in 0..self.n {
            let i1 = (start + k) % self.n;
            if i1 == i2 {
                continue;
            }
            if self.take_step(i1, i2)? {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// C-SVM binary SMO solver.
///
/// Solves the dual SVM problem for binary classification with labels ±1.
///
/// # Arguments
///
/// * `x`      – Training inputs, `n` feature vectors of equal dimension.
/// * `y`      – Binary labels ±1.0, length `n`.
/// * `kernel` – Any `Arc<dyn Kernel>` (must be PSD for guaranteed convergence).
/// * `config` – Solver hyperparameters (C, tol, max_iter).
///
/// # Returns
///
/// `(alpha, b)` where:
/// - `alpha[i]` ∈ [0, C] is the Lagrange multiplier for training point `i`.
/// - `b` is the threshold (bias term).
///
/// The decision function is: `f(x) = Σ_i α_i y_i K(x_i, x) - b`.
///
/// # Errors
///
/// Returns [`KernelError::ComputationError`] if the algorithm does not converge
/// within `config.max_iter` passes.  Returns [`KernelError::DimensionMismatch`]
/// for shape mismatches, [`KernelError::InvalidParameter`] for bad parameters.
pub fn smo_svc(
    x: &[Vec<f64>],
    y: &[f64],
    kernel: &Arc<dyn Kernel>,
    config: &SmoConfig,
) -> Result<(Vec<f64>, f64)> {
    let n = x.len();

    if n == 0 {
        return Err(KernelError::DimensionMismatch {
            expected: vec![1],
            got: vec![0],
            context: "smo_svc: training set cannot be empty".to_string(),
        });
    }
    if y.len() != n {
        return Err(KernelError::DimensionMismatch {
            expected: vec![n],
            got: vec![y.len()],
            context: "smo_svc: y must have the same length as x".to_string(),
        });
    }
    if config.c <= 0.0 {
        return Err(KernelError::InvalidParameter {
            parameter: "C".to_string(),
            value: config.c.to_string(),
            reason: "C must be strictly positive".to_string(),
        });
    }
    // Validate that labels are ±1.
    for (i, &yi) in y.iter().enumerate() {
        if (yi - 1.0).abs() > 1e-9 && (yi + 1.0).abs() > 1e-9 {
            return Err(KernelError::InvalidParameter {
                parameter: "y".to_string(),
                value: yi.to_string(),
                reason: format!("label at index {} must be +1 or -1", i),
            });
        }
    }
    // Check all inputs have the same dimension.
    if n > 1 {
        let d0 = x[0].len();
        for (i, xi) in x.iter().enumerate().skip(1) {
            if xi.len() != d0 {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![d0],
                    got: vec![xi.len()],
                    context: format!("smo_svc: x[{}] has wrong dimension", i),
                });
            }
        }
    }

    // Pre-compute full n×n kernel matrix. This is O(n²) space and time.
    // For large datasets a cached or lazy approach would be preferred, but
    // for correctness and clarity we precompute everything.
    let mut kernel_matrix = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let k_val = kernel.compute(&x[i], &x[j])?;
            kernel_matrix[i][j] = k_val;
            kernel_matrix[j][i] = k_val;
        }
    }

    // Initialize solver state.
    let alpha = vec![0.0_f64; n];
    let b = 0.0_f64;

    // Initialize error cache: E_i = f(x_i) - y_i.
    // With α=0 and b=0: f(x_i) = 0, so E_i = -y_i.
    let error_cache: Vec<f64> = y.iter().map(|&yi| -yi).collect();

    let mut state = SmoState {
        alpha,
        b,
        error_cache,
        n,
        c: config.c,
        tol: config.tol,
        y: y.to_vec(),
        kernel_matrix,
    };

    // SMO main loop (Platt 1998, Figure 2).
    //
    // Loop invariant:
    //   - Start each pass with examine_all indicating whether to scan all examples.
    //   - After a full-all pass: set examine_all=false, continue with non-bound passes.
    //   - After a non-bound pass with numChanged > 0: continue non-bound passes.
    //   - After a non-bound pass with numChanged == 0: set examine_all=true (do a full
    //     verification pass).
    //   - After a full-all pass with numChanged == 0: CONVERGED.
    //
    // This correctly implements Platt's termination criterion.
    let mut examine_all = true;
    let mut total_passes = 0usize;
    // A deterministic but varied offset to break symmetry in tie-breaking.
    let mut random_offset: usize = 17;

    loop {
        let mut num_changed = 0usize;
        let was_examine_all = examine_all;

        if examine_all {
            // Scan all training examples.
            for i2 in 0..n {
                if state.examine_example(i2, random_offset)? {
                    num_changed += 1;
                }
                random_offset = random_offset
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    & 0xFFFF;
            }
        } else {
            // Scan only non-bound examples (those with 0 < α_i < C).
            let non_bound_indices: Vec<usize> = (0..n).filter(|&j| state.is_non_bound(j)).collect();
            for i2 in non_bound_indices {
                if state.examine_example(i2, random_offset)? {
                    num_changed += 1;
                }
                random_offset = random_offset
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    & 0xFFFF;
            }
        }

        total_passes += 1;
        if total_passes > config.max_iter {
            return Err(KernelError::ComputationError(format!(
                "SMO did not converge after {} passes (tol={}, C={}). \
                 Consider increasing max_iter, relaxing tol, or adjusting C.",
                config.max_iter, config.tol, config.c
            )));
        }

        // Convergence check: if we just completed a full-all pass with 0 changes,
        // we are at KKT optimality (within tol).
        if was_examine_all && num_changed == 0 {
            break;
        }

        // Update the examine_all flag for the next pass.
        if was_examine_all {
            // After a full-all pass that made changes: switch to non-bound-only scans.
            examine_all = false;
        } else if num_changed == 0 {
            // After a non-bound pass with 0 changes: do a full verification pass.
            examine_all = true;
        }
        // (If was non-bound pass with changes, keep examine_all=false: scan non-bound again.)
    }

    Ok((state.alpha, state.b))
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    struct TestLinearKernel;
    impl crate::types::Kernel for TestLinearKernel {
        fn compute(&self, x: &[f64], y: &[f64]) -> crate::error::Result<f64> {
            Ok(x.iter().zip(y.iter()).map(|(a, b)| a * b).sum())
        }
        fn name(&self) -> &str {
            "TestLinear"
        }
    }

    #[test]
    fn test_smo_config_default() {
        let cfg = SmoConfig::default();
        assert_eq!(cfg.c, 1.0);
        assert_eq!(cfg.tol, 1e-3);
        assert_eq!(cfg.epsilon, 0.1);
        assert_eq!(cfg.max_iter, 10_000);
    }

    #[test]
    fn test_smo_empty_data() {
        let kernel: Arc<dyn crate::types::Kernel> = Arc::new(TestLinearKernel);
        let cfg = SmoConfig::default();
        let result = smo_svc(&[], &[], &kernel, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_smo_invalid_c() {
        let kernel: Arc<dyn crate::types::Kernel> = Arc::new(TestLinearKernel);
        let cfg = SmoConfig {
            c: -1.0,
            ..Default::default()
        };
        let x = vec![vec![1.0, 2.0]];
        let y = vec![1.0];
        let result = smo_svc(&x, &y, &kernel, &cfg);
        assert!(matches!(result, Err(KernelError::InvalidParameter { .. })));
    }
}
