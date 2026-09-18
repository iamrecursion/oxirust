//! Vector-valued GP (VVGP) regression using any [`MultiOutputKernel`].
//!
//! ## Model
//!
//! Given N training inputs `{x_i}` and corresponding output vectors
//! `{y_i ∈ R^p}`, the VVGP prior is:
//!
//! ```text
//! vec(Y) | X  ~  N(0, K_full + noise * I_{Np})
//! ```
//!
//! where `K_full ∈ R^{Np × Np}` is the block Gram matrix produced by any
//! [`MultiOutputKernel`], and `vec(Y)` stacks the output vectors in row-major
//! order: `[y_0[0], …, y_0[p-1], y_1[0], …]`.
//!
//! ## Posterior
//!
//! After fitting, the posterior at a new input `x*` is:
//!
//! ```text
//! μ(x*)   = K_*(x*) · α           where α = (K_full + noise·I)^{-1} vec(Y)
//! Σ(x*)   = K_**(x*) − V^T V      where V = L^{-1} K_*(x*)^T, LL^T = K_full + noise·I
//! ```
//!
//! `K_*(x*)` is the `p × Np` cross-covariance, `K_**(x*)` is the `p × p`
//! prior block covariance at `x*`, and `L` is the lower Cholesky factor.
//!
//! ## Numerical implementation
//!
//! Cholesky decomposition is used throughout for stability.  The Cholesky
//! factor `L` (lower triangular) is stored in [`VvgpFitted`] and reused for
//! both the posterior mean and variance computations without re-factorising.

use std::sync::Arc;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::{cholesky, solve_triangular};

use crate::error::KernelError;

use super::trait_def::MultiOutputKernel;

type Result<T> = std::result::Result<T, KernelError>;

/// Unfitted vector-valued GP model.
///
/// Holds the kernel and noise level; call [`VvgpModel::fit`] to obtain a
/// [`VvgpFitted`] that supports posterior inference.
pub struct VvgpModel {
    kernel: Arc<dyn MultiOutputKernel>,
    noise: f64,
}

/// Fitted vector-valued GP model storing the Cholesky factor and dual variable
/// alpha needed for O(Np) posterior prediction at new inputs.
pub struct VvgpFitted {
    /// Dual variable α = (K + noise·I)^{-1} vec(Y) of length `N·p`.
    pub alpha: Array1<f64>,
    /// Training inputs (N × d feature vectors).
    inputs: Vec<Vec<f64>>,
    /// Lower-triangular Cholesky factor L of `K_full + noise·I`.
    chol: Array2<f64>,
    /// The kernel, shared with the parent model.
    kernel: Arc<dyn MultiOutputKernel>,
    /// Observation noise added to the diagonal.
    noise: f64,
    /// Number of outputs p.
    n_outputs: usize,
}

impl VvgpModel {
    /// Create a new unfitted VVGP model.
    ///
    /// # Arguments
    /// * `kernel` – Any `Arc<dyn MultiOutputKernel>` implementing the
    ///   `p`-output covariance structure.
    /// * `noise` – Non-negative isotropic observation noise added to the
    ///   block Gram matrix diagonal before Cholesky factorisation.
    pub fn new(kernel: Arc<dyn MultiOutputKernel>, noise: f64) -> Result<Self> {
        if noise < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "noise".to_string(),
                value: noise.to_string(),
                reason: "noise variance must be >= 0".to_string(),
            });
        }
        Ok(Self { kernel, noise })
    }

    /// Fit the GP to N training pairs `(inputs[n], targets[n])`.
    ///
    /// `targets[n]` must be a `Vec<f64>` of length `p = kernel.n_outputs()`.
    /// The method:
    ///
    /// 1. Assembles the `(N·p × N·p)` block Gram matrix.
    /// 2. Adds `noise · I_{Np}` to the diagonal.
    /// 3. Computes the lower Cholesky factor `L`.
    /// 4. Solves `(K + noise·I) α = vec(Y)` via two triangular solves.
    pub fn fit(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<VvgpFitted> {
        let n = inputs.len();
        let p = self.kernel.n_outputs();

        if targets.len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n],
                got: vec![targets.len()],
                context: "VvgpModel::fit: targets.len() must equal inputs.len()".to_string(),
            });
        }
        for (idx, t) in targets.iter().enumerate() {
            if t.len() != p {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![p],
                    got: vec![t.len()],
                    context: format!("VvgpModel::fit: targets[{}] must have length p={}", idx, p),
                });
            }
        }

        let np = n * p;

        // Assemble block Gram matrix and add noise to diagonal.
        let mut k_full = self.kernel.block_gram_matrix(inputs)?;
        for i in 0..np {
            k_full[[i, i]] += self.noise;
        }

        // Stack targets into flat vector (row-major: n-th output vector occupies
        // rows [n*p .. (n+1)*p]).
        let mut y_flat = Array1::<f64>::zeros(np);
        for (n_idx, target) in targets.iter().enumerate() {
            for (p_idx, &v) in target.iter().enumerate() {
                y_flat[n_idx * p + p_idx] = v;
            }
        }

        // Lower Cholesky factor L such that L L^T = K_full + noise·I.
        let chol = cholesky(&k_full.view(), None).map_err(|e| {
            KernelError::ComputationError(format!(
                "VvgpModel::fit: Cholesky failed (matrix may not be PSD): {}",
                e
            ))
        })?;

        // Solve L w = y_flat  (forward substitution).
        let w = solve_triangular(&chol.view(), &y_flat.view(), true, false).map_err(|e| {
            KernelError::ComputationError(format!(
                "VvgpModel::fit: forward triangular solve failed: {}",
                e
            ))
        })?;

        // Solve L^T alpha = w  (back substitution with transpose).
        let chol_t = chol.t().to_owned();
        let alpha = solve_triangular(&chol_t.view(), &w.view(), false, false).map_err(|e| {
            KernelError::ComputationError(format!(
                "VvgpModel::fit: back triangular solve failed: {}",
                e
            ))
        })?;

        Ok(VvgpFitted {
            alpha,
            inputs: inputs.to_vec(),
            chol,
            kernel: Arc::clone(&self.kernel),
            noise: self.noise,
            n_outputs: p,
        })
    }
}

impl VvgpFitted {
    /// Compute the posterior predictive mean and covariance at a test point
    /// `x_star`.
    ///
    /// # Returns
    ///
    /// `(mean, cov)` where:
    /// - `mean` is a `Vec<f64>` of length `p`.
    /// - `cov` is an `Array2<f64>` of shape `(p, p)`.
    ///
    /// The posterior mean is `K_*(x*) α` and the posterior covariance is
    /// `K_**(x*) − V^T V` where `V = L^{-1} K_*(x*)^T` (shape `Np × p`).
    pub fn predict(&self, x_star: &[f64]) -> Result<(Vec<f64>, Array2<f64>)> {
        let n = self.inputs.len();
        let p = self.n_outputs;
        let np = n * p;

        // Build K_star: p × Np cross-covariance.
        // K_star[output_idx, j*p + ci] = K_block(x_star, x_j)[output_idx, ci]
        let mut k_star = Array2::<f64>::zeros((p, np));
        for j in 0..n {
            let block = self.kernel.compute_block(x_star, &self.inputs[j])?;
            for ri in 0..p {
                for ci in 0..p {
                    k_star[[ri, j * p + ci]] = block[[ri, ci]];
                }
            }
        }

        // Prior covariance at x_star: K_ss = K_block(x_star, x_star).
        let k_ss = self.kernel.compute_block(x_star, x_star)?;

        // Posterior mean: μ = K_star · α  (p-vector).
        let mean_arr = k_star.dot(&self.alpha);
        let mean: Vec<f64> = mean_arr.into_raw_vec_and_offset().0;

        // Posterior covariance: Σ = K_ss − V^T V where V = L^{-1} K_star^T.
        // K_star^T has shape Np × p; we solve L V = K_star^T column by column.
        let k_star_t = k_star.t().to_owned(); // Np × p
        let mut v = Array2::<f64>::zeros((np, p));
        for col_idx in 0..p {
            let col = k_star_t.column(col_idx).to_owned();
            let v_col =
                solve_triangular(&self.chol.view(), &col.view(), true, false).map_err(|e| {
                    KernelError::ComputationError(format!(
                        "VvgpFitted::predict: triangular solve for column {} failed: {}",
                        col_idx, e
                    ))
                })?;
            for row_idx in 0..np {
                v[[row_idx, col_idx]] = v_col[row_idx];
            }
        }

        // Cov = K_ss - V^T V  where V^T has shape p × Np.
        let cov = k_ss - v.t().dot(&v);

        Ok((mean, cov))
    }

    /// Number of training points.
    pub fn n_train(&self) -> usize {
        self.inputs.len()
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Observation noise used during fitting.
    pub fn noise(&self) -> f64 {
        self.noise
    }
}
