//! Core [`LearnedMixtureKernel`] type.
//!
//! Implements the forward pass `K_mix = sum_i p_i * K_i` and the analytical
//! gradient `dK_mix/dw_i = p_i * (K_i - K_mix)` with numerically stable
//! softmax (max subtraction).

use std::fmt;
use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// A differentiable mixture over a library of base kernels.
///
/// The mixture is parameterised by a vector of logits `w`. Weights
/// `p = softmax(w)` are always strictly positive and sum to 1. The
/// evaluation is
///
/// ```text
/// K_mix(x, y) = sum_i p_i * K_i(x, y).
/// ```
///
/// Logits are unconstrained real numbers; the softmax parameterisation
/// guarantees a valid convex combination on the simplex, which keeps the
/// mixture positive semi-definite when every base kernel is PSD.
#[derive(Clone)]
pub struct LearnedMixtureKernel {
    base_kernels: Vec<Arc<dyn Kernel>>,
    logits: Vec<f64>,
}

impl LearnedMixtureKernel {
    /// Build a mixture from a non-empty library and matching logits.
    ///
    /// Errors when the library is empty, the vectors disagree in length,
    /// or any logit is non-finite.
    pub fn new(base_kernels: Vec<Arc<dyn Kernel>>, logits: Vec<f64>) -> Result<Self> {
        if base_kernels.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "base_kernels".to_string(),
                value: "[]".to_string(),
                reason: "learned mixture requires at least one base kernel".to_string(),
            });
        }
        if base_kernels.len() != logits.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![base_kernels.len()],
                got: vec![logits.len()],
                context: "LearnedMixtureKernel logits length".to_string(),
            });
        }
        for (i, &w) in logits.iter().enumerate() {
            if !w.is_finite() {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("logits[{}]", i),
                    value: w.to_string(),
                    reason: "logits must be finite".to_string(),
                });
            }
        }
        Ok(Self {
            base_kernels,
            logits,
        })
    }

    /// Build a mixture with uniform logits (all zeros → equal weights).
    pub fn uniform(base_kernels: Vec<Arc<dyn Kernel>>) -> Result<Self> {
        let n = base_kernels.len();
        Self::new(base_kernels, vec![0.0; n])
    }

    /// Number of base kernels in the library.
    pub fn num_kernels(&self) -> usize {
        self.base_kernels.len()
    }

    /// Immutable view of the raw logits.
    pub fn logits(&self) -> &[f64] {
        &self.logits
    }

    /// Softmax weights `p_i = softmax(w)_i`. Always strictly positive,
    /// always sums to 1 in exact arithmetic.
    pub fn weights(&self) -> Vec<f64> {
        softmax(&self.logits)
    }

    /// Replace the logits. The new vector must match `num_kernels()` and
    /// every element must be finite.
    pub fn set_logits(&mut self, new_logits: Vec<f64>) -> Result<()> {
        if new_logits.len() != self.base_kernels.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.base_kernels.len()],
                got: vec![new_logits.len()],
                context: "LearnedMixtureKernel::set_logits".to_string(),
            });
        }
        for (i, &w) in new_logits.iter().enumerate() {
            if !w.is_finite() {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("logits[{}]", i),
                    value: w.to_string(),
                    reason: "logits must be finite".to_string(),
                });
            }
        }
        self.logits = new_logits;
        Ok(())
    }

    /// Apply a raw gradient update `w_i <- w_i - lr * g_i` in place.
    /// Used by [`crate::learned_composition::TrainableKernelMixture`].
    pub fn apply_gradient_step(&mut self, gradient: &[f64], learning_rate: f64) -> Result<()> {
        if gradient.len() != self.logits.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.logits.len()],
                got: vec![gradient.len()],
                context: "LearnedMixtureKernel::apply_gradient_step".to_string(),
            });
        }
        if !learning_rate.is_finite() {
            return Err(KernelError::InvalidParameter {
                parameter: "learning_rate".to_string(),
                value: learning_rate.to_string(),
                reason: "must be finite".to_string(),
            });
        }
        for (w, &g) in self.logits.iter_mut().zip(gradient.iter()) {
            if !g.is_finite() {
                return Err(KernelError::InvalidParameter {
                    parameter: "gradient".to_string(),
                    value: g.to_string(),
                    reason: "gradient entries must be finite".to_string(),
                });
            }
            *w -= learning_rate * g;
        }
        Ok(())
    }

    /// Compute base-kernel values `[K_1(x,y), ..., K_n(x,y)]`.
    fn per_kernel_values(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(self.base_kernels.len());
        for kernel in &self.base_kernels {
            values.push(kernel.compute(x, y)?);
        }
        Ok(values)
    }

    /// Evaluate the mixture on a single input pair.
    pub fn evaluate(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let weights = self.weights();
        let mut acc = 0.0;
        for (kernel, &w) in self.base_kernels.iter().zip(weights.iter()) {
            acc += w * kernel.compute(x, y)?;
        }
        Ok(acc)
    }

    /// Return the analytical gradient `dK_mix/dw_i = p_i * (K_i - K_mix)`.
    ///
    /// This form is numerically cleaner than routing through the full
    /// softmax Jacobian (it stays bounded as `p_i` concentrates mass).
    pub fn gradient_wrt_logits(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        let weights = self.weights();
        let k_vals = self.per_kernel_values(x, y)?;
        let k_mix: f64 = weights.iter().zip(k_vals.iter()).map(|(p, k)| p * k).sum();
        Ok(weights
            .iter()
            .zip(k_vals.iter())
            .map(|(p, k)| p * (k - k_mix))
            .collect())
    }

    /// Return the forward value and the full gradient in one pass — the
    /// preferred API for optimizer steps (avoids redundant evaluations).
    pub fn evaluate_with_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, Vec<f64>)> {
        let weights = self.weights();
        let k_vals = self.per_kernel_values(x, y)?;
        let k_mix: f64 = weights.iter().zip(k_vals.iter()).map(|(p, k)| p * k).sum();
        let grad: Vec<f64> = weights
            .iter()
            .zip(k_vals.iter())
            .map(|(p, k)| p * (k - k_mix))
            .collect();
        Ok((k_mix, grad))
    }

    /// Compute a Gram matrix `G[i,j] = K_mix(xs[i], ys[j])` over two sets
    /// of raw slices. Works for square `xs == ys` and rectangular cross-
    /// evaluation alike.
    pub fn compute_gram(&self, xs: &[&[f64]], ys: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
        let mut matrix = vec![vec![0.0; ys.len()]; xs.len()];
        for (i, &xi) in xs.iter().enumerate() {
            for (j, &yj) in ys.iter().enumerate() {
                matrix[i][j] = self.evaluate(xi, yj)?;
            }
        }
        Ok(matrix)
    }
}

impl fmt::Debug for LearnedMixtureKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.base_kernels.iter().map(|k| k.name()).collect();
        f.debug_struct("LearnedMixtureKernel")
            .field("base_kernels", &names)
            .field("logits", &self.logits)
            .finish()
    }
}

impl Kernel for LearnedMixtureKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.evaluate(x, y)
    }

    fn name(&self) -> &str {
        "LearnedMixture"
    }

    fn is_psd(&self) -> bool {
        // Softmax weights are strictly positive and sum to 1; the mixture
        // is PSD whenever every base kernel is PSD.
        self.base_kernels.iter().all(|k| k.is_psd())
    }
}

/// Numerically stable softmax: subtract the max before exponentiating.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // `max` is finite by construction in `new()` / `set_logits()`.
    let shifted: Vec<f64> = logits.iter().map(|&w| (w - max).exp()).collect();
    let denom: f64 = shifted.iter().sum();
    if denom <= 0.0 || !denom.is_finite() {
        // Degenerate fallback — all exponentials underflowed or overflowed.
        // Return the uniform distribution.
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }
    shifted.iter().map(|&e| e / denom).collect()
}
