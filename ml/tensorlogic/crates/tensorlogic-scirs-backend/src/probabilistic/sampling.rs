//! Seeded Monte Carlo samplers for probabilistic execution.
//!
//! This module provides deterministic (seedable) samplers for the most common
//! distributions used in probabilistic programming and Monte Carlo methods.
//! All samplers return `Scirs2Tensor = ArrayD<f64>`.
//!
//! ## RNG pattern
//! The seeded vs unseeded branches explicitly duplicate the `ArrayD::random_bulk(…)`
//! call because `Random<StdRng>` and `Random<ThreadRng>` are different concrete types
//! that cannot be unified behind a single mutable reference in Rust's type system.

use crate::error::{TlBackendError, TlBackendResult};
use crate::Scirs2Tensor;
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};
use scirs2_core::random::arrays::OptimizedArrayRandom;
use scirs2_core::random::prelude::*;

// ============================================================================
// Bernoulli sampler
// ============================================================================

/// Sample Bernoulli random variables, returning values in {0.0, 1.0}.
///
/// Each element is independently drawn from Bernoulli(p).
///
/// # Arguments
/// * `shape`  — desired output shape
/// * `p`      — success probability in [0, 1]
/// * `seed`   — optional RNG seed for reproducibility
///
/// # Errors
/// Returns `TlBackendError::InvalidOperation` if `p` is outside [0, 1].
pub fn sample_bernoulli(
    shape: &[usize],
    p: f64,
    seed: Option<u64>,
) -> TlBackendResult<Scirs2Tensor> {
    if !(0.0..=1.0).contains(&p) {
        return Err(TlBackendError::InvalidOperation(format!(
            "Bernoulli probability p must be in [0, 1], got {p}"
        )));
    }

    let n_elems: usize = shape.iter().product();
    let dyn_shape = IxDyn(shape);
    let uniform_dist =
        Uniform::new(0.0_f64, 1.0).map_err(|e| TlBackendError::InvalidOperation(e.to_string()))?;

    let raw = if let Some(s) = seed {
        let mut rng = seeded_rng(s);
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    } else {
        let mut rng = thread_rng();
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    };

    let _ = n_elems; // consumed by product above, kept for documentation clarity
    Ok(raw.mapv(|u| if u < p { 1.0 } else { 0.0 }))
}

// ============================================================================
// Uniform sampler
// ============================================================================

/// Sample uniformly from [lo, hi).
///
/// # Errors
/// Returns `TlBackendError::InvalidOperation` if `lo >= hi`.
pub fn sample_uniform(
    shape: &[usize],
    lo: f64,
    hi: f64,
    seed: Option<u64>,
) -> TlBackendResult<Scirs2Tensor> {
    if lo >= hi {
        return Err(TlBackendError::InvalidOperation(format!(
            "Uniform requires lo < hi, got lo={lo} hi={hi}"
        )));
    }

    let dyn_shape = IxDyn(shape);
    let uniform_dist =
        Uniform::new(lo, hi).map_err(|e| TlBackendError::InvalidOperation(e.to_string()))?;

    if let Some(s) = seed {
        let mut rng = seeded_rng(s);
        Ok(ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng))
    } else {
        let mut rng = thread_rng();
        Ok(ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng))
    }
}

// ============================================================================
// Normal sampler
// ============================================================================

/// Sample from N(mean, std_dev²) via the reparameterisation: mean + std_dev * ε.
///
/// This function is differentiable with respect to `mean` and `std_dev`.
///
/// # Errors
/// Returns `TlBackendError::InvalidOperation` if `std_dev <= 0`.
pub fn sample_normal(
    shape: &[usize],
    mean: f64,
    std_dev: f64,
    seed: Option<u64>,
) -> TlBackendResult<Scirs2Tensor> {
    if std_dev <= 0.0 {
        return Err(TlBackendError::InvalidOperation(format!(
            "Normal std_dev must be > 0, got {std_dev}"
        )));
    }

    let dyn_shape = IxDyn(shape);
    // Sample standard Normal ε first, then reparameterise
    let normal_dist =
        Normal::new(0.0_f64, 1.0).map_err(|e| TlBackendError::InvalidOperation(e.to_string()))?;

    let eps = if let Some(s) = seed {
        let mut rng = seeded_rng(s);
        ArrayD::random_bulk(dyn_shape, normal_dist, &mut rng)
    } else {
        let mut rng = thread_rng();
        ArrayD::random_bulk(dyn_shape, normal_dist, &mut rng)
    };

    // mean + std_dev * ε
    Ok(eps.mapv(|e| mean + std_dev * e))
}

// ============================================================================
// Categorical sampler via Gumbel-max trick
// ============================================================================

/// Hard sample from a categorical distribution via the Gumbel-max trick.
///
/// Adds Gumbel(0,1) noise to `logits` and takes the argmax along `axis`,
/// returning a one-hot encoded tensor with the same shape as `logits`
/// (the selected class has value 1.0, all others 0.0).
///
/// # Arguments
/// * `logits` — unnormalised log-probabilities; any shape is accepted
/// * `axis`   — the class axis along which to sample
/// * `seed`   — optional RNG seed
///
/// # Errors
/// Returns an error if `axis` is out of range or the logits tensor is empty.
pub fn sample_categorical(
    logits: &Scirs2Tensor,
    axis: usize,
    seed: Option<u64>,
) -> TlBackendResult<Scirs2Tensor> {
    let ndim = logits.ndim();
    if ndim == 0 {
        return Err(TlBackendError::InvalidOperation(
            "sample_categorical: logits tensor must have at least one dimension".to_string(),
        ));
    }
    if axis >= ndim {
        return Err(TlBackendError::InvalidOperation(format!(
            "sample_categorical: axis {axis} is out of range for tensor with {ndim} dimensions"
        )));
    }

    let shape = logits.shape().to_vec();
    let dyn_shape = IxDyn(&shape);

    // Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0,1)
    let uniform_dist = Uniform::new(1e-10_f64, 1.0 - 1e-10)
        .map_err(|e| TlBackendError::InvalidOperation(e.to_string()))?;

    let gumbel_noise = if let Some(s) = seed {
        let mut rng = seeded_rng(s);
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    } else {
        let mut rng = thread_rng();
        ArrayD::random_bulk(dyn_shape, uniform_dist, &mut rng)
    };

    let gumbel_noise = gumbel_noise.mapv(|u: f64| -(-u.ln()).ln());

    // Perturbed logits = logits + Gumbel noise
    let perturbed = logits + &gumbel_noise;

    // Argmax along axis → one-hot
    let n_classes = shape[axis];
    let argmax_indices = perturbed.map_axis(Axis(axis), |lane| {
        lane.iter()
            .enumerate()
            .fold(
                (0_usize, f64::NEG_INFINITY),
                |(best_idx, best_val), (i, &v)| {
                    if v > best_val {
                        (i, v)
                    } else {
                        (best_idx, best_val)
                    }
                },
            )
            .0
    });

    // Build one-hot output
    let mut one_hot = ArrayD::zeros(IxDyn(&shape));
    for (idx_in_argmax, &class_idx) in argmax_indices.iter().enumerate() {
        // Convert linear index in argmax_indices to full index in one_hot
        let argmax_shape = argmax_indices.shape();
        let mut full_idx: Vec<usize> = Vec::with_capacity(ndim);

        // Reconstruct the multi-dimensional index from the linear index,
        // inserting `class_idx` at the `axis` position.
        let mut remainder = idx_in_argmax;
        let mut collapsed_strides: Vec<usize> = Vec::with_capacity(argmax_shape.len());
        let mut stride = 1_usize;
        for &dim in argmax_shape.iter().rev() {
            collapsed_strides.push(stride);
            stride *= dim;
        }
        collapsed_strides.reverse();

        for (dim_i, (&dim, &s)) in argmax_shape
            .iter()
            .zip(collapsed_strides.iter())
            .enumerate()
        {
            let coord = remainder / s;
            remainder %= dim;

            // Insert the axis coordinate at the right position
            if dim_i == axis {
                full_idx.push(class_idx);
            }
            full_idx.push(coord);
        }
        // Handle the axis position if it wasn't inserted during the loop
        // (this happens when axis == ndim-1 but the loop runs over ndim-1 dims)
        if full_idx.len() == ndim - 1 {
            full_idx.insert(axis, class_idx);
        }

        if full_idx.len() == ndim && full_idx[axis] < n_classes {
            one_hot[IxDyn(&full_idx)] = 1.0;
        }
    }

    Ok(one_hot)
}

// ============================================================================
// Monte Carlo integration
// ============================================================================

/// Configuration for Monte Carlo integration.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of MC samples to draw.
    pub num_samples: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            seed: None,
        }
    }
}

/// Estimate E_{p}[f(z)] for p = Uniform\[0, 1\]^shape by Monte Carlo averaging.
///
/// Draws `config.num_samples` samples z ~ Uniform\[0, 1\]^`shape`, evaluates `f(z)`,
/// and returns the element-wise mean. This is an importance-sampling-ready baseline:
/// the caller can reweight by multiplying by p(z) / q(z) where q is Uniform.
///
/// # Errors
/// Returns an error if `f` returns an error on any sample, or if `config.num_samples == 0`.
pub fn mc_integrate<F>(
    f: F,
    shape: &[usize],
    config: MonteCarloConfig,
) -> TlBackendResult<Scirs2Tensor>
where
    F: Fn(&Scirs2Tensor) -> TlBackendResult<Scirs2Tensor>,
{
    if config.num_samples == 0 {
        return Err(TlBackendError::InvalidOperation(
            "mc_integrate: num_samples must be > 0".to_string(),
        ));
    }

    let uniform_dist =
        Uniform::new(0.0_f64, 1.0).map_err(|e| TlBackendError::InvalidOperation(e.to_string()))?;
    let dyn_shape = IxDyn(shape);

    // We accumulate into a running sum and divide at the end.
    let first_seed = config.seed.map(|s| s.wrapping_add(0));
    let z0 = if let Some(s) = first_seed {
        let mut rng = seeded_rng(s);
        ArrayD::random_bulk(dyn_shape.clone(), uniform_dist, &mut rng)
    } else {
        let mut rng = thread_rng();
        ArrayD::random_bulk(dyn_shape.clone(), uniform_dist, &mut rng)
    };

    let mut accumulator = f(&z0)?;

    for sample_idx in 1..config.num_samples {
        let next_seed = config.seed.map(|s| s.wrapping_add(sample_idx as u64));
        let z = if let Some(s) = next_seed {
            let mut rng = seeded_rng(s);
            ArrayD::random_bulk(dyn_shape.clone(), uniform_dist, &mut rng)
        } else {
            let mut rng = thread_rng();
            ArrayD::random_bulk(dyn_shape.clone(), uniform_dist, &mut rng)
        };

        let fz = f(&z)?;
        accumulator = accumulator + fz;
    }

    let n = config.num_samples as f64;
    Ok(accumulator.mapv(|v| v / n))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    #[test]
    fn sample_normal_mean_std() {
        // 10 000 samples; check empirical mean ≈ 2.5, std ≈ 1.3 (within 5%)
        let mean_true = 2.5_f64;
        let std_true = 1.3_f64;
        let samples =
            sample_normal(&[10_000], mean_true, std_true, Some(42)).expect("sample_normal failed");
        assert_eq!(samples.len(), 10_000);

        let empirical_mean = samples.iter().sum::<f64>() / 10_000.0;
        let empirical_var = samples
            .iter()
            .map(|&x| (x - empirical_mean).powi(2))
            .sum::<f64>()
            / 10_000.0;
        let empirical_std = empirical_var.sqrt();

        assert!(
            (empirical_mean - mean_true).abs() < 0.05 * mean_true.abs().max(1.0),
            "mean {empirical_mean} not close to {mean_true}"
        );
        assert!(
            (empirical_std - std_true).abs() < 0.05 * std_true,
            "std {empirical_std} not close to {std_true}"
        );
    }

    #[test]
    fn sample_bernoulli_mean() {
        let p = 0.3_f64;
        let samples = sample_bernoulli(&[10_000], p, Some(99)).expect("sample_bernoulli failed");
        let empirical_mean = samples.iter().sum::<f64>() / 10_000.0;
        // All values must be 0 or 1
        for &v in samples.iter() {
            assert!(v == 0.0 || v == 1.0, "got non-binary value {v}");
        }
        assert!(
            (empirical_mean - p).abs() < 0.05,
            "empirical mean {empirical_mean} not close to p={p}"
        );
    }

    #[test]
    fn sample_uniform_range() {
        let lo = -2.0_f64;
        let hi = 5.0_f64;
        let samples = sample_uniform(&[5000], lo, hi, Some(7)).expect("sample_uniform failed");
        for &v in samples.iter() {
            assert!(v >= lo && v < hi, "value {v} outside [{lo}, {hi})");
        }
    }

    #[test]
    fn sample_categorical_shape() {
        // logits: shape [4, 3] → output must be same shape, row-wise one-hot
        let logits = ArrayD::zeros(IxDyn(&[4, 3]));
        let out = sample_categorical(&logits, 1, Some(11)).expect("sample_categorical failed");
        assert_eq!(out.shape(), &[4, 3]);

        // Each row must sum to 1.0 (exactly one hot per row)
        for row in out.rows() {
            let row_sum: f64 = row.iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row sum {row_sum} != 1.0");
        }
    }

    #[test]
    fn mc_integrate_constant() {
        // f(z) = 3.7 always → E[f] = 3.7
        let constant = 3.7_f64;
        let config = MonteCarloConfig {
            num_samples: 500,
            seed: Some(55),
        };
        let result = mc_integrate(
            |_z| Ok(ArrayD::from_elem(IxDyn(&[1]), constant)),
            &[2],
            config,
        )
        .expect("mc_integrate failed");
        for &v in result.iter() {
            assert!((v - constant).abs() < 1e-10, "got {v}, expected {constant}");
        }
    }
}
