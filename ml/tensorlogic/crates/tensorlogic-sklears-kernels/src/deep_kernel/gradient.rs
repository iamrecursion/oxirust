//! Gradient helpers for [`DeepKernel`]s.
//!
//! Two paths are provided:
//!
//! * [`finite_difference_gradient`] — central differences over the flat
//!   parameter buffer. Works for any base kernel and any feature
//!   extractor that implements [`NeuralFeatureMap`]; `O(2P)` forward
//!   passes in the number of parameters `P` and used by the crate's own
//!   correctness tests as a reference.
//! * [`rbf_dkl_gradient`] — the analytical gradient for the
//!   RBF-base / MLP-extractor special case. Closed form:
//!
//!   `∂K_DKL / ∂θ = K_DKL · (-2γ) · Σ_k (g(x) - g(y))_k · ∂g_k(x)/∂θ`
//!   `             + K_DKL · ( 2γ) · Σ_k (g(x) - g(y))_k · ∂g_k(y)/∂θ`
//!
//!   (the two sums come from `∂/∂θ || g(x) - g(y) ||²`). The Jacobians
//!   `∂g_k(·)/∂θ` are obtained by standard MLP backprop, reusing the
//!   per-layer pre/post-activation cache produced by
//!   [`MLPFeatureExtractor::forward_with_cache`].
//!
//! # Scope (v0.2.0 preview)
//!
//! * Analytical chain rule is implemented for the
//!   [`MLPFeatureExtractor`] + [`RbfKernel`] pair only — i.e. the
//!   paradigmatic DKL configuration. Other combinations must be
//!   gradient-checked via finite differences; autodiff integration is
//!   out of scope for this release.
//! * Gradients w.r.t. base-kernel hyperparameters (e.g. the RBF `γ`)
//!   are **not** implemented here; the mixture side of the workspace
//!   (`learned_composition`) handles that use case.

use crate::deep_kernel::feature_extractor::{LayerCache, MLPFeatureExtractor, NeuralFeatureMap};
use crate::deep_kernel::kernel::DeepKernel;
use crate::deep_kernel::layer::Activation;
use crate::error::{KernelError, Result};
use crate::tensor_kernels::RbfKernel;
use crate::types::Kernel;

/// Numerical gradient `∂K_DKL/∂θ` via central finite differences on the
/// flat parameter buffer. Returns a vector of length
/// `kernel.feature_extractor().parameter_count()`.
///
/// The caller must pass an `MLPFeatureExtractor` (or any extractor that
/// shares the `parameters` / `sync_from_flat` contract) — the helper
/// needs to perturb the flat buffer and then push the update back into
/// the layer weights before the next forward pass.
pub fn finite_difference_gradient<K: Kernel>(
    kernel: &mut DeepKernel<MLPFeatureExtractor, K>,
    x: &[f64],
    y: &[f64],
    h: f64,
) -> Result<Vec<f64>> {
    if !(h.is_finite() && h > 0.0) {
        return Err(KernelError::InvalidParameter {
            parameter: "h".to_string(),
            value: h.to_string(),
            reason: "finite-difference step must be a positive finite number".to_string(),
        });
    }
    let p = kernel.feature_extractor().parameter_count();
    let mut grad = Vec::with_capacity(p);
    let baseline = kernel.feature_extractor().parameters().to_vec();
    for i in 0..p {
        let mut plus = baseline.clone();
        plus[i] += h;
        kernel
            .feature_extractor_mut()
            .parameters_mut()
            .copy_from_slice(&plus);
        kernel.feature_extractor_mut().sync_from_flat()?;
        let f_plus = kernel.compute(x, y)?;

        let mut minus = baseline.clone();
        minus[i] -= h;
        kernel
            .feature_extractor_mut()
            .parameters_mut()
            .copy_from_slice(&minus);
        kernel.feature_extractor_mut().sync_from_flat()?;
        let f_minus = kernel.compute(x, y)?;

        grad.push((f_plus - f_minus) / (2.0 * h));
    }
    // Restore the original parameter buffer so the kernel is unchanged
    // from the caller's point of view.
    kernel
        .feature_extractor_mut()
        .parameters_mut()
        .copy_from_slice(&baseline);
    kernel.feature_extractor_mut().sync_from_flat()?;
    Ok(grad)
}

/// Analytical gradient of `K_DKL(x, y)` w.r.t. the MLP parameters for
/// the RBF-base case. Returns a vector of length
/// `kernel.feature_extractor().parameter_count()` whose entries mirror
/// the flat parameter layout
/// `layer0.weights(row-major) ++ layer0.biases ++ layer1.weights ++ ...`.
///
/// The closed form is derived in the module doc-comment. The implementation
/// performs one forward pass with per-layer caches on each of `x` and
/// `y`, computes the output-space difference vector `Δ = g(x) - g(y)`,
/// and back-propagates it through both networks to accumulate the
/// per-parameter gradient.
pub fn rbf_dkl_gradient(
    kernel: &DeepKernel<MLPFeatureExtractor, RbfKernel>,
    x: &[f64],
    y: &[f64],
) -> Result<Vec<f64>> {
    let mlp = kernel.feature_extractor();
    let (fx, cache_x) = mlp.forward_with_cache(x)?;
    let (fy, cache_y) = mlp.forward_with_cache(y)?;
    if fx.len() != fy.len() {
        return Err(KernelError::DimensionMismatch {
            expected: vec![fx.len()],
            got: vec![fy.len()],
            context: "rbf_dkl_gradient: feature dims".to_string(),
        });
    }
    let diff: Vec<f64> = fx.iter().zip(fy.iter()).map(|(a, b)| a - b).collect();
    let sq_dist: f64 = diff.iter().map(|d| d * d).sum();
    let gamma = kernel.base_kernel().gamma();
    let k_val = (-gamma * sq_dist).exp();
    // Seed vectors for backprop: ∂K/∂g(x)_k = -2γ·Δ_k·K, ∂K/∂g(y)_k = +2γ·Δ_k·K.
    let seed_x: Vec<f64> = diff.iter().map(|d| -2.0 * gamma * d * k_val).collect();
    let seed_y: Vec<f64> = diff.iter().map(|d| 2.0 * gamma * d * k_val).collect();

    let mut grad = vec![0.0; mlp.parameter_count()];
    accumulate_backward(mlp, &cache_x, x, &seed_x, &mut grad)?;
    accumulate_backward(mlp, &cache_y, y, &seed_y, &mut grad)?;
    Ok(grad)
}

/// Backpropagate an output-space gradient through an MLP and accumulate
/// the per-parameter gradient into `out_grad`.
///
/// * `caches` holds `(pre_activation, post_activation)` for each layer
///   as produced by `MLPFeatureExtractor::forward_with_cache`.
/// * `input` is the original input to layer 0 (so we can form the
///   Jacobian of layer 0's weights w.r.t. the input).
/// * `seed` is `∂K/∂(output of MLP)`, length = output dimension.
///
/// Mutates `out_grad` in place. The layout matches
/// `flatten_layers` in `feature_extractor.rs`:
/// `(layer0.weights row-major, layer0.biases, layer1.weights, ...)`.
fn accumulate_backward(
    mlp: &MLPFeatureExtractor,
    caches: &[LayerCache],
    input: &[f64],
    seed: &[f64],
    out_grad: &mut [f64],
) -> Result<()> {
    let layers = mlp.layers();
    if caches.len() != layers.len() {
        return Err(KernelError::DimensionMismatch {
            expected: vec![layers.len()],
            got: vec![caches.len()],
            context: "accumulate_backward: cache length".to_string(),
        });
    }
    // Build a reverse offset table — offsets[i] points at the first
    // parameter slot for layer `i` inside the flat buffer.
    let mut offsets = Vec::with_capacity(layers.len());
    let mut running = 0usize;
    for layer in layers {
        offsets.push(running);
        running += layer.parameter_count();
    }

    // `delta` holds ∂K / ∂(post-activation of the current layer).
    let mut delta = seed.to_vec();
    if delta.len() != layers[layers.len() - 1].output_dim() {
        return Err(KernelError::DimensionMismatch {
            expected: vec![layers[layers.len() - 1].output_dim()],
            got: vec![delta.len()],
            context: "accumulate_backward: seed length".to_string(),
        });
    }

    for layer_idx in (0..layers.len()).rev() {
        let layer = &layers[layer_idx];
        let (pre, _post) = &caches[layer_idx];
        // ∂K / ∂(pre-activation of this layer) = delta * f'(pre).
        let activation = layer.activation();
        let mut delta_pre = Vec::with_capacity(pre.len());
        for (d, &p) in delta.iter().zip(pre.iter()) {
            delta_pre.push(d * derivative(activation, p));
        }
        // Previous-layer activations (input to this layer).
        let prev_activation: &[f64] = if layer_idx == 0 {
            input
        } else {
            &caches[layer_idx - 1].1
        };
        // Gradients on this layer's parameters.
        // weight grad: (∂K/∂pre[i]) * prev_activation[j] for the
        // entry weights[i][j].
        let w_base = offsets[layer_idx];
        let in_dim = layer.input_dim();
        let out_dim = layer.output_dim();
        for (i, &dpre_i) in delta_pre.iter().enumerate() {
            let row_offset = w_base + i * in_dim;
            for (j, &prev_j) in prev_activation.iter().enumerate() {
                out_grad[row_offset + j] += dpre_i * prev_j;
            }
        }
        let b_base = w_base + out_dim * in_dim;
        for (i, &dpre_i) in delta_pre.iter().enumerate() {
            out_grad[b_base + i] += dpre_i;
        }
        // Propagate delta backward through the affine layer.
        if layer_idx > 0 {
            let mut new_delta = vec![0.0; in_dim];
            for (i, &dpre_i) in delta_pre.iter().enumerate() {
                let row = &layer.weights[i];
                for (j, &w_ij) in row.iter().enumerate() {
                    new_delta[j] += dpre_i * w_ij;
                }
            }
            delta = new_delta;
        }
    }
    Ok(())
}

/// Local wrapper around [`Activation::derivative`] so `accumulate_backward`
/// does not need to import the enum directly (keeps imports tight).
fn derivative(activation: Activation, z: f64) -> f64 {
    activation.derivative(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deep_kernel::feature_extractor::MLPFeatureExtractor;
    use crate::deep_kernel::kernel::DeepKernel;
    use crate::deep_kernel::layer::Activation;
    use crate::types::RbfKernelConfig;

    fn mini_mlp(seed: u64) -> MLPFeatureExtractor {
        MLPFeatureExtractor::xavier_init(
            &[2, 3, 2],
            &[Activation::Tanh, Activation::Identity],
            seed,
        )
        .expect("xavier init")
    }

    #[test]
    fn analytical_matches_finite_difference_for_rbf_mlp() {
        let mlp = mini_mlp(17);
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.8)).expect("valid");
        let mut dkl = DeepKernel::new(mlp, rbf);

        let x = vec![0.3, -0.5];
        let y = vec![-0.2, 0.4];
        let analytical = rbf_dkl_gradient(&dkl, &x, &y).expect("analytical");
        let numerical = finite_difference_gradient(&mut dkl, &x, &y, 1e-5).expect("finite diff");
        assert_eq!(analytical.len(), numerical.len());
        for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            assert!(
                (a - n).abs() < 1e-3,
                "param {} mismatch: analytical={}, numerical={}",
                i,
                a,
                n
            );
        }
    }

    #[test]
    fn finite_difference_restores_parameters() {
        let mlp = mini_mlp(11);
        let before = mlp.parameters().to_vec();
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let mut dkl = DeepKernel::new(mlp, rbf);
        let _ = finite_difference_gradient(&mut dkl, &[0.2, 0.1], &[-0.1, 0.3], 1e-5)
            .expect("finite diff");
        let after = dkl.feature_extractor().parameters().to_vec();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn finite_difference_rejects_zero_step() {
        let mlp = mini_mlp(0);
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let mut dkl = DeepKernel::new(mlp, rbf);
        let err = finite_difference_gradient(&mut dkl, &[0.0, 0.0], &[0.0, 0.0], 0.0)
            .expect_err("zero step must fail");
        assert!(matches!(err, KernelError::InvalidParameter { .. }));
    }
}
