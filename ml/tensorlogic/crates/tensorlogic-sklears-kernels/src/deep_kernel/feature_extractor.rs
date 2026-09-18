//! Differentiable feature extractors for Deep Kernel Learning.
//!
//! [`NeuralFeatureMap`] is the trait an `F` must satisfy to be plugged
//! into [`crate::deep_kernel::DeepKernel`]. The v0.2.0 preview ships a
//! single implementation — [`MLPFeatureExtractor`] — a stack of
//! [`DenseLayer`]s with ReLU / Tanh / Identity activations. Future
//! releases may add CNN / Transformer extractors; the trait is designed
//! to keep those additions out-of-tree until they are ready.
//!
//! # Naming note
//!
//! The trait is named `NeuralFeatureMap` (not `FeatureExtractor`) to
//! avoid colliding with the pre-existing
//! [`crate::feature_extraction::FeatureExtractor`] struct, which is
//! specifically for turning [`tensorlogic_ir::TLExpr`] into numeric
//! features. The two types coexist in the same crate and serve
//! complementary purposes.

use scirs2_core::random::{Normal, SeedableRng, StdRng};

use crate::deep_kernel::layer::{Activation, DenseLayer};
use crate::error::{KernelError, Result};

/// Per-layer cache `(pre_activation, post_activation)` used by the
/// analytical backprop path in [`crate::deep_kernel::gradient`].
pub type LayerCache = (Vec<f64>, Vec<f64>);

/// Bundle returned by [`MLPFeatureExtractor::forward_with_cache`] —
/// `(final_output, per_layer_caches)`.
pub type ForwardCache = (Vec<f64>, Vec<LayerCache>);

/// A differentiable map `ℝ^{d_in} → ℝ^{d_out}` used as the feature
/// extractor inside a Deep Kernel.
///
/// Implementations must be deterministic given their parameters
/// (so `forward(x)` produces the same output for the same input at any
/// point in time), and must be `Send + Sync` so they can be shared
/// across threads like every other crate-level kernel.
pub trait NeuralFeatureMap: Send + Sync {
    /// Map an input vector to feature space.
    fn forward(&self, input: &[f64]) -> Result<Vec<f64>>;

    /// Mutable view of the flat parameter vector (weights + biases, in
    /// layer order). Exposed as `&mut [f64]` so that optimisers can
    /// apply updates in place without owning the extractor.
    fn parameters_mut(&mut self) -> &mut [f64];

    /// Immutable view of the flat parameter vector.
    fn parameters(&self) -> &[f64];

    /// Number of trainable scalar parameters.
    fn parameter_count(&self) -> usize;

    /// Input dimension expected by `forward`.
    fn input_dim(&self) -> usize;

    /// Output dimension produced by `forward`.
    fn output_dim(&self) -> usize;
}

/// Multi-layer perceptron feature extractor.
///
/// The network is a sequence of [`DenseLayer`]s applied in order; the
/// output of layer `i` is the input of layer `i + 1`. The layer stack
/// must be non-empty and layer shapes must match transitively.
///
/// Parameters are stored twice on purpose:
///
/// * as structured `layers: Vec<DenseLayer>` — used by `forward`.
/// * as flat `parameters: Vec<f64>` — exposed to optimisers.
///
/// The two views are kept in sync: [`MLPFeatureExtractor::parameters_mut`]
/// returns a borrow into the flat buffer and
/// [`MLPFeatureExtractor::sync_from_flat`] pushes the flat buffer back
/// into the layer weights. Mutating the flat buffer directly requires a
/// subsequent `sync_from_flat` call before the next `forward`.
#[derive(Clone, Debug)]
pub struct MLPFeatureExtractor {
    layers: Vec<DenseLayer>,
    parameters: Vec<f64>,
}

impl MLPFeatureExtractor {
    /// Wrap an existing `Vec<DenseLayer>` as an MLP feature extractor.
    ///
    /// Fails when the layer stack is empty or when consecutive layer
    /// shapes do not match.
    pub fn from_layers(layers: Vec<DenseLayer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "layers".to_string(),
                value: "[]".to_string(),
                reason: "MLPFeatureExtractor requires at least one layer".to_string(),
            });
        }
        for pair in layers.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if a.output_dim() != b.input_dim() {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![a.output_dim()],
                    got: vec![b.input_dim()],
                    context: "MLPFeatureExtractor: layer shape chain".to_string(),
                });
            }
        }
        let parameters = flatten_layers(&layers);
        Ok(Self { layers, parameters })
    }

    /// Build an MLP from a list of layer widths and a parallel list of
    /// activations (one per weight matrix — i.e. `widths.len() - 1`
    /// entries). Weights are Xavier/Glorot-normal initialised via
    /// SciRS2-Core's seeded RNG; biases are zero.
    pub fn xavier_init(widths: &[usize], activations: &[Activation], seed: u64) -> Result<Self> {
        if widths.len() < 2 {
            return Err(KernelError::InvalidParameter {
                parameter: "widths".to_string(),
                value: format!("{:?}", widths),
                reason: "xavier_init requires at least input and output widths".to_string(),
            });
        }
        if widths.contains(&0) {
            return Err(KernelError::InvalidParameter {
                parameter: "widths".to_string(),
                value: format!("{:?}", widths),
                reason: "widths must be strictly positive".to_string(),
            });
        }
        if activations.len() != widths.len() - 1 {
            return Err(KernelError::DimensionMismatch {
                expected: vec![widths.len() - 1],
                got: vec![activations.len()],
                context: "xavier_init: activations length".to_string(),
            });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers = Vec::with_capacity(widths.len() - 1);
        for (pair, &activation) in widths.windows(2).zip(activations.iter()) {
            let fan_in = pair[0];
            let fan_out = pair[1];
            let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
            let dist = Normal::new(0.0, std).map_err(|e| KernelError::InvalidParameter {
                parameter: "xavier stddev".to_string(),
                value: std.to_string(),
                reason: format!("Normal::new failed: {}", e),
            })?;
            let mut weights = Vec::with_capacity(fan_out);
            for _ in 0..fan_out {
                let mut row = Vec::with_capacity(fan_in);
                for _ in 0..fan_in {
                    row.push(rng.sample(dist));
                }
                weights.push(row);
            }
            let biases = vec![0.0; fan_out];
            layers.push(DenseLayer::new(weights, biases, activation)?);
        }
        Self::from_layers(layers)
    }

    /// Immutable view of the layer stack.
    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass with per-layer caches of `(pre_activation,
    /// post_activation)` tensors. Used by the analytical gradient path
    /// in [`crate::deep_kernel::gradient`].
    pub fn forward_with_cache(&self, input: &[f64]) -> Result<ForwardCache> {
        if input.len() != self.input_dim() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.input_dim()],
                got: vec![input.len()],
                context: "MLPFeatureExtractor::forward_with_cache input".to_string(),
            });
        }
        let mut cache = Vec::with_capacity(self.layers.len());
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (pre, post) = layer.forward_with_preactivation(&current)?;
            cache.push((pre, post.clone()));
            current = post;
        }
        Ok((current, cache))
    }

    /// Push the flat parameter buffer back into the per-layer
    /// `weights` / `biases`. Call this after mutating the flat buffer
    /// returned from [`Self::parameters_mut`] but before the next
    /// forward pass.
    pub fn sync_from_flat(&mut self) -> Result<()> {
        let mut idx = 0;
        for layer in self.layers.iter_mut() {
            for row in layer.weights.iter_mut() {
                for w in row.iter_mut() {
                    let v = *self.parameters.get(idx).ok_or_else(|| {
                        KernelError::ComputationError(
                            "parameter buffer too short during sync_from_flat".to_string(),
                        )
                    })?;
                    if !v.is_finite() {
                        return Err(KernelError::InvalidParameter {
                            parameter: format!("parameters[{}]", idx),
                            value: v.to_string(),
                            reason: "parameters must remain finite".to_string(),
                        });
                    }
                    *w = v;
                    idx += 1;
                }
            }
            for b in layer.biases.iter_mut() {
                let v = *self.parameters.get(idx).ok_or_else(|| {
                    KernelError::ComputationError(
                        "parameter buffer too short during sync_from_flat".to_string(),
                    )
                })?;
                if !v.is_finite() {
                    return Err(KernelError::InvalidParameter {
                        parameter: format!("parameters[{}]", idx),
                        value: v.to_string(),
                        reason: "parameters must remain finite".to_string(),
                    });
                }
                *b = v;
                idx += 1;
            }
        }
        Ok(())
    }
}

impl NeuralFeatureMap for MLPFeatureExtractor {
    fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.input_dim() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.input_dim()],
                got: vec![input.len()],
                context: "MLPFeatureExtractor::forward input".to_string(),
            });
        }
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        &mut self.parameters
    }

    fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    fn input_dim(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.input_dim())
            .unwrap_or_default()
    }

    fn output_dim(&self) -> usize {
        self.layers
            .last()
            .map(|l| l.output_dim())
            .unwrap_or_default()
    }
}

/// Serialise the weights / biases of a layer stack into a flat vector,
/// in the canonical `layer0.weights ++ layer0.biases ++ layer1.weights
/// ++ ...` order.
fn flatten_layers(layers: &[DenseLayer]) -> Vec<f64> {
    let mut out = Vec::with_capacity(layers.iter().map(DenseLayer::parameter_count).sum());
    for layer in layers {
        for row in &layer.weights {
            out.extend_from_slice(row);
        }
        out.extend_from_slice(&layer.biases);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_forward_identity_of_1x1() {
        // A 1→1 identity-MLP should reproduce its input exactly.
        let layer =
            DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Identity).expect("valid");
        let mlp = MLPFeatureExtractor::from_layers(vec![layer]).expect("valid mlp");
        let out = mlp.forward(&[2.5]).expect("forward");
        assert_eq!(out, vec![2.5]);
    }

    #[test]
    fn mlp_rejects_shape_chain_mismatch() {
        let a =
            DenseLayer::new(vec![vec![1.0, 0.0]], vec![0.0], Activation::Identity).expect("valid");
        // Output of `a` is 1, but `b` expects 3.
        let b = DenseLayer::new(vec![vec![1.0, 1.0, 1.0]], vec![0.0], Activation::Identity)
            .expect("valid");
        let err = MLPFeatureExtractor::from_layers(vec![a, b]).expect_err("must fail");
        assert!(matches!(err, KernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn mlp_parameter_roundtrip() {
        let layer = DenseLayer::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0.5, -0.5],
            Activation::ReLU,
        )
        .expect("valid");
        let mlp = MLPFeatureExtractor::from_layers(vec![layer]).expect("valid");
        // 2x2 weights + 2 biases = 6 params.
        assert_eq!(mlp.parameter_count(), 6);
        assert_eq!(mlp.parameters(), &[1.0, 2.0, 3.0, 4.0, 0.5, -0.5]);
    }
}
