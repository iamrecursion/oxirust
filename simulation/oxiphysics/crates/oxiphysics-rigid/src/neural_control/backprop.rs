// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Back-propagation through the [`NeuralController`] network.
//!
//! The learners of this module (policy gradient, actor-critic, behavioural
//! cloning, the neural dynamics model) all minimise a scalar loss over the
//! parameters of a feed-forward network.  Estimating that gradient by central
//! finite differences costs two forward passes plus several `O(P)` copies of
//! the parameter vector per parameter, i.e. `O(P²)` per training sample for a
//! network with `P` parameters.  Back-propagation gives the exact gradient for
//! the price of one forward and one backward pass, `O(P)`.
//!
//! For the network
//!
//! ```text
//! a₀ = input,   zₗ = Wₗ·aₗ + bₗ,   aₗ₊₁ = f(zₗ)  (hidden layers),
//! out = action_scale · tanh(z_out)
//! ```
//!
//! and the vector `d_out = ∂L/∂out` of a loss `L`, [`NeuralController::backward`]
//! returns `∂L/∂θ` for the flat parameter vector `θ` of
//! [`NeuralController::get_parameters`]: with `δ_out = d_out · action_scale ·
//! (1 − tanh²(z_out))` and `δₗ = (Wₗ₊₁ᵀ δₗ₊₁) ⊙ f′(zₗ)`, the gradients are
//! `∂L/∂Wₗ = δₗ aₗᵀ` and `∂L/∂bₗ = δₗ`.

use super::functions::{relu, sigmoid, softmax, tanh_act};
use super::types::{Activation, DenseLayer, NeuralController};

impl Activation {
    /// The activation function evaluated at the pre-activation `z`.
    pub fn apply(self, z: f64) -> f64 {
        match self {
            Activation::Relu => relu(z),
            Activation::Tanh => tanh_act(z),
            Activation::Sigmoid => sigmoid(z),
            Activation::Linear => z,
        }
    }

    /// The derivative of the activation function at the pre-activation `z`,
    /// given its value `activation = self.apply(z)`.
    ///
    /// `tanh′ = 1 − tanh²`, `σ′ = σ(1 − σ)`, `relu′ = [z > 0]` (the
    /// sub-gradient 0 is used at the kink) and `linear′ = 1`.
    pub fn derivative(self, z: f64, activation: f64) -> f64 {
        match self {
            Activation::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - activation * activation,
            Activation::Sigmoid => activation * (1.0 - activation),
            Activation::Linear => 1.0,
        }
    }
}

/// What the forward pass of one [`DenseLayer`] keeps for the backward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerCache {
    /// The input the layer was evaluated at.
    pub input: Vec<f64>,
    /// The pre-activation `W·input + b`.
    pub pre_activation: Vec<f64>,
}

/// What a forward pass through a [`NeuralController`] keeps for the backward
/// pass ([`NeuralController::backward_cached`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardCache {
    /// One cache per hidden layer, followed by the cache of the output layer.
    ///
    /// The input of layer `l + 1` is the activation of layer `l`.
    pub layers: Vec<LayerCache>,
    /// The network output, identical to [`NeuralController::forward`].
    pub output: Vec<f64>,
}

impl DenseLayer {
    /// Linear forward pass `W·input + b` that keeps its input and its
    /// pre-activation for back-propagation.
    pub fn forward_cached(&self, input: &[f64]) -> LayerCache {
        LayerCache {
            input: input.to_vec(),
            pre_activation: self.forward_linear(input),
        }
    }

    /// Backward pass of the linear map `z = W·x + b`.
    ///
    /// `delta` is `∂L/∂z`.  Appends `∂L/∂W` (row by row, as in
    /// [`NeuralController::get_parameters`]) followed by `∂L/∂b` to `grad` and
    /// returns `∂L/∂x = Wᵀ·delta`.
    ///
    /// The forward pass pairs entries up like `zip`: inputs beyond the length of
    /// a weight row and rows beyond the length of `delta` do not take part.
    /// Their parameters receive a zero gradient, so the layout of `grad` is
    /// that of the parameter vector for every layer shape.
    fn backward_linear(&self, cache: &LayerCache, delta: &[f64], grad: &mut Vec<f64>) -> Vec<f64> {
        let mut input_grad = vec![0.0_f64; cache.input.len()];
        for (j, row) in self.weights.iter().enumerate() {
            let delta_j = delta.get(j).copied().unwrap_or(0.0);
            grad.extend(
                (0..row.len()).map(|i| delta_j * cache.input.get(i).copied().unwrap_or(0.0)),
            );
            for (acc, weight) in input_grad.iter_mut().zip(row.iter()) {
                *acc += delta_j * weight;
            }
        }
        grad.extend((0..self.biases.len()).map(|j| delta.get(j).copied().unwrap_or(0.0)));
        input_grad
    }
}

impl NeuralController {
    /// Forward pass that keeps the intermediate values needed by
    /// [`NeuralController::backward_cached`].
    ///
    /// `cache.output` equals `self.forward(input)`.
    pub fn forward_cached(&self, input: &[f64]) -> ForwardCache {
        let mut layers = Vec::with_capacity(self.layers.len() + 1);
        let mut activation = input.to_vec();
        for layer in &self.layers {
            let cache = layer.forward_cached(&activation);
            activation = cache
                .pre_activation
                .iter()
                .map(|&z| self.activation.apply(z))
                .collect();
            layers.push(cache);
        }
        let output_cache = self.output_layer.forward_cached(&activation);
        let output = output_cache
            .pre_activation
            .iter()
            .map(|&raw| tanh_act(raw) * self.action_scale)
            .collect();
        layers.push(output_cache);
        ForwardCache { layers, output }
    }

    /// Back-propagate `d_out = ∂L/∂output` through the pass recorded in `cache`.
    ///
    /// Returns `∂L/∂θ` in the layout of [`NeuralController::get_parameters`]:
    /// for every hidden layer the weight matrix row by row and then the
    /// biases, followed by the output layer.  Like [`NeuralController::forward`]
    /// the routine pairs entries up like `zip`: missing entries of `d_out` count
    /// as zero and surplus entries are ignored.
    ///
    /// `cache` must come from [`NeuralController::forward_cached`] of this
    /// network; a cache of a network with a different number of layers yields
    /// an all-zero gradient.
    pub fn backward_cached(&self, cache: &ForwardCache, d_out: &[f64]) -> Vec<f64> {
        if cache.layers.len() != self.layers.len() + 1 {
            return vec![0.0; self.num_parameters()];
        }
        let n_hidden = self.layers.len();

        // δ at the pre-activations of the output layer: out = scale·tanh(z).
        let output_cache = &cache.layers[n_hidden];
        let mut delta: Vec<f64> = output_cache
            .pre_activation
            .iter()
            .zip(d_out.iter())
            .map(|(&raw, &d)| {
                let squashed = tanh_act(raw);
                d * self.action_scale * (1.0 - squashed * squashed)
            })
            .collect();

        // Gradient blocks from the output layer towards the input.
        let mut blocks: Vec<Vec<f64>> = Vec::with_capacity(n_hidden + 1);
        let mut block = Vec::new();
        let mut upstream = self
            .output_layer
            .backward_linear(output_cache, &delta, &mut block);
        blocks.push(block);

        for (l, layer) in self.layers.iter().enumerate().rev() {
            let layer_cache = &cache.layers[l];
            // The activation of layer `l` is the input of layer `l + 1`.
            let activations = &cache.layers[l + 1].input;
            delta = layer_cache
                .pre_activation
                .iter()
                .zip(activations.iter())
                .zip(upstream.iter())
                .map(|((&z, &a), &g)| g * self.activation.derivative(z, a))
                .collect();
            let mut block = Vec::new();
            upstream = layer.backward_linear(layer_cache, &delta, &mut block);
            blocks.push(block);
        }
        blocks.into_iter().rev().flatten().collect()
    }

    /// Gradient `∂L/∂θ` of a loss `L` with `∂L/∂output = d_out`, in the layout
    /// of [`NeuralController::get_parameters`].
    ///
    /// For the scalar `L = d_out · self.forward(input)` this is the exact
    /// gradient of the network output along `d_out`; it costs one forward and
    /// one backward pass, `O(P)` for `P` parameters, where a finite-difference
    /// estimate needs `O(P²)`.  Use [`NeuralController::forward_cached`] and
    /// [`NeuralController::backward_cached`] to reuse the forward pass.
    pub fn backward(&self, input: &[f64], d_out: &[f64]) -> Vec<f64> {
        self.backward_cached(&self.forward_cached(input), d_out)
    }
}

/// Gradient of `log softmax(logits)[action]` with respect to the logits:
/// `onehot(action) − softmax(logits)`.
///
/// `None` when `action` is not an index into `logits`.
pub(super) fn log_softmax_gradient(logits: &[f64], action: usize) -> Option<Vec<f64>> {
    if action >= logits.len() {
        return None;
    }
    let mut gradient: Vec<f64> = softmax(logits).into_iter().map(|p| -p).collect();
    if let Some(entry) = gradient.get_mut(action) {
        *entry += 1.0;
    }
    Some(gradient)
}

#[cfg(test)]
mod tests;
