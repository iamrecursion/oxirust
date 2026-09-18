// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Neural network-accelerated Lattice Boltzmann methods.
//!
//! This module provides deep-learning augmented LBM components:
//! - Fully-connected neural networks for collision closures, trainable by
//!   backpropagation ([`NeuralLbm::train`])
//! - Physics-informed LBM: a D2Q9 BGK solver whose collision carries a
//!   mass- and momentum-conserving neural correction ([`PhysicsInformedLbm`])
//! - DeepONet-based solution operators
//! - Data-driven reduced-order models by exact dynamic mode decomposition
//!   ([`DataDrivenLbm`])

use crate::error::{Error, Result};
use crate::lattice::{CS2, D2Q9_VELOCITIES, D2Q9_WEIGHTS, equilibrium_d2q9, macros_from_d2q9};

mod complex_eigen;
mod dmd;

pub use dmd::DataDrivenLbm;

// ─────────────────────────────────────────────────────────────────────────────
// Activation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Activation function used in neural network layers.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationFn {
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid: 1 / (1 + e^{-x}).
    Sigmoid,
    /// Identity (no nonlinearity).
    Linear,
}

impl ActivationFn {
    /// Apply this activation function to a single value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::Relu => x.max(0.0),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Linear => x,
        }
    }

    /// Apply the element-wise derivative of this activation function.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFn::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFn::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFn::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFn::Linear => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single dense layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single fully-connected (dense) layer in a neural network.
///
/// `weights[i][j]` is the weight from input neuron `j` to output neuron `i`.
#[derive(Debug, Clone)]
pub struct NeuralLbmLayer {
    /// Weight matrix stored as `[output_neurons][input_neurons]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector with one entry per output neuron.
    pub biases: Vec<f64>,
    /// Activation function applied after the linear transform.
    pub activation: ActivationFn,
}

impl NeuralLbmLayer {
    /// Create a new layer with given weight matrix, biases, and activation.
    ///
    /// # Panics
    /// Panics if `weights` and `biases` have different lengths.
    pub fn new(weights: Vec<Vec<f64>>, biases: Vec<f64>, activation: ActivationFn) -> Self {
        assert_eq!(
            weights.len(),
            biases.len(),
            "weights rows must match biases length"
        );
        Self {
            weights,
            biases,
            activation,
        }
    }

    /// Number of input features expected by this layer.
    pub fn input_size(&self) -> usize {
        self.weights.first().map_or(0, |row| row.len())
    }

    /// Number of output neurons in this layer.
    pub fn output_size(&self) -> usize {
        self.weights.len()
    }

    /// Pre-activation `W * x + b` of the layer (the argument of the activation
    /// function, needed by backpropagation).
    pub fn pre_activation(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + b
            })
            .collect()
    }

    /// Forward pass: compute `activation(W * x + b)`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.pre_activation(input)
            .into_iter()
            .map(|z| self.activation.apply(z))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-layer perceptron
// ─────────────────────────────────────────────────────────────────────────────

/// First-order optimiser used by [`NeuralLbm::train_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// Plain gradient descent, `θ ← θ − η ∇L`.
    GradientDescent,
    /// Adam (Kingma & Ba, 2015) with the standard `β1 = 0.9`, `β2 = 0.999`,
    /// `ε = 1e-8` and bias-corrected moments: every parameter moves by roughly `η`
    /// per step whatever the scale of its gradient, so it trains reliably on the
    /// small residuals of lattice-Boltzmann collisions, where the gradients are
    /// far too small for a sensible fixed gradient-descent step.
    Adam,
}

/// Exponential decay rate of Adam's first moment estimate.
const ADAM_BETA1: f64 = 0.9;
/// Exponential decay rate of Adam's second moment estimate.
const ADAM_BETA2: f64 = 0.999;
/// Adam's denominator offset.
const ADAM_EPSILON: f64 = 1.0e-8;

/// A multi-layer perceptron (MLP) used to accelerate LBM computations.
///
/// The network is a stack of [`NeuralLbmLayer`] instances evaluated
/// sequentially during the forward pass.
#[derive(Debug, Clone)]
pub struct NeuralLbm {
    /// Ordered list of dense layers.
    pub layers: Vec<NeuralLbmLayer>,
    /// Number of inputs to the first layer.
    pub input_size: usize,
    /// Number of outputs from the final layer.
    pub output_size: usize,
}

impl NeuralLbm {
    /// Build a `NeuralLbm` from pre-constructed layers.
    ///
    /// `input_size` and `output_size` are inferred from the first and last
    /// layer if `layers` is non-empty; otherwise the supplied values are used.
    pub fn new(layers: Vec<NeuralLbmLayer>, input_size: usize, output_size: usize) -> Self {
        Self {
            layers,
            input_size,
            output_size,
        }
    }

    /// Create a small default network with random-like (deterministic) weights.
    ///
    /// Architecture: `input_size` → 16 (Tanh) → 16 (Tanh) → `output_size` (Linear).
    pub fn default_network(input_size: usize, output_size: usize) -> Self {
        let hidden = 16usize;
        // Use a simple deterministic pseudo-random initialisation so we stay
        // dependency-free (no external rand crate needed here).
        let layer1_w: Vec<Vec<f64>> = (0..hidden)
            .map(|i| {
                (0..input_size)
                    .map(|j| simple_weight(i, j, input_size) * 0.5)
                    .collect()
            })
            .collect();
        let layer1_b = vec![0.0f64; hidden];
        let layer2_w: Vec<Vec<f64>> = (0..hidden)
            .map(|i| {
                (0..hidden)
                    .map(|j| simple_weight(i, j, hidden) * 0.3)
                    .collect()
            })
            .collect();
        let layer2_b = vec![0.0f64; hidden];
        let layer3_w: Vec<Vec<f64>> = (0..output_size)
            .map(|i| {
                (0..hidden)
                    .map(|j| simple_weight(i, j, hidden) * 0.2)
                    .collect()
            })
            .collect();
        let layer3_b = vec![0.0f64; output_size];

        let layers = vec![
            NeuralLbmLayer::new(layer1_w, layer1_b, ActivationFn::Tanh),
            NeuralLbmLayer::new(layer2_w, layer2_b, ActivationFn::Tanh),
            NeuralLbmLayer::new(layer3_w, layer3_b, ActivationFn::Linear),
        ];
        Self::new(layers, input_size, output_size)
    }

    /// Run the forward pass through all layers.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Predict the equilibrium distribution `f_eq` from density `rho` and
    /// velocity components `u` using the neural network.
    ///
    /// The feature vector is `[rho, u[0\], u[1], ...]`.
    pub fn predict_equilibrium(&self, rho: f64, u: &[f64]) -> Vec<f64> {
        let mut features = vec![rho];
        features.extend_from_slice(u);
        // Pad or truncate to `input_size`.
        features.resize(self.input_size, 0.0);
        self.forward(&features)
    }

    /// Evaluate a turbulence closure model returning the eddy viscosity
    /// correction for a given `strain_rate` magnitude.
    pub fn closure_model(&self, strain_rate: f64) -> f64 {
        let features: Vec<f64> = std::iter::once(strain_rate)
            .chain(std::iter::repeat_n(0.0, self.input_size.saturating_sub(1)))
            .take(self.input_size)
            .collect();
        let out = self.forward(&features);
        *out.first().unwrap_or(&0.0)
    }

    /// Predict the turbulent (sub-grid) viscosity from a feature vector.
    ///
    /// `features` should contain flow-field descriptors such as local velocity
    /// gradients, vorticity, etc.
    pub fn turbulent_viscosity(&self, features: &[f64]) -> f64 {
        let mut padded = features.to_vec();
        padded.resize(self.input_size, 0.0);
        let out = self.forward(&padded);
        out.first().copied().unwrap_or(0.0).abs()
    }

    /// Set every weight and bias of the output layer to zero, so that the network
    /// outputs exactly zero until it is trained.  Does nothing for a network
    /// without layers.
    pub fn zero_output_layer(&mut self) {
        if let Some(last) = self.layers.last_mut() {
            for row in last.weights.iter_mut() {
                row.iter_mut().for_each(|w| *w = 0.0);
            }
            last.biases.iter_mut().for_each(|b| *b = 0.0);
        }
    }

    /// Widths `(inputs, outputs)` of the layer chain, after verifying that every
    /// weight matrix is rectangular, matches its bias vector and consumes exactly
    /// what the preceding layer produces.
    fn chain_widths(&self) -> Result<(usize, usize)> {
        let Some(first) = self.layers.first() else {
            return Err(Error::IncompatibleParameters {
                message: "the network has no layers".to_string(),
            });
        };
        let inputs = first.input_size();
        let mut width = inputs;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.biases.len() != layer.weights.len() {
                return Err(Error::ComponentCountMismatch {
                    expected: layer.weights.len(),
                    found: layer.biases.len(),
                    field: format!("layer {index} biases"),
                });
            }
            if let Some(row) = layer.weights.iter().find(|row| row.len() != width) {
                return Err(Error::ComponentCountMismatch {
                    expected: width,
                    found: row.len(),
                    field: format!("layer {index} input size"),
                });
            }
            width = layer.output_size();
        }
        Ok((inputs, width))
    }

    /// Verify that the network maps `inputs` features to `outputs` values, both as
    /// declared by `input_size` / `output_size` and by the layer weights.
    fn check_shape(&self, inputs: usize, outputs: usize) -> Result<()> {
        if self.input_size != inputs {
            return Err(Error::ComponentCountMismatch {
                expected: inputs,
                found: self.input_size,
                field: "network input size".to_string(),
            });
        }
        if self.output_size != outputs {
            return Err(Error::ComponentCountMismatch {
                expected: outputs,
                found: self.output_size,
                field: "network output size".to_string(),
            });
        }
        let (layer_inputs, layer_outputs) = self.chain_widths()?;
        if layer_inputs != inputs {
            return Err(Error::ComponentCountMismatch {
                expected: inputs,
                found: layer_inputs,
                field: "first layer input size".to_string(),
            });
        }
        if layer_outputs != outputs {
            return Err(Error::ComponentCountMismatch {
                expected: outputs,
                found: layer_outputs,
                field: "last layer output size".to_string(),
            });
        }
        Ok(())
    }

    /// Check that `inputs` / `targets` form a usable training set for the layers.
    fn validate_training_set(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<()> {
        let (input_width, output_width) = self.chain_widths()?;
        if input_width == 0 || output_width == 0 {
            return Err(Error::IncompatibleParameters {
                message: "the network has an empty input or output layer".to_string(),
            });
        }
        if inputs.is_empty() {
            return Err(Error::IncompatibleParameters {
                message: "the training set is empty".to_string(),
            });
        }
        if inputs.len() != targets.len() {
            return Err(Error::ComponentCountMismatch {
                expected: inputs.len(),
                found: targets.len(),
                field: "training targets".to_string(),
            });
        }
        for (n, (x, t)) in inputs.iter().zip(targets.iter()).enumerate() {
            if x.len() != input_width {
                return Err(Error::ComponentCountMismatch {
                    expected: input_width,
                    found: x.len(),
                    field: format!("training input {n}"),
                });
            }
            if t.len() != output_width {
                return Err(Error::ComponentCountMismatch {
                    expected: output_width,
                    found: t.len(),
                    field: format!("training target {n}"),
                });
            }
            for (name, values) in [("training input", x), ("training target", t)] {
                if let Some((index, &value)) =
                    values.iter().enumerate().find(|(_, v)| !v.is_finite())
                {
                    return Err(Error::NonFiniteField {
                        field: format!("{name} {n}"),
                        index,
                        value,
                    });
                }
            }
        }
        Ok(())
    }

    /// Mean squared error of an already validated training set.
    fn mean_squared_error_unchecked(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        let outputs = targets.first().map_or(1, Vec::len);
        let mut sum = 0.0_f64;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            let y = self.forward(x);
            sum += y
                .iter()
                .zip(t.iter())
                .map(|(&yi, &ti)| (yi - ti) * (yi - ti))
                .sum::<f64>();
        }
        sum / (inputs.len() * outputs) as f64
    }

    /// Mean squared error `1/(N K) Σ_n Σ_k (y_nk − t_nk)²` of the network over a data
    /// set of `N` samples with `K` outputs each.
    ///
    /// # Errors
    /// [`Error::ComponentCountMismatch`] / [`Error::NonFiniteField`] /
    /// [`Error::IncompatibleParameters`] if the data does not fit the network.
    pub fn mean_squared_error(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<f64> {
        self.validate_training_set(inputs, targets)?;
        Ok(self.mean_squared_error_unchecked(inputs, targets))
    }

    /// Number of trainable parameters: all weights and biases.
    fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.weights.iter().map(Vec::len).sum::<usize>() + layer.biases.len())
            .sum()
    }

    /// Gradient of the mean squared error with respect to every weight and bias by
    /// backpropagation: the pre-activations `z_l = W_l a_{l-1} + b_l` of every
    /// layer are cached by the forward pass and the error is propagated back with
    /// `δ_l = (W_{l+1}^T δ_{l+1}) ⊙ φ_l'(z_l)`, where `φ'` is
    /// [`ActivationFn::derivative`] evaluated at the pre-activation.
    ///
    /// The gradient is flat, in the order of [`add_to_parameters`](Self::add_to_parameters):
    /// for every layer the weights row by row, then the biases.  The data must have
    /// been validated.
    fn gradients(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> Vec<f64> {
        let mut offsets = Vec::with_capacity(self.layers.len());
        let mut total = 0_usize;
        for layer in &self.layers {
            offsets.push(total);
            total += layer.weights.iter().map(Vec::len).sum::<usize>() + layer.biases.len();
        }
        let mut gradient = vec![0.0_f64; total];
        let outputs = targets.first().map_or(1, Vec::len);
        let scale = 2.0 / (inputs.len() * outputs) as f64;
        let last = self.layers.len() - 1;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            // Forward pass keeping pre-activations z_l and activations a_l (a_0 = x).
            let mut activations: Vec<Vec<f64>> = vec![x.clone()];
            let mut pre_activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
            for layer in &self.layers {
                let z = layer.pre_activation(&activations[activations.len() - 1]);
                let a = z.iter().map(|&zi| layer.activation.apply(zi)).collect();
                pre_activations.push(z);
                activations.push(a);
            }
            // Output error δ_L = dL/dy ⊙ φ'(z_L).
            let mut delta: Vec<f64> = activations[last + 1]
                .iter()
                .zip(t.iter())
                .zip(pre_activations[last].iter())
                .map(|((&y, &target), &z)| {
                    scale * (y - target) * self.layers[last].activation.derivative(z)
                })
                .collect();
            for l in (0..=last).rev() {
                let layer_input = &activations[l];
                let columns = layer_input.len();
                let bias_base = offsets[l] + delta.len() * columns;
                for (i, &d) in delta.iter().enumerate() {
                    gradient[bias_base + i] += d;
                    let row_base = offsets[l] + i * columns;
                    for (g, &a) in gradient[row_base..row_base + columns]
                        .iter_mut()
                        .zip(layer_input.iter())
                    {
                        *g += d * a;
                    }
                }
                if l > 0 {
                    let mut previous = vec![0.0_f64; columns];
                    for (row, &d) in self.layers[l].weights.iter().zip(delta.iter()) {
                        for (p, &w) in previous.iter_mut().zip(row.iter()) {
                            *p += w * d;
                        }
                    }
                    let activation = &self.layers[l - 1].activation;
                    for (p, &z) in previous.iter_mut().zip(pre_activations[l - 1].iter()) {
                        *p *= activation.derivative(z);
                    }
                    delta = previous;
                }
            }
        }
        gradient
    }

    /// Add `step` (one entry per parameter, in the order of the flat gradient) to
    /// the weights and biases.
    fn add_to_parameters(&mut self, step: &[f64]) {
        let mut values = step.iter();
        for layer in self.layers.iter_mut() {
            for w in layer.weights.iter_mut().flatten() {
                if let Some(s) = values.next() {
                    *w += s;
                }
            }
            for b in layer.biases.iter_mut() {
                if let Some(s) = values.next() {
                    *b += s;
                }
            }
        }
    }

    /// Train the network on `inputs` / `targets` by full-batch gradient descent on
    /// the mean squared error ([`mean_squared_error`](Self::mean_squared_error)),
    /// with the gradient obtained by backpropagation through
    /// [`ActivationFn::derivative`].
    ///
    /// Runs `epochs` updates `W ← W − learning_rate · ∂L/∂W` (and likewise for the
    /// biases) and returns the mean squared error of the trained network.  With
    /// `epochs == 0` nothing is changed and the current error is returned.  See
    /// [`train_with`](Self::train_with) for the Adam optimiser, which copes much
    /// better with small or badly scaled data.
    ///
    /// # Errors
    /// As [`train_with`](Self::train_with).
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<f64> {
        self.train_with(
            Optimizer::GradientDescent,
            inputs,
            targets,
            learning_rate,
            epochs,
        )
    }

    /// Train the network with the chosen [`Optimizer`]: `epochs` full-batch updates
    /// on the gradient of the mean squared error, returning the mean squared error
    /// of the trained network.
    ///
    /// # Errors
    /// * [`Error::ComponentCountMismatch`], [`Error::NonFiniteField`],
    ///   [`Error::IncompatibleParameters`]: the data does not fit the network (wrong
    ///   sample widths, unequal counts, empty set, non-finite entries) or the
    ///   network is ill-formed.
    /// * [`Error::ParameterOutOfBounds`]: `learning_rate` is negative or not finite.
    /// * [`Error::NonFiniteField`]: the training diverged (non-finite loss); the
    ///   network is then restored to its state before the call.
    pub fn train_with(
        &mut self,
        optimizer: Optimizer,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<f64> {
        self.validate_training_set(inputs, targets)?;
        if !(learning_rate.is_finite() && learning_rate >= 0.0) {
            return Err(Error::ParameterOutOfBounds {
                name: "learning_rate".to_string(),
                value: learning_rate,
                min: 0.0,
                max: f64::MAX,
            });
        }
        let backup = self.layers.clone();
        let moment_size = match optimizer {
            Optimizer::GradientDescent => 0,
            Optimizer::Adam => self.parameter_count(),
        };
        let mut first_moment = vec![0.0_f64; moment_size];
        let mut second_moment = vec![0.0_f64; moment_size];
        for epoch in 1..=epochs {
            let gradient = self.gradients(inputs, targets);
            let step: Vec<f64> = match optimizer {
                Optimizer::GradientDescent => gradient.iter().map(|g| -learning_rate * g).collect(),
                Optimizer::Adam => {
                    let t = epoch as f64;
                    let first_correction = 1.0 - ADAM_BETA1.powf(t);
                    let second_correction = 1.0 - ADAM_BETA2.powf(t);
                    gradient
                        .iter()
                        .zip(first_moment.iter_mut())
                        .zip(second_moment.iter_mut())
                        .map(|((&g, m), v)| {
                            *m = ADAM_BETA1 * *m + (1.0 - ADAM_BETA1) * g;
                            *v = ADAM_BETA2 * *v + (1.0 - ADAM_BETA2) * g * g;
                            -learning_rate * (*m / first_correction)
                                / ((*v / second_correction).sqrt() + ADAM_EPSILON)
                        })
                        .collect()
                }
            };
            self.add_to_parameters(&step);
        }
        let loss = self.mean_squared_error_unchecked(inputs, targets);
        if !loss.is_finite() {
            self.layers = backup;
            return Err(Error::NonFiniteField {
                field: "training loss".to_string(),
                index: epochs,
                value: loss,
            });
        }
        Ok(loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics-informed LBM
// ─────────────────────────────────────────────────────────────────────────────

/// Number of lattice directions of D2Q9, the only lattice the physics-informed
/// solver runs on.
const Q_D2Q9: usize = 9;

/// Project a population correction onto the collision invariants.
///
/// With the moments `M0 = Σ c_i` and `M1 = Σ c_i e_i` the projected correction
/// `c_i − w_i (M0 + e_i · M1 / cs²)` has vanishing mass and momentum moments,
/// `Σ c_i = 0` and `Σ c_i e_i = 0` (because `Σ w_i = 1`, `Σ w_i e_i = 0` and
/// `Σ w_i e_i e_i = cs² I`), and leaves every correction that is already
/// conservative unchanged.  Added to a collision it therefore changes the
/// stress and the higher moments, but never the mass or the momentum of a node.
fn project_conservative(correction: &mut [f64; Q_D2Q9]) {
    let mut mass = 0.0_f64;
    let mut momentum = [0.0_f64; 2];
    for (c, velocity) in correction.iter().zip(D2Q9_VELOCITIES.iter()) {
        mass += c;
        momentum[0] += c * f64::from(velocity[0]);
        momentum[1] += c * f64::from(velocity[1]);
    }
    for ((c, velocity), weight) in correction
        .iter_mut()
        .zip(D2Q9_VELOCITIES.iter())
        .zip(D2Q9_WEIGHTS.iter())
    {
        let projected_momentum =
            f64::from(velocity[0]) * momentum[0] + f64::from(velocity[1]) * momentum[1];
        *c -= weight * (mass + projected_momentum / CS2);
    }
}

/// Source node of the pull-streaming step: the node `shift` cells upstream of
/// `coord` on a periodic line of `n` cells.
fn upstream(coord: usize, shift: i32, n: usize) -> usize {
    let wrapped = (coord as i64 - i64::from(shift)).rem_euclid(n as i64);
    wrapped as usize
}

/// Physics-informed LBM: a D2Q9 BGK lattice Boltzmann solver on a periodic
/// `nx × ny` grid whose collision is augmented by a learned correction.
///
/// One time step is
///
/// ```text
/// f*_i(x)       = f_i(x) − (f_i(x) − f_i^eq(ρ, u)) / τ + c_i(f(x))     (collision)
/// f_i(x, t + 1) = f*_i(x − e_i)                                       (streaming)
/// ```
///
/// with `ρ`, `u` the moments of the node and `f^eq` the second-order D2Q9
/// equilibrium.  The correction `c(f)` is the output of the network
/// [`neural_correction`](Self::neural_correction), evaluated on the features
/// `[ρ, f_0, …, f_8]` of the node and projected onto the collision invariants
/// (`Σ c_i = 0`, `Σ c_i e_i = 0`): the network can learn the residual between
/// the coarse-grained LBM and a reference (DNS / experimental) solution in the
/// stress and higher moments, but it can neither create nor destroy mass or
/// momentum.
///
/// A freshly constructed solver has a zero output layer, so its correction vanishes
/// and it is exactly the plain BGK-LBM until the network is trained
/// ([`train_collision_correction`](Self::train_collision_correction)).
#[derive(Debug, Clone)]
pub struct PhysicsInformedLbm {
    /// Neural network that provides the collision correction (`q + 1` features in,
    /// `q` corrections out).
    pub neural_correction: NeuralLbm,
    /// Distribution functions, node-major: `lbm_grid[(y * nx + x) * q + i]` is the
    /// population of direction `i` at node `(x, y)` (`Q * Nx * Ny` values).
    pub lbm_grid: Vec<f64>,
    /// Relaxation time τ (BGK stability requires `τ > 1/2`); the lattice kinematic
    /// viscosity is `cs² (τ − 1/2)`.
    pub tau: f64,
    /// Number of velocity directions `Q` (always 9: D2Q9).
    pub q: usize,
    /// Number of grid nodes `nx * ny`.
    pub n_nodes: usize,
    /// Number of nodes in x (periodic).
    pub nx: usize,
    /// Number of nodes in y (periodic).
    pub ny: usize,
}

impl PhysicsInformedLbm {
    /// Create a solver on an `nx × ny` periodic D2Q9 grid at rest (`ρ = 1`,
    /// `u = 0`) with relaxation time `tau` and a default correction network whose
    /// output layer is zero, i.e. an untrained solver that reproduces plain BGK.
    ///
    /// # Errors
    /// [`Error::InvalidGridDimensions`] for an empty grid,
    /// [`Error::UnsupportedVelocitySet`] unless `q == 9` and
    /// [`Error::UnstableRelaxationTime`] unless `tau > 0.5`.
    pub fn new(nx: usize, ny: usize, q: usize, tau: f64) -> Result<Self> {
        Self::validate_lattice(nx, ny, q, tau)?;
        let mut network = NeuralLbm::default_network(q + 1, q);
        network.zero_output_layer();
        Self::with_network(nx, ny, q, tau, network)
    }

    /// Like [`new`](Self::new) with a caller-supplied correction network, which
    /// must map the `q + 1 = 10` features `[ρ, f_0, …, f_8]` to `q = 9` values.
    ///
    /// # Errors
    /// The errors of [`new`](Self::new) and [`Error::ComponentCountMismatch`] for a
    /// network of the wrong shape.
    pub fn with_network(
        nx: usize,
        ny: usize,
        q: usize,
        tau: f64,
        network: NeuralLbm,
    ) -> Result<Self> {
        Self::validate_lattice(nx, ny, q, tau)?;
        network.check_shape(q + 1, q)?;
        let n_nodes = nx
            .checked_mul(ny)
            .filter(|n| n.checked_mul(q).is_some())
            .ok_or(Error::InvalidGridDimensions { nx, ny })?;
        let rest = equilibrium_d2q9(1.0, 0.0, 0.0);
        let mut lbm_grid = Vec::with_capacity(n_nodes * q);
        for _ in 0..n_nodes {
            lbm_grid.extend_from_slice(&rest);
        }
        Ok(Self {
            neural_correction: network,
            lbm_grid,
            tau,
            q,
            n_nodes,
            nx,
            ny,
        })
    }

    fn validate_lattice(nx: usize, ny: usize, q: usize, tau: f64) -> Result<()> {
        Error::check_grid(nx, ny)?;
        if q != Q_D2Q9 {
            return Err(Error::UnsupportedVelocitySet {
                name: format!("D2Q{q}"),
            });
        }
        Error::check_tau(tau)
    }

    /// Check that the public fields still describe a consistent D2Q9 grid.
    fn check_state(&self) -> Result<()> {
        let expected = Q_D2Q9 * self.nx * self.ny;
        if self.q != Q_D2Q9 || self.n_nodes != self.nx * self.ny || self.lbm_grid.len() != expected
        {
            return Err(Error::ComponentCountMismatch {
                expected,
                found: self.lbm_grid.len(),
                field: "lbm_grid".to_string(),
            });
        }
        Ok(())
    }

    /// Network features `[ρ, f_0, …, f_8]` of a node.
    fn collision_features(fi: &[f64; Q_D2Q9]) -> Vec<f64> {
        let mut features = Vec::with_capacity(Q_D2Q9 + 1);
        features.push(fi.iter().sum());
        features.extend_from_slice(fi);
        features
    }

    /// Reset every node to the equilibrium of the fields `field(x, y) = (ρ, ux, uy)`.
    ///
    /// # Errors
    /// [`Error::ComponentCountMismatch`] if the public fields are inconsistent.
    pub fn initialize_equilibrium<F>(&mut self, field: F) -> Result<()>
    where
        F: Fn(usize, usize) -> (f64, f64, f64),
    {
        self.check_state()?;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let (rho, ux, uy) = field(x, y);
                let base = (y * self.nx + x) * Q_D2Q9;
                self.lbm_grid[base..base + Q_D2Q9].copy_from_slice(&equilibrium_d2q9(rho, ux, uy));
            }
        }
        Ok(())
    }

    /// Density and velocity `(ρ, ux, uy)` of the node `(x, y)`.
    ///
    /// # Errors
    /// [`Error::IndexOutOfBounds`] outside the grid.
    pub fn macroscopic(&self, x: usize, y: usize) -> Result<(f64, f64, f64)> {
        self.check_state()?;
        if x >= self.nx || y >= self.ny {
            return Err(Error::IndexOutOfBounds {
                x,
                y,
                nx: self.nx,
                ny: self.ny,
            });
        }
        let base = (y * self.nx + x) * Q_D2Q9;
        let mut fi = [0.0_f64; Q_D2Q9];
        fi.copy_from_slice(&self.lbm_grid[base..base + Q_D2Q9]);
        Ok(macros_from_d2q9(&fi))
    }

    /// Total mass `Σ f_i` of the grid (conserved by every step).
    pub fn total_mass(&self) -> f64 {
        self.lbm_grid.iter().sum()
    }

    /// Total momentum `Σ f_i e_i` of the grid (conserved by every step).
    pub fn total_momentum(&self) -> [f64; 2] {
        let mut momentum = [0.0_f64; 2];
        for node in self.lbm_grid.chunks_exact(Q_D2Q9) {
            for (f, velocity) in node.iter().zip(D2Q9_VELOCITIES.iter()) {
                momentum[0] += f * f64::from(velocity[0]);
                momentum[1] += f * f64::from(velocity[1]);
            }
        }
        momentum
    }

    /// Conservative neural collision correction `c(f)` of a node: the network
    /// output for the features `[ρ, f_0, …, f_8]`, projected so that it carries
    /// neither mass nor momentum.
    ///
    /// Network outputs beyond the ninth are ignored and missing ones count as zero,
    /// so a network of the wrong shape cannot make the step panic.
    pub fn neural_force_correction(&self, fi: &[f64; Q_D2Q9]) -> [f64; Q_D2Q9] {
        let mut features = Self::collision_features(fi);
        features.resize(self.neural_correction.input_size, 0.0);
        let raw = self.neural_correction.forward(&features);
        let mut correction = [0.0_f64; Q_D2Q9];
        for (c, r) in correction.iter_mut().zip(raw.iter()) {
            *c = *r;
        }
        project_conservative(&mut correction);
        correction
    }

    /// The state one time step ahead: BGK collision with the neural correction at
    /// every node, then periodic pull-streaming through a scratch buffer.
    fn advance(&self) -> Result<Vec<f64>> {
        self.check_state()?;
        let (nx, ny) = (self.nx, self.ny);
        let mut post = vec![0.0_f64; self.lbm_grid.len()];
        for (node, collided) in self
            .lbm_grid
            .chunks_exact(Q_D2Q9)
            .zip(post.chunks_exact_mut(Q_D2Q9))
        {
            let mut fi = [0.0_f64; Q_D2Q9];
            fi.copy_from_slice(node);
            let (rho, ux, uy) = macros_from_d2q9(&fi);
            let feq = equilibrium_d2q9(rho, ux, uy);
            let correction = self.neural_force_correction(&fi);
            for i in 0..Q_D2Q9 {
                collided[i] = fi[i] - (fi[i] - feq[i]) / self.tau + correction[i];
            }
        }
        let mut next = vec![0.0_f64; post.len()];
        for y in 0..ny {
            for x in 0..nx {
                let destination = (y * nx + x) * Q_D2Q9;
                for (i, velocity) in D2Q9_VELOCITIES.iter().enumerate() {
                    let source = upstream(y, velocity[1], ny) * nx + upstream(x, velocity[0], nx);
                    next[destination + i] = post[source * Q_D2Q9 + i];
                }
            }
        }
        Ok(next)
    }

    /// Advance the simulation by one LBM time step with neural correction
    /// (collision, then periodic streaming).
    ///
    /// # Errors
    /// [`Error::ComponentCountMismatch`] if the public fields no longer describe a
    /// consistent `q = 9` grid of `nx * ny` nodes.
    pub fn step(&mut self) -> Result<()> {
        self.lbm_grid = self.advance()?;
        Ok(())
    }

    /// Steady-state residual of the current state: the root-mean-square change
    /// of the distribution functions over one step,
    /// `sqrt(mean (f_{n+1} − f_n)²)`.
    ///
    /// It vanishes exactly when the state is a fixed point of the (corrected)
    /// LBM update and is evaluated on a copy, leaving the state untouched.
    ///
    /// # Errors
    /// As [`step`](Self::step).
    pub fn compute_residual(&self) -> Result<f64> {
        let next = self.advance()?;
        let sum_sq: f64 = next
            .iter()
            .zip(self.lbm_grid.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        Ok((sum_sq / next.len() as f64).sqrt())
    }

    /// Fit the neural correction to reference collision data by gradient descent.
    ///
    /// Each sample pairs the pre-collision populations `f` of a node with the
    /// post-collision populations `f_ref` the reference solver produced from them.
    /// The target of the network is the part of `f_ref − BGK(f)` that conserves
    /// mass and momentum (the conservative projection is exactly what
    /// [`neural_force_correction`](Self::neural_force_correction) applies), so that
    /// afterwards `BGK(f) + c(f)` approximates `f_ref`.
    ///
    /// Returns the final mean squared error over the samples.  The collision
    /// residuals are small, so [`Optimizer::Adam`] is the sensible choice unless the
    /// learning rate of plain gradient descent has been tuned to their scale.
    ///
    /// # Errors
    /// [`Error::ComponentCountMismatch`] for slices of different length and the
    /// errors of [`NeuralLbm::train_with`].
    pub fn train_collision_correction(
        &mut self,
        pre_collision: &[[f64; Q_D2Q9]],
        reference_post_collision: &[[f64; Q_D2Q9]],
        optimizer: Optimizer,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<f64> {
        if pre_collision.len() != reference_post_collision.len() {
            return Err(Error::ComponentCountMismatch {
                expected: pre_collision.len(),
                found: reference_post_collision.len(),
                field: "reference post-collision populations".to_string(),
            });
        }
        let mut inputs = Vec::with_capacity(pre_collision.len());
        let mut targets = Vec::with_capacity(pre_collision.len());
        for (fi, reference) in pre_collision.iter().zip(reference_post_collision.iter()) {
            let (rho, ux, uy) = macros_from_d2q9(fi);
            let feq = equilibrium_d2q9(rho, ux, uy);
            let mut residual = [0.0_f64; Q_D2Q9];
            for i in 0..Q_D2Q9 {
                residual[i] = reference[i] - (fi[i] - (fi[i] - feq[i]) / self.tau);
            }
            project_conservative(&mut residual);
            inputs.push(Self::collision_features(fi));
            targets.push(residual.to_vec());
        }
        self.neural_correction
            .train_with(optimizer, &inputs, &targets, learning_rate, epochs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeepONet LBM
// ─────────────────────────────────────────────────────────────────────────────

/// Deep operator network (DeepONet) for LBM solution operators.
///
/// A DeepONet consists of a **branch net** encoding the input function (e.g.
/// initial/boundary conditions) and a **trunk net** encoding the query
/// location (space-time coordinates).  The output is their inner product.
#[derive(Debug, Clone)]
pub struct DeepONetLbm {
    /// Branch network — encodes the input function.
    pub branch_net: NeuralLbm,
    /// Trunk network — encodes the query coordinates.
    pub trunk_net: NeuralLbm,
}

impl DeepONetLbm {
    /// Create a `DeepONetLbm` with given branch and trunk networks.
    pub fn new(branch_net: NeuralLbm, trunk_net: NeuralLbm) -> Self {
        Self {
            branch_net,
            trunk_net,
        }
    }

    /// Create a default `DeepONetLbm` with small networks.
    pub fn default_network(branch_input: usize, trunk_input: usize, basis_size: usize) -> Self {
        let branch_net = NeuralLbm::default_network(branch_input, basis_size);
        let trunk_net = NeuralLbm::default_network(trunk_input, basis_size);
        Self::new(branch_net, trunk_net)
    }

    /// Evaluate the DeepONet: compute `branch(u) · trunk(y)`.
    ///
    /// - `branch_input`: the discrete input function values.
    /// - `trunk_input`: the query coordinates (e.g. `[x, y, t]`).
    pub fn evaluate(&self, branch_input: &[f64], trunk_input: &[f64]) -> f64 {
        let b = self.branch_net.forward(branch_input);
        let t = self.trunk_net.forward(trunk_input);
        b.iter().zip(t.iter()).map(|(&bi, &ti)| bi * ti).sum()
    }

    /// Predict the LBM solution at multiple query points given the input
    /// function encoded in `branch_input`.
    ///
    /// Returns one scalar per query point.
    pub fn predict_solution(&self, branch_input: &[f64], query_points: &[Vec<f64>]) -> Vec<f64> {
        let b = self.branch_net.forward(branch_input);
        query_points
            .iter()
            .map(|qp| {
                let t = self.trunk_net.forward(qp);
                b.iter().zip(t.iter()).map(|(&bi, &ti)| bi * ti).sum()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural BGK collision
// ─────────────────────────────────────────────────────────────────────────────

/// Neural-network augmented BGK collision operator.
///
/// Adds a learned correction term `correction[i]` to the standard BGK update:
///
/// ```text
/// f_i* = f_i - (f_i - f_i^eq) / τ + correction_i
/// ```
pub fn neural_bgk_collision(fi: &[f64], fi_eq: &[f64], tau: f64, correction: &[f64]) -> Vec<f64> {
    assert_eq!(
        fi.len(),
        fi_eq.len(),
        "fi and fi_eq must have the same length"
    );
    let corr_len = correction.len();
    fi.iter()
        .zip(fi_eq.iter())
        .enumerate()
        .map(|(i, (&f, &feq))| {
            let corr = if i < corr_len { correction[i] } else { 0.0 };
            f - (f - feq) / tau + corr
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Data-driven LBM (DMD / ROM)
// ─────────────────────────────────────────────────────────────────────────────
//
// `DataDrivenLbm`, the exact dynamic mode decomposition of distribution-function
// snapshots and the reduced-order model built on it, lives in the `dmd` submodule
// and is re-exported at the top of this module.

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Simple deterministic weight initialiser (no external dependencies).
///
/// Produces values in `[-1, 1]` based on row/column indices.
fn simple_weight(row: usize, col: usize, _n_cols: usize) -> f64 {
    ((row * 7 + col * 13) % 17) as f64 / 8.5 - 1.0
}

/// Compute the macroscopic density from a distribution function slice.
///
/// `rho = sum_i f_i`
pub fn compute_density(fi: &[f64]) -> f64 {
    fi.iter().sum()
}

/// Compute the macroscopic velocity component from distributions.
///
/// Assumes D2Q9 conventions with `e[i]` being the x-component of the
/// `i`-th velocity direction.
pub fn compute_velocity(fi: &[f64], e: &[f64]) -> f64 {
    let rho = compute_density(fi);
    if rho.abs() < 1e-15 {
        return 0.0;
    }
    fi.iter().zip(e.iter()).map(|(&f, &ei)| f * ei).sum::<f64>() / rho
}

/// Compute the Maxwell-Boltzmann equilibrium distribution for a single
/// velocity direction with weight `w`, lattice velocity `e·u`, and speed of
/// sound `cs`.
pub fn equilibrium_fi(rho: f64, eu: f64, u2: f64, w: f64, cs: f64) -> f64 {
    let cs2 = cs * cs;
    rho * w * (1.0 + eu / cs2 + eu * eu / (2.0 * cs2 * cs2) - u2 / (2.0 * cs2))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- ActivationFn tests ---

    #[test]
    fn test_relu_positive() {
        assert_eq!(ActivationFn::Relu.apply(3.0), 3.0);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(ActivationFn::Relu.apply(-2.0), 0.0);
    }

    #[test]
    fn test_tanh_zero() {
        assert!((ActivationFn::Tanh.apply(0.0)).abs() < 1e-15);
    }

    #[test]
    fn test_sigmoid_zero() {
        let s = ActivationFn::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_passthrough() {
        assert_eq!(ActivationFn::Linear.apply(42.0), 42.0);
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(ActivationFn::Relu.derivative(1.0), 1.0);
        assert_eq!(ActivationFn::Relu.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_sigmoid_derivative_zero() {
        let d = ActivationFn::Sigmoid.derivative(0.0);
        assert!((d - 0.25).abs() < 1e-10);
    }

    // --- NeuralLbmLayer tests ---

    #[test]
    fn test_layer_forward_identity() {
        // 2×2 identity weight, zero bias, Linear activation.
        let layer = NeuralLbmLayer::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
            ActivationFn::Linear,
        );
        let out = layer.forward(&[3.0, 5.0]);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_layer_sizes() {
        let layer = NeuralLbmLayer::new(
            vec![vec![1.0, 2.0, 3.0]; 4],
            vec![0.0; 4],
            ActivationFn::Relu,
        );
        assert_eq!(layer.input_size(), 3);
        assert_eq!(layer.output_size(), 4);
    }

    #[test]
    fn test_layer_relu_clamps_negative() {
        let layer = NeuralLbmLayer::new(vec![vec![-1.0]], vec![0.0], ActivationFn::Relu);
        let out = layer.forward(&[1.0]);
        assert_eq!(out[0], 0.0);
    }

    // --- NeuralLbm tests ---

    #[test]
    fn test_neural_lbm_forward_output_size() {
        let net = NeuralLbm::default_network(4, 9);
        let out = net.forward(&[0.1, 0.2, 0.05, 0.0]);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn test_predict_equilibrium_output_size() {
        let net = NeuralLbm::default_network(3, 9);
        let out = net.predict_equilibrium(1.0, &[0.1, 0.05]);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn test_closure_model_returns_scalar() {
        let net = NeuralLbm::default_network(1, 1);
        let nu_t = net.closure_model(0.5);
        assert!(nu_t.is_finite());
    }

    #[test]
    fn test_turbulent_viscosity_non_negative() {
        let net = NeuralLbm::default_network(4, 1);
        let nu_t = net.turbulent_viscosity(&[0.1, 0.2, 0.3, 0.4]);
        assert!(nu_t >= 0.0);
    }

    #[test]
    fn test_neural_lbm_empty_layers() {
        let net = NeuralLbm::new(vec![], 3, 3);
        let out = net.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    // --- PhysicsInformedLbm tests ---

    #[test]
    fn test_pilbm_residual_initial() {
        let pil = PhysicsInformedLbm::new(5, 2, 9, 1.0).unwrap();
        let res = pil.compute_residual().unwrap();
        assert!(res.is_finite());
    }

    #[test]
    fn test_pilbm_step_preserves_size() {
        let mut pil = PhysicsInformedLbm::new(2, 2, 9, 0.6).unwrap();
        let before = pil.lbm_grid.len();
        pil.step().unwrap();
        assert_eq!(pil.lbm_grid.len(), before);
    }

    #[test]
    fn test_pilbm_neural_correction_size() {
        let pil = PhysicsInformedLbm::new(2, 1, 9, 0.6).unwrap();
        let fi = [1.0 / 9.0; 9];
        let corr = pil.neural_force_correction(&fi);
        assert_eq!(corr.len(), 9);
    }

    #[test]
    fn test_pilbm_multiple_steps() {
        let mut pil = PhysicsInformedLbm::new(2, 2, 9, 1.0).unwrap();
        for _ in 0..5 {
            pil.step().unwrap();
        }
        assert!(pil.compute_residual().unwrap().is_finite());
    }

    // --- DeepONetLbm tests ---

    #[test]
    fn test_deeponet_evaluate_finite() {
        let net = DeepONetLbm::default_network(9, 3, 8);
        let branch_in = vec![1.0 / 9.0; 9];
        let trunk_in = vec![0.5, 0.5, 0.0];
        let val = net.evaluate(&branch_in, &trunk_in);
        assert!(val.is_finite());
    }

    #[test]
    fn test_deeponet_predict_solution_count() {
        let net = DeepONetLbm::default_network(9, 2, 8);
        let branch_in = vec![0.1; 9];
        let queries = vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]];
        let preds = net.predict_solution(&branch_in, &queries);
        assert_eq!(preds.len(), 3);
    }

    // --- neural_bgk_collision tests ---

    #[test]
    fn test_bgk_no_correction_equilibrium() {
        let feq: Vec<f64> = vec![1.0 / 9.0; 9];
        let fi = feq.clone();
        let out = neural_bgk_collision(&fi, &feq, 1.0, &[]);
        for (a, b) in out.iter().zip(fi.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bgk_correction_applied() {
        let fi = vec![0.2, 0.1];
        let feq = vec![0.15, 0.15];
        let corr = vec![0.01, -0.01];
        let out = neural_bgk_collision(&fi, &feq, 1.0, &corr);
        // out[0] = 0.2 - (0.2-0.15)/1.0 + 0.01 = 0.2 - 0.05 + 0.01 = 0.16
        assert!((out[0] - 0.16).abs() < 1e-12);
    }

    #[test]
    fn test_bgk_output_length() {
        let fi = vec![0.1; 5];
        let feq = vec![0.1; 5];
        let out = neural_bgk_collision(&fi, &feq, 1.5, &[0.0; 5]);
        assert_eq!(out.len(), 5);
    }

    // --- DataDrivenLbm tests ---

    #[test]
    fn test_dmd_modes_empty() {
        let mut ddl = DataDrivenLbm::new();
        // Without snapshots there is nothing to decompose: an error, not a panic.
        assert!(ddl.dmd_modes().is_err());
        assert!(ddl.modes.is_empty());
    }

    #[test]
    fn test_dmd_modes_two_snapshots() {
        let mut ddl = DataDrivenLbm::new();
        ddl.add_snapshot(vec![1.0, 0.0, 0.0]);
        ddl.add_snapshot(vec![0.9, 0.1, 0.0]);
        ddl.dmd_modes().unwrap();
        assert_eq!(ddl.modes.len(), 1);
        assert_eq!(ddl.eigenvalues.len(), 1);
    }

    #[test]
    fn test_rom_predict_size() {
        let mut ddl = DataDrivenLbm::new();
        for k in 0..5 {
            let snap: Vec<f64> = (0..9).map(|i| (i + k) as f64 * 0.1).collect();
            ddl.add_snapshot(snap);
        }
        ddl.dmd_modes().unwrap();
        let pred = ddl.rom_predict(2);
        assert_eq!(pred.len(), 9);
    }

    #[test]
    fn test_snapshot_count() {
        let mut ddl = DataDrivenLbm::new();
        assert_eq!(ddl.snapshot_count(), 0);
        ddl.add_snapshot(vec![1.0, 2.0]);
        assert_eq!(ddl.snapshot_count(), 1);
    }

    #[test]
    fn test_reconstruction_error_out_of_bounds() {
        let ddl = DataDrivenLbm::new();
        assert!(ddl.reconstruction_error(0).is_nan());
    }

    // --- Helper utilities tests ---

    #[test]
    fn test_compute_density() {
        let fi = vec![0.1, 0.2, 0.3, 0.4];
        assert!((compute_density(&fi) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_velocity_zero_rho() {
        let fi = vec![0.0; 4];
        let e = vec![1.0, 0.0, -1.0, 0.0];
        assert_eq!(compute_velocity(&fi, &e), 0.0);
    }

    #[test]
    fn test_equilibrium_fi_uniform() {
        // At rest (u=0), equilibrium should be rho * w.
        let val = equilibrium_fi(1.0, 0.0, 0.0, 4.0 / 9.0, (1.0f64 / 3.0).sqrt());
        assert!((val - 4.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_weight_range() {
        for r in 0..8 {
            for c in 0..8 {
                let w = simple_weight(r, c, 8);
                assert!((-1.0_f64..=1.0).contains(&w), "weight {w} out of range");
            }
        }
    }
}

#[cfg(test)]
mod regression_tests;
