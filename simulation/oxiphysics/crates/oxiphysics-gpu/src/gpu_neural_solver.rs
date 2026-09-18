// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU neural network solver for physics (CPU mock backend).
//!
//! Provides a multi-layer perceptron (MLP) framework with back-propagation and
//! physics-informed neural network (PINN) utilities:
//! - [`NeuralLayer`]: single dense layer with several activation modes and
//!   Xavier-uniform initialisation.
//! - [`GpuNeuralSolver`]: stacked MLP (tanh hidden layers, linear output) with
//!   forward pass and SGD training ([`GpuNeuralSolver::train_step`]).
//! - [`PhysicsNeuralNet`]: PINN wrapper with PDE + boundary-condition loss and
//!   a trainer ([`PhysicsNeuralNet::train`]) for the 1-D Poisson equation.
//! - Activation functions: [`ns_relu`], [`ns_sigmoid`], [`ns_softmax`].
//! - Loss functions: [`ns_mse_loss`], [`ns_mae_loss`].
//! - PINN residuals: [`pinn_residual`], [`pinn_boundary_loss`].
//!
//! Failures (bad layer sizes, mismatched data lengths, diverging training) are
//! reported as [`crate::error::Error`]; nothing here panics on bad input.

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use crate::error::{Error, Result};

/// Seed of the default weight initialisation: [`GpuNeuralSolver::new`],
/// [`PhysicsNeuralNet::new`] and [`NeuralLayer::new`] are deterministic and
/// draw their Xavier weights from a generator seeded with (a function of) this
/// value.  Use the `with_seed` constructors for other initialisations.
pub const DEFAULT_INIT_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Step `h` of the central difference that gives the PINN's second derivative
/// `u_xx ≈ (u(x + h) − 2 u(x) + u(x − h)) / h²`.
pub const PINN_FD_STEP: f64 = 1.0e-3;

// ── Activation functions ──────────────────────────────────────────────────────

/// Rectified linear unit activation: `max(0, x)`.
pub fn ns_relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Logistic sigmoid activation: `1 / (1 + e^{-x})`.
pub fn ns_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax of a slice: normalised exponentials `exp(x_i) / Σ exp(x_j)`.
///
/// Uses the max-subtraction trick for numerical stability.
/// Returns an empty `Vec` when `x` is empty.
pub fn ns_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ── Loss functions ────────────────────────────────────────────────────────────

/// Mean-squared error loss: `mean((predicted_i − target_i)²)`.
///
/// Returns `0.0` when `predicted` is empty.
pub fn ns_mse_loss(predicted: &[f64], target: &[f64]) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }
    let n = predicted.len().min(target.len());
    predicted[..n]
        .iter()
        .zip(target[..n].iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Mean absolute error loss: `mean(|predicted_i − target_i|)`.
///
/// Returns `0.0` when `predicted` is empty.
pub fn ns_mae_loss(predicted: &[f64], target: &[f64]) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }
    let n = predicted.len().min(target.len());
    predicted[..n]
        .iter()
        .zip(target[..n].iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>()
        / n as f64
}

// ── PINN residuals ────────────────────────────────────────────────────────────

/// Physics-informed residual for the 1-D Poisson equation `−u_xx = source`.
///
/// Returns `−u_xx − source`; this should be driven to zero during training.
pub fn pinn_residual(u: f64, u_xx: f64, source: f64) -> f64 {
    let _ = u; // u itself is not needed for the Poisson residual
    -u_xx - source
}

/// Boundary-condition loss: MSE between the predicted boundary values and the
/// prescribed Dirichlet data.
pub fn pinn_boundary_loss(u_boundary: &[f64], u_target: &[f64]) -> f64 {
    ns_mse_loss(u_boundary, u_target)
}

// ── NeuralLayer ───────────────────────────────────────────────────────────────

/// A single fully-connected (dense) neural network layer.
///
/// Weights are stored in row-major order: `weights[i * n_in + j]` is the
/// weight from input `j` to output neuron `i`.
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// Flattened weight matrix of shape `[n_out × n_in]`.
    pub weights: Vec<f64>,
    /// Bias vector of length `n_out`.
    pub biases: Vec<f64>,
    /// Number of input features.
    pub n_in: usize,
    /// Number of output neurons.
    pub n_out: usize,
}

impl NeuralLayer {
    /// Create a layer with Xavier-uniform weights and zero biases.
    ///
    /// The weights come from a generator seeded deterministically from
    /// [`DEFAULT_INIT_SEED`] and the layer shape, so equal shapes give equal
    /// layers; use [`NeuralLayer::xavier`] to draw from a generator of your own.
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let seed = DEFAULT_INIT_SEED ^ ((n_in as u64) << 32) ^ (n_out as u64);
        Self::xavier(n_in, n_out, &mut StdRng::seed_from_u64(seed))
    }

    /// Create a layer with Xavier (Glorot) uniform weights, drawn from
    /// `U(−a, a)` with `a = sqrt(6 / (n_in + n_out))`, and zero biases.
    ///
    /// Random weights give every neuron its own response, which a constant
    /// initialisation cannot (identical neurons receive identical gradients and
    /// stay identical).
    pub fn xavier<R: Rng + ?Sized>(n_in: usize, n_out: usize, rng: &mut R) -> Self {
        let limit = (6.0 / (n_in + n_out).max(1) as f64).sqrt();
        let weights = (0..n_out * n_in)
            .map(|_| rng.random_range(-limit..limit))
            .collect();
        Self {
            weights,
            biases: vec![0.0; n_out],
            n_in,
            n_out,
        }
    }

    /// Linear forward pass (no activation): `W·x + b`.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let n = self.n_in.min(input.len());
        (0..self.n_out)
            .map(|i| {
                let base = i * self.n_in;
                let dot: f64 = (0..n).map(|j| self.weights[base + j] * input[j]).sum();
                dot + self.biases[i]
            })
            .collect()
    }

    /// Forward pass with ReLU activation applied element-wise.
    pub fn relu_forward(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).into_iter().map(ns_relu).collect()
    }

    /// Forward pass with tanh activation applied element-wise.
    pub fn tanh_forward(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).into_iter().map(|v| v.tanh()).collect()
    }

    /// Number of output neurons.
    pub fn output_size(&self) -> usize {
        self.n_out
    }

    /// Number of input features.
    pub fn input_size(&self) -> usize {
        self.n_in
    }
}

// ── GpuNeuralSolver ───────────────────────────────────────────────────────────

/// Gradients of the loss with respect to the parameters of one dense layer.
#[derive(Debug, Clone)]
struct LayerGradient {
    weights: Vec<f64>,
    biases: Vec<f64>,
}

/// Multi-layer perceptron running on the CPU mock GPU backend.
///
/// Layers are stacked so that the output of layer `i` feeds into layer `i+1`.
/// All hidden activations use `tanh` (smooth, so that the network has non-zero
/// second derivatives, which physics-informed losses need); the final layer is
/// linear.  Training is plain gradient descent with step [`learning_rate`]
/// (see [`train_step`]).
///
/// [`learning_rate`]: GpuNeuralSolver::learning_rate
/// [`train_step`]: GpuNeuralSolver::train_step
#[derive(Debug, Clone)]
pub struct GpuNeuralSolver {
    /// Ordered list of dense layers.
    pub layers: Vec<NeuralLayer>,
    /// Step size of the gradient-descent updates (finite and non-negative).
    pub learning_rate: f64,
}

impl GpuNeuralSolver {
    /// Build a network from an ordered list of layer widths with the default
    /// initialisation seed ([`DEFAULT_INIT_SEED`]).
    ///
    /// `layer_sizes` must contain at least two elements, all non-zero: the first
    /// is the input dimension and the last is the output dimension.  Returns
    /// [`Error::InvalidConfig`] for fewer than two sizes, a zero size, or a
    /// learning rate that is negative or not finite.
    pub fn new(layer_sizes: &[usize], lr: f64) -> Result<Self> {
        Self::with_seed(layer_sizes, lr, DEFAULT_INIT_SEED)
    }

    /// Like [`new`](GpuNeuralSolver::new), drawing the Xavier-uniform weights of
    /// all layers from one generator seeded with `seed`.
    pub fn with_seed(layer_sizes: &[usize], lr: f64, seed: u64) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(Error::invalid_config(format!(
                "a network needs at least an input and an output size, got {} size(s)",
                layer_sizes.len()
            )));
        }
        if layer_sizes.contains(&0) {
            return Err(Error::invalid_config(
                "layer sizes must be non-zero".to_string(),
            ));
        }
        check_learning_rate(lr)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let layers = layer_sizes
            .windows(2)
            .map(|w| NeuralLayer::xavier(w[0], w[1], &mut rng))
            .collect();
        Ok(Self {
            layers,
            learning_rate: lr,
        })
    }

    /// Run a full forward pass through all layers (tanh hidden, linear output).
    pub fn forward_pass(&self, input: &[f64]) -> Vec<f64> {
        let mut x: Vec<f64> = input.to_vec();
        let last = self.layers.len().saturating_sub(1);
        for (i, layer) in self.layers.iter().enumerate() {
            x = if i < last {
                layer.tanh_forward(&x)
            } else {
                layer.forward(&x)
            };
        }
        x
    }

    /// Number of layers (= number of weight matrices).
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Alias for [`forward_pass`](GpuNeuralSolver::forward_pass).
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.forward_pass(input)
    }

    /// Dimension of the network input (zero for a network without layers).
    pub fn input_size(&self) -> usize {
        self.layers.first().map_or(0, NeuralLayer::input_size)
    }

    /// Dimension of the network output (zero for a network without layers).
    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, NeuralLayer::output_size)
    }

    /// One gradient-descent step on a single training sample.
    ///
    /// Minimises the squared error `mean_k (ŷ_k − y_k)²` of the network output
    /// `ŷ = predict(input)` against `target`: the gradient is back-propagated
    /// through the layers and every parameter moves by
    /// `−learning_rate × ∂loss/∂parameter`.  Returns the loss *before* the
    /// update.
    ///
    /// Fails with [`Error::DimensionMismatch`] when `input` or `target` does not
    /// match the network's input or output size, and with
    /// [`Error::InvalidConfig`] for inconsistent layers or an invalid learning
    /// rate; the weights are untouched on error.
    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> Result<f64> {
        check_learning_rate(self.learning_rate)?;
        let (loss, gradients) = self.mse_gradients(input, target)?;
        self.apply_gradients(&gradients);
        Ok(loss)
    }

    /// Squared-error loss of one sample and its parameter gradients.
    fn mse_gradients(&self, input: &[f64], target: &[f64]) -> Result<(f64, Vec<LayerGradient>)> {
        self.check_layers()?;
        check_len("network input", self.input_size(), input.len())?;
        check_len("training target", self.output_size(), target.len())?;
        let activations = self.forward_cache(input);
        let output = activations.last().map_or(&[][..], Vec::as_slice);
        let count = target.len() as f64;
        let loss = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t) * (o - t))
            .sum::<f64>()
            / count;
        let output_gradient: Vec<f64> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| 2.0 * (o - t) / count)
            .collect();
        let mut gradients = self.zero_gradients();
        self.backward(&activations, &output_gradient, &mut gradients);
        Ok((loss, gradients))
    }

    /// Verify that every layer's arrays match its shape and that consecutive
    /// layers chain (`n_out` of one is `n_in` of the next).
    fn check_layers(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(Error::invalid_config(
                "the network has no layers".to_string(),
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.weights.len() != layer.n_out * layer.n_in || layer.biases.len() != layer.n_out
            {
                return Err(Error::invalid_config(format!(
                    "layer {index}: {} weights and {} biases do not match shape {} x {}",
                    layer.weights.len(),
                    layer.biases.len(),
                    layer.n_out,
                    layer.n_in
                )));
            }
        }
        for (index, pair) in self.layers.windows(2).enumerate() {
            if pair[0].n_out != pair[1].n_in {
                return Err(Error::invalid_config(format!(
                    "layer {index} outputs {} values but layer {} takes {}",
                    pair[0].n_out,
                    index + 1,
                    pair[1].n_in
                )));
            }
        }
        Ok(())
    }

    /// Forward pass that keeps every layer's activations: entry 0 is the input,
    /// entry `l + 1` the output of layer `l` (tanh for hidden layers, linear for
    /// the last one).
    fn forward_cache(&self, input: &[f64]) -> Vec<Vec<f64>> {
        let last = self.layers.len().saturating_sub(1);
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        for (index, layer) in self.layers.iter().enumerate() {
            let previous = activations.last().map_or(&[][..], Vec::as_slice);
            let next = if index < last {
                layer.tanh_forward(previous)
            } else {
                layer.forward(previous)
            };
            activations.push(next);
        }
        activations
    }

    fn zero_gradients(&self) -> Vec<LayerGradient> {
        self.layers
            .iter()
            .map(|layer| LayerGradient {
                weights: vec![0.0; layer.weights.len()],
                biases: vec![0.0; layer.biases.len()],
            })
            .collect()
    }

    /// Back-propagate `output_gradient = ∂loss/∂output` through the network and
    /// *add* the resulting parameter gradients to `gradients`.
    ///
    /// `activations` is the result of [`forward_cache`](Self::forward_cache) for
    /// the same input.  The derivative of `tanh` is `1 − a²` in terms of the
    /// activation `a`.
    fn backward(
        &self,
        activations: &[Vec<f64>],
        output_gradient: &[f64],
        gradients: &mut [LayerGradient],
    ) {
        let mut delta: Vec<f64> = output_gradient.to_vec();
        for (index, layer) in self.layers.iter().enumerate().rev() {
            let (Some(gradient), Some(inputs)) = (gradients.get_mut(index), activations.get(index))
            else {
                return;
            };
            let width = layer.n_in.max(1);
            for (bias_gradient, &d) in gradient.biases.iter_mut().zip(delta.iter()) {
                *bias_gradient += d;
            }
            for (row, &d) in gradient.weights.chunks_exact_mut(width).zip(delta.iter()) {
                for (weight_gradient, &a) in row.iter_mut().zip(inputs.iter()) {
                    *weight_gradient += d * a;
                }
            }
            if index == 0 {
                break;
            }
            // Gradient with respect to this layer's input, then through the
            // tanh of the previous layer.
            let mut upstream = vec![0.0; layer.n_in];
            for (row, &d) in layer.weights.chunks_exact(width).zip(delta.iter()) {
                for (sum, &w) in upstream.iter_mut().zip(row.iter()) {
                    *sum += w * d;
                }
            }
            for (sum, &a) in upstream.iter_mut().zip(inputs.iter()) {
                *sum *= 1.0 - a * a;
            }
            delta = upstream;
        }
    }

    /// Gradient-descent update: `parameter -= learning_rate × gradient`.
    fn apply_gradients(&mut self, gradients: &[LayerGradient]) {
        let rate = self.learning_rate;
        for (layer, gradient) in self.layers.iter_mut().zip(gradients.iter()) {
            for (w, g) in layer.weights.iter_mut().zip(gradient.weights.iter()) {
                *w -= rate * g;
            }
            for (b, g) in layer.biases.iter_mut().zip(gradient.biases.iter()) {
                *b -= rate * g;
            }
        }
    }
}

/// `Ok` for a finite, non-negative learning rate.
fn check_learning_rate(lr: f64) -> Result<()> {
    if lr.is_finite() && lr >= 0.0 {
        Ok(())
    } else {
        Err(Error::invalid_config(format!(
            "the learning rate must be finite and non-negative, got {lr}"
        )))
    }
}

/// `Ok` when `got == expected`.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::dimension_mismatch(what, expected, got))
    }
}

// ── PhysicsNeuralNet ──────────────────────────────────────────────────────────

/// Physics-informed neural network (PINN).
///
/// Wraps a [`GpuNeuralSolver`] and provides a composite loss that combines
/// a PDE residual term with a boundary-condition MSE term, and a trainer for the
/// 1-D Poisson problem `−u''(x) = f(x)` with Dirichlet data
/// ([`train`](PhysicsNeuralNet::train)).
#[derive(Debug, Clone)]
pub struct PhysicsNeuralNet {
    /// Underlying neural solver.
    pub solver: GpuNeuralSolver,
    /// Weight applied to the PDE residual in the total loss.
    pub pde_weight: f64,
    /// Weight applied to the boundary-condition loss in the total loss.
    pub bc_weight: f64,
}

impl PhysicsNeuralNet {
    /// Create a new PINN from layer sizes, PDE weight, and BC weight, with
    /// learning rate `1e-3` and the default initialisation seed.
    ///
    /// Fails like [`GpuNeuralSolver::new`].  The trainer needs a network with
    /// one input and one output (`layer_sizes` starting and ending in `1`).
    pub fn new(layer_sizes: &[usize], pde_weight: f64, bc_weight: f64) -> Result<Self> {
        Self::with_seed(layer_sizes, pde_weight, bc_weight, DEFAULT_INIT_SEED)
    }

    /// Like [`new`](PhysicsNeuralNet::new), with Xavier weights drawn from a
    /// generator seeded with `seed`.
    pub fn with_seed(
        layer_sizes: &[usize],
        pde_weight: f64,
        bc_weight: f64,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            solver: GpuNeuralSolver::with_seed(layer_sizes, 1e-3, seed)?,
            pde_weight,
            bc_weight,
        })
    }

    /// Compute the weighted total loss:
    /// `pde_weight * |pde_residual| + bc_weight * bc_loss`.
    pub fn total_loss(&self, pde_residual: f64, bc_loss: f64) -> f64 {
        self.pde_weight * pde_residual.abs() + self.bc_weight * bc_loss
    }

    /// Predict the solution at a 1-D coordinate `x`.
    ///
    /// Wraps the scalar input into a slice and extracts the first output.
    pub fn predict(&self, x: f64) -> f64 {
        let out = self.solver.predict(&[x]);
        out.first().copied().unwrap_or(0.0)
    }

    /// Second derivative `u_xx(x)` of the network by the central difference
    /// `(u(x + h) − 2 u(x) + u(x − h)) / h²` with `h =` [`PINN_FD_STEP`].
    pub fn second_derivative(&self, x: f64) -> f64 {
        let h = PINN_FD_STEP;
        (self.predict(x + h) - 2.0 * self.predict(x) + self.predict(x - h)) / (h * h)
    }

    /// Residual `−u_xx(x) − source` of the Poisson equation at `x`
    /// (see [`pinn_residual`]); training drives it to zero.
    pub fn residual_at(&self, x: f64, source: f64) -> f64 {
        pinn_residual(self.predict(x), self.second_derivative(x), source)
    }

    /// Total loss of the Poisson problem for the given training data, without
    /// changing the network:
    /// `total_loss(mean(residual²), mean((u(x_b) − g_b)²))`.
    ///
    /// `collocation` holds the interior points `x_i`, `source[i]` the source
    /// term `f(x_i)` there, and `boundary` the Dirichlet data `(x_b, g_b)`.
    /// See [`train`](PhysicsNeuralNet::train) for the errors.
    pub fn loss(
        &self,
        collocation: &[f64],
        boundary: &[(f64, f64)],
        source: &[f64],
    ) -> Result<f64> {
        self.check_training_data(collocation, boundary, source)?;
        let residual_sq: f64 = collocation
            .iter()
            .zip(source.iter())
            .map(|(&x, &f)| self.residual_at(x, f).powi(2))
            .sum();
        let pde_mse = mean_or_zero(residual_sq, collocation.len());
        let predicted: Vec<f64> = boundary.iter().map(|&(x, _)| self.predict(x)).collect();
        let prescribed: Vec<f64> = boundary.iter().map(|&(_, g)| g).collect();
        Ok(self.total_loss(pde_mse, pinn_boundary_loss(&predicted, &prescribed)))
    }

    /// Train the network on the 1-D Poisson problem `−u'' = f` with Dirichlet
    /// boundary data by full-batch gradient descent, `epochs` times.
    ///
    /// The loss is
    /// `pde_weight · mean_i r_i² + bc_weight · mean_b (u(x_b) − g_b)²` with the
    /// residual `r_i = −u_xx(x_i) − f(x_i)`, where `u_xx` is the central
    /// difference of the network over three evaluations
    /// (`x_i − h`, `x_i`, `x_i + h`, `h =` [`PINN_FD_STEP`]).  The gradient of
    /// the loss is back-propagated through all three evaluations and every
    /// parameter moves by `−solver.learning_rate × gradient`.
    ///
    /// Returns the loss at the start of each epoch (that is, `epochs` values;
    /// call [`loss`](PhysicsNeuralNet::loss) for the final one).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidConfig`] — the network does not have one input and one
    ///   output, the layers are inconsistent, the learning rate or a loss weight
    ///   is negative or not finite, there is no training point at all, or a
    ///   training value is not finite.
    /// * [`Error::DimensionMismatch`] — `source` and `collocation` differ in
    ///   length.
    /// * [`Error::General`] — the loss became non-finite (the learning rate is
    ///   too large); the network is restored to the last weights with a finite
    ///   loss.
    pub fn train(
        &mut self,
        collocation: &[f64],
        boundary: &[(f64, f64)],
        source: &[f64],
        epochs: usize,
    ) -> Result<Vec<f64>> {
        self.check_training_data(collocation, boundary, source)?;
        check_learning_rate(self.solver.learning_rate)?;
        let mut history = Vec::with_capacity(epochs);
        let mut last_good: Option<Vec<NeuralLayer>> = None;
        for epoch in 0..epochs {
            let (loss, gradients) = self.loss_and_gradients(collocation, boundary, source);
            if !loss.is_finite() {
                if let Some(layers) = last_good {
                    self.solver.layers = layers;
                }
                return Err(Error::general(format!(
                    "PINN training diverged: the loss is not finite at epoch {epoch} \
                     (reduce the learning rate, now {})",
                    self.solver.learning_rate
                )));
            }
            history.push(loss);
            last_good = Some(self.solver.layers.clone());
            self.solver.apply_gradients(&gradients);
        }
        Ok(history)
    }

    /// Validate the network and the training data shared by `loss` and `train`.
    fn check_training_data(
        &self,
        collocation: &[f64],
        boundary: &[(f64, f64)],
        source: &[f64],
    ) -> Result<()> {
        self.solver.check_layers()?;
        if self.solver.input_size() != 1 || self.solver.output_size() != 1 {
            return Err(Error::invalid_config(format!(
                "the 1-D PINN needs one input and one output, the network maps {} -> {}",
                self.solver.input_size(),
                self.solver.output_size()
            )));
        }
        check_len("PINN source terms", collocation.len(), source.len())?;
        if collocation.is_empty() && boundary.is_empty() {
            return Err(Error::invalid_config(
                "PINN training needs at least one collocation or boundary point".to_string(),
            ));
        }
        let finite = collocation.iter().all(|x| x.is_finite())
            && source.iter().all(|f| f.is_finite())
            && boundary
                .iter()
                .all(|&(x, g)| x.is_finite() && g.is_finite())
            && self.pde_weight.is_finite()
            && self.bc_weight.is_finite();
        if !finite {
            return Err(Error::invalid_config(
                "PINN training data and loss weights must be finite".to_string(),
            ));
        }
        Ok(())
    }

    /// Loss value and its gradient with respect to all network parameters.
    ///
    /// Each collocation point needs the network at `x − h`, `x` and `x + h`;
    /// the loss depends on the three outputs through
    /// `r = −(u₊ − 2u₀ + u₋)/h² − f`, so `∂L/∂u₊ = ∂L/∂u₋ = −2 w r /(N h²)` and
    /// `∂L/∂u₀ = 4 w r /(N h²)` for `L = w r²/N`, and the three evaluations
    /// are back-propagated separately and their parameter gradients summed.
    fn loss_and_gradients(
        &self,
        collocation: &[f64],
        boundary: &[(f64, f64)],
        source: &[f64],
    ) -> (f64, Vec<LayerGradient>) {
        let h = PINN_FD_STEP;
        let inv_h2 = 1.0 / (h * h);
        let mut gradients = self.solver.zero_gradients();
        let output_of = |activations: &[Vec<f64>]| {
            activations
                .last()
                .and_then(|out| out.first())
                .copied()
                .unwrap_or(0.0)
        };

        let n_collocation = collocation.len() as f64;
        let mut residual_sq = 0.0;
        for (&x, &f) in collocation.iter().zip(source.iter()) {
            let below = self.solver.forward_cache(&[x - h]);
            let centre = self.solver.forward_cache(&[x]);
            let above = self.solver.forward_cache(&[x + h]);
            let u_xx = (output_of(&above) - 2.0 * output_of(&centre) + output_of(&below)) * inv_h2;
            let r = pinn_residual(output_of(&centre), u_xx, f);
            residual_sq += r * r;
            let c = self.pde_weight * 2.0 * r / n_collocation;
            self.solver.backward(&below, &[-c * inv_h2], &mut gradients);
            self.solver
                .backward(&centre, &[2.0 * c * inv_h2], &mut gradients);
            self.solver.backward(&above, &[-c * inv_h2], &mut gradients);
        }

        let n_boundary = boundary.len() as f64;
        let mut boundary_sq = 0.0;
        for &(x, g) in boundary {
            let activations = self.solver.forward_cache(&[x]);
            let error = output_of(&activations) - g;
            boundary_sq += error * error;
            let c = self.bc_weight * 2.0 * error / n_boundary;
            self.solver.backward(&activations, &[c], &mut gradients);
        }

        let loss = self.total_loss(
            mean_or_zero(residual_sq, collocation.len()),
            mean_or_zero(boundary_sq, boundary.len()),
        );
        (loss, gradients)
    }
}

/// `sum / count` as a mean, or zero when there is nothing to average.
fn mean_or_zero(sum: f64, count: usize) -> f64 {
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ns_relu ──────────────────────────────────────────────────────────

    #[test]
    fn relu_negative_is_zero() {
        assert!((ns_relu(-1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn relu_positive_identity() {
        assert!((ns_relu(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn relu_zero_boundary() {
        assert!((ns_relu(0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn relu_large_positive() {
        assert!((ns_relu(1000.0) - 1000.0).abs() < 1e-8);
    }

    // ── ns_sigmoid ───────────────────────────────────────────────────────

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((ns_sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_large_positive_near_one() {
        assert!(ns_sigmoid(100.0) > 0.999);
    }

    #[test]
    fn sigmoid_large_negative_near_zero() {
        assert!(ns_sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn sigmoid_symmetry() {
        let s = ns_sigmoid(2.0);
        assert!((ns_sigmoid(-2.0) - (1.0 - s)).abs() < 1e-12);
    }

    // ── ns_softmax ────────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let x = [1.0, 2.0, 3.0];
        let s = ns_softmax(&x);
        let total: f64 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_empty_input() {
        let s = ns_softmax(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let s = ns_softmax(&[42.0]);
        assert!((s[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_uniform_input() {
        let x = [1.0f64; 4];
        let s = ns_softmax(&x);
        for &v in &s {
            assert!((v - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_all_non_negative() {
        let x = [-3.0, 0.0, 1.0, 5.0];
        let s = ns_softmax(&x);
        for &v in &s {
            assert!(v >= 0.0);
        }
    }

    // ── ns_mse_loss ──────────────────────────────────────────────────────

    #[test]
    fn mse_zero_for_identical() {
        let v = [1.0, 2.0, 3.0];
        assert!((ns_mse_loss(&v, &v) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mse_known_value() {
        let pred = [3.0];
        let target = [1.0];
        // (3-1)^2 / 1 = 4
        assert!((ns_mse_loss(&pred, &target) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn mse_empty_returns_zero() {
        assert!((ns_mse_loss(&[], &[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mse_positive_values() {
        let pred = [1.0, 2.0];
        let target = [0.0, 0.0];
        let loss = ns_mse_loss(&pred, &target);
        assert!(loss > 0.0);
    }

    // ── ns_mae_loss ──────────────────────────────────────────────────────

    #[test]
    fn mae_zero_for_identical() {
        let v = [1.0, 2.0, 3.0];
        assert!((ns_mae_loss(&v, &v) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mae_known_value() {
        let pred = [3.0, 1.0];
        let target = [1.0, 1.0];
        // |3-1| + |1-1| = 2; / 2 = 1
        assert!((ns_mae_loss(&pred, &target) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mae_empty_returns_zero() {
        assert!((ns_mae_loss(&[], &[]) - 0.0).abs() < 1e-12);
    }

    // ── NeuralLayer ───────────────────────────────────────────────────────

    #[test]
    fn neural_layer_output_size() {
        let layer = NeuralLayer::new(4, 3);
        assert_eq!(layer.output_size(), 3);
    }

    #[test]
    fn neural_layer_input_size() {
        let layer = NeuralLayer::new(4, 3);
        assert_eq!(layer.input_size(), 4);
    }

    #[test]
    fn neural_layer_forward_output_length() {
        let layer = NeuralLayer::new(4, 3);
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn neural_layer_relu_forward_non_negative() {
        let layer = NeuralLayer::new(2, 4);
        let out = layer.relu_forward(&[-10.0, -10.0]);
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn neural_layer_tanh_bounded() {
        let layer = NeuralLayer::new(3, 3);
        let out = layer.tanh_forward(&[1.0, 2.0, 3.0]);
        for &v in &out {
            assert!(v > -1.0 && v < 1.0);
        }
    }

    #[test]
    fn neural_layer_zero_input() {
        // With all weights=0.1 and biases=0, output should be 0
        let mut layer = NeuralLayer::new(3, 2);
        layer.weights = vec![0.0; 6];
        let out = layer.forward(&[0.0, 0.0, 0.0]);
        for &v in &out {
            assert!(v.abs() < 1e-12);
        }
    }

    // ── GpuNeuralSolver ───────────────────────────────────────────────────

    #[test]
    fn solver_layer_count() {
        let s = GpuNeuralSolver::new(&[4, 8, 8, 2], 1e-3).expect("valid layer sizes");
        assert_eq!(s.layer_count(), 3);
    }

    #[test]
    fn solver_forward_output_shape() {
        let s = GpuNeuralSolver::new(&[3, 5, 2], 1e-3).expect("valid layer sizes");
        let out = s.forward_pass(&[1.0, 0.0, -1.0]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn solver_predict_same_as_forward() {
        let s = GpuNeuralSolver::new(&[2, 4, 1], 1e-3).expect("valid layer sizes");
        let input = [0.5, -0.5];
        let a = s.forward_pass(&input);
        let b = s.predict(&input);
        assert_eq!(a, b);
    }

    #[test]
    fn solver_single_layer() {
        let s = GpuNeuralSolver::new(&[2, 1], 1e-3).expect("valid layer sizes");
        let out = s.forward_pass(&[1.0, 1.0]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn solver_deep_network_no_panic() {
        let s = GpuNeuralSolver::new(&[10, 20, 20, 20, 5], 1e-4).expect("valid layer sizes");
        let input = vec![0.1; 10];
        let out = s.forward_pass(&input);
        assert_eq!(out.len(), 5);
    }

    // ── pinn_residual ────────────────────────────────────────────────────

    #[test]
    fn pinn_residual_formula() {
        // residual = -u_xx - source
        let r = pinn_residual(0.0, 2.0, 1.0);
        assert!((r - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn pinn_residual_zero_when_satisfied() {
        // If -u_xx == source then residual == 0
        let u_xx = -1.0;
        let source = 1.0;
        let r = pinn_residual(0.0, u_xx, source);
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn pinn_boundary_loss_zero_for_equal() {
        let v = [1.0, 0.0, -1.0];
        assert!((pinn_boundary_loss(&v, &v) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn pinn_boundary_loss_positive_for_different() {
        let u_boundary = [1.0, 2.0];
        let u_target = [0.0, 0.0];
        assert!(pinn_boundary_loss(&u_boundary, &u_target) > 0.0);
    }

    // ── PhysicsNeuralNet ─────────────────────────────────────────────────

    #[test]
    fn pinn_total_loss_formula() {
        let net = PhysicsNeuralNet::new(&[1, 4, 1], 2.0, 3.0).expect("valid layer sizes");
        // pde_residual = 1, bc_loss = 1 => 2*1 + 3*1 = 5
        let loss = net.total_loss(1.0, 1.0);
        assert!((loss - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pinn_total_loss_zero_when_both_zero() {
        let net = PhysicsNeuralNet::new(&[1, 4, 1], 1.0, 1.0).expect("valid layer sizes");
        assert!((net.total_loss(0.0, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn pinn_predict_returns_scalar() {
        let net = PhysicsNeuralNet::new(&[1, 8, 1], 1.0, 1.0).expect("valid layer sizes");
        let _v = net.predict(0.5); // should not panic
    }

    #[test]
    fn pinn_total_loss_pde_only() {
        let net = PhysicsNeuralNet::new(&[1, 4, 1], 5.0, 0.0).expect("valid layer sizes");
        assert!((net.total_loss(2.0, 100.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn pinn_total_loss_bc_only() {
        let net = PhysicsNeuralNet::new(&[1, 4, 1], 0.0, 4.0).expect("valid layer sizes");
        assert!((net.total_loss(100.0, 3.0) - 12.0).abs() < 1e-12);
    }

    // ── Initialisation ────────────────────────────────────────────────────

    #[test]
    fn xavier_layer_has_distinct_bounded_weights() {
        let mut rng = StdRng::seed_from_u64(1);
        let layer = NeuralLayer::xavier(6, 10, &mut rng);
        let limit = (6.0_f64 / 16.0).sqrt();
        assert_eq!(layer.weights.len(), 60);
        assert_eq!(layer.biases, vec![0.0; 10]);
        assert!(layer.weights.iter().all(|w| w.abs() <= limit));
        // The weights spread over the whole range and average out to ~0.
        let largest = layer.weights.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        assert!(largest > 0.7 * limit, "largest weight {largest}");
        let mean = layer.weights.iter().sum::<f64>() / 60.0;
        assert!(mean.abs() < 0.3 * limit, "mean weight {mean}");
        // Symmetry is broken: no two neurons (rows) are identical.
        let rows: Vec<&[f64]> = layer.weights.chunks(6).collect();
        for (a, row_a) in rows.iter().enumerate() {
            for row_b in rows.iter().skip(a + 1) {
                assert_ne!(row_a, row_b);
            }
        }
    }

    #[test]
    fn new_layer_is_deterministic_and_not_constant() {
        let a = NeuralLayer::new(4, 3);
        let b = NeuralLayer::new(4, 3);
        assert_eq!(a.weights, b.weights);
        // The old initialisation set every weight to 0.1.
        assert!(a.weights.windows(2).any(|w| w[0] != w[1]));
        assert!(a.weights.iter().any(|&w| (w - 0.1).abs() > 1e-3));
        // Different shapes draw from different streams.
        assert_ne!(NeuralLayer::new(3, 4).weights, a.weights);
        assert_eq!(a.biases, vec![0.0; 3]);
        // A degenerate shape does not panic.
        assert!(NeuralLayer::new(0, 0).weights.is_empty());
        assert!(NeuralLayer::new(0, 3).weights.is_empty());
        assert!(NeuralLayer::new(3, 0).weights.is_empty());
    }

    #[test]
    fn solver_seeds_control_the_initialisation() {
        let a = GpuNeuralSolver::with_seed(&[2, 5, 1], 0.1, 7).expect("valid");
        let b = GpuNeuralSolver::with_seed(&[2, 5, 1], 0.1, 7).expect("valid");
        let c = GpuNeuralSolver::with_seed(&[2, 5, 1], 0.1, 8).expect("valid");
        assert_eq!(a.layers[0].weights, b.layers[0].weights);
        assert_ne!(a.layers[0].weights, c.layers[0].weights);
        // `new` is `with_seed(DEFAULT_INIT_SEED)`.
        let d = GpuNeuralSolver::new(&[2, 5, 1], 0.1).expect("valid");
        let e = GpuNeuralSolver::with_seed(&[2, 5, 1], 0.1, DEFAULT_INIT_SEED).expect("valid");
        assert_eq!(d.layers[1].weights, e.layers[1].weights);
        // Neurons of a hidden layer respond differently to the same input.
        let out = a.layers[0].forward(&[1.0, -0.5]);
        for (i, x) in out.iter().enumerate() {
            for y in out.iter().skip(i + 1) {
                assert_ne!(x, y);
            }
        }
    }

    #[test]
    fn solver_rejects_invalid_configuration() {
        for sizes in [&[][..], &[3], &[3, 0, 2], &[0, 3], &[2, 3, 0]] {
            let err = GpuNeuralSolver::new(sizes, 1e-3).expect_err("invalid layer sizes");
            assert!(matches!(err, Error::InvalidConfig(_)), "{sizes:?}: {err:?}");
        }
        for lr in [-1e-3, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = GpuNeuralSolver::new(&[2, 2], lr).expect_err("invalid learning rate");
            assert!(matches!(err, Error::InvalidConfig(_)), "lr {lr}: {err:?}");
        }
        assert!(GpuNeuralSolver::new(&[2, 2], 0.0).is_ok());
        assert!(PhysicsNeuralNet::new(&[1], 1.0, 1.0).is_err());
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    #[test]
    fn hidden_layers_use_tanh_and_the_output_is_linear() {
        let mut net = GpuNeuralSolver::new(&[1, 2, 1], 1e-3).expect("valid");
        net.layers[0].weights = vec![0.5, -1.0];
        net.layers[0].biases = vec![0.1, 0.2];
        net.layers[1].weights = vec![2.0, 3.0];
        net.layers[1].biases = vec![-0.5];
        let x = 0.7_f64;
        let expected = 2.0 * (0.5 * x + 0.1).tanh() + 3.0 * (-x + 0.2).tanh() - 0.5;
        assert!((net.forward_pass(&[x])[0] - expected).abs() < 1e-14);
        // Large input: the hidden units saturate (ReLU would not), the output
        // stays bounded by the output weights.
        let huge = net.forward_pass(&[1.0e6])[0];
        assert!((huge - (2.0 + (-3.0) - 0.5)).abs() < 1e-9, "{huge}");
        assert_eq!(net.input_size(), 1);
        assert_eq!(net.output_size(), 1);
    }

    // ── Back-propagation ──────────────────────────────────────────────────

    /// `∂loss/∂parameter` by central differences, perturbing one weight or bias.
    fn numeric_mse_gradient(
        net: &GpuNeuralSolver,
        input: &[f64],
        target: &[f64],
        layer: usize,
        bias: bool,
        index: usize,
    ) -> f64 {
        let h = 1.0e-6;
        let loss_at = |delta: f64| {
            let mut perturbed = net.clone();
            let slot = if bias {
                &mut perturbed.layers[layer].biases[index]
            } else {
                &mut perturbed.layers[layer].weights[index]
            };
            *slot += delta;
            perturbed
                .mse_gradients(input, target)
                .expect("valid sample")
                .0
        };
        (loss_at(h) - loss_at(-h)) / (2.0 * h)
    }

    #[test]
    fn backprop_matches_finite_differences() {
        let net = GpuNeuralSolver::with_seed(&[3, 4, 3, 2], 0.1, 5).expect("valid");
        let input = [0.4, -1.1, 0.7];
        let target = [0.3, -0.6];
        let (loss, gradients) = net.mse_gradients(&input, &target).expect("valid sample");
        let output = net.predict(&input);
        assert!((loss - ns_mse_loss(&output, &target)).abs() < 1e-14);
        let mut checked = 0;
        for (layer, gradient) in gradients.iter().enumerate() {
            for (index, &analytic) in gradient.weights.iter().enumerate() {
                let numeric = numeric_mse_gradient(&net, &input, &target, layer, false, index);
                assert!(
                    (numeric - analytic).abs() < 1e-7 * (1.0 + analytic.abs()),
                    "weight {layer}/{index}: {numeric} vs {analytic}"
                );
                checked += 1;
            }
            for (index, &analytic) in gradient.biases.iter().enumerate() {
                let numeric = numeric_mse_gradient(&net, &input, &target, layer, true, index);
                assert!(
                    (numeric - analytic).abs() < 1e-7 * (1.0 + analytic.abs()),
                    "bias {layer}/{index}: {numeric} vs {analytic}"
                );
                checked += 1;
            }
        }
        // 3*4 + 4 + 4*3 + 3 + 3*2 + 2
        assert_eq!(checked, 39);
        // The gradients are not trivially zero.
        assert!(
            gradients
                .iter()
                .any(|g| g.weights.iter().any(|w| w.abs() > 1e-3))
        );
    }

    #[test]
    fn train_step_is_gradient_descent_on_the_squared_error() {
        // Single linear neuron: loss (w x + b - y)^2, d/dw = 2 e x, d/db = 2 e.
        let mut net = GpuNeuralSolver::new(&[1, 1], 0.1).expect("valid");
        net.layers[0].weights = vec![0.5];
        net.layers[0].biases = vec![-0.2];
        let (x, y) = (2.0, 1.5);
        let error = 0.5 * x - 0.2 - y; // -0.7
        let loss = net.train_step(&[x], &[y]).expect("train");
        assert!((loss - error * error).abs() < 1e-14, "pre-update loss");
        assert!((net.layers[0].weights[0] - (0.5 - 0.1 * 2.0 * error * x)).abs() < 1e-14);
        assert!((net.layers[0].biases[0] - (-0.2 - 0.1 * 2.0 * error)).abs() < 1e-14);
        // A zero learning rate only evaluates.
        net.learning_rate = 0.0;
        let before = net.layers[0].weights.clone();
        net.train_step(&[x], &[y]).expect("train");
        assert_eq!(net.layers[0].weights, before);
    }

    #[test]
    fn train_step_fits_a_nonlinear_function() {
        let mut net = GpuNeuralSolver::with_seed(&[1, 12, 1], 0.05, 4).expect("valid");
        let samples: Vec<(f64, f64)> = (0..21)
            .map(|i| {
                let x = -1.0 + 0.1 * i as f64;
                (x, (2.0 * x).sin())
            })
            .collect();
        let mse = |net: &GpuNeuralSolver| {
            let predicted: Vec<f64> = samples.iter().map(|&(x, _)| net.predict(&[x])[0]).collect();
            let wanted: Vec<f64> = samples.iter().map(|&(_, y)| y).collect();
            ns_mse_loss(&predicted, &wanted)
        };
        let before = mse(&net);
        for _ in 0..1500 {
            for &(x, y) in &samples {
                net.train_step(&[x], &[y]).expect("train");
            }
        }
        let after = mse(&net);
        assert!(before > 0.05, "untrained network error {before}");
        assert!(after < 1e-3, "trained error {after} (was {before})");
    }

    #[test]
    fn train_step_reports_dimension_errors_without_touching_the_weights() {
        let mut net = GpuNeuralSolver::new(&[3, 4, 2], 0.1).expect("valid");
        let before = net.layers.clone();
        let err = net
            .train_step(&[1.0, 2.0], &[0.0, 0.0])
            .expect_err("bad input");
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                what: "network input",
                expected: 3,
                got: 2
            }
        ));
        let err = net
            .train_step(&[1.0, 2.0, 3.0], &[0.0])
            .expect_err("bad target");
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                what: "training target",
                expected: 2,
                got: 1
            }
        ));
        net.learning_rate = f64::NAN;
        assert!(net.train_step(&[1.0, 2.0, 3.0], &[0.0, 0.0]).is_err());
        net.learning_rate = 0.1;
        // Broken layers are reported, not indexed out of bounds.
        net.layers[1].weights.pop();
        assert!(matches!(
            net.train_step(&[1.0, 2.0, 3.0], &[0.0, 0.0]),
            Err(Error::InvalidConfig(_))
        ));
        net.layers[1].weights.push(0.0);
        net.layers[0].n_out = 5;
        assert!(net.train_step(&[1.0, 2.0, 3.0], &[0.0, 0.0]).is_err());
        net.layers.clear();
        assert!(net.train_step(&[1.0], &[1.0]).is_err());
        assert!(
            before
                .iter()
                .zip(
                    GpuNeuralSolver::new(&[3, 4, 2], 0.1)
                        .expect("valid")
                        .layers
                        .iter()
                )
                .all(|(a, b)| a.weights == b.weights)
        );
    }

    // ── PINN: derivatives and loss ────────────────────────────────────────

    /// `u(x) = 0.8 tanh(1.5 x + 0.2) - 0.6 tanh(-0.7 x + 0.1) + 0.3`.
    fn known_network() -> PhysicsNeuralNet {
        let mut net = PhysicsNeuralNet::new(&[1, 2, 1], 1.0, 1.0).expect("valid");
        net.solver.layers[0].weights = vec![1.5, -0.7];
        net.solver.layers[0].biases = vec![0.2, 0.1];
        net.solver.layers[1].weights = vec![0.8, -0.6];
        net.solver.layers[1].biases = vec![0.3];
        net
    }

    #[test]
    fn second_derivative_matches_the_analytic_value() {
        // d²/dx² tanh(a x + b) = -2 a² t (1 - t²) with t = tanh(a x + b).
        let net = known_network();
        let term = |c: f64, a: f64, b: f64, x: f64| {
            let t = (a * x + b).tanh();
            -2.0 * c * a * a * t * (1.0 - t * t)
        };
        for k in 0..11 {
            let x = -1.0 + 0.2 * k as f64;
            let analytic = term(0.8, 1.5, 0.2, x) + term(-0.6, -0.7, 0.1, x);
            let numeric = net.second_derivative(x);
            assert!(
                (numeric - analytic).abs() < 1e-5,
                "x = {x}: {numeric} vs {analytic}"
            );
            // residual = -u_xx - f.
            let f = 0.37;
            assert!((net.residual_at(x, f) - (-numeric - f)).abs() < 1e-12);
        }
    }

    #[test]
    fn pinn_loss_is_the_weighted_pde_and_boundary_mean_squares() {
        let mut net = known_network();
        net.pde_weight = 2.0;
        net.bc_weight = 3.0;
        let collocation = [0.2, 0.5, 0.9];
        let source = [1.0, -2.0, 0.5];
        let boundary = [(0.0, 0.25), (1.0, -0.5)];
        let pde: f64 = collocation
            .iter()
            .zip(source.iter())
            .map(|(&x, &f)| net.residual_at(x, f).powi(2))
            .sum::<f64>()
            / 3.0;
        let bc: f64 = boundary
            .iter()
            .map(|&(x, g)| (net.predict(x) - g).powi(2))
            .sum::<f64>()
            / 2.0;
        let loss = net.loss(&collocation, &boundary, &source).expect("loss");
        assert!((loss - (2.0 * pde + 3.0 * bc)).abs() < 1e-9 * (1.0 + loss));
        // Only boundary data, or only interior data, is fine too.
        assert!(net.loss(&[], &boundary, &[]).is_ok());
        assert!(net.loss(&collocation, &[], &source).is_ok());
    }

    /// Analytic gradient of the PINN loss (back-propagation through the three
    /// finite-difference evaluations) against central differences of the loss.
    #[test]
    fn pinn_gradient_matches_finite_differences() {
        let mut net = PhysicsNeuralNet::with_seed(&[1, 5, 5, 1], 1.3, 2.1, 9).expect("valid");
        let collocation = [0.1, 0.3, 0.55, 0.8, 0.95];
        let source = [1.0, -0.5, 2.0, 0.3, -1.2];
        let boundary = [(0.0, 0.2), (1.0, -0.4)];
        let (loss, gradients) = net.loss_and_gradients(&collocation, &boundary, &source);
        let reported = net.loss(&collocation, &boundary, &source).expect("loss");
        assert!(
            (loss - reported).abs() < 1e-9 * (1.0 + loss),
            "{loss} vs {reported}"
        );

        let h = 1.0e-4;
        let mut checked = 0;
        for (layer, gradient) in gradients.iter().enumerate() {
            for (kind, values) in [("weight", &gradient.weights), ("bias", &gradient.biases)] {
                for (index, &analytic) in values.iter().enumerate() {
                    let mut loss_at = |delta: f64| {
                        let slot = if kind == "weight" {
                            &mut net.solver.layers[layer].weights[index]
                        } else {
                            &mut net.solver.layers[layer].biases[index]
                        };
                        *slot += delta;
                        let value = net.loss(&collocation, &boundary, &source).expect("loss");
                        let slot = if kind == "weight" {
                            &mut net.solver.layers[layer].weights[index]
                        } else {
                            &mut net.solver.layers[layer].biases[index]
                        };
                        *slot -= delta;
                        value
                    };
                    let numeric = (loss_at(h) - loss_at(-h)) / (2.0 * h);
                    assert!(
                        (numeric - analytic).abs() < 2e-4 * (1.0 + analytic.abs()),
                        "{kind} {layer}/{index}: {numeric} vs {analytic}"
                    );
                    checked += 1;
                }
            }
        }
        // 1*5 + 5 + 5*5 + 5 + 5*1 + 1
        assert_eq!(checked, 46);
    }

    // ── PINN: training ────────────────────────────────────────────────────

    /// `−u'' = π² sin(π x)`, `u(0) = u(1) = 0` has the solution `sin(π x)`.
    fn poisson_problem() -> (Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
        let pi = std::f64::consts::PI;
        let collocation: Vec<f64> = (1..=24).map(|i| i as f64 / 25.0).collect();
        let source = collocation
            .iter()
            .map(|x| pi * pi * (pi * x).sin())
            .collect();
        (collocation, source, vec![(0.0, 0.0), (1.0, 0.0)])
    }

    fn max_error_against_sin_pi_x(net: &PhysicsNeuralNet) -> f64 {
        let pi = std::f64::consts::PI;
        (0..=100)
            .map(|i| {
                let x = i as f64 / 100.0;
                (net.predict(x) - (pi * x).sin()).abs()
            })
            .fold(0.0, f64::max)
    }

    /// The PINN learns the solution of `−u'' = π² sin(π x)`: the maximum error
    /// against `sin(π x)` drops below 1e-2 (an untrained network is far off).
    #[test]
    fn pinn_solves_the_1d_poisson_problem() {
        let (collocation, source, boundary) = poisson_problem();
        for seed in [1_u64, 2, 3] {
            let mut net =
                PhysicsNeuralNet::with_seed(&[1, 16, 16, 1], 1.0, 10.0, seed).expect("valid");
            net.solver.learning_rate = 2e-3;
            let untrained = max_error_against_sin_pi_x(&net);
            let history = net
                .train(&collocation, &boundary, &source, 4000)
                .expect("train");
            let error = max_error_against_sin_pi_x(&net);
            assert!(untrained > 0.1, "seed {seed}: untrained error {untrained}");
            assert!(error < 1e-2, "seed {seed}: max error {error}");
            // The loss fell by several orders of magnitude and ended small.
            assert_eq!(history.len(), 4000);
            assert!(
                history[0] > 10.0,
                "seed {seed}: initial loss {}",
                history[0]
            );
            assert!(
                history[3999] < 0.05,
                "seed {seed}: final loss {}",
                history[3999]
            );
            // The reported final loss agrees with a fresh evaluation.
            let final_loss = net.loss(&collocation, &boundary, &source).expect("loss");
            assert!(final_loss <= history[3999] * 1.05 + 1e-12);
            // The residual at a point between collocation points is small too.
            let x = 0.5 + 0.5 / 25.0;
            let f = std::f64::consts::PI.powi(2) * (std::f64::consts::PI * x).sin();
            assert!(net.residual_at(x, f).abs() < 0.5, "seed {seed}");
        }
    }

    #[test]
    fn pinn_training_is_deterministic_and_loss_starts_at_the_initial_value() {
        let (collocation, source, boundary) = poisson_problem();
        let train = || {
            let mut net = PhysicsNeuralNet::with_seed(&[1, 8, 1], 1.0, 10.0, 21).expect("valid");
            let initial = net.loss(&collocation, &boundary, &source).expect("loss");
            let history = net
                .train(&collocation, &boundary, &source, 30)
                .expect("train");
            (initial, history, net.solver.layers[0].weights.clone())
        };
        let (initial_a, history_a, weights_a) = train();
        let (initial_b, history_b, weights_b) = train();
        assert!((history_a[0] - initial_a).abs() < 1e-9 * (1.0 + initial_a));
        assert_eq!(history_a, history_b);
        assert_eq!(weights_a, weights_b);
        assert_eq!(initial_a, initial_b);
        // Gradient descent with a small step lowers the loss.
        assert!(history_a[29] < history_a[0]);
    }

    #[test]
    fn pinn_zero_epochs_and_zero_learning_rate_leave_the_network_alone() {
        let (collocation, source, boundary) = poisson_problem();
        let mut net = PhysicsNeuralNet::with_seed(&[1, 4, 1], 1.0, 1.0, 2).expect("valid");
        let before = net.solver.layers[0].weights.clone();
        assert!(
            net.train(&collocation, &boundary, &source, 0)
                .expect("train")
                .is_empty()
        );
        assert_eq!(net.solver.layers[0].weights, before);
        net.solver.learning_rate = 0.0;
        let history = net
            .train(&collocation, &boundary, &source, 3)
            .expect("train");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0], history[2]);
        assert_eq!(net.solver.layers[0].weights, before);
    }

    #[test]
    fn pinn_train_validates_its_inputs() {
        let (collocation, source, boundary) = poisson_problem();
        let mut net = PhysicsNeuralNet::with_seed(&[1, 4, 1], 1.0, 1.0, 2).expect("valid");
        // Source terms must match the collocation points.
        assert!(matches!(
            net.train(&collocation, &boundary, &source[..5], 1),
            Err(Error::DimensionMismatch { .. })
        ));
        // Something to train on.
        assert!(matches!(
            net.train(&[], &[], &[], 1),
            Err(Error::InvalidConfig(_))
        ));
        // Finite data and weights.
        let mut bad = source.clone();
        bad[3] = f64::NAN;
        assert!(matches!(
            net.train(&collocation, &boundary, &bad, 1),
            Err(Error::InvalidConfig(_))
        ));
        assert!(
            net.train(&collocation, &[(0.0, f64::INFINITY)], &source, 1)
                .is_err()
        );
        let mut x = collocation.clone();
        x[0] = f64::NAN;
        assert!(net.train(&x, &boundary, &source, 1).is_err());
        net.pde_weight = f64::NAN;
        assert!(net.train(&collocation, &boundary, &source, 1).is_err());
        net.pde_weight = 1.0;
        net.solver.learning_rate = -1.0;
        assert!(net.train(&collocation, &boundary, &source, 1).is_err());
        net.solver.learning_rate = 1e-3;
        // The 1-D trainer needs a scalar-to-scalar network.
        let mut wide = PhysicsNeuralNet::new(&[2, 4, 1], 1.0, 1.0).expect("valid");
        assert!(matches!(
            wide.train(&collocation, &boundary, &source, 1),
            Err(Error::InvalidConfig(_))
        ));
        assert!(wide.loss(&collocation, &boundary, &source).is_err());
        // None of that changed the (valid) network.
        assert!(net.train(&collocation, &boundary, &source, 2).is_ok());
    }

    /// A far too large learning rate blows the loss up: the trainer reports it
    /// and rolls back to the last weights with a finite loss.
    #[test]
    fn pinn_divergence_is_reported_and_rolled_back() {
        let (collocation, source, boundary) = poisson_problem();
        let mut net = PhysicsNeuralNet::with_seed(&[1, 8, 1], 1.0, 10.0, 6).expect("valid");
        net.solver.learning_rate = 1.0e6;
        let err = net
            .train(&collocation, &boundary, &source, 400)
            .expect_err("training must diverge");
        assert!(matches!(err, Error::General(_)), "{err:?}");
        let finite = net.solver.layers.iter().all(|l| {
            l.weights
                .iter()
                .chain(l.biases.iter())
                .all(|v| v.is_finite())
        });
        assert!(finite, "the weights were rolled back to finite values");
        let loss = net.loss(&collocation, &boundary, &source).expect("loss");
        assert!(loss.is_finite());
    }
}
