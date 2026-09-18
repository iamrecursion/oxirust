//! Thin trainable-parameter adapter around [`LearnedMixtureKernel`].
//!
//! Exposes the mixture as a parameter container compatible with
//! `tensorlogic-train`:
//!
//! * `parameters()` / `parameters_mut()` access the logits.
//! * `step(gradient, learning_rate)` applies a vanilla gradient-descent
//!   update to the logits.
//!
//! The adapter intentionally does not own an optimizer — choice of
//! optimizer (SGD, Adam, etc.) stays with the caller. It just bundles the
//! forward evaluation and the analytical gradient so the caller can write
//!
//! ```text
//! let (k, g) = mixture.evaluate_with_gradient(x, y)?;
//! let grad = dloss_dk * g;                       // scale by upstream grad
//! mixture.step(&grad, learning_rate)?;           // vanilla SGD step
//! ```

use crate::error::Result;
use crate::learned_composition::mixture::LearnedMixtureKernel;

/// Trainable adapter around a [`LearnedMixtureKernel`].
#[derive(Clone, Debug)]
pub struct TrainableKernelMixture {
    inner: LearnedMixtureKernel,
}

impl TrainableKernelMixture {
    /// Wrap an existing mixture kernel.
    pub fn new(inner: LearnedMixtureKernel) -> Self {
        Self { inner }
    }

    /// Number of trainable logits.
    pub fn num_parameters(&self) -> usize {
        self.inner.num_kernels()
    }

    /// Borrow the trainable parameters (logits).
    pub fn parameters(&self) -> &[f64] {
        self.inner.logits()
    }

    /// Mixture weights after softmax.
    pub fn weights(&self) -> Vec<f64> {
        self.inner.weights()
    }

    /// Forward pass — mixture kernel value.
    pub fn evaluate(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.inner.evaluate(x, y)
    }

    /// Forward + gradient in a single pass.
    pub fn evaluate_with_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.inner.evaluate_with_gradient(x, y)
    }

    /// Pure gradient (no forward re-use).
    pub fn gradient(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
        self.inner.gradient_wrt_logits(x, y)
    }

    /// Apply a vanilla gradient-descent step `w <- w - lr * g`.
    pub fn step(&mut self, gradient: &[f64], learning_rate: f64) -> Result<()> {
        self.inner.apply_gradient_step(gradient, learning_rate)
    }

    /// Replace the logits outright (useful for optimizer-driven updates).
    pub fn set_parameters(&mut self, new_logits: Vec<f64>) -> Result<()> {
        self.inner.set_logits(new_logits)
    }

    /// Borrow the underlying mixture kernel (read-only).
    pub fn inner(&self) -> &LearnedMixtureKernel {
        &self.inner
    }

    /// Consume the adapter and return the underlying mixture kernel.
    pub fn into_inner(self) -> LearnedMixtureKernel {
        self.inner
    }
}

impl From<LearnedMixtureKernel> for TrainableKernelMixture {
    fn from(inner: LearnedMixtureKernel) -> Self {
        Self::new(inner)
    }
}
