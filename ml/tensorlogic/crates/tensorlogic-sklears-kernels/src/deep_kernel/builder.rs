//! Fluent builder for common Deep Kernel topologies.
//!
//! Mirrors the style of
//! [`crate::learned_composition::LearnedMixtureBuilder`]: chained
//! setters plus a terminating [`DeepKernelBuilder::build`] that returns
//! `Result<DeepKernel<MLPFeatureExtractor, K>>`.
//!
//! Typical usage:
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{
//!     deep_kernel::{Activation, DeepKernelBuilder},
//!     RbfKernel, RbfKernelConfig,
//! };
//!
//! let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
//! let dkl = DeepKernelBuilder::new()
//!     .input_dim(4)
//!     .hidden_layer(8, Activation::ReLU)
//!     .hidden_layer(4, Activation::Tanh)
//!     .output_dim(2, Activation::Identity)
//!     .seed(42)
//!     .build(rbf)
//!     .expect("valid topology");
//! let _ = dkl;
//! ```

use crate::deep_kernel::feature_extractor::MLPFeatureExtractor;
use crate::deep_kernel::kernel::DeepKernel;
use crate::deep_kernel::layer::Activation;
use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Fluent builder for Deep Kernel networks.
///
/// The builder records layer widths and activations in order. The first
/// width is the input dimension (set via [`Self::input_dim`]). Every
/// subsequent hidden layer appends a new width and its activation. The
/// terminating [`Self::output_dim`] records the final width and the
/// output-layer activation.
#[derive(Clone, Debug, Default)]
pub struct DeepKernelBuilder {
    widths: Vec<usize>,
    activations: Vec<Activation>,
    seed: Option<u64>,
    has_output: bool,
}

impl DeepKernelBuilder {
    /// Empty builder — no layers configured yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input dimension. Must be called exactly once before the
    /// first hidden or output layer.
    pub fn input_dim(mut self, dim: usize) -> Self {
        if self.widths.is_empty() {
            self.widths.push(dim);
        } else {
            self.widths[0] = dim;
        }
        self
    }

    /// Append a hidden layer with the given width and activation.
    /// Ignored if the builder has already been closed with
    /// [`Self::output_dim`] — the builder reports that as an error at
    /// build time via
    /// [`KernelError::InvalidParameter`].
    pub fn hidden_layer(mut self, width: usize, activation: Activation) -> Self {
        if !self.has_output {
            self.widths.push(width);
            self.activations.push(activation);
        }
        self
    }

    /// Finalise the topology by appending the output layer. Must be
    /// called exactly once.
    pub fn output_dim(mut self, width: usize, activation: Activation) -> Self {
        if !self.has_output {
            self.widths.push(width);
            self.activations.push(activation);
            self.has_output = true;
        }
        self
    }

    /// Set the RNG seed used for Xavier initialisation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Produce an owned [`MLPFeatureExtractor`] for the configured
    /// topology, without a base kernel. Useful when the caller wants to
    /// combine the MLP with a kernel that needs additional plumbing.
    pub fn build_extractor(&self) -> Result<MLPFeatureExtractor> {
        if self.widths.len() < 2 {
            return Err(KernelError::InvalidParameter {
                parameter: "widths".to_string(),
                value: format!("{:?}", self.widths),
                reason: "builder needs at least input_dim + output_dim".to_string(),
            });
        }
        if !self.has_output {
            return Err(KernelError::InvalidParameter {
                parameter: "output_dim".to_string(),
                value: "unset".to_string(),
                reason: "call output_dim before build".to_string(),
            });
        }
        let seed = self.seed.unwrap_or(0);
        MLPFeatureExtractor::xavier_init(&self.widths, &self.activations, seed)
    }

    /// Finalise the builder against a base kernel and produce a fully
    /// wired [`DeepKernel`].
    pub fn build<K: Kernel>(self, base: K) -> Result<DeepKernel<MLPFeatureExtractor, K>> {
        let extractor = self.build_extractor()?;
        Ok(DeepKernel::new(extractor, base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RbfKernelConfig;
    use crate::RbfKernel;

    #[test]
    fn builder_assembles_three_layer_mlp() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let dkl = DeepKernelBuilder::new()
            .input_dim(3)
            .hidden_layer(5, Activation::ReLU)
            .output_dim(2, Activation::Identity)
            .seed(123)
            .build(rbf)
            .expect("valid build");
        assert_eq!(dkl.feature_extractor().num_layers(), 2);
    }

    #[test]
    fn builder_fails_without_output_dim() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let result = DeepKernelBuilder::new()
            .input_dim(3)
            .hidden_layer(5, Activation::ReLU)
            .build(rbf);
        match result {
            Ok(_) => panic!("missing output_dim must fail"),
            Err(KernelError::InvalidParameter { .. }) => {}
            Err(other) => panic!("unexpected error variant: {}", other),
        }
    }

    #[test]
    fn builder_fails_when_only_input_set() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid");
        let result = DeepKernelBuilder::new().input_dim(3).build(rbf);
        match result {
            Ok(_) => panic!("only input_dim set must fail"),
            Err(KernelError::InvalidParameter { .. }) => {}
            Err(other) => panic!("unexpected error variant: {}", other),
        }
    }
}
