//! Deep Kernel Learning (DKL).
//!
//! This module implements the Deep Kernel Learning architecture of
//! Wilson et al. (2016). A DKL wraps a classical base kernel
//! `K_base(·, ·)` with a differentiable feature extractor `g_θ` and
//! evaluates
//!
//! ```text
//! K_DKL(x, y) = K_base(g_θ(x), g_θ(y)).
//! ```
//!
//! In the v0.2.0 research preview we ship a single reference feature
//! extractor — an MLP with Xavier/Glorot-normal initialisation (via
//! SciRS2-Core's seeded RNG) and support for ReLU / Tanh / Identity
//! activations. The generic [`DeepKernel`] is parameterised by the
//! extractor and the base kernel, so any other kernel in this crate
//! (RBF, Linear, Matern, …) plugs in without modification.
//!
//! # Relationship to `learned_composition`
//!
//! This module is the "nonlinear feature composition" counterpart of
//! [`crate::learned_composition`]. Where
//! [`crate::learned_composition::LearnedMixtureKernel`] learns a
//! softmax-weighted mixture *over* a library of kernels,
//! [`DeepKernel`] learns a nonlinear feature map that *transforms*
//! inputs before a single base kernel is applied. The two modules are
//! intended to be used together for expressive, trainable similarity
//! metrics.
//!
//! # Module layout
//!
//! * [`layer`] — [`DenseLayer`] / [`Activation`] primitives.
//! * [`feature_extractor`] — [`NeuralFeatureMap`] trait and
//!   [`MLPFeatureExtractor`] reference implementation.
//! * [`kernel`] — the [`DeepKernel`] wrapper that composes a feature
//!   map with a base kernel and implements [`crate::Kernel`].
//! * [`gradient`] — finite-difference verification and an analytical
//!   gradient path for the RBF-base case.
//! * [`builder`] — fluent [`DeepKernelBuilder`] for common MLP
//!   topologies.
//!
//! # Gradient semantics
//!
//! * **Analytical**: the closed form `∂K_DKL/∂θ` for the
//!   MLP-extractor + RBF-base case is available via
//!   [`gradient::rbf_dkl_gradient`] — one forward+backward pass, no
//!   autodiff.
//! * **Numerical**: every `DeepKernel<MLPFeatureExtractor, K>` supports
//!   [`gradient::finite_difference_gradient`] as a correctness check or
//!   as a stand-in for base kernels whose analytical chain rule has not
//!   yet been derived.
//! * **Base-kernel hyperparameters**: gradients w.r.t. e.g. the RBF
//!   `γ` are **not** produced here — callers should go through
//!   [`crate::tensor_kernels::RbfKernel::compute_with_gradient`] or a
//!   future autodiff layer.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{
//!     deep_kernel::{Activation, DeepKernelBuilder},
//!     Kernel, RbfKernel, RbfKernelConfig,
//! };
//!
//! let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid gamma");
//! let dkl = DeepKernelBuilder::new()
//!     .input_dim(2)
//!     .hidden_layer(4, Activation::Tanh)
//!     .output_dim(2, Activation::Identity)
//!     .seed(42)
//!     .build(rbf)
//!     .expect("valid topology");
//!
//! let x = vec![0.1, -0.2];
//! let y = vec![0.3, 0.4];
//! let _value = dkl.compute(&x, &y).expect("dkl value");
//! ```

pub mod builder;
pub mod feature_extractor;
pub mod gradient;
pub mod kernel;
pub mod layer;

#[cfg(test)]
mod tests;

pub use builder::DeepKernelBuilder;
pub use feature_extractor::{ForwardCache, LayerCache, MLPFeatureExtractor, NeuralFeatureMap};
pub use gradient::{finite_difference_gradient, rbf_dkl_gradient};
pub use kernel::{DeepKernel, DeepKernelSummary, FeatureMapShape};
pub use layer::{Activation, DenseLayer};
