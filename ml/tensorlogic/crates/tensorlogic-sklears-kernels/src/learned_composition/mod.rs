//! Learned (differentiable) kernel composition.
//!
//! This module provides a differentiable mixture over a library of base
//! kernels. Given a library `{K_1, ..., K_n}` and a vector of trainable
//! logits `w = [w_1, ..., w_n]`, the mixture kernel is defined as
//!
//! ```text
//! p = softmax(w),
//! K_mix(x, y) = sum_i p_i * K_i(x, y).
//! ```
//!
//! The mixture is gradient-aware with respect to the logits. Using the
//! Jacobian of softmax `d p_j / d w_i = p_j * (delta_{ij} - p_i)`, the
//! gradient of the mixture with respect to logit `w_i` reduces to the
//! numerically clean identity
//!
//! ```text
//! d K_mix / d w_i = p_i * (K_i - K_mix).
//! ```
//!
//! This identity is implemented exactly in [`LearnedMixtureKernel::gradient_wrt_logits`]
//! and unit-tested against a finite-difference reference.
//!
//! # Scope (v0.2.0 preview)
//!
//! * Differentiability is w.r.t. mixture logits only — base kernel
//!   hyperparameters are treated as frozen.
//! * Weights are the softmax of logits; they are always strictly positive
//!   and sum to 1, so the mixture is PSD whenever every base kernel is PSD.
//! * The [`TrainableKernelMixture`] adapter exposes a single gradient step
//!   (take a gradient, update logits) — the optimizer loop lives with the
//!   caller (`tensorlogic-train`).
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//! use tensorlogic_sklears_kernels::{
//!     learned_composition::{LearnedMixtureBuilder, LearnedMixtureKernel},
//!     LinearKernel, RbfKernel, RbfKernelConfig,
//! };
//!
//! let kernel: LearnedMixtureKernel = LearnedMixtureBuilder::new()
//!     .push_kernel(Arc::new(LinearKernel::new()))
//!     .push_kernel(Arc::new(
//!         RbfKernel::new(RbfKernelConfig::new(0.5)).expect("valid gamma"),
//!     ))
//!     .build()
//!     .expect("non-empty library");
//!
//! let x = vec![1.0, 2.0];
//! let y = vec![0.5, 1.0];
//! let _value = kernel.evaluate(&x, &y).expect("compatible dims");
//! ```

pub mod builder;
pub mod mixture;
pub mod trainable;

#[cfg(test)]
mod tests;

pub use builder::LearnedMixtureBuilder;
pub use mixture::LearnedMixtureKernel;
pub use trainable::TrainableKernelMixture;
