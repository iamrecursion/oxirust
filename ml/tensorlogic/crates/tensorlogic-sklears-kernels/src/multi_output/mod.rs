//! Multi-output / vector-valued kernels.
//!
//! A `MultiOutputKernel` maps two feature vectors to a `p×p` covariance block
//! `K(x, x') ∈ R^{p×p}`.  This enables vector-valued Gaussian Process
//! regression where `p` outputs are jointly modelled with shared correlation
//! structure.
//!
//! ## Key types
//!
//! - [`MultiOutputKernel`] — the trait ([`trait_def`])
//! - [`KroneckerICMKernel`] — ICM (`B ⊗ k`) multi-output kernel ([`icm`])
//! - [`KroneckerLMCKernel`] — LMC (`Σ_q B_q ⊗ k_q`) multi-output kernel ([`lmc`])
//! - [`VvgpModel`] / [`VvgpFitted`] — vector-valued GP inference ([`vvgp`])
//!
//! ## Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use tensorlogic_sklears_kernels::{
//!     multi_output::{KroneckerICMKernel, MultiOutputKernel, VvgpModel},
//!     RbfKernel, RbfKernelConfig,
//! };
//!
//! // 2-output ICM kernel with an RBF base.
//! let base = Box::new(RbfKernel::new(RbfKernelConfig::new(1.0)).expect("valid"));
//! let covariance = vec![vec![2.0, 0.5], vec![0.5, 1.5]];
//! let icm = KroneckerICMKernel::from_base(base, covariance).expect("valid ICM");
//!
//! // Compute a single 2×2 block.
//! let block = icm.compute_block(&[0.0_f64], &[1.0_f64]).expect("block");
//! assert_eq!(block.shape(), &[2, 2]);
//!
//! // Fit a vector-valued GP (3 training points, 2 outputs each).
//! let inputs = vec![vec![0.0_f64], vec![1.0], vec![2.0]];
//! let targets = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
//! let model = VvgpModel::new(Arc::new(icm), 1e-4).expect("valid noise");
//! let fitted = model.fit(&inputs, &targets).expect("fit");
//!
//! let (mean, cov) = fitted.predict(&[1.5_f64]).expect("predict");
//! assert_eq!(mean.len(), 2);
//! assert_eq!(cov.shape(), &[2, 2]);
//! ```

pub mod icm;
pub mod lmc;
pub mod trait_def;
pub mod vvgp;

#[cfg(test)]
mod tests;

pub use icm::KroneckerICMKernel;
pub use lmc::KroneckerLMCKernel;
pub use trait_def::MultiOutputKernel;
pub use vvgp::{VvgpFitted, VvgpModel};
