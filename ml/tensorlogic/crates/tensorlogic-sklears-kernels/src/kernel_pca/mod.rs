//! Kernel Principal Component Analysis (Scholkopf, Smola & Muller, 1998).
//!
//! This module provides a fully self-contained implementation of Kernel PCA
//! that operates through the crate's [`Kernel`](crate::types::Kernel) trait.
//! Any kernel in the crate (RBF, Linear, Polynomial, Symbolic, etc.) can be
//! plugged in; the only additional requirement for fitting is `Clone + 'static`,
//! which every shipped kernel satisfies.
//!
//! # Module map
//!
//! | Submodule        | Purpose                                                     |
//! |------------------|-------------------------------------------------------------|
//! | [`centering`]    | Double-centering of the Gram matrix + test-time centering   |
//! | [`eigendecomp`]  | Wrapper around `scirs2_linalg::eigh` for top-k eigenpairs   |
//! | [`error`]        | `KernelPcaError` / `KernelPcaResult` types                  |
//! | [`model`]        | `KernelPCA`, `KernelPcaConfig`, `FittedKernelPCA`           |
//!
//! # Minimal example
//!
//! ```rust,no_run
//! use tensorlogic_sklears_kernels::kernel_pca::{KernelPCA, KernelPcaConfig};
//! use tensorlogic_sklears_kernels::RbfKernel;
//! use tensorlogic_sklears_kernels::RbfKernelConfig;
//!
//! let kernel = RbfKernel::new(RbfKernelConfig::new(1.0)).expect("kernel");
//! let config = KernelPcaConfig::new(2);
//! let model = KernelPCA::build(kernel, config).expect("model");
//!
//! let data = vec![
//!     vec![1.0, 2.0],
//!     vec![3.0, 4.0],
//!     vec![5.0, 6.0],
//! ];
//! let fitted = model.fit(&data).expect("fit");
//! let embedding = fitted.transform(&data).expect("transform");
//! assert_eq!(embedding.nrows(), 3);
//! assert_eq!(embedding.ncols(), 2);
//! ```
//!
//! # References
//!
//! - Scholkopf, B., Smola, A. & Muller, K.-R. (1998). *Nonlinear
//!   Component Analysis as a Kernel Eigenvalue Problem*. Neural
//!   Computation 10(5), 1299--1319.

pub mod centering;
pub mod eigendecomp;
pub mod error;
pub mod model;

#[cfg(test)]
mod tests;

pub use centering::{center_test_kernel, double_center, KernelCenteringStats};
pub use eigendecomp::{symmetric_eigendecomp, TopKEigen};
pub use error::{KernelPcaError, KernelPcaResult};
pub use model::{FittedKernelPCA, KernelPCA, KernelPcaConfig};
