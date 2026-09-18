//! Variational Message Passing (VMP) for conjugate exponential families.
//!
//! This module implements the algorithm of Winn & Bishop (2005) for the three
//! families shipped in the v0.2.0 research preview — Gaussian (mean-unknown,
//! precision-known), Categorical, and Dirichlet — along with their conjugate
//! factor relationships (Gaussian observation, Gaussian step, Dirichlet-Categorical,
//! Categorical observation).
//!
//! All updates happen in natural-parameter space. The engine drives a
//! coordinate-ascent loop that monotonically increases the evidence lower bound
//! (ELBO) until either |ΔELBO| or the L∞ residual of the natural-parameter
//! vectors falls below the configured tolerance. Divergence (an ELBO decrease
//! beyond `divergence_tolerance`) surfaces as a [`crate::PgmError::ConvergenceFailure`]
//! so a numerically broken run never silently returns a garbage posterior.
//!
//! # Module map
//!
//! | Submodule             | Purpose                                                    |
//! |-----------------------|------------------------------------------------------------|
//! | [`exponential_family`]| Trait contract every VMP-compatible distribution satisfies |
//! | [`distributions`]     | `GaussianNP`, `CategoricalNP`, `DirichletNP` + KL helpers  |
//! | [`messages`]          | `VmpMessage` / `MessageDirection` primitives               |
//! | [`engine`]            | `VmpConfig` + `VariationalMessagePassing` coordinate engine|
//! | [`special`]           | Local `ln_gamma` / `digamma` (scirs2-core free)            |
//!
//! # Minimal example
//!
//! ```
//! use tensorlogic_quantrs_hooks::vmp::{
//!     VariationalMessagePassing, VmpConfig, VmpFactor,
//! };
//!
//! // y ~ N(μ, 1) with one observation y = 3, prior μ ~ N(0, 1).
//! let config = VmpConfig::new()
//!     .with_gaussian("mu", 0.0, 1.0).expect("register mu")
//!     .with_factor(VmpFactor::GaussianObservation {
//!         target: "mu".to_string(),
//!         observation: 3.0,
//!         precision: 1.0,
//!     })
//!     .with_limits(50, 1e-8);
//!
//! let mut engine = VariationalMessagePassing::new(config).expect("engine");
//! let result = engine.run().expect("run");
//! assert!(result.converged);
//! ```
//!
//! # References
//!
//! - Winn, J. M. & Bishop, C. M. (2005). *Variational Message Passing*.
//!   Journal of Machine Learning Research 6, 661-694.

pub mod beta;
pub mod distributions;
pub mod engine;
pub mod exponential_family;
pub mod gamma;
pub mod messages;
pub mod mixture;
pub mod special;

#[cfg(test)]
mod tests;

pub use beta::{BetaBernoulliObservation, BetaNP};
pub use distributions::{
    categorical_kl, dirichlet_kl, gaussian_kl, gaussian_kl_fixed_precision, CategoricalNP,
    DirichletNP, GaussianNP,
};
pub use engine::{
    Family, VariationalMessagePassing, VariationalState, VmpConfig, VmpFactor, VmpResult,
};
pub use exponential_family::ExponentialFamily;
pub use gamma::{GammaNP, GammaPoissonObservation};
pub use messages::{MessageDirection, VmpMessage};
pub use mixture::{VariationalGaussianMixture, VgmmConfig, VgmmResult};
