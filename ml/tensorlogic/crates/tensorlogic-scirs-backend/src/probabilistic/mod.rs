//! Probabilistic execution sub-system for TensorLogic.
//!
//! This module provides Monte Carlo sampling, uncertainty quantification, and
//! mean-field variational inference, enabling probabilistic reasoning and
//! Bayesian deep learning within the SciRS2-backed executor.
//!
//! ## Sub-modules
//!
//! - [`sampling`] — Seeded Monte Carlo samplers (Bernoulli, Uniform, Normal,
//!   Categorical via Gumbel-max, MC integration)
//! - [`uncertainty`] — Uncertainty estimation over ensembles/samples (credible
//!   intervals, predictive entropy, BALD epistemic uncertainty)
//! - [`variational`] — Mean-field Gaussian variational inference with Adam optimiser

pub mod sampling;
pub mod uncertainty;
pub mod variational;

pub use sampling::{
    mc_integrate, sample_bernoulli, sample_categorical, sample_normal, sample_uniform,
    MonteCarloConfig,
};
pub use uncertainty::{
    bald_epistemic_uncertainty, predictive_entropy, MonteCarloEstimator, UncertaintyEstimate,
};
pub use variational::{MeanFieldGaussian, VariationalConfig, VariationalInference};
