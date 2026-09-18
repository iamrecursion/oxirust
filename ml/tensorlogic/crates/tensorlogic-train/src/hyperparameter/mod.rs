//! Hyperparameter optimization utilities.
//!
//! Provides grid search, random search, and Bayesian optimization for
//! tuning model hyperparameters.

mod acquisition;
mod bayesian;
mod gp;
mod kernel;
mod search;
mod space;
mod value;

#[cfg(test)]
mod tests;

pub use acquisition::AcquisitionFunction;
pub use bayesian::BayesianOptimization;
pub use gp::GaussianProcess;
pub use kernel::GpKernel;
pub use search::{GridSearch, RandomSearch};
pub use space::HyperparamSpace;
pub use value::{HyperparamConfig, HyperparamResult, HyperparamValue};
