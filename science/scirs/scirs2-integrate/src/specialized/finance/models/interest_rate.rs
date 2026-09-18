//! Interest rate models for financial derivatives

use crate::error::IntegrateResult as Result;

/// Trait for interest rate models
pub trait InterestRateModel: Send + Sync {
    /// Calculate the short rate at time t given the current state
    fn short_rate(&self, t: f64, state: &[f64]) -> f64;

    /// Calculate drift for the interest rate process
    fn drift(&self, t: f64, state: &[f64]) -> Vec<f64>;

    /// Calculate diffusion for the interest rate process
    fn diffusion(&self, t: f64, state: &[f64]) -> Vec<f64>;

    /// Number of state variables
    fn state_dimension(&self) -> usize;
}

/// Hull-White one-factor model
#[derive(Debug, Clone)]
pub struct HullWhiteModel {
    /// Mean reversion speed
    pub a: f64,
    /// Long-term mean level
    pub b: f64,
    /// Volatility
    pub sigma: f64,
}

impl InterestRateModel for HullWhiteModel {
    fn short_rate(&self, t: f64, state: &[f64]) -> f64 {
        state[0]
    }

    fn drift(&self, t: f64, state: &[f64]) -> Vec<f64> {
        vec![self.a * (self.b - state[0])]
    }

    fn diffusion(&self, _t: f64, state: &[f64]) -> Vec<f64> {
        vec![self.sigma]
    }

    fn state_dimension(&self) -> usize {
        1
    }
}

/// Cox-Ingersoll-Ross (CIR) model
#[derive(Debug, Clone)]
pub struct CIRModel {
    /// Mean reversion speed
    pub kappa: f64,
    /// Long-term mean
    pub theta: f64,
    /// Volatility
    pub sigma: f64,
}

impl InterestRateModel for CIRModel {
    fn short_rate(&self, t: f64, state: &[f64]) -> f64 {
        state[0].max(0.0)
    }

    fn drift(&self, t: f64, state: &[f64]) -> Vec<f64> {
        vec![self.kappa * (self.theta - state[0].max(0.0))]
    }

    fn diffusion(&self, t: f64, state: &[f64]) -> Vec<f64> {
        vec![self.sigma * state[0].max(0.0).sqrt()]
    }

    fn state_dimension(&self) -> usize {
        1
    }
}
