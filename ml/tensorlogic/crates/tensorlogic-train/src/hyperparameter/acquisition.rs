//! Acquisition functions for Bayesian Optimization.

/// Acquisition function type for Bayesian Optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquisitionFunction {
    /// Expected Improvement - balances exploration and exploitation.
    ExpectedImprovement { xi: f64 },
    /// Upper Confidence Bound - uses uncertainty to guide exploration.
    UpperConfidenceBound { kappa: f64 },
    /// Probability of Improvement - probability of improving over best.
    ProbabilityOfImprovement { xi: f64 },
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        Self::ExpectedImprovement { xi: 0.01 }
    }
}
