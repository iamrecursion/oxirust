//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use std::fmt::Debug;

/// Exponential smoothing parameters
#[derive(Debug, Clone)]
pub struct ExpSmoothingParams {
    /// Level smoothing parameter (alpha)
    pub alpha: f64,
    /// Trend smoothing parameter (beta)
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter (gamma)
    pub gamma: Option<f64>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Whether to use multiplicative trend
    pub multiplicative_trend: bool,
    /// Whether to use multiplicative seasonality
    pub multiplicative_seasonality: bool,
    /// Whether to damp trend
    pub damped_trend: bool,
    /// Damping factor
    pub phi: Option<f64>,
}
/// Options for automatic ARIMA model selection
#[derive(Debug, Clone)]
pub struct AutoArimaOptions {
    /// Maximum AR order (p) to consider
    pub max_p: usize,
    /// Maximum differencing order (d) to consider
    pub max_d: usize,
    /// Maximum MA order (q) to consider
    pub max_q: usize,
    /// Whether to include seasonal components
    pub seasonal: bool,
    /// Seasonal period (required if seasonal is true)
    pub seasonal_period: Option<usize>,
    /// Maximum seasonal AR order (P) to consider
    pub max_seasonal_p: usize,
    /// Maximum seasonal differencing order (D) to consider
    pub max_seasonal_d: usize,
    /// Maximum seasonal MA order (Q) to consider
    pub max_seasonal_q: usize,
    /// Whether to automatically determine differencing order
    pub auto_diff: bool,
    /// Whether to estimate constant/drift term
    pub with_constant: bool,
    /// Information criterion to use for model selection (AIC or BIC)
    pub information_criterion: String,
    /// Number of steps for out-of-sample cross-validation
    pub stepwise: bool,
    /// Maximum total parameters to consider (to avoid overfitting)
    pub max_order: usize,
}
/// Result of time series forecasting
#[derive(Debug, Clone)]
pub struct ForecastResult<F> {
    /// Point forecasts
    pub forecast: Array1<F>,
    /// Lower confidence interval
    pub lower_ci: Array1<F>,
    /// Upper confidence interval
    pub upper_ci: Array1<F>,
}
/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ArimaParams {
    /// Autoregressive order (p)
    pub p: usize,
    /// Integration order (d)
    pub d: usize,
    /// Moving average order (q)
    pub q: usize,
    /// Seasonal order (P)
    pub seasonal_p: Option<usize>,
    /// Seasonal integration order (D)
    pub seasonal_d: Option<usize>,
    /// Seasonal moving average order (Q)
    pub seasonal_q: Option<usize>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
    /// Fit intercept
    pub fit_intercept: bool,
    /// Trend component
    pub trend: Option<String>,
}
/// Model fit metrics
#[derive(Debug, Clone)]
pub(super) struct ModelFitMetrics<F> {
    /// Akaike Information Criterion (AIC)
    pub(super) aic: F,
    /// Bayesian Information Criterion (BIC)
    pub(super) bic: F,
    /// Hannan-Quinn Information Criterion (HQIC)
    #[allow(dead_code)]
    pub(super) hqic: F,
    /// Log-likelihood
    #[allow(dead_code)]
    pub(super) log_likelihood: F,
    /// Mean Squared Error
    #[allow(dead_code)]
    pub(super) mse: F,
}
