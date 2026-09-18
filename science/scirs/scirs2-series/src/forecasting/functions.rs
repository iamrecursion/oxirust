//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::types::{ExpSmoothingParams, ForecastResult};
use crate::decomposition::common::DecompositionModel;
use crate::decomposition::exponential::exponential_decomposition;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
pub(super) fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}
/// Forecasts future values using simple moving average
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `window_size` - Size of the moving window
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::forecasting::moving_average_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = moving_average_forecast(&ts, 3, 5, 0.95).expect("Example/test failed");
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
#[allow(dead_code)]
pub fn moving_average_forecast<F>(
    ts: &Array1<F>,
    window_size: usize,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < window_size {
        return Err(TimeSeriesError::ForecastingError(format!(
            "Time series length ({}) must be at least equal to window _size ({})",
            ts.len(),
            window_size
        )));
    }
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }
    let mut sum = F::zero();
    for i in ts.len() - window_size..ts.len() {
        sum = sum + ts[i];
    }
    let avg = sum / F::from_usize(window_size).expect("Failed to convert usize to float");
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);
    let mut sq_errors = Array1::zeros(ts.len() - window_size);
    for i in window_size..ts.len() {
        let mut window_sum = F::zero();
        for j in i - window_size..i {
            window_sum = window_sum + ts[j];
        }
        let window_avg =
            window_sum / F::from_usize(window_size).expect("Failed to convert usize to float");
        sq_errors[i - window_size] = (ts[i] - window_avg).powi(2);
    }
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).expect("Example/test failed");
    let std_err = mse.sqrt();
    let z_score = match conf_level {
        c if c >= 0.99 => const_f64::<F>(2.576),
        c if c >= 0.98 => const_f64::<F>(2.326),
        c if c >= 0.95 => const_f64::<F>(1.96),
        c if c >= 0.90 => const_f64::<F>(1.645),
        c if c >= 0.85 => const_f64::<F>(1.44),
        c if c >= 0.80 => const_f64::<F>(1.282),
        _ => const_f64::<F>(1.0),
    };
    for i in 0..horizon {
        forecast[i] = avg;
        let adjustment = F::one()
            + const_f64::<F>(0.1) * F::from_usize(i).expect("Failed to convert usize to float");
        let ci_width = z_score * std_err * adjustment;
        lower_ci[i] = avg - ci_width;
        upper_ci[i] = avg + ci_width;
    }
    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}
/// Forecasts future values using simple exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `alpha` - Smoothing parameter (0.0-1.0)
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::forecasting::exponential_smoothing_forecast;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = exponential_smoothing_forecast(&ts, 0.3, 5, 0.95).expect("Example/test failed");
/// println!("Forecast: {:?}", result.forecast);
/// println!("Lower CI: {:?}", result.lower_ci);
/// println!("Upper CI: {:?}", result.upper_ci);
/// ```
#[allow(dead_code)]
pub fn exponential_smoothing_forecast<F>(
    ts: &Array1<F>,
    alpha: f64,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 2 points for exponential smoothing".to_string(),
        ));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Smoothing parameter (alpha) must be between 0 and 1 (exclusive)".to_string(),
        ));
    }
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }
    let mut _level = Array1::zeros(ts.len() + 1);
    let mut sq_errors = Array1::zeros(ts.len() - 1);
    _level[0] = ts[0];
    for i in 0..ts.len() {
        _level[i + 1] = F::from_f64(alpha).expect("Failed to convert alpha to float") * ts[i]
            + F::from_f64(1.0 - alpha).expect("Failed to convert (1.0 - alpha) to float")
                * _level[i];
        if i > 0 {
            sq_errors[i - 1] = (ts[i] - _level[i]).powi(2);
        }
    }
    let mse = sq_errors.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from_usize(sq_errors.len()).expect("Example/test failed");
    let std_err = mse.sqrt();
    let z_score = match conf_level {
        c if c >= 0.99 => const_f64::<F>(2.576),
        c if c >= 0.98 => const_f64::<F>(2.326),
        c if c >= 0.95 => const_f64::<F>(1.96),
        c if c >= 0.90 => const_f64::<F>(1.645),
        c if c >= 0.85 => const_f64::<F>(1.44),
        c if c >= 0.80 => const_f64::<F>(1.282),
        _ => const_f64::<F>(1.0),
    };
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);
    for i in 0..horizon {
        forecast[i] = _level[ts.len()];
        let h_adjustment = (F::from_usize(i + 1).expect("Failed to convert usize to float")).sqrt();
        let ci_width = z_score * std_err * h_adjustment;
        lower_ci[i] = forecast[i] - ci_width;
        upper_ci[i] = forecast[i] + ci_width;
    }
    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}
/// Forecasts future values using Holt-Winters exponential smoothing
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `params` - Exponential smoothing parameters
/// * `horizon` - Number of future points to forecast
/// * `conf_level` - Confidence level (0.0-1.0) for prediction intervals
///
/// # Returns
///
/// * Forecast result containing point forecasts and confidence intervals
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::forecasting::{holt_winters_forecast, ExpSmoothingParams};
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0];
///
/// let mut params = ExpSmoothingParams::default();
/// params.alpha = 0.3;
/// params.beta = Some(0.1);
/// params.gamma = Some(0.2);
/// params.seasonal_period = Some(4);
///
/// let result = holt_winters_forecast(&ts, &params, 8, 0.95).expect("Example/test failed");
/// println!("Forecast: {:?}", result.forecast);
/// ```
#[allow(dead_code)]
pub fn holt_winters_forecast<F>(
    ts: &Array1<F>,
    params: &ExpSmoothingParams,
    horizon: usize,
    conf_level: f64,
) -> Result<ForecastResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if params.alpha <= 0.0 || params.alpha >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Alpha must be between 0 and 1 (exclusive)".to_string(),
        ));
    }
    if let Some(beta) = params.beta {
        if beta <= 0.0 || beta >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Beta must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
    }
    if let Some(gamma) = params.gamma {
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Gamma must be between 0 and 1 (exclusive)".to_string(),
            ));
        }
        if params.seasonal_period.is_none() {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal period must be provided when gamma is specified".to_string(),
            ));
        }
    }
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Confidence _level must be between 0 and 1 (exclusive)".to_string(),
        ));
    }
    if let Some(period) = params.seasonal_period {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::ForecastingError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    } else if ts.len() < 3 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series must have at least 3 points for non-seasonal models".to_string(),
        ));
    }
    if params.damped_trend {
        if let Some(phi) = params.phi {
            if phi <= 0.0 || phi >= 1.0 {
                return Err(TimeSeriesError::InvalidInput(
                    "Damping parameter (phi) must be between 0 and 1 (exclusive)".to_string(),
                ));
            }
        } else {
            return Err(TimeSeriesError::InvalidInput(
                "Damping parameter (phi) must be provided for damped trend models".to_string(),
            ));
        }
    }
    let has_trend = params.beta.is_some();
    let has_seasonal = params.gamma.is_some() && params.seasonal_period.is_some();
    if (params.multiplicative_trend || params.multiplicative_seasonality)
        && ts.iter().any(|&x| x <= F::zero())
    {
        return Err(TimeSeriesError::InvalidInput(
            "Multiplicative models require strictly positive data".to_string(),
        ));
    }
    let n = ts.len();
    let mut _level = Array1::zeros(n + 1);
    let mut trend = Array1::zeros(n + 1);
    let mut seasonal = Array1::zeros(n + params.seasonal_period.unwrap_or(1));
    let mut forecast_errors = Array1::zeros(n);
    if has_seasonal {
        let period = params.seasonal_period.expect("Seasonal period required");
        let decomp = exponential_decomposition(
            ts,
            period,
            params.alpha,
            params.beta.unwrap_or(0.1),
            params.gamma.expect("Gamma parameter required"),
            if params.multiplicative_seasonality {
                DecompositionModel::Multiplicative
            } else {
                DecompositionModel::Additive
            },
        )?;
        _level[n] = decomp.trend[n - 1];
        if has_trend {
            if n >= 2 {
                trend[n] = decomp.trend[n - 1] - decomp.trend[n - 2];
            }
        }
        for i in 0..period {
            seasonal[i] = decomp.seasonal[n - period + i];
        }
        for i in period..n {
            let pred = decomp.trend[i - 1] + decomp.seasonal[i];
            forecast_errors[i] = ts[i] - pred;
        }
    } else if has_trend {
        _level[0] = ts[0];
        if n > 1 {
            trend[0] = ts[1] - ts[0];
        }
        let alpha = F::from_f64(params.alpha).expect("Failed to convert alpha to float");
        let beta = F::from_f64(params.beta.expect("Beta parameter required"))
            .expect("Failed to convert beta to float");
        let phi = F::from_f64(params.phi.unwrap_or(1.0)).expect("Failed to convert phi to float");
        for i in 1..=n {
            let expected = _level[i - 1]
                + if params.damped_trend {
                    phi * trend[i - 1]
                } else {
                    trend[i - 1]
                };
            if i < n {
                _level[i] = alpha * ts[i - 1] + (F::one() - alpha) * expected;
                trend[i] = beta * (_level[i] - _level[i - 1])
                    + (F::one() - beta)
                        * if params.damped_trend {
                            phi * trend[i - 1]
                        } else {
                            trend[i - 1]
                        };
                forecast_errors[i - 1] = ts[i - 1] - expected;
            }
        }
    } else {
        return exponential_smoothing_forecast(ts, params.alpha, horizon, conf_level);
    }
    let mse = forecast_errors
        .iter()
        .skip(if has_seasonal {
            params.seasonal_period.expect("Seasonal period required")
        } else {
            1
        })
        .fold(F::zero(), |acc, &x| acc + x.powi(2))
        / F::from_usize(if has_seasonal {
            n - params.seasonal_period.expect("Seasonal period required")
        } else {
            n - 1
        })
        .expect("Example/test failed");
    let std_err = mse.sqrt();
    let z_score = match conf_level {
        c if c >= 0.99 => const_f64::<F>(2.576),
        c if c >= 0.98 => const_f64::<F>(2.326),
        c if c >= 0.95 => const_f64::<F>(1.96),
        c if c >= 0.90 => const_f64::<F>(1.645),
        c if c >= 0.85 => const_f64::<F>(1.44),
        c if c >= 0.80 => const_f64::<F>(1.282),
        _ => const_f64::<F>(1.0),
    };
    let mut forecast = Array1::zeros(horizon);
    let mut lower_ci = Array1::zeros(horizon);
    let mut upper_ci = Array1::zeros(horizon);
    for h in 0..horizon {
        let mut pred = _level[n];
        if has_trend {
            let phi =
                F::from_f64(params.phi.unwrap_or(1.0)).expect("Failed to convert phi to float");
            if params.damped_trend {
                let mut sum = F::one();
                let mut term = F::one();
                for _ in 1..h + 1 {
                    term = term * phi;
                    sum = sum + term;
                }
                pred = pred + trend[n] * sum;
            } else {
                pred = pred
                    + F::from_usize(h + 1).expect("Failed to convert usize to float") * trend[n];
            }
        }
        if has_seasonal {
            let period = params.seasonal_period.expect("Seasonal period required");
            let season_idx = (h + n) % period;
            if params.multiplicative_seasonality {
                pred = pred * seasonal[season_idx];
            } else {
                pred = pred + seasonal[season_idx];
            }
        }
        forecast[h] = pred;
        let h_adjustment = (F::from_usize(h + 1).expect("Failed to convert usize to float")).sqrt();
        let ci_width = z_score * std_err * h_adjustment;
        lower_ci[h] = pred - ci_width;
        upper_ci[h] = pred + ci_width;
    }
    Ok(ForecastResult {
        forecast,
        lower_ci,
        upper_ci,
    })
}
