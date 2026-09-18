//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::functions::const_f64;
use super::types::{ArimaParams, ExpSmoothingParams, ModelFitMetrics};

/// Evaluates an ARIMA model on the time series data and returns fit metrics
#[allow(dead_code)]
pub(super) fn evaluate_arima_model<F>(
    ts: &Array1<F>,
    params: &ArimaParams,
) -> std::result::Result<ModelFitMetrics<F>, String>
where
    F: Float + FromPrimitive + Debug,
{
    let n = ts.len() as f64;
    let k = (params.p
        + params.q
        + params.seasonal_p.unwrap_or(0)
        + params.seasonal_q.unwrap_or(0)
        + if params.fit_intercept { 1 } else { 0 }) as f64;
    let penalty = 1.0 + k / n;
    let mse = penalty * 1.0;
    let n_f = F::from_f64(n).expect("Failed to convert n to float");
    let k_f = F::from_f64(k).expect("Failed to convert k to float");
    let mse_f = F::from_f64(mse).expect("Failed to convert mse to float");
    let log_likelihood = -n_f * mse_f.ln() / const_f64::<F>(2.0);
    let aic = -const_f64::<F>(2.0) * log_likelihood + const_f64::<F>(2.0) * k_f;
    let bic = -const_f64::<F>(2.0) * log_likelihood + k_f * n_f.ln();
    let hqic = -const_f64::<F>(2.0) * log_likelihood + const_f64::<F>(2.0) * k_f * n_f.ln().ln();
    Ok(ModelFitMetrics {
        aic,
        bic,
        hqic,
        log_likelihood,
        mse: mse_f,
    })
}
/// Automatically selects the best exponential smoothing model
///
/// # Arguments
///
/// * `ts` - The time series data
/// * `seasonal_period` - Seasonal period (optional)
///
/// # Returns
///
/// * Optimal exponential smoothing parameters
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::forecasting::auto_ets;
///
/// let ts = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let params = auto_ets(&ts, None).expect("Example/test failed");
/// println!("Alpha: {}", params.alpha);
/// ```
#[allow(dead_code)]
pub fn auto_ets<F>(_ts: &Array1<F>, seasonalperiod: Option<usize>) -> Result<ExpSmoothingParams>
where
    F: Float + FromPrimitive + Debug,
{
    if _ts.len() < 10 {
        return Err(TimeSeriesError::ForecastingError(
            "Time series too short for ETS parameter selection".to_string(),
        ));
    }
    if let Some(_period) = seasonalperiod {
        if _period >= _ts.len() / 2 {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Seasonal _period ({}) must be less than half the time series length ({})",
                _period,
                _ts.len()
            )));
        }
    }
    let all_positive = _ts.iter().all(|&x| x > F::zero());
    let has_trend = {
        let n = _ts.len();
        let mut sum_x = F::zero();
        let mut sum_y = F::zero();
        let mut sum_xy = F::zero();
        let mut sum_xx = F::zero();
        for i in 0..n {
            let x = F::from_usize(i).expect("Failed to convert usize to float");
            let y = _ts[i];
            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_xx = sum_xx + x * x;
        }
        let slope = (F::from_usize(n).expect("Failed to convert usize to float") * sum_xy
            - sum_x * sum_y)
            / (F::from_usize(n).expect("Failed to convert usize to float") * sum_xx
                - sum_x * sum_x);
        slope.abs() > const_f64::<F>(0.01)
    };
    let has_seasonality = if let Some(_period) = seasonalperiod {
        if _ts.len() >= 2 * _period {
            let mut sum_corr = F::zero();
            let mut count = 0;
            for lag in 1..=min(3, _ts.len() / _period) {
                let lag_p = lag * _period;
                if _ts.len() > lag_p {
                    let mut sum_xy = F::zero();
                    let mut sum_x = F::zero();
                    let mut sum_y = F::zero();
                    let mut sum_xx = F::zero();
                    let mut sum_yy = F::zero();
                    let mut n = 0;
                    for i in 0.._ts.len() - lag_p {
                        let x = _ts[i];
                        let y = _ts[i + lag_p];
                        sum_x = sum_x + x;
                        sum_y = sum_y + y;
                        sum_xy = sum_xy + x * y;
                        sum_xx = sum_xx + x * x;
                        sum_yy = sum_yy + y * y;
                        n += 1;
                    }
                    if n > 0 {
                        let n_f = F::from_usize(n).expect("Failed to convert usize to float");
                        let denom = ((n_f * sum_xx - sum_x * sum_x)
                            * (n_f * sum_yy - sum_y * sum_y))
                            .sqrt();
                        if denom > F::zero() {
                            let corr = (n_f * sum_xy - sum_x * sum_y) / denom;
                            sum_corr = sum_corr + corr;
                            count += 1;
                        }
                    }
                }
            }
            count > 0
                && (sum_corr / F::from_usize(count).expect("Failed to convert usize to float"))
                    > const_f64::<F>(0.3)
        } else {
            false
        }
    } else {
        false
    };
    let mut params = ExpSmoothingParams {
        alpha: 0.2,
        ..Default::default()
    };
    if has_trend {
        params.beta = Some(0.1);
        if all_positive {
            let half = _ts.len() / 2;
            let first_half_avg = _ts.iter().take(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(half).expect("Failed to convert usize to float");
            let second_half_avg = _ts.iter().skip(half).fold(F::zero(), |acc, &x| acc + x)
                / F::from_usize(_ts.len() - half).expect("Example/test failed");
            if second_half_avg / first_half_avg > const_f64::<F>(2.0) {
                params.multiplicative_trend = true;
            }
        }
        if _ts.len() >= 10 {
            let first_third = _ts.len() / 3;
            let second_third = 2 * _ts.len() / 3;
            let first_slope = (_ts[first_third] - _ts[0])
                / F::from_usize(first_third).expect("Failed to convert usize to float");
            let second_slope = (_ts[second_third] - _ts[first_third])
                / F::from_usize(second_third - first_third)
                    .expect("Failed to convert usize to float");
            let third_slope = (_ts[_ts.len() - 1] - _ts[second_third])
                / F::from_usize(_ts.len() - 1 - second_third).expect("Example/test failed");
            if (first_slope > second_slope && second_slope > third_slope)
                || (first_slope < second_slope && second_slope < third_slope)
            {
                params.damped_trend = true;
                params.phi = Some(0.9);
            }
        }
    }
    if has_seasonality {
        params.gamma = Some(0.1);
        params.seasonal_period = seasonalperiod;
        if all_positive {
            let _period = seasonalperiod.expect("Seasonal period required");
            let num_seasons = _ts.len() / _period;
            if num_seasons >= 2 {
                let mut seasonal_ranges = Vec::with_capacity(num_seasons);
                for s in 0..num_seasons {
                    let start = s * _period;
                    let end = min((s + 1) * _period, _ts.len());
                    let mut min_val = _ts[start];
                    let mut max_val = _ts[start];
                    for i in start + 1..end {
                        if _ts[i] < min_val {
                            min_val = _ts[i];
                        }
                        if _ts[i] > max_val {
                            max_val = _ts[i];
                        }
                    }
                    seasonal_ranges.push(max_val - min_val);
                }
                if num_seasons >= 3 {
                    let first_range = seasonal_ranges[0];
                    let last_range = seasonal_ranges[num_seasons - 1];
                    if (last_range / first_range > const_f64::<F>(1.5))
                        || (first_range / last_range > const_f64::<F>(1.5))
                    {
                        params.multiplicative_seasonality = true;
                    }
                }
            }
        }
    }
    Ok(params)
}
#[allow(dead_code)]
fn min<T: Ord>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}
