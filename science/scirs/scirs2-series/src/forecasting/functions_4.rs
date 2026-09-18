//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::functions::{
    const_f64, exponential_smoothing_forecast, holt_winters_forecast, moving_average_forecast,
};
use super::functions_2::arima_forecast;
use super::types::{ArimaParams, ExpSmoothingParams};

/// Ensemble forecasting methods
pub mod ensemble {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::numeric::{Float, FromPrimitive};
    use std::fmt::Debug;
    /// Configuration for ensemble forecasting
    #[derive(Debug, Clone)]
    pub struct EnsembleConfig {
        /// Include ARIMA model in ensemble
        pub use_arima: bool,
        /// Include exponential smoothing in ensemble
        pub use_exp_smoothing: bool,
        /// Include Holt-Winters in ensemble
        pub use_holt_winters: bool,
        /// Include moving average in ensemble
        pub use_moving_average: bool,
        /// Weights for models (if empty, uses equal weights)
        pub weights: Vec<f64>,
        /// Use adaptive weighting based on historical performance
        pub adaptive_weights: bool,
        /// Forecast horizon
        pub horizon: usize,
    }
    impl Default for EnsembleConfig {
        fn default() -> Self {
            Self {
                use_arima: true,
                use_exp_smoothing: true,
                use_holt_winters: true,
                use_moving_average: true,
                weights: vec![],
                adaptive_weights: false,
                horizon: 12,
            }
        }
    }
    /// Result of ensemble forecasting
    #[derive(Debug, Clone)]
    pub struct EnsembleResult<F> {
        /// Combined ensemble forecast
        pub ensemble_forecast: Array1<F>,
        /// Individual model forecasts
        pub individual_forecasts: Vec<Array1<F>>,
        /// Model names
        pub model_names: Vec<String>,
        /// Final weights used for combination
        pub weights: Vec<f64>,
        /// Lower confidence interval
        pub lower_ci: Array1<F>,
        /// Upper confidence interval
        pub upper_ci: Array1<F>,
    }
    /// Simple averaging ensemble
    pub fn simple_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }
        let mut individual_forecasts = Vec::new();
        let mut model_names = Vec::new();
        if config.use_moving_average {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("MovingAverage".to_string());
            }
        }
        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("ExponentialSmoothing".to_string());
            }
        }
        if config.use_holt_winters && data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                individual_forecasts.push(forecast.forecast);
                model_names.push("HoltWinters".to_string());
            }
        }
        if config.use_arima && data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                individual_forecasts.push(forecast.forecast);
                model_names.push("ARIMA".to_string());
            }
        }
        if individual_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No valid forecasts could be generated".to_string(),
            ));
        }
        let weights =
            if config.weights.is_empty() || config.weights.len() != individual_forecasts.len() {
                vec![1.0 / individual_forecasts.len() as f64; individual_forecasts.len()]
            } else {
                let sum: f64 = config.weights.iter().sum();
                config.weights.iter().map(|w| w / sum).collect()
            };
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            let w = F::from_f64(*weight).expect("Failed to convert weight to float");
            for j in 0..config.horizon {
                if j < individual_forecasts[i].len() {
                    ensemble_forecast[j] = ensemble_forecast[j] + w * individual_forecasts[i][j];
                }
            }
        }
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);
        for j in 0..config.horizon {
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;
            for forecast in &individual_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }
            if count > 1 {
                variance =
                    variance / F::from_usize(count).expect("Failed to convert usize to float");
                let std_dev = variance.sqrt();
                let margin = std_dev * const_f64::<F>(1.96);
                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }
        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }
    /// Weighted ensemble based on historical performance
    pub fn weighted_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 20 observations for weighted ensemble".to_string(),
                required: 20,
                actual: data.len(),
            });
        }
        let split_point = data.len() - config.horizon;
        let train_data = data
            .slice(scirs2_core::ndarray::s![..split_point])
            .to_owned();
        let validation_data = data
            .slice(scirs2_core::ndarray::s![split_point..])
            .to_owned();
        let mut individual_forecasts = Vec::new();
        let mut model_names = Vec::new();
        let mut validation_errors = Vec::new();
        if config.use_moving_average {
            let window = std::cmp::min(12, train_data.len() / 2);
            if let Ok(forecast) =
                super::moving_average_forecast(&train_data, window, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("MovingAverage".to_string());
                validation_errors.push(error);
            }
        }
        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) = super::exponential_smoothing_forecast(
                &train_data,
                params.alpha,
                config.horizon,
                0.95,
            ) {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("ExponentialSmoothing".to_string());
                validation_errors.push(error);
            }
        }
        if config.use_holt_winters && train_data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) =
                super::holt_winters_forecast(&train_data, &params, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("HoltWinters".to_string());
                validation_errors.push(error);
            }
        }
        if config.use_arima && train_data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(&train_data, &params, config.horizon, 0.95)
            {
                let error = calculate_mse(&forecast.forecast, &validation_data);
                individual_forecasts.push(forecast.forecast);
                model_names.push("ARIMA".to_string());
                validation_errors.push(error);
            }
        }
        if individual_forecasts.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No valid forecasts could be generated".to_string(),
            ));
        }
        let weights: Vec<f64> = if validation_errors.iter().all(|&e| e > 0.0) {
            let inv_errors: Vec<f64> = validation_errors.iter().map(|&e| 1.0 / e).collect();
            let sum: f64 = inv_errors.iter().sum();
            inv_errors.iter().map(|&w| w / sum).collect()
        } else {
            vec![1.0 / individual_forecasts.len() as f64; individual_forecasts.len()]
        };
        let mut final_forecasts = Vec::new();
        if config.use_moving_average && model_names.contains(&"MovingAverage".to_string()) {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_exp_smoothing && model_names.contains(&"ExponentialSmoothing".to_string()) {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_holt_winters && model_names.contains(&"HoltWinters".to_string()) {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_arima && model_names.contains(&"ARIMA".to_string()) {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                final_forecasts.push(forecast.forecast);
            }
        }
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            if i < final_forecasts.len() {
                let w = F::from_f64(*weight).expect("Failed to convert weight to float");
                for j in 0..config.horizon {
                    if j < final_forecasts[i].len() {
                        ensemble_forecast[j] = ensemble_forecast[j] + w * final_forecasts[i][j];
                    }
                }
            }
        }
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);
        for j in 0..config.horizon {
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;
            for forecast in &final_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }
            if count > 1 {
                variance =
                    variance / F::from_usize(count).expect("Failed to convert usize to float");
                let std_dev = variance.sqrt();
                let margin = std_dev * const_f64::<F>(1.96);
                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }
        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts: final_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }
    /// Stacked ensemble using simple linear combination
    pub fn stacked_ensemble_forecast<F>(
        data: &Array1<F>,
        config: &EnsembleConfig,
    ) -> Result<EnsembleResult<F>>
    where
        F: Float + FromPrimitive + Debug + Clone,
    {
        if data.len() < 30 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 30 observations for stacked ensemble".to_string(),
                required: 30,
                actual: data.len(),
            });
        }
        let cv_folds = 3;
        let fold_size = data.len() / cv_folds;
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        let mut model_names = Vec::new();
        if config.use_moving_average {
            model_names.push("MovingAverage".to_string());
        }
        if config.use_exp_smoothing {
            model_names.push("ExponentialSmoothing".to_string());
        }
        if config.use_holt_winters {
            model_names.push("HoltWinters".to_string());
        }
        if config.use_arima {
            model_names.push("ARIMA".to_string());
        }
        for fold in 0..cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == cv_folds - 1 {
                data.len()
            } else {
                (fold + 1) * fold_size
            };
            if test_start >= data.len() - 1 {
                continue;
            }
            let train_data = if test_start == 0 {
                data.slice(scirs2_core::ndarray::s![test_end..]).to_owned()
            } else {
                let train_part1 = data
                    .slice(scirs2_core::ndarray::s![..test_start])
                    .to_owned();
                let train_part2 = data.slice(scirs2_core::ndarray::s![test_end..]).to_owned();
                let mut combined = Array1::zeros(train_part1.len() + train_part2.len());
                combined
                    .slice_mut(scirs2_core::ndarray::s![..train_part1.len()])
                    .assign(&train_part1);
                combined
                    .slice_mut(scirs2_core::ndarray::s![train_part1.len()..])
                    .assign(&train_part2);
                combined
            };
            let test_data = data
                .slice(scirs2_core::ndarray::s![test_start..test_end])
                .to_owned();
            let horizon = test_data.len();
            if train_data.len() < 10 {
                continue;
            }
            let mut fold_predictions = Vec::new();
            if config.use_moving_average {
                let window = std::cmp::min(12, train_data.len() / 2);
                if let Ok(forecast) =
                    super::moving_average_forecast(&train_data, window, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }
            if config.use_exp_smoothing {
                let params = super::ExpSmoothingParams::default();
                if let Ok(forecast) =
                    super::exponential_smoothing_forecast(&train_data, params.alpha, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }
            if config.use_holt_winters && train_data.len() >= 24 {
                let params = super::ExpSmoothingParams {
                    alpha: 0.3,
                    beta: Some(0.1),
                    gamma: Some(0.1),
                    seasonal_period: Some(12),
                    ..Default::default()
                };
                if let Ok(forecast) =
                    super::holt_winters_forecast(&train_data, &params, horizon, 0.95)
                {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }
            if config.use_arima && train_data.len() >= 20 {
                let params = super::ArimaParams {
                    p: 1,
                    d: 1,
                    q: 1,
                    ..Default::default()
                };
                if let Ok(forecast) = super::arima_forecast(&train_data, &params, horizon, 0.95) {
                    fold_predictions.push(forecast.forecast);
                } else {
                    fold_predictions.push(Array1::zeros(horizon));
                }
            }
            all_predictions.push(fold_predictions);
            all_targets.push(test_data);
        }
        let num_models = model_names.len();
        let mut weights = vec![1.0 / num_models as f64; num_models];
        if !all_predictions.is_empty() {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for (fold_predictions, targets) in all_predictions.iter().zip(all_targets.iter()) {
                for i in 0..targets.len() {
                    let mut row = Vec::new();
                    for model_pred in fold_predictions {
                        if i < model_pred.len() {
                            row.push(model_pred[i].to_f64().unwrap_or(0.0));
                        } else {
                            row.push(0.0);
                        }
                    }
                    if row.len() == num_models {
                        x.push(row);
                        y.push(targets[i].to_f64().unwrap_or(0.0));
                    }
                }
            }
            if x.len() > num_models && x.iter().all(|row| row.len() == num_models) {
                weights = solve_linear_regression(&x, &y);
                let mut sum = weights.iter().sum::<f64>();
                if sum <= 0.0 {
                    weights = vec![1.0 / num_models as f64; num_models];
                } else {
                    for w in weights.iter_mut() {
                        *w = w.max(0.0);
                    }
                    sum = weights.iter().sum::<f64>();
                    if sum > 0.0 {
                        for w in weights.iter_mut() {
                            *w /= sum;
                        }
                    } else {
                        weights = vec![1.0 / num_models as f64; num_models];
                    }
                }
            }
        }
        let mut final_forecasts = Vec::new();
        if config.use_moving_average {
            let window = std::cmp::min(12, data.len() / 2);
            if let Ok(forecast) = super::moving_average_forecast(data, window, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_exp_smoothing {
            let params = super::ExpSmoothingParams::default();
            if let Ok(forecast) =
                super::exponential_smoothing_forecast(data, params.alpha, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_holt_winters && data.len() >= 24 {
            let params = super::ExpSmoothingParams {
                alpha: 0.3,
                beta: Some(0.1),
                gamma: Some(0.1),
                seasonal_period: Some(12),
                ..Default::default()
            };
            if let Ok(forecast) = super::holt_winters_forecast(data, &params, config.horizon, 0.95)
            {
                final_forecasts.push(forecast.forecast);
            }
        }
        if config.use_arima && data.len() >= 20 {
            let params = super::ArimaParams {
                p: 1,
                d: 1,
                q: 1,
                ..Default::default()
            };
            if let Ok(forecast) = super::arima_forecast(data, &params, config.horizon, 0.95) {
                final_forecasts.push(forecast.forecast);
            }
        }
        let mut ensemble_forecast = Array1::zeros(config.horizon);
        for (i, weight) in weights.iter().enumerate() {
            if i < final_forecasts.len() {
                let w = F::from_f64(*weight).expect("Failed to convert weight to float");
                for j in 0..config.horizon {
                    if j < final_forecasts[i].len() {
                        ensemble_forecast[j] = ensemble_forecast[j] + w * final_forecasts[i][j];
                    }
                }
            }
        }
        let mut lower_ci = Array1::zeros(config.horizon);
        let mut upper_ci = Array1::zeros(config.horizon);
        for j in 0..config.horizon {
            let mean = ensemble_forecast[j];
            let mut variance = F::zero();
            let mut count = 0;
            for forecast in &final_forecasts {
                if j < forecast.len() {
                    let diff = forecast[j] - mean;
                    variance = variance + diff * diff;
                    count += 1;
                }
            }
            if count > 1 {
                variance =
                    variance / F::from_usize(count).expect("Failed to convert usize to float");
                let std_dev = variance.sqrt();
                let margin = std_dev * const_f64::<F>(1.96);
                lower_ci[j] = mean - margin;
                upper_ci[j] = mean + margin;
            } else {
                lower_ci[j] = mean;
                upper_ci[j] = mean;
            }
        }
        Ok(EnsembleResult {
            ensemble_forecast,
            individual_forecasts: final_forecasts,
            model_names,
            weights,
            lower_ci,
            upper_ci,
        })
    }
    /// Calculate Mean Squared Error between forecast and actual values
    fn calculate_mse<F: Float>(forecast: &Array1<F>, actual: &Array1<F>) -> f64 {
        let min_len = forecast.len().min(actual.len());
        if min_len == 0 {
            return f64::INFINITY;
        }
        let mut sum_sq_error = 0.0;
        for i in 0..min_len {
            let error = forecast[i].to_f64().unwrap_or(0.0) - actual[i].to_f64().unwrap_or(0.0);
            sum_sq_error += error * error;
        }
        sum_sq_error / min_len as f64
    }
    /// Simple linear regression solver for stacking weights
    fn solve_linear_regression(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        let m = if n > 0 { x[0].len() } else { 0 };
        if n == 0 || m == 0 || n != y.len() {
            return vec![1.0 / m.max(1) as f64; m.max(1)];
        }
        let mut weights = vec![1.0 / m as f64; m];
        let learning_rate = 0.01;
        let iterations = 100;
        for _ in 0..iterations {
            let mut gradients = vec![0.0; m];
            for (x_row, &y_val) in x.iter().zip(y.iter()) {
                let prediction = weights
                    .iter()
                    .zip(x_row.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
                let error = prediction - y_val;
                for (grad, &x_val) in gradients.iter_mut().zip(x_row.iter()) {
                    *grad += (2.0 / n as f64) * error * x_val;
                }
            }
            for j in 0..m {
                weights[j] -= learning_rate * gradients[j];
                weights[j] = weights[j].max(0.0);
            }
            let sum: f64 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }
        weights
    }
}
