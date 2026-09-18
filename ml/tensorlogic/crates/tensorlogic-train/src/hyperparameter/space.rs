//! Hyperparameter space definition for search algorithms.

use crate::{TrainError, TrainResult};
use scirs2_core::random::{RngExt, StdRng};

use super::value::HyperparamValue;

/// Hyperparameter space definition.
#[derive(Debug, Clone)]
pub enum HyperparamSpace {
    /// Discrete choices.
    Discrete(Vec<HyperparamValue>),
    /// Continuous range [min, max].
    Continuous { min: f64, max: f64 },
    /// Log-uniform distribution [min, max].
    LogUniform { min: f64, max: f64 },
    /// Integer range [min, max].
    IntRange { min: i64, max: i64 },
}

impl HyperparamSpace {
    /// Create a discrete choice space.
    pub fn discrete(values: Vec<HyperparamValue>) -> TrainResult<Self> {
        if values.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Discrete space cannot be empty".to_string(),
            ));
        }
        Ok(Self::Discrete(values))
    }

    /// Create a continuous range space.
    pub fn continuous(min: f64, max: f64) -> TrainResult<Self> {
        if min >= max {
            return Err(TrainError::InvalidParameter(
                "min must be less than max".to_string(),
            ));
        }
        Ok(Self::Continuous { min, max })
    }

    /// Create a log-uniform distribution space.
    pub fn log_uniform(min: f64, max: f64) -> TrainResult<Self> {
        if min <= 0.0 || max <= 0.0 || min >= max {
            return Err(TrainError::InvalidParameter(
                "min and max must be positive and min < max".to_string(),
            ));
        }
        Ok(Self::LogUniform { min, max })
    }

    /// Create an integer range space.
    pub fn int_range(min: i64, max: i64) -> TrainResult<Self> {
        if min >= max {
            return Err(TrainError::InvalidParameter(
                "min must be less than max".to_string(),
            ));
        }
        Ok(Self::IntRange { min, max })
    }

    /// Sample a value from this space.
    pub fn sample(&self, rng: &mut StdRng) -> HyperparamValue {
        match self {
            HyperparamSpace::Discrete(values) => {
                let idx = rng.gen_range(0..values.len());
                values[idx].clone()
            }
            HyperparamSpace::Continuous { min, max } => {
                let value = min + (max - min) * rng.random::<f64>();
                HyperparamValue::Float(value)
            }
            HyperparamSpace::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_value = log_min + (log_max - log_min) * rng.random::<f64>();
                HyperparamValue::Float(log_value.exp())
            }
            HyperparamSpace::IntRange { min, max } => {
                let value = rng.gen_range(*min..=*max);
                HyperparamValue::Int(value)
            }
        }
    }

    /// Get all possible values for grid search (for discrete/int spaces).
    pub fn grid_values(&self, num_samples: usize) -> Vec<HyperparamValue> {
        match self {
            HyperparamSpace::Discrete(values) => values.clone(),
            HyperparamSpace::IntRange { min, max } => {
                let range_size = (max - min + 1) as usize;
                let step = (range_size / num_samples).max(1);
                (*min..=*max)
                    .step_by(step)
                    .map(HyperparamValue::Int)
                    .collect()
            }
            HyperparamSpace::Continuous { min, max } => {
                let step = (max - min) / (num_samples as f64);
                (0..num_samples)
                    .map(|i| HyperparamValue::Float(min + step * i as f64))
                    .collect()
            }
            HyperparamSpace::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_step = (log_max - log_min) / (num_samples as f64);
                (0..num_samples)
                    .map(|i| HyperparamValue::Float((log_min + log_step * i as f64).exp()))
                    .collect()
            }
        }
    }
}
