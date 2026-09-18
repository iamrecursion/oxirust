//! Hyperparameter value, result, and config type alias.

use std::collections::HashMap;

/// Hyperparameter configuration (a single point in parameter space).
pub type HyperparamConfig = HashMap<String, HyperparamValue>;

/// Hyperparameter value type.
#[derive(Debug, Clone, PartialEq)]
pub enum HyperparamValue {
    /// Floating-point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
}

impl HyperparamValue {
    /// Get as f64, if possible.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            HyperparamValue::Float(v) => Some(*v),
            HyperparamValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as i64, if possible.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            HyperparamValue::Int(v) => Some(*v),
            HyperparamValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as bool, if possible.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HyperparamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string, if possible.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            HyperparamValue::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Result of a hyperparameter evaluation.
#[derive(Debug, Clone)]
pub struct HyperparamResult {
    /// Hyperparameter configuration used.
    pub config: HyperparamConfig,
    /// Evaluation score (higher is better).
    pub score: f64,
    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

impl HyperparamResult {
    /// Create a new result.
    pub fn new(config: HyperparamConfig, score: f64) -> Self {
        Self {
            config,
            score,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the result.
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }
}
