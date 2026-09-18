//! Alert condition definitions.

use serde::{Deserialize, Serialize};

/// Alert condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AlertCondition {
    /// Threshold condition.
    Threshold(ThresholdCondition),
    /// Rate condition.
    Rate(RateCondition),
    /// Absence condition.
    Absence(AbsenceCondition),
    /// Anomaly detection.
    Anomaly(AnomalyCondition),
    /// Composite condition.
    Composite(CompositeCondition),
}

impl AlertCondition {
    /// Evaluates whether the condition is met for the given value.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> bool {
        match self {
            Self::Threshold(t) => t.evaluate(value),
            _ => false,
        }
    }

    /// Returns the configured threshold value when this is a threshold
    /// condition, otherwise `None`.
    #[must_use]
    pub fn threshold(&self) -> Option<f64> {
        match self {
            Self::Threshold(t) => Some(t.value),
            _ => None,
        }
    }
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than.
    GreaterThan,
    /// Less than.
    LessThan,
    /// Equal to.
    Equal,
}

/// Threshold condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCondition {
    /// Comparison operator.
    pub operator: ComparisonOperator,
    /// Threshold value.
    pub value: f64,
    /// Duration in seconds the condition must be true.
    pub duration_secs: u64,
}

impl ThresholdCondition {
    /// Evaluates whether the given value meets the threshold condition.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> bool {
        match self.operator {
            ComparisonOperator::GreaterThan => value > self.value,
            ComparisonOperator::LessThan => value < self.value,
            ComparisonOperator::Equal => (value - self.value).abs() < f64::EPSILON,
        }
    }
}

/// Rate condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateCondition {
    /// Rate threshold (per minute).
    pub rate: f64,
    /// Window size in seconds.
    pub window_secs: u64,
}

/// Absence condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsenceCondition {
    /// Timeout in seconds.
    pub timeout_secs: u64,
}

/// Anomaly detection condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyCondition {
    /// Baseline value.
    pub baseline: f64,
    /// Deviation threshold (percentage).
    pub deviation_percent: f64,
}

/// Composite condition (AND/OR).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeCondition {
    /// Operator.
    pub operator: LogicalOperator,
    /// Sub-conditions.
    pub conditions: Vec<AlertCondition>,
}

/// Logical operator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogicalOperator {
    /// AND.
    And,
    /// OR.
    Or,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_condition() {
        let cond = ThresholdCondition {
            operator: ComparisonOperator::GreaterThan,
            value: 90.0,
            duration_secs: 60,
        };

        assert!(cond.evaluate(95.0));
        assert!(!cond.evaluate(85.0));
    }
}
