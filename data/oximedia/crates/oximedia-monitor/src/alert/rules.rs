//! Alert rule definitions.

use crate::alert::{conditions::AlertCondition, AlertSeverity};
use serde::{Deserialize, Serialize};

/// Alert rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name.
    pub name: String,
    /// Metric name to monitor.
    pub metric_name: String,
    /// Alert condition.
    pub condition: AlertCondition,
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Alert message template.
    pub message: String,
    /// Rule enabled.
    pub enabled: bool,
}

impl AlertRule {
    /// Create a new alert rule.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        metric_name: impl Into<String>,
        condition: AlertCondition,
        severity: AlertSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            metric_name: metric_name.into(),
            condition,
            severity,
            message: message.into(),
            enabled: true,
        }
    }

    /// Evaluate the rule against a metric value.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> bool {
        if !self.enabled {
            return false;
        }
        self.condition.evaluate(value)
    }
}

/// Alert rule builder.
pub struct AlertRuleBuilder {
    name: String,
    metric_name: String,
    condition: Option<AlertCondition>,
    severity: AlertSeverity,
    message: String,
    enabled: bool,
}

impl AlertRuleBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metric_name: String::new(),
            condition: None,
            severity: AlertSeverity::Warning,
            message: String::new(),
            enabled: true,
        }
    }

    /// Set the metric name.
    #[must_use]
    pub fn metric(mut self, metric_name: impl Into<String>) -> Self {
        self.metric_name = metric_name.into();
        self
    }

    /// Set the condition.
    #[must_use]
    pub fn condition(mut self, condition: AlertCondition) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Set the severity.
    #[must_use]
    pub fn severity(mut self, severity: AlertSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the message.
    #[must_use]
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Set whether the rule is enabled.
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Build the rule.
    #[must_use]
    pub fn build(self) -> AlertRule {
        AlertRule {
            name: self.name,
            metric_name: self.metric_name,
            condition: self.condition.unwrap_or(AlertCondition::Threshold(
                crate::alert::conditions::ThresholdCondition {
                    operator: crate::alert::conditions::ComparisonOperator::GreaterThan,
                    value: 0.0,
                    duration_secs: 0,
                },
            )),
            severity: self.severity,
            message: self.message,
            enabled: self.enabled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::conditions::{ComparisonOperator, ThresholdCondition};

    #[test]
    fn test_alert_rule() {
        let condition = AlertCondition::Threshold(ThresholdCondition {
            operator: ComparisonOperator::GreaterThan,
            value: 90.0,
            duration_secs: 60,
        });

        let rule = AlertRule::new(
            "cpu_high",
            "cpu.usage",
            condition,
            AlertSeverity::Warning,
            "CPU usage is high",
        );

        assert_eq!(rule.name, "cpu_high");
        assert!(rule.enabled);
        assert!(rule.evaluate(95.0));
        assert!(!rule.evaluate(85.0));
    }

    #[test]
    fn test_rule_builder() {
        let condition = AlertCondition::Threshold(ThresholdCondition {
            operator: ComparisonOperator::GreaterThan,
            value: 90.0,
            duration_secs: 60,
        });

        let rule = AlertRuleBuilder::new("cpu_high")
            .metric("cpu.usage")
            .condition(condition)
            .severity(AlertSeverity::Critical)
            .message("CPU usage critical")
            .build();

        assert_eq!(rule.name, "cpu_high");
        assert_eq!(rule.severity, AlertSeverity::Critical);
    }
}
