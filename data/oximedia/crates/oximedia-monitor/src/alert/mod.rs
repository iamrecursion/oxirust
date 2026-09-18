//! Alert system for system monitoring.

pub mod channels;
pub mod conditions;
pub mod history;
#[cfg(not(target_arch = "wasm32"))]
pub mod manager;
pub mod rules;
pub mod severity;

pub use channels::AlertChannel;
#[cfg(all(not(target_arch = "wasm32"), feature = "email"))]
pub use channels::EmailChannel;
#[cfg(not(target_arch = "wasm32"))]
pub use channels::{DiscordChannel, FileChannel, SlackChannel, SmsChannel, WebhookChannel};
pub use conditions::{
    AbsenceCondition, AlertCondition, AnomalyCondition, CompositeCondition, RateCondition,
    ThresholdCondition,
};
pub use history::{AlertHistory, AlertRecord};
#[cfg(not(target_arch = "wasm32"))]
pub use manager::{AlertDeduplicator, AlertManager};
pub use rules::{AlertRule, AlertRuleBuilder};
pub use severity::AlertSeverity;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// An alert instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID.
    pub id: String,
    /// Alert rule name.
    pub rule_name: String,
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Alert message.
    pub message: String,
    /// Metric name that triggered the alert.
    pub metric_name: String,
    /// Current metric value.
    pub metric_value: f64,
    /// Threshold value (if applicable).
    pub threshold: Option<f64>,
    /// Alert timestamp.
    pub timestamp: DateTime<Utc>,
    /// Alert state.
    pub state: AlertState,
    /// Labels.
    pub labels: std::collections::HashMap<String, String>,
}

impl Alert {
    /// Create a new alert.
    #[must_use]
    pub fn new(
        rule_name: impl Into<String>,
        severity: AlertSeverity,
        message: impl Into<String>,
        metric_name: impl Into<String>,
        metric_value: f64,
    ) -> Self {
        let rule_name_str = rule_name.into();
        Self {
            id: format!("{}-{}", rule_name_str, Utc::now().timestamp()),
            rule_name: rule_name_str,
            severity,
            message: message.into(),
            metric_name: metric_name.into(),
            metric_value,
            threshold: None,
            timestamp: Utc::now(),
            state: AlertState::Firing,
            labels: std::collections::HashMap::new(),
        }
    }

    /// Set the threshold value.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Add a label.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Acknowledge the alert.
    pub fn acknowledge(&mut self) {
        self.state = AlertState::Acknowledged;
    }

    /// Resolve the alert.
    pub fn resolve(&mut self) {
        self.state = AlertState::Resolved;
    }
}

/// Alert state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertState {
    /// Alert is firing.
    Firing,
    /// Alert has been acknowledged.
    Acknowledged,
    /// Alert has been resolved.
    Resolved,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            "cpu_high",
            AlertSeverity::Warning,
            "CPU usage is high",
            "cpu.usage",
            95.0,
        )
        .with_threshold(90.0)
        .with_label("host", "server-1");

        assert_eq!(alert.rule_name, "cpu_high");
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.metric_value, 95.0);
        assert_eq!(alert.threshold, Some(90.0));
        assert_eq!(alert.state, AlertState::Firing);
        assert_eq!(alert.labels.get("host"), Some(&"server-1".to_string()));
    }

    #[test]
    fn test_alert_state_transitions() {
        let mut alert = Alert::new(
            "test",
            AlertSeverity::Critical,
            "Test alert",
            "test.metric",
            100.0,
        );

        assert_eq!(alert.state, AlertState::Firing);

        alert.acknowledge();
        assert_eq!(alert.state, AlertState::Acknowledged);

        alert.resolve();
        assert_eq!(alert.state, AlertState::Resolved);
    }
}
