//! Regression alerts.

use super::detect::RegressionInfo;
use serde::{Deserialize, Serialize};

/// Alert level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Critical regression.
    Critical,

    /// Warning level.
    Warning,

    /// Info level.
    Info,
}

impl AlertLevel {
    /// Get alert level from regression percentage.
    pub fn from_regression(regression_percent: f64) -> Self {
        if regression_percent > 20.0 {
            Self::Critical
        } else if regression_percent > 10.0 {
            Self::Warning
        } else {
            Self::Info
        }
    }

    /// Get color for this alert level.
    pub fn color(&self) -> &str {
        match self {
            Self::Critical => "red",
            Self::Warning => "orange",
            Self::Info => "yellow",
        }
    }

    /// Get emoji for this alert level.
    pub fn emoji(&self) -> &str {
        match self {
            Self::Critical => "🔴",
            Self::Warning => "🟡",
            Self::Info => "🔵",
        }
    }
}

/// Regression alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    /// Alert level.
    pub level: AlertLevel,

    /// Regression info.
    pub regression: RegressionInfo,

    /// Alert message.
    pub message: String,

    /// Timestamp.
    #[serde(skip, default = "std::time::Instant::now")]
    pub timestamp: std::time::Instant,
}

impl RegressionAlert {
    /// Create a new regression alert.
    pub fn new(regression: RegressionInfo) -> Self {
        let level = AlertLevel::from_regression(regression.regression_percent);
        let message = format!(
            "Performance regression in '{}': {:.2}% slower",
            regression.name, regression.regression_percent
        );

        Self {
            level,
            regression,
            message,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Check if this is a critical alert.
    pub fn is_critical(&self) -> bool {
        self.level == AlertLevel::Critical
    }

    /// Check if this is a warning alert.
    pub fn is_warning(&self) -> bool {
        self.level == AlertLevel::Warning
    }

    /// Format the alert for display.
    pub fn format(&self) -> String {
        format!("{} [{:?}] {}", self.level.emoji(), self.level, self.message)
    }

    /// Get detailed information.
    pub fn details(&self) -> String {
        let mut details = self.format();
        details.push_str(&format!("\n  Baseline: {:?}", self.regression.baseline));
        details.push_str(&format!("\n  Current:  {:?}", self.regression.current));
        details.push_str(&format!(
            "\n  Std Deviations: {:.2}",
            self.regression.std_deviations
        ));
        details
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_alert_level_from_regression() {
        assert_eq!(AlertLevel::from_regression(25.0), AlertLevel::Critical);
        assert_eq!(AlertLevel::from_regression(15.0), AlertLevel::Warning);
        assert_eq!(AlertLevel::from_regression(5.0), AlertLevel::Info);
    }

    #[test]
    fn test_regression_alert() {
        let regression = RegressionInfo {
            name: "test".to_string(),
            baseline: Duration::from_millis(100),
            current: Duration::from_millis(130),
            regression_percent: 30.0,
            std_deviations: 3.0,
            is_significant: true,
        };

        let alert = RegressionAlert::new(regression);
        assert!(alert.is_critical());
        assert!(!alert.is_warning());
    }

    #[test]
    fn test_alert_format() {
        let regression = RegressionInfo {
            name: "test".to_string(),
            baseline: Duration::from_millis(100),
            current: Duration::from_millis(120),
            regression_percent: 20.0,
            std_deviations: 2.0,
            is_significant: true,
        };

        let alert = RegressionAlert::new(regression);
        let formatted = alert.format();

        assert!(formatted.contains("test"));
        assert!(formatted.contains("20"));
    }
}
