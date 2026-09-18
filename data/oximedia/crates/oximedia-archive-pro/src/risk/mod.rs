//! Format obsolescence risk assessment

pub mod alert;
pub mod assess;
pub mod migration_trigger;
pub mod monitor;

pub use alert::{RiskAlert, RiskAlertLevel};
pub use assess::{FormatRisk, RiskAssessor};
pub use migration_trigger::MigrationTriggerPolicy;
pub use monitor::{MonitoringReport, RiskMonitor};

use serde::{Deserialize, Serialize};

/// Risk level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// No known risks
    None,
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk - immediate action required
    Critical,
}

impl RiskLevel {
    /// Returns the risk level name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }

    /// Returns numeric score (0-100)
    #[must_use]
    pub const fn score(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Low => 25,
            Self::Medium => 50,
            Self::High => 75,
            Self::Critical => 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Low < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    #[test]
    fn test_risk_scores() {
        assert_eq!(RiskLevel::None.score(), 0);
        assert_eq!(RiskLevel::Critical.score(), 100);
    }
}
