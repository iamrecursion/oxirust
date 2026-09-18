//! Risk monitoring for format collections

use super::{FormatRisk, RiskLevel};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Monitoring report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringReport {
    /// Risk distribution
    pub risk_distribution: HashMap<RiskLevel, usize>,
    /// Total files monitored
    pub total_files: usize,
    /// High-risk files
    pub high_risk_files: Vec<String>,
    /// Report timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MonitoringReport {
    /// Generate summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Monitoring Report: {} files total, {} at high risk",
            self.total_files,
            self.high_risk_files.len()
        )
    }
}

/// Risk monitor
pub struct RiskMonitor {
    assessments: Vec<FormatRisk>,
}

impl Default for RiskMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskMonitor {
    /// Create a new risk monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            assessments: Vec::new(),
        }
    }

    /// Add assessment
    pub fn add_assessment(&mut self, assessment: FormatRisk) {
        self.assessments.push(assessment);
    }

    /// Generate monitoring report
    #[must_use]
    pub fn generate_report(&self) -> MonitoringReport {
        let mut risk_distribution = HashMap::new();
        let mut high_risk_files = Vec::new();

        for assessment in &self.assessments {
            *risk_distribution.entry(assessment.risk_level).or_insert(0) += 1;

            if assessment.risk_level >= RiskLevel::High {
                high_risk_files.push(assessment.format.clone());
            }
        }

        MonitoringReport {
            risk_distribution,
            total_files: self.assessments.len(),
            high_risk_files,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get assessments by risk level
    #[must_use]
    pub fn get_by_risk_level(&self, level: RiskLevel) -> Vec<&FormatRisk> {
        self.assessments
            .iter()
            .filter(|a| a.risk_level == level)
            .collect()
    }

    /// Save report to JSON file
    ///
    /// # Errors
    ///
    /// Returns an error if save fails
    pub fn save_report(&self, path: &std::path::Path) -> Result<()> {
        let report = self.generate_report();
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_monitor() {
        let mut monitor = RiskMonitor::new();

        monitor.add_assessment(FormatRisk {
            format: "mkv".to_string(),
            risk_level: RiskLevel::None,
            factors: Vec::new(),
            recommendation: String::new(),
            timestamp: chrono::Utc::now(),
        });

        monitor.add_assessment(FormatRisk {
            format: "wmv".to_string(),
            risk_level: RiskLevel::Critical,
            factors: Vec::new(),
            recommendation: String::new(),
            timestamp: chrono::Utc::now(),
        });

        let report = monitor.generate_report();
        assert_eq!(report.total_files, 2);
        assert_eq!(report.high_risk_files.len(), 1);
    }
}
