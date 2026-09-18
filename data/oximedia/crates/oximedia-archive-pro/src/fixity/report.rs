//! Fixity verification reports

use super::FixityResult;
use serde::{Deserialize, Serialize};

/// Fixity status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixityStatus {
    /// All checks passed
    Healthy,
    /// Some checks failed
    Degraded,
    /// Many checks failed
    Critical,
}

/// Fixity report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixityReport {
    /// Check results
    pub results: Vec<FixityResult>,
    /// Overall status
    pub status: FixityStatus,
    /// Report timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FixityReport {
    /// Create a new fixity report from results
    #[must_use]
    pub fn from_results(results: Vec<FixityResult>) -> Self {
        let total = results.len();
        let failed = results.iter().filter(|r| !r.passed).count();

        let status = if failed == 0 {
            FixityStatus::Healthy
        } else if failed <= total / 2 {
            FixityStatus::Degraded
        } else {
            FixityStatus::Critical
        };

        Self {
            results,
            status,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get the number of passed checks
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get the number of failed checks
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.len() - self.passed_count()
    }

    /// Generate a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Fixity Check: {:?} - {} passed, {} failed out of {} total",
            self.status,
            self.passed_count(),
            self.failed_count(),
            self.results.len()
        )
    }

    /// Get failed results
    #[must_use]
    pub fn failed_results(&self) -> Vec<&FixityResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_healthy_report() {
        let results = vec![
            FixityResult {
                path: PathBuf::from("file1.mkv"),
                passed: true,
                expected_checksum: Some("abc".to_string()),
                actual_checksum: Some("abc".to_string()),
                timestamp: chrono::Utc::now(),
                error: None,
            },
            FixityResult {
                path: PathBuf::from("file2.mkv"),
                passed: true,
                expected_checksum: Some("def".to_string()),
                actual_checksum: Some("def".to_string()),
                timestamp: chrono::Utc::now(),
                error: None,
            },
        ];

        let report = FixityReport::from_results(results);
        assert_eq!(report.status, FixityStatus::Healthy);
        assert_eq!(report.passed_count(), 2);
        assert_eq!(report.failed_count(), 0);
    }

    #[test]
    fn test_degraded_report() {
        let results = vec![
            FixityResult {
                path: PathBuf::from("file1.mkv"),
                passed: true,
                expected_checksum: Some("abc".to_string()),
                actual_checksum: Some("abc".to_string()),
                timestamp: chrono::Utc::now(),
                error: None,
            },
            FixityResult {
                path: PathBuf::from("file2.mkv"),
                passed: false,
                expected_checksum: Some("def".to_string()),
                actual_checksum: Some("xyz".to_string()),
                timestamp: chrono::Utc::now(),
                error: None,
            },
        ];

        let report = FixityReport::from_results(results);
        assert_eq!(report.status, FixityStatus::Degraded);
        assert_eq!(report.failed_count(), 1);
    }
}
