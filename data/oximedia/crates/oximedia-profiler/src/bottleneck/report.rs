//! Bottleneck reporting.

use super::classify::{BottleneckClassifier, BottleneckType};
use super::detect::Bottleneck;
use serde::{Deserialize, Serialize};

/// Bottleneck report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckReport {
    /// All detected bottlenecks.
    pub bottlenecks: Vec<(Bottleneck, BottleneckType)>,

    /// Number of critical bottlenecks.
    pub critical_count: usize,

    /// Number of significant bottlenecks.
    pub significant_count: usize,

    /// Summary.
    pub summary: String,
}

impl BottleneckReport {
    /// Create a new bottleneck report.
    pub fn new(bottlenecks: Vec<Bottleneck>) -> Self {
        let mut classified = Vec::new();
        let mut critical_count = 0;
        let mut significant_count = 0;

        for bottleneck in bottlenecks {
            let bottleneck_type = BottleneckClassifier::classify(&bottleneck);

            if bottleneck.is_critical() {
                critical_count += 1;
            }
            if bottleneck.is_significant() {
                significant_count += 1;
            }

            classified.push((bottleneck, bottleneck_type));
        }

        let summary = format!(
            "Found {} bottlenecks ({} critical, {} significant)",
            classified.len(),
            critical_count,
            significant_count
        );

        Self {
            bottlenecks: classified,
            critical_count,
            significant_count,
            summary,
        }
    }

    /// Get bottlenecks by type.
    pub fn by_type(&self, bottleneck_type: BottleneckType) -> Vec<&Bottleneck> {
        self.bottlenecks
            .iter()
            .filter(|(_, t)| *t == bottleneck_type)
            .map(|(b, _)| b)
            .collect()
    }

    /// Generate a detailed report.
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Bottleneck Analysis Report ===\n\n");
        report.push_str(&format!("{}\n\n", self.summary));

        if self.bottlenecks.is_empty() {
            report.push_str("No bottlenecks detected.\n");
            return report;
        }

        for (i, (bottleneck, bottleneck_type)) in self.bottlenecks.iter().enumerate() {
            let severity = if bottleneck.is_critical() {
                "CRITICAL"
            } else if bottleneck.is_significant() {
                "SIGNIFICANT"
            } else {
                "MINOR"
            };

            report.push_str(&format!(
                "{}. [{}] {}\n",
                i + 1,
                severity,
                bottleneck.description
            ));
            report.push_str(&format!("   Location: {}\n", bottleneck.location));
            report.push_str(&format!("   Time Impact: {:?}\n", bottleneck.time_impact));
            report.push_str(&format!(
                "   Impact: {:.2}%\n",
                bottleneck.impact_percentage
            ));
            report.push_str(&format!("   Type: {:?}\n", bottleneck_type));
            report.push_str(&format!("   {}\n", bottleneck_type.description()));

            if let Some(ref suggestion) = bottleneck.suggestion {
                report.push_str(&format!("   Suggestion: {}\n", suggestion));
            }

            let all_suggestions = BottleneckClassifier::get_all_suggestions(bottleneck);
            if all_suggestions.len() > 1 {
                report.push_str("   Other suggestions:\n");
                for suggestion in all_suggestions.iter().skip(1) {
                    report.push_str(&format!("     - {}\n", suggestion));
                }
            }

            report.push('\n');
        }

        report
    }

    /// Get a short summary.
    pub fn short_summary(&self) -> String {
        let mut summary = self.summary.clone();

        if self.critical_count > 0 {
            summary.push_str(&format!(
                "\nTop critical: {}",
                self.bottlenecks[0].0.location
            ));
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_bottleneck_report() {
        let bottlenecks = vec![
            Bottleneck::new(
                "Test 1".to_string(),
                "compute_function".to_string(),
                Duration::from_secs(1),
            )
            .with_impact(80.0),
            Bottleneck::new(
                "Test 2".to_string(),
                "memory_alloc".to_string(),
                Duration::from_millis(500),
            )
            .with_impact(20.0),
        ];

        let report = BottleneckReport::new(bottlenecks);
        assert_eq!(report.bottlenecks.len(), 2);
        assert_eq!(report.critical_count, 1);
    }

    #[test]
    fn test_bottleneck_report_by_type() {
        let bottlenecks = vec![
            Bottleneck::new(
                "Test 1".to_string(),
                "compute_function".to_string(),
                Duration::from_secs(1),
            ),
            Bottleneck::new(
                "Test 2".to_string(),
                "memory_alloc".to_string(),
                Duration::from_millis(500),
            ),
        ];

        let report = BottleneckReport::new(bottlenecks);
        let cpu_bottlenecks = report.by_type(BottleneckType::CPU);
        assert_eq!(cpu_bottlenecks.len(), 1);
    }

    #[test]
    fn test_detailed_report() {
        let bottlenecks = vec![Bottleneck::new(
            "Test".to_string(),
            "test_function".to_string(),
            Duration::from_secs(1),
        )
        .with_impact(50.0)];

        let report = BottleneckReport::new(bottlenecks);
        let detailed = report.detailed_report();

        assert!(detailed.contains("Bottleneck Analysis Report"));
        assert!(detailed.contains("test_function"));
    }
}
