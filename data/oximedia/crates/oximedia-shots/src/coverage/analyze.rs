//! Coverage analysis for shot sequences.

use crate::types::{CoverageType, Shot};

/// Coverage analyzer.
pub struct CoverageAnalyzer;

impl CoverageAnalyzer {
    /// Create a new coverage analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze coverage patterns in shots.
    #[must_use]
    pub fn analyze(&self, shots: &[Shot]) -> CoverageReport {
        if shots.is_empty() {
            return CoverageReport::default();
        }

        let mut coverage_counts = std::collections::HashMap::new();
        for shot in shots {
            *coverage_counts.entry(shot.coverage).or_insert(0) += 1;
        }

        let total = shots.len();
        let master_shots = *coverage_counts.get(&CoverageType::Master).unwrap_or(&0);
        let single_shots = *coverage_counts.get(&CoverageType::Single).unwrap_or(&0);
        let two_shots = *coverage_counts.get(&CoverageType::TwoShot).unwrap_or(&0);

        CoverageReport {
            total_shots: total,
            master_shots,
            single_shots,
            two_shots,
            coverage_distribution: coverage_counts,
        }
    }
}

impl Default for CoverageAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Coverage analysis report.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Total number of shots.
    pub total_shots: usize,
    /// Number of master shots.
    pub master_shots: usize,
    /// Number of single shots.
    pub single_shots: usize,
    /// Number of two-shots.
    pub two_shots: usize,
    /// Coverage distribution.
    pub coverage_distribution: std::collections::HashMap<CoverageType, usize>,
}

impl Default for CoverageReport {
    fn default() -> Self {
        Self {
            total_shots: 0,
            master_shots: 0,
            single_shots: 0,
            two_shots: 0,
            coverage_distribution: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_analyzer_creation() {
        let _analyzer = CoverageAnalyzer::new();
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = CoverageAnalyzer::new();
        let report = analyzer.analyze(&[]);
        assert_eq!(report.total_shots, 0);
    }
}
