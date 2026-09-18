//! Shot duration analysis.

// use crate::error::ShotResult; // unused
use crate::types::{Shot, ShotStatistics};

/// Shot duration analyzer.
pub struct DurationAnalyzer;

impl DurationAnalyzer {
    /// Create a new duration analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze shot durations and generate statistics.
    #[must_use]
    pub fn analyze(&self, shots: &[Shot]) -> ShotStatistics {
        if shots.is_empty() {
            return ShotStatistics::default();
        }

        let mut durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let total_shots = shots.len();
        let average_shot_duration = durations.iter().sum::<f64>() / total_shots as f64;
        let median_shot_duration = if total_shots % 2 == 0 {
            (durations[total_shots / 2 - 1] + durations[total_shots / 2]) / 2.0
        } else {
            durations[total_shots / 2]
        };
        let min_shot_duration = durations[0];
        let max_shot_duration = durations[total_shots - 1];

        // Calculate distributions
        let mut shot_type_map = std::collections::HashMap::new();
        let mut coverage_map = std::collections::HashMap::new();
        let mut transition_map = std::collections::HashMap::new();

        for shot in shots {
            *shot_type_map.entry(shot.shot_type).or_insert(0) += 1;
            *coverage_map.entry(shot.coverage).or_insert(0) += 1;
            *transition_map.entry(shot.transition).or_insert(0) += 1;
        }

        let shot_type_distribution = shot_type_map.into_iter().collect();
        let coverage_distribution = coverage_map.into_iter().collect();
        let transition_distribution = transition_map.into_iter().collect();

        ShotStatistics {
            total_shots,
            total_scenes: 0,
            average_shot_duration,
            median_shot_duration,
            min_shot_duration,
            max_shot_duration,
            shot_type_distribution,
            coverage_distribution,
            transition_distribution,
            average_shots_per_scene: 0.0,
        }
    }
}

impl Default for DurationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ShotType;
    use oximedia_core::types::{Rational, Timestamp};

    #[test]
    fn test_duration_analyzer_creation() {
        let _analyzer = DurationAnalyzer::new();
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = DurationAnalyzer::new();
        let stats = analyzer.analyze(&[]);
        assert_eq!(stats.total_shots, 0);
    }

    #[test]
    fn test_analyze_single_shot() {
        let analyzer = DurationAnalyzer::new();
        let shot = Shot {
            id: 1,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(60, Rational::new(1, 30)),
            shot_type: ShotType::MediumShot,
            angle: crate::types::CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: crate::types::CompositionAnalysis {
                rule_of_thirds: 0.5,
                symmetry: 0.5,
                balance: 0.5,
                leading_lines: 0.5,
                depth: 0.5,
            },
            coverage: crate::types::CoverageType::Master,
            confidence: 0.8,
            transition: crate::types::TransitionType::Cut,
        };

        let stats = analyzer.analyze(&[shot]);
        assert_eq!(stats.total_shots, 1);
        assert!(stats.average_shot_duration > 0.0);
    }
}
