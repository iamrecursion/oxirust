//! Editing rhythm analysis.

use crate::types::Shot;

/// Rhythm analyzer.
pub struct RhythmAnalyzer;

impl RhythmAnalyzer {
    /// Create a new rhythm analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze editing rhythm.
    #[must_use]
    pub fn analyze(&self, shots: &[Shot]) -> RhythmAnalysis {
        if shots.is_empty() {
            return RhythmAnalysis::default();
        }

        let durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();

        // Calculate beat (average cut rate)
        let total_duration: f64 = durations.iter().sum();
        let beat = if total_duration > 0.0 {
            shots.len() as f64 / total_duration
        } else {
            0.0
        };

        // Calculate regularity (how consistent the rhythm is)
        let mean = total_duration / shots.len() as f64;
        let variance = durations
            .iter()
            .map(|d| (d - mean) * (d - mean))
            .sum::<f64>()
            / shots.len() as f64;
        let regularity = (1.0 / (1.0 + variance)) as f32;

        // Detect accelerations and decelerations
        let mut accelerations = 0;
        let mut decelerations = 0;

        for i in 1..durations.len() {
            if durations[i] < durations[i - 1] * 0.8 {
                accelerations += 1;
            } else if durations[i] > durations[i - 1] * 1.2 {
                decelerations += 1;
            }
        }

        RhythmAnalysis {
            beat,
            regularity,
            accelerations,
            decelerations,
        }
    }
}

impl Default for RhythmAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rhythm analysis results.
#[derive(Debug, Clone, Copy)]
pub struct RhythmAnalysis {
    /// Beat (cuts per second).
    pub beat: f64,
    /// Regularity score (0.0 to 1.0).
    pub regularity: f32,
    /// Number of accelerations.
    pub accelerations: usize,
    /// Number of decelerations.
    pub decelerations: usize,
}

impl Default for RhythmAnalysis {
    fn default() -> Self {
        Self {
            beat: 0.0,
            regularity: 0.0,
            accelerations: 0,
            decelerations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_analyzer_creation() {
        let _analyzer = RhythmAnalyzer::new();
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = RhythmAnalyzer::new();
        let analysis = analyzer.analyze(&[]);
        assert!((analysis.beat - 0.0).abs() < f64::EPSILON);
    }
}
