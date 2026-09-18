//! Pacing analysis for shot sequences.

use crate::types::Shot;

/// Pacing analyzer.
pub struct PacingAnalyzer;

impl PacingAnalyzer {
    /// Create a new pacing analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze pacing of shots.
    #[must_use]
    pub fn analyze(&self, shots: &[Shot]) -> PacingAnalysis {
        if shots.is_empty() {
            return PacingAnalysis::default();
        }

        let durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();

        // Calculate variance in shot duration
        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance = durations
            .iter()
            .map(|d| (d - mean) * (d - mean))
            .sum::<f64>()
            / durations.len() as f64;

        // Calculate tempo (shots per minute)
        let total_duration: f64 = durations.iter().sum();
        let tempo = if total_duration > 0.0 {
            (shots.len() as f64 * 60.0) / total_duration
        } else {
            0.0
        };

        PacingAnalysis {
            tempo,
            variance,
            rhythm_score: self.calculate_rhythm_score(&durations),
        }
    }

    /// Calculate rhythm score (how regular the pacing is).
    fn calculate_rhythm_score(&self, durations: &[f64]) -> f32 {
        if durations.len() < 2 {
            return 0.0;
        }

        let mut changes = Vec::new();
        for i in 1..durations.len() {
            changes.push((durations[i] - durations[i - 1]).abs());
        }

        let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;

        // Lower variance in changes = more rhythmic
        (1.0 / (1.0 + mean_change)) as f32
    }
}

impl Default for PacingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pacing analysis results.
#[derive(Debug, Clone, Copy)]
pub struct PacingAnalysis {
    /// Tempo (shots per minute).
    pub tempo: f64,
    /// Variance in shot durations.
    pub variance: f64,
    /// Rhythm score (0.0 to 1.0).
    pub rhythm_score: f32,
}

impl Default for PacingAnalysis {
    fn default() -> Self {
        Self {
            tempo: 0.0,
            variance: 0.0,
            rhythm_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pacing_analyzer_creation() {
        let _analyzer = PacingAnalyzer::new();
    }
}
