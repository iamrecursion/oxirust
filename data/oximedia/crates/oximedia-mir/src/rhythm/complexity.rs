//! Rhythmic complexity measurement.

/// Rhythmic complexity analyzer.
pub struct RhythmComplexity;

impl RhythmComplexity {
    /// Create a new rhythm complexity analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute rhythmic complexity from onset times.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 2 {
            return 0.0;
        }

        // Compute inter-onset intervals
        let mut intervals = Vec::new();
        for i in 1..onset_times.len() {
            intervals.push(onset_times[i] - onset_times[i - 1]);
        }

        // Complexity based on:
        // 1. Number of unique intervals (normalized)
        let mut unique_intervals = intervals.clone();
        unique_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_intervals.dedup_by(|a, b| (*a - *b).abs() < 0.01);

        let uniqueness = unique_intervals.len() as f32 / intervals.len() as f32;

        // 2. Interval variance
        let mean_interval = crate::utils::mean(&intervals);
        let variance: f32 = intervals
            .iter()
            .map(|i| (i - mean_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;

        let normalized_variance = if mean_interval > 0.0 {
            (variance.sqrt() / mean_interval).min(1.0)
        } else {
            0.0
        };

        // Combined complexity score
        (uniqueness * 0.6 + normalized_variance * 0.4).min(1.0)
    }
}

impl Default for RhythmComplexity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_complexity_creation() {
        let _complexity = RhythmComplexity::new();
    }

    #[test]
    fn test_compute_complexity() {
        let complexity = RhythmComplexity::new();
        let onset_times = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let result = complexity.compute(&onset_times);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_compute_complex_rhythm() {
        let complexity = RhythmComplexity::new();
        let onset_times = vec![0.0, 0.3, 0.7, 1.2, 1.4, 2.0];
        let result = complexity.compute(&onset_times);
        assert!(result > 0.0);
    }
}
