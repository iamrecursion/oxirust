//! Rhythm pattern extraction.

use crate::types::RhythmPattern as RhythmPatternType;

/// Rhythm pattern extractor.
pub struct RhythmPattern;

impl RhythmPattern {
    /// Create a new rhythm pattern extractor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract rhythm patterns from onset times.
    #[must_use]
    pub fn extract(&self, onset_times: &[f32]) -> Vec<RhythmPatternType> {
        let mut patterns = Vec::new();

        // Find repeating interval patterns
        for window_size in [4, 8, 16] {
            if onset_times.len() < window_size {
                continue;
            }

            for i in 0..onset_times.len().saturating_sub(window_size) {
                let window = &onset_times[i..i + window_size];
                let pattern = self.analyze_window(window);

                if pattern.strength > 0.5 {
                    patterns.push(pattern);
                }
            }
        }

        patterns
    }

    /// Analyze a window of onsets for patterns.
    fn analyze_window(&self, window: &[f32]) -> RhythmPatternType {
        if window.len() < 2 {
            return RhythmPatternType {
                start: 0.0,
                duration: 0.0,
                description: "none".to_string(),
                strength: 0.0,
            };
        }

        let start = window[0];
        let duration = window[window.len() - 1] - window[0];

        // Compute interval regularity
        let mut intervals = Vec::new();
        for i in 1..window.len() {
            intervals.push(window[i] - window[i - 1]);
        }

        let mean_interval = crate::utils::mean(&intervals);
        let std_dev = crate::utils::std_dev(&intervals);

        // Pattern strength based on regularity
        let strength = if mean_interval > 0.0 {
            1.0 - (std_dev / mean_interval).min(1.0)
        } else {
            0.0
        };

        let description = if strength > 0.7 {
            "regular".to_string()
        } else if strength > 0.4 {
            "semi-regular".to_string()
        } else {
            "irregular".to_string()
        };

        RhythmPatternType {
            start,
            duration,
            description,
            strength,
        }
    }
}

impl Default for RhythmPattern {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_pattern_creation() {
        let _pattern = RhythmPattern::new();
    }

    #[test]
    fn test_extract_patterns() {
        let pattern = RhythmPattern::new();
        let onset_times = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let patterns = pattern.extract(&onset_times);
        assert!(!patterns.is_empty());
    }
}
