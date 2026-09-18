//! Edit pattern analysis.

use crate::types::Shot;

/// Pattern analyzer.
pub struct PatternAnalyzer;

impl PatternAnalyzer {
    /// Create a new pattern analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze editing patterns in shots.
    #[must_use]
    pub fn analyze(&self, shots: &[Shot]) -> PatternAnalysis {
        if shots.is_empty() {
            return PatternAnalysis::default();
        }

        // Analyze shot-reverse-shot patterns
        let shot_reverse_shot_count = self.count_shot_reverse_shot(shots);

        // Analyze montage sequences (rapid cuts)
        let montage_sequences = self.detect_montage_sequences(shots);

        // Analyze coverage patterns
        let coverage_pattern = self.analyze_coverage_pattern(shots);

        PatternAnalysis {
            shot_reverse_shot_count,
            montage_sequences,
            coverage_pattern,
        }
    }

    /// Count shot-reverse-shot patterns.
    fn count_shot_reverse_shot(&self, shots: &[Shot]) -> usize {
        let mut count = 0;

        for i in 2..shots.len() {
            // Look for alternating singles
            if shots[i - 2].coverage == crate::types::CoverageType::Single
                && shots[i - 1].coverage == crate::types::CoverageType::Single
                && shots[i].coverage == crate::types::CoverageType::Single
                && shots[i - 2].angle == shots[i].angle
            {
                count += 1;
            }
        }

        count
    }

    /// Detect montage sequences (rapid cutting).
    fn detect_montage_sequences(&self, shots: &[Shot]) -> usize {
        let mut sequences = 0;
        let mut rapid_count = 0;

        for i in 1..shots.len() {
            let duration = shots[i].duration_seconds();

            if duration < 2.0 {
                rapid_count += 1;
            } else {
                if rapid_count >= 3 {
                    sequences += 1;
                }
                rapid_count = 0;
            }
        }

        if rapid_count >= 3 {
            sequences += 1;
        }

        sequences
    }

    /// Analyze coverage pattern (master-coverage-master, etc.).
    fn analyze_coverage_pattern(&self, shots: &[Shot]) -> String {
        if shots.is_empty() {
            return String::from("None");
        }

        let mut pattern = String::new();
        for shot in shots.iter().take(5.min(shots.len())) {
            pattern.push_str(match shot.coverage {
                crate::types::CoverageType::Master => "M",
                crate::types::CoverageType::Single => "S",
                crate::types::CoverageType::TwoShot => "T",
                _ => "O",
            });
        }

        pattern
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern analysis results.
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    /// Number of shot-reverse-shot patterns.
    pub shot_reverse_shot_count: usize,
    /// Number of montage sequences.
    pub montage_sequences: usize,
    /// Coverage pattern string.
    pub coverage_pattern: String,
}

impl Default for PatternAnalysis {
    fn default() -> Self {
        Self {
            shot_reverse_shot_count: 0,
            montage_sequences: 0,
            coverage_pattern: String::from("None"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_analyzer_creation() {
        let _analyzer = PatternAnalyzer::new();
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = PatternAnalyzer::new();
        let analysis = analyzer.analyze(&[]);
        assert_eq!(analysis.shot_reverse_shot_count, 0);
    }
}
