//! Melodic contour analysis.

/// Melodic contour analyzer.
pub struct ContourAnalyzer;

impl ContourAnalyzer {
    /// Create a new contour analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyze melodic contour patterns.
    #[must_use]
    pub fn analyze(&self, pitch_contour: &[f32]) -> Vec<String> {
        let mut patterns = Vec::new();

        // Identify ascending and descending passages
        for window in pitch_contour.windows(8) {
            let pattern = self.classify_contour(window);
            if !pattern.is_empty() {
                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Classify contour pattern.
    fn classify_contour(&self, pitches: &[f32]) -> String {
        let valid_pitches: Vec<f32> = pitches.iter().copied().filter(|&p| p > 0.0).collect();

        if valid_pitches.len() < 3 {
            return String::new();
        }

        // Check for ascending pattern
        let ascending = valid_pitches.windows(2).filter(|w| w[1] > w[0]).count();
        let descending = valid_pitches.windows(2).filter(|w| w[1] < w[0]).count();

        if ascending > descending * 2 {
            "ascending".to_string()
        } else if descending > ascending * 2 {
            "descending".to_string()
        } else {
            "undulating".to_string()
        }
    }
}

impl Default for ContourAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_analyzer_creation() {
        let _analyzer = ContourAnalyzer::new();
    }
}
