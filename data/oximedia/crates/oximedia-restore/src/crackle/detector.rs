//! Crackle detection for old recordings.

use crate::error::RestoreResult;

/// Detected crackle region.
#[derive(Debug, Clone)]
pub struct Crackle {
    /// Start sample index.
    pub start: usize,
    /// End sample index.
    pub end: usize,
    /// Intensity.
    pub intensity: f32,
}

/// Crackle detector.
#[derive(Debug, Clone)]
pub struct CrackleDetector {
    threshold: f32,
    #[allow(dead_code)]
    min_duration: usize,
}

impl CrackleDetector {
    /// Create a new crackle detector.
    #[must_use]
    pub fn new(threshold: f32, min_duration: usize) -> Self {
        Self {
            threshold,
            min_duration,
        }
    }

    /// Detect crackle in samples.
    pub fn detect(&self, samples: &[f32]) -> RestoreResult<Vec<Crackle>> {
        if samples.len() < 3 {
            return Ok(Vec::new());
        }

        let mut crackles = Vec::new();

        // Use second derivative to detect rapid transients
        for i in 2..samples.len() - 2 {
            let second_derivative = samples[i + 1] - 2.0 * samples[i] + samples[i - 1];

            if second_derivative.abs() > self.threshold {
                crackles.push(Crackle {
                    start: i.saturating_sub(1),
                    end: i + 2,
                    intensity: second_derivative.abs(),
                });
            }
        }

        Ok(crackles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crackle_detector() {
        let mut samples = vec![0.0; 100];
        samples[50] = 1.0; // Sharp transient

        let detector = CrackleDetector::new(0.5, 1);
        let crackles = detector.detect(&samples).expect("should succeed in test");

        assert!(!crackles.is_empty());
    }
}
