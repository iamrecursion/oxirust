//! Phase alignment correction.

use crate::error::RestoreResult;

/// Phase corrector.
#[derive(Debug, Clone)]
pub struct PhaseCorrector {
    invert_threshold: f32,
}

impl PhaseCorrector {
    /// Create a new phase corrector.
    #[must_use]
    pub fn new(invert_threshold: f32) -> Self {
        Self { invert_threshold }
    }

    /// Correct phase issues.
    ///
    /// Inverts right channel if correlation is negative.
    pub fn correct(&self, left: &[f32], right: &[f32]) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let len = left.len().min(right.len());

        // Compute correlation
        let mut correlation = 0.0;
        let mut sum_ll = 0.0;
        let mut sum_rr = 0.0;

        for i in 0..len {
            correlation += left[i] * right[i];
            sum_ll += left[i] * left[i];
            sum_rr += right[i] * right[i];
        }

        if sum_ll > f32::EPSILON && sum_rr > f32::EPSILON {
            correlation /= (sum_ll * sum_rr).sqrt();
        }

        let mut corrected_right = right.to_vec();

        // If correlation is very negative, invert phase
        if correlation < -self.invert_threshold {
            for sample in &mut corrected_right {
                *sample = -*sample;
            }
        }

        Ok((left.to_vec(), corrected_right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_corrector() {
        let left = vec![1.0; 100];
        let right = vec![-1.0; 100]; // Out of phase

        let corrector = PhaseCorrector::new(0.5);
        let (_, corrected_right) = corrector
            .correct(&left, &right)
            .expect("should succeed in test");

        // Should be inverted back
        assert!((corrected_right[0] - 1.0).abs() < 0.01);
    }
}
