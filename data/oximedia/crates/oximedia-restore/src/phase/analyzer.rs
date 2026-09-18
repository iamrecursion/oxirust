//! Phase correlation analysis.

use crate::error::RestoreResult;

/// Phase correlation result.
#[derive(Debug, Clone)]
pub struct PhaseCorrelation {
    /// Correlation coefficient (-1.0 to 1.0).
    pub correlation: f32,
    /// Phase difference in radians.
    pub phase_difference: f32,
}

/// Phase analyzer.
#[derive(Debug, Clone)]
pub struct PhaseAnalyzer {
    window_size: usize,
}

impl PhaseAnalyzer {
    /// Create a new phase analyzer.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Analyze phase correlation between left and right channels.
    pub fn analyze(&self, left: &[f32], right: &[f32]) -> RestoreResult<PhaseCorrelation> {
        let len = left.len().min(right.len()).min(self.window_size);

        if len == 0 {
            return Ok(PhaseCorrelation {
                correlation: 0.0,
                phase_difference: 0.0,
            });
        }

        // Compute correlation
        let mut sum_lr = 0.0;
        let mut sum_ll = 0.0;
        let mut sum_rr = 0.0;

        for i in 0..len {
            sum_lr += left[i] * right[i];
            sum_ll += left[i] * left[i];
            sum_rr += right[i] * right[i];
        }

        let correlation = if sum_ll > f32::EPSILON && sum_rr > f32::EPSILON {
            sum_lr / (sum_ll * sum_rr).sqrt()
        } else {
            0.0
        };

        Ok(PhaseCorrelation {
            correlation,
            phase_difference: 0.0, // Simplified
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_analyzer() {
        let left = vec![1.0; 100];
        let right = vec![1.0; 100];

        let analyzer = PhaseAnalyzer::new(100);
        let result = analyzer
            .analyze(&left, &right)
            .expect("should succeed in test");

        assert!((result.correlation - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_out_of_phase() {
        let left = vec![1.0; 100];
        let right = vec![-1.0; 100];

        let analyzer = PhaseAnalyzer::new(100);
        let result = analyzer
            .analyze(&left, &right)
            .expect("should succeed in test");

        assert!((result.correlation + 1.0).abs() < 0.01);
    }
}
