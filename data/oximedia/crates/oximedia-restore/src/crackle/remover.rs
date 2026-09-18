//! Crackle removal using morphological filtering.

use crate::crackle::detector::Crackle;
use crate::error::RestoreResult;

/// Crackle remover.
#[derive(Debug, Clone)]
pub struct CrackleRemover {
    window_size: usize,
}

impl CrackleRemover {
    /// Create a new crackle remover.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Remove crackle from samples.
    pub fn remove(&self, samples: &[f32], crackles: &[Crackle]) -> RestoreResult<Vec<f32>> {
        let mut output = samples.to_vec();

        for crackle in crackles {
            if crackle.start >= crackle.end || crackle.end > samples.len() {
                continue;
            }

            // Use median filtering to remove crackle
            for i in crackle.start..crackle.end {
                let start = i.saturating_sub(self.window_size / 2);
                let end = (i + self.window_size / 2 + 1).min(samples.len());

                let mut window: Vec<f32> = samples[start..end].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if window.len() % 2 == 0 {
                    (window[window.len() / 2 - 1] + window[window.len() / 2]) / 2.0
                } else {
                    window[window.len() / 2]
                };

                output[i] = median;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crackle_remover() {
        let mut samples = vec![0.0; 100];
        samples[50] = 1.0;

        let crackle = Crackle {
            start: 50,
            end: 51,
            intensity: 1.0,
        };

        let remover = CrackleRemover::new(5);
        let output = remover
            .remove(&samples, &[crackle])
            .expect("should succeed in test");

        assert_eq!(output.len(), samples.len());
        assert!(output[50].abs() < 0.5);
    }
}
