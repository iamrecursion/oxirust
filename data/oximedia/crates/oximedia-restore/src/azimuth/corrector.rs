//! Azimuth error correction for tape recordings.

use crate::error::RestoreResult;

/// Azimuth corrector configuration.
#[derive(Debug, Clone)]
pub struct AzimuthCorrectorConfig {
    /// Maximum delay in samples to search.
    pub max_delay: usize,
    /// Window size for correlation.
    pub window_size: usize,
}

impl Default for AzimuthCorrectorConfig {
    fn default() -> Self {
        Self {
            max_delay: 10,
            window_size: 1024,
        }
    }
}

/// Azimuth error corrector.
///
/// Corrects tape azimuth errors by aligning left and right channels.
#[derive(Debug, Clone)]
pub struct AzimuthCorrector {
    config: AzimuthCorrectorConfig,
}

impl AzimuthCorrector {
    /// Create a new azimuth corrector.
    #[must_use]
    pub fn new(config: AzimuthCorrectorConfig) -> Self {
        Self { config }
    }

    /// Detect azimuth error between left and right channels.
    ///
    /// Returns the optimal delay in samples (positive = left ahead, negative = right ahead).
    #[must_use]
    pub fn detect_delay(&self, left: &[f32], right: &[f32]) -> isize {
        let len = left.len().min(right.len()).min(self.config.window_size);
        if len == 0 {
            return 0;
        }

        let mut max_corr = f32::NEG_INFINITY;
        let mut best_delay = 0;

        for delay in -(self.config.max_delay as isize)..=(self.config.max_delay as isize) {
            let corr = self.compute_correlation(left, right, delay, len);
            if corr > max_corr {
                max_corr = corr;
                best_delay = delay;
            }
        }

        best_delay
    }

    /// Compute cross-correlation for a given delay.
    fn compute_correlation(&self, left: &[f32], right: &[f32], delay: isize, len: usize) -> f32 {
        let mut corr = 0.0;
        let mut count = 0;

        for i in 0..len {
            #[allow(clippy::cast_possible_wrap)]
            let j = i as isize + delay;
            if j >= 0 && (j as usize) < right.len() {
                corr += left[i] * right[j as usize];
                count += 1;
            }
        }

        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let result = corr / count as f32;
            result
        } else {
            0.0
        }
    }

    /// Correct azimuth error by aligning channels.
    pub fn correct(&self, left: &[f32], right: &[f32]) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let delay = self.detect_delay(left, right);

        let mut corrected_left = left.to_vec();
        let mut corrected_right = right.to_vec();

        if delay > 0 {
            // Left is ahead, delay it
            corrected_left = vec![0.0; delay as usize];
            corrected_left.extend_from_slice(left);
            corrected_left.truncate(left.len());
        } else if delay < 0 {
            // Right is ahead, delay it
            let abs_delay = (-delay) as usize;
            corrected_right = vec![0.0; abs_delay];
            corrected_right.extend_from_slice(right);
            corrected_right.truncate(right.len());
        }

        Ok((corrected_left, corrected_right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azimuth_corrector() {
        let left: Vec<f32> = (0..1000)
            .map(|i| {
                use std::f32::consts::PI;
                (2.0 * PI * 440.0 * i as f32 / 44100.0).sin()
            })
            .collect();

        // Right channel delayed by 5 samples
        let mut right = vec![0.0; 5];
        right.extend_from_slice(&left[..995]);

        let corrector = AzimuthCorrector::new(AzimuthCorrectorConfig::default());
        let delay = corrector.detect_delay(&left, &right);

        assert_eq!(delay, 5);
    }

    #[test]
    fn test_correct() {
        let left = vec![1.0; 100];
        let mut right = vec![0.0; 3];
        right.extend(vec![1.0; 97]);

        let corrector = AzimuthCorrector::new(AzimuthCorrectorConfig::default());
        let (corrected_left, corrected_right) = corrector
            .correct(&left, &right)
            .expect("should succeed in test");

        assert_eq!(corrected_left.len(), left.len());
        assert_eq!(corrected_right.len(), right.len());
    }
}
