//! DC offset removal using high-pass filtering.

use crate::error::RestoreResult;

/// DC offset remover.
///
/// Removes DC bias using a first-order high-pass filter.
#[derive(Debug, Clone)]
pub struct DcRemover {
    cutoff_hz: f32,
    sample_rate: u32,
    prev_input: f32,
    prev_output: f32,
}

impl DcRemover {
    /// Create a new DC remover.
    ///
    /// # Arguments
    ///
    /// * `cutoff_hz` - Cutoff frequency in Hz (typically 5-20 Hz)
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(cutoff_hz: f32, sample_rate: u32) -> Self {
        Self {
            cutoff_hz,
            sample_rate,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    /// Process samples to remove DC offset.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    ///
    /// # Returns
    ///
    /// Samples with DC offset removed.
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        use std::f32::consts::PI;

        // First-order high-pass filter coefficient
        #[allow(clippy::cast_precision_loss)]
        let rc = 1.0 / (2.0 * PI * self.cutoff_hz);
        #[allow(clippy::cast_precision_loss)]
        let dt = 1.0 / self.sample_rate as f32;
        let alpha = rc / (rc + dt);

        let mut output = Vec::with_capacity(samples.len());

        for &sample in samples {
            let filtered = alpha * (self.prev_output + sample - self.prev_input);
            output.push(filtered);
            self.prev_input = sample;
            self.prev_output = filtered;
        }

        Ok(output)
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}

/// Remove DC offset from samples (simple method).
///
/// Subtracts the mean value from all samples.
///
/// # Arguments
///
/// * `samples` - Input samples
///
/// # Returns
///
/// Samples with DC offset removed.
#[must_use]
pub fn remove_dc_simple(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|&s| s - mean).collect()
}

/// Detect DC offset in samples.
///
/// # Arguments
///
/// * `samples` - Input samples
///
/// # Returns
///
/// DC offset value.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_dc_offset(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    samples.iter().sum::<f32>() / samples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_remover() {
        let mut remover = DcRemover::new(10.0, 44100);

        // Create signal with DC offset
        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                use std::f32::consts::PI;
                0.5 + (2.0 * PI * 440.0 * i as f32 / 44100.0).sin()
            })
            .collect();

        let filtered = remover.process(&samples).expect("should succeed in test");
        assert_eq!(filtered.len(), samples.len());

        // DC offset should be reduced
        let original_dc = detect_dc_offset(&samples);
        let filtered_dc = detect_dc_offset(&filtered);
        assert!(filtered_dc.abs() < original_dc.abs());
    }

    #[test]
    fn test_remove_dc_simple() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = remove_dc_simple(&samples);

        let dc = detect_dc_offset(&filtered);
        assert!(dc.abs() < 1e-6);
    }

    #[test]
    fn test_detect_dc_offset() {
        let samples = vec![1.0, 1.0, 1.0, 1.0];
        let dc = detect_dc_offset(&samples);
        assert!((dc - 1.0).abs() < 1e-6);

        let samples = vec![-0.5, -0.5, -0.5];
        let dc = detect_dc_offset(&samples);
        assert!((dc + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut remover = DcRemover::new(10.0, 44100);
        let samples = vec![1.0; 100];
        let _ = remover.process(&samples).expect("should succeed in test");

        remover.reset();
        assert_eq!(remover.prev_input, 0.0);
        assert_eq!(remover.prev_output, 0.0);
    }
}
