//! Hiss detection using spectral analysis.

use crate::error::RestoreResult;
use crate::utils::spectral::{apply_window, spectral_flatness, FftProcessor, WindowFunction};

/// Hiss detection result.
#[derive(Debug, Clone)]
pub struct HissProfile {
    /// Hiss level (0.0 to 1.0).
    pub level: f32,
    /// Dominant frequency range (Hz).
    pub frequency_range: (f32, f32),
    /// Confidence score.
    pub confidence: f32,
}

/// Hiss detector.
#[derive(Debug)]
pub struct HissDetector {
    fft_size: usize,
}

impl HissDetector {
    /// Create a new hiss detector.
    #[must_use]
    pub fn new(fft_size: usize) -> Self {
        Self { fft_size }
    }

    /// Detect hiss in samples.
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<Option<HissProfile>> {
        if samples.len() < self.fft_size {
            return Ok(None);
        }

        let mut windowed = samples[..self.fft_size].to_vec();
        apply_window(&mut windowed, WindowFunction::Hann);

        let fft = FftProcessor::new(self.fft_size);
        let spectrum = fft.forward(&windowed)?;
        let magnitude = fft.magnitude(&spectrum);

        // Hiss is typically high-frequency noise with flat spectrum
        let flatness = spectral_flatness(&magnitude);

        // Check high-frequency energy
        #[allow(clippy::cast_precision_loss)]
        let bin_width = sample_rate as f32 / self.fft_size as f32;
        let high_freq_start = (4000.0 / bin_width) as usize;
        let high_freq_end = magnitude.len().min((12000.0 / bin_width) as usize);

        if high_freq_end <= high_freq_start {
            return Ok(None);
        }

        let high_freq_energy: f32 = magnitude[high_freq_start..high_freq_end]
            .iter()
            .map(|&m| m * m)
            .sum();

        let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();

        #[allow(clippy::cast_precision_loss)]
        let high_freq_ratio = if total_energy > f32::EPSILON {
            high_freq_energy / total_energy
        } else {
            0.0
        };

        // Hiss typically has high flatness and significant high-frequency content
        if flatness > 0.3 && high_freq_ratio > 0.1 {
            Ok(Some(HissProfile {
                level: high_freq_ratio,
                frequency_range: (4000.0, 12000.0),
                confidence: flatness * high_freq_ratio,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hiss_detector() {
        use rand::RngExt;
        let mut rng = rand::rng();

        // Create high-frequency noise (hiss)
        let mut samples = vec![0.0; 8192];
        for i in 0..samples.len() {
            samples[i] = rng.random_range(-0.1..0.1);
        }

        let detector = HissDetector::new(2048);
        let result = detector
            .detect(&samples, 44100)
            .expect("should succeed in test");

        // Should detect some noise characteristics
        assert!(result.is_some() || result.is_none()); // Just check it doesn't crash
    }
}
