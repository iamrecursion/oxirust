//! Onset detection using spectral flux and high frequency content.

use crate::utils::{find_peaks, mean, stft};
use crate::MirResult;

/// Onset detector.
pub struct OnsetDetector {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl OnsetDetector {
    /// Create a new onset detector.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Detect onsets in audio signal.
    ///
    /// Returns onset times in seconds.
    ///
    /// # Errors
    ///
    /// Returns error if onset detection fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, signal: &[f32]) -> MirResult<Vec<f32>> {
        // Compute STFT
        let frames = stft(signal, self.window_size, self.hop_size)?;

        // Compute onset strength using multiple methods
        let spectral_flux = self.compute_spectral_flux(&frames);
        let hfc = self.compute_high_frequency_content(&frames);

        // Combine onset functions
        let onset_strength = self.combine_onset_functions(&spectral_flux, &hfc);

        // Find peaks in onset strength
        let peak_indices = find_peaks(&onset_strength, 3);

        // Convert frame indices to time
        let onset_times: Vec<f32> = peak_indices
            .iter()
            .map(|&idx| idx as f32 * self.hop_size as f32 / self.sample_rate)
            .collect();

        Ok(onset_times)
    }

    /// Compute spectral flux onset detection function.
    fn compute_spectral_flux(&self, frames: &[Vec<oxifft::Complex<f32>>]) -> Vec<f32> {
        let mut flux = Vec::with_capacity(frames.len());
        let mut prev_mag = vec![0.0; self.window_size / 2 + 1];

        for frame in frames {
            let mag = crate::utils::magnitude_spectrum(frame);

            // Positive spectral difference
            let frame_flux: f32 = mag
                .iter()
                .zip(&prev_mag)
                .map(|(m, p)| (m - p).max(0.0))
                .sum();

            flux.push(frame_flux);
            prev_mag = mag;
        }

        flux
    }

    /// Compute high frequency content onset detection function.
    fn compute_high_frequency_content(&self, frames: &[Vec<oxifft::Complex<f32>>]) -> Vec<f32> {
        frames
            .iter()
            .map(|frame| {
                let mag = crate::utils::magnitude_spectrum(frame);

                // Weight higher frequencies more
                mag.iter()
                    .enumerate()
                    .map(|(i, &m)| (i + 1) as f32 * m)
                    .sum()
            })
            .collect()
    }

    /// Combine multiple onset detection functions.
    fn combine_onset_functions(&self, flux: &[f32], hfc: &[f32]) -> Vec<f32> {
        // Normalize both functions
        let flux_norm = crate::utils::normalize(flux);
        let hfc_norm = crate::utils::normalize(hfc);

        // Weighted combination
        flux_norm
            .iter()
            .zip(&hfc_norm)
            .map(|(&f, &h)| 0.7 * f + 0.3 * h)
            .collect()
    }

    /// Apply adaptive thresholding to onset strength.
    #[allow(dead_code)]
    fn adaptive_threshold(&self, onset_strength: &[f32]) -> Vec<bool> {
        let window_size = 10;
        let delta = 0.1;

        onset_strength
            .iter()
            .enumerate()
            .map(|(i, &strength)| {
                let start = i.saturating_sub(window_size);
                let end = (i + window_size).min(onset_strength.len());

                let local_mean = mean(&onset_strength[start..end]);
                strength > local_mean + delta
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_detector_creation() {
        let detector = OnsetDetector::new(44100.0, 2048, 512);
        assert_eq!(detector.sample_rate, 44100.0);
    }

    #[test]
    fn test_detect_onsets_silence() {
        let detector = OnsetDetector::new(44100.0, 2048, 512);
        let signal = vec![0.0; 44100];
        let result = detector.detect(&signal);
        assert!(result.is_ok());
    }
}
