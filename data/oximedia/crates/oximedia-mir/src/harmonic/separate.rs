//! Harmonic-percussive source separation.

use crate::utils::stft;
use crate::MirResult;

/// Harmonic-percussive separator using median filtering.
pub struct HarmonicSeparator {
    #[allow(dead_code)]
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl HarmonicSeparator {
    /// Create a new harmonic-percussive separator.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Separate signal into harmonic and percussive components.
    ///
    /// Returns (`harmonic_energy`, `percussive_energy`) over time.
    ///
    /// # Errors
    ///
    /// Returns error if separation fails.
    pub fn separate(&self, signal: &[f32]) -> MirResult<(Vec<f32>, Vec<f32>)> {
        let frames = stft(signal, self.window_size, self.hop_size)?;

        // Build magnitude spectrogram
        let spectrogram: Vec<Vec<f32>> = frames
            .iter()
            .map(|frame| crate::utils::magnitude_spectrum(frame))
            .collect();

        // Apply median filtering (simplified version)
        let (harmonic_spec, percussive_spec) = self.median_filter_separation(&spectrogram);

        // Compute energy over time for each component
        let harmonic_energy: Vec<f32> = harmonic_spec
            .iter()
            .map(|frame| frame.iter().map(|m| m * m).sum())
            .collect();

        let percussive_energy: Vec<f32> = percussive_spec
            .iter()
            .map(|frame| frame.iter().map(|m| m * m).sum())
            .collect();

        Ok((harmonic_energy, percussive_energy))
    }

    /// Separate using median filtering (simplified).
    fn median_filter_separation(&self, spectrogram: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let num_frames = spectrogram.len();
        if num_frames == 0 {
            return (Vec::new(), Vec::new());
        }

        let num_bins = spectrogram[0].len();

        let mut harmonic_spec = vec![vec![0.0; num_bins]; num_frames];
        let mut percussive_spec = vec![vec![0.0; num_bins]; num_frames];

        // Simplified separation: use frequency continuity for harmonic
        for (frame_idx, frame) in spectrogram.iter().enumerate() {
            for (bin_idx, &magnitude) in frame.iter().enumerate() {
                // Harmonic: stable across time
                let temporal_stability = if frame_idx > 0 && frame_idx < num_frames - 1 {
                    let prev = spectrogram[frame_idx - 1][bin_idx];
                    let next = spectrogram[frame_idx + 1][bin_idx];
                    let avg = (prev + next) / 2.0;
                    1.0 - ((magnitude - avg).abs() / (magnitude + avg + 1e-10)).min(1.0)
                } else {
                    0.5
                };

                // Percussive: stable across frequency
                let spectral_stability = if bin_idx > 0 && bin_idx < num_bins - 1 {
                    let prev = frame[bin_idx - 1];
                    let next = frame[bin_idx + 1];
                    let avg = (prev + next) / 2.0;
                    1.0 - ((magnitude - avg).abs() / (magnitude + avg + 1e-10)).min(1.0)
                } else {
                    0.5
                };

                // Soft masking
                harmonic_spec[frame_idx][bin_idx] = magnitude * temporal_stability;
                percussive_spec[frame_idx][bin_idx] = magnitude * spectral_stability;
            }
        }

        (harmonic_spec, percussive_spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_separator_creation() {
        let separator = HarmonicSeparator::new(44100.0, 2048, 512);
        assert_eq!(separator.sample_rate, 44100.0);
    }
}
