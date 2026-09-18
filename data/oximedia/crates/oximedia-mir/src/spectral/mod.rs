//! Spectral feature extraction.

pub mod centroid;
pub mod contrast;
pub mod flux;
pub mod rolloff;

pub use centroid::SpectralCentroid;
pub use contrast::SpectralContrast;
pub use flux::SpectralFlux;
pub use rolloff::SpectralRolloff;

use crate::types::SpectralResult;
use crate::utils::{mean, stft};
use crate::MirResult;

/// Spectral analyzer.
pub struct SpectralAnalyzer {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Analyze spectral features.
    ///
    /// # Errors
    ///
    /// Returns error if spectral analysis fails.
    pub fn analyze(&self, signal: &[f32]) -> MirResult<SpectralResult> {
        let frames = stft(signal, self.window_size, self.hop_size)?;

        let centroid_analyzer = SpectralCentroid::new(self.sample_rate, self.window_size);
        let rolloff_analyzer = SpectralRolloff::new(self.sample_rate, self.window_size);
        let flux_analyzer = SpectralFlux::new();
        let contrast_analyzer = SpectralContrast::new(self.sample_rate, self.window_size);

        let mut centroid = Vec::new();
        let mut rolloff = Vec::new();
        let mut flux = Vec::new();
        let mut contrast = Vec::new();

        let mut prev_mag = vec![0.0; self.window_size / 2 + 1];

        for frame in &frames {
            let mag = crate::utils::magnitude_spectrum(frame);

            centroid.push(centroid_analyzer.compute(&mag));
            rolloff.push(rolloff_analyzer.compute(&mag));
            flux.push(flux_analyzer.compute(&mag, &prev_mag));
            contrast.push(contrast_analyzer.compute(&mag));

            prev_mag = mag;
        }

        let mean_centroid = mean(&centroid);
        let mean_rolloff = mean(&rolloff);
        let mean_flux = mean(&flux);

        Ok(SpectralResult {
            centroid,
            rolloff,
            flux,
            contrast,
            mean_centroid,
            mean_rolloff,
            mean_flux,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analyzer_creation() {
        let analyzer = SpectralAnalyzer::new(44100.0, 2048, 512);
        assert_eq!(analyzer.sample_rate, 44100.0);
    }
}
