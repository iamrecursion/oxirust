//! Vocal separation from music.

use super::sources::{SeparationResult, SourceSeparator};
use crate::{AnalysisConfig, Result};

/// Separate vocals from instrumental music.
///
/// Uses harmonic-percussive separation as a basis, assuming vocals
/// are primarily harmonic with specific frequency characteristics.
pub fn separate_vocals(
    samples: &[f32],
    sample_rate: f32,
    config: &AnalysisConfig,
) -> Result<SeparationResult> {
    let separator = SourceSeparator::new(config.clone());

    // Perform harmonic-percussive separation
    let hp_result = separator.separate_harmonic_percussive(samples, sample_rate)?;

    // Vocals are primarily in the harmonic component
    // Further processing could be done to isolate vocal frequencies (300-3400 Hz)
    Ok(SeparationResult {
        harmonic: hp_result.harmonic,     // Contains vocals
        percussive: hp_result.percussive, // Instruments without vocals
        residual: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocal_separation() {
        let config = AnalysisConfig::default();
        let samples = vec![0.1; 8192];
        let sample_rate = 44100.0;

        let result = separate_vocals(&samples, sample_rate, &config);
        assert!(result.is_ok());
    }
}
