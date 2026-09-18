//! Drum separation from music.

use super::sources::{SeparationResult, SourceSeparator};
use crate::{AnalysisConfig, Result};

/// Separate drums from music.
///
/// Drums are primarily percussive, so this extracts the percussive component.
pub fn separate_drums(
    samples: &[f32],
    sample_rate: f32,
    config: &AnalysisConfig,
) -> Result<SeparationResult> {
    let separator = SourceSeparator::new(config.clone());

    let hp_result = separator.separate_harmonic_percussive(samples, sample_rate)?;

    // Drums are in the percussive component
    Ok(SeparationResult {
        harmonic: vec![],
        percussive: hp_result.percussive,
        residual: hp_result.harmonic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drum_separation() {
        let config = AnalysisConfig::default();
        let samples = vec![0.1; 8192];
        let sample_rate = 44100.0;

        let result = separate_drums(&samples, sample_rate, &config);
        assert!(result.is_ok());
    }
}
