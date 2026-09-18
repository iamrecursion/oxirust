//! Instrumental separation from vocals.

use super::sources::{SeparationResult, SourceSeparator};
use crate::{AnalysisConfig, Result};

/// Separate instrumental from vocals.
///
/// Complementary to vocal separation.
pub fn separate_instrumental(
    samples: &[f32],
    sample_rate: f32,
    config: &AnalysisConfig,
) -> Result<SeparationResult> {
    let separator = SourceSeparator::new(config.clone());

    let hp_result = separator.separate_harmonic_percussive(samples, sample_rate)?;

    // Instrumental is both harmonic and percussive parts minus vocals
    Ok(SeparationResult {
        harmonic: hp_result.harmonic,
        percussive: hp_result.percussive,
        residual: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrumental_separation() {
        let config = AnalysisConfig::default();
        let samples = vec![0.1; 8192];
        let sample_rate = 44100.0;

        let result = separate_instrumental(&samples, sample_rate, &config);
        assert!(result.is_ok());
    }
}
