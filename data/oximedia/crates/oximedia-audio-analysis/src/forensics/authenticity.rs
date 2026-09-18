//! Audio authenticity verification.

use crate::{AnalysisConfig, Result};

/// Authenticity verifier for detecting manipulated audio.
pub struct AuthenticityVerifier {
    config: AnalysisConfig,
    edit_detector: super::edit::EditDetector,
}

impl AuthenticityVerifier {
    /// Create a new authenticity verifier.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let edit_detector = super::edit::EditDetector::new(config.clone());

        Self {
            config,
            edit_detector,
        }
    }

    /// Verify audio authenticity.
    pub fn verify(&self, samples: &[f32], sample_rate: f32) -> Result<AuthenticityResult> {
        // Detect edits (cuts, splices)
        let edit_result = self.edit_detector.detect(samples, sample_rate)?;

        // Detect compression artifacts
        let compression = super::compression::detect_compression_history(samples, sample_rate);

        // Check noise consistency
        let noise_analyzer = super::noise::NoiseAnalyzer::new(self.config.clone());
        let noise_consistency = noise_analyzer.analyze_consistency(samples, sample_rate)?;

        // Compute authenticity score
        let mut authenticity_score = 1.0;

        // Reduce score for detected edits
        if edit_result.num_edits > 0 {
            authenticity_score *= 1.0 - (edit_result.num_edits as f32 * 0.1).min(0.5);
        }

        // Reduce score for inconsistent noise
        if !noise_consistency.is_consistent {
            authenticity_score *= 0.7;
        }

        // Multiple compression passes indicate manipulation
        if compression.num_compressions > 1 {
            authenticity_score *= 0.8;
        }

        let is_authentic = authenticity_score > 0.7;

        Ok(AuthenticityResult {
            is_authentic,
            authenticity_score,
            detected_edits: edit_result.num_edits,
            noise_consistent: noise_consistency.is_consistent,
            compression_count: compression.num_compressions,
        })
    }
}

/// Authenticity verification result.
#[derive(Debug, Clone)]
pub struct AuthenticityResult {
    /// Whether audio appears authentic
    pub is_authentic: bool,
    /// Authenticity score (0-1, higher = more authentic)
    pub authenticity_score: f32,
    /// Number of detected edits
    pub detected_edits: usize,
    /// Whether background noise is consistent
    pub noise_consistent: bool,
    /// Number of compression passes detected
    pub compression_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authenticity_verifier() {
        let config = AnalysisConfig::default();
        let verifier = AuthenticityVerifier::new(config);

        // Clean, unedited signal
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.3)
            .collect();

        let result = verifier.verify(&samples, sample_rate);
        assert!(result.is_ok());
    }
}
