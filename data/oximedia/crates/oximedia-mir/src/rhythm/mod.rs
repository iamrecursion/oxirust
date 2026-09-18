//! Rhythm feature extraction.

pub mod complexity;
pub mod pattern;
pub mod polyrhythm;

pub use complexity::RhythmComplexity;
pub use pattern::RhythmPattern;
pub use polyrhythm::{
    ExtendedRhythmResult, PolyrhythmAnalyzer, PolyrhythmPattern, SyncopationMetrics,
};

use crate::beat::onset::OnsetDetector;
use crate::types::RhythmResult;
use crate::utils::stft;
use crate::MirResult;

/// Rhythm analyzer.
pub struct RhythmAnalyzer {
    sample_rate: f32,
    hop_size: usize,
}

impl RhythmAnalyzer {
    /// Create a new rhythm analyzer.
    #[must_use]
    pub fn new(sample_rate: f32, hop_size: usize) -> Self {
        Self {
            sample_rate,
            hop_size,
        }
    }

    /// Analyze rhythm features.
    ///
    /// # Errors
    ///
    /// Returns error if rhythm analysis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, signal: &[f32]) -> MirResult<RhythmResult> {
        // Detect onsets
        let onset_detector = OnsetDetector::new(self.sample_rate, 2048, self.hop_size);
        let onset_times = onset_detector.detect(signal)?;

        // Compute onset strength envelope
        let onset_strength = self.compute_onset_strength(signal)?;

        // Extract rhythm patterns
        let pattern_extractor = RhythmPattern::new();
        let patterns = pattern_extractor.extract(&onset_times);

        // Compute rhythm complexity
        let complexity_analyzer = RhythmComplexity::new();
        let complexity = complexity_analyzer.compute(&onset_times);

        // Compute syncopation
        let syncopation = self.compute_syncopation(&onset_times);

        Ok(RhythmResult {
            onset_strength,
            onset_times,
            patterns,
            complexity,
            syncopation,
        })
    }

    /// Compute onset strength envelope.
    fn compute_onset_strength(&self, signal: &[f32]) -> MirResult<Vec<f32>> {
        let window_size = 2048;
        let frames = stft(signal, window_size, self.hop_size)?;

        let mut onset_strength = Vec::with_capacity(frames.len());
        let mut prev_mag = vec![0.0; window_size / 2 + 1];

        for frame in &frames {
            let mag = crate::utils::magnitude_spectrum(frame);

            // Spectral flux
            let flux: f32 = mag
                .iter()
                .zip(&prev_mag)
                .map(|(m, p)| (m - p).max(0.0))
                .sum();

            onset_strength.push(flux);
            prev_mag = mag;
        }

        Ok(onset_strength)
    }

    /// Compute syncopation measure.
    fn compute_syncopation(&self, onset_times: &[f32]) -> f32 {
        if onset_times.len() < 2 {
            return 0.0;
        }

        // Compute inter-onset intervals
        let mut intervals = Vec::new();
        for i in 1..onset_times.len() {
            intervals.push(onset_times[i] - onset_times[i - 1]);
        }

        // Syncopation based on interval irregularity
        let mean_interval = crate::utils::mean(&intervals);
        if mean_interval == 0.0 {
            return 0.0;
        }

        let variance: f32 = intervals
            .iter()
            .map(|i| (i - mean_interval).powi(2))
            .sum::<f32>()
            / intervals.len() as f32;

        let cv = variance.sqrt() / mean_interval;
        cv.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_analyzer_creation() {
        let analyzer = RhythmAnalyzer::new(44100.0, 512);
        assert_eq!(analyzer.sample_rate, 44100.0);
    }
}
