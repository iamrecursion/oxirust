//! Formant tracking over time.

use super::{FormantAnalyzer, FormantResult};
use crate::{AnalysisConfig, Result};

/// Formant tracker for tracking formants over time.
pub struct FormantTracker {
    analyzer: FormantAnalyzer,
    hop_size: usize,
}

impl FormantTracker {
    /// Create a new formant tracker.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let hop_size = config.hop_size;
        Self {
            analyzer: FormantAnalyzer::new(config),
            hop_size,
        }
    }

    /// Track formants over entire audio signal.
    pub fn track(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<FormantResult>> {
        let window_size = 1024;
        let mut results = Vec::new();

        let num_frames = (samples.len() - window_size) / self.hop_size + 1;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = (start + window_size).min(samples.len());

            if end - start < window_size {
                break;
            }

            let frame = &samples[start..end];
            let result = self.analyzer.analyze(frame, sample_rate)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get formant contours (F1, F2, F3, F4 over time).
    #[must_use]
    pub fn get_contours(&self, results: &[FormantResult]) -> Vec<Vec<f32>> {
        let num_formants = 4;
        let mut contours = vec![Vec::new(); num_formants];

        for result in results {
            for (i, &formant) in result.formants.iter().enumerate() {
                if i < num_formants {
                    contours[i].push(formant);
                }
            }
        }

        contours
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_tracking() {
        let config = AnalysisConfig::default();
        let tracker = FormantTracker::new(config);

        let sample_rate = 16000.0;
        let samples = vec![0.1; 16000]; // 1 second of audio

        // Tracking should return a result
        let _ = tracker.track(&samples, sample_rate);
    }
}
