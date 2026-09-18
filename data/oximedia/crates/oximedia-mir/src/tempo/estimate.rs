//! BPM estimation with confidence scoring.

use crate::{MirError, MirResult};

/// BPM estimator with confidence scoring.
#[allow(dead_code)]
pub struct BpmEstimator {
    sample_rate: f32,
    min_bpm: f32,
    max_bpm: f32,
}

impl BpmEstimator {
    /// Create a new BPM estimator.
    #[must_use]
    pub fn new(sample_rate: f32, min_bpm: f32, max_bpm: f32) -> Self {
        Self {
            sample_rate,
            min_bpm,
            max_bpm,
        }
    }

    /// Estimate BPM from inter-onset intervals.
    ///
    /// # Errors
    ///
    /// Returns error if intervals are invalid.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_from_intervals(&self, intervals: &[f32]) -> MirResult<(f32, f32)> {
        if intervals.is_empty() {
            return Err(MirError::InsufficientData(
                "No intervals provided".to_string(),
            ));
        }

        // Convert intervals to BPM
        let bpms: Vec<f32> = intervals
            .iter()
            .filter(|&&i| i > 0.0)
            .map(|&interval| 60.0 / interval)
            .filter(|&bpm| bpm >= self.min_bpm && bpm <= self.max_bpm)
            .collect();

        if bpms.is_empty() {
            return Err(MirError::AnalysisFailed(
                "No valid BPM estimates".to_string(),
            ));
        }

        // Find most common BPM (clustering approach)
        let (bpm, confidence) = self.find_dominant_bpm(&bpms);

        Ok((bpm, confidence))
    }

    /// Find dominant BPM using simple clustering.
    fn find_dominant_bpm(&self, bpms: &[f32]) -> (f32, f32) {
        // Create histogram bins
        let bin_size = 2.0; // 2 BPM bins
        let num_bins = ((self.max_bpm - self.min_bpm) / bin_size).ceil() as usize;
        let mut bins = vec![0_usize; num_bins];
        let mut bin_sums = vec![0.0_f32; num_bins];

        for &bpm in bpms {
            let bin = ((bpm - self.min_bpm) / bin_size).floor() as usize;
            if bin < num_bins {
                bins[bin] += 1;
                bin_sums[bin] += bpm;
            }
        }

        // Find bin with most votes
        let (max_bin, &max_count) = bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .unwrap_or((0, &0));

        if max_count == 0 {
            return (self.min_bpm, 0.0);
        }

        let dominant_bpm = bin_sums[max_bin] / max_count as f32;
        let confidence = max_count as f32 / bpms.len() as f32;

        (dominant_bpm, confidence)
    }

    /// Refine BPM estimate using phase information.
    ///
    /// # Errors
    ///
    /// Returns error if refinement fails.
    pub fn refine_with_phase(
        &self,
        initial_bpm: f32,
        onset_times: &[f32],
    ) -> MirResult<(f32, f32)> {
        if onset_times.len() < 2 {
            return Err(MirError::InsufficientData(
                "Need at least 2 onset times".to_string(),
            ));
        }

        let beat_period = 60.0 / initial_bpm;

        // Try different phases and find the one that best aligns with onsets
        let best_bpm = initial_bpm;
        let mut best_score = 0.0;

        for phase_offset in (0..100).map(|i| i as f32 * beat_period / 100.0) {
            let mut score = 0.0;

            for &onset_time in onset_times {
                // Find closest beat time
                let beat_number = ((onset_time - phase_offset) / beat_period).round();
                let beat_time = phase_offset + beat_number * beat_period;
                let error = (onset_time - beat_time).abs();

                // Gaussian weighting
                let tolerance: f32 = 0.070; // 70ms tolerance
                score += (-error.powi(2) / (2.0 * tolerance.powi(2))).exp();
            }

            if score > best_score {
                best_score = score;
            }
        }

        let confidence = (best_score / onset_times.len() as f32).clamp(0.0, 1.0);

        Ok((best_bpm, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpm_estimator_creation() {
        let estimator = BpmEstimator::new(44100.0, 60.0, 200.0);
        assert_eq!(estimator.sample_rate, 44100.0);
    }

    #[test]
    fn test_estimate_from_intervals() {
        let estimator = BpmEstimator::new(44100.0, 60.0, 200.0);
        // 120 BPM = 0.5 second intervals
        let intervals = vec![0.5, 0.5, 0.5, 0.5];
        let result = estimator.estimate_from_intervals(&intervals);
        assert!(result.is_ok());
        let (bpm, confidence) = result.expect("should succeed in test");
        assert!((bpm - 120.0).abs() < 5.0);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_estimate_empty_intervals() {
        let estimator = BpmEstimator::new(44100.0, 60.0, 200.0);
        let result = estimator.estimate_from_intervals(&[]);
        assert!(result.is_err());
    }
}
