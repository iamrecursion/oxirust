//! Integrated loudness analysis.

use crate::types::LoudnessResult;
use crate::utils::mean;
use crate::{MirError, MirResult};

/// Loudness analyzer.
pub struct LoudnessAnalyzer {
    sample_rate: f32,
}

impl LoudnessAnalyzer {
    /// Create a new loudness analyzer.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Analyze loudness of audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if loudness analysis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, signal: &[f32]) -> MirResult<LoudnessResult> {
        if signal.is_empty() {
            return Err(MirError::InsufficientData(
                "Empty signal for loudness analysis".to_string(),
            ));
        }

        // Compute RMS values in frames
        let frame_size = (self.sample_rate * 0.4) as usize; // 400ms frames
        let hop_size = frame_size / 4;

        let mut rms_values = Vec::new();
        for i in (0..signal.len()).step_by(hop_size) {
            let end = (i + frame_size).min(signal.len());
            if end - i < frame_size / 2 {
                break;
            }

            let frame = &signal[i..end];
            let rms = self.compute_rms(frame);
            rms_values.push(rms);
        }

        // Integrated loudness (simplified, not true EBU R128)
        let integrated_loudness = self.compute_integrated_loudness(&rms_values);

        // Loudness range
        let loudness_range = self.compute_loudness_range(&rms_values);

        // Peak loudness
        let peak_loudness = rms_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // True peak (simplified - would need upsampling)
        let true_peak = signal.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);

        Ok(LoudnessResult {
            integrated_loudness,
            loudness_range,
            peak_loudness,
            true_peak,
        })
    }

    /// Compute RMS of a frame.
    #[allow(clippy::cast_precision_loss)]
    fn compute_rms(&self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = frame.iter().map(|s| s * s).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }

    /// Compute integrated loudness with gating.
    fn compute_integrated_loudness(&self, rms_values: &[f32]) -> f32 {
        if rms_values.is_empty() {
            return 0.0;
        }

        // Simplified gating (not full EBU R128)
        let threshold = mean(rms_values) * 0.1; // -20 LUFS gate approximation

        let gated_values: Vec<f32> = rms_values
            .iter()
            .filter(|&&rms| rms > threshold)
            .copied()
            .collect();

        if gated_values.is_empty() {
            return 0.0;
        }

        mean(&gated_values)
    }

    /// Compute loudness range.
    fn compute_loudness_range(&self, rms_values: &[f32]) -> f32 {
        if rms_values.is_empty() {
            return 0.0;
        }

        let mut sorted = rms_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // LRA is difference between 10th and 95th percentile
        let idx_10 = (sorted.len() as f32 * 0.10) as usize;
        let idx_95 = (sorted.len() as f32 * 0.95) as usize;

        sorted[idx_95] - sorted[idx_10]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_analyzer_creation() {
        let analyzer = LoudnessAnalyzer::new(44100.0);
        assert_eq!(analyzer.sample_rate, 44100.0);
    }

    #[test]
    fn test_compute_rms() {
        let analyzer = LoudnessAnalyzer::new(44100.0);
        let frame = vec![0.5, -0.5, 0.5, -0.5];
        let rms = analyzer.compute_rms(&frame);
        assert!((rms - 0.5).abs() < 1e-6);
    }
}
