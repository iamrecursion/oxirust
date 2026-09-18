//! Beat tracking using dynamic programming.

use crate::beat::downbeat::DownbeatDetector;
use crate::beat::onset::OnsetDetector;
use crate::types::{BeatResult, TempoResult};
use crate::{MirError, MirResult};

/// Beat tracker.
pub struct BeatTracker {
    sample_rate: f32,
    hop_size: usize,
}

impl BeatTracker {
    /// Create a new beat tracker.
    #[must_use]
    pub fn new(sample_rate: f32, hop_size: usize) -> Self {
        Self {
            sample_rate,
            hop_size,
        }
    }

    /// Track beats in audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if beat tracking fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn track(&self, signal: &[f32], tempo: Option<&TempoResult>) -> MirResult<BeatResult> {
        // Detect onsets
        let onset_detector = OnsetDetector::new(self.sample_rate, 2048, self.hop_size);
        let onset_times = onset_detector.detect(signal)?;

        if onset_times.is_empty() {
            return Err(MirError::AnalysisFailed("No onsets detected".to_string()));
        }

        // Get tempo estimate
        let bpm = tempo.map_or(120.0, |t| t.bpm);
        let beat_period = 60.0 / bpm;

        // Track beats using dynamic programming
        let beat_times = self.track_beats_dp(&onset_times, beat_period)?;

        // Detect downbeats
        let downbeat_detector = DownbeatDetector::new(self.sample_rate);
        let downbeat_times = downbeat_detector.detect(signal, &beat_times)?;

        // Compute beat confidence scores
        let beat_confidence = self.compute_beat_confidence(&beat_times, &onset_times);

        // Estimate time signature from beat pattern
        let time_signature = self.estimate_time_signature(&beat_times, &downbeat_times);

        Ok(BeatResult {
            beat_times,
            downbeat_times,
            beat_confidence,
            time_signature,
        })
    }

    /// Track beats using dynamic programming.
    fn track_beats_dp(&self, onset_times: &[f32], beat_period: f32) -> MirResult<Vec<f32>> {
        if onset_times.is_empty() {
            return Err(MirError::InsufficientData(
                "No onset times for beat tracking".to_string(),
            ));
        }

        let duration = onset_times.last().copied().unwrap_or(0.0) + beat_period;
        let num_beats = (duration / beat_period).ceil() as usize;

        // Generate candidate beat times
        let mut beat_times = Vec::new();
        let tolerance = beat_period * 0.2; // 20% tolerance

        for i in 0..num_beats {
            let expected_time = i as f32 * beat_period;

            // Find closest onset within tolerance
            let closest_onset = onset_times
                .iter()
                .filter(|&&t| (t - expected_time).abs() < tolerance)
                .min_by(|&&a, &&b| {
                    (a - expected_time)
                        .abs()
                        .partial_cmp(&(b - expected_time).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(&onset_time) = closest_onset {
                beat_times.push(onset_time);
            } else {
                // No onset found, use expected time
                beat_times.push(expected_time);
            }
        }

        Ok(beat_times)
    }

    /// Compute confidence score for each beat.
    fn compute_beat_confidence(&self, beat_times: &[f32], onset_times: &[f32]) -> Vec<f32> {
        let tolerance: f32 = 0.070; // 70ms tolerance

        beat_times
            .iter()
            .map(|&beat_time| {
                // Find closest onset
                let min_distance = onset_times
                    .iter()
                    .map(|&onset_time| (beat_time - onset_time).abs())
                    .fold(f32::INFINITY, f32::min);

                // Convert distance to confidence (Gaussian)
                (-min_distance.powi(2) / (2.0 * tolerance.powi(2))).exp()
            })
            .collect()
    }

    /// Estimate time signature from beat pattern.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn estimate_time_signature(
        &self,
        beat_times: &[f32],
        downbeat_times: &[f32],
    ) -> Option<(u8, u8)> {
        if downbeat_times.len() < 2 {
            return Some((4, 4)); // Default to 4/4
        }

        // Count beats between downbeats
        let mut beats_per_bar = Vec::new();

        for i in 0..downbeat_times.len() - 1 {
            let start = downbeat_times[i];
            let end = downbeat_times[i + 1];

            let count = beat_times
                .iter()
                .filter(|&&t| t >= start && t < end)
                .count();

            if count > 0 {
                beats_per_bar.push(count);
            }
        }

        if beats_per_bar.is_empty() {
            return Some((4, 4));
        }

        // Find most common beat count
        let mut counts = std::collections::HashMap::new();
        for &count in &beats_per_bar {
            *counts.entry(count).or_insert(0) += 1;
        }

        let most_common = counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&beats, _)| beats)?;

        // Convert to time signature (assume quarter note beat)
        let numerator = most_common as u8;
        let denominator = 4;

        Some((numerator, denominator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_tracker_creation() {
        let tracker = BeatTracker::new(44100.0, 512);
        assert_eq!(tracker.sample_rate, 44100.0);
    }

    #[test]
    fn test_compute_beat_confidence() {
        let tracker = BeatTracker::new(44100.0, 512);
        let beat_times = vec![0.0, 0.5, 1.0];
        let onset_times = vec![0.01, 0.51, 1.02];
        let confidence = tracker.compute_beat_confidence(&beat_times, &onset_times);
        assert_eq!(confidence.len(), 3);
        assert!(confidence[0] > 0.5);
    }

    #[test]
    fn test_estimate_time_signature() {
        let tracker = BeatTracker::new(44100.0, 512);
        let beat_times = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let downbeat_times = vec![0.0, 2.0];
        let time_sig = tracker.estimate_time_signature(&beat_times, &downbeat_times);
        assert_eq!(time_sig, Some((4, 4)));
    }
}
