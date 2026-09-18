//! Downbeat detection using harmonic and percussive analysis.

use crate::utils::stft;
use crate::MirResult;

/// Downbeat detector.
pub struct DownbeatDetector {
    sample_rate: f32,
}

impl DownbeatDetector {
    /// Create a new downbeat detector.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Detect downbeats from audio and beat times.
    ///
    /// # Errors
    ///
    /// Returns error if downbeat detection fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, signal: &[f32], beat_times: &[f32]) -> MirResult<Vec<f32>> {
        if beat_times.is_empty() {
            return Ok(Vec::new());
        }

        // Compute STFT for harmonic analysis
        let window_size = 2048;
        let hop_size = 512;
        let frames = stft(signal, window_size, hop_size)?;

        // Compute harmonic strength for each beat
        let beat_harmonics: Vec<f32> = beat_times
            .iter()
            .map(|&beat_time| {
                let frame_idx = (beat_time * self.sample_rate / hop_size as f32) as usize;
                if frame_idx < frames.len() {
                    self.compute_harmonic_strength(&frames[frame_idx])
                } else {
                    0.0
                }
            })
            .collect();

        // Find downbeats based on harmonic peaks and regularity
        let downbeats = self.find_downbeats_from_pattern(beat_times, &beat_harmonics)?;

        Ok(downbeats)
    }

    /// Compute harmonic strength of a frame.
    fn compute_harmonic_strength(&self, frame: &[oxifft::Complex<f32>]) -> f32 {
        let mag = crate::utils::magnitude_spectrum(frame);

        // Focus on low to mid frequencies (bass and chord content)
        let low_mid_range = mag.len().min(mag.len() / 4);

        mag[..low_mid_range].iter().sum::<f32>() / low_mid_range as f32
    }

    /// Find downbeats from beat pattern and harmonic content.
    #[allow(clippy::unnecessary_wraps)]
    fn find_downbeats_from_pattern(
        &self,
        beat_times: &[f32],
        beat_harmonics: &[f32],
    ) -> MirResult<Vec<f32>> {
        if beat_times.len() < 4 {
            // Not enough beats, assume first beat is downbeat
            return Ok(vec![beat_times.first().copied().unwrap_or(0.0)]);
        }

        // Try different bar lengths (3, 4, 5, 6 beats per bar)
        let mut best_downbeats = Vec::new();
        let mut best_score = 0.0;

        for bar_length in 3..=6 {
            let (downbeats, score) = self.try_bar_length(beat_times, beat_harmonics, bar_length);

            if score > best_score {
                best_score = score;
                best_downbeats = downbeats;
            }
        }

        if best_downbeats.is_empty() {
            // Default: first beat and every 4th beat
            best_downbeats = beat_times.iter().step_by(4).copied().collect();
        }

        Ok(best_downbeats)
    }

    /// Try a specific bar length and return score.
    fn try_bar_length(
        &self,
        beat_times: &[f32],
        beat_harmonics: &[f32],
        bar_length: usize,
    ) -> (Vec<f32>, f32) {
        let mut downbeats = Vec::new();
        let mut total_score = 0.0;

        for phase in 0..bar_length {
            let candidate_downbeats: Vec<f32> = beat_times
                .iter()
                .enumerate()
                .filter(|(i, _)| i % bar_length == phase)
                .map(|(_, &t)| t)
                .collect();

            let score: f32 = candidate_downbeats
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let beat_idx = phase + i * bar_length;
                    if beat_idx < beat_harmonics.len() {
                        beat_harmonics[beat_idx]
                    } else {
                        0.0
                    }
                })
                .sum();

            if score > total_score {
                total_score = score;
                downbeats = candidate_downbeats;
            }
        }

        (downbeats, total_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downbeat_detector_creation() {
        let detector = DownbeatDetector::new(44100.0);
        assert_eq!(detector.sample_rate, 44100.0);
    }

    #[test]
    fn test_detect_no_beats() {
        let detector = DownbeatDetector::new(44100.0);
        let signal = vec![0.0; 44100];
        let beat_times = vec![];
        let result = detector.detect(&signal, &beat_times);
        assert!(result.is_ok());
        assert!(result.expect("should succeed in test").is_empty());
    }

    #[test]
    fn test_detect_few_beats() {
        let detector = DownbeatDetector::new(44100.0);
        let signal = vec![0.0; 44100];
        let beat_times = vec![0.0, 0.5, 1.0];
        let result = detector.detect(&signal, &beat_times);
        assert!(result.is_ok());
    }
}
