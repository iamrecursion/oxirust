//! Melody extraction using salience-based pitch detection.

use crate::types::MelodyResult;
use crate::utils::{find_peaks, stft};
use crate::MirResult;

/// Melody extractor.
pub struct MelodyExtractor {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
}

impl MelodyExtractor {
    /// Create a new melody extractor.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize, hop_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
            hop_size,
        }
    }

    /// Extract melody from audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if melody extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract(&self, signal: &[f32]) -> MirResult<MelodyResult> {
        let frames = stft(signal, self.window_size, self.hop_size)?;

        let mut pitch_contour = Vec::with_capacity(frames.len());
        let mut time_points = Vec::with_capacity(frames.len());
        let mut confidence = Vec::with_capacity(frames.len());

        for (frame_idx, frame) in frames.iter().enumerate() {
            let time = frame_idx as f32 * self.hop_size as f32 / self.sample_rate;
            let (pitch, conf) = self.extract_pitch(frame);

            time_points.push(time);
            pitch_contour.push(pitch);
            confidence.push(conf);
        }

        // Compute melodic range
        let valid_pitches: Vec<f32> = pitch_contour.iter().copied().filter(|&p| p > 0.0).collect();

        let range = if valid_pitches.is_empty() {
            (0.0, 0.0)
        } else {
            let min_pitch = valid_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_pitch = valid_pitches
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min_pitch, max_pitch)
        };

        // Compute melodic complexity
        let complexity = self.compute_complexity(&pitch_contour);

        Ok(MelodyResult {
            pitch_contour,
            time_points,
            confidence,
            range,
            complexity,
        })
    }

    /// Extract dominant pitch from a frame using salience.
    #[allow(clippy::cast_precision_loss)]
    fn extract_pitch(&self, frame: &[oxifft::Complex<f32>]) -> (f32, f32) {
        let mag = crate::utils::magnitude_spectrum(frame);

        // Find peaks in magnitude spectrum
        let peaks = find_peaks(&mag, 5);

        if peaks.is_empty() {
            return (0.0, 0.0); // No pitch detected
        }

        // Select dominant peak (highest magnitude)
        let dominant_peak = peaks
            .iter()
            .max_by(|&&a, &&b| {
                mag[a]
                    .partial_cmp(&mag[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0);

        // Convert bin to frequency
        let freq = dominant_peak as f32 * self.sample_rate / self.window_size as f32;

        // Only consider pitches in musical range (80 Hz to 1000 Hz)
        if !(80.0..=1000.0).contains(&freq) {
            return (0.0, 0.0);
        }

        // Confidence based on peak prominence
        let peak_mag = mag[dominant_peak];
        let median_mag = crate::utils::median(&mag);
        let confidence = if median_mag > 0.0 {
            (peak_mag / (median_mag * 10.0)).min(1.0)
        } else {
            0.0
        };

        (freq, confidence)
    }

    /// Compute melodic complexity based on pitch variation.
    fn compute_complexity(&self, pitch_contour: &[f32]) -> f32 {
        if pitch_contour.len() < 2 {
            return 0.0;
        }

        let mut changes = 0;
        let mut total_change = 0.0;

        for i in 1..pitch_contour.len() {
            if pitch_contour[i] > 0.0 && pitch_contour[i - 1] > 0.0 {
                let diff = (pitch_contour[i] - pitch_contour[i - 1]).abs();
                if diff > 10.0 {
                    // Significant pitch change (> 10 Hz)
                    changes += 1;
                    total_change += diff;
                }
            }
        }

        if changes == 0 {
            return 0.0;
        }

        // Normalize complexity
        let avg_change = total_change / changes as f32;
        (avg_change / 100.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_melody_extractor_creation() {
        let extractor = MelodyExtractor::new(44100.0, 2048, 512);
        assert_eq!(extractor.sample_rate, 44100.0);
    }

    #[test]
    fn test_extract_silence() {
        let extractor = MelodyExtractor::new(44100.0, 2048, 512);
        let signal = vec![0.0; 44100];
        let result = extractor.extract(&signal);
        assert!(result.is_ok());
    }
}
