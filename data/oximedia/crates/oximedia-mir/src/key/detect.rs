//! Key detection using pitch class profiles.

use crate::key::profile::KEY_PROFILES;
use crate::types::KeyResult;
use crate::utils::stft;
use crate::{MirError, MirResult};

/// Key detector using Krumhansl-Schmuckler algorithm.
pub struct KeyDetector {
    sample_rate: f32,
    window_size: usize,
}

impl KeyDetector {
    /// Create a new key detector.
    #[must_use]
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        Self {
            sample_rate,
            window_size,
        }
    }

    /// Detect musical key from audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if key detection fails.
    pub fn detect(&self, signal: &[f32]) -> MirResult<KeyResult> {
        // Compute chromagram
        let chroma = self.compute_chromagram(signal)?;

        if chroma.is_empty() {
            return Err(MirError::InsufficientData(
                "No chroma frames computed".to_string(),
            ));
        }

        // Average chromagram over time
        let avg_chroma = self.average_chroma(&chroma);

        // Correlate with key profiles
        let correlations = self.correlate_with_profiles(&avg_chroma);

        // Find best matching key
        let (key_idx, correlation) = correlations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| MirError::AnalysisFailed("No key correlations".to_string()))?;

        let key_profile = &KEY_PROFILES[key_idx];
        let key_name = format!(
            "{} {}",
            Self::note_name(key_profile.root),
            if key_profile.is_major {
                "major"
            } else {
                "minor"
            }
        );

        // Normalize correlation to confidence score
        let max_correlation = correlations
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let confidence = if max_correlation > 0.0 {
            correlation / max_correlation
        } else {
            0.0
        };

        Ok(KeyResult {
            key: key_name,
            root: key_profile.root,
            is_major: key_profile.is_major,
            confidence,
            profile_correlations: correlations,
        })
    }

    /// Compute chromagram from audio signal.
    fn compute_chromagram(&self, signal: &[f32]) -> MirResult<Vec<[f32; 12]>> {
        let hop_size = self.window_size / 4;
        let frames = stft(signal, self.window_size, hop_size)?;

        let mut chroma_frames = Vec::with_capacity(frames.len());

        for frame in &frames {
            let chroma = self.frame_to_chroma(frame);
            chroma_frames.push(chroma);
        }

        Ok(chroma_frames)
    }

    /// Convert FFT frame to 12-bin chroma vector.
    #[allow(clippy::cast_precision_loss)]
    fn frame_to_chroma(&self, frame: &[oxifft::Complex<f32>]) -> [f32; 12] {
        let mut chroma = [0.0; 12];
        let num_bins = frame.len() / 2;

        // Reference frequency for C0
        let ref_freq = 16.35; // C0 in Hz

        for (bin, complex) in frame[1..num_bins].iter().enumerate() {
            let magnitude = complex.norm();
            let freq = (bin + 1) as f32 * self.sample_rate / self.window_size as f32;

            if freq < 20.0 {
                continue; // Skip very low frequencies
            }

            // Convert frequency to pitch class
            let pitch_class = self.freq_to_pitch_class(freq, ref_freq);

            chroma[pitch_class] += magnitude;
        }

        // Normalize chroma vector
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for c in &mut chroma {
                *c /= sum;
            }
        }

        chroma
    }

    /// Convert frequency to pitch class (0-11).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn freq_to_pitch_class(&self, freq: f32, ref_freq: f32) -> usize {
        let semitones = 12.0 * (freq / ref_freq).log2();
        (semitones.round() as i32).rem_euclid(12) as usize
    }

    /// Average chroma frames over time.
    fn average_chroma(&self, chroma_frames: &[[f32; 12]]) -> [f32; 12] {
        let mut avg = [0.0; 12];

        for frame in chroma_frames {
            for (i, &val) in frame.iter().enumerate() {
                avg[i] += val;
            }
        }

        let count = chroma_frames.len() as f32;
        if count > 0.0 {
            for val in &mut avg {
                *val /= count;
            }
        }

        avg
    }

    /// Correlate chroma with all key profiles.
    fn correlate_with_profiles(&self, chroma: &[f32; 12]) -> Vec<f32> {
        KEY_PROFILES
            .iter()
            .map(|profile| self.correlate(chroma, &profile.weights))
            .collect()
    }

    /// Compute correlation between two vectors.
    fn correlate(&self, a: &[f32; 12], b: &[f32; 12]) -> f32 {
        let mean_a = a.iter().sum::<f32>() / 12.0;
        let mean_b = b.iter().sum::<f32>() / 12.0;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..12 {
            let diff_a = a[i] - mean_a;
            let diff_b = b[i] - mean_b;
            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        if sum_sq_a == 0.0 || sum_sq_b == 0.0 {
            return 0.0;
        }

        numerator / (sum_sq_a * sum_sq_b).sqrt()
    }

    /// Get note name from pitch class.
    #[must_use]
    fn note_name(pitch_class: u8) -> &'static str {
        match pitch_class {
            0 => "C",
            1 => "C#",
            2 => "D",
            3 => "D#",
            4 => "E",
            5 => "F",
            6 => "F#",
            7 => "G",
            8 => "G#",
            9 => "A",
            10 => "A#",
            11 => "B",
            _ => "?",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_detector_creation() {
        let detector = KeyDetector::new(44100.0, 2048);
        assert_eq!(detector.sample_rate, 44100.0);
    }

    #[test]
    fn test_note_name() {
        assert_eq!(KeyDetector::note_name(0), "C");
        assert_eq!(KeyDetector::note_name(4), "E");
        assert_eq!(KeyDetector::note_name(7), "G");
    }

    #[test]
    fn test_average_chroma() {
        let detector = KeyDetector::new(44100.0, 2048);
        let chroma1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let chroma2 = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let avg = detector.average_chroma(&[chroma1, chroma2]);
        assert_eq!(avg[0], 0.5);
        assert_eq!(avg[1], 0.5);
    }
}
