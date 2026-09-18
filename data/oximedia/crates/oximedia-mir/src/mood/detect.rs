//! Mood detection using valence-arousal model.

use crate::types::MoodResult;
use crate::utils::{mean, stft};
use crate::MirResult;
use std::collections::HashMap;

/// Mood detector using valence-arousal model.
pub struct MoodDetector {
    #[allow(dead_code)]
    sample_rate: f32,
}

impl MoodDetector {
    /// Create a new mood detector.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Detect mood from audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if mood detection fails.
    pub fn detect(&self, signal: &[f32]) -> MirResult<MoodResult> {
        let features = self.extract_features(signal)?;

        // Compute valence (negative to positive)
        let valence = self.compute_valence(&features);

        // Compute arousal (calm to energetic)
        let arousal = self.compute_arousal(&features);

        // Map to mood labels
        let moods = self.map_to_moods(valence, arousal);

        // Compute emotional intensity
        let intensity = (valence.abs() + arousal).sqrt() / 2.0_f32.sqrt();

        Ok(MoodResult {
            valence,
            arousal,
            moods,
            intensity,
        })
    }

    /// Extract features for mood detection.
    fn extract_features(&self, signal: &[f32]) -> MirResult<MoodFeatures> {
        let window_size = 2048;
        let hop_size = 512;

        let frames = stft(signal, window_size, hop_size)?;

        let mut spectral_centroids = Vec::new();
        let mut energies = Vec::new();

        for frame in &frames {
            let mag = crate::utils::magnitude_spectrum(frame);

            let centroid = self.compute_spectral_centroid(&mag);
            let energy = mag.iter().map(|m| m * m).sum::<f32>();

            spectral_centroids.push(centroid);
            energies.push(energy);
        }

        Ok(MoodFeatures {
            avg_spectral_centroid: mean(&spectral_centroids),
            avg_energy: mean(&energies),
            energy_variance: crate::utils::std_dev(&energies),
        })
    }

    /// Compute valence from features.
    fn compute_valence(&self, features: &MoodFeatures) -> f32 {
        // High spectral centroid and moderate energy suggests positive valence
        let centroid_factor = (features.avg_spectral_centroid / 1000.0).clamp(0.0, 1.0);
        let energy_factor = (features.avg_energy / 100.0).clamp(0.0, 1.0);

        // Map to [-1, 1] range
        (centroid_factor * 0.6 + energy_factor * 0.4) * 2.0 - 1.0
    }

    /// Compute arousal from features.
    fn compute_arousal(&self, features: &MoodFeatures) -> f32 {
        // High energy and variance suggests high arousal
        let energy_factor = (features.avg_energy / 100.0).clamp(0.0, 1.0);
        let variance_factor = (features.energy_variance / 50.0).clamp(0.0, 1.0);

        energy_factor * 0.7 + variance_factor * 0.3
    }

    /// Map valence-arousal to mood labels.
    fn map_to_moods(&self, valence: f32, arousal: f32) -> HashMap<String, f32> {
        let mut moods = HashMap::new();

        // Quadrant-based mood mapping
        if valence > 0.0 && arousal > 0.5 {
            moods.insert("happy".to_string(), (valence + arousal) / 2.0);
            moods.insert("excited".to_string(), arousal);
        } else if valence > 0.0 && arousal <= 0.5 {
            moods.insert("calm".to_string(), valence * (1.0 - arousal));
            moods.insert("peaceful".to_string(), (valence + (1.0 - arousal)) / 2.0);
        } else if valence <= 0.0 && arousal > 0.5 {
            moods.insert("angry".to_string(), arousal * valence.abs());
            moods.insert("tense".to_string(), arousal);
        } else {
            moods.insert("sad".to_string(), valence.abs() * (1.0 - arousal));
            moods.insert(
                "melancholic".to_string(),
                (valence.abs() + (1.0 - arousal)) / 2.0,
            );
        }

        moods
    }

    /// Compute spectral centroid.
    #[allow(clippy::cast_precision_loss)]
    fn compute_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            weighted_sum += i as f32 * mag;
            total += mag;
        }

        if total > 0.0 {
            weighted_sum / total
        } else {
            0.0
        }
    }
}

/// Features for mood detection.
struct MoodFeatures {
    avg_spectral_centroid: f32,
    avg_energy: f32,
    energy_variance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mood_detector_creation() {
        let detector = MoodDetector::new(44100.0);
        assert_eq!(detector.sample_rate, 44100.0);
    }
}
