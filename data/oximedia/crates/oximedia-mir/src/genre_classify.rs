#![allow(dead_code)]
//! Genre classification for music information retrieval.
//!
//! Provides a rule-based genre classifier that operates on extracted audio
//! features (spectral centroid, tempo, zero-crossing rate, energy, etc.) to
//! assign genre labels with confidence scores.

use std::collections::HashMap;
use std::fmt;

/// Music genre label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MusicGenre {
    /// Rock and alternative rock.
    Rock,
    /// Pop music.
    Pop,
    /// Electronic and EDM.
    Electronic,
    /// Hip-hop and rap.
    HipHop,
    /// Jazz.
    Jazz,
    /// Classical and orchestral.
    Classical,
    /// Country and folk.
    Country,
    /// R&B and soul.
    RnB,
    /// Metal and heavy metal.
    Metal,
    /// Blues.
    Blues,
    /// Reggae and dub.
    Reggae,
    /// Ambient and drone.
    Ambient,
}

impl fmt::Display for MusicGenre {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rock => write!(f, "Rock"),
            Self::Pop => write!(f, "Pop"),
            Self::Electronic => write!(f, "Electronic"),
            Self::HipHop => write!(f, "Hip-Hop"),
            Self::Jazz => write!(f, "Jazz"),
            Self::Classical => write!(f, "Classical"),
            Self::Country => write!(f, "Country"),
            Self::RnB => write!(f, "R&B"),
            Self::Metal => write!(f, "Metal"),
            Self::Blues => write!(f, "Blues"),
            Self::Reggae => write!(f, "Reggae"),
            Self::Ambient => write!(f, "Ambient"),
        }
    }
}

impl MusicGenre {
    /// Return all genre variants.
    #[must_use]
    pub fn all() -> &'static [MusicGenre] {
        &[
            Self::Rock,
            Self::Pop,
            Self::Electronic,
            Self::HipHop,
            Self::Jazz,
            Self::Classical,
            Self::Country,
            Self::RnB,
            Self::Metal,
            Self::Blues,
            Self::Reggae,
            Self::Ambient,
        ]
    }
}

/// Audio features used for genre classification.
#[derive(Debug, Clone, PartialEq)]
pub struct GenreFeatures {
    /// Average spectral centroid in Hz.
    pub spectral_centroid_hz: f64,
    /// Average spectral rolloff in Hz.
    pub spectral_rolloff_hz: f64,
    /// Average zero-crossing rate (crossings per second).
    pub zero_crossing_rate: f64,
    /// Average RMS energy (linear).
    pub rms_energy: f64,
    /// Estimated tempo in BPM.
    pub tempo_bpm: f64,
    /// Spectral flatness (0.0 = tonal, 1.0 = noise-like).
    pub spectral_flatness: f64,
    /// Average MFCC coefficient 1 (roughly related to brightness).
    pub mfcc1: f64,
}

impl GenreFeatures {
    /// Create a new feature vector.
    #[must_use]
    pub fn new(
        spectral_centroid_hz: f64,
        spectral_rolloff_hz: f64,
        zero_crossing_rate: f64,
        rms_energy: f64,
        tempo_bpm: f64,
        spectral_flatness: f64,
        mfcc1: f64,
    ) -> Self {
        Self {
            spectral_centroid_hz,
            spectral_rolloff_hz,
            zero_crossing_rate,
            rms_energy,
            tempo_bpm,
            spectral_flatness,
            mfcc1,
        }
    }
}

/// Result of genre classification.
#[derive(Debug, Clone)]
pub struct GenreClassification {
    /// Per-genre confidence scores (0.0..1.0).
    pub scores: HashMap<MusicGenre, f64>,
}

impl GenreClassification {
    /// Top predicted genre and its confidence.
    #[must_use]
    pub fn top(&self) -> (MusicGenre, f64) {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((MusicGenre::Pop, 0.0), |(&g, &s)| (g, s))
    }

    /// Top-N genres sorted by descending confidence.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(MusicGenre, f64)> {
        let mut sorted: Vec<(MusicGenre, f64)> =
            self.scores.iter().map(|(&g, &s)| (g, s)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Whether the top prediction exceeds a confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.top().1 >= threshold
    }
}

/// Rule-based genre classifier operating on audio features.
#[derive(Debug)]
pub struct GenreClassifier {
    /// Minimum confidence to emit.
    pub min_confidence: f64,
}

impl Default for GenreClassifier {
    fn default() -> Self {
        Self {
            min_confidence: 0.1,
        }
    }
}

impl GenreClassifier {
    /// Create a new genre classifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify genre from extracted features.
    ///
    /// Uses a simple distance-to-prototype approach: each genre has a centroid
    /// in feature space, and the classifier computes a similarity score for each.
    #[must_use]
    pub fn classify(&self, features: &GenreFeatures) -> GenreClassification {
        let prototypes = Self::genre_prototypes();
        let mut raw_scores: HashMap<MusicGenre, f64> = HashMap::new();

        for (genre, proto) in &prototypes {
            let dist = Self::feature_distance(features, proto);
            // Convert distance to similarity: exp(-dist)
            let sim = (-dist * 0.5).exp();
            raw_scores.insert(*genre, sim);
        }

        // Normalize to sum to 1.0
        let total: f64 = raw_scores.values().sum();
        let scores: HashMap<MusicGenre, f64> = if total > 0.0 {
            raw_scores
                .into_iter()
                .map(|(g, s)| (g, s / total))
                .collect()
        } else {
            raw_scores
        };

        GenreClassification { scores }
    }

    /// Euclidean distance in normalized feature space.
    fn feature_distance(a: &GenreFeatures, b: &GenreFeatures) -> f64 {
        let dc = (a.spectral_centroid_hz - b.spectral_centroid_hz) / 2000.0;
        let dr = (a.spectral_rolloff_hz - b.spectral_rolloff_hz) / 4000.0;
        let dz = (a.zero_crossing_rate - b.zero_crossing_rate) / 100.0;
        let de = (a.rms_energy - b.rms_energy) / 0.3;
        let dt = (a.tempo_bpm - b.tempo_bpm) / 40.0;
        let df = (a.spectral_flatness - b.spectral_flatness) / 0.3;
        let dm = (a.mfcc1 - b.mfcc1) / 50.0;
        (dc * dc + dr * dr + dz * dz + de * de + dt * dt + df * df + dm * dm).sqrt()
    }

    /// Return genre prototype feature vectors.
    fn genre_prototypes() -> Vec<(MusicGenre, GenreFeatures)> {
        vec![
            (
                MusicGenre::Rock,
                GenreFeatures::new(3000.0, 6000.0, 80.0, 0.25, 120.0, 0.15, 20.0),
            ),
            (
                MusicGenre::Pop,
                GenreFeatures::new(2500.0, 5000.0, 60.0, 0.20, 115.0, 0.12, 15.0),
            ),
            (
                MusicGenre::Electronic,
                GenreFeatures::new(3500.0, 7000.0, 40.0, 0.30, 128.0, 0.25, 10.0),
            ),
            (
                MusicGenre::HipHop,
                GenreFeatures::new(2000.0, 4500.0, 50.0, 0.22, 90.0, 0.18, 12.0),
            ),
            (
                MusicGenre::Jazz,
                GenreFeatures::new(2200.0, 5500.0, 45.0, 0.12, 100.0, 0.10, 25.0),
            ),
            (
                MusicGenre::Classical,
                GenreFeatures::new(1800.0, 4000.0, 30.0, 0.10, 80.0, 0.05, 30.0),
            ),
            (
                MusicGenre::Country,
                GenreFeatures::new(2800.0, 5500.0, 55.0, 0.18, 110.0, 0.10, 18.0),
            ),
            (
                MusicGenre::RnB,
                GenreFeatures::new(2300.0, 5000.0, 50.0, 0.18, 95.0, 0.14, 14.0),
            ),
            (
                MusicGenre::Metal,
                GenreFeatures::new(4000.0, 8000.0, 120.0, 0.35, 140.0, 0.20, 5.0),
            ),
            (
                MusicGenre::Blues,
                GenreFeatures::new(2100.0, 4500.0, 40.0, 0.14, 85.0, 0.08, 22.0),
            ),
            (
                MusicGenre::Reggae,
                GenreFeatures::new(2000.0, 4200.0, 35.0, 0.15, 75.0, 0.12, 16.0),
            ),
            (
                MusicGenre::Ambient,
                GenreFeatures::new(1500.0, 3500.0, 15.0, 0.05, 60.0, 0.30, 35.0),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_display() {
        assert_eq!(format!("{}", MusicGenre::Rock), "Rock");
        assert_eq!(format!("{}", MusicGenre::HipHop), "Hip-Hop");
        assert_eq!(format!("{}", MusicGenre::RnB), "R&B");
    }

    #[test]
    fn test_genre_all() {
        let all = MusicGenre::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_features_creation() {
        let f = GenreFeatures::new(3000.0, 6000.0, 80.0, 0.25, 120.0, 0.15, 20.0);
        assert!((f.spectral_centroid_hz - 3000.0).abs() < f64::EPSILON);
        assert!((f.tempo_bpm - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_classifier_creation() {
        let c = GenreClassifier::new();
        assert!(c.min_confidence > 0.0);
    }

    #[test]
    fn test_classify_rock_like() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(3000.0, 6000.0, 80.0, 0.25, 120.0, 0.15, 20.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::Rock);
    }

    #[test]
    fn test_classify_classical_like() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(1800.0, 4000.0, 30.0, 0.10, 80.0, 0.05, 30.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::Classical);
    }

    #[test]
    fn test_classify_metal_like() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(4000.0, 8000.0, 120.0, 0.35, 140.0, 0.20, 5.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::Metal);
    }

    #[test]
    fn test_classification_scores_sum_to_one() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(2500.0, 5000.0, 60.0, 0.20, 115.0, 0.12, 15.0);
        let result = c.classify(&features);
        let total: f64 = result.scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_top_n() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(3000.0, 6000.0, 80.0, 0.25, 120.0, 0.15, 20.0);
        let result = c.classify(&features);
        let top3 = result.top_n(3);
        assert_eq!(top3.len(), 3);
        // Scores should be descending
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);
    }

    #[test]
    fn test_is_confident() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(4000.0, 8000.0, 120.0, 0.35, 140.0, 0.20, 5.0);
        let result = c.classify(&features);
        // Metal is quite distinct, confidence should be reasonable
        assert!(result.is_confident(0.05));
    }

    #[test]
    fn test_ambient_classification() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(1500.0, 3500.0, 15.0, 0.05, 60.0, 0.30, 35.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::Ambient);
    }

    #[test]
    fn test_electronic_classification() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(3500.0, 7000.0, 40.0, 0.30, 128.0, 0.25, 10.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::Electronic);
    }

    #[test]
    fn test_hiphop_classification() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(2000.0, 4500.0, 50.0, 0.22, 90.0, 0.18, 12.0);
        let result = c.classify(&features);
        let (top_genre, _) = result.top();
        assert_eq!(top_genre, MusicGenre::HipHop);
    }

    #[test]
    fn test_classification_has_all_genres() {
        let c = GenreClassifier::new();
        let features = GenreFeatures::new(2500.0, 5000.0, 60.0, 0.20, 115.0, 0.12, 15.0);
        let result = c.classify(&features);
        assert_eq!(result.scores.len(), 12);
    }
}
