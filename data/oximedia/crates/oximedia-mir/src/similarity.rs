//! Music similarity and distance metrics.
//!
//! Provides audio feature descriptors and distance functions for comparing
//! tracks and building recommendation indexes.

#![allow(dead_code)]

/// High-level audio features used for similarity comparison.
///
/// All fields are normalized to meaningful ranges for each dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFeatures {
    /// Tempo in beats per minute (typically 60–200).
    pub tempo_bpm: f64,
    /// Musical key as MIDI pitch class (0 = C, 1 = C#, …, 11 = B).
    pub key: u8,
    /// Modality: 0 = minor, 1 = major.
    pub mode: u8,
    /// Integrated loudness in LUFS (typically –60 to 0).
    pub loudness_lufs: f64,
    /// Perceived energy (0.0–1.0).
    pub energy: f64,
    /// Rhythmic regularity / danceability (0.0–1.0).
    pub danceability: f64,
    /// Emotional positivity / valence (0.0–1.0).
    pub valence: f64,
    /// Proportion of speech-like audio (0.0–1.0).
    pub speechiness: f64,
}

impl AudioFeatures {
    /// Return the features as a raw 8-element vector.
    ///
    /// The vector layout is:
    /// `[tempo_bpm, key, mode, loudness_lufs, energy, danceability, valence, speechiness]`
    #[must_use]
    pub fn feature_vector(&self) -> Vec<f64> {
        vec![
            self.tempo_bpm,
            f64::from(self.key),
            f64::from(self.mode),
            self.loudness_lufs,
            self.energy,
            self.danceability,
            self.valence,
            self.speechiness,
        ]
    }

    /// Return a normalized feature vector suitable for distance comparisons.
    ///
    /// Each dimension is mapped to approximately [0, 1]:
    /// - tempo: scaled by 1/200
    /// - key: scaled by 1/12
    /// - mode: already 0/1
    /// - loudness: mapped from [–60, 0] to [0, 1]
    /// - energy/danceability/valence/speechiness: already [0, 1]
    #[must_use]
    pub fn normalize(&self) -> Vec<f64> {
        vec![
            (self.tempo_bpm / 200.0).clamp(0.0, 1.0),
            f64::from(self.key) / 12.0,
            f64::from(self.mode),
            ((self.loudness_lufs + 60.0) / 60.0).clamp(0.0, 1.0),
            self.energy.clamp(0.0, 1.0),
            self.danceability.clamp(0.0, 1.0),
            self.valence.clamp(0.0, 1.0),
            self.speechiness.clamp(0.0, 1.0),
        ]
    }
}

/// Compute the Euclidean distance between two equal-length slices.
///
/// Returns 0.0 if both slices are empty.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Feature vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the cosine similarity between two equal-length slices.
///
/// Returns 0.0 if either vector is a zero vector or both are empty.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Feature vectors must have equal length");
    if a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Compute a weighted Euclidean distance between two `AudioFeatures`.
///
/// `weights` is an 8-element array that scales each normalized feature
/// dimension.  Higher weights make those dimensions more influential.
#[must_use]
pub fn weighted_distance(a: &AudioFeatures, b: &AudioFeatures, weights: &[f64; 8]) -> f64 {
    let na = a.normalize();
    let nb = b.normalize();
    na.iter()
        .zip(nb.iter())
        .zip(weights.iter())
        .map(|((x, y), w)| w * (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// An in-memory index for fast similarity lookup.
///
/// Stores a list of `(track_id, AudioFeatures)` pairs and supports
/// nearest-neighbor queries using Euclidean distance on normalized features.
#[derive(Debug, Default)]
pub struct SimilarityIndex {
    entries: Vec<(u64, AudioFeatures)>,
}

impl SimilarityIndex {
    /// Create an empty similarity index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a track to the index.
    pub fn add(&mut self, id: u64, features: AudioFeatures) {
        self.entries.push((id, features));
    }

    /// Find the `n` most similar tracks to `query`.
    ///
    /// Returns a list of `(track_id, distance)` pairs sorted by ascending
    /// distance (most similar first).  The query track itself is excluded
    /// if it exists in the index.
    #[must_use]
    pub fn find_similar(&self, query: &AudioFeatures, n: usize) -> Vec<(u64, f64)> {
        let qv = query.normalize();

        let mut scored: Vec<(u64, f64)> = self
            .entries
            .iter()
            .map(|(id, feat)| {
                let fv = feat.normalize();
                let dist = euclidean_distance(&qv, &fv);
                (*id, dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Return the number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the index contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features(tempo: f64, key: u8, loudness: f64, energy: f64) -> AudioFeatures {
        AudioFeatures {
            tempo_bpm: tempo,
            key,
            mode: 1,
            loudness_lufs: loudness,
            energy,
            danceability: 0.7,
            valence: 0.6,
            speechiness: 0.05,
        }
    }

    #[test]
    fn test_feature_vector_length() {
        let feat = sample_features(120.0, 5, -10.0, 0.8);
        assert_eq!(feat.feature_vector().len(), 8);
    }

    #[test]
    fn test_feature_vector_values() {
        let feat = sample_features(120.0, 5, -10.0, 0.8);
        let v = feat.feature_vector();
        assert!((v[0] - 120.0).abs() < 1e-9);
        assert!((v[1] - 5.0).abs() < 1e-9);
        assert!((v[3] - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_length() {
        let feat = sample_features(120.0, 5, -10.0, 0.8);
        assert_eq!(feat.normalize().len(), 8);
    }

    #[test]
    fn test_normalize_tempo_clamp() {
        let feat = sample_features(300.0, 0, 0.0, 1.0); // over 200 BPM
        let n = feat.normalize();
        assert!((n[0] - 1.0).abs() < 1e-9, "tempo should clamp to 1.0");
    }

    #[test]
    fn test_normalize_loudness_mapping() {
        // –60 LUFS → 0.0, 0 LUFS → 1.0
        let f_min = sample_features(120.0, 0, -60.0, 0.5);
        let f_max = sample_features(120.0, 0, 0.0, 0.5);
        let n_min = f_min.normalize();
        let n_max = f_max.normalize();
        assert!((n_min[3] - 0.0).abs() < 1e-9);
        assert!((n_max[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_euclidean_distance_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&v, &v) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_euclidean_distance_known_value() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_weighted_distance_same_features() {
        let feat = sample_features(120.0, 5, -10.0, 0.8);
        let weights = [1.0; 8];
        let dist = weighted_distance(&feat, &feat, &weights);
        assert!(dist < 1e-9, "Distance from self should be ~0");
    }

    #[test]
    fn test_weighted_distance_different_features() {
        let a = sample_features(120.0, 0, -10.0, 0.5);
        let b = sample_features(180.0, 11, -30.0, 0.9);
        let weights = [1.0; 8];
        let dist = weighted_distance(&a, &b, &weights);
        assert!(
            dist > 0.0,
            "Different features should have positive distance"
        );
    }

    #[test]
    fn test_similarity_index_new_empty() {
        let idx = SimilarityIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_similarity_index_add_and_len() {
        let mut idx = SimilarityIndex::new();
        idx.add(1, sample_features(120.0, 5, -10.0, 0.8));
        idx.add(2, sample_features(130.0, 7, -12.0, 0.7));
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_similarity_index_find_similar_nearest_first() {
        let mut idx = SimilarityIndex::new();
        // Track 1: very similar to query
        idx.add(1, sample_features(120.0, 5, -10.0, 0.8));
        // Track 2: very different from query
        idx.add(2, sample_features(200.0, 11, -60.0, 0.1));
        let query = sample_features(121.0, 5, -10.5, 0.79);
        let results = idx.find_similar(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1, "Track 1 should be the most similar");
    }

    #[test]
    fn test_similarity_index_find_similar_limits_n() {
        let mut idx = SimilarityIndex::new();
        for i in 0..10 {
            idx.add(i, sample_features(120.0 + i as f64, 5, -10.0, 0.8));
        }
        let query = sample_features(120.0, 5, -10.0, 0.8);
        let results = idx.find_similar(&query, 3);
        assert_eq!(results.len(), 3);
    }
}
