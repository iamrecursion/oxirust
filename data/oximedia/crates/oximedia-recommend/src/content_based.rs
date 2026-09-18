//! Content-based recommendation helpers.
//!
//! This module provides feature-vector representations of media items and the
//! similarity metrics used to find content similar to what a user has already
//! enjoyed.  It is deliberately self-contained so it can be tested and reused
//! independently of the full content recommender in the `content` directory.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// A dense feature vector representing a media item's content characteristics.
///
/// Features may include genre one-hot encodings, normalised duration, year,
/// average rating, and any other continuous or binary signals.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    /// Item identifier.
    pub item_id: Uuid,
    /// Dense floating-point features.
    pub values: Vec<f32>,
}

impl FeatureVector {
    /// Construct a feature vector for `item_id` from raw values.
    ///
    /// # Panics
    ///
    /// Does not panic; caller is responsible for ensuring `values` is
    /// non-empty when computing similarities.
    #[must_use]
    pub fn new(item_id: Uuid, values: Vec<f32>) -> Self {
        Self { item_id, values }
    }

    /// Dimension (length) of the vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// L2 norm of the vector.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Similarity metrics
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two feature vectors.
///
/// Returns `0.0` when either vector has zero norm or when they have
/// incompatible dimensions.
#[must_use]
pub fn cosine_similarity(a: &FeatureVector, b: &FeatureVector) -> f32 {
    if a.dim() != b.dim() || a.dim() == 0 {
        return 0.0;
    }
    let dot: f32 = a.values.iter().zip(&b.values).map(|(x, y)| x * y).sum();
    let norm_a = a.norm();
    let norm_b = b.norm();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Compute Euclidean distance between two feature vectors.
///
/// Returns `f32::INFINITY` when dimensions differ.
#[must_use]
pub fn euclidean_distance(a: &FeatureVector, b: &FeatureVector) -> f32 {
    if a.dim() != b.dim() {
        return f32::INFINITY;
    }
    a.values
        .iter()
        .zip(&b.values)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Convert a Euclidean distance to a similarity score in `[0, 1]`.
#[must_use]
pub fn distance_to_similarity(distance: f32) -> f32 {
    1.0 / (1.0 + distance)
}

// ---------------------------------------------------------------------------
// Item feature store
// ---------------------------------------------------------------------------

/// In-memory store mapping item IDs to their feature vectors.
#[derive(Debug, Clone, Default)]
pub struct FeatureStore {
    items: HashMap<Uuid, FeatureVector>,
}

impl FeatureStore {
    /// Create an empty feature store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or overwrite the feature vector for an item.
    pub fn upsert(&mut self, fv: FeatureVector) {
        self.items.insert(fv.item_id, fv);
    }

    /// Retrieve the feature vector for `item_id`.
    #[must_use]
    pub fn get(&self, item_id: Uuid) -> Option<&FeatureVector> {
        self.items.get(&item_id)
    }

    /// Remove an item from the store.
    pub fn remove(&mut self, item_id: Uuid) -> Option<FeatureVector> {
        self.items.remove(&item_id)
    }

    /// Number of items in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` when the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Compute pairwise cosine similarity between `query_id` and all other
    /// items, returning results sorted descending by similarity.
    ///
    /// Items not found in the store are silently skipped.  The query item
    /// itself is excluded from results.
    #[must_use]
    pub fn most_similar(&self, query_id: Uuid, limit: usize) -> Vec<(Uuid, f32)> {
        let query = match self.items.get(&query_id) {
            Some(q) => q,
            None => return Vec::new(),
        };
        let mut scores: Vec<(Uuid, f32)> = self
            .items
            .iter()
            .filter(|(id, _)| **id != query_id)
            .map(|(id, fv)| (*id, cosine_similarity(query, fv)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }
}

// ---------------------------------------------------------------------------
// Profile-based recommendation
// ---------------------------------------------------------------------------

/// Build a "profile vector" for a user by averaging the feature vectors of
/// items they have positively engaged with (rating ≥ threshold).
///
/// Returns `None` when no valid items are found.
#[must_use]
pub fn build_user_profile(
    store: &FeatureStore,
    liked_items: &[(Uuid, f32)],
    rating_threshold: f32,
) -> Option<FeatureVector> {
    let relevant: Vec<&FeatureVector> = liked_items
        .iter()
        .filter(|(_, r)| *r >= rating_threshold)
        .filter_map(|(id, _)| store.get(*id))
        .collect();

    if relevant.is_empty() {
        return None;
    }

    let dim = relevant[0].dim();
    if dim == 0 {
        return None;
    }

    let n = relevant.len() as f32;
    let mut avg = vec![0.0_f32; dim];
    for fv in &relevant {
        if fv.dim() == dim {
            for (a, v) in avg.iter_mut().zip(&fv.values) {
                *a += v;
            }
        }
    }
    for a in &mut avg {
        *a /= n;
    }

    Some(FeatureVector::new(Uuid::new_v4(), avg))
}

/// Recommend items from `candidate_ids` that are most similar to the user
/// profile vector.
#[must_use]
pub fn recommend_from_profile(
    store: &FeatureStore,
    profile: &FeatureVector,
    candidate_ids: &[Uuid],
    limit: usize,
) -> Vec<(Uuid, f32)> {
    let mut scores: Vec<(Uuid, f32)> = candidate_ids
        .iter()
        .filter_map(|id| {
            store
                .get(*id)
                .map(|fv| (*id, cosine_similarity(profile, fv)))
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);
    scores
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    fn fv(id: Uuid, vals: &[f32]) -> FeatureVector {
        FeatureVector::new(id, vals.to_vec())
    }

    // --- FeatureVector ---

    #[test]
    fn test_feature_vector_dim() {
        let v = fv(uid(), &[1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_feature_vector_norm() {
        let v = fv(uid(), &[3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_vector_zero_norm() {
        let v = fv(uid(), &[0.0, 0.0]);
        assert!((v.norm()).abs() < f32::EPSILON);
    }

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_identical_vectors() {
        let id = uid();
        let v = fv(id, &[1.0, 2.0, 3.0]);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = fv(uid(), &[1.0, 0.0]);
        let b = fv(uid(), &[0.0, 1.0]);
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let a = fv(uid(), &[1.0, 2.0]);
        let b = fv(uid(), &[1.0]);
        assert!((cosine_similarity(&a, &b)).abs() < f32::EPSILON);
    }

    // --- euclidean_distance ---

    #[test]
    fn test_euclidean_same_vector() {
        let v = fv(uid(), &[1.0, 2.0, 3.0]);
        assert!(euclidean_distance(&v, &v) < 1e-5);
    }

    #[test]
    fn test_euclidean_known_distance() {
        let a = fv(uid(), &[0.0, 0.0]);
        let b = fv(uid(), &[3.0, 4.0]);
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_dim_mismatch() {
        let a = fv(uid(), &[1.0, 2.0]);
        let b = fv(uid(), &[1.0]);
        assert!(euclidean_distance(&a, &b).is_infinite());
    }

    // --- distance_to_similarity ---

    #[test]
    fn test_distance_to_similarity_zero() {
        // distance 0 → similarity 1
        assert!((distance_to_similarity(0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distance_to_similarity_large() {
        // As distance → ∞, similarity → 0
        let s = distance_to_similarity(1_000_000.0);
        assert!(s < 0.001);
    }

    // --- FeatureStore ---

    #[test]
    fn test_feature_store_upsert_get() {
        let mut store = FeatureStore::new();
        let id = uid();
        store.upsert(fv(id, &[1.0, 0.0]));
        assert!(store.get(id).is_some());
    }

    #[test]
    fn test_feature_store_remove() {
        let mut store = FeatureStore::new();
        let id = uid();
        store.upsert(fv(id, &[1.0]));
        assert!(store.remove(id).is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_feature_store_most_similar() {
        let mut store = FeatureStore::new();
        let q = uid();
        let near = uid();
        let far = uid();
        store.upsert(fv(q, &[1.0, 0.0]));
        store.upsert(fv(near, &[0.9, 0.1]));
        store.upsert(fv(far, &[0.0, 1.0]));
        let results = store.most_similar(q, 2);
        assert_eq!(results.len(), 2);
        // `near` should rank first
        assert_eq!(results[0].0, near);
    }

    // --- Profile building ---

    #[test]
    fn test_build_user_profile_basic() {
        let mut store = FeatureStore::new();
        let id1 = uid();
        let id2 = uid();
        store.upsert(fv(id1, &[1.0, 0.0]));
        store.upsert(fv(id2, &[0.0, 1.0]));
        let liked = vec![(id1, 5.0), (id2, 4.0)];
        let profile = build_user_profile(&store, &liked, 3.0);
        assert!(profile.is_some());
        let p = profile.expect("should succeed in test");
        assert_eq!(p.dim(), 2);
        assert!((p.values[0] - 0.5).abs() < 1e-5);
        assert!((p.values[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_build_user_profile_threshold_filters() {
        let mut store = FeatureStore::new();
        let id = uid();
        store.upsert(fv(id, &[1.0]));
        // rating below threshold
        let liked = vec![(id, 1.0)];
        let profile = build_user_profile(&store, &liked, 3.0);
        assert!(profile.is_none());
    }

    #[test]
    fn test_recommend_from_profile() {
        let mut store = FeatureStore::new();
        let cand1 = uid();
        let cand2 = uid();
        store.upsert(fv(cand1, &[1.0, 0.0]));
        store.upsert(fv(cand2, &[0.0, 1.0]));
        let profile = fv(uid(), &[1.0, 0.0]);
        let recs = recommend_from_profile(&store, &profile, &[cand1, cand2], 2);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].0, cand1);
    }
}
