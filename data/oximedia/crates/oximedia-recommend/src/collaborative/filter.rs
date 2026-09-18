//! Basic collaborative filter implementation.
//!
//! Provides a simple user-based collaborative filtering algorithm that
//! predicts ratings by computing a weighted average of similar users'
//! ratings, using Pearson correlation as the similarity measure.

use crate::error::{RecommendError, RecommendResult};
use std::collections::HashMap;
use uuid::Uuid;

/// A rating record associating a user, an item, and a numeric score.
#[derive(Debug, Clone)]
pub struct RatingRecord {
    /// User identifier.
    pub user_id: Uuid,
    /// Item identifier.
    pub item_id: Uuid,
    /// Numeric rating (expected range 0.0 – 5.0).
    pub rating: f32,
}

impl RatingRecord {
    /// Create a new rating record.
    #[must_use]
    pub fn new(user_id: Uuid, item_id: Uuid, rating: f32) -> Self {
        Self {
            user_id,
            item_id,
            rating,
        }
    }
}

/// Similarity method used to compare users.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SimilarityMethod {
    /// Pearson correlation coefficient.
    Pearson,
    /// Cosine similarity.
    Cosine,
}

/// Basic collaborative filter based on user-to-user similarity.
///
/// Ratings are stored in an in-memory sparse matrix (user → item → rating).
/// Predictions are computed as a weighted average of neighbour ratings,
/// truncated to the configured neighbourhood size.
pub struct CollaborativeFilter {
    /// Maximum number of neighbours considered per prediction.
    neighbourhood_size: usize,
    /// Similarity method used for neighbour selection.
    similarity_method: SimilarityMethod,
    /// Sparse user–item matrix: `user_id` → (`item_id` → rating)
    ratings: HashMap<Uuid, HashMap<Uuid, f32>>,
}

impl CollaborativeFilter {
    /// Create a new collaborative filter with the given neighbourhood size.
    #[must_use]
    pub fn new(neighbourhood_size: usize) -> Self {
        Self {
            neighbourhood_size,
            similarity_method: SimilarityMethod::Pearson,
            ratings: HashMap::new(),
        }
    }

    /// Set the similarity method.
    #[must_use]
    pub fn with_similarity(mut self, method: SimilarityMethod) -> Self {
        self.similarity_method = method;
        self
    }

    /// Add a batch of ratings to the filter.
    pub fn add_ratings(&mut self, records: &[RatingRecord]) {
        for record in records {
            self.ratings
                .entry(record.user_id)
                .or_default()
                .insert(record.item_id, record.rating);
        }
    }

    /// Add a single rating.
    pub fn add_rating(&mut self, user_id: Uuid, item_id: Uuid, rating: f32) {
        self.ratings
            .entry(user_id)
            .or_default()
            .insert(item_id, rating);
    }

    /// Predict the rating user `user_id` would give to `item_id`.
    ///
    /// Returns [`RecommendError::InsufficientData`] when no neighbours have
    /// rated the target item.
    ///
    /// # Errors
    ///
    /// Returns an error when prediction cannot be computed.
    pub fn predict(&self, user_id: Uuid, item_id: Uuid) -> RecommendResult<f32> {
        // Find neighbours that have rated item_id
        let mut neighbours: Vec<(Uuid, f32, f32)> = self
            .ratings
            .iter()
            .filter(|(&uid, _)| uid != user_id)
            .filter_map(|(&uid, item_map)| {
                item_map.get(&item_id).map(|&r| {
                    let sim = self.user_similarity(user_id, uid);
                    (uid, sim, r)
                })
            })
            .filter(|(_, sim, _)| *sim > 0.0)
            .collect();

        if neighbours.is_empty() {
            return Err(RecommendError::InsufficientData(format!(
                "No neighbours have rated item {item_id}"
            )));
        }

        // Sort by similarity descending and keep only the top-k
        neighbours.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbours.truncate(self.neighbourhood_size);

        // Weighted average
        let numerator: f32 = neighbours.iter().map(|(_, sim, r)| sim * r).sum();
        let denominator: f32 = neighbours.iter().map(|(_, sim, _)| sim).sum();

        if denominator < f32::EPSILON {
            return Err(RecommendError::InsufficientData(
                "Insufficient similarity mass for prediction".to_string(),
            ));
        }

        Ok((numerator / denominator).clamp(0.0, 5.0))
    }

    /// Recommend items to `user_id` that they have not yet rated, sorted by
    /// predicted score descending.
    ///
    /// # Errors
    ///
    /// Returns an error when the user cannot be found.
    pub fn recommend(&self, user_id: Uuid, limit: usize) -> RecommendResult<Vec<(Uuid, f32)>> {
        // Collect all items rated by other users
        let already_rated: std::collections::HashSet<Uuid> = self
            .ratings
            .get(&user_id)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default();

        let candidate_items: std::collections::HashSet<Uuid> = self
            .ratings
            .values()
            .flat_map(|m| m.keys().copied())
            .filter(|id| !already_rated.contains(id))
            .collect();

        let mut predictions: Vec<(Uuid, f32)> = candidate_items
            .into_iter()
            .filter_map(|item_id| {
                self.predict(user_id, item_id)
                    .ok()
                    .map(|score| (item_id, score))
            })
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(limit);

        Ok(predictions)
    }

    /// Compute the similarity between two users.
    #[must_use]
    pub fn user_similarity(&self, user_a: Uuid, user_b: Uuid) -> f32 {
        match self.similarity_method {
            SimilarityMethod::Pearson => self.pearson_similarity(user_a, user_b),
            SimilarityMethod::Cosine => self.cosine_similarity(user_a, user_b),
        }
    }

    /// Pearson correlation between two users over co-rated items.
    fn pearson_similarity(&self, user_a: Uuid, user_b: Uuid) -> f32 {
        let map_a = match self.ratings.get(&user_a) {
            Some(m) => m,
            None => return 0.0,
        };
        let map_b = match self.ratings.get(&user_b) {
            Some(m) => m,
            None => return 0.0,
        };

        let common: Vec<(f32, f32)> = map_a
            .iter()
            .filter_map(|(item, &ra)| map_b.get(item).map(|&rb| (ra, rb)))
            .collect();

        if common.len() < 2 {
            return 0.0;
        }

        let n = common.len() as f32;
        let mean_a: f32 = common.iter().map(|(r, _)| r).sum::<f32>() / n;
        let mean_b: f32 = common.iter().map(|(_, r)| r).sum::<f32>() / n;

        let mut num = 0.0_f32;
        let mut denom_a = 0.0_f32;
        let mut denom_b = 0.0_f32;

        for (ra, rb) in &common {
            let da = ra - mean_a;
            let db = rb - mean_b;
            num += da * db;
            denom_a += da * da;
            denom_b += db * db;
        }

        let denom = (denom_a * denom_b).sqrt();
        if denom < f32::EPSILON {
            return 0.0;
        }

        (num / denom).clamp(-1.0, 1.0).max(0.0) // keep only positive correlations
    }

    /// Cosine similarity between two users over co-rated items.
    fn cosine_similarity(&self, user_a: Uuid, user_b: Uuid) -> f32 {
        let map_a = match self.ratings.get(&user_a) {
            Some(m) => m,
            None => return 0.0,
        };
        let map_b = match self.ratings.get(&user_b) {
            Some(m) => m,
            None => return 0.0,
        };

        let common: Vec<(f32, f32)> = map_a
            .iter()
            .filter_map(|(item, &ra)| map_b.get(item).map(|&rb| (ra, rb)))
            .collect();

        if common.is_empty() {
            return 0.0;
        }

        let dot: f32 = common.iter().map(|(a, b)| a * b).sum();
        let norm_a: f32 = common.iter().map(|(a, _)| a * a).sum::<f32>().sqrt();
        let norm_b: f32 = common.iter().map(|(_, b)| b * b).sum::<f32>().sqrt();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
    }

    /// Number of users in the rating store.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.ratings.len()
    }

    /// Number of distinct items across all users.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.ratings
            .values()
            .flat_map(|m| m.keys())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
}

impl Default for CollaborativeFilter {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter() -> (CollaborativeFilter, Uuid, Uuid, Uuid, Uuid) {
        let mut cf = CollaborativeFilter::new(5);
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        cf.add_rating(u1, i1, 5.0);
        cf.add_rating(u1, i2, 3.0);
        cf.add_rating(u2, i1, 4.0);
        cf.add_rating(u2, i2, 2.0);
        (cf, u1, u2, i1, i2)
    }

    #[test]
    fn test_collaborative_filter_creation() {
        let cf = CollaborativeFilter::new(10);
        assert_eq!(cf.neighbourhood_size, 10);
        assert_eq!(cf.user_count(), 0);
    }

    #[test]
    fn test_add_rating() {
        let mut cf = CollaborativeFilter::new(5);
        let u = Uuid::new_v4();
        let i = Uuid::new_v4();
        cf.add_rating(u, i, 4.0);
        assert_eq!(cf.user_count(), 1);
        assert_eq!(cf.item_count(), 1);
    }

    #[test]
    fn test_pearson_similarity() {
        let (cf, u1, u2, _, _) = make_filter();
        let sim = cf.user_similarity(u1, u2);
        // Both users rated i1 higher than i2, so correlation should be positive
        assert!(sim > 0.0, "Expected positive similarity, got {sim}");
    }

    #[test]
    fn test_cosine_similarity() {
        let mut cf = CollaborativeFilter::new(5).with_similarity(SimilarityMethod::Cosine);
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        cf.add_rating(u1, i1, 3.0);
        cf.add_rating(u1, i2, 4.0);
        cf.add_rating(u2, i1, 3.0);
        cf.add_rating(u2, i2, 4.0);
        let sim = cf.user_similarity(u1, u2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical users, expected sim=1, got {sim}"
        );
    }

    #[test]
    fn test_predict() {
        // Build a fresh filter with enough overlap (>=2 co-rated items) for Pearson.
        let mut cf = CollaborativeFilter::new(5);
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        let new_item = Uuid::new_v4();
        // u1 and u2 both rated i1 and i2 (needed for Pearson, which requires >= 2 co-rated items)
        cf.add_rating(u1, i1, 5.0);
        cf.add_rating(u1, i2, 3.0);
        cf.add_rating(u2, i1, 4.0);
        cf.add_rating(u2, i2, 2.0);
        // Only u2 rated new_item; we want to predict what u1 would give it
        cf.add_rating(u2, new_item, 4.0);

        let result = cf.predict(u1, new_item);
        assert!(result.is_ok(), "Prediction failed: {:?}", result.err());
        let score = result.expect("should succeed in test");
        assert!((0.0..=5.0).contains(&score));
    }

    #[test]
    fn test_predict_insufficient_data() {
        let cf = CollaborativeFilter::new(5);
        let u = Uuid::new_v4();
        let i = Uuid::new_v4();
        let result = cf.predict(u, i);
        assert!(result.is_err());
    }

    #[test]
    fn test_recommend_returns_unrated_items() {
        // Build a fresh filter with u1 and u3 sharing >=2 co-rated items
        // so Pearson similarity is non-zero.
        let mut cf = CollaborativeFilter::new(5);
        let u1 = Uuid::new_v4();
        let u3 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        let new_item = Uuid::new_v4();

        cf.add_rating(u1, i1, 5.0);
        cf.add_rating(u1, i2, 3.0);
        cf.add_rating(u3, i1, 4.0);
        cf.add_rating(u3, i2, 2.0);
        cf.add_rating(u3, new_item, 5.0); // only u3 rated new_item

        let recs = cf.recommend(u1, 10).expect("should succeed in test");
        // new_item should appear since u1 has not rated it and u3 has positive similarity
        assert!(recs.iter().any(|(id, _)| *id == new_item));
    }

    #[test]
    fn test_add_ratings_batch() {
        let mut cf = CollaborativeFilter::new(5);
        let u = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        let records = vec![RatingRecord::new(u, i1, 5.0), RatingRecord::new(u, i2, 3.0)];
        cf.add_ratings(&records);
        assert_eq!(cf.user_count(), 1);
        assert_eq!(cf.item_count(), 2);
    }
}
