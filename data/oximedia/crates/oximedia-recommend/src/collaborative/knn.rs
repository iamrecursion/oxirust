//! K-nearest neighbors for collaborative filtering.

use super::matrix::UserItemMatrix;
use crate::error::{RecommendError, RecommendResult};
use uuid::Uuid;

/// K-nearest neighbors calculator
pub struct KnnCalculator {
    /// Number of neighbors to find
    k: usize,
}

impl KnnCalculator {
    /// Create a new KNN calculator
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Find similar users based on rating patterns.
    ///
    /// Returns an empty list when `user_id` is not yet present in the matrix
    /// (cold-start case) so that callers can gracefully proceed with no neighbours.
    ///
    /// # Errors
    ///
    /// Returns an error only if an internal computation error occurs.
    pub fn find_similar_users(
        &self,
        matrix: &UserItemMatrix,
        user_id: Uuid,
        limit: usize,
    ) -> RecommendResult<Vec<(Uuid, f32)>> {
        let user_ratings = match matrix.get_user_ratings(user_id) {
            Some(ratings) => ratings,
            None => return Ok(Vec::new()),
        };

        let mut similarities = Vec::new();

        // Compare with all other users
        for user_idx in 0..matrix.num_users() {
            if let Some(other_user_id) = matrix.get_user_id(user_idx) {
                if other_user_id == user_id {
                    continue;
                }

                if let Some(other_ratings) = matrix.get_user_ratings(other_user_id) {
                    let similarity = self.calculate_user_similarity(&user_ratings, &other_ratings);
                    if similarity > 0.0 {
                        similarities.push((other_user_id, similarity));
                    }
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Find similar items based on user rating patterns
    ///
    /// # Errors
    ///
    /// Returns an error if item not found or computation fails
    pub fn find_similar_items(
        &self,
        matrix: &UserItemMatrix,
        item_id: Uuid,
        limit: usize,
    ) -> RecommendResult<Vec<(Uuid, f32)>> {
        let item_ratings = matrix
            .get_item_ratings(item_id)
            .ok_or(RecommendError::ContentNotFound(item_id))?;

        let mut similarities = Vec::new();

        // Compare with all other items
        for item_idx in 0..matrix.num_items() {
            if let Some(other_item_id) = matrix.get_item_id(item_idx) {
                if other_item_id == item_id {
                    continue;
                }

                if let Some(other_ratings) = matrix.get_item_ratings(other_item_id) {
                    let similarity = self.calculate_item_similarity(&item_ratings, &other_ratings);
                    if similarity > 0.0 {
                        similarities.push((other_item_id, similarity));
                    }
                }
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Calculate similarity between two user rating vectors
    fn calculate_user_similarity(&self, ratings_a: &[f32], ratings_b: &[f32]) -> f32 {
        if ratings_a.len() != ratings_b.len() {
            return 0.0;
        }

        // Only consider items both users have rated
        let mut common_ratings_a = Vec::new();
        let mut common_ratings_b = Vec::new();

        for (rating_a, rating_b) in ratings_a.iter().zip(ratings_b.iter()) {
            if *rating_a > 0.0 && *rating_b > 0.0 {
                common_ratings_a.push(*rating_a);
                common_ratings_b.push(*rating_b);
            }
        }

        if common_ratings_a.len() < 2 {
            return 0.0;
        }

        // Calculate Pearson correlation
        self.pearson_correlation(&common_ratings_a, &common_ratings_b)
    }

    /// Calculate similarity between two item rating vectors
    fn calculate_item_similarity(&self, ratings_a: &[f32], ratings_b: &[f32]) -> f32 {
        if ratings_a.len() != ratings_b.len() {
            return 0.0;
        }

        // Only consider users who rated both items
        let mut common_ratings_a = Vec::new();
        let mut common_ratings_b = Vec::new();

        for (rating_a, rating_b) in ratings_a.iter().zip(ratings_b.iter()) {
            if *rating_a > 0.0 && *rating_b > 0.0 {
                common_ratings_a.push(*rating_a);
                common_ratings_b.push(*rating_b);
            }
        }

        if common_ratings_a.len() < 2 {
            return 0.0;
        }

        // Calculate cosine similarity
        self.cosine_similarity(&common_ratings_a, &common_ratings_b)
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let n = a.len() as f32;
        let mean_a: f32 = a.iter().sum::<f32>() / n;
        let mean_b: f32 = b.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let diff_a = x - mean_a;
            let diff_b = y - mean_b;
            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator < f32::EPSILON {
            return 0.0;
        }

        (numerator / denominator).clamp(-1.0, 1.0)
    }

    /// Calculate cosine similarity
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return 0.0;
        }

        (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_calculator_creation() {
        let knn = KnnCalculator::new(10);
        assert_eq!(knn.k, 10);
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let knn = KnnCalculator::new(10);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = knn.pearson_correlation(&a, &b);
        assert!((corr - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let knn = KnnCalculator::new(10);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = knn.pearson_correlation(&a, &b);
        assert!((corr + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let knn = KnnCalculator::new(10);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = knn.cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let knn = KnnCalculator::new(10);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = knn.cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
    }
}
