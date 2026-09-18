//! Rating prediction for collaborative filtering.

use super::matrix::UserItemMatrix;
use crate::error::RecommendResult;
use uuid::Uuid;

/// Rating predictor using collaborative filtering
pub struct RatingPredictor {
    /// Baseline predictor
    baseline: BaselinePredictor,
}

impl RatingPredictor {
    /// Create a new rating predictor
    #[must_use]
    pub fn new() -> Self {
        Self {
            baseline: BaselinePredictor::new(),
        }
    }

    /// Predict rating for user-item pair
    ///
    /// # Errors
    ///
    /// Returns an error if prediction fails
    pub fn predict(
        &self,
        matrix: &UserItemMatrix,
        user_id: Uuid,
        item_id: Uuid,
    ) -> RecommendResult<f32> {
        // Try collaborative prediction first
        if let Ok(rating) = self.collaborative_predict(matrix, user_id, item_id) {
            return Ok(rating);
        }

        // Fall back to baseline prediction
        self.baseline.predict(matrix, user_id, item_id)
    }

    /// Collaborative filtering prediction using user-based CF
    fn collaborative_predict(
        &self,
        matrix: &UserItemMatrix,
        user_id: Uuid,
        item_id: Uuid,
    ) -> RecommendResult<f32> {
        // Get users who rated this item
        let item_ratings = matrix
            .get_item_ratings(item_id)
            .ok_or(crate::error::RecommendError::ContentNotFound(item_id))?;

        // Calculate weighted average of similar users' ratings
        let mut weighted_sum = 0.0;
        let mut similarity_sum = 0.0;

        for (idx, &rating) in item_ratings.iter().enumerate() {
            if rating > 0.0 {
                if let Some(other_user_id) = matrix.get_user_id(idx) {
                    if other_user_id != user_id {
                        // Calculate similarity (simplified)
                        let similarity = self.calculate_similarity(matrix, user_id, other_user_id);
                        weighted_sum += similarity * rating;
                        similarity_sum += similarity.abs();
                    }
                }
            }
        }

        if similarity_sum > f32::EPSILON {
            Ok((weighted_sum / similarity_sum).clamp(0.0, 5.0))
        } else {
            Err(crate::error::RecommendError::insufficient_data(
                "No similar users found",
            ))
        }
    }

    /// Calculate user similarity (simplified)
    fn calculate_similarity(&self, matrix: &UserItemMatrix, user_a: Uuid, user_b: Uuid) -> f32 {
        let ratings_a = matrix.get_user_ratings(user_a).unwrap_or_default();
        let ratings_b = matrix.get_user_ratings(user_b).unwrap_or_default();

        // Pearson correlation (simplified)
        let mut common_count = 0;
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;

        for (r_a, r_b) in ratings_a.iter().zip(ratings_b.iter()) {
            if *r_a > 0.0 && *r_b > 0.0 {
                sum_a += r_a;
                sum_b += r_b;
                common_count += 1;
            }
        }

        if common_count < 2 {
            return 0.0;
        }

        let mean_a = sum_a / common_count as f32;
        let mean_b = sum_b / common_count as f32;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for (r_a, r_b) in ratings_a.iter().zip(ratings_b.iter()) {
            if *r_a > 0.0 && *r_b > 0.0 {
                let diff_a = r_a - mean_a;
                let diff_b = r_b - mean_b;
                numerator += diff_a * diff_b;
                sum_sq_a += diff_a * diff_a;
                sum_sq_b += diff_b * diff_b;
            }
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator < f32::EPSILON {
            0.0
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    }
}

impl Default for RatingPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Baseline predictor using global and user/item biases
pub struct BaselinePredictor {
    /// Global mean rating
    global_mean: f32,
}

impl BaselinePredictor {
    /// Create a new baseline predictor
    #[must_use]
    pub fn new() -> Self {
        Self { global_mean: 3.0 }
    }

    /// Calculate global mean from matrix
    pub fn fit(&mut self, matrix: &UserItemMatrix) {
        let mut sum = 0.0;
        let mut count = 0;

        for user_idx in 0..matrix.num_users() {
            if let Some(user_id) = matrix.get_user_id(user_idx) {
                if let Some(ratings) = matrix.get_user_ratings(user_id) {
                    for &rating in &ratings {
                        if rating > 0.0 {
                            sum += rating;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count > 0 {
            self.global_mean = sum / count as f32;
        }
    }

    /// Predict rating using baseline (global mean + biases)
    ///
    /// # Errors
    ///
    /// Returns an error if prediction fails
    pub fn predict(
        &self,
        matrix: &UserItemMatrix,
        user_id: Uuid,
        item_id: Uuid,
    ) -> RecommendResult<f32> {
        // Calculate user bias
        let user_bias = self.calculate_user_bias(matrix, user_id);

        // Calculate item bias
        let item_bias = self.calculate_item_bias(matrix, item_id);

        // Baseline prediction
        Ok((self.global_mean + user_bias + item_bias).clamp(0.0, 5.0))
    }

    /// Calculate user rating bias
    fn calculate_user_bias(&self, matrix: &UserItemMatrix, user_id: Uuid) -> f32 {
        let Some(ratings) = matrix.get_user_ratings(user_id) else {
            return 0.0;
        };

        let rated: Vec<f32> = ratings.iter().copied().filter(|&r| r > 0.0).collect();

        if rated.is_empty() {
            return 0.0;
        }

        let user_mean = rated.iter().sum::<f32>() / rated.len() as f32;
        user_mean - self.global_mean
    }

    /// Calculate item rating bias
    fn calculate_item_bias(&self, matrix: &UserItemMatrix, item_id: Uuid) -> f32 {
        let Some(ratings) = matrix.get_item_ratings(item_id) else {
            return 0.0;
        };

        let rated: Vec<f32> = ratings.iter().copied().filter(|&r| r > 0.0).collect();

        if rated.is_empty() {
            return 0.0;
        }

        let item_mean = rated.iter().sum::<f32>() / rated.len() as f32;
        item_mean - self.global_mean
    }
}

impl Default for BaselinePredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_predictor_creation() {
        let predictor = RatingPredictor::new();
        assert!((predictor.baseline.global_mean - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_baseline_predictor_creation() {
        let predictor = BaselinePredictor::new();
        assert!((predictor.global_mean - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_baseline_fit() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user1 = Uuid::new_v4();
        let item1 = Uuid::new_v4();
        let item2 = Uuid::new_v4();

        matrix.set_rating(user1, item1, 4.0);
        matrix.set_rating(user1, item2, 5.0);

        let mut predictor = BaselinePredictor::new();
        predictor.fit(&matrix);

        assert!((predictor.global_mean - 4.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_baseline_predict() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user1 = Uuid::new_v4();
        let item1 = Uuid::new_v4();

        matrix.set_rating(user1, item1, 4.0);

        let predictor = BaselinePredictor::new();
        let prediction = predictor.predict(&matrix, user1, item1);

        assert!(prediction.is_ok());
        let rating = prediction.expect("should succeed in test");
        assert!((0.0..=5.0).contains(&rating));
    }
}
