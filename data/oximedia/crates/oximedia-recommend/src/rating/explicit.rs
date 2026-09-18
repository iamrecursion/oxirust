//! Explicit rating handling.

use crate::error::{RecommendError, RecommendResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Explicit rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplicitRating {
    /// User ID
    pub user_id: Uuid,
    /// Content ID
    pub content_id: Uuid,
    /// Rating value (0-5)
    pub rating: f32,
    /// Timestamp
    pub timestamp: i64,
    /// Optional review text
    pub review: Option<String>,
}

impl ExplicitRating {
    /// Create a new explicit rating
    ///
    /// # Errors
    ///
    /// Returns an error if rating is out of range
    pub fn new(user_id: Uuid, content_id: Uuid, rating: f32) -> RecommendResult<Self> {
        if !(0.0..=5.0).contains(&rating) {
            return Err(RecommendError::InvalidRating(rating));
        }

        Ok(Self {
            user_id,
            content_id,
            rating,
            timestamp: chrono::Utc::now().timestamp(),
            review: None,
        })
    }

    /// Create with review
    ///
    /// # Errors
    ///
    /// Returns an error if rating is out of range
    pub fn with_review(
        user_id: Uuid,
        content_id: Uuid,
        rating: f32,
        review: String,
    ) -> RecommendResult<Self> {
        let mut explicit_rating = Self::new(user_id, content_id, rating)?;
        explicit_rating.review = Some(review);
        Ok(explicit_rating)
    }
}

/// Rating manager
pub struct RatingManager {
    /// Explicit ratings
    ratings: HashMap<(Uuid, Uuid), ExplicitRating>,
    /// Implicit ratings
    implicit_ratings: HashMap<(Uuid, Uuid), f32>,
}

impl RatingManager {
    /// Create a new rating manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            ratings: HashMap::new(),
            implicit_ratings: HashMap::new(),
        }
    }

    /// Record an explicit rating
    ///
    /// # Errors
    ///
    /// Returns an error if rating is invalid
    pub fn record_rating(
        &mut self,
        user_id: Uuid,
        content_id: Uuid,
        rating: f32,
    ) -> RecommendResult<()> {
        let explicit_rating = ExplicitRating::new(user_id, content_id, rating)?;
        self.ratings.insert((user_id, content_id), explicit_rating);
        Ok(())
    }

    /// Get user's rating for content
    #[must_use]
    pub fn get_rating(&self, user_id: Uuid, content_id: Uuid) -> Option<f32> {
        self.ratings.get(&(user_id, content_id)).map(|r| r.rating)
    }

    /// Get all ratings for content
    #[must_use]
    pub fn get_content_ratings(&self, content_id: Uuid) -> Vec<f32> {
        self.ratings
            .values()
            .filter(|r| r.content_id == content_id)
            .map(|r| r.rating)
            .collect()
    }

    /// Get average rating for content
    #[must_use]
    pub fn get_average_rating(&self, content_id: Uuid) -> Option<f32> {
        let ratings = self.get_content_ratings(content_id);
        if ratings.is_empty() {
            return None;
        }

        let sum: f32 = ratings.iter().sum();
        Some(sum / ratings.len() as f32)
    }

    /// Get user's all ratings
    #[must_use]
    pub fn get_user_ratings(&self, user_id: Uuid) -> Vec<(Uuid, f32)> {
        self.ratings
            .values()
            .filter(|r| r.user_id == user_id)
            .map(|r| (r.content_id, r.rating))
            .collect()
    }

    /// Update implicit rating from viewing behavior
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_implicit_rating(
        &mut self,
        user_id: Uuid,
        content_id: Uuid,
        watch_time_ms: i64,
        completed: bool,
    ) -> RecommendResult<()> {
        // Calculate implicit rating based on behavior
        let base_rating = if completed { 4.0 } else { 3.0 };

        // Adjust based on watch time (normalize to hours)
        let watch_time_hours = watch_time_ms as f32 / 3_600_000.0;
        let time_bonus = (watch_time_hours * 0.5).min(1.0);

        let implicit_rating = (base_rating + time_bonus).min(5.0);

        self.implicit_ratings
            .insert((user_id, content_id), implicit_rating);

        Ok(())
    }

    /// Get combined rating (explicit + implicit)
    #[must_use]
    pub fn get_combined_rating(&self, user_id: Uuid, content_id: Uuid) -> Option<f32> {
        let explicit = self.get_rating(user_id, content_id);
        let implicit = self.implicit_ratings.get(&(user_id, content_id)).copied();

        match (explicit, implicit) {
            (Some(exp), Some(imp)) => Some(exp * 0.7 + imp * 0.3), // Prefer explicit
            (Some(exp), None) => Some(exp),
            (None, Some(imp)) => Some(imp),
            (None, None) => None,
        }
    }

    /// Get rating statistics for content
    #[must_use]
    pub fn get_rating_statistics(&self, content_id: Uuid) -> RatingStatistics {
        let ratings = self.get_content_ratings(content_id);

        if ratings.is_empty() {
            return RatingStatistics::default();
        }

        let count = ratings.len();
        let sum: f32 = ratings.iter().sum();
        let average = sum / count as f32;

        // Calculate standard deviation
        let variance: f32 =
            ratings.iter().map(|r| (r - average).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        // Count distribution
        let mut distribution = [0usize; 6]; // 0-5 stars
        for rating in &ratings {
            let index = rating.round() as usize;
            if index < 6 {
                distribution[index] += 1;
            }
        }

        RatingStatistics {
            count,
            average,
            std_dev,
            distribution,
        }
    }
}

impl Default for RatingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Rating statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatingStatistics {
    /// Number of ratings
    pub count: usize,
    /// Average rating
    pub average: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Rating distribution (0-5 stars)
    pub distribution: [usize; 6],
}

impl Default for RatingStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            average: 0.0,
            std_dev: 0.0,
            distribution: [0; 6],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_rating_creation() {
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();
        let rating = ExplicitRating::new(user_id, content_id, 4.5);

        assert!(rating.is_ok());
        let rating = rating.expect("should succeed in test");
        assert_eq!(rating.rating, 4.5);
    }

    #[test]
    fn test_invalid_rating() {
        let result = ExplicitRating::new(Uuid::new_v4(), Uuid::new_v4(), 6.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rating_manager() {
        let mut manager = RatingManager::new();
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        manager
            .record_rating(user_id, content_id, 4.5)
            .expect("should succeed in test");

        let rating = manager.get_rating(user_id, content_id);
        assert_eq!(rating, Some(4.5));
    }

    #[test]
    fn test_average_rating() {
        let mut manager = RatingManager::new();
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        manager
            .record_rating(user1, content_id, 4.0)
            .expect("should succeed in test");
        manager
            .record_rating(user2, content_id, 5.0)
            .expect("should succeed in test");

        let avg = manager.get_average_rating(content_id);
        assert_eq!(avg, Some(4.5));
    }

    #[test]
    fn test_rating_statistics() {
        let mut manager = RatingManager::new();
        let content_id = Uuid::new_v4();

        for i in 1..=5 {
            manager
                .record_rating(Uuid::new_v4(), content_id, i as f32)
                .expect("should succeed in test");
        }

        let stats = manager.get_rating_statistics(content_id);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.average, 3.0);
    }
}
