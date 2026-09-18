//! User profile management.

use crate::error::{RecommendError, RecommendResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// User profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User ID
    pub user_id: Uuid,
    /// Preferred categories
    pub preferred_categories: HashMap<String, f32>,
    /// Disliked categories
    pub disliked_categories: HashMap<String, f32>,
    /// Average watch duration
    pub avg_watch_duration_ms: i64,
    /// Completion rate
    pub avg_completion_rate: f32,
    /// Viewing patterns
    pub viewing_patterns: ViewingPatterns,
    /// Content preferences
    pub content_preferences: ContentPreferences,
    /// Engagement level
    pub engagement_level: f32,
    /// Profile creation timestamp
    pub created_at: i64,
    /// Last updated timestamp
    pub updated_at: i64,
}

/// Viewing patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingPatterns {
    /// Preferred viewing times (hour of day)
    pub preferred_hours: Vec<u8>,
    /// Preferred days of week
    pub preferred_days: Vec<u8>,
    /// Average session duration (minutes)
    pub avg_session_duration_min: u32,
    /// Binge watching tendency (0-1)
    pub binge_tendency: f32,
}

impl Default for ViewingPatterns {
    fn default() -> Self {
        Self {
            preferred_hours: Vec::new(),
            preferred_days: Vec::new(),
            avg_session_duration_min: 30,
            binge_tendency: 0.5,
        }
    }
}

/// Content preferences
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentPreferences {
    /// Preferred content types
    pub content_types: HashMap<String, f32>,
    /// Preferred duration range (min, max in minutes)
    pub duration_preference: Option<(u32, u32)>,
    /// Quality preference (SD, HD, 4K, etc.)
    pub quality_preference: Option<String>,
    /// Language preferences
    pub language_preferences: Vec<String>,
}

impl UserProfile {
    /// Create a new user profile
    #[must_use]
    pub fn new(user_id: Uuid) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            user_id,
            preferred_categories: HashMap::new(),
            disliked_categories: HashMap::new(),
            avg_watch_duration_ms: 0,
            avg_completion_rate: 0.0,
            viewing_patterns: ViewingPatterns::default(),
            content_preferences: ContentPreferences::default(),
            engagement_level: 0.5,
            created_at: now,
            updated_at: now,
        }
    }

    /// Update category preference
    pub fn update_category_preference(&mut self, category: String, weight: f32) {
        if weight > 0.0 {
            *self.preferred_categories.entry(category).or_insert(0.0) += weight;
        } else {
            *self.disliked_categories.entry(category).or_insert(0.0) += weight.abs();
        }
        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Get category preference score
    #[must_use]
    pub fn get_category_score(&self, category: &str) -> f32 {
        let preference = self.preferred_categories.get(category).unwrap_or(&0.0);
        let dislike = self.disliked_categories.get(category).unwrap_or(&0.0);
        preference - dislike
    }

    /// Update from viewing behavior
    pub fn update_from_view(&mut self, watch_time_ms: i64, completion_rate: f32) {
        // Update average watch duration (exponential moving average)
        let alpha = 0.1;
        self.avg_watch_duration_ms = (alpha * watch_time_ms as f32
            + (1.0 - alpha) * self.avg_watch_duration_ms as f32)
            as i64;

        // Update average completion rate
        self.avg_completion_rate =
            alpha * completion_rate + (1.0 - alpha) * self.avg_completion_rate;

        // Update engagement level
        self.engagement_level = completion_rate.max(self.engagement_level * 0.9);

        self.updated_at = chrono::Utc::now().timestamp();
    }

    /// Get top preferred categories
    #[must_use]
    pub fn get_top_categories(&self, limit: usize) -> Vec<(String, f32)> {
        let mut categories: Vec<(String, f32)> = self
            .preferred_categories
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        categories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        categories.truncate(limit);
        categories
    }
}

/// User profile manager
pub struct UserProfileManager {
    /// User profiles
    profiles: HashMap<Uuid, UserProfile>,
}

impl UserProfileManager {
    /// Create a new profile manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Get or create user profile
    pub fn get_or_create_profile(&mut self, user_id: Uuid) -> &mut UserProfile {
        self.profiles
            .entry(user_id)
            .or_insert_with(|| UserProfile::new(user_id))
    }

    /// Get user profile
    ///
    /// # Errors
    ///
    /// Returns an error if user not found
    pub fn get_profile(&self, user_id: Uuid) -> RecommendResult<UserProfile> {
        self.profiles
            .get(&user_id)
            .cloned()
            .ok_or(RecommendError::UserNotFound(user_id))
    }

    /// Update from view event
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_from_view(
        &mut self,
        user_id: Uuid,
        _content_id: Uuid,
        watch_time_ms: i64,
        completed: bool,
    ) -> RecommendResult<()> {
        let profile = self.get_or_create_profile(user_id);
        let completion_rate = if completed { 1.0 } else { 0.5 };
        profile.update_from_view(watch_time_ms, completion_rate);
        Ok(())
    }

    /// Update from rating
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_from_rating(
        &mut self,
        user_id: Uuid,
        _content_id: Uuid,
        rating: f32,
    ) -> RecommendResult<()> {
        let profile = self.get_or_create_profile(user_id);
        let engagement_boost = (rating / 5.0) * 0.1;
        profile.engagement_level = (profile.engagement_level + engagement_boost).min(1.0);
        Ok(())
    }

    /// Get similar users based on profile similarity
    ///
    /// # Errors
    ///
    /// Returns an error if user not found
    pub fn get_similar_users(&self, user_id: Uuid, limit: usize) -> RecommendResult<Vec<Uuid>> {
        let profile = self
            .profiles
            .get(&user_id)
            .ok_or(RecommendError::UserNotFound(user_id))?;

        let mut similarities: Vec<(Uuid, f32)> = self
            .profiles
            .iter()
            .filter(|(id, _)| **id != user_id)
            .map(|(id, other_profile)| (*id, calculate_profile_similarity(profile, other_profile)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities.into_iter().map(|(id, _)| id).collect())
    }
}

impl Default for UserProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate similarity between two user profiles
fn calculate_profile_similarity(profile_a: &UserProfile, profile_b: &UserProfile) -> f32 {
    // Calculate category overlap (Jaccard similarity)
    let categories_a: std::collections::HashSet<_> =
        profile_a.preferred_categories.keys().collect();
    let categories_b: std::collections::HashSet<_> =
        profile_b.preferred_categories.keys().collect();

    let intersection = categories_a.intersection(&categories_b).count();
    let union = categories_a.union(&categories_b).count();

    if union == 0 {
        return 0.0;
    }

    let category_sim = intersection as f32 / union as f32;

    // Consider engagement similarity
    let engagement_diff = (profile_a.engagement_level - profile_b.engagement_level).abs();
    let engagement_sim = 1.0 - engagement_diff;

    // Weighted combination
    0.7 * category_sim + 0.3 * engagement_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_profile_creation() {
        let user_id = Uuid::new_v4();
        let profile = UserProfile::new(user_id);
        assert_eq!(profile.user_id, user_id);
        assert!(profile.preferred_categories.is_empty());
    }

    #[test]
    fn test_update_category_preference() {
        let mut profile = UserProfile::new(Uuid::new_v4());
        profile.update_category_preference(String::from("Action"), 1.0);
        assert_eq!(profile.get_category_score("Action"), 1.0);
    }

    #[test]
    fn test_update_from_view() {
        let mut profile = UserProfile::new(Uuid::new_v4());
        profile.update_from_view(60000, 0.8);
        assert!(profile.avg_watch_duration_ms > 0);
        assert!(profile.avg_completion_rate > 0.0);
    }

    #[test]
    fn test_get_top_categories() {
        let mut profile = UserProfile::new(Uuid::new_v4());
        profile.update_category_preference(String::from("Action"), 5.0);
        profile.update_category_preference(String::from("Drama"), 3.0);
        profile.update_category_preference(String::from("Comedy"), 4.0);

        let top = profile.get_top_categories(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "Action");
    }

    #[test]
    fn test_profile_manager() {
        let mut manager = UserProfileManager::new();
        let user_id = Uuid::new_v4();

        let profile = manager.get_or_create_profile(user_id);
        assert_eq!(profile.user_id, user_id);
    }
}
