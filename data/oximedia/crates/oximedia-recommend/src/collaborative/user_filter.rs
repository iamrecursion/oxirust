//! User-based collaborative filtering with cosine similarity.
//!
//! This module provides lightweight user-profile management and a
//! collaborative filter that recommends items based on similar users'
//! behaviour.

#![allow(dead_code)]

use std::collections::HashMap;

/// A user profile storing explicit ratings and watch history.
pub struct UserProfile {
    /// Unique user identifier.
    pub user_id: u64,
    /// Map from item ID to rating (0.0 – 5.0 or similar scale).
    pub ratings: HashMap<u64, f32>,
    /// Ordered watch history (item IDs).
    pub watch_history: Vec<u64>,
}

impl UserProfile {
    /// Create a new, empty user profile.
    #[must_use]
    pub fn new(user_id: u64) -> Self {
        Self {
            user_id,
            ratings: HashMap::new(),
            watch_history: Vec::new(),
        }
    }

    /// Record an explicit rating for an item.
    pub fn rate(&mut self, item_id: u64, rating: f32) {
        self.ratings.insert(item_id, rating);
    }

    /// Return `true` if this item appears in the watch history.
    #[must_use]
    pub fn has_watched(&self, item_id: u64) -> bool {
        self.watch_history.contains(&item_id)
    }

    /// Compute the average of all explicit ratings.
    ///
    /// Returns `0.0` if no ratings have been recorded.
    #[must_use]
    pub fn avg_rating(&self) -> f64 {
        if self.ratings.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.ratings.values().map(|&r| f64::from(r)).sum();
        sum / self.ratings.len() as f64
    }
}

/// Compute the cosine similarity between two sparse rating vectors.
///
/// Only items rated by *both* users contribute to the dot product.
/// Returns `0.0` if there are no common items or either vector is zero.
#[must_use]
pub fn cosine_similarity_sparse(a: &HashMap<u64, f32>, b: &HashMap<u64, f32>) -> f64 {
    let dot: f64 = a
        .iter()
        .filter_map(|(id, &va)| b.get(id).map(|&vb| f64::from(va) * f64::from(vb)))
        .sum();

    let norm_a: f64 = a
        .values()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .values()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// A pair of users with a computed similarity score.
pub struct UserSimilarity {
    /// First user ID.
    pub user_a: u64,
    /// Second user ID.
    pub user_b: u64,
    /// Cosine similarity in the range \[0, 1\].
    pub similarity: f64,
}

/// A collaborative filtering engine built on top of user profiles.
pub struct CollaborativeFilter {
    /// All registered user profiles.
    pub profiles: Vec<UserProfile>,
}

impl CollaborativeFilter {
    /// Create a new, empty filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Register a user profile with the filter.
    pub fn add_profile(&mut self, p: UserProfile) {
        self.profiles.push(p);
    }

    /// Return the `k` most similar users to `user_id`, sorted by descending
    /// similarity.  The target user itself is excluded.
    #[must_use]
    pub fn top_k_similar(&self, user_id: u64, k: usize) -> Vec<UserSimilarity> {
        let target = match self.profiles.iter().find(|p| p.user_id == user_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut scored: Vec<UserSimilarity> = self
            .profiles
            .iter()
            .filter(|p| p.user_id != user_id)
            .map(|other| {
                let sim = cosine_similarity_sparse(&target.ratings, &other.ratings);
                UserSimilarity {
                    user_a: user_id,
                    user_b: other.user_id,
                    similarity: sim,
                }
            })
            .collect();

        scored.sort_by(|x, y| {
            y.similarity
                .partial_cmp(&x.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }

    /// Recommend up to `n` items for `user_id`.
    ///
    /// Items are scored by the weighted sum of ratings from similar users.
    /// Items the target user has already rated are excluded.
    #[must_use]
    pub fn recommend(&self, user_id: u64, n: usize) -> Vec<u64> {
        let target = match self.profiles.iter().find(|p| p.user_id == user_id) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let similar = self.top_k_similar(user_id, self.profiles.len());

        let mut scores: HashMap<u64, f64> = HashMap::new();

        for sim_entry in &similar {
            let other = match self.profiles.iter().find(|p| p.user_id == sim_entry.user_b) {
                Some(p) => p,
                None => continue,
            };
            for (&item_id, &rating) in &other.ratings {
                if !target.ratings.contains_key(&item_id) {
                    *scores.entry(item_id).or_insert(0.0) +=
                        sim_entry.similarity * f64::from(rating);
                }
            }
        }

        let mut ranked: Vec<(u64, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(n);
        ranked.into_iter().map(|(id, _)| id).collect()
    }
}

impl Default for CollaborativeFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(user_id: u64, ratings: &[(u64, f32)]) -> UserProfile {
        let mut p = UserProfile::new(user_id);
        for &(item, r) in ratings {
            p.rate(item, r);
        }
        p
    }

    #[test]
    fn test_user_profile_new() {
        let p = UserProfile::new(42);
        assert_eq!(p.user_id, 42);
        assert!(p.ratings.is_empty());
        assert!(p.watch_history.is_empty());
    }

    #[test]
    fn test_rate_and_avg_rating() {
        let mut p = UserProfile::new(1);
        p.rate(10, 4.0);
        p.rate(20, 2.0);
        assert!((p.avg_rating() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_avg_rating_empty() {
        let p = UserProfile::new(1);
        assert!((p.avg_rating()).abs() < 1e-9);
    }

    #[test]
    fn test_has_watched() {
        let mut p = UserProfile::new(1);
        p.watch_history.push(99);
        assert!(p.has_watched(99));
        assert!(!p.has_watched(100));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = HashMap::new();
        a.insert(1u64, 1.0f32);
        a.insert(2u64, 1.0f32);
        let sim = cosine_similarity_sparse(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_no_overlap() {
        let mut a = HashMap::new();
        a.insert(1u64, 1.0f32);
        let mut b = HashMap::new();
        b.insert(2u64, 1.0f32);
        let sim = cosine_similarity_sparse(&a, &b);
        assert!((sim).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: HashMap<u64, f32> = HashMap::new();
        let b: HashMap<u64, f32> = HashMap::new();
        assert!((cosine_similarity_sparse(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_collaborative_filter_new() {
        let cf = CollaborativeFilter::new();
        assert!(cf.profiles.is_empty());
    }

    #[test]
    fn test_top_k_similar_unknown_user() {
        let cf = CollaborativeFilter::new();
        assert!(cf.top_k_similar(999, 5).is_empty());
    }

    #[test]
    fn test_top_k_similar_basic() {
        let mut cf = CollaborativeFilter::new();
        cf.add_profile(make_profile(1, &[(10, 5.0), (20, 3.0)]));
        cf.add_profile(make_profile(2, &[(10, 5.0), (20, 3.0)]));
        cf.add_profile(make_profile(3, &[(10, 1.0), (20, 1.0)]));
        let top = cf.top_k_similar(1, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].user_b, 2);
    }

    #[test]
    fn test_recommend_excludes_rated_items() {
        let mut cf = CollaborativeFilter::new();
        cf.add_profile(make_profile(1, &[(10, 5.0)]));
        cf.add_profile(make_profile(2, &[(10, 5.0), (20, 4.0)]));
        let recs = cf.recommend(1, 5);
        // item 10 already rated by user 1 → should not appear
        assert!(!recs.contains(&10));
        assert!(recs.contains(&20));
    }

    #[test]
    fn test_recommend_unknown_user() {
        let cf = CollaborativeFilter::new();
        assert!(cf.recommend(999, 5).is_empty());
    }
}
