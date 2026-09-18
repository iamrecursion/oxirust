//! User profile management for recommendation personalization.
//!
//! Tracks per-user view history, category affinities, and preference signals
//! to drive content-based and hybrid recommendation strategies.

#![allow(dead_code)]

use std::collections::HashMap;

/// Categories of media content a user may have affinity for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UserCategory {
    /// News and current events
    News,
    /// Sports and athletics
    Sports,
    /// Entertainment and comedy
    Entertainment,
    /// Documentary and factual
    Documentary,
    /// Drama and narrative fiction
    Drama,
    /// Music videos and concerts
    Music,
    /// Science and technology
    Science,
    /// Travel and lifestyle
    Travel,
    /// Education and tutorials
    Education,
    /// Gaming and esports
    Gaming,
}

impl UserCategory {
    /// Human-readable label for this category.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::News => "News",
            Self::Sports => "Sports",
            Self::Entertainment => "Entertainment",
            Self::Documentary => "Documentary",
            Self::Drama => "Drama",
            Self::Music => "Music",
            Self::Science => "Science",
            Self::Travel => "Travel",
            Self::Education => "Education",
            Self::Gaming => "Gaming",
        }
    }

    /// Returns all defined category variants.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::News,
            Self::Sports,
            Self::Entertainment,
            Self::Documentary,
            Self::Drama,
            Self::Music,
            Self::Science,
            Self::Travel,
            Self::Education,
            Self::Gaming,
        ]
    }
}

/// A single view event recorded for a user.
#[derive(Debug, Clone)]
pub struct ViewEvent {
    /// Identifier of the viewed item.
    pub item_id: String,
    /// Categories associated with this item.
    pub categories: Vec<UserCategory>,
    /// Fraction of content watched (0.0–1.0).
    pub completion: f32,
    /// Unix timestamp of the view.
    pub timestamp: i64,
}

impl ViewEvent {
    /// Create a new view event.
    #[must_use]
    pub fn new(
        item_id: impl Into<String>,
        categories: Vec<UserCategory>,
        completion: f32,
        timestamp: i64,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            categories,
            completion: completion.clamp(0.0, 1.0),
            timestamp,
        }
    }
}

/// Per-user profile tracking views and category preferences.
#[derive(Debug, Clone, Default)]
pub struct UserProfile {
    /// Ordered history of view events (oldest first).
    views: Vec<ViewEvent>,
    /// Accumulated affinity scores per category.
    category_scores: HashMap<UserCategory, f64>,
}

impl UserProfile {
    /// Create an empty profile.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a view event and update category affinity scores.
    ///
    /// Affinity is weighted by the completion fraction of the view.
    pub fn add_view(&mut self, event: ViewEvent) {
        let weight = f64::from(event.completion);
        for &cat in &event.categories {
            *self.category_scores.entry(cat).or_insert(0.0) += weight;
        }
        self.views.push(event);
    }

    /// Total number of recorded views.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Returns the accumulated affinity score for a given category (0.0 if unseen).
    #[must_use]
    pub fn category_affinity(&self, category: UserCategory) -> f64 {
        self.category_scores.get(&category).copied().unwrap_or(0.0)
    }

    /// Returns a sorted list of `(category, affinity)` pairs, highest first.
    #[must_use]
    pub fn top_categories(&self) -> Vec<(UserCategory, f64)> {
        let mut pairs: Vec<(UserCategory, f64)> = self
            .category_scores
            .iter()
            .map(|(&cat, &score)| (cat, score))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Returns the most-recently viewed item ID, if any.
    #[must_use]
    pub fn last_viewed(&self) -> Option<&str> {
        self.views.last().map(|e| e.item_id.as_str())
    }

    /// Returns a slice of all view events.
    #[must_use]
    pub fn view_history(&self) -> &[ViewEvent] {
        &self.views
    }

    /// Clears all recorded views and resets affinity scores.
    pub fn reset(&mut self) {
        self.views.clear();
        self.category_scores.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str, cats: Vec<UserCategory>, completion: f32, ts: i64) -> ViewEvent {
        ViewEvent::new(id, cats, completion, ts)
    }

    #[test]
    fn test_category_label_news() {
        assert_eq!(UserCategory::News.label(), "News");
    }

    #[test]
    fn test_category_label_gaming() {
        assert_eq!(UserCategory::Gaming.label(), "Gaming");
    }

    #[test]
    fn test_all_categories_count() {
        assert_eq!(UserCategory::all().len(), 10);
    }

    #[test]
    fn test_profile_starts_empty() {
        let p = UserProfile::new();
        assert_eq!(p.view_count(), 0);
    }

    #[test]
    fn test_add_view_increments_count() {
        let mut p = UserProfile::new();
        p.add_view(make_event("v1", vec![UserCategory::News], 1.0, 1000));
        assert_eq!(p.view_count(), 1);
    }

    #[test]
    fn test_category_affinity_zero_for_unseen() {
        let p = UserProfile::new();
        assert!((p.category_affinity(UserCategory::Sports) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_category_affinity_increases_with_views() {
        let mut p = UserProfile::new();
        p.add_view(make_event("v1", vec![UserCategory::Sports], 0.5, 1000));
        p.add_view(make_event("v2", vec![UserCategory::Sports], 1.0, 2000));
        assert!((p.category_affinity(UserCategory::Sports) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_partial_completion_weighted_correctly() {
        let mut p = UserProfile::new();
        p.add_view(make_event("v1", vec![UserCategory::Music], 0.25, 500));
        let aff = p.category_affinity(UserCategory::Music);
        assert!((aff - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_completion_clamped_to_one() {
        let ev = ViewEvent::new("v1", vec![], 1.5, 0);
        assert!((ev.completion - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_completion_clamped_to_zero() {
        let ev = ViewEvent::new("v1", vec![], -0.3, 0);
        assert!((ev.completion - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_top_categories_ordering() {
        let mut p = UserProfile::new();
        p.add_view(make_event("v1", vec![UserCategory::Drama], 0.4, 100));
        p.add_view(make_event("v2", vec![UserCategory::Drama], 0.8, 200));
        p.add_view(make_event("v3", vec![UserCategory::News], 0.2, 300));
        let top = p.top_categories();
        assert_eq!(top[0].0, UserCategory::Drama);
    }

    #[test]
    fn test_last_viewed_none_when_empty() {
        let p = UserProfile::new();
        assert!(p.last_viewed().is_none());
    }

    #[test]
    fn test_last_viewed_returns_most_recent() {
        let mut p = UserProfile::new();
        p.add_view(make_event("first", vec![], 1.0, 1));
        p.add_view(make_event("second", vec![], 1.0, 2));
        assert_eq!(p.last_viewed(), Some("second"));
    }

    #[test]
    fn test_reset_clears_views_and_affinity() {
        let mut p = UserProfile::new();
        p.add_view(make_event("v1", vec![UserCategory::Gaming], 1.0, 1));
        p.reset();
        assert_eq!(p.view_count(), 0);
        assert!((p.category_affinity(UserCategory::Gaming) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_view_history_slice_length() {
        let mut p = UserProfile::new();
        for i in 0..5_i64 {
            p.add_view(make_event(&format!("v{i}"), vec![], 1.0, i));
        }
        assert_eq!(p.view_history().len(), 5);
    }

    #[test]
    fn test_multi_category_single_view() {
        let mut p = UserProfile::new();
        p.add_view(make_event(
            "v1",
            vec![UserCategory::Science, UserCategory::Education],
            1.0,
            0,
        ));
        assert!((p.category_affinity(UserCategory::Science) - 1.0).abs() < 1e-9);
        assert!((p.category_affinity(UserCategory::Education) - 1.0).abs() < 1e-9);
    }
}
