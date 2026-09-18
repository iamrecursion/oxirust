//! Personalization score computation.
//!
//! `PersonalizationScore` combines multiple signals—category affinity,
//! recency preference, explicit ratings, and context—into a single
//! normalized personalization score that can be blended with other
//! recommendation signals.

use std::collections::HashMap;

/// User preference profile used for score computation.
#[derive(Debug, Clone, Default)]
pub struct UserPreferences {
    /// Affinity weights per category (0.0 – 1.0).
    pub category_affinities: HashMap<String, f32>,
    /// Preferred content duration in milliseconds (optional).
    pub preferred_duration_ms: Option<i64>,
    /// Average rating given by the user (used as baseline, 0.0 – 5.0).
    pub avg_rating_given: f32,
    /// Preferred languages (ISO 639-1 codes).
    pub preferred_languages: Vec<String>,
}

impl UserPreferences {
    /// Create an empty user preference profile.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a category affinity weight.
    pub fn set_affinity(&mut self, category: impl Into<String>, weight: f32) {
        self.category_affinities
            .insert(category.into(), weight.clamp(0.0, 1.0));
    }

    /// Get affinity for a category (0.0 if not set).
    #[must_use]
    pub fn affinity_for(&self, category: &str) -> f32 {
        self.category_affinities
            .get(category)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Factors used when computing the final personalization score.
#[derive(Debug, Clone)]
pub struct ScoreFactors {
    /// Weight of category affinity signal.
    pub category_weight: f32,
    /// Weight of content freshness (recency) signal.
    pub recency_weight: f32,
    /// Weight of average rating signal.
    pub rating_weight: f32,
    /// Weight of view-count popularity signal.
    pub popularity_weight: f32,
}

impl Default for ScoreFactors {
    fn default() -> Self {
        Self {
            category_weight: 0.4,
            recency_weight: 0.2,
            rating_weight: 0.3,
            popularity_weight: 0.1,
        }
    }
}

/// Input signals for a single content item.
#[derive(Debug, Clone)]
pub struct ContentSignals {
    /// Categories this content belongs to.
    pub categories: Vec<String>,
    /// Unix timestamp when the content was created.
    pub created_at: i64,
    /// Average rating for this content (0.0 – 5.0, optional).
    pub avg_rating: Option<f32>,
    /// Total view count.
    pub view_count: u64,
}

/// Computes a unified personalization score for a content item given user preferences.
pub struct PersonalizationScore {
    /// Blending weights for the individual signals.
    factors: ScoreFactors,
    /// Unix timestamp representing "now" (injected for testability).
    now_timestamp: i64,
    /// Recency half-life in seconds (controls how fast freshness decays).
    recency_half_life_secs: f64,
    /// Maximum view count used to normalize the popularity signal.
    max_view_count: u64,
}

impl PersonalizationScore {
    /// Create a scorer using the current wall-clock time and default factors.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factors: ScoreFactors::default(),
            now_timestamp: chrono::Utc::now().timestamp(),
            recency_half_life_secs: 7.0 * 24.0 * 3600.0, // one week
            max_view_count: 10_000_000,
        }
    }

    /// Override the "now" timestamp (useful for deterministic testing).
    #[must_use]
    pub fn with_now(mut self, now: i64) -> Self {
        self.now_timestamp = now;
        self
    }

    /// Set custom blending factors.
    #[must_use]
    pub fn with_factors(mut self, factors: ScoreFactors) -> Self {
        self.factors = factors;
        self
    }

    /// Set the recency half-life in days.
    #[must_use]
    pub fn with_recency_half_life_days(mut self, days: f64) -> Self {
        self.recency_half_life_secs = days * 24.0 * 3600.0;
        self
    }

    /// Set the reference maximum view count for popularity normalization.
    #[must_use]
    pub fn with_max_view_count(mut self, max: u64) -> Self {
        self.max_view_count = max.max(1);
        self
    }

    /// Compute the personalization score in range \[0.0, 1.0\].
    ///
    /// Individual signals are normalized to `[0, 1]` before blending.
    #[must_use]
    pub fn compute(&self, prefs: &UserPreferences, signals: &ContentSignals) -> f32 {
        let category_score = self.compute_category_score(prefs, signals);
        let recency_score = self.compute_recency_score(signals.created_at);
        let rating_score = self.compute_rating_score(signals.avg_rating, prefs.avg_rating_given);
        let popularity_score = self.compute_popularity_score(signals.view_count);

        let score = self.factors.category_weight * category_score
            + self.factors.recency_weight * recency_score
            + self.factors.rating_weight * rating_score
            + self.factors.popularity_weight * popularity_score;

        score.clamp(0.0, 1.0)
    }

    /// Compute the category affinity sub-score.
    #[must_use]
    pub fn compute_category_score(&self, prefs: &UserPreferences, signals: &ContentSignals) -> f32 {
        if signals.categories.is_empty() || prefs.category_affinities.is_empty() {
            return 0.0;
        }

        let total: f32 = signals
            .categories
            .iter()
            .map(|cat| prefs.affinity_for(cat))
            .sum();

        (total / signals.categories.len() as f32).clamp(0.0, 1.0)
    }

    /// Compute the recency sub-score using exponential decay.
    ///
    /// Score = exp(-λ · age) where λ = ln(2) / `half_life`.
    #[must_use]
    pub fn compute_recency_score(&self, created_at: i64) -> f32 {
        let age_secs = (self.now_timestamp - created_at).max(0) as f64;
        let lambda = std::f64::consts::LN_2 / self.recency_half_life_secs;
        ((-lambda * age_secs).exp() as f32).clamp(0.0, 1.0)
    }

    /// Compute the rating sub-score.
    ///
    /// If no content rating is available, falls back to 0.5 (neutral).
    /// The score reflects how well the content's rating matches the user's
    /// average expected quality (normalized 0 – 5 → 0 – 1).
    #[must_use]
    pub fn compute_rating_score(&self, avg_rating: Option<f32>, user_avg_rating: f32) -> f32 {
        let content_rating = avg_rating.unwrap_or(2.5); // neutral default
        let normalized = content_rating / 5.0;

        // Boost content that matches or exceeds the user's average expectations
        let user_normalized = user_avg_rating / 5.0;
        let match_bonus = if content_rating >= user_avg_rating {
            0.1
        } else {
            0.0
        };

        (normalized * 0.9 + user_normalized * 0.1 + match_bonus).clamp(0.0, 1.0)
    }

    /// Compute the popularity sub-score (logarithmic scaling).
    #[must_use]
    pub fn compute_popularity_score(&self, view_count: u64) -> f32 {
        if view_count == 0 {
            return 0.0;
        }

        let log_views = (view_count as f64 + 1.0).ln();
        let log_max = (self.max_view_count as f64 + 1.0).ln();

        (log_views / log_max).clamp(0.0, 1.0) as f32
    }
}

impl Default for PersonalizationScore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_prefs() -> UserPreferences {
        let mut prefs = UserPreferences::new();
        prefs.set_affinity("Action", 0.8);
        prefs.set_affinity("Comedy", 0.5);
        prefs.avg_rating_given = 3.5;
        prefs
    }

    fn base_signals() -> ContentSignals {
        ContentSignals {
            categories: vec!["Action".to_string(), "Thriller".to_string()],
            created_at: chrono::Utc::now().timestamp() - 3600, // 1 hour old
            avg_rating: Some(4.2),
            view_count: 150_000,
        }
    }

    #[test]
    fn test_personalization_score_creation() {
        let scorer = PersonalizationScore::new();
        assert!((scorer.factors.category_weight - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_score_range() {
        let scorer = PersonalizationScore::new();
        let prefs = base_prefs();
        let signals = base_signals();
        let score = scorer.compute(&prefs, &signals);
        assert!((0.0..=1.0).contains(&score), "Score {score} out of range");
    }

    #[test]
    fn test_category_score_known_affinity() {
        let scorer = PersonalizationScore::new();
        let prefs = base_prefs();
        let signals = ContentSignals {
            categories: vec!["Action".to_string()],
            created_at: 0,
            avg_rating: None,
            view_count: 0,
        };
        let cat_score = scorer.compute_category_score(&prefs, &signals);
        assert!((cat_score - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_category_score_no_affinity() {
        let scorer = PersonalizationScore::new();
        let prefs = base_prefs();
        let signals = ContentSignals {
            categories: vec!["Documentary".to_string()],
            created_at: 0,
            avg_rating: None,
            view_count: 0,
        };
        let cat_score = scorer.compute_category_score(&prefs, &signals);
        assert!((cat_score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_recency_score_fresh_content() {
        let now = chrono::Utc::now().timestamp();
        let scorer = PersonalizationScore::new().with_now(now);
        let score = scorer.compute_recency_score(now - 60); // 1 minute old
        assert!(
            score > 0.99,
            "Very recent content should score near 1, got {score}"
        );
    }

    #[test]
    fn test_recency_score_old_content() {
        let now = chrono::Utc::now().timestamp();
        let scorer = PersonalizationScore::new().with_now(now);
        let year_ago = now - 365 * 24 * 3600;
        let score = scorer.compute_recency_score(year_ago);
        assert!(
            score < 0.1,
            "Year-old content should score low, got {score}"
        );
    }

    #[test]
    fn test_rating_score_high_quality() {
        let scorer = PersonalizationScore::new();
        let score = scorer.compute_rating_score(Some(5.0), 3.0);
        assert!(score > 0.9, "5-star content should score high, got {score}");
    }

    #[test]
    fn test_popularity_score_zero_views() {
        let scorer = PersonalizationScore::new();
        let score = scorer.compute_popularity_score(0);
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_popularity_score_grows_with_views() {
        let scorer = PersonalizationScore::new();
        let low = scorer.compute_popularity_score(100);
        let high = scorer.compute_popularity_score(1_000_000);
        assert!(
            high > low,
            "More views should yield higher popularity score"
        );
    }

    #[test]
    fn test_with_custom_factors() {
        let factors = ScoreFactors {
            category_weight: 1.0,
            recency_weight: 0.0,
            rating_weight: 0.0,
            popularity_weight: 0.0,
        };
        let scorer = PersonalizationScore::new().with_factors(factors);
        let mut prefs = UserPreferences::new();
        prefs.set_affinity("Action", 0.7);
        let signals = ContentSignals {
            categories: vec!["Action".to_string()],
            created_at: 0,
            avg_rating: None,
            view_count: 0,
        };
        // With only category weight, score should equal the category affinity
        let score = scorer.compute(&prefs, &signals);
        assert!(
            (score - 0.7).abs() < 0.01,
            "Score should be ~0.7, got {score}"
        );
    }
}
