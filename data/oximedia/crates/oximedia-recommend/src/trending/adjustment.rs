//! Trending adjustment based on view counts.
//!
//! `TrendingAdjustment` applies a multiplier to recommendation scores based on
//! how many views a piece of content has accumulated over a sliding time window.
//! The multiplier grows logarithmically so that viral content gets a meaningful
//! boost without completely drowning out non-trending items.

use crate::Recommendation;

/// Configuration for trending-based score adjustment.
#[derive(Debug, Clone)]
pub struct TrendingAdjustment {
    /// Base of the logarithm used to scale view counts.
    log_base: f32,
    /// Maximum multiplier applied to the base score (caps the boost).
    max_multiplier: f32,
    /// Minimum view count threshold below which no boost is applied.
    min_views_threshold: u64,
    /// Weight controlling how strongly trending adjusts the final score (0.0 – 1.0).
    adjustment_weight: f32,
}

impl TrendingAdjustment {
    /// Create a new `TrendingAdjustment` with default parameters.
    ///
    /// Defaults:
    /// - `log_base`: 10.0
    /// - `max_multiplier`: 2.0
    /// - `min_views_threshold`: 100
    /// - `adjustment_weight`: 0.2
    #[must_use]
    pub fn new() -> Self {
        Self {
            log_base: 10.0,
            max_multiplier: 2.0,
            min_views_threshold: 100,
            adjustment_weight: 0.2,
        }
    }

    /// Set the logarithm base (must be > 1.0; clamped otherwise).
    #[must_use]
    pub fn with_log_base(mut self, base: f32) -> Self {
        self.log_base = base.max(1.001);
        self
    }

    /// Set the maximum multiplier cap (must be >= 1.0).
    #[must_use]
    pub fn with_max_multiplier(mut self, max: f32) -> Self {
        self.max_multiplier = max.max(1.0);
        self
    }

    /// Set the minimum view count before any boost is applied.
    #[must_use]
    pub fn with_min_views_threshold(mut self, threshold: u64) -> Self {
        self.min_views_threshold = threshold;
        self
    }

    /// Set the adjustment weight (clamped to 0.0 – 1.0).
    #[must_use]
    pub fn with_adjustment_weight(mut self, weight: f32) -> Self {
        self.adjustment_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Compute the trending multiplier for the given view count.
    ///
    /// Returns a value in `[1.0, max_multiplier]`.
    #[must_use]
    pub fn compute_multiplier(&self, view_count: u64) -> f32 {
        if view_count < self.min_views_threshold {
            return 1.0;
        }

        let log_views = (view_count as f32).log(self.log_base);
        let normalized = log_views / (self.log_base); // normalize by log_base so values are ~0-1 for typical ranges
        let boost = 1.0 + self.adjustment_weight * normalized;

        boost.min(self.max_multiplier)
    }

    /// Apply trending adjustment to a single recommendation's score.
    ///
    /// The adjusted score is computed as:
    /// `adjusted = score * multiplier`, then clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn adjust_score(&self, score: f32, view_count: u64) -> f32 {
        let multiplier = self.compute_multiplier(view_count);
        (score * multiplier).clamp(0.0, 1.0)
    }

    /// Apply trending adjustments to a list of recommendations in-place,
    /// using each item's `metadata.view_count`.
    pub fn adjust_recommendations(&self, recommendations: &mut Vec<Recommendation>) {
        for rec in recommendations.iter_mut() {
            rec.score = self.adjust_score(rec.score, rec.metadata.view_count);
        }

        // Re-sort by adjusted scores
        recommendations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign ranks
        for (idx, rec) in recommendations.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }
    }

    /// Return a new sorted vector with trending adjustments applied.
    #[must_use]
    pub fn apply(&self, recommendations: Vec<Recommendation>) -> Vec<Recommendation> {
        let mut adjusted = recommendations;
        self.adjust_recommendations(&mut adjusted);
        adjusted
    }
}

impl Default for TrendingAdjustment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentMetadata, Recommendation};
    use uuid::Uuid;

    fn make_rec(score: f32, view_count: u64) -> Recommendation {
        Recommendation {
            content_id: Uuid::new_v4(),
            score,
            rank: 1,
            reasons: vec![],
            metadata: ContentMetadata {
                title: String::from("Test"),
                description: None,
                categories: vec![],
                duration_ms: None,
                thumbnail_url: None,
                created_at: 0,
                avg_rating: None,
                view_count,
            },
            explanation: None,
        }
    }

    #[test]
    fn test_default_creation() {
        let adj = TrendingAdjustment::new();
        assert!((adj.log_base - 10.0).abs() < f32::EPSILON);
        assert!((adj.max_multiplier - 2.0).abs() < f32::EPSILON);
        assert_eq!(adj.min_views_threshold, 100);
    }

    #[test]
    fn test_multiplier_below_threshold_is_one() {
        let adj = TrendingAdjustment::new().with_min_views_threshold(1000);
        assert!((adj.compute_multiplier(50) - 1.0).abs() < f32::EPSILON);
        assert!((adj.compute_multiplier(999) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiplier_above_threshold_grows() {
        let adj = TrendingAdjustment::new();
        let low = adj.compute_multiplier(200);
        let high = adj.compute_multiplier(100_000);
        assert!(
            high >= low,
            "Higher views should yield equal or larger multiplier"
        );
    }

    #[test]
    fn test_multiplier_capped_at_max() {
        let adj = TrendingAdjustment::new().with_max_multiplier(1.5);
        let mult = adj.compute_multiplier(u64::MAX);
        assert!(mult <= 1.5 + f32::EPSILON, "Multiplier must not exceed max");
    }

    #[test]
    fn test_adjust_score_bounded() {
        let adj = TrendingAdjustment::new();
        let adjusted = adj.adjust_score(0.9, 1_000_000);
        assert!((0.0..=1.0).contains(&adjusted));
    }

    #[test]
    fn test_apply_preserves_count() {
        let adj = TrendingAdjustment::new();
        let recs = vec![make_rec(0.8, 10_000), make_rec(0.5, 500), make_rec(0.3, 50)];
        let adjusted = adj.apply(recs);
        assert_eq!(adjusted.len(), 3);
    }

    #[test]
    fn test_apply_reassigns_ranks() {
        let adj = TrendingAdjustment::new();
        let recs = vec![make_rec(0.5, 1_000_000), make_rec(0.9, 10)];
        let adjusted = adj.apply(recs);
        assert_eq!(adjusted[0].rank, 1);
        assert_eq!(adjusted[1].rank, 2);
    }

    #[test]
    fn test_adjust_recommendations_in_place() {
        let adj = TrendingAdjustment::new();
        let mut recs = vec![make_rec(0.5, 500_000)];
        adj.adjust_recommendations(&mut recs);
        // High view count should boost above 0.5
        assert!(recs[0].score >= 0.5);
    }
}
