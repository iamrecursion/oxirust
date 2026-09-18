//! Fresh vs popular content balance.

use crate::error::RecommendResult;
use crate::Recommendation;

/// Freshness balancer
pub struct FreshnessBalancer {
    /// Weight for fresh content (0-1)
    freshness_weight: f32,
    /// Age threshold for "fresh" content (days)
    fresh_threshold_days: u32,
}

impl FreshnessBalancer {
    /// Create a new freshness balancer
    #[must_use]
    pub fn new(freshness_weight: f32, fresh_threshold_days: u32) -> Self {
        Self {
            freshness_weight: freshness_weight.clamp(0.0, 1.0),
            fresh_threshold_days,
        }
    }

    /// Balance recommendations between fresh and popular
    ///
    /// # Errors
    ///
    /// Returns an error if balancing fails
    pub fn balance(
        &self,
        mut recommendations: Vec<Recommendation>,
    ) -> RecommendResult<Vec<Recommendation>> {
        let now = chrono::Utc::now().timestamp();

        // Adjust scores based on freshness
        for rec in &mut recommendations {
            let age_days = (now - rec.metadata.created_at) / 86400;
            let freshness_boost = self.calculate_freshness_boost(age_days as u32);
            rec.score =
                rec.score * (1.0 - self.freshness_weight) + freshness_boost * self.freshness_weight;
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

        Ok(recommendations)
    }

    /// Calculate freshness boost for content age
    fn calculate_freshness_boost(&self, age_days: u32) -> f32 {
        if age_days <= self.fresh_threshold_days {
            1.0
        } else {
            let decay = (age_days - self.fresh_threshold_days) as f32 / 365.0;
            (1.0 - decay).max(0.0)
        }
    }

    /// Check if content is fresh
    #[must_use]
    pub fn is_fresh(&self, created_at: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        let age_days = (now - created_at) / 86400;
        age_days <= i64::from(self.fresh_threshold_days)
    }
}

impl Default for FreshnessBalancer {
    fn default() -> Self {
        Self::new(0.2, 7) // 20% weight, 7 days fresh threshold
    }
}

/// Content recency calculator
pub struct RecencyCalculator;

impl RecencyCalculator {
    /// Calculate recency score (1.0 for brand new, decays with age)
    #[must_use]
    pub fn calculate_score(created_at: i64, half_life_days: u32) -> f32 {
        let now = chrono::Utc::now().timestamp();
        let age_days = (now - created_at) as f32 / 86400.0;

        let decay_rate = (2.0_f32).ln() / half_life_days as f32;
        (-decay_rate * age_days).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freshness_balancer() {
        let balancer = FreshnessBalancer::new(0.3, 7);
        assert_eq!(balancer.fresh_threshold_days, 7);
    }

    #[test]
    fn test_is_fresh() {
        let balancer = FreshnessBalancer::default();
        let now = chrono::Utc::now().timestamp();
        assert!(balancer.is_fresh(now));

        let old = now - (30 * 86400); // 30 days ago
        assert!(!balancer.is_fresh(old));
    }

    #[test]
    fn test_recency_score() {
        let now = chrono::Utc::now().timestamp();
        let score = RecencyCalculator::calculate_score(now, 7);
        assert!((score - 1.0).abs() < 0.01);
    }
}
