//! Trending score calculation.

use serde::{Deserialize, Serialize};

/// Trending score calculator
pub struct TrendingScoreCalculator {
    /// Weight for recency
    recency_weight: f32,
    /// Weight for velocity
    velocity_weight: f32,
    /// Weight for engagement
    engagement_weight: f32,
}

impl TrendingScoreCalculator {
    /// Create a new trending score calculator
    #[must_use]
    pub fn new(recency_weight: f32, velocity_weight: f32, engagement_weight: f32) -> Self {
        Self {
            recency_weight,
            velocity_weight,
            engagement_weight,
        }
    }

    /// Calculate trending score
    #[must_use]
    pub fn calculate(&self, metrics: &TrendingMetrics) -> f32 {
        let recency_score = self.calculate_recency_score(metrics.age_hours);
        let velocity_score = self.calculate_velocity_score(metrics.views_per_hour);
        let engagement_score = metrics.engagement_rate;

        self.recency_weight * recency_score
            + self.velocity_weight * velocity_score
            + self.engagement_weight * engagement_score
    }

    /// Calculate recency score
    fn calculate_recency_score(&self, age_hours: f32) -> f32 {
        // Exponential decay
        let decay_rate = 0.1;
        (-decay_rate * age_hours).exp()
    }

    /// Calculate velocity score
    fn calculate_velocity_score(&self, views_per_hour: f32) -> f32 {
        // Logarithmic scaling
        if views_per_hour < 1.0 {
            return 0.0;
        }
        (views_per_hour.ln() / 10.0).min(1.0)
    }
}

impl Default for TrendingScoreCalculator {
    fn default() -> Self {
        Self::new(0.3, 0.4, 0.3)
    }
}

/// Trending metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingMetrics {
    /// Age in hours
    pub age_hours: f32,
    /// Views per hour
    pub views_per_hour: f32,
    /// Engagement rate
    pub engagement_rate: f32,
    /// Total views
    pub total_views: u64,
}

impl Default for TrendingMetrics {
    fn default() -> Self {
        Self {
            age_hours: 0.0,
            views_per_hour: 0.0,
            engagement_rate: 0.0,
            total_views: 0,
        }
    }
}

/// Viral coefficient calculator
pub struct ViralCoefficientCalculator;

impl ViralCoefficientCalculator {
    /// Calculate viral coefficient
    ///
    /// K = (Average invites per user) × (Conversion rate)
    #[must_use]
    pub fn calculate(avg_shares_per_user: f32, conversion_rate: f32) -> f32 {
        avg_shares_per_user * conversion_rate
    }

    /// Check if content is going viral (K > 1)
    #[must_use]
    pub fn is_viral(avg_shares_per_user: f32, conversion_rate: f32) -> bool {
        Self::calculate(avg_shares_per_user, conversion_rate) > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trending_score_calculator() {
        let calculator = TrendingScoreCalculator::default();
        let metrics = TrendingMetrics {
            age_hours: 1.0,
            views_per_hour: 100.0,
            engagement_rate: 0.5,
            total_views: 100,
        };

        let score = calculator.calculate(&metrics);
        assert!(score > 0.0);
    }

    #[test]
    fn test_viral_coefficient() {
        let k = ViralCoefficientCalculator::calculate(2.0, 0.6);
        assert!((k - 1.2).abs() < f32::EPSILON);
        assert!(ViralCoefficientCalculator::is_viral(2.0, 0.6));
    }
}
