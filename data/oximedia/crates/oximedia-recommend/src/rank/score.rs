//! Final ranking and scoring.

use crate::error::RecommendResult;
use crate::Recommendation;

/// Rank recommendations by score
///
/// # Errors
///
/// Returns an error if ranking fails
pub fn rank_recommendations(
    mut recommendations: Vec<Recommendation>,
) -> RecommendResult<Vec<Recommendation>> {
    // Sort by score (descending)
    recommendations.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign ranks
    for (idx, rec) in recommendations.iter_mut().enumerate() {
        rec.rank = idx + 1;
    }

    Ok(recommendations)
}

/// Learning to rank scorer
pub struct LearningToRankScorer {
    /// Feature weights
    weights: Vec<f32>,
}

impl LearningToRankScorer {
    /// Create a new LTR scorer
    #[must_use]
    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }

    /// Calculate score from features
    #[must_use]
    pub fn score(&self, features: &[f32]) -> f32 {
        features
            .iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }
}

impl Default for LearningToRankScorer {
    fn default() -> Self {
        Self::new(vec![1.0; 10])
    }
}

/// Scoring function types
#[derive(Debug, Clone, Copy)]
pub enum ScoringFunction {
    /// Linear combination
    Linear,
    /// `RankNet`
    RankNet,
    /// `LambdaMART`
    LambdaMART,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContentMetadata;
    use uuid::Uuid;

    #[test]
    fn test_rank_recommendations() {
        let recs = vec![
            Recommendation {
                content_id: Uuid::new_v4(),
                score: 0.5,
                rank: 0,
                reasons: vec![],
                metadata: ContentMetadata {
                    title: String::from("Test"),
                    description: None,
                    categories: vec![],
                    duration_ms: None,
                    thumbnail_url: None,
                    created_at: 0,
                    avg_rating: None,
                    view_count: 0,
                },
                explanation: None,
            },
            Recommendation {
                content_id: Uuid::new_v4(),
                score: 0.9,
                rank: 0,
                reasons: vec![],
                metadata: ContentMetadata {
                    title: String::from("Test2"),
                    description: None,
                    categories: vec![],
                    duration_ms: None,
                    thumbnail_url: None,
                    created_at: 0,
                    avg_rating: None,
                    view_count: 0,
                },
                explanation: None,
            },
        ];

        let ranked = rank_recommendations(recs).expect("should succeed in test");
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[0].score, 0.9);
    }

    #[test]
    fn test_ltr_scorer() {
        let scorer = LearningToRankScorer::new(vec![0.5, 0.3, 0.2]);
        let features = vec![1.0, 1.0, 1.0];
        let score = scorer.score(&features);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }
}
