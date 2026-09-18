//! Hybrid recommendation combiner.

use crate::error::RecommendResult;
use crate::{Recommendation, RecommendationRequest};
use std::collections::HashMap;
use uuid::Uuid;

/// Hybrid recommendation combiner
pub struct HybridCombiner {
    /// Weights for different recommendation methods
    weights: MethodWeights,
}

/// Weights for different recommendation methods
#[derive(Debug, Clone)]
pub struct MethodWeights {
    /// Content-based weight
    pub content_based: f32,
    /// Collaborative filtering weight
    pub collaborative: f32,
    /// Trending weight
    pub trending: f32,
    /// Popularity weight
    pub popularity: f32,
}

impl Default for MethodWeights {
    fn default() -> Self {
        Self {
            content_based: 0.4,
            collaborative: 0.4,
            trending: 0.1,
            popularity: 0.1,
        }
    }
}

impl HybridCombiner {
    /// Create a new hybrid combiner
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: MethodWeights::default(),
        }
    }

    /// Set method weights
    pub fn set_weights(&mut self, weights: MethodWeights) {
        self.weights = weights;
    }

    /// Combine recommendations from multiple methods
    ///
    /// # Errors
    ///
    /// Returns an error if combination fails
    pub fn combine_recommendations(
        &self,
        recommendations_by_method: Vec<(String, Vec<Recommendation>)>,
    ) -> RecommendResult<Vec<Recommendation>> {
        let mut combined_scores: HashMap<Uuid, f32> = HashMap::new();
        let mut recommendation_map: HashMap<Uuid, Recommendation> = HashMap::new();

        // Combine scores from all methods
        for (method, recommendations) in recommendations_by_method {
            let weight = self.get_method_weight(&method);

            for rec in recommendations {
                let weighted_score = rec.score * weight;
                *combined_scores.entry(rec.content_id).or_insert(0.0) += weighted_score;

                // Store recommendation metadata
                recommendation_map.entry(rec.content_id).or_insert(rec);
            }
        }

        // Create combined recommendations
        let mut combined: Vec<Recommendation> = combined_scores
            .into_iter()
            .filter_map(|(content_id, score)| {
                recommendation_map.get(&content_id).map(|rec| {
                    let mut combined_rec = rec.clone();
                    combined_rec.score = score;
                    combined_rec
                })
            })
            .collect();

        // Sort by combined score
        combined.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (idx, rec) in combined.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }

        Ok(combined)
    }

    /// Get weight for a method
    fn get_method_weight(&self, method: &str) -> f32 {
        match method {
            "content_based" => self.weights.content_based,
            "collaborative" => self.weights.collaborative,
            "trending" => self.weights.trending,
            "popularity" => self.weights.popularity,
            _ => 0.0,
        }
    }

    /// Get hybrid recommendations
    ///
    /// # Errors
    ///
    /// Returns an error if recommendation generation fails
    pub fn recommend(
        &self,
        _request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // In a real implementation, this would call multiple recommenders
        // and combine their results using the weights
        Ok(Vec::new())
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize_weights(&mut self) {
        let total = self.weights.content_based
            + self.weights.collaborative
            + self.weights.trending
            + self.weights.popularity;

        if total > f32::EPSILON {
            self.weights.content_based /= total;
            self.weights.collaborative /= total;
            self.weights.trending /= total;
            self.weights.popularity /= total;
        }
    }
}

impl Default for HybridCombiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid strategy
#[derive(Debug, Clone, Copy)]
pub enum HybridStrategy {
    /// Weighted combination
    Weighted,
    /// Switching based on context
    Switching,
    /// Mixed (interleave results)
    Mixed,
    /// Cascade (fallback)
    Cascade,
}

/// Hybrid recommender using multiple strategies
pub struct HybridRecommender {
    /// Combiner
    combiner: HybridCombiner,
    /// Strategy to use
    strategy: HybridStrategy,
}

impl HybridRecommender {
    /// Create a new hybrid recommender
    #[must_use]
    pub fn new(strategy: HybridStrategy) -> Self {
        Self {
            combiner: HybridCombiner::new(),
            strategy,
        }
    }

    /// Get recommendations using hybrid approach
    ///
    /// # Errors
    ///
    /// Returns an error if recommendation generation fails
    pub fn recommend(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        match self.strategy {
            HybridStrategy::Weighted => self.weighted_recommend(request),
            HybridStrategy::Switching => self.switching_recommend(request),
            HybridStrategy::Mixed => self.mixed_recommend(request),
            HybridStrategy::Cascade => self.cascade_recommend(request),
        }
    }

    /// Weighted hybrid recommendation
    fn weighted_recommend(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        self.combiner.recommend(request)
    }

    /// Switching hybrid recommendation
    fn switching_recommend(
        &self,
        _request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // Switch between methods based on context
        Ok(Vec::new())
    }

    /// Mixed hybrid recommendation
    fn mixed_recommend(
        &self,
        _request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // Interleave results from different methods
        Ok(Vec::new())
    }

    /// Cascade hybrid recommendation
    fn cascade_recommend(
        &self,
        _request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // Try methods in sequence with fallback
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_combiner_creation() {
        let combiner = HybridCombiner::new();
        assert!((combiner.weights.content_based - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_method_weights_default() {
        let weights = MethodWeights::default();
        let total =
            weights.content_based + weights.collaborative + weights.trending + weights.popularity;
        assert!((total - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize_weights() {
        let mut combiner = HybridCombiner::new();
        combiner.weights.content_based = 2.0;
        combiner.weights.collaborative = 2.0;
        combiner.weights.trending = 2.0;
        combiner.weights.popularity = 2.0;

        combiner.normalize_weights();

        let total = combiner.weights.content_based
            + combiner.weights.collaborative
            + combiner.weights.trending
            + combiner.weights.popularity;

        assert!((total - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hybrid_recommender_creation() {
        let recommender = HybridRecommender::new(HybridStrategy::Weighted);
        assert!(matches!(recommender.strategy, HybridStrategy::Weighted));
    }
}
