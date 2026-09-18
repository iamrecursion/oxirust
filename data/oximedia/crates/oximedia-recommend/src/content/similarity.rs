//! Content similarity calculation for recommendations.

use crate::error::{RecommendError, RecommendResult};
use crate::{ContentMetadata, Recommendation, RecommendationReason, RecommendationRequest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Content similarity calculator
pub struct ContentSimilarityCalculator {
    /// Feature vectors for content
    feature_vectors: HashMap<Uuid, super::vector::ContentVector>,
}

impl ContentSimilarityCalculator {
    /// Create a new similarity calculator
    #[must_use]
    pub fn new() -> Self {
        Self {
            feature_vectors: HashMap::new(),
        }
    }

    /// Add content with features
    pub fn add_content(&mut self, content_id: Uuid, vector: super::vector::ContentVector) {
        self.feature_vectors.insert(content_id, vector);
    }

    /// Calculate similarity between two content items
    ///
    /// # Errors
    ///
    /// Returns an error if content not found
    pub fn calculate_similarity(&self, content_a: Uuid, content_b: Uuid) -> RecommendResult<f32> {
        let vec_a = self
            .feature_vectors
            .get(&content_a)
            .ok_or(RecommendError::ContentNotFound(content_a))?;
        let vec_b = self
            .feature_vectors
            .get(&content_b)
            .ok_or(RecommendError::ContentNotFound(content_b))?;

        Ok(super::distance::cosine_similarity(vec_a, vec_b))
    }

    /// Find similar content
    ///
    /// # Errors
    ///
    /// Returns an error if content not found
    pub fn find_similar(
        &self,
        content_id: Uuid,
        limit: usize,
    ) -> RecommendResult<Vec<(Uuid, f32)>> {
        let vec = self
            .feature_vectors
            .get(&content_id)
            .ok_or(RecommendError::ContentNotFound(content_id))?;

        let mut similarities: Vec<(Uuid, f32)> = self
            .feature_vectors
            .iter()
            .filter(|(id, _)| **id != content_id)
            .map(|(id, other_vec)| (*id, super::distance::cosine_similarity(vec, other_vec)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);

        Ok(similarities)
    }
}

impl Default for ContentSimilarityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Content-based recommender
pub struct ContentRecommender {
    /// Similarity calculator
    similarity_calculator: ContentSimilarityCalculator,
    /// Content metadata
    content_metadata: HashMap<Uuid, ContentMetadata>,
}

impl ContentRecommender {
    /// Create a new content recommender
    #[must_use]
    pub fn new() -> Self {
        Self {
            similarity_calculator: ContentSimilarityCalculator::new(),
            content_metadata: HashMap::new(),
        }
    }

    /// Add content to the recommender
    pub fn add_content(
        &mut self,
        content_id: Uuid,
        metadata: ContentMetadata,
        features: super::features::ContentFeatures,
    ) {
        let vector = super::vector::ContentVector::from_features(&features);
        self.similarity_calculator.add_content(content_id, vector);
        self.content_metadata.insert(content_id, metadata);
    }

    /// Get content-based recommendations
    ///
    /// When `request.content_id` is `None`, no content seed is available and an empty list
    /// is returned rather than an error, allowing callers to gracefully handle the missing
    /// seed without special-casing the strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if the specified content is not found in the similarity index
    pub fn recommend(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        let base_content = match request.content_id {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };

        let similar = self
            .similarity_calculator
            .find_similar(base_content, request.limit * 2)?;

        let recommendations: Vec<Recommendation> = similar
            .into_iter()
            .enumerate()
            .filter_map(|(idx, (content_id, similarity))| {
                self.content_metadata
                    .get(&content_id)
                    .map(|metadata| Recommendation {
                        content_id,
                        score: similarity,
                        rank: idx + 1,
                        reasons: vec![RecommendationReason::SimilarToLiked {
                            content_id: base_content,
                            similarity,
                        }],
                        metadata: metadata.clone(),
                        explanation: None,
                    })
            })
            .take(request.limit)
            .collect();

        Ok(recommendations)
    }
}

impl Default for ContentRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Similarity metrics for content comparison
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Jaccard index
    Jaccard,
    /// Pearson correlation
    Pearson,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_calculator_creation() {
        let calculator = ContentSimilarityCalculator::new();
        assert_eq!(calculator.feature_vectors.len(), 0);
    }

    #[test]
    fn test_content_recommender_creation() {
        let recommender = ContentRecommender::new();
        assert_eq!(recommender.content_metadata.len(), 0);
    }

    #[test]
    fn test_similarity_metric_variants() {
        let metrics = [
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::Jaccard,
            SimilarityMetric::Pearson,
        ];
        assert_eq!(metrics.len(), 4);
    }
}
