//! Re-ranking algorithms.

use crate::Recommendation;

/// Re-ranking strategy
pub struct Reranker {
    /// Diversity weight
    diversity_weight: f32,
}

impl Reranker {
    /// Create a new reranker
    #[must_use]
    pub fn new(diversity_weight: f32) -> Self {
        Self {
            diversity_weight: diversity_weight.clamp(0.0, 1.0),
        }
    }

    /// Re-rank using Maximum Marginal Relevance (MMR)
    pub fn rerank_mmr(&self, recommendations: &mut [Recommendation]) {
        if recommendations.is_empty() {
            return;
        }

        // Sort by adjusted score
        recommendations.sort_by(|a, b| {
            let score_a = self.calculate_mmr_score(a);
            let score_b = self.calculate_mmr_score(b);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign ranks
        for (idx, rec) in recommendations.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }
    }

    /// Calculate MMR score
    fn calculate_mmr_score(&self, rec: &Recommendation) -> f32 {
        // Simplified MMR - in real implementation would consider similarity to already selected items
        rec.score * (1.0 - self.diversity_weight)
    }

    /// Interleave recommendations from multiple sources
    #[must_use]
    pub fn interleave(sources: Vec<Vec<Recommendation>>) -> Vec<Recommendation> {
        let mut result = Vec::new();
        let max_len = sources.iter().map(Vec::len).max().unwrap_or(0);

        for i in 0..max_len {
            for source in &sources {
                if i < source.len() {
                    result.push(source[i].clone());
                }
            }
        }

        // Re-assign ranks
        for (idx, rec) in result.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }

        result
    }
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_creation() {
        let reranker = Reranker::new(0.5);
        assert!((reranker.diversity_weight - 0.5).abs() < f32::EPSILON);
    }
}
