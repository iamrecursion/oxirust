//! Visual similarity search implementation.

use crate::error::SearchResult;
use uuid::Uuid;

/// Visual search result
#[derive(Debug, Clone)]
pub struct VisualSearchResult {
    /// Asset ID
    pub asset_id: Uuid,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Distance metric
    pub distance: f32,
}

/// Visual search engine
pub struct VisualSearch {
    /// Feature database
    features: Vec<(Uuid, Vec<f32>)>,
}

impl VisualSearch {
    /// Create a new visual search engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add visual features for an asset
    pub fn add_features(&mut self, asset_id: Uuid, features: Vec<f32>) {
        self.features.push((asset_id, features));
    }

    /// Search for similar images
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn search_similar(
        &self,
        query_features: &[f32],
        limit: usize,
    ) -> SearchResult<Vec<VisualSearchResult>> {
        let mut results: Vec<_> = self
            .features
            .iter()
            .map(|(id, features)| {
                let distance = self.euclidean_distance(query_features, features);
                let similarity = 1.0 / (1.0 + distance);

                VisualSearchResult {
                    asset_id: *id,
                    similarity,
                    distance,
                }
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top results
        Ok(results.into_iter().take(limit).collect())
    }

    /// Calculate Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Calculate cosine similarity between two feature vectors
    #[allow(dead_code)]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

impl Default for VisualSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let search = VisualSearch::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let distance = search.euclidean_distance(&a, &b);
        assert!((distance - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        let search = VisualSearch::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let similarity = search.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_search_similar() {
        let mut search = VisualSearch::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        search.add_features(id1, vec![1.0, 2.0, 3.0]);
        search.add_features(id2, vec![4.0, 5.0, 6.0]);

        let results = search
            .search_similar(&[1.0, 2.0, 3.0], 10)
            .expect("should succeed in test");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id1); // Exact match should be first
    }
}
