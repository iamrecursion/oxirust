//! Self-similarity matrix computation.

use crate::{MirError, MirResult};

/// Similarity matrix computer.
pub struct SimilarityMatrix;

impl SimilarityMatrix {
    /// Create a new similarity matrix computer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute self-similarity matrix from features.
    ///
    /// # Errors
    ///
    /// Returns error if computation fails.
    pub fn compute(&self, features: &[Vec<f32>]) -> MirResult<(Vec<f32>, usize)> {
        if features.is_empty() {
            return Err(MirError::InsufficientData(
                "No features for similarity matrix".to_string(),
            ));
        }

        let n = features.len();
        let mut matrix = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let similarity = self.cosine_similarity(&features[i], &features[j]);
                matrix[i * n + j] = similarity;
            }
        }

        Ok((matrix, n))
    }

    /// Compute cosine similarity between two feature vectors.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

impl Default for SimilarityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_matrix_creation() {
        let _sim = SimilarityMatrix::new();
    }

    #[test]
    fn test_cosine_similarity() {
        let sim = SimilarityMatrix::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = sim.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);
    }
}
