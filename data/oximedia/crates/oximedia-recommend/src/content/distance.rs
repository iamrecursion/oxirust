//! Distance and similarity metrics for content comparison.

use super::vector::ContentVector;

/// Calculate cosine similarity between two vectors
#[must_use]
pub fn cosine_similarity(a: &ContentVector, b: &ContentVector) -> f32 {
    if a.dense_features.is_empty() || b.dense_features.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a
        .dense_features
        .iter()
        .zip(b.dense_features.iter())
        .map(|(x, y)| x * y)
        .sum();

    let norm_a = a.norm();
    let norm_b = b.norm();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Calculate Euclidean distance between two vectors
#[must_use]
pub fn euclidean_distance(a: &ContentVector, b: &ContentVector) -> f32 {
    if a.dense_features.len() != b.dense_features.len() {
        return f32::INFINITY;
    }

    a.dense_features
        .iter()
        .zip(b.dense_features.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
#[must_use]
pub fn manhattan_distance(a: &ContentVector, b: &ContentVector) -> f32 {
    if a.dense_features.len() != b.dense_features.len() {
        return f32::INFINITY;
    }

    a.dense_features
        .iter()
        .zip(b.dense_features.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Calculate Pearson correlation between two vectors
#[must_use]
pub fn pearson_correlation(a: &ContentVector, b: &ContentVector) -> f32 {
    if a.dense_features.len() != b.dense_features.len() || a.dense_features.is_empty() {
        return 0.0;
    }

    let n = a.dense_features.len() as f32;
    let mean_a: f32 = a.dense_features.iter().sum::<f32>() / n;
    let mean_b: f32 = b.dense_features.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for (x, y) in a.dense_features.iter().zip(b.dense_features.iter()) {
        let diff_a = x - mean_a;
        let diff_b = y - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator < f32::EPSILON {
        return 0.0;
    }

    (numerator / denominator).clamp(-1.0, 1.0)
}

/// Convert Euclidean distance to similarity score (0-1)
#[must_use]
pub fn distance_to_similarity(distance: f32) -> f32 {
    1.0 / (1.0 + distance)
}

/// Calculate combined similarity (dense + sparse)
#[must_use]
pub fn combined_similarity(
    a: &ContentVector,
    b: &ContentVector,
    dense_weight: f32,
    sparse_weight: f32,
) -> f32 {
    let dense_sim = cosine_similarity(a, b);
    let sparse_sim = a.jaccard_similarity(b);

    let total_weight = dense_weight + sparse_weight;
    if total_weight < f32::EPSILON {
        return 0.0;
    }

    (dense_sim * dense_weight + sparse_sim * sparse_weight) / total_weight
}

/// Distance matrix calculation
pub struct DistanceMatrix {
    /// Pairwise distances
    distances: Vec<Vec<f32>>,
}

impl DistanceMatrix {
    /// Compute distance matrix for a set of vectors
    #[must_use]
    pub fn compute(vectors: &[ContentVector]) -> Self {
        let n = vectors.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&vectors[i], &vectors[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        Self { distances }
    }

    /// Get distance between two items
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i < self.distances.len() && j < self.distances[i].len() {
            self.distances[i][j]
        } else {
            f32::INFINITY
        }
    }

    /// Find k nearest neighbors
    #[must_use]
    pub fn k_nearest(&self, index: usize, k: usize) -> Vec<(usize, f32)> {
        if index >= self.distances.len() {
            return Vec::new();
        }

        let mut neighbors: Vec<(usize, f32)> = self.distances[index]
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(i, &dist)| (i, dist))
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let vec = ContentVector::new(vec![1.0, 2.0, 3.0]);
        let similarity = cosine_similarity(&vec, &vec);
        assert!((similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let vec_a = ContentVector::new(vec![1.0, 0.0]);
        let vec_b = ContentVector::new(vec![0.0, 1.0]);
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert!(similarity.abs() < f32::EPSILON);
    }

    #[test]
    fn test_euclidean_distance() {
        let vec_a = ContentVector::new(vec![0.0, 0.0]);
        let vec_b = ContentVector::new(vec![3.0, 4.0]);
        let distance = euclidean_distance(&vec_a, &vec_b);
        assert!((distance - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_manhattan_distance() {
        let vec_a = ContentVector::new(vec![0.0, 0.0]);
        let vec_b = ContentVector::new(vec![3.0, 4.0]);
        let distance = manhattan_distance(&vec_a, &vec_b);
        assert!((distance - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distance_to_similarity() {
        let sim_zero = distance_to_similarity(0.0);
        assert!((sim_zero - 1.0).abs() < f32::EPSILON);

        let sim_inf = distance_to_similarity(f32::INFINITY);
        assert!(sim_inf < f32::EPSILON);
    }

    #[test]
    fn test_combined_similarity() {
        let vec_a = ContentVector::new(vec![1.0, 2.0]);
        let vec_b = ContentVector::new(vec![1.0, 2.0]);
        let similarity = combined_similarity(&vec_a, &vec_b, 0.5, 0.5);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_distance_matrix() {
        let vectors = vec![
            ContentVector::new(vec![1.0, 0.0]),
            ContentVector::new(vec![0.0, 1.0]),
            ContentVector::new(vec![1.0, 1.0]),
        ];
        let matrix = DistanceMatrix::compute(&vectors);
        let dist = matrix.get(0, 1);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_k_nearest() {
        let vectors = vec![
            ContentVector::new(vec![0.0, 0.0]),
            ContentVector::new(vec![1.0, 0.0]),
            ContentVector::new(vec![100.0, 100.0]),
        ];
        let matrix = DistanceMatrix::compute(&vectors);
        let nearest = matrix.k_nearest(0, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 1); // Closest should be index 1
    }
}
