//! Generic feature vector utilities for Music Information Retrieval.
//!
//! Provides a [`FeatureVector`] type that wraps a named, fixed-length `Vec<f32>`
//! together with common vector operations (normalisation, distance metrics,
//! statistics) used across many MIR tasks.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// DistanceMetric
// ---------------------------------------------------------------------------

/// Supported distance / similarity metrics between feature vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
}

// ---------------------------------------------------------------------------
// FeatureVector
// ---------------------------------------------------------------------------

/// A named, fixed-dimension feature vector.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Descriptive name (e.g. "`mfcc_mean`", "chroma").
    pub name: String,
    /// Raw feature values.
    pub values: Vec<f32>,
}

impl FeatureVector {
    /// Create a new named feature vector.
    #[must_use]
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    /// Create a zero-filled vector of the given dimension.
    #[must_use]
    pub fn zeros(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            values: vec![0.0; dim],
        }
    }

    /// Dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// L2 (Euclidean) norm.
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// L1 (Manhattan) norm.
    #[must_use]
    pub fn l1_norm(&self) -> f32 {
        self.values.iter().map(|v| v.abs()).sum()
    }

    /// Return a unit-length (L2-normalised) copy.
    #[must_use]
    pub fn normalize_l2(&self) -> Self {
        let norm = self.l2_norm();
        if norm < 1e-12 {
            return self.clone();
        }
        Self {
            name: self.name.clone(),
            values: self.values.iter().map(|v| v / norm).collect(),
        }
    }

    /// Return a min-max normalised copy mapping values to `[0, 1]`.
    #[must_use]
    pub fn normalize_min_max(&self) -> Self {
        let min = self.values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range < 1e-12 {
            return self.clone();
        }
        Self {
            name: self.name.clone(),
            values: self.values.iter().map(|v| (v - min) / range).collect(),
        }
    }

    /// Mean of all values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }

    /// Variance of all values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> f32 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let n = self.values.len() as f32;
        self.values.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / n
    }

    /// Standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Compute the distance to another feature vector using the given metric.
    /// Returns `None` if the dimensions differ.
    #[must_use]
    pub fn distance(&self, other: &Self, metric: DistanceMetric) -> Option<f32> {
        if self.dim() != other.dim() {
            return None;
        }
        Some(match metric {
            DistanceMetric::Euclidean => self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Manhattan => self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| (a - b).abs())
                .sum(),
            DistanceMetric::Cosine => {
                let dot: f32 = self
                    .values
                    .iter()
                    .zip(other.values.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let na = self.l2_norm();
                let nb = other.l2_norm();
                if na < 1e-12 || nb < 1e-12 {
                    1.0
                } else {
                    1.0 - dot / (na * nb)
                }
            }
        })
    }

    /// Dot product with another vector. Returns `None` if dimensions differ.
    #[must_use]
    pub fn dot(&self, other: &Self) -> Option<f32> {
        if self.dim() != other.dim() {
            return None;
        }
        Some(
            self.values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a * b)
                .sum(),
        )
    }

    /// Element-wise addition. Returns `None` if dimensions differ.
    #[must_use]
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.dim() != other.dim() {
            return None;
        }
        Some(Self {
            name: self.name.clone(),
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        })
    }

    /// Scale all values by a constant.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            name: self.name.clone(),
            values: self.values.iter().map(|v| v * factor).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// FeatureMatrix
// ---------------------------------------------------------------------------

/// A collection of feature vectors forming a time-series (one per frame).
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    /// Rows (one per time frame).
    pub rows: Vec<FeatureVector>,
}

impl FeatureMatrix {
    /// Create a new empty matrix.
    #[must_use]
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Push a feature vector as the next row.
    pub fn push(&mut self, row: FeatureVector) {
        self.rows.push(row);
    }

    /// Number of rows (frames).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Whether the matrix is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Compute the column-wise (per-dimension) mean across all rows.
    /// Returns `None` if the matrix is empty or rows have inconsistent dimensions.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn column_mean(&self) -> Option<FeatureVector> {
        if self.rows.is_empty() {
            return None;
        }
        let dim = self.rows[0].dim();
        if self.rows.iter().any(|r| r.dim() != dim) {
            return None;
        }
        let n = self.rows.len() as f32;
        let mut mean = vec![0.0f32; dim];
        for row in &self.rows {
            for (i, &v) in row.values.iter().enumerate() {
                mean[i] += v;
            }
        }
        for v in &mut mean {
            *v /= n;
        }
        Some(FeatureVector::new("column_mean", mean))
    }
}

impl Default for FeatureMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fv_new() {
        let fv = FeatureVector::new("test", vec![1.0, 2.0, 3.0]);
        assert_eq!(fv.dim(), 3);
        assert_eq!(fv.name, "test");
    }

    #[test]
    fn test_fv_zeros() {
        let fv = FeatureVector::zeros("z", 5);
        assert_eq!(fv.dim(), 5);
        assert!(fv.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_l2_norm() {
        let fv = FeatureVector::new("n", vec![3.0, 4.0]);
        assert!((fv.l2_norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_l1_norm() {
        let fv = FeatureVector::new("n", vec![-3.0, 4.0]);
        assert!((fv.l1_norm() - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2() {
        let fv = FeatureVector::new("n", vec![3.0, 4.0]);
        let n = fv.normalize_l2();
        assert!((n.l2_norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_min_max() {
        let fv = FeatureVector::new("n", vec![2.0, 4.0, 6.0]);
        let n = fv.normalize_min_max();
        assert!((n.values[0] - 0.0).abs() < 1e-5);
        assert!((n.values[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_variance() {
        let fv = FeatureVector::new("x", vec![2.0, 4.0]);
        assert!((fv.mean() - 3.0).abs() < 1e-5);
        assert!((fv.variance() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_std_dev() {
        let fv = FeatureVector::new("x", vec![2.0, 4.0]);
        assert!((fv.std_dev() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = FeatureVector::new("a", vec![0.0, 0.0]);
        let b = FeatureVector::new("b", vec![3.0, 4.0]);
        let d = a
            .distance(&b, DistanceMetric::Euclidean)
            .expect("should succeed in test");
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = FeatureVector::new("a", vec![0.0, 0.0]);
        let b = FeatureVector::new("b", vec![3.0, 4.0]);
        let d = a
            .distance(&b, DistanceMetric::Manhattan)
            .expect("should succeed in test");
        assert!((d - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = FeatureVector::new("a", vec![1.0, 2.0]);
        let d = a
            .distance(&a, DistanceMetric::Cosine)
            .expect("should succeed in test");
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_distance_dim_mismatch() {
        let a = FeatureVector::new("a", vec![1.0]);
        let b = FeatureVector::new("b", vec![1.0, 2.0]);
        assert!(a.distance(&b, DistanceMetric::Euclidean).is_none());
    }

    #[test]
    fn test_dot() {
        let a = FeatureVector::new("a", vec![1.0, 2.0, 3.0]);
        let b = FeatureVector::new("b", vec![4.0, 5.0, 6.0]);
        assert!((a.dot(&b).expect("should succeed in test") - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_add() {
        let a = FeatureVector::new("a", vec![1.0, 2.0]);
        let b = FeatureVector::new("b", vec![3.0, 4.0]);
        let c = a.add(&b).expect("should succeed in test");
        assert!((c.values[0] - 4.0).abs() < 1e-5);
        assert!((c.values[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let fv = FeatureVector::new("s", vec![1.0, 2.0]);
        let s = fv.scale(3.0);
        assert!((s.values[0] - 3.0).abs() < 1e-5);
        assert!((s.values[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_column_mean() {
        let mut m = FeatureMatrix::new();
        m.push(FeatureVector::new("r1", vec![2.0, 4.0]));
        m.push(FeatureVector::new("r2", vec![4.0, 8.0]));
        let mean = m.column_mean().expect("should succeed in test");
        assert!((mean.values[0] - 3.0).abs() < 1e-5);
        assert!((mean.values[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_empty_column_mean() {
        let m = FeatureMatrix::new();
        assert!(m.column_mean().is_none());
    }

    #[test]
    fn test_matrix_n_rows() {
        let mut m = FeatureMatrix::new();
        assert_eq!(m.n_rows(), 0);
        assert!(m.is_empty());
        m.push(FeatureVector::zeros("r", 3));
        assert_eq!(m.n_rows(), 1);
    }
}
