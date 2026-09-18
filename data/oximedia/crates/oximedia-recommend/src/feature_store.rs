#![allow(dead_code)]
//! Feature storage and retrieval for recommendation models.
//!
//! Provides an in-memory feature store that maps entities (users, items)
//! to dense feature vectors. Supports feature normalization, dot-product
//! similarity, cosine similarity, and batch lookups for efficient
//! recommendation scoring.

use std::collections::HashMap;

/// Type of entity a feature vector belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// A user entity.
    User,
    /// A content item entity.
    Item,
    /// A category/genre entity.
    Category,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "User"),
            Self::Item => write!(f, "Item"),
            Self::Category => write!(f, "Category"),
        }
    }
}

/// A composite key for looking up feature vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureKey {
    /// Entity type.
    pub entity_type: EntityType,
    /// Entity identifier.
    pub entity_id: String,
}

impl FeatureKey {
    /// Create a new feature key.
    pub fn new(entity_type: EntityType, entity_id: impl Into<String>) -> Self {
        Self {
            entity_type,
            entity_id: entity_id.into(),
        }
    }

    /// Create a user feature key.
    pub fn user(id: impl Into<String>) -> Self {
        Self::new(EntityType::User, id)
    }

    /// Create an item feature key.
    pub fn item(id: impl Into<String>) -> Self {
        Self::new(EntityType::Item, id)
    }

    /// Create a category feature key.
    pub fn category(id: impl Into<String>) -> Self {
        Self::new(EntityType::Category, id)
    }
}

/// A dense feature vector with metadata.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// The feature values.
    pub values: Vec<f64>,
    /// Timestamp of last update (ms since epoch).
    pub updated_at: u64,
    /// Version counter.
    pub version: u32,
}

impl FeatureVector {
    /// Create a new feature vector.
    #[must_use]
    pub fn new(values: Vec<f64>, updated_at: u64) -> Self {
        Self {
            values,
            updated_at,
            version: 1,
        }
    }

    /// Dimensionality of this vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Compute the L2 norm.
    #[must_use]
    pub fn l2_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Normalize the vector to unit length (in-place).
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > f64::EPSILON {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }

    /// Return a normalized copy.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Dot product with another vector.
    ///
    /// Returns 0.0 if dimensions do not match.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity with another vector.
    ///
    /// Returns 0.0 if either vector is zero or dimensions mismatch.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
            return 0.0;
        }
        self.dot(other) / (norm_a * norm_b)
    }

    /// Euclidean distance to another vector.
    ///
    /// Returns `f64::MAX` if dimensions mismatch.
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> f64 {
        if self.values.len() != other.values.len() {
            return f64::MAX;
        }
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// Statistics about the feature store.
#[derive(Debug, Clone, Default)]
pub struct FeatureStoreStats {
    /// Total number of stored feature vectors.
    pub total_vectors: usize,
    /// Number of user vectors.
    pub user_vectors: usize,
    /// Number of item vectors.
    pub item_vectors: usize,
    /// Number of category vectors.
    pub category_vectors: usize,
    /// Total lookups performed.
    pub lookups: u64,
    /// Total lookup hits.
    pub hits: u64,
}

impl FeatureStoreStats {
    /// Hit rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}

/// In-memory feature store.
#[derive(Debug)]
pub struct FeatureStore {
    /// Stored vectors.
    vectors: HashMap<FeatureKey, FeatureVector>,
    /// Expected dimensionality (0 = unconstrained).
    expected_dim: usize,
    /// Statistics.
    stats: FeatureStoreStats,
}

impl FeatureStore {
    /// Create a new feature store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            expected_dim: 0,
            stats: FeatureStoreStats::default(),
        }
    }

    /// Create a new feature store with a fixed dimensionality constraint.
    #[must_use]
    pub fn with_dim(dim: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            expected_dim: dim,
            stats: FeatureStoreStats::default(),
        }
    }

    /// Insert or update a feature vector.
    ///
    /// Returns false if the vector's dimensionality does not match the expected dim.
    pub fn put(&mut self, key: FeatureKey, vector: FeatureVector) -> bool {
        if self.expected_dim > 0 && vector.dim() != self.expected_dim {
            return false;
        }

        // Update stats count
        match key.entity_type {
            EntityType::User => {
                if !self.vectors.contains_key(&key) {
                    self.stats.user_vectors += 1;
                }
            }
            EntityType::Item => {
                if !self.vectors.contains_key(&key) {
                    self.stats.item_vectors += 1;
                }
            }
            EntityType::Category => {
                if !self.vectors.contains_key(&key) {
                    self.stats.category_vectors += 1;
                }
            }
        }

        self.vectors.insert(key, vector);
        self.stats.total_vectors = self.vectors.len();
        true
    }

    /// Look up a feature vector.
    pub fn get(&mut self, key: &FeatureKey) -> Option<&FeatureVector> {
        self.stats.lookups += 1;
        let result = self.vectors.get(key);
        if result.is_some() {
            self.stats.hits += 1;
        }
        result
    }

    /// Look up without updating stats (for internal use).
    #[must_use]
    pub fn peek(&self, key: &FeatureKey) -> Option<&FeatureVector> {
        self.vectors.get(key)
    }

    /// Remove a feature vector.
    pub fn remove(&mut self, key: &FeatureKey) -> bool {
        if self.vectors.remove(key).is_some() {
            self.stats.total_vectors = self.vectors.len();
            true
        } else {
            false
        }
    }

    /// Batch lookup of multiple keys.
    pub fn get_batch(&mut self, keys: &[FeatureKey]) -> Vec<Option<&FeatureVector>> {
        keys.iter()
            .map(|k| {
                self.stats.lookups += 1;
                let result = self.vectors.get(k);
                if result.is_some() {
                    self.stats.hits += 1;
                }
                result
            })
            .collect()
    }

    /// Find the k nearest neighbors to a query vector among entities of a given type.
    #[must_use]
    pub fn nearest_neighbors(
        &self,
        query: &FeatureVector,
        entity_type: EntityType,
        k: usize,
    ) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = self
            .vectors
            .iter()
            .filter(|(key, _)| key.entity_type == entity_type)
            .map(|(key, vec)| (key.entity_id.clone(), query.cosine_similarity(vec)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get statistics.
    #[must_use]
    pub fn stats(&self) -> &FeatureStoreStats {
        &self.stats
    }

    /// Clear all stored vectors.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.stats.total_vectors = 0;
        self.stats.user_vectors = 0;
        self.stats.item_vectors = 0;
        self.stats.category_vectors = 0;
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::User.to_string(), "User");
        assert_eq!(EntityType::Item.to_string(), "Item");
        assert_eq!(EntityType::Category.to_string(), "Category");
    }

    #[test]
    fn test_feature_key_constructors() {
        let uk = FeatureKey::user("u1");
        assert_eq!(uk.entity_type, EntityType::User);
        assert_eq!(uk.entity_id, "u1");

        let ik = FeatureKey::item("i1");
        assert_eq!(ik.entity_type, EntityType::Item);

        let ck = FeatureKey::category("c1");
        assert_eq!(ck.entity_type, EntityType::Category);
    }

    #[test]
    fn test_feature_vector_l2_norm() {
        let v = FeatureVector::new(vec![3.0, 4.0], 0);
        assert!((v.l2_norm() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_vector_normalize() {
        let mut v = FeatureVector::new(vec![3.0, 4.0], 0);
        v.normalize();
        assert!((v.l2_norm() - 1.0).abs() < 1e-10);
        assert!((v.values[0] - 0.6).abs() < 1e-10);
        assert!((v.values[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_dot() {
        let a = FeatureVector::new(vec![1.0, 2.0, 3.0], 0);
        let b = FeatureVector::new(vec![4.0, 5.0, 6.0], 0);
        assert!((a.dot(&b) - 32.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_vector_dot_dimension_mismatch() {
        let a = FeatureVector::new(vec![1.0, 2.0], 0);
        let b = FeatureVector::new(vec![1.0, 2.0, 3.0], 0);
        assert!((a.dot(&b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = FeatureVector::new(vec![1.0, 2.0, 3.0], 0);
        let b = FeatureVector::new(vec![1.0, 2.0, 3.0], 0);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = FeatureVector::new(vec![1.0, 0.0], 0);
        let b = FeatureVector::new(vec![0.0, 1.0], 0);
        assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = FeatureVector::new(vec![0.0, 0.0], 0);
        let b = FeatureVector::new(vec![3.0, 4.0], 0);
        assert!((a.euclidean_distance(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_store_put_get() {
        let mut store = FeatureStore::new();
        let key = FeatureKey::user("u1");
        let vec = FeatureVector::new(vec![0.1, 0.2, 0.3], 1000);
        assert!(store.put(key.clone(), vec));
        let retrieved = store.get(&key).expect("should succeed in test");
        assert_eq!(retrieved.dim(), 3);
    }

    #[test]
    fn test_feature_store_dim_constraint() {
        let mut store = FeatureStore::with_dim(3);
        let key = FeatureKey::item("i1");
        let good = FeatureVector::new(vec![1.0, 2.0, 3.0], 0);
        let bad = FeatureVector::new(vec![1.0, 2.0], 0);
        assert!(store.put(key.clone(), good));
        assert!(!store.put(key, bad));
    }

    #[test]
    fn test_feature_store_remove() {
        let mut store = FeatureStore::new();
        let key = FeatureKey::item("i1");
        store.put(key.clone(), FeatureVector::new(vec![1.0], 0));
        assert!(store.remove(&key));
        assert!(store.is_empty());
    }

    #[test]
    fn test_nearest_neighbors() {
        let mut store = FeatureStore::new();
        store.put(
            FeatureKey::item("i1"),
            FeatureVector::new(vec![1.0, 0.0], 0),
        );
        store.put(
            FeatureKey::item("i2"),
            FeatureVector::new(vec![0.9, 0.1], 0),
        );
        store.put(
            FeatureKey::item("i3"),
            FeatureVector::new(vec![0.0, 1.0], 0),
        );

        let query = FeatureVector::new(vec![1.0, 0.0], 0);
        let neighbors = store.nearest_neighbors(&query, EntityType::Item, 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, "i1");
        assert_eq!(neighbors[1].0, "i2");
    }

    #[test]
    fn test_feature_store_stats() {
        let mut store = FeatureStore::new();
        store.put(FeatureKey::user("u1"), FeatureVector::new(vec![1.0], 0));
        store.put(FeatureKey::item("i1"), FeatureVector::new(vec![1.0], 0));
        let _ = store.get(&FeatureKey::user("u1")); // hit
        let _ = store.get(&FeatureKey::user("u999")); // miss
        assert_eq!(store.stats().total_vectors, 2);
        assert_eq!(store.stats().user_vectors, 1);
        assert_eq!(store.stats().item_vectors, 1);
        assert_eq!(store.stats().lookups, 2);
        assert_eq!(store.stats().hits, 1);
        assert!((store.stats().hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_store_clear() {
        let mut store = FeatureStore::new();
        store.put(FeatureKey::user("u1"), FeatureVector::new(vec![1.0], 0));
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.stats().total_vectors, 0);
    }
}
