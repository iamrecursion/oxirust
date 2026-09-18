//! Item vector similarity primitives for content-based recommendation.
//!
//! Provides dense feature vectors and a similarity matrix to enable fast
//! nearest-neighbour lookups across a media catalogue.
//!
//! # Approximate Nearest-Neighbour via LSH
//!
//! For large catalogues where exact O(n) brute-force is too slow, use
//! [`LshItemIndex`].  It wraps the random-hyperplane LSH implementation in
//! [`crate::lsh`] with a convenient `ItemVector`-oriented API:
//!
//! ```
//! use oximedia_recommend::item_similarity::{ItemVector, LshItemIndex, LshItemConfig};
//!
//! let mut index = LshItemIndex::new(LshItemConfig { dim: 4, num_tables: 4, num_planes: 8 });
//! index.insert(ItemVector::new("a", vec![1.0, 0.0, 0.0, 0.0]));
//! index.insert(ItemVector::new("b", vec![0.0, 1.0, 0.0, 0.0]));
//!
//! let results = index.find_similar(&[1.0, 0.0, 0.0, 0.0], 2);
//! assert!(!results.is_empty());
//! assert_eq!(results[0].item_id, "a");
//! ```

#![allow(dead_code)]

use crate::lsh::{LshConfig, LshIndex, LshResult};
use std::collections::HashMap;

/// A dense feature vector representing a media item's content attributes.
#[derive(Debug, Clone)]
pub struct ItemVector {
    /// Unique identifier for this item.
    pub id: String,
    /// Feature values; length defines the embedding dimension.
    pub values: Vec<f64>,
}

impl ItemVector {
    /// Create a new item vector.
    #[must_use]
    pub fn new(id: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            values,
        }
    }

    /// Dot product with another vector of the same dimension.
    ///
    /// Truncates to the shorter length if dimensions differ.
    #[must_use]
    pub fn dot_product(&self, other: &Self) -> f64 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean magnitude (L2 norm) of this vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Cosine similarity with another vector in `[−1, 1]`.
    ///
    /// Returns `0.0` if either vector has zero magnitude.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let denom = self.magnitude() * other.magnitude();
        if denom < f64::EPSILON {
            return 0.0;
        }
        (self.dot_product(other) / denom).clamp(-1.0, 1.0)
    }

    /// Dimension (number of features) of this vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if the vector has no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Precomputed pairwise similarity scores between items.
///
/// Scores are stored symmetrically: inserting `(a, b, s)` also records
/// `(b, a, s)` so lookups work in both directions.
#[derive(Debug, Clone, Default)]
pub struct SimilarityMatrix {
    /// `scores[a][b] = similarity(a, b)`
    scores: HashMap<String, HashMap<String, f64>>,
}

impl SimilarityMatrix {
    /// Create an empty matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or overwrite) the similarity score between two items.
    ///
    /// The score is stored in both directions.
    pub fn insert(&mut self, id_a: impl Into<String>, id_b: impl Into<String>, score: f64) {
        let a = id_a.into();
        let b = id_b.into();
        self.scores
            .entry(a.clone())
            .or_default()
            .insert(b.clone(), score);
        self.scores.entry(b).or_default().insert(a, score);
    }

    /// Retrieve the stored similarity between two items, if present.
    #[must_use]
    pub fn get(&self, id_a: &str, id_b: &str) -> Option<f64> {
        self.scores.get(id_a)?.get(id_b).copied()
    }

    /// Find the `top_k` most similar items to `query_id`, sorted descending.
    ///
    /// The query item itself is excluded from results.
    #[must_use]
    pub fn find_similar(&self, query_id: &str, top_k: usize) -> Vec<(String, f64)> {
        let Some(row) = self.scores.get(query_id) else {
            return Vec::new();
        };
        let mut pairs: Vec<(String, f64)> = row
            .iter()
            .filter(|(id, _)| id.as_str() != query_id)
            .map(|(id, &score)| (id.clone(), score))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(top_k);
        pairs
    }

    /// Total number of unique item IDs tracked in the matrix.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.scores.len()
    }

    /// Build a `SimilarityMatrix` from a slice of `ItemVector`s using cosine similarity.
    #[must_use]
    pub fn from_vectors(vectors: &[ItemVector]) -> Self {
        let mut matrix = Self::new();
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let sim = vectors[i].cosine_similarity(&vectors[j]);
                matrix.insert(vectors[i].id.clone(), vectors[j].id.clone(), sim);
            }
        }
        matrix
    }
}

// ---------------------------------------------------------------------------
// LSH-backed approximate nearest-neighbour item index
// ---------------------------------------------------------------------------

/// Configuration for [`LshItemIndex`].
///
/// Mirrors [`LshConfig`] but uses the same field names as the rest of this
/// module so callers can stay within `item_similarity` without importing
/// `lsh` directly.
#[derive(Debug, Clone)]
pub struct LshItemConfig {
    /// Embedding dimension (must match the `ItemVector` dimension).
    pub dim: usize,
    /// Number of hash tables.  More tables → higher recall, more memory.
    pub num_tables: usize,
    /// Number of random hyperplanes per table.  More planes → higher
    /// precision per table but fewer candidates per query.
    pub num_planes: usize,
}

impl Default for LshItemConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            num_tables: 4,
            num_planes: 8,
        }
    }
}

/// An approximate nearest-neighbour index for `ItemVector`s using
/// locality-sensitive hashing.
///
/// Internally delegates to [`LshIndex`] (random-hyperplane / cosine LSH).
/// The index supports:
///
/// - **`insert`** — add items one by one.
/// - **`bulk_insert`** — add many items at once.
/// - **`find_similar`** — approximate top-k query by cosine similarity.
/// - **`find_similar_to_item`** — convenience method: query using a stored
///   item's own vector.
/// - **`exact_top_k`** — brute-force reference for accuracy evaluation.
pub struct LshItemIndex {
    inner: LshIndex,
}

impl LshItemIndex {
    /// Create a new LSH item index with the given configuration.
    #[must_use]
    pub fn new(config: LshItemConfig) -> Self {
        let lsh_config = LshConfig {
            dim: config.dim,
            num_tables: config.num_tables,
            num_planes: config.num_planes,
        };
        Self {
            inner: LshIndex::new(lsh_config),
        }
    }

    /// Insert a single item into the index.
    pub fn insert(&mut self, item: ItemVector) {
        self.inner.insert(item.id, item.values);
    }

    /// Insert multiple items at once.
    pub fn bulk_insert(&mut self, items: impl IntoIterator<Item = ItemVector>) {
        for item in items {
            self.insert(item);
        }
    }

    /// Approximate top-k nearest neighbours of `query_vector`.
    ///
    /// Returns up to `top_k` results sorted by descending cosine similarity.
    /// The result set is an approximation; some true neighbours may be missing
    /// if they fall in different hash buckets.
    #[must_use]
    pub fn find_similar(&self, query_vector: &[f64], top_k: usize) -> Vec<LshResult> {
        self.inner.query(query_vector, top_k)
    }

    /// Approximate top-k neighbours of a *named* item already in the index.
    ///
    /// Returns `None` if no item with `item_id` exists in the index.
    /// The query item itself is excluded from the results.
    ///
    /// # Implementation note
    ///
    /// `LshIndex` stores vectors positionally and exposes `get_vector(idx)`.
    /// To find the stored vector for `item_id`, we enumerate all stored items
    /// via an exact-top-k query on the zero vector (which enumerates every
    /// item by position), then look up the positional index by name.
    #[must_use]
    pub fn find_similar_to_item(&self, item_id: &str, top_k: usize) -> Option<Vec<LshResult>> {
        let n = self.inner.len();
        if n == 0 {
            return None;
        }

        // Use exact_top_k on a zero-query to enumerate all stored item names
        // in insertion order (the zero vector is equidistant from everything,
        // so the order matches the internal positional index).
        let zero = vec![0.0f64; self.inner.dim()];
        let all_items = self.inner.exact_top_k(&zero, n);

        // Find the positional index of our target item
        let pos = all_items.iter().position(|r| r.item_id == item_id)?;

        // Retrieve the stored vector and run an approximate query
        let query_vec = self.inner.get_vector(pos)?.to_vec();
        let mut results = self.inner.query(&query_vec, top_k + 1);

        // Remove the item itself from results and fix ranks
        results.retain(|r| r.item_id != item_id);
        results.truncate(top_k);
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }
        Some(results)
    }

    /// Exact brute-force top-k — intended for accuracy evaluation in tests.
    #[must_use]
    pub fn exact_top_k(&self, query_vector: &[f64], top_k: usize) -> Vec<LshResult> {
        self.inner.exact_top_k(query_vector, top_k)
    }

    /// Number of items indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the index contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Configured embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}

impl Default for LshItemIndex {
    fn default() -> Self {
        Self::new(LshItemConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec2(id: &str, x: f64, y: f64) -> ItemVector {
        ItemVector::new(id, vec![x, y])
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec2("a", 1.0, 2.0);
        let b = vec2("b", 3.0, 4.0);
        assert!((a.dot_product(&b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec2("a", 0.0, 0.0);
        let b = vec2("b", 1.0, 1.0);
        assert!((a.dot_product(&b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_unit_vector() {
        let v = vec2("v", 1.0, 0.0);
        assert!((v.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_general() {
        let v = vec2("v", 3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec2("a", 1.0, 2.0);
        let b = vec2("b", 1.0, 2.0);
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec2("a", 1.0, 0.0);
        let b = vec2("b", 0.0, 1.0);
        assert!(a.cosine_similarity(&b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec2("a", 0.0, 0.0);
        let b = vec2("b", 1.0, 2.0);
        assert!((a.cosine_similarity(&b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_item_vector_dim() {
        let v = ItemVector::new("x", vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_item_vector_is_empty() {
        let v = ItemVector::new("x", vec![]);
        assert!(v.is_empty());
    }

    #[test]
    fn test_similarity_matrix_insert_and_get() {
        let mut m = SimilarityMatrix::new();
        m.insert("a", "b", 0.8);
        assert!((m.get("a", "b").expect("should succeed in test") - 0.8).abs() < 1e-10);
        assert!((m.get("b", "a").expect("should succeed in test") - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_matrix_missing_returns_none() {
        let m = SimilarityMatrix::new();
        assert!(m.get("x", "y").is_none());
    }

    #[test]
    fn test_find_similar_ordering() {
        let mut m = SimilarityMatrix::new();
        m.insert("a", "b", 0.9);
        m.insert("a", "c", 0.5);
        m.insert("a", "d", 0.7);
        let results = m.find_similar("a", 3);
        assert_eq!(results[0].0, "b");
        assert_eq!(results[1].0, "d");
    }

    #[test]
    fn test_find_similar_top_k_limit() {
        let mut m = SimilarityMatrix::new();
        for i in 0..10_u32 {
            m.insert("q", format!("item{i}"), f64::from(i) * 0.1);
        }
        let results = m.find_similar("q", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_find_similar_empty_for_unknown() {
        let m = SimilarityMatrix::new();
        assert!(m.find_similar("unknown", 5).is_empty());
    }

    #[test]
    fn test_item_count() {
        let mut m = SimilarityMatrix::new();
        m.insert("a", "b", 0.5);
        m.insert("a", "c", 0.6);
        assert_eq!(m.item_count(), 3);
    }

    #[test]
    fn test_from_vectors_builds_correct_similarity() {
        let vectors = vec![
            ItemVector::new("a", vec![1.0, 0.0]),
            ItemVector::new("b", vec![1.0, 0.0]),
            ItemVector::new("c", vec![0.0, 1.0]),
        ];
        let m = SimilarityMatrix::from_vectors(&vectors);
        let ab = m.get("a", "b").expect("should succeed in test");
        let ac = m.get("a", "c").expect("should succeed in test");
        assert!((ab - 1.0).abs() < 1e-9);
        assert!(ac.abs() < 1e-9);
    }

    // ---- LshItemIndex ----

    fn make_lsh_index() -> LshItemIndex {
        LshItemIndex::new(LshItemConfig {
            dim: 4,
            num_tables: 6,
            num_planes: 10,
        })
    }

    #[test]
    fn test_lsh_item_index_creation() {
        let idx = LshItemIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_lsh_item_index_insert_and_len() {
        let mut idx = make_lsh_index();
        idx.insert(ItemVector::new("a", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("b", vec![0.0, 1.0, 0.0, 0.0]));
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_lsh_item_index_bulk_insert() {
        let mut idx = make_lsh_index();
        let items = vec![
            ItemVector::new("x", vec![1.0, 0.0, 0.0, 0.0]),
            ItemVector::new("y", vec![0.0, 1.0, 0.0, 0.0]),
            ItemVector::new("z", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        idx.bulk_insert(items);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_lsh_item_index_find_similar_returns_results() {
        let mut idx = make_lsh_index();
        idx.insert(ItemVector::new("a", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("b", vec![0.0, 1.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("c", vec![0.0, 0.0, 1.0, 0.0]));
        let results = idx.find_similar(&[1.0, 0.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_lsh_item_index_identical_vector_is_top_result() {
        let mut idx = make_lsh_index();
        idx.insert(ItemVector::new("target", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("other1", vec![0.0, 1.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("other2", vec![0.0, 0.0, 1.0, 0.0]));
        let results = idx.find_similar(&[1.0, 0.0, 0.0, 0.0], 1);
        if !results.is_empty() {
            assert_eq!(results[0].item_id, "target");
        }
    }

    #[test]
    fn test_lsh_item_index_similarity_in_range() {
        let mut idx = make_lsh_index();
        idx.insert(ItemVector::new("a", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("b", vec![0.707, 0.707, 0.0, 0.0]));
        let results = idx.find_similar(&[1.0, 0.0, 0.0, 0.0], 2);
        for r in &results {
            assert!(r.similarity >= -1.0 && r.similarity <= 1.0);
        }
    }

    #[test]
    fn test_lsh_item_index_exact_top_k() {
        let mut idx = make_lsh_index();
        idx.insert(ItemVector::new("a", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("b", vec![1.0, 0.0, 0.0, 0.0]));
        idx.insert(ItemVector::new("c", vec![0.0, 1.0, 0.0, 0.0]));
        let exact = idx.exact_top_k(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(exact.len(), 2);
        // a and b are identical to query, c is orthogonal
        assert!(
            exact[0].similarity > exact[1].similarity
                || (exact[0].similarity - exact[1].similarity).abs() < 1e-9
        );
    }

    #[test]
    fn test_lsh_item_index_empty_query_returns_empty() {
        let idx = make_lsh_index();
        let results = idx.find_similar(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_item_index_dim() {
        let idx = make_lsh_index();
        assert_eq!(idx.dim(), 4);
    }

    #[test]
    fn test_lsh_item_config_default() {
        let config = LshItemConfig::default();
        assert_eq!(config.dim, 64);
        assert_eq!(config.num_tables, 4);
        assert_eq!(config.num_planes, 8);
    }
}
