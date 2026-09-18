//! Locality-Sensitive Hashing (LSH) for approximate nearest-neighbour search.
//!
//! Uses the random-hyperplane (sign-LSH / SimHash) family for angular
//! (cosine) similarity.  For each hash table, `num_planes` random hyperplanes
//! are drawn from a deterministic seeded sequence (Halton-like construction),
//! each hyperplane partitions the space into two half-spaces, and the
//! resulting bit pattern is used as a hash bucket key.
//!
//! # Design
//!
//! - **`LshIndex`** – the main index.  Items are added via `insert`, then
//!   `query` retrieves the approximate top-k neighbours.
//! - Multiple hash tables (`num_tables`) reduce the probability of missing
//!   true neighbours at the cost of more memory.
//! - The index performs an exact cosine-similarity re-rank among all
//!   candidates retrieved from the bucket union to return the best `top_k`.
//!
//! # Limitations
//!
//! - O(d · L · b) query time, where d = dimension, L = tables, b = bucket size.
//! - No support for deletion (rebuild required after heavy updates).

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Hyperplane set (one per hash table)
// ---------------------------------------------------------------------------

/// A set of random hyperplanes used to generate one LSH hash value.
///
/// The hyperplanes are generated deterministically from a seed so that
/// the index is reproducible without an external RNG dependency.
#[derive(Debug, Clone)]
struct HyperplaneSet {
    /// Each inner Vec is one hyperplane normal vector of length `dim`.
    planes: Vec<Vec<f64>>,
}

impl HyperplaneSet {
    /// Generate `num_planes` hyperplanes in `dim` dimensions from `seed`.
    fn new(dim: usize, num_planes: usize, seed: u64) -> Self {
        let mut planes = Vec::with_capacity(num_planes);
        for p in 0..num_planes {
            let mut normal = Vec::with_capacity(dim);
            for d in 0..dim {
                // Deterministic pseudo-normal via a linear-congruential sequence
                // seeded differently for each (plane, dimension, table) triple.
                let s = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add((p as u64).wrapping_mul(2_654_435_761))
                    .wrapping_add(d as u64)
                    .wrapping_mul(1_442_695_040_888_963_407)
                    .wrapping_add(p as u64);
                // Map to [-1, 1] via fractional part
                let frac = (s >> 11) as f64 / (1u64 << 53) as f64;
                normal.push(frac * 2.0 - 1.0);
            }
            planes.push(normal);
        }
        Self { planes }
    }

    /// Compute the hash of `vector` as a bitmask (one bit per plane).
    fn hash(&self, vector: &[f64]) -> u64 {
        let mut bits: u64 = 0;
        for (i, plane) in self.planes.iter().enumerate() {
            let dot: f64 = plane.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                bits |= 1u64 << (i % 64);
            }
        }
        bits
    }
}

// ---------------------------------------------------------------------------
// LSH Index
// ---------------------------------------------------------------------------

/// Configuration for the LSH index.
#[derive(Debug, Clone)]
pub struct LshConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of hash tables.
    pub num_tables: usize,
    /// Number of random hyperplanes per table (= bits per hash key).
    pub num_planes: usize,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            num_tables: 4,
            num_planes: 8,
        }
    }
}

/// An approximate nearest-neighbour index using locality-sensitive hashing.
pub struct LshIndex {
    /// Configuration.
    config: LshConfig,
    /// One hyperplane set per hash table.
    hyperplane_sets: Vec<HyperplaneSet>,
    /// Hash tables: `table_idx` → bucket_hash → list of item indices.
    tables: Vec<HashMap<u64, Vec<usize>>>,
    /// Stored item vectors.
    vectors: Vec<Vec<f64>>,
    /// Stored item identifiers.
    item_ids: Vec<String>,
    /// Total items indexed.
    total_indexed: usize,
}

impl LshIndex {
    /// Create a new LSH index.
    #[must_use]
    pub fn new(config: LshConfig) -> Self {
        let hyperplane_sets: Vec<HyperplaneSet> = (0..config.num_tables)
            .map(|t| HyperplaneSet::new(config.dim, config.num_planes, t as u64 * 12_345_678 + 1))
            .collect();
        let tables = vec![HashMap::new(); config.num_tables];
        Self {
            config,
            hyperplane_sets,
            tables,
            vectors: Vec::new(),
            item_ids: Vec::new(),
            total_indexed: 0,
        }
    }

    /// Insert an item into the index.
    ///
    /// `vector` must have length equal to `config.dim`.  Vectors of different
    /// length will be zero-padded or truncated to `config.dim`.
    pub fn insert(&mut self, item_id: impl Into<String>, vector: Vec<f64>) {
        let id = item_id.into();
        let item_idx = self.vectors.len();

        // Normalise to config.dim
        let mut v = vector;
        v.resize(self.config.dim, 0.0);

        for (t, hpset) in self.hyperplane_sets.iter().enumerate() {
            let bucket = hpset.hash(&v);
            self.tables[t].entry(bucket).or_default().push(item_idx);
        }

        self.vectors.push(v);
        self.item_ids.push(id);
        self.total_indexed += 1;
    }

    /// Query for approximate top-k nearest neighbours of `query_vector`.
    ///
    /// Retrieves all candidates from matching buckets across all tables,
    /// then re-ranks by exact cosine similarity.
    #[must_use]
    pub fn query(&self, query_vector: &[f64], top_k: usize) -> Vec<LshResult> {
        if self.vectors.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Normalise query to config.dim
        let mut qv = query_vector.to_vec();
        qv.resize(self.config.dim, 0.0);

        // Collect candidate indices from all tables
        let mut candidates: HashSet<usize> = HashSet::new();
        for (t, hpset) in self.hyperplane_sets.iter().enumerate() {
            let bucket = hpset.hash(&qv);
            if let Some(idxs) = self.tables[t].get(&bucket) {
                candidates.extend(idxs.iter().copied());
            }
        }

        // Exact cosine re-rank
        let q_norm = l2_norm(&qv);
        let mut scored: Vec<(usize, f64)> = candidates
            .into_iter()
            .map(|idx| {
                let sim = cosine_sim(&qv, &self.vectors[idx], q_norm);
                (idx, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, sim))| LshResult {
                item_id: self.item_ids[idx].clone(),
                similarity: sim,
                rank: rank + 1,
            })
            .collect()
    }

    /// Number of items in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_indexed
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_indexed == 0
    }

    /// Number of hash tables.
    #[must_use]
    pub fn num_tables(&self) -> usize {
        self.config.num_tables
    }

    /// Dimension of vectors in the index.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Retrieve the stored vector for an item by its internal index.
    #[must_use]
    pub fn get_vector(&self, item_idx: usize) -> Option<&[f64]> {
        self.vectors.get(item_idx).map(Vec::as_slice)
    }

    /// Compute the exact brute-force top-k neighbours for comparison.
    ///
    /// This is only intended for accuracy evaluation in tests.
    #[must_use]
    pub fn exact_top_k(&self, query_vector: &[f64], top_k: usize) -> Vec<LshResult> {
        let mut qv = query_vector.to_vec();
        qv.resize(self.config.dim, 0.0);
        let q_norm = l2_norm(&qv);

        let mut scored: Vec<(usize, f64)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, cosine_sim(&qv, v, q_norm)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, sim))| LshResult {
                item_id: self.item_ids[idx].clone(),
                similarity: sim,
                rank: rank + 1,
            })
            .collect()
    }
}

impl Default for LshIndex {
    fn default() -> Self {
        Self::new(LshConfig::default())
    }
}

/// Result of an LSH nearest-neighbour query.
#[derive(Debug, Clone)]
pub struct LshResult {
    /// Item identifier.
    pub item_id: String,
    /// Cosine similarity to the query vector.
    pub similarity: f64,
    /// Rank in the result list (1-indexed).
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_sim(a: &[f64], b: &[f64], a_norm: f64) -> f64 {
    let b_norm = l2_norm(b);
    if a_norm < f64::EPSILON || b_norm < f64::EPSILON {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (dot / (a_norm * b_norm)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, hot: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        if hot < dim {
            v[hot] = 1.0;
        }
        v
    }

    fn make_index(num_tables: usize, num_planes: usize, dim: usize) -> LshIndex {
        LshIndex::new(LshConfig {
            dim,
            num_tables,
            num_planes,
        })
    }

    #[test]
    fn test_lsh_index_creation() {
        let idx = LshIndex::default();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_lsh_insert_and_len() {
        let mut idx = make_index(4, 8, 4);
        idx.insert("a", vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert("b", vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_lsh_query_empty_index() {
        let idx = LshIndex::default();
        let results = idx.query(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_query_top_k_limit() {
        let mut idx = make_index(4, 8, 2);
        for i in 0..10_u32 {
            idx.insert(format!("item{i}"), vec![i as f64, 0.0]);
        }
        let results = idx.query(&[1.0, 0.0], 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_lsh_identical_vector_is_top_result() {
        let mut idx = make_index(6, 10, 4);
        idx.insert("target", vec![1.0, 0.0, 0.0, 0.0]);
        idx.insert("noise1", vec![0.0, 1.0, 0.0, 0.0]);
        idx.insert("noise2", vec![0.0, 0.0, 1.0, 0.0]);
        idx.insert("noise3", vec![0.0, 0.0, 0.0, 1.0]);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = idx.query(&query, 1);
        if !results.is_empty() {
            assert_eq!(results[0].item_id, "target");
        }
    }

    #[test]
    fn test_lsh_rank_ordering() {
        let mut idx = make_index(4, 8, 2);
        idx.insert("high", vec![1.0, 0.0]);
        idx.insert("low", vec![0.0, 1.0]);
        let results = idx.query(&[1.0, 0.01], 2);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    #[test]
    fn test_lsh_similarity_in_range() {
        let mut idx = make_index(4, 8, 4);
        for i in 0..5_usize {
            idx.insert(format!("v{i}"), unit_vec(4, i % 4));
        }
        let results = idx.query(&unit_vec(4, 0), 5);
        for r in &results {
            assert!(r.similarity >= -1.0 && r.similarity <= 1.0);
        }
    }

    #[test]
    fn test_lsh_num_tables() {
        let idx = make_index(3, 5, 2);
        assert_eq!(idx.num_tables(), 3);
    }

    #[test]
    fn test_lsh_dim() {
        let idx = make_index(2, 4, 16);
        assert_eq!(idx.dim(), 16);
    }

    #[test]
    fn test_lsh_get_vector() {
        let mut idx = make_index(2, 4, 3);
        idx.insert("x", vec![1.0, 2.0, 3.0]);
        let v = idx.get_vector(0).expect("should exist");
        assert!((v[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lsh_exact_top_k_returns_correct_order() {
        let mut idx = make_index(4, 8, 2);
        idx.insert("a", vec![1.0, 0.0]);
        idx.insert("b", vec![0.707, 0.707]);
        idx.insert("c", vec![0.0, 1.0]);

        // Query: [1, 0] → a is most similar
        let exact = idx.exact_top_k(&[1.0, 0.0], 3);
        assert!(!exact.is_empty());
        assert_eq!(exact[0].item_id, "a");
        assert!((exact[0].similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lsh_short_vector_padded() {
        let mut idx = make_index(2, 4, 8);
        // Insert a 2D vector into an 8D index — should be zero-padded
        idx.insert("short", vec![1.0, 0.0]);
        assert_eq!(idx.len(), 1);
        // Query with another short vector
        let results = idx.query(&[1.0, 0.0], 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_lsh_large_index_has_results() {
        let mut idx = make_index(4, 8, 8);
        for i in 0..100_usize {
            let v: Vec<f64> = (0..8).map(|d| if d == i % 8 { 1.0 } else { 0.0 }).collect();
            idx.insert(format!("item{i}"), v);
        }
        let query: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = idx.query(&query, 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_lsh_zero_top_k() {
        let mut idx = make_index(2, 4, 2);
        idx.insert("a", vec![1.0, 0.0]);
        let results = idx.query(&[1.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_config_default() {
        let config = LshConfig::default();
        assert_eq!(config.dim, 64);
        assert_eq!(config.num_tables, 4);
        assert_eq!(config.num_planes, 8);
    }
}
