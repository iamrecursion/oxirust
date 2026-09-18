#![allow(dead_code)]
//! Vector embedding similarity search for media assets.
//!
//! Provides approximate nearest-neighbor (ANN) search over dense vector
//! embeddings, enabling semantic and visual similarity queries using
//! learned representations (e.g. CLIP-like patent-free models).
//!
//! # Architecture
//!
//! The [`EmbeddingIndex`] stores normalized embedding vectors alongside
//! asset identifiers. Similarity is computed using cosine similarity,
//! with optional quantization for memory-efficient storage of large
//! collections.
//!
//! For collections up to ~100K vectors, brute-force search with SIMD-friendly
//! dot product is used. For larger collections, a partitioned index with
//! coarse quantization centroids enables sub-linear lookup.
//!
//! # Patent-free
//!
//! This module uses only cosine similarity and k-means clustering — no
//! patented ANN algorithms (HNSW patents, etc.) are used.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Embedding vector
// ---------------------------------------------------------------------------

/// A dense embedding vector with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The asset this embedding represents.
    pub asset_id: Uuid,
    /// Dense vector (L2-normalized).
    pub vector: Vec<f32>,
    /// Embedding model identifier (e.g. "clip-vit-b32").
    pub model: String,
    /// Optional label or category for the embedding.
    pub label: Option<String>,
}

impl Embedding {
    /// Create a new embedding, automatically L2-normalizing the vector.
    pub fn new(asset_id: Uuid, vector: Vec<f32>, model: impl Into<String>) -> Self {
        let normalized = l2_normalize(&vector);
        Self {
            asset_id,
            vector: normalized,
            model: model.into(),
            label: None,
        }
    }

    /// Create an embedding with a label.
    pub fn with_label(
        asset_id: Uuid,
        vector: Vec<f32>,
        model: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        let normalized = l2_normalize(&vector);
        Self {
            asset_id,
            vector: normalized,
            model: model.into(),
            label: Some(label.into()),
        }
    }

    /// Dimensionality of the embedding.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

// ---------------------------------------------------------------------------
// Similarity result
// ---------------------------------------------------------------------------

/// A single result from an embedding similarity search.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The matched asset ID.
    pub asset_id: Uuid,
    /// Cosine similarity score in `[-1.0, 1.0]` (higher is more similar).
    pub similarity: f32,
    /// Label if available.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// Embedding index
// ---------------------------------------------------------------------------

/// Configuration for the embedding index.
#[derive(Debug, Clone)]
pub struct EmbeddingIndexConfig {
    /// Expected dimensionality for all embeddings.
    pub dimension: usize,
    /// Number of partitions for coarse quantization (0 = brute force).
    pub num_partitions: usize,
    /// Similarity threshold below which results are discarded.
    pub min_similarity: f32,
}

impl Default for EmbeddingIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            num_partitions: 0,
            min_similarity: 0.0,
        }
    }
}

/// In-memory vector embedding index supporting cosine similarity search.
///
/// Stores dense floating-point vectors and supports both brute-force and
/// partitioned (IVF-flat style) approximate search.
#[derive(Debug)]
pub struct EmbeddingIndex {
    /// Configuration.
    config: EmbeddingIndexConfig,
    /// Stored embeddings.
    embeddings: Vec<Embedding>,
    /// Asset ID to index mapping for fast deletion.
    id_map: HashMap<Uuid, usize>,
    /// Partition centroids (only populated when `num_partitions > 0`).
    centroids: Vec<Vec<f32>>,
    /// Partition assignments (index in `embeddings` -> partition id).
    partition_assignments: Vec<usize>,
    /// Whether the index needs re-partitioning.
    dirty: bool,
}

impl EmbeddingIndex {
    /// Create a new embedding index with the given configuration.
    #[must_use]
    pub fn new(config: EmbeddingIndexConfig) -> Self {
        Self {
            config,
            embeddings: Vec::new(),
            id_map: HashMap::new(),
            centroids: Vec::new(),
            partition_assignments: Vec::new(),
            dirty: false,
        }
    }

    /// Create an index with default configuration and the given dimensionality.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(EmbeddingIndexConfig {
            dimension,
            ..Default::default()
        })
    }

    /// Add an embedding to the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding dimension doesn't match the index.
    pub fn add(&mut self, embedding: Embedding) -> SearchResult<()> {
        if embedding.vector.len() != self.config.dimension {
            return Err(SearchError::FeatureExtraction(format!(
                "Expected dimension {}, got {}",
                self.config.dimension,
                embedding.vector.len()
            )));
        }

        // If asset already exists, replace it
        if let Some(&idx) = self.id_map.get(&embedding.asset_id) {
            self.embeddings[idx] = embedding;
        } else {
            let idx = self.embeddings.len();
            self.id_map.insert(embedding.asset_id, idx);
            self.embeddings.push(embedding);
            self.partition_assignments.push(0);
        }

        self.dirty = true;
        Ok(())
    }

    /// Remove an embedding by asset ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the asset is not found.
    pub fn remove(&mut self, asset_id: Uuid) -> SearchResult<()> {
        let idx = self
            .id_map
            .remove(&asset_id)
            .ok_or_else(|| SearchError::DocumentNotFound(asset_id.to_string()))?;

        // Swap-remove for O(1) deletion
        let last_idx = self.embeddings.len() - 1;
        if idx != last_idx {
            let moved_id = self.embeddings[last_idx].asset_id;
            self.embeddings.swap(idx, last_idx);
            self.partition_assignments.swap(idx, last_idx);
            self.id_map.insert(moved_id, idx);
        }
        self.embeddings.pop();
        self.partition_assignments.pop();
        self.dirty = true;
        Ok(())
    }

    /// Search for the `k` most similar embeddings to the query vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index.
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<SimilarityResult>> {
        if query.len() != self.config.dimension {
            return Err(SearchError::InvalidQuery(format!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.config.dimension
            )));
        }

        let normalized_query = l2_normalize(query);

        if self.config.num_partitions > 0 && !self.centroids.is_empty() && !self.dirty {
            self.search_partitioned(&normalized_query, k)
        } else {
            self.search_brute_force(&normalized_query, k)
        }
    }

    /// Brute-force search: compute cosine similarity against every embedding.
    fn search_brute_force(&self, query: &[f32], k: usize) -> SearchResult<Vec<SimilarityResult>> {
        let mut scored: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .map(|emb| {
                let sim = dot_product(query, &emb.vector);
                SimilarityResult {
                    asset_id: emb.asset_id,
                    similarity: sim,
                    label: emb.label.clone(),
                }
            })
            .filter(|r| r.similarity >= self.config.min_similarity)
            .collect();

        // Partial sort: only keep top-k
        scored.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Partitioned search: only scan the nearest partition(s).
    fn search_partitioned(&self, query: &[f32], k: usize) -> SearchResult<Vec<SimilarityResult>> {
        // Find the nearest centroid(s) — probe the top 2 partitions
        let n_probe = 2.min(self.centroids.len());
        let mut centroid_sims: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot_product(query, c)))
            .collect();
        centroid_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let probe_partitions: Vec<usize> = centroid_sims
            .iter()
            .take(n_probe)
            .map(|&(i, _)| i)
            .collect();

        let mut scored: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .enumerate()
            .filter(|&(i, _)| {
                i < self.partition_assignments.len()
                    && probe_partitions.contains(&self.partition_assignments[i])
            })
            .map(|(_, emb)| {
                let sim = dot_product(query, &emb.vector);
                SimilarityResult {
                    asset_id: emb.asset_id,
                    similarity: sim,
                    label: emb.label.clone(),
                }
            })
            .filter(|r| r.similarity >= self.config.min_similarity)
            .collect();

        scored.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Build or rebuild partition centroids using simple k-means.
    ///
    /// Only meaningful when `config.num_partitions > 0`.
    pub fn build_partitions(&mut self) {
        let k = self.config.num_partitions;
        if k == 0 || self.embeddings.is_empty() {
            return;
        }

        let dim = self.config.dimension;
        let n = self.embeddings.len();
        let k = k.min(n);

        // Initialize centroids from evenly spaced embeddings
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| {
                let idx = i * n / k;
                self.embeddings[idx].vector.clone()
            })
            .collect();

        // Run k-means for a fixed number of iterations
        let max_iters = 20;
        let mut assignments = vec![0usize; n];

        for _ in 0..max_iters {
            // Assignment step
            let mut changed = false;
            for (i, emb) in self.embeddings.iter().enumerate() {
                let mut best_partition = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (j, centroid) in centroids.iter().enumerate() {
                    let sim = dot_product(&emb.vector, centroid);
                    if sim > best_sim {
                        best_sim = sim;
                        best_partition = j;
                    }
                }
                if assignments[i] != best_partition {
                    assignments[i] = best_partition;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step: recompute centroids
            let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, emb) in self.embeddings.iter().enumerate() {
                let p = assignments[i];
                counts[p] += 1;
                for (j, &v) in emb.vector.iter().enumerate() {
                    sums[p][j] += v;
                }
            }

            for (p, sum) in sums.iter().enumerate() {
                if counts[p] > 0 {
                    let c: Vec<f32> = sum.iter().map(|&s| s / counts[p] as f32).collect();
                    centroids[p] = l2_normalize(&c);
                }
            }
        }

        self.centroids = centroids;
        self.partition_assignments = assignments;
        self.dirty = false;
    }

    /// Number of embeddings in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get the configured dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Number of partitions configured.
    #[must_use]
    pub fn num_partitions(&self) -> usize {
        self.config.num_partitions
    }

    /// Check if the index has an embedding for the given asset.
    #[must_use]
    pub fn contains(&self, asset_id: Uuid) -> bool {
        self.id_map.contains_key(&asset_id)
    }

    /// Get the embedding for a specific asset.
    #[must_use]
    pub fn get(&self, asset_id: Uuid) -> Option<&Embedding> {
        self.id_map.get(&asset_id).map(|&idx| &self.embeddings[idx])
    }
}

// ---------------------------------------------------------------------------
// Vector math utilities
// ---------------------------------------------------------------------------

/// Compute the dot product of two vectors.
#[must_use]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize a vector. Returns a zero vector if the input has zero magnitude.
#[must_use]
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Compute cosine similarity between two raw (not necessarily normalized) vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Compute Euclidean distance between two vectors.
#[must_use]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(id: Uuid, values: &[f32]) -> Embedding {
        Embedding::new(id, values.to_vec(), "test-model")
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x.abs() < f32::EPSILON));
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &a);
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_new_normalizes() {
        let id = Uuid::new_v4();
        let emb = Embedding::new(id, vec![3.0, 4.0], "model");
        let mag: f32 = emb.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-5);
        assert_eq!(emb.dimension(), 2);
        assert_eq!(emb.model, "model");
        assert!(emb.label.is_none());
    }

    #[test]
    fn test_embedding_with_label() {
        let id = Uuid::new_v4();
        let emb = Embedding::with_label(id, vec![1.0, 0.0], "model", "cat");
        assert_eq!(emb.label.as_deref(), Some("cat"));
    }

    #[test]
    fn test_index_add_and_search() {
        let mut index = EmbeddingIndex::with_dimension(3);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        index
            .add(make_embedding(id1, &[1.0, 0.0, 0.0]))
            .expect("should add");
        index
            .add(make_embedding(id2, &[0.9, 0.1, 0.0]))
            .expect("should add");
        index
            .add(make_embedding(id3, &[0.0, 0.0, 1.0]))
            .expect("should add");

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
        assert!(index.contains(id1));

        let results = index.search(&[1.0, 0.0, 0.0], 2).expect("should search");
        assert_eq!(results.len(), 2);
        // id1 or id2 should be most similar to [1, 0, 0]
        assert!(results[0].similarity > results[1].similarity);
        // The most similar should be id1 (exact match direction)
        assert_eq!(results[0].asset_id, id1);
    }

    #[test]
    fn test_index_dimension_mismatch_add() {
        let mut index = EmbeddingIndex::with_dimension(3);
        let id = Uuid::new_v4();
        let result = index.add(Embedding::new(id, vec![1.0, 0.0], "model"));
        assert!(result.is_err());
    }

    #[test]
    fn test_index_dimension_mismatch_search() {
        let index = EmbeddingIndex::with_dimension(3);
        let result = index.search(&[1.0, 0.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_remove() {
        let mut index = EmbeddingIndex::with_dimension(2);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index
            .add(make_embedding(id1, &[1.0, 0.0]))
            .expect("should add");
        index
            .add(make_embedding(id2, &[0.0, 1.0]))
            .expect("should add");
        assert_eq!(index.len(), 2);

        index.remove(id1).expect("should remove");
        assert_eq!(index.len(), 1);
        assert!(!index.contains(id1));
        assert!(index.contains(id2));
    }

    #[test]
    fn test_index_remove_not_found() {
        let mut index = EmbeddingIndex::with_dimension(2);
        let result = index.remove(Uuid::new_v4());
        assert!(result.is_err());
    }

    #[test]
    fn test_index_replace_existing() {
        let mut index = EmbeddingIndex::with_dimension(2);
        let id = Uuid::new_v4();

        index
            .add(make_embedding(id, &[1.0, 0.0]))
            .expect("should add");
        index
            .add(make_embedding(id, &[0.0, 1.0]))
            .expect("should replace");

        assert_eq!(index.len(), 1);
        let results = index.search(&[0.0, 1.0], 1).expect("should search");
        assert_eq!(results[0].asset_id, id);
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_index_get() {
        let mut index = EmbeddingIndex::with_dimension(2);
        let id = Uuid::new_v4();
        index
            .add(make_embedding(id, &[1.0, 0.0]))
            .expect("should add");

        let emb = index.get(id);
        assert!(emb.is_some());
        assert_eq!(emb.map(|e| e.asset_id), Some(id));

        assert!(index.get(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_index_empty_search() {
        let index = EmbeddingIndex::with_dimension(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5).expect("should search");
        assert!(results.is_empty());
    }

    #[test]
    fn test_min_similarity_filter() {
        let config = EmbeddingIndexConfig {
            dimension: 2,
            num_partitions: 0,
            min_similarity: 0.9,
        };
        let mut index = EmbeddingIndex::new(config);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index
            .add(make_embedding(id1, &[1.0, 0.0]))
            .expect("should add");
        index
            .add(make_embedding(id2, &[0.0, 1.0]))
            .expect("should add");

        let results = index.search(&[1.0, 0.0], 10).expect("should search");
        // Only id1 should pass the 0.9 similarity threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].asset_id, id1);
    }

    #[test]
    fn test_partitioned_search() {
        let config = EmbeddingIndexConfig {
            dimension: 4,
            num_partitions: 2,
            min_similarity: 0.0,
        };
        let mut index = EmbeddingIndex::new(config);

        // Add embeddings in two clusters
        for i in 0..10 {
            let id = Uuid::new_v4();
            let v = if i < 5 {
                vec![1.0, 0.1 * i as f32, 0.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0, 0.1 * i as f32]
            };
            index
                .add(Embedding::new(id, v, "test"))
                .expect("should add");
        }

        index.build_partitions();

        let results = index
            .search(&[1.0, 0.0, 0.0, 0.0], 3)
            .expect("should search");
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // All results should have positive similarity to [1,0,0,0]
        for r in &results {
            assert!(r.similarity > 0.0);
        }
    }

    #[test]
    fn test_build_partitions_empty() {
        let config = EmbeddingIndexConfig {
            dimension: 2,
            num_partitions: 3,
            min_similarity: 0.0,
        };
        let mut index = EmbeddingIndex::new(config);
        // Should not panic on empty index
        index.build_partitions();
        assert!(index.centroids.is_empty());
    }

    #[test]
    fn test_build_partitions_fewer_than_k() {
        let config = EmbeddingIndexConfig {
            dimension: 2,
            num_partitions: 10,
            min_similarity: 0.0,
        };
        let mut index = EmbeddingIndex::new(config);
        // Only 3 embeddings but 10 partitions requested
        for _ in 0..3 {
            let id = Uuid::new_v4();
            index
                .add(make_embedding(id, &[1.0, 0.0]))
                .expect("should add");
        }
        index.build_partitions();
        // Should clamp to min(k, n)
        assert!(index.centroids.len() <= 3);
    }

    #[test]
    fn test_similarity_result_label() {
        let mut index = EmbeddingIndex::with_dimension(2);
        let id = Uuid::new_v4();
        index
            .add(Embedding::with_label(id, vec![1.0, 0.0], "model", "sunset"))
            .expect("should add");

        let results = index.search(&[1.0, 0.0], 1).expect("should search");
        assert_eq!(results[0].label.as_deref(), Some("sunset"));
    }

    #[test]
    fn test_index_config_defaults() {
        let config = EmbeddingIndexConfig::default();
        assert_eq!(config.dimension, 512);
        assert_eq!(config.num_partitions, 0);
        assert!((config.min_similarity).abs() < f32::EPSILON);
    }

    #[test]
    fn test_large_batch_search() {
        let mut index = EmbeddingIndex::with_dimension(8);
        // Add 100 embeddings with varying directions
        for i in 0..100 {
            let id = Uuid::new_v4();
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            v[(i + 1) % 8] = 0.5;
            index
                .add(Embedding::new(id, v, "test"))
                .expect("should add");
        }

        let query = vec![1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 5).expect("should search");
        assert_eq!(results.len(), 5);
        // Results should be sorted by similarity descending
        for w in results.windows(2) {
            assert!(w[0].similarity >= w[1].similarity);
        }
    }
}
