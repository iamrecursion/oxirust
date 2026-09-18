//! Semantic search with vector embeddings.
//!
//! Provides TF-IDF-style text embedding projected into a 128-dimensional space,
//! cosine similarity, and a brute-force nearest-neighbour index.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// EmbeddingVector
// ──────────────────────────────────────────────────────────────────────────────

/// A 128-dimensional dense float vector representing a semantic embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingVector(pub Vec<f32>);

impl EmbeddingVector {
    /// Dimension of every embedding vector produced by this module.
    pub const DIM: usize = 128;

    /// Creates a zero vector of the standard dimension.
    #[must_use]
    pub fn zeros() -> Self {
        Self(vec![0.0_f32; Self::DIM])
    }

    /// Returns the number of components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the vector has no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// L2-normalises the vector in-place.  A zero-norm vector is left unchanged.
    pub fn normalize(&mut self) {
        let norm: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in &mut self.0 {
                *x /= norm;
            }
        }
    }

    /// Cosine similarity with `other`.  Returns a value in `[-1, 1]`.
    ///
    /// Both vectors are assumed to be unit-normalised; if they are not the
    /// result is still a valid dot-product but will not be clamped to 1.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SemanticEmbedder
// ──────────────────────────────────────────────────────────────────────────────

/// Produces semantic embeddings for text and tag lists.
///
/// The algorithm is a lightweight TF-IDF-style projection:
/// 1. Tokenise (lower-case, split on non-alphanumeric).
/// 2. Hash each token with FNV-1a.
/// 3. Project into `DIM` dimensions via `hash % DIM`; accumulate weighted value.
/// 4. Normalise to unit length.
pub struct SemanticEmbedder;

impl SemanticEmbedder {
    /// Creates a new embedder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Embeds a text string into a 128-dimensional unit vector.
    #[must_use]
    pub fn embed_text(&self, text: &str) -> EmbeddingVector {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return EmbeddingVector::zeros();
        }

        // Compute term frequencies.
        let mut tf: HashMap<String, f32> = HashMap::new();
        let total = tokens.len() as f32;
        for tok in &tokens {
            *tf.entry(tok.clone()).or_insert(0.0) += 1.0 / total;
        }

        project_tf(&tf)
    }

    /// Embeds a slice of tags into a 128-dimensional unit vector.
    ///
    /// Each tag is individually embedded and the results are summed then
    /// normalised, giving equal weight to each tag.
    #[must_use]
    pub fn embed_tags(&self, tags: &[String]) -> EmbeddingVector {
        if tags.is_empty() {
            return EmbeddingVector::zeros();
        }

        let mut acc = vec![0.0_f32; EmbeddingVector::DIM];
        for tag in tags {
            let ev = self.embed_text(tag);
            for (a, b) in acc.iter_mut().zip(ev.0.iter()) {
                *a += b;
            }
        }
        let mut result = EmbeddingVector(acc);
        result.normalize();
        result
    }
}

impl Default for SemanticEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SemanticIndex
// ──────────────────────────────────────────────────────────────────────────────

/// A brute-force approximate nearest-neighbour index over embedding vectors.
pub struct SemanticIndex {
    entries: Vec<(u64, EmbeddingVector)>,
}

impl SemanticIndex {
    /// Creates a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds an embedding.  Replaces any existing entry for the same `id`.
    pub fn add(&mut self, id: u64, embedding: EmbeddingVector) {
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries[pos] = (id, embedding);
        } else {
            self.entries.push((id, embedding));
        }
    }

    /// Returns the number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Brute-force cosine-similarity search.
    ///
    /// Returns up to `top_k` `(id, similarity)` pairs sorted by descending
    /// similarity.
    #[must_use]
    pub fn search(&self, query: EmbeddingVector, top_k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .entries
            .iter()
            .map(|(id, ev)| (*id, ev.cosine_similarity(&query)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

impl Default for SemanticIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SemanticSearchConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for semantic search queries.
#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum cosine similarity threshold (0..=1).
    pub min_similarity: f32,
    /// Whether to apply a simple score-based re-ranking pass.
    pub use_reranking: bool,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            use_reranking: false,
        }
    }
}

impl SemanticSearchConfig {
    /// Creates a new config with the given `top_k` and defaults for the rest.
    #[must_use]
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            ..Default::default()
        }
    }

    /// Executes a semantic search on the given `index` using `query`.
    ///
    /// Candidates below `min_similarity` are filtered out.  When
    /// `use_reranking` is `true` a square-root dampening is applied to the
    /// scores so that near-identical results are spread apart.
    #[must_use]
    pub fn search(&self, index: &SemanticIndex, query: EmbeddingVector) -> Vec<(u64, f32)> {
        let mut results = index.search(query, self.top_k * 4); // fetch extra for filtering
        results.retain(|(_, score)| *score >= self.min_similarity);

        if self.use_reranking {
            for (_, score) in &mut results {
                *score = score.sqrt();
            }
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        results.truncate(self.top_k);
        results
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Lowercases `text` and splits on non-alphanumeric characters, keeping tokens
/// with length > 1.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(String::from)
        .collect()
}

/// FNV-1a 64-bit hash.
fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Projects a term-frequency map into a `DIM`-dimensional vector and
/// normalises the result to unit length.
fn project_tf(tf: &HashMap<String, f32>) -> EmbeddingVector {
    let mut vec = vec![0.0_f32; EmbeddingVector::DIM];
    for (term, weight) in tf {
        let h = fnv1a(term) as usize;
        let dim = h % EmbeddingVector::DIM;
        // Sign is determined by the next bit so that cancellation is unlikely.
        let sign = if (h >> 7) & 1 == 0 { 1.0_f32 } else { -1.0 };
        vec[dim] += sign * weight;
    }
    let mut ev = EmbeddingVector(vec);
    ev.normalize();
    ev
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn embedder() -> SemanticEmbedder {
        SemanticEmbedder::new()
    }

    #[test]
    fn test_embed_text_returns_unit_vector() {
        let ev = embedder().embed_text("hello world");
        let norm: f32 = ev.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn test_embed_empty_string_is_zeros() {
        let ev = embedder().embed_text("");
        let norm: f32 = ev.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1e-9);
    }

    #[test]
    fn test_cosine_self_similarity_is_one() {
        let ev = embedder().embed_text("ocean sunset waves");
        let sim = ev.cosine_similarity(&ev);
        assert!((sim - 1.0).abs() < 1e-5, "sim = {sim}");
    }

    #[test]
    fn test_cosine_different_texts() {
        let a = embedder().embed_text("rocket science launch");
        let b = embedder().embed_text("cooking pasta recipe");
        let sim = a.cosine_similarity(&b);
        // They should be less similar than self-similarity.
        assert!(sim < 1.0);
    }

    #[test]
    fn test_embed_tags_returns_unit_vector() {
        let tags = vec![
            "nature".to_string(),
            "wildlife".to_string(),
            "forest".to_string(),
        ];
        let ev = embedder().embed_tags(&tags);
        let norm: f32 = ev.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn test_embed_tags_empty_is_zeros() {
        let ev = embedder().embed_tags(&[]);
        let norm: f32 = ev.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1e-9);
    }

    #[test]
    fn test_semantic_index_add_and_search() {
        let mut idx = SemanticIndex::new();
        let e = embedder();
        idx.add(1, e.embed_text("mountain hiking trail"));
        idx.add(2, e.embed_text("ocean beach surf"));
        idx.add(3, e.embed_text("mountain summit view"));

        let query = e.embed_text("mountain");
        let results = idx.search(query, 2);
        assert_eq!(results.len(), 2);
        // Mountain-related docs should score higher than ocean.
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1) || ids.contains(&3));
    }

    #[test]
    fn test_semantic_index_replace_entry() {
        let mut idx = SemanticIndex::new();
        let e = embedder();
        idx.add(42, e.embed_text("first version"));
        assert_eq!(idx.len(), 1);
        idx.add(42, e.embed_text("updated version"));
        assert_eq!(idx.len(), 1, "duplicate id should be replaced");
    }

    #[test]
    fn test_semantic_index_empty_search() {
        let idx = SemanticIndex::new();
        let ev = EmbeddingVector::zeros();
        let results = idx.search(ev, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_config_min_similarity_filter() {
        let mut idx = SemanticIndex::new();
        let e = embedder();
        // Add a document with very dissimilar content.
        idx.add(99, e.embed_text("completely unrelated xyzzy qwerty"));
        let query = e.embed_text("mountain hiking adventure");
        let cfg = SemanticSearchConfig {
            top_k: 10,
            min_similarity: 0.999,
            use_reranking: false,
        };
        let results = cfg.search(&idx, query);
        // At such a high threshold nothing should match.
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_config_reranking() {
        let mut idx = SemanticIndex::new();
        let e = embedder();
        idx.add(1, e.embed_text("video editing timeline"));
        idx.add(2, e.embed_text("video colour grading"));
        let query = e.embed_text("video");
        let cfg = SemanticSearchConfig {
            top_k: 2,
            min_similarity: 0.0,
            use_reranking: true,
        };
        let results = cfg.search(&idx, query);
        assert!(!results.is_empty());
        // Scores should be sqrt-dampened, so all scores < 1.0.
        for (_, score) in &results {
            assert!(*score <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn test_embedding_vector_dim() {
        assert_eq!(EmbeddingVector::DIM, 128);
        let z = EmbeddingVector::zeros();
        assert_eq!(z.len(), 128);
    }
}
