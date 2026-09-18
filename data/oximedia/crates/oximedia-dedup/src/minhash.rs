#![allow(dead_code)]
//! `MinHash`-based approximate similarity estimation.
//!
//! Implements the `MinHash` algorithm for efficiently estimating Jaccard
//! similarity between sets, useful for near-duplicate detection on
//! large media collections without pairwise comparison.

use std::collections::HashSet;

/// Default number of hash functions in a `MinHash` signature.
const DEFAULT_NUM_HASHES: usize = 128;

/// Large prime for hash computation.
const HASH_PRIME: u64 = 0x0001_0000_01B3;

/// Modulus for hash computation.
const HASH_MOD: u64 = (1u64 << 61) - 1;

/// A single hash function parameterized by (a, b).
#[derive(Debug, Clone, Copy)]
pub struct HashParams {
    /// Multiplier.
    a: u64,
    /// Offset.
    b: u64,
}

impl HashParams {
    /// Create new hash parameters.
    pub fn new(a: u64, b: u64) -> Self {
        Self { a, b }
    }

    /// Apply this hash function to a value.
    pub fn apply(&self, value: u64) -> u64 {
        // (a * value + b) mod HASH_MOD
        let av = self.a.wrapping_mul(value);
        let avb = av.wrapping_add(self.b);
        avb % HASH_MOD
    }
}

/// Generate deterministic hash parameters from a seed.
fn generate_hash_params(num_hashes: usize, seed: u64) -> Vec<HashParams> {
    let mut params = Vec::with_capacity(num_hashes);
    let mut state = seed;
    for _ in 0..num_hashes {
        state = state.wrapping_mul(HASH_PRIME).wrapping_add(1);
        let a = (state % HASH_MOD).max(1);
        state = state.wrapping_mul(HASH_PRIME).wrapping_add(1);
        let b = state % HASH_MOD;
        params.push(HashParams::new(a, b));
    }
    params
}

/// A `MinHash` signature representing a set.
#[derive(Debug, Clone)]
pub struct MinHashSignature {
    /// The minimum hash values for each hash function.
    values: Vec<u64>,
}

impl MinHashSignature {
    /// Create a new empty signature with the given number of hashes.
    pub fn new(num_hashes: usize) -> Self {
        Self {
            values: vec![u64::MAX; num_hashes],
        }
    }

    /// Get the signature values.
    pub fn values(&self) -> &[u64] {
        &self.values
    }

    /// Get the number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.values.len()
    }

    /// Estimate Jaccard similarity with another signature.
    #[allow(clippy::cast_precision_loss)]
    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        if self.values.is_empty() {
            return 1.0;
        }
        let matches = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.values.len() as f64
    }
}

/// `MinHash` engine for computing signatures.
#[derive(Debug, Clone)]
pub struct MinHasher {
    /// Hash function parameters.
    params: Vec<HashParams>,
    /// Number of hash functions.
    num_hashes: usize,
}

impl MinHasher {
    /// Create a new `MinHash` engine with the default number of hashes.
    pub fn new() -> Self {
        Self::with_num_hashes(DEFAULT_NUM_HASHES)
    }

    /// Create a new `MinHash` engine with a specific number of hashes.
    pub fn with_num_hashes(num_hashes: usize) -> Self {
        let num_hashes = num_hashes.max(1);
        let params = generate_hash_params(num_hashes, 42);
        Self { params, num_hashes }
    }

    /// Create with a custom seed for reproducibility.
    pub fn with_seed(num_hashes: usize, seed: u64) -> Self {
        let num_hashes = num_hashes.max(1);
        let params = generate_hash_params(num_hashes, seed);
        Self { params, num_hashes }
    }

    /// Compute a `MinHash` signature from a set of elements.
    pub fn compute_signature(&self, elements: &[u64]) -> MinHashSignature {
        let mut sig = MinHashSignature::new(self.num_hashes);
        for &elem in elements {
            for (i, param) in self.params.iter().enumerate() {
                let h = param.apply(elem);
                if h < sig.values[i] {
                    sig.values[i] = h;
                }
            }
        }
        sig
    }

    /// Compute signature from byte shingles of a given width.
    pub fn compute_from_bytes(&self, data: &[u8], shingle_width: usize) -> MinHashSignature {
        let width = shingle_width.max(1);
        if data.len() < width {
            return MinHashSignature::new(self.num_hashes);
        }

        let mut elements = Vec::new();
        for window in data.windows(width) {
            let mut h: u64 = 0;
            for &b in window {
                h = h.wrapping_mul(31).wrapping_add(u64::from(b));
            }
            elements.push(h);
        }
        self.compute_signature(&elements)
    }

    /// Get the number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of `MinHash` signatures for batch similarity queries.
#[derive(Debug, Clone)]
pub struct MinHashIndex {
    /// Stored signatures with labels.
    entries: Vec<(String, MinHashSignature)>,
    /// Similarity threshold for duplicate detection.
    threshold: f64,
}

impl MinHashIndex {
    /// Create a new index with the given similarity threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            entries: Vec::new(),
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Add a signature to the index.
    pub fn insert(&mut self, label: String, signature: MinHashSignature) {
        self.entries.push((label, signature));
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find all entries similar to a query signature.
    pub fn query_similar(&self, query: &MinHashSignature) -> Vec<(&str, f64)> {
        let mut results = Vec::new();
        for (label, sig) in &self.entries {
            let sim = query.jaccard_similarity(sig);
            if sim >= self.threshold {
                results.push((label.as_str(), sim));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find all duplicate pairs above the threshold.
    pub fn find_duplicates(&self) -> Vec<(&str, &str, f64)> {
        let mut pairs = Vec::new();
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                let sim = self.entries[i].1.jaccard_similarity(&self.entries[j].1);
                if sim >= self.threshold {
                    pairs.push((self.entries[i].0.as_str(), self.entries[j].0.as_str(), sim));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Compute exact Jaccard similarity between two sets for comparison.
#[allow(clippy::cast_precision_loss)]
pub fn exact_jaccard(set_a: &[u64], set_b: &[u64]) -> f64 {
    let a: HashSet<u64> = set_a.iter().copied().collect();
    let b: HashSet<u64> = set_b.iter().copied().collect();
    let intersection = a.intersection(&b).count();
    let union = a.union(&b).count();
    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_params_apply() {
        let p = HashParams::new(3, 7);
        let h1 = p.apply(10);
        let h2 = p.apply(10);
        assert_eq!(h1, h2); // deterministic
    }

    #[test]
    fn test_generate_hash_params() {
        let params = generate_hash_params(10, 42);
        assert_eq!(params.len(), 10);
        // All a values should be >= 1
        for p in &params {
            assert!(p.a >= 1);
        }
    }

    #[test]
    fn test_minhash_signature_new() {
        let sig = MinHashSignature::new(64);
        assert_eq!(sig.num_hashes(), 64);
        assert!(sig.values().iter().all(|&v| v == u64::MAX));
    }

    #[test]
    fn test_jaccard_identical() {
        let hasher = MinHasher::with_num_hashes(64);
        let elements = vec![1, 2, 3, 4, 5];
        let sig1 = hasher.compute_signature(&elements);
        let sig2 = hasher.compute_signature(&elements);
        assert!((sig1.jaccard_similarity(&sig2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let hasher = MinHasher::with_num_hashes(256);
        let a: Vec<u64> = (0..100).collect();
        let b: Vec<u64> = (10000..10100).collect();
        let sig_a = hasher.compute_signature(&a);
        let sig_b = hasher.compute_signature(&b);
        let sim = sig_a.jaccard_similarity(&sig_b);
        // Disjoint sets should have very low similarity
        assert!(sim < 0.15, "Expected low similarity, got {sim}");
    }

    #[test]
    fn test_jaccard_similar_sets() {
        let hasher = MinHasher::with_num_hashes(256);
        // 90% overlap
        let a: Vec<u64> = (0..100).collect();
        let b: Vec<u64> = (0..90).chain(200..210).collect();
        let sig_a = hasher.compute_signature(&a);
        let sig_b = hasher.compute_signature(&b);
        let estimated = sig_a.jaccard_similarity(&sig_b);
        let exact = exact_jaccard(&a, &b);
        // Estimated should be reasonably close to exact
        assert!(
            (estimated - exact).abs() < 0.2,
            "Estimated {estimated} vs exact {exact}"
        );
    }

    #[test]
    fn test_jaccard_different_lengths() {
        let sig_a = MinHashSignature::new(10);
        let sig_b = MinHashSignature::new(20);
        assert!((sig_a.jaccard_similarity(&sig_b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_empty_signature() {
        let sig_a = MinHashSignature::new(0);
        let sig_b = MinHashSignature::new(0);
        assert!((sig_a.jaccard_similarity(&sig_b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_minhasher_default() {
        let hasher = MinHasher::default();
        assert_eq!(hasher.num_hashes(), DEFAULT_NUM_HASHES);
    }

    #[test]
    fn test_compute_from_bytes() {
        let hasher = MinHasher::with_num_hashes(64);
        let data = b"Hello, World! This is a test string for minhash.";
        let sig = hasher.compute_from_bytes(data, 4);
        assert_eq!(sig.num_hashes(), 64);
        // Should have found some minimums
        assert!(sig.values().iter().any(|&v| v < u64::MAX));
    }

    #[test]
    fn test_compute_from_bytes_too_short() {
        let hasher = MinHasher::with_num_hashes(16);
        let sig = hasher.compute_from_bytes(b"ab", 5);
        // Data too short for shingle_width=5, should be all MAX
        assert!(sig.values().iter().all(|&v| v == u64::MAX));
    }

    #[test]
    fn test_minhash_index_insert_and_query() {
        let hasher = MinHasher::with_num_hashes(64);
        let mut index = MinHashIndex::new(0.5);
        let sig1 = hasher.compute_signature(&[1, 2, 3, 4, 5]);
        let sig2 = hasher.compute_signature(&[1, 2, 3, 4, 5]); // identical
        let sig3 = hasher.compute_signature(&[100, 200, 300, 400, 500]);

        index.insert("file1".to_string(), sig1);
        index.insert("file2".to_string(), sig3);

        let results = index.query_similar(&sig2);
        // sig2 == sig1, so file1 should be found
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "file1");
    }

    #[test]
    fn test_minhash_index_find_duplicates() {
        let hasher = MinHasher::with_num_hashes(64);
        let mut index = MinHashIndex::new(0.9);
        let sig1 = hasher.compute_signature(&[1, 2, 3, 4, 5]);
        let sig2 = hasher.compute_signature(&[1, 2, 3, 4, 5]);
        let sig3 = hasher.compute_signature(&[100, 200, 300]);

        index.insert("a.mp4".to_string(), sig1);
        index.insert("b.mp4".to_string(), sig2);
        index.insert("c.mp4".to_string(), sig3);

        let dupes = index.find_duplicates();
        // a.mp4 and b.mp4 should be duplicates
        assert!(dupes.iter().any(|(a, b, _)| *a == "a.mp4" && *b == "b.mp4"));
    }

    #[test]
    fn test_minhash_index_empty() {
        let index = MinHashIndex::new(0.5);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_exact_jaccard() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5, 6, 7];
        let j = exact_jaccard(&a, &b);
        // intersection = {3,4,5} = 3, union = {1,2,3,4,5,6,7} = 7
        assert!((j - 3.0 / 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_jaccard_empty() {
        let j = exact_jaccard(&[], &[]);
        assert!((j - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_seed_deterministic() {
        let h1 = MinHasher::with_seed(32, 12345);
        let h2 = MinHasher::with_seed(32, 12345);
        let data = vec![10, 20, 30];
        let s1 = h1.compute_signature(&data);
        let s2 = h2.compute_signature(&data);
        assert_eq!(s1.values(), s2.values());
    }
}
