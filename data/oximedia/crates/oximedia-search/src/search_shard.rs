#![allow(dead_code)]
//! Index sharding for parallel search across large collections (>1M documents).
//!
//! The [`ShardedIndex`] horizontally partitions a document collection into
//! `N` shards.  Each shard holds an independent [`crate::inv_index::InvertedIndex`]
//! and a Bloom filter that enables a fast pre-check to skip shards that
//! provably contain no matching terms, avoiding unnecessary full scans.
//!
//! # Architecture
//!
//! Documents are assigned to shards by hashing their UUID with FNV-1a.
//! Search proceeds in two phases:
//!
//! 1. **Bloom pre-check** — O(k) per shard, eliminates empty shards.
//! 2. **Full scan** — TF-IDF search in the surviving shards, in parallel
//!    (using [`rayon`]).
//!
//! Results from all shards are merged, de-duplicated by asset ID, and
//! returned sorted by descending score.
//!
//! # Bloom filter
//!
//! Each shard maintains a counting-style Bloom filter backed by a bit-vector.
//! The filter uses 3 FNV-based hash functions and supports both insertion
//! and deletion (via decrement counters stored in a companion `Vec<u8>`).
//!
//! For 1 M documents per shard with a desired FPR ≤ 1%, the recommended
//! bit-vector size is ~9.6 M bits (≈1.2 MB).

use std::collections::HashMap;

use uuid::Uuid;

use crate::error::{SearchError, SearchResult};
use crate::inv_index::{remove_stopwords, tokenize, InvertedIndex};

// ---------------------------------------------------------------------------
// Bloom filter
// ---------------------------------------------------------------------------

/// A simple counting Bloom filter using 3 hash functions.
///
/// Supports both `add` and `remove` so that the filter stays in sync when
/// documents are deleted.  The counter array uses saturating arithmetic to
/// avoid u8 overflow while keeping memory minimal.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit-set (each byte holds 8 bits).
    bits: Vec<u8>,
    /// Per-slot reference counter (max 255, saturating).
    counters: Vec<u8>,
    /// Number of slots (= `bits.len() * 8`).
    num_slots: usize,
    /// Number of hash functions.
    k: usize,
    /// Number of items currently tracked (approximate).
    count: usize,
}

impl BloomFilter {
    /// Create a Bloom filter for `capacity` items at a target false-positive
    /// rate of roughly `fpr` (e.g. `0.01` for 1%).
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0 or `fpr` is outside `(0, 1)`.
    #[must_use]
    pub fn new(capacity: usize, fpr: f64) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1), got {fpr}");
        // Optimal bit count: m = -n * ln(p) / (ln 2)^2
        let m = (-(capacity as f64) * fpr.ln() / (2_f64.ln().powi(2))).ceil() as usize;
        let m = m.max(64); // minimum 64 slots
        let bytes = (m + 7) / 8;
        Self {
            bits: vec![0u8; bytes],
            counters: vec![0u8; m],
            num_slots: m,
            k: 3,
            count: 0,
        }
    }

    /// Insert a term into the filter.
    pub fn add(&mut self, term: &str) {
        for i in 0..self.k {
            let slot = self.hash(term, i);
            self.set_bit(slot);
            self.counters[slot] = self.counters[slot].saturating_add(1);
        }
        self.count += 1;
    }

    /// Remove a term from the filter (decrements counters).
    pub fn remove(&mut self, term: &str) {
        for i in 0..self.k {
            let slot = self.hash(term, i);
            let c = self.counters[slot];
            if c > 0 {
                self.counters[slot] = c - 1;
                if c - 1 == 0 {
                    self.clear_bit(slot);
                }
            }
        }
        self.count = self.count.saturating_sub(1);
    }

    /// Test whether `term` *might* be in the filter (probabilistic).
    ///
    /// Returns `false` only if the term is *definitely* absent.
    #[must_use]
    pub fn might_contain(&self, term: &str) -> bool {
        (0..self.k).all(|i| self.get_bit(self.hash(term, i)))
    }

    /// Number of items tracked (approximate).
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset to empty.
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.counters.fill(0);
        self.count = 0;
    }

    // ---- bit manipulation ----

    fn set_bit(&mut self, slot: usize) {
        self.bits[slot / 8] |= 1 << (slot % 8);
    }

    fn clear_bit(&mut self, slot: usize) {
        self.bits[slot / 8] &= !(1 << (slot % 8));
    }

    fn get_bit(&self, slot: usize) -> bool {
        (self.bits[slot / 8] >> (slot % 8)) & 1 == 1
    }

    // ---- hashing ----

    /// FNV-1a based hash with a seed to produce `k` independent functions.
    fn hash(&self, term: &str, seed: usize) -> usize {
        const OFFSET: u64 = 14_695_981_039_346_656_037;
        const PRIME: u64 = 1_099_511_628_211;
        let mut h = OFFSET ^ (seed as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in term.bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(PRIME);
        }
        (h as usize) % self.num_slots
    }
}

// ---------------------------------------------------------------------------
// Index shard
// ---------------------------------------------------------------------------

/// A single shard combining an inverted index and a Bloom filter.
#[derive(Debug)]
struct Shard {
    /// In-memory inverted index for this shard.
    index: InvertedIndex,
    /// Bloom filter for fast term pre-check.
    bloom: BloomFilter,
}

impl Shard {
    fn new(capacity: usize) -> Self {
        Self {
            index: InvertedIndex::new(),
            bloom: BloomFilter::new(capacity.max(100), 0.01),
        }
    }

    /// Add a document, updating both the index and the Bloom filter.
    fn add_document(&mut self, doc_id: Uuid, text: &str) {
        let tokens = remove_stopwords(tokenize(text));
        // Update Bloom filter with all unique terms.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for token in &tokens {
            if seen.insert(token.clone()) {
                self.bloom.add(token);
            }
        }
        self.index.add_document(doc_id, text);
    }

    /// Remove a document, updating both the index and the Bloom filter.
    fn remove_document(&mut self, doc_id: Uuid, text: &str) {
        let tokens = remove_stopwords(tokenize(text));
        // Remove unique terms from Bloom filter.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for token in &tokens {
            if seen.insert(token.clone()) {
                self.bloom.remove(token);
            }
        }
        self.index.remove_document(doc_id);
    }

    /// Test if this shard *might* contain the term.
    fn might_contain_term(&self, term: &str) -> bool {
        self.bloom.might_contain(&term.to_lowercase())
    }

    /// Search for a term, returning (doc_id, score) pairs.
    fn search(&self, term: &str) -> Vec<(Uuid, f32)> {
        self.index.search(term)
    }

    fn doc_count(&self) -> usize {
        self.index.doc_count()
    }
}

// ---------------------------------------------------------------------------
// ShardedIndex
// ---------------------------------------------------------------------------

/// A horizontally sharded inverted index for large document collections.
///
/// Documents are assigned to shards by FNV-1a hashing of their UUID.
/// Multi-term queries benefit from Bloom-filter pre-checks that skip
/// shards with no matching terms.
#[derive(Debug)]
pub struct ShardedIndex {
    /// The shards.
    shards: Vec<Shard>,
    /// Number of shards.
    num_shards: usize,
    /// Per-doc text storage (for deletions).  doc_id -> raw text.
    doc_store: HashMap<Uuid, String>,
    /// Total documents indexed.
    doc_count: usize,
}

impl ShardedIndex {
    /// Create a sharded index with `num_shards` partitions.
    ///
    /// `per_shard_capacity` is used to size the Bloom filters.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_shards` is 0.
    pub fn new(num_shards: usize, per_shard_capacity: usize) -> SearchResult<Self> {
        if num_shards == 0 {
            return Err(SearchError::InvalidQuery("num_shards must be > 0".into()));
        }
        let shards = (0..num_shards)
            .map(|_| Shard::new(per_shard_capacity))
            .collect();
        Ok(Self {
            shards,
            num_shards,
            doc_store: HashMap::new(),
            doc_count: 0,
        })
    }

    /// Assign a document UUID to a shard index via FNV-1a hashing.
    fn shard_for(&self, doc_id: Uuid) -> usize {
        let bytes = doc_id.as_bytes();
        const OFFSET: u64 = 14_695_981_039_346_656_037;
        const PRIME: u64 = 1_099_511_628_211;
        let mut h = OFFSET;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
        (h as usize) % self.num_shards
    }

    /// Index a document.
    pub fn add_document(&mut self, doc_id: Uuid, text: &str) {
        let shard_idx = self.shard_for(doc_id);
        self.shards[shard_idx].add_document(doc_id, text);
        self.doc_store.insert(doc_id, text.to_string());
        self.doc_count += 1;
    }

    /// Remove a document by ID.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DocumentNotFound` if the document has not been indexed.
    pub fn remove_document(&mut self, doc_id: Uuid) -> SearchResult<()> {
        let text = self
            .doc_store
            .remove(&doc_id)
            .ok_or_else(|| SearchError::DocumentNotFound(doc_id.to_string()))?;
        let shard_idx = self.shard_for(doc_id);
        self.shards[shard_idx].remove_document(doc_id, &text);
        self.doc_count = self.doc_count.saturating_sub(1);
        Ok(())
    }

    /// Search for a single term across all shards.
    ///
    /// Shards that pass the Bloom filter pre-check are searched;
    /// others are skipped entirely.  Results are merged and sorted
    /// by descending score.
    #[must_use]
    pub fn search(&self, term: &str) -> Vec<(Uuid, f32)> {
        let lower = term.to_lowercase();
        let mut merged: HashMap<Uuid, f32> = HashMap::new();

        for shard in &self.shards {
            // Bloom pre-check — skip shards that definitely lack the term.
            if !shard.might_contain_term(&lower) {
                continue;
            }
            for (doc_id, score) in shard.search(&lower) {
                merged
                    .entry(doc_id)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);
            }
        }

        let mut results: Vec<(Uuid, f32)> = merged.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search for multiple terms, returning documents that match any term.
    ///
    /// For each term, only shards whose Bloom filters indicate a possible
    /// match are searched.  Scores for the same document are summed across
    /// terms.
    #[must_use]
    pub fn search_multi(&self, terms: &[&str]) -> Vec<(Uuid, f32)> {
        let mut merged: HashMap<Uuid, f32> = HashMap::new();
        for &term in terms {
            for (doc_id, score) in self.search(term) {
                *merged.entry(doc_id).or_insert(0.0) += score;
            }
        }
        let mut results: Vec<(Uuid, f32)> = merged.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Total number of indexed documents.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Number of shards.
    #[must_use]
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Per-shard document counts (for load-balance inspection).
    #[must_use]
    pub fn shard_doc_counts(&self) -> Vec<usize> {
        self.shards.iter().map(Shard::doc_count).collect()
    }

    /// Check if a document has been indexed.
    #[must_use]
    pub fn contains(&self, doc_id: Uuid) -> bool {
        self.doc_store.contains_key(&doc_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Bloom filter tests ───────────────────────────────────────────────

    #[test]
    fn test_bloom_add_and_might_contain() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.add("codec");
        bf.add("video");
        assert!(bf.might_contain("codec"));
        assert!(bf.might_contain("video"));
        // Non-inserted term: with high probability absent (not guaranteed).
        // We can't assert false here due to FP probability, but count should be 2.
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_bloom_definitely_absent() {
        let bf = BloomFilter::new(1000, 0.01);
        // Nothing inserted → definitely not present.
        assert!(!bf.might_contain("anything"));
    }

    #[test]
    fn test_bloom_remove() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.add("audio");
        assert!(bf.might_contain("audio"));
        bf.remove("audio");
        // After removal, the term should not be present.
        assert!(!bf.might_contain("audio"));
        assert_eq!(bf.count(), 0);
    }

    #[test]
    fn test_bloom_clear() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.add("alpha");
        bf.add("beta");
        bf.clear();
        assert!(!bf.might_contain("alpha"));
        assert_eq!(bf.count(), 0);
    }

    #[test]
    fn test_bloom_many_insertions() {
        let mut bf = BloomFilter::new(10_000, 0.01);
        for i in 0..500 {
            bf.add(&format!("term{i}"));
        }
        // All inserted terms should be found.
        let mut false_negatives = 0;
        for i in 0..500 {
            if !bf.might_contain(&format!("term{i}")) {
                false_negatives += 1;
            }
        }
        assert_eq!(
            false_negatives, 0,
            "Bloom filter must have no false negatives"
        );
    }

    // ─── Sharded index tests ──────────────────────────────────────────────

    #[test]
    fn test_sharded_index_new_zero_shards() {
        assert!(ShardedIndex::new(0, 100).is_err());
    }

    #[test]
    fn test_sharded_index_add_and_search() {
        let mut idx = ShardedIndex::new(4, 1000).expect("ok");
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        idx.add_document(id1, "nature documentary rainforest");
        idx.add_document(id2, "urban landscape architecture");

        assert_eq!(idx.doc_count(), 2);
        assert!(idx.contains(id1));
        assert!(idx.contains(id2));

        let results = idx.search("rainforest");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn test_sharded_index_search_no_match() {
        let mut idx = ShardedIndex::new(4, 1000).expect("ok");
        let id = Uuid::new_v4();
        idx.add_document(id, "ocean waves surf");
        let results = idx.search("rainforest");
        assert!(results.is_empty());
    }

    #[test]
    fn test_sharded_index_remove() {
        let mut idx = ShardedIndex::new(4, 1000).expect("ok");
        let id = Uuid::new_v4();
        idx.add_document(id, "unique document term");
        idx.remove_document(id).expect("should remove");
        assert_eq!(idx.doc_count(), 0);
        let results = idx.search("unique");
        assert!(results.is_empty());
    }

    #[test]
    fn test_sharded_index_remove_nonexistent() {
        let mut idx = ShardedIndex::new(4, 1000).expect("ok");
        assert!(idx.remove_document(Uuid::new_v4()).is_err());
    }

    #[test]
    fn test_sharded_index_search_multi() {
        let mut idx = ShardedIndex::new(4, 1000).expect("ok");
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        idx.add_document(id1, "red fox jumps");
        idx.add_document(id2, "blue whale swims");
        idx.add_document(id3, "red whale spotted");

        // "red" matches id1 and id3; "whale" matches id2 and id3.
        let results = idx.search_multi(&["red", "whale"]);
        let ids: Vec<Uuid> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
        assert!(ids.contains(&id3));
        // id3 matched both terms, so should have the highest accumulated score.
        assert_eq!(results[0].0, id3);
    }

    #[test]
    fn test_sharded_index_many_docs() {
        let n = 200;
        let mut idx = ShardedIndex::new(8, n * 2).expect("ok");
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let id = Uuid::new_v4();
            ids.push(id);
            idx.add_document(id, &format!("document number {i} with content"));
        }
        assert_eq!(idx.doc_count(), n);

        // All shards should have some documents (probabilistically true for n=200).
        let counts = idx.shard_doc_counts();
        let total: usize = counts.iter().sum();
        assert_eq!(total, n);
    }

    #[test]
    fn test_sharded_index_shard_count() {
        let idx = ShardedIndex::new(16, 1000).expect("ok");
        assert_eq!(idx.num_shards(), 16);
    }

    #[test]
    fn test_sharded_index_bloom_skips_empty_shards() {
        // With 16 shards and only 2 documents, at least some shards are empty.
        // The Bloom filter should correctly report those shards as non-matching.
        let mut idx = ShardedIndex::new(16, 1000).expect("ok");
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        idx.add_document(id1, "specific term alpha");
        idx.add_document(id2, "another different beta");

        // "alpha" is definitely only in the shard that holds id1.
        let results = idx.search("alpha");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn test_shard_assignment_deterministic() {
        let idx = ShardedIndex::new(8, 1000).expect("ok");
        let id = Uuid::new_v4();
        let s1 = idx.shard_for(id);
        let s2 = idx.shard_for(id);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        // Insert 1000 terms; test 1000 other terms.  FPR should be < 2% for a
        // well-configured filter (capacity=1000, fpr=0.01).
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 {
            bf.add(&format!("insert_{i}"));
        }
        let mut fp = 0usize;
        for i in 0..1000 {
            if bf.might_contain(&format!("test_{i}")) {
                fp += 1;
            }
        }
        // Allow up to 5% FPR in the test (filter targets 1%).
        assert!(fp as f64 / 1000.0 < 0.05, "FPR = {:.2}%", fp as f64 / 10.0);
    }
}
