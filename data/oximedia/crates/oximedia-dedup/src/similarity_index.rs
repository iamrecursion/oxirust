//! Similarity index: fast lookup structures for near-duplicate candidate retrieval.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A compact binary hash used for similarity comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryHash {
    /// Raw bytes of the hash.
    pub data: Vec<u8>,
    /// Length in bits.
    pub bits: usize,
}

impl BinaryHash {
    /// Create a new binary hash from raw bytes.
    #[must_use]
    pub fn new(data: Vec<u8>, bits: usize) -> Self {
        Self { data, bits }
    }

    /// Compute the Hamming distance to another hash of the same length.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Normalised similarity in [0, 1] (1.0 = identical).
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        if self.bits == 0 {
            return 1.0;
        }
        let dist = self.hamming_distance(other);
        1.0 - (dist as f64 / self.bits as f64)
    }
}

/// An entry in the similarity index.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// File path.
    pub path: PathBuf,
    /// Perceptual hash for this file.
    pub hash: BinaryHash,
    /// Optional file size in bytes.
    pub file_size: Option<u64>,
}

impl IndexEntry {
    /// Create a new index entry.
    #[must_use]
    pub fn new(path: PathBuf, hash: BinaryHash) -> Self {
        Self {
            path,
            hash,
            file_size: None,
        }
    }

    /// Attach a file size.
    #[must_use]
    pub fn with_size(mut self, size: u64) -> Self {
        self.file_size = Some(size);
        self
    }
}

/// Candidate pair returned by a similarity search.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Path to the query file.
    pub query: PathBuf,
    /// Path to the candidate file.
    pub candidate: PathBuf,
    /// Similarity score in [0, 1].
    pub score: f64,
    /// Hamming distance between hashes.
    pub distance: u32,
}

/// A linear-scan similarity index.  For small-to-medium collections.
#[derive(Debug, Default)]
pub struct LinearSimilarityIndex {
    entries: Vec<IndexEntry>,
}

impl LinearSimilarityIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an entry.
    pub fn insert(&mut self, entry: IndexEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find all entries within `max_distance` Hamming bits of `query_hash`.
    #[must_use]
    pub fn query(
        &self,
        query_path: &Path,
        query_hash: &BinaryHash,
        max_distance: u32,
    ) -> Vec<Candidate> {
        self.entries
            .iter()
            .filter(|e| e.path != query_path)
            .filter_map(|e| {
                let dist = query_hash.hamming_distance(&e.hash);
                if dist <= max_distance {
                    Some(Candidate {
                        query: query_path.to_path_buf(),
                        candidate: e.path.clone(),
                        score: query_hash.similarity(&e.hash),
                        distance: dist,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find all pairs within `max_distance` of each other.
    #[must_use]
    pub fn all_pairs(&self, max_distance: u32) -> Vec<Candidate> {
        let mut result = Vec::new();
        let n = self.entries.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.entries[i].hash.hamming_distance(&self.entries[j].hash);
                if dist <= max_distance {
                    result.push(Candidate {
                        query: self.entries[i].path.clone(),
                        candidate: self.entries[j].path.clone(),
                        score: self.entries[i].hash.similarity(&self.entries[j].hash),
                        distance: dist,
                    });
                }
            }
        }
        result
    }

    /// Remove an entry by path. Returns `true` if it was present.
    pub fn remove(&mut self, path: &Path) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.path != path);
        self.entries.len() < before
    }
}

/// Bucket-based index keyed by truncated hash prefix for faster lookup.
#[derive(Debug, Default)]
pub struct BucketSimilarityIndex {
    /// Prefix length in bytes used for bucketing.
    prefix_len: usize,
    buckets: HashMap<Vec<u8>, Vec<IndexEntry>>,
}

impl BucketSimilarityIndex {
    /// Create a new bucket index with the given prefix length.
    #[must_use]
    pub fn new(prefix_len: usize) -> Self {
        Self {
            prefix_len,
            buckets: HashMap::new(),
        }
    }

    /// Insert an entry into the appropriate bucket.
    pub fn insert(&mut self, entry: IndexEntry) {
        let prefix = entry.hash.data[..self.prefix_len.min(entry.hash.data.len())].to_vec();
        self.buckets.entry(prefix).or_default().push(entry);
    }

    /// Total number of entries across all buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    /// Returns `true` if no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Query the bucket matching the prefix of `query_hash`, returning candidates
    /// within `max_distance`.
    #[must_use]
    pub fn query(
        &self,
        query_path: &Path,
        query_hash: &BinaryHash,
        max_distance: u32,
    ) -> Vec<Candidate> {
        let prefix = query_hash.data[..self.prefix_len.min(query_hash.data.len())].to_vec();
        let bucket = match self.buckets.get(&prefix) {
            Some(b) => b,
            None => return Vec::new(),
        };
        bucket
            .iter()
            .filter(|e| e.path != query_path)
            .filter_map(|e| {
                let dist = query_hash.hamming_distance(&e.hash);
                if dist <= max_distance {
                    Some(Candidate {
                        query: query_path.to_path_buf(),
                        candidate: e.path.clone(),
                        score: query_hash.similarity(&e.hash),
                        distance: dist,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(bytes: Vec<u8>) -> BinaryHash {
        let bits = bytes.len() * 8;
        BinaryHash::new(bytes, bits)
    }

    fn pb(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn test_hamming_identical() {
        let h = make_hash(vec![0b1010_1010]);
        assert_eq!(h.hamming_distance(&h), 0);
    }

    #[test]
    fn test_hamming_one_bit() {
        let a = make_hash(vec![0b0000_0001]);
        let b = make_hash(vec![0b0000_0000]);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_similarity_identical() {
        let h = make_hash(vec![0xFF, 0xFF]);
        assert!((h.similarity(&h) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_zero_bits() {
        let h = BinaryHash::new(vec![], 0);
        assert!((h.similarity(&h) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_partial() {
        let a = make_hash(vec![0b1111_1111]); // 8 bits
        let b = make_hash(vec![0b0000_1111]); // 4 bits differ
        let sim = a.similarity(&b);
        assert!((sim - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_index_entry_with_size() {
        let h = make_hash(vec![0xAB]);
        let entry = IndexEntry::new(pb("file.mp4"), h).with_size(1024);
        assert_eq!(entry.file_size, Some(1024));
    }

    #[test]
    fn test_linear_index_insert_and_len() {
        let mut idx = LinearSimilarityIndex::new();
        assert!(idx.is_empty());
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0xFF])));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_linear_index_query_finds_near() {
        let mut idx = LinearSimilarityIndex::new();
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0b0000_0000])));
        idx.insert(IndexEntry::new(pb("b.mp4"), make_hash(vec![0b0000_0001])));
        idx.insert(IndexEntry::new(pb("c.mp4"), make_hash(vec![0b1111_1111])));
        let query_hash = make_hash(vec![0b0000_0000]);
        let results = idx.query(&pb("query.mp4"), &query_hash, 2);
        assert!(results.iter().any(|r| r.candidate == pb("a.mp4")));
        assert!(results.iter().any(|r| r.candidate == pb("b.mp4")));
        assert!(!results.iter().any(|r| r.candidate == pb("c.mp4")));
    }

    #[test]
    fn test_linear_index_excludes_self() {
        let mut idx = LinearSimilarityIndex::new();
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0x00])));
        let results = idx.query(&pb("a.mp4"), &make_hash(vec![0x00]), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_linear_index_all_pairs() {
        let mut idx = LinearSimilarityIndex::new();
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0b0000_0000])));
        idx.insert(IndexEntry::new(pb("b.mp4"), make_hash(vec![0b0000_0001])));
        let pairs = idx.all_pairs(2);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].score - 0.875).abs() < 1e-9); // 1/8 bits differ
    }

    #[test]
    fn test_linear_index_remove() {
        let mut idx = LinearSimilarityIndex::new();
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0x00])));
        assert!(idx.remove(&pb("a.mp4")));
        assert!(idx.is_empty());
        assert!(!idx.remove(&pb("missing.mp4")));
    }

    #[test]
    fn test_bucket_index_insert_and_query() {
        let mut idx = BucketSimilarityIndex::new(1);
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0x00, 0x01])));
        idx.insert(IndexEntry::new(pb("b.mp4"), make_hash(vec![0x00, 0x02])));
        assert_eq!(idx.len(), 2);
        let query = make_hash(vec![0x00, 0x03]);
        let results = idx.query(&pb("q.mp4"), &query, 4);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bucket_index_miss() {
        let mut idx = BucketSimilarityIndex::new(1);
        idx.insert(IndexEntry::new(pb("a.mp4"), make_hash(vec![0xFF, 0x00])));
        let query = make_hash(vec![0x00, 0x00]);
        let results = idx.query(&pb("q.mp4"), &query, 1);
        assert!(results.is_empty());
    }
}
