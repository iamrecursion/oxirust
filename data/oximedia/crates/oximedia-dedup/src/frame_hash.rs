//! Frame-level hash types for fast perceptual deduplication.
//!
//! Provides `HashAlgorithm`, `FrameHash`, and `FrameHashStore` for
//! inserting and finding similar frames by Hamming distance.

#![allow(dead_code)]

use std::collections::HashMap;

/// Hashing algorithm used to produce a frame hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    /// 64-bit difference hash (dHash).
    DHash64,
    /// 64-bit perceptual hash (pHash).
    PHash64,
    /// 128-bit average hash.
    AHash128,
    /// 256-bit wavelet hash.
    WHash256,
}

impl HashAlgorithm {
    /// Return the bit length of the hash produced by this algorithm.
    #[must_use]
    pub const fn bit_length(self) -> u32 {
        match self {
            Self::DHash64 | Self::PHash64 => 64,
            Self::AHash128 => 128,
            Self::WHash256 => 256,
        }
    }

    /// Whether this algorithm is considered a perceptual hash.
    #[must_use]
    pub const fn is_perceptual(self) -> bool {
        matches!(self, Self::PHash64 | Self::WHash256)
    }

    /// Return a human-readable name for the algorithm.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DHash64 => "dHash-64",
            Self::PHash64 => "pHash-64",
            Self::AHash128 => "aHash-128",
            Self::WHash256 => "wHash-256",
        }
    }
}

/// A compact hash value representing one video frame.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrameHash {
    /// Algorithm that produced this hash.
    pub algorithm: HashAlgorithm,
    /// Raw hash bits stored as a byte vector.
    pub bits: Vec<u8>,
    /// Frame index within the source media.
    pub frame_index: u64,
}

impl FrameHash {
    /// Construct a new `FrameHash`.
    #[must_use]
    pub fn new(algorithm: HashAlgorithm, bits: Vec<u8>, frame_index: u64) -> Self {
        Self {
            algorithm,
            bits,
            frame_index,
        }
    }

    /// Compute the Hamming distance between `self` and `other`.
    ///
    /// Returns `None` if the two hashes have different algorithms or lengths.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> Option<u32> {
        if self.algorithm != other.algorithm || self.bits.len() != other.bits.len() {
            return None;
        }
        let dist = self
            .bits
            .iter()
            .zip(&other.bits)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        Some(dist)
    }

    /// Return `true` if the two hashes are within `max_distance` bits of each other.
    #[must_use]
    pub fn is_similar(&self, other: &Self, max_distance: u32) -> bool {
        self.hamming_distance(other)
            .map_or(false, |d| d <= max_distance)
    }

    /// Return the number of bytes in the hash.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bits.len()
    }
}

/// In-memory store of `FrameHash` values supporting similarity search.
#[derive(Debug, Default)]
pub struct FrameHashStore {
    entries: HashMap<u64, FrameHash>,
}

impl FrameHashStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a `FrameHash` into the store, keyed by its `frame_index`.
    pub fn insert(&mut self, hash: FrameHash) {
        self.entries.insert(hash.frame_index, hash);
    }

    /// Find all hashes within `max_distance` of `query`.
    ///
    /// Returns a `Vec` of `(frame_index, distance)` pairs sorted by distance.
    #[must_use]
    pub fn find_similar(&self, query: &FrameHash, max_distance: u32) -> Vec<(u64, u32)> {
        let mut results: Vec<(u64, u32)> = self
            .entries
            .values()
            .filter(|h| h.frame_index != query.frame_index)
            .filter_map(|h| {
                query
                    .hamming_distance(h)
                    .filter(|&d| d <= max_distance)
                    .map(|d| (h.frame_index, d))
            })
            .collect();
        results.sort_by_key(|&(_, d)| d);
        results
    }

    /// Return the total number of hashes in the store.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Remove the hash for a given frame index, returning it if present.
    pub fn remove(&mut self, frame_index: u64) -> Option<FrameHash> {
        self.entries.remove(&frame_index)
    }

    /// Check whether the store contains a hash for `frame_index`.
    #[must_use]
    pub fn contains(&self, frame_index: u64) -> bool {
        self.entries.contains_key(&frame_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(algo: HashAlgorithm, bits: Vec<u8>, idx: u64) -> FrameHash {
        FrameHash::new(algo, bits, idx)
    }

    #[test]
    fn test_hash_algorithm_bit_length() {
        assert_eq!(HashAlgorithm::DHash64.bit_length(), 64);
        assert_eq!(HashAlgorithm::PHash64.bit_length(), 64);
        assert_eq!(HashAlgorithm::AHash128.bit_length(), 128);
        assert_eq!(HashAlgorithm::WHash256.bit_length(), 256);
    }

    #[test]
    fn test_hash_algorithm_is_perceptual() {
        assert!(!HashAlgorithm::DHash64.is_perceptual());
        assert!(HashAlgorithm::PHash64.is_perceptual());
        assert!(!HashAlgorithm::AHash128.is_perceptual());
        assert!(HashAlgorithm::WHash256.is_perceptual());
    }

    #[test]
    fn test_hash_algorithm_name() {
        assert_eq!(HashAlgorithm::DHash64.name(), "dHash-64");
        assert_eq!(HashAlgorithm::PHash64.name(), "pHash-64");
    }

    #[test]
    fn test_hamming_distance_identical() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0xAA; 8], 0);
        let h2 = make_hash(HashAlgorithm::DHash64, vec![0xAA; 8], 1);
        assert_eq!(h1.hamming_distance(&h2), Some(0));
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0x00; 8], 0);
        let h2 = make_hash(HashAlgorithm::DHash64, vec![0xFF; 8], 1);
        assert_eq!(h1.hamming_distance(&h2), Some(64));
    }

    #[test]
    fn test_hamming_distance_one_bit() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0b0000_0000; 8], 0);
        let h2 = make_hash(
            HashAlgorithm::DHash64,
            vec![0b0000_0001, 0, 0, 0, 0, 0, 0, 0],
            1,
        );
        assert_eq!(h1.hamming_distance(&h2), Some(1));
    }

    #[test]
    fn test_hamming_distance_algorithm_mismatch() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0xAA; 8], 0);
        let h2 = make_hash(HashAlgorithm::PHash64, vec![0xAA; 8], 1);
        assert_eq!(h1.hamming_distance(&h2), None);
    }

    #[test]
    fn test_hamming_distance_length_mismatch() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0xAA; 8], 0);
        let h2 = make_hash(HashAlgorithm::DHash64, vec![0xAA; 4], 1);
        assert_eq!(h1.hamming_distance(&h2), None);
    }

    #[test]
    fn test_is_similar_true() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0b1111_1111; 8], 0);
        let h2 = make_hash(
            HashAlgorithm::DHash64,
            vec![0b1111_1110, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            1,
        );
        assert!(h1.is_similar(&h2, 1));
    }

    #[test]
    fn test_is_similar_false() {
        let h1 = make_hash(HashAlgorithm::DHash64, vec![0x00; 8], 0);
        let h2 = make_hash(HashAlgorithm::DHash64, vec![0xFF; 8], 1);
        assert!(!h1.is_similar(&h2, 10));
    }

    #[test]
    fn test_store_insert_and_count() {
        let mut store = FrameHashStore::new();
        assert_eq!(store.count(), 0);
        store.insert(make_hash(HashAlgorithm::DHash64, vec![0xAA; 8], 0));
        store.insert(make_hash(HashAlgorithm::DHash64, vec![0xBB; 8], 1));
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_store_find_similar() {
        let mut store = FrameHashStore::new();
        let query = make_hash(HashAlgorithm::DHash64, vec![0b0000_0000; 8], 99);
        // Very close
        store.insert(make_hash(
            HashAlgorithm::DHash64,
            vec![0b0000_0001, 0, 0, 0, 0, 0, 0, 0],
            0,
        ));
        // Far
        store.insert(make_hash(HashAlgorithm::DHash64, vec![0xFF; 8], 1));
        let results = store.find_similar(&query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[0].1, 1);
    }

    #[test]
    fn test_store_contains_and_remove() {
        let mut store = FrameHashStore::new();
        store.insert(make_hash(HashAlgorithm::PHash64, vec![0x12; 8], 42));
        assert!(store.contains(42));
        let removed = store.remove(42);
        assert!(removed.is_some());
        assert!(!store.contains(42));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_store_find_similar_sorted_by_distance() {
        let mut store = FrameHashStore::new();
        let query = make_hash(HashAlgorithm::DHash64, vec![0x00; 8], 99);
        // distance 2
        store.insert(make_hash(
            HashAlgorithm::DHash64,
            vec![0b0000_0011, 0, 0, 0, 0, 0, 0, 0],
            1,
        ));
        // distance 1
        store.insert(make_hash(
            HashAlgorithm::DHash64,
            vec![0b0000_0001, 0, 0, 0, 0, 0, 0, 0],
            2,
        ));
        let results = store.find_similar(&query, 10);
        assert_eq!(results.len(), 2);
        assert!(
            results[0].1 <= results[1].1,
            "should be sorted ascending by distance"
        );
    }

    #[test]
    fn test_frame_hash_byte_len() {
        let h = make_hash(HashAlgorithm::AHash128, vec![0u8; 16], 0);
        assert_eq!(h.byte_len(), 16);
    }
}
