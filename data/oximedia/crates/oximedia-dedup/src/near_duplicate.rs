//! Near-duplicate detection using locality-sensitive hashing (LSH).
//!
//! This module provides:
//! - `SimHash`: SimHash fingerprint for text/feature vectors
//! - `MinHash`: MinHash with K independent hash functions
//! - `LshIndex`: LSH bucket index for approximate nearest-neighbour search

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// SimHash
// ---------------------------------------------------------------------------

/// SimHash fingerprint for text or feature vectors.
///
/// Computed by weighted XOR accumulation with threshold by bit position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimHash(pub u64);

impl SimHash {
    /// Compute a SimHash from a slice of feature values.
    ///
    /// Each feature votes for/against each of the 64 bit positions.
    /// If the weighted vote for a position is positive, that bit is 1.
    #[must_use]
    pub fn compute(features: &[u64]) -> Self {
        let mut counts = [0i64; 64];

        for &feature in features {
            for bit in 0..64u32 {
                if (feature >> bit) & 1 == 1 {
                    counts[bit as usize] += 1;
                } else {
                    counts[bit as usize] -= 1;
                }
            }
        }

        let mut hash = 0u64;
        for (bit, &count) in counts.iter().enumerate() {
            if count > 0 {
                hash |= 1u64 << bit;
            }
        }

        Self(hash)
    }

    /// Compute Hamming distance to another SimHash.
    #[must_use]
    pub fn hamming_distance(self, other: &Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Return the raw 64-bit value.
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Similarity in [0.0, 1.0].
    #[must_use]
    pub fn similarity(self, other: &Self) -> f32 {
        1.0 - self.hamming_distance(other) as f32 / 64.0
    }
}

// ---------------------------------------------------------------------------
// MinHash
// ---------------------------------------------------------------------------

/// MinHash with K independent hash functions.
///
/// Useful for estimating Jaccard similarity between sets of shingles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHash {
    /// Number of hash functions (k).
    pub k: usize,
    /// MinHash signatures: k minimum hash values.
    pub signatures: Vec<u64>,
}

/// FNV-1a-based hash mixing: combines `seed` with `value`.
fn fnv_hash(seed: u64, value: u64) -> u64 {
    let mut h: u64 = seed ^ 0xcbf2_9ce4_8422_2325;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h ^= value;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);
    h
}

impl MinHash {
    /// Compute MinHash signatures for a set of shingles using `k` hash functions.
    ///
    /// Each hash function is simulated by FNV mixing with a distinct seed.
    #[must_use]
    pub fn compute(shingles: &[u64], k: usize) -> Self {
        if shingles.is_empty() || k == 0 {
            return Self {
                k,
                signatures: vec![u64::MAX; k],
            };
        }

        let mut signatures = vec![u64::MAX; k];

        for &shingle in shingles {
            for (i, sig) in signatures.iter_mut().enumerate() {
                let h = fnv_hash(i as u64, shingle);
                if h < *sig {
                    *sig = h;
                }
            }
        }

        Self { k, signatures }
    }

    /// Estimate Jaccard similarity as the fraction of matching min-hashes.
    #[must_use]
    pub fn jaccard_estimate(&self, other: &Self) -> f32 {
        let len = self.signatures.len().min(other.signatures.len());
        if len == 0 {
            return 0.0;
        }

        let matches = self
            .signatures
            .iter()
            .zip(other.signatures.iter())
            .filter(|(a, b)| a == b)
            .count();

        matches as f32 / len as f32
    }
}

// ---------------------------------------------------------------------------
// LSH Index
// ---------------------------------------------------------------------------

/// A single LSH bucket.
#[derive(Debug, Clone)]
pub struct LshBucket {
    /// The bucket hash (band signature).
    pub hash: u64,
    /// IDs of items in this bucket.
    pub item_ids: Vec<u64>,
}

/// LSH index for approximate nearest-neighbour search over MinHash signatures.
///
/// Uses a band-based approach: signatures are divided into `b` bands of `r` rows.
/// Items sharing the same band hash are candidates.
pub struct LshIndex {
    /// Number of bands.
    bands: usize,
    /// Number of rows per band.
    rows_per_band: usize,
    /// Stored (id, minhash) pairs.
    items: Vec<(u64, MinHash)>,
}

impl LshIndex {
    /// Create a new LSH index.
    ///
    /// `bands` × `rows_per_band` should equal the MinHash `k` value.
    #[must_use]
    pub fn new(bands: usize, rows_per_band: usize) -> Self {
        Self {
            bands,
            rows_per_band,
            items: Vec::new(),
        }
    }

    /// Add an item with given `id` and its `minhash` signature.
    pub fn add(&mut self, id: u64, minhash: &MinHash) {
        self.items.push((id, minhash.clone()));
    }

    /// Find candidate IDs whose estimated Jaccard similarity to `query` is at least `threshold`.
    #[must_use]
    pub fn find_candidates(&self, query: &MinHash, threshold: f32) -> Vec<u64> {
        let mut candidates = Vec::new();

        for &(id, ref minhash) in &self.items {
            // Check band collision (any band matches → candidate)
            let mut band_collision = false;
            for band in 0..self.bands {
                let start = band * self.rows_per_band;
                let end = (start + self.rows_per_band).min(minhash.signatures.len());
                let qend = (start + self.rows_per_band).min(query.signatures.len());

                if start >= minhash.signatures.len() || start >= query.signatures.len() {
                    break;
                }

                let band_hash_a = hash_band(&minhash.signatures[start..end]);
                let band_hash_b = hash_band(&query.signatures[start..qend]);

                if band_hash_a == band_hash_b {
                    band_collision = true;
                    break;
                }
            }

            if band_collision {
                // Verify with full Jaccard estimate
                let sim = minhash.jaccard_estimate(query);
                if sim >= threshold {
                    candidates.push(id);
                }
            }
        }

        candidates
    }

    /// Return the number of indexed items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Compute a hash for a band of signature rows.
fn hash_band(rows: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in rows {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SimHash tests ---

    #[test]
    fn test_simhash_empty() {
        let h = SimHash::compute(&[]);
        // No features: all counts 0 → all bits 0
        assert_eq!(h.0, 0);
    }

    #[test]
    fn test_simhash_all_ones() {
        // Feature with all bits set → all counts positive → all bits 1
        let h = SimHash::compute(&[u64::MAX]);
        assert_eq!(h.0, u64::MAX);
    }

    #[test]
    fn test_simhash_deterministic() {
        let features = vec![42u64, 123, 9999, 0xDEAD_BEEF];
        let h1 = SimHash::compute(&features);
        let h2 = SimHash::compute(&features);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_simhash_hamming_same() {
        let h = SimHash::compute(&[42, 100, 200]);
        assert_eq!(h.hamming_distance(&h), 0);
    }

    #[test]
    fn test_simhash_hamming_range() {
        let h1 = SimHash::compute(&[1, 2, 3]);
        let h2 = SimHash::compute(&[4, 5, 6]);
        let dist = h1.hamming_distance(&h2);
        assert!(dist <= 64);
    }

    #[test]
    fn test_simhash_similarity_range() {
        let h1 = SimHash::compute(&[1, 2, 3]);
        let h2 = SimHash::compute(&[4, 5, 6]);
        let sim = h1.similarity(&h2);
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_simhash_identical_features_equal() {
        let features = vec![11u64, 22, 33, 44, 55];
        let h1 = SimHash::compute(&features);
        let h2 = SimHash::compute(&features);
        assert_eq!(h1.similarity(&h2), 1.0);
    }

    // --- MinHash tests ---

    #[test]
    fn test_minhash_empty_shingles() {
        let mh = MinHash::compute(&[], 10);
        assert_eq!(mh.signatures.len(), 10);
        assert!(mh.signatures.iter().all(|&s| s == u64::MAX));
    }

    #[test]
    fn test_minhash_identical_sets_max_similarity() {
        let shingles = vec![1u64, 2, 3, 4, 5];
        let mh1 = MinHash::compute(&shingles, 64);
        let mh2 = MinHash::compute(&shingles, 64);
        assert_eq!(mh1.jaccard_estimate(&mh2), 1.0);
    }

    #[test]
    fn test_minhash_disjoint_sets_low_similarity() {
        let set_a: Vec<u64> = (0..50).collect();
        let set_b: Vec<u64> = (1000..1050).collect();
        let mh1 = MinHash::compute(&set_a, 128);
        let mh2 = MinHash::compute(&set_b, 128);
        // Disjoint sets → estimate near 0
        assert!(mh1.jaccard_estimate(&mh2) < 0.1);
    }

    #[test]
    fn test_minhash_partial_overlap() {
        // 50% overlap
        let set_a: Vec<u64> = (0..100).collect();
        let set_b: Vec<u64> = (50..150).collect();
        let mh1 = MinHash::compute(&set_a, 256);
        let mh2 = MinHash::compute(&set_b, 256);
        let sim = mh1.jaccard_estimate(&mh2);
        // True Jaccard = 50/150 ≈ 0.33; estimate may vary
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_minhash_deterministic() {
        let shingles = vec![7u64, 8, 9, 10];
        let mh1 = MinHash::compute(&shingles, 32);
        let mh2 = MinHash::compute(&shingles, 32);
        assert_eq!(mh1, mh2);
    }

    // --- LshIndex tests ---

    #[test]
    fn test_lsh_empty_index() {
        let index = LshIndex::new(4, 4);
        let query = MinHash::compute(&[1, 2, 3], 16);
        let candidates = index.find_candidates(&query, 0.5);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_lsh_identical_item_found() {
        let mut index = LshIndex::new(4, 4);
        let shingles = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mh = MinHash::compute(&shingles, 16);
        index.add(42, &mh);

        let candidates = index.find_candidates(&mh, 0.9);
        assert!(candidates.contains(&42));
    }

    #[test]
    fn test_lsh_different_item_not_found_high_threshold() {
        let mut index = LshIndex::new(4, 4);
        let shingles_a: Vec<u64> = (0..20).collect();
        let shingles_b: Vec<u64> = (1000..1020).collect();
        let mh_a = MinHash::compute(&shingles_a, 16);
        let mh_b = MinHash::compute(&shingles_b, 16);
        index.add(1, &mh_a);

        let candidates = index.find_candidates(&mh_b, 0.9);
        assert!(!candidates.contains(&1));
    }

    #[test]
    fn test_lsh_len() {
        let mut index = LshIndex::new(4, 4);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        let mh = MinHash::compute(&[1, 2, 3], 16);
        index.add(1, &mh);
        index.add(2, &mh);
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_lsh_multiple_items() {
        let mut index = LshIndex::new(4, 4);
        let base: Vec<u64> = (0..30).collect();
        for i in 0..5u64 {
            let mh = MinHash::compute(&base, 16);
            index.add(i, &mh);
        }

        let query = MinHash::compute(&base, 16);
        let candidates = index.find_candidates(&query, 0.9);
        // All 5 should be found (same shingles)
        assert_eq!(candidates.len(), 5);
    }
}
