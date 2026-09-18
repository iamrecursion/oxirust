//! Music similarity search using fingerprints and Locality-Sensitive Hashing (LSH).
//!
//! # Overview
//!
//! [`SimilarityIndex`](crate::similarity_search::SimilarityIndex) stores a collection of audio fingerprints (u32 hash sequences
//! produced by [`crate::fingerprint::AcoustidEncoder`]) and answers approximate
//! nearest-neighbour queries in sub-linear time.
//!
//! ## LSH scheme
//!
//! Fingerprints are sequences of 32-bit hashes.  We use a **band-based MinHash LSH**
//! adapted for integer data:
//!
//! 1. The fingerprint vector is divided into `n_bands` contiguous bands of
//!    `band_width` hashes each.
//! 2. Each band is hashed with a fast FNV-1a accumulator to produce a single
//!    `u64` bucket key.
//! 3. A candidate is retrieved if it shares ≥1 bucket key with the query.
//! 4. Candidates are re-ranked by exact bit-similarity before returning.
//!
//! This gives a tunable recall / speed trade-off: more bands → higher recall,
//! narrower bands → faster bucket lookups.
//!
//! All storage uses `Vec<…>` and `HashMap` — no external dependencies.

use crate::fingerprint::{AcoustidFingerprint, FingerprintMatcher};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the similarity index.
#[derive(Debug, Clone)]
pub struct SimilarityIndexConfig {
    /// Number of LSH bands.  More bands → higher recall, more memory.
    pub n_bands: usize,
    /// Width (in hash words) of each band.
    pub band_width: usize,
    /// Minimum bit-similarity score to include in results (0.0–1.0).
    pub min_similarity: f32,
    /// Maximum number of results to return.
    pub top_k: usize,
}

impl Default for SimilarityIndexConfig {
    fn default() -> Self {
        Self {
            n_bands: 20,
            band_width: 4,
            min_similarity: 0.5,
            top_k: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// SimilarityIndex
// ---------------------------------------------------------------------------

/// Entry stored in the index.
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Stable identifier for this track (e.g. file path, UUID).
    id: String,
    /// The full fingerprint stored for exact re-ranking.
    fingerprint: AcoustidFingerprint,
}

/// A query result.
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    /// Identifier of the matched track.
    pub id: String,
    /// Bit-similarity score in [0, 1] (higher = more similar).
    pub score: f32,
}

/// In-memory similarity search index.
///
/// # Example
///
/// ```no_run
/// use oximedia_mir::similarity_search::{SimilarityIndex, SimilarityIndexConfig};
/// use oximedia_mir::fingerprint::AcoustidEncoder;
///
/// let mut index = SimilarityIndex::new(SimilarityIndexConfig::default());
///
/// let audio_a = vec![0.0f32; 44100];
/// let fp_a = AcoustidEncoder::compute(&audio_a, 44100);
/// index.insert("track_a".to_string(), fp_a);
///
/// let audio_q = vec![0.0f32; 44100];
/// let fp_q = AcoustidEncoder::compute(&audio_q, 44100);
/// let matches = index.search(&fp_q);
/// ```
pub struct SimilarityIndex {
    config: SimilarityIndexConfig,
    /// All indexed entries (for re-ranking).
    entries: Vec<IndexEntry>,
    /// LSH buckets: band_hash → list of entry indices.
    buckets: HashMap<u64, Vec<usize>>,
}

impl SimilarityIndex {
    /// Create a new empty index.
    #[must_use]
    pub fn new(config: SimilarityIndexConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            buckets: HashMap::new(),
        }
    }

    /// Create an index with default configuration.
    #[must_use]
    pub fn default_index() -> Self {
        Self::new(SimilarityIndexConfig::default())
    }

    /// Number of fingerprints stored in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a fingerprint into the index.
    pub fn insert(&mut self, id: String, fingerprint: AcoustidFingerprint) {
        let entry_idx = self.entries.len();
        let band_keys = self.compute_band_keys(&fingerprint);

        for key in band_keys {
            self.buckets.entry(key).or_default().push(entry_idx);
        }

        self.entries.push(IndexEntry { id, fingerprint });
    }

    /// Search for the most similar fingerprints.
    ///
    /// Returns up to `config.top_k` matches whose bit-similarity exceeds
    /// `config.min_similarity`, sorted descending by similarity score.
    #[must_use]
    pub fn search(&self, query: &AcoustidFingerprint) -> Vec<SimilarityMatch> {
        if self.entries.is_empty() || query.is_empty() {
            return Vec::new();
        }

        // ── Step 1: Collect candidate indices via LSH buckets ──────────────
        let band_keys = self.compute_band_keys(query);
        let mut candidate_indices: Vec<usize> = Vec::new();

        for key in &band_keys {
            if let Some(bucket) = self.buckets.get(key) {
                for &idx in bucket {
                    // Deduplicate with a simple linear scan (index is small)
                    if !candidate_indices.contains(&idx) {
                        candidate_indices.push(idx);
                    }
                }
            }
        }

        // Fall back to brute-force if LSH returns no candidates
        // (can happen with very short fingerprints that don't fill bands)
        if candidate_indices.is_empty() {
            candidate_indices = (0..self.entries.len()).collect();
        }

        // ── Step 2: Re-rank candidates by exact bit-similarity ──────────────
        let mut scored: Vec<(usize, f32)> = candidate_indices
            .into_iter()
            .map(|idx| {
                let sim = FingerprintMatcher::bit_similarity(query, &self.entries[idx].fingerprint);
                (idx, sim)
            })
            .filter(|&(_, sim)| sim >= self.config.min_similarity)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);

        scored
            .into_iter()
            .map(|(idx, score)| SimilarityMatch {
                id: self.entries[idx].id.clone(),
                score,
            })
            .collect()
    }

    /// Remove all entries from the index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.buckets.clear();
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// Compute per-band FNV-1a bucket keys for a fingerprint.
    fn compute_band_keys(&self, fp: &AcoustidFingerprint) -> Vec<u64> {
        let bands = self.config.n_bands;
        let bw = self.config.band_width;
        let n = fp.fingerprint.len();

        if n == 0 {
            return Vec::new();
        }

        let mut keys = Vec::with_capacity(bands);

        for band in 0..bands {
            let start = (band * bw) % n; // wrap around if fingerprint is short
            let mut hash = 0xcbf29ce484222325_u64; // FNV-1a offset basis (64-bit)
            const PRIME: u64 = 0x100000001b3;

            // Hash `bw` words starting at `start` (wrapping)
            for i in 0..bw {
                let idx = (start + i) % n;
                let word = fp.fingerprint[idx];
                // Mix each byte of the 32-bit word into the FNV accumulator
                for byte_shift in [0_u32, 8, 16, 24] {
                    let byte = ((word >> byte_shift) & 0xFF) as u64;
                    hash ^= byte;
                    hash = hash.wrapping_mul(PRIME);
                }
            }

            // Include the band index in the key so bands don't collide with each other
            hash ^= band as u64;
            hash = hash.wrapping_mul(PRIME);

            keys.push(hash);
        }

        keys
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::AcoustidEncoder;
    use std::f32::consts::TAU;

    fn sine_fingerprint(freq: f32, dur_secs: f32) -> AcoustidFingerprint {
        let sr = 8000_u32;
        let n = (sr as f32 * dur_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (TAU * freq * i as f32 / sr as f32).sin())
            .collect();
        AcoustidEncoder::compute_with_frame_size(&samples, sr, 256)
    }

    #[test]
    fn test_index_empty() {
        let index = SimilarityIndex::default_index();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_insert_and_len() {
        let mut index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 1.0);
        index.insert("a".to_string(), fp);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_self_similarity() {
        let mut index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 1.0);
        index.insert("a440".to_string(), fp.clone());
        let results = index.search(&fp);
        assert!(
            !results.is_empty(),
            "Self-search must return at least one result"
        );
        assert_eq!(results[0].id, "a440");
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "Self-similarity should be ~1.0"
        );
    }

    #[test]
    fn test_search_different_signals() {
        let mut index = SimilarityIndex::new(SimilarityIndexConfig {
            min_similarity: 0.0,
            ..SimilarityIndexConfig::default()
        });
        let fp_a = sine_fingerprint(440.0, 1.0);
        let fp_b = sine_fingerprint(523.25, 1.0); // C5

        index.insert("A4".to_string(), fp_a.clone());
        index.insert("C5".to_string(), fp_b.clone());

        let results = index.search(&fp_a);
        assert!(!results.is_empty());
        // Best match should be A4 itself
        assert_eq!(results[0].id, "A4");
    }

    #[test]
    fn test_search_empty_query() {
        let mut index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 1.0);
        index.insert("a".to_string(), fp);
        let empty = AcoustidFingerprint {
            fingerprint: vec![],
            duration_secs: 0.0,
        };
        let results = index.search(&empty);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_index() {
        let index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 1.0);
        let results = index.search(&fp);
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear_index() {
        let mut index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 1.0);
        index.insert("a".to_string(), fp);
        assert_eq!(index.len(), 1);
        index.clear();
        assert!(index.is_empty());
    }

    #[test]
    fn test_top_k_respected() {
        let config = SimilarityIndexConfig {
            top_k: 2,
            min_similarity: 0.0,
            ..SimilarityIndexConfig::default()
        };
        let mut index = SimilarityIndex::new(config);
        for i in 0..5 {
            let fp = sine_fingerprint(200.0 + i as f32 * 50.0, 0.5);
            index.insert(format!("track_{i}"), fp);
        }
        let query = sine_fingerprint(300.0, 0.5);
        let results = index.search(&query);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_band_keys_not_empty() {
        let index = SimilarityIndex::default_index();
        let fp = sine_fingerprint(440.0, 0.5);
        let keys = index.compute_band_keys(&fp);
        assert!(!keys.is_empty());
        assert_eq!(keys.len(), index.config.n_bands);
    }

    #[test]
    fn test_results_sorted_descending() {
        let config = SimilarityIndexConfig {
            min_similarity: 0.0,
            ..SimilarityIndexConfig::default()
        };
        let mut index = SimilarityIndex::new(config);
        let fp_ref = sine_fingerprint(440.0, 1.0);
        index.insert("ref".to_string(), fp_ref.clone());
        let fp_other = sine_fingerprint(1000.0, 1.0);
        index.insert("other".to_string(), fp_other);

        let results = index.search(&fp_ref);
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Results must be sorted descending: {:.4} < {:.4}",
                w[0].score,
                w[1].score
            );
        }
    }
}
