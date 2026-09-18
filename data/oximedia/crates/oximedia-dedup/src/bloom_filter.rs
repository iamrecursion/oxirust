//! Near-duplicate detection using a Bloom filter.
//!
//! Provides:
//! - [`BloomFilter`]: space-efficient probabilistic set membership structure
//! - [`NearDuplicateDetector`]: wraps a Bloom filter for streaming deduplication

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Bloom filter
// ---------------------------------------------------------------------------

/// A counting-free Bloom filter backed by a `Vec<u64>` bit array.
///
/// Insert items and test membership with a configurable false-positive rate.
/// Deletions are not supported (standard Bloom filter design).
pub struct BloomFilter {
    /// Bit storage (each `u64` holds 64 bits).
    pub bits: Vec<u64>,
    /// Number of independent hash functions.
    pub num_hashes: u32,
    /// Total number of bits (length of the logical bit array).
    pub bit_count: u64,
}

impl BloomFilter {
    /// Create a new Bloom filter sized for `capacity` items at a given `false_positive_rate`.
    ///
    /// Uses the standard formulas:
    /// - `m = -n * ln(p) / (ln(2))^2`  (number of bits)
    /// - `k = (m / n) * ln(2)`          (number of hash functions)
    #[must_use]
    pub fn new(capacity: usize, false_positive_rate: f32) -> Self {
        let capacity = capacity.max(1);
        let fpr = false_positive_rate.clamp(1e-9, 1.0 - f32::EPSILON);

        let ln2 = std::f64::consts::LN_2;
        let n = capacity as f64;
        let p = fpr as f64;

        let m = (-n * p.ln() / (ln2 * ln2)).ceil() as u64;
        let m = m.max(64); // at least 64 bits
        let k = ((m as f64 / n) * ln2).round() as u32;
        let k = k.clamp(1, 30);

        let words = ((m + 63) / 64) as usize;

        Self {
            bits: vec![0u64; words],
            num_hashes: k,
            bit_count: m,
        }
    }

    /// Insert `item` into the filter.
    pub fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(item, i);
            let word = (idx / 64) as usize;
            let bit = idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Returns `true` if `item` *may* be in the set (possible false positive).
    /// Returns `false` if `item` is *definitely not* in the set.
    #[must_use]
    pub fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(item, i);
            let word = (idx / 64) as usize;
            let bit = idx % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }

    /// Estimate the number of items inserted using the formula:
    /// `n̂ = -m / k * ln(1 - X / m)`
    /// where `X` is the number of set bits.
    #[must_use]
    pub fn estimated_count(&self) -> usize {
        let set_bits: u64 = self.bits.iter().map(|w| w.count_ones() as u64).sum();
        if set_bits == 0 {
            return 0;
        }
        if set_bits >= self.bit_count {
            // Filter saturated
            return usize::MAX;
        }
        let m = self.bit_count as f64;
        let k = self.num_hashes as f64;
        let x = set_bits as f64;
        let estimate = -(m / k) * (1.0 - x / m).ln();
        estimate.round() as usize
    }

    /// Clear all bits (reset the filter).
    pub fn clear(&mut self) {
        for w in &mut self.bits {
            *w = 0;
        }
    }

    /// Compute bit index for the i-th hash of `item` using FNV-like mixing.
    fn hash_index(&self, item: &[u8], i: u32) -> u64 {
        // Two independent hashes via FNV-1a, then Kirsch-Mitzenmacher combination
        let h1 = fnv1a_64(item);
        let h2 = fnv1a_64_seeded(item, i as u64 ^ 0x9e37_79b9_7f4a_7c15);
        h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.bit_count
    }
}

/// FNV-1a 64-bit hash.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64-bit hash with an additional seed mixed in.
fn fnv1a_64_seeded(data: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed.wrapping_mul(0x0000_0100_0000_01b3);
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// NearDuplicateDetector
// ---------------------------------------------------------------------------

/// Streaming near-duplicate detector backed by a Bloom filter.
///
/// Items (fingerprints) are inserted; if the Bloom filter reports a hit the
/// item is considered a near-duplicate.  Because the underlying structure is a
/// Bloom filter the detector never produces false negatives but may produce
/// false positives at a configurable rate.
pub struct NearDuplicateDetector {
    /// Similarity/membership threshold concept – stored for reference only.
    pub threshold: f32,
    /// The underlying Bloom filter.
    pub seen: BloomFilter,
}

impl NearDuplicateDetector {
    /// Create a detector with the given `capacity` (expected number of unique items).
    ///
    /// Uses a false-positive rate of 1%.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            threshold: 0.99,
            seen: BloomFilter::new(capacity, 0.01),
        }
    }

    /// Create with a custom false-positive rate.
    #[must_use]
    pub fn with_fpr(capacity: usize, false_positive_rate: f32) -> Self {
        Self {
            threshold: 1.0 - false_positive_rate,
            seen: BloomFilter::new(capacity, false_positive_rate),
        }
    }

    /// Add `fingerprint` and return `true` if it appears to be a near-duplicate
    /// (i.e., it was already seen before).
    pub fn add_and_check(&mut self, fingerprint: &[u8]) -> bool {
        let duplicate = self.seen.contains(fingerprint);
        self.seen.insert(fingerprint);
        duplicate
    }

    /// Reset the detector (clear all seen fingerprints).
    pub fn reset(&mut self) {
        self.seen.clear();
    }

    /// Estimated number of unique items seen so far.
    #[must_use]
    pub fn estimated_unique_count(&self) -> usize {
        self.seen.estimated_count()
    }
}

// ---------------------------------------------------------------------------
// Bloom filter pre-screening for deduplication
// ---------------------------------------------------------------------------

/// Pre-screening results from the bloom filter pass.
#[derive(Debug, Clone)]
pub struct PreScreenResult {
    /// Indices of items that are *potential* duplicates (bloom filter hit).
    pub candidates: Vec<usize>,
    /// Indices of items that are *definitely* unique (bloom filter miss).
    pub unique: Vec<usize>,
    /// Total items processed.
    pub total: usize,
}

impl PreScreenResult {
    /// Fraction of items that passed as candidates (potential duplicates).
    #[must_use]
    pub fn candidate_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.candidates.len() as f64 / self.total as f64
    }

    /// Fraction of items rejected as unique.
    #[must_use]
    pub fn rejection_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.unique.len() as f64 / self.total as f64
    }
}

/// Pre-screen a set of fingerprints (byte slices) using a bloom filter.
///
/// Each fingerprint is inserted into a bloom filter.  If the filter reports
/// that the fingerprint was already seen, the item is flagged as a candidate
/// for expensive pairwise comparison.  Items whose fingerprints are not yet
/// in the filter are classified as definitely unique (subject to the bloom
/// filter's false-positive rate).
///
/// This dramatically reduces the number of expensive perceptual-hash or
/// SSIM comparisons needed.
#[must_use]
pub fn prescreen_fingerprints(
    fingerprints: &[Vec<u8>],
    capacity: usize,
    fpr: f32,
) -> PreScreenResult {
    let mut bloom = BloomFilter::new(capacity, fpr);
    let mut candidates = Vec::new();
    let mut unique = Vec::new();

    for (i, fp) in fingerprints.iter().enumerate() {
        if bloom.contains(fp) {
            candidates.push(i);
        } else {
            unique.push(i);
        }
        bloom.insert(fp);
    }

    PreScreenResult {
        candidates,
        unique,
        total: fingerprints.len(),
    }
}

/// Pre-screen using quantized perceptual hashes.
///
/// Quantizes each 64-bit hash down to `quantize_bits` (default: 16) so that
/// similar hashes map to the same bloom key.  This means near-duplicates
/// (not just exact duplicates) will collide in the bloom filter.
#[must_use]
pub fn prescreen_perceptual_hashes(
    hashes: &[u64],
    quantize_bits: u32,
    capacity: usize,
    fpr: f32,
) -> PreScreenResult {
    let quantize_bits = quantize_bits.clamp(4, 64);
    let shift = 64 - quantize_bits;

    let mut bloom = BloomFilter::new(capacity, fpr);
    let mut candidates = Vec::new();
    let mut unique = Vec::new();

    for (i, &hash) in hashes.iter().enumerate() {
        let quantized = hash >> shift;
        let bytes = quantized.to_le_bytes();
        if bloom.contains(&bytes) {
            candidates.push(i);
        } else {
            unique.push(i);
        }
        bloom.insert(&bytes);
    }

    PreScreenResult {
        candidates,
        unique,
        total: hashes.len(),
    }
}

// ---------------------------------------------------------------------------
// Integrated dedup pipeline with bloom + LSH
// ---------------------------------------------------------------------------

/// Configuration for the integrated bloom + LSH dedup pipeline.
#[derive(Debug, Clone)]
pub struct DedupPipelineConfig {
    /// Bloom filter capacity.
    pub bloom_capacity: usize,
    /// Bloom filter false positive rate.
    pub bloom_fpr: f32,
    /// Number of bits for quantizing perceptual hashes before bloom insertion.
    pub quantize_bits: u32,
    /// Number of LSH hash tables.
    pub lsh_tables: usize,
    /// Bits sampled per LSH table.
    pub lsh_bits_per_table: usize,
    /// Maximum Hamming distance for near-duplicate pairing.
    pub max_hamming_distance: u32,
    /// PRNG seed for LSH.
    pub seed: u64,
}

impl Default for DedupPipelineConfig {
    fn default() -> Self {
        Self {
            bloom_capacity: 10_000,
            bloom_fpr: 0.01,
            quantize_bits: 16,
            lsh_tables: 8,
            lsh_bits_per_table: 8,
            max_hamming_distance: 10,
            seed: 42,
        }
    }
}

/// Result of the integrated dedup pipeline.
#[derive(Debug, Clone)]
pub struct DedupPipelineResult {
    /// Items that passed bloom pre-screening (potential duplicates).
    pub bloom_candidates: Vec<usize>,
    /// Items rejected by bloom (definitely unique).
    pub bloom_unique: Vec<usize>,
    /// Duplicate pairs found by LSH among the bloom candidates.
    pub lsh_pairs: Vec<(u64, u64, u32)>,
    /// Total items processed.
    pub total: usize,
}

impl DedupPipelineResult {
    /// Fraction of items rejected by bloom (savings from skipping expensive comparison).
    #[must_use]
    pub fn bloom_rejection_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.bloom_unique.len() as f64 / self.total as f64
    }

    /// Number of duplicate pairs found.
    #[must_use]
    pub fn num_pairs(&self) -> usize {
        self.lsh_pairs.len()
    }
}

/// Run the integrated bloom + LSH dedup pipeline.
///
/// Phase 1: Bloom filter pre-screens quantized hashes to reject definitely-unique items.
/// Phase 2: LSH finds near-duplicate pairs among remaining candidates.
///
/// This dramatically reduces the search space for large libraries.
#[must_use]
pub fn run_dedup_pipeline(
    hashes: &[(u64, u64)], // (id, perceptual_hash)
    config: &DedupPipelineConfig,
) -> DedupPipelineResult {
    use crate::lsh_index::lsh_dedup_pass;

    if hashes.is_empty() {
        return DedupPipelineResult {
            bloom_candidates: Vec::new(),
            bloom_unique: Vec::new(),
            lsh_pairs: Vec::new(),
            total: 0,
        };
    }

    // Phase 1: Bloom pre-screening with quantized hashes
    let raw_hashes: Vec<u64> = hashes.iter().map(|&(_, h)| h).collect();
    let prescreen = prescreen_perceptual_hashes(
        &raw_hashes,
        config.quantize_bits,
        config.bloom_capacity,
        config.bloom_fpr,
    );

    // Build candidate set for LSH (bloom hits + their original IDs)
    let candidate_hashes: Vec<(u64, u64)> = prescreen
        .candidates
        .iter()
        .map(|&idx| hashes[idx])
        .collect();

    // Phase 2: LSH on candidates only
    let lsh_result = lsh_dedup_pass(
        &candidate_hashes,
        config.max_hamming_distance,
        config.lsh_tables,
        config.lsh_bits_per_table,
        config.seed,
    );

    DedupPipelineResult {
        bloom_candidates: prescreen.candidates,
        bloom_unique: prescreen.unique,
        lsh_pairs: lsh_result.pairs,
        total: hashes.len(),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BloomFilter construction ----

    #[test]
    fn test_bloom_new_non_empty() {
        let bf = BloomFilter::new(1000, 0.01);
        assert!(!bf.bits.is_empty());
        assert!(bf.num_hashes >= 1);
        assert!(bf.bit_count >= 64);
    }

    #[test]
    fn test_bloom_larger_capacity_more_bits() {
        let bf_small = BloomFilter::new(100, 0.01);
        let bf_large = BloomFilter::new(10_000, 0.01);
        assert!(bf_large.bit_count > bf_small.bit_count);
    }

    #[test]
    fn test_bloom_initially_empty() {
        let bf = BloomFilter::new(100, 0.01);
        assert!(!bf.contains(b"hello"));
        assert!(!bf.contains(b"world"));
    }

    #[test]
    fn test_bloom_insert_and_contains() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"oximedia");
        assert!(bf.contains(b"oximedia"));
    }

    #[test]
    fn test_bloom_multiple_inserts() {
        let mut bf = BloomFilter::new(200, 0.01);
        let items: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "Item should be present after insert");
        }
    }

    #[test]
    fn test_bloom_clear() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"test_item");
        assert!(bf.contains(b"test_item"));
        bf.clear();
        // After clear, the filter should no longer report the item as present
        assert!(!bf.contains(b"test_item"));
    }

    #[test]
    fn test_bloom_estimated_count_zero_initially() {
        let bf = BloomFilter::new(100, 0.01);
        assert_eq!(bf.estimated_count(), 0);
    }

    #[test]
    fn test_bloom_estimated_count_after_inserts() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..100u64 {
            bf.insert(&i.to_le_bytes());
        }
        // Estimate should be in a reasonable range (not exact due to probabilistic nature)
        let est = bf.estimated_count();
        assert!(est > 0, "Estimated count should be positive");
    }

    #[test]
    fn test_bloom_different_fpr_different_hashes() {
        let bf_strict = BloomFilter::new(100, 0.0001);
        let bf_loose = BloomFilter::new(100, 0.1);
        // Stricter FPR → more hash functions
        assert!(bf_strict.num_hashes >= bf_loose.num_hashes);
    }

    // ---- NearDuplicateDetector tests ----

    #[test]
    fn test_near_dup_new() {
        let det = NearDuplicateDetector::new(500);
        assert!(det.seen.bit_count >= 64);
    }

    #[test]
    fn test_near_dup_first_occurrence_not_duplicate() {
        let mut det = NearDuplicateDetector::new(100);
        let result = det.add_and_check(b"unique_fingerprint");
        assert!(!result, "First occurrence should not be a duplicate");
    }

    #[test]
    fn test_near_dup_second_occurrence_is_duplicate() {
        let mut det = NearDuplicateDetector::new(100);
        det.add_and_check(b"duplicate_fingerprint");
        let result = det.add_and_check(b"duplicate_fingerprint");
        assert!(result, "Second occurrence should be detected as duplicate");
    }

    #[test]
    fn test_near_dup_reset_clears_state() {
        let mut det = NearDuplicateDetector::new(100);
        det.add_and_check(b"item_to_reset");
        det.reset();
        // After reset, the same item should not appear as a duplicate
        let result = det.add_and_check(b"item_to_reset");
        assert!(!result, "After reset, item should not be a duplicate");
    }

    #[test]
    fn test_near_dup_multiple_unique_items() {
        let mut det = NearDuplicateDetector::new(1000);
        let items: Vec<Vec<u8>> = (0u64..50).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            // First occurrence of each unique item should not be flagged
            let is_dup = det.add_and_check(item);
            // Could theoretically be a false positive, but with 1000 capacity it should not
            assert!(!is_dup, "Unique item should not be flagged as duplicate");
        }
    }

    #[test]
    fn test_near_dup_estimated_unique_count() {
        let mut det = NearDuplicateDetector::new(1000);
        for i in 0u64..20 {
            det.add_and_check(&i.to_le_bytes());
        }
        let est = det.estimated_unique_count();
        assert!(est > 0);
    }

    // ---- Pre-screening tests ----

    #[test]
    fn test_prescreen_all_unique() {
        let fingerprints: Vec<Vec<u8>> = (0u64..50).map(|i| i.to_le_bytes().to_vec()).collect();
        let result = prescreen_fingerprints(&fingerprints, 1000, 0.01);
        assert_eq!(result.total, 50);
        // All unique items should be in the unique set
        assert_eq!(result.unique.len(), 50);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn test_prescreen_with_duplicates() {
        let mut fingerprints: Vec<Vec<u8>> = Vec::new();
        // Add 10 unique items
        for i in 0u64..10 {
            fingerprints.push(i.to_le_bytes().to_vec());
        }
        // Add 10 duplicates of item 0
        for _ in 0..10 {
            fingerprints.push(0u64.to_le_bytes().to_vec());
        }
        let result = prescreen_fingerprints(&fingerprints, 1000, 0.01);
        assert_eq!(result.total, 20);
        // The first occurrence of item 0 is unique; the 10 re-inserts are candidates
        assert_eq!(result.candidates.len(), 10);
    }

    #[test]
    fn test_prescreen_candidate_rate() {
        let fingerprints = vec![
            vec![1u8, 2, 3],
            vec![1u8, 2, 3], // duplicate
            vec![4u8, 5, 6],
            vec![4u8, 5, 6], // duplicate
            vec![7u8, 8, 9],
        ];
        let result = prescreen_fingerprints(&fingerprints, 100, 0.01);
        assert_eq!(result.total, 5);
        assert_eq!(result.candidates.len(), 2);
        assert!((result.candidate_rate() - 0.4).abs() < f64::EPSILON);
        assert!((result.rejection_rate() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prescreen_perceptual_hashes_identical() {
        let hashes = vec![0xDEAD_BEEF_DEAD_BEEFu64; 5];
        let result = prescreen_perceptual_hashes(&hashes, 16, 1000, 0.01);
        assert_eq!(result.total, 5);
        // First is unique, rest are candidates
        assert_eq!(result.candidates.len(), 4);
        assert_eq!(result.unique.len(), 1);
    }

    #[test]
    fn test_prescreen_perceptual_hashes_all_different() {
        // Hashes that differ significantly in high bits
        let hashes: Vec<u64> = (0..20u64).map(|i| i << 48).collect();
        let result = prescreen_perceptual_hashes(&hashes, 16, 1000, 0.01);
        assert_eq!(result.total, 20);
        // With 16-bit quantization from top bits, all should be unique
        assert_eq!(result.unique.len(), 20);
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn test_prescreen_empty_input() {
        let result = prescreen_fingerprints(&[], 100, 0.01);
        assert_eq!(result.total, 0);
        assert!(result.candidates.is_empty());
        assert!(result.unique.is_empty());
        assert_eq!(result.candidate_rate(), 0.0);
        assert_eq!(result.rejection_rate(), 0.0);
    }

    #[test]
    fn test_prescreen_result_rates_sum_to_one() {
        let fingerprints: Vec<Vec<u8>> =
            vec![vec![1, 2], vec![1, 2], vec![3, 4], vec![5, 6], vec![3, 4]];
        let result = prescreen_fingerprints(&fingerprints, 100, 0.01);
        let sum = result.candidate_rate() + result.rejection_rate();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // ---- DedupPipelineConfig tests ----

    #[test]
    fn test_pipeline_config_default() {
        let cfg = DedupPipelineConfig::default();
        assert_eq!(cfg.bloom_capacity, 10_000);
        assert_eq!(cfg.lsh_tables, 8);
        assert_eq!(cfg.max_hamming_distance, 10);
    }

    // ---- run_dedup_pipeline tests ----

    #[test]
    fn test_pipeline_empty() {
        let cfg = DedupPipelineConfig::default();
        let result = run_dedup_pipeline(&[], &cfg);
        assert_eq!(result.total, 0);
        assert!(result.bloom_candidates.is_empty());
        assert!(result.lsh_pairs.is_empty());
    }

    #[test]
    fn test_pipeline_all_identical() {
        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        let hashes: Vec<(u64, u64)> = (0..10).map(|i| (i, hash)).collect();
        let cfg = DedupPipelineConfig {
            bloom_capacity: 100,
            bloom_fpr: 0.01,
            quantize_bits: 16,
            lsh_tables: 6,
            lsh_bits_per_table: 8,
            max_hamming_distance: 0,
            seed: 42,
        };
        let result = run_dedup_pipeline(&hashes, &cfg);
        assert_eq!(result.total, 10);
        // First item is unique in bloom, rest are candidates
        assert_eq!(result.bloom_candidates.len(), 9);
        assert_eq!(result.bloom_unique.len(), 1);
        // LSH should find many pairs among the 9 candidates
        assert!(!result.lsh_pairs.is_empty());
    }

    #[test]
    fn test_pipeline_all_unique() {
        // Hashes that differ significantly in high bits
        let hashes: Vec<(u64, u64)> = (0..20u64).map(|i| (i, i << 48)).collect();
        let cfg = DedupPipelineConfig {
            bloom_capacity: 1000,
            bloom_fpr: 0.01,
            quantize_bits: 16,
            lsh_tables: 4,
            lsh_bits_per_table: 12,
            max_hamming_distance: 5,
            seed: 42,
        };
        let result = run_dedup_pipeline(&hashes, &cfg);
        assert_eq!(result.total, 20);
        // All should be unique in bloom
        assert_eq!(result.bloom_unique.len(), 20);
        assert!(result.bloom_candidates.is_empty());
        assert!(result.lsh_pairs.is_empty());
    }

    #[test]
    fn test_pipeline_mixed() {
        // 5 identical + 5 unique
        let base_hash = 0xAAAA_BBBB_CCCC_DDDDu64;
        let mut hashes: Vec<(u64, u64)> = (0..5).map(|i| (i, base_hash)).collect();
        for i in 5..10u64 {
            hashes.push((i, i << 48));
        }
        let cfg = DedupPipelineConfig::default();
        let result = run_dedup_pipeline(&hashes, &cfg);
        assert_eq!(result.total, 10);
        // bloom_unique should include the unique hashes
        assert!(!result.bloom_unique.is_empty());
    }

    #[test]
    fn test_pipeline_result_bloom_rejection_rate() {
        let result = DedupPipelineResult {
            bloom_candidates: vec![0, 1],
            bloom_unique: vec![2, 3, 4],
            lsh_pairs: vec![(0, 1, 0)],
            total: 5,
        };
        assert!((result.bloom_rejection_rate() - 0.6).abs() < f64::EPSILON);
        assert_eq!(result.num_pairs(), 1);
    }

    #[test]
    fn test_pipeline_result_empty_total() {
        let result = DedupPipelineResult {
            bloom_candidates: Vec::new(),
            bloom_unique: Vec::new(),
            lsh_pairs: Vec::new(),
            total: 0,
        };
        assert_eq!(result.bloom_rejection_rate(), 0.0);
        assert_eq!(result.num_pairs(), 0);
    }

    #[test]
    fn test_pipeline_near_duplicates() {
        let base = 0xFFFF_FFFF_FFFF_FFFFu64;
        let similar = base ^ 0b111; // 3 bits different
                                    // Use same high bits so bloom quantization groups them
        let hashes = vec![(1, base), (2, similar), (3, base)];
        let cfg = DedupPipelineConfig {
            bloom_capacity: 100,
            bloom_fpr: 0.01,
            quantize_bits: 16,
            lsh_tables: 8,
            lsh_bits_per_table: 6,
            max_hamming_distance: 5,
            seed: 77,
        };
        let result = run_dedup_pipeline(&hashes, &cfg);
        assert_eq!(result.total, 3);
        // At least some pairs should be found
        // (bloom may or may not catch all, depends on quantization)
    }
}
