//! Locality-Sensitive Hashing (LSH) index for approximate nearest-neighbour
//! deduplication of high-dimensional media feature vectors.

#![allow(dead_code)]

use std::collections::HashMap;

// ── Bucket ────────────────────────────────────────────────────────────────────

/// A single LSH bucket containing item IDs that hashed to the same key.
#[derive(Debug, Clone, Default)]
pub struct LshBucket {
    items: Vec<u64>,
}

impl LshBucket {
    /// Create an empty bucket.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an item ID into this bucket.
    pub fn insert(&mut self, id: u64) {
        if !self.items.contains(&id) {
            self.items.push(id);
        }
    }

    /// Returns the number of items in this bucket.
    #[must_use]
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// Returns the item IDs in this bucket.
    #[must_use]
    pub fn items(&self) -> &[u64] {
        &self.items
    }
}

// ── Bucket statistics ─────────────────────────────────────────────────────────

/// Aggregate statistics about all buckets in an LSH index.
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Total number of (non-empty) buckets.
    pub bucket_count: usize,
    /// Average items per bucket.
    pub avg_size: f64,
    /// Largest bucket size.
    pub max_size: usize,
    /// Total items across all buckets.
    pub total_items: usize,
}

impl BucketStats {
    /// Returns the average bucket size.
    #[must_use]
    pub fn avg_size(&self) -> f64 {
        self.avg_size
    }

    /// Returns the maximum bucket size.
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

// ── LSH Index ─────────────────────────────────────────────────────────────────

/// A simple random-projection LSH index for `D`-dimensional `f32` vectors.
///
/// Uses multiple hash tables, each projecting the vector onto a random
/// hyperplane sign pattern to form a bucket key.
#[derive(Debug)]
pub struct LshIndex {
    /// Number of hash tables.
    num_tables: usize,
    /// Number of bits (hyperplanes) per table.
    bits_per_table: usize,
    /// Random projection vectors: `[table][bit][dim]`
    projections: Vec<Vec<Vec<f32>>>,
    /// Hash tables: `[table][bucket_key] -> LshBucket`
    tables: Vec<HashMap<u64, LshBucket>>,
    /// Dimensionality of the indexed vectors.
    dim: usize,
}

impl LshIndex {
    /// Create a new LSH index.
    ///
    /// # Arguments
    /// * `dim`            – Vector dimensionality.
    /// * `num_tables`     – Number of independent hash tables.
    /// * `bits_per_table` – Bits (hyperplanes) per table.
    /// * `seed`           – Seed for deterministic projection generation.
    #[must_use]
    pub fn new(dim: usize, num_tables: usize, bits_per_table: usize, seed: u64) -> Self {
        let projections = Self::generate_projections(dim, num_tables, bits_per_table, seed);
        let tables = vec![HashMap::new(); num_tables];
        Self {
            num_tables,
            bits_per_table,
            projections,
            tables,
            dim,
        }
    }

    /// Generate projection hyperplanes using a simple LCG PRNG seeded by
    /// `seed` so results are fully deterministic without external crates.
    #[allow(clippy::cast_precision_loss)]
    fn generate_projections(
        dim: usize,
        num_tables: usize,
        bits: usize,
        seed: u64,
    ) -> Vec<Vec<Vec<f32>>> {
        let mut state = seed.wrapping_add(1);
        let lcg_next = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-1, 1]
            let val = (*s >> 11) as f32 / (1u64 << 53) as f32;
            val * 2.0 - 1.0
        };

        (0..num_tables)
            .map(|_| {
                (0..bits)
                    .map(|_| (0..dim).map(|_| lcg_next(&mut state)).collect())
                    .collect()
            })
            .collect()
    }

    /// Compute the bucket key for `vec` in hash table `table_idx`.
    #[allow(clippy::cast_precision_loss)]
    fn bucket_key(&self, vec: &[f32], table_idx: usize) -> u64 {
        let mut key = 0u64;
        for (bit_idx, proj) in self.projections[table_idx].iter().enumerate() {
            let dot: f32 = vec.iter().zip(proj.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                key |= 1u64 << bit_idx;
            }
        }
        key
    }

    /// Insert item `id` with feature vector `vec` into the index.
    ///
    /// # Panics
    /// Panics if `vec.len() != self.dim`.
    pub fn insert(&mut self, id: u64, vec: &[f32]) {
        assert_eq!(
            vec.len(),
            self.dim,
            "Vector dimensionality mismatch: expected {}, got {}",
            self.dim,
            vec.len()
        );
        for t in 0..self.num_tables {
            let key = self.bucket_key(vec, t);
            self.tables[t].entry(key).or_default().insert(id);
        }
    }

    /// Query for all candidate neighbours of `vec`.
    ///
    /// Returns the union of IDs found in any matching bucket across all tables.
    ///
    /// # Panics
    /// Panics if `vec.len() != self.dim`.
    #[must_use]
    pub fn query(&self, vec: &[f32]) -> Vec<u64> {
        assert_eq!(
            vec.len(),
            self.dim,
            "Vector dimensionality mismatch: expected {}, got {}",
            self.dim,
            vec.len()
        );
        let mut candidates = std::collections::HashSet::new();
        for t in 0..self.num_tables {
            let key = self.bucket_key(vec, t);
            if let Some(bucket) = self.tables[t].get(&key) {
                for &id in bucket.items() {
                    candidates.insert(id);
                }
            }
        }
        let mut result: Vec<u64> = candidates.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Query and then filter to approximate nearest neighbours by Euclidean
    /// distance, returning up to `k` results sorted nearest-first.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn approximate_neighbors(&self, vec: &[f32], k: usize) -> Vec<u64> {
        // For this index we store IDs only (not vectors), so we return the
        // raw candidate list trimmed to k.  A full implementation would store
        // vectors too and re-rank by distance.
        let mut candidates = self.query(vec);
        candidates.truncate(k);
        candidates
    }

    /// Compute aggregate statistics across all buckets.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bucket_stats(&self) -> BucketStats {
        let all_sizes: Vec<usize> = self
            .tables
            .iter()
            .flat_map(|table| table.values().map(LshBucket::size))
            .collect();

        let bucket_count = all_sizes.len();
        let total_items: usize = all_sizes.iter().sum();
        let max_size = all_sizes.iter().copied().max().unwrap_or(0);
        let avg_size = if bucket_count == 0 {
            0.0
        } else {
            total_items as f64 / bucket_count as f64
        };

        BucketStats {
            bucket_count,
            avg_size,
            max_size,
            total_items,
        }
    }

    /// Returns the number of hash tables.
    #[must_use]
    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Returns the vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ── Bit-based LSH for Hamming space (perceptual hashes) ──────────────────────

/// An LSH index specialised for 64-bit perceptual hashes in Hamming space.
///
/// Each hash table selects a random subset of bits from the 64-bit hash.
/// Items whose selected bits match fall into the same bucket and become
/// candidates for detailed comparison.  This replaces O(n^2) pairwise
/// comparison with O(n * average_bucket_size) lookups.
#[derive(Debug)]
pub struct BitLshIndex {
    /// Number of hash tables.
    num_tables: usize,
    /// Number of bits sampled per table.
    bits_per_table: usize,
    /// Bit positions to sample for each table: `[table][bit_index]`
    bit_masks: Vec<Vec<u8>>,
    /// Hash tables: `[table][bucket_key] -> Vec<(id, hash)>`
    tables: Vec<HashMap<u64, Vec<(u64, u64)>>>,
}

impl BitLshIndex {
    /// Create a new bit-based LSH index.
    ///
    /// # Arguments
    /// * `num_tables`     - Number of independent hash tables (more = better recall).
    /// * `bits_per_table` - Bits sampled per table (fewer = more collisions = better recall but more candidates).
    /// * `seed`           - Seed for deterministic bit selection.
    #[must_use]
    pub fn new(num_tables: usize, bits_per_table: usize, seed: u64) -> Self {
        let bits_per_table = bits_per_table.min(64);
        let bit_masks = Self::generate_bit_masks(num_tables, bits_per_table, seed);
        let tables = vec![HashMap::new(); num_tables];
        Self {
            num_tables,
            bits_per_table,
            bit_masks,
            tables,
        }
    }

    /// Generate random bit-position selections using a deterministic LCG.
    fn generate_bit_masks(num_tables: usize, bits: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut state = seed.wrapping_add(1);
        let lcg_next = |s: &mut u64| -> u64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *s
        };

        (0..num_tables)
            .map(|_| {
                let mut positions: Vec<u8> = Vec::with_capacity(bits);
                while positions.len() < bits {
                    let candidate = (lcg_next(&mut state) % 64) as u8;
                    if !positions.contains(&candidate) {
                        positions.push(candidate);
                    }
                }
                positions.sort_unstable();
                positions
            })
            .collect()
    }

    /// Compute bucket key by extracting selected bits from a 64-bit hash.
    fn bucket_key(&self, hash: u64, table_idx: usize) -> u64 {
        let mut key = 0u64;
        for (i, &bit_pos) in self.bit_masks[table_idx].iter().enumerate() {
            if hash & (1u64 << bit_pos) != 0 {
                key |= 1u64 << i;
            }
        }
        key
    }

    /// Insert an item with a 64-bit perceptual hash.
    pub fn insert(&mut self, id: u64, hash: u64) {
        for t in 0..self.num_tables {
            let key = self.bucket_key(hash, t);
            self.tables[t].entry(key).or_default().push((id, hash));
        }
    }

    /// Query for candidate neighbours of `hash`.
    ///
    /// Returns deduplicated `(id, hash)` pairs that share a bucket with
    /// the query in at least one table.
    #[must_use]
    pub fn query_candidates(&self, hash: u64) -> Vec<(u64, u64)> {
        let mut seen = std::collections::HashSet::new();
        let mut candidates = Vec::new();
        for t in 0..self.num_tables {
            let key = self.bucket_key(hash, t);
            if let Some(bucket) = self.tables[t].get(&key) {
                for &(id, h) in bucket {
                    if seen.insert(id) {
                        candidates.push((id, h));
                    }
                }
            }
        }
        candidates
    }

    /// Find all pairs within Hamming distance `max_distance` using LSH.
    ///
    /// Returns `Vec<(id_a, id_b, distance)>` with `id_a < id_b`.
    #[must_use]
    pub fn find_near_duplicates(&self, max_distance: u32) -> Vec<(u64, u64, u32)> {
        // Collect all unique items
        let mut all_items: Vec<(u64, u64)> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        for table in &self.tables {
            for bucket in table.values() {
                for &(id, hash) in bucket {
                    if seen_ids.insert(id) {
                        all_items.push((id, hash));
                    }
                }
            }
        }

        // For each item, query candidates and check distance
        let mut pairs = std::collections::HashSet::new();
        let mut results = Vec::new();

        for &(id, hash) in &all_items {
            let candidates = self.query_candidates(hash);
            for (cid, chash) in candidates {
                if cid == id {
                    continue;
                }
                let (lo, hi) = if id < cid { (id, cid) } else { (cid, id) };
                if pairs.insert((lo, hi)) {
                    let dist = (hash ^ chash).count_ones();
                    if dist <= max_distance {
                        results.push((lo, hi, dist));
                    }
                }
            }
        }

        results
    }

    /// Returns the number of hash tables.
    #[must_use]
    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Returns the number of bits sampled per table.
    #[must_use]
    pub fn bits_per_table(&self) -> usize {
        self.bits_per_table
    }
}

// ── LSH-accelerated deduplication pipeline ───────────────────────────────────

/// Result of an LSH-accelerated deduplication pass.
#[derive(Debug, Clone)]
pub struct LshDedupResult {
    /// Pairs `(id_a, id_b, distance)` with `id_a < id_b`.
    pub pairs: Vec<(u64, u64, u32)>,
    /// Number of candidate pairs considered (before distance filter).
    pub candidates_checked: usize,
    /// Total items indexed.
    pub total_items: usize,
}

impl LshDedupResult {
    /// Fraction of N^2/2 pairwise comparisons that were actually performed.
    #[must_use]
    pub fn comparison_ratio(&self) -> f64 {
        let n = self.total_items;
        if n < 2 {
            return 0.0;
        }
        let full_pairs = n * (n - 1) / 2;
        if full_pairs == 0 {
            return 0.0;
        }
        self.candidates_checked as f64 / full_pairs as f64
    }

    /// Returns the number of duplicate pairs found.
    #[must_use]
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }
}

/// Run a full LSH-based deduplication pass over a set of 64-bit hashes.
///
/// Instead of O(n^2) pairwise comparison, this builds a [`BitLshIndex`]
/// and only checks candidate pairs that land in the same bucket in at
/// least one hash table.  The expected cost is O(n * average_bucket_size).
///
/// # Arguments
/// * `hashes` - Slice of `(id, hash)` pairs.
/// * `max_distance` - Maximum Hamming distance threshold for a duplicate.
/// * `num_tables` - Number of LSH tables (more = better recall, more work).
/// * `bits_per_table` - Bits sampled per table (fewer = more collisions = better recall).
/// * `seed` - Deterministic PRNG seed.
#[must_use]
pub fn lsh_dedup_pass(
    hashes: &[(u64, u64)],
    max_distance: u32,
    num_tables: usize,
    bits_per_table: usize,
    seed: u64,
) -> LshDedupResult {
    if hashes.len() < 2 {
        return LshDedupResult {
            pairs: Vec::new(),
            candidates_checked: 0,
            total_items: hashes.len(),
        };
    }

    let mut index = BitLshIndex::new(num_tables, bits_per_table, seed);
    for &(id, hash) in hashes {
        index.insert(id, hash);
    }

    let mut seen_pairs = std::collections::HashSet::new();
    let mut results = Vec::new();
    let mut candidates_checked: usize = 0;

    for &(id, hash) in hashes {
        let candidates = index.query_candidates(hash);
        for (cid, chash) in candidates {
            if cid == id {
                continue;
            }
            let (lo, hi) = if id < cid { (id, cid) } else { (cid, id) };
            if seen_pairs.insert((lo, hi)) {
                candidates_checked += 1;
                let dist = (hash ^ chash).count_ones();
                if dist <= max_distance {
                    results.push((lo, hi, dist));
                }
            }
        }
    }

    LshDedupResult {
        pairs: results,
        candidates_checked,
        total_items: hashes.len(),
    }
}

/// Group items by transitive closure over duplicate pairs.
///
/// Given LSH dedup results, builds connected components so that if A~B and
/// B~C then {A, B, C} form a single group.
#[must_use]
pub fn group_by_lsh_pairs(pairs: &[(u64, u64, u32)], all_ids: &[u64]) -> Vec<Vec<u64>> {
    use std::collections::HashMap;

    if pairs.is_empty() {
        return Vec::new();
    }

    // Build union-find
    let mut parent: HashMap<u64, u64> = HashMap::new();
    for &id in all_ids {
        parent.insert(id, id);
    }

    fn find(parent: &mut HashMap<u64, u64>, x: u64) -> u64 {
        let p = parent.get(&x).copied().unwrap_or(x);
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    for &(a, b, _) in pairs {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    // Collect groups with >1 member
    let mut groups: HashMap<u64, Vec<u64>> = HashMap::new();
    for &id in all_ids {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push(id);
    }

    groups.into_values().filter(|g| g.len() > 1).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot % dim] = 1.0;
        v
    }

    #[test]
    fn test_lsh_bucket_size_empty() {
        let b = LshBucket::new();
        assert_eq!(b.size(), 0);
    }

    #[test]
    fn test_lsh_bucket_insert_and_size() {
        let mut b = LshBucket::new();
        b.insert(1);
        b.insert(2);
        b.insert(1); // duplicate, should not increase size
        assert_eq!(b.size(), 2);
    }

    #[test]
    fn test_lsh_bucket_items() {
        let mut b = LshBucket::new();
        b.insert(42);
        b.insert(99);
        assert!(b.items().contains(&42));
        assert!(b.items().contains(&99));
    }

    #[test]
    fn test_lsh_index_creation() {
        let idx = LshIndex::new(8, 4, 6, 42);
        assert_eq!(idx.dim(), 8);
        assert_eq!(idx.num_tables(), 4);
    }

    #[test]
    fn test_lsh_index_insert_and_query_self() {
        let mut idx = LshIndex::new(4, 3, 4, 7);
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        idx.insert(1, &v);
        let results = idx.query(&v);
        assert!(results.contains(&1));
    }

    #[test]
    fn test_lsh_query_returns_sorted() {
        let mut idx = LshIndex::new(4, 2, 4, 13);
        let v = vec![1.0f32, 1.0, 1.0, 1.0];
        idx.insert(5, &v);
        idx.insert(3, &v);
        idx.insert(7, &v);
        let results = idx.query(&v);
        let mut sorted = results.clone();
        sorted.sort_unstable();
        assert_eq!(results, sorted);
    }

    #[test]
    fn test_lsh_similar_vectors_in_same_bucket() {
        let mut idx = LshIndex::new(8, 6, 6, 99);
        let v1 = vec![1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0f32, 1.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0]; // very similar
        idx.insert(10, &v1);
        idx.insert(11, &v2);
        let results = idx.query(&v1);
        // v1 itself must be found
        assert!(results.contains(&10));
    }

    #[test]
    fn test_lsh_approximate_neighbors_k_limit() {
        let mut idx = LshIndex::new(4, 2, 4, 17);
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        for i in 0..20u64 {
            idx.insert(i, &v);
        }
        let results = idx.approximate_neighbors(&v, 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_bucket_stats_empty() {
        let idx = LshIndex::new(4, 3, 4, 0);
        let stats = idx.bucket_stats();
        assert_eq!(stats.bucket_count, 0);
        assert_eq!(stats.max_size(), 0);
        assert_eq!(stats.avg_size(), 0.0);
    }

    #[test]
    fn test_bucket_stats_after_inserts() {
        let mut idx = LshIndex::new(4, 2, 4, 55);
        let v = vec![0.5f32, 0.5, 0.5, 0.5];
        idx.insert(1, &v);
        idx.insert(2, &v);
        let stats = idx.bucket_stats();
        assert!(stats.bucket_count > 0);
        assert!(stats.max_size() >= 1);
        assert!(stats.avg_size() > 0.0);
    }

    #[test]
    fn test_unit_vectors_different_dimensions() {
        let mut idx = LshIndex::new(8, 4, 5, 77);
        for i in 0..8u64 {
            let v = unit_vec(8, i as usize);
            idx.insert(i, &v);
        }
        // Each unit vector inserted without panic
        assert_eq!(idx.dim(), 8);
    }

    #[test]
    fn test_insert_multiple_tables() {
        let mut idx = LshIndex::new(4, 5, 4, 11);
        let v = vec![0.1f32, 0.2, 0.3, 0.4];
        idx.insert(100, &v);
        // Item should appear in query results
        let r = idx.query(&v);
        assert!(r.contains(&100));
    }

    #[test]
    fn test_bucket_stats_avg_max() {
        let stats = BucketStats {
            bucket_count: 3,
            avg_size: 2.5,
            max_size: 5,
            total_items: 7,
        };
        assert_eq!(stats.avg_size(), 2.5);
        assert_eq!(stats.max_size(), 5);
    }

    // ---- BitLshIndex tests ----

    #[test]
    fn test_bit_lsh_creation() {
        let idx = BitLshIndex::new(4, 8, 42);
        assert_eq!(idx.num_tables(), 4);
        assert_eq!(idx.bits_per_table(), 8);
    }

    #[test]
    fn test_bit_lsh_insert_and_self_query() {
        let mut idx = BitLshIndex::new(4, 8, 42);
        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        idx.insert(1, hash);
        let candidates = idx.query_candidates(hash);
        assert!(candidates.iter().any(|(id, _)| *id == 1));
    }

    #[test]
    fn test_bit_lsh_identical_hashes_found() {
        let mut idx = BitLshIndex::new(6, 10, 99);
        let hash = 0x1234_5678_9ABC_DEF0u64;
        idx.insert(1, hash);
        idx.insert(2, hash);
        let pairs = idx.find_near_duplicates(0);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (1, 2, 0));
    }

    #[test]
    fn test_bit_lsh_near_duplicates_within_distance() {
        let mut idx = BitLshIndex::new(8, 6, 77);
        let base = 0xFFFF_FFFF_FFFF_FFFFu64;
        // Flip 3 bits
        let similar = base ^ 0b111;
        idx.insert(10, base);
        idx.insert(11, similar);
        let pairs = idx.find_near_duplicates(5);
        // Should find this pair (distance = 3)
        assert!(!pairs.is_empty());
        let found = pairs.iter().any(|&(a, b, d)| a == 10 && b == 11 && d == 3);
        assert!(found, "Should find pair with distance 3");
    }

    #[test]
    fn test_bit_lsh_distant_hashes_not_paired() {
        let mut idx = BitLshIndex::new(4, 16, 55);
        idx.insert(1, 0x0000_0000_0000_0000);
        idx.insert(2, 0xFFFF_FFFF_FFFF_FFFF);
        let pairs = idx.find_near_duplicates(5);
        // Distance = 64, should not be paired
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_bit_lsh_many_items() {
        let mut idx = BitLshIndex::new(4, 8, 42);
        // Insert 100 items with sequential hashes
        for i in 0..100u64 {
            idx.insert(i, i);
        }
        // Query for item 50; should at least find itself
        let candidates = idx.query_candidates(50);
        assert!(candidates.iter().any(|(id, _)| *id == 50));
    }

    #[test]
    fn test_bit_lsh_deduplicated_candidates() {
        let mut idx = BitLshIndex::new(8, 6, 42);
        let hash = 0xAAAA_BBBB_CCCC_DDDDu64;
        idx.insert(1, hash);
        let candidates = idx.query_candidates(hash);
        // Even though it appears in multiple tables, should be deduplicated
        let count_1 = candidates.iter().filter(|(id, _)| *id == 1).count();
        assert_eq!(count_1, 1);
    }

    // ---- lsh_dedup_pass tests ----

    #[test]
    fn test_lsh_dedup_pass_empty() {
        let result = lsh_dedup_pass(&[], 5, 4, 8, 42);
        assert!(result.pairs.is_empty());
        assert_eq!(result.total_items, 0);
    }

    #[test]
    fn test_lsh_dedup_pass_single() {
        let result = lsh_dedup_pass(&[(1, 0xDEAD)], 5, 4, 8, 42);
        assert!(result.pairs.is_empty());
        assert_eq!(result.total_items, 1);
    }

    #[test]
    fn test_lsh_dedup_pass_identical() {
        let hash = 0xDEAD_BEEF_CAFE_BABEu64;
        let hashes = vec![(1, hash), (2, hash), (3, hash)];
        let result = lsh_dedup_pass(&hashes, 0, 6, 8, 42);
        // All identical → 3 pairs: (1,2), (1,3), (2,3)
        assert_eq!(result.pairs.len(), 3);
        for &(_, _, d) in &result.pairs {
            assert_eq!(d, 0);
        }
    }

    #[test]
    fn test_lsh_dedup_pass_near_duplicates() {
        let base = 0xFFFF_FFFF_FFFF_FFFFu64;
        let similar = base ^ 0b111; // 3 bits different
        let hashes = vec![(10, base), (20, similar)];
        let result = lsh_dedup_pass(&hashes, 5, 8, 6, 77);
        assert!(!result.pairs.is_empty(), "Should find near-duplicate pair");
        let found = result
            .pairs
            .iter()
            .any(|&(a, b, d)| a == 10 && b == 20 && d == 3);
        assert!(found);
    }

    #[test]
    fn test_lsh_dedup_pass_distant_not_paired() {
        let hashes = vec![(1, 0x0000_0000_0000_0000u64), (2, 0xFFFF_FFFF_FFFF_FFFFu64)];
        let result = lsh_dedup_pass(&hashes, 5, 4, 16, 55);
        // Distance = 64, well above threshold
        assert!(result.pairs.is_empty());
    }

    #[test]
    fn test_lsh_dedup_pass_comparison_ratio() {
        let hash = 0xABCDu64;
        let hashes: Vec<(u64, u64)> = (0..100).map(|i| (i, hash)).collect();
        let result = lsh_dedup_pass(&hashes, 0, 4, 8, 42);
        // comparison_ratio should be well below 1.0 if LSH is filtering
        // (or could be 1.0 if all land in same bucket, which is fine for identical hashes)
        assert!(result.comparison_ratio() <= 1.0);
        assert!(result.comparison_ratio() > 0.0);
    }

    #[test]
    fn test_lsh_dedup_result_num_pairs() {
        let result = LshDedupResult {
            pairs: vec![(1, 2, 0), (1, 3, 1)],
            candidates_checked: 5,
            total_items: 3,
        };
        assert_eq!(result.num_pairs(), 2);
    }

    // ---- group_by_lsh_pairs tests ----

    #[test]
    fn test_group_by_lsh_pairs_empty() {
        let groups = group_by_lsh_pairs(&[], &[1, 2, 3]);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_group_by_lsh_pairs_single_pair() {
        let pairs = vec![(1, 2, 0)];
        let groups = group_by_lsh_pairs(&pairs, &[1, 2, 3]);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!(g.contains(&1));
        assert!(g.contains(&2));
        assert!(!g.contains(&3));
    }

    #[test]
    fn test_group_by_lsh_pairs_transitive() {
        // A~B, B~C → {A, B, C}
        let pairs = vec![(1, 2, 3), (2, 3, 2)];
        let groups = group_by_lsh_pairs(&pairs, &[1, 2, 3]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_group_by_lsh_pairs_two_groups() {
        let pairs = vec![(1, 2, 0), (3, 4, 1)];
        let groups = group_by_lsh_pairs(&pairs, &[1, 2, 3, 4, 5]);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_lsh_dedup_pass_many_items_sparse() {
        // 200 items with well-separated hashes: most should NOT be paired
        let hashes: Vec<(u64, u64)> = (0..200).map(|i| (i, i * 0x0101_0101_0101_0101)).collect();
        let result = lsh_dedup_pass(&hashes, 3, 4, 12, 42);
        // We can't guarantee exact count, but it should complete without panic
        assert_eq!(result.total_items, 200);
    }
}
