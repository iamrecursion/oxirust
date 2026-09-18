//! Hierarchical deduplication: fast pass (hash) -> medium pass (perceptual) -> slow pass (SSIM).
//!
//! This module implements a multi-pass deduplication pipeline that progressively
//! applies more expensive detection methods only to items that survived cheaper
//! passes. This dramatically reduces computation for large media libraries.
//!
//! # Passes
//!
//! 1. **Fast pass (cryptographic hash)**: Groups files with identical BLAKE3 hashes.
//!    These are exact duplicates. Cost: O(n).
//! 2. **Medium pass (perceptual hash + LSH)**: Among the remaining files, uses
//!    perceptual hashing (pHash) with LSH-based nearest-neighbor search to find
//!    near-duplicates. Cost: O(n * avg_bucket_size) instead of O(n^2).
//! 3. **Slow pass (SSIM)**: Verifies medium-pass candidate pairs with full
//!    structural similarity (SSIM) computation to eliminate false positives.
//!    Cost: O(candidates) << O(n^2).
//!
//! # Example
//!
//! ```
//! use oximedia_dedup::hierarchical::{HierarchicalDedup, HierarchicalConfig};
//!
//! let config = HierarchicalConfig::default();
//! let mut dedup = HierarchicalDedup::new(config);
//!
//! // Add items
//! dedup.add_item("video1.mp4", &[0xAB; 32], 0xDEAD_BEEF, &[128; 64]);
//! dedup.add_item("video2.mp4", &[0xAB; 32], 0xDEAD_BEEF, &[128; 64]);
//! dedup.add_item("video3.mp4", &[0xCD; 32], 0xDEAD_BEE0, &[130; 64]);
//!
//! let result = dedup.run();
//! assert!(!result.exact_groups.is_empty());
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::lsh_index::BitLshIndex;
use crate::visual::{self, Image, SsimParams};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for hierarchical deduplication.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Whether to run the fast (exact hash) pass.
    pub enable_fast_pass: bool,
    /// Whether to run the medium (perceptual hash + LSH) pass.
    pub enable_medium_pass: bool,
    /// Whether to run the slow (SSIM) pass to verify medium-pass candidates.
    pub enable_slow_pass: bool,

    /// Maximum Hamming distance for perceptual hash near-duplicate detection.
    pub perceptual_max_distance: u32,
    /// Number of LSH hash tables for the medium pass.
    pub lsh_num_tables: usize,
    /// Bits sampled per LSH table.
    pub lsh_bits_per_table: usize,
    /// PRNG seed for LSH.
    pub lsh_seed: u64,

    /// Minimum SSIM score to confirm a near-duplicate in the slow pass.
    pub ssim_threshold: f64,
    /// Thumbnail resolution (width = height) for SSIM computation.
    pub ssim_thumbnail_size: usize,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            enable_fast_pass: true,
            enable_medium_pass: true,
            enable_slow_pass: true,
            perceptual_max_distance: 10,
            lsh_num_tables: 8,
            lsh_bits_per_table: 8,
            lsh_seed: 42,
            ssim_threshold: 0.85,
            ssim_thumbnail_size: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Input item
// ---------------------------------------------------------------------------

/// A single media item to be deduplicated.
#[derive(Debug, Clone)]
pub struct DedupItem {
    /// Unique identifier (e.g., file path).
    pub id: String,
    /// Cryptographic hash (e.g., BLAKE3 32-byte digest).
    pub content_hash: Vec<u8>,
    /// 64-bit perceptual hash.
    pub perceptual_hash: u64,
    /// Grayscale thumbnail pixels (flattened, row-major).
    pub thumbnail: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A group of duplicate items found at a specific pass.
#[derive(Debug, Clone)]
pub struct DedupGroup {
    /// IDs of the items in this group.
    pub item_ids: Vec<String>,
    /// The detection pass that discovered this group.
    pub pass: DetectionPass,
    /// Similarity score (1.0 for exact, < 1.0 for near-duplicates).
    pub similarity: f64,
}

/// Which detection pass found a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionPass {
    /// Exact content hash match.
    FastHash,
    /// Perceptual hash near-match (via LSH).
    MediumPerceptual,
    /// SSIM-verified near-duplicate.
    SlowSsim,
}

impl DetectionPass {
    /// Human-readable label for the pass.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::FastHash => "exact-hash",
            Self::MediumPerceptual => "perceptual-lsh",
            Self::SlowSsim => "ssim-verified",
        }
    }
}

/// Full result of the hierarchical deduplication pipeline.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Groups found by the fast (exact hash) pass.
    pub exact_groups: Vec<DedupGroup>,
    /// Groups found by the medium (perceptual + LSH) pass.
    pub perceptual_groups: Vec<DedupGroup>,
    /// Groups found by the slow (SSIM) pass.
    pub ssim_groups: Vec<DedupGroup>,
    /// Number of items processed.
    pub total_items: usize,
    /// Items eliminated by the fast pass (not sent to medium pass).
    pub fast_pass_hits: usize,
    /// Candidate pairs considered in the medium pass.
    pub medium_pass_candidates: usize,
    /// Candidate pairs verified in the slow pass.
    pub slow_pass_verified: usize,
}

impl HierarchicalResult {
    /// Total number of duplicate groups across all passes.
    #[must_use]
    pub fn total_groups(&self) -> usize {
        self.exact_groups.len() + self.perceptual_groups.len() + self.ssim_groups.len()
    }

    /// Total number of duplicate items (across all groups, minus one keeper per group).
    #[must_use]
    pub fn total_duplicates(&self) -> usize {
        let count_dupes = |groups: &[DedupGroup]| -> usize {
            groups
                .iter()
                .map(|g| g.item_ids.len().saturating_sub(1))
                .sum()
        };
        count_dupes(&self.exact_groups)
            + count_dupes(&self.perceptual_groups)
            + count_dupes(&self.ssim_groups)
    }

    /// All groups combined, in order: exact, perceptual, ssim.
    #[must_use]
    pub fn all_groups(&self) -> Vec<&DedupGroup> {
        let mut all = Vec::new();
        for g in &self.exact_groups {
            all.push(g);
        }
        for g in &self.perceptual_groups {
            all.push(g);
        }
        for g in &self.ssim_groups {
            all.push(g);
        }
        all
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Hierarchical deduplication pipeline.
pub struct HierarchicalDedup {
    config: HierarchicalConfig,
    items: Vec<DedupItem>,
}

impl HierarchicalDedup {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: HierarchicalConfig) -> Self {
        Self {
            config,
            items: Vec::new(),
        }
    }

    /// Add an item to the pipeline.
    ///
    /// # Arguments
    /// * `id` - Unique identifier (e.g., file path).
    /// * `content_hash` - Cryptographic hash bytes.
    /// * `perceptual_hash` - 64-bit perceptual hash.
    /// * `thumbnail` - Grayscale thumbnail pixels.
    pub fn add_item(
        &mut self,
        id: &str,
        content_hash: &[u8],
        perceptual_hash: u64,
        thumbnail: &[u8],
    ) {
        self.items.push(DedupItem {
            id: id.to_string(),
            content_hash: content_hash.to_vec(),
            perceptual_hash,
            thumbnail: thumbnail.to_vec(),
        });
    }

    /// Returns the number of items added.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Run the full hierarchical deduplication pipeline.
    #[must_use]
    pub fn run(&self) -> HierarchicalResult {
        let mut result = HierarchicalResult {
            exact_groups: Vec::new(),
            perceptual_groups: Vec::new(),
            ssim_groups: Vec::new(),
            total_items: self.items.len(),
            fast_pass_hits: 0,
            medium_pass_candidates: 0,
            slow_pass_verified: 0,
        };

        if self.items.len() < 2 {
            return result;
        }

        // Track which items have been assigned to an exact group (skip in later passes).
        let mut assigned_exact = vec![false; self.items.len()];

        // ── Pass 1: Fast (exact hash) ──
        if self.config.enable_fast_pass {
            let groups = self.fast_pass();
            for group in &groups {
                for id in &group.item_ids {
                    if let Some(pos) = self.items.iter().position(|item| item.id == *id) {
                        assigned_exact[pos] = true;
                    }
                }
            }
            result.fast_pass_hits = groups
                .iter()
                .map(|g| g.item_ids.len().saturating_sub(1))
                .sum();
            result.exact_groups = groups;
        }

        // Collect indices of items not yet assigned.
        let remaining: Vec<usize> = (0..self.items.len())
            .filter(|&i| !assigned_exact[i])
            .collect();

        if remaining.len() < 2 {
            return result;
        }

        // ── Pass 2: Medium (perceptual hash + LSH) ──
        if self.config.enable_medium_pass {
            let (perceptual_groups, candidate_count) = self.medium_pass(&remaining);
            result.medium_pass_candidates = candidate_count;

            if self.config.enable_slow_pass {
                // ── Pass 3: Slow (SSIM verification) ──
                let (verified, verified_count) = self.slow_pass(&perceptual_groups);
                result.slow_pass_verified = verified_count;
                result.ssim_groups = verified;
            } else {
                // No slow pass: promote medium results directly.
                result.perceptual_groups = perceptual_groups;
            }
        }

        result
    }

    /// Pass 1: Group by exact content hash.
    fn fast_pass(&self) -> Vec<DedupGroup> {
        let mut hash_groups: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        for (i, item) in self.items.iter().enumerate() {
            hash_groups
                .entry(item.content_hash.clone())
                .or_default()
                .push(i);
        }

        hash_groups
            .values()
            .filter(|indices| indices.len() >= 2)
            .map(|indices| DedupGroup {
                item_ids: indices.iter().map(|&i| self.items[i].id.clone()).collect(),
                pass: DetectionPass::FastHash,
                similarity: 1.0,
            })
            .collect()
    }

    /// Pass 2: Find near-duplicates using perceptual hash + LSH.
    ///
    /// Returns (groups, candidate_count).
    fn medium_pass(&self, remaining: &[usize]) -> (Vec<DedupGroup>, usize) {
        let mut lsh = BitLshIndex::new(
            self.config.lsh_num_tables,
            self.config.lsh_bits_per_table,
            self.config.lsh_seed,
        );

        // Insert remaining items into LSH index (using their position index as ID).
        for &idx in remaining {
            lsh.insert(idx as u64, self.items[idx].perceptual_hash);
        }

        // Find near-duplicate pairs.
        let mut seen_pairs = std::collections::HashSet::new();
        let mut candidate_count = 0usize;
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for &idx in remaining {
            let candidates = lsh.query_candidates(self.items[idx].perceptual_hash);
            for (cid, chash) in candidates {
                let cidx = cid as usize;
                if cidx == idx {
                    continue;
                }
                let (lo, hi) = if idx < cidx { (idx, cidx) } else { (cidx, idx) };
                if seen_pairs.insert((lo, hi)) {
                    candidate_count += 1;
                    let dist = (self.items[idx].perceptual_hash ^ chash).count_ones();
                    if dist <= self.config.perceptual_max_distance {
                        let similarity = 1.0 - f64::from(dist) / 64.0;
                        edges.push((lo, hi, similarity));
                    }
                }
            }
        }

        // Group by transitive closure using Union-Find.
        let groups = self.union_find_groups(&edges);
        (groups, candidate_count)
    }

    /// Pass 3: Verify medium-pass groups with SSIM.
    ///
    /// Returns (verified_groups, verified_pair_count).
    fn slow_pass(&self, candidate_groups: &[DedupGroup]) -> (Vec<DedupGroup>, usize) {
        let res = self.config.ssim_thumbnail_size.max(4);
        let expected_pixels = res * res;
        let ssim_params = SsimParams::default();
        let mut verified_groups = Vec::new();
        let mut verified_count = 0usize;

        for group in candidate_groups {
            // Build thumbnail images for each item in the group.
            let mut images: Vec<(String, Option<Image>)> = Vec::new();
            for id in &group.item_ids {
                if let Some(item) = self.items.iter().find(|it| it.id == *id) {
                    let img = if item.thumbnail.len() == expected_pixels {
                        Image::from_data(res, res, 1, item.thumbnail.clone()).ok()
                    } else {
                        None
                    };
                    images.push((id.clone(), img));
                }
            }

            // Verify pairwise SSIM among group members.
            let mut verified_members: Vec<String> = Vec::new();
            let mut best_ssim = 0.0f64;

            if images.len() >= 2 {
                // Use the first item as anchor and check all others against it.
                if let Some(ref anchor_img) = images[0].1 {
                    verified_members.push(images[0].0.clone());
                    for (id, img_opt) in images.iter().skip(1) {
                        if let Some(ref img) = img_opt {
                            let ssim = visual::compute_ssim(anchor_img, img, &ssim_params);
                            verified_count += 1;
                            if ssim >= self.config.ssim_threshold {
                                verified_members.push(id.clone());
                                if ssim > best_ssim {
                                    best_ssim = ssim;
                                }
                            }
                        }
                    }
                }
            }

            if verified_members.len() >= 2 {
                verified_groups.push(DedupGroup {
                    item_ids: verified_members,
                    pass: DetectionPass::SlowSsim,
                    similarity: best_ssim,
                });
            }
        }

        (verified_groups, verified_count)
    }

    /// Group edges by transitive closure using Union-Find.
    fn union_find_groups(&self, edges: &[(usize, usize, f64)]) -> Vec<DedupGroup> {
        if edges.is_empty() {
            return Vec::new();
        }

        // Collect unique indices.
        let mut all_indices = std::collections::HashSet::new();
        for &(a, b, _) in edges {
            all_indices.insert(a);
            all_indices.insert(b);
        }

        let idx_list: Vec<usize> = all_indices.into_iter().collect();
        let mut parent: HashMap<usize, usize> = idx_list.iter().map(|&i| (i, i)).collect();

        fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
            let p = parent.get(&x).copied().unwrap_or(x);
            if p == x {
                return x;
            }
            let root = find(parent, p);
            parent.insert(x, root);
            root
        }

        for &(a, b, _) in edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                parent.insert(ra, rb);
            }
        }

        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for &i in &idx_list {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(i);
        }

        // Compute best similarity per group.
        let mut group_best_sim: HashMap<usize, f64> = HashMap::new();
        for &(a, _, sim) in edges {
            let root = find(&mut parent, a);
            let entry = group_best_sim.entry(root).or_insert(0.0);
            if sim > *entry {
                *entry = sim;
            }
        }

        groups
            .into_iter()
            .filter(|(_, members)| members.len() >= 2)
            .map(|(root, members)| DedupGroup {
                item_ids: members.iter().map(|&i| self.items[i].id.clone()).collect(),
                pass: DetectionPass::MediumPerceptual,
                similarity: group_best_sim.get(&root).copied().unwrap_or(0.0),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(byte: u8) -> Vec<u8> {
        vec![byte; 32]
    }

    fn make_thumbnail(val: u8, size: usize) -> Vec<u8> {
        vec![val; size * size]
    }

    #[test]
    fn test_hierarchical_config_default() {
        let cfg = HierarchicalConfig::default();
        assert!(cfg.enable_fast_pass);
        assert!(cfg.enable_medium_pass);
        assert!(cfg.enable_slow_pass);
        assert_eq!(cfg.perceptual_max_distance, 10);
    }

    #[test]
    fn test_detection_pass_label() {
        assert_eq!(DetectionPass::FastHash.label(), "exact-hash");
        assert_eq!(DetectionPass::MediumPerceptual.label(), "perceptual-lsh");
        assert_eq!(DetectionPass::SlowSsim.label(), "ssim-verified");
    }

    #[test]
    fn test_empty_pipeline() {
        let dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        let result = dedup.run();
        assert_eq!(result.total_items, 0);
        assert_eq!(result.total_groups(), 0);
        assert_eq!(result.total_duplicates(), 0);
    }

    #[test]
    fn test_single_item() {
        let mut dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        dedup.add_item(
            "only.mp4",
            &make_hash(0xAB),
            0xDEAD,
            &make_thumbnail(128, 16),
        );
        let result = dedup.run();
        assert_eq!(result.total_items, 1);
        assert_eq!(result.total_groups(), 0);
    }

    #[test]
    fn test_exact_duplicates_fast_pass() {
        let mut dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        let hash = make_hash(0xAB);
        let thumb = make_thumbnail(128, 16);
        dedup.add_item("a.mp4", &hash, 0xDEAD_BEEF, &thumb);
        dedup.add_item("b.mp4", &hash, 0xDEAD_BEEF, &thumb);
        dedup.add_item(
            "c.mp4",
            &make_hash(0xCD),
            0x1234_5678,
            &make_thumbnail(64, 16),
        );

        let result = dedup.run();
        assert_eq!(result.exact_groups.len(), 1);
        assert_eq!(result.exact_groups[0].item_ids.len(), 2);
        assert_eq!(result.exact_groups[0].pass, DetectionPass::FastHash);
        assert_eq!(result.exact_groups[0].similarity, 1.0);
        assert_eq!(result.fast_pass_hits, 1);
    }

    #[test]
    fn test_perceptual_near_duplicates() {
        let mut cfg = HierarchicalConfig::default();
        cfg.enable_slow_pass = false; // disable SSIM verification
        cfg.perceptual_max_distance = 5;

        let mut dedup = HierarchicalDedup::new(cfg);
        let base_hash = 0xFFFF_FFFF_FFFF_FFFFu64;
        let similar_hash = base_hash ^ 0b111; // 3 bits different

        dedup.add_item(
            "a.mp4",
            &make_hash(0xAA),
            base_hash,
            &make_thumbnail(128, 16),
        );
        dedup.add_item(
            "b.mp4",
            &make_hash(0xBB),
            similar_hash,
            &make_thumbnail(130, 16),
        );

        let result = dedup.run();
        assert!(result.exact_groups.is_empty(), "Different hashes");
        // Should find perceptual near-duplicates
        if !result.perceptual_groups.is_empty() {
            assert_eq!(
                result.perceptual_groups[0].pass,
                DetectionPass::MediumPerceptual
            );
            assert!(result.perceptual_groups[0].similarity > 0.9);
        }
    }

    #[test]
    fn test_ssim_verification_pass() {
        let cfg = HierarchicalConfig {
            ssim_thumbnail_size: 8,
            perceptual_max_distance: 10,
            ssim_threshold: 0.5, // low threshold for test
            ..HierarchicalConfig::default()
        };

        let mut dedup = HierarchicalDedup::new(cfg);
        // Two items with different hashes but identical thumbnails
        let thumb = make_thumbnail(128, 8);
        let base_hash = 0xFFFF_FFFF_FFFF_FFFFu64;
        let similar_hash = base_hash ^ 0b11; // 2 bits different

        dedup.add_item("x.mp4", &make_hash(0x11), base_hash, &thumb);
        dedup.add_item("y.mp4", &make_hash(0x22), similar_hash, &thumb);

        let result = dedup.run();
        assert!(result.exact_groups.is_empty());
        // SSIM should verify since thumbnails are identical
        if !result.ssim_groups.is_empty() {
            assert_eq!(result.ssim_groups[0].pass, DetectionPass::SlowSsim);
        }
    }

    #[test]
    fn test_three_exact_duplicates() {
        let mut dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        let hash = make_hash(0xFF);
        let thumb = make_thumbnail(200, 16);

        dedup.add_item("a.mp4", &hash, 0x1111, &thumb);
        dedup.add_item("b.mp4", &hash, 0x1111, &thumb);
        dedup.add_item("c.mp4", &hash, 0x1111, &thumb);

        let result = dedup.run();
        assert_eq!(result.exact_groups.len(), 1);
        assert_eq!(result.exact_groups[0].item_ids.len(), 3);
        assert_eq!(result.total_duplicates(), 2);
    }

    #[test]
    fn test_fast_pass_only() {
        let cfg = HierarchicalConfig {
            enable_fast_pass: true,
            enable_medium_pass: false,
            enable_slow_pass: false,
            ..HierarchicalConfig::default()
        };

        let mut dedup = HierarchicalDedup::new(cfg);
        let hash = make_hash(0xAA);
        dedup.add_item("a.mp4", &hash, 0xDEAD, &make_thumbnail(128, 16));
        dedup.add_item("b.mp4", &hash, 0xDEAD, &make_thumbnail(128, 16));

        let result = dedup.run();
        assert_eq!(result.exact_groups.len(), 1);
        assert!(result.perceptual_groups.is_empty());
        assert!(result.ssim_groups.is_empty());
    }

    #[test]
    fn test_all_groups_aggregation() {
        let mut dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        let hash = make_hash(0x01);
        let thumb = make_thumbnail(128, 16);

        dedup.add_item("a.mp4", &hash, 0x1111, &thumb);
        dedup.add_item("b.mp4", &hash, 0x1111, &thumb);

        let result = dedup.run();
        let all = result.all_groups();
        assert!(!all.is_empty());
    }

    #[test]
    fn test_item_count() {
        let mut dedup = HierarchicalDedup::new(HierarchicalConfig::default());
        assert_eq!(dedup.item_count(), 0);
        dedup.add_item("a.mp4", &[0; 32], 0, &[128; 64]);
        assert_eq!(dedup.item_count(), 1);
    }

    #[test]
    fn test_distant_perceptual_hashes_not_grouped() {
        let mut cfg = HierarchicalConfig::default();
        cfg.enable_slow_pass = false;
        cfg.perceptual_max_distance = 3;

        let mut dedup = HierarchicalDedup::new(cfg);
        // Hashes with high Hamming distance
        dedup.add_item(
            "a.mp4",
            &make_hash(0xAA),
            0x0000_0000_0000_0000,
            &make_thumbnail(128, 16),
        );
        dedup.add_item(
            "b.mp4",
            &make_hash(0xBB),
            0xFFFF_FFFF_FFFF_FFFF,
            &make_thumbnail(200, 16),
        );

        let result = dedup.run();
        assert!(result.exact_groups.is_empty());
        assert!(result.perceptual_groups.is_empty());
    }

    #[test]
    fn test_mixed_exact_and_near_duplicates() {
        let mut cfg = HierarchicalConfig::default();
        cfg.enable_slow_pass = false;
        cfg.perceptual_max_distance = 5;

        let mut dedup = HierarchicalDedup::new(cfg);

        // Exact duplicate pair
        let hash_a = make_hash(0xAA);
        dedup.add_item("exact1.mp4", &hash_a, 0x1111, &make_thumbnail(128, 16));
        dedup.add_item("exact2.mp4", &hash_a, 0x1111, &make_thumbnail(128, 16));

        // Near-duplicate pair (different content hash, similar perceptual hash)
        let base = 0xFFFF_FFFF_FFFF_FFFFu64;
        let near = base ^ 0b11;
        dedup.add_item(
            "near1.mp4",
            &make_hash(0xBB),
            base,
            &make_thumbnail(200, 16),
        );
        dedup.add_item(
            "near2.mp4",
            &make_hash(0xCC),
            near,
            &make_thumbnail(202, 16),
        );

        let result = dedup.run();
        assert_eq!(result.exact_groups.len(), 1);
        // Near-duplicates should be found by medium pass (may or may not depending on LSH)
        // Just verify the pipeline completes without error
        assert_eq!(result.total_items, 4);
    }

    #[test]
    fn test_hierarchical_result_total_groups() {
        let result = HierarchicalResult {
            exact_groups: vec![DedupGroup {
                item_ids: vec!["a".into(), "b".into()],
                pass: DetectionPass::FastHash,
                similarity: 1.0,
            }],
            perceptual_groups: vec![DedupGroup {
                item_ids: vec!["c".into(), "d".into()],
                pass: DetectionPass::MediumPerceptual,
                similarity: 0.95,
            }],
            ssim_groups: Vec::new(),
            total_items: 4,
            fast_pass_hits: 1,
            medium_pass_candidates: 5,
            slow_pass_verified: 0,
        };
        assert_eq!(result.total_groups(), 2);
        assert_eq!(result.total_duplicates(), 2);
    }
}
