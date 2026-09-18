//! Copy detection for identifying duplicate or near-duplicate media.
//!
//! Provides fingerprint-based matching, partial copy detection, and
//! similarity scoring to detect when media has been copied or repurposed.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A perceptual hash fingerprint for media content
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaFingerprint {
    /// 64-bit perceptual hash
    pub hash: u64,
    /// Source identifier
    pub source_id: String,
    /// Timestamp in the source (milliseconds)
    pub timestamp_ms: u64,
    /// Fingerprint type
    pub fp_type: FingerprintType,
}

impl MediaFingerprint {
    /// Create a new fingerprint
    #[must_use]
    pub fn new(
        hash: u64,
        source_id: impl Into<String>,
        timestamp_ms: u64,
        fp_type: FingerprintType,
    ) -> Self {
        Self {
            hash,
            source_id: source_id.into(),
            timestamp_ms,
            fp_type,
        }
    }

    /// Compute Hamming distance to another fingerprint
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }

    /// Similarity score (0.0 = completely different, 1.0 = identical)
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let dist = self.hamming_distance(other);
        1.0 - dist as f64 / 64.0
    }
}

/// Type of media fingerprint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FingerprintType {
    /// Perceptual hash of video frame
    VideoFrame,
    /// Audio fingerprint segment
    AudioSegment,
    /// Combined A/V fingerprint
    Combined,
    /// Color histogram hash
    ColorHistogram,
    /// DCT-based hash
    DctHash,
}

/// A detected copy match between two media segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyMatch {
    /// Source fingerprint
    pub source: MediaFingerprint,
    /// Query fingerprint that matched
    pub query: MediaFingerprint,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Whether this is a partial copy
    pub is_partial: bool,
    /// Estimated transformation applied (if any)
    pub transformation: Option<CopyTransformation>,
}

impl CopyMatch {
    /// Create a new copy match
    #[must_use]
    pub fn new(source: MediaFingerprint, query: MediaFingerprint) -> Self {
        let similarity = source.similarity(&query);
        let is_partial = similarity > 0.7 && similarity < 1.0;
        Self {
            source,
            query,
            similarity,
            is_partial,
            transformation: None,
        }
    }

    /// Is this an exact or near-exact copy?
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.similarity >= 0.95
    }
}

/// Transformation applied during copying
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTransformation {
    /// Was the image cropped?
    pub cropped: bool,
    /// Was the image scaled?
    pub scaled: bool,
    /// Was color grading applied?
    pub color_graded: bool,
    /// Was a watermark added?
    pub watermarked: bool,
    /// Estimated compression applied (0 = none, 100 = heavy)
    pub compression_level: u8,
}

impl Default for CopyTransformation {
    fn default() -> Self {
        Self {
            cropped: false,
            scaled: false,
            color_graded: false,
            watermarked: false,
            compression_level: 0,
        }
    }
}

/// Database of fingerprints for matching
#[derive(Debug, Clone)]
pub struct FingerprintDatabase {
    /// Stored fingerprints indexed by source_id
    fingerprints: HashMap<String, Vec<MediaFingerprint>>,
    /// Total count
    total_count: usize,
}

impl FingerprintDatabase {
    /// Create a new empty database
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprints: HashMap::new(),
            total_count: 0,
        }
    }

    /// Add a fingerprint to the database
    pub fn add(&mut self, fp: MediaFingerprint) {
        self.fingerprints
            .entry(fp.source_id.clone())
            .or_default()
            .push(fp);
        self.total_count += 1;
    }

    /// Add multiple fingerprints
    pub fn add_all(&mut self, fps: impl IntoIterator<Item = MediaFingerprint>) {
        for fp in fps {
            self.add(fp);
        }
    }

    /// Total number of fingerprints
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_count
    }

    /// Returns true if database is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Number of distinct source IDs
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.fingerprints.len()
    }

    /// Get all fingerprints for a source
    #[must_use]
    pub fn get_source(&self, source_id: &str) -> Option<&Vec<MediaFingerprint>> {
        self.fingerprints.get(source_id)
    }

    /// Remove all fingerprints for a source
    pub fn remove_source(&mut self, source_id: &str) -> usize {
        if let Some(fps) = self.fingerprints.remove(source_id) {
            let count = fps.len();
            self.total_count -= count;
            count
        } else {
            0
        }
    }

    /// Find all matching fingerprints for a query
    #[must_use]
    pub fn find_matches(&self, query: &MediaFingerprint, threshold: f64) -> Vec<CopyMatch> {
        let mut matches = Vec::new();

        for fps in self.fingerprints.values() {
            for fp in fps {
                if fp.source_id == query.source_id {
                    continue; // Skip self-matches
                }
                let sim = fp.similarity(query);
                if sim >= threshold {
                    let mut m = CopyMatch::new(fp.clone(), query.clone());
                    m.similarity = sim;
                    matches.push(m);
                }
            }
        }

        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }
}

impl Default for FingerprintDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Partial copy detector - finds when a portion of media appears in another
#[derive(Debug, Clone)]
pub struct PartialCopyDetector {
    /// Minimum similarity to consider a partial copy
    pub min_similarity: f64,
    /// Minimum fraction of the query that must match
    pub min_coverage: f64,
    /// Window size for sliding window analysis (in fingerprint count)
    pub window_size: usize,
}

impl PartialCopyDetector {
    /// Create a new partial copy detector
    #[must_use]
    pub fn new(min_similarity: f64, min_coverage: f64, window_size: usize) -> Self {
        Self {
            min_similarity,
            min_coverage,
            window_size,
        }
    }

    /// Find partial copies of `query_fps` within `database_fps`
    /// Returns a list of (start_idx, end_idx, similarity) matches
    #[must_use]
    pub fn find_partial_copies(
        &self,
        query_fps: &[MediaFingerprint],
        database_fps: &[MediaFingerprint],
    ) -> Vec<PartialCopyRegion> {
        let mut regions = Vec::new();

        if query_fps.is_empty() || database_fps.is_empty() {
            return regions;
        }

        let w = self
            .window_size
            .min(query_fps.len())
            .min(database_fps.len());
        if w == 0 {
            return regions;
        }

        for query_start in 0..=(query_fps.len().saturating_sub(w)) {
            for db_start in 0..=(database_fps.len().saturating_sub(w)) {
                let sim = window_similarity(
                    &query_fps[query_start..query_start + w],
                    &database_fps[db_start..db_start + w],
                );

                if sim >= self.min_similarity {
                    let coverage = w as f64 / query_fps.len() as f64;
                    if coverage >= self.min_coverage {
                        regions.push(PartialCopyRegion {
                            query_start,
                            query_end: query_start + w,
                            db_start,
                            db_end: db_start + w,
                            similarity: sim,
                            coverage,
                        });
                    }
                }
            }
        }

        // Merge overlapping regions
        merge_regions(regions)
    }
}

impl Default for PartialCopyDetector {
    fn default() -> Self {
        Self::new(0.75, 0.1, 10)
    }
}

/// A region identified as a partial copy
#[derive(Debug, Clone)]
pub struct PartialCopyRegion {
    /// Start index in query fingerprints
    pub query_start: usize,
    /// End index in query fingerprints
    pub query_end: usize,
    /// Start index in database fingerprints
    pub db_start: usize,
    /// End index in database fingerprints
    pub db_end: usize,
    /// Average similarity in this region
    pub similarity: f64,
    /// Fraction of query covered
    pub coverage: f64,
}

impl PartialCopyRegion {
    /// Length of the matching region
    #[must_use]
    pub fn len(&self) -> usize {
        self.query_end - self.query_start
    }

    /// Returns true if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.query_start >= self.query_end
    }
}

/// Compute average similarity across a window of fingerprint pairs
fn window_similarity(a: &[MediaFingerprint], b: &[MediaFingerprint]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let total: f64 = a.iter().zip(b.iter()).map(|(x, y)| x.similarity(y)).sum();
    total / n as f64
}

/// Merge overlapping partial copy regions
fn merge_regions(mut regions: Vec<PartialCopyRegion>) -> Vec<PartialCopyRegion> {
    if regions.len() <= 1 {
        return regions;
    }

    regions.sort_by_key(|r| r.query_start);
    let mut merged = Vec::new();
    let mut current = regions.remove(0);

    for next in regions {
        if next.query_start < current.query_end {
            // Overlapping: merge
            current.query_end = current.query_end.max(next.query_end);
            current.db_end = current.db_end.max(next.db_end);
            current.similarity = (current.similarity + next.similarity) / 2.0;
            current.coverage = current.coverage.max(next.coverage);
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
}

// ─── Spatial-hash block copy detection ───────────────────────────────────────

/// A spatial block with integer grid coordinates and a hash fingerprint.
///
/// `x` and `y` are **block-grid** coordinates (i.e. block index, not pixel
/// coordinates).  The `hash` may be any perceptual or content hash of the
/// block's pixel data.
#[derive(Debug, Clone)]
pub struct SpatialBlock {
    /// Block column index.
    pub x: i32,
    /// Block row index.
    pub y: i32,
    /// Content hash of this block (e.g. dHash, pHash, etc.).
    pub hash: u64,
}

impl SpatialBlock {
    /// Create a new spatial block.
    #[must_use]
    pub fn new(x: i32, y: i32, hash: u64) -> Self {
        Self { x, y, hash }
    }

    /// Hamming distance to another block's hash.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }
}

/// A matched pair of spatial blocks that are considered copies.
#[derive(Debug, Clone)]
pub struct BlockCopyMatch {
    /// Index of the first block in the input slice.
    pub index_a: usize,
    /// Index of the second block in the input slice.
    pub index_b: usize,
    /// Hamming distance between the two block hashes.
    pub hamming_distance: u32,
}

/// Spatial-hash grid used internally by [`BlockSpatialMatcher`].
///
/// Blocks are bucketed by `(x, y)` block-grid position so that the
/// neighbor-search window only touches blocks within Manhattan distance 1
/// in block-grid space rather than the entire corpus.
struct BlockGrid {
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl BlockGrid {
    fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    fn insert(&mut self, bx: i32, by: i32, idx: usize) {
        self.cells.entry((bx, by)).or_default().push(idx);
    }

    /// Iterate over all block indices in the 3×3 neighborhood of `(bx, by)`.
    fn neighbors(&self, bx: i32, by: i32) -> impl Iterator<Item = usize> + '_ {
        (-1i32..=1).flat_map(move |dx| {
            (-1i32..=1).flat_map(move |dy| {
                self.cells
                    .get(&(bx + dx, by + dy))
                    .into_iter()
                    .flat_map(|v| v.iter().copied())
            })
        })
    }
}

/// Block-level copy matcher using a spatial hash grid for O(n) neighbor lookup.
///
/// Rather than comparing every block against every other block (O(n²)), blocks
/// are indexed into a grid keyed by their `(x, y)` block-grid position.  Only
/// blocks that share the same 3×3 grid cell neighborhood are compared, which
/// keeps the comparison count proportional to n for typical media content.
pub struct BlockSpatialMatcher {
    /// Maximum Hamming distance at which two blocks are considered copies
    /// (default 5 — approximately 92 % bit-similarity for a 64-bit hash).
    pub max_hamming_distance: u32,
}

impl Default for BlockSpatialMatcher {
    fn default() -> Self {
        Self {
            max_hamming_distance: 5,
        }
    }
}

impl BlockSpatialMatcher {
    /// Create a new matcher with the default Hamming distance threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a matcher with a custom Hamming distance threshold.
    #[must_use]
    pub fn with_threshold(max_hamming_distance: u32) -> Self {
        Self {
            max_hamming_distance,
        }
    }

    /// Find copy-pairs within `blocks` using the spatial hash grid.
    ///
    /// Returns every `(i, j)` pair where `j > i`, both blocks are within the
    /// 3×3 block-grid neighborhood, and `hamming(blocks[i], blocks[j]) <=
    /// self.max_hamming_distance`.
    #[must_use]
    pub fn find_copies(&self, blocks: &[SpatialBlock]) -> Vec<BlockCopyMatch> {
        let mut grid = BlockGrid::new();
        for (i, block) in blocks.iter().enumerate() {
            grid.insert(block.x, block.y, i);
        }

        let mut matches = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            for j in grid.neighbors(block.x, block.y) {
                if j <= i {
                    continue;
                }
                let dist = block.hamming_distance(&blocks[j]);
                if dist <= self.max_hamming_distance {
                    matches.push(BlockCopyMatch {
                        index_a: i,
                        index_b: j,
                        hamming_distance: dist,
                    });
                }
            }
        }

        matches
    }
}

// ─── Overall copy detection report ───────────────────────────────────────────

/// Overall copy detection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyDetectionReport {
    /// Source being analyzed
    pub source_id: String,
    /// Number of fingerprints analyzed
    pub fingerprint_count: usize,
    /// Exact copy matches found
    pub exact_matches: Vec<String>,
    /// Partial copy regions found
    #[serde(skip)]
    pub partial_regions: Vec<PartialCopyRegion>,
    /// Overall similarity score (0.0 = unique, 1.0 = exact copy)
    pub similarity_score: f64,
    /// Whether this content is considered a copy
    pub is_copy: bool,
}

impl CopyDetectionReport {
    /// Create a new report
    #[must_use]
    pub fn new(source_id: impl Into<String>, fingerprint_count: usize) -> Self {
        Self {
            source_id: source_id.into(),
            fingerprint_count,
            exact_matches: Vec::new(),
            partial_regions: Vec::new(),
            similarity_score: 0.0,
            is_copy: false,
        }
    }

    /// Finalize the report by computing the overall assessment
    pub fn finalize(&mut self) {
        // Compute score from exact matches and partial regions
        let exact_score: f64 = if self.exact_matches.is_empty() {
            0.0
        } else {
            1.0
        };
        let partial_score = self
            .partial_regions
            .iter()
            .map(|r| r.similarity * r.coverage)
            .fold(0.0_f64, f64::max);

        self.similarity_score = exact_score.max(partial_score);
        self.is_copy = self.similarity_score > 0.7 || !self.exact_matches.is_empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp(hash: u64, source: &str, ts: u64) -> MediaFingerprint {
        MediaFingerprint::new(hash, source, ts, FingerprintType::VideoFrame)
    }

    #[test]
    fn test_fingerprint_creation() {
        let fp = make_fp(0xDEADBEEF, "source1", 1000);
        assert_eq!(fp.hash, 0xDEADBEEF);
        assert_eq!(fp.source_id, "source1");
        assert_eq!(fp.timestamp_ms, 1000);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let fp = make_fp(0xFF00FF00, "s1", 0);
        assert_eq!(fp.hamming_distance(&fp), 0);
    }

    #[test]
    fn test_hamming_distance_one_bit() {
        let fp1 = make_fp(0b0000_0001u64, "s1", 0);
        let fp2 = make_fp(0b0000_0000u64, "s2", 0);
        assert_eq!(fp1.hamming_distance(&fp2), 1);
    }

    #[test]
    fn test_similarity_identical() {
        let fp = make_fp(12345, "s1", 0);
        assert!((fp.similarity(&fp) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_all_different_bits() {
        let fp1 = make_fp(0u64, "s1", 0);
        let fp2 = make_fp(u64::MAX, "s2", 0);
        assert!((fp1.similarity(&fp2)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_copy_match_creation() {
        let fp1 = make_fp(0xFF00u64, "s1", 0);
        let fp2 = make_fp(0xFF00u64, "s2", 0);
        let m = CopyMatch::new(fp1, fp2);
        assert!((m.similarity - 1.0).abs() < f64::EPSILON);
        assert!(m.is_exact());
    }

    #[test]
    fn test_copy_match_not_exact() {
        let fp1 = make_fp(0b0000_0000_1111_1111u64, "s1", 0);
        let fp2 = make_fp(0b0000_0000_0000_1111u64, "s2", 0);
        let m = CopyMatch::new(fp1, fp2);
        assert!(!m.is_exact());
    }

    #[test]
    fn test_fingerprint_database_empty() {
        let db = FingerprintDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.source_count(), 0);
    }

    #[test]
    fn test_fingerprint_database_add() {
        let mut db = FingerprintDatabase::new();
        db.add(make_fp(100, "s1", 0));
        db.add(make_fp(200, "s1", 1000));
        db.add(make_fp(300, "s2", 0));
        assert_eq!(db.len(), 3);
        assert_eq!(db.source_count(), 2);
    }

    #[test]
    fn test_fingerprint_database_find_exact_match() {
        let mut db = FingerprintDatabase::new();
        db.add(make_fp(0xABCDu64, "reference", 0));
        let query = make_fp(0xABCDu64, "query", 0);
        let matches = db.find_matches(&query, 0.9);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].is_exact());
    }

    #[test]
    fn test_fingerprint_database_no_self_match() {
        let mut db = FingerprintDatabase::new();
        db.add(make_fp(0xABCDu64, "s1", 0));
        let query = make_fp(0xABCDu64, "s1", 0); // same source_id
        let matches = db.find_matches(&query, 0.5);
        assert_eq!(matches.len(), 0); // Self-match filtered
    }

    #[test]
    fn test_fingerprint_database_remove_source() {
        let mut db = FingerprintDatabase::new();
        db.add(make_fp(1, "s1", 0));
        db.add(make_fp(2, "s1", 1));
        db.add(make_fp(3, "s2", 0));
        let removed = db.remove_source("s1");
        assert_eq!(removed, 2);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_partial_copy_detector_no_copies() {
        let detector = PartialCopyDetector::new(0.95, 0.1, 3);
        let query: Vec<_> = (0..5u64).map(|i| make_fp(i * 1000, "q", i)).collect();
        let db: Vec<_> = (0..5u64).map(|i| make_fp(i, "db", i)).collect();
        // Very different hashes, should find no partial copies at 95% threshold
        let regions = detector.find_partial_copies(&query, &db);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_partial_copy_detector_exact_copy() {
        let detector = PartialCopyDetector::new(0.9, 0.2, 3);
        let fps: Vec<_> = (0..10u64)
            .map(|i| make_fp(0xAAAA_0000 + i, "q", i))
            .collect();
        let db_fps: Vec<_> = (0..10u64)
            .map(|i| make_fp(0xAAAA_0000 + i, "db", i))
            .collect();
        let regions = detector.find_partial_copies(&fps, &db_fps);
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_partial_copy_region_len() {
        let region = PartialCopyRegion {
            query_start: 2,
            query_end: 7,
            db_start: 0,
            db_end: 5,
            similarity: 0.9,
            coverage: 0.5,
        };
        assert_eq!(region.len(), 5);
        assert!(!region.is_empty());
    }

    #[test]
    fn test_copy_detection_report_finalize_no_copies() {
        let mut report = CopyDetectionReport::new("test", 100);
        report.finalize();
        assert!(!report.is_copy);
        assert_eq!(report.similarity_score, 0.0);
    }

    #[test]
    fn test_copy_detection_report_finalize_exact_match() {
        let mut report = CopyDetectionReport::new("test", 100);
        report.exact_matches.push("reference_source".to_string());
        report.finalize();
        assert!(report.is_copy);
        assert!((report.similarity_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_similarity_identical() {
        let fps_a: Vec<_> = vec![make_fp(100, "a", 0), make_fp(200, "a", 1)];
        let fps_b: Vec<_> = vec![make_fp(100, "b", 0), make_fp(200, "b", 1)];
        let sim = window_similarity(&fps_a, &fps_b);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_copy_transformation_default() {
        let t = CopyTransformation::default();
        assert!(!t.cropped);
        assert!(!t.scaled);
        assert_eq!(t.compression_level, 0);
    }

    // ── BlockSpatialMatcher tests ─────────────────────────────────────────────

    /// Build a corpus of blocks at regularly-spaced grid positions with known
    /// identical pairs (same hash, adjacent positions).
    fn make_blocks(n: usize, identical_stride: usize) -> Vec<SpatialBlock> {
        (0..n)
            .map(|i| {
                let x = (i % 10) as i32;
                let y = (i / 10) as i32;
                // Every `identical_stride`-th block gets the same hash as its
                // predecessor so it forms a known copy-pair.
                let hash = if identical_stride > 0 && i > 0 && i % identical_stride == 0 {
                    (i - 1) as u64 * 17
                } else {
                    i as u64 * 17
                };
                SpatialBlock::new(x, y, hash)
            })
            .collect()
    }

    /// Brute-force reference: compare all (i, j) pairs with j > i.
    fn brute_force_matches(blocks: &[SpatialBlock], max_dist: u32) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..blocks.len() {
            for j in i + 1..blocks.len() {
                if blocks[i].hamming_distance(&blocks[j]) <= max_dist {
                    out.push((i, j));
                }
            }
        }
        out
    }

    #[test]
    fn test_copy_detect_spatial_matches_brute_force() {
        // 100-block corpus with known identical pairs every 5 blocks.
        let blocks = make_blocks(100, 5);
        let matcher = BlockSpatialMatcher::with_threshold(0); // exact matches only

        let spatial: std::collections::HashSet<(usize, usize)> = matcher
            .find_copies(&blocks)
            .into_iter()
            .map(|m| (m.index_a, m.index_b))
            .collect();

        let brute: std::collections::HashSet<(usize, usize)> =
            brute_force_matches(&blocks, 0).into_iter().collect();

        // Spatial results must contain every brute-force match that falls
        // within the 3×3 neighborhood; since our blocks are laid out in a
        // 10-wide grid and identical pairs are consecutive, they always share
        // a grid cell.  We check that for every brute-force hit the match is
        // also found by the spatial matcher (superset check restricted to
        // close neighbors).
        let neighbors_only: std::collections::HashSet<(usize, usize)> = brute
            .iter()
            .copied()
            .filter(|&(a, b)| {
                let dx = (blocks[a].x - blocks[b].x).abs();
                let dy = (blocks[a].y - blocks[b].y).abs();
                dx <= 1 && dy <= 1
            })
            .collect();

        for pair in &neighbors_only {
            assert!(
                spatial.contains(pair),
                "Spatial matcher missed brute-force pair {:?}",
                pair
            );
        }
    }

    #[test]
    fn test_copy_detect_10k_blocks_completes() {
        use std::time::Instant;

        // 10 000 blocks spread over a 100×100 grid — no intentional copies.
        let blocks: Vec<SpatialBlock> = (0..10_000usize)
            .map(|i| SpatialBlock::new((i % 100) as i32, (i / 100) as i32, i as u64 * 31337))
            .collect();

        let matcher = BlockSpatialMatcher::with_threshold(3);
        let start = Instant::now();
        let _matches = matcher.find_copies(&blocks);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs() < 2,
            "10 000-block spatial match should complete in < 2 s, took {:.2?}",
            elapsed
        );
    }

    #[test]
    fn test_spatial_block_hamming_distance() {
        let a = SpatialBlock::new(0, 0, 0b0000_1111u64);
        let b = SpatialBlock::new(0, 0, 0b0000_0000u64);
        assert_eq!(a.hamming_distance(&b), 4);
    }

    #[test]
    fn test_block_spatial_matcher_no_copies() {
        // Blocks with maximally different hashes should not match.
        let blocks: Vec<SpatialBlock> = (0..20usize)
            .map(|i| {
                SpatialBlock::new(
                    (i % 5) as i32,
                    (i / 5) as i32,
                    i as u64 * 0x0101_0101_0101_0101,
                )
            })
            .collect();
        let matcher = BlockSpatialMatcher::with_threshold(2);
        let matches = matcher.find_copies(&blocks);
        // Very spread-out hashes → unlikely to match within tight Hamming bound.
        for m in &matches {
            assert!(
                m.hamming_distance <= 2,
                "Reported match exceeds threshold: dist {}",
                m.hamming_distance
            );
        }
    }
}
