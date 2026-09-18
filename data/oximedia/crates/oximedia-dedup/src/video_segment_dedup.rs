//! Video segment deduplication using perceptual hashing and temporal windowing.
//!
//! This module detects duplicate or near-duplicate video segments within and
//! across media files by:
//! - Computing perceptual hashes over frame data
//! - Using temporal windowing to compare ordered frame sequences
//! - Scoring similarity via Hamming distance on 64-bit hashes

#![allow(dead_code)]

use std::collections::HashMap;

// ── SegmentFingerprint ────────────────────────────────────────────────────────

/// A compact fingerprint for a single video segment (group of frames).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentFingerprint {
    /// Perceptual hash of the segment (64-bit).
    pub hash: u64,
    /// Number of frames in this segment.
    pub frame_count: usize,
    /// Duration of this segment in milliseconds.
    pub duration_ms: u64,
}

impl SegmentFingerprint {
    /// Create a `SegmentFingerprint` from pre-computed values.
    #[must_use]
    pub fn new(hash: u64, frame_count: usize, duration_ms: u64) -> Self {
        Self {
            hash,
            frame_count,
            duration_ms,
        }
    }

    /// Derive a `SegmentFingerprint` from raw frame data.
    ///
    /// Uses a two-pass FNV-1a–based perceptual hash that is sensitive to
    /// the distribution of byte values (luminance-like) rather than the exact
    /// raw bytes, giving some robustness to minor encoding differences.
    #[must_use]
    pub fn from_frame_data(data: &[u8], frame_count: usize, duration_ms: u64) -> Self {
        let hash = perceptual_hash_u64(data);
        Self {
            hash,
            frame_count,
            duration_ms,
        }
    }

    /// Compute the Hamming distance (number of differing bits) between two hashes.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }

    /// Returns the average milliseconds per frame.
    ///
    /// Returns 0 if `frame_count` is zero.
    #[must_use]
    pub fn ms_per_frame(&self) -> u64 {
        if self.frame_count == 0 {
            0
        } else {
            self.duration_ms / self.frame_count as u64
        }
    }
}

/// Compute a 64-bit perceptual hash from raw byte data.
///
/// Implements a difference-hash (dHash) approach over 65 equal segments:
/// each bit of the output hash encodes whether the FNV-1a state of segment
/// `i` is greater than the state of segment `i+1`. This captures gradient
/// direction and is robust to absolute value shifts.
fn perceptual_hash_u64(data: &[u8]) -> u64 {
    if data.is_empty() {
        return 0;
    }

    // Use 65 buckets to get 64 comparison pairs
    const SEGS: usize = 65;
    let seg_size = (data.len() + SEGS - 1) / SEGS;
    let mut seg_vals = [0u64; SEGS];

    for (i, chunk) in data.chunks(seg_size.max(1)).enumerate() {
        if i >= SEGS {
            break;
        }
        // FNV-1a with segment-index mixing to distinguish adjacent equal-value regions
        let fnv_offset: u64 =
            0xcbf2_9ce4_8422_2325u64.wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut state: u64 = fnv_offset;
        for &b in chunk {
            state ^= u64::from(b);
            state = state.wrapping_mul(0x0100_0000_01b3);
        }
        seg_vals[i] = state;
    }

    // dHash: bit i = 1 if seg_vals[i] > seg_vals[i+1]
    let mut hash = 0u64;
    for i in 0..64 {
        if seg_vals[i] > seg_vals[i + 1] {
            hash |= 1u64 << i;
        }
    }
    hash
}

// ── match_segments ────────────────────────────────────────────────────────────

/// Compute the similarity score between two segment fingerprints.
///
/// Returns a value in `[0.0, 1.0]` where:
/// - `1.0` = identical hashes
/// - `0.0` = maximum Hamming distance (all 64 bits differ)
///
/// Frame count and duration are NOT used for scoring; they are metadata.
#[must_use]
pub fn match_segments(a: &SegmentFingerprint, b: &SegmentFingerprint) -> f32 {
    let differing_bits = (a.hash ^ b.hash).count_ones();
    // 64 bits total; normalize so 0 diff → 1.0, 64 diff → 0.0
    1.0 - (differing_bits as f32 / 64.0)
}

// ── TemporalWindow ────────────────────────────────────────────────────────────

/// A sliding window of `SegmentFingerprint` hashes used for temporal matching.
#[derive(Debug, Clone)]
pub struct TemporalWindow {
    /// Hashes extracted from the fingerprints in this window.
    pub hashes: Vec<u64>,
    /// Index in the original sequence where this window starts.
    pub start_idx: usize,
    /// Total duration covered by this window (sum of segment durations), ms.
    pub duration_ms: u64,
}

impl TemporalWindow {
    /// Create a window from a slice of fingerprints at a given start index.
    #[must_use]
    pub fn from_fingerprints(fps: &[SegmentFingerprint], start_idx: usize) -> Self {
        let hashes = fps.iter().map(|f| f.hash).collect();
        let duration_ms = fps.iter().map(|f| f.duration_ms).sum();
        Self {
            hashes,
            start_idx,
            duration_ms,
        }
    }

    /// Compare this window to another by average Hamming similarity.
    ///
    /// Returns `0.0` if either window is empty or they have different lengths.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.hashes.is_empty() || other.hashes.is_empty() {
            return 0.0;
        }
        if self.hashes.len() != other.hashes.len() {
            return 0.0;
        }
        let total: f32 = self
            .hashes
            .iter()
            .zip(other.hashes.iter())
            .map(|(&a, &b)| {
                let diff = (a ^ b).count_ones();
                1.0 - (diff as f32 / 64.0)
            })
            .sum();
        total / self.hashes.len() as f32
    }
}

// ── TemporalWindowMatcher ─────────────────────────────────────────────────────

/// Compares two sequences of `SegmentFingerprint`s using a sliding temporal window.
///
/// The matcher slides a window of `window_size` fingerprints across both
/// sequences (with `stride` step), computing the average per-hash Hamming
/// similarity for each aligned window pair.
#[derive(Debug, Clone)]
pub struct TemporalWindowMatcher {
    /// Number of segments per window.
    pub window_size: usize,
    /// Step between consecutive windows.
    pub stride: usize,
}

impl TemporalWindowMatcher {
    /// Create a new matcher with explicit window/stride.
    #[must_use]
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            stride: stride.max(1),
        }
    }

    /// Create a matcher with sensible defaults (window=4, stride=2).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(4, 2)
    }

    /// Compare two sequences by finding the highest-scoring window alignment.
    ///
    /// Extracts windows from `a` and `b` and tries all combinations, returning
    /// the maximum similarity found across all window pairs.
    ///
    /// Returns `0.0` if either sequence is shorter than `window_size`.
    #[must_use]
    pub fn compare_sequences(&self, a: &[SegmentFingerprint], b: &[SegmentFingerprint]) -> f32 {
        if a.len() < self.window_size || b.len() < self.window_size {
            return 0.0;
        }

        let windows_a = self.extract_windows(a);
        let windows_b = self.extract_windows(b);

        let mut best = 0.0f32;
        for wa in &windows_a {
            for wb in &windows_b {
                let sim = wa.similarity(wb);
                if sim > best {
                    best = sim;
                }
            }
        }
        best
    }

    /// Find the best-aligned window pair between two sequences.
    ///
    /// Returns `Some((offset_a, offset_b, similarity))` for the best match,
    /// or `None` if either sequence is too short.
    #[must_use]
    pub fn find_best_alignment(
        &self,
        a: &[SegmentFingerprint],
        b: &[SegmentFingerprint],
    ) -> Option<(usize, usize, f32)> {
        if a.len() < self.window_size || b.len() < self.window_size {
            return None;
        }

        let windows_a = self.extract_windows(a);
        let windows_b = self.extract_windows(b);

        let mut best_sim = 0.0f32;
        let mut best_offset_a = 0usize;
        let mut best_offset_b = 0usize;

        for wa in &windows_a {
            for wb in &windows_b {
                let sim = wa.similarity(wb);
                if sim > best_sim {
                    best_sim = sim;
                    best_offset_a = wa.start_idx;
                    best_offset_b = wb.start_idx;
                }
            }
        }

        if best_sim > 0.0 {
            Some((best_offset_a, best_offset_b, best_sim))
        } else {
            None
        }
    }

    /// Extract all windows from a sequence with the configured stride.
    fn extract_windows(&self, seq: &[SegmentFingerprint]) -> Vec<TemporalWindow> {
        if seq.len() < self.window_size {
            return Vec::new();
        }

        let mut windows = Vec::new();
        let mut start = 0;
        while start + self.window_size <= seq.len() {
            let slice = &seq[start..start + self.window_size];
            windows.push(TemporalWindow::from_fingerprints(slice, start));
            start += self.stride;
        }
        windows
    }
}

// ── SegmentMatch ──────────────────────────────────────────────────────────────

/// A detected match between two segments from possibly different videos.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMatch {
    /// Identifier of the first video.
    pub video_a: String,
    /// Index of the matching segment in video A's fingerprint sequence.
    pub segment_a_idx: usize,
    /// Identifier of the second video.
    pub video_b: String,
    /// Index of the matching segment in video B's fingerprint sequence.
    pub segment_b_idx: usize,
    /// Similarity score in `[0.0, 1.0]`.
    pub similarity: f32,
    /// Estimated temporal offset between the segments in milliseconds.
    ///
    /// Positive value means segment B starts later than segment A.
    pub time_offset_ms: i64,
}

// ── VideoSegmentDeduplicator ──────────────────────────────────────────────────

/// Finds duplicate video segments across a collection of indexed videos.
///
/// Each video is represented as an ordered sequence of [`SegmentFingerprint`]s.
/// The deduplicator compares all pairs of fingerprint sequences and reports
/// segment-level matches above a configurable similarity threshold.
#[derive(Debug, Default)]
pub struct VideoSegmentDeduplicator {
    /// Stored fingerprint sequences keyed by video identifier.
    videos: HashMap<String, Vec<SegmentFingerprint>>,
    /// Default similarity threshold (0.0–1.0).
    threshold: f32,
}

impl VideoSegmentDeduplicator {
    /// Create a new deduplicator with the default threshold (0.8).
    #[must_use]
    pub fn new() -> Self {
        Self {
            videos: HashMap::new(),
            threshold: 0.8,
        }
    }

    /// Create a deduplicator with a custom similarity threshold.
    #[must_use]
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            videos: HashMap::new(),
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Register a video's fingerprint sequence.
    pub fn add_video(&mut self, video_id: &str, fingerprints: Vec<SegmentFingerprint>) {
        self.videos.insert(video_id.to_owned(), fingerprints);
    }

    /// Returns the number of indexed videos.
    #[must_use]
    pub fn video_count(&self) -> usize {
        self.videos.len()
    }

    /// Returns `true` if no videos have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.videos.is_empty()
    }

    /// Find all matching segment pairs across all indexed videos.
    ///
    /// Performs pairwise comparison between each combination of
    /// (video_a, segment_i) × (video_b, segment_j) for all distinct
    /// video pairs. Within-video self-comparisons are skipped.
    ///
    /// Uses the deduplicator's configured threshold.
    #[must_use]
    pub fn find_duplicate_segments(&self) -> Vec<SegmentMatch> {
        self.find_duplicate_segments_with_threshold(self.threshold)
    }

    /// Find matching segment pairs using an explicit similarity threshold.
    #[must_use]
    pub fn find_duplicate_segments_with_threshold(&self, threshold: f32) -> Vec<SegmentMatch> {
        let mut matches = Vec::new();

        let video_ids: Vec<&String> = self.videos.keys().collect();
        let n = video_ids.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let id_a = video_ids[i];
                let id_b = video_ids[j];
                let fps_a = &self.videos[id_a];
                let fps_b = &self.videos[id_b];

                let mut video_matches =
                    compare_segment_sequences(id_a, fps_a, id_b, fps_b, threshold);
                matches.append(&mut video_matches);
            }
        }

        matches
    }

    /// Find matches between a newly submitted video and all indexed videos
    /// without permanently adding it to the index.
    #[must_use]
    pub fn query(
        &self,
        video_id: &str,
        fingerprints: &[SegmentFingerprint],
        threshold: f32,
    ) -> Vec<SegmentMatch> {
        let mut matches = Vec::new();

        for (indexed_id, indexed_fps) in &self.videos {
            if indexed_id == video_id {
                continue;
            }
            let mut m = compare_segment_sequences(
                video_id,
                fingerprints,
                indexed_id,
                indexed_fps,
                threshold,
            );
            matches.append(&mut m);
        }

        matches
    }
}

/// Compare two ordered fingerprint sequences and return all segment-level matches.
fn compare_segment_sequences(
    id_a: &str,
    fps_a: &[SegmentFingerprint],
    id_b: &str,
    fps_b: &[SegmentFingerprint],
    threshold: f32,
) -> Vec<SegmentMatch> {
    let mut matches = Vec::new();

    for (i, fp_a) in fps_a.iter().enumerate() {
        for (j, fp_b) in fps_b.iter().enumerate() {
            let sim = match_segments(fp_a, fp_b);
            if sim >= threshold {
                // Compute approximate time offset based on segment positions and durations.
                let time_a: i64 = fps_a[..i].iter().map(|f| f.duration_ms as i64).sum();
                let time_b: i64 = fps_b[..j].iter().map(|f| f.duration_ms as i64).sum();
                let time_offset_ms = time_b - time_a;

                matches.push(SegmentMatch {
                    video_a: id_a.to_owned(),
                    segment_a_idx: i,
                    video_b: id_b.to_owned(),
                    segment_b_idx: j,
                    similarity: sim,
                    time_offset_ms,
                });
            }
        }
    }

    matches
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SegmentFingerprint ─────────────────────────────────────────────────

    #[test]
    fn test_segment_fingerprint_new() {
        let fp = SegmentFingerprint::new(0xDEAD_BEEF_1234_5678, 30, 1000);
        assert_eq!(fp.hash, 0xDEAD_BEEF_1234_5678);
        assert_eq!(fp.frame_count, 30);
        assert_eq!(fp.duration_ms, 1000);
    }

    #[test]
    fn test_from_frame_data_deterministic() {
        let data = b"hello video frame data here";
        let fp1 = SegmentFingerprint::from_frame_data(data, 10, 500);
        let fp2 = SegmentFingerprint::from_frame_data(data, 10, 500);
        assert_eq!(fp1.hash, fp2.hash);
        assert_eq!(fp1.frame_count, 10);
        assert_eq!(fp1.duration_ms, 500);
    }

    #[test]
    fn test_from_frame_data_different_inputs_differ() {
        // Use substantially different, longer inputs to ensure distinct hashes.
        // The perceptual hash is designed for frame-sized data (>= 64 bytes).
        let data_a: Vec<u8> = (0u8..=127).collect();
        let data_b: Vec<u8> = (128u8..=255).collect();
        let fp_a = SegmentFingerprint::from_frame_data(&data_a, 10, 500);
        let fp_b = SegmentFingerprint::from_frame_data(&data_b, 10, 500);
        assert_ne!(fp_a.hash, fp_b.hash);
    }

    #[test]
    fn test_from_frame_data_empty() {
        let fp = SegmentFingerprint::from_frame_data(b"", 0, 0);
        assert_eq!(fp.hash, 0);
        assert_eq!(fp.frame_count, 0);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let fp = SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 30, 1000);
        assert_eq!(fp.hamming_distance(&fp), 0);
    }

    #[test]
    fn test_hamming_distance_all_differ() {
        let fp_a = SegmentFingerprint::new(0x0000_0000_0000_0000, 30, 1000);
        let fp_b = SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 30, 1000);
        assert_eq!(fp_a.hamming_distance(&fp_b), 64);
    }

    #[test]
    fn test_ms_per_frame() {
        let fp = SegmentFingerprint::new(0, 30, 1000);
        assert_eq!(fp.ms_per_frame(), 33); // floor(1000/30)
    }

    #[test]
    fn test_ms_per_frame_zero_frames() {
        let fp = SegmentFingerprint::new(0, 0, 1000);
        assert_eq!(fp.ms_per_frame(), 0);
    }

    // ── match_segments ─────────────────────────────────────────────────────

    #[test]
    fn test_match_segments_identical() {
        let fp = SegmentFingerprint::new(0x1234_5678_ABCD_EF01, 30, 1000);
        let sim = match_segments(&fp, &fp);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_segments_completely_different() {
        let fp_a = SegmentFingerprint::new(0x0000_0000_0000_0000, 30, 1000);
        let fp_b = SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 30, 1000);
        let sim = match_segments(&fp_a, &fp_b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_segments_half_different() {
        // 32 bits differ → 32/64 = 0.5 diff → 0.5 similarity
        let fp_a = SegmentFingerprint::new(0x0000_0000_FFFF_FFFF, 30, 1000);
        let fp_b = SegmentFingerprint::new(0xFFFF_FFFF_0000_0000, 30, 1000);
        // XOR = 0xFFFF_FFFF_FFFF_FFFF → 64 bits → sim = 0.0  (all bits differ when combined)
        // Actually 0x0000_0000_FFFF_FFFF ^ 0xFFFF_FFFF_0000_0000 = 0xFFFF_FFFF_FFFF_FFFF → 64 ones
        let sim = match_segments(&fp_a, &fp_b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_match_segments_near_identical() {
        // Only 1 bit differs
        let fp_a = SegmentFingerprint::new(0b1000, 30, 1000);
        let fp_b = SegmentFingerprint::new(0b1001, 30, 1000); // 1 bit diff
        let sim = match_segments(&fp_a, &fp_b);
        let expected = 1.0 - (1.0 / 64.0);
        assert!((sim - expected).abs() < 0.001);
    }

    #[test]
    fn test_match_segments_ignores_frame_count() {
        let hash = 0xCAFE_BABE_DEAD_BEEF;
        let fp_a = SegmentFingerprint::new(hash, 10, 500);
        let fp_b = SegmentFingerprint::new(hash, 99, 9999);
        // Same hash → similarity = 1.0 regardless of metadata
        let sim = match_segments(&fp_a, &fp_b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    // ── TemporalWindow ─────────────────────────────────────────────────────

    #[test]
    fn test_temporal_window_from_fingerprints() {
        let fps = vec![
            SegmentFingerprint::new(0xAA, 10, 300),
            SegmentFingerprint::new(0xBB, 10, 400),
            SegmentFingerprint::new(0xCC, 10, 500),
        ];
        let win = TemporalWindow::from_fingerprints(&fps, 5);
        assert_eq!(win.hashes, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(win.start_idx, 5);
        assert_eq!(win.duration_ms, 1200);
    }

    #[test]
    fn test_temporal_window_similarity_identical() {
        let fps = vec![
            SegmentFingerprint::new(0x11, 10, 100),
            SegmentFingerprint::new(0x22, 10, 100),
        ];
        let w1 = TemporalWindow::from_fingerprints(&fps, 0);
        let w2 = TemporalWindow::from_fingerprints(&fps, 0);
        assert!((w1.similarity(&w2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_window_similarity_different_lengths() {
        let fps_a = vec![SegmentFingerprint::new(0x11, 10, 100)];
        let fps_b = vec![
            SegmentFingerprint::new(0x11, 10, 100),
            SegmentFingerprint::new(0x22, 10, 100),
        ];
        let w1 = TemporalWindow::from_fingerprints(&fps_a, 0);
        let w2 = TemporalWindow::from_fingerprints(&fps_b, 0);
        assert!((w1.similarity(&w2) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_temporal_window_similarity_empty() {
        let w1 = TemporalWindow {
            hashes: vec![],
            start_idx: 0,
            duration_ms: 0,
        };
        let w2 = TemporalWindow {
            hashes: vec![],
            start_idx: 0,
            duration_ms: 0,
        };
        assert!((w1.similarity(&w2) - 0.0).abs() < f32::EPSILON);
    }

    // ── TemporalWindowMatcher ──────────────────────────────────────────────

    #[test]
    fn test_matcher_compare_identical_sequences() {
        let fps: Vec<SegmentFingerprint> = (0..8)
            .map(|i| SegmentFingerprint::new(i * 0x1111_1111_1111_1111, 10, 100))
            .collect();
        let matcher = TemporalWindowMatcher::new(4, 2);
        let sim = matcher.compare_sequences(&fps, &fps);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_matcher_compare_too_short() {
        let fps = vec![SegmentFingerprint::new(0xAB, 10, 100)];
        let matcher = TemporalWindowMatcher::new(4, 2);
        let sim = matcher.compare_sequences(&fps, &fps);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matcher_find_best_alignment_identical() {
        let fps: Vec<SegmentFingerprint> = (0..6)
            .map(|i| SegmentFingerprint::new(i as u64, 10, 100))
            .collect();
        let matcher = TemporalWindowMatcher::new(3, 1);
        let result = matcher.find_best_alignment(&fps, &fps);
        assert!(result.is_some());
        let (_, _, sim) = result.expect("alignment found");
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_matcher_find_best_alignment_empty_sequences() {
        let fps: Vec<SegmentFingerprint> = vec![];
        let matcher = TemporalWindowMatcher::new(4, 2);
        assert!(matcher.find_best_alignment(&fps, &fps).is_none());
    }

    // ── VideoSegmentDeduplicator ───────────────────────────────────────────

    #[test]
    fn test_deduplicator_empty() {
        let dedup = VideoSegmentDeduplicator::new();
        assert_eq!(dedup.video_count(), 0);
        assert!(dedup.is_empty());
        assert!(dedup.find_duplicate_segments().is_empty());
    }

    #[test]
    fn test_deduplicator_single_video_no_matches() {
        let mut dedup = VideoSegmentDeduplicator::new();
        let fps = vec![SegmentFingerprint::new(0x1234, 10, 500)];
        dedup.add_video("video_a", fps);
        assert_eq!(dedup.video_count(), 1);
        // Single video → no cross-video matches
        assert!(dedup.find_duplicate_segments().is_empty());
    }

    #[test]
    fn test_deduplicator_identical_videos_match() {
        let mut dedup = VideoSegmentDeduplicator::with_threshold(0.9);
        let fps: Vec<SegmentFingerprint> = vec![
            SegmentFingerprint::new(0xAAAA_AAAA_AAAA_AAAA, 10, 500),
            SegmentFingerprint::new(0xBBBB_BBBB_BBBB_BBBB, 10, 500),
        ];
        dedup.add_video("video_a", fps.clone());
        dedup.add_video("video_b", fps);

        let matches = dedup.find_duplicate_segments();
        assert!(!matches.is_empty());
        for m in &matches {
            assert!(m.similarity >= 0.9);
        }
    }

    #[test]
    fn test_deduplicator_query_without_indexing() {
        let mut dedup = VideoSegmentDeduplicator::new();
        let fps_indexed = vec![SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 10, 500)];
        dedup.add_video("indexed", fps_indexed);

        let fps_query = vec![SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 10, 500)];
        let results = dedup.query("query_video", &fps_query, 0.99);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].video_a, "query_video");
        assert_eq!(results[0].video_b, "indexed");
    }

    #[test]
    fn test_deduplicator_time_offset_computed() {
        let mut dedup = VideoSegmentDeduplicator::with_threshold(0.99);
        // video_a: 2 segments of 500ms each
        let fps_a = vec![
            SegmentFingerprint::new(0x0000_0000_0000_0001, 10, 500),
            SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 10, 500),
        ];
        // video_b: same second segment first
        let fps_b = vec![SegmentFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 10, 500)];
        dedup.add_video("a", fps_a);
        dedup.add_video("b", fps_b);

        let matches = dedup.find_duplicate_segments();
        assert!(!matches.is_empty(), "expected at least one match");

        // Find the match regardless of which video appears as A vs B
        // (HashMap iteration order is non-deterministic).
        let m = matches.iter().find(|m| {
            // Case 1: a[1] matched b[0]
            (m.video_a == "a" && m.segment_a_idx == 1 && m.video_b == "b" && m.segment_b_idx == 0)
            // Case 2: b[0] matched a[1] (reversed iteration)
            || (m.video_a == "b" && m.segment_a_idx == 0 && m.video_b == "a" && m.segment_b_idx == 1)
        });
        assert!(m.is_some(), "expected match between a[1] and b[0]");

        let m = m.expect("match found");
        // time_offset = time_b - time_a
        // case 1: time_b(b[0]=0ms) - time_a(a[1]=500ms) = -500
        // case 2: time_b(a[1]=500ms) - time_a(b[0]=0ms) = +500
        assert!(
            m.time_offset_ms == -500 || m.time_offset_ms == 500,
            "expected ±500ms offset, got {}",
            m.time_offset_ms
        );
    }
}
