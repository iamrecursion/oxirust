//! Segment-level deduplication for media streams.
//!
//! Detects duplicate segments within or across media files using
//! rolling window hashing over frame sequences.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::DedupResult;

/// Hash value representing a media segment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentHash {
    /// Raw hash bytes (32-byte BLAKE3-style)
    data: [u8; 32],
    /// Number of frames covered
    frame_count: usize,
}

impl SegmentHash {
    /// Create a new segment hash from raw bytes and frame count.
    #[must_use]
    pub fn new(data: [u8; 32], frame_count: usize) -> Self {
        Self { data, frame_count }
    }

    /// Construct a segment hash by XOR-folding a byte slice into 32 bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8], frame_count: usize) -> Self {
        // Use a simple FNV-1a–style rolling hash that feeds into the 32-byte
        // output, ensuring different byte sequences produce different hashes
        // even when the same byte value repeats at stride-32 offsets.
        let mut data = [0u8; 32];
        let mut state: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for &b in bytes {
            state ^= u64::from(b);
            state = state.wrapping_mul(0x0100_0000_01b3); // FNV prime
        }
        // Spread the 64-bit state into the first 8 bytes
        let state_bytes = state.to_le_bytes();
        data[..8].copy_from_slice(&state_bytes);
        // Fill the remaining 24 bytes with additional mixing rounds
        for chunk_idx in 1..4u64 {
            state ^= chunk_idx;
            for &b in bytes {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            let s = state.to_le_bytes();
            let offset = chunk_idx as usize * 8;
            data[offset..offset + 8].copy_from_slice(&s);
        }
        Self { data, frame_count }
    }

    /// Returns `true` if this hash matches another within a given
    /// Hamming-bit tolerance.
    #[must_use]
    pub fn is_match(&self, other: &Self, max_diff_bits: u32) -> bool {
        if self.frame_count != other.frame_count {
            return false;
        }
        let diff: u32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        diff <= max_diff_bits
    }

    /// Returns the raw bytes of this hash.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }

    /// Returns the number of frames this hash covers.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for segment-level deduplication.
#[derive(Debug, Clone)]
pub struct SegmentDedupConfig {
    /// Number of frames per dedup window.
    pub window_size_frames: usize,
    /// Step between consecutive windows (stride).
    pub stride_frames: usize,
    /// Minimum number of consecutive matching frames to report as a shared clip.
    pub min_match_length_frames: usize,
    /// Maximum Hamming-bit distance to consider two segments identical.
    pub max_diff_bits: u32,
}

impl Default for SegmentDedupConfig {
    fn default() -> Self {
        Self {
            window_size_frames: 30,
            stride_frames: 15,
            min_match_length_frames: 15,
            max_diff_bits: 4,
        }
    }
}

impl SegmentDedupConfig {
    /// Create a new config with explicit parameters.
    #[must_use]
    pub fn new(window_size_frames: usize, stride_frames: usize, max_diff_bits: u32) -> Self {
        Self {
            window_size_frames,
            stride_frames,
            min_match_length_frames: window_size_frames / 2,
            max_diff_bits,
        }
    }

    /// Returns the window size in frames.
    #[must_use]
    pub fn window_size_frames(&self) -> usize {
        self.window_size_frames
    }

    /// Returns the stride in frames.
    #[must_use]
    pub fn stride_frames(&self) -> usize {
        self.stride_frames
    }

    /// Returns the minimum match length in frames required to emit a result.
    #[must_use]
    pub fn min_match_length_frames(&self) -> usize {
        self.min_match_length_frames
    }

    /// Returns the maximum allowed Hamming-bit difference.
    #[must_use]
    pub fn max_diff_bits(&self) -> u32 {
        self.max_diff_bits
    }
}

// ── Segment record ────────────────────────────────────────────────────────────

/// A record of one segment inserted into the deduplicator.
#[derive(Debug, Clone)]
pub struct SegmentRecord {
    /// Source identifier (e.g. file path or stream ID).
    pub source_id: String,
    /// Frame offset where this segment begins.
    pub frame_offset: usize,
    /// The hash of this segment.
    pub hash: SegmentHash,
}

// ── Deduplicator ──────────────────────────────────────────────────────────────

/// Performs segment-level deduplication over a corpus of media segments.
#[derive(Debug, Default)]
pub struct SegmentDeduplicator {
    config: SegmentDedupConfig,
    /// Map from hash -> list of segment records sharing that hash.
    index: HashMap<[u8; 32], Vec<SegmentRecord>>,
    /// Total unique hashes inserted.
    unique_count: usize,
}

impl SegmentDeduplicator {
    /// Create a new deduplicator with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SegmentDedupConfig::default())
    }

    /// Create a new deduplicator with explicit configuration.
    #[must_use]
    pub fn with_config(config: SegmentDedupConfig) -> Self {
        Self {
            config,
            index: HashMap::new(),
            unique_count: 0,
        }
    }

    /// Add a segment identified by `source_id` starting at `frame_offset`
    /// with the given raw bytes (representing that segment's content).
    pub fn add_segment(&mut self, source_id: &str, frame_offset: usize, bytes: &[u8]) {
        let hash = SegmentHash::from_bytes(bytes, self.config.window_size_frames);
        let key = *hash.as_bytes();
        let is_new = !self.index.contains_key(&key);
        self.index.entry(key).or_default().push(SegmentRecord {
            source_id: source_id.to_string(),
            frame_offset,
            hash,
        });
        if is_new {
            self.unique_count += 1;
        }
    }

    /// Find all duplicate segment groups (groups where 2+ sources share a hash).
    #[must_use]
    pub fn find_duplicates(&self) -> Vec<Vec<&SegmentRecord>> {
        self.index
            .values()
            .filter(|group| group.len() > 1)
            .map(|group| group.iter().collect())
            .collect()
    }

    /// Returns the number of unique segment hashes stored.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.unique_count
    }

    /// Returns the total number of segment records stored.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.index.values().map(Vec::len).sum()
    }

    /// Returns the underlying config.
    #[must_use]
    pub fn config(&self) -> &SegmentDedupConfig {
        &self.config
    }
}

// ── Shared-clip detection ─────────────────────────────────────────────────────

/// A match describing a shared clip found between two video files.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedClipMatch {
    /// First file in the pair.
    pub file_a: PathBuf,
    /// Second file in the pair.
    pub file_b: PathBuf,
    /// Frame offset in `file_a` where the shared clip begins.
    pub offset_a_frames: usize,
    /// Frame offset in `file_b` where the shared clip begins.
    pub offset_b_frames: usize,
    /// Length of the shared clip in frames.
    pub length_frames: usize,
    /// Similarity confidence in [0.0, 1.0].
    pub confidence: f32,
}

/// Compute a compact 64-bit rolling hash over a window of `u64` pHash values.
///
/// Uses a Rabin-fingerprint-style polynomial with multiplication + XOR fold so
/// that the hash changes with every slide.
fn hash_window(frames: &[u64]) -> u64 {
    const BASE: u64 = 0x517c_c1b7_2722_0a95;
    frames
        .iter()
        .fold(0u64, |acc, &h| acc.wrapping_mul(BASE).wrapping_add(h))
}

/// Find shared clips across pairs of pHash frame sequences.
///
/// For each `(file_a, file_b)` pair in `file_pairs`, the function receives
/// the pHash frame sequence for each file via `hash_provider`, then slides a
/// window of `config.window_size_frames` over both sequences in strides of
/// `config.stride_frames`.  Matching windows (by hash value) that can be
/// extended to at least `config.min_match_length_frames` are returned as
/// [`SharedClipMatch`] entries.
///
/// The `hash_provider` closure is called with a file path and must return the
/// ordered sequence of per-frame pHash values for that file.
pub fn find_shared_clips_with_hashes(
    file_pairs: &[(PathBuf, PathBuf)],
    frame_hashes_a: &[Vec<u64>],
    frame_hashes_b: &[Vec<u64>],
    config: &SegmentDedupConfig,
) -> DedupResult<Vec<SharedClipMatch>> {
    assert_eq!(
        file_pairs.len(),
        frame_hashes_a.len(),
        "file_pairs and frame_hashes_a must have the same length"
    );
    assert_eq!(
        file_pairs.len(),
        frame_hashes_b.len(),
        "file_pairs and frame_hashes_b must have the same length"
    );

    let win = config.window_size_frames().max(1);
    let stride = config.stride_frames().max(1);
    let min_len = config.min_match_length_frames().max(1);

    let mut matches = Vec::new();

    for idx in 0..file_pairs.len() {
        let (ref path_a, ref path_b) = file_pairs[idx];
        let hashes_a = &frame_hashes_a[idx];
        let hashes_b = &frame_hashes_b[idx];

        // Build a BTreeMap<window_hash, Vec<offset_in_b>> from file_b's windows.
        let mut index_b: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut off = 0;
        while off + win <= hashes_b.len() {
            let wh = hash_window(&hashes_b[off..off + win]);
            index_b.entry(wh).or_default().push(off);
            off += stride;
        }

        // Slide over file_a and look up matching windows.
        let mut off_a = 0;
        while off_a + win <= hashes_a.len() {
            let wh = hash_window(&hashes_a[off_a..off_a + win]);
            if let Some(b_offsets) = index_b.get(&wh) {
                for &off_b in b_offsets {
                    // Verify element-wise equality (guard against hash collisions).
                    let equal = hashes_a[off_a..off_a + win] == hashes_b[off_b..off_b + win];
                    if !equal {
                        off_a += stride;
                        continue;
                    }

                    // Extend the match as far as frames are identical.
                    let mut length = win;
                    while off_a + length < hashes_a.len()
                        && off_b + length < hashes_b.len()
                        && hashes_a[off_a + length] == hashes_b[off_b + length]
                    {
                        length += 1;
                    }

                    if length >= min_len {
                        // Compute confidence: fraction of matched frames vs pairwise union.
                        let max_possible = hashes_a.len().max(hashes_b.len());
                        #[allow(clippy::cast_precision_loss)]
                        let confidence = (length as f32) / (max_possible as f32).max(1.0);
                        let confidence = confidence.min(1.0_f32);

                        matches.push(SharedClipMatch {
                            file_a: path_a.clone(),
                            file_b: path_b.clone(),
                            offset_a_frames: off_a,
                            offset_b_frames: off_b,
                            length_frames: length,
                            confidence,
                        });
                    }
                }
            }
            off_a += stride;
        }
    }

    Ok(matches)
}

/// Find shared clips across file pairs.
///
/// This is a convenience wrapper that uses a caller-supplied closure to obtain
/// per-frame pHash sequences for each file path, then delegates to
/// [`find_shared_clips_with_hashes`].
pub fn find_shared_clips<F>(
    file_pairs: &[(PathBuf, PathBuf)],
    config: &SegmentDedupConfig,
    mut hash_provider: F,
) -> DedupResult<Vec<SharedClipMatch>>
where
    F: FnMut(&Path) -> DedupResult<Vec<u64>>,
{
    let mut hashes_a = Vec::with_capacity(file_pairs.len());
    let mut hashes_b = Vec::with_capacity(file_pairs.len());

    for (path_a, path_b) in file_pairs {
        hashes_a.push(hash_provider(path_a)?);
        hashes_b.push(hash_provider(path_b)?);
    }

    find_shared_clips_with_hashes(file_pairs, &hashes_a, &hashes_b, config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(byte: u8, frames: usize) -> SegmentHash {
        let mut data = [0u8; 32];
        data[0] = byte;
        SegmentHash::new(data, frames)
    }

    #[test]
    fn test_segment_hash_is_match_exact() {
        let h1 = make_hash(0xAB, 30);
        let h2 = make_hash(0xAB, 30);
        assert!(h1.is_match(&h2, 0));
    }

    #[test]
    fn test_segment_hash_no_match_different_frames() {
        let h1 = make_hash(0xAB, 30);
        let h2 = make_hash(0xAB, 60);
        assert!(!h1.is_match(&h2, 100));
    }

    #[test]
    fn test_segment_hash_hamming_tolerance() {
        let mut d1 = [0u8; 32];
        let mut d2 = [0u8; 32];
        d1[0] = 0b0000_0001;
        d2[0] = 0b0000_0011; // 1-bit difference
        let h1 = SegmentHash::new(d1, 30);
        let h2 = SegmentHash::new(d2, 30);
        assert!(h1.is_match(&h2, 1));
        assert!(!h1.is_match(&h2, 0));
    }

    #[test]
    fn test_segment_hash_from_bytes() {
        let h = SegmentHash::from_bytes(b"hello world", 15);
        assert_eq!(h.frame_count(), 15);
        assert_ne!(h.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_config_window_size_frames() {
        let cfg = SegmentDedupConfig::new(48, 24, 8);
        assert_eq!(cfg.window_size_frames(), 48);
        assert_eq!(cfg.stride_frames(), 24);
        assert_eq!(cfg.max_diff_bits(), 8);
    }

    #[test]
    fn test_config_default() {
        let cfg = SegmentDedupConfig::default();
        assert_eq!(cfg.window_size_frames(), 30);
    }

    #[test]
    fn test_add_segment_unique_count() {
        let mut dedup = SegmentDeduplicator::new();
        dedup.add_segment("source_a", 0, b"segment_content_one");
        dedup.add_segment("source_b", 0, b"segment_content_two");
        assert_eq!(dedup.unique_count(), 2);
    }

    #[test]
    fn test_add_segment_duplicate_increments_total_not_unique() {
        let mut dedup = SegmentDeduplicator::new();
        dedup.add_segment("source_a", 0, b"same_content");
        dedup.add_segment("source_b", 0, b"same_content");
        // same bytes → same hash → unique_count stays 1, total = 2
        assert_eq!(dedup.unique_count(), 1);
        assert_eq!(dedup.total_count(), 2);
    }

    #[test]
    fn test_find_duplicates_empty() {
        let dedup = SegmentDeduplicator::new();
        assert!(dedup.find_duplicates().is_empty());
    }

    #[test]
    fn test_find_duplicates_no_dups() {
        let mut dedup = SegmentDeduplicator::new();
        dedup.add_segment("a", 0, b"aaa");
        dedup.add_segment("b", 0, b"bbb");
        assert!(dedup.find_duplicates().is_empty());
    }

    #[test]
    fn test_find_duplicates_with_dups() {
        let mut dedup = SegmentDeduplicator::new();
        dedup.add_segment("src_a", 0, b"identical_bytes");
        dedup.add_segment("src_b", 0, b"identical_bytes");
        dedup.add_segment("src_c", 30, b"different");
        let dups = dedup.find_duplicates();
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].len(), 2);
    }

    #[test]
    fn test_with_config_preserves_config() {
        let cfg = SegmentDedupConfig::new(60, 30, 2);
        let dedup = SegmentDeduplicator::with_config(cfg);
        assert_eq!(dedup.config().window_size_frames(), 60);
    }

    #[test]
    fn test_segment_record_fields() {
        let mut dedup = SegmentDeduplicator::new();
        dedup.add_segment("my_video.mp4", 120, b"frame_data_xyz");
        // Verify the record is stored correctly.
        let total = dedup.total_count();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_multiple_sources_multiple_segments() {
        let mut dedup = SegmentDeduplicator::new();
        for i in 0u8..5 {
            dedup.add_segment("fileA", (i as usize) * 30, &[i; 64]);
            dedup.add_segment("fileB", (i as usize) * 30, &[i; 64]);
        }
        // 5 unique hashes, 10 total
        assert_eq!(dedup.unique_count(), 5);
        assert_eq!(dedup.total_count(), 10);
        assert_eq!(dedup.find_duplicates().len(), 5);
    }

    // ── find_shared_clips tests ───────────────────────────────────────────────

    /// Helper: build a Vec<u64> from a repeating pattern of values.
    fn phash_seq(values: &[u64], repeat: usize) -> Vec<u64> {
        values
            .iter()
            .cloned()
            .cycle()
            .take(values.len() * repeat)
            .collect()
    }

    #[test]
    fn test_segment_dedup_identical_content_detected() {
        // Both files have the exact same 60-frame pHash sequence.
        let frames: Vec<u64> = (0u64..60).map(|i| i * 0x1234_5678).collect();
        let config = SegmentDedupConfig {
            window_size_frames: 10,
            stride_frames: 5,
            min_match_length_frames: 10,
            max_diff_bits: 0,
        };
        let path_a = PathBuf::from("video_a.mp4");
        let path_b = PathBuf::from("video_b.mp4");
        let pairs = vec![(path_a.clone(), path_b.clone())];

        let result = find_shared_clips_with_hashes(
            &pairs,
            std::slice::from_ref(&frames),
            std::slice::from_ref(&frames),
            &config,
        )
        .expect("find_shared_clips_with_hashes should succeed");

        assert!(!result.is_empty(), "should detect at least one shared clip");
        // The best match should have confidence = 1.0 when sequences are identical.
        let best = result
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .expect("at least one match");
        assert!(
            (best.confidence - 1.0).abs() < f32::EPSILON,
            "confidence should be 1.0 for identical sequences, got {}",
            best.confidence
        );
    }

    #[test]
    fn test_segment_dedup_no_match_returns_empty() {
        // File A and File B have completely different pHash sequences.
        let frames_a: Vec<u64> = (0u64..60).map(|i| i * 0xAAAA_AAAA).collect();
        let frames_b: Vec<u64> = (0u64..60).map(|i| i * 0x5555_5555 + 1).collect();

        let config = SegmentDedupConfig {
            window_size_frames: 10,
            stride_frames: 5,
            min_match_length_frames: 10,
            max_diff_bits: 0,
        };
        let pairs = vec![(PathBuf::from("unique_a.mp4"), PathBuf::from("unique_b.mp4"))];

        let result = find_shared_clips_with_hashes(&pairs, &[frames_a], &[frames_b], &config)
            .expect("find_shared_clips_with_hashes should succeed");

        assert!(
            result.is_empty(),
            "completely different sequences should produce no matches"
        );
    }
}
