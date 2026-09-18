#![allow(dead_code)]
//! Clip fingerprinting for deduplication and identification.
//!
//! This module computes perceptual fingerprints for video clips, enabling
//! duplicate detection, similarity matching, and content identification.
//! Fingerprints are computed from visual features (color histograms, edge
//! patterns) and temporal structure (scene transitions, motion patterns).

use std::collections::HashMap;

/// Size of each fingerprint block in bytes.
const BLOCK_SIZE: usize = 32;

/// A perceptual fingerprint for a single video frame.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrameFingerprint {
    /// Compact hash of the frame's visual features.
    pub hash: [u8; BLOCK_SIZE],
    /// Frame number within the clip.
    pub frame_number: u64,
}

impl FrameFingerprint {
    /// Creates a new frame fingerprint from a hash and frame number.
    pub fn new(hash: [u8; BLOCK_SIZE], frame_number: u64) -> Self {
        Self { hash, frame_number }
    }

    /// Creates a fingerprint from a raw byte slice (truncated or zero-padded to `BLOCK_SIZE`).
    pub fn from_bytes(data: &[u8], frame_number: u64) -> Self {
        let mut hash = [0u8; BLOCK_SIZE];
        let len = data.len().min(BLOCK_SIZE);
        hash[..len].copy_from_slice(&data[..len]);
        Self { hash, frame_number }
    }

    /// Computes the Hamming distance between two fingerprints.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.hash
            .iter()
            .zip(other.hash.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Returns whether two fingerprints are considered similar (distance below threshold).
    pub fn is_similar(&self, other: &Self, threshold: u32) -> bool {
        self.hamming_distance(other) <= threshold
    }
}

/// A complete clip fingerprint composed of sampled frame fingerprints.
#[derive(Debug, Clone)]
pub struct ClipFingerprint {
    /// Clip identifier.
    pub clip_id: String,
    /// Ordered frame fingerprints sampled from the clip.
    pub frames: Vec<FrameFingerprint>,
    /// Duration of the clip in milliseconds.
    pub duration_ms: u64,
    /// Frame rate used during fingerprinting.
    pub sample_fps: f64,
}

impl ClipFingerprint {
    /// Creates a new empty clip fingerprint.
    pub fn new(clip_id: &str, duration_ms: u64, sample_fps: f64) -> Self {
        Self {
            clip_id: clip_id.to_string(),
            frames: Vec::new(),
            duration_ms,
            sample_fps,
        }
    }

    /// Adds a frame fingerprint.
    pub fn add_frame(&mut self, fp: FrameFingerprint) {
        self.frames.push(fp);
    }

    /// Returns the number of sampled frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Computes the average Hamming distance to another clip fingerprint.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_distance(&self, other: &Self) -> f64 {
        let count = self.frames.len().min(other.frames.len());
        if count == 0 {
            return f64::MAX;
        }
        let total: u64 = self
            .frames
            .iter()
            .zip(other.frames.iter())
            .map(|(a, b)| u64::from(a.hamming_distance(b)))
            .sum();
        total as f64 / count as f64
    }

    /// Determines whether this clip is a likely duplicate of another.
    pub fn is_duplicate(&self, other: &Self, distance_threshold: f64) -> bool {
        self.average_distance(other) <= distance_threshold
    }

    /// Finds the best matching offset between two clips (sliding window match).
    #[allow(clippy::cast_precision_loss)]
    pub fn find_best_offset(&self, other: &Self) -> (i64, f64) {
        if self.frames.is_empty() || other.frames.is_empty() {
            return (0, f64::MAX);
        }
        let max_offset = self.frames.len().max(other.frames.len()) as i64;
        let mut best_offset: i64 = 0;
        let mut best_distance = f64::MAX;

        for offset in -max_offset..=max_offset {
            let mut total_dist = 0u64;
            let mut count = 0u64;
            for (i, fp_a) in self.frames.iter().enumerate() {
                let j = i as i64 + offset;
                if j >= 0 && (j as usize) < other.frames.len() {
                    total_dist += u64::from(fp_a.hamming_distance(&other.frames[j as usize]));
                    count += 1;
                }
            }
            if count > 0 {
                let avg = total_dist as f64 / count as f64;
                if avg < best_distance {
                    best_distance = avg;
                    best_offset = offset;
                }
            }
        }

        (best_offset, best_distance)
    }
}

/// Similarity result between two clips.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// ID of clip A.
    pub clip_a: String,
    /// ID of clip B.
    pub clip_b: String,
    /// Average frame distance.
    pub distance: f64,
    /// Similarity score (0.0-1.0, higher = more similar).
    pub similarity: f64,
    /// Whether the clips are considered duplicates.
    pub is_duplicate: bool,
}

/// A fingerprint database for matching clips.
#[derive(Debug)]
pub struct FingerprintDb {
    /// Stored clip fingerprints keyed by clip ID.
    entries: HashMap<String, ClipFingerprint>,
    /// Similarity threshold for duplicate detection.
    duplicate_threshold: f64,
}

impl FingerprintDb {
    /// Creates a new fingerprint database.
    pub fn new(duplicate_threshold: f64) -> Self {
        Self {
            entries: HashMap::new(),
            duplicate_threshold,
        }
    }

    /// Creates a database with a default threshold.
    pub fn with_default_threshold() -> Self {
        Self::new(8.0)
    }

    /// Inserts a clip fingerprint into the database.
    pub fn insert(&mut self, fp: ClipFingerprint) {
        self.entries.insert(fp.clip_id.clone(), fp);
    }

    /// Returns the number of fingerprints in the database.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up a fingerprint by clip ID.
    pub fn get(&self, clip_id: &str) -> Option<&ClipFingerprint> {
        self.entries.get(clip_id)
    }

    /// Finds all clips similar to the query clip.
    #[allow(clippy::cast_precision_loss)]
    pub fn find_similar(&self, query: &ClipFingerprint) -> Vec<SimilarityResult> {
        let max_bits = (BLOCK_SIZE * 8) as f64;
        self.entries
            .values()
            .filter(|stored| stored.clip_id != query.clip_id)
            .map(|stored| {
                let distance = query.average_distance(stored);
                let similarity = (1.0 - distance / max_bits).max(0.0);
                let is_dup = distance <= self.duplicate_threshold;
                SimilarityResult {
                    clip_a: query.clip_id.clone(),
                    clip_b: stored.clip_id.clone(),
                    distance,
                    similarity,
                    is_duplicate: is_dup,
                }
            })
            .collect()
    }

    /// Removes a fingerprint from the database.
    pub fn remove(&mut self, clip_id: &str) -> bool {
        self.entries.remove(clip_id).is_some()
    }
}

/// Computes a simple perceptual hash from pixel luminance values.
pub fn compute_phash(luminance: &[u8], width: usize, height: usize) -> [u8; BLOCK_SIZE] {
    let mut hash = [0u8; BLOCK_SIZE];
    if luminance.is_empty() || width == 0 || height == 0 {
        return hash;
    }

    #[allow(clippy::cast_precision_loss)]
    let mean: f64 = luminance.iter().map(|&v| f64::from(v)).sum::<f64>() / luminance.len() as f64;

    // Generate hash bits: each bit is 1 if pixel > mean, 0 otherwise
    for (i, &pixel) in luminance.iter().enumerate() {
        if i / 8 >= BLOCK_SIZE {
            break;
        }
        if f64::from(pixel) > mean {
            hash[i / 8] |= 1 << (i % 8);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_fingerprint_new() {
        let hash = [0u8; BLOCK_SIZE];
        let fp = FrameFingerprint::new(hash, 42);
        assert_eq!(fp.frame_number, 42);
        assert_eq!(fp.hash, [0u8; BLOCK_SIZE]);
    }

    #[test]
    fn test_frame_fingerprint_from_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let fp = FrameFingerprint::from_bytes(&data, 0);
        assert_eq!(fp.hash[0], 1);
        assert_eq!(fp.hash[4], 5);
        assert_eq!(fp.hash[5], 0);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let fp_a = FrameFingerprint::new([0xFF; BLOCK_SIZE], 0);
        let fp_b = FrameFingerprint::new([0xFF; BLOCK_SIZE], 1);
        assert_eq!(fp_a.hamming_distance(&fp_b), 0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let fp_a = FrameFingerprint::new([0x00; BLOCK_SIZE], 0);
        let fp_b = FrameFingerprint::new([0xFF; BLOCK_SIZE], 1);
        assert_eq!(fp_a.hamming_distance(&fp_b), (BLOCK_SIZE * 8) as u32);
    }

    #[test]
    fn test_is_similar() {
        let fp_a = FrameFingerprint::new([0x00; BLOCK_SIZE], 0);
        let mut hash_b = [0x00; BLOCK_SIZE];
        hash_b[0] = 0x01; // 1 bit different
        let fp_b = FrameFingerprint::new(hash_b, 1);
        assert!(fp_a.is_similar(&fp_b, 2));
        assert!(!fp_a.is_similar(&fp_b, 0));
    }

    #[test]
    fn test_clip_fingerprint_new() {
        let cf = ClipFingerprint::new("clip1", 5000, 2.0);
        assert_eq!(cf.clip_id, "clip1");
        assert_eq!(cf.duration_ms, 5000);
        assert_eq!(cf.frame_count(), 0);
    }

    #[test]
    fn test_clip_fingerprint_add_frame() {
        let mut cf = ClipFingerprint::new("clip1", 5000, 2.0);
        cf.add_frame(FrameFingerprint::new([0; BLOCK_SIZE], 0));
        cf.add_frame(FrameFingerprint::new([0; BLOCK_SIZE], 1));
        assert_eq!(cf.frame_count(), 2);
    }

    #[test]
    fn test_clip_average_distance_identical() {
        let mut cf_a = ClipFingerprint::new("a", 1000, 1.0);
        let mut cf_b = ClipFingerprint::new("b", 1000, 1.0);
        for i in 0..5 {
            let hash = [i as u8; BLOCK_SIZE];
            cf_a.add_frame(FrameFingerprint::new(hash, i));
            cf_b.add_frame(FrameFingerprint::new(hash, i));
        }
        assert!((cf_a.average_distance(&cf_b) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_is_duplicate() {
        let mut cf_a = ClipFingerprint::new("a", 1000, 1.0);
        let mut cf_b = ClipFingerprint::new("b", 1000, 1.0);
        let hash = [0xAA; BLOCK_SIZE];
        cf_a.add_frame(FrameFingerprint::new(hash, 0));
        cf_b.add_frame(FrameFingerprint::new(hash, 0));
        assert!(cf_a.is_duplicate(&cf_b, 1.0));
    }

    #[test]
    fn test_fingerprint_db_insert_get() {
        let mut db = FingerprintDb::with_default_threshold();
        let cf = ClipFingerprint::new("c1", 2000, 1.0);
        db.insert(cf);
        assert_eq!(db.len(), 1);
        assert!(db.get("c1").is_some());
        assert!(db.get("c2").is_none());
    }

    #[test]
    fn test_fingerprint_db_remove() {
        let mut db = FingerprintDb::with_default_threshold();
        db.insert(ClipFingerprint::new("c1", 1000, 1.0));
        assert!(db.remove("c1"));
        assert!(!db.remove("c1"));
        assert!(db.is_empty());
    }

    #[test]
    fn test_fingerprint_db_find_similar() {
        let mut db = FingerprintDb::new(10.0);
        let mut cf1 = ClipFingerprint::new("stored", 1000, 1.0);
        cf1.add_frame(FrameFingerprint::new([0x00; BLOCK_SIZE], 0));
        db.insert(cf1);

        let mut query = ClipFingerprint::new("query", 1000, 1.0);
        query.add_frame(FrameFingerprint::new([0x00; BLOCK_SIZE], 0));

        let results = db.find_similar(&query);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_duplicate);
        assert!((results[0].distance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_phash_uniform() {
        let data = vec![128u8; 64];
        let hash = compute_phash(&data, 8, 8);
        // All pixels equal the mean, so no bits should be set
        assert_eq!(hash, [0u8; BLOCK_SIZE]);
    }

    #[test]
    fn test_compute_phash_half_bright() {
        let mut data = vec![0u8; 256];
        for v in data.iter_mut().take(128) {
            *v = 255;
        }
        let hash = compute_phash(&data, 16, 16);
        // First half brighter than mean => bits set
        assert_ne!(hash, [0u8; BLOCK_SIZE]);
    }

    #[test]
    fn test_compute_phash_empty() {
        let hash = compute_phash(&[], 0, 0);
        assert_eq!(hash, [0u8; BLOCK_SIZE]);
    }
}
