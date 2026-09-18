#![allow(dead_code)]
//! Duplicate detection combining visual perceptual hash and audio fingerprint.
//!
//! This module finds near-duplicate media assets by combining two signals:
//!
//! 1. **Visual hash** — a 64-bit perceptual hash (pHash) of representative frames.
//!    Hamming distance ≤ threshold indicates visually similar content.
//! 2. **Audio fingerprint** — a compact binary fingerprint of the audio stream.
//!    Hamming distance ≤ threshold indicates acoustically similar content.
//!
//! # Fusion strategy
//!
//! Similarity scores from both modalities are fused with configurable weights:
//!
//! ```text
//! similarity = visual_weight * visual_sim + audio_weight * audio_sim
//! ```
//!
//! where each `sim` is normalised to `[0.0, 1.0]` (1.0 = identical).
//! A pair is flagged as a duplicate if `similarity >= config.min_similarity`.
//!
//! # Patent-free
//!
//! Only standard Hamming distance and weighted linear fusion are used.
//! No patented fingerprinting algorithms (e.g. Shazam, ACRCloud) are needed.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the duplicate detector.
#[derive(Debug, Clone)]
pub struct DuplicateDetectorConfig {
    /// Maximum Hamming distance (bits) for two visual hashes to be considered
    /// visually similar.  Range: 0 (exact) to 64 (any).  Default: 10.
    pub visual_threshold: u32,
    /// Maximum Hamming distance (bits) for two audio fingerprints to be
    /// considered acoustically similar.  Default: 12.
    pub audio_threshold: u32,
    /// Weight applied to the visual similarity score [0.0, 1.0]. Default: 0.6.
    pub visual_weight: f32,
    /// Weight applied to the audio similarity score [0.0, 1.0]. Default: 0.4.
    pub audio_weight: f32,
    /// Minimum fused similarity score [0.0, 1.0] to report a candidate pair
    /// as a duplicate.  Default: 0.75.
    pub min_similarity: f32,
    /// If `true`, a candidate pair is only reported if *both* visual and audio
    /// similarities exceed their respective thresholds.  Default: false.
    pub require_both_signals: bool,
}

impl Default for DuplicateDetectorConfig {
    fn default() -> Self {
        Self {
            visual_threshold: 10,
            audio_threshold: 12,
            visual_weight: 0.6,
            audio_weight: 0.4,
            min_similarity: 0.75,
            require_both_signals: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Asset descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a single media asset used in duplicate detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetDescriptor {
    /// Unique asset identifier.
    pub asset_id: Uuid,
    /// 64-bit perceptual hash of the visual content (e.g. pHash of a key frame).
    /// `None` if the asset has no video/image track.
    pub visual_hash: Option<u64>,
    /// Compact binary audio fingerprint.  The length should be consistent
    /// across all assets indexed by the same detector.
    /// `None` if the asset has no audio track.
    pub audio_fingerprint: Option<Vec<u8>>,
}

impl AssetDescriptor {
    /// Create a new asset descriptor with both modalities.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        visual_hash: Option<u64>,
        audio_fingerprint: Option<Vec<u8>>,
    ) -> Self {
        Self {
            asset_id,
            visual_hash,
            audio_fingerprint,
        }
    }

    /// Create an audio-only descriptor (e.g. podcast, music file).
    #[must_use]
    pub fn audio_only(asset_id: Uuid, audio_fingerprint: Vec<u8>) -> Self {
        Self {
            asset_id,
            visual_hash: None,
            audio_fingerprint: Some(audio_fingerprint),
        }
    }

    /// Create a visual-only descriptor (e.g. muted video, image).
    #[must_use]
    pub fn visual_only(asset_id: Uuid, visual_hash: u64) -> Self {
        Self {
            asset_id,
            visual_hash: Some(visual_hash),
            audio_fingerprint: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Duplicate pair
// ---------------------------------------------------------------------------

/// A detected duplicate pair with fused similarity information.
#[derive(Debug, Clone)]
pub struct DuplicatePair {
    /// First asset in the pair.
    pub asset_a: Uuid,
    /// Second asset in the pair (always `asset_a < asset_b` by UUID string order).
    pub asset_b: Uuid,
    /// Visual similarity [0.0, 1.0].  `None` if at least one asset lacks a visual hash.
    pub visual_similarity: Option<f32>,
    /// Audio similarity [0.0, 1.0].  `None` if at least one asset lacks an audio fingerprint.
    pub audio_similarity: Option<f32>,
    /// Fused weighted similarity score [0.0, 1.0].
    pub fused_similarity: f32,
}

impl DuplicatePair {
    /// Returns `true` if the two assets are considered exact duplicates
    /// (both modalities available and both similarities are 1.0).
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.visual_similarity
            .map_or(false, |v| v >= 1.0 - f32::EPSILON)
            && self
                .audio_similarity
                .map_or(false, |a| a >= 1.0 - f32::EPSILON)
    }
}

// ---------------------------------------------------------------------------
// Hamming distance helpers
// ---------------------------------------------------------------------------

/// Compute the Hamming distance between two 64-bit integers (popcount of XOR).
#[inline]
fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Compute the byte-level Hamming distance between two equal-length byte slices.
///
/// Returns `None` if the slices differ in length.
fn hamming_bytes(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    let dist = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    Some(dist)
}

/// Normalise a Hamming distance over `max_bits` to a similarity in `[0.0, 1.0]`.
#[inline]
fn hamming_to_similarity(distance: u32, max_bits: u32) -> f32 {
    if max_bits == 0 {
        return 1.0;
    }
    1.0 - (distance as f32 / max_bits as f32)
}

// ---------------------------------------------------------------------------
// DuplicateDetector
// ---------------------------------------------------------------------------

/// Detects near-duplicate media assets using fused visual + audio similarity.
///
/// Assets are indexed in memory.  Call [`DuplicateDetector::find_duplicates`]
/// to retrieve all pairs above the configured similarity threshold, or
/// [`DuplicateDetector::find_duplicates_for`] to query a single asset.
#[derive(Debug)]
pub struct DuplicateDetector {
    config: DuplicateDetectorConfig,
    /// Indexed assets keyed by UUID.
    assets: HashMap<Uuid, AssetDescriptor>,
}

impl DuplicateDetector {
    /// Create a new detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DuplicateDetectorConfig::default())
    }

    /// Create a detector with a custom configuration.
    #[must_use]
    pub fn with_config(config: DuplicateDetectorConfig) -> Self {
        Self {
            config,
            assets: HashMap::new(),
        }
    }

    /// Add or update an asset in the index.
    ///
    /// If the asset was previously added, its descriptor is replaced.
    pub fn add_asset(&mut self, descriptor: AssetDescriptor) {
        self.assets.insert(descriptor.asset_id, descriptor);
    }

    /// Remove an asset from the index by UUID.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DocumentNotFound` if the asset is not indexed.
    pub fn remove_asset(&mut self, asset_id: Uuid) -> SearchResult<()> {
        if self.assets.remove(&asset_id).is_none() {
            return Err(SearchError::DocumentNotFound(asset_id.to_string()));
        }
        Ok(())
    }

    /// Return the number of indexed assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Compute the fused similarity between two descriptors.
    ///
    /// Returns `None` if neither modality is available in both assets.
    fn compute_similarity(
        &self,
        a: &AssetDescriptor,
        b: &AssetDescriptor,
    ) -> Option<DuplicatePair> {
        let visual_sim = match (a.visual_hash, b.visual_hash) {
            (Some(va), Some(vb)) => {
                let dist = hamming64(va, vb);
                Some(hamming_to_similarity(dist, 64))
            }
            _ => None,
        };

        let audio_sim = match (&a.audio_fingerprint, &b.audio_fingerprint) {
            (Some(fa), Some(fb)) => {
                let dist = hamming_bytes(fa, fb);
                let max_bits = (fa.len() * 8) as u32;
                dist.map(|d| hamming_to_similarity(d, max_bits))
            }
            _ => None,
        };

        // At least one signal must be available.
        if visual_sim.is_none() && audio_sim.is_none() {
            return None;
        }

        // Apply threshold gates.
        if self.config.require_both_signals {
            if visual_sim.is_none() || audio_sim.is_none() {
                return None;
            }
        }

        // Check individual thresholds when signals are present.
        if let Some(vs) = visual_sim {
            let dist = ((1.0 - vs) * 64.0).round() as u32;
            if dist > self.config.visual_threshold {
                // Visual signal is available but exceeds threshold.
                if self.config.require_both_signals || audio_sim.is_none() {
                    return None;
                }
            }
        }
        if let Some(aus) = audio_sim {
            if let (Some(fa), Some(_fb)) = (&a.audio_fingerprint, &b.audio_fingerprint) {
                let max_bits = (fa.len() * 8) as u32;
                let dist = ((1.0 - aus) * max_bits as f32).round() as u32;
                if dist > self.config.audio_threshold {
                    if self.config.require_both_signals || visual_sim.is_none() {
                        return None;
                    }
                }
            }
        }

        // Fuse scores.
        let fused = compute_fused_score(
            visual_sim,
            audio_sim,
            self.config.visual_weight,
            self.config.audio_weight,
        );

        if fused < self.config.min_similarity {
            return None;
        }

        // Canonical pair order by UUID string (deterministic).
        let (asset_a, asset_b) = if a.asset_id.to_string() <= b.asset_id.to_string() {
            (a.asset_id, b.asset_id)
        } else {
            (b.asset_id, a.asset_id)
        };

        Some(DuplicatePair {
            asset_a,
            asset_b,
            visual_similarity: visual_sim,
            audio_similarity: audio_sim,
            fused_similarity: fused,
        })
    }

    /// Find all duplicate pairs in the index above the configured similarity threshold.
    ///
    /// Returns pairs in descending order of `fused_similarity`.
    /// Complexity is O(n²) — for large collections use [`DuplicateDetector::find_duplicates_for`]
    /// to limit the search to a single query asset.
    #[must_use]
    pub fn find_duplicates(&self) -> Vec<DuplicatePair> {
        let descriptors: Vec<&AssetDescriptor> = self.assets.values().collect();
        let n = descriptors.len();
        let mut pairs = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(pair) = self.compute_similarity(descriptors[i], descriptors[j]) {
                    pairs.push(pair);
                }
            }
        }

        pairs.sort_by(|a, b| b.fused_similarity.total_cmp(&a.fused_similarity));
        pairs
    }

    /// Find duplicates of a specific query asset against all indexed assets.
    ///
    /// The query asset does **not** need to be in the index.
    /// Returns pairs sorted by descending `fused_similarity`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::FeatureExtraction` if the query descriptor has no
    /// usable modality (both `visual_hash` and `audio_fingerprint` are `None`).
    pub fn find_duplicates_for(&self, query: &AssetDescriptor) -> SearchResult<Vec<DuplicatePair>> {
        if query.visual_hash.is_none() && query.audio_fingerprint.is_none() {
            return Err(SearchError::FeatureExtraction(
                "query descriptor has neither visual_hash nor audio_fingerprint".into(),
            ));
        }

        let mut pairs: Vec<DuplicatePair> = self
            .assets
            .values()
            .filter(|a| a.asset_id != query.asset_id)
            .filter_map(|candidate| self.compute_similarity(query, candidate))
            .collect();

        pairs.sort_by(|a, b| b.fused_similarity.total_cmp(&a.fused_similarity));
        Ok(pairs)
    }

    /// Return the configured minimum similarity threshold.
    #[must_use]
    pub fn min_similarity(&self) -> f32 {
        self.config.min_similarity
    }

    /// Clear all indexed assets.
    pub fn clear(&mut self) {
        self.assets.clear();
    }
}

impl Default for DuplicateDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Score fusion helper
// ---------------------------------------------------------------------------

/// Compute a weighted fused score from optional visual and audio similarity values.
///
/// If only one signal is available, the full weight of both signals is applied
/// to the available one (renormalisation).
fn compute_fused_score(
    visual_sim: Option<f32>,
    audio_sim: Option<f32>,
    visual_weight: f32,
    audio_weight: f32,
) -> f32 {
    match (visual_sim, audio_sim) {
        (Some(v), Some(a)) => {
            let total_weight = visual_weight + audio_weight;
            if total_weight < f32::EPSILON {
                (v + a) / 2.0
            } else {
                (v * visual_weight + a * audio_weight) / total_weight
            }
        }
        (Some(v), None) => v,
        (None, Some(a)) => a,
        (None, None) => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_desc(visual: Option<u64>, audio: Option<Vec<u8>>) -> AssetDescriptor {
        AssetDescriptor::new(Uuid::new_v4(), visual, audio)
    }

    #[test]
    fn test_hamming64_identical() {
        assert_eq!(hamming64(0xDEAD_BEEF, 0xDEAD_BEEF), 0);
    }

    #[test]
    fn test_hamming64_all_differ() {
        assert_eq!(hamming64(0u64, u64::MAX), 64);
    }

    #[test]
    fn test_hamming_bytes_equal() {
        let a = vec![0u8, 0, 0, 0];
        let b = vec![0u8, 0, 0, 0];
        assert_eq!(hamming_bytes(&a, &b), Some(0));
    }

    #[test]
    fn test_hamming_bytes_different_length() {
        assert_eq!(hamming_bytes(&[0u8, 1], &[0u8]), None);
    }

    #[test]
    fn test_hamming_bytes_one_bit() {
        let a = vec![0b0000_0001u8];
        let b = vec![0b0000_0000u8];
        assert_eq!(hamming_bytes(&a, &b), Some(1));
    }

    #[test]
    fn test_hamming_to_similarity_zero_dist() {
        assert!((hamming_to_similarity(0, 64) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_to_similarity_full_dist() {
        assert!((hamming_to_similarity(64, 64) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_to_similarity_half() {
        assert!((hamming_to_similarity(32, 64) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_fused_score_both_equal_weights() {
        let s = compute_fused_score(Some(0.8), Some(0.6), 0.5, 0.5);
        assert!((s - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_fused_score_visual_only() {
        let s = compute_fused_score(Some(0.9), None, 0.6, 0.4);
        assert!((s - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_fused_score_audio_only() {
        let s = compute_fused_score(None, Some(0.7), 0.6, 0.4);
        assert!((s - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_fused_score_none_none() {
        let s = compute_fused_score(None, None, 0.6, 0.4);
        assert!((s - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_detector_exact_duplicate() {
        let mut detector = DuplicateDetector::new();
        let hash = 0xABCD_1234_5678_EF00u64;
        let fp = vec![0b1010_1010u8; 8];

        let a = AssetDescriptor::new(Uuid::new_v4(), Some(hash), Some(fp.clone()));
        let b = AssetDescriptor::new(Uuid::new_v4(), Some(hash), Some(fp));
        detector.add_asset(a);
        detector.add_asset(b);

        let pairs = detector.find_duplicates();
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].fused_similarity - 1.0).abs() < 1e-5);
        assert!(pairs[0].is_exact());
    }

    #[test]
    fn test_detector_no_duplicates_when_very_different() {
        let mut detector = DuplicateDetector::new();
        // visual hash with 40 bits different → well above default threshold of 10
        let a = AssetDescriptor::new(Uuid::new_v4(), Some(0x0000_0000_0000_0000u64), None);
        let b = AssetDescriptor::new(Uuid::new_v4(), Some(0xFFFF_FFFF_FFFF_FFFFu64), None);
        detector.add_asset(a);
        detector.add_asset(b);

        let pairs = detector.find_duplicates();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_detector_close_visual_hash() {
        let mut detector = DuplicateDetector::new();
        let base: u64 = 0xFFFF_FFFF_FFFF_FFFFu64;
        // 5 bits different — within threshold of 10
        let similar = base ^ 0b1_1111u64;
        let a = AssetDescriptor::new(Uuid::new_v4(), Some(base), None);
        let b = AssetDescriptor::new(Uuid::new_v4(), Some(similar), None);
        detector.add_asset(a);
        detector.add_asset(b);

        let pairs = detector.find_duplicates();
        assert_eq!(pairs.len(), 1);
        let vis_sim = pairs[0].visual_similarity.expect("should have visual sim");
        assert!(vis_sim > 0.9);
    }

    #[test]
    fn test_detector_asset_count() {
        let mut detector = DuplicateDetector::new();
        assert_eq!(detector.asset_count(), 0);
        detector.add_asset(make_desc(Some(42), None));
        detector.add_asset(make_desc(Some(99), None));
        assert_eq!(detector.asset_count(), 2);
    }

    #[test]
    fn test_detector_remove_asset() {
        let mut detector = DuplicateDetector::new();
        let id = Uuid::new_v4();
        detector.add_asset(AssetDescriptor::new(id, Some(1234), None));
        assert_eq!(detector.asset_count(), 1);
        detector.remove_asset(id).expect("should succeed");
        assert_eq!(detector.asset_count(), 0);
    }

    #[test]
    fn test_detector_remove_nonexistent() {
        let mut detector = DuplicateDetector::new();
        assert!(detector.remove_asset(Uuid::new_v4()).is_err());
    }

    #[test]
    fn test_find_duplicates_for_query() {
        let mut detector = DuplicateDetector::new();
        let hash = 0xBEEF_CAFE_0102_0304u64;
        // Index two assets, one near-duplicate of query
        let near = AssetDescriptor::new(Uuid::new_v4(), Some(hash ^ 0b11u64), None); // 2 bits diff
        let far = AssetDescriptor::new(Uuid::new_v4(), Some(hash ^ u64::MAX), None); // 64 bits diff
        let near_id = near.asset_id;
        detector.add_asset(near);
        detector.add_asset(far);

        let query = AssetDescriptor::new(Uuid::new_v4(), Some(hash), None);
        let pairs = detector.find_duplicates_for(&query).expect("ok");
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].asset_a == near_id || pairs[0].asset_b == near_id);
    }

    #[test]
    fn test_find_duplicates_for_empty_query_errors() {
        let detector = DuplicateDetector::new();
        let empty = AssetDescriptor::new(Uuid::new_v4(), None, None);
        assert!(detector.find_duplicates_for(&empty).is_err());
    }

    #[test]
    fn test_detector_clear() {
        let mut detector = DuplicateDetector::new();
        detector.add_asset(make_desc(Some(1), None));
        detector.add_asset(make_desc(Some(2), None));
        detector.clear();
        assert_eq!(detector.asset_count(), 0);
        assert!(detector.find_duplicates().is_empty());
    }

    #[test]
    fn test_asset_descriptor_constructors() {
        let id = Uuid::new_v4();
        let visual = AssetDescriptor::visual_only(id, 42u64);
        assert!(visual.visual_hash.is_some());
        assert!(visual.audio_fingerprint.is_none());

        let id2 = Uuid::new_v4();
        let audio = AssetDescriptor::audio_only(id2, vec![1u8, 2, 3]);
        assert!(audio.visual_hash.is_none());
        assert!(audio.audio_fingerprint.is_some());
    }

    #[test]
    fn test_duplicate_pair_is_not_exact_when_audio_missing() {
        let pair = DuplicatePair {
            asset_a: Uuid::new_v4(),
            asset_b: Uuid::new_v4(),
            visual_similarity: Some(1.0),
            audio_similarity: None,
            fused_similarity: 1.0,
        };
        // Only one modality present — not considered "exact" by policy.
        assert!(!pair.is_exact());
    }

    #[test]
    fn test_require_both_signals_config() {
        let config = DuplicateDetectorConfig {
            require_both_signals: true,
            visual_threshold: 10,
            audio_threshold: 12,
            visual_weight: 0.5,
            audio_weight: 0.5,
            min_similarity: 0.5,
        };
        let mut detector = DuplicateDetector::with_config(config);

        // Asset with only visual hash — with require_both=true, no pair should
        // form because the audio signal is absent.
        let a = AssetDescriptor::visual_only(Uuid::new_v4(), 0x1234u64);
        let b = AssetDescriptor::visual_only(Uuid::new_v4(), 0x1234u64); // identical hash
        detector.add_asset(a);
        detector.add_asset(b);

        let pairs = detector.find_duplicates();
        assert!(
            pairs.is_empty(),
            "require_both_signals=true must suppress single-modality matches"
        );
    }

    #[test]
    fn test_audio_fingerprint_similarity() {
        let mut detector = DuplicateDetector::with_config(DuplicateDetectorConfig {
            audio_threshold: 4,
            visual_threshold: 10,
            visual_weight: 0.0,
            audio_weight: 1.0,
            min_similarity: 0.9,
            require_both_signals: false,
        });
        // 1 bit different in 8 bytes (64 bits) → similarity = 63/64 ≈ 0.984
        let fp_a = vec![0b0000_0000u8; 8];
        let mut fp_b = vec![0b0000_0000u8; 8];
        fp_b[0] = 0b0000_0001; // 1 bit diff

        let a = AssetDescriptor::audio_only(Uuid::new_v4(), fp_a);
        let b = AssetDescriptor::audio_only(Uuid::new_v4(), fp_b);
        detector.add_asset(a);
        detector.add_asset(b);

        let pairs = detector.find_duplicates();
        assert_eq!(pairs.len(), 1);
        let aus = pairs[0]
            .audio_similarity
            .expect("audio sim should be present");
        assert!(aus > 0.9);
    }

    #[test]
    fn test_sorted_by_fused_similarity_descending() {
        let mut detector = DuplicateDetector::with_config(DuplicateDetectorConfig {
            visual_threshold: 64,
            audio_threshold: 64,
            min_similarity: 0.0,
            ..Default::default()
        });
        let base: u64 = 0;
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();
        // A vs B: 2 bits diff → high similarity
        // A vs C: 20 bits diff → lower similarity
        // B vs C: 22 bits diff → lowest
        detector.add_asset(AssetDescriptor::visual_only(id_a, base));
        detector.add_asset(AssetDescriptor::visual_only(id_b, base ^ 0b11u64));
        detector.add_asset(AssetDescriptor::visual_only(
            id_c,
            base ^ ((1u64 << 20) - 1),
        ));

        let pairs = detector.find_duplicates();
        // Should be sorted descending
        for w in pairs.windows(2) {
            assert!(w[0].fused_similarity >= w[1].fused_similarity);
        }
    }
}
