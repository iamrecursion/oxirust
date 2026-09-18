//! Video-level deduplication.
//!
//! This module provides:
//! - `VideoFingerprint`: frame-level hash + temporal hash for a video
//! - `VideoDuplicateDetector`: find duplicate or near-duplicate videos
//! - `TrimmedDuplicateDetector`: sliding-window for trimmed/shifted duplicates
//! - `DuplicatePair` / `DuplicateCluster`: result types

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// VideoFingerprint
// ---------------------------------------------------------------------------

/// A compact fingerprint for a video.
///
/// Contains hashes of sampled keyframes plus a temporal hash that encodes
/// the overall frame sequence order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFingerprint {
    /// Unique identifier for this video.
    pub video_id: u64,
    /// Hashes of sampled keyframes.
    pub keyframe_hashes: Vec<u64>,
    /// Temporal hash computed from the ordered keyframe sequence.
    pub temporal_hash: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

impl VideoFingerprint {
    /// Create a new fingerprint.
    #[must_use]
    pub fn new(video_id: u64, keyframe_hashes: Vec<u64>, duration_ms: u64) -> Self {
        let temporal_hash = Self::compute_temporal_hash(&keyframe_hashes);
        Self {
            video_id,
            keyframe_hashes,
            temporal_hash,
            duration_ms,
        }
    }

    /// Compute a temporal hash from an ordered sequence of frame hashes.
    ///
    /// Accumulates with XOR and a left-rotation so order matters.
    #[must_use]
    pub fn compute_temporal_hash(frame_hashes: &[u64]) -> u64 {
        let mut acc: u64 = 0;
        for &h in frame_hashes {
            acc = acc.rotate_left(7) ^ h;
        }
        acc
    }

    /// Estimate similarity to another fingerprint using set-intersection / union
    /// of keyframe hash sets.
    #[must_use]
    pub fn keyframe_similarity(&self, other: &Self) -> f32 {
        if self.keyframe_hashes.is_empty() && other.keyframe_hashes.is_empty() {
            return 1.0;
        }

        let intersection = self
            .keyframe_hashes
            .iter()
            .filter(|h| other.keyframe_hashes.contains(h))
            .count();

        // Union = |A| + |B| - |A ∩ B|
        let union = self.keyframe_hashes.len() + other.keyframe_hashes.len() - intersection;

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

// ---------------------------------------------------------------------------
// DuplicatePair
// ---------------------------------------------------------------------------

/// A detected duplicate pair of videos.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicatePair {
    /// ID of the first video.
    pub a_id: u64,
    /// ID of the second video.
    pub b_id: u64,
    /// Similarity score in [0.0, 1.0].
    pub similarity: f32,
    /// Estimated temporal offset in milliseconds (positive = b starts later).
    pub offset_ms: i64,
    /// True if detected as a trimmed (start/end cut) duplicate.
    pub is_trimmed: bool,
}

// ---------------------------------------------------------------------------
// VideoDuplicateDetector
// ---------------------------------------------------------------------------

/// Detects duplicate and near-duplicate videos by comparing fingerprints.
pub struct VideoDuplicateDetector {
    /// Stored fingerprints.
    fingerprints: Vec<VideoFingerprint>,
}

impl VideoDuplicateDetector {
    /// Create an empty detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
        }
    }

    /// Add a fingerprint to the detector.
    pub fn add(&mut self, fingerprint: VideoFingerprint) {
        self.fingerprints.push(fingerprint);
    }

    /// Find all duplicate pairs above the given similarity threshold.
    ///
    /// Returns a list of `(id_a, id_b, similarity)` tuples.
    #[must_use]
    pub fn find_duplicates(&self, threshold: f32) -> Vec<(u64, u64, f32)> {
        let mut results = Vec::new();

        let n = self.fingerprints.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let fp_a = &self.fingerprints[i];
                let fp_b = &self.fingerprints[j];

                let sim = fp_a.keyframe_similarity(fp_b);
                if sim >= threshold {
                    results.push((fp_a.video_id, fp_b.video_id, sim));
                }
            }
        }

        results
    }

    /// Return the number of indexed fingerprints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Return true if no fingerprints have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }
}

impl Default for VideoDuplicateDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TrimmedDuplicateDetector
// ---------------------------------------------------------------------------

/// Detects trimmed (start/end cut) duplicates using sliding-window matching.
pub struct TrimmedDuplicateDetector;

impl TrimmedDuplicateDetector {
    /// Find trimmed duplicate pairs among the given fingerprints.
    ///
    /// Uses a sliding window over keyframe hash sequences.
    /// `threshold` is the minimum fraction of matching hashes in the window.
    #[must_use]
    pub fn find_trimmed_duplicates(fingerprints: &[VideoFingerprint]) -> Vec<DuplicatePair> {
        let mut pairs = Vec::new();
        let n = fingerprints.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let fp_a = &fingerprints[i];
                let fp_b = &fingerprints[j];

                if let Some((sim, offset_ms)) = sliding_window_match(fp_a, fp_b) {
                    pairs.push(DuplicatePair {
                        a_id: fp_a.video_id,
                        b_id: fp_b.video_id,
                        similarity: sim,
                        offset_ms,
                        is_trimmed: offset_ms != 0,
                    });
                }
            }
        }

        pairs
    }
}

/// Find the best sliding-window alignment between two keyframe hash sequences.
///
/// Returns `Some((similarity, offset_ms))` if the best window alignment
/// produces a match ratio ≥ 0.5, or `None` otherwise.
fn sliding_window_match(fp_a: &VideoFingerprint, fp_b: &VideoFingerprint) -> Option<(f32, i64)> {
    let a = &fp_a.keyframe_hashes;
    let b = &fp_b.keyframe_hashes;

    if a.is_empty() || b.is_empty() {
        return None;
    }

    // ms per keyframe (approximate)
    let ms_per_frame_a = if a.len() > 1 {
        fp_a.duration_ms as i64 / a.len() as i64
    } else {
        0
    };

    let mut best_sim = 0.0f32;
    let mut best_offset_ms: i64 = 0;

    // Try all offsets where the shorter sequence aligns against the longer
    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let max_offset = longer.len().saturating_sub(shorter.len()) + 1;

    for offset in 0..max_offset {
        let window = &longer[offset..offset + shorter.len()];
        let matches = shorter
            .iter()
            .zip(window.iter())
            .filter(|(x, y)| x == y)
            .count();
        let sim = matches as f32 / shorter.len() as f32;
        if sim > best_sim {
            best_sim = sim;
            best_offset_ms = offset as i64 * ms_per_frame_a;
        }
    }

    if best_sim >= 0.5 {
        Some((best_sim, best_offset_ms))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// DuplicateCluster
// ---------------------------------------------------------------------------

/// A cluster of duplicate videos.
#[derive(Debug, Clone)]
pub struct DuplicateCluster {
    /// ID of the representative (first seen) video in the cluster.
    pub representative: u64,
    /// IDs of all other members of the cluster.
    pub members: Vec<u64>,
}

impl DuplicateCluster {
    /// Build duplicate clusters from a list of pairs using union-find.
    #[must_use]
    pub fn build_clusters(pairs: &[DuplicatePair]) -> Vec<Self> {
        // Collect all unique IDs
        let mut ids: Vec<u64> = Vec::new();
        for pair in pairs {
            if !ids.contains(&pair.a_id) {
                ids.push(pair.a_id);
            }
            if !ids.contains(&pair.b_id) {
                ids.push(pair.b_id);
            }
        }

        // Union-find: parent[i] = index of parent in `ids`
        let mut parent: Vec<usize> = (0..ids.len()).collect();

        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        };

        // Union sets for each pair
        for pair in pairs {
            if let (Some(ai), Some(bi)) = (
                ids.iter().position(|&id| id == pair.a_id),
                ids.iter().position(|&id| id == pair.b_id),
            ) {
                let ra = find(&mut parent, ai);
                let rb = find(&mut parent, bi);
                if ra != rb {
                    parent[rb] = ra;
                }
            }
        }

        // Flatten all nodes to roots
        let roots: Vec<usize> = (0..ids.len()).map(|i| find(&mut parent, i)).collect();

        // Group by root
        let mut cluster_map: std::collections::HashMap<usize, Vec<u64>> =
            std::collections::HashMap::new();
        for (i, &root) in roots.iter().enumerate() {
            cluster_map.entry(root).or_default().push(ids[i]);
        }

        // Build result: only include clusters with more than one member
        cluster_map
            .into_values()
            .filter(|members| members.len() > 1)
            .map(|mut members| {
                members.sort_unstable();
                let representative = members[0];
                let rest = members[1..].to_vec();
                DuplicateCluster {
                    representative,
                    members: rest,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp(id: u64, hashes: Vec<u64>, duration_ms: u64) -> VideoFingerprint {
        VideoFingerprint::new(id, hashes, duration_ms)
    }

    // --- VideoFingerprint tests ---

    #[test]
    fn test_temporal_hash_empty() {
        let h = VideoFingerprint::compute_temporal_hash(&[]);
        assert_eq!(h, 0);
    }

    #[test]
    fn test_temporal_hash_deterministic() {
        let hashes = vec![1u64, 2, 3, 4, 5];
        let h1 = VideoFingerprint::compute_temporal_hash(&hashes);
        let h2 = VideoFingerprint::compute_temporal_hash(&hashes);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_temporal_hash_order_sensitive() {
        let h1 = VideoFingerprint::compute_temporal_hash(&[1, 2, 3]);
        let h2 = VideoFingerprint::compute_temporal_hash(&[3, 2, 1]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_keyframe_similarity_identical() {
        let fp = make_fp(1, vec![1, 2, 3, 4], 4000);
        assert_eq!(fp.keyframe_similarity(&fp), 1.0);
    }

    #[test]
    fn test_keyframe_similarity_disjoint() {
        let fp_a = make_fp(1, vec![1, 2, 3], 3000);
        let fp_b = make_fp(2, vec![4, 5, 6], 3000);
        assert_eq!(fp_a.keyframe_similarity(&fp_b), 0.0);
    }

    #[test]
    fn test_keyframe_similarity_partial() {
        let fp_a = make_fp(1, vec![1, 2, 3, 4], 4000);
        let fp_b = make_fp(2, vec![3, 4, 5, 6], 4000);
        let sim = fp_a.keyframe_similarity(&fp_b);
        // intersection=2, union=6 → 2/6 ≈ 0.333
        assert!((sim - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_keyframe_similarity_empty_both() {
        let fp_a = make_fp(1, vec![], 0);
        let fp_b = make_fp(2, vec![], 0);
        assert_eq!(fp_a.keyframe_similarity(&fp_b), 1.0);
    }

    // --- VideoDuplicateDetector tests ---

    #[test]
    fn test_detector_empty() {
        let detector = VideoDuplicateDetector::new();
        let dups = detector.find_duplicates(0.5);
        assert!(dups.is_empty());
    }

    #[test]
    fn test_detector_single() {
        let mut detector = VideoDuplicateDetector::new();
        detector.add(make_fp(1, vec![1, 2, 3], 3000));
        let dups = detector.find_duplicates(0.5);
        assert!(dups.is_empty());
    }

    #[test]
    fn test_detector_identical_pair() {
        let mut detector = VideoDuplicateDetector::new();
        detector.add(make_fp(1, vec![1, 2, 3, 4, 5], 5000));
        detector.add(make_fp(2, vec![1, 2, 3, 4, 5], 5000));
        let dups = detector.find_duplicates(0.9);
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].0, 1);
        assert_eq!(dups[0].1, 2);
        assert_eq!(dups[0].2, 1.0);
    }

    #[test]
    fn test_detector_no_match_below_threshold() {
        let mut detector = VideoDuplicateDetector::new();
        detector.add(make_fp(1, vec![1, 2, 3], 3000));
        detector.add(make_fp(2, vec![4, 5, 6], 3000));
        let dups = detector.find_duplicates(0.5);
        assert!(dups.is_empty());
    }

    // --- TrimmedDuplicateDetector tests ---

    #[test]
    fn test_trimmed_identical() {
        let hashes = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let fps = vec![
            make_fp(1, hashes.clone(), 8000),
            make_fp(2, hashes.clone(), 8000),
        ];
        let pairs = TrimmedDuplicateDetector::find_trimmed_duplicates(&fps);
        assert!(!pairs.is_empty());
        assert_eq!(pairs[0].similarity, 1.0);
        assert!(!pairs[0].is_trimmed);
    }

    #[test]
    fn test_trimmed_offset() {
        // B is A with 2 frames prepended → trimmed duplicate
        let hashes_a = vec![3u64, 4, 5, 6, 7];
        let hashes_b = vec![1u64, 2, 3, 4, 5, 6, 7];
        let fps = vec![make_fp(1, hashes_a, 5000), make_fp(2, hashes_b, 7000)];
        let pairs = TrimmedDuplicateDetector::find_trimmed_duplicates(&fps);
        assert!(!pairs.is_empty());
        assert_eq!(pairs[0].similarity, 1.0);
        assert!(pairs[0].is_trimmed);
    }

    #[test]
    fn test_trimmed_no_match() {
        let fps = vec![
            make_fp(1, vec![1, 2, 3], 3000),
            make_fp(2, vec![7, 8, 9], 3000),
        ];
        let pairs = TrimmedDuplicateDetector::find_trimmed_duplicates(&fps);
        assert!(pairs.is_empty());
    }

    // --- DuplicateCluster tests ---

    #[test]
    fn test_clusters_empty_pairs() {
        let clusters = DuplicateCluster::build_clusters(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_clusters_single_pair() {
        let pairs = vec![DuplicatePair {
            a_id: 1,
            b_id: 2,
            similarity: 1.0,
            offset_ms: 0,
            is_trimmed: false,
        }];
        let clusters = DuplicateCluster::build_clusters(&pairs);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].representative, 1);
        assert!(clusters[0].members.contains(&2));
    }

    #[test]
    fn test_clusters_chain() {
        // 1-2, 2-3, 3-4 → single cluster {1,2,3,4}
        let pairs = vec![
            DuplicatePair {
                a_id: 1,
                b_id: 2,
                similarity: 1.0,
                offset_ms: 0,
                is_trimmed: false,
            },
            DuplicatePair {
                a_id: 2,
                b_id: 3,
                similarity: 1.0,
                offset_ms: 0,
                is_trimmed: false,
            },
            DuplicatePair {
                a_id: 3,
                b_id: 4,
                similarity: 1.0,
                offset_ms: 0,
                is_trimmed: false,
            },
        ];
        let clusters = DuplicateCluster::build_clusters(&pairs);
        assert_eq!(clusters.len(), 1);
        let total: usize = 1 + clusters[0].members.len();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_clusters_two_separate() {
        // {1,2} and {3,4} are independent clusters
        let pairs = vec![
            DuplicatePair {
                a_id: 1,
                b_id: 2,
                similarity: 1.0,
                offset_ms: 0,
                is_trimmed: false,
            },
            DuplicatePair {
                a_id: 3,
                b_id: 4,
                similarity: 1.0,
                offset_ms: 0,
                is_trimmed: false,
            },
        ];
        let clusters = DuplicateCluster::build_clusters(&pairs);
        assert_eq!(clusters.len(), 2);
    }
}
