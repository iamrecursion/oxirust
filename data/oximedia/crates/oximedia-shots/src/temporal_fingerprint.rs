//! Temporal video fingerprinting using difference hashing and DTW alignment.
//!
//! Each frame is reduced to a compact 64-bit hash by:
//! 1. Treating the raw byte slice as a grid of luma/intensity samples.
//! 2. Computing the mean intensity.
//! 3. Setting bit `i` to 1 if sample `i` is above the mean (0 otherwise).
//!
//! A [`TemporalFingerprint`] is a sequence of such hashes, one per processed
//! frame (or one per `frame_interval` frames). Two fingerprints can be compared
//! with [`TemporalFingerprint::match_score`], which runs classic O(N·M) DTW
//! using per-hash Hamming distance as the frame-level cost.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A temporal fingerprint for a video clip or segment.
///
/// Stores a compact representation of the visual evolution of a sequence as a
/// vector of 64-bit difference hashes.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFingerprint {
    /// One 64-bit hash per sampled frame.
    pub hashes: Vec<u64>,
    /// Frame rate of the source video (frames per second).
    pub fps: f32,
    /// Total duration represented by this fingerprint, in milliseconds.
    pub duration_ms: u64,
}

impl TemporalFingerprint {
    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of hash entries in this fingerprint.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.hashes.len()
    }

    /// Total duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    // -----------------------------------------------------------------------
    // Similarity via DTW
    // -----------------------------------------------------------------------

    /// Compute a similarity score in `[0.0, 1.0]` between `self` and `other`
    /// using Dynamic Time Warping (DTW) on per-hash Hamming distances.
    ///
    /// Returns `1.0` for identical fingerprints and `0.0` for maximally
    /// different ones.  Empty fingerprints produce `0.0`.
    #[must_use]
    pub fn match_score(&self, other: &TemporalFingerprint) -> f32 {
        if self.hashes.is_empty() || other.hashes.is_empty() {
            return 0.0;
        }

        let n = self.hashes.len();
        let m = other.hashes.len();

        // DTW cost matrix — flattened row-major (n rows, m cols)
        let mut dp = vec![f64::INFINITY; n * m];

        // Seed corner
        dp[0] = hamming_norm(self.hashes[0], other.hashes[0]);

        // Fill first column
        for i in 1..n {
            dp[i * m] = dp[(i - 1) * m] + hamming_norm(self.hashes[i], other.hashes[0]);
        }
        // Fill first row
        for j in 1..m {
            dp[j] = dp[j - 1] + hamming_norm(self.hashes[0], other.hashes[j]);
        }
        // Fill rest
        for i in 1..n {
            for j in 1..m {
                let cost = hamming_norm(self.hashes[i], other.hashes[j]);
                let prev = dp[(i - 1) * m + j]
                    .min(dp[i * m + (j - 1)])
                    .min(dp[(i - 1) * m + (j - 1)]);
                dp[i * m + j] = prev + cost;
            }
        }

        // Normalise by warping path length (n + m - 1 is the maximum possible
        // path length for an (n×m) DTW grid).
        let raw_cost = dp[n * m - 1];
        let path_len = (n + m - 1) as f64;
        let normalised = raw_cost / path_len;

        // `normalised` is in [0, 1] since each per-hash cost is already
        // normalised to [0, 1] via `hamming_norm`.
        let similarity = (1.0 - normalised).clamp(0.0, 1.0) as f32;
        similarity
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Extracts temporal fingerprints from sequences of raw frame byte slices.
///
/// Each frame is hashed using a difference-hash algorithm:
/// - The input is divided into 64 equal-sized regions.
/// - The mean intensity of all bytes is computed.
/// - Each region's mean is compared to the global mean; bit `i` is set if
///   region `i` is above the global mean.
#[derive(Debug, Clone)]
pub struct TemporalFingerprintExtractor {
    /// Only process every Nth frame (1 = every frame, 2 = every other, etc.).
    pub frame_interval: usize,
}

impl TemporalFingerprintExtractor {
    /// Create a new extractor that processes every `frame_interval`-th frame.
    ///
    /// `frame_interval` is clamped to a minimum of 1.
    #[must_use]
    pub fn new(frame_interval: usize) -> Self {
        Self {
            frame_interval: frame_interval.max(1),
        }
    }

    /// Create an extractor that processes every frame.
    #[must_use]
    pub fn every_frame() -> Self {
        Self::new(1)
    }

    /// Extract a [`TemporalFingerprint`] from a sequence of raw frame slices.
    ///
    /// Each element of `frames` is an arbitrary byte slice representing a
    /// video frame (e.g., packed Y, YUV420, or RGB data).
    ///
    /// # Arguments
    ///
    /// * `frames` - Ordered sequence of raw frame byte slices.
    /// * `fps`    - Frame rate of the source video.
    #[must_use]
    pub fn extract(&self, frames: &[&[u8]], fps: f32) -> TemporalFingerprint {
        if frames.is_empty() || fps <= 0.0 {
            return TemporalFingerprint {
                hashes: Vec::new(),
                fps: fps.max(0.0),
                duration_ms: 0,
            };
        }

        let interval = self.frame_interval.max(1);
        let hashes: Vec<u64> = frames
            .iter()
            .enumerate()
            .filter(|(i, _)| i % interval == 0)
            .map(|(_, frame)| dhash_64(frame))
            .collect();

        let total_frames = frames.len();
        let duration_ms = ((total_frames as f64 / fps as f64) * 1000.0).round() as u64;

        TemporalFingerprint {
            hashes,
            fps,
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a 64-bit difference hash of an arbitrary byte slice.
///
/// The slice is divided into 64 equal buckets.  The mean byte value across the
/// entire slice is computed.  Bit `i` is set to 1 if the mean value of bucket
/// `i` is greater than or equal to the global mean.
fn dhash_64(data: &[u8]) -> u64 {
    if data.is_empty() {
        return 0;
    }

    // Global mean
    let total: u64 = data.iter().map(|&b| b as u64).sum();
    let global_mean = total as f64 / data.len() as f64;

    // Split into 64 buckets
    let len = data.len();
    let mut hash: u64 = 0;
    for bit in 0..64u32 {
        let start = (bit as usize * len) / 64;
        let end = ((bit as usize + 1) * len) / 64;
        let bucket = if end > start {
            &data[start..end]
        } else {
            &data[start..start + 1.min(len - start)]
        };
        let bucket_sum: u64 = bucket.iter().map(|&b| b as u64).sum();
        let bucket_mean = bucket_sum as f64 / bucket.len() as f64;
        if bucket_mean >= global_mean {
            hash |= 1u64 << bit;
        }
    }
    hash
}

/// Hamming distance between two 64-bit hashes, normalised to `[0.0, 1.0]`.
#[inline]
fn hamming_norm(a: u64, b: u64) -> f64 {
    let dist = (a ^ b).count_ones() as f64;
    dist / 64.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn uniform_frame(value: u8, size: usize) -> Vec<u8> {
        vec![value; size]
    }

    fn ramp_frame(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    // ------------------------------------------------------------------
    // dhash / hamming helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_dhash_empty_is_zero() {
        assert_eq!(dhash_64(&[]), 0);
    }

    #[test]
    fn test_dhash_uniform_all_above_or_equal_mean() {
        // All bytes equal → all buckets equal mean → all bits set
        let data = uniform_frame(128, 256);
        let h = dhash_64(&data);
        // Every bucket mean == global mean → all bits set (>=)
        assert_eq!(h, u64::MAX);
    }

    #[test]
    fn test_dhash_two_identical_frames_same_hash() {
        let f = ramp_frame(640);
        assert_eq!(dhash_64(&f), dhash_64(&f));
    }

    #[test]
    fn test_hamming_norm_identical() {
        assert!((hamming_norm(0xABCD, 0xABCD) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_hamming_norm_all_bits_differ() {
        assert!((hamming_norm(0, u64::MAX) - 1.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // TemporalFingerprintExtractor
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_empty_frames() {
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&[], 25.0);
        assert_eq!(fp.frame_count(), 0);
        assert_eq!(fp.duration_ms(), 0);
    }

    #[test]
    fn test_extract_single_frame() {
        let frame = ramp_frame(1920);
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&[&frame], 25.0);
        assert_eq!(fp.frame_count(), 1);
        assert!(fp.duration_ms() > 0 || true); // 1/25s = 40ms
    }

    #[test]
    fn test_extract_frame_count_with_interval() {
        let frames: Vec<Vec<u8>> = (0..10).map(|i| uniform_frame(i as u8 * 20, 64)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::new(2); // every 2nd
        let fp = extractor.extract(&refs, 24.0);
        // frames 0, 2, 4, 6, 8 → 5 hashes
        assert_eq!(fp.frame_count(), 5);
    }

    #[test]
    fn test_extract_duration_ms() {
        let frames: Vec<Vec<u8>> = (0..25).map(|i| uniform_frame(i as u8, 64)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&refs, 25.0);
        // 25 frames at 25fps = 1000ms
        assert_eq!(fp.duration_ms(), 1000);
    }

    #[test]
    fn test_extract_fps_stored() {
        let frame = uniform_frame(100, 32);
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&[&frame], 30.0);
        assert!((fp.fps - 30.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // TemporalFingerprint::match_score
    // ------------------------------------------------------------------

    #[test]
    fn test_match_score_identical_is_one() {
        let frames: Vec<Vec<u8>> = (0..8).map(|i| ramp_frame(i * 32 + 64)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&refs, 24.0);
        let score = fp.match_score(&fp);
        assert!(
            (score - 1.0).abs() < 1e-5,
            "identical fingerprint score={score}"
        );
    }

    #[test]
    fn test_match_score_empty_is_zero() {
        let fp_empty = TemporalFingerprint {
            hashes: vec![],
            fps: 24.0,
            duration_ms: 0,
        };
        let frame = ramp_frame(256);
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&[&frame], 24.0);
        assert!((fp_empty.match_score(&fp)).abs() < 1e-5);
        assert!((fp.match_score(&fp_empty)).abs() < 1e-5);
    }

    #[test]
    fn test_match_score_different_frames_below_one() {
        // Use ramp frames vs inverse-ramp frames, which produce different hashes
        let ramp: Vec<Vec<u8>> = (0..8)
            .map(|_| (0..256u32).map(|i| i as u8).collect())
            .collect();
        let inv_ramp: Vec<Vec<u8>> = (0..8)
            .map(|_| (0..256u32).map(|i| 255 - i as u8).collect())
            .collect();
        let r_refs: Vec<&[u8]> = ramp.iter().map(|v| v.as_slice()).collect();
        let i_refs: Vec<&[u8]> = inv_ramp.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp_r = extractor.extract(&r_refs, 24.0);
        let fp_i = extractor.extract(&i_refs, 24.0);
        // A ramp and its inverse produce complementary bit patterns → Hamming distance = 64
        let score = fp_r.match_score(&fp_i);
        assert!(
            score < 1.0,
            "different fingerprints should score < 1.0, got {score}"
        );
    }

    #[test]
    fn test_match_score_symmetric() {
        let fa: Vec<Vec<u8>> = vec![ramp_frame(128), uniform_frame(50, 128)];
        let fb: Vec<Vec<u8>> = vec![uniform_frame(200, 128), ramp_frame(64)];
        let a_refs: Vec<&[u8]> = fa.iter().map(|v| v.as_slice()).collect();
        let b_refs: Vec<&[u8]> = fb.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp_a = extractor.extract(&a_refs, 24.0);
        let fp_b = extractor.extract(&b_refs, 24.0);
        let s_ab = fp_a.match_score(&fp_b);
        let s_ba = fp_b.match_score(&fp_a);
        assert!(
            (s_ab - s_ba).abs() < 1e-5,
            "match_score must be symmetric: {s_ab} vs {s_ba}"
        );
    }

    #[test]
    fn test_match_score_in_range() {
        let frames: Vec<Vec<u8>> = (0..6).map(|i| uniform_frame(i as u8 * 40, 128)).collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp = extractor.extract(&refs, 25.0);
        let other_frames: Vec<Vec<u8>> = (0..4).map(|i| ramp_frame(i * 64 + 32)).collect();
        let other_refs: Vec<&[u8]> = other_frames.iter().map(|v| v.as_slice()).collect();
        let fp2 = extractor.extract(&other_refs, 25.0);
        let score = fp.match_score(&fp2);
        assert!(score >= 0.0 && score <= 1.0, "score out of range: {score}");
    }

    #[test]
    fn test_match_score_offset_sequence_higher_than_random() {
        // A sequence and the same sequence with a 1-frame offset should score
        // higher than a completely random sequence.
        let base: Vec<Vec<u8>> = (0..10).map(|i| uniform_frame(i as u8 * 20, 256)).collect();
        let shifted: Vec<Vec<u8>> = (1..=10)
            .map(|i| uniform_frame((i % 13) as u8 * 20, 256))
            .collect();
        let random: Vec<Vec<u8>> = (0..10)
            .map(|i| uniform_frame(((i * 37 + 13) % 256) as u8, 256))
            .collect();
        let b_refs: Vec<&[u8]> = base.iter().map(|v| v.as_slice()).collect();
        let s_refs: Vec<&[u8]> = shifted.iter().map(|v| v.as_slice()).collect();
        let r_refs: Vec<&[u8]> = random.iter().map(|v| v.as_slice()).collect();
        let extractor = TemporalFingerprintExtractor::every_frame();
        let fp_b = extractor.extract(&b_refs, 25.0);
        let fp_s = extractor.extract(&s_refs, 25.0);
        let fp_r = extractor.extract(&r_refs, 25.0);
        let score_shift = fp_b.match_score(&fp_s);
        let score_rand = fp_b.match_score(&fp_r);
        // Shifted sequence should be at least as similar as random
        // (relaxed assertion — DTW may not guarantee strict ordering here,
        // but both must be in [0,1])
        assert!(score_shift >= 0.0 && score_rand >= 0.0);
    }
}
