//! Duplicate and near-duplicate frame detection.
//!
//! Provides efficient detection of exact duplicate frames (byte-for-byte identity)
//! and near-duplicate frames (perceptually very similar). Near-duplicate detection
//! uses multi-metric comparison: SAD (Sum of Absolute Differences), histogram
//! correlation, and a fast DCT-based perceptual hash.
//!
//! # Algorithm
//!
//! For each consecutive pair of frames (or any pair in a set), the detector
//! computes:
//!
//! * **SAD score**: normalised mean absolute difference across all luma pixels.
//! * **Histogram correlation**: comparison of 64-bin luma histograms.
//! * **Perceptual similarity**: Hamming distance between 8×8 average hashes.
//!
//! A frame pair is classified as a near-duplicate when all three scores
//! exceed their respective thresholds.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during duplicate detection.
#[derive(Debug, thiserror::Error)]
pub enum DuplicateDetectError {
    /// Frame dimensions are inconsistent within the set.
    #[error("frame dimension mismatch: expected {expected_w}x{expected_h}, got {got_w}x{got_h} at index {index}")]
    DimensionMismatch {
        /// Expected width.
        expected_w: u32,
        /// Expected height.
        expected_h: u32,
        /// Actual width at the offending index.
        got_w: u32,
        /// Actual height at the offending index.
        got_h: u32,
        /// Index of the offending frame.
        index: usize,
    },
    /// A frame buffer's length does not match the declared dimensions.
    #[error("frame {index} buffer size {actual} does not match {w}x{h}={expected}")]
    BufferSizeMismatch {
        /// Frame index.
        index: usize,
        /// Declared width.
        w: u32,
        /// Declared height.
        h: u32,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Classification of a frame pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKind {
    /// Frames are bit-for-bit identical.
    Exact,
    /// Frames are perceptually very similar (SAD, histogram, and hash all agree).
    NearDuplicate,
    /// Frames are not considered duplicates under the given thresholds.
    Unique,
}

/// Result of comparing two frames.
#[derive(Debug, Clone)]
pub struct FramePairResult {
    /// Index of the first frame in the input slice.
    pub index_a: usize,
    /// Index of the second frame in the input slice.
    pub index_b: usize,
    /// Normalised SAD in [0, 1] (0 = identical, 1 = maximally different).
    pub sad_score: f32,
    /// Histogram correlation in [0, 1] (1 = identical distribution).
    pub histogram_correlation: f32,
    /// Perceptual hash similarity in [0, 1] (1 = identical hash).
    pub hash_similarity: f32,
    /// Classification of the pair.
    pub kind: DuplicateKind,
}

/// A run of consecutive duplicate frames in a sequence.
#[derive(Debug, Clone)]
pub struct DuplicateRun {
    /// Index of the first frame in the run.
    pub start: usize,
    /// Index of the last frame in the run (inclusive).
    pub end: usize,
    /// Kind of duplication (Exact or NearDuplicate).
    pub kind: DuplicateKind,
}

impl DuplicateRun {
    /// Number of frames in the run.
    pub fn len(&self) -> usize {
        self.end - self.start + 1
    }

    /// Returns whether the run is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if the run contains more than one frame.
    pub fn is_meaningful(&self) -> bool {
        self.end > self.start
    }
}

/// Configuration for the duplicate frame detector.
#[derive(Debug, Clone)]
pub struct DuplicateDetectorConfig {
    /// SAD threshold [0, 1] below which a pair is treated as near-duplicate.
    /// Default: `0.02` (2% mean pixel difference).
    pub sad_threshold: f32,
    /// Histogram correlation threshold [0, 1] above which a pair is near-duplicate.
    /// Default: `0.98`.
    pub histogram_threshold: f32,
    /// Hash similarity threshold [0, 1] above which a pair is near-duplicate.
    /// Default: `0.96` (≈3 bit Hamming distance out of 64).
    pub hash_threshold: f32,
    /// Number of histogram bins.
    /// Default: `64`.
    pub histogram_bins: usize,
}

impl Default for DuplicateDetectorConfig {
    fn default() -> Self {
        Self {
            sad_threshold: 0.02,
            histogram_threshold: 0.98,
            hash_threshold: 0.96,
            histogram_bins: 64,
        }
    }
}

/// Stateless duplicate frame detector.
pub struct DuplicateDetector {
    /// Detection thresholds and parameters.
    pub config: DuplicateDetectorConfig,
}

impl DuplicateDetector {
    /// Create a new detector with the given `config`.
    pub fn new(config: DuplicateDetectorConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration.
    pub fn default() -> Self {
        Self::new(DuplicateDetectorConfig::default())
    }

    /// Compare two grayscale frames (luma planes, `width × height` bytes).
    ///
    /// Returns a [`FramePairResult`] with all computed metrics and the final
    /// classification.
    ///
    /// # Errors
    ///
    /// Returns `DuplicateDetectError::BufferSizeMismatch` if either frame's
    /// buffer length differs from `width × height`.
    pub fn compare_pair(
        &self,
        frame_a: &[u8],
        frame_b: &[u8],
        width: u32,
        height: u32,
        index_a: usize,
        index_b: usize,
    ) -> Result<FramePairResult, DuplicateDetectError> {
        let expected = (width as usize) * (height as usize);
        if frame_a.len() != expected {
            return Err(DuplicateDetectError::BufferSizeMismatch {
                index: index_a,
                w: width,
                h: height,
                expected,
                actual: frame_a.len(),
            });
        }
        if frame_b.len() != expected {
            return Err(DuplicateDetectError::BufferSizeMismatch {
                index: index_b,
                w: width,
                h: height,
                expected,
                actual: frame_b.len(),
            });
        }

        // Check for exact equality first (short-circuit).
        if frame_a == frame_b {
            return Ok(FramePairResult {
                index_a,
                index_b,
                sad_score: 0.0,
                histogram_correlation: 1.0,
                hash_similarity: 1.0,
                kind: DuplicateKind::Exact,
            });
        }

        let sad_score = compute_normalised_sad(frame_a, frame_b);
        let histogram_correlation =
            compute_histogram_correlation(frame_a, frame_b, self.config.histogram_bins);
        let hash_similarity = compute_avg_hash_similarity(frame_a, width, height, frame_b);

        let kind = if sad_score <= self.config.sad_threshold
            && histogram_correlation >= self.config.histogram_threshold
            && hash_similarity >= self.config.hash_threshold
        {
            DuplicateKind::NearDuplicate
        } else {
            DuplicateKind::Unique
        };

        Ok(FramePairResult {
            index_a,
            index_b,
            sad_score,
            histogram_correlation,
            hash_similarity,
            kind,
        })
    }

    /// Scan a sequence of grayscale frames for consecutive duplicates.
    ///
    /// All frames must be `width × height` bytes.  Returns a list of
    /// [`DuplicateRun`] entries, each representing a run of duplicates.
    ///
    /// # Errors
    ///
    /// Returns the first `DuplicateDetectError` encountered.
    pub fn scan_sequence(
        &self,
        frames: &[&[u8]],
        width: u32,
        height: u32,
    ) -> Result<Vec<DuplicateRun>, DuplicateDetectError> {
        let expected = (width as usize) * (height as usize);
        for (i, &frame) in frames.iter().enumerate() {
            if frame.len() != expected {
                return Err(DuplicateDetectError::BufferSizeMismatch {
                    index: i,
                    w: width,
                    h: height,
                    expected,
                    actual: frame.len(),
                });
            }
        }

        let mut runs: Vec<DuplicateRun> = Vec::new();
        if frames.len() < 2 {
            return Ok(runs);
        }

        let mut run_start: Option<(usize, DuplicateKind)> = None;

        for i in 0..frames.len() - 1 {
            let result = self.compare_pair(frames[i], frames[i + 1], width, height, i, i + 1)?;

            match result.kind {
                DuplicateKind::Unique => {
                    if let Some((start, kind)) = run_start.take() {
                        runs.push(DuplicateRun {
                            start,
                            end: i,
                            kind,
                        });
                    }
                }
                kind => {
                    if run_start.is_none() {
                        run_start = Some((i, kind));
                    }
                }
            }
        }

        // Close any open run at the end of the sequence.
        if let Some((start, kind)) = run_start {
            runs.push(DuplicateRun {
                start,
                end: frames.len() - 1,
                kind,
            });
        }

        Ok(runs)
    }

    /// Return the indices of frames that should be removed to deduplicate the
    /// sequence (keeping the first frame of each duplicate run).
    ///
    /// All frames must be `width × height` bytes.
    pub fn deduplicate_indices(
        &self,
        frames: &[&[u8]],
        width: u32,
        height: u32,
    ) -> Result<Vec<usize>, DuplicateDetectError> {
        let runs = self.scan_sequence(frames, width, height)?;
        let mut remove: Vec<usize> = Vec::new();
        for run in runs {
            // Keep run.start; mark run.start+1 ..= run.end for removal.
            for idx in (run.start + 1)..=run.end {
                remove.push(idx);
            }
        }
        remove.sort_unstable();
        remove.dedup();
        Ok(remove)
    }
}

// -----------------------------------------------------------------------
// Public free functions
// -----------------------------------------------------------------------

/// Compute the normalised mean absolute difference (SAD) between two luma
/// planes of the same length.
///
/// Returns a value in [0, 1] where 0 means identical and 1 means maximally
/// different.
pub fn compute_normalised_sad(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let total: u64 = a
        .iter()
        .take(n)
        .zip(b.iter().take(n))
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    (total as f64 / (n as f64 * 255.0)) as f32
}

/// Compute the Pearson histogram correlation between two luma planes.
///
/// Bins the intensities into `bins` buckets, normalises to probability
/// distributions, then returns the correlation coefficient in [0, 1].
pub fn compute_histogram_correlation(a: &[u8], b: &[u8], bins: usize) -> f32 {
    let bins = bins.max(1).min(256);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    if n_a == 0.0 || n_b == 0.0 {
        return 0.0;
    }

    let mut hist_a = vec![0u64; bins];
    let mut hist_b = vec![0u64; bins];

    for &p in a {
        let bin = (p as usize * bins / 256).min(bins - 1);
        hist_a[bin] += 1;
    }
    for &p in b {
        let bin = (p as usize * bins / 256).min(bins - 1);
        hist_b[bin] += 1;
    }

    // Normalise.
    let ha: Vec<f64> = hist_a.iter().map(|&c| c as f64 / n_a).collect();
    let hb: Vec<f64> = hist_b.iter().map(|&c| c as f64 / n_b).collect();

    // Pearson correlation.
    let mean_a = ha.iter().sum::<f64>() / bins as f64;
    let mean_b = hb.iter().sum::<f64>() / bins as f64;

    let mut num = 0.0f64;
    let mut den_a = 0.0f64;
    let mut den_b = 0.0f64;

    for i in 0..bins {
        let da = ha[i] - mean_a;
        let db = hb[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let den = (den_a * den_b).sqrt();
    if den < 1e-12 {
        // Both histograms are flat — treat as perfectly correlated.
        return 1.0;
    }
    // Map [-1, 1] → [0, 1] for convenience.
    ((num / den + 1.0) / 2.0).clamp(0.0, 1.0) as f32
}

/// Compute the average-hash (aHash) similarity between two frames.
///
/// Downsamples each frame to 8×8, computes the mean, binarises, then
/// returns `1 - hamming_distance(hash_a, hash_b) / 64.0`.
pub fn compute_avg_hash_similarity(frame_a: &[u8], width: u32, height: u32, frame_b: &[u8]) -> f32 {
    let hash_a = avg_hash_8x8(frame_a, width, height);
    let hash_b = avg_hash_8x8(frame_b, width, height);
    1.0 - (hash_a ^ hash_b).count_ones() as f32 / 64.0
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Compute a 64-bit average hash by downsampling a luma plane to 8×8.
fn avg_hash_8x8(frame: &[u8], width: u32, height: u32) -> u64 {
    let sw = width as usize;
    let sh = height as usize;
    if sw == 0 || sh == 0 || frame.is_empty() {
        return 0;
    }

    // Bilinear downscale to 8×8.
    let mut small = [0u8; 64];
    for dy in 0..8usize {
        let src_y_f = (dy as f64 + 0.5) * sh as f64 / 8.0 - 0.5;
        let y0 = (src_y_f.floor() as isize).max(0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (src_y_f - y0 as f64).clamp(0.0, 1.0);

        for dx in 0..8usize {
            let src_x_f = (dx as f64 + 0.5) * sw as f64 / 8.0 - 0.5;
            let x0 = (src_x_f.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (src_x_f - x0 as f64).clamp(0.0, 1.0);

            let p00 = frame.get(y0 * sw + x0).copied().unwrap_or(0) as f64;
            let p01 = frame.get(y0 * sw + x1).copied().unwrap_or(0) as f64;
            let p10 = frame.get(y1 * sw + x0).copied().unwrap_or(0) as f64;
            let p11 = frame.get(y1 * sw + x1).copied().unwrap_or(0) as f64;

            let top = p00 * (1.0 - fx) + p01 * fx;
            let bot = p10 * (1.0 - fx) + p11 * fx;
            let val = top * (1.0 - fy) + bot * fy;
            small[dy * 8 + dx] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    let mean = small.iter().map(|&p| p as u64).sum::<u64>() / 64;
    let mut hash = 0u64;
    for (i, &p) in small.iter().enumerate() {
        if p as u64 >= mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    fn ramp(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    // 1. compute_normalised_sad: identical frames → 0.0
    #[test]
    fn test_sad_identical_zero() {
        let frame = ramp(16, 16);
        let score = compute_normalised_sad(&frame, &frame);
        assert_eq!(score, 0.0);
    }

    // 2. compute_normalised_sad: 0 vs 255 → 1.0
    #[test]
    fn test_sad_opposite_one() {
        let black = flat(8, 8, 0);
        let white = flat(8, 8, 255);
        let score = compute_normalised_sad(&black, &white);
        assert!((score - 1.0).abs() < 1e-5, "expected 1.0, got {score}");
    }

    // 3. compute_normalised_sad is in [0, 1]
    #[test]
    fn test_sad_range() {
        let a = ramp(16, 16);
        let b = flat(16, 16, 128);
        let score = compute_normalised_sad(&a, &b);
        assert!(score >= 0.0 && score <= 1.0);
    }

    // 4. compute_histogram_correlation: identical frames → 1.0
    #[test]
    fn test_histogram_identical() {
        let frame = ramp(16, 16);
        let corr = compute_histogram_correlation(&frame, &frame, 64);
        assert!((corr - 1.0).abs() < 1e-5, "expected 1.0, got {corr}");
    }

    // 5. compute_histogram_correlation: is in [0, 1]
    #[test]
    fn test_histogram_range() {
        let a = flat(16, 16, 0);
        let b = flat(16, 16, 255);
        let corr = compute_histogram_correlation(&a, &b, 64);
        assert!(corr >= 0.0 && corr <= 1.0, "corr {corr} out of range");
    }

    // 6. compute_avg_hash_similarity: identical frames → 1.0
    #[test]
    fn test_hash_identical() {
        let frame = ramp(32, 32);
        let sim = compute_avg_hash_similarity(&frame, 32, 32, &frame);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    // 7. compare_pair: exact identical frames → Exact
    #[test]
    fn test_compare_pair_exact() {
        let det = DuplicateDetector::default();
        let frame = ramp(16, 16);
        let result = det.compare_pair(&frame, &frame, 16, 16, 0, 1).expect("ok");
        assert_eq!(result.kind, DuplicateKind::Exact);
        assert_eq!(result.sad_score, 0.0);
        assert!((result.histogram_correlation - 1.0).abs() < 1e-5);
    }

    // 8. compare_pair: very similar frames → NearDuplicate
    #[test]
    fn test_compare_pair_near_duplicate() {
        let det = DuplicateDetector::default();
        let frame_a = flat(16, 16, 100);
        let mut frame_b = frame_a.clone();
        // Flip one pixel slightly.
        frame_b[0] = 101;
        let result = det
            .compare_pair(&frame_a, &frame_b, 16, 16, 0, 1)
            .expect("ok");
        assert!(
            matches!(
                result.kind,
                DuplicateKind::NearDuplicate | DuplicateKind::Exact
            ),
            "expected near-dup or exact, got {:?}",
            result.kind
        );
    }

    // 9. compare_pair: very different frames → Unique
    #[test]
    fn test_compare_pair_unique() {
        let det = DuplicateDetector::default();
        let frame_a = flat(16, 16, 0);
        let frame_b = flat(16, 16, 255);
        let result = det
            .compare_pair(&frame_a, &frame_b, 16, 16, 0, 1)
            .expect("ok");
        assert_eq!(result.kind, DuplicateKind::Unique);
    }

    // 10. compare_pair: buffer size mismatch → error
    #[test]
    fn test_compare_pair_size_mismatch_error() {
        let det = DuplicateDetector::default();
        let frame_a = flat(16, 16, 100);
        let frame_b = flat(8, 8, 100); // wrong size
        let result = det.compare_pair(&frame_a, &frame_b, 16, 16, 0, 1);
        assert!(result.is_err());
    }

    // 11. scan_sequence: identical frames → single run covering all
    #[test]
    fn test_scan_sequence_all_identical() {
        let det = DuplicateDetector::default();
        let frame = flat(8, 8, 128);
        let frames: Vec<&[u8]> = (0..5).map(|_| frame.as_slice()).collect();
        let runs = det.scan_sequence(&frames, 8, 8).expect("ok");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].start, 0);
        assert_eq!(runs[0].end, 4);
    }

    // 12. scan_sequence: no duplicates → empty run list
    #[test]
    fn test_scan_sequence_no_duplicates() {
        let det = DuplicateDetector::default();
        let frames_owned: Vec<Vec<u8>> = vec![flat(8, 8, 0), flat(8, 8, 128), flat(8, 8, 255)];
        let frames: Vec<&[u8]> = frames_owned.iter().map(|v| v.as_slice()).collect();
        let runs = det.scan_sequence(&frames, 8, 8).expect("ok");
        assert!(runs.is_empty(), "expected no runs, got {} runs", runs.len());
    }

    // 13. scan_sequence: fewer than 2 frames → empty
    #[test]
    fn test_scan_sequence_single_frame_empty() {
        let det = DuplicateDetector::default();
        let frame = flat(8, 8, 128);
        let frames: Vec<&[u8]> = vec![frame.as_slice()];
        let runs = det.scan_sequence(&frames, 8, 8).expect("ok");
        assert!(runs.is_empty());
    }

    // 14. deduplicate_indices: returns correct remove list
    #[test]
    fn test_deduplicate_indices_all_identical() {
        let det = DuplicateDetector::default();
        let frame = flat(8, 8, 100);
        let frames: Vec<&[u8]> = (0..4).map(|_| frame.as_slice()).collect();
        let remove = det.deduplicate_indices(&frames, 8, 8).expect("ok");
        // Keep index 0; remove 1, 2, 3.
        assert_eq!(remove, vec![1, 2, 3]);
    }

    // 15. DuplicateRun::len returns correct count
    #[test]
    fn test_duplicate_run_len() {
        let run = DuplicateRun {
            start: 2,
            end: 6,
            kind: DuplicateKind::Exact,
        };
        assert_eq!(run.len(), 5);
        assert!(run.is_meaningful());
    }

    // 16. DuplicateRun single frame is not meaningful
    #[test]
    fn test_duplicate_run_single_not_meaningful() {
        let run = DuplicateRun {
            start: 3,
            end: 3,
            kind: DuplicateKind::NearDuplicate,
        };
        assert!(!run.is_meaningful());
    }

    // 17. compute_normalised_sad: empty slices → 0.0
    #[test]
    fn test_sad_empty_slices() {
        let score = compute_normalised_sad(&[], &[]);
        assert_eq!(score, 0.0);
    }

    // 18. DuplicateDetector::new stores config
    #[test]
    fn test_detector_new_stores_config() {
        let config = DuplicateDetectorConfig {
            sad_threshold: 0.05,
            histogram_threshold: 0.95,
            hash_threshold: 0.90,
            histogram_bins: 32,
        };
        let det = DuplicateDetector::new(config.clone());
        assert!((det.config.sad_threshold - 0.05).abs() < 1e-6);
        assert_eq!(det.config.histogram_bins, 32);
    }

    // 19. compute_histogram_correlation: empty slices → 0.0
    #[test]
    fn test_histogram_empty() {
        let corr = compute_histogram_correlation(&[], &[], 64);
        assert_eq!(corr, 0.0);
    }

    // 20. scan_sequence: middle duplicate run detected
    #[test]
    fn test_scan_sequence_middle_run() {
        let det = DuplicateDetector::default();
        let frame_unique_a = flat(8, 8, 0);
        let frame_dup = flat(8, 8, 128);
        let frame_unique_b = flat(8, 8, 255);
        let frames: Vec<&[u8]> = vec![
            frame_unique_a.as_slice(),
            frame_dup.as_slice(),
            frame_dup.as_slice(),
            frame_dup.as_slice(),
            frame_unique_b.as_slice(),
        ];
        let runs = det.scan_sequence(&frames, 8, 8).expect("ok");
        // The dup run should be somewhere in the middle.
        let found_dup = runs.iter().any(|r| r.start >= 1 && r.end <= 3);
        assert!(
            found_dup,
            "expected a duplicate run in middle, got {runs:?}"
        );
    }

    // 21. DuplicateKind variants are distinguishable
    #[test]
    fn test_duplicate_kind_variants() {
        assert_ne!(DuplicateKind::Exact, DuplicateKind::NearDuplicate);
        assert_ne!(DuplicateKind::NearDuplicate, DuplicateKind::Unique);
    }

    // 22. compare_pair reports correct indices
    #[test]
    fn test_compare_pair_indices() {
        let det = DuplicateDetector::default();
        let f = flat(8, 8, 100);
        let result = det.compare_pair(&f, &f, 8, 8, 3, 7).expect("ok");
        assert_eq!(result.index_a, 3);
        assert_eq!(result.index_b, 7);
    }

    // 23. hash similarity is in [0, 1] for arbitrary frames
    #[test]
    fn test_hash_similarity_range() {
        let a = ramp(32, 32);
        let b = flat(32, 32, 200);
        let sim = compute_avg_hash_similarity(&a, 32, 32, &b);
        assert!(sim >= 0.0 && sim <= 1.0, "sim {sim} out of range");
    }

    // 24. scan_sequence: wrong buffer size → error
    #[test]
    fn test_scan_sequence_wrong_size_error() {
        let det = DuplicateDetector::default();
        let frame_ok = flat(8, 8, 100);
        let frame_bad = flat(4, 4, 100); // 16 bytes, but 8×8=64 expected
        let frames: Vec<&[u8]> = vec![frame_ok.as_slice(), frame_bad.as_slice()];
        let result = det.scan_sequence(&frames, 8, 8);
        assert!(result.is_err());
    }

    // 25. deduplicate_indices: no duplicates → empty remove list
    #[test]
    fn test_deduplicate_indices_no_dups() {
        let det = DuplicateDetector::default();
        let frames_owned: Vec<Vec<u8>> = vec![flat(8, 8, 10), flat(8, 8, 128), flat(8, 8, 245)];
        let frames: Vec<&[u8]> = frames_owned.iter().map(|v| v.as_slice()).collect();
        let remove = det.deduplicate_indices(&frames, 8, 8).expect("ok");
        assert!(remove.is_empty(), "expected no removals, got {remove:?}");
    }
}
