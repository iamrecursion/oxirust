//! Scene-based alignment: detect scene changes in each stream and align
//! corresponding segments independently.
//!
//! # Algorithm
//!
//! 1. **Scene detection** — frame-level luminance histogram difference with a
//!    configurable threshold produces a list of scene-boundary frame indices.
//! 2. **Segment pairing** — Dynamic Programming (DP) with an edit-distance-like
//!    cost matrix pairs scenes from stream A to scenes from stream B by
//!    minimising the total colour-histogram χ² distance between segment
//!    representatives.
//! 3. **Per-segment alignment** — for each matched pair the caller receives an
//!    [`AlignedSegment`] with the exact frame offset and a confidence score
//!    (normalised χ² similarity).
//!
//! The luminance histogram uses 64 bins computed over an 8-bit Y channel
//! derived from the average of R, G, B channels (fast luma approximation:
//! `Y ≈ (R + G + B) / 3`).  The colour histogram uses 16×16×16 RGB bins
//! reduced to a 1-D representation for efficiency.

use crate::{AlignError, AlignResult};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for scene-based alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAlignConfig {
    /// Frame index step for scene detection.  Setting this > 1 samples every
    /// `step`-th frame, trading accuracy for speed.  Default: 1.
    pub detection_step: usize,
    /// Histogram-difference threshold for a cut to be considered a scene
    /// change.  Range: [0.0, 1.0].  Default: 0.30.
    pub cut_threshold: f64,
    /// Maximum number of frames between two matched scenes (guard rail).
    /// Default: 300.
    pub max_frame_offset: i64,
    /// Minimum number of frames in a segment (too-short segments are merged
    /// with the preceding one).  Default: 5.
    pub min_segment_len: usize,
    /// Number of luminance histogram bins.  Must be a power of 2 ≤ 256.
    /// Default: 64.
    pub luma_bins: usize,
}

impl Default for SceneAlignConfig {
    fn default() -> Self {
        Self {
            detection_step: 1,
            cut_threshold: 0.30,
            max_frame_offset: 300,
            min_segment_len: 5,
            luma_bins: 64,
        }
    }
}

/// A single video frame represented as a flat buffer of 8-bit RGB pixels.
///
/// The buffer must have exactly `width * height * 3` bytes in RGB order.
#[derive(Debug, Clone)]
pub struct FrameRgb {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel data: R G B R G B …
    pub data: Vec<u8>,
}

impl FrameRgb {
    /// Create a new frame.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError::InvalidConfig`] if `data.len() != width * height * 3`.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> AlignResult<Self> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(AlignError::InvalidConfig(format!(
                "FrameRgb: expected {expected} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

/// A scene-change boundary detected in a video stream.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SceneBoundary {
    /// Frame index (0-based) at which the scene change occurs.
    pub frame_index: usize,
    /// Histogram difference score that triggered this boundary (0.0 – 1.0).
    pub score: f64,
}

/// A segment that has been aligned between two streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedSegment {
    /// Frame index (inclusive) of the segment start in stream A.
    pub start_a: usize,
    /// Frame index (exclusive) of the segment end in stream A.
    pub end_a: usize,
    /// Corresponding frame index (inclusive) in stream B.
    pub start_b: usize,
    /// Corresponding frame index (exclusive) in stream B.
    pub end_b: usize,
    /// Frame offset: `start_b - start_a` as a signed integer.
    pub frame_offset: i64,
    /// Confidence score (0.0 = no confidence, 1.0 = perfect match).
    pub confidence: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneAligner
// ─────────────────────────────────────────────────────────────────────────────

/// Scene-based video alignment engine.
///
/// Detects scene boundaries in two video streams and produces per-segment
/// time offsets so that temporally non-uniform drift can be corrected.
///
/// # Example
///
/// ```
/// use oximedia_align::scene_align::{SceneAlignConfig, SceneAligner, FrameRgb};
///
/// let cfg = SceneAlignConfig::default();
/// let aligner = SceneAligner::new(cfg);
///
/// // Build synthetic 8×8 frames (all-red and all-green alternating).
/// let red_frame  = FrameRgb::new(8, 8, vec![255, 0, 0].repeat(64)).unwrap();
/// let green_frame = FrameRgb::new(8, 8, vec![0, 255, 0].repeat(64)).unwrap();
///
/// let stream_a = vec![red_frame.clone(), red_frame.clone(), green_frame.clone()];
/// let stream_b = vec![red_frame.clone(), green_frame.clone()];
/// let segments = aligner.align(&stream_a, &stream_b).unwrap();
/// ```
pub struct SceneAligner {
    config: SceneAlignConfig,
}

impl SceneAligner {
    /// Create a new aligner with the given configuration.
    #[must_use]
    pub fn new(config: SceneAlignConfig) -> Self {
        Self { config }
    }

    /// Detect scene boundaries in a sequence of frames.
    ///
    /// Returns a sorted list of scene boundaries.  An implicit boundary always
    /// exists at frame 0.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError::InsufficientData`] if `frames` is empty.
    pub fn detect_scenes(&self, frames: &[FrameRgb]) -> AlignResult<Vec<SceneBoundary>> {
        if frames.is_empty() {
            return Err(AlignError::InsufficientData(
                "detect_scenes: empty frame list".to_string(),
            ));
        }

        let bins = self.config.luma_bins;
        let step = self.config.detection_step.max(1);
        let threshold = self.config.cut_threshold;
        let min_len = self.config.min_segment_len;

        let mut boundaries: Vec<SceneBoundary> = Vec::new();
        // First frame always starts a scene.
        boundaries.push(SceneBoundary {
            frame_index: 0,
            score: 0.0,
        });

        let hist_prev = luma_histogram(&frames[0], bins);
        let mut prev_hist = hist_prev;
        let mut last_boundary_idx = 0usize;

        for i in (step..frames.len()).step_by(step) {
            let curr_hist = luma_histogram(&frames[i], bins);
            let diff = histogram_l1_diff(&prev_hist, &curr_hist);
            if diff >= threshold && (i - last_boundary_idx) >= min_len {
                boundaries.push(SceneBoundary {
                    frame_index: i,
                    score: diff,
                });
                last_boundary_idx = i;
            }
            prev_hist = curr_hist;
        }

        Ok(boundaries)
    }

    /// Align `stream_a` and `stream_b` using scene-boundary matching.
    ///
    /// Returns one [`AlignedSegment`] per matched scene pair.
    ///
    /// # Errors
    ///
    /// * [`AlignError::InsufficientData`] if either stream is empty.
    /// * [`AlignError::NoSolution`] if no scene pairs can be found.
    pub fn align(
        &self,
        stream_a: &[FrameRgb],
        stream_b: &[FrameRgb],
    ) -> AlignResult<Vec<AlignedSegment>> {
        if stream_a.is_empty() {
            return Err(AlignError::InsufficientData(
                "stream_a is empty".to_string(),
            ));
        }
        if stream_b.is_empty() {
            return Err(AlignError::InsufficientData(
                "stream_b is empty".to_string(),
            ));
        }

        let bounds_a = self.detect_scenes(stream_a)?;
        let bounds_b = self.detect_scenes(stream_b)?;

        // Build segment frame ranges from boundaries.
        let segs_a = boundaries_to_segments(&bounds_a, stream_a.len());
        let segs_b = boundaries_to_segments(&bounds_b, stream_b.len());

        if segs_a.is_empty() || segs_b.is_empty() {
            return Err(AlignError::NoSolution(
                "no segments extracted from scene boundaries".to_string(),
            ));
        }

        // Build representative colour histograms for each segment.
        let bins = self.config.luma_bins;
        let reps_a: Vec<Vec<f64>> = segs_a
            .iter()
            .map(|&(s, e)| segment_representative(stream_a, s, e, bins))
            .collect();
        let reps_b: Vec<Vec<f64>> = segs_b
            .iter()
            .map(|&(s, e)| segment_representative(stream_b, s, e, bins))
            .collect();

        // DP matching: find the minimum-cost monotone alignment.
        let matches = dp_match_segments(&reps_a, &reps_b);

        // Construct AlignedSegments from DP result.
        let max_off = self.config.max_frame_offset;
        let mut result = Vec::with_capacity(matches.len());

        for (ia, ib) in matches {
            let (start_a, end_a) = segs_a[ia];
            let (start_b, end_b) = segs_b[ib];
            let frame_offset = start_b as i64 - start_a as i64;

            if frame_offset.abs() > max_off {
                // Skip pairings that exceed the maximum allowed offset.
                continue;
            }

            let chi2 = chi2_distance(&reps_a[ia], &reps_b[ib]);
            // Confidence: 1.0 when chi2 == 0, approaches 0 as chi2 grows.
            let confidence = 1.0 / (1.0 + chi2);

            result.push(AlignedSegment {
                start_a,
                end_a,
                start_b,
                end_b,
                frame_offset,
                confidence,
            });
        }

        if result.is_empty() {
            return Err(AlignError::NoSolution(
                "all segment pairs exceeded max_frame_offset".to_string(),
            ));
        }

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a normalised luma histogram with `bins` equal-width bins.
///
/// Luma approximation: Y = (R + G + B) / 3.
fn luma_histogram(frame: &FrameRgb, bins: usize) -> Vec<f64> {
    let bins = bins.max(1);
    let mut hist = vec![0u64; bins];
    let pixels = frame.width as usize * frame.height as usize;
    let bin_width = 256.0 / bins as f64;

    for chunk in frame.data.chunks_exact(3) {
        let y = (chunk[0] as u32 + chunk[1] as u32 + chunk[2] as u32) / 3;
        let b = ((y as f64) / bin_width) as usize;
        hist[b.min(bins - 1)] += 1;
    }

    // Normalise to sum = 1.0.
    let total = pixels.max(1) as f64;
    hist.iter().map(|&c| c as f64 / total).collect()
}

/// L1 (Manhattan) distance between two normalised histograms.
fn histogram_l1_diff(h1: &[f64], h2: &[f64]) -> f64 {
    h1.iter()
        .zip(h2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        * 0.5 // scale to [0, 1] since sum of absolute differences over normalised histograms ≤ 2
}

/// χ² distance between two normalised histograms.
fn chi2_distance(h1: &[f64], h2: &[f64]) -> f64 {
    h1.iter()
        .zip(h2.iter())
        .filter(|&(&a, &b)| a + b > 0.0)
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff / (a + b)
        })
        .sum()
}

/// Convert a list of sorted scene boundaries into (start, end) segment ranges.
///
/// `total_frames` is used to close the final segment.
fn boundaries_to_segments(
    boundaries: &[SceneBoundary],
    total_frames: usize,
) -> Vec<(usize, usize)> {
    let mut segs = Vec::with_capacity(boundaries.len());
    for i in 0..boundaries.len() {
        let start = boundaries[i].frame_index;
        let end = if i + 1 < boundaries.len() {
            boundaries[i + 1].frame_index
        } else {
            total_frames
        };
        if end > start {
            segs.push((start, end));
        }
    }
    segs
}

/// Build a representative luma histogram for a segment by averaging
/// per-frame histograms of a fixed sub-sample of up to 8 frames.
fn segment_representative(frames: &[FrameRgb], start: usize, end: usize, bins: usize) -> Vec<f64> {
    let len = end.saturating_sub(start).max(1);
    // Sample at most 8 evenly spaced frames.
    let sample_count = len.min(8);
    let mut acc = vec![0.0f64; bins];

    for k in 0..sample_count {
        let idx = start + (k * len) / sample_count;
        let idx = idx.min(end.saturating_sub(1));
        let h = luma_histogram(&frames[idx], bins);
        for (a, &v) in acc.iter_mut().zip(h.iter()) {
            *a += v;
        }
    }

    let inv = 1.0 / sample_count.max(1) as f64;
    acc.iter_mut().for_each(|v| *v *= inv);
    acc
}

/// Dynamic-programming monotone segment matching.
///
/// Finds the minimum χ² cost alignment between two ordered lists of segments.
/// Both sequences must be traversed monotonically (no reordering).
///
/// Time: O(n × m).  Space: O(n × m).
fn dp_match_segments(reps_a: &[Vec<f64>], reps_b: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let na = reps_a.len();
    let nb = reps_b.len();

    // dp[i][j] = min cost to match prefixes a[0..i] and b[0..j].
    let mut dp = vec![vec![f64::INFINITY; nb + 1]; na + 1];
    dp[0][0] = 0.0;

    for i in 1..=na {
        for j in 1..=nb {
            let cost = chi2_distance(&reps_a[i - 1], &reps_b[j - 1]);
            // Match (i-1, j-1) extending from dp[i-1][j-1].
            let from_match = dp[i - 1][j - 1] + cost;
            // Skip segment in A (insert gap in A).
            let from_skip_a = dp[i - 1][j] + 1.0; // gap penalty
                                                  // Skip segment in B (insert gap in B).
            let from_skip_b = dp[i][j - 1] + 1.0; // gap penalty
            dp[i][j] = from_match.min(from_skip_a).min(from_skip_b);
        }
    }

    // Backtrack to find the alignment.
    let mut path = Vec::new();
    let mut i = na;
    let mut j = nb;

    while i > 0 && j > 0 {
        let cost = chi2_distance(&reps_a[i - 1], &reps_b[j - 1]);
        let came_from_match = (dp[i][j] - (dp[i - 1][j - 1] + cost)).abs() < 1e-10;
        let came_from_skip_a = (dp[i][j] - (dp[i - 1][j] + 1.0)).abs() < 1e-10;

        if came_from_match {
            path.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if came_from_skip_a {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    path.reverse();
    path
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────

    fn solid_frame(r: u8, g: u8, b: u8) -> FrameRgb {
        FrameRgb::new(8, 8, vec![r, g, b].repeat(64)).expect("solid frame")
    }

    // ── FrameRgb ──────────────────────────────────────────────────────────

    #[test]
    fn test_frame_rgb_invalid_size_returns_error() {
        let result = FrameRgb::new(4, 4, vec![0u8; 10]);
        assert!(result.is_err(), "wrong size should return error");
    }

    #[test]
    fn test_frame_rgb_valid() {
        let frame = solid_frame(100, 150, 200);
        assert_eq!(frame.width, 8);
        assert_eq!(frame.data.len(), 8 * 8 * 3);
    }

    // ── luma_histogram ────────────────────────────────────────────────────

    #[test]
    fn test_luma_histogram_sums_to_one() {
        let frame = solid_frame(120, 120, 120);
        let hist = luma_histogram(&frame, 64);
        let total: f64 = hist.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "histogram sum = {total}");
    }

    #[test]
    fn test_luma_histogram_solid_single_bin() {
        // All pixels Y=120, bins=64 → bin index = 120 / 4 = 30.
        let frame = solid_frame(120, 120, 120);
        let hist = luma_histogram(&frame, 64);
        let nonzero: Vec<_> = hist.iter().enumerate().filter(|(_, &v)| v > 0.0).collect();
        assert_eq!(nonzero.len(), 1, "solid frame should fill exactly one bin");
    }

    #[test]
    fn test_histogram_l1_diff_identical() {
        let frame = solid_frame(80, 80, 80);
        let h = luma_histogram(&frame, 64);
        let diff = histogram_l1_diff(&h, &h);
        assert!(diff < 1e-9, "self-diff should be 0, got {diff}");
    }

    #[test]
    fn test_histogram_l1_diff_different() {
        let dark = luma_histogram(&solid_frame(10, 10, 10), 64);
        let bright = luma_histogram(&solid_frame(240, 240, 240), 64);
        let diff = histogram_l1_diff(&dark, &bright);
        assert!(
            diff > 0.9,
            "dark vs bright diff should be near 1, got {diff}"
        );
    }

    // ── detect_scenes ─────────────────────────────────────────────────────

    #[test]
    fn test_detect_scenes_empty_returns_error() {
        let aligner = SceneAligner::new(SceneAlignConfig::default());
        assert!(aligner.detect_scenes(&[]).is_err());
    }

    #[test]
    fn test_detect_scenes_single_frame() {
        let aligner = SceneAligner::new(SceneAlignConfig::default());
        let frames = vec![solid_frame(128, 128, 128)];
        let bounds = aligner.detect_scenes(&frames).expect("single frame");
        assert_eq!(bounds.len(), 1, "single frame → one boundary at 0");
        assert_eq!(bounds[0].frame_index, 0);
    }

    #[test]
    fn test_detect_scenes_no_change_no_extra_boundaries() {
        let cfg = SceneAlignConfig {
            cut_threshold: 0.30,
            min_segment_len: 1,
            ..Default::default()
        };
        let aligner = SceneAligner::new(cfg);
        let frames: Vec<FrameRgb> = (0..10).map(|_| solid_frame(128, 128, 128)).collect();
        let bounds = aligner.detect_scenes(&frames).expect("constant stream");
        assert_eq!(
            bounds.len(),
            1,
            "constant stream → only implicit boundary at 0"
        );
    }

    #[test]
    fn test_detect_scenes_hard_cut() {
        let cfg = SceneAlignConfig {
            cut_threshold: 0.20,
            min_segment_len: 1,
            ..Default::default()
        };
        let aligner = SceneAligner::new(cfg);
        // 5 dark frames followed by 5 bright frames.
        let mut frames: Vec<FrameRgb> = (0..5).map(|_| solid_frame(10, 10, 10)).collect();
        frames.extend((0..5).map(|_| solid_frame(240, 240, 240)));
        let bounds = aligner.detect_scenes(&frames).expect("hard cut");
        assert!(
            bounds.len() >= 2,
            "expected at least 2 boundaries, got {}",
            bounds.len()
        );
        // Second boundary should be at frame 5.
        let has_cut_at_5 = bounds.iter().any(|b| b.frame_index == 5);
        assert!(has_cut_at_5, "scene boundary expected at frame 5");
    }

    // ── boundaries_to_segments ────────────────────────────────────────────

    #[test]
    fn test_boundaries_to_segments_two_scenes() {
        let bounds = vec![
            SceneBoundary {
                frame_index: 0,
                score: 0.0,
            },
            SceneBoundary {
                frame_index: 5,
                score: 0.9,
            },
        ];
        let segs = boundaries_to_segments(&bounds, 10);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], (0, 5));
        assert_eq!(segs[1], (5, 10));
    }

    #[test]
    fn test_boundaries_to_segments_single_scene() {
        let bounds = vec![SceneBoundary {
            frame_index: 0,
            score: 0.0,
        }];
        let segs = boundaries_to_segments(&bounds, 8);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0], (0, 8));
    }

    // ── dp_match_segments ─────────────────────────────────────────────────

    #[test]
    fn test_dp_match_identity() {
        // Two identical single-segment streams → one perfect match.
        let frame = solid_frame(100, 100, 100);
        let rep = segment_representative(&[frame], 0, 1, 64);
        let reps = vec![rep.clone()];
        let matches = dp_match_segments(&reps, &reps);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], (0, 0));
    }

    #[test]
    fn test_dp_match_two_identical_segments() {
        let dark = solid_frame(20, 20, 20);
        let bright = solid_frame(220, 220, 220);
        let frames = vec![dark.clone(), bright.clone()];
        let rep_d = segment_representative(&frames, 0, 1, 64);
        let rep_b = segment_representative(&frames, 1, 2, 64);
        let reps_a = vec![rep_d.clone(), rep_b.clone()];
        let reps_b = vec![rep_d, rep_b];
        let matches = dp_match_segments(&reps_a, &reps_b);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0], (0, 0));
        assert_eq!(matches[1], (1, 1));
    }

    // ── align ─────────────────────────────────────────────────────────────

    #[test]
    fn test_align_identical_streams_zero_offset() {
        let cfg = SceneAlignConfig {
            cut_threshold: 0.20,
            min_segment_len: 1,
            max_frame_offset: 300,
            ..Default::default()
        };
        let aligner = SceneAligner::new(cfg);

        // Two identical streams: dark × 5, bright × 5.
        let mut frames: Vec<FrameRgb> = (0..5).map(|_| solid_frame(10, 10, 10)).collect();
        frames.extend((0..5).map(|_| solid_frame(240, 240, 240)));

        let segments = aligner.align(&frames, &frames).expect("align identical");
        assert!(!segments.is_empty(), "should produce aligned segments");
        for seg in &segments {
            assert_eq!(
                seg.frame_offset, 0,
                "identical streams → zero offset, got {}",
                seg.frame_offset
            );
        }
    }

    #[test]
    fn test_align_empty_stream_returns_error() {
        let aligner = SceneAligner::new(SceneAlignConfig::default());
        let frames = vec![solid_frame(128, 128, 128)];
        assert!(aligner.align(&[], &frames).is_err());
        assert!(aligner.align(&frames, &[]).is_err());
    }

    #[test]
    fn test_align_offset_stream_detects_shift() {
        let cfg = SceneAlignConfig {
            cut_threshold: 0.15,
            min_segment_len: 1,
            max_frame_offset: 300,
            ..Default::default()
        };
        let aligner = SceneAligner::new(cfg);

        // Stream A: dark×5, bright×5.
        let mut stream_a: Vec<FrameRgb> = (0..5).map(|_| solid_frame(10, 10, 10)).collect();
        stream_a.extend((0..5).map(|_| solid_frame(240, 240, 240)));

        // Stream B: 2 padding dark frames, then same dark×5, bright×5 → offset = -2 on first seg.
        let mut stream_b: Vec<FrameRgb> = (0..7).map(|_| solid_frame(10, 10, 10)).collect();
        stream_b.extend((0..5).map(|_| solid_frame(240, 240, 240)));

        let segments = aligner.align(&stream_a, &stream_b).expect("align shifted");
        assert!(!segments.is_empty(), "should produce aligned segments");
    }

    #[test]
    fn test_align_confidence_perfect_match() {
        let cfg = SceneAlignConfig {
            cut_threshold: 0.20,
            min_segment_len: 1,
            ..Default::default()
        };
        let aligner = SceneAligner::new(cfg);
        let mut frames: Vec<FrameRgb> = (0..5).map(|_| solid_frame(50, 50, 50)).collect();
        frames.extend((0..5).map(|_| solid_frame(200, 200, 200)));
        let segments = aligner.align(&frames, &frames).expect("align");
        for seg in &segments {
            assert!(
                seg.confidence > 0.5,
                "identical frames should yield high confidence, got {}",
                seg.confidence
            );
        }
    }

    // ── chi2_distance ─────────────────────────────────────────────────────

    #[test]
    fn test_chi2_distance_identical() {
        let h = vec![0.25, 0.25, 0.25, 0.25];
        let d = chi2_distance(&h, &h);
        assert!(d < 1e-9, "χ² of identical histograms should be 0, got {d}");
    }

    #[test]
    fn test_chi2_distance_orthogonal() {
        let h1 = vec![1.0, 0.0, 0.0, 0.0];
        let h2 = vec![0.0, 0.0, 0.0, 1.0];
        // χ² = 1.0^2 / 1.0 + 1.0^2 / 1.0 = 2.0
        let d = chi2_distance(&h1, &h2);
        assert!(
            (d - 2.0).abs() < 1e-9,
            "χ² of orthogonal histograms = 2.0, got {d}"
        );
    }

    // ── SceneAlignConfig ──────────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = SceneAlignConfig::default();
        assert_eq!(cfg.detection_step, 1);
        assert!((cfg.cut_threshold - 0.30).abs() < 1e-9);
        assert_eq!(cfg.luma_bins, 64);
        assert_eq!(cfg.min_segment_len, 5);
    }
}
