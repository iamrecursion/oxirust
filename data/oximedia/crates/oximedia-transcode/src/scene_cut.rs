//! Scene cut detection for smart encoding decisions.
//!
//! Provides histogram-based and threshold-based scene change detection
//! that can guide encoder I-frame placement.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Algorithm used to detect scene cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutDetectionMethod {
    /// Simple per-pixel difference threshold.
    Threshold,
    /// Colour histogram comparison.
    Histogram,
    /// Edge-map difference between frames.
    EdgeDiff,
    /// Phase correlation of luminance planes.
    PhaseCor,
}

impl CutDetectionMethod {
    /// Returns the typical false-positive rate (0.0–1.0) for this method.
    #[must_use]
    pub fn typical_false_positive_rate(&self) -> f32 {
        match self {
            Self::Threshold => 0.15,
            Self::Histogram => 0.08,
            Self::EdgeDiff => 0.05,
            Self::PhaseCor => 0.03,
        }
    }
}

/// A detected scene cut at a specific frame.
#[derive(Debug, Clone)]
pub struct SceneCut {
    /// Frame index (0-based) where the cut was detected.
    pub frame: u64,
    /// Confidence value in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Detection method that produced this cut.
    pub method: CutDetectionMethod,
}

impl SceneCut {
    /// Creates a new `SceneCut`.
    #[must_use]
    pub fn new(frame: u64, confidence: f32, method: CutDetectionMethod) -> Self {
        Self {
            frame,
            confidence,
            method,
        }
    }

    /// Returns `true` if the confidence exceeds the hard-cut threshold (0.85).
    #[must_use]
    pub fn is_hard_cut(&self) -> bool {
        self.confidence > 0.85
    }
}

/// Computes a normalised histogram difference between two histograms.
///
/// Both slices must be the same length; returns `0.0` if empty.
/// The result is the sum of absolute differences divided by the total
/// number of pixels represented in `hist_a`.
#[must_use]
pub fn compute_histogram_diff(hist_a: &[u32], hist_b: &[u32]) -> f32 {
    if hist_a.is_empty() || hist_a.len() != hist_b.len() {
        return 0.0;
    }
    let total: u64 = hist_a.iter().map(|&v| u64::from(v)).sum();
    if total == 0 {
        return 0.0;
    }
    let sad: u64 = hist_a
        .iter()
        .zip(hist_b.iter())
        .map(|(&a, &b)| (i64::from(a) - i64::from(b)).unsigned_abs())
        .sum();
    sad as f32 / total as f32
}

/// Detects scene cuts in a sequence of per-frame histograms.
#[derive(Debug, Clone)]
pub struct SceneCutDetector {
    /// Detection method to use.
    pub method: CutDetectionMethod,
    /// Confidence threshold above which a candidate is reported as a cut.
    pub threshold: f32,
}

impl Default for SceneCutDetector {
    fn default() -> Self {
        Self {
            method: CutDetectionMethod::Histogram,
            threshold: 0.4,
        }
    }
}

impl SceneCutDetector {
    /// Creates a new detector with the given method and threshold.
    #[must_use]
    pub fn new(method: CutDetectionMethod, threshold: f32) -> Self {
        Self { method, threshold }
    }

    /// Analyses consecutive frame histograms and returns detected cuts.
    ///
    /// Each element of `frame_histograms` is the histogram for one frame.
    /// Consecutive pairs are compared; if the normalised difference exceeds
    /// `self.threshold` a [`SceneCut`] is recorded at the later frame index.
    #[must_use]
    pub fn detect_cuts(&self, frame_histograms: &[Vec<u32>]) -> Vec<SceneCut> {
        let mut cuts = Vec::new();
        for (i, pair) in frame_histograms.windows(2).enumerate() {
            let diff = compute_histogram_diff(&pair[0], &pair[1]);
            if diff >= self.threshold {
                // Clamp confidence to [0.0, 1.0]
                let confidence = diff.min(1.0);
                cuts.push(SceneCut::new((i + 1) as u64, confidence, self.method));
            }
        }
        cuts
    }

    /// Convenience method: returns the number of cuts in a slice.
    #[must_use]
    pub fn cut_count(cuts: &[SceneCut]) -> usize {
        cuts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CutDetectionMethod ---

    #[test]
    fn test_threshold_fpr() {
        assert!((CutDetectionMethod::Threshold.typical_false_positive_rate() - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_fpr() {
        assert!((CutDetectionMethod::Histogram.typical_false_positive_rate() - 0.08).abs() < 1e-6);
    }

    #[test]
    fn test_edge_diff_fpr() {
        assert!((CutDetectionMethod::EdgeDiff.typical_false_positive_rate() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_phase_cor_fpr() {
        assert!((CutDetectionMethod::PhaseCor.typical_false_positive_rate() - 0.03).abs() < 1e-6);
    }

    // --- SceneCut ---

    #[test]
    fn test_scene_cut_is_hard_cut_true() {
        let cut = SceneCut::new(5, 0.90, CutDetectionMethod::Histogram);
        assert!(cut.is_hard_cut());
    }

    #[test]
    fn test_scene_cut_is_hard_cut_false() {
        let cut = SceneCut::new(5, 0.80, CutDetectionMethod::Histogram);
        assert!(!cut.is_hard_cut());
    }

    #[test]
    fn test_scene_cut_boundary_085() {
        // Exactly 0.85 is NOT a hard cut (strictly greater than)
        let cut = SceneCut::new(1, 0.85, CutDetectionMethod::Threshold);
        assert!(!cut.is_hard_cut());
    }

    // --- compute_histogram_diff ---

    #[test]
    fn test_histogram_diff_identical() {
        let h = vec![10u32, 20, 30, 40];
        assert!((compute_histogram_diff(&h, &h) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_diff_completely_different() {
        let a = vec![100u32, 0, 0, 0];
        let b = vec![0u32, 100, 0, 0];
        // SAD = 200, total = 100 → ratio = 2.0, clamped later at display but raw = 2.0
        let diff = compute_histogram_diff(&a, &b);
        assert!((diff - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_diff_empty() {
        assert_eq!(compute_histogram_diff(&[], &[]), 0.0);
    }

    #[test]
    fn test_histogram_diff_length_mismatch() {
        let a = vec![1u32, 2];
        let b = vec![1u32, 2, 3];
        assert_eq!(compute_histogram_diff(&a, &b), 0.0);
    }

    // --- SceneCutDetector ---

    #[test]
    fn test_default_detector() {
        let det = SceneCutDetector::default();
        assert_eq!(det.method, CutDetectionMethod::Histogram);
        assert!((det.threshold - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_detect_no_cuts_identical_frames() {
        let det = SceneCutDetector::default();
        let frame = vec![50u32; 256];
        let histograms = vec![frame.clone(), frame.clone(), frame.clone()];
        let cuts = det.detect_cuts(&histograms);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_detect_single_cut() {
        let det = SceneCutDetector::new(CutDetectionMethod::Histogram, 0.4);
        // Frame 0: all pixels in bin 0; Frame 1: completely different distribution
        let frame_a = {
            let mut h = vec![0u32; 256];
            h[0] = 1000;
            h
        };
        let frame_b = {
            let mut h = vec![0u32; 256];
            h[255] = 1000;
            h
        };
        let histograms = vec![frame_a, frame_b];
        let cuts = det.detect_cuts(&histograms);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].frame, 1);
    }

    #[test]
    fn test_cut_count_helper() {
        let cuts = vec![
            SceneCut::new(1, 0.9, CutDetectionMethod::Histogram),
            SceneCut::new(5, 0.95, CutDetectionMethod::Histogram),
        ];
        assert_eq!(SceneCutDetector::cut_count(&cuts), 2);
    }
}
