//! Automatic highlight detection for sports and event production.
//!
//! Combines cut density (editorial rhythm) with audio energy peaks to
//! identify exciting segments in a multi-camera recording.
//!
//! # Algorithm
//!
//! For each candidate window:
//! 1. Count the number of `CutPoint`s whose `frame_idx` falls within the
//!    window.  Divide by the window size to obtain a normalised *cut density*.
//! 2. Average the `FrameEnergy::energy` values inside the window for the
//!    *mean audio energy*.
//! 3. Compute a weighted highlight score:
//!    `score = w_energy * energy_norm + w_density * density_norm`
//!    where `w_energy + w_density = 1`.
//! 4. Emit a [`HighlightSegment`] when `score ≥ highlight_threshold`.
//!
//! Adjacent or overlapping segments are optionally merged with
//! [`HighlightDetector::merge_overlapping`].

use crate::cut_point::CutPoint;

// ── FrameEnergy ───────────────────────────────────────────────────────────────

/// Per-frame audio energy measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameEnergy {
    /// Frame index (0-based).
    pub frame_idx: u64,
    /// RMS energy for this frame (non-negative).
    pub energy: f32,
}

impl FrameEnergy {
    /// Create a new `FrameEnergy`, clamping `energy` to `[0, ∞)`.
    #[must_use]
    pub fn new(frame_idx: u64, energy: f32) -> Self {
        Self {
            frame_idx,
            energy: energy.max(0.0),
        }
    }
}

// ── HighlightSegment ──────────────────────────────────────────────────────────

/// A detected highlight segment in the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightSegment {
    /// First frame of the highlight (inclusive).
    pub start_frame: u64,
    /// Last frame of the highlight (inclusive).
    pub end_frame: u64,
    /// Normalised highlight score in \[0.0, 1.0\].
    pub score: f32,
}

impl HighlightSegment {
    /// Duration of the segment in frames (always ≥ 1).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// `true` when this segment overlaps or is adjacent to `other`.
    #[must_use]
    pub fn overlaps_or_adjacent(&self, other: &Self) -> bool {
        // Adjacent means end+1 == other.start or vice versa.
        self.start_frame <= other.end_frame + 1 && other.start_frame <= self.end_frame + 1
    }

    /// Merge this segment with `other`, taking the maximum score.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            start_frame: self.start_frame.min(other.start_frame),
            end_frame: self.end_frame.max(other.end_frame),
            score: self.score.max(other.score),
        }
    }
}

// ── HighlightDetector ─────────────────────────────────────────────────────────

/// Detects highlight segments from audio energy and cut density.
///
/// # Example
///
/// ```
/// use oximedia_multicam::highlight::{HighlightDetector, FrameEnergy};
/// use oximedia_multicam::cut_point::{CutPoint, CutPointKind};
///
/// let detector = HighlightDetector::new(0.6, 0.3);
///
/// // Build 120 frames of moderate energy with a spike at frames 50–70.
/// let mut frames: Vec<FrameEnergy> = (0..120)
///     .map(|i| FrameEnergy::new(i, 0.1))
///     .collect();
/// for i in 50..70 {
///     frames[i] = FrameEnergy::new(i as u64, 0.95);
/// }
///
/// // Several cuts in the high-energy region.
/// let cuts = vec![
///     CutPoint::new(52, CutPointKind::AudioEnergy, 0.9),
///     CutPoint::new(58, CutPointKind::AudioEnergy, 0.85),
///     CutPoint::new(64, CutPointKind::AudioEnergy, 0.88),
/// ];
///
/// let highlights = detector.detect(&frames, &cuts);
/// assert!(!highlights.is_empty());
/// assert!(highlights[0].score >= 0.3);
/// ```
#[derive(Debug, Clone)]
pub struct HighlightDetector {
    /// Minimum mean audio energy (in the window) to be considered high energy.
    pub energy_threshold: f32,
    /// Minimum cut density (cuts per frame) to be considered high density.
    pub cut_density_threshold: f32,
    /// Length of the analysis window in frames.
    pub window_frames: u64,
    /// Minimum score required to emit a [`HighlightSegment`].
    pub highlight_threshold: f32,
    /// Relative weight of the energy component in the score (0.0–1.0).
    pub energy_weight: f32,
}

impl HighlightDetector {
    /// Create a `HighlightDetector` with the given primary thresholds.
    ///
    /// `energy_threshold` — minimum mean frame energy in a window to score
    /// non-zero on the energy axis.
    ///
    /// `cut_density_threshold` — minimum cuts-per-frame in a window to score
    /// non-zero on the density axis.
    ///
    /// The analysis window defaults to 30 frames; the minimum highlight score
    /// defaults to 0.3; energy and density are weighted 60%/40%.
    #[must_use]
    pub fn new(energy_threshold: f32, cut_density_threshold: f32) -> Self {
        Self {
            energy_threshold: energy_threshold.max(0.0),
            cut_density_threshold: cut_density_threshold.max(0.0),
            window_frames: 30,
            highlight_threshold: 0.3,
            energy_weight: 0.6,
        }
    }

    /// Override the analysis window size (frames).
    #[must_use]
    pub fn with_window(mut self, window_frames: u64) -> Self {
        self.window_frames = window_frames.max(1);
        self
    }

    /// Override the minimum score to emit a highlight.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.highlight_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Override the energy weight (density weight becomes `1 - energy_weight`).
    #[must_use]
    pub fn with_energy_weight(mut self, w: f32) -> Self {
        self.energy_weight = w.clamp(0.0, 1.0);
        self
    }

    /// Detect highlight segments from `frames` (per-frame energy) and `cuts`.
    ///
    /// Returns a list of [`HighlightSegment`]s sorted by `start_frame`.
    #[must_use]
    pub fn detect(&self, frames: &[FrameEnergy], cuts: &[CutPoint]) -> Vec<HighlightSegment> {
        if frames.is_empty() {
            return Vec::new();
        }

        let total_frames = frames.len() as u64;
        let step = (self.window_frames / 2).max(1); // 50% overlap

        // Pre-compute global energy statistics for normalisation.
        let max_energy = frames.iter().map(|f| f.energy).fold(0.0f32, f32::max);
        let energy_scale = if max_energy > 0.0 { max_energy } else { 1.0 };

        // Maximum possible cuts in a window (for density normalisation).
        let max_possible_cuts = self.window_frames as f32 * self.cut_density_threshold * 4.0;
        let density_scale = if max_possible_cuts > 0.0 {
            max_possible_cuts
        } else {
            1.0
        };

        let density_weight = 1.0 - self.energy_weight;
        let mut segments = Vec::new();

        let mut window_start = 0u64;
        while window_start < total_frames {
            let window_end = (window_start + self.window_frames - 1).min(total_frames - 1);

            // Mean energy in the window.
            let window_energy = self.mean_energy_in_window(frames, window_start, window_end);

            // Cut density in the window.
            let cut_count = self.cuts_in_window(cuts, window_start, window_end) as f32;
            let density = cut_count / density_scale;

            // Normalise energy relative to global max.
            let energy_norm = (window_energy / energy_scale).clamp(0.0, 1.0);
            let density_norm = density.clamp(0.0, 1.0);

            // Weighted score.
            let score = self.energy_weight * energy_norm + density_weight * density_norm;

            // Threshold gate: also require at least one criterion to be met.
            let energy_met = window_energy >= self.energy_threshold;
            let density_met = cut_count / (self.window_frames as f32) >= self.cut_density_threshold;

            if score >= self.highlight_threshold && (energy_met || density_met) {
                segments.push(HighlightSegment {
                    start_frame: window_start,
                    end_frame: window_end,
                    score,
                });
            }

            window_start += step;
        }

        // Merge overlapping segments.
        Self::merge_overlapping(segments)
    }

    /// Mean energy of `frames` within `[start, end]` (inclusive).
    fn mean_energy_in_window(&self, frames: &[FrameEnergy], start: u64, end: u64) -> f32 {
        let slice: Vec<f32> = frames
            .iter()
            .filter(|f| f.frame_idx >= start && f.frame_idx <= end)
            .map(|f| f.energy)
            .collect();

        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Number of cut points within `[start, end]` (inclusive).
    fn cuts_in_window(&self, cuts: &[CutPoint], start: u64, end: u64) -> usize {
        cuts.iter()
            .filter(|c| c.frame_idx >= start && c.frame_idx <= end)
            .count()
    }

    /// Merge a sorted-by-start_frame list of segments that overlap or are
    /// adjacent, taking the maximum score.
    #[must_use]
    pub fn merge_overlapping(mut segments: Vec<HighlightSegment>) -> Vec<HighlightSegment> {
        if segments.is_empty() {
            return segments;
        }
        segments.sort_by_key(|s| s.start_frame);
        let mut merged: Vec<HighlightSegment> = Vec::with_capacity(segments.len());

        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if last.overlaps_or_adjacent(&seg) {
                    *last = last.merge(&seg);
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut_point::{CutPoint, CutPointKind};

    // ── FrameEnergy ──────────────────────────────────────────────────────────

    #[test]
    fn test_frame_energy_clamps_negative() {
        let fe = FrameEnergy::new(0, -1.0);
        assert_eq!(fe.energy, 0.0);
    }

    #[test]
    fn test_frame_energy_stores_values() {
        let fe = FrameEnergy::new(42, 0.75);
        assert_eq!(fe.frame_idx, 42);
        assert!((fe.energy - 0.75).abs() < 1e-6);
    }

    // ── HighlightSegment ─────────────────────────────────────────────────────

    #[test]
    fn test_highlight_segment_duration() {
        let seg = HighlightSegment {
            start_frame: 10,
            end_frame: 19,
            score: 0.8,
        };
        assert_eq!(seg.duration_frames(), 10);
    }

    #[test]
    fn test_highlight_segment_overlaps_adjacent() {
        let a = HighlightSegment {
            start_frame: 0,
            end_frame: 9,
            score: 0.5,
        };
        let b = HighlightSegment {
            start_frame: 10,
            end_frame: 19,
            score: 0.6,
        };
        assert!(a.overlaps_or_adjacent(&b));
    }

    #[test]
    fn test_highlight_segment_does_not_overlap_gap() {
        let a = HighlightSegment {
            start_frame: 0,
            end_frame: 8,
            score: 0.5,
        };
        let b = HighlightSegment {
            start_frame: 10,
            end_frame: 19,
            score: 0.6,
        };
        assert!(!a.overlaps_or_adjacent(&b));
    }

    #[test]
    fn test_highlight_segment_merge_takes_max_score() {
        let a = HighlightSegment {
            start_frame: 0,
            end_frame: 10,
            score: 0.5,
        };
        let b = HighlightSegment {
            start_frame: 8,
            end_frame: 20,
            score: 0.9,
        };
        let m = a.merge(&b);
        assert_eq!(m.start_frame, 0);
        assert_eq!(m.end_frame, 20);
        assert!((m.score - 0.9).abs() < 1e-6);
    }

    // ── HighlightDetector ────────────────────────────────────────────────────

    /// No frames → empty output.
    #[test]
    fn test_detect_empty_frames() {
        let d = HighlightDetector::new(0.5, 0.1);
        let result = d.detect(&[], &[]);
        assert!(result.is_empty());
    }

    /// All-silence signal with no cuts → no highlights.
    #[test]
    fn test_detect_silent_no_cuts_no_highlights() {
        let d = HighlightDetector::new(0.5, 0.1).with_threshold(0.4);
        let frames: Vec<FrameEnergy> = (0..120).map(|i| FrameEnergy::new(i, 0.0)).collect();
        let result = d.detect(&frames, &[]);
        assert!(result.is_empty(), "Got {} highlights", result.len());
    }

    /// High energy region should be detected.
    #[test]
    fn test_detect_high_energy_region() {
        let d = HighlightDetector::new(0.5, 0.05)
            .with_window(20)
            .with_threshold(0.25);

        let mut frames: Vec<FrameEnergy> = (0..100u64).map(|i| FrameEnergy::new(i, 0.05)).collect();
        // Inject high energy at frames 40–60.
        for i in 40..60 {
            frames[i] = FrameEnergy::new(i as u64, 0.95);
        }
        let result = d.detect(&frames, &[]);
        assert!(!result.is_empty(), "Expected at least one highlight");
        // The highlight should cover the high-energy region.
        let covers = result
            .iter()
            .any(|s| s.start_frame <= 40 && s.end_frame >= 59);
        assert!(covers, "No segment covers frames 40-59; got {:?}", result);
    }

    /// Dense cuts region should be detected.
    #[test]
    fn test_detect_dense_cuts_region() {
        let d = HighlightDetector::new(0.3, 0.05)
            .with_window(20)
            .with_threshold(0.15)
            .with_energy_weight(0.2); // weight density heavily

        let frames: Vec<FrameEnergy> = (0..100u64).map(|i| FrameEnergy::new(i, 0.4)).collect();
        // Many cuts at frames 50–70.
        let cuts: Vec<CutPoint> = (50u64..70)
            .step_by(3)
            .map(|i| CutPoint::new(i, CutPointKind::AudioEnergy, 0.9))
            .collect();

        let result = d.detect(&frames, &cuts);
        assert!(!result.is_empty(), "Expected highlight from dense cuts");
    }

    /// Combined high energy + high cut density → strong highlight score.
    #[test]
    fn test_detect_combined_energy_and_cuts() {
        let d = HighlightDetector::new(0.5, 0.05)
            .with_window(20)
            .with_threshold(0.2);

        let mut frames: Vec<FrameEnergy> = (0..120u64).map(|i| FrameEnergy::new(i, 0.1)).collect();
        for i in 50..70 {
            frames[i] = FrameEnergy::new(i as u64, 0.9);
        }
        let cuts: Vec<CutPoint> = (52u64..68)
            .step_by(4)
            .map(|i| CutPoint::new(i, CutPointKind::AudioEnergy, 0.85))
            .collect();

        let result = d.detect(&frames, &cuts);
        assert!(!result.is_empty());
        let max_score = result.iter().map(|s| s.score).fold(0.0f32, f32::max);
        assert!(max_score >= 0.4, "Expected score ≥ 0.4, got {max_score}");
    }

    /// Scores are within [0, 1].
    #[test]
    fn test_detect_scores_bounded() {
        let d = HighlightDetector::new(0.3, 0.05)
            .with_window(15)
            .with_threshold(0.1);
        let frames: Vec<FrameEnergy> = (0..60u64).map(|i| FrameEnergy::new(i, 0.8)).collect();
        let cuts: Vec<CutPoint> = (0u64..60)
            .step_by(5)
            .map(|i| CutPoint::new(i, CutPointKind::AudioEnergy, 1.0))
            .collect();
        let result = d.detect(&frames, &cuts);
        for seg in &result {
            assert!(
                seg.score >= 0.0 && seg.score <= 1.0,
                "score out of range: {}",
                seg.score
            );
        }
    }

    /// merge_overlapping merges touching segments.
    #[test]
    fn test_merge_overlapping_merges_adjacent() {
        let segs = vec![
            HighlightSegment {
                start_frame: 0,
                end_frame: 9,
                score: 0.5,
            },
            HighlightSegment {
                start_frame: 10,
                end_frame: 20,
                score: 0.7,
            },
        ];
        let merged = HighlightDetector::merge_overlapping(segs);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].start_frame, 0);
        assert_eq!(merged[0].end_frame, 20);
        assert!((merged[0].score - 0.7).abs() < 1e-6);
    }

    /// merge_overlapping keeps non-adjacent segments separate.
    #[test]
    fn test_merge_overlapping_keeps_separated() {
        let segs = vec![
            HighlightSegment {
                start_frame: 0,
                end_frame: 8,
                score: 0.5,
            },
            HighlightSegment {
                start_frame: 20,
                end_frame: 30,
                score: 0.7,
            },
        ];
        let merged = HighlightDetector::merge_overlapping(segs);
        assert_eq!(merged.len(), 2);
    }
}
