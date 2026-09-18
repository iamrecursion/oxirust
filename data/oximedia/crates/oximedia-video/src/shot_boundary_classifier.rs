//! Shot boundary classification for video streams.
//!
//! Classifies transitions between shots as hard cuts, dissolves, wipes, or fades
//! by analysing a sequence of per-frame pixel difference summaries.
//!
//! # Algorithm overview
//!
//! * **Hard cut**: a single frame where `diff` is large (above `hard_cut_threshold`).
//! * **Dissolve**: `diff` values gradually rise then fall over several frames.
//! * **Fade to/from black/white**: `diff` values rise *or* fall monotonically while
//!   the absolute brightness drifts towards 0 or 255.
//! * **Wipe**: spatial analysis — one half of the frame changes while the other
//!   remains static, with a travelling edge.

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Per-frame difference summary fed to the classifier.
#[derive(Debug, Clone)]
pub struct PixelDiff {
    /// Mean absolute per-pixel difference from the previous frame (0–255 range).
    pub mean_diff: f32,
    /// Mean absolute difference for the left half of the frame.
    pub left_diff: f32,
    /// Mean absolute difference for the right half of the frame.
    pub right_diff: f32,
    /// Mean absolute difference for the top half of the frame.
    pub top_diff: f32,
    /// Mean absolute difference for the bottom half of the frame.
    pub bottom_diff: f32,
    /// Mean luma (brightness) of the frame, in [0, 255].
    pub mean_luma: f32,
}

impl PixelDiff {
    /// Create a `PixelDiff` with all spatial halves set to `mean_diff`.
    pub fn uniform(mean_diff: f32, mean_luma: f32) -> Self {
        Self {
            mean_diff,
            left_diff: mean_diff,
            right_diff: mean_diff,
            top_diff: mean_diff,
            bottom_diff: mean_diff,
            mean_luma,
        }
    }
}

/// Wipe travel direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDir {
    /// The wipe edge moves from left to right.
    Left,
    /// The wipe edge moves from right to left.
    Right,
    /// The wipe edge moves from top to bottom.
    Up,
    /// The wipe edge moves from bottom to top.
    Down,
}

/// Fade endpoint description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeType {
    /// Scene fades out to black.
    ToBlack,
    /// Scene fades in from black.
    FromBlack,
    /// Scene fades out to white (overexposure).
    ToWhite,
    /// Scene fades in from white.
    FromWhite,
}

/// The classified type of shot boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum ShotBoundaryType {
    /// Instantaneous cut — one frame with a large `mean_diff`.
    HardCut,
    /// Gradual cross-dissolve spanning `duration_frames` frames.
    Dissolve {
        /// Number of frames over which the dissolve occurs.
        duration_frames: u32,
    },
    /// Wipe transition with a travelling edge in `direction`.
    Wipe {
        /// Direction the wipe edge is travelling.
        direction: WipeDir,
    },
    /// Fade to or from a neutral colour.
    Fade {
        /// Which kind of fade.
        from_to: FadeType,
    },
    /// No significant transition detected.
    None,
}

/// Configuration for the shot boundary classifier.
pub struct ShotBoundaryClassifier {
    /// Mean diff threshold above which a single frame is considered a hard cut.
    pub hard_cut_threshold: f32,
    /// Mean diff threshold for the start of a soft transition.
    pub soft_transition_threshold: f32,
    /// Luma value below which a frame is considered "near black".
    pub black_luma_threshold: f32,
    /// Luma value above which a frame is considered "near white".
    pub white_luma_threshold: f32,
    /// Maximum fraction by which left/right (or top/bottom) diffs may differ to
    /// still be classified as a wipe rather than a dissolve.
    pub wipe_asymmetry_min: f32,
}

impl Default for ShotBoundaryClassifier {
    fn default() -> Self {
        Self {
            hard_cut_threshold: 30.0,
            soft_transition_threshold: 8.0,
            black_luma_threshold: 20.0,
            white_luma_threshold: 235.0,
            wipe_asymmetry_min: 2.0,
        }
    }
}

impl ShotBoundaryClassifier {
    /// Create a classifier with explicit parameters.
    pub fn new(
        hard_cut_threshold: f32,
        soft_transition_threshold: f32,
        black_luma_threshold: f32,
        white_luma_threshold: f32,
        wipe_asymmetry_min: f32,
    ) -> Self {
        Self {
            hard_cut_threshold,
            soft_transition_threshold,
            black_luma_threshold,
            white_luma_threshold,
            wipe_asymmetry_min,
        }
    }

    /// Classify the shot boundary represented by `frames`.
    ///
    /// * Pass a single-element slice for hard-cut detection.
    /// * Pass multiple consecutive frames for dissolve / fade / wipe detection.
    ///
    /// Returns [`ShotBoundaryType::None`] when no significant boundary is found.
    pub fn classify(&self, frames: &[PixelDiff]) -> ShotBoundaryType {
        if frames.is_empty() {
            return ShotBoundaryType::None;
        }

        // --- Hard cut: single frame with very large diff ---
        if frames.len() == 1 {
            if frames[0].mean_diff >= self.hard_cut_threshold {
                return ShotBoundaryType::HardCut;
            }
            return ShotBoundaryType::None;
        }

        // Check if *any* frame qualifies as a hard cut first.
        let max_diff = frames
            .iter()
            .map(|f| f.mean_diff)
            .fold(f32::NEG_INFINITY, f32::max);

        if max_diff >= self.hard_cut_threshold {
            return ShotBoundaryType::HardCut;
        }

        // --- Soft transitions: diff is above the soft threshold for ≥2 frames ---
        let active: Vec<&PixelDiff> = frames
            .iter()
            .filter(|f| f.mean_diff >= self.soft_transition_threshold)
            .collect();

        if active.is_empty() {
            return ShotBoundaryType::None;
        }

        // --- Wipe: spatial asymmetry — one half has significantly higher diff ---
        if let Some(dir) = self.detect_wipe(frames) {
            return ShotBoundaryType::Wipe { direction: dir };
        }

        // --- Fade: monotonic diff + brightness change ---
        if let Some(fade) = self.detect_fade(frames) {
            return ShotBoundaryType::Fade { from_to: fade };
        }

        // --- Dissolve: gradual symmetric increase and decrease ---
        let duration = active.len() as u32;
        ShotBoundaryType::Dissolve {
            duration_frames: duration,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Detect wipe direction from spatial half-differences.
    fn detect_wipe(&self, frames: &[PixelDiff]) -> Option<WipeDir> {
        // For a left→right wipe, left_diff is consistently higher than right_diff.
        // For a right→left wipe, right_diff > left_diff.
        // Similarly for top/bottom.

        let mut left_wins = 0i32;
        let mut right_wins = 0i32;
        let mut top_wins = 0i32;
        let mut bottom_wins = 0i32;

        for f in frames {
            let lr_ratio = if f.right_diff > 1e-6 {
                f.left_diff / f.right_diff
            } else {
                1.0
            };
            let rl_ratio = if f.left_diff > 1e-6 {
                f.right_diff / f.left_diff
            } else {
                1.0
            };
            let tb_ratio = if f.bottom_diff > 1e-6 {
                f.top_diff / f.bottom_diff
            } else {
                1.0
            };
            let bt_ratio = if f.top_diff > 1e-6 {
                f.bottom_diff / f.top_diff
            } else {
                1.0
            };

            if lr_ratio >= self.wipe_asymmetry_min {
                left_wins += 1;
            }
            if rl_ratio >= self.wipe_asymmetry_min {
                right_wins += 1;
            }
            if tb_ratio >= self.wipe_asymmetry_min {
                top_wins += 1;
            }
            if bt_ratio >= self.wipe_asymmetry_min {
                bottom_wins += 1;
            }
        }

        let thresh = (frames.len() as i32 + 1) / 2;
        if left_wins >= thresh {
            return Some(WipeDir::Left);
        }
        if right_wins >= thresh {
            return Some(WipeDir::Right);
        }
        if top_wins >= thresh {
            return Some(WipeDir::Up);
        }
        if bottom_wins >= thresh {
            return Some(WipeDir::Down);
        }
        None
    }

    /// Detect a fade by checking for monotonic diff + luma trend.
    fn detect_fade(&self, frames: &[PixelDiff]) -> Option<FadeType> {
        if frames.len() < 2 {
            return None;
        }

        let first_luma = frames[0].mean_luma;
        let last_luma = frames[frames.len() - 1].mean_luma;

        let diffs: Vec<f32> = frames.iter().map(|f| f.mean_diff).collect();
        let lumas: Vec<f32> = frames.iter().map(|f| f.mean_luma).collect();

        // Check monotonic decrease in luma (fade to dark) or increase (fade to bright).
        let luma_decreasing = is_mostly_monotonic(&lumas, false);
        let luma_increasing = is_mostly_monotonic(&lumas, true);
        let diff_unimodal = is_unimodal_or_monotonic(&diffs);

        if !diff_unimodal {
            return None;
        }

        if luma_decreasing && last_luma <= self.black_luma_threshold {
            return Some(FadeType::ToBlack);
        }
        if luma_increasing && first_luma <= self.black_luma_threshold {
            return Some(FadeType::FromBlack);
        }
        if luma_increasing && last_luma >= self.white_luma_threshold {
            return Some(FadeType::ToWhite);
        }
        if luma_decreasing && first_luma >= self.white_luma_threshold {
            return Some(FadeType::FromWhite);
        }

        None
    }
}

// -----------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------

/// Return `true` if `values` is mostly monotonic (≥70 % of consecutive pairs
/// satisfy the direction).
fn is_mostly_monotonic(values: &[f32], increasing: bool) -> bool {
    if values.len() < 2 {
        // A single-element (or empty) sequence is trivially monotonic.
        return true;
    }
    let conforming = values
        .windows(2)
        .filter(|w| {
            if increasing {
                w[1] >= w[0]
            } else {
                w[1] <= w[0]
            }
        })
        .count();
    conforming * 10 >= values.len().saturating_sub(1) * 7
}

/// Return `true` if `values` is unimodal (single peak) or monotonic.
fn is_unimodal_or_monotonic(values: &[f32]) -> bool {
    if values.len() < 2 {
        return true;
    }
    // Find the peak
    let peak_idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Values before peak should be non-decreasing, values after peak non-increasing.
    let before_ok = is_mostly_monotonic(&values[..=peak_idx], true);
    let after_ok = is_mostly_monotonic(&values[peak_idx..], false);
    before_ok && after_ok
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform(diff: f32, luma: f32) -> PixelDiff {
        PixelDiff::uniform(diff, luma)
    }

    // 1. Single high-diff frame → HardCut
    #[test]
    fn test_hard_cut_single_frame() {
        let cls = ShotBoundaryClassifier::default();
        let frames = vec![make_uniform(50.0, 128.0)];
        assert_eq!(cls.classify(&frames), ShotBoundaryType::HardCut);
    }

    // 2. Single low-diff frame → None
    #[test]
    fn test_no_transition_single_frame() {
        let cls = ShotBoundaryClassifier::default();
        let frames = vec![make_uniform(2.0, 128.0)];
        assert_eq!(cls.classify(&frames), ShotBoundaryType::None);
    }

    // 3. Multiple frames, one very high → HardCut
    #[test]
    fn test_hard_cut_in_sequence() {
        let cls = ShotBoundaryClassifier::default();
        let frames = vec![
            make_uniform(5.0, 128.0),
            make_uniform(60.0, 128.0), // hard cut
            make_uniform(5.0, 128.0),
        ];
        assert_eq!(cls.classify(&frames), ShotBoundaryType::HardCut);
    }

    // 4. Gradually rising and falling diff → Dissolve
    #[test]
    fn test_dissolve_gradual_diff() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = [10.0, 14.0, 18.0, 15.0, 12.0, 9.0]
            .iter()
            .map(|&d| make_uniform(d, 128.0))
            .collect();
        let result = cls.classify(&frames);
        assert!(
            matches!(result, ShotBoundaryType::Dissolve { .. }),
            "expected Dissolve, got {result:?}"
        );
    }

    // 5. Dissolve duration_frames > 0
    #[test]
    fn test_dissolve_duration_nonzero() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = [10.0, 14.0, 18.0, 14.0, 10.0]
            .iter()
            .map(|&d| make_uniform(d, 128.0))
            .collect();
        if let ShotBoundaryType::Dissolve { duration_frames } = cls.classify(&frames) {
            assert!(duration_frames > 0);
        }
    }

    // 6. Fade to black: luma decreasing, last frame dark
    #[test]
    fn test_fade_to_black() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..8u32)
            .map(|i| {
                let luma = 128.0 - (i as f32 * 16.0);
                let diff = 10.0 + i as f32 * 2.0;
                make_uniform(diff, luma.max(0.0))
            })
            .collect();
        let result = cls.classify(&frames);
        assert_eq!(
            result,
            ShotBoundaryType::Fade {
                from_to: FadeType::ToBlack
            },
            "expected FadeToBlack, got {result:?}"
        );
    }

    // 7. Fade from black: luma increasing, first frame dark
    #[test]
    fn test_fade_from_black() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..8u32)
            .map(|i| {
                let luma = i as f32 * 16.0;
                let diff = 8.0 + i as f32;
                make_uniform(diff, luma)
            })
            .collect();
        let result = cls.classify(&frames);
        assert_eq!(
            result,
            ShotBoundaryType::Fade {
                from_to: FadeType::FromBlack
            },
            "expected FadeFromBlack, got {result:?}"
        );
    }

    // 8. Wipe left: left half has consistently higher diff
    #[test]
    fn test_wipe_left() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..6)
            .map(|_| PixelDiff {
                mean_diff: 20.0,
                left_diff: 40.0, // left is active
                right_diff: 5.0, // right is static
                top_diff: 20.0,
                bottom_diff: 20.0,
                mean_luma: 128.0,
            })
            .collect();
        assert_eq!(
            cls.classify(&frames),
            ShotBoundaryType::Wipe {
                direction: WipeDir::Left
            }
        );
    }

    // 9. Wipe right: right half has consistently higher diff
    #[test]
    fn test_wipe_right() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..6)
            .map(|_| PixelDiff {
                mean_diff: 20.0,
                left_diff: 5.0,
                right_diff: 40.0,
                top_diff: 20.0,
                bottom_diff: 20.0,
                mean_luma: 128.0,
            })
            .collect();
        assert_eq!(
            cls.classify(&frames),
            ShotBoundaryType::Wipe {
                direction: WipeDir::Right
            }
        );
    }

    // 10. Empty frames → None
    #[test]
    fn test_empty_frames_returns_none() {
        let cls = ShotBoundaryClassifier::default();
        assert_eq!(cls.classify(&[]), ShotBoundaryType::None);
    }

    // 11. All low-diff frames → None
    #[test]
    fn test_all_low_diff_none() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..5).map(|_| make_uniform(1.0, 128.0)).collect();
        assert_eq!(cls.classify(&frames), ShotBoundaryType::None);
    }

    // 12. Fade to white: luma increasing, last frame bright
    #[test]
    fn test_fade_to_white() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..8u32)
            .map(|i| {
                let luma = 128.0 + (i as f32 * 16.0);
                let diff = 8.0 + i as f32;
                make_uniform(diff, luma.min(255.0))
            })
            .collect();
        let result = cls.classify(&frames);
        assert_eq!(
            result,
            ShotBoundaryType::Fade {
                from_to: FadeType::ToWhite
            },
            "expected FadeToWhite, got {result:?}"
        );
    }

    // 13. Fade from white: luma decreasing, first frame bright
    #[test]
    fn test_fade_from_white() {
        let cls = ShotBoundaryClassifier::default();
        let frames: Vec<PixelDiff> = (0..8u32)
            .map(|i| {
                let luma = 255.0 - (i as f32 * 28.0);
                let diff = 8.0 + i as f32;
                make_uniform(diff, luma.max(0.0))
            })
            .collect();
        let result = cls.classify(&frames);
        assert_eq!(
            result,
            ShotBoundaryType::Fade {
                from_to: FadeType::FromWhite
            },
            "expected FadeFromWhite, got {result:?}"
        );
    }

    // 14. PixelDiff::uniform fields
    #[test]
    fn test_pixel_diff_uniform() {
        let d = PixelDiff::uniform(15.0, 200.0);
        assert_eq!(d.mean_diff, 15.0);
        assert_eq!(d.left_diff, 15.0);
        assert_eq!(d.right_diff, 15.0);
        assert_eq!(d.top_diff, 15.0);
        assert_eq!(d.bottom_diff, 15.0);
        assert_eq!(d.mean_luma, 200.0);
    }
}
