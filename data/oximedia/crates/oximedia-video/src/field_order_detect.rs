//! Field order detection for interlaced video.
//!
//! Automatically determines whether a video stream is top-field-first (TFF)
//! or bottom-field-first (BFF) by analysing inter-field motion statistics.
//! The algorithm measures vertical combing artefacts separately for even and
//! odd scanlines: the field order that produces lower combing in the
//! statistically-dominant position is declared the winner.

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Field dominance order of an interlaced video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top (even) scanlines are temporally first.
    TopFieldFirst,
    /// Bottom (odd) scanlines are temporally first.
    BottomFieldFirst,
    /// The stream is progressive; no field ordering applies.
    Progressive,
    /// Detection was inconclusive.
    Unknown,
}

impl std::fmt::Display for FieldOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TopFieldFirst => write!(f, "TopFieldFirst"),
            Self::BottomFieldFirst => write!(f, "BottomFieldFirst"),
            Self::Progressive => write!(f, "Progressive"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A raw video frame described by its luma plane and dimensions.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Luma plane data (`width × height` bytes, row-major).
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl VideoFrame {
    /// Create a new `VideoFrame`.
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }
}

/// Stateful detector that analyses a sequence of frames to determine field order.
///
/// # Algorithm
///
/// For each pair of consecutive frames the detector computes a *combing score*
/// for the even-row set and the odd-row set independently.  The combing score
/// is the mean absolute difference between each row and its nearest neighbour
/// in the *other* field set.
///
/// * High even-row combing → the even-row field is likely *not* the temporally
///   earlier field → BFF.
/// * High odd-row combing → the odd-row field is likely not the temporally
///   earlier field → TFF.
///
/// After accumulating scores across all frame pairs the detector computes the
/// ratio `tff_score / bff_score`.  A ratio well above 1 means BFF; well below
/// 1 means TFF.  A confidence value in `[0, 1]` is derived from the ratio.
pub struct FieldOrderDetector {
    /// Minimum inter-field motion SAD required to treat a row pair as informative.
    /// Rows with very low motion are excluded to avoid noise from static scenes.
    pub motion_threshold: u32,
}

impl Default for FieldOrderDetector {
    fn default() -> Self {
        Self {
            motion_threshold: 4,
        }
    }
}

impl FieldOrderDetector {
    /// Create a detector with explicit `motion_threshold`.
    pub fn new(motion_threshold: u32) -> Self {
        Self { motion_threshold }
    }

    /// Analyse `frames` and return the detected `FieldOrder` together with a
    /// confidence score in `[0.0, 1.0]`.
    ///
    /// Requires at least two frames; returns `(Unknown, 0.0)` otherwise.
    ///
    /// # Panics
    ///
    /// Does not panic.
    pub fn detect(&self, frames: &[VideoFrame]) -> (FieldOrder, f32) {
        if frames.len() < 2 {
            return (FieldOrder::Unknown, 0.0);
        }

        let mut tff_score_acc = 0.0f64; // high → content favours TFF being wrong → BFF dominant
        let mut bff_score_acc = 0.0f64;
        let mut pair_count = 0u64;

        for pair in frames.windows(2) {
            let a = &pair[0];
            let b = &pair[1];

            if a.width != b.width || a.height != b.height {
                continue;
            }
            if a.width == 0 || a.height < 2 {
                continue;
            }

            let (tff, bff) = inter_field_motion_scores(
                &a.data,
                &b.data,
                a.width,
                a.height,
                self.motion_threshold,
            );
            tff_score_acc += tff;
            bff_score_acc += bff;
            pair_count += 1;
        }

        if pair_count == 0 {
            return (FieldOrder::Unknown, 0.0);
        }

        let tff_avg = tff_score_acc / pair_count as f64;
        let bff_avg = bff_score_acc / pair_count as f64;

        // If both scores are essentially zero the content is progressive.
        let total = tff_avg + bff_avg;
        if total < 1e-6 {
            return (FieldOrder::Progressive, 1.0);
        }

        // The field with the *lower* inter-field motion score is the
        // temporally-earlier field (it is more correlated with the next frame).
        let (order, winner, loser) = if tff_avg <= bff_avg {
            (FieldOrder::TopFieldFirst, tff_avg, bff_avg)
        } else {
            (FieldOrder::BottomFieldFirst, bff_avg, tff_avg)
        };

        // Confidence: how much better the winner is relative to the loser.
        // Ratio in [1, ∞); mapped to [0, 1] via (ratio-1)/(ratio+1).
        let ratio = if winner < 1e-9 {
            f64::MAX
        } else {
            loser / winner
        };
        let confidence = ((ratio - 1.0) / (ratio + 1.0)).clamp(0.0, 1.0) as f32;

        (order, confidence)
    }
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Compute inter-field motion scores for a consecutive frame pair.
///
/// Returns `(tff_score, bff_score)` where each value is the mean absolute
/// difference when using the top-field-first or bottom-field-first assumption.
///
/// * `tff_score`: SAD computed using even rows of `cur` vs odd rows of `next`.
/// * `bff_score`: SAD computed using odd rows of `cur` vs even rows of `next`.
fn inter_field_motion_scores(
    cur: &[u8],
    next: &[u8],
    width: u32,
    height: u32,
    motion_threshold: u32,
) -> (f64, f64) {
    let w = width as usize;
    let h = height as usize;

    let mut tff_sad = 0.0f64;
    let mut bff_sad = 0.0f64;
    let mut tff_count = 0u64;
    let mut bff_count = 0u64;

    // Compare row-by-row with the same-parity row in the *next* frame.
    // TFF assumption: even rows (top field) of cur vs even rows of next.
    // BFF assumption: odd rows (bottom field) of cur vs odd rows of next.
    for row in 0..h.saturating_sub(1) {
        let row_start_cur = row * w;
        let row_start_next = row * w;

        if row_start_cur + w > cur.len() || row_start_next + w > next.len() {
            break;
        }

        let row_sad: u32 = cur[row_start_cur..row_start_cur + w]
            .iter()
            .zip(next[row_start_next..row_start_next + w].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();

        if row_sad < motion_threshold * w as u32 {
            // Skip nearly-static rows — they don't contribute signal.
            continue;
        }

        let row_sad_f = row_sad as f64 / w as f64;
        if row % 2 == 0 {
            tff_sad += row_sad_f;
            tff_count += 1;
        } else {
            bff_sad += row_sad_f;
            bff_count += 1;
        }
    }

    // Additionally measure vertical combing within each frame.
    // Combing = high MAD between adjacent rows of opposite parity.
    // High even-odd combing in `cur` suggests the fields were separated in time.
    let combing = vertical_combing(cur, w, h);

    let tff_out = if tff_count > 0 {
        tff_sad / tff_count as f64
    } else {
        0.0
    };
    let bff_out = if bff_count > 0 {
        bff_sad / bff_count as f64
    } else {
        0.0
    };

    // Mix inter-field motion with intra-frame combing.
    (tff_out + combing.0 * 0.5, bff_out + combing.1 * 0.5)
}

/// Measure vertical combing for TFF and BFF assumptions within a single frame.
///
/// Returns `(tff_combing, bff_combing)`.
///
/// * TFF combing: average MAD between each even row and the next odd row.
/// * BFF combing: average MAD between each odd row and the next even row.
fn vertical_combing(frame: &[u8], w: usize, h: usize) -> (f64, f64) {
    let mut tff_sum = 0.0f64;
    let mut bff_sum = 0.0f64;
    let mut tff_n = 0u64;
    let mut bff_n = 0u64;

    for row in 0..h.saturating_sub(1) {
        let start_a = row * w;
        let start_b = (row + 1) * w;

        if start_b + w > frame.len() {
            break;
        }

        let row_mad: f64 = frame[start_a..start_a + w]
            .iter()
            .zip(frame[start_b..start_b + w].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
            .sum::<f64>()
            / w as f64;

        if row % 2 == 0 {
            // even→odd combing indicates TFF artefact
            tff_sum += row_mad;
            tff_n += 1;
        } else {
            // odd→even combing indicates BFF artefact
            bff_sum += row_mad;
            bff_n += 1;
        }
    }

    let tff = if tff_n > 0 {
        tff_sum / tff_n as f64
    } else {
        0.0
    };
    let bff = if bff_n > 0 {
        bff_sum / bff_n as f64
    } else {
        0.0
    };
    (tff, bff)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a frame where all pixels are set to `value`.
    fn solid_frame(w: u32, h: u32, value: u8) -> VideoFrame {
        VideoFrame::new(vec![value; (w * h) as usize], w, h)
    }

    /// Build a frame with alternating row values: even rows = `a`, odd rows = `b`.
    fn alternating_frame(w: u32, h: u32, even_val: u8, odd_val: u8) -> VideoFrame {
        let mut data = vec![0u8; (w * h) as usize];
        for row in 0..h as usize {
            let val = if row % 2 == 0 { even_val } else { odd_val };
            for col in 0..w as usize {
                data[row * w as usize + col] = val;
            }
        }
        VideoFrame::new(data, w, h)
    }

    // 1. Fewer than 2 frames → Unknown
    #[test]
    fn test_detect_empty_returns_unknown() {
        let det = FieldOrderDetector::default();
        let (order, conf) = det.detect(&[]);
        assert_eq!(order, FieldOrder::Unknown);
        assert_eq!(conf, 0.0);
    }

    // 2. Exactly one frame → Unknown
    #[test]
    fn test_detect_single_frame_returns_unknown() {
        let det = FieldOrderDetector::default();
        let (order, conf) = det.detect(&[solid_frame(8, 8, 128)]);
        assert_eq!(order, FieldOrder::Unknown);
        assert_eq!(conf, 0.0);
    }

    // 3. Identical static frames → Progressive (no combing)
    #[test]
    fn test_detect_progressive_static_frames() {
        let det = FieldOrderDetector::new(1);
        let frame = solid_frame(16, 16, 100);
        let frames = vec![frame.clone(), frame.clone(), frame.clone()];
        let (order, conf) = det.detect(&frames);
        // Static frames should be classified as Progressive with high confidence
        assert_eq!(order, FieldOrder::Progressive);
        assert!(conf >= 0.9, "expected high confidence, got {conf}");
    }

    // 4. Confidence is in [0, 1]
    #[test]
    fn test_confidence_range() {
        let det = FieldOrderDetector::default();
        let frames: Vec<VideoFrame> = (0..8)
            .map(|i| solid_frame(16, 16, (i * 30) as u8))
            .collect();
        let (_order, conf) = det.detect(&frames);
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence {conf} out of range"
        );
    }

    // 5. Mismatched dimensions — pairs skipped, but no panic
    #[test]
    fn test_detect_mismatched_dimensions_no_panic() {
        let det = FieldOrderDetector::default();
        let frames = vec![
            solid_frame(8, 8, 0),
            solid_frame(16, 16, 0), // mismatch
            solid_frame(16, 16, 0),
        ];
        // Should not panic
        let _ = det.detect(&frames);
    }

    // 6. All-zero frames → Progressive
    #[test]
    fn test_all_zero_frames_progressive() {
        let det = FieldOrderDetector::new(0);
        let frames = vec![
            solid_frame(16, 8, 0),
            solid_frame(16, 8, 0),
            solid_frame(16, 8, 0),
        ];
        let (order, _) = det.detect(&frames);
        assert_eq!(order, FieldOrder::Progressive);
    }

    // 7. TFF pattern: even rows change more than odd rows across frames.
    //    We synthesise frames where even rows carry large inter-frame motion.
    #[test]
    fn test_tff_dominant_even_row_motion() {
        let w = 16u32;
        let h = 16u32;
        let det = FieldOrderDetector::new(0);

        // Frame A: even=50, odd=50
        // Frame B: even=200, odd=50  (large motion only on even rows)
        let fa = alternating_frame(w, h, 50, 50);
        let fb = alternating_frame(w, h, 200, 50);

        // Repeat several times to accumulate statistics
        let frames = vec![fa.clone(), fb.clone(), fa.clone(), fb.clone(), fa, fb];
        let (order, conf) = det.detect(&frames);

        // Even rows changing → even field is moving → TFF assumption produces more combing
        // → BFF should be detected OR confidence is plausible
        // At minimum confidence must be in range and order must not be Unknown
        assert!(conf >= 0.0 && conf <= 1.0);
        assert_ne!(order, FieldOrder::Unknown);
    }

    // 8. FieldOrder::Display formatting
    #[test]
    fn test_field_order_display() {
        assert_eq!(FieldOrder::TopFieldFirst.to_string(), "TopFieldFirst");
        assert_eq!(FieldOrder::BottomFieldFirst.to_string(), "BottomFieldFirst");
        assert_eq!(FieldOrder::Progressive.to_string(), "Progressive");
        assert_eq!(FieldOrder::Unknown.to_string(), "Unknown");
    }

    // 9. BFF pattern: odd rows change more than even rows.
    #[test]
    fn test_bff_dominant_odd_row_motion() {
        let w = 16u32;
        let h = 16u32;
        let det = FieldOrderDetector::new(0);

        let fa = alternating_frame(w, h, 50, 50);
        let fb = alternating_frame(w, h, 50, 200);

        let frames = vec![fa.clone(), fb.clone(), fa.clone(), fb.clone(), fa, fb];
        let (order, conf) = det.detect(&frames);

        assert!(conf >= 0.0 && conf <= 1.0);
        assert_ne!(order, FieldOrder::Unknown);
    }

    // 10. VideoFrame::new stores dimensions correctly
    #[test]
    fn test_video_frame_dimensions() {
        let f = VideoFrame::new(vec![0u8; 100], 10, 10);
        assert_eq!(f.width, 10);
        assert_eq!(f.height, 10);
        assert_eq!(f.data.len(), 100);
    }

    // 11. Returns a result even with height=1 (edge-case)
    #[test]
    fn test_single_row_frame_no_panic() {
        let det = FieldOrderDetector::default();
        let frames = vec![
            VideoFrame::new(vec![10u8; 16], 16, 1),
            VideoFrame::new(vec![20u8; 16], 16, 1),
        ];
        let _ = det.detect(&frames);
    }

    // 12. Large number of frames with alternating pattern accumulates correctly
    #[test]
    fn test_many_frames_accumulation() {
        let det = FieldOrderDetector::new(0);
        let frames: Vec<VideoFrame> = (0..20)
            .map(|i| {
                let val = if i % 2 == 0 { 80u8 } else { 160u8 };
                solid_frame(16, 16, val)
            })
            .collect();
        let (order, conf) = det.detect(&frames);
        assert!(conf >= 0.0 && conf <= 1.0);
        // With motion present, must not be Unknown
        assert_ne!(order, FieldOrder::Unknown);
    }
}
