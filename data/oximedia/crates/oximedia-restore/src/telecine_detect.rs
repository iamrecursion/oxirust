//! Telecine detection and 3:2 pulldown analysis.
//!
//! Provides cadence detection, field-order analysis, and inverse telecine helpers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Standard telecine pulldown patterns (field cadences).
///
/// For 3:2 pulldown (NTSC film-to-video), the pattern of fields per frame repeats with period 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulldownPattern {
    /// 3:2 pulldown (NTSC 24→29.97 fps).  Fields pattern: AABBC AABBC ...
    ThreeTwo,
    /// 2:2 pulldown (PAL 25fps progressive sources).
    TwoTwo,
    /// 2:3:3:2 pulldown (alternative 24→29.97).
    TwoThreeThreeTwo,
    /// Unknown / not detected.
    Unknown,
}

impl PulldownPattern {
    /// Get the repeat period in frames.
    pub fn period(&self) -> usize {
        match self {
            PulldownPattern::ThreeTwo => 5,
            PulldownPattern::TwoTwo => 2,
            PulldownPattern::TwoThreeThreeTwo => 4,
            PulldownPattern::Unknown => 0,
        }
    }

    /// Source frames per period.
    pub fn source_frames(&self) -> usize {
        match self {
            PulldownPattern::ThreeTwo => 4,
            PulldownPattern::TwoTwo => 2,
            PulldownPattern::TwoThreeThreeTwo => 4,
            PulldownPattern::Unknown => 0,
        }
    }
}

/// Field order of a video signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top field (even lines) first.
    TopFirst,
    /// Bottom field (odd lines) first.
    BottomFirst,
    /// Progressive (no fields).
    Progressive,
    /// Unknown.
    Unknown,
}

/// Inter-field difference statistics for a single frame pair.
#[derive(Debug, Clone, Copy)]
pub struct FieldDiff {
    /// Frame index.
    pub frame_idx: usize,
    /// Difference between top-top fields of consecutive frames.
    pub tt_diff: f32,
    /// Difference between bottom-bottom fields.
    pub bb_diff: f32,
    /// Difference between top-bottom (within one frame).
    pub tb_diff: f32,
}

impl FieldDiff {
    /// Detect whether this frame is likely a repeated field.
    pub fn is_repeated_field(&self, threshold: f32) -> bool {
        self.tt_diff < threshold || self.bb_diff < threshold
    }
}

/// Compute the mean absolute difference between two equal-length pixel slices.
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

/// Compute field difference statistics for a pair of consecutive frames.
///
/// Each frame is represented as a flat array of pixels in raster order.
/// `width` and `height` describe frame dimensions.
pub fn compute_field_diff(
    frame_a: &[f32],
    frame_b: &[f32],
    width: usize,
    height: usize,
    frame_idx: usize,
) -> Option<FieldDiff> {
    if width == 0 || height < 2 {
        return None;
    }
    if frame_a.len() < width * height || frame_b.len() < width * height {
        return None;
    }

    // Extract top fields (even rows: 0, 2, 4, ...)
    let top_a: Vec<f32> = (0..height)
        .step_by(2)
        .flat_map(|row| frame_a[row * width..(row + 1) * width].iter().cloned())
        .collect();
    let top_b: Vec<f32> = (0..height)
        .step_by(2)
        .flat_map(|row| frame_b[row * width..(row + 1) * width].iter().cloned())
        .collect();

    // Extract bottom fields (odd rows: 1, 3, 5, ...)
    let bot_a: Vec<f32> = (1..height)
        .step_by(2)
        .flat_map(|row| frame_a[row * width..(row + 1) * width].iter().cloned())
        .collect();
    let bot_b: Vec<f32> = (1..height)
        .step_by(2)
        .flat_map(|row| frame_b[row * width..(row + 1) * width].iter().cloned())
        .collect();

    let tt_diff = mean_abs_diff(&top_a, &top_b);
    let bb_diff = mean_abs_diff(&bot_a, &bot_b);
    let tb_diff = mean_abs_diff(&top_a, &bot_a);

    Some(FieldDiff {
        frame_idx,
        tt_diff,
        bb_diff,
        tb_diff,
    })
}

/// Telecine cadence detector.
#[derive(Debug, Clone)]
pub struct TelecineDetector {
    /// Threshold for considering a field diff as "low" (repeated).
    pub repeat_threshold: f32,
    /// Accumulated field diffs for cadence analysis.
    diffs: Vec<FieldDiff>,
    /// Maximum history length.
    max_history: usize,
}

impl TelecineDetector {
    /// Create a new detector.
    pub fn new(repeat_threshold: f32, max_history: usize) -> Self {
        Self {
            repeat_threshold,
            diffs: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Feed a computed field diff.
    pub fn feed(&mut self, diff: FieldDiff) {
        if self.diffs.len() >= self.max_history {
            self.diffs.remove(0);
        }
        self.diffs.push(diff);
    }

    /// Detect the pulldown pattern from accumulated diffs.
    pub fn detect_pattern(&self) -> PulldownPattern {
        if self.diffs.len() < 5 {
            return PulldownPattern::Unknown;
        }

        // Count how many frames show repeated fields (low diff)
        let repeated: Vec<bool> = self
            .diffs
            .iter()
            .map(|d| d.is_repeated_field(self.repeat_threshold))
            .collect();

        // Try to find 3:2 cadence: in every window of 5, exactly 2 should have low diff
        let period = 5;
        let windows = repeated.len() / period;
        if windows == 0 {
            return PulldownPattern::Unknown;
        }

        let mut three_two_score = 0u32;
        for w in 0..windows {
            let count = repeated[w * period..(w + 1) * period]
                .iter()
                .filter(|&&x| x)
                .count();
            if count == 2 {
                three_two_score += 1;
            }
        }

        if three_two_score as f32 / windows as f32 >= 0.7 {
            return PulldownPattern::ThreeTwo;
        }

        // Try 2:2 (all frames progressive)
        if repeated.iter().all(|&r| !r) {
            return PulldownPattern::TwoTwo;
        }

        PulldownPattern::Unknown
    }

    /// Detect the field order from the accumulated diffs.
    pub fn detect_field_order(&self) -> FieldOrder {
        if self.diffs.is_empty() {
            return FieldOrder::Unknown;
        }
        let avg_tb: f32 =
            self.diffs.iter().map(|d| d.tb_diff).sum::<f32>() / self.diffs.len() as f32;
        let avg_tt: f32 =
            self.diffs.iter().map(|d| d.tt_diff).sum::<f32>() / self.diffs.len() as f32;

        if avg_tb < 0.01 {
            FieldOrder::Progressive
        } else if avg_tt < avg_tb {
            FieldOrder::TopFirst
        } else {
            FieldOrder::BottomFirst
        }
    }

    /// Get accumulated diff count.
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.diffs.clear();
    }

    /// Get the repeat flags for accumulated frames.
    pub fn repeat_flags(&self) -> Vec<bool> {
        self.diffs
            .iter()
            .map(|d| d.is_repeated_field(self.repeat_threshold))
            .collect()
    }
}

/// Find the cadence phase offset within a 3:2 pulldown sequence.
///
/// Returns an offset 0–4 indicating where in the 5-frame cycle the sequence starts.
pub fn find_cadence_phase(repeat_flags: &[bool]) -> Option<usize> {
    if repeat_flags.len() < 5 {
        return None;
    }

    // In a 3:2 pattern: two adjacent frames are "repeated" (high field reuse)
    // Find the position of the first repeated frame
    for start in 0..5 {
        let window = &repeat_flags[start..];
        if window.len() >= 5 {
            let count = window[..5].iter().filter(|&&x| x).count();
            if count == 2 {
                return Some(start);
            }
        }
    }
    None
}

/// Inverse telecine frame selector.
///
/// Given a cadence phase, this selects which frames to keep to reconstruct 24fps content.
#[derive(Debug, Clone)]
pub struct InverseTelecine {
    pattern: PulldownPattern,
    phase: usize,
}

impl InverseTelecine {
    /// Create with detected pattern and phase.
    pub fn new(pattern: PulldownPattern, phase: usize) -> Self {
        Self { pattern, phase }
    }

    /// Check if the given frame index should be kept in the output.
    pub fn should_keep(&self, frame_idx: usize) -> bool {
        match self.pattern {
            PulldownPattern::ThreeTwo => {
                let pos = (frame_idx + self.phase) % 5;
                // In 3:2: keep frames at positions 0, 1, 2, 4 (skip position 3 or 4)
                pos != 3
            }
            PulldownPattern::TwoTwo => true,
            _ => true,
        }
    }

    /// Get the pattern being used.
    pub fn pattern(&self) -> PulldownPattern {
        self.pattern
    }

    /// Get the phase offset.
    pub fn phase(&self) -> usize {
        self.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulldown_pattern_period() {
        assert_eq!(PulldownPattern::ThreeTwo.period(), 5);
        assert_eq!(PulldownPattern::TwoTwo.period(), 2);
        assert_eq!(PulldownPattern::Unknown.period(), 0);
    }

    #[test]
    fn test_pulldown_source_frames() {
        assert_eq!(PulldownPattern::ThreeTwo.source_frames(), 4);
        assert_eq!(PulldownPattern::TwoTwo.source_frames(), 2);
    }

    #[test]
    fn test_mean_abs_diff_same() {
        let a = vec![0.5f32; 10];
        let b = vec![0.5f32; 10];
        assert!((mean_abs_diff(&a, &b)).abs() < 1e-7);
    }

    #[test]
    fn test_mean_abs_diff_different() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        assert!((mean_abs_diff(&a, &b) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_field_diff_too_small() {
        let frame = vec![0.5f32; 4];
        let result = compute_field_diff(&frame, &frame, 2, 1, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_compute_field_diff_basic() {
        let w = 4usize;
        let h = 4usize;
        let frame_a = vec![0.5f32; w * h];
        let frame_b = vec![0.5f32; w * h];
        let diff = compute_field_diff(&frame_a, &frame_b, w, h, 0).expect("should succeed in test");
        assert!((diff.tt_diff).abs() < 1e-6);
        assert!((diff.bb_diff).abs() < 1e-6);
    }

    #[test]
    fn test_field_diff_repeated_detection() {
        let diff = FieldDiff {
            frame_idx: 0,
            tt_diff: 0.001,
            bb_diff: 0.5,
            tb_diff: 0.5,
        };
        assert!(diff.is_repeated_field(0.01));
        assert!(!diff.is_repeated_field(0.0001));
    }

    #[test]
    fn test_telecine_detector_insufficient_data() {
        let detector = TelecineDetector::new(0.05, 100);
        assert_eq!(detector.detect_pattern(), PulldownPattern::Unknown);
    }

    #[test]
    fn test_telecine_detector_progressive() {
        let mut detector = TelecineDetector::new(0.05, 100);
        for i in 0..10 {
            detector.feed(FieldDiff {
                frame_idx: i,
                tt_diff: 0.5,
                bb_diff: 0.5,
                tb_diff: 0.001, // low tb -> progressive
            });
        }
        assert_eq!(detector.detect_field_order(), FieldOrder::Progressive);
    }

    #[test]
    fn test_telecine_detector_reset() {
        let mut detector = TelecineDetector::new(0.05, 100);
        detector.feed(FieldDiff {
            frame_idx: 0,
            tt_diff: 0.1,
            bb_diff: 0.1,
            tb_diff: 0.1,
        });
        detector.reset();
        assert_eq!(detector.diff_count(), 0);
    }

    #[test]
    fn test_telecine_detector_repeat_flags() {
        let mut detector = TelecineDetector::new(0.05, 100);
        detector.feed(FieldDiff {
            frame_idx: 0,
            tt_diff: 0.001,
            bb_diff: 0.5,
            tb_diff: 0.5,
        });
        detector.feed(FieldDiff {
            frame_idx: 1,
            tt_diff: 0.5,
            bb_diff: 0.5,
            tb_diff: 0.5,
        });
        let flags = detector.repeat_flags();
        assert_eq!(flags[0], true);
        assert_eq!(flags[1], false);
    }

    #[test]
    fn test_find_cadence_phase_insufficient() {
        let flags = vec![true, false];
        assert!(find_cadence_phase(&flags).is_none());
    }

    #[test]
    fn test_find_cadence_phase_detects() {
        // 3:2 cadence: positions 2 and 3 are repeated
        let flags = vec![false, false, true, true, false];
        let phase = find_cadence_phase(&flags);
        assert!(phase.is_some());
    }

    #[test]
    fn test_inverse_telecine_two_two_keeps_all() {
        let itc = InverseTelecine::new(PulldownPattern::TwoTwo, 0);
        for i in 0..20 {
            assert!(itc.should_keep(i));
        }
    }

    #[test]
    fn test_inverse_telecine_three_two_skips_frame_3() {
        let itc = InverseTelecine::new(PulldownPattern::ThreeTwo, 0);
        // Position 3 should be skipped
        assert!(!itc.should_keep(3));
        // Position 0,1,2,4 should be kept
        assert!(itc.should_keep(0));
        assert!(itc.should_keep(1));
        assert!(itc.should_keep(2));
        assert!(itc.should_keep(4));
    }

    #[test]
    fn test_inverse_telecine_accessors() {
        let itc = InverseTelecine::new(PulldownPattern::ThreeTwo, 2);
        assert_eq!(itc.pattern(), PulldownPattern::ThreeTwo);
        assert_eq!(itc.phase(), 2);
    }
}
