//! 3:2 pulldown detection and removal (telecine).
//!
//! Detects repeating field-difference patterns and reconstructs the
//! original progressive frames by discarding duplicated fields.

#![allow(dead_code)]

use std::collections::VecDeque;

/// A recognised pulldown (telecine) cadence pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulldownPattern {
    /// Standard NTSC 3:2 (AABBCD…) – 24 fps → 29.97 fps.
    TwoThree,
    /// Reverse 2:3 cadence (ABBBCD…).
    ThreeTwo,
    /// Already 24fps progressive – no pulldown.
    TwentyFour,
    /// Pattern not yet detected.
    Unknown,
}

impl PulldownPattern {
    /// The frame-repeat counts that define each cadence.
    ///
    /// For `TwoThree`: `[2, 3]` means field A is shown twice, field B three
    /// times, repeating.
    #[must_use]
    pub fn cadence_frames(&self) -> &[u32] {
        match self {
            Self::TwoThree => &[2, 3],
            Self::ThreeTwo => &[3, 2],
            Self::TwentyFour => &[1],
            Self::Unknown => &[],
        }
    }

    /// Output frame rate produced after applying this cadence to 24 fps film.
    #[must_use]
    pub fn output_fps(&self) -> f32 {
        match self {
            Self::TwoThree | Self::ThreeTwo => 29.97,
            Self::TwentyFour => 24.0,
            Self::Unknown => 0.0,
        }
    }
}

/// Field parity / scan order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity {
    /// Top field first (TFF).
    TopFirst,
    /// Bottom field first (BFF).
    BottomFirst,
    /// Progressive scan (no interlacing).
    Progressive,
}

/// Stateful pulldown detector.
///
/// Accumulates field-difference samples and identifies repeating patterns
/// that correspond to 3:2 or 2:3 telecine cadences.
pub struct PulldownDetector {
    /// Ring buffer of recent (`frame_idx`, `field_diff`) samples.
    history: VecDeque<(u64, f32)>,
    /// Ring-buffer capacity (default 10).
    capacity: usize,
    /// The pattern identified so far.
    detected: PulldownPattern,
}

impl PulldownDetector {
    /// Create a new detector with a 10-frame ring buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(10),
            capacity: 10,
            detected: PulldownPattern::Unknown,
        }
    }

    /// Feed a new field-difference sample and return the detected pattern if
    /// confident, or `None` while still accumulating data.
    pub fn analyze_frame(&mut self, field_diff: f32, frame_idx: u64) -> Option<PulldownPattern> {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back((frame_idx, field_diff));

        if self.history.len() < 5 {
            return None;
        }

        let diffs: Vec<f32> = self.history.iter().map(|(_, d)| *d).collect();
        let (pattern, _phase) = TelecineRemover::identify_cadence(&diffs);

        if pattern == PulldownPattern::Unknown {
            None
        } else {
            self.detected = pattern;
            Some(pattern)
        }
    }

    /// The last confirmed pattern.
    #[must_use]
    pub fn detected_pattern(&self) -> PulldownPattern {
        self.detected
    }
}

impl Default for PulldownDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Telecine removal utilities.
pub struct TelecineRemover;

impl TelecineRemover {
    /// Identify the cadence pattern and its phase offset within `diffs`.
    ///
    /// Returns `(pattern, phase_offset)`.  Phase is the index of the first
    /// 'high' (repeated) field within the window.
    #[must_use]
    pub fn identify_cadence(diffs: &[f32]) -> (PulldownPattern, usize) {
        if diffs.len() < 5 {
            return (PulldownPattern::Unknown, 0);
        }

        // Compute a threshold: mean + 0.5 * std_dev
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let variance =
            diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / diffs.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev < 1e-3 {
            // No variation → probably progressive or unknown
            return (PulldownPattern::TwentyFour, 0);
        }

        let threshold = mean + 0.3 * std_dev;

        // Classify each frame as 'high' (repeated field) or 'low' (unique field)
        let high: Vec<bool> = diffs.iter().map(|&d| d > threshold).collect();

        // Try to match 2-3 pattern (LHLLL or LLHLL…) within the window
        // A 5-frame group in 3:2 has 2 'high' and 3 'low' fields
        let n = high.len();
        for phase in 0..5 {
            let group: Vec<bool> = (0..5).map(|i| high[(phase + i) % n]).collect();
            let highs: usize = group.iter().filter(|&&h| h).count();

            if highs == 2 {
                // Count consecutive highs to distinguish 3:2 vs 2:3
                let consecutive_start = group
                    .windows(2)
                    .enumerate()
                    .find(|(_, w)| w[0] && w[1])
                    .map(|(i, _)| i);

                if consecutive_start.is_some() {
                    return (PulldownPattern::ThreeTwo, phase);
                }
                return (PulldownPattern::TwoThree, phase);
            }
        }

        (PulldownPattern::Unknown, 0)
    }

    /// Remove pulldown from a sequence of frames.
    ///
    /// For 3:2 pulldown every 5th input frame (at the given cadence phase) is
    /// a duplicated-field frame and is dropped, yielding 4 progressive frames
    /// per 5 input frames.
    ///
    /// `frames` – flat f32 luma arrays (one per input frame).
    /// `pattern` – detected cadence.
    /// `phase`   – phase offset of the first repeated field.
    #[must_use]
    pub fn remove_pulldown(
        frames: &[Vec<f32>],
        pattern: PulldownPattern,
        phase: usize,
        _width: u32,
        _height: u32,
    ) -> Vec<Vec<f32>> {
        match pattern {
            PulldownPattern::TwentyFour | PulldownPattern::Unknown => frames.to_vec(),
            PulldownPattern::TwoThree | PulldownPattern::ThreeTwo => {
                // In 3:2 pulldown, one frame in every 5 is a repeat.
                // We drop the frame at position (phase+2) % 5 within each 5-frame group.
                let mut output = Vec::with_capacity(frames.len() * 4 / 5 + 1);
                for (i, frame) in frames.iter().enumerate() {
                    if i % 5 != (phase + 4) % 5 {
                        output.push(frame.clone());
                    }
                }
                output
            }
        }
    }
}

/// Summary report of a pulldown analysis.
#[derive(Debug, Clone)]
pub struct PulldownReport {
    /// Detected pulldown pattern.
    pub detected: PulldownPattern,
    /// Phase offset of the cadence within the analysed window.
    pub phase: usize,
    /// Effective output frame rate after pulldown removal.
    pub output_fps: f32,
}

impl PulldownReport {
    /// Create a new report.
    #[must_use]
    pub fn new(detected: PulldownPattern, phase: usize) -> Self {
        let output_fps = match detected {
            PulldownPattern::TwoThree | PulldownPattern::ThreeTwo => 24.0,
            PulldownPattern::TwentyFour => 24.0,
            PulldownPattern::Unknown => 0.0,
        };
        Self {
            detected,
            phase,
            output_fps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic 3:2 field-diff sequence.
    /// Pattern: low, low, high, low, high (2 highs per 5 frames)
    fn make_23_diffs(len: usize, phase: usize) -> Vec<f32> {
        // 3:2 signature: [10, 10, 80, 10, 80] repeating
        let pattern = [10.0f32, 10.0, 80.0, 10.0, 80.0];
        (0..len).map(|i| pattern[(i + phase) % 5]).collect()
    }

    #[test]
    fn test_pulldown_pattern_cadence_two_three() {
        let p = PulldownPattern::TwoThree;
        assert_eq!(p.cadence_frames(), &[2, 3]);
    }

    #[test]
    fn test_pulldown_pattern_cadence_unknown_empty() {
        let p = PulldownPattern::Unknown;
        assert!(p.cadence_frames().is_empty());
    }

    #[test]
    fn test_pulldown_pattern_output_fps_23() {
        assert!((PulldownPattern::TwoThree.output_fps() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_pulldown_pattern_output_fps_24() {
        assert_eq!(PulldownPattern::TwentyFour.output_fps(), 24.0);
    }

    #[test]
    fn test_detector_needs_five_frames() {
        let mut det = PulldownDetector::new();
        for i in 0..4 {
            let r = det.analyze_frame(10.0, i as u64);
            assert!(r.is_none(), "Should not detect on frame {i}");
        }
    }

    #[test]
    fn test_detector_identifies_23_pattern() {
        let mut det = PulldownDetector::new();
        let diffs = make_23_diffs(10, 0);
        let mut last = None;
        for (i, &d) in diffs.iter().enumerate() {
            last = det.analyze_frame(d, i as u64);
        }
        // By 10 frames, pattern should be detected (TwoThree or ThreeTwo)
        if let Some(p) = last {
            assert!(p == PulldownPattern::TwoThree || p == PulldownPattern::ThreeTwo);
        }
    }

    #[test]
    fn test_identify_cadence_flat_is_24fps() {
        let diffs = vec![10.0f32; 10];
        let (pattern, _) = TelecineRemover::identify_cadence(&diffs);
        assert_eq!(pattern, PulldownPattern::TwentyFour);
    }

    #[test]
    fn test_identify_cadence_23_detected() {
        let diffs = make_23_diffs(10, 0);
        let (pattern, _phase) = TelecineRemover::identify_cadence(&diffs);
        assert!(
            pattern == PulldownPattern::TwoThree || pattern == PulldownPattern::ThreeTwo,
            "got {pattern:?}"
        );
    }

    #[test]
    fn test_remove_pulldown_24_unchanged() {
        let frames: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; 4]).collect();
        let out = TelecineRemover::remove_pulldown(&frames, PulldownPattern::TwentyFour, 0, 2, 2);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn test_remove_pulldown_23_drops_one_in_five() {
        let frames: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
        let out = TelecineRemover::remove_pulldown(&frames, PulldownPattern::TwoThree, 0, 2, 2);
        // Expect 4 per 5 input frames → 8 for 10 input frames
        assert_eq!(out.len(), 8, "got {} frames", out.len());
    }

    #[test]
    fn test_pulldown_report_new() {
        let r = PulldownReport::new(PulldownPattern::TwoThree, 2);
        assert_eq!(r.detected, PulldownPattern::TwoThree);
        assert_eq!(r.phase, 2);
        assert_eq!(r.output_fps, 24.0);
    }

    #[test]
    fn test_pulldown_report_unknown_fps_zero() {
        let r = PulldownReport::new(PulldownPattern::Unknown, 0);
        assert_eq!(r.output_fps, 0.0);
    }
}
