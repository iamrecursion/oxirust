//! GOP (Group of Pictures) structure analysis and planning.
//!
//! Provides GOP boundary detection, hierarchical B-frame pyramid layouts,
//! scene-change–based keyframe insertion, and GOP statistics.

#![allow(dead_code)]

/// The role a frame plays within the B-pyramid hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PyramidLevel {
    /// Top-level anchor frame (P or I); level 0 has highest quality.
    Anchor = 0,
    /// Level-1 B-frame (references two anchors).
    L1 = 1,
    /// Level-2 B-frame (references two level-1 frames).
    L2 = 2,
    /// Level-3 B-frame (leaf; references two level-2 frames).
    L3 = 3,
}

impl PyramidLevel {
    /// Returns the suggested QP delta relative to the top-level anchor.
    #[must_use]
    pub fn qp_delta(self) -> i8 {
        match self {
            Self::Anchor => 0,
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L3 => 4,
        }
    }
}

/// Describes a single frame's position within a planned GOP.
#[derive(Debug, Clone)]
pub struct GopFrame {
    /// Display-order position (0-based within the GOP).
    pub position: u32,
    /// Whether this frame is the GOP's opening I- or IDR-frame.
    pub is_keyframe: bool,
    /// Whether this frame is a B-frame.
    pub is_b_frame: bool,
    /// The B-pyramid level for this frame.
    pub pyramid_level: PyramidLevel,
}

impl GopFrame {
    /// Convenience constructor for a keyframe.
    #[must_use]
    pub fn keyframe(position: u32) -> Self {
        Self {
            position,
            is_keyframe: true,
            is_b_frame: false,
            pyramid_level: PyramidLevel::Anchor,
        }
    }

    /// Convenience constructor for a P-frame anchor.
    #[must_use]
    pub fn p_frame(position: u32) -> Self {
        Self {
            position,
            is_keyframe: false,
            is_b_frame: false,
            pyramid_level: PyramidLevel::Anchor,
        }
    }

    /// Convenience constructor for a B-frame at a given pyramid level.
    #[must_use]
    pub fn b_frame(position: u32, level: PyramidLevel) -> Self {
        Self {
            position,
            is_keyframe: false,
            is_b_frame: true,
            pyramid_level: level,
        }
    }
}

/// Aggregate statistics about a planned or observed GOP.
#[derive(Debug, Clone, Default)]
pub struct GopStatistics {
    /// Total number of frames in the GOP.
    pub total_frames: u32,
    /// Number of I-frames (including IDR).
    pub i_frame_count: u32,
    /// Number of P-frames.
    pub p_frame_count: u32,
    /// Number of B-frames.
    pub b_frame_count: u32,
    /// Average pyramid level across all frames.
    pub avg_pyramid_level: f32,
}

impl GopStatistics {
    /// Computes the B-frame ratio (0.0–1.0).
    #[must_use]
    pub fn b_ratio(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.b_frame_count as f32 / self.total_frames as f32
    }
}

/// Plans a mini-GOP with a 2-level B-pyramid for `gop_size` frames.
///
/// Returns a `Vec<GopFrame>` in display order where frame 0 is always a
/// keyframe and subsequent anchors appear at intervals of `anchor_interval`.
#[must_use]
pub fn plan_gop(gop_size: u32, anchor_interval: u32) -> Vec<GopFrame> {
    let mut frames = Vec::with_capacity(gop_size as usize);
    if gop_size == 0 {
        return frames;
    }
    for pos in 0..gop_size {
        let frame = if pos == 0 {
            GopFrame::keyframe(pos)
        } else if anchor_interval == 0 || pos % anchor_interval == 0 {
            GopFrame::p_frame(pos)
        } else {
            let offset = pos % anchor_interval;
            let half = anchor_interval / 2;
            if offset == half {
                GopFrame::b_frame(pos, PyramidLevel::L1)
            } else if offset % 2 == 0 {
                GopFrame::b_frame(pos, PyramidLevel::L2)
            } else {
                GopFrame::b_frame(pos, PyramidLevel::L3)
            }
        };
        frames.push(frame);
    }
    frames
}

/// Computes aggregate statistics over a slice of `GopFrame` descriptors.
#[must_use]
pub fn compute_statistics(frames: &[GopFrame]) -> GopStatistics {
    let total = frames.len() as u32;
    let mut i_count = 0u32;
    let mut p_count = 0u32;
    let mut b_count = 0u32;
    let mut level_sum = 0u32;

    for f in frames {
        if f.is_keyframe {
            i_count += 1;
        } else if f.is_b_frame {
            b_count += 1;
        } else {
            p_count += 1;
        }
        level_sum += f.pyramid_level as u32;
    }

    let avg_level = if total > 0 {
        level_sum as f32 / total as f32
    } else {
        0.0
    };

    GopStatistics {
        total_frames: total,
        i_frame_count: i_count,
        p_frame_count: p_count,
        b_frame_count: b_count,
        avg_pyramid_level: avg_level,
    }
}

/// Scene-change detector that decides whether a new keyframe should be
/// inserted based on a simple sum-of-absolute-differences threshold.
#[derive(Debug)]
pub struct SceneChangeDetector {
    /// SAD threshold above which a scene change is declared.
    pub threshold: u64,
    /// Minimum number of frames between forced keyframes.
    pub min_keyframe_interval: u32,
    frames_since_last_key: u32,
}

impl SceneChangeDetector {
    /// Creates a new detector.
    #[must_use]
    pub fn new(threshold: u64, min_keyframe_interval: u32) -> Self {
        Self {
            threshold,
            min_keyframe_interval,
            frames_since_last_key: 0,
        }
    }

    /// Updates the detector with the SAD value for the current frame.
    /// Returns `true` if a scene change keyframe should be inserted.
    pub fn update(&mut self, sad: u64) -> bool {
        self.frames_since_last_key += 1;
        let is_change =
            self.frames_since_last_key >= self.min_keyframe_interval && sad >= self.threshold;
        if is_change {
            self.frames_since_last_key = 0;
        }
        is_change
    }

    /// Forces a keyframe reset (call on IDR insertion).
    pub fn reset(&mut self) {
        self.frames_since_last_key = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_level_ordering() {
        assert!(PyramidLevel::Anchor < PyramidLevel::L1);
        assert!(PyramidLevel::L1 < PyramidLevel::L2);
        assert!(PyramidLevel::L2 < PyramidLevel::L3);
    }

    #[test]
    fn test_pyramid_qp_delta_increases() {
        assert!(PyramidLevel::L3.qp_delta() > PyramidLevel::L2.qp_delta());
        assert!(PyramidLevel::L2.qp_delta() > PyramidLevel::L1.qp_delta());
        assert!(PyramidLevel::L1.qp_delta() > PyramidLevel::Anchor.qp_delta());
    }

    #[test]
    fn test_gop_frame_keyframe() {
        let f = GopFrame::keyframe(0);
        assert!(f.is_keyframe);
        assert!(!f.is_b_frame);
    }

    #[test]
    fn test_gop_frame_b_frame() {
        let f = GopFrame::b_frame(3, PyramidLevel::L2);
        assert!(f.is_b_frame);
        assert_eq!(f.pyramid_level, PyramidLevel::L2);
    }

    #[test]
    fn test_plan_gop_first_frame_is_keyframe() {
        let frames = plan_gop(16, 4);
        assert!(frames[0].is_keyframe);
    }

    #[test]
    fn test_plan_gop_length() {
        let frames = plan_gop(30, 5);
        assert_eq!(frames.len(), 30);
    }

    #[test]
    fn test_plan_gop_zero_returns_empty() {
        let frames = plan_gop(0, 4);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_statistics_total_equals_i_plus_p_plus_b() {
        let frames = plan_gop(16, 4);
        let stats = compute_statistics(&frames);
        assert_eq!(stats.total_frames, 16);
        assert_eq!(
            stats.i_frame_count + stats.p_frame_count + stats.b_frame_count,
            stats.total_frames
        );
    }

    #[test]
    fn test_statistics_i_frame_count_at_least_one() {
        let frames = plan_gop(10, 4);
        let stats = compute_statistics(&frames);
        assert!(stats.i_frame_count >= 1);
    }

    #[test]
    fn test_b_ratio_range() {
        let frames = plan_gop(20, 4);
        let stats = compute_statistics(&frames);
        assert!(stats.b_ratio() >= 0.0 && stats.b_ratio() <= 1.0);
    }

    #[test]
    fn test_scene_change_not_triggered_below_interval() {
        let mut det = SceneChangeDetector::new(1000, 5);
        // Below min_keyframe_interval – should never trigger.
        for _ in 0..4 {
            assert!(!det.update(u64::MAX));
        }
    }

    #[test]
    fn test_scene_change_triggered_above_threshold() {
        let mut det = SceneChangeDetector::new(500, 2);
        det.update(0); // frame 1
        let triggered = det.update(1000); // frame 2 – above threshold and at min interval
        assert!(triggered);
    }

    #[test]
    fn test_scene_change_reset() {
        let mut det = SceneChangeDetector::new(100, 1);
        det.update(200); // triggers keyframe, resets counter
                         // After reset, next frame should NOT trigger (counter = 0, then 1 = interval).
        let triggered = det.update(200);
        assert!(triggered); // min_interval=1, so immediately eligible again
        det.reset();
        // Now forcibly reset – one more update needed.
        assert!(det.update(200));
    }

    #[test]
    fn test_statistics_empty_gop() {
        let stats = compute_statistics(&[]);
        assert_eq!(stats.total_frames, 0);
        assert!((stats.b_ratio() - 0.0).abs() < 1e-6);
    }
}
