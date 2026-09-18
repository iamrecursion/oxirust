//! Action detection for identifying dynamic activity segments in video.

#![allow(dead_code)]

/// Category of action detected in a video segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionType {
    /// High-speed motion or rapid changes.
    HighMotion,
    /// Slow or minimal movement.
    LowMotion,
    /// Camera pan detected.
    CameraPan,
    /// Camera tilt detected.
    CameraTilt,
    /// Object enters or exits frame.
    ObjectEntry,
    /// Stable, nearly static scene.
    Static,
}

impl ActionType {
    /// Return a human-readable label for the action type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::HighMotion => "high_motion",
            Self::LowMotion => "low_motion",
            Self::CameraPan => "camera_pan",
            Self::CameraTilt => "camera_tilt",
            Self::ObjectEntry => "object_entry",
            Self::Static => "static",
        }
    }
}

/// A contiguous video segment identified as a particular action.
#[derive(Debug, Clone)]
pub struct ActionSegment {
    /// Frame index where the segment starts (inclusive).
    pub start_frame: usize,
    /// Frame index where the segment ends (exclusive).
    pub end_frame: usize,
    /// Frames per second (used to calculate wall-clock duration).
    pub fps: f32,
    /// Action category for this segment.
    pub action_type: ActionType,
    /// Mean motion magnitude over the segment (pixels/frame).
    pub mean_motion: f32,
}

impl ActionSegment {
    /// Create a new [`ActionSegment`].
    #[must_use]
    pub fn new(
        start_frame: usize,
        end_frame: usize,
        fps: f32,
        action_type: ActionType,
        mean_motion: f32,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            fps: fps.max(0.001),
            action_type,
            mean_motion,
        }
    }

    /// Return the duration of the segment in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_ms(&self) -> f64 {
        let frames = self.end_frame.saturating_sub(self.start_frame);
        frames as f64 / f64::from(self.fps) * 1000.0
    }

    /// Return the number of frames in the segment.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// Detects action segments from per-frame motion magnitude values.
pub struct ActionDetector {
    /// Motion magnitude per frame (pixels/frame).
    frame_motions: Vec<f32>,
    /// Frames per second of the input stream.
    fps: f32,
    /// Motion threshold separating high from low motion (pixels/frame).
    motion_threshold: f32,
    /// Minimum number of frames to form a valid segment.
    min_segment_frames: usize,
}

impl ActionDetector {
    /// Create a new [`ActionDetector`].
    #[must_use]
    pub fn new(fps: f32, motion_threshold: f32, min_segment_frames: usize) -> Self {
        Self {
            frame_motions: Vec::new(),
            fps: fps.max(0.001),
            motion_threshold,
            min_segment_frames: min_segment_frames.max(1),
        }
    }

    /// Register the motion magnitude for one frame.
    pub fn add_frame(&mut self, motion_magnitude: f32) {
        self.frame_motions.push(motion_magnitude.max(0.0));
    }

    /// Classify a motion value into an [`ActionType`].
    #[must_use]
    fn classify(&self, motion: f32) -> ActionType {
        if motion < 0.5 {
            ActionType::Static
        } else if motion < self.motion_threshold * 0.5 {
            ActionType::LowMotion
        } else {
            ActionType::HighMotion
        }
    }

    /// Return the detected action segments from all registered frames.
    ///
    /// Consecutive frames with the same classification are grouped.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detected_segments(&self) -> Vec<ActionSegment> {
        if self.frame_motions.is_empty() {
            return Vec::new();
        }

        let mut segments: Vec<ActionSegment> = Vec::new();
        let first_type = self.classify(self.frame_motions[0]);
        let mut seg_start = 0usize;
        let mut seg_type = first_type;
        let mut seg_sum = self.frame_motions[0];
        let mut seg_count = 1usize;

        for (i, &m) in self.frame_motions.iter().enumerate().skip(1) {
            let t = self.classify(m);
            if t == seg_type {
                seg_sum += m;
                seg_count += 1;
            } else {
                if seg_count >= self.min_segment_frames {
                    segments.push(ActionSegment::new(
                        seg_start,
                        i,
                        self.fps,
                        seg_type.clone(),
                        seg_sum / seg_count as f32,
                    ));
                }
                seg_start = i;
                seg_type = t;
                seg_sum = m;
                seg_count = 1;
            }
        }

        // Flush final segment.
        let total = self.frame_motions.len();
        if seg_count >= self.min_segment_frames {
            segments.push(ActionSegment::new(
                seg_start,
                total,
                self.fps,
                seg_type,
                seg_sum / seg_count as f32,
            ));
        }

        segments
    }
}

/// Summary report from an action detection pass.
#[derive(Debug, Clone)]
pub struct ActionReport {
    /// All detected segments, in temporal order.
    pub segments: Vec<ActionSegment>,
    /// Total number of frames analysed.
    pub total_frames: usize,
    /// Frames per second.
    pub fps: f32,
}

impl ActionReport {
    /// Build an [`ActionReport`] from a detector.
    #[must_use]
    pub fn from_detector(detector: &ActionDetector) -> Self {
        let segments = detector.detected_segments();
        let total_frames = detector.frame_motions.len();
        Self {
            segments,
            total_frames,
            fps: detector.fps,
        }
    }

    /// Return the fraction of total frames that belong to active (non-static) segments.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn action_density(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let active_frames: usize = self
            .segments
            .iter()
            .filter(|s| s.action_type != ActionType::Static)
            .map(ActionSegment::frame_count)
            .sum();
        active_frames as f32 / self.total_frames as f32
    }

    /// Return the number of detected segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_type_label() {
        assert_eq!(ActionType::HighMotion.label(), "high_motion");
        assert_eq!(ActionType::Static.label(), "static");
        assert_eq!(ActionType::CameraPan.label(), "camera_pan");
    }

    #[test]
    fn test_action_segment_duration_ms_basic() {
        let seg = ActionSegment::new(0, 30, 30.0, ActionType::HighMotion, 5.0);
        // 30 frames at 30 fps = 1000 ms
        assert!((seg.duration_ms() - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_action_segment_frame_count() {
        let seg = ActionSegment::new(10, 40, 25.0, ActionType::LowMotion, 1.0);
        assert_eq!(seg.frame_count(), 30);
    }

    #[test]
    fn test_action_segment_empty_range() {
        let seg = ActionSegment::new(10, 10, 25.0, ActionType::Static, 0.0);
        assert_eq!(seg.frame_count(), 0);
        assert!((seg.duration_ms() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_action_detector_no_frames() {
        let det = ActionDetector::new(25.0, 5.0, 2);
        assert!(det.detected_segments().is_empty());
    }

    #[test]
    fn test_action_detector_all_static() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..10 {
            det.add_frame(0.1);
        }
        let segs = det.detected_segments();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].action_type, ActionType::Static);
    }

    #[test]
    fn test_action_detector_high_motion_segment() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..10 {
            det.add_frame(10.0);
        }
        let segs = det.detected_segments();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].action_type, ActionType::HighMotion);
    }

    #[test]
    fn test_action_detector_mixed_segments() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..5 {
            det.add_frame(0.1); // static
        }
        for _ in 0..5 {
            det.add_frame(10.0); // high motion
        }
        let segs = det.detected_segments();
        assert!(segs.len() >= 2);
    }

    #[test]
    fn test_action_detector_min_segment_filters_short() {
        let mut det = ActionDetector::new(25.0, 5.0, 5);
        det.add_frame(10.0); // only 1 high-motion frame — below min
        det.add_frame(0.0); // static
        let segs = det.detected_segments();
        // The 1-frame high-motion block should be dropped.
        assert!(segs
            .iter()
            .all(|s| s.frame_count() >= 5 || s.frame_count() == 0));
    }

    #[test]
    fn test_action_report_action_density_all_static() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..10 {
            det.add_frame(0.1);
        }
        let report = ActionReport::from_detector(&det);
        assert!((report.action_density() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_action_report_action_density_all_active() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..10 {
            det.add_frame(10.0);
        }
        let report = ActionReport::from_detector(&det);
        assert!((report.action_density() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_action_report_no_frames_density_zero() {
        let det = ActionDetector::new(25.0, 5.0, 1);
        let report = ActionReport::from_detector(&det);
        assert!((report.action_density() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_action_report_segment_count() {
        let mut det = ActionDetector::new(25.0, 5.0, 1);
        for _ in 0..5 {
            det.add_frame(0.1);
        }
        for _ in 0..5 {
            det.add_frame(10.0);
        }
        let report = ActionReport::from_detector(&det);
        assert_eq!(report.segment_count(), report.segments.len());
    }
}
