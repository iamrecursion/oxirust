#![allow(dead_code)]
//! Insert mode semantics for clip placement operations.

/// How a clip insertion interacts with existing timeline content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InsertMode {
    /// Overwrites existing content at the target position.
    Overwrite,
    /// Pushes all subsequent clips downstream to make room.
    Ripple,
    /// Pulls downstream clips and fills gaps with handles.
    PushPull,
}

impl InsertMode {
    /// Short description of the insert mode.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            InsertMode::Overwrite => "Overwrite existing clips",
            InsertMode::Ripple => "Ripple downstream clips to make room",
            InsertMode::PushPull => "Push/pull downstream with handle preservation",
        }
    }

    /// Returns true if this mode shifts existing clips in time.
    #[must_use]
    pub fn shifts_clips(&self) -> bool {
        matches!(self, InsertMode::Ripple | InsertMode::PushPull)
    }
}

/// A candidate position for inserting a clip, with snap metadata.
#[derive(Debug, Clone)]
pub struct InsertPoint {
    /// Frame position on the timeline.
    pub frame: i64,
    /// Whether this point was snapped from an original request.
    pub snapped: bool,
    /// The original requested frame (before snapping).
    pub requested_frame: i64,
}

impl InsertPoint {
    /// Creates a new insert point.
    #[must_use]
    pub fn new(frame: i64) -> Self {
        Self {
            frame,
            snapped: false,
            requested_frame: frame,
        }
    }

    /// Snaps this insert point to the nearest snap position from `candidates`.
    /// Returns `self` unchanged when `candidates` is empty.
    #[must_use]
    pub fn snap_to_nearest(mut self, candidates: &[i64], threshold: i64) -> Self {
        if candidates.is_empty() {
            return self;
        }
        let (nearest, dist) = candidates
            .iter()
            .fold((candidates[0], i64::MAX), |best, &c| {
                let d = (c - self.frame).abs();
                if d < best.1 {
                    (c, d)
                } else {
                    best
                }
            });
        if dist <= threshold {
            self.requested_frame = self.frame;
            self.frame = nearest;
            self.snapped = true;
        }
        self
    }
}

/// Describes a pending insert operation and its expected outcome.
#[derive(Debug, Clone)]
pub struct InsertOperation {
    /// Mode to use for insertion.
    pub mode: InsertMode,
    /// Where to insert.
    pub point: InsertPoint,
    /// Duration of the clip being inserted (frames).
    pub clip_duration: u64,
    /// Duration of the timeline before the insert (frames).
    pub timeline_duration_before: u64,
}

impl InsertOperation {
    /// Creates a new insert operation.
    #[must_use]
    pub fn new(
        mode: InsertMode,
        point: InsertPoint,
        clip_duration: u64,
        timeline_duration_before: u64,
    ) -> Self {
        Self {
            mode,
            point,
            clip_duration,
            timeline_duration_before,
        }
    }

    /// Predicted resulting timeline duration after applying this operation.
    #[must_use]
    pub fn resulting_duration(&self) -> u64 {
        match self.mode {
            InsertMode::Overwrite => {
                // Overwrite does not extend beyond the clip's footprint unless it's at the tail.
                let end = self.point.frame.max(0) as u64 + self.clip_duration;
                self.timeline_duration_before.max(end)
            }
            InsertMode::Ripple | InsertMode::PushPull => {
                // Ripple always adds the full clip duration.
                self.timeline_duration_before + self.clip_duration
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overwrite_description() {
        assert_eq!(
            InsertMode::Overwrite.description(),
            "Overwrite existing clips"
        );
    }

    #[test]
    fn test_ripple_description() {
        assert!(InsertMode::Ripple.description().contains("Ripple"));
    }

    #[test]
    fn test_push_pull_description() {
        assert!(InsertMode::PushPull.description().contains("Push/pull"));
    }

    #[test]
    fn test_overwrite_does_not_shift() {
        assert!(!InsertMode::Overwrite.shifts_clips());
    }

    #[test]
    fn test_ripple_shifts() {
        assert!(InsertMode::Ripple.shifts_clips());
    }

    #[test]
    fn test_push_pull_shifts() {
        assert!(InsertMode::PushPull.shifts_clips());
    }

    #[test]
    fn test_insert_point_new() {
        let p = InsertPoint::new(100);
        assert_eq!(p.frame, 100);
        assert!(!p.snapped);
    }

    #[test]
    fn test_snap_to_nearest_within_threshold() {
        let p = InsertPoint::new(98).snap_to_nearest(&[100, 200, 50], 5);
        assert_eq!(p.frame, 100);
        assert!(p.snapped);
        assert_eq!(p.requested_frame, 98);
    }

    #[test]
    fn test_snap_outside_threshold_no_snap() {
        let p = InsertPoint::new(50).snap_to_nearest(&[100], 10);
        assert_eq!(p.frame, 50);
        assert!(!p.snapped);
    }

    #[test]
    fn test_snap_empty_candidates() {
        let p = InsertPoint::new(30).snap_to_nearest(&[], 5);
        assert_eq!(p.frame, 30);
    }

    #[test]
    fn test_ripple_resulting_duration() {
        let pt = InsertPoint::new(50);
        let op = InsertOperation::new(InsertMode::Ripple, pt, 30, 200);
        assert_eq!(op.resulting_duration(), 230);
    }

    #[test]
    fn test_overwrite_within_timeline() {
        let pt = InsertPoint::new(10);
        let op = InsertOperation::new(InsertMode::Overwrite, pt, 20, 200);
        // clip ends at frame 30, timeline stays 200
        assert_eq!(op.resulting_duration(), 200);
    }

    #[test]
    fn test_overwrite_extending_timeline() {
        let pt = InsertPoint::new(190);
        let op = InsertOperation::new(InsertMode::Overwrite, pt, 30, 200);
        // clip ends at 220, extends timeline
        assert_eq!(op.resulting_duration(), 220);
    }

    #[test]
    fn test_push_pull_resulting_duration() {
        let pt = InsertPoint::new(0);
        let op = InsertOperation::new(InsertMode::PushPull, pt, 48, 100);
        assert_eq!(op.resulting_duration(), 148);
    }
}
