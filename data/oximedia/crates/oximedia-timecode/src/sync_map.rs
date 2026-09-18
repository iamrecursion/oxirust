//! Timecode sync/offset mapping
//!
//! Provides `TcOffset`, `TcSyncPoint`, and `SyncMap` for translating frame
//! positions between two timecode domains using linear interpolation.

#[allow(dead_code)]
/// A fixed frame offset with an optional description
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcOffset {
    /// Signed offset in frames
    pub offset_frames: i64,
    /// Human-readable description of this offset
    pub description: String,
}

impl TcOffset {
    /// Create a new `TcOffset`
    #[must_use]
    pub fn new(offset_frames: i64, description: String) -> Self {
        Self {
            offset_frames,
            description,
        }
    }

    /// Returns `true` when the offset is exactly zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.offset_frames == 0
    }

    /// Return a new `TcOffset` with the sign negated
    #[must_use]
    pub fn negate(&self) -> Self {
        Self {
            offset_frames: -self.offset_frames,
            description: self.description.clone(),
        }
    }
}

#[allow(dead_code)]
/// A correspondence between a source frame and a target frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcSyncPoint {
    /// Frame number in the source domain
    pub source_frame: i64,
    /// Corresponding frame number in the target domain
    pub target_frame: i64,
}

impl TcSyncPoint {
    /// Create a new sync point
    #[must_use]
    pub fn new(source_frame: i64, target_frame: i64) -> Self {
        Self {
            source_frame,
            target_frame,
        }
    }

    /// Returns `target_frame - source_frame`
    #[must_use]
    pub fn offset(&self) -> i64 {
        self.target_frame - self.source_frame
    }
}

#[allow(dead_code)]
/// A collection of sync points that defines a piecewise-linear mapping between
/// source and target frame domains.
#[derive(Debug, Clone, Default)]
pub struct SyncMap {
    /// Ordered list of sync points (sorted by source_frame)
    pub sync_points: Vec<TcSyncPoint>,
}

impl SyncMap {
    /// Create an empty `SyncMap`
    #[must_use]
    pub fn new() -> Self {
        Self {
            sync_points: Vec::new(),
        }
    }

    /// Add a sync point. The internal list is kept sorted by `source_frame`.
    pub fn add_sync_point(&mut self, point: TcSyncPoint) {
        self.sync_points.push(point);
        self.sync_points.sort_by_key(|p| p.source_frame);
    }

    /// Return the number of sync points
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.sync_points.len()
    }

    /// Convert a source frame to a target frame using linear interpolation.
    ///
    /// - If there are no sync points, returns `source_frame` unchanged.
    /// - If `source_frame` is before the first sync point or after the last,
    ///   the nearest endpoint's offset is extrapolated.
    /// - Otherwise, interpolates linearly between the bounding sync points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn source_to_target(&self, frame: i64) -> i64 {
        if self.sync_points.is_empty() {
            return frame;
        }
        if self.sync_points.len() == 1 {
            return frame + self.sync_points[0].offset();
        }

        let first = &self.sync_points[0];
        let last = &self.sync_points[self.sync_points.len() - 1];

        if frame <= first.source_frame {
            return frame + first.offset();
        }
        if frame >= last.source_frame {
            return frame + last.offset();
        }

        // Binary search for the segment
        let idx = self
            .sync_points
            .partition_point(|p| p.source_frame <= frame);
        let lo = &self.sync_points[idx - 1];
        let hi = &self.sync_points[idx];

        let span = (hi.source_frame - lo.source_frame) as f64;
        let t = (frame - lo.source_frame) as f64 / span;
        let interp_target = lo.target_frame as f64 + t * (hi.target_frame - lo.target_frame) as f64;
        interp_target.round() as i64
    }

    /// Convert a target frame to a source frame using linear interpolation.
    ///
    /// Uses the same piecewise-linear logic but operates on the target axis.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn target_to_source(&self, frame: i64) -> i64 {
        if self.sync_points.is_empty() {
            return frame;
        }
        if self.sync_points.len() == 1 {
            return frame - self.sync_points[0].offset();
        }

        // Sort by target_frame for this lookup
        let mut by_target = self.sync_points.clone();
        by_target.sort_by_key(|p| p.target_frame);

        let first = &by_target[0];
        let last = &by_target[by_target.len() - 1];

        if frame <= first.target_frame {
            return frame - first.offset();
        }
        if frame >= last.target_frame {
            return frame - last.offset();
        }

        let idx = by_target.partition_point(|p| p.target_frame <= frame);
        let lo = &by_target[idx - 1];
        let hi = &by_target[idx];

        let span = (hi.target_frame - lo.target_frame) as f64;
        let t = (frame - lo.target_frame) as f64 / span;
        let interp_source = lo.source_frame as f64 + t * (hi.source_frame - lo.source_frame) as f64;
        interp_source.round() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tc_offset_is_zero_true() {
        let o = TcOffset::new(0, "zero".into());
        assert!(o.is_zero());
    }

    #[test]
    fn test_tc_offset_is_zero_false() {
        let o = TcOffset::new(10, "shift".into());
        assert!(!o.is_zero());
    }

    #[test]
    fn test_tc_offset_negate() {
        let o = TcOffset::new(25, "shift".into());
        let neg = o.negate();
        assert_eq!(neg.offset_frames, -25);
    }

    #[test]
    fn test_tc_offset_negate_zero() {
        let o = TcOffset::new(0, "zero".into());
        assert_eq!(o.negate().offset_frames, 0);
    }

    #[test]
    fn test_tc_sync_point_offset() {
        let p = TcSyncPoint::new(100, 150);
        assert_eq!(p.offset(), 50);
    }

    #[test]
    fn test_tc_sync_point_negative_offset() {
        let p = TcSyncPoint::new(200, 100);
        assert_eq!(p.offset(), -100);
    }

    #[test]
    fn test_sync_map_empty_passthrough() {
        let map = SyncMap::new();
        assert_eq!(map.source_to_target(42), 42);
        assert_eq!(map.target_to_source(42), 42);
    }

    #[test]
    fn test_sync_map_single_point() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(0, 100));
        // source frame 0 maps to target 100; frame 50 maps to 150
        assert_eq!(map.source_to_target(0), 100);
        assert_eq!(map.source_to_target(50), 150);
    }

    #[test]
    fn test_sync_map_two_points_interpolation() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(0, 0));
        map.add_sync_point(TcSyncPoint::new(100, 200));
        // At midpoint source=50, target should be 100
        assert_eq!(map.source_to_target(50), 100);
    }

    #[test]
    fn test_sync_map_extrapolate_before_first() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(100, 110));
        map.add_sync_point(TcSyncPoint::new(200, 220));
        // Before first point: uses first offset (+10)
        assert_eq!(map.source_to_target(50), 60);
    }

    #[test]
    fn test_sync_map_extrapolate_after_last() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(0, 0));
        map.add_sync_point(TcSyncPoint::new(100, 200));
        // After last point: uses last offset (+100)
        assert_eq!(map.source_to_target(150), 250);
    }

    #[test]
    fn test_sync_map_target_to_source_single_point() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(0, 100));
        assert_eq!(map.target_to_source(100), 0);
        assert_eq!(map.target_to_source(150), 50);
    }

    #[test]
    fn test_sync_map_point_count() {
        let mut map = SyncMap::new();
        assert_eq!(map.point_count(), 0);
        map.add_sync_point(TcSyncPoint::new(0, 0));
        assert_eq!(map.point_count(), 1);
        map.add_sync_point(TcSyncPoint::new(100, 100));
        assert_eq!(map.point_count(), 2);
    }

    #[test]
    fn test_sync_map_sorted_after_add() {
        let mut map = SyncMap::new();
        map.add_sync_point(TcSyncPoint::new(200, 200));
        map.add_sync_point(TcSyncPoint::new(0, 0));
        map.add_sync_point(TcSyncPoint::new(100, 100));
        assert_eq!(map.sync_points[0].source_frame, 0);
        assert_eq!(map.sync_points[1].source_frame, 100);
        assert_eq!(map.sync_points[2].source_frame, 200);
    }
}
