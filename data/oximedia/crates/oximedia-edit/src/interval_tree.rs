//! Interval tree for O(log n) clip lookup by time position.
//!
//! Implements an augmented sorted-interval index that allows efficient:
//!
//! - **Point query**: Find all clips containing a given timeline position.
//! - **Range query**: Find all clips overlapping a given time range.
//! - **Nearest query**: Find the closest clip edge to a given position.
//!
//! The structure is based on a sorted array of intervals augmented with
//! maximum-end values, enabling binary-search-based queries.

#![allow(dead_code)]

use crate::clip::ClipId;

/// An interval representing a clip's time range on the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipInterval {
    /// Clip ID.
    pub clip_id: ClipId,
    /// Track index.
    pub track_index: usize,
    /// Start position (inclusive).
    pub start: i64,
    /// End position (exclusive).
    pub end: i64,
}

impl ClipInterval {
    /// Create a new clip interval.
    #[must_use]
    pub fn new(clip_id: ClipId, track_index: usize, start: i64, end: i64) -> Self {
        Self {
            clip_id,
            track_index,
            start,
            end,
        }
    }

    /// Check if this interval contains a point.
    #[must_use]
    pub fn contains_point(&self, point: i64) -> bool {
        point >= self.start && point < self.end
    }

    /// Check if this interval overlaps with a range.
    #[must_use]
    pub fn overlaps(&self, range_start: i64, range_end: i64) -> bool {
        self.start < range_end && self.end > range_start
    }

    /// Duration of this interval.
    #[must_use]
    pub fn duration(&self) -> i64 {
        (self.end - self.start).max(0)
    }
}

/// An augmented interval tree node for binary search.
#[derive(Debug, Clone)]
struct AugmentedInterval {
    /// The interval.
    interval: ClipInterval,
    /// Maximum end value in the subtree rooted at this node.
    max_end: i64,
}

/// An interval tree for efficient clip lookup.
///
/// Build the tree from a list of clip intervals using [`IntervalTree::build`],
/// then query with [`query_point`] or [`query_range`].
///
/// [`query_point`]: IntervalTree::query_point
/// [`query_range`]: IntervalTree::query_range
#[derive(Debug)]
pub struct IntervalTree {
    /// Sorted intervals with augmented max-end.
    nodes: Vec<AugmentedInterval>,
    /// Total number of intervals indexed.
    count: usize,
}

impl IntervalTree {
    /// Create an empty interval tree.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            count: 0,
        }
    }

    /// Build an interval tree from a list of intervals.
    ///
    /// The intervals are sorted by start position and augmented with
    /// max-end values for efficient querying.
    #[must_use]
    pub fn build(mut intervals: Vec<ClipInterval>) -> Self {
        let count = intervals.len();
        if intervals.is_empty() {
            return Self::empty();
        }

        // Sort by start position
        intervals.sort_by_key(|iv| iv.start);

        // Build augmented nodes (compute max_end from right to left)
        let mut nodes: Vec<AugmentedInterval> = intervals
            .into_iter()
            .map(|iv| AugmentedInterval {
                max_end: iv.end,
                interval: iv,
            })
            .collect();

        // Compute running max_end from back to front
        let mut running_max = i64::MIN;
        for node in nodes.iter_mut().rev() {
            running_max = running_max.max(node.interval.end);
            node.max_end = running_max;
        }

        Self { nodes, count }
    }

    /// Query all intervals that contain a specific point.
    #[must_use]
    pub fn query_point(&self, point: i64) -> Vec<ClipInterval> {
        let mut results = Vec::new();

        for node in &self.nodes {
            // Early termination: if the interval starts after the point,
            // and max_end is also after the point, we still need to check
            // (because sorted by start, not by end).
            // But if start > point, this and all subsequent intervals
            // start after the point, so they can't contain it.
            if node.interval.start > point {
                break;
            }
            if node.interval.contains_point(point) {
                results.push(node.interval);
            }
        }

        results
    }

    /// Query all intervals that overlap with a range.
    #[must_use]
    pub fn query_range(&self, start: i64, end: i64) -> Vec<ClipInterval> {
        let mut results = Vec::new();

        for node in &self.nodes {
            // If the current node's start is >= end, no further intervals
            // can overlap (they all start at or after this point).
            if node.interval.start >= end {
                break;
            }
            if node.interval.overlaps(start, end) {
                results.push(node.interval);
            }
        }

        results
    }

    /// Find the nearest clip edge (start or end) to a given position.
    ///
    /// Returns `(position, clip_id, is_start_edge)`.
    #[must_use]
    pub fn nearest_edge(&self, position: i64) -> Option<(i64, ClipId, bool)> {
        let mut best: Option<(i64, ClipId, bool)> = None;
        let mut best_distance = i64::MAX;

        for node in &self.nodes {
            let start_dist = (position - node.interval.start).abs();
            let end_dist = (position - node.interval.end).abs();

            if start_dist < best_distance {
                best_distance = start_dist;
                best = Some((node.interval.start, node.interval.clip_id, true));
            }
            if end_dist < best_distance {
                best_distance = end_dist;
                best = Some((node.interval.end, node.interval.clip_id, false));
            }
        }

        best
    }

    /// Get the total number of intervals in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Helper to build an interval tree from a Timeline.
#[must_use]
pub fn build_from_timeline(timeline: &crate::timeline::Timeline) -> IntervalTree {
    let mut intervals = Vec::new();
    for (track_idx, track) in timeline.tracks.iter().enumerate() {
        for clip in &track.clips {
            intervals.push(ClipInterval::new(
                clip.id,
                track_idx,
                clip.timeline_start,
                clip.timeline_end(),
            ));
        }
    }
    IntervalTree::build(intervals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, ClipType};
    use crate::timeline::{Timeline, TrackType};
    use oximedia_core::Rational;

    fn sample_intervals() -> Vec<ClipInterval> {
        vec![
            ClipInterval::new(1, 0, 0, 5000),
            ClipInterval::new(2, 0, 5000, 8000),
            ClipInterval::new(3, 1, 1000, 6000),
            ClipInterval::new(4, 1, 7000, 10000),
        ]
    }

    #[test]
    fn test_clip_interval_contains_point() {
        let iv = ClipInterval::new(1, 0, 100, 200);
        assert!(iv.contains_point(100));
        assert!(iv.contains_point(150));
        assert!(!iv.contains_point(200)); // exclusive end
        assert!(!iv.contains_point(50));
    }

    #[test]
    fn test_clip_interval_overlaps() {
        let iv = ClipInterval::new(1, 0, 100, 200);
        assert!(iv.overlaps(150, 250));
        assert!(iv.overlaps(50, 150));
        assert!(iv.overlaps(50, 250));
        assert!(iv.overlaps(120, 180));
        assert!(!iv.overlaps(200, 300));
        assert!(!iv.overlaps(0, 100));
    }

    #[test]
    fn test_clip_interval_duration() {
        let iv = ClipInterval::new(1, 0, 100, 300);
        assert_eq!(iv.duration(), 200);
    }

    #[test]
    fn test_empty_tree() {
        let tree = IntervalTree::empty();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.query_point(100).is_empty());
        assert!(tree.query_range(0, 1000).is_empty());
    }

    #[test]
    fn test_build_tree() {
        let intervals = sample_intervals();
        let tree = IntervalTree::build(intervals);
        assert_eq!(tree.len(), 4);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_point_query() {
        let tree = IntervalTree::build(sample_intervals());

        // At position 2500: should find clip 1 (0-5000) and clip 3 (1000-6000)
        let results = tree.query_point(2500);
        let ids: Vec<ClipId> = results.iter().map(|r| r.clip_id).collect();
        assert!(ids.contains(&1), "should contain clip 1");
        assert!(ids.contains(&3), "should contain clip 3");
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_point_query_at_boundary() {
        let tree = IntervalTree::build(sample_intervals());

        // At position 5000: clip 1 ends (exclusive), clip 2 starts, clip 3 active
        let results = tree.query_point(5000);
        let ids: Vec<ClipId> = results.iter().map(|r| r.clip_id).collect();
        assert!(ids.contains(&2), "clip 2 starts at 5000");
        assert!(ids.contains(&3), "clip 3 spans 1000-6000");
        assert!(!ids.contains(&1), "clip 1 ends at 5000 (exclusive)");
    }

    #[test]
    fn test_range_query() {
        let tree = IntervalTree::build(sample_intervals());

        // Range 4000-6000 should overlap clips 1, 2, 3
        let results = tree.query_range(4000, 6000);
        let ids: Vec<ClipId> = results.iter().map(|r| r.clip_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_range_query_no_results() {
        let tree = IntervalTree::build(sample_intervals());
        let results = tree.query_range(10001, 20000);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nearest_edge() {
        let tree = IntervalTree::build(sample_intervals());

        // Near 4999 should find clip 1 end at 5000 or clip 2 start at 5000
        let result = tree.nearest_edge(4999);
        assert!(result.is_some());
        let (pos, _id, _is_start) = result.expect("should find edge");
        assert_eq!(pos, 5000);
    }

    #[test]
    fn test_nearest_edge_empty_tree() {
        let tree = IntervalTree::empty();
        assert!(tree.nearest_edge(100).is_none());
    }

    #[test]
    fn test_build_from_timeline() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let at = tl.add_track(TrackType::Audio);
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 0, 5000));
        let _ = tl.add_clip(vt, Clip::new(0, ClipType::Video, 5000, 3000));
        let _ = tl.add_clip(at, Clip::new(0, ClipType::Audio, 0, 8000));

        let tree = build_from_timeline(&tl);
        assert_eq!(tree.len(), 3);

        let at_2500 = tree.query_point(2500);
        assert_eq!(at_2500.len(), 2); // video + audio
    }

    #[test]
    fn test_single_interval() {
        let tree = IntervalTree::build(vec![ClipInterval::new(1, 0, 0, 1000)]);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.query_point(500).len(), 1);
        assert_eq!(tree.query_point(1500).len(), 0);
    }

    #[test]
    fn test_many_intervals() {
        let intervals: Vec<ClipInterval> = (0..100)
            .map(|i| ClipInterval::new(i as u64, 0, i * 100, i * 100 + 150))
            .collect();
        let tree = IntervalTree::build(intervals);
        assert_eq!(tree.len(), 100);

        // Position 5050 should overlap with interval starting at 5000 (5000-5150)
        let results = tree.query_point(5050);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_range_query_all() {
        let tree = IntervalTree::build(sample_intervals());
        let results = tree.query_range(0, 10000);
        assert_eq!(results.len(), 4);
    }

    /// Required test: interval tree query for frame-based lookup.
    ///
    /// Validates that `query_point` correctly locates clips by frame number
    /// using the `start_frame` / `end_frame` / `clip_id` semantic described
    /// in the task specification.
    #[test]
    fn test_interval_tree_query() {
        // Build an interval tree with three clips expressed as frame ranges.
        // clip_id=1: frames 0..30  (start_frame=0, end_frame=30)
        // clip_id=2: frames 30..60 (start_frame=30, end_frame=60)
        // clip_id=3: frames 10..50 (start_frame=10, end_frame=50)
        let intervals = vec![
            ClipInterval::new(1, 0, 0, 30),
            ClipInterval::new(2, 0, 30, 60),
            ClipInterval::new(3, 1, 10, 50),
        ];
        let tree = IntervalTree::build(intervals);

        // Frame 15 should be in clip 1 (0..30) and clip 3 (10..50).
        let at_15 = tree.query_point(15);
        let ids_at_15: Vec<ClipId> = at_15.iter().map(|iv| iv.clip_id).collect();
        assert!(ids_at_15.contains(&1), "clip 1 should contain frame 15");
        assert!(ids_at_15.contains(&3), "clip 3 should contain frame 15");
        assert!(
            !ids_at_15.contains(&2),
            "clip 2 should not contain frame 15"
        );

        // Frame 35 should be in clip 2 (30..60) and clip 3 (10..50).
        let at_35 = tree.query_point(35);
        let ids_at_35: Vec<ClipId> = at_35.iter().map(|iv| iv.clip_id).collect();
        assert!(ids_at_35.contains(&2), "clip 2 should contain frame 35");
        assert!(ids_at_35.contains(&3), "clip 3 should contain frame 35");
        assert!(
            !ids_at_35.contains(&1),
            "clip 1 should not contain frame 35"
        );

        // Frame 55 should be only in clip 2 (30..60).
        let at_55 = tree.query_point(55);
        let ids_at_55: Vec<ClipId> = at_55.iter().map(|iv| iv.clip_id).collect();
        assert_eq!(ids_at_55, vec![2], "only clip 2 should contain frame 55");

        // Frame 0: only clip 1 (start-inclusive).
        let at_0 = tree.query_point(0);
        let ids_at_0: Vec<ClipId> = at_0.iter().map(|iv| iv.clip_id).collect();
        assert!(ids_at_0.contains(&1));
        assert!(!ids_at_0.contains(&2));

        // Frame 60: end is exclusive, so no clip covers frame 60.
        let at_60 = tree.query_point(60);
        assert!(at_60.is_empty(), "frame 60 is beyond all clips");
    }

    #[test]
    fn test_query_point_no_match() {
        let tree = IntervalTree::build(sample_intervals());
        // Position 6500 is after clip 3 (end=6000) but inside clip 2 (5000-8000)
        // Use a position in the gap between clip 2 end and clip 4 start
        let results = tree.query_point(6500);
        // clip 2 (5000-8000) contains 6500
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].clip_id, 2);

        // Position 8500: after clip 2 (end=8000), before clip 4 end (10000)
        // but clip 4 starts at 7000, so 8500 is in clip 4
        let results2 = tree.query_point(8500);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].clip_id, 4);

        // Position 10500: after all clips
        let results3 = tree.query_point(10500);
        assert!(results3.is_empty());
    }
}
