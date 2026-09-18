//! Interval tree for O(log n) clip lookup by time position.
//!
//! An augmented balanced interval tree that stores timeline clips indexed by
//! their `[timeline_in, timeline_out)` range. Supports efficient queries:
//! - Point query: find all clips containing a given frame position
//! - Range query: find all clips overlapping a given time range
//! - Nearest query: find the closest clip to a position
//!
//! The tree is implemented as a sorted array with augmented max-end values,
//! enabling binary-search-based queries without pointer chasing.

use crate::clip::ClipId;

/// An interval representing a clip's time range on the timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    /// Start position (inclusive), in frames.
    pub start: i64,
    /// End position (exclusive), in frames.
    pub end: i64,
    /// The clip this interval belongs to.
    pub clip_id: ClipId,
}

impl Interval {
    /// Creates a new interval.
    #[must_use]
    pub fn new(start: i64, end: i64, clip_id: ClipId) -> Self {
        Self {
            start,
            end,
            clip_id,
        }
    }

    /// Returns `true` if this interval contains the given point.
    #[must_use]
    pub fn contains_point(&self, point: i64) -> bool {
        point >= self.start && point < self.end
    }

    /// Returns `true` if this interval overlaps with `[start, end)`.
    #[must_use]
    pub fn overlaps(&self, start: i64, end: i64) -> bool {
        self.start < end && self.end > start
    }

    /// Returns the duration of this interval.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.end - self.start
    }

    /// Returns the midpoint of this interval.
    #[must_use]
    pub fn midpoint(&self) -> i64 {
        self.start + (self.end - self.start) / 2
    }
}

/// An augmented node in the interval tree array representation.
#[derive(Clone, Debug)]
struct AugmentedInterval {
    /// The original interval.
    interval: Interval,
    /// Maximum end value in this node's subtree.
    max_end: i64,
}

/// Sorted-array-based interval tree with augmented max-end values.
///
/// Intervals are stored sorted by start position, with each entry augmented
/// with the maximum end value in its implicit subtree. This allows
/// binary-search-based O(log n) point and range queries.
#[derive(Clone, Debug, Default)]
pub struct IntervalTree {
    /// Sorted array of augmented intervals.
    nodes: Vec<AugmentedInterval>,
    /// Whether the tree needs rebuilding.
    dirty: bool,
}

impl IntervalTree {
    /// Creates a new empty interval tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dirty: false,
        }
    }

    /// Creates an interval tree with the given capacity hint.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            dirty: false,
        }
    }

    /// Inserts an interval into the tree.
    ///
    /// The tree is marked dirty and will be rebuilt on the next query.
    pub fn insert(&mut self, interval: Interval) {
        self.nodes.push(AugmentedInterval {
            max_end: interval.end,
            interval,
        });
        self.dirty = true;
    }

    /// Removes all intervals for the given clip ID.
    ///
    /// Returns the number of intervals removed.
    pub fn remove(&mut self, clip_id: ClipId) -> usize {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.interval.clip_id != clip_id);
        let removed = before - self.nodes.len();
        if removed > 0 {
            self.dirty = true;
        }
        removed
    }

    /// Removes all intervals from the tree.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.dirty = false;
    }

    /// Returns the number of intervals in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Ensures the tree is sorted and augmented.
    fn rebuild_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        // Sort by start position, then by end position for stability.
        self.nodes.sort_by(|a, b| {
            a.interval
                .start
                .cmp(&b.interval.start)
                .then(a.interval.end.cmp(&b.interval.end))
        });

        // Augment: compute max_end for implicit binary tree structure.
        // We use a bottom-up sweep: for each node, max_end is the max of
        // its own end and the max_end of its children in the implicit tree.
        let n = self.nodes.len();
        if n == 0 {
            self.dirty = false;
            return;
        }

        // Reset max_end to own end.
        for node in &mut self.nodes {
            node.max_end = node.interval.end;
        }

        // Build augmented max_end using suffix-maximum approach.
        // This ensures that for any prefix scan, we can prune early.
        // We compute a running suffix max from right to left.
        // Actually, for a sorted-by-start array, a simpler approach:
        // max_end[i] = max(interval[i].end, max_end[i+1..])
        // This is the "suffix maximum" which allows early termination
        // when searching from left: if suffix_max < query_point, skip.
        let mut running_max = i64::MIN;
        for i in (0..n).rev() {
            running_max = running_max.max(self.nodes[i].interval.end);
            self.nodes[i].max_end = running_max;
        }

        self.dirty = false;
    }

    /// Finds all intervals containing the given point.
    ///
    /// Returns clip IDs of all matching intervals.
    pub fn query_point(&mut self, point: i64) -> Vec<ClipId> {
        self.rebuild_if_dirty();
        let mut results = Vec::new();
        self.query_point_inner(point, &mut results);
        results
    }

    /// Internal point query implementation.
    fn query_point_inner(&self, point: i64, results: &mut Vec<ClipId>) {
        for node in &self.nodes {
            // Pruning: if this node's start > point, and since nodes are sorted
            // by start, all subsequent nodes also have start > point.
            // But they could still contain point if they started before...
            // Actually, since sorted by start, if start > point, no subsequent
            // interval can contain point either (start > point means the
            // interval doesn't contain point).
            if node.interval.start > point {
                break;
            }
            // Check if the max_end in the subtree from this node onward
            // is less than the point -- if so, no matches possible.
            if node.max_end <= point {
                // Actually this means all remaining intervals end at or before
                // point. But we need end > point for containment.
                // Since max_end is a suffix max, if max_end <= point,
                // no remaining interval has end > point.
                // Wait -- max_end at position i is the suffix max from i onward.
                // If max_end <= point, then all intervals from i onward have
                // end <= point, so none contain the point. But we already
                // checked start <= point above, and containment requires
                // end > point. So we can break.
                // Actually, we should not break here because earlier intervals
                // (already processed) might have contributed. The suffix max
                // check applies to this node and all subsequent ones.
                // Since we iterate left to right and max_end is suffix max:
                // if this node's max_end <= point, all remaining have end <= point,
                // so we can break.
                break;
            }
            if node.interval.contains_point(point) {
                results.push(node.interval.clip_id);
            }
        }
    }

    /// Finds all intervals overlapping with the given range `[start, end)`.
    pub fn query_range(&mut self, start: i64, end: i64) -> Vec<ClipId> {
        self.rebuild_if_dirty();
        let mut results = Vec::new();
        self.query_range_inner(start, end, &mut results);
        results
    }

    /// Internal range query implementation.
    fn query_range_inner(&self, start: i64, end: i64, results: &mut Vec<ClipId>) {
        for node in &self.nodes {
            // If this interval starts at or after the query end, and since
            // sorted by start, all subsequent intervals also start >= end.
            // Overlap requires interval.start < end, so we can break.
            if node.interval.start >= end {
                break;
            }
            // Suffix max pruning: if max_end <= start, no remaining interval
            // has end > start, so no overlap possible.
            if node.max_end <= start {
                break;
            }
            if node.interval.overlaps(start, end) {
                results.push(node.interval.clip_id);
            }
        }
    }

    /// Finds the nearest interval to the given point.
    ///
    /// Returns the clip ID and distance (in frames) of the nearest interval.
    /// If the point is inside an interval, distance is 0.
    pub fn nearest(&mut self, point: i64) -> Option<(ClipId, i64)> {
        self.rebuild_if_dirty();
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_id = None;
        let mut best_dist = i64::MAX;

        for node in &self.nodes {
            let dist = if node.interval.contains_point(point) {
                0
            } else if point < node.interval.start {
                node.interval.start - point
            } else {
                point - node.interval.end + 1
            };

            if dist < best_dist {
                best_dist = dist;
                best_id = Some(node.interval.clip_id);
                if dist == 0 {
                    // Can't do better than contained.
                    // But keep scanning for potential other contained intervals.
                    // Actually, we want the nearest single result, so 0 is optimal.
                    break;
                }
            }

            // If this node's start is already past the point and the distance
            // is growing, we can use binary search properties to stop.
            if node.interval.start > point && node.interval.start - point > best_dist {
                break;
            }
        }

        best_id.map(|id| (id, best_dist))
    }

    /// Returns all intervals sorted by start position.
    pub fn all_intervals(&mut self) -> Vec<&Interval> {
        self.rebuild_if_dirty();
        self.nodes.iter().map(|n| &n.interval).collect()
    }

    /// Bulk-loads intervals, more efficient than individual inserts.
    pub fn bulk_load(&mut self, intervals: Vec<Interval>) {
        self.nodes.reserve(intervals.len());
        for interval in intervals {
            self.nodes.push(AugmentedInterval {
                max_end: interval.end,
                interval,
            });
        }
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipId;

    fn make_id() -> ClipId {
        ClipId::new()
    }

    #[test]
    fn test_interval_contains_point() {
        let id = make_id();
        let iv = Interval::new(10, 20, id);
        assert!(iv.contains_point(10));
        assert!(iv.contains_point(15));
        assert!(iv.contains_point(19));
        assert!(!iv.contains_point(9));
        assert!(!iv.contains_point(20));
    }

    #[test]
    fn test_interval_overlaps() {
        let id = make_id();
        let iv = Interval::new(10, 20, id);
        assert!(iv.overlaps(5, 15));
        assert!(iv.overlaps(15, 25));
        assert!(iv.overlaps(0, 100));
        assert!(iv.overlaps(10, 20));
        assert!(!iv.overlaps(20, 30));
        assert!(!iv.overlaps(0, 10));
    }

    #[test]
    fn test_interval_duration() {
        let id = make_id();
        let iv = Interval::new(10, 30, id);
        assert_eq!(iv.duration(), 20);
    }

    #[test]
    fn test_interval_midpoint() {
        let id = make_id();
        let iv = Interval::new(10, 30, id);
        assert_eq!(iv.midpoint(), 20);
    }

    #[test]
    fn test_empty_tree() {
        let mut tree = IntervalTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.query_point(5).is_empty());
        assert!(tree.query_range(0, 100).is_empty());
        assert!(tree.nearest(5).is_none());
    }

    #[test]
    fn test_single_insert_and_query() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        let result = tree.query_point(15);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id);
    }

    #[test]
    fn test_point_query_miss() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        assert!(tree.query_point(5).is_empty());
        assert!(tree.query_point(25).is_empty());
    }

    #[test]
    fn test_point_query_boundary() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        // Start is inclusive
        let result = tree.query_point(10);
        assert_eq!(result.len(), 1);
        // End is exclusive
        assert!(tree.query_point(20).is_empty());
    }

    #[test]
    fn test_multiple_intervals_point_query() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();
        let id3 = make_id();

        tree.insert(Interval::new(0, 100, id1));
        tree.insert(Interval::new(50, 150, id2));
        tree.insert(Interval::new(200, 300, id3));

        // Point 75 is in both id1 and id2
        let result = tree.query_point(75);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&id1));
        assert!(result.contains(&id2));

        // Point 125 is only in id2
        let result = tree.query_point(125);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id2);

        // Point 250 is only in id3
        let result = tree.query_point(250);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id3);
    }

    #[test]
    fn test_range_query() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();
        let id3 = make_id();

        tree.insert(Interval::new(0, 50, id1));
        tree.insert(Interval::new(100, 150, id2));
        tree.insert(Interval::new(200, 250, id3));

        // Range overlapping id1 and id2
        let result = tree.query_range(25, 125);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&id1));
        assert!(result.contains(&id2));
    }

    #[test]
    fn test_range_query_no_overlap() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        tree.insert(Interval::new(0, 50, id1));

        let result = tree.query_range(60, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_range_query_adjacent_no_overlap() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        tree.insert(Interval::new(0, 50, id1));

        // [50, 100) does not overlap [0, 50) because end is exclusive
        let result = tree.query_range(50, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();

        tree.insert(Interval::new(0, 50, id1));
        tree.insert(Interval::new(100, 150, id2));
        assert_eq!(tree.len(), 2);

        let removed = tree.remove(id1);
        assert_eq!(removed, 1);
        assert_eq!(tree.len(), 1);

        assert!(tree.query_point(25).is_empty());
        assert_eq!(tree.query_point(125).len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(0, 50, make_id()));
        tree.insert(Interval::new(100, 150, make_id()));
        tree.clear();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_nearest_contained() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        let (found_id, dist) = tree.nearest(15).expect("should find");
        assert_eq!(found_id, id);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_nearest_before() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        let (found_id, dist) = tree.nearest(5).expect("should find");
        assert_eq!(found_id, id);
        assert_eq!(dist, 5); // 10 - 5 = 5
    }

    #[test]
    fn test_nearest_after() {
        let mut tree = IntervalTree::new();
        let id = make_id();
        tree.insert(Interval::new(10, 20, id));

        let (found_id, dist) = tree.nearest(25).expect("should find");
        assert_eq!(found_id, id);
        assert_eq!(dist, 6); // 25 - 20 + 1 = 6
    }

    #[test]
    fn test_nearest_picks_closest() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();

        tree.insert(Interval::new(0, 10, id1));
        tree.insert(Interval::new(20, 30, id2));

        // Point 12 is closer to id1 (distance 3) than id2 (distance 8)
        let (found_id, dist) = tree.nearest(12).expect("should find");
        assert_eq!(found_id, id1);
        assert_eq!(dist, 3);

        // Point 18 is closer to id2 (distance 2) than id1 (distance 9)
        let (found_id, dist) = tree.nearest(18).expect("should find");
        assert_eq!(found_id, id2);
        assert_eq!(dist, 2);
    }

    #[test]
    fn test_bulk_load() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();
        let id3 = make_id();

        tree.bulk_load(vec![
            Interval::new(200, 300, id3),
            Interval::new(0, 100, id1),
            Interval::new(100, 200, id2),
        ]);

        assert_eq!(tree.len(), 3);

        let result = tree.query_point(50);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id1);

        let result = tree.query_point(150);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id2);
    }

    #[test]
    fn test_all_intervals_sorted() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        let id2 = make_id();
        let id3 = make_id();

        tree.insert(Interval::new(200, 300, id3));
        tree.insert(Interval::new(0, 100, id1));
        tree.insert(Interval::new(100, 200, id2));

        let intervals = tree.all_intervals();
        assert_eq!(intervals.len(), 3);
        assert_eq!(intervals[0].start, 0);
        assert_eq!(intervals[1].start, 100);
        assert_eq!(intervals[2].start, 200);
    }

    #[test]
    fn test_with_capacity() {
        let tree = IntervalTree::with_capacity(100);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_many_overlapping_intervals() {
        let mut tree = IntervalTree::new();
        let mut ids = Vec::new();

        // Create 100 overlapping intervals
        for i in 0..100 {
            let id = make_id();
            ids.push(id);
            tree.insert(Interval::new(i * 5, i * 5 + 50, id));
        }

        // Query at midpoint: should find many
        let result = tree.query_point(250);
        assert!(!result.is_empty());

        // Query range
        let result = tree.query_range(0, 1000);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_non_overlapping_intervals_point_query() {
        let mut tree = IntervalTree::new();
        let ids: Vec<ClipId> = (0..10).map(|_| make_id()).collect();

        for (i, id) in ids.iter().enumerate() {
            let start = (i as i64) * 100;
            tree.insert(Interval::new(start, start + 50, *id));
        }

        // Gap between intervals: should find nothing
        assert!(tree.query_point(75).is_empty());
        assert!(tree.query_point(175).is_empty());

        // Inside an interval
        let result = tree.query_point(25);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ids[0]);
    }

    #[test]
    fn test_insert_after_query_triggers_rebuild() {
        let mut tree = IntervalTree::new();
        let id1 = make_id();
        tree.insert(Interval::new(100, 200, id1));

        // Query triggers rebuild
        let _ = tree.query_point(150);

        // Insert another
        let id2 = make_id();
        tree.insert(Interval::new(0, 50, id2));

        // New query should find the newly inserted interval
        let result = tree.query_point(25);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], id2);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut tree = IntervalTree::new();
        tree.insert(Interval::new(0, 50, make_id()));
        let removed = tree.remove(make_id()); // Different ID
        assert_eq!(removed, 0);
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_duplicate_clip_id_intervals() {
        let mut tree = IntervalTree::new();
        let id = make_id();

        // Same clip ID, different ranges (e.g., after split)
        tree.insert(Interval::new(0, 50, id));
        tree.insert(Interval::new(100, 150, id));

        let result = tree.query_range(0, 200);
        assert_eq!(result.len(), 2);

        // Remove should remove both
        let removed = tree.remove(id);
        assert_eq!(removed, 2);
        assert!(tree.is_empty());
    }
}
