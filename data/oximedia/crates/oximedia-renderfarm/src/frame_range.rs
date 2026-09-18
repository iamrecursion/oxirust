//! Frame range utilities for render job specification.
//!
//! [`FrameRange`] describes a contiguous or strided range of frame numbers.
//! [`FrameRangeList`] aggregates multiple such ranges for sparse frame sets
//! common in VFX pipelines (e.g. "render frames 1-10 and 50-60").

#![allow(dead_code)]

/// A contiguous range of frame numbers with an optional step.
///
/// # Example
///
/// ```
/// use oximedia_renderfarm::frame_range::FrameRange;
///
/// let r = FrameRange::new(1, 10);
/// assert_eq!(r.len(), 10);
/// let frames: Vec<i32> = r.iter().collect();
/// assert_eq!(frames[0], 1);
/// assert_eq!(frames[9], 10);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRange {
    /// First frame (inclusive).
    pub first: i32,
    /// Last frame (inclusive).
    pub last: i32,
    /// Step between consecutive frames (must be ≥ 1).
    pub step: u32,
}

impl FrameRange {
    /// Create a new frame range with step 1.
    ///
    /// If `first > last` the range is considered empty.
    #[must_use]
    pub fn new(first: i32, last: i32) -> Self {
        Self {
            first,
            last,
            step: 1,
        }
    }

    /// Create a new frame range with a custom step.
    ///
    /// `step` is clamped to a minimum of 1.
    #[must_use]
    pub fn with_step(first: i32, last: i32, step: u32) -> Self {
        Self {
            first,
            last,
            step: step.max(1),
        }
    }

    /// Number of frames this range covers.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.first > self.last {
            return 0;
        }
        let span = (self.last - self.first) as u64;
        let step = u64::from(self.step);
        ((span / step) + 1) as usize
    }

    /// Return `true` if the range contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.first > self.last
    }

    /// Return `true` if `frame` falls within this range.
    #[must_use]
    pub fn contains(&self, frame: i32) -> bool {
        if frame < self.first || frame > self.last {
            return false;
        }
        let offset = (frame - self.first) as u64;
        offset % u64::from(self.step) == 0
    }

    /// Return an iterator over every frame number in this range.
    #[must_use]
    pub fn iter(&self) -> FrameRangeIter {
        FrameRangeIter {
            current: self.first,
            last: self.last,
            step: self.step as i32,
        }
    }

    /// Split the range into chunks of at most `size` frames each.
    ///
    /// Returns an empty `Vec` if the range is empty or `size` is zero.
    #[must_use]
    pub fn chunks(&self, size: usize) -> Vec<FrameRange> {
        if size == 0 || self.is_empty() {
            return Vec::new();
        }
        let frames: Vec<i32> = self.iter().collect();
        frames
            .chunks(size)
            .map(|chunk| {
                FrameRange::with_step(
                    *chunk
                        .first()
                        .expect("invariant: chunks() produces non-empty slices"),
                    *chunk
                        .last()
                        .expect("invariant: chunks() produces non-empty slices"),
                    self.step,
                )
            })
            .collect()
    }

    /// Duration in frames (alias for [`len`](Self::len)).
    #[must_use]
    pub fn duration(&self) -> usize {
        self.len()
    }
}

impl std::fmt::Display for FrameRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.step == 1 {
            write!(f, "{}-{}", self.first, self.last)
        } else {
            write!(f, "{}-{}x{}", self.first, self.last, self.step)
        }
    }
}

/// Iterator produced by [`FrameRange::iter`].
#[derive(Debug)]
pub struct FrameRangeIter {
    current: i32,
    last: i32,
    step: i32,
}

impl Iterator for FrameRangeIter {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.last {
            return None;
        }
        let val = self.current;
        self.current += self.step;
        Some(val)
    }
}

/// A collection of [`FrameRange`] values representing a (possibly sparse)
/// set of frames.
///
/// # Example
///
/// ```
/// use oximedia_renderfarm::frame_range::{FrameRange, FrameRangeList};
///
/// let mut list = FrameRangeList::new();
/// list.add(FrameRange::new(1, 10));
/// list.add(FrameRange::new(50, 60));
/// assert_eq!(list.total_frames(), 21);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FrameRangeList {
    ranges: Vec<FrameRange>,
}

impl FrameRangeList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a range to the list.
    pub fn add(&mut self, range: FrameRange) {
        self.ranges.push(range);
    }

    /// Total number of frames across all ranges (duplicates are **not**
    /// deduplicated).
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.ranges.iter().map(FrameRange::len).sum()
    }

    /// Iterate over all frame numbers in insertion order.
    pub fn iter_frames(&self) -> impl Iterator<Item = i32> + '_ {
        self.ranges.iter().flat_map(FrameRange::iter)
    }

    /// Number of ranges in the list.
    #[must_use]
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Return `true` if this list contains no ranges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Flatten all ranges into a sorted, deduplicated list of frame numbers.
    #[must_use]
    pub fn unique_frames(&self) -> Vec<i32> {
        let mut frames: Vec<i32> = self.iter_frames().collect();
        frames.sort_unstable();
        frames.dedup();
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_range_len_basic() {
        assert_eq!(FrameRange::new(1, 10).len(), 10);
    }

    #[test]
    fn frame_range_len_single_frame() {
        assert_eq!(FrameRange::new(5, 5).len(), 1);
    }

    #[test]
    fn frame_range_empty_when_first_gt_last() {
        let r = FrameRange::new(10, 5);
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn frame_range_step_reduces_len() {
        let r = FrameRange::with_step(0, 10, 2);
        assert_eq!(r.len(), 6); // 0,2,4,6,8,10
    }

    #[test]
    fn frame_range_contains_in_range() {
        let r = FrameRange::new(1, 10);
        assert!(r.contains(5));
        assert!(!r.contains(0));
        assert!(!r.contains(11));
    }

    #[test]
    fn frame_range_contains_with_step() {
        let r = FrameRange::with_step(0, 10, 2);
        assert!(r.contains(4));
        assert!(!r.contains(3));
    }

    #[test]
    fn frame_range_iter_collects_all() {
        let frames: Vec<i32> = FrameRange::new(1, 5).iter().collect();
        assert_eq!(frames, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn frame_range_iter_with_step() {
        let frames: Vec<i32> = FrameRange::with_step(0, 6, 2).iter().collect();
        assert_eq!(frames, vec![0, 2, 4, 6]);
    }

    #[test]
    fn frame_range_chunks_splits_evenly() {
        let r = FrameRange::new(1, 6);
        let chunks = r.chunks(2);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn frame_range_chunks_zero_returns_empty() {
        let r = FrameRange::new(1, 5);
        assert!(r.chunks(0).is_empty());
    }

    #[test]
    fn frame_range_display_no_step() {
        assert_eq!(FrameRange::new(1, 10).to_string(), "1-10");
    }

    #[test]
    fn frame_range_display_with_step() {
        assert_eq!(FrameRange::with_step(0, 10, 2).to_string(), "0-10x2");
    }

    #[test]
    fn frame_range_list_total_frames() {
        let mut list = FrameRangeList::new();
        list.add(FrameRange::new(1, 10));
        list.add(FrameRange::new(50, 60));
        assert_eq!(list.total_frames(), 21);
    }

    #[test]
    fn frame_range_list_unique_frames_deduplicates() {
        let mut list = FrameRangeList::new();
        list.add(FrameRange::new(1, 5));
        list.add(FrameRange::new(3, 7));
        let u = list.unique_frames();
        assert_eq!(u, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn frame_range_list_is_empty() {
        let list = FrameRangeList::new();
        assert!(list.is_empty());
    }
}
