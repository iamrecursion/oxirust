//! Sequence range types for frame-accurate in/out point management.
//!
//! `SequenceRange` models a contiguous span of frames within a sequence,
//! with support for in-point variants (absolute or relative), overlap
//! detection, and ordered lists of ranges.

#![allow(dead_code)]

/// Describes the in-point of a sequence range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequenceIn {
    /// An absolute frame number in the parent sequence.
    Absolute(u64),
    /// A frame offset relative to the previous edit point (or sequence start).
    Relative(i64),
}

impl SequenceIn {
    /// Returns the frame offset this in-point represents.
    ///
    /// For `Absolute`, the offset is simply the frame number (as `i64`).
    /// For `Relative`, the stored signed offset is returned directly.
    #[must_use]
    pub fn frame_offset(&self) -> i64 {
        match self {
            Self::Absolute(n) => *n as i64,
            Self::Relative(off) => *off,
        }
    }

    /// Returns `true` if this is an absolute in-point.
    #[must_use]
    pub const fn is_absolute(&self) -> bool {
        matches!(self, Self::Absolute(_))
    }

    /// Returns `true` if this is a relative in-point.
    #[must_use]
    pub const fn is_relative(&self) -> bool {
        matches!(self, Self::Relative(_))
    }

    /// Resolves the in-point to an absolute frame number given a
    /// `reference` frame for relative offsets.
    ///
    /// Returns `None` if the resolved value would underflow below zero.
    #[must_use]
    pub fn resolve(&self, reference: u64) -> Option<u64> {
        match self {
            Self::Absolute(n) => Some(*n),
            Self::Relative(off) => {
                let resolved = reference as i64 + off;
                if resolved < 0 {
                    None
                } else {
                    Some(resolved as u64)
                }
            }
        }
    }
}

/// A contiguous, half-open range of frames `[start_frame, start_frame + length)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceRange {
    /// The first frame of this range (inclusive).
    pub start_frame: u64,
    /// The number of frames in this range.
    pub length: u64,
}

impl SequenceRange {
    /// Creates a new `SequenceRange`.
    ///
    /// # Panics
    /// Panics if `length` is zero.
    #[must_use]
    pub fn new(start_frame: u64, length: u64) -> Self {
        assert!(length > 0, "SequenceRange length must be non-zero");
        Self {
            start_frame,
            length,
        }
    }

    /// Returns the exclusive end frame (one past the last frame).
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.start_frame + self.length
    }

    /// Returns the duration of this range in frames (same as `length`).
    #[must_use]
    pub const fn duration_frames(&self) -> u64 {
        self.length
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_frame < other.end_frame() && other.start_frame < self.end_frame()
    }

    /// Returns `true` if `frame` falls within `[start_frame, end_frame)`.
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame()
    }

    /// Returns the intersection of this range with `other`, or `None` if
    /// they do not overlap.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start = self.start_frame.max(other.start_frame);
        let end = self.end_frame().min(other.end_frame());
        if end > start {
            Some(Self::new(start, end - start))
        } else {
            None
        }
    }
}

/// An ordered, non-overlapping list of `SequenceRange` entries.
///
/// Ranges are kept sorted by `start_frame`.  Overlapping ranges are rejected.
#[derive(Debug, Default, Clone)]
pub struct SequenceRangeList {
    ranges: Vec<SequenceRange>,
}

impl SequenceRangeList {
    /// Creates a new, empty `SequenceRangeList`.
    #[must_use]
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Attempts to add `range` to the list.
    ///
    /// Returns `Err(&str)` if `range` overlaps with any existing entry.
    ///
    /// # Errors
    /// Returns a static error message if the range overlaps an existing one.
    pub fn add(&mut self, range: SequenceRange) -> Result<(), &'static str> {
        for existing in &self.ranges {
            if existing.overlaps(&range) {
                return Err("SequenceRange overlaps an existing range");
            }
        }
        self.ranges.push(range);
        self.ranges.sort_by_key(|r| r.start_frame);
        Ok(())
    }

    /// Returns the total number of frames across all ranges.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.ranges.iter().map(|r| r.length).sum()
    }

    /// Returns the number of ranges in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Returns `true` if the list contains no ranges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Returns a reference to all ranges in start-frame order.
    #[must_use]
    pub fn ranges(&self) -> &[SequenceRange] {
        &self.ranges
    }

    /// Returns `true` if any range in the list contains `frame`.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        self.ranges.iter().any(|r| r.contains(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_in_absolute_offset() {
        let si = SequenceIn::Absolute(100);
        assert_eq!(si.frame_offset(), 100);
    }

    #[test]
    fn test_sequence_in_relative_offset() {
        let si = SequenceIn::Relative(-5);
        assert_eq!(si.frame_offset(), -5);
    }

    #[test]
    fn test_sequence_in_is_absolute() {
        assert!(SequenceIn::Absolute(0).is_absolute());
        assert!(!SequenceIn::Relative(0).is_absolute());
    }

    #[test]
    fn test_sequence_in_is_relative() {
        assert!(SequenceIn::Relative(0).is_relative());
        assert!(!SequenceIn::Absolute(0).is_relative());
    }

    #[test]
    fn test_sequence_in_resolve_absolute() {
        let si = SequenceIn::Absolute(42);
        assert_eq!(si.resolve(0), Some(42));
    }

    #[test]
    fn test_sequence_in_resolve_relative_positive() {
        let si = SequenceIn::Relative(10);
        assert_eq!(si.resolve(20), Some(30));
    }

    #[test]
    fn test_sequence_in_resolve_relative_underflow() {
        let si = SequenceIn::Relative(-50);
        assert_eq!(si.resolve(10), None);
    }

    #[test]
    fn test_sequence_range_end_frame() {
        let r = SequenceRange::new(10, 20);
        assert_eq!(r.end_frame(), 30);
    }

    #[test]
    fn test_sequence_range_duration_frames() {
        let r = SequenceRange::new(0, 100);
        assert_eq!(r.duration_frames(), 100);
    }

    #[test]
    fn test_sequence_range_overlaps_true() {
        let a = SequenceRange::new(0, 50);
        let b = SequenceRange::new(25, 50);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_sequence_range_overlaps_false() {
        let a = SequenceRange::new(0, 50);
        let b = SequenceRange::new(50, 50);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_sequence_range_contains() {
        let r = SequenceRange::new(10, 20);
        assert!(r.contains(10));
        assert!(r.contains(29));
        assert!(!r.contains(9));
        assert!(!r.contains(30));
    }

    #[test]
    fn test_sequence_range_intersection_some() {
        let a = SequenceRange::new(0, 50);
        let b = SequenceRange::new(25, 50);
        let inter = a.intersection(&b).expect("should succeed in test");
        assert_eq!(inter.start_frame, 25);
        assert_eq!(inter.length, 25);
    }

    #[test]
    fn test_sequence_range_intersection_none() {
        let a = SequenceRange::new(0, 25);
        let b = SequenceRange::new(50, 25);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_range_list_add_and_total() {
        let mut list = SequenceRangeList::new();
        list.add(SequenceRange::new(0, 10))
            .expect("should succeed in test");
        list.add(SequenceRange::new(10, 10))
            .expect("should succeed in test");
        assert_eq!(list.total_frames(), 20);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_range_list_add_overlap_rejected() {
        let mut list = SequenceRangeList::new();
        list.add(SequenceRange::new(0, 20))
            .expect("should succeed in test");
        let result = list.add(SequenceRange::new(10, 20));
        assert!(result.is_err());
    }

    #[test]
    fn test_range_list_sorted() {
        let mut list = SequenceRangeList::new();
        list.add(SequenceRange::new(100, 10))
            .expect("should succeed in test");
        list.add(SequenceRange::new(0, 10))
            .expect("should succeed in test");
        let ranges = list.ranges();
        assert_eq!(ranges[0].start_frame, 0);
        assert_eq!(ranges[1].start_frame, 100);
    }

    #[test]
    fn test_range_list_contains_frame() {
        let mut list = SequenceRangeList::new();
        list.add(SequenceRange::new(50, 10))
            .expect("should succeed in test");
        assert!(list.contains_frame(55));
        assert!(!list.contains_frame(60));
    }

    #[test]
    fn test_range_list_empty() {
        let list = SequenceRangeList::new();
        assert!(list.is_empty());
        assert_eq!(list.total_frames(), 0);
    }
}
