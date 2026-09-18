//! Gap detection and filling utilities for the timeline.
//!
//! `GapFiller` scans a track's clip ranges and either reports gaps or
//! automatically fills them with a user-specified strategy.

#![allow(dead_code)]

/// A gap between two adjacent clips on a track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    /// Track on which the gap was found.
    pub track_id: u32,
    /// First frame of the gap (inclusive).
    pub start_frame: u64,
    /// First frame *after* the gap (exclusive).
    pub end_frame: u64,
}

impl Gap {
    /// Duration of the gap in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when the gap is shorter than `max_frames`.
    #[must_use]
    pub fn is_short(&self, max_frames: u64) -> bool {
        self.duration_frames() < max_frames
    }
}

/// Strategy for filling detected gaps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillStrategy {
    /// Leave gaps unfilled — only detect them.
    ReportOnly,
    /// Fill with black (or silence for audio).
    Black,
    /// Extend the preceding clip to cover the gap.
    ExtendPrevious,
    /// Extend the following clip backwards to cover the gap.
    ExtendNext,
}

impl FillStrategy {
    /// Returns `true` when this strategy actively modifies the timeline.
    #[must_use]
    pub fn modifies_timeline(&self) -> bool {
        !matches!(self, Self::ReportOnly)
    }
}

/// A clip range used as input to gap analysis (start frame, end frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClipRange {
    /// Inclusive start frame.
    pub start: u64,
    /// Exclusive end frame.
    pub end: u64,
}

impl ClipRange {
    /// Create a new range.  Panics in debug if `start >= end`.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        debug_assert!(start < end, "ClipRange: start must be less than end");
        Self { start, end }
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Scans a sorted list of clip ranges and finds gaps between them.
#[derive(Debug, Default)]
pub struct GapFiller {
    /// Only gaps longer than this many frames are reported.
    pub min_gap_frames: u64,
}

impl GapFiller {
    /// Create a `GapFiller` with a minimum gap threshold.
    #[must_use]
    pub fn new(min_gap_frames: u64) -> Self {
        Self { min_gap_frames }
    }

    /// Detect all gaps on a single track.
    ///
    /// `ranges` need not be pre-sorted; the method sorts them internally.
    #[must_use]
    pub fn find_gaps(&self, track_id: u32, ranges: &[ClipRange]) -> Vec<Gap> {
        if ranges.len() < 2 {
            return Vec::new();
        }

        let mut sorted = ranges.to_vec();
        sorted.sort_unstable();

        let mut gaps = Vec::new();
        for window in sorted.windows(2) {
            let a = &window[0];
            let b = &window[1];
            if b.start > a.end {
                let gap_len = b.start - a.end;
                if gap_len > self.min_gap_frames {
                    gaps.push(Gap {
                        track_id,
                        start_frame: a.end,
                        end_frame: b.start,
                    });
                }
            }
        }
        gaps
    }

    /// Detect overlaps on a single track.  Returns pairs of overlapping ranges.
    #[must_use]
    pub fn find_overlaps(&self, ranges: &[ClipRange]) -> Vec<(ClipRange, ClipRange)> {
        let mut sorted = ranges.to_vec();
        sorted.sort_unstable();
        let mut overlaps = Vec::new();
        for window in sorted.windows(2) {
            if window[0].overlaps(&window[1]) {
                overlaps.push((window[0], window[1]));
            }
        }
        overlaps
    }

    /// Returns `true` when there are no gaps on the track (above threshold).
    #[must_use]
    pub fn is_gapless(&self, track_id: u32, ranges: &[ClipRange]) -> bool {
        self.find_gaps(track_id, ranges).is_empty()
    }

    /// Total number of frames covered by gaps.
    #[must_use]
    pub fn total_gap_frames(&self, track_id: u32, ranges: &[ClipRange]) -> u64 {
        self.find_gaps(track_id, ranges)
            .iter()
            .map(Gap::duration_frames)
            .sum()
    }
}

/// Summary report produced after analysing one or more tracks.
#[derive(Debug, Default)]
pub struct GapReport {
    /// All detected gaps across all tracks.
    pub gaps: Vec<Gap>,
}

impl GapReport {
    /// Add gaps from another track.
    pub fn extend(&mut self, more: impl IntoIterator<Item = Gap>) {
        self.gaps.extend(more);
    }

    /// Gaps on a specific track.
    #[must_use]
    pub fn gaps_on_track(&self, track_id: u32) -> Vec<&Gap> {
        self.gaps
            .iter()
            .filter(|g| g.track_id == track_id)
            .collect()
    }

    /// Total gap duration across all tracks.
    #[must_use]
    pub fn total_gap_frames(&self) -> u64 {
        self.gaps.iter().map(Gap::duration_frames).sum()
    }

    /// Returns `true` when no gaps were found.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.gaps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(s: u64, e: u64) -> ClipRange {
        ClipRange::new(s, e)
    }

    #[test]
    fn gap_duration() {
        let g = Gap {
            track_id: 0,
            start_frame: 10,
            end_frame: 20,
        };
        assert_eq!(g.duration_frames(), 10);
    }

    #[test]
    fn gap_is_short() {
        let g = Gap {
            track_id: 0,
            start_frame: 0,
            end_frame: 5,
        };
        assert!(g.is_short(10));
        assert!(!g.is_short(4));
    }

    #[test]
    fn fill_strategy_modifies_timeline() {
        assert!(!FillStrategy::ReportOnly.modifies_timeline());
        assert!(FillStrategy::Black.modifies_timeline());
        assert!(FillStrategy::ExtendPrevious.modifies_timeline());
        assert!(FillStrategy::ExtendNext.modifies_timeline());
    }

    #[test]
    fn clip_range_duration() {
        let r = ClipRange::new(5, 15);
        assert_eq!(r.duration(), 10);
    }

    #[test]
    fn clip_range_overlaps() {
        assert!(r(0, 10).overlaps(&r(5, 15)));
        assert!(!r(0, 5).overlaps(&r(5, 10)));
    }

    #[test]
    fn find_gaps_basic() {
        let filler = GapFiller::new(0);
        let ranges = [r(0, 10), r(15, 25)];
        let gaps = filler.find_gaps(0, &ranges);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 10);
        assert_eq!(gaps[0].end_frame, 15);
    }

    #[test]
    fn find_gaps_min_threshold_filters_small_gaps() {
        let filler = GapFiller::new(5);
        let ranges = [r(0, 10), r(13, 25)]; // gap of 3 frames
        let gaps = filler.find_gaps(0, &ranges);
        assert!(gaps.is_empty());
    }

    #[test]
    fn find_gaps_unsorted_input() {
        let filler = GapFiller::new(0);
        let ranges = [r(15, 25), r(0, 10)]; // reversed
        let gaps = filler.find_gaps(0, &ranges);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 10);
    }

    #[test]
    fn find_gaps_no_gap_when_contiguous() {
        let filler = GapFiller::new(0);
        let ranges = [r(0, 10), r(10, 20)];
        assert!(filler.find_gaps(0, &ranges).is_empty());
    }

    #[test]
    fn find_overlaps_basic() {
        let filler = GapFiller::new(0);
        let ranges = [r(0, 15), r(10, 25)];
        let overlaps = filler.find_overlaps(&ranges);
        assert_eq!(overlaps.len(), 1);
    }

    #[test]
    fn find_overlaps_no_overlap() {
        let filler = GapFiller::new(0);
        let ranges = [r(0, 10), r(10, 20)];
        assert!(filler.find_overlaps(&ranges).is_empty());
    }

    #[test]
    fn is_gapless_true_when_contiguous() {
        let filler = GapFiller::new(0);
        assert!(filler.is_gapless(0, &[r(0, 10), r(10, 20)]));
    }

    #[test]
    fn total_gap_frames() {
        let filler = GapFiller::new(0);
        let ranges = [r(0, 10), r(15, 25), r(30, 40)];
        // gaps: 10-15 (5 frames), 25-30 (5 frames) → total 10
        assert_eq!(filler.total_gap_frames(0, &ranges), 10);
    }

    #[test]
    fn gap_report_gaps_on_track() {
        let mut report = GapReport::default();
        report.extend([
            Gap {
                track_id: 0,
                start_frame: 5,
                end_frame: 10,
            },
            Gap {
                track_id: 1,
                start_frame: 20,
                end_frame: 30,
            },
        ]);
        assert_eq!(report.gaps_on_track(0).len(), 1);
        assert_eq!(report.gaps_on_track(1).len(), 1);
        assert_eq!(report.gaps_on_track(2).len(), 0);
    }

    #[test]
    fn gap_report_is_clean() {
        let report = GapReport::default();
        assert!(report.is_clean());
    }
}
