#![allow(dead_code)]
//! ISO Base Media File Format edit list (`elst`) abstraction.
//!
//! An edit list maps the presentation timeline of a track to its media
//! timeline, supporting trimming, offsets, dwell (still frame), and
//! rate-scaled segments. This module provides a high-level [`EditList`]
//! for building, querying, and applying edit lists.

/// Rate at which media plays during an edit segment.
///
/// Encoded as a 16.16 fixed-point value in the file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaRate {
    /// Playback rate (1.0 = normal, 0.0 = dwell / still frame).
    pub rate: f64,
}

impl MediaRate {
    /// Normal playback speed (1x).
    pub const NORMAL: Self = Self { rate: 1.0 };
    /// Dwell / still-frame rate.
    pub const DWELL: Self = Self { rate: 0.0 };

    /// Creates a rate from a floating-point value.
    #[must_use]
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    /// Returns `true` if this is a dwell (still frame) segment.
    #[must_use]
    pub fn is_dwell(self) -> bool {
        self.rate.abs() < f64::EPSILON
    }

    /// Returns `true` if this is normal 1x playback.
    #[must_use]
    pub fn is_normal(self) -> bool {
        (self.rate - 1.0).abs() < 1e-9
    }

    /// Converts to 16.16 fixed-point representation.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn to_fixed_point(self) -> i32 {
        (self.rate * 65536.0) as i32
    }

    /// Parses from 16.16 fixed-point representation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_fixed_point(fp: i32) -> Self {
        Self {
            rate: f64::from(fp) / 65536.0,
        }
    }
}

impl Default for MediaRate {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// A single entry in the edit list (`elst` box).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditListEntry {
    /// Segment duration in movie timescale units.
    pub segment_duration: u64,
    /// Media time (start time in the media timescale).
    /// A value of `-1` means an empty edit (gap).
    pub media_time: i64,
    /// Media rate for this segment.
    pub media_rate: MediaRate,
}

impl EditListEntry {
    /// Creates a normal-rate edit entry.
    #[must_use]
    pub fn normal(segment_duration: u64, media_time: i64) -> Self {
        Self {
            segment_duration,
            media_time,
            media_rate: MediaRate::NORMAL,
        }
    }

    /// Creates an empty edit (gap / delay).
    #[must_use]
    pub fn empty(segment_duration: u64) -> Self {
        Self {
            segment_duration,
            media_time: -1,
            media_rate: MediaRate::NORMAL,
        }
    }

    /// Creates a dwell (still-frame) edit.
    #[must_use]
    pub fn dwell(segment_duration: u64, media_time: i64) -> Self {
        Self {
            segment_duration,
            media_time,
            media_rate: MediaRate::DWELL,
        }
    }

    /// Returns `true` if this is an empty edit (gap).
    #[must_use]
    pub fn is_empty_edit(&self) -> bool {
        self.media_time < 0
    }

    /// Returns `true` if this is a dwell segment.
    #[must_use]
    pub fn is_dwell(&self) -> bool {
        self.media_rate.is_dwell() && !self.is_empty_edit()
    }
}

/// Outcome of mapping a presentation time to media time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EditMapping {
    /// Presentation time maps to this media time.
    Mapped {
        /// Media time in media timescale units.
        media_time: u64,
        /// Playback rate.
        rate: f64,
    },
    /// Presentation time falls in an empty edit (gap).
    Gap,
    /// Presentation time is beyond the edit list range.
    OutOfRange,
}

/// A complete edit list for a track.
#[derive(Debug, Clone)]
pub struct EditList {
    /// Ordered entries.
    entries: Vec<EditListEntry>,
    /// Movie timescale (ticks per second) for segment durations.
    movie_timescale: u32,
    /// Media timescale (ticks per second) for media times.
    media_timescale: u32,
}

impl EditList {
    /// Creates an empty edit list with the given timescales.
    #[must_use]
    pub fn new(movie_timescale: u32, media_timescale: u32) -> Self {
        Self {
            entries: Vec::new(),
            movie_timescale,
            media_timescale,
        }
    }

    /// Appends an entry to the edit list.
    pub fn push(&mut self, entry: EditListEntry) {
        self.entries.push(entry);
    }

    /// Returns a slice of all entries.
    #[must_use]
    pub fn entries(&self) -> &[EditListEntry] {
        &self.entries
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total presentation duration in movie timescale ticks.
    #[must_use]
    pub fn total_duration_ticks(&self) -> u64 {
        self.entries.iter().map(|e| e.segment_duration).sum()
    }

    /// Total presentation duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn total_duration_seconds(&self) -> f64 {
        if self.movie_timescale == 0 {
            return 0.0;
        }
        self.total_duration_ticks() as f64 / f64::from(self.movie_timescale)
    }

    /// Returns the initial offset (empty edit at the beginning) in movie
    /// timescale ticks, or 0 if the list doesn't start with an empty edit.
    #[must_use]
    pub fn initial_offset_ticks(&self) -> u64 {
        self.entries
            .first()
            .filter(|e| e.is_empty_edit())
            .map_or(0, |e| e.segment_duration)
    }

    /// Returns the initial offset in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn initial_offset_seconds(&self) -> f64 {
        if self.movie_timescale == 0 {
            return 0.0;
        }
        self.initial_offset_ticks() as f64 / f64::from(self.movie_timescale)
    }

    /// Maps a presentation time (in movie timescale ticks) to a media time.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn map_presentation_time(&self, presentation_ticks: u64) -> EditMapping {
        let mut offset: u64 = 0;

        for entry in &self.entries {
            let seg_end = offset + entry.segment_duration;
            if presentation_ticks < seg_end {
                if entry.is_empty_edit() {
                    return EditMapping::Gap;
                }
                let local = presentation_ticks - offset;
                // Convert from movie timescale to media timescale.
                let media_offset = if self.movie_timescale > 0 {
                    (local as f64 / f64::from(self.movie_timescale)
                        * f64::from(self.media_timescale)
                        * entry.media_rate.rate) as u64
                } else {
                    0
                };
                let media_time = (entry.media_time as u64) + media_offset;
                return EditMapping::Mapped {
                    media_time,
                    rate: entry.media_rate.rate,
                };
            }
            offset = seg_end;
        }

        EditMapping::OutOfRange
    }

    /// Returns `true` if the edit list is trivial (single entry at normal
    /// rate starting from media time 0).
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        if self.entries.len() != 1 {
            return false;
        }
        let e = &self.entries[0];
        e.media_time == 0 && e.media_rate.is_normal() && !e.is_empty_edit()
    }

    /// Returns `true` if any entry is a dwell (still frame).
    #[must_use]
    pub fn has_dwell(&self) -> bool {
        self.entries.iter().any(EditListEntry::is_dwell)
    }

    /// Returns `true` if any entry is an empty edit.
    #[must_use]
    pub fn has_empty_edit(&self) -> bool {
        self.entries.iter().any(EditListEntry::is_empty_edit)
    }

    /// Returns the movie timescale.
    #[must_use]
    pub const fn movie_timescale(&self) -> u32 {
        self.movie_timescale
    }

    /// Returns the media timescale.
    #[must_use]
    pub const fn media_timescale(&self) -> u32 {
        self.media_timescale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_rate_normal() {
        let r = MediaRate::NORMAL;
        assert!(r.is_normal());
        assert!(!r.is_dwell());
    }

    #[test]
    fn test_media_rate_dwell() {
        let r = MediaRate::DWELL;
        assert!(r.is_dwell());
        assert!(!r.is_normal());
    }

    #[test]
    fn test_media_rate_fixed_point_round_trip() {
        let r = MediaRate::new(1.5);
        let fp = r.to_fixed_point();
        let r2 = MediaRate::from_fixed_point(fp);
        assert!((r2.rate - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_entry_normal() {
        let e = EditListEntry::normal(1000, 500);
        assert!(!e.is_empty_edit());
        assert!(!e.is_dwell());
        assert_eq!(e.segment_duration, 1000);
        assert_eq!(e.media_time, 500);
    }

    #[test]
    fn test_entry_empty() {
        let e = EditListEntry::empty(2000);
        assert!(e.is_empty_edit());
        assert!(!e.is_dwell());
    }

    #[test]
    fn test_entry_dwell() {
        let e = EditListEntry::dwell(500, 100);
        assert!(e.is_dwell());
        assert!(!e.is_empty_edit());
    }

    #[test]
    fn test_edit_list_empty() {
        let el = EditList::new(90000, 48000);
        assert!(el.is_empty());
        assert_eq!(el.len(), 0);
        assert_eq!(el.total_duration_ticks(), 0);
    }

    #[test]
    fn test_edit_list_single_normal() {
        let mut el = EditList::new(90000, 90000);
        el.push(EditListEntry::normal(90000, 0));
        assert_eq!(el.len(), 1);
        assert!(el.is_trivial());
        assert!((el.total_duration_seconds() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_is_trivial_false_for_offset() {
        let mut el = EditList::new(90000, 90000);
        el.push(EditListEntry::normal(90000, 1000));
        assert!(!el.is_trivial()); // media_time != 0
    }

    #[test]
    fn test_initial_offset() {
        let mut el = EditList::new(90000, 90000);
        el.push(EditListEntry::empty(45000));
        el.push(EditListEntry::normal(90000, 0));

        assert_eq!(el.initial_offset_ticks(), 45000);
        assert!((el.initial_offset_seconds() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_map_gap() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::empty(500));
        el.push(EditListEntry::normal(1000, 0));

        let result = el.map_presentation_time(250);
        assert_eq!(result, EditMapping::Gap);
    }

    #[test]
    fn test_map_normal() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::normal(2000, 100));

        if let EditMapping::Mapped { media_time, rate } = el.map_presentation_time(500) {
            assert_eq!(media_time, 600); // 100 + 500
            assert!((rate - 1.0).abs() < f64::EPSILON);
        } else {
            panic!("Expected Mapped");
        }
    }

    #[test]
    fn test_map_out_of_range() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::normal(100, 0));
        assert_eq!(el.map_presentation_time(200), EditMapping::OutOfRange);
    }

    #[test]
    fn test_has_dwell() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::dwell(500, 0));
        assert!(el.has_dwell());
    }

    #[test]
    fn test_has_empty_edit() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::empty(300));
        assert!(el.has_empty_edit());
    }

    #[test]
    fn test_total_duration_zero_timescale() {
        let el = EditList::new(0, 0);
        assert!((el.total_duration_seconds()).abs() < f64::EPSILON);
        assert!((el.initial_offset_seconds()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entries_accessor() {
        let mut el = EditList::new(1000, 1000);
        el.push(EditListEntry::normal(1000, 0));
        el.push(EditListEntry::empty(500));
        assert_eq!(el.entries().len(), 2);
    }
}
