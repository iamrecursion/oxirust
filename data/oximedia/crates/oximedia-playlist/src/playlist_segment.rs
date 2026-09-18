#![allow(dead_code)]
//! Playlist segmentation into time-bounded blocks.
//!
//! Provides [`SegmentBoundary`], [`PlaylistSegment`], and [`SegmentSplitter`]
//! for breaking a long playlist into logical time blocks (hours, shows, etc.).

use std::time::Duration;

// ---------------------------------------------------------------------------
// Segment boundary
// ---------------------------------------------------------------------------

/// Defines where a segment boundary should be placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentBoundary {
    /// Fixed time offset from the start of the playlist.
    AtOffset(Duration),
    /// After every N items.
    EveryNItems(usize),
    /// Whenever the cumulative duration reaches a multiple of this value.
    EveryDuration(Duration),
}

// ---------------------------------------------------------------------------
// Segment item (lightweight reference)
// ---------------------------------------------------------------------------

/// A lightweight representation of an item inside a segment.
#[derive(Debug, Clone)]
pub struct SegmentItem {
    /// Original index in the source playlist.
    pub source_index: usize,
    /// URI or path to the media file.
    pub uri: String,
    /// Duration of this item, if known.
    pub duration: Option<Duration>,
}

impl SegmentItem {
    /// Creates a new segment item.
    pub fn new(source_index: usize, uri: impl Into<String>) -> Self {
        Self {
            source_index,
            uri: uri.into(),
            duration: None,
        }
    }

    /// Attaches a duration.
    pub fn with_duration(mut self, dur: Duration) -> Self {
        self.duration = Some(dur);
        self
    }
}

// ---------------------------------------------------------------------------
// PlaylistSegment
// ---------------------------------------------------------------------------

/// A contiguous slice of a playlist with its own time budget.
#[derive(Debug, Clone)]
pub struct PlaylistSegment {
    /// Zero-based segment index.
    pub index: usize,
    /// Optional label (e.g., "Hour 1", "News Block").
    pub label: Option<String>,
    /// Items belonging to this segment.
    pub items: Vec<SegmentItem>,
    /// Maximum allowed duration for this segment.
    pub max_duration: Option<Duration>,
}

impl PlaylistSegment {
    /// Creates a new unlabelled segment.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            label: None,
            items: Vec::new(),
            max_duration: None,
        }
    }

    /// Attaches a label to the segment.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the maximum duration budget.
    pub fn with_max_duration(mut self, dur: Duration) -> Self {
        self.max_duration = Some(dur);
        self
    }

    /// Adds an item to this segment.
    pub fn add_item(&mut self, item: SegmentItem) {
        self.items.push(item);
    }

    /// Returns the total duration of all items that have durations.
    pub fn total_duration(&self) -> Duration {
        self.items
            .iter()
            .filter_map(|i| i.duration)
            .fold(Duration::ZERO, |acc, d| acc + d)
    }

    /// Returns the number of items in the segment.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns `true` if the total duration exceeds the budget (if set).
    pub fn is_over_budget(&self) -> bool {
        if let Some(max) = self.max_duration {
            self.total_duration() > max
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentSplitter
// ---------------------------------------------------------------------------

/// Splits a flat list of items into [`PlaylistSegment`]s according to a
/// [`SegmentBoundary`] rule.
#[derive(Debug, Clone)]
pub struct SegmentSplitter {
    boundary: SegmentBoundary,
}

impl SegmentSplitter {
    /// Creates a splitter with the given boundary rule.
    pub fn new(boundary: SegmentBoundary) -> Self {
        Self { boundary }
    }

    /// Splits `items` into segments according to the configured boundary.
    ///
    /// Items are cloned into the segments.
    pub fn split(&self, items: &[SegmentItem]) -> Vec<PlaylistSegment> {
        match &self.boundary {
            SegmentBoundary::EveryNItems(n) => self.split_by_count(items, *n),
            SegmentBoundary::EveryDuration(budget) => self.split_by_duration(items, *budget),
            SegmentBoundary::AtOffset(_) => self.split_by_offset(items),
        }
    }

    fn split_by_count(&self, items: &[SegmentItem], n: usize) -> Vec<PlaylistSegment> {
        if n == 0 {
            return Vec::new();
        }
        items
            .chunks(n)
            .enumerate()
            .map(|(seg_idx, chunk)| {
                let mut seg = PlaylistSegment::new(seg_idx);
                for item in chunk {
                    seg.add_item(item.clone());
                }
                seg
            })
            .collect()
    }

    fn split_by_duration(&self, items: &[SegmentItem], budget: Duration) -> Vec<PlaylistSegment> {
        let mut segments: Vec<PlaylistSegment> = Vec::new();
        let mut current = PlaylistSegment::new(0).with_max_duration(budget);
        let mut seg_idx = 0usize;

        for item in items {
            let item_dur = item.duration.unwrap_or(Duration::ZERO);
            if !current.is_empty() && current.total_duration() + item_dur > budget {
                segments.push(current);
                seg_idx += 1;
                current = PlaylistSegment::new(seg_idx).with_max_duration(budget);
            }
            current.add_item(item.clone());
        }
        if !current.is_empty() {
            segments.push(current);
        }
        segments
    }

    fn split_by_offset(&self, items: &[SegmentItem]) -> Vec<PlaylistSegment> {
        // For AtOffset: everything before the offset is segment 0,
        // everything at/after is segment 1.
        let offset = match &self.boundary {
            SegmentBoundary::AtOffset(d) => *d,
            _ => unreachable!(),
        };

        let mut seg0 = PlaylistSegment::new(0);
        let mut seg1 = PlaylistSegment::new(1);
        let mut elapsed = Duration::ZERO;

        for item in items {
            if elapsed < offset {
                seg0.add_item(item.clone());
            } else {
                seg1.add_item(item.clone());
            }
            elapsed += item.duration.unwrap_or(Duration::ZERO);
        }

        let mut out = vec![seg0];
        if !seg1.is_empty() {
            out.push(seg1);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn items_with_durations(durations_secs: &[u64]) -> Vec<SegmentItem> {
        durations_secs
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                SegmentItem::new(i, format!("item_{}.mp4", i)).with_duration(Duration::from_secs(d))
            })
            .collect()
    }

    #[test]
    fn test_split_every_n_items_even() {
        let items = items_with_durations(&[10, 20, 30, 40]);
        let splitter = SegmentSplitter::new(SegmentBoundary::EveryNItems(2));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].item_count(), 2);
        assert_eq!(segs[1].item_count(), 2);
    }

    #[test]
    fn test_split_every_n_items_odd() {
        let items = items_with_durations(&[10, 20, 30]);
        let splitter = SegmentSplitter::new(SegmentBoundary::EveryNItems(2));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].item_count(), 1);
    }

    #[test]
    fn test_split_every_n_zero_returns_empty() {
        let items = items_with_durations(&[10, 20]);
        let splitter = SegmentSplitter::new(SegmentBoundary::EveryNItems(0));
        assert!(splitter.split(&items).is_empty());
    }

    #[test]
    fn test_split_by_duration_basic() {
        // Each item is 60 s; budget is 120 s → 2 items per segment
        let items = items_with_durations(&[60, 60, 60, 60]);
        let splitter =
            SegmentSplitter::new(SegmentBoundary::EveryDuration(Duration::from_secs(120)));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn test_split_by_duration_single_large_item() {
        // A single item larger than budget still goes into one segment.
        let items = items_with_durations(&[300]);
        let splitter =
            SegmentSplitter::new(SegmentBoundary::EveryDuration(Duration::from_secs(60)));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_segment_total_duration() {
        let mut seg = PlaylistSegment::new(0);
        seg.add_item(SegmentItem::new(0, "a.mp4").with_duration(Duration::from_secs(30)));
        seg.add_item(SegmentItem::new(1, "b.mp4").with_duration(Duration::from_secs(70)));
        assert_eq!(seg.total_duration(), Duration::from_secs(100));
    }

    #[test]
    fn test_segment_is_over_budget() {
        let mut seg = PlaylistSegment::new(0).with_max_duration(Duration::from_secs(60));
        seg.add_item(SegmentItem::new(0, "a.mp4").with_duration(Duration::from_secs(90)));
        assert!(seg.is_over_budget());
    }

    #[test]
    fn test_segment_not_over_budget() {
        let mut seg = PlaylistSegment::new(0).with_max_duration(Duration::from_secs(120));
        seg.add_item(SegmentItem::new(0, "a.mp4").with_duration(Duration::from_secs(60)));
        assert!(!seg.is_over_budget());
    }

    #[test]
    fn test_segment_empty_is_empty() {
        let seg = PlaylistSegment::new(0);
        assert!(seg.is_empty());
    }

    #[test]
    fn test_segment_label() {
        let seg = PlaylistSegment::new(0).with_label("Hour 1");
        assert_eq!(seg.label.as_deref(), Some("Hour 1"));
    }

    #[test]
    fn test_split_at_offset() {
        // Items: 30s, 30s, 30s, 30s — offset at 60s
        let items = items_with_durations(&[30, 30, 30, 30]);
        let splitter = SegmentSplitter::new(SegmentBoundary::AtOffset(Duration::from_secs(60)));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].item_count(), 2);
        assert_eq!(segs[1].item_count(), 2);
    }

    #[test]
    fn test_split_at_offset_all_before() {
        let items = items_with_durations(&[10, 10]);
        // Offset beyond all content → only segment 0
        let splitter = SegmentSplitter::new(SegmentBoundary::AtOffset(Duration::from_secs(9999)));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_segment_item_source_index() {
        let item = SegmentItem::new(42, "track.mp4");
        assert_eq!(item.source_index, 42);
    }

    #[test]
    fn test_segment_no_max_duration_not_over_budget() {
        let mut seg = PlaylistSegment::new(0);
        seg.add_item(SegmentItem::new(0, "x.mp4").with_duration(Duration::from_secs(999)));
        assert!(!seg.is_over_budget());
    }

    #[test]
    fn test_split_empty_input() {
        let items: Vec<SegmentItem> = vec![];
        let splitter = SegmentSplitter::new(SegmentBoundary::EveryNItems(3));
        assert!(splitter.split(&items).is_empty());
    }

    #[test]
    fn test_split_by_duration_three_segments() {
        // 6 items × 30 s each, budget = 60 s → 3 segments
        let items = items_with_durations(&[30, 30, 30, 30, 30, 30]);
        let splitter =
            SegmentSplitter::new(SegmentBoundary::EveryDuration(Duration::from_secs(60)));
        let segs = splitter.split(&items);
        assert_eq!(segs.len(), 3);
    }
}
