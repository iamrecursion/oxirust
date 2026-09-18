//! Selection management for timeline editing.
//!
//! Provides range-based and clip-based selection with multiple selection modes.

#![allow(dead_code)]

/// A time range on the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionRange {
    /// Inclusive start of the range (in timebase units).
    pub start: u64,
    /// Exclusive end of the range (in timebase units).
    pub end: u64,
}

impl SelectionRange {
    /// Creates a new `SelectionRange`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `end < start`.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        debug_assert!(end >= start, "SelectionRange: end must be >= start");
        Self { start, end }
    }

    /// Returns the duration of this range.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Returns `true` if this range is adjacent to or overlapping `other`
    /// (i.e., they can be merged into one).
    #[must_use]
    pub fn can_merge(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Merges two ranges into their union.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Returns `true` if `t` falls within `[start, end)`.
    #[must_use]
    pub fn contains_time(&self, t: u64) -> bool {
        t >= self.start && t < self.end
    }
}

/// Determines how a new selection interacts with the existing selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Replace the entire current selection with the new one.
    Replace,
    /// Add the new range/item to the existing selection (union).
    Add,
    /// Remove the new range/item from the existing selection (difference).
    Subtract,
    /// Toggle the new range/item (add if not selected, remove if selected).
    Toggle,
}

/// A lightweight reference to a clip on the timeline, used for range-based
/// selection queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineClipRef {
    /// Unique clip identifier.
    pub id: u64,
    /// Timeline start position (inclusive).
    pub start: u64,
    /// Timeline end position (exclusive).
    pub end: u64,
}

impl TimelineClipRef {
    /// Creates a new `TimelineClipRef`.
    #[must_use]
    pub const fn new(id: u64, start: u64, end: u64) -> Self {
        Self { id, start, end }
    }

    /// Returns `true` if this clip overlaps with `range`.
    #[must_use]
    pub fn overlaps_range(&self, range: &SelectionRange) -> bool {
        self.start < range.end && range.start < self.end
    }
}

/// Manages the current selection state for timeline editing.
///
/// A `Selection` tracks three orthogonal selection types:
/// * **Ranges** – time spans on the timeline ruler.
/// * **Clips** – individual clip IDs.
/// * **Tracks** – track indices.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Time ranges currently selected (kept non-overlapping and sorted).
    ranges: Vec<SelectionRange>,
    /// IDs of selected clips.
    selected_clips: Vec<u64>,
    /// Indices of selected tracks.
    selected_tracks: Vec<u32>,
}

impl Selection {
    /// Creates a new, empty `Selection`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // Range operations
    // ------------------------------------------------------------------

    /// Applies `range` to the time-range selection using `mode`.
    pub fn select_range(&mut self, range: SelectionRange, mode: SelectionMode) {
        match mode {
            SelectionMode::Replace => {
                self.ranges.clear();
                if range.duration() > 0 {
                    self.ranges.push(range);
                }
            }
            SelectionMode::Add => {
                if range.duration() > 0 {
                    self.ranges.push(range);
                    self.merge_overlapping_ranges();
                }
            }
            SelectionMode::Subtract => {
                self.subtract_range(range);
            }
            SelectionMode::Toggle => {
                if self.ranges.iter().any(|r| r.overlaps(&range)) {
                    self.subtract_range(range);
                } else {
                    self.ranges.push(range);
                    self.merge_overlapping_ranges();
                }
            }
        }
    }

    /// Subtracts a range from the current selection.
    fn subtract_range(&mut self, sub: SelectionRange) {
        let mut result = Vec::new();
        for r in self.ranges.drain(..) {
            if r.end <= sub.start || r.start >= sub.end {
                // No overlap – keep as-is.
                result.push(r);
            } else {
                // Left fragment.
                if r.start < sub.start {
                    result.push(SelectionRange::new(r.start, sub.start));
                }
                // Right fragment.
                if r.end > sub.end {
                    result.push(SelectionRange::new(sub.end, r.end));
                }
            }
        }
        self.ranges = result;
    }

    /// Merges all overlapping or adjacent ranges in `self.ranges`.
    pub fn merge_overlapping_ranges(&mut self) {
        if self.ranges.len() < 2 {
            return;
        }
        self.ranges.sort_by_key(|r| r.start);
        let mut merged: Vec<SelectionRange> = Vec::new();
        for r in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.can_merge(&r) {
                    *last = last.merge(&r);
                    continue;
                }
            }
            merged.push(r);
        }
        self.ranges = merged;
    }

    /// Returns the total duration covered by all selected ranges.
    #[must_use]
    pub fn selected_duration(&self) -> u64 {
        self.ranges.iter().map(SelectionRange::duration).sum()
    }

    /// Returns a slice of the currently selected ranges.
    #[must_use]
    pub fn ranges(&self) -> &[SelectionRange] {
        &self.ranges
    }

    // ------------------------------------------------------------------
    // Clip operations
    // ------------------------------------------------------------------

    /// Applies `id` to the clip selection using `mode`.
    pub fn select_clip(&mut self, id: u64, mode: SelectionMode) {
        match mode {
            SelectionMode::Replace => {
                self.selected_clips.clear();
                self.selected_clips.push(id);
            }
            SelectionMode::Add => {
                if !self.selected_clips.contains(&id) {
                    self.selected_clips.push(id);
                }
            }
            SelectionMode::Subtract => {
                self.selected_clips.retain(|&c| c != id);
            }
            SelectionMode::Toggle => {
                if self.selected_clips.contains(&id) {
                    self.selected_clips.retain(|&c| c != id);
                } else {
                    self.selected_clips.push(id);
                }
            }
        }
    }

    /// Returns `true` if the clip with `id` is currently selected.
    #[must_use]
    pub fn is_clip_selected(&self, id: u64) -> bool {
        self.selected_clips.contains(&id)
    }

    /// Returns a slice of all currently selected clip IDs.
    #[must_use]
    pub fn selected_clips(&self) -> &[u64] {
        &self.selected_clips
    }

    /// Selects all clips from `clips` whose time span overlaps `range`.
    pub fn select_all_in_range(&mut self, clips: &[TimelineClipRef], range: SelectionRange) {
        for clip in clips {
            if clip.overlaps_range(&range) && !self.selected_clips.contains(&clip.id) {
                self.selected_clips.push(clip.id);
            }
        }
    }

    // ------------------------------------------------------------------
    // Track operations
    // ------------------------------------------------------------------

    /// Selects the track with the given index using `mode`.
    pub fn select_track(&mut self, track: u32, mode: SelectionMode) {
        match mode {
            SelectionMode::Replace => {
                self.selected_tracks.clear();
                self.selected_tracks.push(track);
            }
            SelectionMode::Add => {
                if !self.selected_tracks.contains(&track) {
                    self.selected_tracks.push(track);
                }
            }
            SelectionMode::Subtract => {
                self.selected_tracks.retain(|&t| t != track);
            }
            SelectionMode::Toggle => {
                if self.selected_tracks.contains(&track) {
                    self.selected_tracks.retain(|&t| t != track);
                } else {
                    self.selected_tracks.push(track);
                }
            }
        }
    }

    /// Returns `true` if the track with `index` is currently selected.
    #[must_use]
    pub fn is_track_selected(&self, index: u32) -> bool {
        self.selected_tracks.contains(&index)
    }

    /// Returns a slice of all currently selected track indices.
    #[must_use]
    pub fn selected_tracks(&self) -> &[u32] {
        &self.selected_tracks
    }

    // ------------------------------------------------------------------
    // General
    // ------------------------------------------------------------------

    /// Clears all selection state (ranges, clips, and tracks).
    pub fn clear(&mut self) {
        self.ranges.clear();
        self.selected_clips.clear();
        self.selected_tracks.clear();
    }

    /// Returns `true` if nothing at all is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.selected_clips.is_empty() && self.selected_tracks.is_empty()
    }
}

/// A reference to a clip on a specific track, used in multi-track selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionItem {
    /// Clip identifier.
    pub clip_id: u64,
    /// Track identifier.
    pub track_id: u32,
}

impl SelectionItem {
    /// Create a new selection item.
    #[must_use]
    pub fn new(clip_id: u64, track_id: u32) -> Self {
        Self { clip_id, track_id }
    }

    /// Returns `true` when both items reside on the same track.
    #[must_use]
    pub fn same_track(&self, other: &SelectionItem) -> bool {
        self.track_id == other.track_id
    }
}

/// A multi-track clip selection backed by a flat list of [`SelectionItem`]s.
#[derive(Debug, Clone, Default)]
pub struct EditSelection {
    /// Items currently selected.
    pub items: Vec<SelectionItem>,
}

impl EditSelection {
    /// Create an empty selection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item; silently ignored if a clip with the same `clip_id` is
    /// already present.
    pub fn add(&mut self, item: SelectionItem) {
        if !self.contains(item.clip_id) {
            self.items.push(item);
        }
    }

    /// Remove the item whose `clip_id` matches.
    pub fn remove(&mut self, clip_id: u64) {
        self.items.retain(|i| i.clip_id != clip_id);
    }

    /// Returns `true` if a clip with the given ID is selected.
    #[must_use]
    pub fn contains(&self, clip_id: u64) -> bool {
        self.items.iter().any(|i| i.clip_id == clip_id)
    }

    /// Clear all selected items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Number of selected items.
    #[must_use]
    pub fn count(&self) -> usize {
        self.items.len()
    }

    /// Returns the unique track IDs present in the selection (order unspecified).
    #[must_use]
    pub fn tracks(&self) -> Vec<u32> {
        let mut seen: Vec<u32> = Vec::new();
        for item in &self.items {
            if !seen.contains(&item.track_id) {
                seen.push(item.track_id);
            }
        }
        seen
    }
}

/// Manages groups of clips that are linked together so that selecting one
/// automatically includes its linked peers.
#[derive(Debug, Clone, Default)]
pub struct LinkedSelection {
    /// Each inner `Vec<u64>` is a group of linked clip IDs.
    pub groups: Vec<Vec<u64>>,
}

impl LinkedSelection {
    /// Create an empty linked-selection manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new group of linked clips.
    pub fn add_linked_group(&mut self, ids: Vec<u64>) {
        if !ids.is_empty() {
            self.groups.push(ids);
        }
    }

    /// Return all clip IDs that are linked to `clip_id` (not including
    /// `clip_id` itself).  Returns an empty `Vec` when the clip has no links.
    #[must_use]
    pub fn linked_clips(&self, clip_id: u64) -> Vec<u64> {
        for group in &self.groups {
            if group.contains(&clip_id) {
                return group.iter().copied().filter(|&id| id != clip_id).collect();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // SelectionRange tests
    // ------------------------------------------------------------------

    #[test]
    fn test_range_duration() {
        let r = SelectionRange::new(10, 30);
        assert_eq!(r.duration(), 20);
    }

    #[test]
    fn test_range_overlaps() {
        let a = SelectionRange::new(0, 10);
        let b = SelectionRange::new(5, 15);
        let c = SelectionRange::new(10, 20);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c)); // touching but not overlapping
    }

    #[test]
    fn test_range_merge() {
        let a = SelectionRange::new(0, 10);
        let b = SelectionRange::new(8, 20);
        let m = a.merge(&b);
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 20);
    }

    #[test]
    fn test_range_contains_time() {
        let r = SelectionRange::new(10, 20);
        assert!(r.contains_time(10));
        assert!(r.contains_time(15));
        assert!(!r.contains_time(20)); // exclusive end
    }

    // ------------------------------------------------------------------
    // Selection – range mode tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_range_replace() {
        let mut s = Selection::new();
        s.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
        s.select_range(SelectionRange::new(200, 300), SelectionMode::Replace);
        assert_eq!(s.ranges().len(), 1);
        assert_eq!(s.ranges()[0], SelectionRange::new(200, 300));
    }

    #[test]
    fn test_select_range_add_merges() {
        let mut s = Selection::new();
        s.select_range(SelectionRange::new(0, 50), SelectionMode::Add);
        s.select_range(SelectionRange::new(40, 100), SelectionMode::Add);
        assert_eq!(s.ranges().len(), 1);
        assert_eq!(s.ranges()[0].end, 100);
    }

    #[test]
    fn test_select_range_subtract() {
        let mut s = Selection::new();
        s.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
        s.select_range(SelectionRange::new(40, 60), SelectionMode::Subtract);
        assert_eq!(s.ranges().len(), 2);
        assert_eq!(s.ranges()[0], SelectionRange::new(0, 40));
        assert_eq!(s.ranges()[1], SelectionRange::new(60, 100));
    }

    #[test]
    fn test_selected_duration() {
        let mut s = Selection::new();
        s.select_range(SelectionRange::new(0, 50), SelectionMode::Add);
        s.select_range(SelectionRange::new(100, 150), SelectionMode::Add);
        assert_eq!(s.selected_duration(), 100);
    }

    // ------------------------------------------------------------------
    // Selection – clip mode tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_clip_replace() {
        let mut s = Selection::new();
        s.select_clip(1, SelectionMode::Add);
        s.select_clip(2, SelectionMode::Add);
        s.select_clip(3, SelectionMode::Replace);
        assert_eq!(s.selected_clips().len(), 1);
        assert!(s.is_clip_selected(3));
    }

    #[test]
    fn test_select_clip_add_no_duplicates() {
        let mut s = Selection::new();
        s.select_clip(5, SelectionMode::Add);
        s.select_clip(5, SelectionMode::Add);
        assert_eq!(s.selected_clips().len(), 1);
    }

    #[test]
    fn test_select_clip_toggle() {
        let mut s = Selection::new();
        s.select_clip(7, SelectionMode::Toggle);
        assert!(s.is_clip_selected(7));
        s.select_clip(7, SelectionMode::Toggle);
        assert!(!s.is_clip_selected(7));
    }

    #[test]
    fn test_select_all_in_range() {
        let clips = vec![
            TimelineClipRef::new(1, 0, 50),
            TimelineClipRef::new(2, 50, 100),
            TimelineClipRef::new(3, 200, 300),
        ];
        let mut s = Selection::new();
        s.select_all_in_range(&clips, SelectionRange::new(0, 100));
        assert!(s.is_clip_selected(1));
        assert!(s.is_clip_selected(2));
        assert!(!s.is_clip_selected(3));
    }

    // ------------------------------------------------------------------
    // Selection – general tests
    // ------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut s = Selection::new();
        s.select_clip(1, SelectionMode::Add);
        s.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
        s.select_track(0, SelectionMode::Add);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn test_merge_overlapping_three_ranges() {
        let mut s = Selection::new();
        s.select_range(SelectionRange::new(0, 30), SelectionMode::Add);
        s.select_range(SelectionRange::new(20, 60), SelectionMode::Add);
        s.select_range(SelectionRange::new(55, 100), SelectionMode::Add);
        assert_eq!(s.ranges().len(), 1);
        assert_eq!(s.ranges()[0], SelectionRange::new(0, 100));
    }

    #[test]
    fn test_is_empty_initially() {
        let s = Selection::new();
        assert!(s.is_empty());
    }

    #[test]
    fn test_track_selection() {
        let mut s = Selection::new();
        s.select_track(0, SelectionMode::Add);
        s.select_track(1, SelectionMode::Add);
        assert!(s.is_track_selected(0));
        assert!(s.is_track_selected(1));
        s.select_track(0, SelectionMode::Subtract);
        assert!(!s.is_track_selected(0));
    }

    // ------------------------------------------------------------------
    // SelectionItem tests
    // ------------------------------------------------------------------

    #[test]
    fn test_selection_item_same_track_true() {
        let a = SelectionItem::new(1, 0);
        let b = SelectionItem::new(2, 0);
        assert!(a.same_track(&b));
    }

    #[test]
    fn test_selection_item_same_track_false() {
        let a = SelectionItem::new(1, 0);
        let b = SelectionItem::new(2, 1);
        assert!(!a.same_track(&b));
    }

    // ------------------------------------------------------------------
    // EditSelection tests
    // ------------------------------------------------------------------

    #[test]
    fn test_edit_selection_add_and_contains() {
        let mut sel = EditSelection::new();
        sel.add(SelectionItem::new(10, 0));
        assert!(sel.contains(10));
        assert!(!sel.contains(99));
    }

    #[test]
    fn test_edit_selection_add_no_duplicate() {
        let mut sel = EditSelection::new();
        sel.add(SelectionItem::new(5, 1));
        sel.add(SelectionItem::new(5, 1));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn test_edit_selection_remove() {
        let mut sel = EditSelection::new();
        sel.add(SelectionItem::new(3, 0));
        sel.remove(3);
        assert!(!sel.contains(3));
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn test_edit_selection_clear() {
        let mut sel = EditSelection::new();
        sel.add(SelectionItem::new(1, 0));
        sel.add(SelectionItem::new(2, 1));
        sel.clear();
        assert_eq!(sel.count(), 0);
    }

    #[test]
    fn test_edit_selection_tracks_unique() {
        let mut sel = EditSelection::new();
        sel.add(SelectionItem::new(1, 0));
        sel.add(SelectionItem::new(2, 0));
        sel.add(SelectionItem::new(3, 1));
        let tracks = sel.tracks();
        assert_eq!(tracks.len(), 2);
        assert!(tracks.contains(&0));
        assert!(tracks.contains(&1));
    }

    #[test]
    fn test_edit_selection_count() {
        let mut sel = EditSelection::new();
        assert_eq!(sel.count(), 0);
        sel.add(SelectionItem::new(7, 2));
        assert_eq!(sel.count(), 1);
    }

    // ------------------------------------------------------------------
    // LinkedSelection tests
    // ------------------------------------------------------------------

    #[test]
    fn test_linked_selection_no_links() {
        let ls = LinkedSelection::new();
        assert!(ls.linked_clips(42).is_empty());
    }

    #[test]
    fn test_linked_selection_add_group() {
        let mut ls = LinkedSelection::new();
        ls.add_linked_group(vec![1, 2, 3]);
        let linked = ls.linked_clips(1);
        assert!(linked.contains(&2));
        assert!(linked.contains(&3));
        assert!(!linked.contains(&1)); // self excluded
    }

    #[test]
    fn test_linked_selection_multiple_groups() {
        let mut ls = LinkedSelection::new();
        ls.add_linked_group(vec![10, 11]);
        ls.add_linked_group(vec![20, 21, 22]);
        // clip 10 should only know about 11
        assert_eq!(ls.linked_clips(10), vec![11]);
        // clip 22 should know about 20, 21
        let linked = ls.linked_clips(22);
        assert!(linked.contains(&20));
        assert!(linked.contains(&21));
    }

    #[test]
    fn test_linked_selection_empty_group_ignored() {
        let mut ls = LinkedSelection::new();
        ls.add_linked_group(vec![]);
        assert_eq!(ls.groups.len(), 0);
    }

    #[test]
    fn test_linked_selection_clip_not_in_any_group() {
        let mut ls = LinkedSelection::new();
        ls.add_linked_group(vec![1, 2]);
        assert!(ls.linked_clips(99).is_empty());
    }
}
