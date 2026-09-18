//! Timeline markers and notes.
//!
//! Provides `TimelineMarkerType`, `TimelineMarker`, and `MarkerCollection`
//! for attaching named, typed markers to frame positions in a timeline.

/// The semantic category of a timeline marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TimelineMarkerType {
    /// Chapter division point (used for chapter menus, streaming segments).
    Chapter,
    /// Informational note attached to a frame.
    Note,
    /// In-point for a range selection.
    InPoint,
    /// Out-point for a range selection.
    OutPoint,
    /// Synchronisation reference point.
    Sync,
    /// Warning flag, e.g. for a problematic frame or clip.
    Warning,
}

impl TimelineMarkerType {
    /// Returns a CSS hex colour string associated with this marker type.
    #[must_use]
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Chapter => "#E67E22",
            Self::Note => "#3498DB",
            Self::InPoint => "#27AE60",
            Self::OutPoint => "#C0392B",
            Self::Sync => "#8E44AD",
            Self::Warning => "#F1C40F",
        }
    }

    /// Returns `true` if this marker type is intended to mark a *range*
    /// (i.e. it has a conceptual partner on the other end of a span).
    #[must_use]
    pub fn is_range(&self) -> bool {
        matches!(self, Self::InPoint | Self::OutPoint)
    }
}

/// A single marker placed at a specific frame in a timeline.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TimelineMarker {
    /// Unique identifier for this marker.
    pub id: u64,
    /// Frame position where the marker is placed.
    pub frame: u64,
    /// Human-readable label.
    pub name: String,
    /// Semantic type of this marker.
    pub marker_type: TimelineMarkerType,
    /// Optional longer description or note text.
    pub note: Option<String>,
    /// Duration in frames (0 = point marker; > 0 = range marker).
    pub duration_frames: u32,
}

impl TimelineMarker {
    /// Returns the exclusive end frame.  For point markers this equals `frame`.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.frame + u64::from(self.duration_frames)
    }

    /// Returns `true` if this is a point marker (zero duration).
    #[must_use]
    pub fn is_point(&self) -> bool {
        self.duration_frames == 0
    }

    /// Returns `true` if this marker spans more than a single frame.
    #[must_use]
    pub fn has_duration(&self) -> bool {
        self.duration_frames > 0
    }
}

/// An ordered, searchable collection of timeline markers.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct MarkerCollection {
    /// All markers in insertion order.
    pub markers: Vec<TimelineMarker>,
    /// Next ID to assign when a marker is added.
    pub next_id: u64,
}

impl MarkerCollection {
    /// Create a new, empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a marker to the collection, assigning it a unique ID.
    ///
    /// Returns the assigned ID.
    pub fn add(&mut self, mut m: TimelineMarker) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        m.id = id;
        self.markers.push(m);
        id
    }

    /// Remove the marker with the given ID.
    ///
    /// Returns `true` if the marker was found and removed.
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.markers.iter().position(|m| m.id == id) {
            self.markers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Return all markers whose `frame` exactly matches `frame`.
    #[must_use]
    pub fn at_frame(&self, frame: u64) -> Vec<&TimelineMarker> {
        self.markers.iter().filter(|m| m.frame == frame).collect()
    }

    /// Return all markers that start within `[start, end)`.
    #[must_use]
    pub fn in_range(&self, start: u64, end: u64) -> Vec<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.frame >= start && m.frame < end)
            .collect()
    }

    /// Return all markers whose type is `Chapter`.
    #[must_use]
    pub fn chapters(&self) -> Vec<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.marker_type == TimelineMarkerType::Chapter)
            .collect()
    }

    /// Return all markers sorted by ascending frame number.
    #[must_use]
    pub fn sorted_by_frame(&self) -> Vec<&TimelineMarker> {
        let mut sorted: Vec<&TimelineMarker> = self.markers.iter().collect();
        sorted.sort_by_key(|m| m.frame);
        sorted
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_marker(frame: u64, marker_type: TimelineMarkerType, dur: u32) -> TimelineMarker {
        TimelineMarker {
            id: 0, // will be assigned by collection
            frame,
            name: format!("M@{frame}"),
            marker_type,
            note: None,
            duration_frames: dur,
        }
    }

    // --- TimelineMarkerType ---

    #[test]
    fn test_color_hex_starts_with_hash() {
        for t in [
            TimelineMarkerType::Chapter,
            TimelineMarkerType::Note,
            TimelineMarkerType::InPoint,
            TimelineMarkerType::OutPoint,
            TimelineMarkerType::Sync,
            TimelineMarkerType::Warning,
        ] {
            assert!(t.color_hex().starts_with('#'), "missing '#': {:?}", t);
        }
    }

    #[test]
    fn test_in_point_is_range() {
        assert!(TimelineMarkerType::InPoint.is_range());
    }

    #[test]
    fn test_out_point_is_range() {
        assert!(TimelineMarkerType::OutPoint.is_range());
    }

    #[test]
    fn test_chapter_is_not_range() {
        assert!(!TimelineMarkerType::Chapter.is_range());
    }

    #[test]
    fn test_note_is_not_range() {
        assert!(!TimelineMarkerType::Note.is_range());
    }

    // --- TimelineMarker ---

    #[test]
    fn test_marker_end_frame_point() {
        let m = make_marker(100, TimelineMarkerType::Note, 0);
        assert_eq!(m.end_frame(), 100);
    }

    #[test]
    fn test_marker_end_frame_range() {
        let m = make_marker(200, TimelineMarkerType::Chapter, 48);
        assert_eq!(m.end_frame(), 248);
    }

    #[test]
    fn test_marker_is_point_true() {
        let m = make_marker(50, TimelineMarkerType::Sync, 0);
        assert!(m.is_point());
    }

    #[test]
    fn test_marker_is_point_false_when_duration() {
        let m = make_marker(50, TimelineMarkerType::Chapter, 24);
        assert!(!m.is_point());
    }

    #[test]
    fn test_marker_has_duration_true() {
        let m = make_marker(10, TimelineMarkerType::Chapter, 100);
        assert!(m.has_duration());
    }

    #[test]
    fn test_marker_has_duration_false_for_point() {
        let m = make_marker(10, TimelineMarkerType::Warning, 0);
        assert!(!m.has_duration());
    }

    // --- MarkerCollection ---

    #[test]
    fn test_collection_add_assigns_ids() {
        let mut col = MarkerCollection::new();
        let id1 = col.add(make_marker(0, TimelineMarkerType::Note, 0));
        let id2 = col.add(make_marker(10, TimelineMarkerType::Chapter, 0));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_collection_remove_existing() {
        let mut col = MarkerCollection::new();
        col.add(make_marker(0, TimelineMarkerType::Note, 0));
        assert!(col.remove(1));
        assert!(col.markers.is_empty());
    }

    #[test]
    fn test_collection_remove_nonexistent() {
        let mut col = MarkerCollection::new();
        assert!(!col.remove(999));
    }

    #[test]
    fn test_collection_at_frame() {
        let mut col = MarkerCollection::new();
        col.add(make_marker(100, TimelineMarkerType::Note, 0));
        col.add(make_marker(200, TimelineMarkerType::Note, 0));
        col.add(make_marker(100, TimelineMarkerType::Chapter, 0));
        let at_100 = col.at_frame(100);
        assert_eq!(at_100.len(), 2);
    }

    #[test]
    fn test_collection_in_range() {
        let mut col = MarkerCollection::new();
        col.add(make_marker(10, TimelineMarkerType::Note, 0));
        col.add(make_marker(50, TimelineMarkerType::Chapter, 0));
        col.add(make_marker(100, TimelineMarkerType::Warning, 0));
        // range [10, 100) should include frames 10 and 50 but not 100
        let in_range = col.in_range(10, 100);
        assert_eq!(in_range.len(), 2);
    }

    #[test]
    fn test_collection_chapters() {
        let mut col = MarkerCollection::new();
        col.add(make_marker(0, TimelineMarkerType::Chapter, 0));
        col.add(make_marker(100, TimelineMarkerType::Note, 0));
        col.add(make_marker(200, TimelineMarkerType::Chapter, 0));
        assert_eq!(col.chapters().len(), 2);
    }

    #[test]
    fn test_collection_sorted_by_frame() {
        let mut col = MarkerCollection::new();
        col.add(make_marker(300, TimelineMarkerType::Note, 0));
        col.add(make_marker(100, TimelineMarkerType::Chapter, 0));
        col.add(make_marker(200, TimelineMarkerType::Warning, 0));
        let sorted = col.sorted_by_frame();
        assert_eq!(sorted[0].frame, 100);
        assert_eq!(sorted[1].frame, 200);
        assert_eq!(sorted[2].frame, 300);
    }
}
