//! Cue points for marking significant moments in a subtitle track.
//!
//! Cue points are time-stamped annotations attached to a subtitle stream that
//! flag events such as chapter boundaries, advertisement breaks, or explicit
//! subtitle positions.  They complement subtitle cues but are distinct from
//! them: a cue point carries a *type* and optional *label* rather than
//! displayable text.

#![allow(dead_code)]

/// The semantic category of a cue point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CueType {
    /// Start of a chapter or named section.
    Chapter,
    /// An advertisement break boundary.
    AdBreak,
    /// A bookmark set interactively by the user.
    Bookmark,
    /// A scene transition detected automatically.
    SceneChange,
    /// A forced subtitle display event.
    ForcedSubtitle,
    /// Any application-defined custom event.
    Custom(String),
}

impl CueType {
    /// Return a human-readable label for the cue type.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Chapter => "Chapter",
            Self::AdBreak => "AdBreak",
            Self::Bookmark => "Bookmark",
            Self::SceneChange => "SceneChange",
            Self::ForcedSubtitle => "ForcedSubtitle",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// A single cue point: a typed, timestamped annotation on the subtitle track.
///
/// # Example
///
/// ```
/// use oximedia_subtitle::cue_point::{CuePoint, CueType};
///
/// let cp = CuePoint::new(CueType::Chapter, 5000)
///     .with_label("Act 1");
/// assert_eq!(cp.label.as_deref(), Some("Act 1"));
/// ```
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Semantic type of this cue point.
    pub cue_type: CueType,
    /// Timestamp of the cue point in milliseconds.
    pub timestamp_ms: i64,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Optional duration in milliseconds (e.g. for ad breaks).
    pub duration_ms: Option<i64>,
}

impl CuePoint {
    /// Create a new cue point at the given timestamp.
    #[must_use]
    pub fn new(cue_type: CueType, timestamp_ms: i64) -> Self {
        Self {
            cue_type,
            timestamp_ms,
            label: None,
            duration_ms: None,
        }
    }

    /// Attach a label to this cue point.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Attach a duration to this cue point.
    #[must_use]
    pub fn with_duration(mut self, duration_ms: i64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Return `true` if the cue point falls within `[from_ms, to_ms)`.
    #[must_use]
    pub fn in_range(&self, from_ms: i64, to_ms: i64) -> bool {
        self.timestamp_ms >= from_ms && self.timestamp_ms < to_ms
    }
}

/// An ordered, mutable collection of [`CuePoint`]s.
///
/// # Example
///
/// ```
/// use oximedia_subtitle::cue_point::{CuePoint, CuePointList, CueType};
///
/// let mut list = CuePointList::new();
/// list.add(CuePoint::new(CueType::Chapter, 0));
/// list.add(CuePoint::new(CueType::AdBreak, 30_000));
/// list.add(CuePoint::new(CueType::Chapter, 60_000));
///
/// let hits = list.in_range(25_000, 65_000);
/// assert_eq!(hits.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct CuePointList {
    points: Vec<CuePoint>,
}

impl CuePointList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Append a cue point.  The list is kept sorted by timestamp.
    pub fn add(&mut self, cue: CuePoint) {
        let pos = self
            .points
            .partition_point(|p| p.timestamp_ms <= cue.timestamp_ms);
        self.points.insert(pos, cue);
    }

    /// Return the number of cue points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the list contains no cue points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return all cue points whose timestamp falls within `[from_ms, to_ms)`.
    #[must_use]
    pub fn in_range(&self, from_ms: i64, to_ms: i64) -> Vec<&CuePoint> {
        self.points
            .iter()
            .filter(|p| p.in_range(from_ms, to_ms))
            .collect()
    }

    /// Return all cue points of a specific type.
    #[must_use]
    pub fn by_type(&self, cue_type: &CueType) -> Vec<&CuePoint> {
        self.points
            .iter()
            .filter(|p| &p.cue_type == cue_type)
            .collect()
    }

    /// Return the cue point immediately at or before the given timestamp,
    /// if any.
    #[must_use]
    pub fn latest_at(&self, timestamp_ms: i64) -> Option<&CuePoint> {
        self.points
            .iter()
            .rev()
            .find(|p| p.timestamp_ms <= timestamp_ms)
    }

    /// Remove all cue points of the specified type.
    pub fn remove_type(&mut self, cue_type: &CueType) {
        self.points.retain(|p| &p.cue_type != cue_type);
    }

    /// Return an iterator over all cue points in timestamp order.
    pub fn iter(&self) -> impl Iterator<Item = &CuePoint> {
        self.points.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_list() -> CuePointList {
        let mut list = CuePointList::new();
        list.add(CuePoint::new(CueType::Chapter, 0).with_label("Intro"));
        list.add(CuePoint::new(CueType::AdBreak, 30_000).with_duration(15_000));
        list.add(CuePoint::new(CueType::Chapter, 60_000).with_label("Act 1"));
        list.add(CuePoint::new(CueType::SceneChange, 45_000));
        list.add(CuePoint::new(CueType::Bookmark, 90_000).with_label("Fav"));
        list
    }

    #[test]
    fn test_list_len() {
        let list = make_list();
        assert_eq!(list.len(), 5);
    }

    #[test]
    fn test_list_is_sorted() {
        let list = make_list();
        let ts: Vec<i64> = list.iter().map(|p| p.timestamp_ms).collect();
        let mut sorted = ts.clone();
        sorted.sort_unstable();
        assert_eq!(ts, sorted);
    }

    #[test]
    fn test_in_range_basic() {
        let list = make_list();
        let hits = list.in_range(0, 31_000);
        // should include 0 (Chapter) and 30_000 (AdBreak), not 45_000+
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_in_range_exclusive_end() {
        let list = make_list();
        // 30_000 should NOT be in [0, 30_000)
        let hits = list.in_range(0, 30_000);
        assert!(!hits.iter().any(|p| p.timestamp_ms == 30_000));
    }

    #[test]
    fn test_in_range_empty() {
        let list = make_list();
        let hits = list.in_range(200_000, 300_000);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_by_type_chapter() {
        let list = make_list();
        let chapters = list.by_type(&CueType::Chapter);
        assert_eq!(chapters.len(), 2);
    }

    #[test]
    fn test_by_type_no_match() {
        let list = make_list();
        let custom = list.by_type(&CueType::Custom("foo".to_string()));
        assert!(custom.is_empty());
    }

    #[test]
    fn test_latest_at() {
        let list = make_list();
        let cp = list.latest_at(50_000).expect("should succeed in test");
        assert_eq!(cp.timestamp_ms, 45_000);
    }

    #[test]
    fn test_latest_at_none() {
        let list = make_list();
        assert!(list.latest_at(-1).is_none());
    }

    #[test]
    fn test_remove_type() {
        let mut list = make_list();
        list.remove_type(&CueType::Chapter);
        assert!(list.by_type(&CueType::Chapter).is_empty());
        // Others unaffected
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_cue_point_in_range() {
        let cp = CuePoint::new(CueType::Bookmark, 5000);
        assert!(cp.in_range(4000, 6000));
        // in_range is [from, to): from_ms=5000 and timestamp_ms=5000 => 5000 >= 5000 is true
        assert!(cp.in_range(5000, 6000));
        assert!(!cp.in_range(5001, 6000));
        assert!(!cp.in_range(6000, 9000));
    }

    #[test]
    fn test_cue_type_label_custom() {
        let t = CueType::Custom("my-event".to_string());
        assert_eq!(t.label(), "my-event");
    }

    #[test]
    fn test_cue_type_label_standard() {
        assert_eq!(CueType::Chapter.label(), "Chapter");
        assert_eq!(CueType::AdBreak.label(), "AdBreak");
        assert_eq!(CueType::SceneChange.label(), "SceneChange");
    }

    #[test]
    fn test_with_duration() {
        let cp = CuePoint::new(CueType::AdBreak, 0).with_duration(30_000);
        assert_eq!(cp.duration_ms, Some(30_000));
    }

    #[test]
    fn test_empty_list() {
        let list = CuePointList::new();
        assert!(list.is_empty());
        assert!(list.latest_at(9999).is_none());
        assert!(list.in_range(0, 1000).is_empty());
    }
}
