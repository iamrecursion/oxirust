//! Review markers for frame-level annotations on a timeline.

use crate::drawing::color::Color;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Category of a review marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerCategory {
    /// Issue that needs correction.
    Issue,
    /// Informational marker.
    Info,
    /// Scene or section boundary.
    Scene,
    /// Approval checkpoint.
    Approval,
    /// Custom / other category.
    Custom,
}

impl std::fmt::Display for MarkerCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Issue => write!(f, "Issue"),
            Self::Info => write!(f, "Info"),
            Self::Scene => write!(f, "Scene"),
            Self::Approval => write!(f, "Approval"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A marker placed at a specific frame in the review timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewMarker {
    /// Frame number where the marker is placed (0-indexed).
    pub frame_number: i64,
    /// Display color of the marker.
    pub color: Color,
    /// Short label text.
    pub label: String,
    /// Category of the marker.
    pub category: MarkerCategory,
    /// Author who placed the marker.
    pub author: String,
    /// Optional extended note.
    pub note: Option<String>,
    /// When the marker was created.
    pub created_at: DateTime<Utc>,
}

impl ReviewMarker {
    /// Create a new review marker.
    #[must_use]
    pub fn new(
        frame_number: i64,
        color: Color,
        label: impl Into<String>,
        category: MarkerCategory,
        author: impl Into<String>,
    ) -> Self {
        Self {
            frame_number,
            color,
            label: label.into(),
            category,
            author: author.into(),
            note: None,
            created_at: Utc::now(),
        }
    }

    /// Attach a detailed note to the marker.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Returns true if this marker is an issue.
    #[must_use]
    pub fn is_issue(&self) -> bool {
        self.category == MarkerCategory::Issue
    }

    /// Returns true if this marker signals an approval point.
    #[must_use]
    pub fn is_approval(&self) -> bool {
        self.category == MarkerCategory::Approval
    }
}

/// Collection of review markers, sorted by frame number.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MarkerSet {
    markers: Vec<ReviewMarker>,
}

impl MarkerSet {
    /// Create an empty marker set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a marker (maintains sorted order by frame number).
    pub fn insert(&mut self, marker: ReviewMarker) {
        let pos = self
            .markers
            .partition_point(|m| m.frame_number <= marker.frame_number);
        self.markers.insert(pos, marker);
    }

    /// Return all markers.
    #[must_use]
    pub fn all(&self) -> &[ReviewMarker] {
        &self.markers
    }

    /// Return markers in a frame range [start, end] (inclusive).
    #[must_use]
    pub fn in_range(&self, start: i64, end: i64) -> Vec<&ReviewMarker> {
        self.markers
            .iter()
            .filter(|m| m.frame_number >= start && m.frame_number <= end)
            .collect()
    }

    /// Return markers of a specific category.
    #[must_use]
    pub fn by_category(&self, category: MarkerCategory) -> Vec<&ReviewMarker> {
        self.markers
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }

    /// Return the total number of markers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Returns true when there are no markers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> Color {
        Color::new(255, 0, 0, 1.0)
    }

    fn blue() -> Color {
        Color::new(0, 0, 255, 1.0)
    }

    #[test]
    fn test_marker_creation() {
        let m = ReviewMarker::new(42, red(), "Bad cut", MarkerCategory::Issue, "alice");
        assert_eq!(m.frame_number, 42);
        assert_eq!(m.label, "Bad cut");
        assert_eq!(m.category, MarkerCategory::Issue);
        assert_eq!(m.author, "alice");
        assert!(m.note.is_none());
    }

    #[test]
    fn test_marker_with_note() {
        let m = ReviewMarker::new(10, red(), "Scene end", MarkerCategory::Scene, "bob")
            .with_note("Confirm chapter boundary");
        assert!(m.note.is_some());
        assert_eq!(
            m.note.expect("should succeed in test"),
            "Confirm chapter boundary"
        );
    }

    #[test]
    fn test_marker_is_issue() {
        let issue = ReviewMarker::new(1, red(), "Glitch", MarkerCategory::Issue, "alice");
        let info = ReviewMarker::new(2, blue(), "Note", MarkerCategory::Info, "alice");
        assert!(issue.is_issue());
        assert!(!info.is_issue());
    }

    #[test]
    fn test_marker_is_approval() {
        let m = ReviewMarker::new(100, blue(), "OK", MarkerCategory::Approval, "alice");
        assert!(m.is_approval());
        assert!(!m.is_issue());
    }

    #[test]
    fn test_marker_set_insert_sorted() {
        let mut set = MarkerSet::new();
        set.insert(ReviewMarker::new(50, red(), "B", MarkerCategory::Info, "x"));
        set.insert(ReviewMarker::new(10, red(), "A", MarkerCategory::Info, "x"));
        set.insert(ReviewMarker::new(80, red(), "C", MarkerCategory::Info, "x"));
        let frames: Vec<i64> = set.all().iter().map(|m| m.frame_number).collect();
        assert_eq!(frames, vec![10, 50, 80]);
    }

    #[test]
    fn test_marker_set_in_range() {
        let mut set = MarkerSet::new();
        for f in [5, 20, 35, 50] {
            set.insert(ReviewMarker::new(f, red(), "x", MarkerCategory::Info, "x"));
        }
        let range = set.in_range(15, 40);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].frame_number, 20);
        assert_eq!(range[1].frame_number, 35);
    }

    #[test]
    fn test_marker_set_by_category() {
        let mut set = MarkerSet::new();
        set.insert(ReviewMarker::new(1, red(), "a", MarkerCategory::Issue, "x"));
        set.insert(ReviewMarker::new(
            2,
            blue(),
            "b",
            MarkerCategory::Approval,
            "x",
        ));
        set.insert(ReviewMarker::new(3, red(), "c", MarkerCategory::Issue, "x"));
        let issues = set.by_category(MarkerCategory::Issue);
        assert_eq!(issues.len(), 2);
        let approvals = set.by_category(MarkerCategory::Approval);
        assert_eq!(approvals.len(), 1);
    }

    #[test]
    fn test_marker_category_display() {
        assert_eq!(MarkerCategory::Issue.to_string(), "Issue");
        assert_eq!(MarkerCategory::Approval.to_string(), "Approval");
    }
}
