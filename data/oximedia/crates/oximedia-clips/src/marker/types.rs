//! Marker types and structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MarkerId(Uuid);

impl MarkerId {
    /// Creates a new random marker ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a marker ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for MarkerId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MarkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerType {
    /// Standard marker.
    Standard,
    /// Chapter marker for navigation.
    Chapter,
    /// To-do marker for tasks.
    ToDo,
    /// Comment marker.
    Comment,
    /// Beat marker for story structure.
    Beat,
    /// Sync marker for synchronization.
    Sync,
    /// Custom marker type.
    Custom,
}

impl MarkerType {
    /// Returns all marker types.
    #[must_use]
    pub const fn all() -> [Self; 7] {
        [
            Self::Standard,
            Self::Chapter,
            Self::ToDo,
            Self::Comment,
            Self::Beat,
            Self::Sync,
            Self::Custom,
        ]
    }
}

impl std::fmt::Display for MarkerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::Chapter => write!(f, "Chapter"),
            Self::ToDo => write!(f, "To-Do"),
            Self::Comment => write!(f, "Comment"),
            Self::Beat => write!(f, "Beat"),
            Self::Sync => write!(f, "Sync"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// A frame-accurate marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Unique identifier.
    pub id: MarkerId,

    /// Marker type.
    pub marker_type: MarkerType,

    /// Frame position.
    pub frame: i64,

    /// Marker name/title.
    pub name: String,

    /// Optional comment/note.
    pub comment: Option<String>,

    /// Color (RGB hex, e.g., "#FF0000").
    pub color: Option<String>,

    /// Duration in frames (for range markers).
    pub duration: Option<i64>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Created by user.
    pub created_by: Option<String>,

    /// Whether this is completed (for to-do markers).
    pub is_completed: bool,
}

impl Marker {
    /// Creates a new marker.
    #[must_use]
    pub fn new(marker_type: MarkerType, frame: i64, name: impl Into<String>) -> Self {
        Self {
            id: MarkerId::new(),
            marker_type,
            frame,
            name: name.into(),
            comment: None,
            color: None,
            duration: None,
            created_at: Utc::now(),
            created_by: None,
            is_completed: false,
        }
    }

    /// Creates a chapter marker.
    #[must_use]
    pub fn chapter(frame: i64, name: impl Into<String>) -> Self {
        Self::new(MarkerType::Chapter, frame, name)
    }

    /// Creates a to-do marker.
    #[must_use]
    pub fn todo(frame: i64, name: impl Into<String>) -> Self {
        Self::new(MarkerType::ToDo, frame, name)
    }

    /// Creates a comment marker.
    #[must_use]
    pub fn comment_marker(frame: i64, name: impl Into<String>) -> Self {
        Self::new(MarkerType::Comment, frame, name)
    }

    /// Sets the comment.
    pub fn set_comment(&mut self, comment: impl Into<String>) {
        self.comment = Some(comment.into());
    }

    /// Sets the color.
    pub fn set_color(&mut self, color: impl Into<String>) {
        self.color = Some(color.into());
    }

    /// Sets the duration (for range markers).
    pub fn set_duration(&mut self, duration: i64) {
        self.duration = Some(duration);
    }

    /// Sets the created by user.
    pub fn set_created_by(&mut self, user: impl Into<String>) {
        self.created_by = Some(user.into());
    }

    /// Marks the to-do as completed.
    pub fn set_completed(&mut self, completed: bool) {
        self.is_completed = completed;
    }

    /// Returns the end frame (for range markers).
    #[must_use]
    pub fn end_frame(&self) -> Option<i64> {
        self.duration.map(|d| self.frame + d)
    }

    /// Checks if this is a range marker.
    #[must_use]
    pub fn is_range(&self) -> bool {
        self.duration.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_creation() {
        let marker = Marker::new(MarkerType::Standard, 100, "Test Marker");
        assert_eq!(marker.frame, 100);
        assert_eq!(marker.name, "Test Marker");
        assert_eq!(marker.marker_type, MarkerType::Standard);
    }

    #[test]
    fn test_chapter_marker() {
        let marker = Marker::chapter(100, "Act 1");
        assert_eq!(marker.marker_type, MarkerType::Chapter);
        assert_eq!(marker.name, "Act 1");
    }

    #[test]
    fn test_range_marker() {
        let mut marker = Marker::new(MarkerType::Standard, 100, "Range");
        marker.set_duration(50);
        assert!(marker.is_range());
        assert_eq!(marker.end_frame(), Some(150));
    }

    #[test]
    fn test_todo_marker() {
        let mut marker = Marker::todo(100, "Fix color");
        assert!(!marker.is_completed);
        marker.set_completed(true);
        assert!(marker.is_completed);
    }
}
