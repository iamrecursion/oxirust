//! Annotation and note types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NoteId(Uuid);

impl NoteId {
    /// Creates a new random note ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a note ID from a UUID.
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

impl Default for NoteId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NoteId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A note or annotation on a clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    /// Unique identifier.
    pub id: NoteId,

    /// Note content.
    pub content: String,

    /// Optional frame position.
    pub frame: Option<i64>,

    /// Created timestamp.
    pub created_at: DateTime<Utc>,

    /// Created by user.
    pub created_by: Option<String>,

    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,

    /// Reply to another note (for threading).
    pub reply_to: Option<NoteId>,
}

impl Note {
    /// Creates a new note.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: NoteId::new(),
            content: content.into(),
            frame: None,
            created_at: now,
            created_by: None,
            modified_at: now,
            reply_to: None,
        }
    }

    /// Creates a note at a specific frame.
    #[must_use]
    pub fn at_frame(content: impl Into<String>, frame: i64) -> Self {
        let mut note = Self::new(content);
        note.frame = Some(frame);
        note
    }

    /// Creates a reply to another note.
    #[must_use]
    pub fn reply_to(content: impl Into<String>, reply_to: NoteId) -> Self {
        let mut note = Self::new(content);
        note.reply_to = Some(reply_to);
        note
    }

    /// Sets the frame position.
    pub fn set_frame(&mut self, frame: i64) {
        self.frame = Some(frame);
        self.modified_at = Utc::now();
    }

    /// Sets the creator.
    pub fn set_created_by(&mut self, user: impl Into<String>) {
        self.created_by = Some(user.into());
    }

    /// Updates the content.
    pub fn set_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.modified_at = Utc::now();
    }

    /// Checks if this is a reply.
    #[must_use]
    pub const fn is_reply(&self) -> bool {
        self.reply_to.is_some()
    }
}

/// An annotation with drawing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Unique identifier.
    pub id: NoteId,

    /// Associated note.
    pub note: Note,

    /// Frame position.
    pub frame: i64,

    /// Annotation type.
    pub annotation_type: AnnotationType,

    /// Drawing data (format depends on type).
    pub data: AnnotationData,
}

/// Type of annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationType {
    /// Free-form drawing.
    Drawing,
    /// Arrow annotation.
    Arrow,
    /// Rectangle/box.
    Rectangle,
    /// Circle/ellipse.
    Circle,
    /// Text annotation.
    Text,
}

/// Annotation drawing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationData {
    /// Path data for drawings.
    Path(Vec<(f64, f64)>),
    /// Arrow with start and end points.
    Arrow {
        /// Start point (x, y).
        start: (f64, f64),
        /// End point (x, y).
        end: (f64, f64),
    },
    /// Rectangle with position and size.
    Rectangle {
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
        /// Width.
        width: f64,
        /// Height.
        height: f64,
    },
    /// Circle with center and radius.
    Circle {
        /// Center X coordinate.
        cx: f64,
        /// Center Y coordinate.
        cy: f64,
        /// Radius.
        radius: f64,
    },
    /// Text with position and content.
    Text {
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
        /// Text content.
        text: String,
    },
}

impl Annotation {
    /// Creates a new annotation.
    #[must_use]
    pub fn new(
        note: Note,
        frame: i64,
        annotation_type: AnnotationType,
        data: AnnotationData,
    ) -> Self {
        Self {
            id: NoteId::new(),
            note,
            frame,
            annotation_type,
            data,
        }
    }

    /// Creates an arrow annotation.
    #[must_use]
    pub fn arrow(note: Note, frame: i64, start: (f64, f64), end: (f64, f64)) -> Self {
        Self::new(
            note,
            frame,
            AnnotationType::Arrow,
            AnnotationData::Arrow { start, end },
        )
    }

    /// Creates a rectangle annotation.
    #[must_use]
    pub fn rectangle(note: Note, frame: i64, x: f64, y: f64, width: f64, height: f64) -> Self {
        Self::new(
            note,
            frame,
            AnnotationType::Rectangle,
            AnnotationData::Rectangle {
                x,
                y,
                width,
                height,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_creation() {
        let note = Note::new("This is a test note");
        assert_eq!(note.content, "This is a test note");
        assert!(note.frame.is_none());
        assert!(!note.is_reply());
    }

    #[test]
    fn test_note_at_frame() {
        let note = Note::at_frame("Frame note", 100);
        assert_eq!(note.frame, Some(100));
    }

    #[test]
    fn test_note_reply() {
        let original = Note::new("Original note");
        let reply = Note::reply_to("Reply", original.id);
        assert!(reply.is_reply());
        assert_eq!(reply.reply_to, Some(original.id));
    }

    #[test]
    fn test_annotation() {
        let note = Note::new("Arrow pointing here");
        let annotation = Annotation::arrow(note, 100, (10.0, 20.0), (100.0, 200.0));
        assert_eq!(annotation.frame, 100);
        assert_eq!(annotation.annotation_type, AnnotationType::Arrow);
    }
}
