//! Temporal notes and markers for review sessions.

use std::time::SystemTime;

/// A time range in a media clip.
#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    /// First frame of the range (inclusive).
    pub start_frame: u64,
    /// Last frame of the range (inclusive).
    pub end_frame: u64,
    /// Frames per second.
    pub frame_rate: f64,
}

impl TimeRange {
    /// Create a new time range.
    #[must_use]
    pub fn new(start: u64, end: u64, fps: f64) -> Self {
        Self {
            start_frame: start,
            end_frame: end,
            frame_rate: fps,
        }
    }

    /// Number of frames in the range (inclusive both ends).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        if self.end_frame >= self.start_frame {
            self.end_frame - self.start_frame + 1
        } else {
            0
        }
    }

    /// Duration of the range in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.frame_rate > 0.0 {
            self.duration_frames() as f64 / self.frame_rate
        } else {
            0.0
        }
    }

    /// Check whether `frame` falls within this range (inclusive).
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }

    /// Check whether this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start_frame <= other.end_frame && other.start_frame <= self.end_frame
    }
}

/// Classification of a timeline note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteType {
    /// General feedback.
    General,
    /// Visual issue (color, exposure, etc.).
    Visual,
    /// Audio issue.
    Audio,
    /// Edit pacing / timing issue.
    Pacing,
    /// Technical issue (codec, format, quality).
    Technical,
    /// Legal concern (clearances, rights).
    Legal,
}

impl NoteType {
    /// Short human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            NoteType::General => "General",
            NoteType::Visual => "Visual",
            NoteType::Audio => "Audio",
            NoteType::Pacing => "Pacing",
            NoteType::Technical => "Technical",
            NoteType::Legal => "Legal",
        }
    }

    /// Representative emoji for the note type.
    #[must_use]
    pub fn emoji(&self) -> &'static str {
        match self {
            NoteType::General => "\u{1F4DD}",       // 📝
            NoteType::Visual => "\u{1F441}",        // 👁
            NoteType::Audio => "\u{1F50A}",         // 🔊
            NoteType::Pacing => "\u{2702}\u{FE0F}", // ✂️
            NoteType::Technical => "\u{1F527}",     // 🔧
            NoteType::Legal => "\u{2696}\u{FE0F}",  // ⚖️
        }
    }
}

/// A temporal note attached to a specific time range in the media.
#[derive(Debug, Clone)]
pub struct TimelineNote {
    /// Unique identifier.
    pub id: String,
    /// Author name.
    pub author: String,
    /// Time range this note covers.
    pub time_range: TimeRange,
    /// Text of the note.
    pub text: String,
    /// Classification of the note.
    pub note_type: NoteType,
    /// Whether the note has been resolved.
    pub resolved: bool,
    /// Creation timestamp.
    pub created_at: SystemTime,
    /// User-defined tags.
    pub tags: Vec<String>,
}

impl TimelineNote {
    /// Create a new timeline note.
    #[must_use]
    pub fn new(author: &str, range: TimeRange, text: &str, note_type: NoteType) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate a deterministic-looking ID from author + time + text
        let mut hasher = DefaultHasher::new();
        author.hash(&mut hasher);
        text.hash(&mut hasher);
        range.start_frame.hash(&mut hasher);
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
            .hash(&mut hasher);
        let id = format!("note-{:016x}", hasher.finish());

        Self {
            id,
            author: author.to_string(),
            time_range: range,
            text: text.to_string(),
            note_type,
            resolved: false,
            created_at: SystemTime::now(),
            tags: Vec::new(),
        }
    }

    /// Attach a tag to this note.
    #[must_use]
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Mark this note as resolved.
    pub fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Check whether this note covers the given frame.
    #[must_use]
    pub fn overlaps_frame(&self, frame: u64) -> bool {
        self.time_range.contains_frame(frame)
    }
}

/// A collection of timeline notes for a review session.
pub struct TimelineNoteCollection {
    notes: Vec<TimelineNote>,
    /// Total duration of the associated media in frames.
    #[allow(dead_code)]
    media_duration_frames: u64,
}

impl TimelineNoteCollection {
    /// Create a new collection for media with the given duration.
    #[must_use]
    pub fn new(duration_frames: u64) -> Self {
        Self {
            notes: Vec::new(),
            media_duration_frames: duration_frames,
        }
    }

    /// Add a note to the collection.
    pub fn add_note(&mut self, note: TimelineNote) {
        self.notes.push(note);
    }

    /// Look up a note by its ID.
    #[must_use]
    pub fn get_note(&self, id: &str) -> Option<&TimelineNote> {
        self.notes.iter().find(|n| n.id == id)
    }

    /// Mark the note with the given ID as resolved.  Returns `true` on success.
    pub fn resolve_note(&mut self, id: &str) -> bool {
        if let Some(note) = self.notes.iter_mut().find(|n| n.id == id) {
            note.resolve();
            true
        } else {
            false
        }
    }

    /// All notes that cover the given frame.
    #[must_use]
    pub fn notes_at_frame(&self, frame: u64) -> Vec<&TimelineNote> {
        self.notes
            .iter()
            .filter(|n| n.overlaps_frame(frame))
            .collect()
    }

    /// All notes written by a specific author.
    #[must_use]
    pub fn notes_by_author(&self, author: &str) -> Vec<&TimelineNote> {
        self.notes.iter().filter(|n| n.author == author).collect()
    }

    /// Number of unresolved notes.
    #[must_use]
    pub fn unresolved_count(&self) -> usize {
        self.notes.iter().filter(|n| !n.resolved).count()
    }

    /// All notes in insertion order.
    #[must_use]
    pub fn all_notes(&self) -> &[TimelineNote] {
        &self.notes
    }

    /// All notes of a given type.
    #[must_use]
    pub fn notes_by_type(&self, note_type: NoteType) -> Vec<&TimelineNote> {
        self.notes
            .iter()
            .filter(|n| n.note_type == note_type)
            .collect()
    }

    /// Generate a plain-text summary of all notes.
    #[must_use]
    pub fn export_summary(&self) -> String {
        let mut out = String::from("Timeline Note Summary\n");
        out.push_str("=====================\n\n");
        if self.notes.is_empty() {
            out.push_str("No notes.\n");
            return out;
        }
        for note in &self.notes {
            out.push_str(&format!(
                "[{}] {} ({}) frames {}-{}\n",
                if note.resolved { "RESOLVED" } else { "OPEN" },
                note.note_type.name(),
                note.author,
                note.time_range.start_frame,
                note.time_range.end_frame,
            ));
            out.push_str(&format!("  {}\n", note.text));
            if !note.tags.is_empty() {
                out.push_str(&format!("  Tags: {}\n", note.tags.join(", ")));
            }
            out.push('\n');
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

    fn make_range(start: u64, end: u64) -> TimeRange {
        TimeRange::new(start, end, 24.0)
    }

    fn make_note(author: &str, start: u64, end: u64) -> TimelineNote {
        TimelineNote::new(
            author,
            make_range(start, end),
            "test note",
            NoteType::General,
        )
    }

    #[test]
    fn test_time_range_new() {
        let r = make_range(10, 20);
        assert_eq!(r.start_frame, 10);
        assert_eq!(r.end_frame, 20);
        assert_eq!(r.duration_frames(), 11); // inclusive
    }

    #[test]
    fn test_time_range_contains() {
        let r = make_range(10, 20);
        assert!(r.contains_frame(10));
        assert!(r.contains_frame(15));
        assert!(r.contains_frame(20));
        assert!(!r.contains_frame(9));
        assert!(!r.contains_frame(21));
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = make_range(0, 10);
        let r2 = make_range(5, 15);
        let r3 = make_range(11, 20);
        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
        assert!(!r1.overlaps(&r3));
        assert!(!r3.overlaps(&r1));
        // Edge: adjacent ranges share no frame, but boundary touches
        let r4 = make_range(10, 10);
        assert!(r1.overlaps(&r4));
    }

    #[test]
    fn test_time_range_duration_seconds() {
        let r = TimeRange::new(0, 23, 24.0); // 24 frames at 24 fps = 1.0 s
        let secs = r.duration_seconds();
        assert!((secs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_note_new() {
        let note = make_note("Alice", 0, 10);
        assert_eq!(note.author, "Alice");
        assert!(!note.resolved);
        assert!(note.tags.is_empty());
        assert!(!note.id.is_empty());
    }

    #[test]
    fn test_note_with_tag() {
        let note = make_note("Bob", 0, 5).with_tag("color").with_tag("urgent");
        assert_eq!(note.tags.len(), 2);
        assert!(note.tags.contains(&"color".to_string()));
        assert!(note.tags.contains(&"urgent".to_string()));
    }

    #[test]
    fn test_note_resolve() {
        let mut note = make_note("Alice", 0, 10);
        assert!(!note.resolved);
        note.resolve();
        assert!(note.resolved);
    }

    #[test]
    fn test_collection_add() {
        let mut col = TimelineNoteCollection::new(1000);
        assert_eq!(col.all_notes().len(), 0);
        col.add_note(make_note("Alice", 0, 10));
        col.add_note(make_note("Bob", 5, 15));
        assert_eq!(col.all_notes().len(), 2);
    }

    #[test]
    fn test_collection_notes_at_frame() {
        let mut col = TimelineNoteCollection::new(1000);
        col.add_note(make_note("Alice", 0, 10));
        col.add_note(make_note("Bob", 20, 30));
        let at_5 = col.notes_at_frame(5);
        assert_eq!(at_5.len(), 1);
        assert_eq!(at_5[0].author, "Alice");
        let at_25 = col.notes_at_frame(25);
        assert_eq!(at_25.len(), 1);
        assert_eq!(at_25[0].author, "Bob");
        let at_15 = col.notes_at_frame(15);
        assert_eq!(at_15.len(), 0);
    }

    #[test]
    fn test_collection_unresolved() {
        let mut col = TimelineNoteCollection::new(1000);
        col.add_note(make_note("Alice", 0, 10));
        col.add_note(make_note("Bob", 20, 30));
        assert_eq!(col.unresolved_count(), 2);
        let id = col.all_notes()[0].id.clone();
        assert!(col.resolve_note(&id));
        assert_eq!(col.unresolved_count(), 1);
    }

    #[test]
    fn test_note_type_name() {
        for nt in [
            NoteType::General,
            NoteType::Visual,
            NoteType::Audio,
            NoteType::Pacing,
            NoteType::Technical,
            NoteType::Legal,
        ] {
            assert!(!nt.name().is_empty());
        }
    }

    #[test]
    fn test_export_summary_not_empty() {
        let mut col = TimelineNoteCollection::new(1000);
        col.add_note(make_note("Alice", 0, 10));
        let summary = col.export_summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Alice"));
    }
}
