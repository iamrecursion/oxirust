//! Extended EDL event model.
//!
//! Provides `EditType`, `EdlEvent` (local, distinct from `event::EdlEvent`),
//! and `EdlEventList` for building and querying EDL event sequences.

#![allow(dead_code)]

/// The type of edit at a cut point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    /// Hard cut (instantaneous transition).
    Cut,
    /// Dissolve (cross-fade between outgoing and incoming).
    Dissolve,
    /// Wipe (spatial transition defined by a wipe pattern number).
    Wipe,
    /// Key (matte / chroma key composite).
    Key,
}

impl EditType {
    /// Returns `true` if the edit type is a transition (not a hard cut).
    #[must_use]
    pub fn is_transition(self) -> bool {
        !matches!(self, Self::Cut)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cut => "C",
            Self::Dissolve => "D",
            Self::Wipe => "W",
            Self::Key => "K",
        }
    }
}

/// A single event in an EDL, with source/record timecodes in frames.
#[derive(Debug, Clone)]
pub struct EdlEvent {
    /// 1-based event number.
    pub number: u32,
    /// Source reel name.
    pub reel: String,
    /// Edit type at this cut point.
    pub edit_type: EditType,
    /// Source-in frame (inclusive).
    pub source_in: u64,
    /// Source-out frame (exclusive).
    pub source_out: u64,
    /// Record-in frame on the timeline (inclusive).
    pub record_in: u64,
    /// Record-out frame on the timeline (exclusive).
    pub record_out: u64,
    /// Optional wipe pattern number (only meaningful for `Wipe`).
    pub wipe_number: Option<u32>,
    /// Optional comment text.
    pub comment: Option<String>,
}

impl EdlEvent {
    /// Create a new `EdlEvent`.
    #[must_use]
    pub fn new(
        number: u32,
        reel: impl Into<String>,
        edit_type: EditType,
        source_in: u64,
        source_out: u64,
        record_in: u64,
        record_out: u64,
    ) -> Self {
        Self {
            number,
            reel: reel.into(),
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
            wipe_number: None,
            comment: None,
        }
    }

    /// Duration of this event on the record timeline in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.record_out.saturating_sub(self.record_in)
    }

    /// Returns `true` if the edit type is a transition (not a cut).
    #[must_use]
    pub fn is_transition(&self) -> bool {
        self.edit_type.is_transition()
    }

    /// Set a comment on the event.
    pub fn set_comment(&mut self, comment: impl Into<String>) {
        self.comment = Some(comment.into());
    }

    /// Set the wipe pattern number.
    pub fn set_wipe_number(&mut self, n: u32) {
        self.wipe_number = Some(n);
    }
}

/// An ordered collection of `EdlEvent` entries.
#[derive(Debug, Clone, Default)]
pub struct EdlEventList {
    events: Vec<EdlEvent>,
}

impl EdlEventList {
    /// Create an empty event list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the list.
    pub fn add(&mut self, event: EdlEvent) {
        self.events.push(event);
    }

    /// Return all events that are transitions (non-cut edits).
    #[must_use]
    pub fn transitions(&self) -> Vec<&EdlEvent> {
        self.events.iter().filter(|e| e.is_transition()).collect()
    }

    /// Sum of `duration_frames()` across all events.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.events.iter().map(|e| e.duration_frames()).sum()
    }

    /// Total number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return events as a slice.
    #[must_use]
    pub fn events(&self) -> &[EdlEvent] {
        &self.events
    }

    /// Find an event by number.
    #[must_use]
    pub fn find(&self, number: u32) -> Option<&EdlEvent> {
        self.events.iter().find(|e| e.number == number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EditType tests ---

    #[test]
    fn test_cut_is_not_transition() {
        assert!(!EditType::Cut.is_transition());
    }

    #[test]
    fn test_dissolve_is_transition() {
        assert!(EditType::Dissolve.is_transition());
    }

    #[test]
    fn test_wipe_is_transition() {
        assert!(EditType::Wipe.is_transition());
    }

    #[test]
    fn test_key_is_transition() {
        assert!(EditType::Key.is_transition());
    }

    #[test]
    fn test_edit_type_label() {
        assert_eq!(EditType::Cut.label(), "C");
        assert_eq!(EditType::Dissolve.label(), "D");
        assert_eq!(EditType::Wipe.label(), "W");
        assert_eq!(EditType::Key.label(), "K");
    }

    // --- EdlEvent tests ---

    #[test]
    fn test_event_duration_frames() {
        let ev = EdlEvent::new(1, "R001", EditType::Cut, 0, 100, 1000, 1100);
        assert_eq!(ev.duration_frames(), 100);
    }

    #[test]
    fn test_event_duration_zero_when_inverted() {
        let ev = EdlEvent::new(1, "R001", EditType::Cut, 0, 0, 50, 30);
        assert_eq!(ev.duration_frames(), 0);
    }

    #[test]
    fn test_event_is_transition_cut() {
        let ev = EdlEvent::new(1, "R001", EditType::Cut, 0, 25, 0, 25);
        assert!(!ev.is_transition());
    }

    #[test]
    fn test_event_is_transition_dissolve() {
        let ev = EdlEvent::new(2, "R002", EditType::Dissolve, 0, 12, 0, 12);
        assert!(ev.is_transition());
    }

    #[test]
    fn test_event_set_comment() {
        let mut ev = EdlEvent::new(1, "R001", EditType::Cut, 0, 10, 0, 10);
        assert!(ev.comment.is_none());
        ev.set_comment("my note");
        assert_eq!(ev.comment.as_deref(), Some("my note"));
    }

    #[test]
    fn test_event_set_wipe_number() {
        let mut ev = EdlEvent::new(1, "R001", EditType::Wipe, 0, 10, 0, 10);
        ev.set_wipe_number(7);
        assert_eq!(ev.wipe_number, Some(7));
    }

    // --- EdlEventList tests ---

    #[test]
    fn test_list_add_and_len() {
        let mut list = EdlEventList::new();
        assert!(list.is_empty());
        list.add(EdlEvent::new(1, "A", EditType::Cut, 0, 25, 0, 25));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_list_transitions_only() {
        let mut list = EdlEventList::new();
        list.add(EdlEvent::new(1, "A", EditType::Cut, 0, 25, 0, 25));
        list.add(EdlEvent::new(2, "B", EditType::Dissolve, 0, 12, 25, 37));
        list.add(EdlEvent::new(3, "C", EditType::Wipe, 0, 6, 37, 43));
        let t = list.transitions();
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_list_total_duration() {
        let mut list = EdlEventList::new();
        list.add(EdlEvent::new(1, "A", EditType::Cut, 0, 25, 0, 25));
        list.add(EdlEvent::new(2, "B", EditType::Cut, 25, 50, 25, 50));
        assert_eq!(list.total_duration_frames(), 50);
    }

    #[test]
    fn test_list_find() {
        let mut list = EdlEventList::new();
        list.add(EdlEvent::new(42, "X", EditType::Cut, 0, 10, 0, 10));
        assert!(list.find(42).is_some());
        assert!(list.find(1).is_none());
    }
}
