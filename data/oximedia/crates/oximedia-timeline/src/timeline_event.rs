//! Timeline event types for markers, chapters, cue points, and comments.
//!
//! `TimelineEvent` provides a uniform way to attach navigable and
//! informational markers to a timeline, queried by time window or
//! navigation capability.

#![allow(dead_code)]

/// Classifies a timeline event by its intended use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A generic named marker (e.g. editorial notes).
    Marker,
    /// A chapter boundary (navigable by players).
    Chapter,
    /// A cue point used for live insertion or ad-break triggering.
    CuePoint,
    /// A free-text comment attached to a frame.
    Comment,
}

impl EventType {
    /// Returns `true` if events of this type should appear in a
    /// navigation / chapter-jump UI.
    #[must_use]
    pub const fn is_navigable(&self) -> bool {
        matches!(self, Self::Chapter | Self::CuePoint)
    }

    /// Returns a short string label for the event type.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Marker => "marker",
            Self::Chapter => "chapter",
            Self::CuePoint => "cue",
            Self::Comment => "comment",
        }
    }
}

/// A single event attached to a timeline position.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Unique identifier for this event.
    pub id: u64,
    /// The frame at which this event occurs.
    pub frame: u64,
    /// The kind of event.
    pub event_type: EventType,
    /// Human-readable label.
    pub label: String,
    /// Optional duration in frames (for ranged events).  `None` = instantaneous.
    pub duration_frames: Option<u64>,
    /// Optional colour hint (packed RGBA u32).
    pub color: Option<u32>,
}

impl TimelineEvent {
    /// Creates a new instantaneous `TimelineEvent`.
    #[must_use]
    pub fn new(id: u64, frame: u64, event_type: EventType, label: impl Into<String>) -> Self {
        Self {
            id,
            frame,
            event_type,
            label: label.into(),
            duration_frames: None,
            color: None,
        }
    }

    /// Creates a ranged event that spans `duration_frames` frames.
    #[must_use]
    pub fn with_duration(
        id: u64,
        frame: u64,
        duration_frames: u64,
        event_type: EventType,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id,
            frame,
            event_type,
            label: label.into(),
            duration_frames: Some(duration_frames),
            color: None,
        }
    }

    /// Returns `true` if this event is navigable (chapter / cue point).
    #[must_use]
    pub fn is_navigable(&self) -> bool {
        self.event_type.is_navigable()
    }

    /// Returns `true` if this event overlaps the half-open frame window
    /// `[window_start, window_end)`.
    ///
    /// An instantaneous event at frame `f` is considered "in range" if
    /// `f` is within the window.  A ranged event overlaps if any part of
    /// `[frame, frame + duration_frames)` intersects the window.
    #[must_use]
    pub fn is_in_range(&self, window_start: u64, window_end: u64) -> bool {
        let event_end = self
            .duration_frames
            .map_or(self.frame + 1, |d| self.frame + d);
        self.frame < window_end && event_end > window_start
    }

    /// Returns the exclusive end frame of this event.
    ///
    /// For instantaneous events this is `frame + 1`.
    #[must_use]
    pub fn end_frame(&self) -> u64 {
        self.duration_frames
            .map_or(self.frame + 1, |d| self.frame + d)
    }
}

/// An ordered collection of `TimelineEvent` entries.
#[derive(Debug, Default, Clone)]
pub struct EventList {
    events: Vec<TimelineEvent>,
    next_id: u64,
}

impl EventList {
    /// Creates a new, empty `EventList`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
        }
    }

    /// Adds an event, automatically assigning a unique ID.
    ///
    /// Events are kept sorted by frame position.
    pub fn add(&mut self, mut event: TimelineEvent) -> u64 {
        event.id = self.next_id;
        self.next_id += 1;
        self.events.push(event);
        self.events.sort_by_key(|e| e.frame);
        self.next_id - 1
    }

    /// Returns all events whose positions intersect the window
    /// `[window_start, window_end)`.
    #[must_use]
    pub fn in_window(&self, window_start: u64, window_end: u64) -> Vec<&TimelineEvent> {
        self.events
            .iter()
            .filter(|e| e.is_in_range(window_start, window_end))
            .collect()
    }

    /// Returns only navigable events (chapters / cue points) in the list.
    #[must_use]
    pub fn navigable_events(&self) -> Vec<&TimelineEvent> {
        self.events.iter().filter(|e| e.is_navigable()).collect()
    }

    /// Returns the total number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Removes the event with the given `id`, returning `true` on success.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.events.len();
        self.events.retain(|e| e.id != id);
        self.events.len() < before
    }

    /// Returns a reference to all events in frame order.
    #[must_use]
    pub fn all(&self) -> &[TimelineEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_chapter_is_navigable() {
        assert!(EventType::Chapter.is_navigable());
    }

    #[test]
    fn test_event_type_cue_is_navigable() {
        assert!(EventType::CuePoint.is_navigable());
    }

    #[test]
    fn test_event_type_marker_not_navigable() {
        assert!(!EventType::Marker.is_navigable());
    }

    #[test]
    fn test_event_type_comment_not_navigable() {
        assert!(!EventType::Comment.is_navigable());
    }

    #[test]
    fn test_event_type_labels() {
        assert_eq!(EventType::Marker.label(), "marker");
        assert_eq!(EventType::Chapter.label(), "chapter");
        assert_eq!(EventType::CuePoint.label(), "cue");
        assert_eq!(EventType::Comment.label(), "comment");
    }

    #[test]
    fn test_event_new_instantaneous() {
        let ev = TimelineEvent::new(1, 100, EventType::Marker, "test");
        assert_eq!(ev.frame, 100);
        assert!(ev.duration_frames.is_none());
    }

    #[test]
    fn test_event_with_duration() {
        let ev = TimelineEvent::with_duration(1, 50, 24, EventType::Chapter, "intro");
        assert_eq!(ev.end_frame(), 74);
    }

    #[test]
    fn test_event_is_navigable_chapter() {
        let ev = TimelineEvent::new(1, 0, EventType::Chapter, "ch1");
        assert!(ev.is_navigable());
    }

    #[test]
    fn test_event_is_navigable_comment_false() {
        let ev = TimelineEvent::new(1, 0, EventType::Comment, "note");
        assert!(!ev.is_navigable());
    }

    #[test]
    fn test_event_is_in_range_instantaneous() {
        let ev = TimelineEvent::new(1, 100, EventType::Marker, "m");
        assert!(ev.is_in_range(90, 110));
        assert!(!ev.is_in_range(0, 100));
        assert!(!ev.is_in_range(101, 200));
    }

    #[test]
    fn test_event_is_in_range_ranged() {
        let ev = TimelineEvent::with_duration(1, 50, 30, EventType::Chapter, "c");
        // Event spans [50, 80)
        assert!(ev.is_in_range(60, 70));
        assert!(ev.is_in_range(40, 60)); // partial overlap at start
        assert!(ev.is_in_range(75, 90)); // partial overlap at end
        assert!(!ev.is_in_range(80, 100)); // just after
    }

    #[test]
    fn test_event_list_add_assigns_id() {
        let mut list = EventList::new();
        let id = list.add(TimelineEvent::new(0, 10, EventType::Marker, "a"));
        assert_eq!(id, 1);
        let id2 = list.add(TimelineEvent::new(0, 20, EventType::Marker, "b"));
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_event_list_sorted_by_frame() {
        let mut list = EventList::new();
        list.add(TimelineEvent::new(0, 200, EventType::Marker, "late"));
        list.add(TimelineEvent::new(0, 10, EventType::Chapter, "early"));
        let all = list.all();
        assert_eq!(all[0].frame, 10);
        assert_eq!(all[1].frame, 200);
    }

    #[test]
    fn test_event_list_in_window() {
        let mut list = EventList::new();
        list.add(TimelineEvent::new(0, 50, EventType::Marker, "a"));
        list.add(TimelineEvent::new(0, 150, EventType::Marker, "b"));
        let found = list.in_window(0, 100);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].frame, 50);
    }

    #[test]
    fn test_event_list_navigable_events() {
        let mut list = EventList::new();
        list.add(TimelineEvent::new(0, 0, EventType::Chapter, "ch"));
        list.add(TimelineEvent::new(0, 50, EventType::Comment, "note"));
        list.add(TimelineEvent::new(0, 100, EventType::CuePoint, "ad"));
        let nav = list.navigable_events();
        assert_eq!(nav.len(), 2);
    }

    #[test]
    fn test_event_list_remove() {
        let mut list = EventList::new();
        let id = list.add(TimelineEvent::new(0, 10, EventType::Marker, "m"));
        assert!(list.remove(id));
        assert!(list.is_empty());
        assert!(!list.remove(999));
    }

    #[test]
    fn test_event_list_len_and_empty() {
        let list = EventList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }
}
