//! Event detection in video streams.
//!
//! This module provides types and logic for detecting and tracking
//! audio/visual events (cuts, fades, speech starts, silence, etc.)
//! throughout a media timeline.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of an event detected in the media.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A hard cut between shots.
    Cut,
    /// A fade transition (in or out).
    Fade,
    /// A wipe transition.
    Wipe,
    /// A flash frame or bright flash.
    Flash,
    /// Speech begins.
    SpeechStart,
    /// Speech ends.
    SpeechEnd,
    /// Music begins.
    MusicStart,
    /// Silence begins.
    SilenceStart,
}

impl EventType {
    /// Returns true if this event is visual in nature.
    pub fn is_visual(&self) -> bool {
        matches!(self, Self::Cut | Self::Fade | Self::Wipe | Self::Flash)
    }

    /// Returns true if this event is audio in nature.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            Self::SpeechStart | Self::SpeechEnd | Self::MusicStart | Self::SilenceStart
        )
    }
}

/// A single detected event.
#[derive(Debug, Clone)]
pub struct DetectedEvent {
    /// The frame at which this event occurs.
    pub frame: u64,
    /// The type of event detected.
    pub event_type: EventType,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Duration of the event in frames (0 means instantaneous).
    pub duration_frames: u32,
}

impl DetectedEvent {
    /// Create a new detected event.
    pub fn new(frame: u64, event_type: EventType, confidence: f32, duration_frames: u32) -> Self {
        Self {
            frame,
            event_type,
            confidence,
            duration_frames,
        }
    }

    /// Returns true if the confidence is at or above the threshold.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Returns true if the event is instantaneous (zero duration).
    pub fn is_instant(&self) -> bool {
        self.duration_frames == 0
    }
}

/// A timeline of detected events.
#[derive(Debug, Clone, Default)]
pub struct EventTimeline {
    /// All detected events.
    pub events: Vec<DetectedEvent>,
}

impl EventTimeline {
    /// Create an empty event timeline.
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event to the timeline.
    pub fn add(&mut self, event: DetectedEvent) {
        self.events.push(event);
    }

    /// Returns all events whose start frame falls within [start, end].
    pub fn events_in_range(&self, start: u64, end: u64) -> Vec<&DetectedEvent> {
        self.events
            .iter()
            .filter(|e| e.frame >= start && e.frame <= end)
            .collect()
    }

    /// Returns all events of a specific type.
    pub fn events_of_type(&self, t: &EventType) -> Vec<&DetectedEvent> {
        self.events.iter().filter(|e| &e.event_type == t).collect()
    }

    /// Returns the density of events (events per frame) within a rolling window.
    ///
    /// If `window_frames` is 0, returns 0.0 to avoid division by zero.
    pub fn event_density(&self, window_frames: u64) -> f32 {
        if window_frames == 0 || self.events.is_empty() {
            return 0.0;
        }
        self.events.len() as f32 / window_frames as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(frame: u64, etype: EventType, confidence: f32, duration: u32) -> DetectedEvent {
        DetectedEvent::new(frame, etype, confidence, duration)
    }

    #[test]
    fn test_event_type_is_visual() {
        assert!(EventType::Cut.is_visual());
        assert!(EventType::Fade.is_visual());
        assert!(EventType::Wipe.is_visual());
        assert!(EventType::Flash.is_visual());
        assert!(!EventType::SpeechStart.is_visual());
        assert!(!EventType::SilenceStart.is_visual());
    }

    #[test]
    fn test_event_type_is_audio() {
        assert!(!EventType::Cut.is_audio());
        assert!(!EventType::Fade.is_audio());
        assert!(EventType::SpeechStart.is_audio());
        assert!(EventType::SpeechEnd.is_audio());
        assert!(EventType::MusicStart.is_audio());
        assert!(EventType::SilenceStart.is_audio());
    }

    #[test]
    fn test_detected_event_is_confident() {
        let event = make_event(10, EventType::Cut, 0.85, 0);
        assert!(event.is_confident(0.8));
        assert!(event.is_confident(0.85));
        assert!(!event.is_confident(0.9));
    }

    #[test]
    fn test_detected_event_is_instant() {
        let instant = make_event(5, EventType::Cut, 0.9, 0);
        assert!(instant.is_instant());

        let non_instant = make_event(5, EventType::Fade, 0.9, 15);
        assert!(!non_instant.is_instant());
    }

    #[test]
    fn test_timeline_empty() {
        let timeline = EventTimeline::new();
        assert!(timeline.events.is_empty());
        assert!(timeline.events_in_range(0, 1000).is_empty());
        assert!(timeline.events_of_type(&EventType::Cut).is_empty());
        assert_eq!(timeline.event_density(100), 0.0);
    }

    #[test]
    fn test_timeline_add() {
        let mut timeline = EventTimeline::new();
        timeline.add(make_event(10, EventType::Cut, 0.9, 0));
        timeline.add(make_event(50, EventType::Fade, 0.7, 12));
        assert_eq!(timeline.events.len(), 2);
    }

    #[test]
    fn test_timeline_events_in_range() {
        let mut timeline = EventTimeline::new();
        timeline.add(make_event(10, EventType::Cut, 0.9, 0));
        timeline.add(make_event(50, EventType::Fade, 0.7, 12));
        timeline.add(make_event(100, EventType::SpeechStart, 0.95, 0));
        timeline.add(make_event(200, EventType::SilenceStart, 0.8, 0));

        let range = timeline.events_in_range(20, 150);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].frame, 50);
        assert_eq!(range[1].frame, 100);
    }

    #[test]
    fn test_timeline_events_in_range_inclusive() {
        let mut timeline = EventTimeline::new();
        timeline.add(make_event(0, EventType::Cut, 1.0, 0));
        timeline.add(make_event(100, EventType::Cut, 1.0, 0));

        let range = timeline.events_in_range(0, 100);
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_timeline_events_of_type() {
        let mut timeline = EventTimeline::new();
        timeline.add(make_event(10, EventType::Cut, 0.9, 0));
        timeline.add(make_event(30, EventType::Cut, 0.8, 0));
        timeline.add(make_event(60, EventType::Fade, 0.7, 10));
        timeline.add(make_event(90, EventType::SpeechStart, 0.95, 0));

        let cuts = timeline.events_of_type(&EventType::Cut);
        assert_eq!(cuts.len(), 2);

        let fades = timeline.events_of_type(&EventType::Fade);
        assert_eq!(fades.len(), 1);

        let wipes = timeline.events_of_type(&EventType::Wipe);
        assert!(wipes.is_empty());
    }

    #[test]
    fn test_timeline_event_density() {
        let mut timeline = EventTimeline::new();
        for i in 0..10 {
            timeline.add(make_event(i * 10, EventType::Cut, 0.9, 0));
        }
        // 10 events in a 100-frame window => density = 0.1
        let density = timeline.event_density(100);
        assert!((density - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_timeline_event_density_zero_window() {
        let mut timeline = EventTimeline::new();
        timeline.add(make_event(10, EventType::Cut, 0.9, 0));
        assert_eq!(timeline.event_density(0), 0.0);
    }

    #[test]
    fn test_detected_event_fields() {
        let event = make_event(42, EventType::Flash, 0.77, 3);
        assert_eq!(event.frame, 42);
        assert!((event.confidence - 0.77).abs() < 1e-5);
        assert_eq!(event.duration_frames, 3);
        assert_eq!(event.event_type, EventType::Flash);
    }
}
