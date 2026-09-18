//! Timed metadata support for adaptive streaming (ID3, EMSG, SCTE-35, event streams).
//!
//! This module provides types for embedding and managing timed metadata events
//! within media streams, including ID3 tags, MPEG-DASH EMSG boxes, SCTE-35 splice
//! info sections, and HLS `EXT-X-DATERANGE` events.

#![allow(dead_code)]

/// The type of timed metadata carried in a [`TimedMetaEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimedMetaType {
    /// ID3 timed metadata (used in HLS audio/video segments).
    Id3,
    /// MPEG-DASH Event Message box (EMSG).
    Emsg,
    /// SCTE-35 splice information section.
    Scte35,
    /// Generic HLS/DASH event stream entry.
    EventStream,
    /// HLS `EXT-X-DATERANGE` date-range metadata.
    DateRange,
}

impl TimedMetaType {
    /// Return `true` if this metadata type is carried as raw binary data.
    ///
    /// `Id3`, `Emsg`, and `Scte35` are binary; `EventStream` and `DateRange` are textual.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Id3 | Self::Emsg | Self::Scte35)
    }
}

/// A single timed metadata event associated with a presentation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedMetaEvent {
    /// Presentation time of this event in milliseconds.
    pub presentation_time_ms: u64,
    /// Optional duration of this event in milliseconds.  `None` means instant.
    pub duration_ms: Option<u64>,
    /// The type of metadata this event carries.
    pub meta_type: TimedMetaType,
    /// Raw metadata payload.
    pub data: Vec<u8>,
    /// Unique event identifier (scheme-specific).
    pub event_id: String,
}

impl TimedMetaEvent {
    /// Create a new timed metadata event.
    #[must_use]
    pub fn new(
        presentation_time_ms: u64,
        duration_ms: Option<u64>,
        meta_type: TimedMetaType,
        data: Vec<u8>,
        event_id: impl Into<String>,
    ) -> Self {
        Self {
            presentation_time_ms,
            duration_ms,
            meta_type,
            data,
            event_id: event_id.into(),
        }
    }

    /// Return `true` if this event has no duration (i.e., it is a point-in-time event).
    #[must_use]
    pub fn is_instant(&self) -> bool {
        self.duration_ms.is_none()
    }

    /// Return the size of the metadata payload in bytes.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

/// A stream of timed metadata events sharing a common scheme URI.
#[derive(Debug, Clone, Default)]
pub struct TimedMetaStream {
    /// All events in this stream, in insertion order.
    pub events: Vec<TimedMetaEvent>,
    /// Scheme ID URI identifying the metadata format.
    pub scheme_id: String,
}

impl TimedMetaStream {
    /// Create a new timed metadata stream with the given scheme ID.
    #[must_use]
    pub fn new(scheme_id: impl Into<String>) -> Self {
        Self {
            events: Vec::new(),
            scheme_id: scheme_id.into(),
        }
    }

    /// Add an event to the stream.
    pub fn add_event(&mut self, event: TimedMetaEvent) {
        self.events.push(event);
    }

    /// Return all events whose presentation time falls within the window
    /// `[time_ms, time_ms + window_ms)`.
    #[must_use]
    pub fn events_at(&self, time_ms: u64, window_ms: u64) -> Vec<&TimedMetaEvent> {
        let end = time_ms.saturating_add(window_ms);
        self.events
            .iter()
            .filter(|e| e.presentation_time_ms >= time_ms && e.presentation_time_ms < end)
            .collect()
    }

    /// Return the total number of events in the stream.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Return all events of type [`TimedMetaType::Scte35`].
    #[must_use]
    pub fn scte35_events(&self) -> Vec<&TimedMetaEvent> {
        self.events
            .iter()
            .filter(|e| e.meta_type == TimedMetaType::Scte35)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TimedMetaType tests ---

    #[test]
    fn test_id3_is_binary() {
        assert!(TimedMetaType::Id3.is_binary());
    }

    #[test]
    fn test_emsg_is_binary() {
        assert!(TimedMetaType::Emsg.is_binary());
    }

    #[test]
    fn test_scte35_is_binary() {
        assert!(TimedMetaType::Scte35.is_binary());
    }

    #[test]
    fn test_event_stream_not_binary() {
        assert!(!TimedMetaType::EventStream.is_binary());
    }

    #[test]
    fn test_date_range_not_binary() {
        assert!(!TimedMetaType::DateRange.is_binary());
    }

    // --- TimedMetaEvent tests ---

    #[test]
    fn test_event_is_instant_when_no_duration() {
        let event = TimedMetaEvent::new(1000, None, TimedMetaType::Id3, vec![0x01, 0x02], "id3-1");
        assert!(event.is_instant());
    }

    #[test]
    fn test_event_not_instant_when_duration_present() {
        let event = TimedMetaEvent::new(1000, Some(5000), TimedMetaType::Scte35, vec![], "scte-1");
        assert!(!event.is_instant());
    }

    #[test]
    fn test_event_data_size() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let event = TimedMetaEvent::new(0, None, TimedMetaType::Emsg, data, "emsg-1");
        assert_eq!(event.data_size(), 4);
    }

    #[test]
    fn test_event_data_size_empty() {
        let event = TimedMetaEvent::new(0, None, TimedMetaType::DateRange, vec![], "dr-1");
        assert_eq!(event.data_size(), 0);
    }

    // --- TimedMetaStream tests ---

    #[test]
    fn test_stream_new_is_empty() {
        let stream = TimedMetaStream::new("urn:example:id3");
        assert_eq!(stream.event_count(), 0);
    }

    #[test]
    fn test_stream_add_event() {
        let mut stream = TimedMetaStream::new("urn:example:id3");
        stream.add_event(TimedMetaEvent::new(
            1000,
            None,
            TimedMetaType::Id3,
            vec![],
            "e1",
        ));
        assert_eq!(stream.event_count(), 1);
    }

    #[test]
    fn test_stream_events_at_window() {
        let mut stream = TimedMetaStream::new("urn:scte:scte35:2013:bin");
        stream.add_event(TimedMetaEvent::new(
            0,
            None,
            TimedMetaType::Scte35,
            vec![],
            "e1",
        ));
        stream.add_event(TimedMetaEvent::new(
            500,
            None,
            TimedMetaType::Scte35,
            vec![],
            "e2",
        ));
        stream.add_event(TimedMetaEvent::new(
            1500,
            None,
            TimedMetaType::Scte35,
            vec![],
            "e3",
        ));
        let found = stream.events_at(0, 1000);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_stream_events_at_empty_window() {
        let mut stream = TimedMetaStream::new("urn:example");
        stream.add_event(TimedMetaEvent::new(
            5000,
            None,
            TimedMetaType::Id3,
            vec![],
            "e1",
        ));
        let found = stream.events_at(0, 1000);
        assert_eq!(found.len(), 0);
    }

    #[test]
    fn test_stream_scte35_events_filter() {
        let mut stream = TimedMetaStream::new("urn:mixed");
        stream.add_event(TimedMetaEvent::new(
            0,
            None,
            TimedMetaType::Id3,
            vec![],
            "i1",
        ));
        stream.add_event(TimedMetaEvent::new(
            1000,
            None,
            TimedMetaType::Scte35,
            vec![],
            "s1",
        ));
        stream.add_event(TimedMetaEvent::new(
            2000,
            None,
            TimedMetaType::Emsg,
            vec![],
            "em1",
        ));
        stream.add_event(TimedMetaEvent::new(
            3000,
            None,
            TimedMetaType::Scte35,
            vec![],
            "s2",
        ));
        let scte35 = stream.scte35_events();
        assert_eq!(scte35.len(), 2);
        assert_eq!(scte35[0].event_id, "s1");
        assert_eq!(scte35[1].event_id, "s2");
    }

    #[test]
    fn test_stream_event_count_multiple() {
        let mut stream = TimedMetaStream::new("urn:example");
        for i in 0..5u64 {
            stream.add_event(TimedMetaEvent::new(
                i * 1000,
                None,
                TimedMetaType::EventStream,
                vec![],
                format!("e{i}"),
            ));
        }
        assert_eq!(stream.event_count(), 5);
    }

    #[test]
    fn test_stream_scheme_id_stored() {
        let stream = TimedMetaStream::new("urn:mpeg:dash:event:2012");
        assert_eq!(stream.scheme_id, "urn:mpeg:dash:event:2012");
    }
}
