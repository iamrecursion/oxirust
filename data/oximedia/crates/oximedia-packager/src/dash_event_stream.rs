// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! DASH event stream support for timed metadata.
//!
//! DASH event streams allow in-manifest signalling of timed metadata events,
//! such as SCTE-35 splice points, ID3 equivalents, chapter markers, and ad
//! insertion opportunities.  They are described in ISO 23009-1 (DASH) and the
//! DASH-IF Interoperability Points.
//!
//! # Overview
//!
//! A DASH **EventStream** is an `<EventStream>` element embedded inside a
//! `<Period>` element of an MPD.  Each `<Event>` child carries:
//!
//! | Attribute | Description |
//! |-----------|-------------|
//! | `presentationTime` | Relative to the Period start in timescale ticks. |
//! | `duration` | Duration in timescale ticks. |
//! | `id` | An integer identifier unique within the stream. |
//! | `messageData` | Optional base-64 payload (SCTE-35, ID3, …). |
//!
//! # Example
//!
//! ```
//! use oximedia_packager::dash_event_stream::{
//!     DashEvent, DashEventStream, DashEventStreamSet,
//! };
//!
//! let mut stream = DashEventStream::new(
//!     "urn:scte:scte35:2013:bin",
//!     90_000, // 90 kHz timescale
//! );
//! stream.add_event(DashEvent::new(1, 0, Some(270_000)));   // 3-second event at t=0
//! stream.add_event(DashEvent::new(2, 810_000, None));       // point event at t=9s
//!
//! let xml = stream.to_xml_element().expect("xml generation should succeed");
//! assert!(xml.contains("EventStream"));
//! assert!(xml.contains("urn:scte:scte35:2013:bin"));
//! assert!(xml.contains("<Event"));
//! ```

use crate::error::{PackagerError, PackagerResult};
use base64::Engine as _;
use std::time::Duration;

// ---------------------------------------------------------------------------
// DashEvent
// ---------------------------------------------------------------------------

/// A single timed event within a DASH `EventStream`.
#[derive(Debug, Clone)]
pub struct DashEvent {
    /// Integer identifier, unique within the parent `EventStream`.
    pub id: u32,
    /// Presentation time in timescale ticks (relative to Period start).
    pub presentation_time: u64,
    /// Optional event duration in timescale ticks.
    pub duration_ticks: Option<u64>,
    /// Optional binary payload (will be base-64 encoded in XML output).
    pub payload: Vec<u8>,
    /// Optional human-readable message data (used when `payload` is empty).
    pub message_data: Option<String>,
}

impl DashEvent {
    /// Create a new event.
    ///
    /// `presentation_time` and `duration_ticks` are in media timescale units.
    #[must_use]
    pub fn new(id: u32, presentation_time: u64, duration_ticks: Option<u64>) -> Self {
        Self {
            id,
            presentation_time,
            duration_ticks,
            payload: Vec::new(),
            message_data: None,
        }
    }

    /// Set the binary payload.
    #[must_use]
    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    /// Set a text message data string.
    #[must_use]
    pub fn with_message_data(mut self, data: impl Into<String>) -> Self {
        self.message_data = Some(data.into());
        self
    }

    /// Compute the presentation time as a [`Duration`] given a `timescale`.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if `timescale` is zero.
    pub fn presentation_time_duration(&self, timescale: u32) -> PackagerResult<Duration> {
        if timescale == 0 {
            return Err(PackagerError::InvalidConfig(
                "timescale must not be zero".into(),
            ));
        }
        let nanos = (self.presentation_time as u128 * 1_000_000_000) / timescale as u128;
        Ok(Duration::from_nanos(nanos as u64))
    }

    /// Compute the event duration as a [`Duration`] given a `timescale`.
    ///
    /// Returns `None` if no duration is set.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if `timescale` is zero.
    pub fn event_duration(&self, timescale: u32) -> PackagerResult<Option<Duration>> {
        if timescale == 0 {
            return Err(PackagerError::InvalidConfig(
                "timescale must not be zero".into(),
            ));
        }
        let dur = self.duration_ticks.map(|d| {
            let nanos = (d as u128 * 1_000_000_000) / timescale as u128;
            Duration::from_nanos(nanos as u64)
        });
        Ok(dur)
    }

    /// Render this event as an `<Event …/>` XML element string.
    ///
    /// If `payload` is non-empty it is base-64 encoded and emitted as
    /// `messageData`.  Otherwise the `message_data` text field is used.
    #[must_use]
    pub fn to_xml_element(&self) -> String {
        let mut attrs = format!(
            "presentationTime=\"{}\" id=\"{}\"",
            self.presentation_time, self.id
        );

        if let Some(dur) = self.duration_ticks {
            attrs.push_str(&format!(" duration=\"{dur}\""));
        }

        let message_data = if !self.payload.is_empty() {
            Some(base64::engine::general_purpose::STANDARD.encode(&self.payload))
        } else {
            self.message_data.clone()
        };

        match message_data {
            Some(md) => format!("<Event {attrs} messageData=\"{md}\"/>"),
            None => format!("<Event {attrs}/>"),
        }
    }
}

// ---------------------------------------------------------------------------
// DashEventStream
// ---------------------------------------------------------------------------

/// A DASH `EventStream` element, containing timed events for one scheme.
#[derive(Debug, Clone)]
pub struct DashEventStream {
    /// Scheme ID URI identifying the event format
    /// (e.g. `"urn:scte:scte35:2013:bin"`, `"urn:mpeg:dash:event:2012"`).
    pub scheme_id_uri: String,
    /// Optional value attribute for the scheme.
    pub value: Option<String>,
    /// Media timescale (ticks per second).
    pub timescale: u32,
    /// Ordered list of events.
    events: Vec<DashEvent>,
}

impl DashEventStream {
    /// Create a new event stream with the given scheme URI and timescale.
    #[must_use]
    pub fn new(scheme_id_uri: impl Into<String>, timescale: u32) -> Self {
        Self {
            scheme_id_uri: scheme_id_uri.into(),
            value: None,
            timescale,
            events: Vec::new(),
        }
    }

    /// Set an optional scheme value.
    #[must_use]
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Add a single event.
    pub fn add_event(&mut self, event: DashEvent) {
        self.events.push(event);
    }

    /// Add multiple events.
    pub fn add_events(&mut self, events: impl IntoIterator<Item = DashEvent>) {
        self.events.extend(events);
    }

    /// Return all events.
    #[must_use]
    pub fn events(&self) -> &[DashEvent] {
        &self.events
    }

    /// Return the number of events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Return events whose presentation time falls within `[start, end)` ticks.
    #[must_use]
    pub fn events_in_range(&self, start_ticks: u64, end_ticks: u64) -> Vec<&DashEvent> {
        self.events
            .iter()
            .filter(|e| e.presentation_time >= start_ticks && e.presentation_time < end_ticks)
            .collect()
    }

    /// Validate the event stream.
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if the timescale is zero or
    /// the scheme URI is empty.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.timescale == 0 {
            return Err(PackagerError::InvalidConfig(
                "EventStream timescale must not be zero".into(),
            ));
        }
        if self.scheme_id_uri.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "EventStream schemeIdUri must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Render as an `<EventStream …>…</EventStream>` XML element.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is invalid.
    pub fn to_xml_element(&self) -> PackagerResult<String> {
        self.validate()?;

        let mut out = format!(
            "<EventStream schemeIdUri=\"{}\" timescale=\"{}\"",
            self.scheme_id_uri, self.timescale
        );

        if let Some(val) = &self.value {
            out.push_str(&format!(" value=\"{val}\""));
        }

        if self.events.is_empty() {
            out.push_str("/>");
        } else {
            out.push('>');
            for event in &self.events {
                out.push_str(&event.to_xml_element());
            }
            out.push_str("</EventStream>");
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Well-known scheme constructors
// ---------------------------------------------------------------------------

impl DashEventStream {
    /// Create a SCTE-35 binary event stream.
    ///
    /// Uses scheme URI `urn:scte:scte35:2013:bin` with the given timescale.
    #[must_use]
    pub fn scte35(timescale: u32) -> Self {
        Self::new("urn:scte:scte35:2013:bin", timescale)
    }

    /// Create a SCTE-35 XML event stream.
    ///
    /// Uses scheme URI `urn:scte:scte35:2014:xml+bin`.
    #[must_use]
    pub fn scte35_xml(timescale: u32) -> Self {
        Self::new("urn:scte:scte35:2014:xml+bin", timescale)
    }

    /// Create a MPEG-DASH "urn:mpeg:dash:event:2012" event stream
    /// (generic timed metadata).
    #[must_use]
    pub fn mpeg_dash_event(timescale: u32) -> Self {
        Self::new("urn:mpeg:dash:event:2012", timescale)
    }

    /// Create a custom event stream with the given URI.
    #[must_use]
    pub fn custom(scheme_id_uri: impl Into<String>, timescale: u32) -> Self {
        Self::new(scheme_id_uri, timescale)
    }
}

// ---------------------------------------------------------------------------
// DashEventStreamSet
// ---------------------------------------------------------------------------

/// A collection of [`DashEventStream`] instances for a DASH Period.
///
/// Multiple event streams with different scheme URIs may coexist in the same
/// Period (e.g. both SCTE-35 and a custom ad-tracking stream).
#[derive(Debug, Clone, Default)]
pub struct DashEventStreamSet {
    streams: Vec<DashEventStream>,
}

impl DashEventStreamSet {
    /// Create an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream fails validation.
    pub fn add(&mut self, stream: DashEventStream) -> PackagerResult<()> {
        stream.validate()?;
        self.streams.push(stream);
        Ok(())
    }

    /// Return the number of streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Return `true` if there are no streams.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Return all streams.
    #[must_use]
    pub fn streams(&self) -> &[DashEventStream] {
        &self.streams
    }

    /// Find a stream by scheme URI.
    #[must_use]
    pub fn find_by_scheme(&self, scheme_uri: &str) -> Option<&DashEventStream> {
        self.streams.iter().find(|s| s.scheme_id_uri == scheme_uri)
    }

    /// Render all streams as concatenated XML `<EventStream>` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if any stream fails validation.
    pub fn to_xml_elements(&self) -> PackagerResult<String> {
        let mut out = String::new();
        for stream in &self.streams {
            out.push_str(&stream.to_xml_element()?);
        }
        Ok(out)
    }

    /// Total event count across all streams.
    #[must_use]
    pub fn total_event_count(&self) -> usize {
        self.streams.iter().map(|s| s.event_count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- DashEvent ----------------------------------------------------------

    #[test]
    fn test_event_new() {
        let e = DashEvent::new(1, 90_000, Some(270_000));
        assert_eq!(e.id, 1);
        assert_eq!(e.presentation_time, 90_000);
        assert_eq!(e.duration_ticks, Some(270_000));
        assert!(e.payload.is_empty());
        assert!(e.message_data.is_none());
    }

    #[test]
    fn test_event_presentation_time_duration() {
        let e = DashEvent::new(1, 90_000, None);
        let dur = e
            .presentation_time_duration(90_000)
            .expect("should succeed");
        assert_eq!(dur, Duration::from_secs(1));
    }

    #[test]
    fn test_event_presentation_time_duration_zero_timescale() {
        let e = DashEvent::new(1, 90_000, None);
        assert!(e.presentation_time_duration(0).is_err());
    }

    #[test]
    fn test_event_duration_seconds() {
        let e = DashEvent::new(1, 0, Some(270_000));
        let dur = e
            .event_duration(90_000)
            .expect("should succeed")
            .expect("should be some");
        assert_eq!(dur, Duration::from_secs(3));
    }

    #[test]
    fn test_event_no_duration() {
        let e = DashEvent::new(1, 0, None);
        let dur = e.event_duration(90_000).expect("should succeed");
        assert!(dur.is_none());
    }

    #[test]
    fn test_event_xml_minimal() {
        let e = DashEvent::new(5, 180_000, None);
        let xml = e.to_xml_element();
        assert!(xml.starts_with("<Event"));
        assert!(xml.contains("id=\"5\""));
        assert!(xml.contains("presentationTime=\"180000\""));
    }

    #[test]
    fn test_event_xml_with_duration() {
        let e = DashEvent::new(1, 0, Some(90_000));
        let xml = e.to_xml_element();
        assert!(xml.contains("duration=\"90000\""));
    }

    #[test]
    fn test_event_xml_with_payload() {
        let e = DashEvent::new(1, 0, None).with_payload(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        let xml = e.to_xml_element();
        assert!(xml.contains("messageData=\""));
        // base-64 of 0xDEADBEEF
        assert!(xml.contains("3q2+7w=="));
    }

    #[test]
    fn test_event_xml_with_message_data() {
        let e = DashEvent::new(1, 0, None).with_message_data("hello world");
        let xml = e.to_xml_element();
        assert!(xml.contains("messageData=\"hello world\""));
    }

    #[test]
    fn test_event_payload_takes_priority_over_message_data() {
        // payload should override message_data in XML output
        let mut e = DashEvent::new(1, 0, None).with_payload(vec![0xAA, 0xBB]);
        e.message_data = Some("should be ignored".to_string());
        let xml = e.to_xml_element();
        assert!(!xml.contains("should be ignored"));
        assert!(xml.contains("messageData=\""));
    }

    // --- DashEventStream validation ------------------------------------------

    #[test]
    fn test_stream_validate_ok() {
        let stream = DashEventStream::new("urn:scte:scte35:2013:bin", 90_000);
        assert!(stream.validate().is_ok());
    }

    #[test]
    fn test_stream_validate_zero_timescale() {
        let stream = DashEventStream::new("urn:example", 0);
        assert!(stream.validate().is_err());
    }

    #[test]
    fn test_stream_validate_empty_scheme() {
        let stream = DashEventStream::new("", 90_000);
        assert!(stream.validate().is_err());
    }

    // --- DashEventStream XML generation -------------------------------------

    #[test]
    fn test_stream_empty_xml() {
        let stream = DashEventStream::new("urn:mpeg:dash:event:2012", 1000);
        let xml = stream.to_xml_element().expect("should succeed");
        assert!(xml.contains("EventStream"));
        assert!(xml.contains("urn:mpeg:dash:event:2012"));
        assert!(xml.contains("timescale=\"1000\""));
        // Self-closing when empty
        assert!(xml.ends_with("/>"));
    }

    #[test]
    fn test_stream_with_events_xml() {
        let mut stream = DashEventStream::new("urn:scte:scte35:2013:bin", 90_000);
        stream.add_event(DashEvent::new(1, 0, Some(270_000)));
        stream.add_event(DashEvent::new(2, 810_000, None));

        let xml = stream.to_xml_element().expect("should succeed");
        assert!(xml.contains("<EventStream"));
        assert!(xml.contains("</EventStream>"));
        // Count only <Event presentationTime="..." elements (not </EventStream>)
        assert_eq!(xml.matches("<Event presentationTime").count(), 2);
    }

    #[test]
    fn test_stream_with_value_attribute() {
        let stream = DashEventStream::new("urn:scte:scte35:2013:bin", 90_000).with_value("1");
        let xml = stream.to_xml_element().expect("should succeed");
        assert!(xml.contains("value=\"1\""));
    }

    #[test]
    fn test_stream_events_in_range() {
        let mut stream = DashEventStream::scte35(90_000);
        stream.add_event(DashEvent::new(1, 0, None));
        stream.add_event(DashEvent::new(2, 90_000, None));
        stream.add_event(DashEvent::new(3, 270_000, None));

        let found = stream.events_in_range(0, 180_000);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, 1);
        assert_eq!(found[1].id, 2);
    }

    #[test]
    fn test_stream_events_in_range_exclusive_end() {
        let mut stream = DashEventStream::scte35(90_000);
        stream.add_event(DashEvent::new(1, 180_000, None));
        // Exactly at end_ticks — should be excluded
        let found = stream.events_in_range(0, 180_000);
        assert!(found.is_empty());
    }

    // --- Well-known constructors --------------------------------------------

    #[test]
    fn test_scte35_constructor() {
        let s = DashEventStream::scte35(90_000);
        assert_eq!(s.scheme_id_uri, "urn:scte:scte35:2013:bin");
        assert_eq!(s.timescale, 90_000);
    }

    #[test]
    fn test_scte35_xml_constructor() {
        let s = DashEventStream::scte35_xml(90_000);
        assert_eq!(s.scheme_id_uri, "urn:scte:scte35:2014:xml+bin");
    }

    #[test]
    fn test_mpeg_dash_event_constructor() {
        let s = DashEventStream::mpeg_dash_event(1000);
        assert_eq!(s.scheme_id_uri, "urn:mpeg:dash:event:2012");
    }

    #[test]
    fn test_custom_constructor() {
        let s = DashEventStream::custom("urn:example:custom", 48_000);
        assert_eq!(s.scheme_id_uri, "urn:example:custom");
        assert_eq!(s.timescale, 48_000);
    }

    // --- DashEventStreamSet -------------------------------------------------

    #[test]
    fn test_set_empty() {
        let set = DashEventStreamSet::new();
        assert!(set.is_empty());
        assert_eq!(set.stream_count(), 0);
        assert_eq!(set.total_event_count(), 0);
    }

    #[test]
    fn test_set_add_valid_stream() {
        let mut set = DashEventStreamSet::new();
        let stream = DashEventStream::scte35(90_000);
        assert!(set.add(stream).is_ok());
        assert_eq!(set.stream_count(), 1);
    }

    #[test]
    fn test_set_add_invalid_stream_fails() {
        let mut set = DashEventStreamSet::new();
        let bad = DashEventStream::new("", 0); // Both invalid
        assert!(set.add(bad).is_err());
        assert!(set.is_empty());
    }

    #[test]
    fn test_set_find_by_scheme() {
        let mut set = DashEventStreamSet::new();
        set.add(DashEventStream::scte35(90_000))
            .expect("add should succeed");
        set.add(DashEventStream::mpeg_dash_event(1000))
            .expect("add should succeed");

        let found = set.find_by_scheme("urn:scte:scte35:2013:bin");
        assert!(found.is_some());
        assert!(set.find_by_scheme("urn:not:found").is_none());
    }

    #[test]
    fn test_set_total_event_count() {
        let mut set = DashEventStreamSet::new();

        let mut s1 = DashEventStream::scte35(90_000);
        s1.add_event(DashEvent::new(1, 0, None));
        s1.add_event(DashEvent::new(2, 90_000, None));

        let mut s2 = DashEventStream::mpeg_dash_event(1000);
        s2.add_event(DashEvent::new(1, 0, None));

        set.add(s1).expect("should succeed");
        set.add(s2).expect("should succeed");

        assert_eq!(set.total_event_count(), 3);
    }

    #[test]
    fn test_set_to_xml_elements() {
        let mut set = DashEventStreamSet::new();

        let mut stream = DashEventStream::scte35(90_000);
        stream.add_event(DashEvent::new(1, 0, None));
        set.add(stream).expect("should succeed");

        let xml = set.to_xml_elements().expect("should succeed");
        assert!(xml.contains("EventStream"));
        assert!(xml.contains("<Event"));
    }

    #[test]
    fn test_set_to_xml_elements_empty() {
        let set = DashEventStreamSet::new();
        let xml = set.to_xml_elements().expect("should succeed");
        assert!(xml.is_empty());
    }

    #[test]
    fn test_set_streams_accessor() {
        let mut set = DashEventStreamSet::new();
        set.add(DashEventStream::scte35(90_000))
            .expect("should succeed");
        assert_eq!(set.streams().len(), 1);
    }

    #[test]
    fn test_add_events_batch() {
        let mut stream = DashEventStream::scte35(90_000);
        let events: Vec<DashEvent> = (0..5u32)
            .map(|i| DashEvent::new(i, i as u64 * 90_000, None))
            .collect();
        stream.add_events(events);
        assert_eq!(stream.event_count(), 5);
    }
}
