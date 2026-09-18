//! Chrome Tracing JSON exporter.
//!
//! Produces JSON output compatible with `chrome://tracing` (also known as
//! Perfetto UI).  The format is described at
//! <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU>.
//!
//! ## Supported event phases
//!
//! | Phase | Meaning |
//! |-------|---------|
//! | `B`   | Duration begin |
//! | `E`   | Duration end |
//! | `X`   | Complete event (begin + duration encoded together) |
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::chrome_trace::{ChromeTracingExporter, ChromeTraceEvent, ChromePhase};
//!
//! let mut exporter = ChromeTracingExporter::new();
//! exporter.add_begin("render", "video", 0.0, 1, 1);
//! exporter.add_end("render", "video", 500.0, 1, 1);
//! let json = exporter.to_json().expect("serialisation must not fail");
//! assert!(json.contains("traceEvents"));
//! ```

#![allow(dead_code)]

use crate::span::{Span, SpanTracker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ChromePhase
// ---------------------------------------------------------------------------

/// Chrome Tracing event phase character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromePhase {
    /// Duration event begin (`B`).
    Begin,
    /// Duration event end (`E`).
    End,
    /// Complete event — start + duration in a single record (`X`).
    Complete,
    /// Instant / point-in-time event (`i`).
    Instant,
    /// Counter event (`C`).
    Counter,
    /// Async event begin (`b`).
    AsyncBegin,
    /// Async event end (`e`).
    AsyncEnd,
}

impl ChromePhase {
    /// Returns the single-character string used in the JSON output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Begin => "B",
            Self::End => "E",
            Self::Complete => "X",
            Self::Instant => "i",
            Self::Counter => "C",
            Self::AsyncBegin => "b",
            Self::AsyncEnd => "e",
        }
    }
}

// ---------------------------------------------------------------------------
// ChromeTraceEvent
// ---------------------------------------------------------------------------

/// A single event in Chrome Tracing JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeTraceEvent {
    /// Process id.
    pub pid: u64,
    /// Thread id.
    pub tid: u64,
    /// Timestamp in **microseconds** from the trace start.
    pub ts: f64,
    /// Phase (single character).
    pub ph: String,
    /// Event name.
    pub name: String,
    /// Category (comma-separated list of categories).
    pub cat: String,
    /// Duration in microseconds (only for `X` phase).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    /// Optional key-value arguments.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub args: HashMap<String, String>,
}

impl ChromeTraceEvent {
    /// Creates a begin event.
    #[must_use]
    pub fn begin(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            pid,
            tid,
            ts: ts_us,
            ph: ChromePhase::Begin.as_str().to_owned(),
            name: name.into(),
            cat: cat.into(),
            dur: None,
            args: HashMap::new(),
        }
    }

    /// Creates an end event.
    #[must_use]
    pub fn end(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            pid,
            tid,
            ts: ts_us,
            ph: ChromePhase::End.as_str().to_owned(),
            name: name.into(),
            cat: cat.into(),
            dur: None,
            args: HashMap::new(),
        }
    }

    /// Creates a complete (`X`) event.
    #[must_use]
    pub fn complete(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        dur_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            pid,
            tid,
            ts: ts_us,
            ph: ChromePhase::Complete.as_str().to_owned(),
            name: name.into(),
            cat: cat.into(),
            dur: Some(dur_us),
            args: HashMap::new(),
        }
    }

    /// Adds a string argument.
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ChromeTracingExporter
// ---------------------------------------------------------------------------

/// Builds and serialises Chrome Tracing JSON.
///
/// Events are stored in insertion order; the serialised output wraps them in
/// `{"traceEvents": [...]}` as required by the format spec.
#[derive(Debug, Default)]
pub struct ChromeTracingExporter {
    events: Vec<ChromeTraceEvent>,
}

impl ChromeTracingExporter {
    /// Creates an empty exporter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a pre-built event.
    pub fn push(&mut self, event: ChromeTraceEvent) {
        self.events.push(event);
    }

    /// Adds a `B` (begin) event.
    pub fn add_begin(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(ChromeTraceEvent::begin(name, cat, ts_us, pid, tid));
    }

    /// Adds an `E` (end) event.
    pub fn add_end(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(ChromeTraceEvent::end(name, cat, ts_us, pid, tid));
    }

    /// Adds an `X` (complete) event.
    pub fn add_complete(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        dur_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events.push(ChromeTraceEvent::complete(
            name, cat, ts_us, dur_us, pid, tid,
        ));
    }

    /// Returns the number of events recorded.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns an immutable slice of the recorded events.
    #[must_use]
    pub fn events(&self) -> &[ChromeTraceEvent] {
        &self.events
    }

    /// Serialises to Chrome Tracing JSON.
    ///
    /// Returns `Err` only if `serde_json` encounters an internal failure
    /// (highly unlikely for this simple structure).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct Output<'a> {
            #[serde(rename = "traceEvents")]
            trace_events: &'a [ChromeTraceEvent],
        }

        serde_json::to_string_pretty(&Output {
            trace_events: &self.events,
        })
    }

    /// Clears all stored events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Exports all closed spans from a `SpanTracker` as `X` (complete) events.
    ///
    /// Open spans (no `end_ns`) are silently skipped.
    /// The `pid` and `tid` are taken from the provided values since
    /// `SpanTracker` does not store OS-level ids.
    pub fn export_from_tracker(&mut self, tracker: &SpanTracker, pid: u64, tid: u64) {
        let mut spans = tracker.all_spans();
        // Sort by start time for deterministic ordering.
        spans.sort_by_key(|s| s.start_ns);

        for span in spans {
            if let (Some(end_ns), _) = (span.end_ns, span.start_ns) {
                let ts_us = span.start_ns as f64 / 1_000.0;
                let dur_us = (end_ns.saturating_sub(span.start_ns)) as f64 / 1_000.0;
                let mut event =
                    ChromeTraceEvent::complete(&span.name, "span", ts_us, dur_us, pid, tid);
                event
                    .args
                    .insert("span_id".to_owned(), span.id.value().to_string());
                if let Some(pid_val) = span.parent_id {
                    event
                        .args
                        .insert("parent_id".to_owned(), pid_val.value().to_string());
                }
                self.events.push(event);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: export a slice of Spans directly
// ---------------------------------------------------------------------------

/// Converts a slice of `Span`s to a full Chrome Tracing JSON string.
pub fn spans_to_chrome_json(
    spans: &[Span],
    pid: u64,
    tid: u64,
) -> Result<String, serde_json::Error> {
    let mut exporter = ChromeTracingExporter::new();
    let mut sorted: Vec<&Span> = spans.iter().collect();
    sorted.sort_by_key(|s| s.start_ns);

    for span in sorted {
        if let Some(end_ns) = span.end_ns {
            let ts_us = span.start_ns as f64 / 1_000.0;
            let dur_us = (end_ns.saturating_sub(span.start_ns)) as f64 / 1_000.0;
            exporter.add_complete(&span.name, "span", ts_us, dur_us, pid, tid);
        }
    }
    exporter.to_json()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SpanTracker;
    use std::thread;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // JSON structure tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_exporter_produces_valid_json() {
        let exporter = ChromeTracingExporter::new();
        let json = exporter.to_json().expect("serialisation must not fail");
        assert!(json.contains("traceEvents"));
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("must be valid JSON");
        assert!(parsed["traceEvents"].is_array());
        assert_eq!(parsed["traceEvents"].as_array().expect("array").len(), 0);
    }

    #[test]
    fn test_begin_end_events_in_json() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("encode", "codec", 0.0, 1, 1);
        e.add_end("encode", "codec", 1000.0, 1, 1);
        let json = e.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let events = v["traceEvents"].as_array().expect("array");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["ph"], "B");
        assert_eq!(events[1]["ph"], "E");
    }

    #[test]
    fn test_complete_event_has_dur() {
        let mut e = ChromeTracingExporter::new();
        e.add_complete("decode", "codec", 500.0, 250.0, 1, 1);
        let json = e.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let event = &v["traceEvents"][0];
        assert_eq!(event["ph"], "X");
        assert!((event["dur"].as_f64().expect("dur") - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_event_name_and_category_preserved() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("my_function", "render", 0.0, 2, 3);
        let json = e.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["traceEvents"][0]["name"], "my_function");
        assert_eq!(v["traceEvents"][0]["cat"], "render");
        assert_eq!(v["traceEvents"][0]["pid"], 2);
        assert_eq!(v["traceEvents"][0]["tid"], 3);
    }

    #[test]
    fn test_event_ordering_preserved() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("first", "cat", 0.0, 1, 1);
        e.add_begin("second", "cat", 100.0, 1, 1);
        e.add_end("second", "cat", 200.0, 1, 1);
        e.add_end("first", "cat", 300.0, 1, 1);
        let json = e.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        let events = v["traceEvents"].as_array().expect("array");
        assert_eq!(events[0]["name"], "first");
        assert_eq!(events[1]["name"], "second");
        assert_eq!(events[2]["name"], "second");
        assert_eq!(events[3]["name"], "first");
    }

    #[test]
    fn test_event_count() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("a", "c", 0.0, 1, 1);
        e.add_end("a", "c", 1.0, 1, 1);
        assert_eq!(e.event_count(), 2);
    }

    #[test]
    fn test_clear_resets_events() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("a", "c", 0.0, 1, 1);
        e.clear();
        assert_eq!(e.event_count(), 0);
    }

    #[test]
    fn test_ts_field_is_microseconds() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("fn", "cat", 12345.678, 1, 1);
        let json = e.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        let ts = v["traceEvents"][0]["ts"].as_f64().expect("ts");
        assert!((ts - 12345.678).abs() < 0.001);
    }

    #[test]
    fn test_export_from_tracker_produces_complete_events() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("render_frame");
            thread::sleep(Duration::from_millis(5));
        }
        let mut exporter = ChromeTracingExporter::new();
        exporter.export_from_tracker(&tracker, 1, 1);
        assert_eq!(exporter.event_count(), 1);
        let json = exporter.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["traceEvents"][0]["ph"], "X");
        assert_eq!(v["traceEvents"][0]["name"], "render_frame");
    }

    #[test]
    fn test_export_from_tracker_span_id_in_args() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("some_span");
        }
        let mut exporter = ChromeTracingExporter::new();
        exporter.export_from_tracker(&tracker, 1, 1);
        let json = exporter.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        // args.span_id should be present
        assert!(v["traceEvents"][0]["args"]["span_id"].is_string());
    }

    #[test]
    fn test_chrome_phase_as_str() {
        assert_eq!(ChromePhase::Begin.as_str(), "B");
        assert_eq!(ChromePhase::End.as_str(), "E");
        assert_eq!(ChromePhase::Complete.as_str(), "X");
        assert_eq!(ChromePhase::Instant.as_str(), "i");
    }

    #[test]
    fn test_nested_spans_exported_as_separate_events() {
        let tracker = SpanTracker::new();
        {
            let _outer = tracker.enter("outer");
            {
                let _inner = tracker.enter("inner");
            }
        }
        let mut exporter = ChromeTracingExporter::new();
        exporter.export_from_tracker(&tracker, 1, 1);
        assert_eq!(exporter.event_count(), 2);
        let json = exporter.to_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let names: Vec<&str> = v["traceEvents"]
            .as_array()
            .expect("array")
            .iter()
            .map(|e| e["name"].as_str().expect("name"))
            .collect();
        assert!(names.contains(&"outer"));
        assert!(names.contains(&"inner"));
    }
}
