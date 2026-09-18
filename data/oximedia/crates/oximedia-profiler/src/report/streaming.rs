//! Channel-based live streaming reporter for real-time profiling data.
//!
//! Provides a `StreamingReporter` that pushes `ProfilingEvent` values
//! over a standard `std::sync::mpsc` channel so consumers can receive
//! profiling events in real time without any external network deps.

use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, Sender};

/// A single profiling event emitted by `StreamingReporter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingEvent {
    /// Nanosecond timestamp (caller-supplied; use e.g. `Instant::now()` converted).
    pub timestamp_ns: u64,
    /// Classifies what kind of event this is.
    pub event_type: ProfilingEventType,
    /// Human-readable label / span name / metric name.
    pub label: String,
    /// Elapsed nanoseconds for `SpanEnd` events; `None` otherwise.
    pub duration_ns: Option<u64>,
    /// Numeric value for `Counter` / `Gauge` events; `None` otherwise.
    pub value: Option<f64>,
}

/// Discriminant for `ProfilingEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilingEventType {
    /// A named span has begun.
    SpanStart,
    /// A named span has ended (includes measured duration).
    SpanEnd,
    /// A monotonically-increasing counter observation.
    Counter,
    /// An instantaneous gauge reading.
    Gauge,
}

/// A live profiling reporter that pushes events over an `mpsc` channel.
///
/// Create one with [`StreamingReporter::new`], keep the `StreamingReporter`
/// on the producer side, and hand the `Receiver` to any consumer thread.
///
/// # Example
///
/// ```rust
/// use oximedia_profiler::report::streaming::{StreamingReporter, ProfilingEventType};
///
/// let (reporter, rx) = StreamingReporter::new();
/// reporter.span_start("encode", 0);
/// reporter.span_end("encode", 1_000_000, 1_000_000);
/// reporter.record("fps", 2_000_000, ProfilingEventType::Gauge, 60.0);
///
/// let json = StreamingReporter::drain_to_json(&rx);
/// assert!(json.contains("encode"));
/// assert!(json.contains("fps"));
/// ```
pub struct StreamingReporter {
    sender: Sender<ProfilingEvent>,
}

impl StreamingReporter {
    /// Create a new streaming reporter paired with its receiver endpoint.
    ///
    /// Returns `(reporter, receiver)`.  The reporter side is `Send` and may
    /// be cloned via [`clone_sender`](Self::clone_sender) to fan out to
    /// multiple producer threads.
    pub fn new() -> (Self, Receiver<ProfilingEvent>) {
        let (sender, receiver) = channel();
        (Self { sender }, receiver)
    }

    /// Clone the underlying `Sender` so multiple threads can share one channel.
    pub fn clone_sender(&self) -> Sender<ProfilingEvent> {
        self.sender.clone()
    }

    /// Emit a `SpanStart` event.
    ///
    /// `timestamp_ns` should be a monotonic nanosecond counter (e.g.
    /// `Instant::now().elapsed().as_nanos() as u64`).
    pub fn span_start(&self, label: impl Into<String>, timestamp_ns: u64) {
        // Deliberately ignore send errors: the receiver may have been dropped.
        let _ = self.sender.send(ProfilingEvent {
            timestamp_ns,
            event_type: ProfilingEventType::SpanStart,
            label: label.into(),
            duration_ns: None,
            value: None,
        });
    }

    /// Emit a `SpanEnd` event including the measured duration.
    pub fn span_end(&self, label: impl Into<String>, timestamp_ns: u64, duration_ns: u64) {
        let _ = self.sender.send(ProfilingEvent {
            timestamp_ns,
            event_type: ProfilingEventType::SpanEnd,
            label: label.into(),
            duration_ns: Some(duration_ns),
            value: None,
        });
    }

    /// Emit a `Counter` or `Gauge` event with an associated numeric value.
    pub fn record(
        &self,
        label: impl Into<String>,
        timestamp_ns: u64,
        event_type: ProfilingEventType,
        value: f64,
    ) {
        let _ = self.sender.send(ProfilingEvent {
            timestamp_ns,
            event_type,
            label: label.into(),
            duration_ns: None,
            value: Some(value),
        });
    }

    /// Drain all pending events from `receiver` and serialise them to a JSON
    /// array string.
    ///
    /// Non-blocking: only events already in the channel buffer are collected.
    /// Returns `"[]"` when the channel is empty or serialisation fails.
    pub fn drain_to_json(receiver: &Receiver<ProfilingEvent>) -> String {
        let events: Vec<ProfilingEvent> = receiver.try_iter().collect();
        serde_json::to_string(&events).unwrap_or_else(|_| "[]".to_string())
    }
}

impl Default for StreamingReporter {
    /// Creates a default `StreamingReporter`, discarding the `Receiver`.
    ///
    /// Useful when a value is required for a field default but you do not
    /// intend to consume the stream.  Prefer [`StreamingReporter::new`] when
    /// you need the receiver.
    fn default() -> Self {
        let (reporter, _rx) = Self::new();
        reporter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_reporter_span_start_end() {
        let (reporter, rx) = StreamingReporter::new();

        reporter.span_start("my_span", 1_000);
        reporter.span_end("my_span", 2_000, 1_000);

        let json = StreamingReporter::drain_to_json(&rx);

        // Both events must appear in the output.
        assert!(json.contains("my_span"), "label missing: {json}");
        assert!(
            json.contains("SpanStart"),
            "SpanStart event_type missing: {json}"
        );
        assert!(
            json.contains("SpanEnd"),
            "SpanEnd event_type missing: {json}"
        );
        // The duration_ns field of the SpanEnd event should be present.
        assert!(
            json.contains("duration_ns"),
            "duration_ns field missing: {json}"
        );
    }

    #[test]
    fn test_streaming_reporter_counter() {
        let (reporter, rx) = StreamingReporter::new();

        reporter.record("frames_encoded", 5_000, ProfilingEventType::Counter, 42.0);

        let json = StreamingReporter::drain_to_json(&rx);

        assert!(json.contains("frames_encoded"), "label missing: {json}");
        assert!(json.contains("Counter"), "Counter type missing: {json}");
        assert!(json.contains("42"), "value missing: {json}");
    }

    #[test]
    fn test_streaming_reporter_empty() {
        let (_reporter, rx) = StreamingReporter::new();

        let json = StreamingReporter::drain_to_json(&rx);

        assert_eq!(json, "[]", "expected empty JSON array");
    }

    #[test]
    fn test_streaming_reporter_gauge() {
        let (reporter, rx) = StreamingReporter::new();

        reporter.record("cpu_usage", 10_000, ProfilingEventType::Gauge, 75.5);

        let json = StreamingReporter::drain_to_json(&rx);
        assert!(json.contains("cpu_usage"));
        assert!(json.contains("Gauge"));
        assert!(json.contains("75.5"));
    }

    #[test]
    fn test_streaming_reporter_multiple_events_ordering() {
        let (reporter, rx) = StreamingReporter::new();

        reporter.span_start("a", 100);
        reporter.span_start("b", 200);
        reporter.span_end("b", 300, 100);
        reporter.span_end("a", 400, 300);

        let events: Vec<ProfilingEvent> = rx.try_iter().collect();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].label, "a");
        assert_eq!(events[1].label, "b");
        assert_eq!(events[2].label, "b");
        assert_eq!(events[3].label, "a");
    }

    #[test]
    fn test_streaming_reporter_drain_twice() {
        let (reporter, rx) = StreamingReporter::new();

        reporter.span_start("first", 0);
        let json1 = StreamingReporter::drain_to_json(&rx);
        assert!(json1.contains("first"));

        // After drain, channel should be empty.
        let json2 = StreamingReporter::drain_to_json(&rx);
        assert_eq!(json2, "[]");
    }

    #[test]
    fn test_streaming_reporter_clone_sender() {
        let (reporter, rx) = StreamingReporter::new();
        let sender2 = reporter.clone_sender();

        reporter.span_start("from_reporter", 0);
        let _ = sender2.send(ProfilingEvent {
            timestamp_ns: 1,
            event_type: ProfilingEventType::Counter,
            label: "from_sender".to_string(),
            duration_ns: None,
            value: Some(1.0),
        });

        let events: Vec<ProfilingEvent> = rx.try_iter().collect();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_streaming_reporter_default() {
        // Default should not panic.
        let _reporter = StreamingReporter::default();
    }
}
