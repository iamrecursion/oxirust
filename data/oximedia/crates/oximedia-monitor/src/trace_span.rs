//! Distributed trace-span tracking for monitoring request lifecycles.
//!
//! Provides a lightweight span model for measuring latency in media
//! processing pipelines without requiring a full tracing backend.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Completion state of a trace span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// Span is currently running.
    Running,
    /// Span completed successfully.
    Ok,
    /// Span completed with an error.
    Error(String),
    /// Span was cancelled before completion.
    Cancelled,
}

impl SpanStatus {
    /// Returns `true` if the span has reached a terminal state.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Returns `true` if the span finished without error.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Short label for display.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Ok => "ok",
            Self::Error(_) => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// An individual trace span covering a named operation.
#[derive(Debug)]
pub struct TraceSpan {
    /// Unique span id.
    pub id: u64,
    /// Human-readable operation name.
    pub name: String,
    /// When the span started.
    start: Instant,
    /// How long the span ran (set when finished).
    elapsed: Option<Duration>,
    /// Current status.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes.
    pub attributes: HashMap<String, String>,
    /// Threshold in milliseconds above which the span is considered slow.
    pub slow_threshold_ms: u64,
}

impl TraceSpan {
    /// Create a new span with the given name and slow threshold.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, slow_threshold_ms: u64) -> Self {
        Self {
            id,
            name: name.into(),
            start: Instant::now(),
            elapsed: None,
            status: SpanStatus::Running,
            attributes: HashMap::new(),
            slow_threshold_ms,
        }
    }

    /// Duration in milliseconds — available only after the span finishes.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.elapsed.map(|d| d.as_millis() as u64)
    }

    /// Returns `true` if the span finished and exceeded the slow threshold.
    #[must_use]
    pub fn is_slow(&self) -> bool {
        self.duration_ms()
            .is_some_and(|ms| ms > self.slow_threshold_ms)
    }

    /// Finish the span with a given status, recording elapsed time.
    pub fn finish(&mut self, status: SpanStatus) {
        self.elapsed = Some(self.start.elapsed());
        self.status = status;
    }

    /// Attach an attribute to this span.
    pub fn set_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }
}

/// Manages active and finished trace spans.
#[derive(Debug, Default)]
pub struct SpanTracer {
    active: HashMap<u64, TraceSpan>,
    finished: Vec<TraceSpan>,
    next_id: u64,
    default_slow_ms: u64,
}

impl SpanTracer {
    /// Create a tracer with a default slow-threshold in milliseconds.
    #[must_use]
    pub fn new(default_slow_ms: u64) -> Self {
        Self {
            default_slow_ms,
            ..Default::default()
        }
    }

    /// Start a new span, returning its id.
    pub fn start(&mut self, name: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let span = TraceSpan::new(id, name, self.default_slow_ms);
        self.active.insert(id, span);
        id
    }

    /// Finish a span by id with the given status.
    ///
    /// Returns `true` if the span was found and finished.
    pub fn finish(&mut self, id: u64, status: SpanStatus) -> bool {
        if let Some(mut span) = self.active.remove(&id) {
            span.finish(status);
            self.finished.push(span);
            true
        } else {
            false
        }
    }

    /// Number of currently active (running) spans.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// All finished spans.
    #[must_use]
    pub fn finished_spans(&self) -> &[TraceSpan] {
        &self.finished
    }

    /// Finished spans that exceeded the slow threshold.
    #[must_use]
    pub fn slow_spans(&self) -> Vec<&TraceSpan> {
        self.finished.iter().filter(|s| s.is_slow()).collect()
    }

    /// Get a reference to an active span by id.
    #[must_use]
    pub fn get_active(&self, id: u64) -> Option<&TraceSpan> {
        self.active.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_span_status_is_complete() {
        assert!(!SpanStatus::Running.is_complete());
        assert!(SpanStatus::Ok.is_complete());
        assert!(SpanStatus::Error("oops".to_string()).is_complete());
        assert!(SpanStatus::Cancelled.is_complete());
    }

    #[test]
    fn test_span_status_is_ok() {
        assert!(SpanStatus::Ok.is_ok());
        assert!(!SpanStatus::Running.is_ok());
        assert!(!SpanStatus::Error("e".to_string()).is_ok());
        assert!(!SpanStatus::Cancelled.is_ok());
    }

    #[test]
    fn test_span_status_label() {
        assert_eq!(SpanStatus::Running.label(), "running");
        assert_eq!(SpanStatus::Ok.label(), "ok");
        assert_eq!(SpanStatus::Error("x".to_string()).label(), "error");
        assert_eq!(SpanStatus::Cancelled.label(), "cancelled");
    }

    #[test]
    fn test_trace_span_new() {
        let span = TraceSpan::new(1, "encode", 100);
        assert_eq!(span.id, 1);
        assert_eq!(span.name, "encode");
        assert_eq!(span.status, SpanStatus::Running);
        assert!(span.duration_ms().is_none());
        assert!(!span.is_slow());
    }

    #[test]
    fn test_trace_span_finish_records_duration() {
        let mut span = TraceSpan::new(0, "op", 1000);
        span.finish(SpanStatus::Ok);
        assert!(span.duration_ms().is_some());
        assert!(span.status.is_complete());
    }

    #[test]
    fn test_trace_span_is_slow() {
        let mut span = TraceSpan::new(0, "op", 0); // 0 ms threshold — always slow once finished
        thread::sleep(Duration::from_millis(2));
        span.finish(SpanStatus::Ok);
        assert!(span.is_slow());
    }

    #[test]
    fn test_trace_span_not_slow() {
        let mut span = TraceSpan::new(0, "op", 100_000); // very high threshold
        span.finish(SpanStatus::Ok);
        assert!(!span.is_slow());
    }

    #[test]
    fn test_trace_span_set_attr() {
        let mut span = TraceSpan::new(0, "op", 100);
        span.set_attr("codec", "h264");
        assert_eq!(
            span.attributes.get("codec").map(|s| s.as_str()),
            Some("h264")
        );
    }

    #[test]
    fn test_span_tracer_start_and_active_count() {
        let mut tracer = SpanTracer::new(500);
        let _id1 = tracer.start("decode");
        let _id2 = tracer.start("encode");
        assert_eq!(tracer.active_count(), 2);
    }

    #[test]
    fn test_span_tracer_finish_decrements_active() {
        let mut tracer = SpanTracer::new(500);
        let id = tracer.start("mux");
        assert_eq!(tracer.active_count(), 1);
        assert!(tracer.finish(id, SpanStatus::Ok));
        assert_eq!(tracer.active_count(), 0);
        assert_eq!(tracer.finished_spans().len(), 1);
    }

    #[test]
    fn test_span_tracer_finish_missing_returns_false() {
        let mut tracer = SpanTracer::new(100);
        assert!(!tracer.finish(999, SpanStatus::Ok));
    }

    #[test]
    fn test_span_tracer_slow_spans() {
        let mut tracer = SpanTracer::new(0); // 0 ms → every finished span is slow
        let id = tracer.start("slow_op");
        thread::sleep(Duration::from_millis(2));
        tracer.finish(id, SpanStatus::Ok);
        assert_eq!(tracer.slow_spans().len(), 1);
    }

    #[test]
    fn test_span_tracer_get_active() {
        let mut tracer = SpanTracer::new(100);
        let id = tracer.start("check");
        assert!(tracer.get_active(id).is_some());
        tracer.finish(id, SpanStatus::Ok);
        assert!(tracer.get_active(id).is_none());
    }
}
