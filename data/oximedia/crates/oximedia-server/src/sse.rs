//! Server-Sent Events (SSE) endpoint support per RFC 6202.
//!
//! This module provides a pure-Rust SSE implementation that formats events
//! according to the W3C Server-Sent Events specification (also known as
//! EventSource / RFC 6202).
//!
//! # Wire Format
//!
//! Each SSE event is a sequence of field lines followed by a blank line:
//!
//! ```text
//! id: 42\n
//! event: update\n
//! data: {"count":1}\n
//! retry: 3000\n
//! \n
//! ```
//!
//! Multi-line data is split across multiple `data:` lines. Comments start with
//! a colon (`:`).
//!
//! # Example
//!
//! ```rust
//! use oximedia_server::sse::{SseEvent, SseEventStream};
//!
//! let event = SseEvent::new("hello world")
//!     .with_id("1")
//!     .with_event_type("message");
//!
//! let bytes = event.to_bytes();
//! let text = String::from_utf8(bytes).unwrap();
//! assert!(text.contains("data: hello world\n"));
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

// ── SseEvent ──────────────────────────────────────────────────────────────────

/// A single Server-Sent Event.
///
/// Fields correspond directly to the SSE wire format fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Optional `id:` field (sets the client's `lastEventId`).
    pub id: Option<String>,
    /// Optional `event:` field (event type; defaults to `"message"`).
    pub event_type: Option<String>,
    /// The `data:` payload. May contain newlines, which are split into
    /// multiple `data:` lines on serialisation.
    pub data: String,
    /// Optional `retry:` field (reconnect delay in milliseconds).
    pub retry: Option<u32>,
    /// Optional comment line (prefix `:` in the wire format).
    pub comment: Option<String>,
}

impl SseEvent {
    /// Creates a new event with only the `data` field set.
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            id: None,
            event_type: None,
            data: data.into(),
            retry: None,
            comment: None,
        }
    }

    /// Sets the `id` field (builder pattern).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the `event` type field (builder pattern).
    #[must_use]
    pub fn with_event_type(mut self, event_type: impl Into<String>) -> Self {
        self.event_type = Some(event_type.into());
        self
    }

    /// Sets the `retry` field in milliseconds (builder pattern).
    #[must_use]
    pub fn with_retry(mut self, retry_ms: u32) -> Self {
        self.retry = Some(retry_ms);
        self
    }

    /// Sets the comment line (builder pattern).
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Serialises this event to RFC 6202 UTF-8 bytes.
    ///
    /// Field ordering: comment → id → event → retry → data → blank line.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_string_repr().into_bytes()
    }

    /// Serialises this event to a UTF-8 string in RFC 6202 format.
    pub fn to_string_repr(&self) -> String {
        let mut out = String::new();

        // Comment line
        if let Some(ref c) = self.comment {
            // Comments may not contain newlines; split if necessary
            for line in c.lines() {
                out.push(':');
                if !line.is_empty() {
                    out.push(' ');
                    out.push_str(line);
                }
                out.push('\n');
            }
        }

        // id field
        if let Some(ref id) = self.id {
            out.push_str("id: ");
            out.push_str(id);
            out.push('\n');
        }

        // event field
        if let Some(ref event_type) = self.event_type {
            out.push_str("event: ");
            out.push_str(event_type);
            out.push('\n');
        }

        // retry field
        if let Some(retry_ms) = self.retry {
            out.push_str("retry: ");
            out.push_str(&retry_ms.to_string());
            out.push('\n');
        }

        // data field — split on \n and \r\n
        for line in self.data.split('\n') {
            // Strip trailing \r if present (handles \r\n)
            let line = line.trim_end_matches('\r');
            out.push_str("data: ");
            out.push_str(line);
            out.push('\n');
        }

        // Terminating blank line
        out.push('\n');
        out
    }
}

// ── SseEventStream ────────────────────────────────────────────────────────────

/// A queue of [`SseEvent`]s awaiting delivery.
///
/// Suitable for use in an HTTP streaming response handler: events are pushed
/// by the application and popped (consumed) by the response writer.
#[derive(Debug, Default)]
pub struct SseEventStream {
    events: VecDeque<SseEvent>,
}

impl SseEventStream {
    /// Creates an empty event stream.
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    /// Pushes an event to the back of the queue.
    pub fn push(&mut self, event: SseEvent) {
        self.events.push_back(event);
    }

    /// Pops the next event from the front of the queue.
    pub fn pop(&mut self) -> Option<SseEvent> {
        self.events.pop_front()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the number of queued events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns a keep-alive heartbeat event (comment-only, no data).
    ///
    /// Browsers and proxies use SSE comments to detect connection liveness.
    pub fn heartbeat() -> SseEvent {
        SseEvent {
            id: None,
            event_type: None,
            data: String::new(),
            retry: None,
            comment: Some("heartbeat".to_string()),
        }
    }

    /// Drains all queued events and serialises them into a single byte buffer.
    pub fn drain_to_bytes(&mut self) -> Vec<u8> {
        let mut buf = Vec::new();
        while let Some(event) = self.pop() {
            buf.extend_from_slice(&event.to_bytes());
        }
        buf
    }

    /// Returns a reference to the front event without removing it.
    pub fn peek(&self) -> Option<&SseEvent> {
        self.events.front()
    }
}

// ── SseFormatter ─────────────────────────────────────────────────────────────

/// Stateless helper for formatting SSE wire-format fragments.
pub struct SseFormatter;

impl SseFormatter {
    /// Formats a complete SSE event to bytes.
    pub fn format_event(event: &SseEvent) -> Vec<u8> {
        event.to_bytes()
    }

    /// Formats a standalone comment line to bytes.
    ///
    /// The comment must not contain newlines; they are stripped.
    pub fn format_comment(comment: &str) -> Vec<u8> {
        let safe = comment.lines().next().unwrap_or("");
        if safe.is_empty() {
            b":\n".to_vec()
        } else {
            format!(": {}\n", safe).into_bytes()
        }
    }

    /// Formats a `retry:` directive to bytes.
    pub fn format_retry(retry_ms: u32) -> Vec<u8> {
        format!("retry: {}\n", retry_ms).into_bytes()
    }

    /// Formats a single `data:` line to bytes (no trailing blank line).
    pub fn format_data_line(data: &str) -> Vec<u8> {
        format!("data: {}\n", data).into_bytes()
    }

    /// Formats a blank line (event terminator) to bytes.
    pub fn format_terminator() -> Vec<u8> {
        b"\n".to_vec()
    }
}

// ── SseEventBuilder ───────────────────────────────────────────────────────────

/// Fluent builder that mirrors [`SseEvent`] construction with validation.
#[derive(Debug, Default)]
pub struct SseEventBuilder {
    id: Option<String>,
    event_type: Option<String>,
    data: Option<String>,
    retry: Option<u32>,
    comment: Option<String>,
}

impl SseEventBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the event ID.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the event type.
    #[must_use]
    pub fn event_type(mut self, et: impl Into<String>) -> Self {
        self.event_type = Some(et.into());
        self
    }

    /// Sets the data payload.
    #[must_use]
    pub fn data(mut self, data: impl Into<String>) -> Self {
        self.data = Some(data.into());
        self
    }

    /// Sets the retry interval in milliseconds.
    #[must_use]
    pub fn retry(mut self, ms: u32) -> Self {
        self.retry = Some(ms);
        self
    }

    /// Sets the comment.
    #[must_use]
    pub fn comment(mut self, c: impl Into<String>) -> Self {
        self.comment = Some(c.into());
        self
    }

    /// Builds the [`SseEvent`].  Data defaults to empty string if not set.
    pub fn build(self) -> SseEvent {
        SseEvent {
            id: self.id,
            event_type: self.event_type,
            data: self.data.unwrap_or_default(),
            retry: self.retry,
            comment: self.comment,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SseEvent::to_bytes / to_string_repr ───────────────────────────────────

    #[test]
    fn test_basic_event_to_bytes() {
        let event = SseEvent::new("hello");
        let text = String::from_utf8(event.to_bytes()).expect("valid utf-8");
        assert!(text.contains("data: hello\n"));
        // Must end with blank line
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_event_with_id() {
        let event = SseEvent::new("payload").with_id("42");
        let text = event.to_string_repr();
        assert!(text.contains("id: 42\n"));
    }

    #[test]
    fn test_event_with_event_type() {
        let event = SseEvent::new("data").with_event_type("update");
        let text = event.to_string_repr();
        assert!(text.contains("event: update\n"));
    }

    #[test]
    fn test_event_with_retry() {
        let event = SseEvent::new("x").with_retry(3000);
        let text = event.to_string_repr();
        assert!(text.contains("retry: 3000\n"));
    }

    #[test]
    fn test_event_with_comment() {
        let event = SseEvent::new("y").with_comment("keep-alive");
        let text = event.to_string_repr();
        assert!(text.contains(": keep-alive\n"));
    }

    #[test]
    fn test_event_all_fields() {
        let event = SseEvent::new("body")
            .with_id("7")
            .with_event_type("ping")
            .with_retry(5000)
            .with_comment("note");
        let text = event.to_string_repr();
        assert!(text.contains("id: 7\n"));
        assert!(text.contains("event: ping\n"));
        assert!(text.contains("retry: 5000\n"));
        assert!(text.contains("data: body\n"));
        assert!(text.contains(": note\n"));
    }

    #[test]
    fn test_event_no_optional_fields() {
        let event = SseEvent::new("minimal");
        let text = event.to_string_repr();
        assert!(!text.contains("id:"));
        assert!(!text.contains("event:"));
        assert!(!text.contains("retry:"));
        // No comment line (comment lines start with ': ')
        assert!(!text.starts_with(':'));
        assert!(text.contains("data: minimal\n"));
    }

    #[test]
    fn test_multiline_data_split() {
        let event = SseEvent::new("line1\nline2\nline3");
        let text = event.to_string_repr();
        assert!(text.contains("data: line1\n"));
        assert!(text.contains("data: line2\n"));
        assert!(text.contains("data: line3\n"));
    }

    #[test]
    fn test_multiline_data_crlf_split() {
        let event = SseEvent::new("alpha\r\nbeta");
        let text = event.to_string_repr();
        assert!(text.contains("data: alpha\n"));
        assert!(text.contains("data: beta\n"));
    }

    #[test]
    fn test_rfc_blank_line_terminator() {
        let event = SseEvent::new("test");
        let text = event.to_string_repr();
        // Event must be terminated by exactly one blank line
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_to_string_repr_equals_from_bytes() {
        let event = SseEvent::new("check").with_id("1").with_event_type("msg");
        let from_str = event.to_string_repr();
        let from_bytes = String::from_utf8(event.to_bytes()).expect("utf-8");
        assert_eq!(from_str, from_bytes);
    }

    // ── SseEventStream ────────────────────────────────────────────────────────

    #[test]
    fn test_stream_push_pop() {
        let mut stream = SseEventStream::new();
        stream.push(SseEvent::new("a"));
        stream.push(SseEvent::new("b"));
        assert_eq!(stream.len(), 2);
        let first = stream.pop().expect("should have event");
        assert_eq!(first.data, "a");
        assert_eq!(stream.len(), 1);
    }

    #[test]
    fn test_stream_is_empty() {
        let mut stream = SseEventStream::new();
        assert!(stream.is_empty());
        stream.push(SseEvent::new("x"));
        assert!(!stream.is_empty());
    }

    #[test]
    fn test_stream_heartbeat() {
        let hb = SseEventStream::heartbeat();
        // Heartbeat is a comment-only event
        assert!(hb.comment.is_some());
        let text = hb.to_string_repr();
        assert!(text.starts_with(':'));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_stream_empty_pop_returns_none() {
        let mut stream = SseEventStream::new();
        assert!(stream.pop().is_none());
    }

    #[test]
    fn test_stream_drain_to_bytes() {
        let mut stream = SseEventStream::new();
        stream.push(SseEvent::new("msg1"));
        stream.push(SseEvent::new("msg2"));
        let bytes = stream.drain_to_bytes();
        assert!(stream.is_empty());
        let text = String::from_utf8(bytes).expect("utf-8");
        assert!(text.contains("data: msg1\n"));
        assert!(text.contains("data: msg2\n"));
    }

    // ── SseFormatter ─────────────────────────────────────────────────────────

    #[test]
    fn test_formatter_format_comment_non_empty() {
        let bytes = SseFormatter::format_comment("ping");
        let text = String::from_utf8(bytes).expect("utf-8");
        assert_eq!(text, ": ping\n");
    }

    #[test]
    fn test_formatter_format_comment_empty() {
        let bytes = SseFormatter::format_comment("");
        let text = String::from_utf8(bytes).expect("utf-8");
        assert_eq!(text, ":\n");
    }

    #[test]
    fn test_formatter_format_retry() {
        let bytes = SseFormatter::format_retry(1500);
        let text = String::from_utf8(bytes).expect("utf-8");
        assert_eq!(text, "retry: 1500\n");
    }

    #[test]
    fn test_formatter_format_terminator() {
        let bytes = SseFormatter::format_terminator();
        assert_eq!(bytes, b"\n");
    }

    // ── SseEventBuilder ───────────────────────────────────────────────────────

    #[test]
    fn test_builder_constructs_event() {
        let event = SseEventBuilder::new()
            .id("99")
            .event_type("click")
            .data("payload")
            .retry(2000)
            .comment("debug")
            .build();
        assert_eq!(event.id.as_deref(), Some("99"));
        assert_eq!(event.event_type.as_deref(), Some("click"));
        assert_eq!(event.data, "payload");
        assert_eq!(event.retry, Some(2000));
        assert!(event.comment.is_some());
    }

    #[test]
    fn test_builder_default_data_is_empty() {
        let event = SseEventBuilder::new().build();
        assert_eq!(event.data, "");
    }
}
