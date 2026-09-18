//! W3C Trace Context distributed tracing support.
//!
//! Implements the W3C Trace Context specification:
//! <https://www.w3.org/TR/trace-context/>
//!
//! The `traceparent` header format is:
//! `{version}-{trace-id}-{parent-id}-{trace-flags}`
//!
//! Example: `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`
//!
//! # Usage
//!
//! ```rust
//! use oximedia_server::tracing_ctx::{TraceContext, TraceMiddleware};
//! use std::collections::HashMap;
//!
//! let mut headers = HashMap::new();
//! let ctx = TraceMiddleware::inject_trace(&mut headers);
//! assert!(headers.contains_key("traceparent"));
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// W3C traceparent header name.
pub const TRACEPARENT_HEADER: &str = "traceparent";
/// W3C tracestate header name.
pub const TRACESTATE_HEADER: &str = "tracestate";
/// W3C Trace Context version 0.
pub const TRACE_VERSION: u8 = 0x00;
/// Sampled flag bit.
pub const FLAG_SAMPLED: u8 = 0x01;

// ── TraceContext ──────────────────────────────────────────────────────────────

/// A W3C Trace Context carrying trace and span identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContext {
    /// Trace ID: 32 lowercase hex characters (128-bit).
    pub trace_id: String,
    /// Span ID: 16 lowercase hex characters (64-bit).
    pub span_id: String,
    /// Parent span ID (the span that created this one), 16 hex chars.
    pub parent_span_id: Option<String>,
    /// Trace flags byte (bit 0 = sampled).
    pub flags: u8,
    /// Trace Context version (always 0x00 currently).
    pub version: u8,
    /// Optional vendor-specific tracestate key-value pairs.
    pub tracestate: Vec<(String, String)>,
}

impl TraceContext {
    /// Creates a fresh root trace context (no parent).
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            parent_span_id: None,
            flags: FLAG_SAMPLED,
            version: TRACE_VERSION,
            tracestate: Vec::new(),
        }
    }

    /// Creates a child span from this context.
    ///
    /// The child shares the same `trace_id` but gets a new `span_id`.
    /// The current `span_id` becomes the child's `parent_span_id`.
    pub fn new_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: generate_span_id(),
            parent_span_id: Some(self.span_id.clone()),
            flags: self.flags,
            version: self.version,
            tracestate: self.tracestate.clone(),
        }
    }

    /// Serialises this context to a W3C `traceparent` header value.
    ///
    /// Format: `{version:02x}-{trace_id}-{span_id}-{flags:02x}`
    pub fn to_traceparent(&self) -> String {
        format!(
            "{:02x}-{}-{}-{:02x}",
            self.version, self.trace_id, self.span_id, self.flags
        )
    }

    /// Parses a `traceparent` header value.
    ///
    /// Returns `None` if the format is invalid.
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.splitn(4, '-').collect();
        if parts.len() != 4 {
            return None;
        }
        let version = u8::from_str_radix(parts[0], 16).ok()?;
        let trace_id = parts[1];
        let span_id = parts[2];
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        // Validate lengths
        if trace_id.len() != 32 || span_id.len() != 16 {
            return None;
        }
        // Validate hex characters
        if !is_hex(trace_id) || !is_hex(span_id) {
            return None;
        }
        // All-zeros are forbidden by the spec
        if trace_id == "0".repeat(32) || span_id == "0".repeat(16) {
            return None;
        }

        Some(Self {
            trace_id: trace_id.to_lowercase(),
            span_id: span_id.to_lowercase(),
            parent_span_id: None,
            flags,
            version,
            tracestate: Vec::new(),
        })
    }

    /// Returns `true` if the sampled flag is set.
    pub fn is_sampled(&self) -> bool {
        self.flags & FLAG_SAMPLED != 0
    }

    /// Serialises the `tracestate` entries to the W3C header format.
    ///
    /// Format: `key=value,key2=value2`
    pub fn to_tracestate(&self) -> String {
        self.tracestate
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Adds a tracestate entry (builder pattern).
    #[must_use]
    pub fn with_tracestate(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tracestate.push((key.into(), value.into()));
        self
    }

    /// Sets the sampled flag (builder pattern).
    #[must_use]
    pub fn with_sampled(mut self, sampled: bool) -> Self {
        if sampled {
            self.flags |= FLAG_SAMPLED;
        } else {
            self.flags &= !FLAG_SAMPLED;
        }
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── TraceContextBuilder ───────────────────────────────────────────────────────

/// Fluent builder for constructing a [`TraceContext`] with precise control.
#[derive(Debug, Default)]
pub struct TraceContextBuilder {
    trace_id: Option<String>,
    span_id: Option<String>,
    parent_span_id: Option<String>,
    flags: u8,
    version: u8,
    tracestate: Vec<(String, String)>,
}

impl TraceContextBuilder {
    /// Creates a new builder with default (sampled, version 0) settings.
    pub fn new() -> Self {
        Self {
            flags: FLAG_SAMPLED,
            version: TRACE_VERSION,
            ..Default::default()
        }
    }

    /// Sets a specific trace ID (must be 32 hex chars).
    pub fn trace_id(mut self, id: impl Into<String>) -> Self {
        self.trace_id = Some(id.into());
        self
    }

    /// Sets a specific span ID (must be 16 hex chars).
    pub fn span_id(mut self, id: impl Into<String>) -> Self {
        self.span_id = Some(id.into());
        self
    }

    /// Sets the parent span ID.
    pub fn parent_span_id(mut self, id: impl Into<String>) -> Self {
        self.parent_span_id = Some(id.into());
        self
    }

    /// Sets the sampled flag.
    pub fn sampled(mut self, sampled: bool) -> Self {
        if sampled {
            self.flags |= FLAG_SAMPLED;
        } else {
            self.flags &= !FLAG_SAMPLED;
        }
        self
    }

    /// Adds a tracestate entry.
    pub fn tracestate(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tracestate.push((key.into(), value.into()));
        self
    }

    /// Builds the [`TraceContext`], generating missing IDs.
    pub fn build(self) -> TraceContext {
        TraceContext {
            trace_id: self.trace_id.unwrap_or_else(generate_trace_id),
            span_id: self.span_id.unwrap_or_else(generate_span_id),
            parent_span_id: self.parent_span_id,
            flags: self.flags,
            version: self.version,
            tracestate: self.tracestate,
        }
    }
}

// ── TraceMiddleware ───────────────────────────────────────────────────────────

/// Middleware helper for injecting and extracting W3C Trace Context headers.
pub struct TraceMiddleware;

impl TraceMiddleware {
    /// Injects a new (or refreshed) `traceparent` into `headers` and returns the context.
    ///
    /// If `headers` already contain a valid `traceparent`, a child span is created.
    /// Otherwise a fresh root trace is started.
    pub fn inject_trace(headers: &mut HashMap<String, String>) -> TraceContext {
        let ctx = if let Some(existing) = Self::extract_trace(headers) {
            existing.new_child()
        } else {
            TraceContext::new()
        };
        headers.insert(TRACEPARENT_HEADER.to_string(), ctx.to_traceparent());
        if !ctx.tracestate.is_empty() {
            headers.insert(TRACESTATE_HEADER.to_string(), ctx.to_tracestate());
        }
        ctx
    }

    /// Extracts a [`TraceContext`] from `headers`, if a valid `traceparent` is present.
    pub fn extract_trace(headers: &HashMap<String, String>) -> Option<TraceContext> {
        let value = headers
            .get(TRACEPARENT_HEADER)
            .or_else(|| headers.get("Traceparent"))
            .or_else(|| headers.get("TRACEPARENT"))?;
        let mut ctx = TraceContext::from_traceparent(value)?;
        // Also extract tracestate if present
        if let Some(ts) = headers
            .get(TRACESTATE_HEADER)
            .or_else(|| headers.get("Tracestate"))
        {
            ctx.tracestate = parse_tracestate(ts);
        }
        Some(ctx)
    }

    /// Creates a child span context from the given parent, updating headers.
    pub fn create_child_span(parent: &TraceContext) -> TraceContext {
        parent.new_child()
    }

    /// Propagates trace context into outgoing headers.
    pub fn propagate(ctx: &TraceContext, headers: &mut HashMap<String, String>) {
        headers.insert(TRACEPARENT_HEADER.to_string(), ctx.to_traceparent());
        let ts = ctx.to_tracestate();
        if !ts.is_empty() {
            headers.insert(TRACESTATE_HEADER.to_string(), ts);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Generates a 32-character lowercase hex trace ID using simple PRNG.
fn generate_trace_id() -> String {
    format!("{:016x}{:016x}", pseudo_rand_u64(), pseudo_rand_u64())
}

/// Generates a 16-character lowercase hex span ID.
fn generate_span_id() -> String {
    // Keep retrying until we get a non-zero value (required by spec)
    loop {
        let v = pseudo_rand_u64();
        if v != 0 {
            return format!("{:016x}", v);
        }
    }
}

/// Very lightweight non-cryptographic ID generator using the system clock.
///
/// In production code a proper CSRNG or UUID library should be used;
/// this suffices for unit tests and integration scaffolding.
fn pseudo_rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345) as u64;

    // Mix with thread ID and a counter for better uniqueness
    let tid = {
        // Use a simple thread-local counter
        thread_local! {
            static COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
        }
        COUNTER.with(|c| {
            let v = c.get();
            c.set(v.wrapping_add(0x9e37_79b9_7f4a_7c15));
            v
        })
    };

    // xorshift64 mix
    let mut x = nanos.wrapping_add(tid).wrapping_add(0x1234_5678_9abc_def0);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Returns `true` if all characters in `s` are ASCII hexadecimal digits.
fn is_hex(s: &str) -> bool {
    s.bytes().all(|b: u8| b.is_ascii_hexdigit())
}

/// Parses a `tracestate` header value into key-value pairs.
fn parse_tracestate(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            let eq = entry.find('=')?;
            let key = entry[..eq].trim().to_string();
            let val = entry[eq + 1..].trim().to_string();
            if key.is_empty() {
                None
            } else {
                Some((key, val))
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TraceContext::new ─────────────────────────────────────────────────────

    #[test]
    fn test_new_generates_valid_trace_id() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.trace_id.len(), 32, "trace_id must be 32 hex chars");
        assert!(is_hex(&ctx.trace_id), "trace_id must be hex");
    }

    #[test]
    fn test_new_generates_valid_span_id() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.span_id.len(), 16, "span_id must be 16 hex chars");
        assert!(is_hex(&ctx.span_id), "span_id must be hex");
    }

    #[test]
    fn test_new_has_no_parent_span() {
        let ctx = TraceContext::new();
        assert!(ctx.parent_span_id.is_none());
    }

    #[test]
    fn test_new_is_sampled_by_default() {
        let ctx = TraceContext::new();
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_new_has_version_zero() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.version, 0x00);
    }

    // ── to_traceparent ────────────────────────────────────────────────────────

    #[test]
    fn test_to_traceparent_format() {
        let ctx = TraceContextBuilder::new()
            .trace_id("4bf92f3577b34da6a3ce929d0e0e4736")
            .span_id("00f067aa0ba902b7")
            .sampled(true)
            .build();
        assert_eq!(
            ctx.to_traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        );
    }

    #[test]
    fn test_to_traceparent_unsampled() {
        let ctx = TraceContextBuilder::new()
            .trace_id("4bf92f3577b34da6a3ce929d0e0e4736")
            .span_id("00f067aa0ba902b7")
            .sampled(false)
            .build();
        assert!(ctx.to_traceparent().ends_with("-00"));
    }

    // ── from_traceparent ──────────────────────────────────────────────────────

    #[test]
    fn test_from_traceparent_valid() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let ctx = TraceContext::from_traceparent(tp).expect("should parse");
        assert_eq!(ctx.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(ctx.span_id, "00f067aa0ba902b7");
        assert_eq!(ctx.flags, 0x01);
        assert_eq!(ctx.version, 0x00);
    }

    #[test]
    fn test_from_traceparent_invalid_too_few_parts() {
        assert!(TraceContext::from_traceparent("00-abc-def").is_none());
    }

    #[test]
    fn test_from_traceparent_invalid_trace_id_length() {
        // trace_id only 8 chars
        assert!(TraceContext::from_traceparent("00-4bf92f35-00f067aa0ba902b7-01").is_none());
    }

    #[test]
    fn test_from_traceparent_invalid_span_id_length() {
        // span_id only 8 chars
        assert!(
            TraceContext::from_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa-01")
                .is_none()
        );
    }

    #[test]
    fn test_from_traceparent_invalid_hex() {
        let tp = "00-GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG-00f067aa0ba902b7-01";
        assert!(TraceContext::from_traceparent(tp).is_none());
    }

    #[test]
    fn test_from_traceparent_all_zero_trace_id_rejected() {
        let tp = "00-00000000000000000000000000000000-00f067aa0ba902b7-01";
        assert!(TraceContext::from_traceparent(tp).is_none());
    }

    // ── Round-trip ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_from_traceparent_roundtrip() {
        let ctx = TraceContext::new();
        let tp = ctx.to_traceparent();
        let restored = TraceContext::from_traceparent(&tp).expect("round-trip should succeed");
        assert_eq!(restored.trace_id, ctx.trace_id);
        assert_eq!(restored.span_id, ctx.span_id);
        assert_eq!(restored.flags, ctx.flags);
    }

    // ── Child span ────────────────────────────────────────────────────────────

    #[test]
    fn test_new_child_inherits_trace_id() {
        let parent = TraceContext::new();
        let child = parent.new_child();
        assert_eq!(child.trace_id, parent.trace_id);
    }

    #[test]
    fn test_new_child_has_new_span_id() {
        let parent = TraceContext::new();
        let child = parent.new_child();
        assert_ne!(child.span_id, parent.span_id);
    }

    #[test]
    fn test_new_child_parent_span_id_set() {
        let parent = TraceContext::new();
        let child = parent.new_child();
        assert_eq!(
            child.parent_span_id.as_deref(),
            Some(parent.span_id.as_str())
        );
    }

    // ── TraceMiddleware ───────────────────────────────────────────────────────

    #[test]
    fn test_inject_trace_sets_traceparent_header() {
        let mut headers = HashMap::new();
        TraceMiddleware::inject_trace(&mut headers);
        assert!(headers.contains_key(TRACEPARENT_HEADER));
    }

    #[test]
    fn test_extract_trace_round_trip() {
        let mut headers = HashMap::new();
        let injected = TraceMiddleware::inject_trace(&mut headers);
        let extracted = TraceMiddleware::extract_trace(&headers).expect("should extract");
        // The extracted context's span_id matches the injected one
        assert_eq!(extracted.span_id, injected.span_id);
        assert_eq!(extracted.trace_id, injected.trace_id);
    }

    #[test]
    fn test_extract_trace_absent_returns_none() {
        let headers = HashMap::new();
        assert!(TraceMiddleware::extract_trace(&headers).is_none());
    }

    #[test]
    fn test_inject_on_existing_trace_creates_child() {
        let mut headers = HashMap::new();
        let parent = TraceMiddleware::inject_trace(&mut headers);
        // Second inject should create a child span
        let child = TraceMiddleware::inject_trace(&mut headers);
        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
    }

    // ── Tracestate ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_tracestate_empty() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.to_tracestate(), "");
    }

    #[test]
    fn test_to_tracestate_with_entries() {
        let ctx = TraceContext::new()
            .with_tracestate("vendor", "value1")
            .with_tracestate("other", "value2");
        let ts = ctx.to_tracestate();
        assert!(ts.contains("vendor=value1"));
        assert!(ts.contains("other=value2"));
    }

    // ── TraceContextBuilder ───────────────────────────────────────────────────

    #[test]
    fn test_builder_sets_all_fields() {
        let ctx = TraceContextBuilder::new()
            .trace_id("aaaabbbbccccddddaaaabbbbccccdddd")
            .span_id("1111222233334444")
            .parent_span_id("5555666677778888")
            .sampled(false)
            .tracestate("k", "v")
            .build();
        assert_eq!(ctx.trace_id, "aaaabbbbccccddddaaaabbbbccccdddd");
        assert_eq!(ctx.span_id, "1111222233334444");
        assert_eq!(ctx.parent_span_id.as_deref(), Some("5555666677778888"));
        assert!(!ctx.is_sampled());
        assert_eq!(ctx.tracestate.len(), 1);
    }

    #[test]
    fn test_create_child_span_via_middleware() {
        let parent = TraceContext::new();
        let child = TraceMiddleware::create_child_span(&parent);
        assert_eq!(child.trace_id, parent.trace_id);
        assert_eq!(
            child.parent_span_id.as_deref(),
            Some(parent.span_id.as_str())
        );
    }
}
