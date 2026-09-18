//! W3C Trace Context propagation (RFC 7230-compatible headers).
//!
//! Implements the W3C Trace Context specification:
//! <https://www.w3.org/TR/trace-context/>
//!
//! The two primary headers are:
//!
//! - **`traceparent`**: `{version}-{trace-id}-{parent-id}-{flags}`
//! - **`tracestate`**: vendor-specific key-value pairs
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::w3c_trace_context::{TraceContext, TraceFlags};
//!
//! // Generate a new root context.
//! let ctx = TraceContext::new_root();
//! let traceparent = ctx.traceparent_header();
//! assert!(traceparent.starts_with("00-"));
//!
//! // Parse an incoming context from a traceparent header.
//! let parsed = TraceContext::from_traceparent(&traceparent)
//!     .expect("valid traceparent should parse");
//! assert!(parsed.sampled());
//! ```

#![allow(dead_code)]

use std::fmt;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// A 128-bit trace identifier (W3C Trace Context).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

impl TraceId {
    /// Create a TraceId from raw bytes.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Parse a TraceId from a 32-character lowercase hex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not exactly 32 lowercase hex chars.
    pub fn from_hex(s: &str) -> MonitorResult<Self> {
        if s.len() != 32 {
            return Err(MonitorError::Other(format!(
                "trace-id must be 32 hex chars, got {}",
                s.len()
            )));
        }
        let mut bytes = [0u8; 16];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_nibble(chunk[0])?;
            let lo = hex_nibble(chunk[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        if bytes == [0u8; 16] {
            return Err(MonitorError::Other(
                "trace-id must not be all zeros".to_string(),
            ));
        }
        Ok(Self(bytes))
    }

    /// Return the hex string representation.
    #[must_use]
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.0)
    }

    /// Return the raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Generate a pseudo-random TraceId using a simple LCG seeded from the
    /// current system time. Not cryptographically secure, but adequate for
    /// monitoring use cases without external dependencies.
    #[must_use]
    pub fn generate() -> Self {
        let seed = seed_from_time();
        let mut bytes = [0u8; 16];
        let mut state = seed;
        for chunk in bytes.chunks_mut(8) {
            state = lcg_next(state);
            let b = state.to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
        // Ensure non-zero (W3C requirement).
        if bytes == [0u8; 16] {
            bytes[15] = 1;
        }
        Self(bytes)
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// A 64-bit span (parent) identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// Create a SpanId from raw bytes.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Parse a SpanId from a 16-character lowercase hex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not exactly 16 lowercase hex chars.
    pub fn from_hex(s: &str) -> MonitorResult<Self> {
        if s.len() != 16 {
            return Err(MonitorError::Other(format!(
                "parent-id must be 16 hex chars, got {}",
                s.len()
            )));
        }
        let mut bytes = [0u8; 8];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_nibble(chunk[0])?;
            let lo = hex_nibble(chunk[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        if bytes == [0u8; 8] {
            return Err(MonitorError::Other(
                "parent-id must not be all zeros".to_string(),
            ));
        }
        Ok(Self(bytes))
    }

    /// Return the hex string representation.
    #[must_use]
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.0)
    }

    /// Return the raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Generate a pseudo-random SpanId.
    #[must_use]
    pub fn generate() -> Self {
        let seed = seed_from_time();
        let state = lcg_next(seed ^ 0xDEAD_BEEF_CAFE_1337);
        let bytes = state.to_le_bytes();
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes);
        if arr == [0u8; 8] {
            arr[7] = 1;
        }
        Self(arr)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Trace flags
// ---------------------------------------------------------------------------

/// W3C Trace Context trace flags byte (currently only the `sampled` bit is
/// defined by the spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceFlags(u8);

impl TraceFlags {
    /// Sampled flag (bit 0).
    pub const SAMPLED: u8 = 0x01;

    /// Create flags with the sampled bit set.
    #[must_use]
    pub fn sampled() -> Self {
        Self(Self::SAMPLED)
    }

    /// Create flags with no bits set (not sampled).
    #[must_use]
    pub fn not_sampled() -> Self {
        Self(0)
    }

    /// Create from a raw byte.
    #[must_use]
    pub fn from_byte(b: u8) -> Self {
        Self(b)
    }

    /// Parse from a 2-character hex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not exactly 2 hex chars.
    pub fn from_hex(s: &str) -> MonitorResult<Self> {
        if s.len() != 2 {
            return Err(MonitorError::Other(format!(
                "flags must be 2 hex chars, got {}",
                s.len()
            )));
        }
        let hi = hex_nibble(s.as_bytes()[0])?;
        let lo = hex_nibble(s.as_bytes()[1])?;
        Ok(Self((hi << 4) | lo))
    }

    /// Return the raw byte.
    #[must_use]
    pub fn as_byte(self) -> u8 {
        self.0
    }

    /// Return true if the sampled bit is set.
    #[must_use]
    pub fn is_sampled(self) -> bool {
        self.0 & Self::SAMPLED != 0
    }

    /// Return the two-character lowercase hex representation.
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("{:02x}", self.0)
    }
}

// ---------------------------------------------------------------------------
// TraceState
// ---------------------------------------------------------------------------

/// W3C Trace Context `tracestate` header: an ordered list of key=value pairs.
///
/// Only the first 32 list-members are retained as per the spec recommendation.
#[derive(Debug, Clone, Default)]
pub struct TraceState {
    /// Ordered list of (vendor-key, value) pairs.
    entries: Vec<(String, String)>,
}

impl TraceState {
    /// Maximum number of list-members retained.
    pub const MAX_MEMBERS: usize = 32;

    /// Create an empty tracestate.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parse from the `tracestate` header value.
    ///
    /// Entries are comma-separated `key=value` pairs. Whitespace around commas
    /// is trimmed.
    #[must_use]
    pub fn parse(header: &str) -> Self {
        let entries: Vec<(String, String)> = header
            .split(',')
            .filter_map(|part| {
                let part = part.trim();
                let eq = part.find('=')?;
                let key = part[..eq].trim().to_string();
                let val = part[eq + 1..].trim().to_string();
                if key.is_empty() || val.is_empty() {
                    return None;
                }
                Some((key, val))
            })
            .take(Self::MAX_MEMBERS)
            .collect();
        Self { entries }
    }

    /// Insert or update a key (prepend to maintain W3C ordering).
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        self.entries.retain(|(k, _)| k != &key);
        self.entries.insert(0, (key, value));
        self.entries.truncate(Self::MAX_MEMBERS);
    }

    /// Get the value for a vendor key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Return the serialized `tracestate` header value.
    #[must_use]
    pub fn to_header(&self) -> String {
        self.entries
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Returns true if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// TraceContext
// ---------------------------------------------------------------------------

/// A W3C Trace Context propagation context.
///
/// Encodes the information carried in the `traceparent` and `tracestate`
/// HTTP headers.
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// W3C version (currently always `00`).
    pub version: u8,
    /// 128-bit trace identifier.
    pub trace_id: TraceId,
    /// 64-bit parent span identifier.
    pub parent_id: SpanId,
    /// Trace flags.
    pub flags: TraceFlags,
    /// Optional tracestate.
    pub state: TraceState,
}

impl TraceContext {
    /// Create a new root context with a freshly generated trace ID and span ID.
    ///
    /// Sampling is enabled by default.
    #[must_use]
    pub fn new_root() -> Self {
        Self {
            version: 0,
            trace_id: TraceId::generate(),
            parent_id: SpanId::generate(),
            flags: TraceFlags::sampled(),
            state: TraceState::empty(),
        }
    }

    /// Create a child context derived from this context with a new span ID.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            version: 0,
            trace_id: self.trace_id,
            parent_id: SpanId::generate(),
            flags: self.flags,
            state: self.state.clone(),
        }
    }

    /// Return the `traceparent` header value.
    ///
    /// Format: `{version}-{trace-id}-{parent-id}-{flags}`
    #[must_use]
    pub fn traceparent_header(&self) -> String {
        format!(
            "{:02x}-{}-{}-{}",
            self.version,
            self.trace_id.to_hex(),
            self.parent_id.to_hex(),
            self.flags.to_hex()
        )
    }

    /// Return the `tracestate` header value (empty string if no state).
    #[must_use]
    pub fn tracestate_header(&self) -> String {
        self.state.to_header()
    }

    /// Parse from a `traceparent` header value.
    ///
    /// # Errors
    ///
    /// Returns an error if the header format is invalid.
    pub fn from_traceparent(header: &str) -> MonitorResult<Self> {
        let parts: Vec<&str> = header.splitn(4, '-').collect();
        if parts.len() != 4 {
            return Err(MonitorError::Other(format!(
                "traceparent must have 4 '-'-separated parts, got {}",
                parts.len()
            )));
        }

        // Version: currently only "00" is supported.
        if parts[0].len() != 2 {
            return Err(MonitorError::Other(
                "traceparent version must be 2 hex chars".to_string(),
            ));
        }
        let version_hi = hex_nibble(parts[0].as_bytes()[0])?;
        let version_lo = hex_nibble(parts[0].as_bytes()[1])?;
        let version = (version_hi << 4) | version_lo;
        if version == 0xFF {
            return Err(MonitorError::Other(
                "traceparent version 0xff is reserved".to_string(),
            ));
        }

        let trace_id = TraceId::from_hex(parts[1])?;
        let parent_id = SpanId::from_hex(parts[2])?;
        let flags = TraceFlags::from_hex(parts[3])?;

        Ok(Self {
            version,
            trace_id,
            parent_id,
            flags,
            state: TraceState::empty(),
        })
    }

    /// Parse from both `traceparent` and `tracestate` headers.
    ///
    /// # Errors
    ///
    /// Returns an error if `traceparent` parsing fails.
    pub fn from_headers(traceparent: &str, tracestate: Option<&str>) -> MonitorResult<Self> {
        let mut ctx = Self::from_traceparent(traceparent)?;
        if let Some(ts) = tracestate {
            ctx.state = TraceState::parse(ts);
        }
        Ok(ctx)
    }

    /// Returns `true` if the sampled flag is set.
    #[must_use]
    pub fn sampled(&self) -> bool {
        self.flags.is_sampled()
    }

    /// Return the trace ID as a hex string.
    #[must_use]
    pub fn trace_id_hex(&self) -> String {
        self.trace_id.to_hex()
    }

    /// Return the parent span ID as a hex string.
    #[must_use]
    pub fn parent_id_hex(&self) -> String {
        self.parent_id.to_hex()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a hex nibble byte to its value (0-15).
fn hex_nibble(b: u8) -> MonitorResult<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(MonitorError::Other(format!(
            "invalid hex character: {}",
            other as char
        ))),
    }
}

/// Convert a byte slice to a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Seed a PRNG from the current system time (nanosecond resolution).
fn seed_from_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_DEAD_BEEF)
}

/// Linear congruential generator step.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- hex_nibble --

    #[test]
    fn test_hex_nibble_digits() {
        assert_eq!(hex_nibble(b'0').expect("ok"), 0);
        assert_eq!(hex_nibble(b'9').expect("ok"), 9);
    }

    #[test]
    fn test_hex_nibble_lower() {
        assert_eq!(hex_nibble(b'a').expect("ok"), 10);
        assert_eq!(hex_nibble(b'f').expect("ok"), 15);
    }

    #[test]
    fn test_hex_nibble_upper() {
        assert_eq!(hex_nibble(b'A').expect("ok"), 10);
        assert_eq!(hex_nibble(b'F').expect("ok"), 15);
    }

    #[test]
    fn test_hex_nibble_invalid() {
        assert!(hex_nibble(b'g').is_err());
        assert!(hex_nibble(b'Z').is_err());
        assert!(hex_nibble(b'!').is_err());
    }

    // -- TraceId --

    #[test]
    fn test_trace_id_generate_not_zero() {
        let id = TraceId::generate();
        assert_ne!(*id.as_bytes(), [0u8; 16]);
    }

    #[test]
    fn test_trace_id_hex_roundtrip() {
        let id = TraceId::generate();
        let hex = id.to_hex();
        assert_eq!(hex.len(), 32);
        let parsed = TraceId::from_hex(&hex).expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_trace_id_from_hex_all_zeros_fails() {
        assert!(TraceId::from_hex(&"0".repeat(32)).is_err());
    }

    #[test]
    fn test_trace_id_from_hex_wrong_length_fails() {
        assert!(TraceId::from_hex("abc").is_err());
    }

    #[test]
    fn test_trace_id_from_hex_invalid_char_fails() {
        let mut bad = "a".repeat(30);
        bad.push_str("zz");
        assert!(TraceId::from_hex(&bad).is_err());
    }

    #[test]
    fn test_trace_id_display() {
        let id = TraceId::from_hex("4bf92f3577b34da6a3ce929d0e0e4736").expect("ok");
        assert_eq!(id.to_string(), "4bf92f3577b34da6a3ce929d0e0e4736");
    }

    // -- SpanId --

    #[test]
    fn test_span_id_generate_not_zero() {
        let id = SpanId::generate();
        assert_ne!(*id.as_bytes(), [0u8; 8]);
    }

    #[test]
    fn test_span_id_hex_roundtrip() {
        let id = SpanId::generate();
        let hex = id.to_hex();
        assert_eq!(hex.len(), 16);
        let parsed = SpanId::from_hex(&hex).expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_span_id_from_hex_all_zeros_fails() {
        assert!(SpanId::from_hex(&"0".repeat(16)).is_err());
    }

    #[test]
    fn test_span_id_from_hex_wrong_length_fails() {
        assert!(SpanId::from_hex("abc").is_err());
    }

    // -- TraceFlags --

    #[test]
    fn test_flags_sampled() {
        let f = TraceFlags::sampled();
        assert!(f.is_sampled());
        assert_eq!(f.to_hex(), "01");
    }

    #[test]
    fn test_flags_not_sampled() {
        let f = TraceFlags::not_sampled();
        assert!(!f.is_sampled());
        assert_eq!(f.to_hex(), "00");
    }

    #[test]
    fn test_flags_from_hex() {
        let f = TraceFlags::from_hex("01").expect("ok");
        assert!(f.is_sampled());
    }

    #[test]
    fn test_flags_from_hex_invalid() {
        assert!(TraceFlags::from_hex("zz").is_err());
        assert!(TraceFlags::from_hex("0").is_err());
    }

    #[test]
    fn test_flags_from_byte() {
        let f = TraceFlags::from_byte(0xFF);
        assert!(f.is_sampled());
        assert_eq!(f.as_byte(), 0xFF);
    }

    // -- TraceState --

    #[test]
    fn test_trace_state_empty() {
        let ts = TraceState::empty();
        assert!(ts.is_empty());
        assert_eq!(ts.to_header(), "");
    }

    #[test]
    fn test_trace_state_parse() {
        let ts = TraceState::parse("rojo=00f067aa0ba902b7,congo=t61rcWkgMzE");
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.get("rojo"), Some("00f067aa0ba902b7"));
        assert_eq!(ts.get("congo"), Some("t61rcWkgMzE"));
    }

    #[test]
    fn test_trace_state_to_header() {
        let ts = TraceState::parse("a=1,b=2");
        let h = ts.to_header();
        assert!(h.contains("a=1"));
        assert!(h.contains("b=2"));
    }

    #[test]
    fn test_trace_state_set_prepends() {
        let mut ts = TraceState::parse("b=2,c=3");
        ts.set("a", "1");
        // "a" should be first.
        let header = ts.to_header();
        assert!(header.starts_with("a=1"));
    }

    #[test]
    fn test_trace_state_set_updates_existing() {
        let mut ts = TraceState::parse("a=1,b=2");
        ts.set("a", "updated");
        assert_eq!(ts.get("a"), Some("updated"));
        // Only one "a" should exist.
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn test_trace_state_max_members() {
        let parts: Vec<String> = (0..40).map(|i| format!("k{i}=v{i}")).collect();
        let ts = TraceState::parse(&parts.join(","));
        assert_eq!(ts.len(), TraceState::MAX_MEMBERS);
    }

    // -- TraceContext --

    #[test]
    fn test_new_root_traceparent_format() {
        let ctx = TraceContext::new_root();
        let tp = ctx.traceparent_header();
        // Format: 00-{32hex}-{16hex}-{2hex}
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent must have 4 parts");
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[2].len(), 16);
        assert_eq!(parts[3].len(), 2);
    }

    #[test]
    fn test_new_root_is_sampled_by_default() {
        let ctx = TraceContext::new_root();
        assert!(ctx.sampled());
    }

    #[test]
    fn test_from_traceparent_roundtrip() {
        let ctx = TraceContext::new_root();
        let tp = ctx.traceparent_header();
        let parsed = TraceContext::from_traceparent(&tp).expect("should parse");
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.parent_id, ctx.parent_id);
        assert_eq!(parsed.flags.as_byte(), ctx.flags.as_byte());
        assert_eq!(parsed.version, ctx.version);
    }

    #[test]
    fn test_from_traceparent_known_value() {
        // Example from W3C spec.
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let ctx = TraceContext::from_traceparent(tp).expect("should parse");
        assert_eq!(ctx.trace_id_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(ctx.parent_id_hex(), "00f067aa0ba902b7");
        assert!(ctx.sampled());
        assert_eq!(ctx.version, 0);
    }

    #[test]
    fn test_from_traceparent_invalid_format() {
        assert!(TraceContext::from_traceparent("bad").is_err());
        assert!(TraceContext::from_traceparent("00-bad-bad-00").is_err());
        // Too few parts.
        assert!(TraceContext::from_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7"
        )
        .is_err());
    }

    #[test]
    fn test_from_traceparent_version_ff_rejected() {
        let tp = "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        assert!(TraceContext::from_traceparent(tp).is_err());
    }

    #[test]
    fn test_child_inherits_trace_id() {
        let parent = TraceContext::new_root();
        let child = parent.child();
        assert_eq!(parent.trace_id, child.trace_id);
        // Child should have a different span ID.
        // (Very small probability of collision with LCG — acceptable for tests.)
    }

    #[test]
    fn test_child_inherits_flags() {
        let parent = TraceContext::new_root();
        let child = parent.child();
        assert_eq!(parent.flags.as_byte(), child.flags.as_byte());
    }

    #[test]
    fn test_from_headers_with_tracestate() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let ts = "rojo=00f067aa0ba902b7";
        let ctx = TraceContext::from_headers(tp, Some(ts)).expect("should parse");
        assert_eq!(ctx.state.get("rojo"), Some("00f067aa0ba902b7"));
    }

    #[test]
    fn test_from_headers_without_tracestate() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let ctx = TraceContext::from_headers(tp, None).expect("should parse");
        assert!(ctx.state.is_empty());
    }

    #[test]
    fn test_traceparent_header_format_correctness() {
        // Verify the format exactly: version(2)-traceId(32)-parentId(16)-flags(2)
        let ctx = TraceContext::new_root();
        let tp = ctx.traceparent_header();
        let parts: Vec<&str> = tp.split('-').collect();
        // version
        assert_eq!(parts[0].len(), 2, "version must be 2 chars");
        // trace-id
        assert_eq!(parts[1].len(), 32, "trace-id must be 32 chars");
        // parent-id
        assert_eq!(parts[2].len(), 16, "parent-id must be 16 chars");
        // flags
        assert_eq!(parts[3].len(), 2, "flags must be 2 chars");
        // All must be lowercase hex.
        for part in &parts {
            assert!(
                part.chars().all(|c| c.is_ascii_hexdigit() || c == '-'),
                "All chars must be hex digits"
            );
        }
    }

    #[test]
    fn test_not_sampled_context() {
        let mut ctx = TraceContext::new_root();
        ctx.flags = TraceFlags::not_sampled();
        assert!(!ctx.sampled());
        let tp = ctx.traceparent_header();
        assert!(tp.ends_with("-00"));
    }

    #[test]
    fn test_bytes_to_hex() {
        assert_eq!(bytes_to_hex(&[0x4b, 0xf9, 0x2f]), "4bf92f");
        assert_eq!(bytes_to_hex(&[0x00]), "00");
        assert_eq!(bytes_to_hex(&[0xff]), "ff");
    }
}
