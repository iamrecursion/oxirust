//! HTTP request logging with ring-buffer storage and query helpers.
//!
//! Provides `RequestMethod`, `RequestEntry`, and `RequestLog` for recording
//! per-request data and querying slow or error responses.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── RequestMethod ─────────────────────────────────────────────────────────────

/// HTTP methods recognized by the server.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RequestMethod {
    /// HTTP GET — retrieve a resource.
    Get,
    /// HTTP POST — submit data to create or trigger an action.
    Post,
    /// HTTP PUT — replace a resource entirely.
    Put,
    /// HTTP PATCH — partially update a resource.
    Patch,
    /// HTTP DELETE — remove a resource.
    Delete,
    /// HTTP HEAD — like GET but response body omitted.
    Head,
    /// HTTP OPTIONS — describe communication options.
    Options,
    /// HTTP TRACE — message loop-back test.
    Trace,
    /// HTTP CONNECT — establish a tunnel.
    Connect,
}

impl RequestMethod {
    /// Returns `true` for methods that modify server state
    /// (POST, PUT, PATCH, DELETE).
    pub fn is_write(&self) -> bool {
        matches!(self, Self::Post | Self::Put | Self::Patch | Self::Delete)
    }

    /// Returns `true` for read-only methods (GET, HEAD, OPTIONS).
    pub fn is_read(&self) -> bool {
        matches!(self, Self::Get | Self::Head | Self::Options)
    }

    /// Parse from a string (case-insensitive).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "PATCH" => Some(Self::Patch),
            "DELETE" => Some(Self::Delete),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            "TRACE" => Some(Self::Trace),
            "CONNECT" => Some(Self::Connect),
            _ => None,
        }
    }

    /// Canonical uppercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Trace => "TRACE",
            Self::Connect => "CONNECT",
        }
    }
}

// ── RequestEntry ──────────────────────────────────────────────────────────────

/// A single recorded HTTP request.
#[derive(Debug, Clone)]
pub struct RequestEntry {
    /// Unix timestamp (milliseconds) when the request was received.
    pub timestamp_ms: u64,
    /// HTTP method.
    pub method: RequestMethod,
    /// Request URI path (without query string).
    pub path: String,
    /// HTTP status code returned.
    pub status_code: u16,
    /// Total response time in milliseconds.
    pub elapsed_ms: u64,
    /// Optional remote IP address.
    pub remote_ip: Option<String>,
    /// Optional authenticated subject (user ID).
    pub subject: Option<String>,
    /// Response body size in bytes.
    pub response_bytes: u64,
}

impl RequestEntry {
    /// Create a new entry.  `elapsed_ms` is the end-to-end request duration.
    pub fn new(
        method: RequestMethod,
        path: impl Into<String>,
        status_code: u16,
        elapsed_ms: u64,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            timestamp_ms,
            method,
            path: path.into(),
            status_code,
            elapsed_ms,
            remote_ip: None,
            subject: None,
            response_bytes: 0,
        }
    }

    /// Attach a remote IP address.
    #[must_use]
    pub fn with_remote_ip(mut self, ip: impl Into<String>) -> Self {
        self.remote_ip = Some(ip.into());
        self
    }

    /// Attach an authenticated subject.
    #[must_use]
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Attach response body size.
    #[must_use]
    pub fn with_response_bytes(mut self, bytes: u64) -> Self {
        self.response_bytes = bytes;
        self
    }

    /// Returns the request duration.
    pub fn duration_ms(&self) -> u64 {
        self.elapsed_ms
    }

    /// Returns `true` if the status code indicates success (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status_code)
    }

    /// Returns `true` if the status code indicates a client or server error.
    pub fn is_error(&self) -> bool {
        self.status_code >= 400
    }

    /// Returns `true` if the status code is a 5xx server error.
    pub fn is_server_error(&self) -> bool {
        self.status_code >= 500
    }
}

// ── RequestLog ────────────────────────────────────────────────────────────────

/// Ring-buffer request log with configurable capacity.
pub struct RequestLog {
    /// Internal ring buffer.
    entries: VecDeque<RequestEntry>,
    /// Maximum number of entries retained.
    capacity: usize,
    /// Threshold for "slow" requests (milliseconds).
    slow_threshold_ms: u64,
}

impl RequestLog {
    /// Create a new log with the given capacity and slow-request threshold.
    pub fn new(capacity: usize, slow_threshold_ms: u64) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            slow_threshold_ms,
        }
    }

    /// Record a new request entry. Oldest entry is evicted when at capacity.
    pub fn record(&mut self, entry: RequestEntry) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Return all entries where `elapsed_ms >= slow_threshold_ms`.
    pub fn slow_requests(&self) -> Vec<&RequestEntry> {
        self.entries
            .iter()
            .filter(|e| e.elapsed_ms >= self.slow_threshold_ms)
            .collect()
    }

    /// Return all entries where `status_code >= 400`.
    pub fn error_requests(&self) -> Vec<&RequestEntry> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }

    /// Total number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the last `n` entries, most-recent last.
    pub fn last_n(&self, n: usize) -> Vec<&RequestEntry> {
        let start = self.entries.len().saturating_sub(n);
        self.entries.range(start..).collect()
    }

    /// Average response time across all stored entries, or `0` when empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_elapsed_ms(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let total: u64 = self.entries.iter().map(|e| e.elapsed_ms).sum();
        total as f64 / self.entries.len() as f64
    }

    /// Count entries within the given time window (milliseconds from now).
    pub fn count_within_ms(&self, window_ms: u64) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        let cutoff = now.saturating_sub(window_ms);
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= cutoff)
            .count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(status: u16, elapsed: u64) -> RequestEntry {
        RequestEntry::new(RequestMethod::Get, "/test", status, elapsed)
    }

    fn write_entry(status: u16, elapsed: u64) -> RequestEntry {
        RequestEntry::new(RequestMethod::Post, "/data", status, elapsed)
    }

    // RequestMethod

    #[test]
    fn method_is_write() {
        assert!(RequestMethod::Post.is_write());
        assert!(RequestMethod::Put.is_write());
        assert!(RequestMethod::Patch.is_write());
        assert!(RequestMethod::Delete.is_write());
        assert!(!RequestMethod::Get.is_write());
        assert!(!RequestMethod::Head.is_write());
    }

    #[test]
    fn method_is_read() {
        assert!(RequestMethod::Get.is_read());
        assert!(RequestMethod::Head.is_read());
        assert!(!RequestMethod::Post.is_read());
    }

    #[test]
    fn method_from_str_known() {
        assert_eq!(RequestMethod::from_str("get"), Some(RequestMethod::Get));
        assert_eq!(
            RequestMethod::from_str("DELETE"),
            Some(RequestMethod::Delete)
        );
        assert_eq!(
            RequestMethod::from_str("options"),
            Some(RequestMethod::Options)
        );
    }

    #[test]
    fn method_from_str_unknown() {
        assert_eq!(RequestMethod::from_str("FOOBAR"), None);
    }

    #[test]
    fn method_as_str() {
        assert_eq!(RequestMethod::Get.as_str(), "GET");
        assert_eq!(RequestMethod::Post.as_str(), "POST");
    }

    // RequestEntry

    #[test]
    fn entry_duration_ms() {
        let e = entry(200, 42);
        assert_eq!(e.duration_ms(), 42);
    }

    #[test]
    fn entry_is_success() {
        assert!(entry(200, 10).is_success());
        assert!(entry(204, 10).is_success());
        assert!(!entry(404, 10).is_success());
    }

    #[test]
    fn entry_is_error() {
        assert!(entry(400, 10).is_error());
        assert!(entry(500, 10).is_error());
        assert!(!entry(200, 10).is_error());
    }

    #[test]
    fn entry_is_server_error() {
        assert!(entry(500, 10).is_server_error());
        assert!(!entry(404, 10).is_server_error());
    }

    #[test]
    fn entry_builder_chaining() {
        let e = write_entry(201, 55)
            .with_remote_ip("10.0.0.1")
            .with_subject("alice")
            .with_response_bytes(1024);
        assert_eq!(e.remote_ip.as_deref(), Some("10.0.0.1"));
        assert_eq!(e.subject.as_deref(), Some("alice"));
        assert_eq!(e.response_bytes, 1024);
    }

    // RequestLog

    #[test]
    fn log_record_and_len() {
        let mut log = RequestLog::new(10, 500);
        assert!(log.is_empty());
        log.record(entry(200, 100));
        log.record(entry(404, 200));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn log_evicts_oldest_at_capacity() {
        let mut log = RequestLog::new(3, 500);
        for _ in 0..4 {
            log.record(entry(200, 10));
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn log_slow_requests() {
        let mut log = RequestLog::new(10, 500);
        log.record(entry(200, 100)); // fast
        log.record(entry(200, 600)); // slow
        log.record(entry(200, 500)); // exactly at threshold → slow
        assert_eq!(log.slow_requests().len(), 2);
    }

    #[test]
    fn log_error_requests() {
        let mut log = RequestLog::new(10, 500);
        log.record(entry(200, 10));
        log.record(entry(400, 10));
        log.record(entry(500, 10));
        assert_eq!(log.error_requests().len(), 2);
    }

    #[test]
    fn log_last_n() {
        let mut log = RequestLog::new(10, 500);
        for i in 0..5_u16 {
            log.record(entry(200 + i, 10));
        }
        assert_eq!(log.last_n(3).len(), 3);
        assert_eq!(log.last_n(100).len(), 5);
    }

    #[test]
    fn log_average_elapsed_empty() {
        let log = RequestLog::new(10, 500);
        assert_eq!(log.average_elapsed_ms(), 0.0);
    }

    #[test]
    fn log_average_elapsed() {
        let mut log = RequestLog::new(10, 500);
        log.record(entry(200, 100));
        log.record(entry(200, 200));
        let avg = log.average_elapsed_ms();
        assert!((avg - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn log_count_within_ms_all_recent() {
        let mut log = RequestLog::new(10, 500);
        log.record(entry(200, 10));
        log.record(entry(200, 20));
        // window of 60 seconds — all entries should be within it
        assert_eq!(log.count_within_ms(60_000), 2);
    }
}
