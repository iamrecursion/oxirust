//! Structured access log for the OxiMedia server.
//!
//! Records HTTP request/response pairs with timing, method, path,
//! status code, and byte counts.  Provides filtering, summary statistics,
//! and serialisation helpers.

#![allow(dead_code)]
#![allow(missing_docs)]

/// A single access log entry capturing one HTTP exchange.
#[derive(Debug, Clone)]
pub struct AccessEntry {
    /// Unique request identifier.
    pub request_id: String,
    /// Remote client IP address.
    pub remote_ip: String,
    /// HTTP method (e.g. "GET").
    pub method: String,
    /// Request path (e.g. "/api/v1/media").
    pub path: String,
    /// HTTP status code returned.
    pub status: u16,
    /// Response body size in bytes.
    pub response_bytes: u64,
    /// Total request handling time in milliseconds.
    pub duration_ms: f64,
    /// Timestamp (ms since epoch) when the request was received.
    pub timestamp_ms: u64,
    /// Optional authenticated user identifier.
    pub user_id: Option<String>,
    /// Optional referrer header value.
    pub referrer: Option<String>,
    /// Optional user-agent header value.
    pub user_agent: Option<String>,
}

impl AccessEntry {
    /// Creates a minimal access entry.
    pub fn new(
        request_id: impl Into<String>,
        remote_ip: impl Into<String>,
        method: impl Into<String>,
        path: impl Into<String>,
        status: u16,
        response_bytes: u64,
        duration_ms: f64,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            remote_ip: remote_ip.into(),
            method: method.into(),
            path: path.into(),
            status,
            response_bytes,
            duration_ms,
            timestamp_ms,
            user_id: None,
            referrer: None,
            user_agent: None,
        }
    }

    /// Attaches an authenticated user identifier.
    #[must_use]
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Returns `true` when the status code indicates a client error (4xx).
    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.status)
    }

    /// Returns `true` when the status code indicates a server error (5xx).
    pub fn is_server_error(&self) -> bool {
        self.status >= 500
    }

    /// Returns `true` when the status code indicates success (2xx).
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Formats the entry as a CLF-like (Combined Log Format) string.
    ///
    /// `<remote_ip> - <user> [<timestamp_ms>] "<method> <path>" <status> <bytes>`
    pub fn to_clf(&self) -> String {
        let user = self.user_id.as_deref().unwrap_or("-");
        format!(
            r#"{} - {} [{}] "{} {}" {} {}"#,
            self.remote_ip,
            user,
            self.timestamp_ms,
            self.method,
            self.path,
            self.status,
            self.response_bytes,
        )
    }
}

/// Summary statistics computed from a collection of access entries.
#[derive(Debug, Clone, Default)]
pub struct AccessSummary {
    /// Total number of requests.
    pub total_requests: u64,
    /// Number of 2xx responses.
    pub successful: u64,
    /// Number of 4xx responses.
    pub client_errors: u64,
    /// Number of 5xx responses.
    pub server_errors: u64,
    /// Total bytes sent in responses.
    pub total_bytes: u64,
    /// Average response duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Slowest request duration in milliseconds.
    pub max_duration_ms: f64,
    /// Fastest request duration in milliseconds.
    pub min_duration_ms: f64,
}

impl AccessSummary {
    /// Returns the error rate as a fraction in `[0.0, 1.0]`.
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.client_errors + self.server_errors) as f64 / self.total_requests as f64
    }
}

/// A filter applied when querying the access log.
#[derive(Debug, Clone, Default)]
pub struct AccessFilter {
    /// Restrict to entries with the given HTTP method.
    pub method: Option<String>,
    /// Restrict to entries with status codes in this range (inclusive).
    pub status_range: Option<(u16, u16)>,
    /// Restrict to entries at or after this timestamp (ms).
    pub since_ms: Option<u64>,
    /// Restrict to entries at or before this timestamp (ms).
    pub until_ms: Option<u64>,
    /// Restrict to entries from this remote IP.
    pub remote_ip: Option<String>,
}

impl AccessFilter {
    /// Creates an empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Tests whether an entry satisfies all filter conditions.
    pub fn matches(&self, entry: &AccessEntry) -> bool {
        if let Some(ref m) = self.method {
            if !entry.method.eq_ignore_ascii_case(m) {
                return false;
            }
        }
        if let Some((lo, hi)) = self.status_range {
            if entry.status < lo || entry.status > hi {
                return false;
            }
        }
        if let Some(since) = self.since_ms {
            if entry.timestamp_ms < since {
                return false;
            }
        }
        if let Some(until) = self.until_ms {
            if entry.timestamp_ms > until {
                return false;
            }
        }
        if let Some(ref ip) = self.remote_ip {
            if entry.remote_ip != *ip {
                return false;
            }
        }
        true
    }
}

/// A bounded ring buffer that stores recent access log entries.
#[derive(Debug)]
pub struct AccessLog {
    entries: Vec<AccessEntry>,
    /// Maximum number of entries to retain.
    capacity: usize,
    /// Index of the next write position.
    head: usize,
    /// Total entries ever appended (for statistics).
    total_appended: u64,
}

impl AccessLog {
    /// Creates a new access log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            total_appended: 0,
        }
    }

    /// Appends an entry to the log.  When full, the oldest entry is overwritten.
    pub fn append(&mut self, entry: AccessEntry) {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            self.entries[self.head] = entry;
        }
        self.head = (self.head + 1) % self.capacity;
        self.total_appended += 1;
    }

    /// Returns the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the total number of entries ever appended (including overwritten ones).
    pub fn total_appended(&self) -> u64 {
        self.total_appended
    }

    /// Queries entries that satisfy `filter`, sorted oldest-first by `timestamp_ms`.
    pub fn query(&self, filter: &AccessFilter) -> Vec<&AccessEntry> {
        let mut results: Vec<&AccessEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();
        results.sort_by_key(|e| e.timestamp_ms);
        results
    }

    /// Computes summary statistics over all stored entries that match `filter`.
    pub fn summarize(&self, filter: &AccessFilter) -> AccessSummary {
        let entries: Vec<&AccessEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();
        if entries.is_empty() {
            return AccessSummary::default();
        }
        let total = entries.len() as u64;
        let successful = entries.iter().filter(|e| e.is_success()).count() as u64;
        let client_errors = entries.iter().filter(|e| e.is_client_error()).count() as u64;
        let server_errors = entries.iter().filter(|e| e.is_server_error()).count() as u64;
        let total_bytes: u64 = entries.iter().map(|e| e.response_bytes).sum();
        let sum_dur: f64 = entries.iter().map(|e| e.duration_ms).sum();
        let max_dur = entries
            .iter()
            .map(|e| e.duration_ms)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_dur = entries
            .iter()
            .map(|e| e.duration_ms)
            .fold(f64::INFINITY, f64::min);

        AccessSummary {
            total_requests: total,
            successful,
            client_errors,
            server_errors,
            total_bytes,
            avg_duration_ms: sum_dur / total as f64,
            max_duration_ms: max_dur,
            min_duration_ms: min_dur,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(request_id: &str, status: u16, duration_ms: f64, ts: u64) -> AccessEntry {
        AccessEntry::new(
            request_id,
            "127.0.0.1",
            "GET",
            "/api/v1/test",
            status,
            256,
            duration_ms,
            ts,
        )
    }

    // --- AccessEntry ---

    #[test]
    fn test_entry_is_success() {
        assert!(make_entry("r1", 200, 10.0, 0).is_success());
        assert!(make_entry("r2", 201, 10.0, 0).is_success());
        assert!(!make_entry("r3", 400, 10.0, 0).is_success());
    }

    #[test]
    fn test_entry_is_client_error() {
        assert!(make_entry("r1", 404, 5.0, 0).is_client_error());
        assert!(!make_entry("r2", 500, 5.0, 0).is_client_error());
    }

    #[test]
    fn test_entry_is_server_error() {
        assert!(make_entry("r1", 500, 5.0, 0).is_server_error());
        assert!(make_entry("r2", 503, 5.0, 0).is_server_error());
        assert!(!make_entry("r3", 200, 5.0, 0).is_server_error());
    }

    #[test]
    fn test_entry_clf_format() {
        let e = make_entry("r1", 200, 10.0, 1_000_000);
        let clf = e.to_clf();
        assert!(clf.contains("127.0.0.1"));
        assert!(clf.contains("200"));
        assert!(clf.contains("GET"));
    }

    #[test]
    fn test_entry_with_user() {
        let e = make_entry("r1", 200, 10.0, 0).with_user("alice");
        let clf = e.to_clf();
        assert!(clf.contains("alice"));
    }

    // --- AccessFilter ---

    #[test]
    fn test_filter_by_method() {
        let mut f = AccessFilter::new();
        f.method = Some("POST".to_string());
        let e_get = AccessEntry::new("r1", "1.1.1.1", "GET", "/", 200, 0, 1.0, 100);
        let e_post = AccessEntry::new("r2", "1.1.1.1", "POST", "/", 201, 0, 1.0, 100);
        assert!(!f.matches(&e_get));
        assert!(f.matches(&e_post));
    }

    #[test]
    fn test_filter_by_status_range() {
        let mut f = AccessFilter::new();
        f.status_range = Some((400, 499));
        assert!(f.matches(&make_entry("r1", 404, 1.0, 0)));
        assert!(!f.matches(&make_entry("r2", 200, 1.0, 0)));
        assert!(!f.matches(&make_entry("r3", 500, 1.0, 0)));
    }

    #[test]
    fn test_filter_by_time_range() {
        let mut f = AccessFilter::new();
        f.since_ms = Some(500);
        f.until_ms = Some(1500);
        assert!(f.matches(&make_entry("r1", 200, 1.0, 1000)));
        assert!(!f.matches(&make_entry("r2", 200, 1.0, 200)));
        assert!(!f.matches(&make_entry("r3", 200, 1.0, 2000)));
    }

    #[test]
    fn test_filter_by_remote_ip() {
        let mut f = AccessFilter::new();
        f.remote_ip = Some("10.0.0.1".to_string());
        let e1 = AccessEntry::new("r1", "10.0.0.1", "GET", "/", 200, 0, 1.0, 0);
        let e2 = AccessEntry::new("r2", "10.0.0.2", "GET", "/", 200, 0, 1.0, 0);
        assert!(f.matches(&e1));
        assert!(!f.matches(&e2));
    }

    // --- AccessLog ---

    #[test]
    fn test_log_append_and_len() {
        let mut log = AccessLog::new(10);
        log.append(make_entry("r1", 200, 10.0, 1));
        log.append(make_entry("r2", 200, 20.0, 2));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_wraps_at_capacity() {
        let mut log = AccessLog::new(3);
        for i in 0..5u64 {
            log.append(make_entry(&format!("r{i}"), 200, 1.0, i));
        }
        assert_eq!(log.len(), 3);
        assert_eq!(log.total_appended(), 5);
    }

    #[test]
    fn test_log_is_empty_initially() {
        let log = AccessLog::new(100);
        assert!(log.is_empty());
    }

    #[test]
    fn test_log_query_returns_filtered_sorted() {
        let mut log = AccessLog::new(20);
        log.append(make_entry("r1", 404, 5.0, 300));
        log.append(make_entry("r2", 200, 10.0, 100));
        log.append(make_entry("r3", 500, 15.0, 200));
        let mut f = AccessFilter::new();
        f.status_range = Some((200, 299));
        let results = log.query(&f);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, 200);
    }

    #[test]
    fn test_log_summarize_statistics() {
        let mut log = AccessLog::new(20);
        log.append(make_entry("r1", 200, 10.0, 1));
        log.append(make_entry("r2", 200, 20.0, 2));
        log.append(make_entry("r3", 404, 5.0, 3));
        log.append(make_entry("r4", 500, 30.0, 4));
        let summary = log.summarize(&AccessFilter::new());
        assert_eq!(summary.total_requests, 4);
        assert_eq!(summary.successful, 2);
        assert_eq!(summary.client_errors, 1);
        assert_eq!(summary.server_errors, 1);
    }

    #[test]
    fn test_summarize_error_rate() {
        let mut log = AccessLog::new(10);
        log.append(make_entry("r1", 200, 1.0, 0));
        log.append(make_entry("r2", 500, 1.0, 1));
        let s = log.summarize(&AccessFilter::new());
        assert!((s.error_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_summarize_avg_duration() {
        let mut log = AccessLog::new(10);
        log.append(make_entry("r1", 200, 10.0, 0));
        log.append(make_entry("r2", 200, 30.0, 1));
        let s = log.summarize(&AccessFilter::new());
        assert!((s.avg_duration_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_summarize_min_max_duration() {
        let mut log = AccessLog::new(10);
        log.append(make_entry("r1", 200, 5.0, 0));
        log.append(make_entry("r2", 200, 50.0, 1));
        let s = log.summarize(&AccessFilter::new());
        assert!((s.min_duration_ms - 5.0).abs() < 1e-9);
        assert!((s.max_duration_ms - 50.0).abs() < 1e-9);
    }
}
