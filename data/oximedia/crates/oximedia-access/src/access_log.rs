//! Access event logging for `OxiMedia`.
//!
//! Records read and write access events per user, with filtering and
//! recency queries for audit and compliance workflows.

#![allow(dead_code)]

use std::collections::HashMap;

/// The type of access event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessEvent {
    /// A read-only access (e.g. media playback, metadata fetch).
    Read,
    /// A write operation (e.g. metadata update, file creation).
    Write,
    /// An administrative action (e.g. permission change, user management).
    Admin,
    /// A delete operation.
    Delete,
}

impl AccessEvent {
    /// Returns `true` if this event is a write-class operation
    /// (modifies or removes data).
    #[must_use]
    pub fn is_write(&self) -> bool {
        matches!(self, Self::Write | Self::Delete | Self::Admin)
    }

    /// Returns a short human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
            Self::Delete => "delete",
        }
    }
}

/// A single entry in the access log.
#[derive(Debug, Clone)]
pub struct AccessLogEntry {
    /// The user performing the action.
    pub username: String,
    /// The resource being accessed (e.g. a path or asset ID).
    pub resource: String,
    /// The type of access.
    pub event: AccessEvent,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
}

impl AccessLogEntry {
    /// Create a new log entry.
    #[must_use]
    pub fn new(
        username: impl Into<String>,
        resource: impl Into<String>,
        event: AccessEvent,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            username: username.into(),
            resource: resource.into(),
            event,
            timestamp_ms,
        }
    }

    /// Returns `true` if the entry is within `window_ms` of `now_ms`.
    #[must_use]
    pub fn is_recent(&self, now_ms: u64, window_ms: u64) -> bool {
        now_ms.saturating_sub(self.timestamp_ms) < window_ms
    }
}

/// Append-only access log with filtering capabilities.
#[derive(Debug, Default)]
pub struct AccessLog {
    entries: Vec<AccessLogEntry>,
}

impl AccessLog {
    /// Create an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry to the log.
    pub fn record(&mut self, entry: AccessLogEntry) {
        self.entries.push(entry);
    }

    /// Return all entries belonging to `username`.
    #[must_use]
    pub fn filter_by_user(&self, username: &str) -> Vec<&AccessLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.username == username)
            .collect()
    }

    /// Return all write-class entries within `window_ms` of `now_ms`.
    #[must_use]
    pub fn recent_writes(&self, now_ms: u64, window_ms: u64) -> Vec<&AccessLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.event.is_write() && e.is_recent(now_ms, window_ms))
            .collect()
    }

    /// Return entries for a specific resource.
    #[must_use]
    pub fn filter_by_resource(&self, resource: &str) -> Vec<&AccessLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.resource == resource)
            .collect()
    }

    /// Count all entries in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute a per-user event count summary.
    #[must_use]
    pub fn user_event_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for e in &self.entries {
            *counts.entry(e.username.clone()).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(user: &str, res: &str, ev: AccessEvent, ts: u64) -> AccessLogEntry {
        AccessLogEntry::new(user, res, ev, ts)
    }

    #[test]
    fn test_read_is_not_write() {
        assert!(!AccessEvent::Read.is_write());
    }

    #[test]
    fn test_write_is_write() {
        assert!(AccessEvent::Write.is_write());
    }

    #[test]
    fn test_delete_is_write() {
        assert!(AccessEvent::Delete.is_write());
    }

    #[test]
    fn test_admin_is_write() {
        assert!(AccessEvent::Admin.is_write());
    }

    #[test]
    fn test_event_labels() {
        assert_eq!(AccessEvent::Read.label(), "read");
        assert_eq!(AccessEvent::Delete.label(), "delete");
    }

    #[test]
    fn test_entry_is_recent_within_window() {
        let e = entry("alice", "/asset/1", AccessEvent::Read, 1000);
        assert!(e.is_recent(1500, 1000));
    }

    #[test]
    fn test_entry_is_recent_outside_window() {
        let e = entry("alice", "/asset/1", AccessEvent::Read, 1000);
        assert!(!e.is_recent(2001, 1000));
    }

    #[test]
    fn test_log_empty_initially() {
        let log = AccessLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_record_increments_len() {
        let mut log = AccessLog::new();
        log.record(entry("bob", "/clip/2", AccessEvent::Write, 5000));
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_filter_by_user() {
        let mut log = AccessLog::new();
        log.record(entry("carol", "/a", AccessEvent::Read, 1000));
        log.record(entry("dave", "/b", AccessEvent::Write, 2000));
        log.record(entry("carol", "/c", AccessEvent::Delete, 3000));
        let carol_entries = log.filter_by_user("carol");
        assert_eq!(carol_entries.len(), 2);
    }

    #[test]
    fn test_filter_by_user_none_found() {
        let log = AccessLog::new();
        assert!(log.filter_by_user("nobody").is_empty());
    }

    #[test]
    fn test_recent_writes_returns_only_writes() {
        let mut log = AccessLog::new();
        let now = 10_000u64;
        log.record(entry("eve", "/x", AccessEvent::Read, now - 500));
        log.record(entry("eve", "/y", AccessEvent::Write, now - 500));
        log.record(entry("eve", "/z", AccessEvent::Delete, now - 500));
        let writes = log.recent_writes(now, 1000);
        assert_eq!(writes.len(), 2);
    }

    #[test]
    fn test_recent_writes_excludes_old() {
        let mut log = AccessLog::new();
        let now = 10_000u64;
        log.record(entry("frank", "/old", AccessEvent::Write, 1));
        let writes = log.recent_writes(now, 1000);
        assert!(writes.is_empty());
    }

    #[test]
    fn test_filter_by_resource() {
        let mut log = AccessLog::new();
        log.record(entry("grace", "/res/A", AccessEvent::Read, 100));
        log.record(entry("heidi", "/res/B", AccessEvent::Write, 200));
        log.record(entry("ivan", "/res/A", AccessEvent::Delete, 300));
        let res_a = log.filter_by_resource("/res/A");
        assert_eq!(res_a.len(), 2);
    }

    #[test]
    fn test_user_event_counts() {
        let mut log = AccessLog::new();
        log.record(entry("judy", "/a", AccessEvent::Read, 100));
        log.record(entry("judy", "/b", AccessEvent::Write, 200));
        log.record(entry("karl", "/c", AccessEvent::Read, 300));
        let counts = log.user_event_counts();
        assert_eq!(counts["judy"], 2);
        assert_eq!(counts["karl"], 1);
    }
}
