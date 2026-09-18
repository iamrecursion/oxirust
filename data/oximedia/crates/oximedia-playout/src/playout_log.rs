#![allow(dead_code)]
//! Detailed playout logging and audit trail.
//!
//! This module records every significant event during playout: clip start/stop,
//! errors, failovers, graphics triggers, and ad-break transitions.  The log
//! supports querying by time range, severity, and event type for compliance
//! reporting and post-broadcast analysis.

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational — normal operation.
    Info,
    /// Warning — non-critical issue.
    Warning,
    /// Error — something failed.
    Error,
    /// Critical — requires immediate attention.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Event type
// ---------------------------------------------------------------------------

/// Type of playout event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayoutEventType {
    /// A clip started playing.
    ClipStart,
    /// A clip finished playing.
    ClipEnd,
    /// A clip was skipped.
    ClipSkip,
    /// A graphics layer was triggered on.
    GraphicsOn,
    /// A graphics layer was triggered off.
    GraphicsOff,
    /// An ad break started.
    AdBreakStart,
    /// An ad break ended.
    AdBreakEnd,
    /// Emergency failover was activated.
    Failover,
    /// A manual override occurred.
    ManualOverride,
    /// A configuration change was applied.
    ConfigChange,
    /// A frame drop was detected.
    FrameDrop,
    /// A sync error occurred.
    SyncError,
    /// Server started.
    ServerStart,
    /// Server stopped.
    ServerStop,
}

impl fmt::Display for PlayoutEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::ClipStart => "CLIP_START",
            Self::ClipEnd => "CLIP_END",
            Self::ClipSkip => "CLIP_SKIP",
            Self::GraphicsOn => "GFX_ON",
            Self::GraphicsOff => "GFX_OFF",
            Self::AdBreakStart => "AD_START",
            Self::AdBreakEnd => "AD_END",
            Self::Failover => "FAILOVER",
            Self::ManualOverride => "MANUAL",
            Self::ConfigChange => "CONFIG",
            Self::FrameDrop => "FRAME_DROP",
            Self::SyncError => "SYNC_ERR",
            Self::ServerStart => "SVR_START",
            Self::ServerStop => "SVR_STOP",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Log entry
// ---------------------------------------------------------------------------

/// A single playout log entry.
#[derive(Debug, Clone)]
pub struct PlayoutLogEntry {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Timestamp as seconds since midnight.
    pub timestamp_sec: f64,
    /// Event type.
    pub event_type: PlayoutEventType,
    /// Severity.
    pub severity: Severity,
    /// Channel / output name.
    pub channel: String,
    /// Human-readable message.
    pub message: String,
    /// Optional clip or asset identifier.
    pub asset_id: Option<String>,
    /// Optional duration in seconds.
    pub duration_sec: Option<f64>,
}

impl fmt::Display for PlayoutLogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:08}] {:.3}s {} {} [{}] {}",
            self.seq,
            self.timestamp_sec,
            self.severity,
            self.event_type,
            self.channel,
            self.message,
        )
    }
}

// ---------------------------------------------------------------------------
// Log query
// ---------------------------------------------------------------------------

/// A query filter for searching the playout log.
#[derive(Debug, Clone, Default)]
pub struct LogQuery {
    /// Only entries at or above this severity.
    pub min_severity: Option<Severity>,
    /// Only entries of this event type.
    pub event_type: Option<PlayoutEventType>,
    /// Only entries for this channel.
    pub channel: Option<String>,
    /// Only entries at or after this timestamp.
    pub from_sec: Option<f64>,
    /// Only entries at or before this timestamp.
    pub to_sec: Option<f64>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl LogQuery {
    /// Create a new empty query (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by minimum severity.
    pub fn with_min_severity(mut self, sev: Severity) -> Self {
        self.min_severity = Some(sev);
        self
    }

    /// Filter by event type.
    pub fn with_event_type(mut self, et: PlayoutEventType) -> Self {
        self.event_type = Some(et);
        self
    }

    /// Filter by channel name.
    pub fn with_channel(mut self, ch: &str) -> Self {
        self.channel = Some(ch.to_string());
        self
    }

    /// Filter by time range.
    pub fn with_time_range(mut self, from: f64, to: f64) -> Self {
        self.from_sec = Some(from);
        self.to_sec = Some(to);
        self
    }

    /// Limit the number of results.
    pub fn with_limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Test whether an entry matches this query.
    fn matches(&self, entry: &PlayoutLogEntry) -> bool {
        if let Some(min_sev) = self.min_severity {
            if entry.severity < min_sev {
                return false;
            }
        }
        if let Some(et) = self.event_type {
            if entry.event_type != et {
                return false;
            }
        }
        if let Some(ref ch) = self.channel {
            if &entry.channel != ch {
                return false;
            }
        }
        if let Some(from) = self.from_sec {
            if entry.timestamp_sec < from {
                return false;
            }
        }
        if let Some(to) = self.to_sec {
            if entry.timestamp_sec > to {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Playout log
// ---------------------------------------------------------------------------

/// Ring-buffer-backed playout log with query capabilities.
#[derive(Debug, Clone)]
pub struct PlayoutLog {
    /// Maximum number of entries to retain.
    pub max_entries: usize,
    /// The log entries.
    entries: VecDeque<PlayoutLogEntry>,
    /// Next sequence number.
    next_seq: u64,
}

impl PlayoutLog {
    /// Create a new playout log.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: VecDeque::with_capacity(max_entries.min(10_000)),
            next_seq: 1,
        }
    }

    /// Append a new log entry.
    pub fn append(
        &mut self,
        timestamp_sec: f64,
        event_type: PlayoutEventType,
        severity: Severity,
        channel: &str,
        message: &str,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = PlayoutLogEntry {
            seq,
            timestamp_sec,
            event_type,
            severity,
            channel: channel.to_string(),
            message: message.to_string(),
            asset_id: None,
            duration_sec: None,
        };
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        seq
    }

    /// Append an entry with asset and duration metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn append_with_meta(
        &mut self,
        timestamp_sec: f64,
        event_type: PlayoutEventType,
        severity: Severity,
        channel: &str,
        message: &str,
        asset_id: &str,
        duration_sec: f64,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = PlayoutLogEntry {
            seq,
            timestamp_sec,
            event_type,
            severity,
            channel: channel.to_string(),
            message: message.to_string(),
            asset_id: Some(asset_id.to_string()),
            duration_sec: Some(duration_sec),
        };
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        seq
    }

    /// Query the log.
    pub fn query(&self, q: &LogQuery) -> Vec<&PlayoutLogEntry> {
        let mut results: Vec<&PlayoutLogEntry> =
            self.entries.iter().filter(|e| q.matches(e)).collect();
        if let Some(limit) = q.limit {
            results.truncate(limit);
        }
        results
    }

    /// Total number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Return the last N entries (most recent first).
    pub fn tail(&self, n: usize) -> Vec<&PlayoutLogEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Count entries matching a query.
    pub fn count(&self, q: &LogQuery) -> usize {
        self.entries.iter().filter(|e| q.matches(e)).count()
    }

    /// Summary statistics.
    pub fn summary(&self) -> LogSummary {
        let mut info = 0u64;
        let mut warning = 0u64;
        let mut error = 0u64;
        let mut critical = 0u64;

        for e in &self.entries {
            match e.severity {
                Severity::Info => info += 1,
                Severity::Warning => warning += 1,
                Severity::Error => error += 1,
                Severity::Critical => critical += 1,
            }
        }

        LogSummary {
            total: self.entries.len() as u64,
            info,
            warning,
            error,
            critical,
        }
    }
}

// ---------------------------------------------------------------------------
// Log summary
// ---------------------------------------------------------------------------

/// Summary statistics for a playout log.
#[derive(Debug, Clone)]
pub struct LogSummary {
    /// Total entries.
    pub total: u64,
    /// Info entries.
    pub info: u64,
    /// Warning entries.
    pub warning: u64,
    /// Error entries.
    pub error: u64,
    /// Critical entries.
    pub critical: u64,
}

impl LogSummary {
    /// Error rate as a fraction of total entries.
    #[allow(clippy::cast_precision_loss)]
    pub fn error_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.error + self.critical) as f64 / self.total as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_log() -> PlayoutLog {
        let mut log = PlayoutLog::new(100);
        log.append(
            0.0,
            PlayoutEventType::ServerStart,
            Severity::Info,
            "CH1",
            "Server started",
        );
        log.append(
            1.0,
            PlayoutEventType::ClipStart,
            Severity::Info,
            "CH1",
            "Playing intro",
        );
        log.append(
            30.0,
            PlayoutEventType::ClipEnd,
            Severity::Info,
            "CH1",
            "Intro done",
        );
        log.append(
            30.5,
            PlayoutEventType::FrameDrop,
            Severity::Warning,
            "CH1",
            "1 frame dropped",
        );
        log.append(
            31.0,
            PlayoutEventType::ClipStart,
            Severity::Info,
            "CH1",
            "Playing main",
        );
        log.append(
            60.0,
            PlayoutEventType::Failover,
            Severity::Critical,
            "CH1",
            "Source lost",
        );
        log
    }

    #[test]
    fn test_log_append() {
        let mut log = PlayoutLog::new(10);
        let seq = log.append(
            0.0,
            PlayoutEventType::ServerStart,
            Severity::Info,
            "CH1",
            "Start",
        );
        assert_eq!(seq, 1);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_log_ring_buffer() {
        let mut log = PlayoutLog::new(3);
        log.append(
            0.0,
            PlayoutEventType::ServerStart,
            Severity::Info,
            "CH1",
            "a",
        );
        log.append(1.0, PlayoutEventType::ClipStart, Severity::Info, "CH1", "b");
        log.append(2.0, PlayoutEventType::ClipEnd, Severity::Info, "CH1", "c");
        log.append(3.0, PlayoutEventType::ClipStart, Severity::Info, "CH1", "d");
        assert_eq!(log.len(), 3);
        // First entry should have been evicted
        let tail = log.tail(3);
        assert_eq!(tail[0].message, "d");
    }

    #[test]
    fn test_query_by_severity() {
        let log = sample_log();
        let q = LogQuery::new().with_min_severity(Severity::Warning);
        let results = log.query(&q);
        assert_eq!(results.len(), 2); // Warning + Critical
    }

    #[test]
    fn test_query_by_event_type() {
        let log = sample_log();
        let q = LogQuery::new().with_event_type(PlayoutEventType::ClipStart);
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_time_range() {
        let log = sample_log();
        let q = LogQuery::new().with_time_range(1.0, 31.0);
        let results = log.query(&q);
        assert_eq!(results.len(), 4); // timestamps 1.0, 30.0, 30.5, 31.0
    }

    #[test]
    fn test_query_with_limit() {
        let log = sample_log();
        let q = LogQuery::new().with_limit(2);
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_summary() {
        let log = sample_log();
        let s = log.summary();
        assert_eq!(s.total, 6);
        assert_eq!(s.info, 4);
        assert_eq!(s.warning, 1);
        assert_eq!(s.critical, 1);
        assert_eq!(s.error, 0);
    }

    #[test]
    fn test_error_rate() {
        let log = sample_log();
        let s = log.summary();
        // 1 critical / 6 total
        assert!((s.error_rate() - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_append_with_meta() {
        let mut log = PlayoutLog::new(10);
        let seq = log.append_with_meta(
            5.0,
            PlayoutEventType::ClipStart,
            Severity::Info,
            "CH1",
            "Playing clip",
            "CLIP_001",
            120.0,
        );
        assert_eq!(seq, 1);
        let entry = &log.tail(1)[0];
        assert_eq!(entry.asset_id.as_deref(), Some("CLIP_001"));
        assert_eq!(entry.duration_sec, Some(120.0));
    }

    #[test]
    fn test_clear() {
        let mut log = sample_log();
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_count() {
        let log = sample_log();
        let q = LogQuery::new().with_event_type(PlayoutEventType::Failover);
        assert_eq!(log.count(&q), 1);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
        assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
    }

    #[test]
    fn test_entry_display() {
        let entry = PlayoutLogEntry {
            seq: 42,
            timestamp_sec: 10.5,
            event_type: PlayoutEventType::ClipStart,
            severity: Severity::Info,
            channel: "CH1".to_string(),
            message: "Playing clip X".to_string(),
            asset_id: None,
            duration_sec: None,
        };
        let s = format!("{entry}");
        assert!(s.contains("CLIP_START"));
        assert!(s.contains("CH1"));
    }

    #[test]
    fn test_empty_log_summary() {
        let log = PlayoutLog::new(10);
        let s = log.summary();
        assert_eq!(s.total, 0);
        assert!((s.error_rate() - 0.0).abs() < 1e-12);
    }
}
