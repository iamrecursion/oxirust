//! Structured log aggregation with level filtering for `OxiMedia` monitoring.
//!
//! Collects `LogEntry` records from pipeline components and exposes
//! query helpers to retrieve entries filtered by level, component,
//! or message pattern.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Severity level of a log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Fine-grained diagnostic information.
    Trace = 0,
    /// General development-time information.
    Debug = 1,
    /// Normal operational messages.
    Info = 2,
    /// Potentially harmful situations.
    Warn = 3,
    /// Error events that may still allow continued operation.
    Error = 4,
    /// Severe errors causing abnormal termination.
    Fatal = 5,
}

impl LogLevel {
    /// Human-readable name of the level.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
        }
    }

    /// Returns `true` when the level is at least `Warn`.
    #[must_use]
    pub fn is_warning_or_above(self) -> bool {
        self >= Self::Warn
    }
}

/// A single structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Severity of the entry.
    pub level: LogLevel,
    /// Component or module that emitted the entry.
    pub component: String,
    /// Human-readable message.
    pub message: String,
    /// Optional key-value context pairs.
    pub fields: Vec<(String, String)>,
    /// Monotonic timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
}

impl LogEntry {
    /// Create a simple entry with no extra fields.
    #[must_use]
    pub fn new(
        level: LogLevel,
        component: impl Into<String>,
        message: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            level,
            component: component.into(),
            message: message.into(),
            fields: Vec::new(),
            timestamp_ms,
        }
    }

    /// Attach a key-value context field to this entry.
    #[must_use]
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.push((key.into(), value.into()));
        self
    }

    /// Returns `true` when the entry is at least `Warn` severity.
    #[must_use]
    pub fn is_notable(&self) -> bool {
        self.level.is_warning_or_above()
    }
}

/// Configuration for `LogAggregator`.
#[derive(Debug, Clone)]
pub struct LogAggregatorConfig {
    /// Maximum number of entries to keep in the ring buffer.
    pub capacity: usize,
    /// Minimum level to store; entries below this are dropped.
    pub min_level: LogLevel,
}

impl Default for LogAggregatorConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            min_level: LogLevel::Debug,
        }
    }
}

/// Aggregates log entries from multiple pipeline components.
///
/// Entries are stored in a fixed-capacity ring buffer; oldest entries
/// are evicted when the buffer is full.
#[derive(Debug)]
pub struct LogAggregator {
    config: LogAggregatorConfig,
    entries: VecDeque<LogEntry>,
    total_received: u64,
    total_dropped: u64,
}

impl LogAggregator {
    /// Create a new aggregator with the given configuration.
    #[must_use]
    pub fn new(config: LogAggregatorConfig) -> Self {
        Self {
            entries: VecDeque::with_capacity(config.capacity),
            config,
            total_received: 0,
            total_dropped: 0,
        }
    }

    /// Ingest one log entry. Returns `false` if the entry was filtered by
    /// `min_level`.
    pub fn ingest(&mut self, entry: LogEntry) -> bool {
        self.total_received += 1;

        if entry.level < self.config.min_level {
            self.total_dropped += 1;
            return false;
        }

        if self.entries.len() == self.config.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        true
    }

    /// Return entries whose level is exactly `level` or above.
    #[must_use]
    pub fn filter_by_level(&self, level: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level >= level).collect()
    }

    /// Return entries emitted by the named `component`.
    #[must_use]
    pub fn filter_by_component<'a>(&'a self, component: &str) -> Vec<&'a LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.component == component)
            .collect()
    }

    /// Return entries whose message contains `needle` (case-insensitive).
    #[must_use]
    pub fn search(&self, needle: &str) -> Vec<&LogEntry> {
        let needle_lower = needle.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.message.to_lowercase().contains(&needle_lower))
            .collect()
    }

    /// Current number of entries stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total entries received since creation (including filtered ones).
    #[must_use]
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Total entries dropped due to `min_level` filtering.
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// Clear all stored entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drain entries older than `cutoff_ms`.
    pub fn prune_before(&mut self, cutoff_ms: u64) {
        self.entries.retain(|e| e.timestamp_ms >= cutoff_ms);
    }
}

impl Default for LogAggregator {
    fn default() -> Self {
        Self::new(LogAggregatorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(level: LogLevel, component: &str, message: &str, ts: u64) -> LogEntry {
        LogEntry::new(level, component, message, ts)
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Fatal.as_str(), "FATAL");
    }

    #[test]
    fn test_is_warning_or_above() {
        assert!(!LogLevel::Info.is_warning_or_above());
        assert!(LogLevel::Warn.is_warning_or_above());
        assert!(LogLevel::Error.is_warning_or_above());
    }

    #[test]
    fn test_ingest_basic() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Info, "encoder", "started", 1000));
        assert_eq!(agg.len(), 1);
    }

    #[test]
    fn test_ingest_filtered_by_min_level() {
        let mut agg = LogAggregator::new(LogAggregatorConfig {
            capacity: 100,
            min_level: LogLevel::Warn,
        });
        let accepted = agg.ingest(entry(LogLevel::Info, "comp", "verbose", 1));
        assert!(!accepted);
        assert_eq!(agg.len(), 0);
        assert_eq!(agg.total_dropped(), 1);
    }

    #[test]
    fn test_ingest_ring_buffer_eviction() {
        let mut agg = LogAggregator::new(LogAggregatorConfig {
            capacity: 3,
            min_level: LogLevel::Trace,
        });
        for i in 0_u64..5 {
            agg.ingest(entry(LogLevel::Info, "c", &format!("msg{i}"), i));
        }
        assert_eq!(agg.len(), 3);
    }

    #[test]
    fn test_filter_by_level_returns_correct_entries() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Debug, "c", "dbg", 1));
        agg.ingest(entry(LogLevel::Warn, "c", "warn", 2));
        agg.ingest(entry(LogLevel::Error, "c", "err", 3));
        let warns = agg.filter_by_level(LogLevel::Warn);
        assert_eq!(warns.len(), 2);
    }

    #[test]
    fn test_filter_by_component() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Info, "encoder", "e1", 1));
        agg.ingest(entry(LogLevel::Info, "muxer", "m1", 2));
        agg.ingest(entry(LogLevel::Info, "encoder", "e2", 3));
        let enc = agg.filter_by_component("encoder");
        assert_eq!(enc.len(), 2);
    }

    #[test]
    fn test_search_case_insensitive() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Info, "net", "Connection RESET", 1));
        agg.ingest(entry(LogLevel::Info, "net", "frame dropped", 2));
        let hits = agg.search("reset");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Info, "c", "m", 1));
        agg.clear();
        assert!(agg.is_empty());
    }

    #[test]
    fn test_prune_before() {
        let mut agg = LogAggregator::default();
        agg.ingest(entry(LogLevel::Info, "c", "old", 100));
        agg.ingest(entry(LogLevel::Info, "c", "new", 200));
        agg.prune_before(150);
        assert_eq!(agg.len(), 1);
        assert_eq!(agg.filter_by_component("c")[0].message, "new");
    }

    #[test]
    fn test_total_received_counter() {
        let mut agg = LogAggregator::default();
        for _ in 0..5 {
            agg.ingest(entry(LogLevel::Info, "c", "m", 1));
        }
        assert_eq!(agg.total_received(), 5);
    }

    #[test]
    fn test_entry_with_field() {
        let e = LogEntry::new(LogLevel::Info, "c", "m", 0)
            .with_field("job_id", "42")
            .with_field("frame", "100");
        assert_eq!(e.fields.len(), 2);
        assert_eq!(e.fields[0].0, "job_id");
    }

    #[test]
    fn test_is_notable() {
        let e_info = entry(LogLevel::Info, "c", "m", 0);
        let e_err = entry(LogLevel::Error, "c", "m", 0);
        assert!(!e_info.is_notable());
        assert!(e_err.is_notable());
    }
}
