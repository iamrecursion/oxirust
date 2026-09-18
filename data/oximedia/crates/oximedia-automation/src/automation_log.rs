#![allow(dead_code)]
//! Structured logging for broadcast automation events.
//!
//! This module provides a comprehensive logging framework for recording
//! automation events, state transitions, operator actions, and system alerts.
//! Logs are structured for easy querying and compliance reporting.

use std::collections::VecDeque;
use std::fmt;

/// Severity level for automation log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogSeverity {
    /// Trace-level detail for debugging.
    Trace,
    /// Informational debug messages.
    Debug,
    /// Normal operational information.
    Info,
    /// Warning conditions that may need attention.
    Warning,
    /// Error conditions requiring investigation.
    Error,
    /// Critical failures requiring immediate action.
    Critical,
}

impl fmt::Display for LogSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRIT"),
        }
    }
}

/// Category of automation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogCategory {
    /// Playout operations (play, stop, cue).
    Playout,
    /// Source switching events.
    Switching,
    /// Device control commands and responses.
    DeviceControl,
    /// Operator actions (manual overrides, etc.).
    Operator,
    /// System health and monitoring.
    SystemHealth,
    /// Failover events.
    Failover,
    /// Schedule and playlist events.
    Schedule,
    /// EAS (Emergency Alert System) events.
    EmergencyAlert,
    /// Configuration changes.
    Configuration,
    /// General automation events.
    General,
}

impl fmt::Display for LogCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Playout => write!(f, "PLAYOUT"),
            Self::Switching => write!(f, "SWITCHING"),
            Self::DeviceControl => write!(f, "DEVICE"),
            Self::Operator => write!(f, "OPERATOR"),
            Self::SystemHealth => write!(f, "HEALTH"),
            Self::Failover => write!(f, "FAILOVER"),
            Self::Schedule => write!(f, "SCHEDULE"),
            Self::EmergencyAlert => write!(f, "EAS"),
            Self::Configuration => write!(f, "CONFIG"),
            Self::General => write!(f, "GENERAL"),
        }
    }
}

/// A single structured automation log entry.
#[derive(Debug, Clone)]
pub struct AutomationLogEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Timestamp as milliseconds since epoch.
    pub timestamp_ms: i64,
    /// Severity level.
    pub severity: LogSeverity,
    /// Event category.
    pub category: LogCategory,
    /// Channel identifier (if applicable).
    pub channel_id: Option<String>,
    /// Human-readable message.
    pub message: String,
    /// Optional structured detail key-value pairs.
    pub details: Vec<(String, String)>,
    /// Operator who initiated the event (if applicable).
    pub operator: Option<String>,
}

impl AutomationLogEntry {
    /// Create a new log entry with the given parameters.
    pub fn new(
        sequence: u64,
        timestamp_ms: i64,
        severity: LogSeverity,
        category: LogCategory,
        message: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            timestamp_ms,
            severity,
            category,
            channel_id: None,
            message: message.into(),
            details: Vec::new(),
            operator: None,
        }
    }

    /// Attach a channel ID to this entry.
    pub fn with_channel(mut self, channel_id: impl Into<String>) -> Self {
        self.channel_id = Some(channel_id.into());
        self
    }

    /// Attach an operator name.
    pub fn with_operator(mut self, operator: impl Into<String>) -> Self {
        self.operator = Some(operator.into());
        self
    }

    /// Add a detail key-value pair.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.push((key.into(), value.into()));
        self
    }

    /// Check if the entry matches a minimum severity filter.
    pub fn matches_severity(&self, min: LogSeverity) -> bool {
        self.severity >= min
    }

    /// Check if the entry matches a category filter.
    pub fn matches_category(&self, category: LogCategory) -> bool {
        self.category == category
    }

    /// Format the entry as a single-line log string.
    pub fn format_line(&self) -> String {
        let channel = self.channel_id.as_deref().unwrap_or("-");
        let operator = self.operator.as_deref().unwrap_or("-");
        format!(
            "[{}] {} {} ch={} op={} {}",
            self.sequence, self.severity, self.category, channel, operator, self.message
        )
    }
}

/// Query filter for searching automation logs.
#[derive(Debug, Clone, Default)]
pub struct LogFilter {
    /// Minimum severity to include.
    pub min_severity: Option<LogSeverity>,
    /// Filter by category.
    pub category: Option<LogCategory>,
    /// Filter by channel ID.
    pub channel_id: Option<String>,
    /// Filter by operator.
    pub operator: Option<String>,
    /// Start timestamp (inclusive).
    pub from_timestamp_ms: Option<i64>,
    /// End timestamp (inclusive).
    pub to_timestamp_ms: Option<i64>,
    /// Maximum number of entries to return.
    pub limit: Option<usize>,
}

impl LogFilter {
    /// Create a new empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum severity.
    pub fn with_min_severity(mut self, severity: LogSeverity) -> Self {
        self.min_severity = Some(severity);
        self
    }

    /// Set category filter.
    pub fn with_category(mut self, category: LogCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Set channel filter.
    pub fn with_channel(mut self, channel_id: impl Into<String>) -> Self {
        self.channel_id = Some(channel_id.into());
        self
    }

    /// Set operator filter.
    pub fn with_operator(mut self, operator: impl Into<String>) -> Self {
        self.operator = Some(operator.into());
        self
    }

    /// Set time range filter.
    pub fn with_time_range(mut self, from_ms: i64, to_ms: i64) -> Self {
        self.from_timestamp_ms = Some(from_ms);
        self.to_timestamp_ms = Some(to_ms);
        self
    }

    /// Set result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if a log entry matches this filter.
    pub fn matches(&self, entry: &AutomationLogEntry) -> bool {
        if let Some(min_sev) = self.min_severity {
            if entry.severity < min_sev {
                return false;
            }
        }
        if let Some(cat) = self.category {
            if entry.category != cat {
                return false;
            }
        }
        if let Some(ref ch) = self.channel_id {
            if entry.channel_id.as_ref() != Some(ch) {
                return false;
            }
        }
        if let Some(ref op) = self.operator {
            if entry.operator.as_ref() != Some(op) {
                return false;
            }
        }
        if let Some(from) = self.from_timestamp_ms {
            if entry.timestamp_ms < from {
                return false;
            }
        }
        if let Some(to) = self.to_timestamp_ms {
            if entry.timestamp_ms > to {
                return false;
            }
        }
        true
    }
}

/// Ring-buffer backed automation log with query capabilities.
#[derive(Debug)]
pub struct AutomationLog {
    /// Log entries stored in a ring buffer.
    entries: VecDeque<AutomationLogEntry>,
    /// Maximum capacity.
    capacity: usize,
    /// Next sequence number.
    next_sequence: u64,
    /// Number of entries that have been evicted due to capacity.
    evicted_count: u64,
}

impl AutomationLog {
    /// Create a new automation log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.min(65536)),
            capacity,
            next_sequence: 1,
            evicted_count: 0,
        }
    }

    /// Append a log entry, auto-assigning the sequence number.
    pub fn append(
        &mut self,
        timestamp_ms: i64,
        severity: LogSeverity,
        category: LogCategory,
        message: impl Into<String>,
    ) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        let entry = AutomationLogEntry::new(seq, timestamp_ms, severity, category, message);
        self.push_entry(entry);
        seq
    }

    /// Push a pre-built entry into the log.
    pub fn push_entry(&mut self, mut entry: AutomationLogEntry) {
        entry.sequence = self.next_sequence;
        self.next_sequence += 1;
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
            self.evicted_count += 1;
        }
        self.entries.push_back(entry);
    }

    /// Query log entries with a filter.
    pub fn query(&self, filter: &LogFilter) -> Vec<&AutomationLogEntry> {
        let mut results: Vec<&AutomationLogEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        results
    }

    /// Get the total number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of evicted entries.
    pub fn evicted_count(&self) -> u64 {
        self.evicted_count
    }

    /// Get the last N entries.
    pub fn tail(&self, n: usize) -> Vec<&AutomationLogEntry> {
        let skip = self.entries.len().saturating_sub(n);
        self.entries.iter().skip(skip).collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Count entries matching a severity level.
    pub fn count_by_severity(&self, severity: LogSeverity) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == severity)
            .count()
    }

    /// Count entries matching a category.
    pub fn count_by_category(&self, category: LogCategory) -> usize {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .count()
    }
}

impl Default for AutomationLog {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_severity_display() {
        assert_eq!(LogSeverity::Trace.to_string(), "TRACE");
        assert_eq!(LogSeverity::Debug.to_string(), "DEBUG");
        assert_eq!(LogSeverity::Info.to_string(), "INFO");
        assert_eq!(LogSeverity::Warning.to_string(), "WARN");
        assert_eq!(LogSeverity::Error.to_string(), "ERROR");
        assert_eq!(LogSeverity::Critical.to_string(), "CRIT");
    }

    #[test]
    fn test_log_severity_ordering() {
        assert!(LogSeverity::Critical > LogSeverity::Error);
        assert!(LogSeverity::Error > LogSeverity::Warning);
        assert!(LogSeverity::Warning > LogSeverity::Info);
        assert!(LogSeverity::Info > LogSeverity::Debug);
        assert!(LogSeverity::Debug > LogSeverity::Trace);
    }

    #[test]
    fn test_log_category_display() {
        assert_eq!(LogCategory::Playout.to_string(), "PLAYOUT");
        assert_eq!(LogCategory::Switching.to_string(), "SWITCHING");
        assert_eq!(LogCategory::EmergencyAlert.to_string(), "EAS");
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = AutomationLogEntry::new(
            1,
            1000,
            LogSeverity::Info,
            LogCategory::Playout,
            "Test message",
        );
        assert_eq!(entry.sequence, 1);
        assert_eq!(entry.timestamp_ms, 1000);
        assert_eq!(entry.severity, LogSeverity::Info);
        assert_eq!(entry.message, "Test message");
        assert!(entry.channel_id.is_none());
    }

    #[test]
    fn test_log_entry_builder() {
        let entry = AutomationLogEntry::new(
            1,
            2000,
            LogSeverity::Warning,
            LogCategory::Operator,
            "Override",
        )
        .with_channel("CH1")
        .with_operator("admin")
        .with_detail("source", "CAM1");
        assert_eq!(entry.channel_id.as_deref(), Some("CH1"));
        assert_eq!(entry.operator.as_deref(), Some("admin"));
        assert_eq!(entry.details.len(), 1);
        assert_eq!(entry.details[0].0, "source");
    }

    #[test]
    fn test_log_entry_format_line() {
        let entry = AutomationLogEntry::new(
            42,
            3000,
            LogSeverity::Error,
            LogCategory::Failover,
            "Primary down",
        )
        .with_channel("CH2");
        let line = entry.format_line();
        assert!(line.contains("[42]"));
        assert!(line.contains("ERROR"));
        assert!(line.contains("FAILOVER"));
        assert!(line.contains("Primary down"));
    }

    #[test]
    fn test_log_entry_matches_severity() {
        let entry =
            AutomationLogEntry::new(1, 1000, LogSeverity::Warning, LogCategory::General, "msg");
        assert!(entry.matches_severity(LogSeverity::Warning));
        assert!(entry.matches_severity(LogSeverity::Info));
        assert!(!entry.matches_severity(LogSeverity::Error));
    }

    #[test]
    fn test_automation_log_append() {
        let mut log = AutomationLog::new(100);
        let seq = log.append(1000, LogSeverity::Info, LogCategory::Playout, "Started");
        assert!(seq > 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_automation_log_capacity() {
        let mut log = AutomationLog::new(3);
        log.append(1, LogSeverity::Info, LogCategory::General, "A");
        log.append(2, LogSeverity::Info, LogCategory::General, "B");
        log.append(3, LogSeverity::Info, LogCategory::General, "C");
        log.append(4, LogSeverity::Info, LogCategory::General, "D");
        assert_eq!(log.len(), 3);
        assert_eq!(log.evicted_count(), 1);
    }

    #[test]
    fn test_automation_log_query_by_severity() {
        let mut log = AutomationLog::new(100);
        log.append(1, LogSeverity::Info, LogCategory::General, "Info msg");
        log.append(2, LogSeverity::Error, LogCategory::General, "Error msg");
        log.append(3, LogSeverity::Warning, LogCategory::General, "Warn msg");

        let filter = LogFilter::new().with_min_severity(LogSeverity::Warning);
        let results = log.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_automation_log_query_by_category() {
        let mut log = AutomationLog::new(100);
        log.append(1, LogSeverity::Info, LogCategory::Playout, "Play");
        log.append(2, LogSeverity::Info, LogCategory::Switching, "Switch");
        log.append(3, LogSeverity::Info, LogCategory::Playout, "Stop");

        let filter = LogFilter::new().with_category(LogCategory::Playout);
        let results = log.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_automation_log_tail() {
        let mut log = AutomationLog::new(100);
        for i in 0..10 {
            log.append(
                i,
                LogSeverity::Info,
                LogCategory::General,
                format!("msg {i}"),
            );
        }
        let tail = log.tail(3);
        assert_eq!(tail.len(), 3);
        assert!(tail[2].message.contains("msg 9"));
    }

    #[test]
    fn test_automation_log_count_by_severity() {
        let mut log = AutomationLog::new(100);
        log.append(1, LogSeverity::Info, LogCategory::General, "a");
        log.append(2, LogSeverity::Error, LogCategory::General, "b");
        log.append(3, LogSeverity::Error, LogCategory::General, "c");
        assert_eq!(log.count_by_severity(LogSeverity::Error), 2);
        assert_eq!(log.count_by_severity(LogSeverity::Info), 1);
    }

    #[test]
    fn test_automation_log_clear() {
        let mut log = AutomationLog::new(100);
        log.append(1, LogSeverity::Info, LogCategory::General, "a");
        log.append(2, LogSeverity::Info, LogCategory::General, "b");
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_log_filter_time_range() {
        let mut log = AutomationLog::new(100);
        log.append(100, LogSeverity::Info, LogCategory::General, "early");
        log.append(500, LogSeverity::Info, LogCategory::General, "mid");
        log.append(900, LogSeverity::Info, LogCategory::General, "late");

        let filter = LogFilter::new().with_time_range(200, 600);
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
    }
}
