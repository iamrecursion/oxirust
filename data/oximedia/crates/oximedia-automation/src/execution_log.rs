//! Structured execution log for automation scripts and action sequences.
//!
//! Provides `LogLevel`, `ExecLogEntry`, and `ExecutionLog` for capturing
//! timestamped log lines produced during the execution of automation rules.

#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity level of a log entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Highly detailed diagnostic information.
    Trace,
    /// Standard debug output.
    Debug,
    /// Informational message about normal operation.
    Info,
    /// A potentially problematic situation.
    Warn,
    /// A recoverable error condition.
    Error,
    /// A fatal, non-recoverable error.
    Fatal,
}

impl LogLevel {
    /// Numeric priority (higher = more severe).
    ///
    /// Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Fatal = 5.
    #[must_use]
    pub fn numeric_level(&self) -> u8 {
        match self {
            Self::Trace => 0,
            Self::Debug => 1,
            Self::Info => 2,
            Self::Warn => 3,
            Self::Error => 4,
            Self::Fatal => 5,
        }
    }

    /// Short uppercase label.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// ExecLogEntry
// ---------------------------------------------------------------------------

/// A single log line emitted during automation execution.
#[derive(Debug, Clone)]
pub struct ExecLogEntry {
    /// Unix seconds at which the entry was created.
    pub timestamp_secs: u64,
    /// Severity of this entry.
    pub level: LogLevel,
    /// The module or action that produced this entry.
    pub source: String,
    /// Log message text.
    pub message: String,
}

impl ExecLogEntry {
    /// Create a new entry using the current wall-clock time.
    #[must_use]
    pub fn new(level: LogLevel, source: impl Into<String>, message: impl Into<String>) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            timestamp_secs,
            level,
            source: source.into(),
            message: message.into(),
        }
    }

    /// Create an entry with an explicit timestamp (useful in tests).
    #[must_use]
    pub fn with_timestamp(
        timestamp_secs: u64,
        level: LogLevel,
        source: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_secs,
            level,
            source: source.into(),
            message: message.into(),
        }
    }

    /// Returns `true` if this entry is at `Error` or `Fatal` level.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self.level, LogLevel::Error | LogLevel::Fatal)
    }

    /// Formatted single-line representation.
    #[must_use]
    pub fn format_line(&self) -> String {
        format!(
            "[{}] {} [{}] {}",
            self.timestamp_secs,
            self.level.as_str(),
            self.source,
            self.message,
        )
    }
}

// ---------------------------------------------------------------------------
// ExecutionLog
// ---------------------------------------------------------------------------

/// Append-only log of `ExecLogEntry` items produced during a single
/// automation execution session.
#[derive(Debug, Default)]
pub struct ExecutionLog {
    entries: Vec<ExecLogEntry>,
}

impl ExecutionLog {
    /// Create an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry.
    pub fn push(&mut self, entry: ExecLogEntry) {
        self.entries.push(entry);
    }

    /// Convenience helper — log at the given level.
    pub fn log(&mut self, level: LogLevel, source: impl Into<String>, message: impl Into<String>) {
        self.push(ExecLogEntry::new(level, source, message));
    }

    /// Return all entries at `Error` or `Fatal` level.
    #[must_use]
    pub fn errors(&self) -> Vec<&ExecLogEntry> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }

    /// Return entries whose level numeric value is ≥ that of `min_level`.
    #[must_use]
    pub fn filter_level(&self, min_level: &LogLevel) -> Vec<&ExecLogEntry> {
        let threshold = min_level.numeric_level();
        self.entries
            .iter()
            .filter(|e| e.level.numeric_level() >= threshold)
            .collect()
    }

    /// Total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all entries in insertion order.
    #[must_use]
    pub fn all(&self) -> &[ExecLogEntry] {
        &self.entries
    }

    /// Last `n` entries (most recent last slice).
    #[must_use]
    pub fn tail(&self, n: usize) -> &[ExecLogEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_numeric_trace() {
        assert_eq!(LogLevel::Trace.numeric_level(), 0);
    }

    #[test]
    fn test_log_level_numeric_fatal() {
        assert_eq!(LogLevel::Fatal.numeric_level(), 5);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(format!("{}", LogLevel::Debug), "DEBUG");
    }

    #[test]
    fn test_exec_log_entry_is_error_true() {
        let e = ExecLogEntry::new(LogLevel::Error, "src", "msg");
        assert!(e.is_error());
        let f = ExecLogEntry::new(LogLevel::Fatal, "src", "msg");
        assert!(f.is_error());
    }

    #[test]
    fn test_exec_log_entry_is_error_false() {
        let e = ExecLogEntry::new(LogLevel::Warn, "src", "msg");
        assert!(!e.is_error());
        let i = ExecLogEntry::new(LogLevel::Info, "src", "msg");
        assert!(!i.is_error());
    }

    #[test]
    fn test_exec_log_entry_format_line() {
        let e = ExecLogEntry::with_timestamp(1000, LogLevel::Info, "sched", "started");
        let line = e.format_line();
        assert!(line.contains("INFO"));
        assert!(line.contains("sched"));
        assert!(line.contains("started"));
        assert!(line.contains("1000"));
    }

    #[test]
    fn test_execution_log_push_and_len() {
        let mut log = ExecutionLog::new();
        assert!(log.is_empty());
        log.push(ExecLogEntry::new(LogLevel::Info, "a", "msg1"));
        log.push(ExecLogEntry::new(LogLevel::Error, "a", "msg2"));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_execution_log_errors() {
        let mut log = ExecutionLog::new();
        log.push(ExecLogEntry::new(LogLevel::Info, "x", "ok"));
        log.push(ExecLogEntry::new(LogLevel::Error, "x", "bad"));
        log.push(ExecLogEntry::new(LogLevel::Fatal, "x", "worse"));
        let errs = log.errors();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_execution_log_filter_level() {
        let mut log = ExecutionLog::new();
        log.push(ExecLogEntry::new(LogLevel::Trace, "x", "t"));
        log.push(ExecLogEntry::new(LogLevel::Info, "x", "i"));
        log.push(ExecLogEntry::new(LogLevel::Error, "x", "e"));
        let filtered = log.filter_level(&LogLevel::Info);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_execution_log_filter_level_warn() {
        let mut log = ExecutionLog::new();
        log.push(ExecLogEntry::new(LogLevel::Debug, "x", "d"));
        log.push(ExecLogEntry::new(LogLevel::Warn, "x", "w"));
        log.push(ExecLogEntry::new(LogLevel::Error, "x", "e"));
        let filtered = log.filter_level(&LogLevel::Warn);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_execution_log_tail() {
        let mut log = ExecutionLog::new();
        for i in 0..5u64 {
            log.push(ExecLogEntry::with_timestamp(i, LogLevel::Info, "x", "m"));
        }
        let tail = log.tail(3);
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[0].timestamp_secs, 2);
        assert_eq!(tail[2].timestamp_secs, 4);
    }

    #[test]
    fn test_execution_log_tail_larger_than_log() {
        let mut log = ExecutionLog::new();
        log.push(ExecLogEntry::new(LogLevel::Info, "x", "m"));
        let tail = log.tail(10);
        assert_eq!(tail.len(), 1);
    }
}
