#![allow(dead_code)]
//! Conform session logging and audit trail.
//!
//! This module provides structured logging for conform operations, capturing
//! every match decision, media relinking, timeline change, and export action
//! for full auditability and troubleshooting of conform sessions.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

/// Severity level for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LogLevel {
    /// Debug-level detail.
    Debug,
    /// Informational messages.
    Info,
    /// Potential issues.
    Warning,
    /// Errors that may affect the conform.
    Error,
    /// Critical failures.
    Critical,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Category of the log entry for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogCategory {
    /// Session lifecycle events.
    Session,
    /// Import operations.
    Import,
    /// Match operations.
    Match,
    /// Relink operations.
    Relink,
    /// Timeline reconstruction.
    Timeline,
    /// Export operations.
    Export,
    /// Quality control.
    Qc,
    /// General/other.
    General,
}

impl fmt::Display for LogCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session => write!(f, "SESSION"),
            Self::Import => write!(f, "IMPORT"),
            Self::Match => write!(f, "MATCH"),
            Self::Relink => write!(f, "RELINK"),
            Self::Timeline => write!(f, "TIMELINE"),
            Self::Export => write!(f, "EXPORT"),
            Self::Qc => write!(f, "QC"),
            Self::General => write!(f, "GENERAL"),
        }
    }
}

/// A single conform log entry.
#[derive(Debug, Clone)]
pub struct ConformLogEntry {
    /// Unique sequence number.
    pub sequence: u64,
    /// Timestamp of the event.
    pub timestamp: SystemTime,
    /// Severity level.
    pub level: LogLevel,
    /// Category.
    pub category: LogCategory,
    /// Main message.
    pub message: String,
    /// Optional clip or asset identifier.
    pub clip_id: Option<String>,
    /// Optional source path.
    pub source_path: Option<String>,
    /// Additional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ConformLogEntry {
    /// Create a new log entry.
    pub fn new(level: LogLevel, category: LogCategory, message: impl Into<String>) -> Self {
        Self {
            sequence: 0,
            timestamp: SystemTime::now(),
            level,
            category,
            message: message.into(),
            clip_id: None,
            source_path: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach a clip ID to this entry.
    pub fn with_clip_id(mut self, clip_id: impl Into<String>) -> Self {
        self.clip_id = Some(clip_id.into());
        self
    }

    /// Attach a source path to this entry.
    pub fn with_source(mut self, path: impl Into<String>) -> Self {
        self.source_path = Some(path.into());
        self
    }

    /// Add a metadata key-value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Format the entry as a human-readable string.
    #[must_use]
    pub fn format_line(&self) -> String {
        let clip = self.clip_id.as_deref().unwrap_or("-");
        format!(
            "[{}] [{}] clip={} {}",
            self.level, self.category, clip, self.message
        )
    }
}

/// A filter for querying log entries.
#[derive(Debug, Clone, Default)]
pub struct LogFilter {
    /// Minimum log level to include.
    pub min_level: Option<LogLevel>,
    /// Filter by category.
    pub category: Option<LogCategory>,
    /// Filter by clip ID substring.
    pub clip_id_contains: Option<String>,
    /// Filter by message substring.
    pub message_contains: Option<String>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl LogFilter {
    /// Create a new empty filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum level.
    #[must_use]
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    /// Set category filter.
    #[must_use]
    pub fn with_category(mut self, category: LogCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Set message substring filter.
    pub fn with_message_contains(mut self, text: impl Into<String>) -> Self {
        self.message_contains = Some(text.into());
        self
    }

    /// Set result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if an entry matches this filter.
    #[must_use]
    pub fn matches(&self, entry: &ConformLogEntry) -> bool {
        if let Some(min) = self.min_level {
            if entry.level < min {
                return false;
            }
        }
        if let Some(cat) = self.category {
            if entry.category != cat {
                return false;
            }
        }
        if let Some(ref sub) = self.clip_id_contains {
            if let Some(ref clip) = entry.clip_id {
                if !clip.contains(sub.as_str()) {
                    return false;
                }
            } else {
                return false;
            }
        }
        if let Some(ref sub) = self.message_contains {
            if !entry.message.contains(sub.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Summary statistics for a conform log.
#[derive(Debug, Clone, Default)]
pub struct LogSummary {
    /// Total entries.
    pub total_entries: usize,
    /// Count by level.
    pub by_level: HashMap<LogLevel, usize>,
    /// Count by category.
    pub by_category: HashMap<LogCategory, usize>,
    /// Number of distinct clips referenced.
    pub distinct_clips: usize,
}

/// The conform log aggregator.
#[derive(Debug, Clone)]
pub struct ConformLog {
    /// All log entries in chronological order.
    entries: Vec<ConformLogEntry>,
    /// Next sequence number.
    next_seq: u64,
}

impl Default for ConformLog {
    fn default() -> Self {
        Self::new()
    }
}

impl ConformLog {
    /// Create a new empty log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 1,
        }
    }

    /// Add an entry to the log, assigning a sequence number.
    pub fn add(&mut self, mut entry: ConformLogEntry) {
        entry.sequence = self.next_seq;
        self.next_seq += 1;
        self.entries.push(entry);
    }

    /// Add a simple info message.
    pub fn info(&mut self, category: LogCategory, message: impl Into<String>) {
        self.add(ConformLogEntry::new(LogLevel::Info, category, message));
    }

    /// Add a warning message.
    pub fn warn(&mut self, category: LogCategory, message: impl Into<String>) {
        self.add(ConformLogEntry::new(LogLevel::Warning, category, message));
    }

    /// Add an error message.
    pub fn error(&mut self, category: LogCategory, message: impl Into<String>) {
        self.add(ConformLogEntry::new(LogLevel::Error, category, message));
    }

    /// Get the total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Query entries using a filter.
    #[must_use]
    pub fn query(&self, filter: &LogFilter) -> Vec<&ConformLogEntry> {
        let mut results: Vec<&ConformLogEntry> =
            self.entries.iter().filter(|e| filter.matches(e)).collect();
        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }
        results
    }

    /// Generate a summary of the log.
    #[must_use]
    pub fn summary(&self) -> LogSummary {
        let mut by_level: HashMap<LogLevel, usize> = HashMap::new();
        let mut by_category: HashMap<LogCategory, usize> = HashMap::new();
        let mut clips = std::collections::HashSet::new();

        for entry in &self.entries {
            *by_level.entry(entry.level).or_insert(0) += 1;
            *by_category.entry(entry.category).or_insert(0) += 1;
            if let Some(ref cid) = entry.clip_id {
                clips.insert(cid.clone());
            }
        }

        LogSummary {
            total_entries: self.entries.len(),
            by_level,
            by_category,
            distinct_clips: clips.len(),
        }
    }

    /// Get all entries (read-only).
    #[must_use]
    pub fn entries(&self) -> &[ConformLogEntry] {
        &self.entries
    }

    /// Check if there are any errors or critical entries.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.level == LogLevel::Error || e.level == LogLevel::Critical)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_seq = 1;
    }

    /// Format the entire log as lines.
    #[must_use]
    pub fn format_all(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(ConformLogEntry::format_line)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_creation() {
        let entry = ConformLogEntry::new(LogLevel::Info, LogCategory::Session, "Session started");
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.category, LogCategory::Session);
        assert_eq!(entry.message, "Session started");
    }

    #[test]
    fn test_log_entry_with_clip_id() {
        let entry = ConformLogEntry::new(LogLevel::Info, LogCategory::Match, "Matched")
            .with_clip_id("clip_001");
        assert_eq!(entry.clip_id.as_deref(), Some("clip_001"));
    }

    #[test]
    fn test_log_entry_with_source() {
        let entry = ConformLogEntry::new(LogLevel::Info, LogCategory::Import, "Imported")
            .with_source("/media/file.mxf");
        assert_eq!(entry.source_path.as_deref(), Some("/media/file.mxf"));
    }

    #[test]
    fn test_log_entry_with_meta() {
        let entry = ConformLogEntry::new(LogLevel::Debug, LogCategory::General, "Test")
            .with_meta("key", "value");
        assert_eq!(entry.metadata.get("key").map(String::as_str), Some("value"));
    }

    #[test]
    fn test_log_entry_format_line() {
        let entry = ConformLogEntry::new(LogLevel::Warning, LogCategory::Match, "No match found")
            .with_clip_id("clip_002");
        let line = entry.format_line();
        assert!(line.contains("WARN"));
        assert!(line.contains("MATCH"));
        assert!(line.contains("clip_002"));
    }

    #[test]
    fn test_conform_log_add_and_len() {
        let mut log = ConformLog::new();
        assert!(log.is_empty());
        log.info(LogCategory::Session, "Start");
        log.info(LogCategory::Session, "End");
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_conform_log_sequence_numbers() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "First");
        log.info(LogCategory::Session, "Second");
        assert_eq!(log.entries()[0].sequence, 1);
        assert_eq!(log.entries()[1].sequence, 2);
    }

    #[test]
    fn test_conform_log_warn_and_error() {
        let mut log = ConformLog::new();
        log.warn(LogCategory::Qc, "Potential issue");
        log.error(LogCategory::Match, "Match failed");
        assert!(log.has_errors());
    }

    #[test]
    fn test_conform_log_no_errors() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "All good");
        assert!(!log.has_errors());
    }

    #[test]
    fn test_log_filter_by_level() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "Info");
        log.warn(LogCategory::Qc, "Warning");
        log.error(LogCategory::Match, "Error");

        let filter = LogFilter::new().with_min_level(LogLevel::Warning);
        let results = log.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_log_filter_by_category() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "Session info");
        log.info(LogCategory::Match, "Match info");
        log.info(LogCategory::Export, "Export info");

        let filter = LogFilter::new().with_category(LogCategory::Match);
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].message, "Match info");
    }

    #[test]
    fn test_log_filter_by_message() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Match, "Found exact match");
        log.info(LogCategory::Match, "No match found");

        let filter = LogFilter::new().with_message_contains("exact");
        let results = log.query(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_log_filter_with_limit() {
        let mut log = ConformLog::new();
        for i in 0..10 {
            log.info(LogCategory::Session, format!("Entry {i}"));
        }
        let filter = LogFilter::new().with_limit(3);
        let results = log.query(&filter);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_log_summary() {
        let mut log = ConformLog::new();
        log.add(ConformLogEntry::new(LogLevel::Info, LogCategory::Match, "M1").with_clip_id("c1"));
        log.add(ConformLogEntry::new(LogLevel::Info, LogCategory::Match, "M2").with_clip_id("c2"));
        log.add(
            ConformLogEntry::new(LogLevel::Error, LogCategory::Qc, "QC fail").with_clip_id("c1"),
        );

        let summary = log.summary();
        assert_eq!(summary.total_entries, 3);
        assert_eq!(*summary.by_level.get(&LogLevel::Info).unwrap_or(&0), 2);
        assert_eq!(*summary.by_level.get(&LogLevel::Error).unwrap_or(&0), 1);
        assert_eq!(summary.distinct_clips, 2);
    }

    #[test]
    fn test_log_clear() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "Data");
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_log_format_all() {
        let mut log = ConformLog::new();
        log.info(LogCategory::Session, "Started");
        let lines = log.format_all();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("INFO"));
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(format!("{}", LogLevel::Debug), "DEBUG");
        assert_eq!(format!("{}", LogLevel::Critical), "CRITICAL");
    }

    #[test]
    fn test_log_category_display() {
        assert_eq!(format!("{}", LogCategory::Session), "SESSION");
        assert_eq!(format!("{}", LogCategory::Relink), "RELINK");
    }
}
