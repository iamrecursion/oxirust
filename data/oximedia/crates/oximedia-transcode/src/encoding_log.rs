#![allow(dead_code)]
//! Structured log for encoding sessions.

use std::time::{Duration, SystemTime};

/// Categories of events that can occur during an encoding session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EncodingEvent {
    /// A non-fatal advisory message.
    Warning(String),
    /// A fatal error that stopped encoding.
    Error(String),
    /// A milestone reporting percentage completion.
    Progress(u8),
    /// A phase boundary (e.g. "pass 1 complete").
    Phase(String),
    /// An informational note.
    Info(String),
}

impl EncodingEvent {
    /// Returns `true` for error-level events.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns `true` for warning-level events.
    #[must_use]
    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warning(_))
    }

    /// Returns `true` for progress milestone events.
    #[must_use]
    pub fn is_progress(&self) -> bool {
        matches!(self, Self::Progress(_))
    }

    /// Extract the human-readable message, if any.
    #[must_use]
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::Warning(m) | Self::Error(m) | Self::Phase(m) | Self::Info(m) => Some(m),
            Self::Progress(_) => None,
        }
    }
}

/// A single entry in the encoding log.
#[derive(Debug, Clone)]
pub struct EncodingLogEntry {
    /// The event that was logged.
    pub event: EncodingEvent,
    /// Wall-clock time when the event occurred.
    pub timestamp: SystemTime,
    /// Session-relative elapsed time at the moment of the event.
    pub elapsed: Duration,
}

impl EncodingLogEntry {
    /// Create a new log entry.
    #[must_use]
    pub fn new(event: EncodingEvent, timestamp: SystemTime, elapsed: Duration) -> Self {
        Self {
            event,
            timestamp,
            elapsed,
        }
    }

    /// Returns `true` if the entry was recorded less than `window` ago.
    #[must_use]
    pub fn is_recent(&self, window: Duration) -> bool {
        self.timestamp
            .elapsed()
            .map(|age| age < window)
            .unwrap_or(false)
    }
}

/// A complete log of encoding events for one session.
#[derive(Debug, Default)]
pub struct EncodingLog {
    entries: Vec<EncodingLogEntry>,
    session_start: Option<SystemTime>,
}

impl EncodingLog {
    /// Create an empty log, recording the current time as session start.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            session_start: Some(SystemTime::now()),
        }
    }

    /// Record a new event, automatically computing elapsed time.
    pub fn record(&mut self, event: EncodingEvent) {
        let now = SystemTime::now();
        let elapsed = self
            .session_start
            .and_then(|s| now.duration_since(s).ok())
            .unwrap_or(Duration::ZERO);
        self.entries
            .push(EncodingLogEntry::new(event, now, elapsed));
    }

    /// All error entries.
    #[must_use]
    pub fn errors(&self) -> Vec<&EncodingLogEntry> {
        self.entries.iter().filter(|e| e.event.is_error()).collect()
    }

    /// All warning entries.
    #[must_use]
    pub fn warnings(&self) -> Vec<&EncodingLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.event.is_warning())
            .collect()
    }

    /// All progress milestone entries.
    #[must_use]
    pub fn progress_events(&self) -> Vec<&EncodingLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.event.is_progress())
            .collect()
    }

    /// All entries in insertion order.
    #[must_use]
    pub fn all_entries(&self) -> &[EncodingLogEntry] {
        &self.entries
    }

    /// Total number of logged entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if any fatal error was recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.entries.iter().any(|e| e.event.is_error())
    }

    /// The most recent progress percentage (0–100), or `None` if not yet reported.
    #[must_use]
    pub fn last_progress_pct(&self) -> Option<u8> {
        self.entries.iter().rev().find_map(|e| {
            if let EncodingEvent::Progress(p) = e.event {
                Some(p)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_event_is_error_true() {
        let e = EncodingEvent::Error("oops".into());
        assert!(e.is_error());
    }

    #[test]
    fn test_encoding_event_is_error_false() {
        assert!(!EncodingEvent::Warning("w".into()).is_error());
        assert!(!EncodingEvent::Progress(50).is_error());
    }

    #[test]
    fn test_encoding_event_is_warning() {
        assert!(EncodingEvent::Warning("low bitrate".into()).is_warning());
        assert!(!EncodingEvent::Info("ok".into()).is_warning());
    }

    #[test]
    fn test_encoding_event_is_progress() {
        assert!(EncodingEvent::Progress(75).is_progress());
        assert!(!EncodingEvent::Error("x".into()).is_progress());
    }

    #[test]
    fn test_encoding_event_message() {
        let e = EncodingEvent::Error("disk full".into());
        assert_eq!(e.message(), Some("disk full"));
        let p = EncodingEvent::Progress(50);
        assert!(p.message().is_none());
    }

    #[test]
    fn test_log_record_and_len() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Info("start".into()));
        log.record(EncodingEvent::Progress(25));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_errors() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Error("bad codec".into()));
        log.record(EncodingEvent::Warning("slow".into()));
        assert_eq!(log.errors().len(), 1);
    }

    #[test]
    fn test_log_warnings() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Warning("W1".into()));
        log.record(EncodingEvent::Warning("W2".into()));
        log.record(EncodingEvent::Error("E".into()));
        assert_eq!(log.warnings().len(), 2);
    }

    #[test]
    fn test_log_progress_events() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Progress(10));
        log.record(EncodingEvent::Progress(50));
        log.record(EncodingEvent::Info("info".into()));
        assert_eq!(log.progress_events().len(), 2);
    }

    #[test]
    fn test_log_has_errors_false() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Warning("w".into()));
        assert!(!log.has_errors());
    }

    #[test]
    fn test_log_has_errors_true() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Error("fatal".into()));
        assert!(log.has_errors());
    }

    #[test]
    fn test_log_last_progress_pct_none() {
        let log = EncodingLog::new();
        assert!(log.last_progress_pct().is_none());
    }

    #[test]
    fn test_log_last_progress_pct_some() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Progress(25));
        log.record(EncodingEvent::Progress(75));
        assert_eq!(log.last_progress_pct(), Some(75));
    }

    #[test]
    fn test_log_is_empty() {
        let log = EncodingLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn test_entry_elapsed_non_negative() {
        let mut log = EncodingLog::new();
        log.record(EncodingEvent::Info("hi".into()));
        assert!(!log.is_empty());
        let entry = &log.all_entries()[0];
        // elapsed should be very small but non-negative
        assert!(entry.elapsed < Duration::from_secs(5));
    }
}
