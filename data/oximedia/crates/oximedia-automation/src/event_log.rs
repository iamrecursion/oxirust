//! Structured event log for the automation system.
//!
//! Records automation events with severity levels and provides
//! filtering, error/warning extraction, and log-level queries.

#![allow(dead_code)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Severity level of an automation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventSeverity {
    /// Verbose diagnostic information
    Trace = 0,
    /// Standard operational information
    Info = 1,
    /// Non-critical issue that may require attention
    Warning = 2,
    /// Recoverable error
    Error = 3,
    /// Unrecoverable critical failure
    Critical = 4,
}

impl EventSeverity {
    /// Returns the numeric level (0–4) of this severity.
    #[must_use]
    pub fn numeric_level(self) -> u8 {
        self as u8
    }

    /// Returns a short string tag for the severity.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
            Self::Critical => "CRIT",
        }
    }
}

/// A single recorded automation event.
#[derive(Debug, Clone)]
pub struct AutomationEvent {
    /// Event severity
    pub severity: EventSeverity,
    /// Source component or subsystem
    pub source: String,
    /// Human-readable message
    pub message: String,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
}

impl AutomationEvent {
    /// Creates a new event with an automatic timestamp.
    #[must_use]
    pub fn new(severity: EventSeverity, source: &str, message: &str) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            severity,
            source: source.to_owned(),
            message: message.to_owned(),
            timestamp_ms,
        }
    }

    /// Creates a new event with an explicit timestamp (useful for tests).
    #[must_use]
    pub fn with_timestamp(
        severity: EventSeverity,
        source: &str,
        message: &str,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            severity,
            source: source.to_owned(),
            message: message.to_owned(),
            timestamp_ms,
        }
    }

    /// Returns `true` if this event is an error or critical failure.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(
            self.severity,
            EventSeverity::Error | EventSeverity::Critical
        )
    }

    /// Returns `true` if this event is a warning.
    #[must_use]
    pub fn is_warning(&self) -> bool {
        matches!(self.severity, EventSeverity::Warning)
    }
}

/// Append-only log of automation events.
#[derive(Debug, Default)]
pub struct EventLog {
    entries: Vec<AutomationEvent>,
}

impl EventLog {
    /// Creates a new empty event log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an event to the log.
    pub fn push(&mut self, event: AutomationEvent) {
        self.entries.push(event);
    }

    /// Convenience: records a new event directly.
    pub fn record(&mut self, severity: EventSeverity, source: &str, message: &str) {
        self.push(AutomationEvent::new(severity, source, message));
    }

    /// Returns all events at or above `Error` severity.
    #[must_use]
    pub fn errors(&self) -> Vec<&AutomationEvent> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }

    /// Returns all `Warning`-level events.
    #[must_use]
    pub fn warnings(&self) -> Vec<&AutomationEvent> {
        self.entries.iter().filter(|e| e.is_warning()).collect()
    }

    /// Returns all events whose severity is exactly `severity`.
    #[must_use]
    pub fn filter_by_severity(&self, severity: EventSeverity) -> Vec<&AutomationEvent> {
        self.entries
            .iter()
            .filter(|e| e.severity == severity)
            .collect()
    }

    /// Returns the total number of events in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(sev: EventSeverity) -> AutomationEvent {
        AutomationEvent::with_timestamp(sev, "test", "msg", 1_000)
    }

    #[test]
    fn test_severity_numeric_level_trace() {
        assert_eq!(EventSeverity::Trace.numeric_level(), 0);
    }

    #[test]
    fn test_severity_numeric_level_critical() {
        assert_eq!(EventSeverity::Critical.numeric_level(), 4);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(EventSeverity::Error > EventSeverity::Warning);
        assert!(EventSeverity::Warning > EventSeverity::Info);
    }

    #[test]
    fn test_severity_tag() {
        assert_eq!(EventSeverity::Warning.tag(), "WARN");
        assert_eq!(EventSeverity::Critical.tag(), "CRIT");
    }

    #[test]
    fn test_event_is_error_for_error() {
        assert!(make_event(EventSeverity::Error).is_error());
    }

    #[test]
    fn test_event_is_error_for_critical() {
        assert!(make_event(EventSeverity::Critical).is_error());
    }

    #[test]
    fn test_event_is_not_error_for_warning() {
        assert!(!make_event(EventSeverity::Warning).is_error());
    }

    #[test]
    fn test_event_is_warning() {
        assert!(make_event(EventSeverity::Warning).is_warning());
        assert!(!make_event(EventSeverity::Info).is_warning());
    }

    #[test]
    fn test_log_push_and_len() {
        let mut log = EventLog::new();
        log.push(make_event(EventSeverity::Info));
        log.push(make_event(EventSeverity::Warning));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_is_empty() {
        let log = EventLog::new();
        assert!(log.is_empty());
    }

    #[test]
    fn test_log_errors() {
        let mut log = EventLog::new();
        log.push(make_event(EventSeverity::Info));
        log.push(make_event(EventSeverity::Error));
        log.push(make_event(EventSeverity::Critical));
        assert_eq!(log.errors().len(), 2);
    }

    #[test]
    fn test_log_warnings() {
        let mut log = EventLog::new();
        log.push(make_event(EventSeverity::Warning));
        log.push(make_event(EventSeverity::Error));
        assert_eq!(log.warnings().len(), 1);
    }

    #[test]
    fn test_log_filter_by_severity() {
        let mut log = EventLog::new();
        log.push(make_event(EventSeverity::Info));
        log.push(make_event(EventSeverity::Info));
        log.push(make_event(EventSeverity::Error));
        assert_eq!(log.filter_by_severity(EventSeverity::Info).len(), 2);
        assert_eq!(log.filter_by_severity(EventSeverity::Trace).len(), 0);
    }

    #[test]
    fn test_log_clear() {
        let mut log = EventLog::new();
        log.push(make_event(EventSeverity::Info));
        log.clear();
        assert!(log.is_empty());
    }
}
