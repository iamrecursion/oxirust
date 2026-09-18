//! Playout event log.
//!
//! Records structured events (clip transitions, errors, operator actions, etc.)
//! that occur during playout, suitable for audit trails and post-broadcast
//! analysis.

#![allow(dead_code)]

/// Severity level of a log event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventSeverity {
    /// Informational — normal operation.
    Info,
    /// A non-critical warning that requires attention.
    Warning,
    /// An error that may have affected output quality.
    Error,
    /// A critical failure; output may have been interrupted.
    Critical,
}

impl EventSeverity {
    /// Returns `true` for `Warning` and above.
    #[must_use]
    pub fn is_significant(self) -> bool {
        self >= Self::Warning
    }

    /// Human-readable tag.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
            Self::Critical => "CRIT",
        }
    }
}

/// Category of playout event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventCategory {
    /// A clip started or ended.
    ClipTransition,
    /// A graphics element was triggered.
    Graphics,
    /// An automation or GPI event fired.
    Automation,
    /// An operator performed a manual action.
    OperatorAction,
    /// A system error or warning was raised.
    SystemFault,
    /// Audio/video quality check result.
    QualityCheck,
    /// Custom / unclassified event.
    Custom(String),
}

impl EventCategory {
    /// Returns `true` if operator intervention caused this event.
    #[must_use]
    pub fn is_operator_driven(&self) -> bool {
        matches!(self, Self::OperatorAction)
    }
}

/// A single event in the playout log.
#[derive(Debug, Clone)]
pub struct PlayoutEvent {
    /// Sequential event identifier (1-based).
    pub id: u64,
    /// Wall-clock timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Severity level.
    pub severity: EventSeverity,
    /// Category of the event.
    pub category: EventCategory,
    /// Human-readable description.
    pub message: String,
    /// Optional source or component that generated the event.
    pub source: Option<String>,
}

impl PlayoutEvent {
    /// Returns `true` for significant (Warning+) events.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.severity.is_significant()
    }
}

/// Append-only structured event log for a playout session.
#[derive(Debug)]
pub struct EventLog {
    events: Vec<PlayoutEvent>,
    next_id: u64,
    /// Maximum events to retain (oldest are dropped when exceeded).
    max_capacity: usize,
}

impl EventLog {
    /// Create a new log with the given capacity.
    #[must_use]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            events: Vec::new(),
            next_id: 1,
            max_capacity: max_capacity.max(1),
        }
    }

    /// Append a new event.  Returns the assigned event ID.
    pub fn append(
        &mut self,
        timestamp_ms: u64,
        severity: EventSeverity,
        category: EventCategory,
        message: impl Into<String>,
        source: Option<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(PlayoutEvent {
            id,
            timestamp_ms,
            severity,
            category,
            message: message.into(),
            source,
        });

        // Trim oldest entries to stay within capacity.
        if self.events.len() > self.max_capacity {
            let excess = self.events.len() - self.max_capacity;
            self.events.drain(0..excess);
        }

        id
    }

    /// Convenience: append an informational event.
    pub fn info(&mut self, ts: u64, category: EventCategory, msg: impl Into<String>) -> u64 {
        self.append(ts, EventSeverity::Info, category, msg, None)
    }

    /// Convenience: append a warning event.
    pub fn warn(&mut self, ts: u64, category: EventCategory, msg: impl Into<String>) -> u64 {
        self.append(ts, EventSeverity::Warning, category, msg, None)
    }

    /// Convenience: append an error event.
    pub fn error(&mut self, ts: u64, category: EventCategory, msg: impl Into<String>) -> u64 {
        self.append(ts, EventSeverity::Error, category, msg, None)
    }

    /// Total number of events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the log contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// All events at or above the given severity.
    #[must_use]
    pub fn by_severity(&self, min: EventSeverity) -> Vec<&PlayoutEvent> {
        self.events.iter().filter(|e| e.severity >= min).collect()
    }

    /// All events in a given time window (inclusive on both ends).
    #[must_use]
    pub fn in_window(&self, start_ms: u64, end_ms: u64) -> Vec<&PlayoutEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// The most recent N events.
    #[must_use]
    pub fn last_n(&self, n: usize) -> Vec<&PlayoutEvent> {
        let skip = self.events.len().saturating_sub(n);
        self.events[skip..].iter().collect()
    }

    /// Returns `true` if any critical event has been logged.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.events
            .iter()
            .any(|e| e.severity == EventSeverity::Critical)
    }

    /// Clear all stored events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> u64 {
        1_700_000_000_000u64 // fixed epoch-ms for tests
    }

    #[test]
    fn severity_ordering() {
        assert!(EventSeverity::Critical > EventSeverity::Error);
        assert!(EventSeverity::Error > EventSeverity::Warning);
        assert!(EventSeverity::Warning > EventSeverity::Info);
    }

    #[test]
    fn severity_is_significant() {
        assert!(EventSeverity::Warning.is_significant());
        assert!(EventSeverity::Error.is_significant());
        assert!(EventSeverity::Critical.is_significant());
        assert!(!EventSeverity::Info.is_significant());
    }

    #[test]
    fn severity_tags() {
        assert_eq!(EventSeverity::Info.tag(), "INFO");
        assert_eq!(EventSeverity::Critical.tag(), "CRIT");
    }

    #[test]
    fn category_is_operator_driven() {
        assert!(EventCategory::OperatorAction.is_operator_driven());
        assert!(!EventCategory::SystemFault.is_operator_driven());
    }

    #[test]
    fn log_append_increments_id() {
        let mut log = EventLog::new(100);
        let id1 = log.info(ts(), EventCategory::ClipTransition, "clip started");
        let id2 = log.info(ts(), EventCategory::ClipTransition, "clip ended");
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn log_len_and_is_empty() {
        let mut log = EventLog::new(100);
        assert!(log.is_empty());
        log.info(ts(), EventCategory::Graphics, "logo on");
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn log_by_severity() {
        let mut log = EventLog::new(100);
        log.info(ts(), EventCategory::ClipTransition, "ok");
        log.warn(ts(), EventCategory::SystemFault, "diskfull");
        log.error(ts(), EventCategory::SystemFault, "crash");
        assert_eq!(log.by_severity(EventSeverity::Warning).len(), 2);
        assert_eq!(log.by_severity(EventSeverity::Error).len(), 1);
    }

    #[test]
    fn log_in_window() {
        let mut log = EventLog::new(100);
        log.append(
            100,
            EventSeverity::Info,
            EventCategory::Automation,
            "a",
            None,
        );
        log.append(
            200,
            EventSeverity::Info,
            EventCategory::Automation,
            "b",
            None,
        );
        log.append(
            300,
            EventSeverity::Info,
            EventCategory::Automation,
            "c",
            None,
        );
        let window = log.in_window(100, 200);
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn log_last_n() {
        let mut log = EventLog::new(100);
        for i in 0..5u64 {
            log.info(ts() + i, EventCategory::QualityCheck, "ok");
        }
        assert_eq!(log.last_n(3).len(), 3);
    }

    #[test]
    fn log_has_critical() {
        let mut log = EventLog::new(100);
        assert!(!log.has_critical());
        log.append(
            ts(),
            EventSeverity::Critical,
            EventCategory::SystemFault,
            "fatal",
            None,
        );
        assert!(log.has_critical());
    }

    #[test]
    fn log_clear() {
        let mut log = EventLog::new(100);
        log.info(ts(), EventCategory::ClipTransition, "x");
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn log_capacity_respected() {
        let mut log = EventLog::new(3);
        for i in 0..5u64 {
            log.info(ts() + i, EventCategory::Automation, "e");
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn event_is_significant() {
        let mut log = EventLog::new(10);
        log.warn(ts(), EventCategory::QualityCheck, "low audio");
        let ev = &log.events[0];
        assert!(ev.is_significant());
    }
}
