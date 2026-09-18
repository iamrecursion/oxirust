//! Structured workflow event log for `oximedia-workflow`.
//!
//! [`WorkflowLog`] stores an in-memory sequence of [`WorkflowEvent`]s and
//! provides filtering helpers such as [`WorkflowLog::recent_errors`] so that
//! operators can quickly surface problems without trawling the full history.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── Event type ────────────────────────────────────────────────────────────────

/// Discriminates the kind of event stored in a [`WorkflowEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowEventType {
    /// The workflow was submitted to the execution queue.
    Submitted,
    /// A task within the workflow started executing.
    TaskStarted,
    /// A task completed successfully.
    TaskCompleted,
    /// A task failed and will (or will not) be retried.
    TaskFailed,
    /// The workflow completed without any failures.
    WorkflowCompleted,
    /// The workflow was cancelled before completion.
    WorkflowCancelled,
    /// A non-fatal warning was recorded (e.g. resource limit approached).
    Warning,
    /// A fatal error occurred; the workflow cannot continue.
    Error,
}

impl WorkflowEventType {
    /// Returns `true` when this event type indicates a failure or error.
    #[must_use]
    pub fn is_error(self) -> bool {
        matches!(self, Self::TaskFailed | Self::Error)
    }

    /// Returns `true` when this event type signals successful completion.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::TaskCompleted | Self::WorkflowCompleted)
    }

    /// Returns a short label suitable for log output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Submitted => "SUBMITTED",
            Self::TaskStarted => "TASK_START",
            Self::TaskCompleted => "TASK_OK",
            Self::TaskFailed => "TASK_FAIL",
            Self::WorkflowCompleted => "WF_OK",
            Self::WorkflowCancelled => "WF_CANCEL",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

impl std::fmt::Display for WorkflowEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ── Event ─────────────────────────────────────────────────────────────────────

/// A single structured log entry within a [`WorkflowLog`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowEvent {
    /// Monotonically increasing sequence number within the log.
    pub seq: u64,
    /// Seconds since the UNIX epoch when this event was recorded.
    pub timestamp_secs: u64,
    /// The kind of event.
    pub event_type: WorkflowEventType,
    /// Identifier of the workflow this event belongs to.
    pub workflow_id: String,
    /// Optional identifier of the specific task (if task-scoped).
    pub task_id: Option<String>,
    /// Human-readable message providing additional context.
    pub message: String,
}

impl WorkflowEvent {
    /// Creates a workflow-level event using the current system time.
    #[must_use]
    pub fn workflow_event(
        seq: u64,
        event_type: WorkflowEventType,
        workflow_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            timestamp_secs: now_secs(),
            event_type,
            workflow_id: workflow_id.into(),
            task_id: None,
            message: message.into(),
        }
    }

    /// Creates a task-scoped event using the current system time.
    #[must_use]
    pub fn task_event(
        seq: u64,
        event_type: WorkflowEventType,
        workflow_id: impl Into<String>,
        task_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            timestamp_secs: now_secs(),
            event_type,
            workflow_id: workflow_id.into(),
            task_id: Some(task_id.into()),
            message: message.into(),
        }
    }

    /// Returns `true` when this event represents a failure or error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.event_type.is_error()
    }
}

/// Returns seconds since UNIX epoch (best-effort; returns 0 on overflow).
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ── Log ───────────────────────────────────────────────────────────────────────

/// An in-memory log that accumulates [`WorkflowEvent`]s for one or more
/// workflows and exposes query helpers.
#[derive(Debug, Default, Clone)]
pub struct WorkflowLog {
    events: Vec<WorkflowEvent>,
    next_seq: u64,
}

impl WorkflowLog {
    /// Creates a new, empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an event to the log, assigning it the next sequence number.
    pub fn append(
        &mut self,
        event_type: WorkflowEventType,
        workflow_id: impl Into<String>,
        message: impl Into<String>,
    ) {
        let ev = WorkflowEvent::workflow_event(self.next_seq, event_type, workflow_id, message);
        self.next_seq += 1;
        self.events.push(ev);
    }

    /// Appends a task-scoped event to the log.
    pub fn append_task(
        &mut self,
        event_type: WorkflowEventType,
        workflow_id: impl Into<String>,
        task_id: impl Into<String>,
        message: impl Into<String>,
    ) {
        let ev =
            WorkflowEvent::task_event(self.next_seq, event_type, workflow_id, task_id, message);
        self.next_seq += 1;
        self.events.push(ev);
    }

    /// Total number of events in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the log contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns a slice of all events in insertion order.
    #[must_use]
    pub fn all(&self) -> &[WorkflowEvent] {
        &self.events
    }

    /// Returns the `n` most recently appended error events (`TaskFailed` or
    /// Error), newest first.
    #[must_use]
    pub fn recent_errors(&self, n: usize) -> Vec<&WorkflowEvent> {
        self.events
            .iter()
            .rev()
            .filter(|e| e.is_error())
            .take(n)
            .collect()
    }

    /// Returns all events for the given workflow ID.
    #[must_use]
    pub fn events_for_workflow<'a>(&'a self, workflow_id: &str) -> Vec<&'a WorkflowEvent> {
        self.events
            .iter()
            .filter(|e| e.workflow_id == workflow_id)
            .collect()
    }

    /// Returns all events matching the given event type.
    #[must_use]
    pub fn events_of_type(&self, event_type: WorkflowEventType) -> Vec<&WorkflowEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Returns the number of error-type events in the log.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.events.iter().filter(|e| e.is_error()).count()
    }

    /// Clears all events from the log and resets the sequence counter.
    pub fn clear(&mut self) {
        self.events.clear();
        self.next_seq = 0;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log() -> WorkflowLog {
        let mut log = WorkflowLog::new();
        log.append(WorkflowEventType::Submitted, "wf-1", "Workflow submitted");
        log.append_task(
            WorkflowEventType::TaskStarted,
            "wf-1",
            "task-a",
            "Started task-a",
        );
        log.append_task(
            WorkflowEventType::TaskFailed,
            "wf-1",
            "task-a",
            "task-a failed: timeout",
        );
        log.append(WorkflowEventType::Error, "wf-1", "Unrecoverable error");
        log.append_task(
            WorkflowEventType::TaskCompleted,
            "wf-2",
            "task-b",
            "task-b done",
        );
        log
    }

    #[test]
    fn test_new_log_empty() {
        let log = WorkflowLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_append_increments_len() {
        let mut log = WorkflowLog::new();
        log.append(WorkflowEventType::Submitted, "wf-x", "msg");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_sequence_numbers_monotonic() {
        let log = make_log();
        for (i, ev) in log.all().iter().enumerate() {
            assert_eq!(ev.seq, i as u64);
        }
    }

    #[test]
    fn test_recent_errors_count() {
        let log = make_log();
        // Two error events: TaskFailed and Error
        let errors = log.recent_errors(10);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_recent_errors_newest_first() {
        let log = make_log();
        let errors = log.recent_errors(10);
        // The Error event was appended after TaskFailed, so it has higher seq
        assert!(errors[0].seq > errors[1].seq);
    }

    #[test]
    fn test_recent_errors_limit() {
        let log = make_log();
        let errors = log.recent_errors(1);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_events_for_workflow() {
        let log = make_log();
        let wf1 = log.events_for_workflow("wf-1");
        assert_eq!(wf1.len(), 4);
        let wf2 = log.events_for_workflow("wf-2");
        assert_eq!(wf2.len(), 1);
    }

    #[test]
    fn test_events_of_type() {
        let log = make_log();
        let starts = log.events_of_type(WorkflowEventType::TaskStarted);
        assert_eq!(starts.len(), 1);
    }

    #[test]
    fn test_error_count() {
        let log = make_log();
        assert_eq!(log.error_count(), 2);
    }

    #[test]
    fn test_clear_resets_log() {
        let mut log = make_log();
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.error_count(), 0);
    }

    #[test]
    fn test_clear_resets_sequence() {
        let mut log = make_log();
        log.clear();
        log.append(WorkflowEventType::Submitted, "wf-new", "msg");
        assert_eq!(log.all()[0].seq, 0);
    }

    #[test]
    fn test_event_type_is_error() {
        assert!(WorkflowEventType::TaskFailed.is_error());
        assert!(WorkflowEventType::Error.is_error());
        assert!(!WorkflowEventType::TaskCompleted.is_error());
        assert!(!WorkflowEventType::Submitted.is_error());
    }

    #[test]
    fn test_event_type_is_success() {
        assert!(WorkflowEventType::TaskCompleted.is_success());
        assert!(WorkflowEventType::WorkflowCompleted.is_success());
        assert!(!WorkflowEventType::TaskFailed.is_success());
    }

    #[test]
    fn test_event_type_label() {
        assert_eq!(WorkflowEventType::Error.label(), "ERROR");
        assert_eq!(WorkflowEventType::Submitted.label(), "SUBMITTED");
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", WorkflowEventType::Warning), "WARN");
    }

    #[test]
    fn test_task_event_has_task_id() {
        let log = make_log();
        let task_ev = log
            .events_of_type(WorkflowEventType::TaskStarted)
            .into_iter()
            .next()
            .expect("should succeed in test");
        assert_eq!(task_ev.task_id.as_deref(), Some("task-a"));
    }

    #[test]
    fn test_workflow_event_no_task_id() {
        let log = make_log();
        let submitted = log
            .events_of_type(WorkflowEventType::Submitted)
            .into_iter()
            .next()
            .expect("should succeed in test");
        assert!(submitted.task_id.is_none());
    }
}
