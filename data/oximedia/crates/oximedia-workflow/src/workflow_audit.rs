//! Workflow audit trail for compliance and debugging.
//!
//! Provides `AuditEventType`, `WorkflowAuditEntry`, and `WorkflowAudit`
//! for recording an immutable history of workflow lifecycle events.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// AuditEventType
// ---------------------------------------------------------------------------

/// Category of audit event emitted by the workflow engine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    /// Workflow was submitted for execution.
    WorkflowSubmitted,
    /// Workflow moved to a different execution state.
    StateChanged,
    /// A task within the workflow changed state.
    TaskStateChanged,
    /// A user-supplied parameter was updated.
    ParameterUpdated,
    /// Workflow was cancelled by an operator.
    WorkflowCancelled,
    /// An error occurred during execution.
    ErrorOccurred,
    /// Workflow completed (success or failure).
    WorkflowCompleted,
    /// A retry was attempted.
    RetryAttempted,
}

impl AuditEventType {
    /// Returns `true` for event types that represent a state transition.
    #[must_use]
    pub fn is_state_change(&self) -> bool {
        matches!(
            self,
            Self::StateChanged
                | Self::TaskStateChanged
                | Self::WorkflowCancelled
                | Self::WorkflowCompleted
        )
    }

    /// Short string identifier for the event type.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WorkflowSubmitted => "workflow_submitted",
            Self::StateChanged => "state_changed",
            Self::TaskStateChanged => "task_state_changed",
            Self::ParameterUpdated => "parameter_updated",
            Self::WorkflowCancelled => "workflow_cancelled",
            Self::ErrorOccurred => "error_occurred",
            Self::WorkflowCompleted => "workflow_completed",
            Self::RetryAttempted => "retry_attempted",
        }
    }
}

// ---------------------------------------------------------------------------
// WorkflowAuditEntry
// ---------------------------------------------------------------------------

/// A single entry in the workflow audit trail.
#[derive(Debug, Clone)]
pub struct WorkflowAuditEntry {
    /// Workflow run this entry belongs to.
    pub run_id: String,
    /// Type of event recorded.
    pub event_type: AuditEventType,
    /// Unix timestamp (seconds since epoch) when the event occurred.
    pub timestamp_secs: u64,
    /// Optional actor (user or service) that triggered the event.
    pub actor: Option<String>,
    /// Raw context payload (e.g. serialised diff).
    pub context: Option<String>,
}

impl WorkflowAuditEntry {
    /// Create a new entry with the current wall-clock time.
    #[must_use]
    pub fn new(run_id: impl Into<String>, event_type: AuditEventType) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            run_id: run_id.into(),
            event_type,
            timestamp_secs,
            actor: None,
            context: None,
        }
    }

    /// Create an entry with an explicit timestamp (useful in tests).
    #[must_use]
    pub fn with_timestamp(
        run_id: impl Into<String>,
        event_type: AuditEventType,
        timestamp_secs: u64,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            event_type,
            timestamp_secs,
            actor: None,
            context: None,
        }
    }

    /// Attach an actor label.
    #[must_use]
    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    /// Attach a context payload.
    #[must_use]
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    /// Human-readable description of this entry.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "[{}] run={} event={}{}",
            self.timestamp_secs,
            self.run_id,
            self.event_type.as_str(),
            self.actor
                .as_ref()
                .map(|a| format!(" actor={a}"))
                .unwrap_or_default(),
        )
    }
}

// ---------------------------------------------------------------------------
// WorkflowAudit
// ---------------------------------------------------------------------------

/// Append-only audit log for all workflow runs.
#[derive(Debug, Default)]
pub struct WorkflowAudit {
    /// Entries indexed by run ID for fast per-run queries.
    index: HashMap<String, Vec<usize>>,
    /// Flat list of all entries in insertion order.
    all: Vec<WorkflowAuditEntry>,
}

impl WorkflowAudit {
    /// Create an empty audit log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry to the audit log.
    pub fn record(&mut self, entry: WorkflowAuditEntry) {
        let idx = self.all.len();
        self.index
            .entry(entry.run_id.clone())
            .or_default()
            .push(idx);
        self.all.push(entry);
    }

    /// Return all entries for a specific run, in insertion order.
    #[must_use]
    pub fn entries_for_run(&self, run_id: &str) -> Vec<&WorkflowAuditEntry> {
        self.index
            .get(run_id)
            .map(|idxs| idxs.iter().map(|&i| &self.all[i]).collect())
            .unwrap_or_default()
    }

    /// Return the most recent `n` entries across all runs.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&WorkflowAuditEntry> {
        self.all.iter().rev().take(n).collect()
    }

    /// Total number of entries across all runs.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.all.len()
    }

    /// All entries (insertion order).
    #[must_use]
    pub fn all_entries(&self) -> &[WorkflowAuditEntry] {
        &self.all
    }

    /// Entries filtered by event type.
    #[must_use]
    pub fn by_event_type(&self, et: &AuditEventType) -> Vec<&WorkflowAuditEntry> {
        self.all.iter().filter(|e| &e.event_type == et).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_is_state_change_true() {
        assert!(AuditEventType::StateChanged.is_state_change());
        assert!(AuditEventType::TaskStateChanged.is_state_change());
        assert!(AuditEventType::WorkflowCancelled.is_state_change());
        assert!(AuditEventType::WorkflowCompleted.is_state_change());
    }

    #[test]
    fn test_event_type_is_state_change_false() {
        assert!(!AuditEventType::WorkflowSubmitted.is_state_change());
        assert!(!AuditEventType::ParameterUpdated.is_state_change());
        assert!(!AuditEventType::ErrorOccurred.is_state_change());
        assert!(!AuditEventType::RetryAttempted.is_state_change());
    }

    #[test]
    fn test_event_type_as_str() {
        assert_eq!(
            AuditEventType::WorkflowSubmitted.as_str(),
            "workflow_submitted"
        );
        assert_eq!(AuditEventType::StateChanged.as_str(), "state_changed");
        assert_eq!(AuditEventType::ErrorOccurred.as_str(), "error_occurred");
    }

    #[test]
    fn test_entry_description_no_actor() {
        let e = WorkflowAuditEntry::with_timestamp("run1", AuditEventType::StateChanged, 1000);
        let desc = e.description();
        assert!(desc.contains("run1"));
        assert!(desc.contains("state_changed"));
        assert!(desc.contains("1000"));
    }

    #[test]
    fn test_entry_description_with_actor() {
        let e = WorkflowAuditEntry::with_timestamp("run2", AuditEventType::StateChanged, 2000)
            .with_actor("operator");
        let desc = e.description();
        assert!(desc.contains("actor=operator"));
    }

    #[test]
    fn test_entry_with_context() {
        let e = WorkflowAuditEntry::new("r", AuditEventType::ParameterUpdated)
            .with_context(r#"{"key":"val"}"#);
        assert!(e.context.is_some());
    }

    #[test]
    fn test_audit_record_and_total() {
        let mut audit = WorkflowAudit::new();
        audit.record(WorkflowAuditEntry::new(
            "r1",
            AuditEventType::WorkflowSubmitted,
        ));
        audit.record(WorkflowAuditEntry::new("r1", AuditEventType::StateChanged));
        audit.record(WorkflowAuditEntry::new(
            "r2",
            AuditEventType::WorkflowSubmitted,
        ));
        assert_eq!(audit.total_entries(), 3);
    }

    #[test]
    fn test_entries_for_run() {
        let mut audit = WorkflowAudit::new();
        audit.record(WorkflowAuditEntry::new(
            "run1",
            AuditEventType::WorkflowSubmitted,
        ));
        audit.record(WorkflowAuditEntry::new(
            "run2",
            AuditEventType::WorkflowSubmitted,
        ));
        audit.record(WorkflowAuditEntry::new(
            "run1",
            AuditEventType::WorkflowCompleted,
        ));
        let entries = audit.entries_for_run("run1");
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_entries_for_unknown_run() {
        let audit = WorkflowAudit::new();
        assert!(audit.entries_for_run("ghost").is_empty());
    }

    #[test]
    fn test_recent_entries() {
        let mut audit = WorkflowAudit::new();
        for i in 0..5u64 {
            audit.record(WorkflowAuditEntry::with_timestamp(
                format!("run{i}"),
                AuditEventType::WorkflowSubmitted,
                i,
            ));
        }
        let recent = audit.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent first (run4 has timestamp 4)
        assert_eq!(recent[0].timestamp_secs, 4);
    }

    #[test]
    fn test_by_event_type() {
        let mut audit = WorkflowAudit::new();
        audit.record(WorkflowAuditEntry::new("r", AuditEventType::ErrorOccurred));
        audit.record(WorkflowAuditEntry::new("r", AuditEventType::StateChanged));
        audit.record(WorkflowAuditEntry::new("r", AuditEventType::ErrorOccurred));
        let errors = audit.by_event_type(&AuditEventType::ErrorOccurred);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_all_entries_insertion_order() {
        let mut audit = WorkflowAudit::new();
        audit.record(WorkflowAuditEntry::with_timestamp(
            "r",
            AuditEventType::WorkflowSubmitted,
            1,
        ));
        audit.record(WorkflowAuditEntry::with_timestamp(
            "r",
            AuditEventType::WorkflowCompleted,
            2,
        ));
        let all = audit.all_entries();
        assert_eq!(all[0].timestamp_secs, 1);
        assert_eq!(all[1].timestamp_secs, 2);
    }
}
