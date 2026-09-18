//! Workflow audit logging: immutable event log, actor tracking, and compliance export.

#![allow(dead_code)]

use std::collections::VecDeque;

/// The actor that performed an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Actor {
    /// Unique identifier (user ID, service account, etc.).
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Actor type.
    pub kind: ActorKind,
}

impl Actor {
    /// Create a human actor.
    #[must_use]
    pub fn human(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            display_name: name.to_string(),
            kind: ActorKind::Human,
        }
    }

    /// Create a system/service actor.
    #[must_use]
    pub fn system(id: &str) -> Self {
        Self {
            id: id.to_string(),
            display_name: id.to_string(),
            kind: ActorKind::System,
        }
    }
}

/// Kind of actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActorKind {
    /// Human user.
    Human,
    /// Automated system or service.
    System,
    /// External API caller.
    ExternalApi,
}

/// Category of an audit event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditEventKind {
    /// Workflow was created.
    WorkflowCreated,
    /// Workflow was started.
    WorkflowStarted,
    /// Workflow was paused.
    WorkflowPaused,
    /// Workflow was resumed.
    WorkflowResumed,
    /// Workflow was cancelled.
    WorkflowCancelled,
    /// Workflow completed successfully.
    WorkflowCompleted,
    /// Workflow failed.
    WorkflowFailed,
    /// A task started.
    TaskStarted,
    /// A task completed.
    TaskCompleted,
    /// A task failed.
    TaskFailed,
    /// A task was retried.
    TaskRetried,
    /// Configuration was changed.
    ConfigChanged,
    /// Permission was changed.
    PermissionChanged,
    /// A custom application event.
    Custom(String),
}

impl AuditEventKind {
    /// Return a string label for the event kind.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::WorkflowCreated => "WORKFLOW_CREATED".to_string(),
            Self::WorkflowStarted => "WORKFLOW_STARTED".to_string(),
            Self::WorkflowPaused => "WORKFLOW_PAUSED".to_string(),
            Self::WorkflowResumed => "WORKFLOW_RESUMED".to_string(),
            Self::WorkflowCancelled => "WORKFLOW_CANCELLED".to_string(),
            Self::WorkflowCompleted => "WORKFLOW_COMPLETED".to_string(),
            Self::WorkflowFailed => "WORKFLOW_FAILED".to_string(),
            Self::TaskStarted => "TASK_STARTED".to_string(),
            Self::TaskCompleted => "TASK_COMPLETED".to_string(),
            Self::TaskFailed => "TASK_FAILED".to_string(),
            Self::TaskRetried => "TASK_RETRIED".to_string(),
            Self::ConfigChanged => "CONFIG_CHANGED".to_string(),
            Self::PermissionChanged => "PERMISSION_CHANGED".to_string(),
            Self::Custom(s) => format!("CUSTOM:{s}"),
        }
    }
}

/// A single immutable audit event.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Event kind.
    pub kind: AuditEventKind,
    /// Actor who triggered the event.
    pub actor: Actor,
    /// Workflow ID this event relates to.
    pub workflow_id: String,
    /// Optional task ID this event relates to.
    pub task_id: Option<String>,
    /// Wall-clock timestamp as a Unix epoch second (for deterministic tests we use u64).
    pub timestamp_epoch_secs: u64,
    /// Optional free-form detail message.
    pub detail: Option<String>,
    /// Key-value metadata.
    pub metadata: Vec<(String, String)>,
}

impl AuditEvent {
    fn label(&self) -> String {
        self.kind.label()
    }
}

/// The append-only audit log.
#[derive(Debug, Default)]
pub struct AuditLog {
    events: VecDeque<AuditEvent>,
    next_sequence: u64,
    max_capacity: Option<usize>,
}

impl AuditLog {
    /// Create a new unbounded audit log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an audit log with a maximum number of retained events (oldest are dropped).
    #[must_use]
    pub fn with_capacity(max: usize) -> Self {
        Self {
            max_capacity: Some(max),
            ..Default::default()
        }
    }

    /// Append a new event to the log.
    pub fn append(
        &mut self,
        kind: AuditEventKind,
        actor: Actor,
        workflow_id: &str,
        task_id: Option<&str>,
        timestamp: u64,
        detail: Option<String>,
    ) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        let event = AuditEvent {
            sequence: seq,
            kind,
            actor,
            workflow_id: workflow_id.to_string(),
            task_id: task_id.map(std::string::ToString::to_string),
            timestamp_epoch_secs: timestamp,
            detail,
            metadata: Vec::new(),
        };
        if let Some(max) = self.max_capacity {
            if self.events.len() >= max {
                self.events.pop_front();
            }
        }
        self.events.push_back(event);
        seq
    }

    /// Append an event with additional metadata.
    pub fn append_with_metadata(
        &mut self,
        kind: AuditEventKind,
        actor: Actor,
        workflow_id: &str,
        task_id: Option<&str>,
        timestamp: u64,
        detail: Option<String>,
        metadata: Vec<(String, String)>,
    ) -> u64 {
        let seq = self.append(kind, actor, workflow_id, task_id, timestamp, detail);
        if let Some(event) = self.events.back_mut() {
            event.metadata = metadata;
        }
        seq
    }

    /// Return the number of stored events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return true if no events are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Filter events by workflow ID.
    #[must_use]
    pub fn events_for_workflow(&self, workflow_id: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.workflow_id == workflow_id)
            .collect()
    }

    /// Filter events by actor ID.
    #[must_use]
    pub fn events_for_actor(&self, actor_id: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.actor.id == actor_id)
            .collect()
    }

    /// Filter events by kind label (string match).
    #[must_use]
    pub fn events_by_kind(&self, kind: &AuditEventKind) -> Vec<&AuditEvent> {
        let label = kind.label();
        self.events
            .iter()
            .filter(|e| e.kind.label() == label)
            .collect()
    }

    /// Return events in a timestamp range (inclusive).
    #[must_use]
    pub fn events_in_range(&self, from_epoch: u64, to_epoch: u64) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_epoch_secs >= from_epoch && e.timestamp_epoch_secs <= to_epoch)
            .collect()
    }

    /// Export all events as CSV lines for compliance reporting.
    #[must_use]
    pub fn export_csv(&self) -> Vec<String> {
        let mut lines = vec![
            "sequence,timestamp,kind,actor_id,actor_name,workflow_id,task_id,detail".to_string(),
        ];
        for e in &self.events {
            let task_id = e.task_id.as_deref().unwrap_or("");
            let detail = e.detail.as_deref().unwrap_or("").replace(',', ";");
            lines.push(format!(
                "{},{},{},{},{},{},{},{}",
                e.sequence,
                e.timestamp_epoch_secs,
                e.label(),
                e.actor.id,
                e.actor.display_name,
                e.workflow_id,
                task_id,
                detail,
            ));
        }
        lines
    }

    /// Return next sequence number (useful for assertions in tests).
    #[must_use]
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }
}

/// Compliance export format options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplianceFormat {
    /// Comma-separated values.
    Csv,
    /// Newline-delimited JSON.
    NdJson,
    /// Plain text summary.
    Text,
}

/// Generate a compliance export in the requested format.
#[must_use]
pub fn export_compliance(log: &AuditLog, format: ComplianceFormat) -> String {
    match format {
        ComplianceFormat::Csv => log.export_csv().join("\n"),
        ComplianceFormat::NdJson => log
            .events
            .iter()
            .map(|e| {
                format!(
                    r#"{{"seq":{},"ts":{},"kind":"{}","actor":"{}","wf":"{}"}}"#,
                    e.sequence,
                    e.timestamp_epoch_secs,
                    e.label(),
                    e.actor.id,
                    e.workflow_id,
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        ComplianceFormat::Text => {
            let mut out = String::new();
            for e in &log.events {
                out.push_str(&format!(
                    "[{}] {} actor={} wf={}\n",
                    e.sequence,
                    e.label(),
                    e.actor.id,
                    e.workflow_id,
                ));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn human() -> Actor {
        Actor::human("user-1", "Alice")
    }

    fn system() -> Actor {
        Actor::system("scheduler")
    }

    #[test]
    fn test_append_increments_sequence() {
        let mut log = AuditLog::new();
        let s1 = log.append(
            AuditEventKind::WorkflowCreated,
            human(),
            "wf-1",
            None,
            1000,
            None,
        );
        let s2 = log.append(
            AuditEventKind::WorkflowStarted,
            human(),
            "wf-1",
            None,
            1001,
            None,
        );
        assert_eq!(s1, 0);
        assert_eq!(s2, 1);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_log_is_empty_initially() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_events_for_workflow() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowCreated,
            human(),
            "wf-1",
            None,
            1000,
            None,
        );
        log.append(
            AuditEventKind::WorkflowCreated,
            human(),
            "wf-2",
            None,
            1001,
            None,
        );
        let wf1_events = log.events_for_workflow("wf-1");
        assert_eq!(wf1_events.len(), 1);
        assert_eq!(wf1_events[0].workflow_id, "wf-1");
    }

    #[test]
    fn test_events_for_actor() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowStarted,
            human(),
            "wf-1",
            None,
            1000,
            None,
        );
        log.append(
            AuditEventKind::WorkflowStarted,
            system(),
            "wf-2",
            None,
            1001,
            None,
        );
        assert_eq!(log.events_for_actor("user-1").len(), 1);
        assert_eq!(log.events_for_actor("scheduler").len(), 1);
    }

    #[test]
    fn test_events_by_kind() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowCreated,
            human(),
            "wf-1",
            None,
            1000,
            None,
        );
        log.append(
            AuditEventKind::WorkflowStarted,
            human(),
            "wf-1",
            None,
            1001,
            None,
        );
        log.append(
            AuditEventKind::WorkflowCreated,
            system(),
            "wf-2",
            None,
            1002,
            None,
        );
        let created = log.events_by_kind(&AuditEventKind::WorkflowCreated);
        assert_eq!(created.len(), 2);
    }

    #[test]
    fn test_events_in_range() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::TaskStarted,
            human(),
            "wf-1",
            Some("t1"),
            100,
            None,
        );
        log.append(
            AuditEventKind::TaskCompleted,
            human(),
            "wf-1",
            Some("t1"),
            200,
            None,
        );
        log.append(
            AuditEventKind::TaskStarted,
            human(),
            "wf-1",
            Some("t2"),
            300,
            None,
        );
        let range = log.events_in_range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].timestamp_epoch_secs, 200);
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut log = AuditLog::with_capacity(3);
        for i in 0..5u64 {
            log.append(AuditEventKind::TaskStarted, system(), "wf-1", None, i, None);
        }
        assert_eq!(log.len(), 3);
        // Oldest (seq=0,1) should be gone; youngest (seq=2,3,4) remain.
        let events = log.events_for_workflow("wf-1");
        assert_eq!(events[0].sequence, 2);
    }

    #[test]
    fn test_export_csv_header() {
        let log = AuditLog::new();
        let csv = log.export_csv();
        assert_eq!(csv.len(), 1);
        assert!(csv[0].starts_with("sequence,"));
    }

    #[test]
    fn test_export_csv_rows() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowCompleted,
            human(),
            "wf-1",
            None,
            9999,
            Some("done".to_string()),
        );
        let csv = log.export_csv();
        assert_eq!(csv.len(), 2);
        assert!(csv[1].contains("WORKFLOW_COMPLETED"));
        assert!(csv[1].contains("user-1"));
    }

    #[test]
    fn test_export_compliance_ndjson() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowFailed,
            system(),
            "wf-9",
            None,
            5000,
            None,
        );
        let output = export_compliance(&log, ComplianceFormat::NdJson);
        assert!(output.contains("WORKFLOW_FAILED"));
        assert!(output.contains("wf-9"));
    }

    #[test]
    fn test_export_compliance_text() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::WorkflowPaused,
            human(),
            "wf-3",
            None,
            2000,
            None,
        );
        let output = export_compliance(&log, ComplianceFormat::Text);
        assert!(output.contains("WORKFLOW_PAUSED"));
        assert!(output.contains("wf-3"));
    }

    #[test]
    fn test_append_with_metadata() {
        let mut log = AuditLog::new();
        let meta = vec![("key".to_string(), "value".to_string())];
        log.append_with_metadata(
            AuditEventKind::ConfigChanged,
            human(),
            "wf-1",
            None,
            3000,
            None,
            meta,
        );
        let events = log.events_for_workflow("wf-1");
        assert_eq!(events[0].metadata.len(), 1);
        assert_eq!(events[0].metadata[0].0, "key");
    }

    #[test]
    fn test_audit_event_kind_labels() {
        assert_eq!(AuditEventKind::WorkflowCreated.label(), "WORKFLOW_CREATED");
        assert_eq!(AuditEventKind::TaskRetried.label(), "TASK_RETRIED");
        assert_eq!(
            AuditEventKind::Custom("MY_EVENT".to_string()).label(),
            "CUSTOM:MY_EVENT"
        );
    }

    #[test]
    fn test_actor_kinds() {
        let h = Actor::human("u1", "Bob");
        assert_eq!(h.kind, ActorKind::Human);
        let s = Actor::system("svc");
        assert_eq!(s.kind, ActorKind::System);
    }

    #[test]
    fn test_next_sequence_counter() {
        let mut log = AuditLog::new();
        assert_eq!(log.next_sequence(), 0);
        log.append(
            AuditEventKind::WorkflowCreated,
            human(),
            "wf-1",
            None,
            0,
            None,
        );
        assert_eq!(log.next_sequence(), 1);
    }

    #[test]
    fn test_task_id_captured() {
        let mut log = AuditLog::new();
        log.append(
            AuditEventKind::TaskStarted,
            system(),
            "wf-1",
            Some("task-42"),
            100,
            None,
        );
        let events = log.events_for_workflow("wf-1");
        assert_eq!(events[0].task_id.as_deref(), Some("task-42"));
    }
}
