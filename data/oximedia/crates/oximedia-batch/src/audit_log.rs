//! Audit log: track who submitted, modified, and cancelled each job.
//!
//! Every state transition and configuration change is recorded as an
//! [`AuditEntry`](crate::audit_log::AuditEntry) with a timestamp, actor, action, and optional details.
//! The log is append-only, thread-safe, and supports filtering and export.

use std::collections::VecDeque;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::types::JobId;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The kind of action that was performed on a job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    /// A new job was submitted.
    JobSubmitted,
    /// A job was started by the execution engine.
    JobStarted,
    /// A job completed successfully.
    JobCompleted,
    /// A job failed.
    JobFailed,
    /// A job was cancelled by an operator.
    JobCancelled,
    /// A job was retried.
    JobRetried {
        /// Which attempt this is (1-based).
        attempt: u32,
    },
    /// A job's priority was changed.
    PriorityChanged {
        /// The old priority level.
        old_priority: String,
        /// The new priority level.
        new_priority: String,
    },
    /// A job was moved to the dead letter queue.
    MovedToDeadLetter,
    /// A job was replayed from the dead letter queue.
    ReplayedFromDeadLetter,
    /// A job's configuration was modified.
    ConfigModified {
        /// Which field was changed.
        field: String,
    },
    /// A job's dependency was added or removed.
    DependencyChanged {
        /// Description of the change.
        change: String,
    },
    /// A checkpoint was saved for a job.
    CheckpointSaved {
        /// Step index at which the checkpoint was taken.
        step: usize,
    },
    /// A custom / free-form action.
    Custom {
        /// Action label.
        action: String,
    },
}

impl std::fmt::Display for AuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobSubmitted => write!(f, "job_submitted"),
            Self::JobStarted => write!(f, "job_started"),
            Self::JobCompleted => write!(f, "job_completed"),
            Self::JobFailed => write!(f, "job_failed"),
            Self::JobCancelled => write!(f, "job_cancelled"),
            Self::JobRetried { attempt } => write!(f, "job_retried(attempt={attempt})"),
            Self::PriorityChanged {
                old_priority,
                new_priority,
            } => write!(f, "priority_changed({old_priority}->{new_priority})"),
            Self::MovedToDeadLetter => write!(f, "moved_to_dead_letter"),
            Self::ReplayedFromDeadLetter => write!(f, "replayed_from_dead_letter"),
            Self::ConfigModified { field } => write!(f, "config_modified({field})"),
            Self::DependencyChanged { change } => write!(f, "dependency_changed({change})"),
            Self::CheckpointSaved { step } => write!(f, "checkpoint_saved(step={step})"),
            Self::Custom { action } => write!(f, "custom({action})"),
        }
    }
}

/// Identifies who performed an action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Actor {
    /// The batch engine (system-initiated).
    System,
    /// A human user.
    User {
        /// Username or identifier.
        username: String,
    },
    /// An API client.
    ApiClient {
        /// Client identifier or API key prefix.
        client_id: String,
    },
    /// A scheduled task or cron trigger.
    Scheduler {
        /// Schedule name or expression.
        schedule: String,
    },
}

impl std::fmt::Display for Actor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User { username } => write!(f, "user:{username}"),
            Self::ApiClient { client_id } => write!(f, "api:{client_id}"),
            Self::Scheduler { schedule } => write!(f, "scheduler:{schedule}"),
        }
    }
}

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID.
    pub entry_id: String,
    /// The job this action pertains to.
    pub job_id: JobId,
    /// Who performed the action.
    pub actor: Actor,
    /// What was done.
    pub action: AuditAction,
    /// Optional human-readable details / reason.
    pub details: Option<String>,
    /// Unix timestamp (seconds since epoch).
    pub timestamp_secs: u64,
    /// IP address or origin (if available).
    pub source_ip: Option<String>,
}

impl AuditEntry {
    /// Create a new audit entry.
    #[must_use]
    pub fn new(job_id: JobId, actor: Actor, action: AuditAction) -> Self {
        Self {
            entry_id: uuid::Uuid::new_v4().to_string(),
            job_id,
            actor,
            action,
            details: None,
            timestamp_secs: current_timestamp(),
            source_ip: None,
        }
    }

    /// Builder: add details.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Builder: add source IP.
    #[must_use]
    pub fn with_source_ip(mut self, ip: impl Into<String>) -> Self {
        self.source_ip = Some(ip.into());
        self
    }

    /// Age of this entry in seconds.
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        current_timestamp().saturating_sub(self.timestamp_secs)
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// Maximum entries kept in memory before oldest are evicted.
const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// Append-only, thread-safe audit log.
#[derive(Debug)]
pub struct AuditLog {
    entries: RwLock<VecDeque<AuditEntry>>,
    max_entries: usize,
}

impl AuditLog {
    /// Create a new audit log with the default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(VecDeque::new()),
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }

    /// Create a new audit log with a custom maximum capacity.
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(VecDeque::new()),
            max_entries: max_entries.max(1),
        }
    }

    /// Append an entry to the log.
    ///
    /// If the log exceeds `max_entries`, the oldest entries are evicted.
    pub fn log(&self, entry: AuditEntry) {
        let mut entries = self.entries.write();
        entries.push_back(entry);
        while entries.len() > self.max_entries {
            entries.pop_front();
        }
    }

    /// Convenience: log a simple action with system actor.
    pub fn log_system(&self, job_id: JobId, action: AuditAction) {
        self.log(AuditEntry::new(job_id, Actor::System, action));
    }

    /// Convenience: log a simple action with a user actor.
    pub fn log_user(&self, job_id: JobId, username: impl Into<String>, action: AuditAction) {
        self.log(AuditEntry::new(
            job_id,
            Actor::User {
                username: username.into(),
            },
            action,
        ));
    }

    /// Total number of entries in the log.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Retrieve all entries for a specific job, in chronological order.
    #[must_use]
    pub fn entries_for_job(&self, job_id: &JobId) -> Vec<AuditEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| &e.job_id == job_id)
            .cloned()
            .collect()
    }

    /// Retrieve all entries by a specific actor.
    #[must_use]
    pub fn entries_by_actor(&self, actor: &Actor) -> Vec<AuditEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| &e.actor == actor)
            .cloned()
            .collect()
    }

    /// Retrieve entries matching a predicate, newest first.
    #[must_use]
    pub fn query<F>(&self, predicate: F) -> Vec<AuditEntry>
    where
        F: Fn(&AuditEntry) -> bool,
    {
        self.entries
            .read()
            .iter()
            .rev()
            .filter(|e| predicate(e))
            .cloned()
            .collect()
    }

    /// Retrieve the most recent `n` entries, newest first.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<AuditEntry> {
        self.entries.read().iter().rev().take(n).cloned().collect()
    }

    /// Retrieve entries within a time range (inclusive).
    #[must_use]
    pub fn entries_in_range(&self, from_secs: u64, to_secs: u64) -> Vec<AuditEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| e.timestamp_secs >= from_secs && e.timestamp_secs <= to_secs)
            .cloned()
            .collect()
    }

    /// Count entries by action type for a specific job.
    #[must_use]
    pub fn action_counts_for_job(
        &self,
        job_id: &JobId,
    ) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for entry in self.entries.read().iter() {
            if &entry.job_id == job_id {
                *counts.entry(entry.action.to_string()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Export the entire log as a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if JSON encoding fails.
    pub fn export_json(&self) -> std::result::Result<String, serde_json::Error> {
        let entries: Vec<AuditEntry> = self.entries.read().iter().cloned().collect();
        serde_json::to_string_pretty(&entries)
    }

    /// Clear the entire log.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Summary statistics.
    #[must_use]
    pub fn stats(&self) -> AuditStats {
        let entries = self.entries.read();
        let total = entries.len();
        let mut by_action = std::collections::HashMap::new();
        let mut by_actor = std::collections::HashMap::new();
        let mut unique_jobs = std::collections::HashSet::new();

        for e in entries.iter() {
            *by_action.entry(e.action.to_string()).or_insert(0usize) += 1;
            *by_actor.entry(e.actor.to_string()).or_insert(0usize) += 1;
            unique_jobs.insert(e.job_id.as_str().to_string());
        }

        AuditStats {
            total_entries: total,
            unique_jobs: unique_jobs.len(),
            by_action,
            by_actor,
        }
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for the audit log.
#[derive(Debug, Clone)]
pub struct AuditStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Number of distinct jobs referenced.
    pub unique_jobs: usize,
    /// Entry count by action type.
    pub by_action: std::collections::HashMap<String, usize>,
    /// Entry count by actor.
    pub by_actor: std::collections::HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn jid(s: &str) -> JobId {
        JobId::from(s)
    }

    #[test]
    fn test_audit_log_basic() {
        let log = AuditLog::new();
        assert!(log.is_empty());
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_audit_log_entries_for_job() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        log.log_system(jid("j1"), AuditAction::JobStarted);
        log.log_system(jid("j2"), AuditAction::JobSubmitted);
        let j1_entries = log.entries_for_job(&jid("j1"));
        assert_eq!(j1_entries.len(), 2);
    }

    #[test]
    fn test_audit_log_entries_by_actor() {
        let log = AuditLog::new();
        log.log_user(jid("j1"), "alice", AuditAction::JobSubmitted);
        log.log_user(jid("j2"), "bob", AuditAction::JobSubmitted);
        log.log_user(jid("j3"), "alice", AuditAction::JobCancelled);

        let alice_entries = log.entries_by_actor(&Actor::User {
            username: "alice".into(),
        });
        assert_eq!(alice_entries.len(), 2);
    }

    #[test]
    fn test_audit_log_recent() {
        let log = AuditLog::new();
        for i in 0..10 {
            log.log_system(jid(&format!("j{i}")), AuditAction::JobSubmitted);
        }
        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        // Newest first: j9, j8, j7
        assert_eq!(recent[0].job_id.as_str(), "j9");
        assert_eq!(recent[1].job_id.as_str(), "j8");
        assert_eq!(recent[2].job_id.as_str(), "j7");
    }

    #[test]
    fn test_audit_log_query_filter() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        log.log_system(jid("j1"), AuditAction::JobStarted);
        log.log_system(jid("j1"), AuditAction::JobFailed);
        log.log_system(jid("j1"), AuditAction::JobRetried { attempt: 1 });
        log.log_system(jid("j1"), AuditAction::JobCompleted);

        let failures = log.query(|e| matches!(&e.action, AuditAction::JobFailed));
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn test_audit_log_eviction() {
        let log = AuditLog::with_capacity(5);
        for i in 0..10 {
            log.log_system(jid(&format!("j{i}")), AuditAction::JobSubmitted);
        }
        assert_eq!(log.len(), 5);
        // Should have j5..j9 (oldest evicted).
        let entries = log.recent(5);
        assert_eq!(entries[0].job_id.as_str(), "j9");
        assert_eq!(entries[4].job_id.as_str(), "j5");
    }

    #[test]
    fn test_audit_log_action_counts() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        log.log_system(jid("j1"), AuditAction::JobStarted);
        log.log_system(jid("j1"), AuditAction::JobFailed);
        log.log_system(jid("j1"), AuditAction::JobRetried { attempt: 1 });
        log.log_system(jid("j1"), AuditAction::JobStarted);
        log.log_system(jid("j1"), AuditAction::JobCompleted);

        let counts = log.action_counts_for_job(&jid("j1"));
        assert_eq!(counts.get("job_submitted").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("job_started").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("job_completed").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("job_failed").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_audit_log_clear() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_audit_log_stats() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        log.log_user(jid("j2"), "alice", AuditAction::JobSubmitted);
        log.log_system(jid("j1"), AuditAction::JobCompleted);

        let stats = log.stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.unique_jobs, 2);
        assert_eq!(
            stats.by_action.get("job_submitted").copied().unwrap_or(0),
            2
        );
    }

    #[test]
    fn test_audit_entry_with_details() {
        let entry = AuditEntry::new(jid("j1"), Actor::System, AuditAction::JobFailed)
            .with_details("OOM killed")
            .with_source_ip("192.168.1.1");
        assert_eq!(entry.details, Some("OOM killed".to_string()));
        assert_eq!(entry.source_ip, Some("192.168.1.1".to_string()));
    }

    #[test]
    fn test_audit_entry_age() {
        let entry = AuditEntry::new(jid("j1"), Actor::System, AuditAction::JobSubmitted);
        assert!(entry.age_secs() < 5);
    }

    #[test]
    fn test_audit_action_display() {
        assert_eq!(AuditAction::JobSubmitted.to_string(), "job_submitted");
        assert_eq!(AuditAction::JobCancelled.to_string(), "job_cancelled");
        assert_eq!(
            AuditAction::JobRetried { attempt: 3 }.to_string(),
            "job_retried(attempt=3)"
        );
        assert_eq!(
            AuditAction::PriorityChanged {
                old_priority: "Normal".into(),
                new_priority: "High".into(),
            }
            .to_string(),
            "priority_changed(Normal->High)"
        );
    }

    #[test]
    fn test_actor_display() {
        assert_eq!(Actor::System.to_string(), "system");
        assert_eq!(
            Actor::User {
                username: "bob".into()
            }
            .to_string(),
            "user:bob"
        );
        assert_eq!(
            Actor::ApiClient {
                client_id: "cli-1".into()
            }
            .to_string(),
            "api:cli-1"
        );
    }

    #[test]
    fn test_audit_log_export_json() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::JobSubmitted);
        let json = log.export_json().expect("JSON export should succeed");
        assert!(json.contains("job_submitted"));
        assert!(json.contains("j1"));
    }

    #[test]
    fn test_audit_log_entries_in_range() {
        let log = AuditLog::new();
        let now = current_timestamp();

        let mut entry1 = AuditEntry::new(jid("j1"), Actor::System, AuditAction::JobSubmitted);
        entry1.timestamp_secs = now - 100;
        log.log(entry1);

        let mut entry2 = AuditEntry::new(jid("j2"), Actor::System, AuditAction::JobStarted);
        entry2.timestamp_secs = now - 50;
        log.log(entry2);

        let entry3 = AuditEntry::new(jid("j3"), Actor::System, AuditAction::JobCompleted);
        log.log(entry3);

        let range = log.entries_in_range(now - 60, now + 10);
        // Should include j2 and j3 but not j1.
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_audit_log_default() {
        let log = AuditLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_audit_log_custom_action() {
        let log = AuditLog::new();
        log.log(AuditEntry::new(
            jid("j1"),
            Actor::System,
            AuditAction::Custom {
                action: "manual_override".into(),
            },
        ));
        let entries = log.entries_for_job(&jid("j1"));
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].action,
            AuditAction::Custom {
                action: "manual_override".into()
            }
        );
    }

    #[test]
    fn test_audit_log_checkpoint_action() {
        let log = AuditLog::new();
        log.log_system(jid("j1"), AuditAction::CheckpointSaved { step: 5 });
        let entries = log.entries_for_job(&jid("j1"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action.to_string(), "checkpoint_saved(step=5)");
    }
}
