//! Dead letter queue for permanently failed jobs.
//!
//! When a job exhausts all retries it is moved to the dead letter queue (DLQ)
//! instead of being silently discarded.  The DLQ supports configurable
//! retention policies, inspection, replay, and purging of old entries.

use std::collections::VecDeque;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};
use crate::types::JobId;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Reason a job was moved to the dead letter queue.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeadLetterReason {
    /// All retry attempts were exhausted.
    RetriesExhausted {
        /// Number of attempts made.
        attempts: u32,
    },
    /// The job exceeded its timeout limit.
    Timeout {
        /// The configured timeout in seconds.
        timeout_secs: u64,
    },
    /// A validation error prevented execution.
    ValidationFailed {
        /// Description of the validation failure.
        details: String,
    },
    /// The job was manually rejected by an operator.
    ManualReject {
        /// Who rejected the job.
        rejected_by: String,
    },
    /// A dependency of this job failed permanently.
    DependencyFailed {
        /// The ID of the failed dependency.
        dependency_id: String,
    },
    /// Catch-all for unexpected failures.
    Other {
        /// Freeform description.
        details: String,
    },
}

impl std::fmt::Display for DeadLetterReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetriesExhausted { attempts } => {
                write!(f, "Retries exhausted after {attempts} attempts")
            }
            Self::Timeout { timeout_secs } => {
                write!(f, "Timed out after {timeout_secs}s")
            }
            Self::ValidationFailed { details } => {
                write!(f, "Validation failed: {details}")
            }
            Self::ManualReject { rejected_by } => {
                write!(f, "Manually rejected by {rejected_by}")
            }
            Self::DependencyFailed { dependency_id } => {
                write!(f, "Dependency '{dependency_id}' failed")
            }
            Self::Other { details } => write!(f, "{details}"),
        }
    }
}

/// A single entry in the dead letter queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// Unique ID for this DLQ entry.
    pub entry_id: String,
    /// Original job ID.
    pub job_id: JobId,
    /// Original job name.
    pub job_name: String,
    /// Why the job ended up here.
    pub reason: DeadLetterReason,
    /// The last error message observed.
    pub last_error: String,
    /// Number of times the job was retried before ending up here.
    pub retry_count: u32,
    /// Unix timestamp (seconds since epoch) when the entry was created.
    pub created_at_secs: u64,
    /// Number of times this entry has been replayed (re-submitted).
    pub replay_count: u32,
    /// Arbitrary metadata carried over from the original job.
    pub metadata: std::collections::HashMap<String, String>,
}

impl DeadLetterEntry {
    /// Create a new dead letter entry.
    #[must_use]
    pub fn new(
        job_id: JobId,
        job_name: impl Into<String>,
        reason: DeadLetterReason,
        last_error: impl Into<String>,
        retry_count: u32,
    ) -> Self {
        Self {
            entry_id: uuid::Uuid::new_v4().to_string(),
            job_id,
            job_name: job_name.into(),
            reason,
            last_error: last_error.into(),
            retry_count,
            created_at_secs: current_timestamp(),
            replay_count: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata and return self (builder style).
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Age of this entry in seconds (returns 0 if the clock went backwards).
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        current_timestamp().saturating_sub(self.created_at_secs)
    }

    /// Mark that this entry was replayed.
    pub fn mark_replayed(&mut self) {
        self.replay_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Retention policy
// ---------------------------------------------------------------------------

/// How old entries are purged from the dead letter queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    /// Keep entries forever.
    Indefinite,
    /// Keep entries for at most `max_age_secs` seconds.
    MaxAge {
        /// Maximum age in seconds.
        max_age_secs: u64,
    },
    /// Keep at most `max_entries` entries; oldest are evicted first.
    MaxCount {
        /// Maximum number of entries.
        max_entries: usize,
    },
    /// Combined age + count limit; whichever triggers first.
    AgeAndCount {
        /// Maximum age in seconds.
        max_age_secs: u64,
        /// Maximum number of entries.
        max_entries: usize,
    },
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        // Default: keep for 7 days, max 10 000 entries.
        Self::AgeAndCount {
            max_age_secs: 7 * 24 * 3600,
            max_entries: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Dead letter queue
// ---------------------------------------------------------------------------

/// Thread-safe dead letter queue with configurable retention.
#[derive(Debug)]
pub struct DeadLetterQueue {
    entries: RwLock<VecDeque<DeadLetterEntry>>,
    retention: RetentionPolicy,
}

impl DeadLetterQueue {
    /// Create a new dead letter queue with the given retention policy.
    #[must_use]
    pub fn new(retention: RetentionPolicy) -> Self {
        Self {
            entries: RwLock::new(VecDeque::new()),
            retention,
        }
    }

    /// Create a DLQ with the default retention policy.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RetentionPolicy::default())
    }

    /// Push a failed job into the dead letter queue.
    ///
    /// Retention enforcement is applied automatically after insertion.
    pub fn push(&self, entry: DeadLetterEntry) {
        let mut entries = self.entries.write();
        entries.push_back(entry);
        drop(entries);
        self.enforce_retention();
    }

    /// Number of entries currently in the DLQ.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the DLQ is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Look up an entry by its DLQ entry ID.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if the entry does not exist.
    pub fn get(&self, entry_id: &str) -> Result<DeadLetterEntry> {
        self.entries
            .read()
            .iter()
            .find(|e| e.entry_id == entry_id)
            .cloned()
            .ok_or_else(|| BatchError::JobNotFound(format!("DLQ entry not found: {entry_id}")))
    }

    /// Look up entries for a specific job ID.
    #[must_use]
    pub fn find_by_job_id(&self, job_id: &JobId) -> Vec<DeadLetterEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| &e.job_id == job_id)
            .cloned()
            .collect()
    }

    /// List all entries, newest first.
    #[must_use]
    pub fn list(&self) -> Vec<DeadLetterEntry> {
        let entries = self.entries.read();
        entries.iter().rev().cloned().collect()
    }

    /// List entries matching a predicate, newest first.
    #[must_use]
    pub fn list_filtered<F>(&self, predicate: F) -> Vec<DeadLetterEntry>
    where
        F: Fn(&DeadLetterEntry) -> bool,
    {
        let entries = self.entries.read();
        entries
            .iter()
            .rev()
            .filter(|e| predicate(e))
            .cloned()
            .collect()
    }

    /// Remove and return an entry by its DLQ entry ID (for replay).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if the entry does not exist.
    pub fn take(&self, entry_id: &str) -> Result<DeadLetterEntry> {
        let mut entries = self.entries.write();
        let pos = entries
            .iter()
            .position(|e| e.entry_id == entry_id)
            .ok_or_else(|| BatchError::JobNotFound(format!("DLQ entry not found: {entry_id}")))?;
        // safe because we just confirmed the index is valid
        let mut entry = entries.remove(pos).ok_or_else(|| {
            BatchError::JobNotFound(format!("DLQ entry vanished unexpectedly: {entry_id}"))
        })?;
        entry.mark_replayed();
        Ok(entry)
    }

    /// Delete an entry permanently.
    ///
    /// Returns `true` if it was found and deleted.
    pub fn delete(&self, entry_id: &str) -> bool {
        let mut entries = self.entries.write();
        if let Some(pos) = entries.iter().position(|e| e.entry_id == entry_id) {
            entries.remove(pos);
            true
        } else {
            false
        }
    }

    /// Purge all entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Purge entries older than `max_age_secs`.
    ///
    /// Returns the number of entries purged.
    pub fn purge_older_than(&self, max_age_secs: u64) -> usize {
        let now = current_timestamp();
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|e| now.saturating_sub(e.created_at_secs) <= max_age_secs);
        before - entries.len()
    }

    /// Summary statistics about the dead letter queue.
    #[must_use]
    pub fn stats(&self) -> DeadLetterStats {
        let entries = self.entries.read();
        let total = entries.len();
        let mut by_reason = std::collections::HashMap::new();
        let mut total_replays = 0u64;
        let mut oldest_secs = 0u64;
        let now = current_timestamp();

        for e in entries.iter() {
            let key = match &e.reason {
                DeadLetterReason::RetriesExhausted { .. } => "retries_exhausted",
                DeadLetterReason::Timeout { .. } => "timeout",
                DeadLetterReason::ValidationFailed { .. } => "validation_failed",
                DeadLetterReason::ManualReject { .. } => "manual_reject",
                DeadLetterReason::DependencyFailed { .. } => "dependency_failed",
                DeadLetterReason::Other { .. } => "other",
            };
            *by_reason.entry(key.to_string()).or_insert(0usize) += 1;
            total_replays += u64::from(e.replay_count);
            let age = now.saturating_sub(e.created_at_secs);
            if age > oldest_secs {
                oldest_secs = age;
            }
        }

        DeadLetterStats {
            total_entries: total,
            by_reason,
            total_replays,
            oldest_entry_age_secs: oldest_secs,
        }
    }

    // -----------------------------------------------------------------------
    // Retention enforcement
    // -----------------------------------------------------------------------

    fn enforce_retention(&self) {
        match self.retention {
            RetentionPolicy::Indefinite => {}
            RetentionPolicy::MaxAge { max_age_secs } => {
                self.purge_older_than(max_age_secs);
            }
            RetentionPolicy::MaxCount { max_entries } => {
                self.enforce_max_count(max_entries);
            }
            RetentionPolicy::AgeAndCount {
                max_age_secs,
                max_entries,
            } => {
                self.purge_older_than(max_age_secs);
                self.enforce_max_count(max_entries);
            }
        }
    }

    fn enforce_max_count(&self, max_entries: usize) {
        let mut entries = self.entries.write();
        while entries.len() > max_entries {
            entries.pop_front();
        }
    }
}

impl Default for DeadLetterQueue {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Summary statistics for the DLQ.
#[derive(Debug, Clone)]
pub struct DeadLetterStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Count of entries grouped by reason category.
    pub by_reason: std::collections::HashMap<String, usize>,
    /// Total number of replay attempts across all entries.
    pub total_replays: u64,
    /// Age of the oldest entry in seconds.
    pub oldest_entry_age_secs: u64,
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

    fn make_entry(job_name: &str, reason: DeadLetterReason) -> DeadLetterEntry {
        DeadLetterEntry::new(
            JobId::from(format!("job-{job_name}")),
            job_name,
            reason,
            "something went wrong",
            3,
        )
    }

    #[test]
    fn test_dlq_push_and_len() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        assert!(dlq.is_empty());
        dlq.push(make_entry(
            "a",
            DeadLetterReason::RetriesExhausted { attempts: 3 },
        ));
        assert_eq!(dlq.len(), 1);
        assert!(!dlq.is_empty());
    }

    #[test]
    fn test_dlq_get_by_entry_id() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        let entry = make_entry("b", DeadLetterReason::Timeout { timeout_secs: 60 });
        let eid = entry.entry_id.clone();
        dlq.push(entry);
        let loaded = dlq.get(&eid).expect("should find entry");
        assert_eq!(loaded.job_name, "b");
    }

    #[test]
    fn test_dlq_get_missing_returns_error() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        assert!(dlq.get("no-such-entry").is_err());
    }

    #[test]
    fn test_dlq_find_by_job_id() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        let jid = JobId::from("j1");
        let e1 = DeadLetterEntry::new(
            jid.clone(),
            "first",
            DeadLetterReason::RetriesExhausted { attempts: 1 },
            "err",
            1,
        );
        let e2 = DeadLetterEntry::new(
            jid.clone(),
            "second",
            DeadLetterReason::RetriesExhausted { attempts: 2 },
            "err",
            2,
        );
        dlq.push(e1);
        dlq.push(e2);
        let found = dlq.find_by_job_id(&jid);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_dlq_list_newest_first() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        dlq.push(make_entry(
            "first",
            DeadLetterReason::Other {
                details: "a".into(),
            },
        ));
        dlq.push(make_entry(
            "second",
            DeadLetterReason::Other {
                details: "b".into(),
            },
        ));
        let list = dlq.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].job_name, "second");
        assert_eq!(list[1].job_name, "first");
    }

    #[test]
    fn test_dlq_list_filtered() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        dlq.push(make_entry(
            "a",
            DeadLetterReason::Timeout { timeout_secs: 30 },
        ));
        dlq.push(make_entry(
            "b",
            DeadLetterReason::RetriesExhausted { attempts: 5 },
        ));
        dlq.push(make_entry(
            "c",
            DeadLetterReason::Timeout { timeout_secs: 60 },
        ));
        let timeouts = dlq.list_filtered(|e| matches!(&e.reason, DeadLetterReason::Timeout { .. }));
        assert_eq!(timeouts.len(), 2);
    }

    #[test]
    fn test_dlq_take_removes_and_marks_replayed() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        let entry = make_entry("r", DeadLetterReason::RetriesExhausted { attempts: 2 });
        let eid = entry.entry_id.clone();
        dlq.push(entry);
        let taken = dlq.take(&eid).expect("should take entry");
        assert_eq!(taken.replay_count, 1);
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_take_missing_returns_error() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        assert!(dlq.take("ghost").is_err());
    }

    #[test]
    fn test_dlq_delete() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        let entry = make_entry(
            "d",
            DeadLetterReason::Other {
                details: "x".into(),
            },
        );
        let eid = entry.entry_id.clone();
        dlq.push(entry);
        assert!(dlq.delete(&eid));
        assert!(dlq.is_empty());
        assert!(!dlq.delete(&eid));
    }

    #[test]
    fn test_dlq_clear() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        for i in 0..5 {
            dlq.push(make_entry(
                &i.to_string(),
                DeadLetterReason::RetriesExhausted { attempts: 1 },
            ));
        }
        assert_eq!(dlq.len(), 5);
        dlq.clear();
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_max_count_retention() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::MaxCount { max_entries: 3 });
        for i in 0..5 {
            dlq.push(make_entry(
                &i.to_string(),
                DeadLetterReason::RetriesExhausted { attempts: 1 },
            ));
        }
        // Only 3 should remain (the newest).
        assert_eq!(dlq.len(), 3);
        let list = dlq.list();
        // Newest first: 4, 3, 2
        assert_eq!(list[0].job_name, "4");
        assert_eq!(list[1].job_name, "3");
        assert_eq!(list[2].job_name, "2");
    }

    #[test]
    fn test_dlq_stats() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        dlq.push(make_entry(
            "a",
            DeadLetterReason::RetriesExhausted { attempts: 3 },
        ));
        dlq.push(make_entry(
            "b",
            DeadLetterReason::Timeout { timeout_secs: 60 },
        ));
        dlq.push(make_entry(
            "c",
            DeadLetterReason::RetriesExhausted { attempts: 5 },
        ));
        let stats = dlq.stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(
            stats
                .by_reason
                .get("retries_exhausted")
                .copied()
                .unwrap_or(0),
            2
        );
        assert_eq!(stats.by_reason.get("timeout").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_dead_letter_entry_with_meta() {
        let entry = make_entry(
            "m",
            DeadLetterReason::Other {
                details: "x".into(),
            },
        )
        .with_meta("project", "test-project")
        .with_meta("user", "admin");
        assert_eq!(
            entry.metadata.get("project").map(|s| s.as_str()),
            Some("test-project")
        );
        assert_eq!(
            entry.metadata.get("user").map(|s| s.as_str()),
            Some("admin")
        );
    }

    #[test]
    fn test_dead_letter_reason_display() {
        let r = DeadLetterReason::RetriesExhausted { attempts: 5 };
        assert_eq!(r.to_string(), "Retries exhausted after 5 attempts");
        let r2 = DeadLetterReason::Timeout { timeout_secs: 120 };
        assert_eq!(r2.to_string(), "Timed out after 120s");
        let r3 = DeadLetterReason::ValidationFailed {
            details: "bad input".into(),
        };
        assert_eq!(r3.to_string(), "Validation failed: bad input");
    }

    #[test]
    fn test_dead_letter_entry_age() {
        let entry = make_entry(
            "age",
            DeadLetterReason::Other {
                details: "x".into(),
            },
        );
        // Just created, age should be very small.
        assert!(entry.age_secs() < 5);
    }

    #[test]
    fn test_retention_policy_default() {
        let policy = RetentionPolicy::default();
        match policy {
            RetentionPolicy::AgeAndCount {
                max_age_secs,
                max_entries,
            } => {
                assert_eq!(max_age_secs, 7 * 24 * 3600);
                assert_eq!(max_entries, 10_000);
            }
            _ => panic!("Expected AgeAndCount default"),
        }
    }

    #[test]
    fn test_dlq_default() {
        let dlq = DeadLetterQueue::default();
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_purge_older_than() {
        let dlq = DeadLetterQueue::new(RetentionPolicy::Indefinite);
        // Push an entry with an old timestamp.
        let mut old_entry = make_entry(
            "old",
            DeadLetterReason::Other {
                details: "x".into(),
            },
        );
        old_entry.created_at_secs = current_timestamp().saturating_sub(1000);
        dlq.push(old_entry);
        dlq.push(make_entry(
            "new",
            DeadLetterReason::Other {
                details: "y".into(),
            },
        ));

        let purged = dlq.purge_older_than(500);
        assert_eq!(purged, 1);
        assert_eq!(dlq.len(), 1);
        let remaining = dlq.list();
        assert_eq!(remaining[0].job_name, "new");
    }

    #[test]
    fn test_dead_letter_entry_mark_replayed() {
        let mut entry = make_entry(
            "rp",
            DeadLetterReason::Other {
                details: "x".into(),
            },
        );
        assert_eq!(entry.replay_count, 0);
        entry.mark_replayed();
        assert_eq!(entry.replay_count, 1);
        entry.mark_replayed();
        assert_eq!(entry.replay_count, 2);
    }
}
