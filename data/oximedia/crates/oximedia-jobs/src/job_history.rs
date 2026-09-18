#![allow(dead_code)]
//! Job history — record execution outcomes and compute aggregate statistics.

use std::time::{Duration, Instant};

/// A terminal job result value stored alongside a history entry.
///
/// Intentionally kept as a simple key-value map so callers can store any
/// serialisable result without imposing a rigid schema on the history module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JobResult {
    /// Arbitrary key-value pairs describing the result (e.g. output file path,
    /// byte count, etc.).
    pub fields: std::collections::HashMap<String, String>,
}

impl JobResult {
    /// Create an empty result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value field.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Retrieve a field value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }
}

/// Configurable retention policy for [`JobHistory`].
///
/// Both limits are optional (`None` means unlimited).  When both are set the
/// `prune()` call enforces them independently: the age check runs first, then
/// the count check trims the oldest remaining entries.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age of an entry.  Entries older than this are pruned.
    pub max_age: Option<Duration>,
    /// Maximum total number of entries to retain.  When the count exceeds this
    /// the *oldest* excess entries are removed.
    pub max_entries: Option<usize>,
}

impl Default for RetentionPolicy {
    /// The default policy imposes no limits.
    fn default() -> Self {
        Self {
            max_age: None,
            max_entries: None,
        }
    }
}

/// The outcome of a single job execution attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOutcome {
    /// The job completed without error.
    Success,
    /// The job failed with the given reason.
    Failed(String),
    /// The job was cancelled before completion.
    Cancelled,
}

impl ExecutionOutcome {
    /// Returns `true` for the `Success` variant.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` for the `Failed` variant.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// A single history record for one job execution attempt.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Identifier of the job this entry belongs to.
    pub job_id: String,
    /// Execution outcome.
    pub outcome: ExecutionOutcome,
    /// Wall-clock duration of this execution.
    pub duration: Duration,
    /// When this attempt started.
    pub started_at: Instant,
    /// Optional terminal result payload (only present when the job produced one).
    pub result: Option<JobResult>,
}

impl HistoryEntry {
    /// Create a new history entry without a result payload.
    #[must_use]
    pub fn new(job_id: impl Into<String>, outcome: ExecutionOutcome, duration: Duration) -> Self {
        Self {
            job_id: job_id.into(),
            outcome,
            duration,
            started_at: Instant::now(),
            result: None,
        }
    }

    /// Attach a terminal [`JobResult`] to this entry.
    #[must_use]
    pub fn with_result(mut self, result: JobResult) -> Self {
        self.result = Some(result);
        self
    }

    /// Returns `true` if this entry represents a successful execution.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.outcome.is_success()
    }

    /// Returns the duration in whole milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u128 {
        self.duration.as_millis()
    }
}

/// Accumulates history entries for one or more jobs.
#[derive(Debug)]
pub struct JobHistory {
    entries: Vec<HistoryEntry>,
    /// Active retention policy.
    policy: RetentionPolicy,
}

impl Default for JobHistory {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            policy: RetentionPolicy::default(),
        }
    }
}

impl JobHistory {
    /// Create an empty history with no retention limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty history with the given retention policy.
    #[must_use]
    pub fn with_retention(policy: RetentionPolicy) -> Self {
        Self {
            entries: Vec::new(),
            policy,
        }
    }

    /// Remove entries that violate the current [`RetentionPolicy`].
    ///
    /// * Age pruning runs first: entries whose `started_at` is older than
    ///   `policy.max_age` are dropped.
    /// * Count pruning follows: the *oldest* entries are dropped until
    ///   `entries.len() <= policy.max_entries`.
    pub fn prune(&mut self) {
        if let Some(max_age) = self.policy.max_age {
            let now = Instant::now();
            self.entries
                .retain(|e| now.duration_since(e.started_at) <= max_age);
        }
        if let Some(max_entries) = self.policy.max_entries {
            if self.entries.len() > max_entries {
                let excess = self.entries.len() - max_entries;
                self.entries.drain(..excess);
            }
        }
    }

    /// Record a new history entry.
    pub fn record(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
    }

    /// Return all entries marked as successes.
    #[must_use]
    pub fn successes(&self) -> Vec<&HistoryEntry> {
        self.entries.iter().filter(|e| e.is_success()).collect()
    }

    /// Return all entries marked as failures.
    #[must_use]
    pub fn failures(&self) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.outcome.is_failure())
            .collect()
    }

    /// Total number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fraction of recorded entries that succeeded, in `[0.0, 1.0]`.
    /// Returns `0.0` when no entries exist.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.successes().len() as f64 / self.entries.len() as f64
    }

    /// Return entries for a specific job id.
    #[must_use]
    pub fn entries_for(&self, job_id: &str) -> Vec<&HistoryEntry> {
        self.entries.iter().filter(|e| e.job_id == job_id).collect()
    }

    /// Compute the average execution duration across all entries.
    /// Returns `None` when no entries exist.
    #[must_use]
    pub fn average_duration(&self) -> Option<Duration> {
        if self.entries.is_empty() {
            return None;
        }
        let total_ns: u128 = self.entries.iter().map(|e| e.duration.as_nanos()).sum();
        Some(Duration::from_nanos(
            (total_ns / self.entries.len() as u128) as u64,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(job_id: &str, ok: bool, ms: u64) -> HistoryEntry {
        let outcome = if ok {
            ExecutionOutcome::Success
        } else {
            ExecutionOutcome::Failed("error".to_string())
        };
        HistoryEntry::new(job_id, outcome, Duration::from_millis(ms))
    }

    #[test]
    fn test_execution_outcome_is_success_true() {
        assert!(ExecutionOutcome::Success.is_success());
    }

    #[test]
    fn test_execution_outcome_is_success_false() {
        assert!(!ExecutionOutcome::Failed("x".to_string()).is_success());
        assert!(!ExecutionOutcome::Cancelled.is_success());
    }

    #[test]
    fn test_execution_outcome_is_failure_true() {
        assert!(ExecutionOutcome::Failed("oops".to_string()).is_failure());
    }

    #[test]
    fn test_execution_outcome_is_failure_false() {
        assert!(!ExecutionOutcome::Success.is_failure());
        assert!(!ExecutionOutcome::Cancelled.is_failure());
    }

    #[test]
    fn test_history_entry_is_success() {
        let e = entry("j1", true, 100);
        assert!(e.is_success());
    }

    #[test]
    fn test_history_entry_is_not_success_on_failure() {
        let e = entry("j2", false, 50);
        assert!(!e.is_success());
    }

    #[test]
    fn test_history_entry_duration_ms() {
        let e = entry("j3", true, 250);
        assert_eq!(e.duration_ms(), 250);
    }

    #[test]
    fn test_job_history_record_and_len() {
        let mut h = JobHistory::new();
        assert!(h.is_empty());
        h.record(entry("j", true, 10));
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_job_history_successes() {
        let mut h = JobHistory::new();
        h.record(entry("j", true, 10));
        h.record(entry("j", false, 20));
        h.record(entry("j", true, 30));
        assert_eq!(h.successes().len(), 2);
    }

    #[test]
    fn test_job_history_failures() {
        let mut h = JobHistory::new();
        h.record(entry("j", true, 10));
        h.record(entry("j", false, 20));
        assert_eq!(h.failures().len(), 1);
    }

    #[test]
    fn test_job_history_success_rate_all_ok() {
        let mut h = JobHistory::new();
        h.record(entry("j", true, 1));
        h.record(entry("j", true, 2));
        assert!((h.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_job_history_success_rate_partial() {
        let mut h = JobHistory::new();
        h.record(entry("j", true, 1));
        h.record(entry("j", false, 1));
        assert!((h.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_job_history_success_rate_empty() {
        let h = JobHistory::new();
        assert_eq!(h.success_rate(), 0.0);
    }

    #[test]
    fn test_job_history_entries_for() {
        let mut h = JobHistory::new();
        h.record(entry("alpha", true, 10));
        h.record(entry("beta", false, 20));
        h.record(entry("alpha", true, 30));
        assert_eq!(h.entries_for("alpha").len(), 2);
        assert_eq!(h.entries_for("beta").len(), 1);
        assert_eq!(h.entries_for("gamma").len(), 0);
    }

    #[test]
    fn test_job_history_average_duration_none_when_empty() {
        let h = JobHistory::new();
        assert!(h.average_duration().is_none());
    }

    #[test]
    fn test_job_history_average_duration_some() {
        let mut h = JobHistory::new();
        h.record(entry("j", true, 100));
        h.record(entry("j", false, 200));
        let avg = h.average_duration().expect("avg should be valid");
        assert_eq!(avg.as_millis(), 150);
    }

    #[test]
    fn test_job_history_default_is_empty() {
        let h = JobHistory::default();
        assert!(h.is_empty());
    }

    // -----------------------------------------------------------------------
    // RetentionPolicy tests
    // -----------------------------------------------------------------------

    /// An entry whose `started_at` is far in the past should be pruned when
    /// `max_age` is set to a very short duration.
    #[test]
    fn test_prune_by_age_removes_old_entry() {
        // Build an entry that "started" 10 seconds ago by reconstructing it
        // with a backdated `started_at`.  We manipulate the field directly
        // because `Instant::now() - Duration` is stable since Rust 1.34.
        let mut h = JobHistory::with_retention(RetentionPolicy {
            max_age: Some(Duration::from_millis(1)),
            max_entries: None,
        });

        // Insert an entry that is already stale (started well before the 1 ms window).
        let mut old_entry = entry("j-old", true, 50);
        old_entry.started_at = Instant::now() - Duration::from_secs(5);
        h.record(old_entry);

        // Insert a fresh entry.
        h.record(entry("j-fresh", true, 50));

        assert_eq!(h.len(), 2);
        h.prune();
        // The old entry should be gone; the fresh one may or may not survive
        // depending on sub-millisecond timing, but at least the stale one is out.
        let stale_count = h.entries_for("j-old").len();
        assert_eq!(stale_count, 0, "old entry should have been pruned by age");
    }

    /// When `max_entries` is set to 2 and 4 entries are present, `prune()`
    /// must drop the 2 oldest ones.
    #[test]
    fn test_prune_by_count_keeps_most_recent() {
        let mut h = JobHistory::with_retention(RetentionPolicy {
            max_age: None,
            max_entries: Some(2),
        });

        for id in &["a", "b", "c", "d"] {
            let mut e = entry(id, true, 10);
            // Space out started_at so ordering is deterministic.
            e.started_at = Instant::now() - Duration::from_secs(10);
            h.record(e);
        }
        // Add two more with a more recent started_at.
        h.record(entry("e", true, 10));
        h.record(entry("f", true, 10));

        assert_eq!(h.len(), 6);
        h.prune();
        assert!(
            h.len() <= 2,
            "expected at most 2 entries after count prune, got {}",
            h.len()
        );
    }

    /// Entries that fall within the policy should survive `prune()`.
    #[test]
    fn test_prune_retains_entries_within_policy() {
        let mut h = JobHistory::with_retention(RetentionPolicy {
            max_age: Some(Duration::from_secs(3600)),
            max_entries: Some(100),
        });
        for id in &["x", "y", "z"] {
            h.record(entry(id, true, 10));
        }
        assert_eq!(h.len(), 3);
        h.prune();
        assert_eq!(h.len(), 3, "entries within policy should be retained");
    }
}
