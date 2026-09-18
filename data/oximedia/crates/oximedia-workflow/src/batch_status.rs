//! Batch task status update manager.
//!
//! Collects task status changes in memory and flushes them to a persistence
//! backend in batches, reducing the number of database write operations
//! compared to individual per-task updates. This is especially important for
//! high-throughput workflows with hundreds of short-lived tasks.
//!
//! The [`BatchStatusWriter`] accumulates [`StatusUpdate`] records and
//! exposes a [`BatchStatusWriter::flush`] method that emits all pending
//! updates at once. A configurable `max_batch_size` prevents unbounded
//! memory growth during long bursts.

use crate::task::TaskState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// StatusUpdate
// ---------------------------------------------------------------------------

/// A single pending task-status change to be persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    /// Workflow this task belongs to.
    pub workflow_id: String,
    /// Task identifier.
    pub task_id: String,
    /// New task state.
    pub state: TaskState,
    /// Optional error message (for failed tasks).
    pub error: Option<String>,
    /// Unix timestamp (seconds) when the change occurred.
    pub timestamp_secs: u64,
    /// Sequence number for ordering within a flush batch.
    pub seq: u64,
}

impl StatusUpdate {
    /// Create a new status update at the current wall-clock time.
    #[must_use]
    pub fn now(
        workflow_id: impl Into<String>,
        task_id: impl Into<String>,
        state: TaskState,
        error: Option<String>,
        seq: u64,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            workflow_id: workflow_id.into(),
            task_id: task_id.into(),
            state,
            error,
            timestamp_secs: ts,
            seq,
        }
    }
}

// ---------------------------------------------------------------------------
// FlushResult
// ---------------------------------------------------------------------------

/// Statistics returned by [`BatchStatusWriter::flush`].
#[derive(Debug, Clone, Default)]
pub struct FlushResult {
    /// Number of status updates emitted.
    pub updates_flushed: usize,
    /// Number of distinct workflows affected.
    pub workflows_touched: usize,
    /// Approximate byte size of the flushed batch (for logging).
    pub estimated_bytes: usize,
}

// ---------------------------------------------------------------------------
// BatchStatusWriter
// ---------------------------------------------------------------------------

/// Accumulates task status changes and flushes them in bulk.
///
/// Updates are keyed by `(workflow_id, task_id)`. If a task transitions
/// multiple times before a flush (e.g., `Pending → Running → Completed`),
/// only the most-recent state is retained — this de-duplicates writes for
/// tasks that complete faster than the flush interval.
#[derive(Debug)]
pub struct BatchStatusWriter {
    /// Pending updates: `(workflow_id, task_id) → StatusUpdate`.
    pending: HashMap<(String, String), StatusUpdate>,
    /// Monotonic sequence counter.
    seq: u64,
    /// Maximum number of updates to hold before forcing a flush.
    max_batch_size: usize,
    /// Total number of updates ever accepted (including de-duplicated ones).
    total_accepted: u64,
    /// Total number of flush operations performed.
    total_flushes: u64,
}

impl BatchStatusWriter {
    /// Create a new writer with the given `max_batch_size`.
    #[must_use]
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: HashMap::new(),
            seq: 0,
            max_batch_size,
            total_accepted: 0,
            total_flushes: 0,
        }
    }

    /// Record a task status change.
    ///
    /// If the batch is full after accepting this update, the caller should
    /// call [`Self::flush`] before continuing (check [`Self::needs_flush`]).
    pub fn record(
        &mut self,
        workflow_id: impl Into<String>,
        task_id: impl Into<String>,
        state: TaskState,
        error: Option<String>,
    ) {
        let wf = workflow_id.into();
        let tid = task_id.into();
        let seq = self.seq;
        self.seq += 1;
        self.total_accepted += 1;

        // Overwrite any existing pending update — keep only latest state.
        let update = StatusUpdate::now(wf.clone(), tid.clone(), state, error, seq);
        self.pending.insert((wf, tid), update);
    }

    /// Return `true` if the batch should be flushed (batch is full or overflowing).
    #[must_use]
    pub fn needs_flush(&self) -> bool {
        self.pending.len() >= self.max_batch_size
    }

    /// Return the number of pending updates.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Flush all pending updates.
    ///
    /// The provided `sink` callback receives the sorted batch of updates.
    /// After calling `sink` the internal buffer is cleared regardless of
    /// whether `sink` succeeds.
    ///
    /// # Errors
    ///
    /// Returns any error propagated from `sink`.
    pub fn flush<E, F>(&mut self, sink: F) -> Result<FlushResult, E>
    where
        F: FnOnce(Vec<StatusUpdate>) -> Result<(), E>,
    {
        if self.pending.is_empty() {
            return Ok(FlushResult::default());
        }

        // Collect, sort by sequence number for deterministic ordering.
        let mut batch: Vec<StatusUpdate> = self.pending.drain().map(|(_, v)| v).collect();
        batch.sort_by_key(|u| u.seq);

        let updates_flushed = batch.len();
        let workflows_touched = batch
            .iter()
            .map(|u| u.workflow_id.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let estimated_bytes = batch
            .iter()
            .map(|u| u.workflow_id.len() + u.task_id.len() + 32)
            .sum();

        sink(batch)?;
        self.total_flushes += 1;

        Ok(FlushResult {
            updates_flushed,
            workflows_touched,
            estimated_bytes,
        })
    }

    /// Flush into a `Vec<StatusUpdate>` and return it (useful for testing).
    pub fn flush_to_vec(&mut self) -> Vec<StatusUpdate> {
        let mut result = Vec::new();
        // Use infallible sink
        let _ = self.flush::<std::convert::Infallible, _>(|batch| {
            result = batch;
            Ok(())
        });
        result
    }

    /// Total number of updates ever accepted.
    #[must_use]
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Total number of flush operations.
    #[must_use]
    pub fn total_flushes(&self) -> u64 {
        self.total_flushes
    }

    /// Change the maximum batch size.
    pub fn set_max_batch_size(&mut self, max: usize) {
        self.max_batch_size = max;
    }

    /// Maximum batch size.
    #[must_use]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_accepts_updates() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "task-a", TaskState::Running, None);
        w.record("wf-1", "task-b", TaskState::Running, None);
        assert_eq!(w.pending_count(), 2);
        assert_eq!(w.total_accepted(), 2);
    }

    #[test]
    fn test_writer_deduplicates_same_task() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "task-a", TaskState::Pending, None);
        w.record("wf-1", "task-a", TaskState::Running, None);
        w.record("wf-1", "task-a", TaskState::Completed, None);
        // Three transitions, but only 1 unique key
        assert_eq!(w.pending_count(), 1);
        assert_eq!(w.total_accepted(), 3);

        let batch = w.flush_to_vec();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].state, TaskState::Completed);
    }

    #[test]
    fn test_writer_needs_flush_at_capacity() {
        let mut w = BatchStatusWriter::new(3);
        assert!(!w.needs_flush());
        w.record("wf-1", "a", TaskState::Running, None);
        w.record("wf-1", "b", TaskState::Running, None);
        assert!(!w.needs_flush());
        w.record("wf-1", "c", TaskState::Running, None);
        assert!(w.needs_flush());
    }

    #[test]
    fn test_flush_clears_pending() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "a", TaskState::Completed, None);
        w.record("wf-1", "b", TaskState::Failed, Some("err".to_string()));

        let batch = w.flush_to_vec();
        assert_eq!(batch.len(), 2);
        assert_eq!(w.pending_count(), 0);
        assert_eq!(w.total_flushes(), 1);
    }

    #[test]
    fn test_flush_empty_is_noop() {
        let mut w = BatchStatusWriter::new(100);
        let batch = w.flush_to_vec();
        assert!(batch.is_empty());
        assert_eq!(w.total_flushes(), 0);
    }

    #[test]
    fn test_flush_sorts_by_sequence() {
        let mut w = BatchStatusWriter::new(100);
        // Records with different task IDs (unique keys, so all retained)
        w.record("wf-1", "z-task", TaskState::Completed, None);
        w.record("wf-1", "a-task", TaskState::Running, None);
        w.record("wf-1", "m-task", TaskState::Pending, None);

        let batch = w.flush_to_vec();
        // Should be sorted by sequence (insertion order): z=0, a=1, m=2
        assert_eq!(batch[0].task_id, "z-task");
        assert_eq!(batch[1].task_id, "a-task");
        assert_eq!(batch[2].task_id, "m-task");
    }

    #[test]
    fn test_flush_workflows_touched() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "task-a", TaskState::Completed, None);
        w.record("wf-2", "task-b", TaskState::Completed, None);
        w.record("wf-2", "task-c", TaskState::Failed, None);

        let mut result_ref = FlushResult::default();
        let _ = w.flush::<std::convert::Infallible, _>(|batch| {
            result_ref = FlushResult {
                updates_flushed: batch.len(),
                workflows_touched: batch
                    .iter()
                    .map(|u| u.workflow_id.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                estimated_bytes: 0,
            };
            Ok(())
        });
        assert_eq!(result_ref.updates_flushed, 3);
        assert_eq!(result_ref.workflows_touched, 2);
    }

    #[test]
    fn test_multiple_flush_cycles() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "t1", TaskState::Completed, None);
        w.flush_to_vec();
        w.record("wf-1", "t2", TaskState::Completed, None);
        w.flush_to_vec();

        assert_eq!(w.total_flushes(), 2);
        assert_eq!(w.total_accepted(), 2);
    }

    #[test]
    fn test_status_update_fields() {
        let u = StatusUpdate::now("wf-x", "task-y", TaskState::Running, None, 42);
        assert_eq!(u.workflow_id, "wf-x");
        assert_eq!(u.task_id, "task-y");
        assert_eq!(u.state, TaskState::Running);
        assert_eq!(u.seq, 42);
        assert!(u.timestamp_secs > 0);
    }

    #[test]
    fn test_set_max_batch_size() {
        let mut w = BatchStatusWriter::new(10);
        assert_eq!(w.max_batch_size(), 10);
        w.set_max_batch_size(50);
        assert_eq!(w.max_batch_size(), 50);
    }

    #[test]
    fn test_error_propagation() {
        let mut w = BatchStatusWriter::new(100);
        w.record("wf-1", "t1", TaskState::Failed, Some("boom".to_string()));

        let result = w.flush::<String, _>(|_| Err("db unavailable".to_string()));
        assert!(result.is_err());
        // After a failed flush, pending should be drained (we drained before calling sink)
        assert_eq!(w.pending_count(), 0);
    }
}
