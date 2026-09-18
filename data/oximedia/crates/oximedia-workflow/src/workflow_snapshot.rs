#![allow(dead_code)]
//! Workflow state snapshotting for rollback, replay, and audit.
//!
//! Captures the full state of a workflow at a given point in time so that
//! execution can be replayed, rolled back to a previous state, or audited.
//! Snapshots are immutable once created and carry a monotonic sequence number.

use std::collections::HashMap;

/// Monotonically increasing snapshot sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotSeq(u64);

impl SnapshotSeq {
    /// Create a snapshot sequence number from a raw value.
    #[must_use]
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the underlying value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SnapshotSeq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq:{}", self.0)
    }
}

/// The state of an individual task captured in the snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturedTaskState {
    /// Task has not started.
    Pending,
    /// Task is currently running.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed with a reason.
    Failed(String),
    /// Task was skipped.
    Skipped,
}

impl std::fmt::Display for CapturedTaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed(reason) => write!(f, "failed: {reason}"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// A snapshot of a single task.
#[derive(Debug, Clone)]
pub struct TaskSnapshot {
    /// Task identifier.
    pub task_id: String,
    /// State at snapshot time.
    pub state: CapturedTaskState,
    /// Number of retries that have occurred.
    pub retry_count: u32,
    /// Optional output data produced by the task.
    pub output: Option<String>,
}

impl TaskSnapshot {
    /// Create a new task snapshot.
    pub fn new(task_id: impl Into<String>, state: CapturedTaskState) -> Self {
        Self {
            task_id: task_id.into(),
            state,
            retry_count: 0,
            output: None,
        }
    }

    /// Set retry count.
    #[must_use]
    pub fn with_retries(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    /// Set output data.
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = Some(output.into());
        self
    }

    /// Return true if the task has finished (completed, failed, or skipped).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            CapturedTaskState::Completed
                | CapturedTaskState::Failed(_)
                | CapturedTaskState::Skipped
        )
    }
}

/// An immutable snapshot of the entire workflow at a point in time.
#[derive(Debug, Clone)]
pub struct WorkflowSnapshot {
    /// Sequence number of this snapshot.
    pub seq: SnapshotSeq,
    /// Workflow identifier.
    pub workflow_id: String,
    /// Wall-clock time the snapshot was taken (seconds since epoch).
    pub timestamp_secs: u64,
    /// Label describing why this snapshot was taken.
    pub label: String,
    /// Per-task state snapshots.
    pub tasks: Vec<TaskSnapshot>,
    /// Arbitrary metadata attached to the snapshot.
    pub metadata: HashMap<String, String>,
}

impl WorkflowSnapshot {
    /// Create a new workflow snapshot.
    pub fn new(
        seq: SnapshotSeq,
        workflow_id: impl Into<String>,
        timestamp_secs: u64,
        label: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            workflow_id: workflow_id.into(),
            timestamp_secs,
            label: label.into(),
            tasks: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a task snapshot.
    pub fn add_task(&mut self, task: TaskSnapshot) {
        self.tasks.push(task);
    }

    /// Attach metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Return the number of tasks in this snapshot.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Count tasks in a given state.
    #[must_use]
    pub fn count_in_state(&self, state: &CapturedTaskState) -> usize {
        self.tasks.iter().filter(|t| &t.state == state).count()
    }

    /// Return the completion ratio (0.0..=1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn completion_ratio(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        let terminal = self.tasks.iter().filter(|t| t.is_terminal()).count();
        terminal as f64 / self.tasks.len() as f64
    }

    /// Find a task snapshot by id.
    #[must_use]
    pub fn find_task(&self, task_id: &str) -> Option<&TaskSnapshot> {
        self.tasks.iter().find(|t| t.task_id == task_id)
    }
}

/// Manages a series of snapshots for a workflow.
#[derive(Debug)]
pub struct SnapshotStore {
    /// Workflow identifier this store belongs to.
    workflow_id: String,
    /// Ordered list of snapshots.
    snapshots: Vec<WorkflowSnapshot>,
    /// Counter for assigning sequence numbers.
    next_seq: u64,
    /// Maximum number of snapshots to keep (0 = unlimited).
    max_snapshots: usize,
}

impl SnapshotStore {
    /// Create a new snapshot store for the given workflow.
    pub fn new(workflow_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            snapshots: Vec::new(),
            next_seq: 1,
            max_snapshots: 0,
        }
    }

    /// Set a maximum retention limit for snapshots.
    #[must_use]
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }

    /// Take a new snapshot and add it to the store.
    pub fn take_snapshot(
        &mut self,
        timestamp_secs: u64,
        label: impl Into<String>,
        tasks: Vec<TaskSnapshot>,
    ) -> SnapshotSeq {
        let seq = SnapshotSeq::new(self.next_seq);
        self.next_seq += 1;

        let mut snapshot =
            WorkflowSnapshot::new(seq, self.workflow_id.clone(), timestamp_secs, label);
        for t in tasks {
            snapshot.add_task(t);
        }

        self.snapshots.push(snapshot);

        // Enforce retention
        if self.max_snapshots > 0 && self.snapshots.len() > self.max_snapshots {
            let to_remove = self.snapshots.len() - self.max_snapshots;
            self.snapshots.drain(..to_remove);
        }

        seq
    }

    /// Return the number of stored snapshots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if the store has no snapshots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get the latest snapshot.
    #[must_use]
    pub fn latest(&self) -> Option<&WorkflowSnapshot> {
        self.snapshots.last()
    }

    /// Get a snapshot by sequence number.
    #[must_use]
    pub fn get(&self, seq: SnapshotSeq) -> Option<&WorkflowSnapshot> {
        self.snapshots.iter().find(|s| s.seq == seq)
    }

    /// List all snapshot sequence numbers and labels.
    #[must_use]
    pub fn list(&self) -> Vec<(SnapshotSeq, &str)> {
        self.snapshots
            .iter()
            .map(|s| (s.seq, s.label.as_str()))
            .collect()
    }

    /// Compare two snapshots and return task ids whose state changed.
    #[must_use]
    pub fn diff(&self, a: SnapshotSeq, b: SnapshotSeq) -> Option<Vec<String>> {
        let snap_a = self.get(a)?;
        let snap_b = self.get(b)?;

        let map_a: HashMap<&str, &CapturedTaskState> = snap_a
            .tasks
            .iter()
            .map(|t| (t.task_id.as_str(), &t.state))
            .collect();

        let mut changed = Vec::new();
        for task in &snap_b.tasks {
            match map_a.get(task.task_id.as_str()) {
                Some(prev_state) if **prev_state != task.state => {
                    changed.push(task.task_id.clone());
                }
                None => {
                    changed.push(task.task_id.clone());
                }
                _ => {}
            }
        }

        Some(changed)
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tasks() -> Vec<TaskSnapshot> {
        vec![
            TaskSnapshot::new("t1", CapturedTaskState::Completed),
            TaskSnapshot::new("t2", CapturedTaskState::Running),
            TaskSnapshot::new("t3", CapturedTaskState::Pending),
        ]
    }

    #[test]
    fn test_snapshot_seq_ordering() {
        let a = SnapshotSeq::new(1);
        let b = SnapshotSeq::new(2);
        assert!(a < b);
        assert_eq!(a.value(), 1);
    }

    #[test]
    fn test_snapshot_seq_display() {
        let s = SnapshotSeq::new(42);
        assert_eq!(format!("{s}"), "seq:42");
    }

    #[test]
    fn test_captured_task_state_display() {
        assert_eq!(format!("{}", CapturedTaskState::Pending), "pending");
        assert_eq!(format!("{}", CapturedTaskState::Running), "running");
        assert_eq!(format!("{}", CapturedTaskState::Completed), "completed");
        assert_eq!(
            format!("{}", CapturedTaskState::Failed("boom".into())),
            "failed: boom"
        );
        assert_eq!(format!("{}", CapturedTaskState::Skipped), "skipped");
    }

    #[test]
    fn test_task_snapshot_terminal() {
        assert!(TaskSnapshot::new("t", CapturedTaskState::Completed).is_terminal());
        assert!(TaskSnapshot::new("t", CapturedTaskState::Failed("x".into())).is_terminal());
        assert!(TaskSnapshot::new("t", CapturedTaskState::Skipped).is_terminal());
        assert!(!TaskSnapshot::new("t", CapturedTaskState::Pending).is_terminal());
        assert!(!TaskSnapshot::new("t", CapturedTaskState::Running).is_terminal());
    }

    #[test]
    fn test_task_snapshot_builder() {
        let ts = TaskSnapshot::new("t1", CapturedTaskState::Completed)
            .with_retries(3)
            .with_output("done");
        assert_eq!(ts.retry_count, 3);
        assert_eq!(ts.output.as_deref(), Some("done"));
    }

    #[test]
    fn test_workflow_snapshot_basic() {
        let mut ws = WorkflowSnapshot::new(SnapshotSeq::new(1), "wf-1", 1000, "initial");
        ws.add_task(TaskSnapshot::new("t1", CapturedTaskState::Pending));
        ws.set_metadata("user", "admin");
        assert_eq!(ws.task_count(), 1);
        assert_eq!(
            ws.metadata.get("user").expect("should succeed in test"),
            "admin"
        );
    }

    #[test]
    fn test_workflow_snapshot_completion_ratio() {
        let mut ws = WorkflowSnapshot::new(SnapshotSeq::new(1), "wf-1", 1000, "mid");
        for t in sample_tasks() {
            ws.add_task(t);
        }
        // 1 completed out of 3
        let ratio = ws.completion_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_workflow_snapshot_empty_ratio() {
        let ws = WorkflowSnapshot::new(SnapshotSeq::new(1), "wf-1", 1000, "empty");
        assert!((ws.completion_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_workflow_snapshot_find_task() {
        let mut ws = WorkflowSnapshot::new(SnapshotSeq::new(1), "wf-1", 1000, "mid");
        for t in sample_tasks() {
            ws.add_task(t);
        }
        assert!(ws.find_task("t1").is_some());
        assert!(ws.find_task("nonexistent").is_none());
    }

    #[test]
    fn test_snapshot_store_take_and_get() {
        let mut store = SnapshotStore::new("wf-1");
        assert!(store.is_empty());
        let seq = store.take_snapshot(1000, "first", sample_tasks());
        assert_eq!(store.len(), 1);
        let snap = store.get(seq).expect("should succeed in test");
        assert_eq!(snap.label, "first");
        assert_eq!(snap.task_count(), 3);
    }

    #[test]
    fn test_snapshot_store_latest() {
        let mut store = SnapshotStore::new("wf-1");
        store.take_snapshot(1000, "first", vec![]);
        store.take_snapshot(2000, "second", vec![]);
        let latest = store.latest().expect("should succeed in test");
        assert_eq!(latest.label, "second");
    }

    #[test]
    fn test_snapshot_store_max_retention() {
        let mut store = SnapshotStore::new("wf-1").with_max_snapshots(2);
        store.take_snapshot(1000, "one", vec![]);
        store.take_snapshot(2000, "two", vec![]);
        store.take_snapshot(3000, "three", vec![]);
        assert_eq!(store.len(), 2);
        // "one" should have been evicted
        let labels: Vec<&str> = store.list().iter().map(|(_, l)| *l).collect();
        assert_eq!(labels, vec!["two", "three"]);
    }

    #[test]
    fn test_snapshot_store_diff() {
        let mut store = SnapshotStore::new("wf-1");
        let tasks_a = vec![
            TaskSnapshot::new("t1", CapturedTaskState::Pending),
            TaskSnapshot::new("t2", CapturedTaskState::Pending),
        ];
        let seq_a = store.take_snapshot(1000, "before", tasks_a);

        let tasks_b = vec![
            TaskSnapshot::new("t1", CapturedTaskState::Completed),
            TaskSnapshot::new("t2", CapturedTaskState::Pending),
        ];
        let seq_b = store.take_snapshot(2000, "after", tasks_b);

        let changed = store.diff(seq_a, seq_b).expect("should succeed in test");
        assert_eq!(changed, vec!["t1"]);
    }

    #[test]
    fn test_snapshot_store_diff_new_task() {
        let mut store = SnapshotStore::new("wf-1");
        let seq_a = store.take_snapshot(1000, "before", vec![]);
        let tasks_b = vec![TaskSnapshot::new("t1", CapturedTaskState::Pending)];
        let seq_b = store.take_snapshot(2000, "after", tasks_b);

        let changed = store.diff(seq_a, seq_b).expect("should succeed in test");
        assert_eq!(changed, vec!["t1"]);
    }

    #[test]
    fn test_snapshot_store_clear() {
        let mut store = SnapshotStore::new("wf-1");
        store.take_snapshot(1000, "one", vec![]);
        store.take_snapshot(2000, "two", vec![]);
        assert_eq!(store.len(), 2);
        store.clear();
        assert!(store.is_empty());
    }
}
