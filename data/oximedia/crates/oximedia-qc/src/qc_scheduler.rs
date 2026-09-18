//! QC job scheduler for prioritised quality-control execution.
//!
//! Provides `QcPriority`, `QcJob`, and `QcScheduler` for managing
//! a queue of QC jobs ordered by priority.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Priority level for a QC job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QcPriority {
    /// Run when nothing else is scheduled.
    Low,
    /// Standard scheduling priority.
    Normal,
    /// Elevated priority; processed before Normal jobs.
    High,
    /// Processed immediately ahead of all other jobs.
    Critical,
}

impl QcPriority {
    /// Returns a numeric value for this priority (higher = more urgent).
    pub fn numeric_value(self) -> u8 {
        match self {
            QcPriority::Low => 1,
            QcPriority::Normal => 2,
            QcPriority::High => 3,
            QcPriority::Critical => 4,
        }
    }

    /// Returns `true` if this priority is at least High.
    pub fn is_elevated(self) -> bool {
        self >= QcPriority::High
    }
}

/// A single QC job awaiting execution.
#[derive(Debug, Clone)]
pub struct QcJob {
    /// Unique identifier for this job.
    pub id: u64,
    /// Path to the media file to be checked.
    pub file_path: String,
    /// Name of the QC profile to apply.
    pub profile_name: String,
    /// Scheduling priority.
    pub priority: QcPriority,
    /// Deadline in Unix seconds, if any.
    pub deadline_unix_secs: Option<u64>,
}

impl QcJob {
    /// Creates a new QC job.
    pub fn new(
        id: u64,
        file_path: impl Into<String>,
        profile_name: impl Into<String>,
        priority: QcPriority,
    ) -> Self {
        Self {
            id,
            file_path: file_path.into(),
            profile_name: profile_name.into(),
            priority,
            deadline_unix_secs: None,
        }
    }

    /// Attaches a deadline (Unix timestamp in seconds) to this job.
    pub fn with_deadline(mut self, unix_secs: u64) -> Self {
        self.deadline_unix_secs = Some(unix_secs);
        self
    }

    /// Returns `true` if this job has Critical or High priority.
    pub fn is_urgent(&self) -> bool {
        self.priority.is_elevated()
    }

    /// Returns `true` if the job has a deadline set.
    pub fn has_deadline(&self) -> bool {
        self.deadline_unix_secs.is_some()
    }
}

/// A priority-aware scheduler for QC jobs.
///
/// Jobs are stored in a `VecDeque` and `next_job()` selects the
/// highest-priority (then lowest id) job from the pending queue.
#[derive(Debug, Default)]
pub struct QcScheduler {
    queue: VecDeque<QcJob>,
    next_id: u64,
}

impl QcScheduler {
    /// Creates a new, empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a new job ID and returns it.
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Enqueues a job. Returns the assigned job ID.
    pub fn enqueue(
        &mut self,
        file_path: impl Into<String>,
        profile_name: impl Into<String>,
        priority: QcPriority,
    ) -> u64 {
        let id = self.alloc_id();
        let job = QcJob::new(id, file_path, profile_name, priority);
        self.queue.push_back(job);
        id
    }

    /// Enqueues a fully constructed job directly.
    pub fn enqueue_job(&mut self, mut job: QcJob) -> u64 {
        let id = self.alloc_id();
        job.id = id;
        self.queue.push_back(job);
        id
    }

    /// Removes and returns the highest-priority pending job.
    ///
    /// When multiple jobs share the same priority, the one with the lowest
    /// ID (earliest enqueued) is returned.
    pub fn next_job(&mut self) -> Option<QcJob> {
        if self.queue.is_empty() {
            return None;
        }
        // Find index of best job.
        let best_idx = self
            .queue
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| {
                a.priority.cmp(&b.priority).then(bi.cmp(ai)) // lower index wins on tie
            })
            .map(|(i, _)| i)?;

        self.queue.remove(best_idx)
    }

    /// Returns the number of jobs currently pending.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if there are no pending jobs.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of pending urgent jobs (High or Critical).
    pub fn urgent_count(&self) -> usize {
        self.queue.iter().filter(|j| j.is_urgent()).count()
    }

    /// Removes all pending jobs and returns them.
    pub fn drain(&mut self) -> Vec<QcJob> {
        self.queue.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_numeric_values_ordered() {
        assert!(QcPriority::Low.numeric_value() < QcPriority::Normal.numeric_value());
        assert!(QcPriority::Normal.numeric_value() < QcPriority::High.numeric_value());
        assert!(QcPriority::High.numeric_value() < QcPriority::Critical.numeric_value());
    }

    #[test]
    fn test_priority_is_elevated() {
        assert!(QcPriority::High.is_elevated());
        assert!(QcPriority::Critical.is_elevated());
        assert!(!QcPriority::Normal.is_elevated());
        assert!(!QcPriority::Low.is_elevated());
    }

    #[test]
    fn test_job_is_urgent() {
        let job = QcJob::new(0, "file.mxf", "Broadcast", QcPriority::Critical);
        assert!(job.is_urgent());
        let low_job = QcJob::new(1, "file.mp4", "Streaming", QcPriority::Low);
        assert!(!low_job.is_urgent());
    }

    #[test]
    fn test_job_has_deadline() {
        let job = QcJob::new(0, "f", "p", QcPriority::Normal).with_deadline(1_700_000_000);
        assert!(job.has_deadline());
        assert_eq!(job.deadline_unix_secs, Some(1_700_000_000));
    }

    #[test]
    fn test_job_no_deadline_by_default() {
        let job = QcJob::new(0, "f", "p", QcPriority::Normal);
        assert!(!job.has_deadline());
    }

    #[test]
    fn test_scheduler_enqueue_increments_pending() {
        let mut sched = QcScheduler::new();
        sched.enqueue("a.mp4", "Broadcast", QcPriority::Normal);
        sched.enqueue("b.mp4", "Streaming", QcPriority::Low);
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn test_scheduler_next_job_returns_highest_priority() {
        let mut sched = QcScheduler::new();
        sched.enqueue("low.mp4", "S", QcPriority::Low);
        sched.enqueue("high.mp4", "B", QcPriority::High);
        sched.enqueue("normal.mp4", "S", QcPriority::Normal);

        let job = sched.next_job().expect("should succeed in test");
        assert_eq!(job.priority, QcPriority::High);
    }

    #[test]
    fn test_scheduler_fifo_on_equal_priority() {
        let mut sched = QcScheduler::new();
        let id1 = sched.enqueue("first.mp4", "S", QcPriority::Normal);
        let _id2 = sched.enqueue("second.mp4", "S", QcPriority::Normal);

        let job = sched.next_job().expect("should succeed in test");
        assert_eq!(job.id, id1);
    }

    #[test]
    fn test_scheduler_empty_returns_none() {
        let mut sched = QcScheduler::new();
        assert!(sched.next_job().is_none());
    }

    #[test]
    fn test_scheduler_is_empty() {
        let mut sched = QcScheduler::new();
        assert!(sched.is_empty());
        sched.enqueue("f", "p", QcPriority::Low);
        assert!(!sched.is_empty());
    }

    #[test]
    fn test_scheduler_urgent_count() {
        let mut sched = QcScheduler::new();
        sched.enqueue("a", "p", QcPriority::Low);
        sched.enqueue("b", "p", QcPriority::High);
        sched.enqueue("c", "p", QcPriority::Critical);
        assert_eq!(sched.urgent_count(), 2);
    }

    #[test]
    fn test_scheduler_drain() {
        let mut sched = QcScheduler::new();
        sched.enqueue("x", "p", QcPriority::Normal);
        sched.enqueue("y", "p", QcPriority::Normal);
        let drained = sched.drain();
        assert_eq!(drained.len(), 2);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_scheduler_critical_before_low() {
        let mut sched = QcScheduler::new();
        sched.enqueue("low1", "p", QcPriority::Low);
        sched.enqueue("low2", "p", QcPriority::Low);
        sched.enqueue("crit", "p", QcPriority::Critical);
        let job = sched.next_job().expect("should succeed in test");
        assert_eq!(job.priority, QcPriority::Critical);
        assert_eq!(sched.pending_count(), 2);
    }
}
