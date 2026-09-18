#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
//! Job priority queue for render farm scheduling.
//!
//! Provides urgency-weighted priority queuing for render jobs,
//! enabling fair and efficient scheduling across heterogeneous workloads.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Urgency level for a render job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RenderUrgency {
    /// Background work, processed when no other jobs are waiting.
    Background,
    /// Normal production priority.
    #[default]
    Normal,
    /// Elevated priority for time-sensitive work.
    High,
    /// Rush deliverable; preempts Normal and Background.
    Rush,
    /// Emergency; must render before any other job class.
    Emergency,
}

impl RenderUrgency {
    /// Numeric weight used to calculate effective priority.
    ///
    /// Higher weight means the job competes more aggressively for slots.
    pub fn weight(self) -> u32 {
        match self {
            Self::Background => 1,
            Self::Normal => 10,
            Self::High => 50,
            Self::Rush => 200,
            Self::Emergency => 1000,
        }
    }

    /// Human-readable label for the urgency level.
    pub fn label(self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Rush => "rush",
            Self::Emergency => "emergency",
        }
    }

    /// Returns `true` if this urgency can preempt currently running jobs.
    pub fn is_preemptive(self) -> bool {
        matches!(self, Self::Rush | Self::Emergency)
    }
}

/// A render job decorated with priority metadata.
#[derive(Debug, Clone)]
pub struct PrioritizedJob {
    /// Unique job identifier.
    pub job_id: u64,
    /// Base numeric priority supplied by the user (0–100).
    pub base_priority: u32,
    /// Urgency classification.
    pub urgency: RenderUrgency,
    /// Wall-clock submission timestamp (Unix seconds).
    pub submitted_at: u64,
    /// Optional project name for display purposes.
    pub project: Option<String>,
}

impl PrioritizedJob {
    /// Creates a new prioritized job.
    pub fn new(job_id: u64, base_priority: u32, urgency: RenderUrgency, submitted_at: u64) -> Self {
        Self {
            job_id,
            base_priority,
            urgency,
            submitted_at,
            project: None,
        }
    }

    /// Attaches a project name to the job.
    pub fn with_project(mut self, name: impl Into<String>) -> Self {
        self.project = Some(name.into());
        self
    }

    /// Computes the effective scheduling priority.
    ///
    /// `effective_priority = base_priority * urgency_weight + age_bonus`
    ///
    /// An age bonus prevents starvation of low-priority jobs.
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_priority(&self, now_secs: u64) -> u64 {
        let age = now_secs.saturating_sub(self.submitted_at);
        let age_bonus = (age as f64).sqrt() as u64;
        let base = u64::from(self.base_priority) * u64::from(self.urgency.weight());
        base + age_bonus
    }
}

// ── BinaryHeap ordering wrapper ────────────────────────────────────────────

/// Internal heap entry that carries the snapshot priority so the heap
/// remains consistent even if wall-clock time advances.
#[derive(Debug, Clone)]
struct HeapEntry {
    priority_snapshot: u64,
    job: PrioritizedJob,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority_snapshot == other.priority_snapshot
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_snapshot.cmp(&other.priority_snapshot)
    }
}

// ── JobPriorityQueue ───────────────────────────────────────────────────────

/// Max-priority queue for render jobs.
///
/// Jobs are ordered by their effective priority (highest first).
/// The queue is not thread-safe; wrap in a `Mutex` for shared access.
#[derive(Debug, Default)]
pub struct JobPriorityQueue {
    heap: BinaryHeap<HeapEntry>,
    now_secs: u64,
}

impl JobPriorityQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a queue with an explicit current time (useful for tests).
    pub fn with_time(now_secs: u64) -> Self {
        Self {
            heap: BinaryHeap::new(),
            now_secs,
        }
    }

    /// Updates the queue's notion of the current time.
    ///
    /// Must be called before `push` to get accurate age bonuses.
    pub fn set_time(&mut self, now_secs: u64) {
        self.now_secs = now_secs;
    }

    /// Pushes a job onto the queue.
    pub fn push(&mut self, job: PrioritizedJob) {
        let priority_snapshot = job.effective_priority(self.now_secs);
        self.heap.push(HeapEntry {
            priority_snapshot,
            job,
        });
    }

    /// Removes and returns the highest-priority job, or `None` if empty.
    pub fn pop(&mut self) -> Option<PrioritizedJob> {
        self.heap.pop().map(|e| e.job)
    }

    /// Returns a reference to the highest-priority job without removing it.
    pub fn peek(&self) -> Option<&PrioritizedJob> {
        self.heap.peek().map(|e| &e.job)
    }

    /// Returns the number of jobs currently in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue contains no jobs.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drains all jobs sorted by priority (highest first).
    pub fn drain_sorted(mut self) -> Vec<PrioritizedJob> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(j) = self.pop() {
            out.push(j);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: u64, base: u32, urgency: RenderUrgency) -> PrioritizedJob {
        PrioritizedJob::new(id, base, urgency, 0)
    }

    #[test]
    fn test_urgency_weights_ordered() {
        assert!(RenderUrgency::Background.weight() < RenderUrgency::Normal.weight());
        assert!(RenderUrgency::Normal.weight() < RenderUrgency::High.weight());
        assert!(RenderUrgency::High.weight() < RenderUrgency::Rush.weight());
        assert!(RenderUrgency::Rush.weight() < RenderUrgency::Emergency.weight());
    }

    #[test]
    fn test_urgency_labels() {
        assert_eq!(RenderUrgency::Background.label(), "background");
        assert_eq!(RenderUrgency::Emergency.label(), "emergency");
    }

    #[test]
    fn test_urgency_preemptive() {
        assert!(!RenderUrgency::Normal.is_preemptive());
        assert!(RenderUrgency::Rush.is_preemptive());
        assert!(RenderUrgency::Emergency.is_preemptive());
    }

    #[test]
    fn test_default_urgency() {
        assert_eq!(RenderUrgency::default(), RenderUrgency::Normal);
    }

    #[test]
    fn test_effective_priority_increases_with_urgency() {
        let low = make_job(1, 10, RenderUrgency::Normal);
        let high = make_job(2, 10, RenderUrgency::Emergency);
        assert!(high.effective_priority(0) > low.effective_priority(0));
    }

    #[test]
    fn test_effective_priority_age_bonus() {
        let job = make_job(1, 10, RenderUrgency::Normal);
        let p_young = job.effective_priority(0);
        let p_old = job.effective_priority(10_000);
        assert!(p_old > p_young);
    }

    #[test]
    fn test_with_project() {
        let job = make_job(1, 50, RenderUrgency::High).with_project("MovieA");
        assert_eq!(job.project.as_deref(), Some("MovieA"));
    }

    #[test]
    fn test_queue_empty_on_new() {
        let q = JobPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_queue_push_pop_single() {
        let mut q = JobPriorityQueue::new();
        q.push(make_job(42, 50, RenderUrgency::Normal));
        assert_eq!(q.len(), 1);
        let job = q.pop().expect("should succeed in test");
        assert_eq!(job.job_id, 42);
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_pop_highest_first() {
        let mut q = JobPriorityQueue::new();
        q.push(make_job(1, 10, RenderUrgency::Normal));
        q.push(make_job(2, 10, RenderUrgency::Emergency));
        q.push(make_job(3, 10, RenderUrgency::Background));
        // Emergency should come out first
        assert_eq!(q.pop().expect("should succeed in test").job_id, 2);
        // Then Normal
        assert_eq!(q.pop().expect("should succeed in test").job_id, 1);
        // Then Background
        assert_eq!(q.pop().expect("should succeed in test").job_id, 3);
    }

    #[test]
    fn test_queue_peek_does_not_remove() {
        let mut q = JobPriorityQueue::new();
        q.push(make_job(7, 10, RenderUrgency::High));
        let _ = q.peek().expect("should succeed in test");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_drain_sorted() {
        let mut q = JobPriorityQueue::new();
        q.push(make_job(1, 5, RenderUrgency::Normal));
        q.push(make_job(2, 5, RenderUrgency::Rush));
        let sorted = q.drain_sorted();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].job_id, 2); // Rush first
    }

    #[test]
    fn test_queue_pop_empty_returns_none() {
        let mut q = JobPriorityQueue::new();
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_queue_with_time() {
        let q = JobPriorityQueue::with_time(9999);
        assert_eq!(q.now_secs, 9999);
        assert!(q.is_empty());
    }
}
