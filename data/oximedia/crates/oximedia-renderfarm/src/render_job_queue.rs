#![allow(dead_code)]
//! Priority-based render job queue for the `OxiMedia` render farm.
//!
//! Provides a `JobUrgency` enum, a `RenderJob` with deadline inspection,
//! and a `RenderJobQueue` that orders work by priority.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Urgency level that determines scheduling priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobUrgency {
    /// Background, low-priority work.
    Background,
    /// Normal production job.
    Normal,
    /// Elevated priority (e.g. client review).
    High,
    /// Urgent delivery.
    Critical,
    /// Drop everything – deadline imminent.
    Emergency,
}

impl JobUrgency {
    /// Numeric priority score: higher = run sooner.
    #[must_use]
    pub fn numeric_priority(&self) -> u8 {
        match self {
            JobUrgency::Background => 0,
            JobUrgency::Normal => 25,
            JobUrgency::High => 50,
            JobUrgency::Critical => 75,
            JobUrgency::Emergency => 100,
        }
    }

    /// Returns `true` for urgency levels that should skip the normal queue.
    #[must_use]
    pub fn is_preemptive(&self) -> bool {
        matches!(self, JobUrgency::Critical | JobUrgency::Emergency)
    }
}

impl PartialOrd for JobUrgency {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JobUrgency {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numeric_priority().cmp(&other.numeric_priority())
    }
}

/// A single render job submitted to the farm queue.
#[derive(Debug, Clone)]
pub struct RenderJob {
    /// Unique job identifier.
    pub id: u64,
    /// Human-readable job name.
    pub name: String,
    /// First frame of the render range (inclusive).
    pub frame_start: u32,
    /// Last frame of the render range (inclusive).
    pub frame_end: u32,
    /// Scheduling urgency.
    pub urgency: JobUrgency,
    /// Optional Unix-epoch deadline timestamp (seconds).
    pub deadline_ts: Option<u64>,
    /// Path to the project file.
    pub project_path: String,
}

impl RenderJob {
    /// Create a new render job.
    #[must_use]
    pub fn new(
        id: u64,
        name: &str,
        frame_start: u32,
        frame_end: u32,
        urgency: JobUrgency,
        project_path: &str,
    ) -> Self {
        Self {
            id,
            name: name.to_string(),
            frame_start,
            frame_end,
            urgency,
            deadline_ts: None,
            project_path: project_path.to_string(),
        }
    }

    /// Returns `true` when the job has a deadline AND is at Critical or Emergency urgency.
    #[must_use]
    pub fn is_deadline_critical(&self) -> bool {
        self.deadline_ts.is_some() && self.urgency.is_preemptive()
    }

    /// Number of frames in the render range.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frame_end.saturating_sub(self.frame_start) + 1
    }

    /// Set a Unix-epoch deadline timestamp.
    #[must_use]
    pub fn with_deadline(mut self, ts: u64) -> Self {
        self.deadline_ts = Some(ts);
        self
    }
}

/// Internal wrapper that allows `RenderJob` to be stored in a `BinaryHeap`
/// ordered by urgency (highest urgency = highest priority).
#[derive(Debug)]
struct QueueEntry {
    job: RenderJob,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.job.urgency == other.job.urgency && self.job.id == other.job.id
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher urgency = higher priority = comes first in BinaryHeap (max-heap)
        self.job
            .urgency
            .cmp(&other.job.urgency)
            .then_with(|| self.job.id.cmp(&other.job.id).reverse())
    }
}

/// A priority-ordered queue of [`RenderJob`] items.
#[derive(Debug, Default)]
pub struct RenderJobQueue {
    heap: BinaryHeap<QueueEntry>,
}

impl RenderJobQueue {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Submit a job to the queue.
    pub fn submit(&mut self, job: RenderJob) {
        self.heap.push(QueueEntry { job });
    }

    /// Remove and return the highest-priority job, or `None` if the queue is empty.
    pub fn pop_next(&mut self) -> Option<RenderJob> {
        self.heap.pop().map(|e| e.job)
    }

    /// Peek at the highest-priority job without removing it.
    #[must_use]
    pub fn peek_next(&self) -> Option<&RenderJob> {
        self.heap.peek().map(|e| &e.job)
    }

    /// Number of jobs currently in the queue.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` when there are no pending jobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Count jobs at or above a given urgency level.
    #[must_use]
    pub fn count_at_least(&self, urgency: JobUrgency) -> usize {
        self.heap
            .iter()
            .filter(|e| e.job.urgency >= urgency)
            .count()
    }

    /// Drain all jobs into a `Vec` ordered from highest to lowest priority.
    pub fn drain_all(&mut self) -> Vec<RenderJob> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(job) = self.pop_next() {
            out.push(job);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: u64, urgency: JobUrgency) -> RenderJob {
        RenderJob::new(
            id,
            &format!("job_{id}"),
            1,
            100,
            urgency,
            "/proj/test.blend",
        )
    }

    #[test]
    fn test_urgency_background_priority_zero() {
        assert_eq!(JobUrgency::Background.numeric_priority(), 0);
    }

    #[test]
    fn test_urgency_emergency_priority_max() {
        assert_eq!(JobUrgency::Emergency.numeric_priority(), 100);
    }

    #[test]
    fn test_urgency_critical_is_preemptive() {
        assert!(JobUrgency::Critical.is_preemptive());
    }

    #[test]
    fn test_urgency_normal_not_preemptive() {
        assert!(!JobUrgency::Normal.is_preemptive());
    }

    #[test]
    fn test_urgency_ordering() {
        assert!(JobUrgency::Emergency > JobUrgency::Background);
    }

    #[test]
    fn test_render_job_frame_count() {
        let j = RenderJob::new(1, "test", 1, 100, JobUrgency::Normal, "/p.blend");
        assert_eq!(j.frame_count(), 100);
    }

    #[test]
    fn test_render_job_single_frame() {
        let j = RenderJob::new(2, "still", 42, 42, JobUrgency::Normal, "/p.blend");
        assert_eq!(j.frame_count(), 1);
    }

    #[test]
    fn test_render_job_not_deadline_critical_without_deadline() {
        let j = job(1, JobUrgency::Critical);
        assert!(!j.is_deadline_critical()); // no deadline set
    }

    #[test]
    fn test_render_job_deadline_critical_with_deadline_and_critical_urgency() {
        let j = job(1, JobUrgency::Critical).with_deadline(9_999_999);
        assert!(j.is_deadline_critical());
    }

    #[test]
    fn test_render_job_deadline_not_critical_with_normal_urgency() {
        let j = job(1, JobUrgency::Normal).with_deadline(9_999_999);
        assert!(!j.is_deadline_critical());
    }

    #[test]
    fn test_queue_initially_empty() {
        let q = RenderJobQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_queue_submit_increases_count() {
        let mut q = RenderJobQueue::new();
        q.submit(job(1, JobUrgency::Normal));
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn test_queue_pop_returns_highest_priority() {
        let mut q = RenderJobQueue::new();
        q.submit(job(1, JobUrgency::Background));
        q.submit(job(2, JobUrgency::Emergency));
        q.submit(job(3, JobUrgency::Normal));
        let first = q.pop_next().expect("should succeed in test");
        assert_eq!(first.urgency, JobUrgency::Emergency);
    }

    #[test]
    fn test_queue_pop_from_empty_is_none() {
        let mut q = RenderJobQueue::new();
        assert!(q.pop_next().is_none());
    }

    #[test]
    fn test_queue_count_at_least() {
        let mut q = RenderJobQueue::new();
        q.submit(job(1, JobUrgency::Background));
        q.submit(job(2, JobUrgency::Critical));
        assert_eq!(q.count_at_least(JobUrgency::Critical), 1);
    }

    #[test]
    fn test_queue_drain_all_ordered() {
        let mut q = RenderJobQueue::new();
        q.submit(job(1, JobUrgency::Background));
        q.submit(job(2, JobUrgency::Emergency));
        q.submit(job(3, JobUrgency::Normal));
        let drained = q.drain_all();
        assert_eq!(drained[0].urgency, JobUrgency::Emergency);
        assert!(q.is_empty());
    }
}
