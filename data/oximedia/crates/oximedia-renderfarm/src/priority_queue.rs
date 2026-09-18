//! Priority queue for render job scheduling.
//!
//! Provides [`RenderPriority`] levels, a [`PrioritizedJob`] wrapper, and a
//! [`RenderPriorityQueue`] backed by a binary heap so that the highest-priority
//! job is always dequeued first.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Numeric priority levels for render jobs.
///
/// Variants are ordered so that `Critical > High > Normal > Low > Background`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPriority {
    /// Reserved for emergency re-renders needed before broadcast.
    Critical,
    /// Elevated priority for client-facing deliverables.
    High,
    /// Standard production renders.
    Normal,
    /// Internal preview or test renders.
    Low,
    /// Overnight or opportunistic work.
    Background,
}

impl RenderPriority {
    /// Numeric weight used for ordering (higher = more urgent).
    #[must_use]
    pub const fn weight(self) -> u8 {
        match self {
            Self::Critical => 100,
            Self::High => 75,
            Self::Normal => 50,
            Self::Low => 25,
            Self::Background => 0,
        }
    }

    /// Return `true` if this priority level is at least `Normal`.
    #[must_use]
    pub const fn is_production(self) -> bool {
        matches!(self, Self::Critical | Self::High | Self::Normal)
    }
}

impl PartialOrd for RenderPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RenderPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl std::fmt::Display for RenderPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "critical"),
            Self::High => write!(f, "high"),
            Self::Normal => write!(f, "normal"),
            Self::Low => write!(f, "low"),
            Self::Background => write!(f, "background"),
        }
    }
}

/// A render job wrapped with scheduling metadata.
///
/// `T` is the application-defined job descriptor (e.g. a job ID `String` or a
/// full job struct).
#[derive(Debug, Clone)]
pub struct PrioritizedJob<T> {
    /// The scheduling priority for this job.
    pub priority: RenderPriority,
    /// Time at which the job was enqueued (used for FIFO tie-breaking).
    pub enqueued_at: Instant,
    /// The application-level job data.
    pub job: T,
}

impl<T> PrioritizedJob<T> {
    /// Wrap a job with a given priority, recording the current time.
    #[must_use]
    pub fn new(priority: RenderPriority, job: T) -> Self {
        Self {
            priority,
            enqueued_at: Instant::now(),
            job,
        }
    }
}

// BinaryHeap is a max-heap; we want higher priority first.
// For equal priority, earlier enqueue time wins (lower Instant = higher order).
impl<T> PartialEq for PrioritizedJob<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.enqueued_at == other.enqueued_at
    }
}

impl<T> Eq for PrioritizedJob<T> {}

impl<T> PartialOrd for PrioritizedJob<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PrioritizedJob<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary sort: higher weight wins.
        match self.priority.weight().cmp(&other.priority.weight()) {
            Ordering::Equal => {
                // Tie-break: earlier enqueue time wins (reverse so that
                // the heap pops the smallest Instant first).
                other.enqueued_at.cmp(&self.enqueued_at)
            }
            ord => ord,
        }
    }
}

/// A max-priority queue for render jobs.
///
/// Jobs are dequeued in descending priority order; within the same priority
/// level the job that was enqueued earliest is returned first.
///
/// # Example
///
/// ```
/// use oximedia_renderfarm::priority_queue::{RenderPriority, RenderPriorityQueue};
///
/// let mut q: RenderPriorityQueue<String> = RenderPriorityQueue::new();
/// q.push(RenderPriority::Low, "background_job".to_string());
/// q.push(RenderPriority::Critical, "urgent_job".to_string());
///
/// let first = q.pop().expect("should succeed in test");
/// assert_eq!(first.priority, RenderPriority::Critical);
/// ```
#[derive(Debug, Default)]
pub struct RenderPriorityQueue<T> {
    heap: BinaryHeap<PrioritizedJob<T>>,
}

impl<T> RenderPriorityQueue<T> {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Enqueue a job with the given priority.
    pub fn push(&mut self, priority: RenderPriority, job: T) {
        self.heap.push(PrioritizedJob::new(priority, job));
    }

    /// Dequeue and return the highest-priority job, or `None` if empty.
    pub fn pop(&mut self) -> Option<PrioritizedJob<T>> {
        self.heap.pop()
    }

    /// Peek at the next job without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PrioritizedJob<T>> {
        self.heap.peek()
    }

    /// Number of jobs currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Return `true` if the queue has no jobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain all jobs in priority order.
    pub fn drain_ordered(&mut self) -> Vec<PrioritizedJob<T>> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(j) = self.heap.pop() {
            out.push(j);
        }
        out
    }

    /// Remove all jobs with priority strictly below `min_priority`.
    pub fn prune_below(&mut self, min_priority: RenderPriority) {
        let kept: Vec<_> = std::mem::take(&mut self.heap)
            .into_iter()
            .filter(|j| j.priority >= min_priority)
            .collect();
        self.heap = BinaryHeap::from(kept);
    }

    /// Count jobs at exactly `priority`.
    #[must_use]
    pub fn count_at(&self, priority: RenderPriority) -> usize {
        self.heap.iter().filter(|j| j.priority == priority).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_weight_ordering() {
        assert!(RenderPriority::Critical.weight() > RenderPriority::High.weight());
        assert!(RenderPriority::High.weight() > RenderPriority::Normal.weight());
        assert!(RenderPriority::Normal.weight() > RenderPriority::Low.weight());
        assert!(RenderPriority::Low.weight() > RenderPriority::Background.weight());
    }

    #[test]
    fn priority_ord_critical_gt_background() {
        assert!(RenderPriority::Critical > RenderPriority::Background);
    }

    #[test]
    fn priority_is_production() {
        assert!(RenderPriority::Normal.is_production());
        assert!(!RenderPriority::Background.is_production());
    }

    #[test]
    fn priority_display() {
        assert_eq!(RenderPriority::High.to_string(), "high");
        assert_eq!(RenderPriority::Background.to_string(), "background");
    }

    #[test]
    fn queue_push_pop_order() {
        let mut q: RenderPriorityQueue<&str> = RenderPriorityQueue::new();
        q.push(RenderPriority::Low, "low");
        q.push(RenderPriority::Critical, "crit");
        q.push(RenderPriority::Normal, "norm");
        let first = q.pop().expect("should succeed in test");
        assert_eq!(first.priority, RenderPriority::Critical);
    }

    #[test]
    fn queue_is_empty_initially() {
        let q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        assert!(q.is_empty());
    }

    #[test]
    fn queue_len_after_pushes() {
        let mut q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        q.push(RenderPriority::Normal, 1);
        q.push(RenderPriority::High, 2);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn queue_pop_empty_returns_none() {
        let mut q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        assert!(q.pop().is_none());
    }

    #[test]
    fn queue_peek_does_not_remove() {
        let mut q: RenderPriorityQueue<&str> = RenderPriorityQueue::new();
        q.push(RenderPriority::High, "hi");
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_drain_ordered_empties_queue() {
        let mut q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        q.push(RenderPriority::Low, 1);
        q.push(RenderPriority::High, 2);
        let ordered = q.drain_ordered();
        assert_eq!(ordered[0].priority, RenderPriority::High);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_prune_below_removes_low() {
        let mut q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        q.push(RenderPriority::Background, 0);
        q.push(RenderPriority::Normal, 1);
        q.prune_below(RenderPriority::Normal);
        assert_eq!(q.len(), 1);
        assert_eq!(
            q.pop().expect("should succeed in test").priority,
            RenderPriority::Normal
        );
    }

    #[test]
    fn queue_count_at_priority() {
        let mut q: RenderPriorityQueue<i32> = RenderPriorityQueue::new();
        q.push(RenderPriority::High, 1);
        q.push(RenderPriority::High, 2);
        q.push(RenderPriority::Low, 3);
        assert_eq!(q.count_at(RenderPriority::High), 2);
        assert_eq!(q.count_at(RenderPriority::Low), 1);
    }

    #[test]
    fn prioritized_job_new_captures_priority() {
        let j = PrioritizedJob::new(RenderPriority::Critical, "job");
        assert_eq!(j.priority, RenderPriority::Critical);
    }

    #[test]
    fn queue_default_is_empty() {
        let q: RenderPriorityQueue<i32> = RenderPriorityQueue::default();
        assert!(q.is_empty());
    }
}
