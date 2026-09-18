#![allow(dead_code)]
//! Priority-based task scheduling queue.
//!
//! Provides a priority queue that orders workflow tasks by priority level,
//! deadline, and submission order. Supports multi-level priority with
//! starvation prevention through priority aging.

use std::collections::BinaryHeap;

/// Priority level for a queued task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityLevel {
    /// Lowest priority — background tasks.
    Low = 0,
    /// Default priority for normal tasks.
    Normal = 1,
    /// Elevated priority for time-sensitive work.
    High = 2,
    /// Highest priority — critical / emergency tasks.
    Critical = 3,
}

impl PriorityLevel {
    /// Return the numeric weight of this priority level.
    #[must_use]
    pub fn weight(self) -> u32 {
        self as u32
    }

    /// Promote to the next higher level (caps at Critical).
    #[must_use]
    pub fn promote(self) -> Self {
        match self {
            Self::Low => Self::Normal,
            Self::Normal => Self::High,
            Self::High | Self::Critical => Self::Critical,
        }
    }
}

impl std::fmt::Display for PriorityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// An entry in the priority queue.
#[derive(Debug, Clone)]
pub struct PriorityEntry {
    /// Task identifier.
    pub task_id: String,
    /// Current effective priority.
    pub priority: PriorityLevel,
    /// Original priority before any aging.
    pub base_priority: PriorityLevel,
    /// Optional deadline (seconds since epoch; 0 = none).
    pub deadline_secs: u64,
    /// Submission timestamp (seconds since epoch).
    pub submitted_secs: u64,
    /// Internal insertion order for tie-breaking.
    insertion_order: u64,
    /// Number of times this entry has been aged.
    pub age_count: u32,
}

impl PriorityEntry {
    /// Create a new priority entry.
    pub fn new(
        task_id: impl Into<String>,
        priority: PriorityLevel,
        submitted_secs: u64,
        insertion_order: u64,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            priority,
            base_priority: priority,
            deadline_secs: 0,
            submitted_secs,
            insertion_order,
            age_count: 0,
        }
    }

    /// Set a deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline_secs: u64) -> Self {
        self.deadline_secs = deadline_secs;
        self
    }

    /// Return true if this entry has a deadline.
    #[must_use]
    pub fn has_deadline(&self) -> bool {
        self.deadline_secs > 0
    }

    /// Check whether the deadline has passed relative to `now_secs`.
    #[must_use]
    pub fn is_overdue(&self, now_secs: u64) -> bool {
        self.has_deadline() && now_secs > self.deadline_secs
    }

    /// Return the time this entry has been waiting (in seconds).
    #[must_use]
    pub fn wait_time(&self, now_secs: u64) -> u64 {
        now_secs.saturating_sub(self.submitted_secs)
    }
}

// Ordering: higher priority first, then earlier deadline, then earlier submission
impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first
        self.priority
            .cmp(&other.priority)
            // Then earlier deadline first (reverse: smaller is better)
            .then_with(|| {
                if self.deadline_secs == 0 && other.deadline_secs == 0 {
                    std::cmp::Ordering::Equal
                } else if self.deadline_secs == 0 {
                    std::cmp::Ordering::Less // no deadline => lower urgency
                } else if other.deadline_secs == 0 {
                    std::cmp::Ordering::Greater
                } else {
                    other.deadline_secs.cmp(&self.deadline_secs) // smaller deadline is more urgent
                }
            })
            // Then earlier submission first (FIFO for ties)
            .then_with(|| other.submitted_secs.cmp(&self.submitted_secs))
            // Final tie-break: earlier insertion wins
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

/// A priority queue for scheduling workflow tasks.
#[derive(Debug)]
pub struct TaskPriorityQueue {
    /// The underlying binary heap.
    heap: BinaryHeap<PriorityEntry>,
    /// Counter for insertion ordering.
    insertion_counter: u64,
    /// How many seconds of wait time before priority is promoted.
    aging_threshold_secs: u64,
}

impl TaskPriorityQueue {
    /// Create a new empty priority queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            insertion_counter: 0,
            aging_threshold_secs: 300, // 5 minutes default
        }
    }

    /// Set the aging threshold (seconds of waiting before priority promotion).
    #[must_use]
    pub fn with_aging_threshold(mut self, secs: u64) -> Self {
        self.aging_threshold_secs = secs;
        self
    }

    /// Enqueue a task with the given priority and submission time.
    pub fn enqueue(
        &mut self,
        task_id: impl Into<String>,
        priority: PriorityLevel,
        submitted_secs: u64,
    ) -> u64 {
        let order = self.insertion_counter;
        self.insertion_counter += 1;
        let entry = PriorityEntry::new(task_id, priority, submitted_secs, order);
        self.heap.push(entry);
        order
    }

    /// Enqueue a task with a deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        task_id: impl Into<String>,
        priority: PriorityLevel,
        submitted_secs: u64,
        deadline_secs: u64,
    ) -> u64 {
        let order = self.insertion_counter;
        self.insertion_counter += 1;
        let entry = PriorityEntry::new(task_id, priority, submitted_secs, order)
            .with_deadline(deadline_secs);
        self.heap.push(entry);
        order
    }

    /// Dequeue the highest-priority task.
    pub fn dequeue(&mut self) -> Option<PriorityEntry> {
        self.heap.pop()
    }

    /// Peek at the highest-priority task without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PriorityEntry> {
        self.heap.peek()
    }

    /// Return the number of enqueued tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Apply priority aging: promote tasks that have been waiting longer than the threshold.
    ///
    /// This drains and rebuilds the heap, so call sparingly.
    pub fn apply_aging(&mut self, now_secs: u64) {
        let threshold = self.aging_threshold_secs;
        let entries: Vec<PriorityEntry> = self.heap.drain().collect();
        for mut entry in entries {
            if entry.wait_time(now_secs) > threshold * (u64::from(entry.age_count) + 1) {
                entry.priority = entry.priority.promote();
                entry.age_count += 1;
            }
            self.heap.push(entry);
        }
    }

    /// Drain all overdue tasks (past their deadline).
    pub fn drain_overdue(&mut self, now_secs: u64) -> Vec<PriorityEntry> {
        let mut overdue = Vec::new();
        let mut remaining = Vec::new();
        while let Some(entry) = self.heap.pop() {
            if entry.is_overdue(now_secs) {
                overdue.push(entry);
            } else {
                remaining.push(entry);
            }
        }
        for e in remaining {
            self.heap.push(e);
        }
        overdue
    }

    /// Clear all tasks from the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Count tasks at each priority level.
    #[must_use]
    pub fn count_by_priority(&self) -> [usize; 4] {
        let mut counts = [0_usize; 4];
        for entry in &self.heap {
            let idx = entry.priority.weight() as usize;
            if idx < 4 {
                counts[idx] += 1;
            }
        }
        counts
    }
}

impl Default for TaskPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_level_ordering() {
        assert!(PriorityLevel::Critical > PriorityLevel::High);
        assert!(PriorityLevel::High > PriorityLevel::Normal);
        assert!(PriorityLevel::Normal > PriorityLevel::Low);
    }

    #[test]
    fn test_priority_level_promote() {
        assert_eq!(PriorityLevel::Low.promote(), PriorityLevel::Normal);
        assert_eq!(PriorityLevel::Normal.promote(), PriorityLevel::High);
        assert_eq!(PriorityLevel::High.promote(), PriorityLevel::Critical);
        assert_eq!(PriorityLevel::Critical.promote(), PriorityLevel::Critical);
    }

    #[test]
    fn test_priority_level_display() {
        assert_eq!(format!("{}", PriorityLevel::Low), "low");
        assert_eq!(format!("{}", PriorityLevel::Normal), "normal");
        assert_eq!(format!("{}", PriorityLevel::High), "high");
        assert_eq!(format!("{}", PriorityLevel::Critical), "critical");
    }

    #[test]
    fn test_priority_entry_deadline() {
        let entry = PriorityEntry::new("t1", PriorityLevel::Normal, 1000, 0).with_deadline(2000);
        assert!(entry.has_deadline());
        assert!(!entry.is_overdue(1500));
        assert!(entry.is_overdue(2500));
    }

    #[test]
    fn test_priority_entry_no_deadline() {
        let entry = PriorityEntry::new("t1", PriorityLevel::Normal, 1000, 0);
        assert!(!entry.has_deadline());
        assert!(!entry.is_overdue(9999));
    }

    #[test]
    fn test_priority_entry_wait_time() {
        let entry = PriorityEntry::new("t1", PriorityLevel::Normal, 1000, 0);
        assert_eq!(entry.wait_time(1500), 500);
        assert_eq!(entry.wait_time(500), 0); // saturating
    }

    #[test]
    fn test_queue_enqueue_dequeue_priority() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue("low", PriorityLevel::Low, 1000);
        q.enqueue("critical", PriorityLevel::Critical, 1000);
        q.enqueue("normal", PriorityLevel::Normal, 1000);

        let first = q.dequeue().expect("should succeed in test");
        assert_eq!(first.task_id, "critical");
        let second = q.dequeue().expect("should succeed in test");
        assert_eq!(second.task_id, "normal");
        let third = q.dequeue().expect("should succeed in test");
        assert_eq!(third.task_id, "low");
    }

    #[test]
    fn test_queue_fifo_within_same_priority() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue("a", PriorityLevel::Normal, 1000);
        q.enqueue("b", PriorityLevel::Normal, 1000);
        q.enqueue("c", PriorityLevel::Normal, 1000);

        assert_eq!(q.dequeue().expect("should succeed in test").task_id, "a");
        assert_eq!(q.dequeue().expect("should succeed in test").task_id, "b");
        assert_eq!(q.dequeue().expect("should succeed in test").task_id, "c");
    }

    #[test]
    fn test_queue_deadline_ordering() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue_with_deadline("far", PriorityLevel::Normal, 1000, 5000);
        q.enqueue_with_deadline("near", PriorityLevel::Normal, 1000, 2000);

        // Nearer deadline should come first
        assert_eq!(q.dequeue().expect("should succeed in test").task_id, "near");
        assert_eq!(q.dequeue().expect("should succeed in test").task_id, "far");
    }

    #[test]
    fn test_queue_len_and_empty() {
        let mut q = TaskPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        q.enqueue("a", PriorityLevel::Normal, 1000);
        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_peek() {
        let mut q = TaskPriorityQueue::new();
        assert!(q.peek().is_none());
        q.enqueue("a", PriorityLevel::High, 1000);
        q.enqueue("b", PriorityLevel::Low, 1000);
        assert_eq!(q.peek().expect("should succeed in test").task_id, "a");
        assert_eq!(q.len(), 2); // peek doesn't remove
    }

    #[test]
    fn test_queue_apply_aging() {
        let mut q = TaskPriorityQueue::new().with_aging_threshold(100);
        q.enqueue("old", PriorityLevel::Low, 0);
        q.enqueue("new", PriorityLevel::Low, 900);

        // At time 200, "old" has waited 200s > threshold 100, should promote
        q.apply_aging(200);

        let first = q.dequeue().expect("should succeed in test");
        assert_eq!(first.task_id, "old");
        assert_eq!(first.priority, PriorityLevel::Normal);
        assert_eq!(first.age_count, 1);
    }

    #[test]
    fn test_queue_drain_overdue() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue_with_deadline("overdue", PriorityLevel::Normal, 1000, 1500);
        q.enqueue_with_deadline("ok", PriorityLevel::Normal, 1000, 3000);
        q.enqueue("no-deadline", PriorityLevel::Normal, 1000);

        let overdue = q.drain_overdue(2000);
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].task_id, "overdue");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_clear() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue("a", PriorityLevel::Normal, 1000);
        q.enqueue("b", PriorityLevel::High, 1000);
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_count_by_priority() {
        let mut q = TaskPriorityQueue::new();
        q.enqueue("a", PriorityLevel::Low, 0);
        q.enqueue("b", PriorityLevel::Normal, 0);
        q.enqueue("c", PriorityLevel::Normal, 0);
        q.enqueue("d", PriorityLevel::Critical, 0);

        let counts = q.count_by_priority();
        assert_eq!(counts[PriorityLevel::Low.weight() as usize], 1);
        assert_eq!(counts[PriorityLevel::Normal.weight() as usize], 2);
        assert_eq!(counts[PriorityLevel::High.weight() as usize], 0);
        assert_eq!(counts[PriorityLevel::Critical.weight() as usize], 1);
    }
}
