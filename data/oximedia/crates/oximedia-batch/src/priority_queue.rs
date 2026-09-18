//! Priority-aware job queue backed by a binary max-heap.
//!
//! [`PriorityJobQueue`] provides FIFO ordering within the same priority tier,
//! plus in-place `promote` and `remove` operations.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A scored job entry for the priority queue.
///
/// Priority levels: 0 = Low, 1 = Normal, 2 = High, 3 = Critical.
#[derive(Debug, Clone)]
pub struct PriorityEntry {
    /// Unique job identifier.
    pub job_id: String,
    /// Priority level: 0 = Low, 1 = Normal, 2 = High, 3 = Critical.
    pub priority: u8,
    /// Unix timestamp (seconds since epoch) used for FIFO tie-breaking.
    pub submit_time: u64,
    /// Estimated execution duration in seconds.
    pub estimated_duration_secs: u64,
    /// Combined CPU + memory resource weight in the range `0.0..=1.0`.
    pub resource_weight: f32,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submit_time == other.submit_time
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Custom `Ord`: higher priority comes first; ties are broken by *earlier*
/// `submit_time` (FIFO — smaller timestamp wins, i.e., is "greater" in the heap).
impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Smaller submit_time means the job arrived earlier → it should
                // be dequeued first, so it is "greater" in the max-heap.
                other.submit_time.cmp(&self.submit_time)
            }
            non_equal => non_equal,
        }
    }
}

/// A bounded priority job queue backed by a `BinaryHeap`.
///
/// Jobs with higher `priority` values are dequeued first.  Within the same
/// priority tier jobs are served in FIFO order (earliest `submit_time` first).
pub struct PriorityJobQueue {
    heap: BinaryHeap<PriorityEntry>,
    max_size: usize,
}

impl PriorityJobQueue {
    /// Create a new queue with the given capacity cap.
    ///
    /// `max_size == 0` means unlimited.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            max_size,
        }
    }

    /// Push a new entry.
    ///
    /// # Errors
    ///
    /// Returns `Err("queue full")` when the queue has reached `max_size` (and
    /// `max_size > 0`).
    pub fn push(&mut self, entry: PriorityEntry) -> Result<(), &'static str> {
        if self.max_size > 0 && self.heap.len() >= self.max_size {
            return Err("queue full");
        }
        self.heap.push(entry);
        Ok(())
    }

    /// Remove and return the highest-priority entry, or `None` if empty.
    pub fn pop(&mut self) -> Option<PriorityEntry> {
        self.heap.pop()
    }

    /// Return a reference to the highest-priority entry without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PriorityEntry> {
        self.heap.peek()
    }

    /// Number of entries currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// `true` when the queue contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain all entries and return them sorted from highest to lowest priority
    /// (FIFO within the same tier).
    pub fn drain_by_priority(&mut self) -> Vec<PriorityEntry> {
        let mut all: Vec<PriorityEntry> = self.heap.drain().collect();
        // `BinaryHeap::drain` does not guarantee order; sort explicitly.
        all.sort_unstable_by(|a, b| b.cmp(a));
        all
    }

    /// Change the priority of a job already in the queue.
    ///
    /// Returns `true` if the job was found and updated, `false` otherwise.
    ///
    /// The implementation rebuilds the heap after mutation, which is O(n).
    pub fn promote(&mut self, job_id: &str, new_priority: u8) -> bool {
        let mut found = false;
        let mut entries: Vec<PriorityEntry> = self.heap.drain().collect();
        for entry in &mut entries {
            if entry.job_id == job_id {
                entry.priority = new_priority;
                found = true;
                break;
            }
        }
        for entry in entries {
            self.heap.push(entry);
        }
        found
    }

    /// Remove a specific job by ID.
    ///
    /// Returns the entry if it was present, otherwise `None`.
    pub fn remove(&mut self, job_id: &str) -> Option<PriorityEntry> {
        let mut entries: Vec<PriorityEntry> = self.heap.drain().collect();
        let pos = entries.iter().position(|e| e.job_id == job_id);
        match pos {
            None => {
                // Put them all back.
                for entry in entries {
                    self.heap.push(entry);
                }
                None
            }
            Some(idx) => {
                let removed = entries.swap_remove(idx);
                for entry in entries {
                    self.heap.push(entry);
                }
                Some(removed)
            }
        }
    }

    /// Return references to all entries sorted from highest to lowest priority
    /// without consuming the queue.
    ///
    /// This is O(n log n) because the heap's internal storage is unordered.
    #[must_use]
    pub fn iter_sorted(&self) -> Vec<&PriorityEntry> {
        let mut refs: Vec<&PriorityEntry> = self.heap.iter().collect();
        refs.sort_unstable_by(|a, b| b.cmp(a));
        refs
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn make_entry(job_id: &str, priority: u8, submit_time: u64) -> PriorityEntry {
    PriorityEntry {
        job_id: job_id.to_string(),
        priority,
        submit_time,
        estimated_duration_secs: 0,
        resource_weight: 0.0,
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_queue_is_empty() {
        let q = PriorityJobQueue::new(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_push_and_pop_single_entry() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("a", 1, 100))
            .expect("push should succeed");
        assert_eq!(q.len(), 1);
        let e = q.pop().expect("pop should return Some");
        assert_eq!(e.job_id, "a");
        assert!(q.is_empty());
    }

    #[test]
    fn test_higher_priority_dequeued_first() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("low", 0, 1))
            .expect("push should succeed");
        q.push(make_entry("high", 3, 2))
            .expect("push should succeed");
        q.push(make_entry("normal", 1, 3))
            .expect("push should succeed");
        let first = q.pop().expect("pop should return Some");
        assert_eq!(first.job_id, "high");
    }

    #[test]
    fn test_fifo_tiebreak_same_priority() {
        let mut q = PriorityJobQueue::new(10);
        // Same priority, earlier submit_time should come out first.
        q.push(make_entry("second", 2, 200))
            .expect("push should succeed");
        q.push(make_entry("first", 2, 100))
            .expect("push should succeed");
        q.push(make_entry("third", 2, 300))
            .expect("push should succeed");
        assert_eq!(q.pop().expect("pop failed").job_id, "first");
        assert_eq!(q.pop().expect("pop failed").job_id, "second");
        assert_eq!(q.pop().expect("pop failed").job_id, "third");
    }

    #[test]
    fn test_pop_returns_none_on_empty() {
        let mut q = PriorityJobQueue::new(5);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("x", 2, 10)).expect("push should succeed");
        let _ = q.peek().expect("peek should return Some");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_full_returns_error() {
        let mut q = PriorityJobQueue::new(2);
        q.push(make_entry("a", 1, 1))
            .expect("push 1 should succeed");
        q.push(make_entry("b", 1, 2))
            .expect("push 2 should succeed");
        let result = q.push(make_entry("c", 1, 3));
        assert!(result.is_err());
    }

    #[test]
    fn test_unlimited_queue_never_full() {
        let mut q = PriorityJobQueue::new(0); // 0 = unlimited
        for i in 0u64..1000 {
            q.push(make_entry(&i.to_string(), 1, i))
                .expect("push should never fail");
        }
        assert_eq!(q.len(), 1000);
    }

    #[test]
    fn test_drain_by_priority_order() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("c", 0, 1)).expect("push should succeed");
        q.push(make_entry("a", 3, 2)).expect("push should succeed");
        q.push(make_entry("b", 1, 3)).expect("push should succeed");
        let drained = q.drain_by_priority();
        assert_eq!(drained[0].job_id, "a");
        assert_eq!(drained[1].job_id, "b");
        assert_eq!(drained[2].job_id, "c");
        assert!(q.is_empty());
    }

    #[test]
    fn test_promote_increases_priority() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("slow", 0, 1))
            .expect("push should succeed");
        q.push(make_entry("fast", 3, 2))
            .expect("push should succeed");
        // Promote "slow" to Critical so it matches "fast"
        let found = q.promote("slow", 3);
        assert!(found);
        // Both are now priority 3; "slow" submitted earlier (t=1) → comes first.
        let first = q.pop().expect("pop should return Some");
        assert_eq!(first.job_id, "slow");
    }

    #[test]
    fn test_promote_non_existent_job_returns_false() {
        let mut q = PriorityJobQueue::new(10);
        assert!(!q.promote("ghost", 3));
    }

    #[test]
    fn test_remove_existing_job() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("keep", 1, 1))
            .expect("push should succeed");
        q.push(make_entry("drop", 2, 2))
            .expect("push should succeed");
        let removed = q.remove("drop");
        assert!(removed.is_some());
        assert_eq!(removed.expect("should be Some").job_id, "drop");
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop().expect("pop should return Some").job_id, "keep");
    }

    #[test]
    fn test_remove_non_existent_returns_none() {
        let mut q = PriorityJobQueue::new(10);
        assert!(q.remove("no-such-job").is_none());
    }

    #[test]
    fn test_iter_sorted_does_not_consume() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("b", 1, 1)).expect("push should succeed");
        q.push(make_entry("a", 3, 2)).expect("push should succeed");
        let sorted = q.iter_sorted();
        assert_eq!(sorted[0].job_id, "a");
        assert_eq!(sorted[1].job_id, "b");
        // Queue still intact.
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_ordering_critical_beats_all() {
        let mut q = PriorityJobQueue::new(10);
        for p in [0u8, 1, 2] {
            q.push(make_entry(&p.to_string(), p, p as u64))
                .expect("push should succeed");
        }
        q.push(make_entry("crit", 3, 99))
            .expect("push should succeed");
        assert_eq!(q.pop().expect("pop failed").job_id, "crit");
    }

    #[test]
    fn test_promote_then_drain_order() {
        let mut q = PriorityJobQueue::new(10);
        q.push(make_entry("alpha", 0, 10))
            .expect("push should succeed");
        q.push(make_entry("beta", 1, 20))
            .expect("push should succeed");
        q.push(make_entry("gamma", 2, 30))
            .expect("push should succeed");
        // Promote alpha to priority 2 — same as gamma but earlier submit time.
        q.promote("alpha", 2);
        let drained = q.drain_by_priority();
        // alpha (priority 2, t=10) before gamma (priority 2, t=30)
        assert_eq!(drained[0].job_id, "alpha");
        assert_eq!(drained[1].job_id, "gamma");
        assert_eq!(drained[2].job_id, "beta");
    }
}
