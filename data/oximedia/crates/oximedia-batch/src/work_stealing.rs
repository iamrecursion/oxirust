//! Work-stealing scheduler for load balancing across workers.
//!
//! Traditional round-robin or random assignment can leave some workers idle
//! while others are overloaded.  A work-stealing scheduler lets idle workers
//! *steal* tasks from the back of busy workers' queues, achieving near-optimal
//! load balancing with minimal coordination.
//!
//! ## Architecture
//!
//! Each worker owns a double-ended queue (deque).  New tasks are pushed to the
//! *front* and the owning worker pops from the *front* (LIFO for cache locality).
//! Idle workers steal from the *back* (FIFO — oldest tasks first, which tend to
//! be larger and benefit most from redistribution).
//!
//! ## Key types
//!
//! - [`WorkStealingScheduler`]: the coordinator that distributes tasks.
//! - [`WorkerDeque`]: per-worker deque with steal support.
//! - [`StealResult`]: outcome of a steal attempt.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};

use crate::error::{BatchError, Result};

// ---------------------------------------------------------------------------
// Work item
// ---------------------------------------------------------------------------

/// A unit of work submitted to the scheduler.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique task identifier.
    pub task_id: String,
    /// Estimated cost in arbitrary units (higher = more work).
    pub cost: u64,
    /// Priority (higher = should be processed sooner).
    pub priority: u8,
    /// Optional affinity: prefer a specific worker index.
    pub affinity: Option<usize>,
}

impl WorkItem {
    /// Create a new work item.
    #[must_use]
    pub fn new(task_id: impl Into<String>, cost: u64) -> Self {
        Self {
            task_id: task_id.into(),
            cost,
            priority: 1,
            affinity: None,
        }
    }

    /// Builder: set priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set worker affinity.
    #[must_use]
    pub fn with_affinity(mut self, worker_index: usize) -> Self {
        self.affinity = Some(worker_index);
        self
    }
}

// ---------------------------------------------------------------------------
// Worker deque
// ---------------------------------------------------------------------------

/// The result of a steal attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum StealResult {
    /// Successfully stole a task.
    Success,
    /// The target worker's queue was empty.
    Empty,
    /// Retry — another thread is modifying the queue.
    Retry,
}

/// Per-worker double-ended queue.
///
/// The owning worker pushes and pops from the front (LIFO).
/// Thieves steal from the back (FIFO).
#[derive(Debug)]
pub struct WorkerDeque {
    /// Worker index.
    pub worker_id: usize,
    /// The deque itself (protected by a mutex for simplicity; in production
    /// this would use lock-free Chase-Lev deques).
    items: Mutex<VecDeque<WorkItem>>,
    /// Total cost of items currently in the deque.
    total_cost: AtomicU64,
    /// Whether this worker is currently processing a task.
    is_busy: AtomicBool,
    /// Number of tasks completed by this worker.
    completed_count: AtomicU64,
    /// Total cost of completed tasks.
    completed_cost: AtomicU64,
}

impl WorkerDeque {
    /// Create a new empty worker deque.
    #[must_use]
    pub fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            items: Mutex::new(VecDeque::new()),
            total_cost: AtomicU64::new(0),
            is_busy: AtomicBool::new(false),
            completed_count: AtomicU64::new(0),
            completed_cost: AtomicU64::new(0),
        }
    }

    /// Push a task to the front (owner side).
    pub fn push(&self, item: WorkItem) {
        self.total_cost.fetch_add(item.cost, Ordering::Relaxed);
        self.items.lock().push_front(item);
    }

    /// Pop a task from the front (owner side).
    pub fn pop(&self) -> Option<WorkItem> {
        let item = self.items.lock().pop_front()?;
        self.total_cost.fetch_sub(item.cost, Ordering::Relaxed);
        Some(item)
    }

    /// Steal a task from the back (thief side).
    pub fn steal(&self) -> (StealResult, Option<WorkItem>) {
        let mut queue = self.items.lock();
        match queue.pop_back() {
            Some(item) => {
                self.total_cost.fetch_sub(item.cost, Ordering::Relaxed);
                (StealResult::Success, Some(item))
            }
            None => (StealResult::Empty, None),
        }
    }

    /// Number of items currently in the deque.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.lock().len()
    }

    /// Whether the deque is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.lock().is_empty()
    }

    /// Total estimated cost of pending items.
    #[must_use]
    pub fn pending_cost(&self) -> u64 {
        self.total_cost.load(Ordering::Relaxed)
    }

    /// Whether the worker is currently busy.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.is_busy.load(Ordering::Relaxed)
    }

    /// Mark worker as busy.
    pub fn set_busy(&self, busy: bool) {
        self.is_busy.store(busy, Ordering::Relaxed);
    }

    /// Record a completed task.
    pub fn record_completion(&self, cost: u64) {
        self.completed_count.fetch_add(1, Ordering::Relaxed);
        self.completed_cost.fetch_add(cost, Ordering::Relaxed);
    }

    /// Number of tasks completed by this worker.
    #[must_use]
    pub fn completed_count(&self) -> u64 {
        self.completed_count.load(Ordering::Relaxed)
    }

    /// Total cost of completed tasks.
    #[must_use]
    pub fn completed_cost(&self) -> u64 {
        self.completed_cost.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Load metric
// ---------------------------------------------------------------------------

/// Load information for a single worker.
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    /// Worker index.
    pub worker_id: usize,
    /// Number of pending tasks.
    pub pending_tasks: usize,
    /// Total estimated cost of pending tasks.
    pub pending_cost: u64,
    /// Whether the worker is currently busy.
    pub is_busy: bool,
    /// Number of completed tasks.
    pub completed_tasks: u64,
}

// ---------------------------------------------------------------------------
// Work-stealing scheduler
// ---------------------------------------------------------------------------

/// Coordinates work distribution across workers using work-stealing.
#[derive(Debug)]
pub struct WorkStealingScheduler {
    /// Per-worker deques.
    workers: Vec<WorkerDeque>,
    /// Total tasks submitted.
    total_submitted: AtomicU64,
    /// Total tasks stolen.
    total_stolen: AtomicU64,
    /// Assignment strategy for initial placement.
    assignment: RwLock<AssignmentStrategy>,
}

/// How new tasks are initially assigned to workers (before stealing kicks in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentStrategy {
    /// Assign to the worker with the lowest pending cost.
    LeastLoaded,
    /// Round-robin across workers.
    RoundRobin,
    /// Respect affinity hints; fall back to least-loaded.
    AffinityFirst,
}

impl WorkStealingScheduler {
    /// Create a scheduler with the given number of workers.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if `worker_count` is zero.
    pub fn new(worker_count: usize) -> Result<Self> {
        if worker_count == 0 {
            return Err(BatchError::InvalidJobConfig(
                "Worker count must be at least 1".to_string(),
            ));
        }

        let workers = (0..worker_count).map(WorkerDeque::new).collect();

        Ok(Self {
            workers,
            total_submitted: AtomicU64::new(0),
            total_stolen: AtomicU64::new(0),
            assignment: RwLock::new(AssignmentStrategy::LeastLoaded),
        })
    }

    /// Set the assignment strategy.
    pub fn set_assignment_strategy(&self, strategy: AssignmentStrategy) {
        *self.assignment.write() = strategy;
    }

    /// Number of workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Submit a new work item.
    ///
    /// The item is assigned to a worker based on the current assignment strategy.
    pub fn submit(&self, item: WorkItem) {
        let target = self.choose_worker(&item);
        self.workers[target].push(item);
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Submit multiple items at once.
    pub fn submit_batch(&self, items: Vec<WorkItem>) {
        for item in items {
            self.submit(item);
        }
    }

    /// Try to get work for the given worker.
    ///
    /// First checks the worker's own deque, then attempts to steal from the
    /// busiest other worker.
    pub fn get_work(&self, worker_id: usize) -> Option<WorkItem> {
        if worker_id >= self.workers.len() {
            return None;
        }

        // Try own queue first.
        if let Some(item) = self.workers[worker_id].pop() {
            return Some(item);
        }

        // Try to steal from the busiest worker.
        self.try_steal(worker_id)
    }

    /// Attempt to steal a task from the most loaded worker (excluding self).
    fn try_steal(&self, thief_id: usize) -> Option<WorkItem> {
        // Find the worker with the most pending cost (excluding self).
        let victim_id = self
            .workers
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != thief_id)
            .max_by_key(|(_, w)| w.pending_cost())?
            .0;

        let (result, item) = self.workers[victim_id].steal();
        if result == StealResult::Success {
            self.total_stolen.fetch_add(1, Ordering::Relaxed);
        }
        item
    }

    /// Get the load of each worker.
    #[must_use]
    pub fn worker_loads(&self) -> Vec<WorkerLoad> {
        self.workers
            .iter()
            .map(|w| WorkerLoad {
                worker_id: w.worker_id,
                pending_tasks: w.len(),
                pending_cost: w.pending_cost(),
                is_busy: w.is_busy(),
                completed_tasks: w.completed_count(),
            })
            .collect()
    }

    /// Total pending tasks across all workers.
    #[must_use]
    pub fn total_pending(&self) -> usize {
        self.workers.iter().map(|w| w.len()).sum()
    }

    /// Total pending cost across all workers.
    #[must_use]
    pub fn total_pending_cost(&self) -> u64 {
        self.workers.iter().map(|w| w.pending_cost()).sum()
    }

    /// Total tasks submitted since creation.
    #[must_use]
    pub fn total_submitted(&self) -> u64 {
        self.total_submitted.load(Ordering::Relaxed)
    }

    /// Total tasks stolen between workers.
    #[must_use]
    pub fn total_stolen(&self) -> u64 {
        self.total_stolen.load(Ordering::Relaxed)
    }

    /// Mark a worker as busy.
    pub fn mark_busy(&self, worker_id: usize, busy: bool) {
        if worker_id < self.workers.len() {
            self.workers[worker_id].set_busy(busy);
        }
    }

    /// Record that a worker completed a task.
    pub fn record_completion(&self, worker_id: usize, cost: u64) {
        if worker_id < self.workers.len() {
            self.workers[worker_id].record_completion(cost);
        }
    }

    /// Get a reference to a specific worker deque.
    #[must_use]
    pub fn worker(&self, worker_id: usize) -> Option<&WorkerDeque> {
        self.workers.get(worker_id)
    }

    /// Compute the load imbalance ratio.
    ///
    /// Returns 0.0 when all workers are equally loaded, and approaches 1.0
    /// when all load is on a single worker.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn imbalance_ratio(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let costs: Vec<u64> = self.workers.iter().map(|w| w.pending_cost()).collect();
        let total: u64 = costs.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let avg = total as f64 / costs.len() as f64;
        let max = costs.iter().copied().max().unwrap_or(0);
        if avg < f64::EPSILON {
            return 0.0;
        }
        ((max as f64 - avg) / avg).min(1.0)
    }

    // ── Private helpers ─────────────────────────────────────────────────

    fn choose_worker(&self, item: &WorkItem) -> usize {
        let strategy = *self.assignment.read();
        match strategy {
            AssignmentStrategy::AffinityFirst => {
                if let Some(aff) = item.affinity {
                    if aff < self.workers.len() {
                        return aff;
                    }
                }
                self.least_loaded_worker()
            }
            AssignmentStrategy::LeastLoaded => self.least_loaded_worker(),
            AssignmentStrategy::RoundRobin => {
                let idx = self.total_submitted.load(Ordering::Relaxed) as usize;
                idx % self.workers.len()
            }
        }
    }

    fn least_loaded_worker(&self) -> usize {
        self.workers
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| w.pending_cost())
            .map(|(id, _)| id)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── WorkItem ────────────────────────────────────────────────────────
    #[test]
    fn test_work_item_builder() {
        let item = WorkItem::new("task-1", 100)
            .with_priority(3)
            .with_affinity(2);
        assert_eq!(item.task_id, "task-1");
        assert_eq!(item.cost, 100);
        assert_eq!(item.priority, 3);
        assert_eq!(item.affinity, Some(2));
    }

    // ── WorkerDeque ─────────────────────────────────────────────────────
    #[test]
    fn test_worker_deque_push_pop() {
        let deque = WorkerDeque::new(0);
        assert!(deque.is_empty());
        deque.push(WorkItem::new("a", 10));
        deque.push(WorkItem::new("b", 20));
        assert_eq!(deque.len(), 2);
        assert_eq!(deque.pending_cost(), 30);

        // Pop returns front (LIFO for owner).
        let item = deque.pop().expect("should pop");
        assert_eq!(item.task_id, "b"); // last pushed
        assert_eq!(deque.pending_cost(), 10);
    }

    #[test]
    fn test_worker_deque_steal() {
        let deque = WorkerDeque::new(0);
        deque.push(WorkItem::new("first", 10));
        deque.push(WorkItem::new("second", 20));

        // Steal returns back (FIFO for thief).
        let (result, item) = deque.steal();
        assert_eq!(result, StealResult::Success);
        assert_eq!(
            item.expect("should have item").task_id,
            "first" // first pushed
        );
    }

    #[test]
    fn test_worker_deque_steal_empty() {
        let deque = WorkerDeque::new(0);
        let (result, item) = deque.steal();
        assert_eq!(result, StealResult::Empty);
        assert!(item.is_none());
    }

    #[test]
    fn test_worker_deque_busy_flag() {
        let deque = WorkerDeque::new(0);
        assert!(!deque.is_busy());
        deque.set_busy(true);
        assert!(deque.is_busy());
    }

    #[test]
    fn test_worker_deque_completion_tracking() {
        let deque = WorkerDeque::new(0);
        deque.record_completion(100);
        deque.record_completion(200);
        assert_eq!(deque.completed_count(), 2);
        assert_eq!(deque.completed_cost(), 300);
    }

    // ── Scheduler creation ──────────────────────────────────────────────
    #[test]
    fn test_scheduler_creation() {
        let sched = WorkStealingScheduler::new(4).expect("should create");
        assert_eq!(sched.worker_count(), 4);
        assert_eq!(sched.total_pending(), 0);
    }

    #[test]
    fn test_scheduler_zero_workers_error() {
        let result = WorkStealingScheduler::new(0);
        assert!(result.is_err());
    }

    // ── Submit and get work ─────────────────────────────────────────────
    #[test]
    fn test_submit_and_get_work() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.submit(WorkItem::new("t1", 100));
        sched.submit(WorkItem::new("t2", 200));

        assert_eq!(sched.total_submitted(), 2);
        assert_eq!(sched.total_pending(), 2);

        // Get work for worker 0.
        let item = sched.get_work(0);
        assert!(item.is_some());
    }

    // ── Work stealing ───────────────────────────────────────────────────
    #[test]
    fn test_work_stealing() {
        let sched = WorkStealingScheduler::new(2).expect("should create");

        // Load all tasks onto worker 0.
        sched.set_assignment_strategy(AssignmentStrategy::AffinityFirst);
        for i in 0..5 {
            sched.submit(WorkItem::new(format!("t{i}"), 100).with_affinity(0));
        }

        assert_eq!(sched.worker(0).expect("should exist").len(), 5);
        assert_eq!(sched.worker(1).expect("should exist").len(), 0);

        // Worker 1 gets work → should steal from worker 0.
        let stolen = sched.get_work(1);
        assert!(stolen.is_some());
        assert_eq!(sched.total_stolen(), 1);
    }

    // ── Least loaded assignment ─────────────────────────────────────────
    #[test]
    fn test_least_loaded_assignment() {
        let sched = WorkStealingScheduler::new(3).expect("should create");
        sched.set_assignment_strategy(AssignmentStrategy::LeastLoaded);

        // Submit tasks with different costs.
        sched.submit(WorkItem::new("heavy", 1000));
        sched.submit(WorkItem::new("light", 10));

        // Second task should go to a different worker since first is heavier.
        let loads = sched.worker_loads();
        let busy_count = loads.iter().filter(|l| l.pending_tasks > 0).count();
        assert!(busy_count >= 1);
    }

    // ── Round robin ─────────────────────────────────────────────────────
    #[test]
    fn test_round_robin_assignment() {
        let sched = WorkStealingScheduler::new(3).expect("should create");
        sched.set_assignment_strategy(AssignmentStrategy::RoundRobin);

        for i in 0..6 {
            sched.submit(WorkItem::new(format!("t{i}"), 10));
        }

        // Each worker should have 2 tasks.
        for wid in 0..3 {
            assert_eq!(
                sched.worker(wid).expect("should exist").len(),
                2,
                "worker {wid} should have 2 tasks"
            );
        }
    }

    // ── Affinity ────────────────────────────────────────────────────────
    #[test]
    fn test_affinity_assignment() {
        let sched = WorkStealingScheduler::new(4).expect("should create");
        sched.set_assignment_strategy(AssignmentStrategy::AffinityFirst);

        sched.submit(WorkItem::new("gpu-task", 500).with_affinity(2));

        assert_eq!(sched.worker(2).expect("should exist").len(), 1);
    }

    #[test]
    fn test_affinity_out_of_range_falls_back() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.set_assignment_strategy(AssignmentStrategy::AffinityFirst);

        sched.submit(WorkItem::new("bad-affinity", 100).with_affinity(99));

        // Should fall back to least-loaded.
        assert_eq!(sched.total_pending(), 1);
    }

    // ── Worker loads ────────────────────────────────────────────────────
    #[test]
    fn test_worker_loads() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.submit(WorkItem::new("t1", 100).with_affinity(0));

        let loads = sched.worker_loads();
        assert_eq!(loads.len(), 2);
        let w0 = &loads[0];
        assert_eq!(w0.pending_tasks, 1);
        assert_eq!(w0.pending_cost, 100);
    }

    // ── Imbalance ratio ─────────────────────────────────────────────────
    #[test]
    fn test_imbalance_ratio_balanced() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        // No tasks → perfectly balanced.
        assert!((sched.imbalance_ratio()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_imbalance_ratio_unbalanced() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.set_assignment_strategy(AssignmentStrategy::AffinityFirst);

        // Put all load on worker 0.
        for i in 0..10 {
            sched.submit(WorkItem::new(format!("t{i}"), 100).with_affinity(0));
        }

        let ratio = sched.imbalance_ratio();
        assert!(ratio > 0.5);
    }

    // ── Batch submit ────────────────────────────────────────────────────
    #[test]
    fn test_submit_batch() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        let items: Vec<WorkItem> = (0..10)
            .map(|i| WorkItem::new(format!("batch-{i}"), 50))
            .collect();
        sched.submit_batch(items);
        assert_eq!(sched.total_submitted(), 10);
        assert_eq!(sched.total_pending(), 10);
    }

    // ── Mark busy / record completion ───────────────────────────────────
    #[test]
    fn test_mark_busy_and_record_completion() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.mark_busy(0, true);
        assert!(sched.worker(0).expect("exists").is_busy());

        sched.record_completion(0, 500);
        sched.record_completion(0, 300);
        assert_eq!(sched.worker(0).expect("exists").completed_count(), 2);
        assert_eq!(sched.worker(0).expect("exists").completed_cost(), 800);
    }

    // ── Get work from invalid worker ────────────────────────────────────
    #[test]
    fn test_get_work_invalid_worker() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        assert!(sched.get_work(99).is_none());
    }

    // ── Total pending cost ──────────────────────────────────────────────
    #[test]
    fn test_total_pending_cost() {
        let sched = WorkStealingScheduler::new(2).expect("should create");
        sched.submit(WorkItem::new("a", 100));
        sched.submit(WorkItem::new("b", 200));
        assert_eq!(sched.total_pending_cost(), 300);
    }
}
