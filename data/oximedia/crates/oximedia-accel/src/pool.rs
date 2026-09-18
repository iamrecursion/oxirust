//! Thread-pool abstraction for accelerated work items.
//!
//! `AccelPool` provides a simple, synchronous work queue that processes items
//! in priority order and tracks per-pool statistics.  The implementation is
//! intentionally single-threaded and allocation-light so that it compiles
//! without additional runtime dependencies.

#![allow(dead_code)]

use std::collections::BinaryHeap;
use std::time::Instant;

/// An item of work submitted to the pool.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique identifier assigned by the caller.
    pub id: u64,
    /// Priority (higher value = higher priority).
    pub priority: i32,
    /// Opaque payload.
    pub data: Vec<u8>,
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for WorkItem {}

impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority comes first; break ties by lower id.
        self.priority
            .cmp(&other.priority)
            .then(other.id.cmp(&self.id))
    }
}

/// The result of processing a [`WorkItem`].
#[derive(Debug, Clone)]
pub struct WorkResult {
    /// Matches the `id` of the originating [`WorkItem`].
    pub id: u64,
    /// Processed output bytes.
    pub output: Vec<u8>,
    /// Wall-clock processing time in milliseconds.
    pub elapsed_ms: u64,
}

/// Aggregate statistics for an [`AccelPool`].
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total items submitted.
    pub submitted: u64,
    /// Total items successfully completed.
    pub completed: u64,
    /// Total items that failed processing.
    pub failed: u64,
    /// Average processing latency in milliseconds.
    pub avg_latency_ms: f64,
}

/// A pool that schedules and processes accelerated work items.
///
/// Items are processed synchronously inside [`AccelPool::drain_completed`].
pub struct AccelPool {
    /// Maximum number of worker threads (informational; pool is sync).
    workers: usize,
    /// Maximum pending items before back-pressure.
    queue_depth: usize,
    /// Pending work ordered by priority.
    queue: BinaryHeap<WorkItem>,
    /// Completed results waiting to be drained.
    completed: Vec<WorkResult>,
    /// Running statistics.
    stats: PoolStats,
    /// Sum of all latencies for average calculation.
    total_latency_ms: u64,
}

impl AccelPool {
    /// Create a new pool with `workers` logical workers and a `queue_depth`
    /// soft limit on pending items.
    #[must_use]
    pub fn new(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
            queue_depth: workers.max(1) * 64,
            queue: BinaryHeap::new(),
            completed: Vec::new(),
            stats: PoolStats::default(),
            total_latency_ms: 0,
        }
    }

    /// Submit a work item.  Returns the item's `id` for tracking.
    ///
    /// Items beyond `queue_depth` are still accepted (soft limit only).
    pub fn submit(&mut self, item: WorkItem) -> u64 {
        let id = item.id;
        self.stats.submitted += 1;
        self.queue.push(item);
        id
    }

    /// Process all queued items and return completed results.
    ///
    /// Each item's payload is passed through a simple identity transform
    /// (copy) to simulate real work.
    pub fn drain_completed(&mut self) -> Vec<WorkResult> {
        while let Some(item) = self.queue.pop() {
            let start = Instant::now();
            // Simulate processing: reverse the data bytes.
            let output: Vec<u8> = item.data.iter().copied().rev().collect();
            let elapsed_ms = start.elapsed().as_millis() as u64;

            self.total_latency_ms += elapsed_ms;
            self.stats.completed += 1;

            self.completed.push(WorkResult {
                id: item.id,
                output,
                elapsed_ms,
            });
        }

        // Update average latency.
        if self.stats.completed > 0 {
            self.stats.avg_latency_ms = self.total_latency_ms as f64 / self.stats.completed as f64;
        }

        std::mem::take(&mut self.completed)
    }

    /// Return the number of currently queued (unprocessed) items.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Return a rough utilization ratio (queued / capacity).
    ///
    /// Returns a value in [0.0, 1.0]; values > 1.0 indicate back-pressure.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.queue_depth == 0 {
            return 0.0;
        }
        self.queue.len() as f64 / self.queue_depth as f64
    }

    /// Return a snapshot of the current pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    /// Return the configured worker count.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Return the configured queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.queue_depth
    }
}

impl Default for AccelPool {
    fn default() -> Self {
        Self::new(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u64, priority: i32, data: Vec<u8>) -> WorkItem {
        WorkItem { id, priority, data }
    }

    #[test]
    fn test_pool_new() {
        let pool = AccelPool::new(4);
        assert_eq!(pool.workers(), 4);
        assert_eq!(pool.pending(), 0);
    }

    #[test]
    fn test_submit_increments_pending() {
        let mut pool = AccelPool::new(2);
        pool.submit(make_item(1, 0, vec![]));
        assert_eq!(pool.pending(), 1);
    }

    #[test]
    fn test_drain_clears_queue() {
        let mut pool = AccelPool::new(2);
        pool.submit(make_item(1, 0, vec![1, 2, 3]));
        pool.submit(make_item(2, 0, vec![4, 5]));
        let results = pool.drain_completed();
        assert_eq!(results.len(), 2);
        assert_eq!(pool.pending(), 0);
    }

    #[test]
    fn test_drain_reverses_data() {
        let mut pool = AccelPool::new(1);
        pool.submit(make_item(42, 0, vec![1, 2, 3]));
        let results = pool.drain_completed();
        assert_eq!(results[0].output, vec![3, 2, 1]);
    }

    #[test]
    fn test_priority_ordering() {
        let mut pool = AccelPool::new(2);
        pool.submit(make_item(1, 0, vec![0]));
        pool.submit(make_item(2, 10, vec![1])); // higher priority
        pool.submit(make_item(3, 5, vec![2]));
        let results = pool.drain_completed();
        // Highest priority item should be first.
        assert_eq!(results[0].id, 2);
        assert_eq!(results[1].id, 3);
        assert_eq!(results[2].id, 1);
    }

    #[test]
    fn test_stats_submitted() {
        let mut pool = AccelPool::new(1);
        pool.submit(make_item(1, 0, vec![]));
        pool.submit(make_item(2, 0, vec![]));
        assert_eq!(pool.stats().submitted, 2);
    }

    #[test]
    fn test_stats_completed_after_drain() {
        let mut pool = AccelPool::new(1);
        pool.submit(make_item(1, 0, vec![]));
        pool.drain_completed();
        assert_eq!(pool.stats().completed, 1);
        assert_eq!(pool.stats().failed, 0);
    }

    #[test]
    fn test_utilization_empty() {
        let pool = AccelPool::new(4);
        assert!((pool.utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_utilization_grows() {
        let mut pool = AccelPool::new(1);
        // queue_depth = 1*64 = 64
        for i in 0..32 {
            pool.submit(make_item(i, 0, vec![]));
        }
        assert!(pool.utilization() > 0.0);
    }

    #[test]
    fn test_default_pool() {
        let pool = AccelPool::default();
        assert_eq!(pool.workers(), 4);
    }

    #[test]
    fn test_drain_twice_returns_only_new_items() {
        let mut pool = AccelPool::new(1);
        pool.submit(make_item(1, 0, vec![]));
        let first = pool.drain_completed();
        assert_eq!(first.len(), 1);
        pool.submit(make_item(2, 0, vec![]));
        let second = pool.drain_completed();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].id, 2);
    }
}
