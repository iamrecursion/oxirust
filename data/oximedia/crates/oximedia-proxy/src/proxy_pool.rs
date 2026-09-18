#![allow(dead_code)]
//! Worker pool management for parallel proxy generation.
//!
//! This module models a fixed-size pool of transcoding workers that can be
//! assigned proxy generation jobs. It tracks worker state, queues overflow
//! jobs, and reports utilization statistics. The design is purely in-memory
//! and does not spawn real OS threads — it is a scheduling data structure
//! suitable for driving an external executor.

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Current state of a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Idle and ready to accept a job.
    Idle,
    /// Currently processing a job.
    Busy,
    /// Drained — will not accept new jobs.
    Drained,
}

/// Identifies a single worker in the pool.
#[derive(Debug, Clone)]
pub struct Worker {
    /// Zero-based index.
    pub index: usize,
    /// Current state.
    pub state: WorkerState,
    /// Number of jobs completed by this worker.
    pub completed_jobs: u64,
    /// Cumulative processing time (estimated).
    pub total_work_time: Duration,
    /// Current job ID (if busy).
    current_job: Option<u64>,
}

impl Worker {
    /// Create a new idle worker.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            state: WorkerState::Idle,
            completed_jobs: 0,
            total_work_time: Duration::ZERO,
            current_job: None,
        }
    }

    /// Assign a job to this worker.
    pub fn assign(&mut self, job_id: u64) -> bool {
        if self.state != WorkerState::Idle {
            return false;
        }
        self.state = WorkerState::Busy;
        self.current_job = Some(job_id);
        true
    }

    /// Mark the current job as complete.
    pub fn complete(&mut self, elapsed: Duration) {
        if self.state == WorkerState::Busy {
            self.state = WorkerState::Idle;
            self.current_job = None;
            self.completed_jobs += 1;
            self.total_work_time += elapsed;
        }
    }

    /// Drain this worker so it will not accept new jobs.
    pub fn drain(&mut self) {
        if self.state == WorkerState::Idle {
            self.state = WorkerState::Drained;
        }
    }

    /// Return the current job ID, if any.
    pub fn current_job(&self) -> Option<u64> {
        self.current_job
    }

    /// Return `true` when the worker is idle.
    pub fn is_idle(&self) -> bool {
        self.state == WorkerState::Idle
    }
}

/// A pending proxy generation job.
#[derive(Debug, Clone)]
pub struct ProxyJob {
    /// Unique job identifier.
    pub id: u64,
    /// Source file path or asset ID.
    pub source: String,
    /// Target proxy path or asset ID.
    pub target: String,
    /// Priority (lower is higher priority).
    pub priority: u32,
}

impl ProxyJob {
    /// Create a new job.
    pub fn new(id: u64, source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            id,
            source: source.into(),
            target: target.into(),
            priority: 100,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// A fixed-capacity pool of proxy transcoding workers.
#[derive(Debug)]
pub struct ProxyWorkerPool {
    /// Workers in the pool.
    workers: Vec<Worker>,
    /// Pending job queue.
    queue: VecDeque<ProxyJob>,
    /// Maximum queue depth (0 = unlimited).
    max_queue: usize,
    /// Total jobs submitted.
    total_submitted: u64,
    /// Total jobs completed.
    total_completed: u64,
    /// Total jobs rejected (queue full).
    total_rejected: u64,
    /// Set to `true` after `drain_all()` — no new jobs are accepted.
    draining: bool,
}

impl ProxyWorkerPool {
    /// Create a pool with the given number of workers.
    pub fn new(worker_count: usize) -> Self {
        let workers = (0..worker_count).map(Worker::new).collect();
        Self {
            workers,
            queue: VecDeque::new(),
            max_queue: 0,
            total_submitted: 0,
            total_completed: 0,
            total_rejected: 0,
            draining: false,
        }
    }

    /// Set maximum queue depth.
    pub fn with_max_queue(mut self, max: usize) -> Self {
        self.max_queue = max;
        self
    }

    /// Submit a job. Returns `true` if accepted (assigned or queued).
    ///
    /// Returns `false` if the pool is draining (after [`drain_all`](Self::drain_all))
    /// or if the queue is at maximum capacity.
    pub fn submit(&mut self, job: ProxyJob) -> bool {
        // Refuse new submissions while draining.
        if self.draining {
            self.total_rejected += 1;
            return false;
        }

        self.total_submitted += 1;

        // Try to assign directly to an idle worker
        if let Some(worker) = self.workers.iter_mut().find(|w| w.is_idle()) {
            worker.assign(job.id);
            return true;
        }

        // Otherwise queue
        if self.max_queue > 0 && self.queue.len() >= self.max_queue {
            self.total_rejected += 1;
            return false;
        }
        self.queue.push_back(job);
        true
    }

    /// Mark a worker's current job as complete and try to assign the next
    /// queued job. Returns the completed job ID if successful.
    pub fn complete_job(&mut self, worker_index: usize, elapsed: Duration) -> Option<u64> {
        if worker_index >= self.workers.len() {
            return None;
        }
        let job_id = self.workers[worker_index].current_job();
        self.workers[worker_index].complete(elapsed);
        self.total_completed += 1;

        // Try to assign next queued job
        if let Some(next_job) = self.queue.pop_front() {
            self.workers[worker_index].assign(next_job.id);
        }
        job_id
    }

    /// Number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of currently idle workers.
    pub fn idle_count(&self) -> usize {
        self.workers.iter().filter(|w| w.is_idle()).count()
    }

    /// Number of currently busy workers.
    pub fn busy_count(&self) -> usize {
        self.workers
            .iter()
            .filter(|w| w.state == WorkerState::Busy)
            .count()
    }

    /// Number of jobs waiting in the queue.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Pool utilization as a ratio (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        self.busy_count() as f64 / self.workers.len() as f64
    }

    /// Total submitted jobs.
    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    /// Total completed jobs.
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Total rejected jobs (due to full queue).
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    /// Drain all idle workers so no new jobs are accepted.
    pub fn drain_all(&mut self) {
        for w in &mut self.workers {
            w.drain();
        }
        // After draining, seal the queue so no new jobs are accepted.
        self.queue.clear();
        self.draining = true;
    }

    /// Get a reference to a worker by index.
    pub fn worker(&self, index: usize) -> Option<&Worker> {
        self.workers.get(index)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_new_is_idle() {
        let w = Worker::new(0);
        assert!(w.is_idle());
        assert_eq!(w.state, WorkerState::Idle);
        assert_eq!(w.completed_jobs, 0);
    }

    #[test]
    fn test_worker_assign_and_complete() {
        let mut w = Worker::new(0);
        assert!(w.assign(1));
        assert_eq!(w.state, WorkerState::Busy);
        assert_eq!(w.current_job(), Some(1));
        w.complete(Duration::from_secs(5));
        assert!(w.is_idle());
        assert_eq!(w.completed_jobs, 1);
        assert_eq!(w.total_work_time, Duration::from_secs(5));
    }

    #[test]
    fn test_worker_cannot_assign_when_busy() {
        let mut w = Worker::new(0);
        w.assign(1);
        assert!(!w.assign(2));
    }

    #[test]
    fn test_worker_drain() {
        let mut w = Worker::new(0);
        w.drain();
        assert_eq!(w.state, WorkerState::Drained);
        assert!(!w.assign(1)); // cannot assign to drained
    }

    #[test]
    fn test_pool_submit_assigns_idle() {
        let mut pool = ProxyWorkerPool::new(2);
        assert!(pool.submit(ProxyJob::new(1, "a.mov", "a_proxy.mp4")));
        assert_eq!(pool.busy_count(), 1);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_pool_submit_queues_when_full() {
        let mut pool = ProxyWorkerPool::new(1);
        pool.submit(ProxyJob::new(1, "a", "ap"));
        assert!(pool.submit(ProxyJob::new(2, "b", "bp"))); // queued
        assert_eq!(pool.queued_count(), 1);
    }

    #[test]
    fn test_pool_rejects_when_queue_full() {
        let mut pool = ProxyWorkerPool::new(1).with_max_queue(1);
        pool.submit(ProxyJob::new(1, "a", "ap")); // assigned
        pool.submit(ProxyJob::new(2, "b", "bp")); // queued
        assert!(!pool.submit(ProxyJob::new(3, "c", "cp"))); // rejected
        assert_eq!(pool.total_rejected(), 1);
    }

    #[test]
    fn test_pool_complete_dequeues_next() {
        let mut pool = ProxyWorkerPool::new(1);
        pool.submit(ProxyJob::new(1, "a", "ap"));
        pool.submit(ProxyJob::new(2, "b", "bp"));
        let done = pool.complete_job(0, Duration::from_secs(1));
        assert_eq!(done, Some(1));
        // Worker should now be busy with job 2
        assert_eq!(
            pool.worker(0)
                .expect("should succeed in test")
                .current_job(),
            Some(2)
        );
    }

    #[test]
    fn test_pool_utilization() {
        let mut pool = ProxyWorkerPool::new(4);
        assert!((pool.utilization() - 0.0).abs() < f64::EPSILON);
        pool.submit(ProxyJob::new(1, "a", "ap"));
        pool.submit(ProxyJob::new(2, "b", "bp"));
        assert!((pool.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pool_drain_all() {
        let mut pool = ProxyWorkerPool::new(2);
        pool.drain_all();
        assert!(!pool.submit(ProxyJob::new(1, "a", "ap")));
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = ProxyWorkerPool::new(1);
        pool.submit(ProxyJob::new(1, "a", "ap"));
        pool.complete_job(0, Duration::from_millis(100));
        assert_eq!(pool.total_submitted(), 1);
        assert_eq!(pool.total_completed(), 1);
    }

    #[test]
    fn test_proxy_job_priority() {
        let job = ProxyJob::new(1, "src", "dst").with_priority(10);
        assert_eq!(job.priority, 10);
    }

    #[test]
    fn test_pool_empty_utilization() {
        let pool = ProxyWorkerPool::new(0);
        assert!((pool.utilization() - 0.0).abs() < f64::EPSILON);
    }
}
