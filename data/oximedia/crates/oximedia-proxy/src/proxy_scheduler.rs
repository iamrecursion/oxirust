//! Proxy task scheduler for managing concurrent proxy generation jobs.
#![allow(dead_code)]

use std::collections::VecDeque;

/// Configuration for the proxy scheduler.
#[derive(Debug, Clone)]
pub struct ProxySchedulerConfig {
    /// Maximum number of concurrent proxy generation tasks.
    pub max_concurrent: usize,
    /// Maximum queue depth before new tasks are rejected.
    pub max_queue_depth: usize,
    /// Default priority for tasks that don't specify one.
    pub default_priority: u8,
}

impl ProxySchedulerConfig {
    /// Create a new scheduler config.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            max_queue_depth: 256,
            default_priority: 50,
        }
    }

    /// Return the maximum number of concurrent tasks.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

impl Default for ProxySchedulerConfig {
    fn default() -> Self {
        Self::new(4)
    }
}

/// A single proxy generation task.
#[derive(Debug, Clone)]
pub struct ProxyTask {
    /// Unique task identifier.
    pub id: u64,
    /// Source file path.
    pub source_path: String,
    /// Destination file path.
    pub dest_path: String,
    /// Estimated duration in seconds.
    pub estimated_secs: f64,
    /// Task priority (0 = lowest, 255 = highest).
    pub priority: u8,
}

impl ProxyTask {
    /// Create a new proxy task.
    pub fn new(id: u64, source_path: &str, dest_path: &str, estimated_secs: f64) -> Self {
        Self {
            id,
            source_path: source_path.to_owned(),
            dest_path: dest_path.to_owned(),
            estimated_secs,
            priority: 50,
        }
    }

    /// Return the estimated processing time in seconds.
    pub fn estimated_secs(&self) -> f64 {
        self.estimated_secs
    }

    /// Set task priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Statistics collected by the proxy scheduler.
#[derive(Debug, Clone, Default)]
pub struct ProxySchedulerStats {
    /// Total tasks completed.
    pub completed: u64,
    /// Total wall-clock seconds elapsed since creation.
    pub elapsed_secs: f64,
    /// Total tasks that failed.
    pub failed: u64,
}

impl ProxySchedulerStats {
    /// Calculate approximate throughput as tasks per hour.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_per_hour(&self) -> f64 {
        if self.elapsed_secs <= 0.0 {
            return 0.0;
        }
        (self.completed as f64 / self.elapsed_secs) * 3600.0
    }

    /// Return total tasks processed (completed + failed).
    pub fn total_processed(&self) -> u64 {
        self.completed + self.failed
    }
}

/// Proxy task scheduler.
#[derive(Debug)]
pub struct ProxyScheduler {
    config: ProxySchedulerConfig,
    queue: VecDeque<ProxyTask>,
    running: Vec<ProxyTask>,
    stats: ProxySchedulerStats,
    next_id: u64,
}

impl ProxyScheduler {
    /// Create a new scheduler with the given config.
    pub fn new(config: ProxySchedulerConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            running: Vec::new(),
            stats: ProxySchedulerStats::default(),
            next_id: 1,
        }
    }

    /// Submit a task to the scheduler queue. Returns the assigned task id.
    pub fn submit(&mut self, source: &str, dest: &str, estimated_secs: f64) -> Option<u64> {
        if self.queue.len() >= self.config.max_queue_depth {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let task = ProxyTask::new(id, source, dest, estimated_secs);
        self.queue.push_back(task);
        self.pump();
        Some(id)
    }

    /// Advance running tasks — mark the first running task as complete.
    /// In a real implementation this would be driven by async I/O callbacks.
    pub fn complete_one(&mut self) {
        if !self.running.is_empty() {
            self.running.remove(0);
            self.stats.completed += 1;
            self.pump();
        }
    }

    /// Internal: move queued tasks into running slots.
    fn pump(&mut self) {
        while self.running.len() < self.config.max_concurrent {
            if let Some(task) = self.queue.pop_front() {
                self.running.push(task);
            } else {
                break;
            }
        }
    }

    /// Return the number of currently running tasks.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Return the number of queued (pending) tasks.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Return a reference to scheduler statistics.
    pub fn stats(&self) -> &ProxySchedulerStats {
        &self.stats
    }

    /// Record elapsed time for statistics calculation.
    pub fn record_elapsed(&mut self, secs: f64) {
        self.stats.elapsed_secs += secs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler(max: usize) -> ProxyScheduler {
        ProxyScheduler::new(ProxySchedulerConfig::new(max))
    }

    #[test]
    fn test_config_max_concurrent() {
        let cfg = ProxySchedulerConfig::new(8);
        assert_eq!(cfg.max_concurrent(), 8);
    }

    #[test]
    fn test_config_default() {
        let cfg = ProxySchedulerConfig::default();
        assert_eq!(cfg.max_concurrent(), 4);
    }

    #[test]
    fn test_task_estimated_secs() {
        let task = ProxyTask::new(1, "a.mov", "a_proxy.mp4", 42.5);
        assert!((task.estimated_secs() - 42.5).abs() < 1e-9);
    }

    #[test]
    fn test_task_priority_default() {
        let task = ProxyTask::new(1, "a.mov", "b.mp4", 10.0);
        assert_eq!(task.priority, 50);
    }

    #[test]
    fn test_task_with_priority() {
        let task = ProxyTask::new(1, "a.mov", "b.mp4", 10.0).with_priority(200);
        assert_eq!(task.priority, 200);
    }

    #[test]
    fn test_submit_increments_id() {
        let mut sched = make_scheduler(2);
        let id1 = sched
            .submit("a.mov", "a_p.mp4", 5.0)
            .expect("should succeed in test");
        let id2 = sched
            .submit("b.mov", "b_p.mp4", 5.0)
            .expect("should succeed in test");
        assert!(id2 > id1);
    }

    #[test]
    fn test_submit_fills_running_slots() {
        let mut sched = make_scheduler(2);
        sched
            .submit("a.mov", "a_p.mp4", 5.0)
            .expect("should succeed in test");
        sched
            .submit("b.mov", "b_p.mp4", 5.0)
            .expect("should succeed in test");
        assert_eq!(sched.running_count(), 2);
        assert_eq!(sched.queued_count(), 0);
    }

    #[test]
    fn test_submit_queues_excess() {
        let mut sched = make_scheduler(1);
        sched
            .submit("a.mov", "a_p.mp4", 5.0)
            .expect("should succeed in test");
        sched
            .submit("b.mov", "b_p.mp4", 5.0)
            .expect("should succeed in test");
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.queued_count(), 1);
    }

    #[test]
    fn test_complete_one_promotes_queued() {
        let mut sched = make_scheduler(1);
        sched
            .submit("a.mov", "a_p.mp4", 5.0)
            .expect("should succeed in test");
        sched
            .submit("b.mov", "b_p.mp4", 5.0)
            .expect("should succeed in test");
        sched.complete_one();
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.queued_count(), 0);
        assert_eq!(sched.stats().completed, 1);
    }

    #[test]
    fn test_queue_depth_limit() {
        let mut cfg = ProxySchedulerConfig::new(1);
        cfg.max_queue_depth = 2;
        let mut sched = ProxyScheduler::new(cfg);
        sched
            .submit("a.mov", "a_p.mp4", 1.0)
            .expect("should succeed in test"); // running
        sched
            .submit("b.mov", "b_p.mp4", 1.0)
            .expect("should succeed in test"); // queue slot 1
        sched
            .submit("c.mov", "c_p.mp4", 1.0)
            .expect("should succeed in test"); // queue slot 2
        let rejected = sched.submit("d.mov", "d_p.mp4", 1.0);
        assert!(rejected.is_none());
    }

    #[test]
    fn test_stats_throughput_zero_elapsed() {
        let stats = ProxySchedulerStats::default();
        assert!((stats.throughput_per_hour() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_throughput_nonzero() {
        let stats = ProxySchedulerStats {
            completed: 3600,
            elapsed_secs: 3600.0,
            failed: 0,
        };
        assert!((stats.throughput_per_hour() - 3600.0).abs() < 1e-3);
    }

    #[test]
    fn test_stats_total_processed() {
        let stats = ProxySchedulerStats {
            completed: 10,
            elapsed_secs: 60.0,
            failed: 3,
        };
        assert_eq!(stats.total_processed(), 13);
    }

    #[test]
    fn test_record_elapsed_accumulates() {
        let mut sched = make_scheduler(2);
        sched.record_elapsed(30.0);
        sched.record_elapsed(30.0);
        assert!((sched.stats().elapsed_secs - 60.0).abs() < 1e-9);
    }
}
