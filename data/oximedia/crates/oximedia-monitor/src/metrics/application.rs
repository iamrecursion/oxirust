//! Application-specific metrics (encoding, jobs, workers).

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Application metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetrics {
    /// Encoding metrics.
    pub encoding: EncodingMetrics,
    /// Job metrics.
    pub jobs: JobMetrics,
    /// Worker metrics.
    pub workers: WorkerMetrics,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Encoding performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EncodingMetrics {
    /// Current encoding throughput in FPS.
    pub fps: f64,
    /// GOPs per second.
    pub gops_per_second: f64,
    /// Average encoding time per frame (milliseconds).
    pub avg_frame_time_ms: f64,
    /// Transcode queue depth.
    pub queue_depth: usize,
    /// Active encoding sessions.
    pub active_sessions: usize,
    /// Total frames encoded.
    pub total_frames: u64,
    /// Total GOPs encoded.
    pub total_gops: u64,
    /// Dropped frames count.
    pub dropped_frames: u64,
    /// Buffer underruns.
    pub buffer_underruns: u64,
}

impl EncodingMetrics {
    /// Calculate the drop rate.
    #[must_use]
    pub fn drop_rate(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.dropped_frames as f64 / self.total_frames as f64) * 100.0
        }
    }

    /// Check if encoding is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.drop_rate() <= 1.0 && self.buffer_underruns == 0
    }
}

/// Job statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobMetrics {
    /// Total jobs completed.
    pub completed: u64,
    /// Total jobs failed.
    pub failed: u64,
    /// Jobs currently queued.
    pub queued: usize,
    /// Jobs currently running.
    pub running: usize,
    /// Average job duration (seconds).
    pub avg_duration_secs: f64,
    /// Jobs per minute (recent rate).
    pub jobs_per_minute: f64,
    /// Error rate (errors per minute).
    pub error_rate: f64,
}

impl JobMetrics {
    /// Calculate the success rate.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.completed + self.failed;
        if total == 0 {
            100.0
        } else {
            (self.completed as f64 / total as f64) * 100.0
        }
    }

    /// Check if job processing is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.success_rate() >= 95.0 && self.error_rate < 1.0
    }
}

/// Worker status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is online and idle.
    Idle,
    /// Worker is busy processing a job.
    Busy,
    /// Worker is offline.
    Offline,
    /// Worker is in an error state.
    Error,
}

/// Worker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID.
    pub id: String,
    /// Worker hostname.
    pub hostname: String,
    /// Worker status.
    pub status: WorkerStatus,
    /// Current job ID (if busy).
    pub current_job_id: Option<String>,
    /// Jobs completed by this worker.
    pub jobs_completed: u64,
    /// Jobs failed by this worker.
    pub jobs_failed: u64,
    /// Last heartbeat timestamp.
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// CPU usage.
    pub cpu_usage: f64,
    /// Memory usage in bytes.
    pub memory_usage: u64,
}

impl WorkerInfo {
    /// Check if the worker is online.
    #[must_use]
    pub fn is_online(&self) -> bool {
        matches!(self.status, WorkerStatus::Idle | WorkerStatus::Busy)
    }

    /// Check if the worker missed its heartbeat.
    #[must_use]
    pub fn is_stale(&self, timeout_secs: i64) -> bool {
        let now = chrono::Utc::now();
        let elapsed = now.signed_duration_since(self.last_heartbeat);
        elapsed.num_seconds() > timeout_secs
    }
}

/// Worker metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkerMetrics {
    /// Workers by ID.
    pub workers: HashMap<String, WorkerInfo>,
    /// Total workers.
    pub total: usize,
    /// Online workers.
    pub online: usize,
    /// Offline workers.
    pub offline: usize,
    /// Busy workers.
    pub busy: usize,
    /// Idle workers.
    pub idle: usize,
    /// Workers in error state.
    pub error: usize,
}

impl WorkerMetrics {
    /// Calculate the utilization percentage.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.online == 0 {
            0.0
        } else {
            (self.busy as f64 / self.online as f64) * 100.0
        }
    }

    /// Check if worker pool is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.online > 0 && self.error == 0
    }
}

/// Application metrics tracker.
pub struct ApplicationMetricsTracker {
    encoding: Arc<RwLock<EncodingMetrics>>,
    jobs: Arc<RwLock<JobMetrics>>,
    workers: Arc<RwLock<WorkerMetrics>>,
}

impl ApplicationMetricsTracker {
    /// Create a new application metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoding: Arc::new(RwLock::new(EncodingMetrics::default())),
            jobs: Arc::new(RwLock::new(JobMetrics::default())),
            workers: Arc::new(RwLock::new(WorkerMetrics::default())),
        }
    }

    /// Update encoding metrics.
    pub fn update_encoding<F>(&self, f: F)
    where
        F: FnOnce(&mut EncodingMetrics),
    {
        let mut encoding = self.encoding.write();
        f(&mut encoding);
    }

    /// Update job metrics.
    pub fn update_jobs<F>(&self, f: F)
    where
        F: FnOnce(&mut JobMetrics),
    {
        let mut jobs = self.jobs.write();
        f(&mut jobs);
    }

    /// Update worker metrics.
    pub fn update_workers<F>(&self, f: F)
    where
        F: FnOnce(&mut WorkerMetrics),
    {
        let mut workers = self.workers.write();
        f(&mut workers);
    }

    /// Get current metrics snapshot.
    #[must_use]
    pub fn snapshot(&self) -> ApplicationMetrics {
        ApplicationMetrics {
            encoding: self.encoding.read().clone(),
            jobs: self.jobs.read().clone(),
            workers: self.workers.read().clone(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Record a frame encoded.
    pub fn record_frame_encoded(&self, encode_time_ms: f64) {
        self.update_encoding(|enc| {
            enc.total_frames += 1;

            // Update average frame time (exponential moving average)
            let alpha = 0.1;
            enc.avg_frame_time_ms = alpha * encode_time_ms + (1.0 - alpha) * enc.avg_frame_time_ms;

            // Calculate FPS from average frame time
            if enc.avg_frame_time_ms > 0.0 {
                enc.fps = 1000.0 / enc.avg_frame_time_ms;
            }
        });
    }

    /// Record a dropped frame.
    pub fn record_dropped_frame(&self) {
        self.update_encoding(|enc| {
            enc.dropped_frames += 1;
        });
    }

    /// Record a GOP encoded.
    pub fn record_gop_encoded(&self) {
        self.update_encoding(|enc| {
            enc.total_gops += 1;
        });
    }

    /// Record a buffer underrun.
    pub fn record_buffer_underrun(&self) {
        self.update_encoding(|enc| {
            enc.buffer_underruns += 1;
        });
    }

    /// Update queue depth.
    pub fn update_queue_depth(&self, depth: usize) {
        self.update_encoding(|enc| {
            enc.queue_depth = depth;
        });
    }

    /// Update active sessions.
    pub fn update_active_sessions(&self, count: usize) {
        self.update_encoding(|enc| {
            enc.active_sessions = count;
        });
    }

    /// Record a job completed.
    pub fn record_job_completed(&self, duration_secs: f64) {
        self.update_jobs(|jobs| {
            jobs.completed += 1;

            // Update average duration (exponential moving average)
            let alpha = 0.1;
            jobs.avg_duration_secs = alpha * duration_secs + (1.0 - alpha) * jobs.avg_duration_secs;
        });
    }

    /// Record a job failed.
    pub fn record_job_failed(&self) {
        self.update_jobs(|jobs| {
            jobs.failed += 1;
        });
    }

    /// Update job queue status.
    pub fn update_job_queue(&self, queued: usize, running: usize) {
        self.update_jobs(|jobs| {
            jobs.queued = queued;
            jobs.running = running;
        });
    }

    /// Register a worker.
    pub fn register_worker(&self, worker: WorkerInfo) {
        self.update_workers(|workers| {
            workers.workers.insert(worker.id.clone(), worker);
            Self::recalculate_worker_stats(workers);
        });
    }

    /// Update worker status.
    pub fn update_worker_status(&self, worker_id: &str, status: WorkerStatus) {
        self.update_workers(|workers| {
            if let Some(worker) = workers.workers.get_mut(worker_id) {
                worker.status = status;
                worker.last_heartbeat = chrono::Utc::now();
            }
            Self::recalculate_worker_stats(workers);
        });
    }

    /// Remove a worker.
    pub fn remove_worker(&self, worker_id: &str) {
        self.update_workers(|workers| {
            workers.workers.remove(worker_id);
            Self::recalculate_worker_stats(workers);
        });
    }

    fn recalculate_worker_stats(workers: &mut WorkerMetrics) {
        workers.total = workers.workers.len();
        workers.online = 0;
        workers.offline = 0;
        workers.busy = 0;
        workers.idle = 0;
        workers.error = 0;

        for worker in workers.workers.values() {
            match worker.status {
                WorkerStatus::Idle => {
                    workers.online += 1;
                    workers.idle += 1;
                }
                WorkerStatus::Busy => {
                    workers.online += 1;
                    workers.busy += 1;
                }
                WorkerStatus::Offline => {
                    workers.offline += 1;
                }
                WorkerStatus::Error => {
                    workers.error += 1;
                }
            }
        }
    }
}

impl Default for ApplicationMetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_metrics_drop_rate() {
        let metrics = EncodingMetrics {
            total_frames: 1000,
            dropped_frames: 10,
            ..Default::default()
        };

        assert_eq!(metrics.drop_rate(), 1.0);
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_job_metrics_success_rate() {
        let metrics = JobMetrics {
            completed: 95,
            failed: 5,
            ..Default::default()
        };

        assert_eq!(metrics.success_rate(), 95.0);
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_worker_metrics_utilization() {
        let metrics = WorkerMetrics {
            total: 10,
            online: 8,
            busy: 6,
            idle: 2,
            offline: 2,
            error: 0,
            workers: HashMap::new(),
        };

        assert_eq!(metrics.utilization(), 75.0);
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_worker_info_is_stale() {
        let mut worker = WorkerInfo {
            id: "worker-1".to_string(),
            hostname: "host-1".to_string(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            jobs_completed: 0,
            jobs_failed: 0,
            last_heartbeat: chrono::Utc::now() - chrono::Duration::seconds(150),
            cpu_usage: 0.0,
            memory_usage: 0,
        };

        assert!(worker.is_stale(120)); // 2 minute timeout
        assert!(!worker.is_stale(200)); // 3+ minute timeout

        worker.last_heartbeat = chrono::Utc::now();
        assert!(!worker.is_stale(120));
    }

    #[test]
    fn test_application_metrics_tracker() {
        let tracker = ApplicationMetricsTracker::new();

        tracker.record_frame_encoded(16.67); // ~60fps
        tracker.record_frame_encoded(16.67);
        tracker.record_dropped_frame();
        tracker.record_gop_encoded();

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.encoding.total_frames, 2);
        assert_eq!(snapshot.encoding.dropped_frames, 1);
        assert_eq!(snapshot.encoding.total_gops, 1);
    }

    #[test]
    fn test_worker_registration() {
        let tracker = ApplicationMetricsTracker::new();

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            hostname: "host-1".to_string(),
            status: WorkerStatus::Idle,
            current_job_id: None,
            jobs_completed: 0,
            jobs_failed: 0,
            last_heartbeat: chrono::Utc::now(),
            cpu_usage: 10.0,
            memory_usage: 1024 * 1024 * 100,
        };

        tracker.register_worker(worker);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.workers.total, 1);
        assert_eq!(snapshot.workers.online, 1);
        assert_eq!(snapshot.workers.idle, 1);

        tracker.update_worker_status("worker-1", WorkerStatus::Busy);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.workers.busy, 1);
        assert_eq!(snapshot.workers.idle, 0);
    }

    #[test]
    fn test_job_tracking() {
        let tracker = ApplicationMetricsTracker::new();

        tracker.record_job_completed(120.5);
        tracker.record_job_completed(130.0);
        tracker.record_job_failed();

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.jobs.completed, 2);
        assert_eq!(snapshot.jobs.failed, 1);
        assert!(snapshot.jobs.avg_duration_secs > 0.0);
    }
}
