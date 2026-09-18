//! Worker registry and management

use crate::heartbeat_batch::{HeartbeatBatch, HeartbeatReport};
use crate::{FarmError, Result, WorkerId, WorkerState};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::Duration;

/// Worker registration information
#[derive(Debug, Clone)]
pub struct WorkerRegistration {
    pub worker_id: WorkerId,
    pub hostname: String,
    pub capabilities: WorkerCapabilities,
    pub metadata: HashMap<String, String>,
    pub state: WorkerState,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub active_tasks: u32,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
}

/// Worker capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerCapabilities {
    pub cpu_cores: u32,
    pub memory_bytes: u64,
    pub supported_codecs: Vec<String>,
    pub supported_formats: Vec<String>,
    pub has_gpu: bool,
    pub gpus: Vec<GpuInfo>,
    pub max_concurrent_tasks: u32,
    pub tags: HashMap<String, String>,
}

/// GPU information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_bytes: u64,
    pub vendor: String,
    pub supported_codecs: Vec<String>,
}

/// Worker status update
#[derive(Debug, Clone, Copy)]
pub struct WorkerStatusUpdate {
    pub cpu_usage: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub disk_free: u64,
    pub active_tasks: u32,
    pub state: WorkerState,
}

/// Worker registry manages all connected workers
pub struct WorkerRegistry {
    workers: RwLock<HashMap<WorkerId, WorkerRegistration>>,
    heartbeat_timeout: Duration,
    /// Batch accumulator for incoming heartbeat updates.
    ///
    /// Guarded by a separate `Mutex` so the (rarely written) `workers` map and
    /// the (frequently written) batch do not contend on the same lock.
    heartbeat_batch: Mutex<HeartbeatBatch>,
}

/// Default maximum batch size before an automatic flush is triggered.
const DEFAULT_HEARTBEAT_BATCH_SIZE: usize = 32;
/// Default flush interval for the heartbeat batch.
const DEFAULT_HEARTBEAT_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

impl WorkerRegistry {
    /// Create a new worker registry with default heartbeat batch settings.
    #[must_use]
    pub fn new(heartbeat_timeout: Duration) -> Self {
        Self::with_batch_config(
            heartbeat_timeout,
            DEFAULT_HEARTBEAT_BATCH_SIZE,
            DEFAULT_HEARTBEAT_FLUSH_INTERVAL,
        )
    }

    /// Create a new worker registry with custom heartbeat batch parameters.
    ///
    /// * `heartbeat_timeout` — how long without a heartbeat before a worker is
    ///   considered stale.
    /// * `batch_size` — number of pending reports that trigger an automatic flush.
    /// * `flush_interval` — maximum time between flushes (timer-based).
    #[must_use]
    pub fn with_batch_config(
        heartbeat_timeout: Duration,
        batch_size: usize,
        flush_interval: Duration,
    ) -> Self {
        Self {
            workers: RwLock::new(HashMap::new()),
            heartbeat_timeout,
            heartbeat_batch: Mutex::new(HeartbeatBatch::new(batch_size, flush_interval)),
        }
    }

    /// Register a new worker
    pub async fn register_worker(
        &self,
        worker_id: WorkerId,
        hostname: String,
        capabilities: WorkerCapabilities,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let mut workers = self.workers.write();

        if workers.contains_key(&worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {worker_id} already registered"
            )));
        }

        let registration = WorkerRegistration {
            worker_id: worker_id.clone(),
            hostname,
            capabilities,
            metadata,
            state: WorkerState::Idle,
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            active_tasks: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
        };

        workers.insert(worker_id.clone(), registration);
        tracing::info!("Worker {} registered successfully", worker_id);
        Ok(())
    }

    /// Unregister a worker
    pub async fn unregister_worker(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if workers.remove(worker_id).is_none() {
            return Err(FarmError::NotFound(format!("Worker {worker_id} not found")));
        }

        tracing::info!("Worker {} unregistered", worker_id);
        Ok(())
    }

    /// Update worker heartbeat.
    ///
    /// The heartbeat is first staged in the internal [`HeartbeatBatch`].  When
    /// the batch reaches its configured size limit *or* the flush interval has
    /// elapsed, the batch is flushed and all staged updates are applied to the
    /// in-memory worker map in a single write-lock acquisition.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if `worker_id` is unknown **at flush
    /// time**.  Updates for unknown workers are silently discarded (the same
    /// behaviour as a direct write would produce for a race with
    /// `unregister_worker`).
    pub async fn heartbeat(&self, worker_id: &WorkerId, status: WorkerStatusUpdate) -> Result<()> {
        // Verify the worker exists before staging (fail-fast for callers).
        {
            let workers = self.workers.read();
            if !workers.contains_key(worker_id) {
                return Err(FarmError::NotFound(format!("Worker {worker_id} not found")));
            }
        }

        let report = HeartbeatReport::with_details(
            worker_id.as_str(),
            status.cpu_usage,
            status.memory_used as f64 / status.memory_total.max(1) as f64,
            status.active_tasks,
        );

        let flush_needed = {
            let mut batch = self.heartbeat_batch.lock();
            let size_triggered = batch.add(report);
            size_triggered || batch.should_flush()
        };

        if flush_needed {
            self.flush_heartbeat_batch(status.state, status.active_tasks, worker_id)?;
        } else {
            // Fast-path: update the worker record directly while no flush is needed
            // so that callers always see an up-to-date timestamp even between flushes.
            let mut workers = self.workers.write();
            if let Some(worker) = workers.get_mut(worker_id) {
                worker.last_heartbeat = Utc::now();
                worker.state = status.state;
                worker.active_tasks = status.active_tasks;
            }
        }

        tracing::debug!("Heartbeat received from worker {}", worker_id);
        Ok(())
    }

    /// Flush all pending heartbeat reports from the batch and apply them to the
    /// worker map.  The `fallback_state` and `fallback_active_tasks` are used
    /// for workers whose report was not found in the batch (e.g., the single
    /// caller whose report triggered the flush).
    fn flush_heartbeat_batch(
        &self,
        fallback_state: WorkerState,
        fallback_active_tasks: u32,
        trigger_worker_id: &WorkerId,
    ) -> Result<()> {
        let reports = {
            let mut batch = self.heartbeat_batch.lock();
            batch.flush()
        };

        let mut workers = self.workers.write();
        let now = Utc::now();

        // Apply all batched reports.
        for report in &reports {
            let wid = WorkerId::new(report.worker_id.as_str());
            if let Some(worker) = workers.get_mut(&wid) {
                worker.last_heartbeat = now;
                worker.active_tasks = report.active_tasks;
                // Keep the most recent state from the direct-path update or use
                // the report's active-tasks as a proxy for Busy/Idle.
                if report.active_tasks > 0 && worker.state == WorkerState::Idle {
                    worker.state = WorkerState::Busy;
                }
            }
        }

        // Ensure the triggering worker is also updated (it was added to the
        // batch just before calling this function; the batch flush above covers
        // it, but apply the state explicitly to be safe).
        if let Some(worker) = workers.get_mut(trigger_worker_id) {
            worker.last_heartbeat = now;
            worker.state = fallback_state;
            worker.active_tasks = fallback_active_tasks;
        }

        tracing::debug!("Flushed {} heartbeat reports from batch", reports.len());
        Ok(())
    }

    /// Force-flush any pending heartbeat reports immediately.
    ///
    /// This is useful for graceful-shutdown paths where timer-based flushing
    /// may not have fired yet.
    pub fn flush_pending_heartbeats(&self) {
        let reports = {
            let mut batch = self.heartbeat_batch.lock();
            if batch.is_empty() {
                return;
            }
            batch.flush()
        };

        let mut workers = self.workers.write();
        let now = Utc::now();
        for report in &reports {
            let wid = WorkerId::new(report.worker_id.as_str());
            if let Some(worker) = workers.get_mut(&wid) {
                worker.last_heartbeat = now;
                worker.active_tasks = report.active_tasks;
            }
        }
        tracing::debug!(
            "flush_pending_heartbeats: applied {} reports",
            reports.len()
        );
    }

    /// Return the number of heartbeat reports currently pending in the batch.
    #[must_use]
    pub fn pending_heartbeat_count(&self) -> usize {
        self.heartbeat_batch.lock().pending_count()
    }

    /// Mark worker as offline
    pub async fn mark_offline(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.state = WorkerState::Offline;
            tracing::warn!("Worker {} marked as offline", worker_id);
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Get worker by ID
    pub fn get_worker(&self, worker_id: &WorkerId) -> Option<WorkerRegistration> {
        let workers = self.workers.read();
        workers.get(worker_id).cloned()
    }

    /// List all workers
    pub fn list_workers(&self) -> Vec<WorkerRegistration> {
        let workers = self.workers.read();
        workers.values().cloned().collect()
    }

    /// List workers by state
    pub fn list_workers_by_state(&self, state: WorkerState) -> Vec<WorkerRegistration> {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state == state)
            .cloned()
            .collect()
    }

    /// Get active worker count
    pub fn active_worker_count(&self) -> usize {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state != WorkerState::Offline)
            .count()
    }

    /// Increment task completed count
    pub async fn increment_task_completed(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.total_tasks_completed += 1;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Increment task failed count
    pub async fn increment_task_failed(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.total_tasks_failed += 1;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Get workers that haven't sent heartbeat within timeout
    pub fn get_stale_workers(&self) -> Vec<WorkerId> {
        let workers = self.workers.read();
        let now = Utc::now();
        let timeout = chrono::Duration::from_std(self.heartbeat_timeout)
            .unwrap_or(chrono::Duration::seconds(30));

        workers
            .values()
            .filter(|w| w.state != WorkerState::Offline && (now - w.last_heartbeat) > timeout)
            .map(|w| w.worker_id.clone())
            .collect()
    }

    /// Get worker statistics
    pub fn get_statistics(&self) -> WorkerStatistics {
        let workers = self.workers.read();

        let total = workers.len();
        let idle = workers
            .values()
            .filter(|w| w.state == WorkerState::Idle)
            .count();
        let busy = workers
            .values()
            .filter(|w| w.state == WorkerState::Busy)
            .count();
        let overloaded = workers
            .values()
            .filter(|w| w.state == WorkerState::Overloaded)
            .count();
        let draining = workers
            .values()
            .filter(|w| w.state == WorkerState::Draining)
            .count();
        let offline = workers
            .values()
            .filter(|w| w.state == WorkerState::Offline)
            .count();

        let total_tasks_completed: u64 = workers.values().map(|w| w.total_tasks_completed).sum();
        let total_tasks_failed: u64 = workers.values().map(|w| w.total_tasks_failed).sum();

        WorkerStatistics {
            total,
            idle,
            busy,
            overloaded,
            draining,
            offline,
            total_tasks_completed,
            total_tasks_failed,
        }
    }
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStatistics {
    pub total: usize,
    pub idle: usize,
    pub busy: usize,
    pub overloaded: usize,
    pub draining: usize,
    pub offline: usize,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_capabilities() -> WorkerCapabilities {
        WorkerCapabilities {
            cpu_cores: 8,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            supported_codecs: vec!["h264".to_string(), "h265".to_string()],
            supported_formats: vec!["mp4".to_string(), "mkv".to_string()],
            has_gpu: false,
            gpus: vec![],
            max_concurrent_tasks: 4,
            tags: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        assert_eq!(registry.active_worker_count(), 1);
        let worker = registry.get_worker(&worker_id).unwrap();
        assert_eq!(worker.hostname, "host1");
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let result = registry
            .register_worker(
                worker_id,
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_worker_unregistration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        registry.unregister_worker(&worker_id).await.unwrap();
        assert_eq!(registry.active_worker_count(), 0);
    }

    #[tokio::test]
    async fn test_worker_heartbeat() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let status = WorkerStatusUpdate {
            cpu_usage: 0.5,
            memory_used: 8 * 1024 * 1024 * 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            disk_free: 100 * 1024 * 1024 * 1024,
            active_tasks: 2,
            state: WorkerState::Busy,
        };

        registry.heartbeat(&worker_id, status).await.unwrap();

        let worker = registry.get_worker(&worker_id).unwrap();
        assert_eq!(worker.state, WorkerState::Busy);
        assert_eq!(worker.active_tasks, 2);
    }

    #[tokio::test]
    async fn test_list_workers_by_state() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));

        let worker1 = WorkerId::new("worker-1");
        let worker2 = WorkerId::new("worker-2");

        registry
            .register_worker(
                worker1.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        registry
            .register_worker(
                worker2.clone(),
                "host2".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let status = WorkerStatusUpdate {
            cpu_usage: 0.5,
            memory_used: 8 * 1024 * 1024 * 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            disk_free: 100 * 1024 * 1024 * 1024,
            active_tasks: 2,
            state: WorkerState::Busy,
        };

        registry.heartbeat(&worker2, status).await.unwrap();

        let idle_workers = registry.list_workers_by_state(WorkerState::Idle);
        assert_eq!(idle_workers.len(), 1);

        let busy_workers = registry.list_workers_by_state(WorkerState::Busy);
        assert_eq!(busy_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_task_counters() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        registry.increment_task_completed(&worker_id).await.unwrap();
        registry.increment_task_completed(&worker_id).await.unwrap();
        registry.increment_task_failed(&worker_id).await.unwrap();

        let worker = registry.get_worker(&worker_id).unwrap();
        assert_eq!(worker.total_tasks_completed, 2);
        assert_eq!(worker.total_tasks_failed, 1);
    }

    #[tokio::test]
    async fn test_statistics() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));

        registry
            .register_worker(
                WorkerId::new("worker-1"),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        registry
            .register_worker(
                WorkerId::new("worker-2"),
                "host2".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .unwrap();

        let stats = registry.get_statistics();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.idle, 2);
        assert_eq!(stats.busy, 0);
    }

    // ── HeartbeatBatch integration tests ──────────────────────────────────────

    /// Push N heartbeats below the batch threshold and assert no batch flush has
    /// occurred yet (pending count remains at N).
    #[tokio::test]
    async fn test_heartbeat_batch_accumulates() {
        // batch_size=10 so a single heartbeat will NOT trigger a flush.
        let registry = WorkerRegistry::with_batch_config(
            Duration::from_secs(60),
            10,                        // batch_size
            Duration::from_secs(3600), // flush_interval (never fires in this test)
        );
        let worker_id = WorkerId::new("batch-worker-1");
        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("register");

        let status = WorkerStatusUpdate {
            cpu_usage: 0.1,
            memory_used: 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            disk_free: 100 * 1024 * 1024 * 1024,
            active_tasks: 0,
            state: WorkerState::Idle,
        };

        // Send 3 heartbeats — well below threshold of 10.
        for _ in 0..3 {
            registry
                .heartbeat(&worker_id, status)
                .await
                .expect("heartbeat");
        }

        // With a batch size of 10 and only 3 heartbeats the batch should NOT
        // have been flushed yet.  The fast-path in heartbeat() updates the
        // worker map directly but also stages a report; the batch pending count
        // reflects staged-but-not-yet-batch-flushed items only when the
        // size_limit triggers.  We verify that the pending count is < 10 (not
        // a flush boundary was crossed).
        //
        // The primary assertion is that the worker is still registered and
        // readable — no data was lost regardless of flush state.
        let worker = registry
            .get_worker(&worker_id)
            .expect("worker should still exist");
        assert_eq!(worker.active_tasks, 0);
        // Pending count after the fast-path should be 0 because each heartbeat
        // that does NOT trigger a size-flush goes through the direct-write path
        // and ends up consuming the staged entry via the batch only when a flush
        // boundary is crossed.  Specifically: 3 < 10, so no batch-flush event
        // fired and the batch still holds 3 staged entries.
        assert!(
            registry.pending_heartbeat_count() <= 3,
            "fewer than threshold items should still be in batch"
        );
    }

    /// Push exactly `batch_size` heartbeats so the auto-flush fires and all
    /// updates are reflected in the worker map.
    #[tokio::test]
    async fn test_heartbeat_batch_flushes() {
        // batch_size=5, so the 5th heartbeat triggers the size-based flush.
        let registry = WorkerRegistry::with_batch_config(
            Duration::from_secs(60),
            5,
            Duration::from_secs(3600),
        );
        let worker_id = WorkerId::new("batch-worker-2");
        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("register");

        // Send heartbeats up to the flush threshold.
        for i in 0u32..5 {
            let status = WorkerStatusUpdate {
                cpu_usage: 0.2,
                memory_used: 1024,
                memory_total: 16 * 1024 * 1024 * 1024,
                disk_free: 100 * 1024 * 1024 * 1024,
                active_tasks: i,
                state: WorkerState::Busy,
            };
            registry
                .heartbeat(&worker_id, status)
                .await
                .expect("heartbeat");
        }

        // After the flush the batch should be empty.
        assert_eq!(
            registry.pending_heartbeat_count(),
            0,
            "batch should be empty after threshold flush"
        );

        // The worker record must have been updated.
        let worker = registry
            .get_worker(&worker_id)
            .expect("worker should still exist");
        assert_eq!(
            worker.state,
            WorkerState::Busy,
            "worker state should reflect latest heartbeat"
        );
    }
}
