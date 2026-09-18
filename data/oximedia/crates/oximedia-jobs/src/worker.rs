// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker pool implementation for job execution.

use crate::job::Job;
use crate::metrics::MetricsCollector;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Worker errors
#[derive(Debug, Error)]
pub enum WorkerError {
    /// Job execution failed
    #[error("Job execution failed: {0}")]
    ExecutionFailed(String),

    /// Worker shutdown
    #[error("Worker shutdown")]
    Shutdown,

    /// Invalid job
    #[error("Invalid job: {0}")]
    InvalidJob(String),

    /// Timeout
    #[error("Job execution timeout")]
    Timeout,

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// Result type for worker operations
pub type Result<T> = std::result::Result<T, WorkerError>;

/// Job executor trait
#[async_trait]
pub trait JobExecutor: Send + Sync {
    /// Execute a job
    ///
    /// # Errors
    ///
    /// Returns an error if job execution fails
    async fn execute(&self, job: &mut Job) -> Result<()>;

    /// Check if executor can handle a job type
    fn can_handle(&self, job: &Job) -> bool;
}

/// Job execution context
pub struct ExecutionContext {
    /// Job being executed
    pub job: Job,
    /// Cancellation token
    pub cancelled: Arc<RwLock<bool>>,
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker pool size
    pub pool_size: usize,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,
    /// Health check timeout in seconds
    pub health_timeout_secs: i64,
    /// Job execution timeout in seconds
    pub execution_timeout_secs: u64,
    /// Enable auto-scaling
    pub auto_scaling: bool,
    /// Minimum pool size for auto-scaling
    pub min_pool_size: usize,
    /// Maximum pool size for auto-scaling
    pub max_pool_size: usize,
    /// Target utilization for scaling (percentage)
    pub target_utilization: f64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            pool_size: 4,
            heartbeat_interval_secs: 30,
            health_timeout_secs: 90,
            execution_timeout_secs: 3600,
            auto_scaling: false,
            min_pool_size: 2,
            max_pool_size: 16,
            target_utilization: 70.0,
        }
    }
}

/// Worker message
enum WorkerMessage {
    /// Execute a job
    Execute(Box<Job>),
    /// Cancel a job
    Cancel(Uuid),
    /// Shutdown worker
    Shutdown,
}

/// Worker instance
struct Worker {
    /// Worker ID
    id: String,
    /// Message receiver
    receiver: mpsc::UnboundedReceiver<WorkerMessage>,
    /// Job executor
    executor: Arc<dyn JobExecutor>,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    /// Configuration
    config: WorkerConfig,
    /// Current job cancellation flag
    current_job_cancelled: Arc<RwLock<Option<Arc<RwLock<bool>>>>>,
    /// Shared active-job counter (shared with the owning WorkerPool).
    active_job_count: Arc<tokio::sync::Mutex<usize>>,
}

impl Worker {
    /// Create a new worker
    fn new(
        id: String,
        receiver: mpsc::UnboundedReceiver<WorkerMessage>,
        executor: Arc<dyn JobExecutor>,
        metrics: Arc<MetricsCollector>,
        config: WorkerConfig,
        active_job_count: Arc<tokio::sync::Mutex<usize>>,
    ) -> Self {
        Self {
            id,
            receiver,
            executor,
            metrics,
            config,
            current_job_cancelled: Arc::new(RwLock::new(None)),
            active_job_count,
        }
    }

    /// Run the worker
    async fn run(mut self) {
        info!("Worker {} started", self.id);
        self.metrics.register_worker(self.id.clone()).await;

        let heartbeat_interval = Duration::from_secs(self.config.heartbeat_interval_secs);
        let mut heartbeat_ticker = tokio::time::interval(heartbeat_interval);

        loop {
            tokio::select! {
                _ = heartbeat_ticker.tick() => {
                    self.metrics.worker_heartbeat(&self.id).await;
                }
                message = self.receiver.recv() => {
                    match message {
                        Some(WorkerMessage::Execute(mut job)) => {
                            self.execute_job(&mut job).await;
                        }
                        Some(WorkerMessage::Cancel(job_id)) => {
                            self.cancel_job(job_id).await;
                        }
                        Some(WorkerMessage::Shutdown) | None => {
                            info!("Worker {} shutting down", self.id);
                            break;
                        }
                    }
                }
            }
        }

        self.metrics.unregister_worker(&self.id).await;
        info!("Worker {} stopped", self.id);
    }

    /// Execute a job
    async fn execute_job(&mut self, job: &mut Job) {
        if !self.executor.can_handle(job) {
            error!("Worker {} cannot handle job {}", self.id, job.id);
            // Decrement active counter: we counted this job in submit() already.
            let mut count = self.active_job_count.lock().await;
            *count = count.saturating_sub(1);
            return;
        }

        info!("Worker {} executing job {}", self.id, job.id);

        job.mark_started(self.id.clone());

        self.metrics
            .record_job_started(
                job.id,
                job.name.clone(),
                job.payload.job_type().to_string(),
                job.priority,
                self.id.clone(),
            )
            .await;

        let cancelled = Arc::new(RwLock::new(false));
        *self.current_job_cancelled.write().await = Some(cancelled.clone());

        let timeout = Duration::from_secs(
            job.resource_quota
                .max_execution_time_secs
                .unwrap_or(self.config.execution_timeout_secs),
        );

        let execution_result = tokio::time::timeout(timeout, self.executor.execute(job)).await;

        *self.current_job_cancelled.write().await = None;

        match execution_result {
            Ok(Ok(())) => {
                if *cancelled.read().await {
                    warn!("Job {} was cancelled", job.id);
                    job.mark_cancelled();
                } else {
                    info!("Job {} completed successfully", job.id);
                    job.mark_completed();
                    self.metrics.record_job_completed(job.id, &self.id).await;
                }
            }
            Ok(Err(e)) => {
                error!("Job {} failed: {}", job.id, e);
                let error_msg = e.to_string();
                job.mark_failed(error_msg.clone());
                self.metrics
                    .record_job_failed(job.id, &self.id, error_msg)
                    .await;
            }
            Err(_) => {
                error!("Job {} timed out", job.id);
                let error_msg = "Execution timeout".to_string();
                job.mark_failed(error_msg.clone());
                self.metrics
                    .record_job_failed(job.id, &self.id, error_msg)
                    .await;
            }
        }

        // Decrement the active job counter so drain() knows when it is safe
        // to declare the pool idle.
        let mut count = self.active_job_count.lock().await;
        *count = count.saturating_sub(1);
    }

    /// Cancel a job
    async fn cancel_job(&self, job_id: Uuid) {
        debug!("Worker {} cancelling job {}", self.id, job_id);
        if let Some(cancelled) = self.current_job_cancelled.read().await.as_ref() {
            *cancelled.write().await = true;
        }
    }
}

/// Worker pool
pub struct WorkerPool {
    /// Worker senders
    workers: Arc<RwLock<Vec<mpsc::UnboundedSender<WorkerMessage>>>>,
    /// Job executor
    executor: Arc<dyn JobExecutor>,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    /// Configuration
    config: WorkerConfig,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
    /// Drain flag — when set, the pool stops accepting new jobs but finishes
    /// already-dispatched ones before shutting down.
    draining: Arc<RwLock<bool>>,
    /// Load balancing semaphore
    semaphore: Arc<Semaphore>,
    /// Counter of jobs currently being executed across all workers.
    active_job_count: Arc<tokio::sync::Mutex<usize>>,
}

impl WorkerPool {
    /// Create a new worker pool
    #[must_use]
    pub fn new(
        executor: Arc<dyn JobExecutor>,
        metrics: Arc<MetricsCollector>,
        config: WorkerConfig,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.pool_size));

        Self {
            workers: Arc::new(RwLock::new(Vec::new())),
            executor,
            metrics,
            config,
            shutdown: Arc::new(RwLock::new(false)),
            draining: Arc::new(RwLock::new(false)),
            semaphore,
            active_job_count: Arc::new(tokio::sync::Mutex::new(0)),
        }
    }

    /// Start the worker pool
    pub async fn start(&self) {
        info!(
            "Starting worker pool with {} workers",
            self.config.pool_size
        );

        let mut workers = self.workers.write().await;
        for i in 0..self.config.pool_size {
            let (sender, receiver) = mpsc::unbounded_channel();
            let worker_id = format!("worker-{i}");

            let worker = Worker::new(
                worker_id,
                receiver,
                self.executor.clone(),
                self.metrics.clone(),
                self.config.clone(),
                self.active_job_count.clone(),
            );

            tokio::spawn(worker.run());
            workers.push(sender);
        }

        if self.config.auto_scaling {
            self.start_auto_scaling();
        }
    }

    /// Submit a job for execution.
    ///
    /// Returns [`WorkerError::Shutdown`] when the pool is shutting down or in
    /// drain mode (drain mode blocks new submissions while in-flight jobs finish).
    ///
    /// # Errors
    ///
    /// Returns an error if no workers are available or if the job cannot be submitted
    pub async fn submit(&self, job: Job) -> Result<()> {
        if *self.shutdown.read().await {
            return Err(WorkerError::Shutdown);
        }
        if *self.draining.read().await {
            return Err(WorkerError::Shutdown);
        }

        let permit = self.semaphore.acquire().await.map_err(|e| {
            WorkerError::ExecutionFailed(format!("Failed to acquire semaphore: {e}"))
        })?;

        let workers = self.workers.read().await;
        if workers.is_empty() {
            return Err(WorkerError::ExecutionFailed(
                "No workers available".to_string(),
            ));
        }

        let worker_idx = (job.id.as_u128() as usize) % workers.len();
        let worker = &workers[worker_idx];

        worker
            .send(WorkerMessage::Execute(Box::new(job)))
            .map_err(|e| WorkerError::ExecutionFailed(format!("Failed to send job: {e}")))?;

        drop(permit);

        // Increment the active job counter so drain() can wait for completion.
        *self.active_job_count.lock().await += 1;

        Ok(())
    }

    /// Enter drain mode: stop accepting new jobs and wait for all currently
    /// in-flight jobs to finish before returning.
    ///
    /// After `drain()` returns the pool is idle and can be safely shut down.
    pub async fn drain(&self) {
        info!("WorkerPool entering drain mode");
        *self.draining.write().await = true;

        // Wait until no active jobs remain.
        let poll_interval = Duration::from_millis(100);
        loop {
            if *self.active_job_count.lock().await == 0 {
                break;
            }
            sleep(poll_interval).await;
        }

        info!("WorkerPool drain complete — all jobs finished");
    }

    /// Returns `true` when the pool is in drain mode.
    pub async fn is_draining(&self) -> bool {
        *self.draining.read().await
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: Uuid) {
        let workers = self.workers.read().await;
        for worker in workers.iter() {
            let _ = worker.send(WorkerMessage::Cancel(job_id));
        }
    }

    /// Shutdown the worker pool
    pub async fn shutdown(&self) {
        info!("Shutting down worker pool");
        *self.shutdown.write().await = true;

        let workers = self.workers.read().await;
        for worker in workers.iter() {
            let _ = worker.send(WorkerMessage::Shutdown);
        }

        sleep(Duration::from_secs(2)).await;
        info!("Worker pool shutdown complete");
    }

    /// Get worker count
    #[must_use]
    pub async fn worker_count(&self) -> usize {
        self.workers.read().await.len()
    }

    /// Start auto-scaling monitoring
    fn start_auto_scaling(&self) {
        let workers = self.workers.clone();
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        let executor = self.executor.clone();
        let shutdown = self.shutdown.clone();
        let active_job_count = self.active_job_count.clone();

        tokio::spawn(async move {
            let mut check_interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                check_interval.tick().await;

                if *shutdown.read().await {
                    break;
                }

                let utilization = metrics.get_worker_utilization().await;
                let current_size = workers.read().await.len();

                debug!(
                    "Worker pool auto-scaling check: utilization={:.2}%, size={}",
                    utilization, current_size
                );

                if utilization > config.target_utilization && current_size < config.max_pool_size {
                    info!(
                        "Scaling up worker pool from {} to {}",
                        current_size,
                        current_size + 1
                    );
                    let (sender, receiver) = mpsc::unbounded_channel();
                    let worker_id = format!("worker-{current_size}");

                    let worker = Worker::new(
                        worker_id,
                        receiver,
                        executor.clone(),
                        metrics.clone(),
                        config.clone(),
                        active_job_count.clone(),
                    );

                    tokio::spawn(worker.run());
                    workers.write().await.push(sender);
                } else if utilization < (config.target_utilization / 2.0)
                    && current_size > config.min_pool_size
                {
                    info!(
                        "Scaling down worker pool from {} to {}",
                        current_size,
                        current_size - 1
                    );
                    if let Some(worker) = workers.write().await.pop() {
                        let _ = worker.send(WorkerMessage::Shutdown);
                    }
                }
            }
        });
    }

    /// Health check
    pub async fn health_check(&self) -> Vec<String> {
        let unhealthy = self
            .metrics
            .get_unhealthy_workers(self.config.health_timeout_secs)
            .await;

        if !unhealthy.is_empty() {
            warn!("Found {} unhealthy workers", unhealthy.len());
        }

        unhealthy
    }
}

/// Default job executor (placeholder)
pub struct DefaultExecutor;

#[async_trait]
impl JobExecutor for DefaultExecutor {
    async fn execute(&self, job: &mut Job) -> Result<()> {
        debug!(
            "Executing job {} of type {}",
            job.id,
            job.payload.job_type()
        );

        sleep(Duration::from_millis(100)).await;

        job.update_progress(50);
        sleep(Duration::from_millis(100)).await;

        job.update_progress(100);

        Ok(())
    }

    fn can_handle(&self, _job: &Job) -> bool {
        true
    }
}
