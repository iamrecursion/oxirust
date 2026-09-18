// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Main render farm coordinator.

use crate::error::{Error, Result};
use crate::job::{Job, JobId, JobState, JobSubmission};
use crate::scheduler::{Scheduler, SchedulingAlgorithm, Task};
use crate::worker::{Worker, WorkerId, WorkerRegistration, WorkerState};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Scheduling algorithm
    pub scheduling_algorithm: SchedulingAlgorithm,
    /// Maximum retries for failed tasks
    pub max_retries: u32,
    /// Worker timeout (seconds)
    pub worker_timeout: u64,
    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u64,
    /// Job cleanup interval (seconds)
    pub cleanup_interval: u64,
    /// Enable auto-scaling
    pub enable_auto_scaling: bool,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            scheduling_algorithm: SchedulingAlgorithm::Priority,
            max_retries: 3,
            worker_timeout: 60,
            heartbeat_interval: 30,
            cleanup_interval: 300,
            enable_auto_scaling: false,
            max_concurrent_jobs: 1000,
        }
    }
}

/// Coordinator events
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// Job submitted
    JobSubmitted {
        /// Job ID
        job_id: JobId,
    },
    /// Job started
    JobStarted {
        /// Job ID
        job_id: JobId,
    },
    /// Job completed
    JobCompleted {
        /// Job ID
        job_id: JobId,
    },
    /// Job failed
    JobFailed {
        /// Job ID
        job_id: JobId,
        /// Error message
        error: String,
    },
    /// Worker registered
    WorkerRegistered {
        /// Worker ID
        worker_id: WorkerId,
    },
    /// Worker offline
    WorkerOffline {
        /// Worker ID
        worker_id: WorkerId,
    },
    /// Task assigned
    TaskAssigned {
        /// Job ID
        job_id: JobId,
        /// Worker ID
        worker_id: WorkerId,
        /// Frame number
        frame: u32,
    },
    /// Task completed
    TaskCompleted {
        /// Job ID
        job_id: JobId,
        /// Worker ID
        worker_id: WorkerId,
        /// Frame number
        frame: u32,
    },
}

/// Main render farm coordinator
pub struct Coordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Job storage
    jobs: Arc<DashMap<JobId, Arc<RwLock<Job>>>>,
    /// Worker storage
    workers: Arc<DashMap<WorkerId, Arc<RwLock<Worker>>>>,
    /// Scheduler
    scheduler: Arc<Scheduler>,
    /// Event sender
    event_tx: mpsc::UnboundedSender<CoordinatorEvent>,
    /// Event receiver
    #[allow(dead_code)]
    event_rx: Arc<RwLock<mpsc::UnboundedReceiver<CoordinatorEvent>>>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl Coordinator {
    /// Create a new coordinator
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub async fn new(config: CoordinatorConfig) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let coordinator = Self {
            config: config.clone(),
            jobs: Arc::new(DashMap::new()),
            workers: Arc::new(DashMap::new()),
            scheduler: Arc::new(Scheduler::new(config.scheduling_algorithm)),
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
            running: Arc::new(RwLock::new(false)),
        };

        Ok(coordinator)
    }

    /// Start the coordinator
    pub async fn start(&self) -> Result<()> {
        *self.running.write() = true;
        info!("Render farm coordinator started");

        // Spawn background tasks
        self.spawn_heartbeat_monitor();
        self.spawn_job_scheduler();
        self.spawn_cleanup_task();

        Ok(())
    }

    /// Stop the coordinator
    pub async fn stop(&self) -> Result<()> {
        *self.running.write() = false;
        info!("Render farm coordinator stopped");
        Ok(())
    }

    /// Submit a new job
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_job(&self, submission: JobSubmission) -> Result<JobId> {
        let job = Job::new(submission);
        let job_id = job.id;

        info!("Submitting job: {}", job_id);

        // Store job
        self.jobs.insert(job_id, Arc::new(RwLock::new(job)));

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::JobSubmitted { job_id });

        // Split job into tasks
        self.split_job_into_tasks(job_id)?;

        Ok(job_id)
    }

    /// Register a new worker
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails
    pub async fn register_worker(&self, registration: WorkerRegistration) -> Result<WorkerId> {
        let worker = Worker::new(registration);
        let worker_id = worker.id;

        info!(
            "Registering worker: {} ({})",
            worker_id, worker.registration.hostname
        );

        // Store worker
        self.workers
            .insert(worker_id, Arc::new(RwLock::new(worker)));

        // Emit event
        let _ = self
            .event_tx
            .send(CoordinatorEvent::WorkerRegistered { worker_id });

        Ok(worker_id)
    }

    /// Get job by ID
    #[must_use]
    pub fn get_job(&self, job_id: JobId) -> Option<Arc<RwLock<Job>>> {
        self.jobs.get(&job_id).map(|entry| entry.value().clone())
    }

    /// Get worker by ID
    #[must_use]
    pub fn get_worker(&self, worker_id: WorkerId) -> Option<Arc<RwLock<Worker>>> {
        self.workers
            .get(&worker_id)
            .map(|entry| entry.value().clone())
    }

    /// List all jobs
    #[must_use]
    pub fn list_jobs(&self) -> Vec<JobId> {
        self.jobs.iter().map(|entry| *entry.key()).collect()
    }

    /// List all workers
    #[must_use]
    pub fn list_workers(&self) -> Vec<WorkerId> {
        self.workers.iter().map(|entry| *entry.key()).collect()
    }

    /// Get available workers
    #[must_use]
    pub fn get_available_workers(&self) -> Vec<WorkerId> {
        self.workers
            .iter()
            .filter(|entry| entry.value().read().is_available())
            .map(|entry| *entry.key())
            .collect()
    }

    /// Cancel a job
    ///
    /// # Errors
    ///
    /// Returns an error if job not found or cancellation fails
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        let job_arc = self
            .get_job(job_id)
            .ok_or_else(|| Error::JobNotFound(job_id.to_string()))?;

        let mut job = job_arc.write();

        if job.is_finished() {
            return Err(Error::InvalidStateTransition {
                from: job.state.to_string(),
                to: "Cancelled".to_string(),
            });
        }

        job.update_state(JobState::Cancelled)?;

        info!("Job cancelled: {}", job_id);

        Ok(())
    }

    /// Update worker heartbeat
    pub fn worker_heartbeat(&self, worker_id: WorkerId) -> Result<()> {
        let worker_arc = self
            .get_worker(worker_id)
            .ok_or_else(|| Error::WorkerNotFound(worker_id.to_string()))?;

        worker_arc.write().heartbeat();

        Ok(())
    }

    /// Split job into tasks
    fn split_job_into_tasks(&self, job_id: JobId) -> Result<()> {
        let job_arc = self
            .get_job(job_id)
            .ok_or_else(|| Error::JobNotFound(job_id.to_string()))?;

        let job = job_arc.read();

        // Get frame range from job type
        let frames = match &job.submission.job_type {
            crate::job::JobType::ImageSequence {
                start_frame,
                end_frame,
            } => (*start_frame..=*end_frame).collect::<Vec<_>>(),
            crate::job::JobType::VideoRender { .. } => {
                // Default to 100 frames for video render
                (1..=100).collect()
            }
            _ => {
                // Single frame for other job types
                vec![1]
            }
        };

        // Create tasks
        for frame in frames {
            let task = Task::new(job_id, frame, job.submission.priority);
            self.scheduler.enqueue(task);
        }

        debug!(
            "Split job {} into {} tasks",
            job_id,
            self.scheduler.queue_size()
        );

        Ok(())
    }

    /// Spawn heartbeat monitor
    fn spawn_heartbeat_monitor(&self) {
        let workers = self.workers.clone();
        let running = self.running.clone();
        let interval = self.config.heartbeat_interval;
        let _timeout = self.config.worker_timeout;
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while *running.read() {
                tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

                for entry in workers.iter() {
                    let worker_id = *entry.key();
                    let worker_arc = entry.value();
                    let mut worker = worker_arc.write();

                    if !worker.is_online() {
                        warn!("Worker {} is offline", worker_id);
                        worker.update_state(WorkerState::Offline);
                        let _ = event_tx.send(CoordinatorEvent::WorkerOffline { worker_id });
                    }
                }
            }
        });
    }

    /// Spawn job scheduler
    fn spawn_job_scheduler(&self) {
        let workers = self.workers.clone();
        let scheduler = self.scheduler.clone();
        let running = self.running.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while *running.read() {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                // Find available workers
                let available_workers: Vec<_> = workers
                    .iter()
                    .filter(|entry| entry.value().read().is_available())
                    .map(|entry| (*entry.key(), entry.value().clone()))
                    .collect();

                // Assign tasks to available workers
                for (worker_id, worker_arc) in available_workers {
                    let worker = worker_arc.read();

                    if let Some(task) = scheduler.schedule(&worker) {
                        drop(worker);

                        if let Ok(_assignment) = scheduler.assign(worker_id, task.clone()) {
                            let mut worker = worker_arc.write();
                            worker.update_state(WorkerState::Busy);
                            worker.current_job_id = Some(task.job_id.to_string());

                            let _ = event_tx.send(CoordinatorEvent::TaskAssigned {
                                job_id: task.job_id,
                                worker_id,
                                frame: task.frame,
                            });

                            debug!("Assigned task {} to worker {}", task.id, worker_id);
                        }
                    }
                }
            }
        });
    }

    /// Spawn cleanup task
    fn spawn_cleanup_task(&self) {
        let jobs = self.jobs.clone();
        let running = self.running.clone();
        let interval = self.config.cleanup_interval;

        tokio::spawn(async move {
            while *running.read() {
                tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

                // Clean up old finished jobs (older than 24 hours)
                let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);

                let mut to_remove = Vec::new();
                for entry in jobs.iter() {
                    let job = entry.value().read();
                    if job.is_finished() {
                        if let Some(completed_at) = job.completed_at {
                            if completed_at < cutoff {
                                to_remove.push(*entry.key());
                            }
                        }
                    }
                }

                for job_id in to_remove {
                    jobs.remove(&job_id);
                    debug!("Cleaned up old job: {}", job_id);
                }
            }
        });
    }

    /// Get statistics
    #[must_use]
    pub fn get_stats(&self) -> CoordinatorStats {
        let total_jobs = self.jobs.len();
        let mut active_jobs = 0;
        let mut completed_jobs = 0;
        let mut failed_jobs = 0;

        for entry in self.jobs.iter() {
            let job = entry.value().read();
            if job.is_active() {
                active_jobs += 1;
            } else if job.state == JobState::Completed {
                completed_jobs += 1;
            } else if job.state == JobState::Failed {
                failed_jobs += 1;
            }
        }

        let total_workers = self.workers.len();
        let mut idle_workers = 0;
        let mut busy_workers = 0;
        let mut offline_workers = 0;

        for entry in self.workers.iter() {
            let worker = entry.value().read();
            match worker.state {
                WorkerState::Idle => idle_workers += 1,
                WorkerState::Busy => busy_workers += 1,
                WorkerState::Offline => offline_workers += 1,
                _ => {}
            }
        }

        CoordinatorStats {
            total_jobs,
            active_jobs,
            completed_jobs,
            failed_jobs,
            total_workers,
            idle_workers,
            busy_workers,
            offline_workers,
            queue_size: self.scheduler.queue_size(),
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CoordinatorStats {
    /// Total jobs
    pub total_jobs: usize,
    /// Active jobs
    pub active_jobs: usize,
    /// Completed jobs
    pub completed_jobs: usize,
    /// Failed jobs
    pub failed_jobs: usize,
    /// Total workers
    pub total_workers: usize,
    /// Idle workers
    pub idle_workers: usize,
    /// Busy workers
    pub busy_workers: usize,
    /// Offline workers
    pub offline_workers: usize,
    /// Queue size
    pub queue_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobSubmission, Priority};
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_coordinator_creation() -> Result<()> {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config).await?;
        assert_eq!(coordinator.list_jobs().len(), 0);
        assert_eq!(coordinator.list_workers().len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_job_submission() -> Result<()> {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config).await?;

        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 10)
            .priority(Priority::Normal)
            .build()?;

        let job_id = coordinator.submit_job(submission).await?;

        assert!(coordinator.get_job(job_id).is_some());
        assert_eq!(coordinator.list_jobs().len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_worker_registration() -> Result<()> {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config).await?;

        let registration = WorkerRegistration {
            hostname: "worker01".to_string(),
            ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            port: 8080,
            capabilities: Default::default(),
            location: None,
            tags: std::collections::HashMap::new(),
        };

        let worker_id = coordinator.register_worker(registration).await?;

        assert!(coordinator.get_worker(worker_id).is_some());
        assert_eq!(coordinator.list_workers().len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_job_cancellation() -> Result<()> {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config).await?;

        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 10)
            .build()?;

        let job_id = coordinator.submit_job(submission).await?;

        // Update job state to queued
        {
            let job_arc = coordinator.get_job(job_id).expect("should succeed in test");
            let mut job = job_arc.write();
            job.update_state(JobState::Validating)?;
            job.update_state(JobState::Queued)?;
        }

        coordinator.cancel_job(job_id).await?;

        let job_arc = coordinator.get_job(job_id).expect("should succeed in test");
        let job = job_arc.read();
        assert_eq!(job.state, JobState::Cancelled);

        Ok(())
    }

    #[tokio::test]
    async fn test_coordinator_stats() -> Result<()> {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config).await?;

        let stats = coordinator.get_stats();
        assert_eq!(stats.total_jobs, 0);
        assert_eq!(stats.total_workers, 0);

        // Submit a job
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 10)
            .build()?;

        coordinator.submit_job(submission).await?;

        let stats = coordinator.get_stats();
        assert_eq!(stats.total_jobs, 1);

        Ok(())
    }
}
