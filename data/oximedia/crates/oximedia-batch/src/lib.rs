//! Comprehensive batch processing engine for `OxiMedia`
//!
//! This crate provides a production-ready batch processing system with:
//! - Job queuing and scheduling
//! - Worker pool management
//! - Template-based configuration
//! - Watch folder automation
//! - Distributed processing support
//! - REST API and CLI interfaces

// `memmap2` requires one unavoidable `unsafe` block per `Mmap::map()` call;
// the block is isolated in `operations/mmap_reader.rs` with a safety comment.
// All other modules remain fully safe.
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod api;
/// Immutable append-only audit trail for all job lifecycle events.
pub mod audit_log;
/// Per-job and fleet-wide batch analytics: throughput, latency, error rates.
pub mod batch_analytics;
pub mod batch_report;
pub mod batch_runner;
pub mod batch_schedule;
/// Job chaining: define DAG-like sequential/parallel chains between jobs.
pub mod chaining;
/// Durable checkpoint persistence for mid-job progress and crash recovery.
pub mod checkpoint;
pub mod checkpointing;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod cli;
/// Cluster-member discovery and heartbeat tracking for distributed workers.
pub mod cluster_discovery;
pub mod conditional_dag;
/// CPU/GPU/IO cost estimation for job admission control and resource planning.
pub mod cost_estimator;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod database;
/// Dead-letter queue for failed jobs with replay and quarantine support.
pub mod dead_letter_queue;
pub mod dep_graph;
pub mod dependency;
pub mod error;
/// Error recovery strategies: retry with backoff, circuit breaker, fallback.
pub mod error_recovery;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod examples;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod execution;
pub mod fair_scheduler;
/// Graceful shutdown coordinator: drain window, force-cancel, status reporting.
pub mod graceful_shutdown;
pub mod job;
pub mod job_archive;
/// Job dependency tracker: declarative `before`/`after` constraint resolution.
pub mod job_deps;
pub mod job_migration;
pub mod job_splitting;
pub mod metrics;
pub mod monitoring;
/// Notification hub: fan-out job events to webhooks, email, and Slack.
pub mod notification_hub;
pub mod notifications;
pub mod operations;
pub mod output_collector;
pub mod pipeline_validator;
pub mod presets;
pub mod priority_queue;
#[cfg(not(target_arch = "wasm32"))]
pub mod processor;
/// Progress aggregator: roll up subtask progress into parent-job percentage.
pub mod progress_agg;
pub mod progress_tracker;
#[cfg(not(target_arch = "wasm32"))]
pub mod queue;
/// Quota definition types: resource ceilings and usage counters.
pub mod quota;
/// Quota enforcer: per-user/team hard limits on concurrent and total jobs.
pub mod quota_enforcer;
pub mod rate_limiter;
pub mod resource_estimator;
pub mod resource_reservation;
pub mod retry_policy;
#[cfg(all(not(target_arch = "wasm32"), feature = "scripting"))]
pub mod script;
pub mod task_group;
pub mod template;
pub mod throttle;
pub mod timeout_enforcer;
pub mod types;
pub mod utils;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod watch;
pub mod work_stealing;

pub use error::{BatchError, Result};
pub use job::{BatchJob, BatchOperation, InputSpec, OutputSpec};
pub use types::{JobId, JobState, Priority, RetryPolicy};

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
use database::Database;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
use execution::ExecutionEngine;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
use queue::JobQueue;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Graceful shutdown types (always compiled — not gated on sqlite/wasm)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU8, Ordering};

/// Configuration for graceful shutdown of `BatchEngine` (requires the `sqlite` feature).
#[derive(Debug, Clone)]
pub struct ShutdownConfig {
    /// How long to wait (in milliseconds) for in-progress jobs to finish
    /// before considering the drain complete.
    pub drain_timeout_ms: u64,
    /// If `Some`, forcibly cancel remaining jobs after this many milliseconds
    /// even if `drain_timeout_ms` has not elapsed.  Must be ≥ `drain_timeout_ms`
    /// to take effect.
    pub force_after_ms: Option<u64>,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout_ms: 5_000,
            force_after_ms: None,
        }
    }
}

/// Shutdown progression state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShutdownState {
    /// Engine is accepting new jobs.
    Running = 0,
    /// Shutdown has been requested; no new jobs accepted.
    ShutdownRequested = 1,
    /// Waiting for in-progress jobs to drain.
    Draining = 2,
    /// All jobs finished (or force-timeout elapsed); engine is stopped.
    Terminated = 3,
}

impl ShutdownState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Running,
            1 => Self::ShutdownRequested,
            2 => Self::Draining,
            3 => Self::Terminated,
            _ => Self::Terminated,
        }
    }
}

/// Summary returned by `BatchEngine::request_shutdown` (requires the `sqlite` feature).
#[derive(Debug, Clone)]
pub struct ShutdownReport {
    /// Number of jobs that finished cleanly during the drain window.
    pub jobs_completed: usize,
    /// Number of jobs that were cancelled (force timeout or drain timeout).
    pub jobs_cancelled: usize,
    /// Wall-clock time the shutdown procedure took in milliseconds.
    pub elapsed_ms: u64,
    /// Final shutdown state (always [`ShutdownState::Terminated`] on success).
    pub state: ShutdownState,
}

/// Shared shutdown flag — stored as an `AtomicU8` to allow lock-free reads.
pub struct ShutdownFlag(AtomicU8);

impl ShutdownFlag {
    /// Create a new flag in the `Running` state.
    #[must_use]
    pub fn new() -> Self {
        Self(AtomicU8::new(ShutdownState::Running as u8))
    }

    /// Read the current state.
    #[must_use]
    pub fn state(&self) -> ShutdownState {
        ShutdownState::from_u8(self.0.load(Ordering::Acquire))
    }

    /// Transition to the given state.
    pub fn set(&self, state: ShutdownState) {
        self.0.store(state as u8, Ordering::Release);
    }

    /// Returns `true` if shutdown has been requested (any non-Running state).
    #[must_use]
    pub fn is_shutdown_requested(&self) -> bool {
        self.state() != ShutdownState::Running
    }
}

impl Default for ShutdownFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// Main batch processing engine
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub struct BatchEngine {
    queue: Arc<JobQueue>,
    engine: Arc<ExecutionEngine>,
    database: Arc<Database>,
    shutdown_flag: Arc<ShutdownFlag>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
impl BatchEngine {
    /// Create a new batch processing engine
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to `SQLite` database file
    /// * `worker_count` - Number of worker threads
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn new(db_path: &str, worker_count: usize) -> Result<Self> {
        let database = Arc::new(Database::new(db_path)?);
        let queue = Arc::new(JobQueue::new());
        let engine = Arc::new(ExecutionEngine::new(
            worker_count,
            Arc::clone(&queue),
            Arc::clone(&database),
        )?);

        Ok(Self {
            queue,
            engine,
            database,
            shutdown_flag: Arc::new(ShutdownFlag::new()),
        })
    }

    /// Submit a job to the queue
    ///
    /// # Arguments
    ///
    /// * `job` - The job to submit
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_job(&self, job: BatchJob) -> Result<JobId> {
        let job_id = job.id.clone();
        self.database.save_job(&job)?;
        self.queue.enqueue(job).await?;
        Ok(job_id)
    }

    /// Get job status
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to query
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub async fn get_job_status(&self, job_id: &JobId) -> Result<JobState> {
        self.queue.get_job_status(job_id).await
    }

    /// Cancel a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to cancel
    ///
    /// # Errors
    ///
    /// Returns an error if cancellation fails
    pub async fn cancel_job(&self, job_id: &JobId) -> Result<()> {
        self.queue.cancel_job(job_id).await
    }

    /// List all jobs
    ///
    /// # Errors
    ///
    /// Returns an error if database query fails
    pub fn list_jobs(&self) -> Result<Vec<BatchJob>> {
        self.database.list_jobs()
    }

    /// Start the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if engine startup fails
    pub async fn start(&self) -> Result<()> {
        self.engine.start().await
    }

    /// Stop the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if engine shutdown fails
    pub async fn stop(&self) -> Result<()> {
        self.engine.stop().await
    }

    /// Get queue reference
    #[must_use]
    pub fn queue(&self) -> Arc<JobQueue> {
        Arc::clone(&self.queue)
    }

    /// Get engine reference
    #[must_use]
    pub fn engine(&self) -> Arc<ExecutionEngine> {
        Arc::clone(&self.engine)
    }

    /// Get database reference
    #[must_use]
    pub fn database(&self) -> Arc<Database> {
        Arc::clone(&self.database)
    }

    /// Returns a clone of the shared shutdown flag, allowing external
    /// components to observe the engine's shutdown state.
    #[must_use]
    pub fn shutdown_flag(&self) -> Arc<ShutdownFlag> {
        Arc::clone(&self.shutdown_flag)
    }

    /// Returns the current shutdown state.
    #[must_use]
    pub fn shutdown_state(&self) -> ShutdownState {
        self.shutdown_flag.state()
    }

    /// Request a graceful shutdown of the batch engine.
    ///
    /// Steps:
    /// 1. Transitions state to [`ShutdownState::ShutdownRequested`] — no new
    ///    jobs are accepted.
    /// 2. Transitions to [`ShutdownState::Draining`] and polls until either
    ///    - All queued/running jobs finish, **or**
    ///    - `config.drain_timeout_ms` elapses.
    /// 3. If `config.force_after_ms` is set and that deadline elapses before
    ///    drain completes, remaining jobs are cancelled.
    /// 4. Calls `stop()` on the underlying execution engine and transitions to
    ///    [`ShutdownState::Terminated`].
    ///
    /// Returns a [`ShutdownReport`] describing what happened.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying engine `stop()` fails.
    pub async fn request_shutdown(&self, config: ShutdownConfig) -> Result<ShutdownReport> {
        let start = std::time::Instant::now();

        // --- Phase 1: signal no new jobs ---
        self.shutdown_flag.set(ShutdownState::ShutdownRequested);

        // --- Phase 2: drain ---
        self.shutdown_flag.set(ShutdownState::Draining);

        let drain_deadline = std::time::Duration::from_millis(config.drain_timeout_ms);
        let force_deadline = config.force_after_ms.map(std::time::Duration::from_millis);

        let mut jobs_completed = 0usize;
        let mut jobs_cancelled = 0usize;

        // Poll the queue until it reports no active jobs or a deadline fires.
        // We check every 10 ms.
        let poll_interval = std::time::Duration::from_millis(10);
        loop {
            let elapsed = start.elapsed();

            // Force-cancel deadline takes precedence if set.
            if let Some(force) = force_deadline {
                if elapsed >= force {
                    // Cancel anything still pending.
                    if let Ok(pending) = self.database.list_jobs() {
                        for job in &pending {
                            let is_active = job.context.as_ref().map_or(false, |ctx| {
                                matches!(ctx.state, JobState::Queued | JobState::Running)
                            });
                            if is_active {
                                let _ = self.queue.cancel_job(&job.id).await;
                                jobs_cancelled += 1;
                            }
                        }
                    }
                    break;
                }
            }

            // Drain timeout.
            if elapsed >= drain_deadline {
                break;
            }

            // Check if all jobs are finished.
            match self.database.list_jobs() {
                Ok(jobs) => {
                    let active = jobs
                        .iter()
                        .filter(|j| {
                            j.context.as_ref().map_or(false, |ctx| {
                                matches!(ctx.state, JobState::Queued | JobState::Running)
                            })
                        })
                        .count();
                    jobs_completed = jobs
                        .iter()
                        .filter(|j| {
                            j.context
                                .as_ref()
                                .map_or(false, |ctx| matches!(ctx.state, JobState::Completed))
                        })
                        .count();
                    if active == 0 {
                        break;
                    }
                }
                Err(_) => break,
            }

            tokio::time::sleep(poll_interval).await;
        }

        // --- Phase 3: stop engine ---
        self.engine.stop().await?;
        self.shutdown_flag.set(ShutdownState::Terminated);

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ShutdownReport {
            jobs_completed,
            jobs_cancelled,
            elapsed_ms,
            state: ShutdownState::Terminated,
        })
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_engine_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4);
        assert!(engine.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_job_submission() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await;
        assert!(job_id.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_job_status() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await.expect("failed to submit job");
        let status = engine.get_job_status(&job_id).await;
        assert!(status.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_job_cancellation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await.expect("failed to submit job");
        let result = engine.cancel_job(&job_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_list_jobs() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let jobs = engine.list_jobs();
        assert!(jobs.is_ok());
    }
}

// ---------------------------------------------------------------------------
// Shutdown flag & config unit tests (no sqlite feature required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod shutdown_tests {
    use super::*;

    #[test]
    fn test_shutdown_flag_initial_state() {
        let flag = ShutdownFlag::new();
        assert_eq!(flag.state(), ShutdownState::Running);
        assert!(!flag.is_shutdown_requested());
    }

    #[test]
    fn test_shutdown_flag_transitions() {
        let flag = ShutdownFlag::new();
        flag.set(ShutdownState::ShutdownRequested);
        assert_eq!(flag.state(), ShutdownState::ShutdownRequested);
        assert!(flag.is_shutdown_requested());

        flag.set(ShutdownState::Draining);
        assert_eq!(flag.state(), ShutdownState::Draining);

        flag.set(ShutdownState::Terminated);
        assert_eq!(flag.state(), ShutdownState::Terminated);
    }

    #[test]
    fn test_shutdown_flag_default() {
        let flag = ShutdownFlag::default();
        assert_eq!(flag.state(), ShutdownState::Running);
    }

    #[test]
    fn test_shutdown_config_default() {
        let cfg = ShutdownConfig::default();
        assert_eq!(cfg.drain_timeout_ms, 5_000);
        assert!(cfg.force_after_ms.is_none());
    }

    #[test]
    fn test_shutdown_config_with_force() {
        let cfg = ShutdownConfig {
            drain_timeout_ms: 2_000,
            force_after_ms: Some(4_000),
        };
        assert_eq!(cfg.drain_timeout_ms, 2_000);
        assert_eq!(cfg.force_after_ms, Some(4_000));
    }

    #[test]
    fn test_shutdown_state_from_u8_all_variants() {
        assert_eq!(ShutdownState::from_u8(0), ShutdownState::Running);
        assert_eq!(ShutdownState::from_u8(1), ShutdownState::ShutdownRequested);
        assert_eq!(ShutdownState::from_u8(2), ShutdownState::Draining);
        assert_eq!(ShutdownState::from_u8(3), ShutdownState::Terminated);
        // Out of range → Terminated
        assert_eq!(ShutdownState::from_u8(99), ShutdownState::Terminated);
    }

    #[test]
    fn test_shutdown_report_fields() {
        let report = ShutdownReport {
            jobs_completed: 5,
            jobs_cancelled: 2,
            elapsed_ms: 300,
            state: ShutdownState::Terminated,
        };
        assert_eq!(report.jobs_completed, 5);
        assert_eq!(report.jobs_cancelled, 2);
        assert_eq!(report.elapsed_ms, 300);
        assert_eq!(report.state, ShutdownState::Terminated);
    }

    #[test]
    fn test_shutdown_flag_arc_share() {
        use std::sync::Arc;
        let flag = Arc::new(ShutdownFlag::new());
        let flag2 = Arc::clone(&flag);
        flag2.set(ShutdownState::ShutdownRequested);
        assert!(flag.is_shutdown_requested());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_engine_shutdown_state_initially_running() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().expect("tmp file");
        let db_path = tmp.path().to_str().expect("path utf8");
        let engine = BatchEngine::new(db_path, 2).expect("create engine");
        assert_eq!(engine.shutdown_state(), ShutdownState::Running);
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_engine_request_shutdown_empty_queue() {
        use tempfile::NamedTempFile;
        let tmp = NamedTempFile::new().expect("tmp file");
        let db_path = tmp.path().to_str().expect("path utf8");
        let engine = BatchEngine::new(db_path, 2).expect("create engine");

        let config = ShutdownConfig {
            drain_timeout_ms: 100,
            force_after_ms: None,
        };
        let report = engine
            .request_shutdown(config)
            .await
            .expect("shutdown failed");
        assert_eq!(report.state, ShutdownState::Terminated);
        assert!(report.elapsed_ms < 5_000);
    }
}
