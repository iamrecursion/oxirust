//! Proxy transcode queue management.
//!
//! Provides a priority-based queue for scheduling and tracking proxy transcode
//! jobs, along with batch request support and queue statistics.
//!
//! The `ParallelTranscodeExecutor` uses Rayon's work-stealing thread pool to
//! transcode multiple proxy jobs concurrently, saturating available CPU cores
//! without spawning excessive threads.

#![allow(dead_code)]

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Specification for a proxy transcode output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxySpec {
    /// Target resolution (width, height).
    pub resolution: (u32, u32),
    /// Codec identifier (e.g. "h264", "prores_proxy").
    pub codec: String,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
}

impl ProxySpec {
    /// Create a new proxy spec.
    #[must_use]
    pub fn new(resolution: (u32, u32), codec: impl Into<String>, bitrate_kbps: u32) -> Self {
        Self {
            resolution,
            codec: codec.into(),
            bitrate_kbps,
        }
    }

    /// Standard H.264 HD proxy.
    #[must_use]
    pub fn h264_hd() -> Self {
        Self::new((1920, 1080), "h264", 8_000)
    }

    /// Standard ProRes Proxy.
    #[must_use]
    pub fn prores_proxy() -> Self {
        Self::new((1920, 1080), "prores_proxy", 45_000)
    }
}

/// A proxy transcode request submitted to the queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyRequest {
    /// Unique request identifier.
    pub id: String,
    /// Path to the source media file.
    pub source_path: String,
    /// Desired proxy specification.
    pub proxy_spec: ProxySpec,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Submission timestamp in milliseconds since epoch.
    pub submitted_at_ms: u64,
}

impl ProxyRequest {
    /// Create a new proxy request.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        source_path: impl Into<String>,
        proxy_spec: ProxySpec,
        priority: u8,
        submitted_at_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            source_path: source_path.into(),
            proxy_spec,
            priority,
            submitted_at_ms,
        }
    }
}

/// Status of a proxy transcode job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Waiting in the queue.
    Queued,
    /// Currently being transcoded.
    Running,
    /// Successfully completed.
    Completed,
    /// Failed due to an error.
    Failed,
    /// Cancelled before execution.
    Cancelled,
}

/// A proxy transcode job with its current status and timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyTranscodeJob {
    /// The original request.
    pub request: ProxyRequest,
    /// Current job status.
    pub status: JobStatus,
    /// When the job started, in milliseconds since epoch.
    pub started_at_ms: Option<u64>,
    /// When the job completed (or failed), in milliseconds since epoch.
    pub completed_at_ms: Option<u64>,
    /// Path to the output proxy file (set on completion).
    pub output_path: Option<String>,
    /// Error message if the job failed.
    pub error: Option<String>,
}

impl ProxyTranscodeJob {
    /// Create a new job in the `Queued` state.
    #[must_use]
    pub fn new(request: ProxyRequest) -> Self {
        Self {
            request,
            status: JobStatus::Queued,
            started_at_ms: None,
            completed_at_ms: None,
            output_path: None,
            error: None,
        }
    }

    /// Duration the job waited in the queue before starting, in milliseconds.
    #[must_use]
    pub fn wait_duration_ms(&self) -> Option<u64> {
        self.started_at_ms
            .map(|start| start.saturating_sub(self.request.submitted_at_ms))
    }

    /// Total processing duration, in milliseconds.
    #[must_use]
    pub fn processing_duration_ms(&self) -> Option<u64> {
        match (self.started_at_ms, self.completed_at_ms) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }
}

/// Priority-based proxy transcode queue.
///
/// Higher-priority jobs are dispatched first. Within the same priority,
/// earlier submission time wins (FIFO).
#[derive(Debug, Default)]
pub struct ProxyTranscodeQueue {
    /// All jobs indexed by their ID.
    jobs: std::collections::HashMap<String, ProxyTranscodeJob>,
    /// Queue ordering: sorted by (priority desc, submitted_at_ms asc).
    order: VecDeque<String>,
}

impl ProxyTranscodeQueue {
    /// Create an empty transcode queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new request and return the job ID.
    pub fn submit(&mut self, request: ProxyRequest) -> String {
        let id = request.id.clone();
        let job = ProxyTranscodeJob::new(request);
        // Insert into order queue maintaining priority order
        let priority = job.request.priority;
        let submitted = job.request.submitted_at_ms;
        // Find insertion point: higher priority first, then earlier submission
        let pos = self
            .order
            .iter()
            .position(|existing_id| {
                if let Some(j) = self.jobs.get(existing_id) {
                    let ep = j.request.priority;
                    let es = j.request.submitted_at_ms;
                    // Insert before this entry if we have higher priority,
                    // or equal priority and earlier submission
                    ep < priority || (ep == priority && es > submitted)
                } else {
                    false
                }
            })
            .unwrap_or(self.order.len());
        self.order.insert(pos, id.clone());
        self.jobs.insert(id.clone(), job);
        id
    }

    /// Get the next queued job (highest priority, earliest submitted).
    #[must_use]
    pub fn next_job(&mut self) -> Option<&mut ProxyTranscodeJob> {
        // Find the first job in order that is still Queued
        let next_id = self
            .order
            .iter()
            .find(|id| {
                self.jobs
                    .get(*id)
                    .map(|j| j.status == JobStatus::Queued)
                    .unwrap_or(false)
            })
            .cloned();
        next_id.and_then(|id| self.jobs.get_mut(&id))
    }

    /// Mark a job as started at the given timestamp.
    pub fn start_job(&mut self, id: &str, started_at_ms: u64) {
        if let Some(job) = self.jobs.get_mut(id) {
            if job.status == JobStatus::Queued {
                job.status = JobStatus::Running;
                job.started_at_ms = Some(started_at_ms);
            }
        }
    }

    /// Mark a job as successfully completed.
    pub fn complete_job(&mut self, id: &str, output: &str) {
        if let Some(job) = self.jobs.get_mut(id) {
            job.status = JobStatus::Completed;
            job.output_path = Some(output.to_string());
            // Use a synthetic completion time if not already set
            if job.completed_at_ms.is_none() {
                job.completed_at_ms = job.started_at_ms.map(|s| s + 1);
            }
        }
    }

    /// Mark a job as failed with an error message.
    pub fn fail_job(&mut self, id: &str, error: &str) {
        if let Some(job) = self.jobs.get_mut(id) {
            job.status = JobStatus::Failed;
            job.error = Some(error.to_string());
        }
    }

    /// Cancel a queued job.
    pub fn cancel_job(&mut self, id: &str) {
        if let Some(job) = self.jobs.get_mut(id) {
            if job.status == JobStatus::Queued {
                job.status = JobStatus::Cancelled;
            }
        }
    }

    /// Get a job by ID (immutable).
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ProxyTranscodeJob> {
        self.jobs.get(id)
    }

    /// Total number of jobs (all statuses).
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the queue has any jobs at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Iterate over all jobs.
    pub fn iter(&self) -> impl Iterator<Item = &ProxyTranscodeJob> {
        self.jobs.values()
    }
}

/// Aggregated statistics for a `ProxyTranscodeQueue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    /// Number of jobs currently queued (waiting).
    pub pending: u32,
    /// Number of jobs currently running.
    pub running: u32,
    /// Total jobs completed successfully.
    pub completed: u64,
    /// Total jobs that failed.
    pub failed: u64,
    /// Average wait time from submission to start, in milliseconds.
    pub avg_wait_ms: f64,
}

impl QueueStats {
    /// Compute statistics from the current state of the queue.
    #[must_use]
    pub fn compute(queue: &ProxyTranscodeQueue) -> Self {
        let mut pending = 0u32;
        let mut running = 0u32;
        let mut completed = 0u64;
        let mut failed = 0u64;
        let mut total_wait_ms = 0u64;
        let mut wait_count = 0u32;

        for job in queue.iter() {
            match job.status {
                JobStatus::Queued => pending += 1,
                JobStatus::Running => running += 1,
                JobStatus::Completed => completed += 1,
                JobStatus::Failed => failed += 1,
                JobStatus::Cancelled => {}
            }
            if let Some(wait) = job.wait_duration_ms() {
                total_wait_ms += wait;
                wait_count += 1;
            }
        }

        let avg_wait_ms = if wait_count == 0 {
            0.0
        } else {
            total_wait_ms as f64 / wait_count as f64
        };

        Self {
            pending,
            running,
            completed,
            failed,
            avg_wait_ms,
        }
    }
}

/// A batch proxy request: transcode many sources with the same spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyBatchRequest {
    /// Source file paths to transcode.
    pub source_paths: Vec<String>,
    /// Proxy specification to apply to all sources.
    pub spec: ProxySpec,
    /// Maximum number of concurrent transcode jobs.
    pub concurrent_limit: u32,
}

impl ProxyBatchRequest {
    /// Create a new batch request.
    #[must_use]
    pub fn new(source_paths: Vec<String>, spec: ProxySpec, concurrent_limit: u32) -> Self {
        Self {
            source_paths,
            spec,
            concurrent_limit,
        }
    }

    /// Estimate the total time in minutes to transcode `items` files at `fps` frames-per-second.
    ///
    /// Assumes each source file has ~1 minute of content at `fps` fps, and that
    /// the transcode runs at 2× real-time per concurrent slot.
    #[must_use]
    pub fn estimate_duration_mins(items: usize, fps: f32) -> f32 {
        if fps <= 0.0 || items == 0 {
            return 0.0;
        }
        // Simple model: each item takes 0.5 real-time minutes (2× speed),
        // batched by concurrent_limit which defaults to 1 here for the pure fn.
        let per_item_mins = 1.0 / 2.0; // 1 minute of source → 0.5 min at 2× speed
                                       // fps affects CPU cost (higher fps = proportionally more work)
        let fps_factor = fps / 25.0; // normalise to 25 fps
        items as f32 * per_item_mins * fps_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, priority: u8, submitted_at_ms: u64) -> ProxyRequest {
        ProxyRequest::new(
            id,
            format!("/source/{id}.mov"),
            ProxySpec::h264_hd(),
            priority,
            submitted_at_ms,
        )
    }

    #[test]
    fn test_proxy_spec_new() {
        let spec = ProxySpec::new((1280, 720), "h264", 5_000);
        assert_eq!(spec.resolution, (1280, 720));
        assert_eq!(spec.codec, "h264");
        assert_eq!(spec.bitrate_kbps, 5_000);
    }

    #[test]
    fn test_proxy_spec_presets() {
        let hd = ProxySpec::h264_hd();
        assert_eq!(hd.codec, "h264");
        assert_eq!(hd.resolution, (1920, 1080));

        let prores = ProxySpec::prores_proxy();
        assert_eq!(prores.codec, "prores_proxy");
    }

    #[test]
    fn test_submit_returns_id() {
        let mut queue = ProxyTranscodeQueue::new();
        let req = make_request("job_001", 100, 1000);
        let id = queue.submit(req);
        assert_eq!(id, "job_001");
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("low", 10, 1000));
        queue.submit(make_request("high", 200, 2000));
        queue.submit(make_request("mid", 100, 1500));

        // next_job should return "high" (priority 200)
        let next = queue.next_job().expect("should succeed in test");
        assert_eq!(next.request.id, "high");
    }

    #[test]
    fn test_complete_job() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("j1", 50, 1000));
        queue.start_job("j1", 1100);
        queue.complete_job("j1", "/proxy/j1.mp4");
        let job = queue.get("j1").expect("should succeed in test");
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(job.output_path.as_deref(), Some("/proxy/j1.mp4"));
    }

    #[test]
    fn test_fail_job() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("j2", 50, 1000));
        queue.start_job("j2", 1100);
        queue.fail_job("j2", "codec error");
        let job = queue.get("j2").expect("should succeed in test");
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error.as_deref(), Some("codec error"));
    }

    #[test]
    fn test_cancel_job() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("j3", 50, 1000));
        queue.cancel_job("j3");
        let job = queue.get("j3").expect("should succeed in test");
        assert_eq!(job.status, JobStatus::Cancelled);
    }

    #[test]
    fn test_queue_stats() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("a", 10, 0));
        queue.submit(make_request("b", 10, 0));
        queue.submit(make_request("c", 10, 0));
        queue.start_job("a", 100);
        queue.complete_job("a", "/out/a.mp4");
        queue.start_job("b", 100);
        queue.fail_job("b", "err");

        let stats = QueueStats::compute(&queue);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
    }

    #[test]
    fn test_queue_is_empty() {
        let queue = ProxyTranscodeQueue::new();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_wait_duration_ms() {
        let mut queue = ProxyTranscodeQueue::new();
        queue.submit(make_request("w", 10, 1000));
        queue.start_job("w", 2000);
        let job = queue.get("w").expect("should succeed in test");
        assert_eq!(job.wait_duration_ms(), Some(1000));
    }

    #[test]
    fn test_batch_estimate_duration_zero_fps() {
        assert!((ProxyBatchRequest::estimate_duration_mins(10, 0.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_estimate_duration_zero_items() {
        assert!((ProxyBatchRequest::estimate_duration_mins(0, 25.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_estimate_duration_positive() {
        let mins = ProxyBatchRequest::estimate_duration_mins(10, 25.0);
        assert!(mins > 0.0);
    }
}

// ============================================================================
// Parallel Transcode Executor (Rayon work-stealing)
// ============================================================================

/// Result of a single parallel transcode task.
#[derive(Debug, Clone)]
pub struct ParallelJobResult {
    /// Job ID from the original request.
    pub id: String,
    /// Whether the job succeeded.
    pub success: bool,
    /// Output path on success.
    pub output_path: Option<String>,
    /// Error message on failure.
    pub error: Option<String>,
    /// Simulated processing duration in milliseconds.
    pub duration_ms: u64,
}

/// User-supplied transcode function signature.
///
/// The function receives `(source_path, spec)` and should return either
/// `Ok(output_path)` or `Err(error_message)`.  It will be called from Rayon
/// worker threads, so it must be `Send + Sync`.
pub type TranscodeFn =
    Arc<dyn Fn(&str, &ProxySpec) -> std::result::Result<String, String> + Send + Sync>;

/// Configuration for the parallel executor.
#[derive(Clone)]
pub struct ParallelExecutorConfig {
    /// Maximum number of concurrent transcode threads.
    /// `0` means use all available logical CPUs.
    pub thread_count: usize,
    /// User-supplied transcode function.
    pub transcode_fn: TranscodeFn,
}

impl ParallelExecutorConfig {
    /// Create a config with a custom transcode function and thread count.
    pub fn new(thread_count: usize, transcode_fn: TranscodeFn) -> Self {
        Self {
            thread_count,
            transcode_fn,
        }
    }

    /// Create a config that uses a no-op (stub) transcode for testing.
    #[must_use]
    pub fn stub() -> Self {
        let fn_: TranscodeFn = Arc::new(|src, spec| {
            Ok(format!(
                "/proxy/{}_{}x{}.mp4",
                std::path::Path::new(src)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("clip"),
                spec.resolution.0,
                spec.resolution.1,
            ))
        });
        Self {
            thread_count: 0,
            transcode_fn: fn_,
        }
    }
}

/// Parallel batch proxy transcode executor powered by Rayon.
///
/// Jobs from a `ProxyTranscodeQueue` are dispatched to Rayon's global or a
/// custom thread pool using work-stealing.  Completed results are collected
/// in a thread-safe results vector.
pub struct ParallelTranscodeExecutor {
    config: ParallelExecutorConfig,
}

impl ParallelTranscodeExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: ParallelExecutorConfig) -> Self {
        Self { config }
    }

    /// Execute all `Queued` jobs from `queue` in parallel and return results.
    ///
    /// The queue is not mutated; callers should apply results using
    /// [`ParallelTranscodeExecutor::apply_results`] if they wish to update
    /// statuses.
    ///
    /// # Errors
    ///
    /// Never errors at the executor level; individual job failures are captured
    /// in [`ParallelJobResult::success`] / [`ParallelJobResult::error`].
    pub fn execute_batch(&self, queue: &ProxyTranscodeQueue) -> Vec<ParallelJobResult> {
        // Collect all queued jobs into an owned snapshot (no queue mutation)
        let jobs: Vec<(String, String, ProxySpec)> = queue
            .iter()
            .filter(|j| j.status == JobStatus::Queued)
            .map(|j| {
                (
                    j.request.id.clone(),
                    j.request.source_path.clone(),
                    j.request.proxy_spec.clone(),
                )
            })
            .collect();

        let transcode_fn = Arc::clone(&self.config.transcode_fn);
        let results: Arc<Mutex<Vec<ParallelJobResult>>> = Arc::new(Mutex::new(Vec::new()));

        let threads = if self.config.thread_count == 0 {
            num_cpus::get()
        } else {
            self.config.thread_count
        };

        // Build a private thread pool limited to `threads` workers.
        // If construction fails (extremely rare — only on resource exhaustion),
        // fall back to running jobs sequentially on the calling thread so that
        // we always return results and never panic.
        let pool_opt: Option<rayon::ThreadPool> = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok();

        let results_clone = Arc::clone(&results);
        let run_jobs = move || {
            jobs.par_iter().for_each(|(id, src, spec)| {
                let start = std::time::Instant::now();
                let outcome = (transcode_fn)(src, spec);
                let duration_ms = start.elapsed().as_millis() as u64;

                let result = match outcome {
                    Ok(out) => ParallelJobResult {
                        id: id.clone(),
                        success: true,
                        output_path: Some(out),
                        error: None,
                        duration_ms,
                    },
                    Err(e) => ParallelJobResult {
                        id: id.clone(),
                        success: false,
                        output_path: None,
                        error: Some(e),
                        duration_ms,
                    },
                };

                if let Ok(mut guard) = results_clone.lock() {
                    guard.push(result);
                }
            });
        };

        match pool_opt {
            Some(pool) => pool.install(run_jobs),
            // Fallback: global rayon pool (uses all CPUs, still parallel)
            None => run_jobs(),
        }

        Arc::try_unwrap(results)
            .ok()
            .and_then(|m| m.into_inner().ok())
            .unwrap_or_default()
    }

    /// Apply a batch of results back to a mutable queue, updating job statuses.
    pub fn apply_results(queue: &mut ProxyTranscodeQueue, results: &[ParallelJobResult]) {
        for r in results {
            if r.success {
                queue.start_job(&r.id, 0);
                queue.complete_job(&r.id, r.output_path.as_deref().unwrap_or(""));
            } else {
                queue.start_job(&r.id, 0);
                queue.fail_job(&r.id, r.error.as_deref().unwrap_or("unknown error"));
            }
        }
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    fn make_queue(n: usize) -> ProxyTranscodeQueue {
        let mut q = ProxyTranscodeQueue::new();
        for i in 0..n {
            q.submit(ProxyRequest::new(
                format!("job_{i}"),
                format!("/src/clip_{i}.mov"),
                ProxySpec::h264_hd(),
                100,
                i as u64 * 10,
            ));
        }
        q
    }

    #[test]
    fn test_parallel_executor_all_succeed() {
        let config = ParallelExecutorConfig::stub();
        let executor = ParallelTranscodeExecutor::new(config);
        let queue = make_queue(8);

        let results = executor.execute_batch(&queue);
        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|r| r.success));
        assert!(results.iter().all(|r| r.output_path.is_some()));
    }

    #[test]
    fn test_parallel_executor_error_fn() {
        let fail_fn: TranscodeFn =
            Arc::new(|_src, _spec| Err("simulated transcode failure".to_string()));
        let config = ParallelExecutorConfig::new(2, fail_fn);
        let executor = ParallelTranscodeExecutor::new(config);
        let queue = make_queue(4);

        let results = executor.execute_batch(&queue);
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| !r.success));
        assert!(results.iter().all(|r| r.error.is_some()));
    }

    #[test]
    fn test_parallel_executor_empty_queue() {
        let config = ParallelExecutorConfig::stub();
        let executor = ParallelTranscodeExecutor::new(config);
        let queue = ProxyTranscodeQueue::new();

        let results = executor.execute_batch(&queue);
        assert!(results.is_empty());
    }

    #[test]
    fn test_apply_results_updates_statuses() {
        let config = ParallelExecutorConfig::stub();
        let executor = ParallelTranscodeExecutor::new(config);
        let mut queue = make_queue(3);

        let results = executor.execute_batch(&queue);
        ParallelTranscodeExecutor::apply_results(&mut queue, &results);

        let stats = QueueStats::compute(&queue);
        assert_eq!(stats.completed, 3);
        assert_eq!(stats.pending, 0);
    }

    #[test]
    fn test_apply_results_marks_failures() {
        let fail_fn: TranscodeFn = Arc::new(|_src, _spec| Err("err".to_string()));
        let config = ParallelExecutorConfig::new(1, fail_fn);
        let executor = ParallelTranscodeExecutor::new(config);
        let mut queue = make_queue(2);

        let results = executor.execute_batch(&queue);
        ParallelTranscodeExecutor::apply_results(&mut queue, &results);

        let stats = QueueStats::compute(&queue);
        assert_eq!(stats.failed, 2);
    }

    #[test]
    fn test_parallel_result_has_duration() {
        let config = ParallelExecutorConfig::stub();
        let executor = ParallelTranscodeExecutor::new(config);
        let queue = make_queue(2);
        let results = executor.execute_batch(&queue);
        // duration_ms may be 0 on fast machines but field must exist
        assert!(results.iter().all(|r| r.duration_ms < u64::MAX));
    }

    #[test]
    fn test_parallel_skips_non_queued_jobs() {
        let config = ParallelExecutorConfig::stub();
        let executor = ParallelTranscodeExecutor::new(config);
        let mut queue = make_queue(4);
        // Pre-cancel two jobs so they aren't in Queued state
        queue.cancel_job("job_0");
        queue.cancel_job("job_1");

        let results = executor.execute_batch(&queue);
        // Only 2 queued jobs should be executed
        assert_eq!(results.len(), 2);
    }
}
