//! Job management and queuing for transcode operations.

use crate::{Result, TranscodeConfig, TranscodeError, TranscodeOutput};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Status of a transcode job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscodeStatus {
    /// Job is queued and waiting to start.
    Queued,
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed with an error.
    Failed,
    /// Job was cancelled.
    Cancelled,
    /// Job is paused.
    Paused,
}

/// Priority levels for transcode jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum JobPriority {
    /// Low priority (background processing).
    Low = 0,
    /// Normal priority (default).
    #[default]
    Normal = 1,
    /// High priority (time-sensitive).
    High = 2,
    /// Critical priority (highest).
    Critical = 3,
}

/// Configuration for a transcode job.
#[derive(Debug, Clone)]
pub struct TranscodeJobConfig {
    /// The transcode configuration.
    pub config: TranscodeConfig,
    /// Job priority.
    pub priority: JobPriority,
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Timeout for the job (None = no timeout).
    pub timeout: Option<Duration>,
    /// Job metadata (tags, labels, etc.).
    pub metadata: std::collections::HashMap<String, String>,
}

impl TranscodeJobConfig {
    /// Creates a new job configuration.
    #[must_use]
    pub fn new(config: TranscodeConfig) -> Self {
        Self {
            config,
            priority: JobPriority::Normal,
            max_retries: 3,
            timeout: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Sets the job priority.
    #[must_use]
    pub fn with_priority(mut self, priority: JobPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the maximum number of retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sets the job timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Adds metadata to the job.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A transcode job with state tracking.
#[derive(Debug, Clone)]
pub struct TranscodeJob {
    /// Unique job ID.
    pub id: String,
    /// Job configuration.
    pub config: TranscodeJobConfig,
    /// Current status.
    pub status: TranscodeStatus,
    /// Number of retry attempts made.
    pub retry_count: u32,
    /// Time when the job was created.
    pub created_at: SystemTime,
    /// Time when the job started.
    pub started_at: Option<SystemTime>,
    /// Time when the job completed or failed.
    pub completed_at: Option<SystemTime>,
    /// Error message if the job failed.
    pub error: Option<String>,
    /// Output if the job completed successfully.
    pub output: Option<TranscodeOutput>,
    /// Progress percentage (0-100).
    pub progress: f64,
}

impl TranscodeJob {
    /// Creates a new transcode job.
    #[must_use]
    pub fn new(config: TranscodeJobConfig) -> Self {
        Self {
            id: Self::generate_id(),
            config,
            status: TranscodeStatus::Queued,
            retry_count: 0,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            error: None,
            output: None,
            progress: 0.0,
        }
    }

    /// Generates a unique job ID.
    fn generate_id() -> String {
        use std::time::UNIX_EPOCH;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();

        format!("job_{timestamp}")
    }

    /// Marks the job as started.
    pub fn start(&mut self) {
        self.status = TranscodeStatus::Running;
        self.started_at = Some(SystemTime::now());
    }

    /// Marks the job as completed with output.
    pub fn complete(&mut self, output: TranscodeOutput) {
        self.status = TranscodeStatus::Completed;
        self.completed_at = Some(SystemTime::now());
        self.output = Some(output);
        self.progress = 100.0;
    }

    /// Marks the job as failed with an error message.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = TranscodeStatus::Failed;
        self.completed_at = Some(SystemTime::now());
        self.error = Some(error.into());
    }

    /// Marks the job as cancelled.
    pub fn cancel(&mut self) {
        self.status = TranscodeStatus::Cancelled;
        self.completed_at = Some(SystemTime::now());
    }

    /// Pauses the job.
    pub fn pause(&mut self) {
        if self.status == TranscodeStatus::Running {
            self.status = TranscodeStatus::Paused;
        }
    }

    /// Resumes the job.
    pub fn resume(&mut self) {
        if self.status == TranscodeStatus::Paused {
            self.status = TranscodeStatus::Running;
        }
    }

    /// Updates the job progress.
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 100.0);
    }

    /// Increments the retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Checks if the job can be retried.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.status == TranscodeStatus::Failed && self.retry_count < self.config.max_retries
    }

    /// Gets the elapsed time since the job started.
    #[must_use]
    pub fn elapsed_time(&self) -> Option<Duration> {
        self.started_at
            .and_then(|start| SystemTime::now().duration_since(start).ok())
    }

    /// Gets the total time from creation to completion.
    #[must_use]
    pub fn total_time(&self) -> Option<Duration> {
        self.completed_at
            .and_then(|end| end.duration_since(self.created_at).ok())
    }

    /// Checks if the job has timed out.
    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        if let Some(timeout) = self.config.timeout {
            if let Some(elapsed) = self.elapsed_time() {
                return elapsed > timeout;
            }
        }
        false
    }

    /// Gets a human-readable status description.
    #[must_use]
    pub fn status_string(&self) -> String {
        match self.status {
            TranscodeStatus::Queued => "Queued".to_string(),
            TranscodeStatus::Running => format!("Running ({:.1}%)", self.progress),
            TranscodeStatus::Completed => "Completed".to_string(),
            TranscodeStatus::Failed => {
                if let Some(ref error) = self.error {
                    format!("Failed: {error}")
                } else {
                    "Failed".to_string()
                }
            }
            TranscodeStatus::Cancelled => "Cancelled".to_string(),
            TranscodeStatus::Paused => format!("Paused ({:.1}%)", self.progress),
        }
    }
}

/// Job queue for managing multiple transcode jobs.
pub struct JobQueue {
    jobs: Vec<TranscodeJob>,
    max_concurrent: usize,
}

impl JobQueue {
    /// Creates a new job queue.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            jobs: Vec::new(),
            max_concurrent,
        }
    }

    /// Adds a job to the queue.
    pub fn enqueue(&mut self, job: TranscodeJob) {
        self.jobs.push(job);
        self.sort_by_priority();
    }

    /// Gets the next job to execute.
    #[must_use]
    pub fn dequeue(&mut self) -> Option<TranscodeJob> {
        let running_count = self
            .jobs
            .iter()
            .filter(|j| j.status == TranscodeStatus::Running)
            .count();

        if running_count >= self.max_concurrent {
            return None;
        }

        // Find first queued job
        if let Some(index) = self
            .jobs
            .iter()
            .position(|j| j.status == TranscodeStatus::Queued)
        {
            let mut job = self.jobs.remove(index);
            job.start();
            self.jobs.push(job.clone());
            Some(job)
        } else {
            None
        }
    }

    /// Gets the number of jobs in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Checks if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Gets the number of running jobs.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == TranscodeStatus::Running)
            .count()
    }

    /// Gets the number of queued jobs.
    #[must_use]
    pub fn queued_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == TranscodeStatus::Queued)
            .count()
    }

    /// Cancels a job by ID.
    #[allow(dead_code)]
    pub fn cancel_job(&mut self, job_id: &str) -> Result<()> {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.cancel();
            Ok(())
        } else {
            Err(TranscodeError::JobError(format!("Job not found: {job_id}")))
        }
    }

    /// Gets a job by ID.
    #[must_use]
    pub fn get_job(&self, job_id: &str) -> Option<&TranscodeJob> {
        #[allow(dead_code)]
        self.jobs.iter().find(|j| j.id == job_id)
    }

    /// Clears completed and failed jobs.
    pub fn clear_finished(&mut self) {
        self.jobs.retain(|j| {
            !matches!(
                j.status,
                TranscodeStatus::Completed | TranscodeStatus::Failed | TranscodeStatus::Cancelled
            )
        });
    }

    fn sort_by_priority(&mut self) {
        self.jobs.sort_by(|a, b| {
            b.config
                .priority
                .cmp(&a.config.priority)
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_creation() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default());
        let job = TranscodeJob::new(config);

        assert_eq!(job.status, TranscodeStatus::Queued);
        assert_eq!(job.retry_count, 0);
        assert_eq!(job.progress, 0.0);
        assert!(job.started_at.is_none());
        assert!(job.completed_at.is_none());
    }

    #[test]
    fn test_job_lifecycle() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default());
        let mut job = TranscodeJob::new(config);

        // Start job
        job.start();
        assert_eq!(job.status, TranscodeStatus::Running);
        assert!(job.started_at.is_some());

        // Update progress
        job.update_progress(50.0);
        assert_eq!(job.progress, 50.0);

        // Complete job
        let output = TranscodeOutput {
            output_path: "test.mp4".to_string(),
            file_size: 1000,
            duration: 60.0,
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            encoding_time: 30.0,
            speed_factor: 2.0,
        };
        job.complete(output);

        assert_eq!(job.status, TranscodeStatus::Completed);
        assert_eq!(job.progress, 100.0);
        assert!(job.completed_at.is_some());
        assert!(job.output.is_some());
    }

    #[test]
    fn test_job_failure() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default());
        let mut job = TranscodeJob::new(config);

        job.start();
        job.fail("Test error");

        assert_eq!(job.status, TranscodeStatus::Failed);
        assert_eq!(job.error, Some("Test error".to_string()));
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_job_retry() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default()).with_max_retries(3);
        let mut job = TranscodeJob::new(config);

        job.fail("Error");
        assert!(job.can_retry());

        job.increment_retry();
        assert_eq!(job.retry_count, 1);
        assert!(job.can_retry());

        job.increment_retry();
        job.increment_retry();
        assert_eq!(job.retry_count, 3);
        assert!(!job.can_retry());
    }

    #[test]
    fn test_job_pause_resume() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default());
        let mut job = TranscodeJob::new(config);

        job.start();
        assert_eq!(job.status, TranscodeStatus::Running);

        job.pause();
        assert_eq!(job.status, TranscodeStatus::Paused);

        job.resume();
        assert_eq!(job.status, TranscodeStatus::Running);
    }

    #[test]
    fn test_job_queue() {
        let mut queue = JobQueue::new(2);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        let config1 = TranscodeJobConfig::new(TranscodeConfig::default());
        let config2 = TranscodeJobConfig::new(TranscodeConfig::default());

        queue.enqueue(TranscodeJob::new(config1));
        queue.enqueue(TranscodeJob::new(config2));

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());
        assert_eq!(queue.queued_count(), 2);
        assert_eq!(queue.running_count(), 0);
    }

    #[test]
    fn test_job_queue_priority() {
        let mut queue = JobQueue::new(5);

        let low =
            TranscodeJobConfig::new(TranscodeConfig::default()).with_priority(JobPriority::Low);
        let high =
            TranscodeJobConfig::new(TranscodeConfig::default()).with_priority(JobPriority::High);
        let normal =
            TranscodeJobConfig::new(TranscodeConfig::default()).with_priority(JobPriority::Normal);

        queue.enqueue(TranscodeJob::new(low));
        queue.enqueue(TranscodeJob::new(high));
        queue.enqueue(TranscodeJob::new(normal));

        // High priority should be first
        let next = queue.dequeue().expect("should succeed in test");
        assert_eq!(next.config.priority, JobPriority::High);
    }

    #[test]
    fn test_job_queue_clear_finished() {
        let mut queue = JobQueue::new(5);

        let config = TranscodeJobConfig::new(TranscodeConfig::default());
        let mut job = TranscodeJob::new(config);
        job.complete(TranscodeOutput {
            output_path: "test.mp4".to_string(),
            file_size: 1000,
            duration: 60.0,
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            encoding_time: 30.0,
            speed_factor: 2.0,
        });

        queue.enqueue(job);
        assert_eq!(queue.len(), 1);

        queue.clear_finished();
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_job_config_builder() {
        let config = TranscodeJobConfig::new(TranscodeConfig::default())
            .with_priority(JobPriority::High)
            .with_max_retries(5)
            .with_timeout(Duration::from_secs(3600))
            .with_metadata("user", "test_user")
            .with_metadata("project", "test_project");

        assert_eq!(config.priority, JobPriority::High);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.timeout, Some(Duration::from_secs(3600)));
        assert_eq!(config.metadata.get("user"), Some(&"test_user".to_string()));
        assert_eq!(
            config.metadata.get("project"),
            Some(&"test_project".to_string())
        );
    }
}
