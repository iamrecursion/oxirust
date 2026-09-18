// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job type definitions and execution logic.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority jobs
    Low = 0,
    /// Normal priority jobs
    Normal = 1,
    /// High priority jobs
    High = 2,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
        }
    }
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is pending execution
    Pending,
    /// Job is currently running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job is waiting for dependencies
    Waiting,
    /// Job is scheduled for future execution
    Scheduled,
}

impl fmt::Display for JobStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Waiting => write!(f, "waiting"),
            Self::Scheduled => write!(f, "scheduled"),
        }
    }
}

/// Transcoding parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeParams {
    /// Input file path
    pub input: String,
    /// Output file path
    pub output: String,
    /// Video codec
    pub video_codec: String,
    /// Audio codec
    pub audio_codec: String,
    /// Video bitrate in bits per second
    pub video_bitrate: u64,
    /// Audio bitrate in bits per second
    pub audio_bitrate: u64,
    /// Target resolution (width, height)
    pub resolution: Option<(u32, u32)>,
    /// Frame rate
    pub framerate: Option<f64>,
    /// Preset (e.g., "fast", "medium", "slow")
    pub preset: String,
    /// Hardware acceleration
    pub hw_accel: Option<String>,
}

/// Thumbnail generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailParams {
    /// Input file path
    pub input: String,
    /// Output directory
    pub output_dir: String,
    /// Number of thumbnails to generate
    pub count: u32,
    /// Thumbnail width
    pub width: u32,
    /// Thumbnail height
    pub height: u32,
    /// Quality (1-100)
    pub quality: u8,
}

/// Sprite sheet parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpriteSheetParams {
    /// Input file path
    pub input: String,
    /// Output file path
    pub output: String,
    /// Number of frames
    pub frame_count: u32,
    /// Columns in sprite sheet
    pub columns: u32,
    /// Rows in sprite sheet
    pub rows: u32,
    /// Frame width
    pub frame_width: u32,
    /// Frame height
    pub frame_height: u32,
}

/// Analysis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisParams {
    /// Input file path
    pub input: String,
    /// Analysis type
    pub analysis_type: AnalysisType,
    /// Output file for results
    pub output: Option<String>,
}

/// Analysis types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisType {
    /// Quality analysis
    Quality,
    /// Scene detection
    Scenes,
    /// Audio analysis
    Audio,
    /// Video analysis
    Video,
    /// Full analysis
    Full,
}

/// Batch operation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchParams {
    /// Input files
    pub inputs: Vec<String>,
    /// Operation to perform
    pub operation: String,
    /// Operation-specific parameters
    pub params: HashMap<String, String>,
}

/// Job payload containing job-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobPayload {
    /// Transcoding job
    Transcode(TranscodeParams),
    /// Thumbnail generation job
    Thumbnail(ThumbnailParams),
    /// Sprite sheet generation job
    SpriteSheet(SpriteSheetParams),
    /// Analysis job
    Analysis(AnalysisParams),
    /// Batch operation job
    Batch(BatchParams),
}

impl JobPayload {
    /// Returns the job type name
    #[must_use]
    pub fn job_type(&self) -> &str {
        match self {
            Self::Transcode(_) => "transcode",
            Self::Thumbnail(_) => "thumbnail",
            Self::SpriteSheet(_) => "sprite_sheet",
            Self::Analysis(_) => "analysis",
            Self::Batch(_) => "batch",
        }
    }
}

/// Retry policy for failed jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial backoff duration in seconds
    pub initial_backoff_secs: u64,
    /// Maximum backoff duration in seconds
    pub max_backoff_secs: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_secs: 60,
            max_backoff_secs: 3600,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Calculate backoff duration for a given attempt
    #[must_use]
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        #[allow(clippy::cast_precision_loss)]
        let backoff_secs =
            self.initial_backoff_secs as f64 * self.backoff_multiplier.powi(attempt as i32);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap
        )]
        let backoff_secs = backoff_secs.min(self.max_backoff_secs as f64) as i64;
        Duration::seconds(backoff_secs)
    }
}

/// Resource quotas for job execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum CPU cores
    pub max_cpu_cores: Option<u32>,
    /// Maximum memory in bytes
    pub max_memory_bytes: Option<u64>,
    /// Maximum GPU count
    pub max_gpu_count: Option<u32>,
    /// Maximum execution time in seconds
    pub max_execution_time_secs: Option<u64>,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_cpu_cores: None,
            max_memory_bytes: None,
            max_gpu_count: None,
            max_execution_time_secs: Some(3600),
        }
    }
}

/// Conditional execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// Always execute
    Always,
    /// Execute if dependency completed successfully
    OnSuccess(Uuid),
    /// Execute if dependency failed
    OnFailure(Uuid),
    /// Execute if dependency completed (success or failure)
    OnCompletion(Uuid),
    /// Execute if all dependencies completed successfully
    AllSuccess(Vec<Uuid>),
    /// Execute if any dependency completed successfully
    AnySuccess(Vec<Uuid>),
}

/// Job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique job identifier
    pub id: Uuid,
    /// Job name
    pub name: String,
    /// Job priority
    pub priority: Priority,
    /// Job status
    pub status: JobStatus,
    /// Job payload
    pub payload: JobPayload,
    /// Retry policy
    pub retry_policy: RetryPolicy,
    /// Resource quotas
    pub resource_quota: ResourceQuota,
    /// Execution condition
    pub condition: Condition,
    /// Dependencies (job IDs that must complete before this job)
    pub dependencies: Vec<Uuid>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Scheduled execution time
    pub scheduled_at: Option<DateTime<Utc>>,
    /// Started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Deadline for execution
    pub deadline: Option<DateTime<Utc>>,
    /// Number of attempts
    pub attempts: u32,
    /// Last error message
    pub error: Option<String>,
    /// Progress percentage (0-100)
    pub progress: u8,
    /// Worker ID executing the job
    pub worker_id: Option<String>,
    /// Next job IDs in pipeline
    pub next_jobs: Vec<Uuid>,
}

impl Job {
    /// Create a new job
    #[must_use]
    pub fn new(name: String, priority: Priority, payload: JobPayload) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            priority,
            status: JobStatus::Pending,
            payload,
            retry_policy: RetryPolicy::default(),
            resource_quota: ResourceQuota::default(),
            condition: Condition::Always,
            dependencies: Vec::new(),
            tags: Vec::new(),
            created_at: Utc::now(),
            scheduled_at: None,
            started_at: None,
            completed_at: None,
            deadline: None,
            attempts: 0,
            error: None,
            progress: 0,
            worker_id: None,
            next_jobs: Vec::new(),
        }
    }

    /// Set retry policy
    #[must_use]
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Set resource quota
    #[must_use]
    pub fn with_resource_quota(mut self, quota: ResourceQuota) -> Self {
        self.resource_quota = quota;
        self
    }

    /// Set execution condition
    #[must_use]
    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.condition = condition;
        self
    }

    /// Add dependency
    #[must_use]
    pub fn with_dependency(mut self, dep_id: Uuid) -> Self {
        self.dependencies.push(dep_id);
        self
    }

    /// Add tag
    #[must_use]
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Set scheduled time
    #[must_use]
    pub fn with_schedule(mut self, scheduled_at: DateTime<Utc>) -> Self {
        self.scheduled_at = Some(scheduled_at);
        self.status = JobStatus::Scheduled;
        self
    }

    /// Set deadline
    #[must_use]
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add next job in pipeline
    #[must_use]
    pub fn with_next_job(mut self, next_id: Uuid) -> Self {
        self.next_jobs.push(next_id);
        self
    }

    /// Check if job can be executed based on conditions
    #[must_use]
    pub fn can_execute(&self, completed_jobs: &HashMap<Uuid, JobStatus>) -> bool {
        match &self.condition {
            Condition::Always => true,
            Condition::OnSuccess(id) => completed_jobs.get(id) == Some(&JobStatus::Completed),
            Condition::OnFailure(id) => completed_jobs.get(id) == Some(&JobStatus::Failed),
            Condition::OnCompletion(id) => {
                matches!(
                    completed_jobs.get(id),
                    Some(JobStatus::Completed | JobStatus::Failed)
                )
            }
            Condition::AllSuccess(ids) => ids
                .iter()
                .all(|id| completed_jobs.get(id) == Some(&JobStatus::Completed)),
            Condition::AnySuccess(ids) => ids
                .iter()
                .any(|id| completed_jobs.get(id) == Some(&JobStatus::Completed)),
        }
    }

    /// Check if job is ready to execute (all dependencies met)
    #[must_use]
    pub fn is_ready(&self, completed_jobs: &HashMap<Uuid, JobStatus>) -> bool {
        if !self.dependencies.is_empty()
            && !self
                .dependencies
                .iter()
                .all(|id| completed_jobs.get(id) == Some(&JobStatus::Completed))
        {
            return false;
        }

        self.can_execute(completed_jobs)
    }

    /// Check if job has exceeded its deadline
    #[must_use]
    pub fn is_past_deadline(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Utc::now() > deadline
        } else {
            false
        }
    }

    /// Check if job should be retried
    #[must_use]
    pub fn should_retry(&self) -> bool {
        self.status == JobStatus::Failed && self.attempts < self.retry_policy.max_attempts
    }

    /// Calculate next retry time
    #[must_use]
    pub fn next_retry_time(&self) -> Option<DateTime<Utc>> {
        if self.should_retry() {
            let backoff = self.retry_policy.backoff_duration(self.attempts);
            Some(Utc::now() + backoff)
        } else {
            None
        }
    }

    /// Update progress
    pub fn update_progress(&mut self, progress: u8) {
        self.progress = progress.min(100);
    }

    /// Mark job as started
    pub fn mark_started(&mut self, worker_id: String) {
        self.status = JobStatus::Running;
        self.started_at = Some(Utc::now());
        self.worker_id = Some(worker_id);
        self.attempts += 1;
    }

    /// Mark job as completed
    pub fn mark_completed(&mut self) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.progress = 100;
    }

    /// Mark job as failed
    pub fn mark_failed(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error = Some(error);
    }

    /// Mark job as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = JobStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }

    /// Reset job for retry
    pub fn reset_for_retry(&mut self) {
        self.status = JobStatus::Pending;
        self.started_at = None;
        self.worker_id = None;
        self.progress = 0;
    }
}

/// Job builder for creating complex job configurations
pub struct JobBuilder {
    job: Job,
}

impl JobBuilder {
    /// Create a new job builder
    #[must_use]
    pub fn new(name: String, priority: Priority, payload: JobPayload) -> Self {
        Self {
            job: Job::new(name, priority, payload),
        }
    }

    /// Set retry policy
    #[must_use]
    pub fn retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.job = self.job.with_retry_policy(policy);
        self
    }

    /// Set resource quota
    #[must_use]
    pub fn resource_quota(mut self, quota: ResourceQuota) -> Self {
        self.job = self.job.with_resource_quota(quota);
        self
    }

    /// Set execution condition
    #[must_use]
    pub fn condition(mut self, condition: Condition) -> Self {
        self.job = self.job.with_condition(condition);
        self
    }

    /// Add dependency
    #[must_use]
    pub fn dependency(mut self, dep_id: Uuid) -> Self {
        self.job = self.job.with_dependency(dep_id);
        self
    }

    /// Add tag
    #[must_use]
    pub fn tag(mut self, tag: String) -> Self {
        self.job = self.job.with_tag(tag);
        self
    }

    /// Set scheduled time
    #[must_use]
    pub fn schedule(mut self, scheduled_at: DateTime<Utc>) -> Self {
        self.job = self.job.with_schedule(scheduled_at);
        self
    }

    /// Set deadline
    #[must_use]
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.job = self.job.with_deadline(deadline);
        self
    }

    /// Add next job in pipeline
    #[must_use]
    pub fn next_job(mut self, next_id: Uuid) -> Self {
        self.job = self.job.with_next_job(next_id);
        self
    }

    /// Build the job
    #[must_use]
    pub fn build(self) -> Job {
        self.job
    }
}
