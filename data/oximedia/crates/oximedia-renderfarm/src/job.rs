// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job management for render farm.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Unique identifier for a job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(Uuid);

impl JobId {
    /// Create a new job ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Low priority (batch processing)
    Low = 0,
    /// Normal priority (default)
    #[default]
    Normal = 1,
    /// High priority (important projects)
    High = 2,
    /// Urgent priority (critical deadlines)
    Urgent = 3,
}

/// Job state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    /// Job is pending
    Pending,
    /// Job is being validated
    Validating,
    /// Job is queued for execution
    Queued,
    /// Job is currently rendering
    Rendering,
    /// Job is paused
    Paused,
    /// Job is being verified
    Verifying,
    /// Job is being assembled
    Assembling,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
}

impl std::fmt::Display for JobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Validating => write!(f, "Validating"),
            Self::Queued => write!(f, "Queued"),
            Self::Rendering => write!(f, "Rendering"),
            Self::Paused => write!(f, "Paused"),
            Self::Verifying => write!(f, "Verifying"),
            Self::Assembling => write!(f, "Assembling"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Job type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobType {
    /// Video rendering from timeline
    VideoRender {
        /// Timeline reference
        timeline_id: String,
    },
    /// Image sequence rendering
    ImageSequence {
        /// Frame range
        start_frame: u32,
        /// End frame
        end_frame: u32,
    },
    /// Audio mixing and rendering
    AudioMix {
        /// Audio tracks
        tracks: Vec<String>,
    },
    /// Distributed transcoding
    Transcoding {
        /// Input file
        input: PathBuf,
        /// Output format
        format: String,
    },
    /// Compositing job
    Compositing {
        /// Layers to composite
        layers: Vec<String>,
    },
    /// Simulation rendering
    Simulation {
        /// Simulation type
        sim_type: String,
        /// Cache path
        cache_path: PathBuf,
    },
}

/// Job submission request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSubmission {
    /// Project file path
    pub project_file: PathBuf,
    /// Job type
    pub job_type: JobType,
    /// Priority level
    pub priority: Priority,
    /// Deadline (optional)
    pub deadline: Option<DateTime<Utc>>,
    /// Budget limit (optional)
    pub budget_limit: Option<f64>,
    /// Dependencies (asset paths)
    pub dependencies: Vec<PathBuf>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Target pool (optional)
    pub pool_id: Option<String>,
    /// Chunk size for splitting
    pub chunk_size: Option<u32>,
}

impl JobSubmission {
    /// Create a new job submission builder
    #[must_use]
    pub fn builder() -> JobSubmissionBuilder {
        JobSubmissionBuilder::default()
    }
}

/// Builder for job submissions
#[derive(Default)]
pub struct JobSubmissionBuilder {
    project_file: Option<PathBuf>,
    job_type: Option<JobType>,
    priority: Priority,
    deadline: Option<DateTime<Utc>>,
    budget_limit: Option<f64>,
    dependencies: Vec<PathBuf>,
    metadata: HashMap<String, String>,
    pool_id: Option<String>,
    chunk_size: Option<u32>,
}

impl JobSubmissionBuilder {
    /// Set project file
    #[must_use]
    pub fn project_file<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.project_file = Some(path.into());
        self
    }

    /// Set job type to video render
    #[must_use]
    pub fn video_render(mut self, timeline_id: String) -> Self {
        self.job_type = Some(JobType::VideoRender { timeline_id });
        self
    }

    /// Set job type to image sequence
    #[must_use]
    pub fn image_sequence(mut self, start: u32, end: u32) -> Self {
        self.job_type = Some(JobType::ImageSequence {
            start_frame: start,
            end_frame: end,
        });
        self
    }

    /// Set frame range (for image sequence)
    #[must_use]
    pub fn frame_range(mut self, start: u32, end: u32) -> Self {
        self.job_type = Some(JobType::ImageSequence {
            start_frame: start,
            end_frame: end,
        });
        self
    }

    /// Set priority
    #[must_use]
    pub const fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline
    #[must_use]
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set budget limit
    #[must_use]
    pub fn budget_limit(mut self, limit: f64) -> Self {
        self.budget_limit = Some(limit);
        self
    }

    /// Add dependency
    #[must_use]
    pub fn dependency<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.dependencies.push(path.into());
        self
    }

    /// Add metadata
    #[must_use]
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set target pool
    #[must_use]
    pub fn pool_id(mut self, pool_id: String) -> Self {
        self.pool_id = Some(pool_id);
        self
    }

    /// Set chunk size
    #[must_use]
    pub const fn chunk_size(mut self, size: u32) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Build the job submission
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing
    pub fn build(self) -> Result<JobSubmission> {
        let project_file = self
            .project_file
            .ok_or_else(|| Error::Configuration("project_file is required".to_string()))?;
        let job_type = self
            .job_type
            .ok_or_else(|| Error::Configuration("job_type is required".to_string()))?;

        Ok(JobSubmission {
            project_file,
            job_type,
            priority: self.priority,
            deadline: self.deadline,
            budget_limit: self.budget_limit,
            dependencies: self.dependencies,
            metadata: self.metadata,
            pool_id: self.pool_id,
            chunk_size: self.chunk_size,
        })
    }
}

/// Render job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Job ID
    pub id: JobId,
    /// Job submission details
    pub submission: JobSubmission,
    /// Current state
    pub state: JobState,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Start time (when rendering began)
    pub started_at: Option<DateTime<Utc>>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Progress (0.0 to 1.0)
    pub progress: f64,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Actual cost
    pub actual_cost: f64,
    /// Assigned workers
    pub assigned_workers: Vec<String>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Retry count
    pub retry_count: u32,
    /// Output files
    pub output_files: Vec<PathBuf>,
}

impl Job {
    /// Create a new job from submission
    #[must_use]
    pub fn new(submission: JobSubmission) -> Self {
        Self {
            id: JobId::new(),
            submission,
            state: JobState::Pending,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            estimated_cost: 0.0,
            actual_cost: 0.0,
            assigned_workers: Vec::new(),
            error_message: None,
            retry_count: 0,
            output_files: Vec::new(),
        }
    }

    /// Check if job is active (rendering or queued)
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(
            self.state,
            JobState::Pending
                | JobState::Queued
                | JobState::Rendering
                | JobState::Validating
                | JobState::Verifying
                | JobState::Assembling
        )
    }

    /// Check if job is finished
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(
            self.state,
            JobState::Completed | JobState::Failed | JobState::Cancelled
        )
    }

    /// Update job state
    ///
    /// # Errors
    ///
    /// Returns an error if the state transition is invalid
    pub fn update_state(&mut self, new_state: JobState) -> Result<()> {
        // Validate state transition
        let valid = match (&self.state, &new_state) {
            (JobState::Pending, JobState::Validating) => true,
            (JobState::Validating, JobState::Queued) => true,
            (JobState::Validating, JobState::Failed) => true,
            (JobState::Queued, JobState::Rendering) => true,
            (JobState::Queued, JobState::Cancelled) => true,
            (JobState::Rendering, JobState::Paused) => true,
            (JobState::Rendering, JobState::Verifying) => true,
            (JobState::Rendering, JobState::Failed) => true,
            (JobState::Paused, JobState::Rendering) => true,
            (JobState::Paused, JobState::Cancelled) => true,
            (JobState::Verifying, JobState::Assembling) => true,
            (JobState::Verifying, JobState::Failed) => true,
            (JobState::Assembling, JobState::Completed) => true,
            (JobState::Assembling, JobState::Failed) => true,
            _ => false,
        };

        if !valid {
            return Err(Error::InvalidStateTransition {
                from: self.state.to_string(),
                to: new_state.to_string(),
            });
        }

        self.state = new_state;

        // Update timestamps
        match new_state {
            JobState::Rendering if self.started_at.is_none() => {
                self.started_at = Some(Utc::now());
            }
            JobState::Completed | JobState::Failed | JobState::Cancelled => {
                self.completed_at = Some(Utc::now());
            }
            _ => {}
        }

        Ok(())
    }

    /// Update progress
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Calculate ETA based on progress and elapsed time
    #[must_use]
    pub fn calculate_eta(&self) -> Option<DateTime<Utc>> {
        if let Some(started_at) = self.started_at {
            if self.progress > 0.0 && self.progress < 1.0 {
                let elapsed = Utc::now() - started_at;
                let total_time = elapsed.num_seconds() as f64 / self.progress;
                let remaining = total_time - elapsed.num_seconds() as f64;
                return Some(Utc::now() + chrono::Duration::seconds(remaining as i64));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id_generation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Urgent);
    }

    #[test]
    fn test_job_submission_builder() -> Result<()> {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .priority(Priority::High)
            .build()?;

        assert_eq!(submission.priority, Priority::High);
        assert_eq!(
            submission.project_file,
            PathBuf::from("/path/to/project.blend")
        );
        Ok(())
    }

    #[test]
    fn test_job_creation() {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .build()
            .expect("should succeed in test");

        let job = Job::new(submission);
        assert_eq!(job.state, JobState::Pending);
        assert_eq!(job.progress, 0.0);
        assert!(job.is_active());
    }

    #[test]
    fn test_job_state_transitions() -> Result<()> {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .build()?;

        let mut job = Job::new(submission);

        // Valid transitions
        job.update_state(JobState::Validating)?;
        assert_eq!(job.state, JobState::Validating);

        job.update_state(JobState::Queued)?;
        assert_eq!(job.state, JobState::Queued);

        job.update_state(JobState::Rendering)?;
        assert_eq!(job.state, JobState::Rendering);
        assert!(job.started_at.is_some());

        Ok(())
    }

    #[test]
    fn test_invalid_state_transition() -> Result<()> {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .build()?;

        let mut job = Job::new(submission);

        // Invalid transition: Pending -> Completed
        let result = job.update_state(JobState::Completed);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_job_progress_update() {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .build()
            .expect("should succeed in test");

        let mut job = Job::new(submission);
        job.update_progress(0.5);
        assert_eq!(job.progress, 0.5);

        // Test clamping
        job.update_progress(1.5);
        assert_eq!(job.progress, 1.0);

        job.update_progress(-0.5);
        assert_eq!(job.progress, 0.0);
    }

    #[test]
    fn test_job_is_finished() {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .build()
            .expect("should succeed in test");

        let mut job = Job::new(submission);
        assert!(!job.is_finished());

        job.state = JobState::Completed;
        assert!(job.is_finished());

        job.state = JobState::Failed;
        assert!(job.is_finished());
    }

    #[test]
    fn test_job_builder_missing_fields() {
        let result = JobSubmission::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_job_metadata() -> Result<()> {
        let submission = JobSubmission::builder()
            .project_file("/path/to/project.blend")
            .frame_range(1, 100)
            .metadata("client".to_string(), "Acme Corp".to_string())
            .metadata("project".to_string(), "Commercial".to_string())
            .build()?;

        assert_eq!(submission.metadata.len(), 2);
        assert_eq!(
            submission
                .metadata
                .get("client")
                .expect("should succeed in test"),
            "Acme Corp"
        );
        Ok(())
    }
}
