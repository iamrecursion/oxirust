//! Core types for batch processing

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique job identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(String);

impl JobId {
    /// Create a new job ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a job ID from a string
    #[must_use]
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the string representation
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for JobId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for JobId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum Priority {
    /// Low priority
    Low = 0,
    /// Normal priority
    #[default]
    Normal = 1,
    /// High priority
    High = 2,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
        }
    }
}

/// Job execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    /// Job is queued and waiting
    Queued,
    /// Job is currently running
    Running,
    /// Job completed successfully
    Completed,
    /// Job failed
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job is waiting for dependencies
    Pending,
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Pending => write!(f, "Pending"),
        }
    }
}

/// Retry policy for failed jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Delay between retries in seconds
    pub retry_delay_secs: u64,
    /// Whether to use exponential backoff
    pub exponential_backoff: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            retry_delay_secs: 60,
            exponential_backoff: true,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy
    #[must_use]
    pub const fn new(max_attempts: u32, retry_delay_secs: u64, exponential_backoff: bool) -> Self {
        Self {
            max_attempts,
            retry_delay_secs,
            exponential_backoff,
        }
    }

    /// No retry policy
    #[must_use]
    pub const fn none() -> Self {
        Self {
            max_attempts: 0,
            retry_delay_secs: 0,
            exponential_backoff: false,
        }
    }

    /// Get delay for a specific attempt
    #[must_use]
    pub fn get_delay(&self, attempt: u32) -> u64 {
        if self.exponential_backoff {
            self.retry_delay_secs * 2_u64.pow(attempt)
        } else {
            self.retry_delay_secs
        }
    }
}

/// Job schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Schedule {
    /// Execute immediately
    #[default]
    Immediate,
    /// Execute at a specific time
    At(DateTime<Utc>),
    /// Execute after a delay (seconds)
    After(u64),
    /// Recurring execution (cron-like)
    Recurring {
        /// Cron expression
        expression: String,
    },
}

/// Resource requirements for a job
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceRequirements {
    /// Number of CPU cores required
    pub cpu_cores: Option<usize>,
    /// Memory requirement in MB
    pub memory_mb: Option<usize>,
    /// GPU requirement
    pub gpu: bool,
    /// Disk space requirement in MB
    pub disk_space_mb: Option<usize>,
}

/// Job execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobContext {
    /// Job ID
    pub job_id: JobId,
    /// Job name
    pub job_name: String,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Start time
    pub started_at: Option<DateTime<Utc>>,
    /// Completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Current state
    pub state: JobState,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Error message if failed
    pub error: Option<String>,
}

impl JobContext {
    /// Create a new job context
    #[must_use]
    pub fn new(job_id: JobId, job_name: String) -> Self {
        Self {
            job_id,
            job_name,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            state: JobState::Queued,
            retry_count: 0,
            error: None,
        }
    }

    /// Mark job as started
    pub fn mark_started(&mut self) {
        self.started_at = Some(Utc::now());
        self.state = JobState::Running;
    }

    /// Mark job as completed
    pub fn mark_completed(&mut self) {
        self.completed_at = Some(Utc::now());
        self.state = JobState::Completed;
    }

    /// Mark job as failed
    pub fn mark_failed(&mut self, error: String) {
        self.completed_at = Some(Utc::now());
        self.state = JobState::Failed;
        self.error = Some(error);
    }

    /// Mark job as cancelled
    pub fn mark_cancelled(&mut self) {
        self.completed_at = Some(Utc::now());
        self.state = JobState::Cancelled;
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Get duration if job is completed
    #[must_use]
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.started_at
            .and_then(|start| self.completed_at.map(|end| end - start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id_creation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_job_id_from_string() {
        let id = JobId::from_string("test-id".to_string());
        assert_eq!(id.as_str(), "test-id");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_default() {
        let priority = Priority::default();
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn test_job_state_display() {
        assert_eq!(JobState::Queued.to_string(), "Queued");
        assert_eq!(JobState::Running.to_string(), "Running");
        assert_eq!(JobState::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.retry_delay_secs, 60);
        assert!(policy.exponential_backoff);
    }

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy::new(5, 10, true);
        assert_eq!(policy.get_delay(0), 10);
        assert_eq!(policy.get_delay(1), 20);
        assert_eq!(policy.get_delay(2), 40);
    }

    #[test]
    fn test_retry_policy_linear() {
        let policy = RetryPolicy::new(5, 10, false);
        assert_eq!(policy.get_delay(0), 10);
        assert_eq!(policy.get_delay(1), 10);
        assert_eq!(policy.get_delay(2), 10);
    }

    #[test]
    fn test_retry_policy_none() {
        let policy = RetryPolicy::none();
        assert_eq!(policy.max_attempts, 0);
    }

    #[test]
    fn test_job_context_creation() {
        let ctx = JobContext::new(JobId::new(), "test-job".to_string());
        assert_eq!(ctx.state, JobState::Queued);
        assert_eq!(ctx.retry_count, 0);
        assert!(ctx.started_at.is_none());
    }

    #[test]
    fn test_job_context_lifecycle() {
        let mut ctx = JobContext::new(JobId::new(), "test-job".to_string());

        ctx.mark_started();
        assert_eq!(ctx.state, JobState::Running);
        assert!(ctx.started_at.is_some());

        ctx.mark_completed();
        assert_eq!(ctx.state, JobState::Completed);
        assert!(ctx.completed_at.is_some());
        assert!(ctx.duration().is_some());
    }

    #[test]
    fn test_job_context_failure() {
        let mut ctx = JobContext::new(JobId::new(), "test-job".to_string());
        ctx.mark_started();
        ctx.mark_failed("error message".to_string());

        assert_eq!(ctx.state, JobState::Failed);
        assert_eq!(ctx.error, Some("error message".to_string()));
    }

    #[test]
    fn test_job_context_cancellation() {
        let mut ctx = JobContext::new(JobId::new(), "test-job".to_string());
        ctx.mark_cancelled();

        assert_eq!(ctx.state, JobState::Cancelled);
    }

    #[test]
    fn test_job_context_retry() {
        let mut ctx = JobContext::new(JobId::new(), "test-job".to_string());
        ctx.increment_retry();
        ctx.increment_retry();

        assert_eq!(ctx.retry_count, 2);
    }

    #[test]
    fn test_resource_requirements_default() {
        let req = ResourceRequirements::default();
        assert!(req.cpu_cores.is_none());
        assert!(req.memory_mb.is_none());
        assert!(!req.gpu);
    }

    #[test]
    fn test_schedule_default() {
        let schedule = Schedule::default();
        matches!(schedule, Schedule::Immediate);
    }
}
