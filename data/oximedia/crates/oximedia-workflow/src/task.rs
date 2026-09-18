//! Task definitions and types.

use crate::error::{Result, WorkflowError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

/// Unique task identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(Uuid);

impl TaskId {
    /// Create a new random task ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for TaskId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// Task state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum TaskState {
    /// Task is pending execution.
    #[default]
    Pending,
    /// Task is queued for execution.
    Queued,
    /// Task is currently running.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
    /// Task is waiting for dependencies.
    Waiting,
    /// Task is retrying after failure.
    Retrying,
    /// Task is skipped due to conditions.
    Skipped,
}

impl TaskState {
    /// Check if task is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Skipped
        )
    }

    /// Check if task is active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Retrying)
    }

    /// Check if task can be started.
    #[must_use]
    pub const fn can_start(&self) -> bool {
        matches!(self, Self::Pending | Self::Queued)
    }
}

/// Task priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum TaskPriority {
    /// Low priority.
    Low = 0,
    /// Normal priority.
    #[default]
    Normal = 1,
    /// High priority.
    High = 2,
    /// Critical priority.
    Critical = 3,
}

/// Retry policy for tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_attempts: u32,
    /// Initial delay before retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Backoff multiplier.
    pub backoff_multiplier: f64,
    /// Whether to use exponential backoff.
    pub exponential_backoff: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            exponential_backoff: true,
        }
    }
}

impl RetryPolicy {
    /// Calculate delay for a specific attempt.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if !self.exponential_backoff {
            return self.initial_delay;
        }

        let delay = self.initial_delay.as_secs_f64()
            * self
                .backoff_multiplier
                .powi(i32::try_from(attempt).unwrap_or(10));
        let delay = delay.min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(delay)
    }

    /// Check if should retry given the attempt count.
    #[must_use]
    pub const fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Task type enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskType {
    /// Transcode media file.
    Transcode {
        /// Input file path.
        input: PathBuf,
        /// Output file path.
        output: PathBuf,
        /// Preset name.
        preset: String,
        /// Additional parameters.
        #[serde(default)]
        params: HashMap<String, serde_json::Value>,
    },

    /// Quality control validation.
    QualityControl {
        /// Input file path.
        input: PathBuf,
        /// QC profile name.
        profile: String,
        /// Validation rules.
        #[serde(default)]
        rules: Vec<String>,
    },

    /// File transfer operation.
    Transfer {
        /// Source path or URL.
        source: String,
        /// Destination path or URL.
        destination: String,
        /// Transfer protocol.
        protocol: TransferProtocol,
        /// Transfer options.
        #[serde(default)]
        options: HashMap<String, String>,
    },

    /// Send notification.
    Notification {
        /// Notification channel.
        channel: NotificationChannel,
        /// Message content.
        message: String,
        /// Additional metadata.
        #[serde(default)]
        metadata: HashMap<String, String>,
    },

    /// Execute custom script.
    CustomScript {
        /// Script path.
        script: PathBuf,
        /// Script arguments.
        #[serde(default)]
        args: Vec<String>,
        /// Environment variables.
        #[serde(default)]
        env: HashMap<String, String>,
    },

    /// Media analysis task.
    Analysis {
        /// Input file path.
        input: PathBuf,
        /// Analysis types to perform.
        analyses: Vec<AnalysisType>,
        /// Output path for results.
        output: Option<PathBuf>,
    },

    /// Conditional decision task.
    Conditional {
        /// Condition expression.
        condition: String,
        /// Task to execute if true.
        true_task: Option<Box<Task>>,
        /// Task to execute if false.
        false_task: Option<Box<Task>>,
    },

    /// Wait for duration.
    Wait {
        /// Duration to wait.
        duration: Duration,
    },

    /// HTTP request task.
    HttpRequest {
        /// Request URL.
        url: String,
        /// HTTP method.
        method: HttpMethod,
        /// Request headers.
        #[serde(default)]
        headers: HashMap<String, String>,
        /// Request body.
        body: Option<String>,
    },
}

/// Transfer protocol types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransferProtocol {
    /// Local file system.
    Local,
    /// FTP transfer.
    Ftp,
    /// SFTP transfer.
    Sftp,
    /// Amazon S3.
    S3,
    /// HTTP/HTTPS.
    Http,
    /// Rsync.
    Rsync,
}

/// Notification channel types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NotificationChannel {
    /// Email notification.
    Email {
        /// Recipient addresses.
        to: Vec<String>,
        /// Subject line.
        subject: String,
    },
    /// Webhook notification.
    Webhook {
        /// Webhook URL.
        url: String,
    },
    /// Slack notification.
    Slack {
        /// Slack channel.
        channel: String,
        /// Webhook URL.
        webhook_url: String,
    },
    /// Discord notification.
    Discord {
        /// Webhook URL.
        webhook_url: String,
    },
}

/// Media analysis types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisType {
    /// Audio level analysis.
    AudioLevels,
    /// Video quality analysis.
    VideoQuality,
    /// Scene detection.
    SceneDetection,
    /// Black frame detection.
    BlackFrames,
    /// Silence detection.
    Silence,
    /// Color analysis.
    Color,
    /// Motion analysis.
    Motion,
}

/// HTTP methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    /// GET method.
    Get,
    /// POST method.
    Post,
    /// PUT method.
    Put,
    /// DELETE method.
    Delete,
    /// PATCH method.
    Patch,
}

/// Task definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    pub id: TaskId,
    /// Task name.
    pub name: String,
    /// Task type and configuration.
    pub task_type: TaskType,
    /// Current state.
    #[serde(default)]
    pub state: TaskState,
    /// Task priority.
    #[serde(default)]
    pub priority: TaskPriority,
    /// Retry policy.
    #[serde(default)]
    pub retry: RetryPolicy,
    /// Execution timeout.
    #[serde(default = "default_timeout")]
    pub timeout: Duration,
    /// Task dependencies.
    #[serde(default)]
    pub dependencies: Vec<TaskId>,
    /// Task metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Number of retry attempts so far.
    #[serde(default)]
    pub retry_count: u32,
    /// Task conditions (must evaluate to true to run).
    #[serde(default)]
    pub conditions: Vec<String>,
}

fn default_timeout() -> Duration {
    Duration::from_secs(3600) // 1 hour
}

impl Task {
    /// Create a new task.
    #[must_use]
    pub fn new(name: impl Into<String>, task_type: TaskType) -> Self {
        Self {
            id: TaskId::new(),
            name: name.into(),
            task_type,
            state: TaskState::Pending,
            priority: TaskPriority::default(),
            retry: RetryPolicy::default(),
            timeout: default_timeout(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
            retry_count: 0,
            conditions: Vec::new(),
        }
    }

    /// Add a dependency to this task.
    pub fn add_dependency(&mut self, task_id: TaskId) {
        if !self.dependencies.contains(&task_id) {
            self.dependencies.push(task_id);
        }
    }

    /// Set task priority.
    #[must_use]
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set retry policy.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add condition.
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Check if task can be executed.
    #[must_use]
    pub const fn can_execute(&self) -> bool {
        self.state.can_start()
    }

    /// Update task state.
    pub fn set_state(&mut self, state: TaskState) -> Result<()> {
        // Validate state transition
        match (&self.state, &state) {
            (TaskState::Completed | TaskState::Failed | TaskState::Cancelled, _)
                if !matches!(state, TaskState::Pending) =>
            {
                return Err(WorkflowError::InvalidStateTransition {
                    from: format!("{:?}", self.state),
                    to: format!("{state:?}"),
                });
            }
            _ => {}
        }

        self.state = state;
        Ok(())
    }

    /// Increment retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if should retry.
    #[must_use]
    pub fn should_retry(&self) -> bool {
        self.retry.should_retry(self.retry_count)
    }

    /// Get retry delay.
    #[must_use]
    pub fn retry_delay(&self) -> Duration {
        self.retry.delay_for_attempt(self.retry_count)
    }
}

/// Task execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task identifier.
    pub task_id: TaskId,
    /// Execution status.
    pub status: TaskState,
    /// Result data.
    pub data: Option<serde_json::Value>,
    /// Error message if failed.
    pub error: Option<String>,
    /// Execution duration.
    pub duration: Duration,
    /// Output files produced.
    #[serde(default)]
    pub outputs: Vec<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_creation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_state_terminal() {
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(!TaskState::Running.is_terminal());
    }

    #[test]
    fn test_task_state_active() {
        assert!(TaskState::Running.is_active());
        assert!(!TaskState::Pending.is_active());
    }

    #[test]
    fn test_retry_policy_delay() {
        let policy = RetryPolicy::default();
        let delay1 = policy.delay_for_attempt(0);
        let delay2 = policy.delay_for_attempt(1);
        assert!(delay2 > delay1);
    }

    #[test]
    fn test_retry_policy_max_attempts() {
        let policy = RetryPolicy {
            max_attempts: 3,
            ..Default::default()
        };
        assert!(policy.should_retry(0));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(
            "test-task",
            TaskType::Wait {
                duration: Duration::from_secs(10),
            },
        );
        assert_eq!(task.name, "test-task");
        assert_eq!(task.state, TaskState::Pending);
    }

    #[test]
    fn test_task_with_priority() {
        let task = Task::new(
            "test",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
        .with_priority(TaskPriority::High);
        assert_eq!(task.priority, TaskPriority::High);
    }

    #[test]
    fn test_task_add_dependency() {
        let mut task = Task::new(
            "test",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let dep_id = TaskId::new();
        task.add_dependency(dep_id);
        assert_eq!(task.dependencies.len(), 1);
        assert_eq!(task.dependencies[0], dep_id);
    }

    #[test]
    fn test_task_state_transition() {
        let mut task = Task::new(
            "test",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        assert!(task.set_state(TaskState::Running).is_ok());
        assert!(task.set_state(TaskState::Completed).is_ok());
        assert!(task.set_state(TaskState::Running).is_err());
    }

    #[test]
    fn test_task_retry_logic() {
        let mut task = Task::new(
            "test",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
        .with_retry(RetryPolicy {
            max_attempts: 2,
            ..Default::default()
        });

        assert!(task.should_retry());
        task.increment_retry();
        assert!(task.should_retry());
        task.increment_retry();
        assert!(!task.should_retry());
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            exponential_backoff: true,
        };

        let delay0 = policy.delay_for_attempt(0);
        let delay1 = policy.delay_for_attempt(1);
        let delay2 = policy.delay_for_attempt(2);

        assert_eq!(delay0.as_secs(), 1);
        assert_eq!(delay1.as_secs(), 2);
        assert_eq!(delay2.as_secs(), 4);
    }
}
