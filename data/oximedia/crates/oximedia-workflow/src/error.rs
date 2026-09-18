//! Error types for workflow orchestration.

use std::path::PathBuf;
use thiserror::Error;

/// Result type for workflow operations.
pub type Result<T> = std::result::Result<T, WorkflowError>;

/// Errors that can occur during workflow orchestration.
#[derive(Debug, Error)]
pub enum WorkflowError {
    /// Workflow not found.
    #[error("Workflow not found: {0}")]
    WorkflowNotFound(String),

    /// Task not found.
    #[error("Task not found: {0}")]
    TaskNotFound(String),

    /// Cycle detected in workflow DAG.
    #[error("Cycle detected in workflow DAG")]
    CycleDetected,

    /// Invalid workflow configuration.
    #[error("Invalid workflow configuration: {0}")]
    InvalidConfiguration(String),

    /// Task execution failed.
    #[error("Task execution failed: {task_id}, reason: {reason}")]
    TaskExecutionFailed {
        /// Task identifier.
        task_id: String,
        /// Failure reason.
        reason: String,
    },

    /// Task timeout.
    #[error("Task timeout: {0}")]
    TaskTimeout(String),

    /// Task cancelled.
    #[error("Task cancelled: {0}")]
    TaskCancelled(String),

    /// Dependency failure.
    #[error("Dependency failed: {0}")]
    DependencyFailed(String),

    /// Database error.
    #[error("Database error: {0}")]
    Database(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    YamlParsing(#[from] serde_yaml::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File watcher error.
    #[error("File watcher error: {0}")]
    FileWatcher(String),

    /// Invalid task type.
    #[error("Invalid task type: {0}")]
    InvalidTaskType(String),

    /// Missing required parameter.
    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    /// Invalid parameter value.
    #[error("Invalid parameter value: {param}, value: {value}")]
    InvalidParameter {
        /// Parameter name.
        param: String,
        /// Invalid value.
        value: String,
    },

    /// Resource limit exceeded.
    #[error("Resource limit exceeded: {resource}, limit: {limit}")]
    ResourceLimitExceeded {
        /// Resource type.
        resource: String,
        /// Limit value.
        limit: String,
    },

    /// Worker pool error.
    #[error("Worker pool error: {0}")]
    WorkerPool(String),

    /// Scheduler error.
    #[error("Scheduler error: {0}")]
    Scheduler(String),

    /// Invalid cron expression.
    #[error("Invalid cron expression: {0}")]
    InvalidCronExpression(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Invalid URL.
    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    /// HTTP error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// WebSocket error.
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    /// Lock poisoned.
    #[error("Lock poisoned")]
    LockPoisoned,

    /// Shutdown in progress.
    #[error("Shutdown in progress")]
    ShutdownInProgress,

    /// Already running.
    #[error("Workflow already running: {0}")]
    AlreadyRunning(String),

    /// Not running.
    #[error("Workflow not running: {0}")]
    NotRunning(String),

    /// Invalid state transition.
    #[error("Invalid state transition: from {from} to {to}")]
    InvalidStateTransition {
        /// Current state.
        from: String,
        /// Target state.
        to: String,
    },

    /// Generic error.
    #[error("{0}")]
    Generic(String),
}

impl WorkflowError {
    /// Create a generic error with a message.
    pub fn generic(msg: impl Into<String>) -> Self {
        Self::Generic(msg.into())
    }

    /// Check if error is recoverable.
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::TaskTimeout(_)
                | Self::ResourceLimitExceeded { .. }
                | Self::WorkerPool(_)
                | Self::Http(_)
        )
    }

    /// Check if error should trigger retry.
    #[must_use]
    pub fn should_retry(&self) -> bool {
        matches!(
            self,
            Self::TaskExecutionFailed { .. }
                | Self::TaskTimeout(_)
                | Self::Http(_)
                | Self::WorkerPool(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WorkflowError::WorkflowNotFound("test-workflow".to_string());
        assert_eq!(err.to_string(), "Workflow not found: test-workflow");
    }

    #[test]
    fn test_task_execution_failed() {
        let err = WorkflowError::TaskExecutionFailed {
            task_id: "task-1".to_string(),
            reason: "connection timeout".to_string(),
        };
        assert!(err.to_string().contains("task-1"));
        assert!(err.to_string().contains("connection timeout"));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(WorkflowError::TaskTimeout("task-1".to_string()).is_recoverable());
        assert!(!WorkflowError::CycleDetected.is_recoverable());
    }

    #[test]
    fn test_should_retry() {
        assert!(WorkflowError::TaskTimeout("task-1".to_string()).should_retry());
        assert!(!WorkflowError::CycleDetected.should_retry());
    }

    #[test]
    fn test_generic_error() {
        let err = WorkflowError::generic("custom error");
        assert_eq!(err.to_string(), "custom error");
    }

    #[test]
    fn test_invalid_state_transition() {
        let err = WorkflowError::InvalidStateTransition {
            from: "running".to_string(),
            to: "pending".to_string(),
        };
        assert!(err.to_string().contains("running"));
        assert!(err.to_string().contains("pending"));
    }
}
