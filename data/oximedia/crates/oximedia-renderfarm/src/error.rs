// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Error types for render farm operations.

use std::io;
use thiserror::Error;

/// Result type for render farm operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for render farm operations
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Job not found
    #[error("Job not found: {0}")]
    JobNotFound(String),

    /// Worker not found
    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    /// Pool not found
    #[error("Pool not found: {0}")]
    PoolNotFound(String),

    /// Invalid job state transition
    #[error("Invalid job state transition from {from} to {to}")]
    InvalidStateTransition {
        /// Current state
        from: String,
        /// Target state
        to: String,
    },

    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Worker offline
    #[error("Worker offline: {0}")]
    WorkerOffline(String),

    /// No workers available
    #[error("No workers available for job")]
    NoWorkersAvailable,

    /// Insufficient resources
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    /// Dependency error
    #[error("Dependency error: {0}")]
    Dependency(String),

    /// Asset not found
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// License unavailable
    #[error("License unavailable: {0}")]
    LicenseUnavailable(String),

    /// Budget exceeded
    #[error("Budget exceeded: allocated={allocated}, spent={spent}")]
    BudgetExceeded {
        /// Allocated budget
        allocated: f64,
        /// Spent amount
        spent: f64,
    },

    /// Deadline exceeded
    #[error("Deadline exceeded for job: {0}")]
    DeadlineExceeded(String),

    /// Verification failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// API error
    #[error("API error: {0}")]
    Api(String),

    /// Cloud provider error
    #[error("Cloud provider error: {0}")]
    CloudProvider(String),

    /// Plugin error
    #[error("Plugin error: {0}")]
    Plugin(String),

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// Recovery error
    #[error("Recovery error: {0}")]
    Recovery(String),

    /// Invalid frame range
    #[error("Invalid frame range: start={start}, end={end}")]
    InvalidFrameRange {
        /// Start frame
        start: u32,
        /// End frame
        end: u32,
    },

    /// Task failed
    #[error("Task failed: {0}")]
    TaskFailed(String),

    /// Timeout
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}

impl From<prometheus::Error> for Error {
    fn from(e: prometheus::Error) -> Self {
        Self::Other(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::JobNotFound("job123".to_string());
        assert_eq!(err.to_string(), "Job not found: job123");
    }

    #[test]
    fn test_budget_exceeded_error() {
        let err = Error::BudgetExceeded {
            allocated: 1000.0,
            spent: 1500.0,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("1500"));
    }

    #[test]
    fn test_invalid_state_transition() {
        let err = Error::InvalidStateTransition {
            from: "Pending".to_string(),
            to: "Completed".to_string(),
        };
        assert!(err.to_string().contains("Pending"));
        assert!(err.to_string().contains("Completed"));
    }

    #[test]
    fn test_string_conversion() {
        let err: Error = "test error".into();
        assert_eq!(err.to_string(), "test error");
    }
}
