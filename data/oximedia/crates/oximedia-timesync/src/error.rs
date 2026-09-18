//! Error types for time synchronization.

use std::io;
use thiserror::Error;

/// Result type for time synchronization operations.
pub type TimeSyncResult<T> = Result<T, TimeSyncError>;

/// Errors that can occur during time synchronization operations.
#[derive(Error, Debug)]
pub enum TimeSyncError {
    /// PTP protocol error
    #[error("PTP protocol error: {0}")]
    Ptp(String),

    /// NTP protocol error
    #[error("NTP protocol error: {0}")]
    Ntp(String),

    /// Timecode error
    #[error("Timecode error: {0}")]
    Timecode(String),

    /// Clock discipline error
    #[error("Clock discipline error: {0}")]
    ClockDiscipline(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Invalid packet
    #[error("Invalid packet: {0}")]
    InvalidPacket(String),

    /// Timeout
    #[error("Operation timed out")]
    Timeout,

    /// Clock not synchronized
    #[error("Clock not synchronized")]
    NotSynchronized,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shared memory error
    #[error("Shared memory error: {0}")]
    SharedMemory(String),

    /// IPC error
    #[error("IPC error: {0}")]
    Ipc(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// System clock adjustment error
    #[error("System clock adjustment error: {0}")]
    ClockAdjust(String),

    /// Invalid timestamp
    #[error("Invalid timestamp")]
    InvalidTimestamp,

    /// Overflow error
    #[error("Numeric overflow")]
    Overflow,
}
