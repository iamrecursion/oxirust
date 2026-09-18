//! Watermarking error types.

use thiserror::Error;

/// Watermarking errors.
#[derive(Debug, Error)]
pub enum WatermarkError {
    /// Invalid audio data.
    #[error("Invalid audio data: {0}")]
    InvalidData(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Insufficient capacity.
    #[error("Insufficient capacity: need {needed} bits, have {have} bits")]
    InsufficientCapacity {
        /// Required capacity in bits.
        needed: usize,
        /// Available capacity in bits.
        have: usize,
    },

    /// Watermark not detected.
    #[error("Watermark not detected")]
    NotDetected,

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Synchronization failed.
    #[error("Synchronization failed: {0}")]
    SyncFailed(String),

    /// Error correction failed.
    #[error("Error correction failed: too many errors")]
    ErrorCorrectionFailed,

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Audio processing error.
    #[error("Audio error: {0}")]
    Audio(#[from] oximedia_audio::AudioError),

    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

/// Result type for watermarking operations.
pub type WatermarkResult<T> = Result<T, WatermarkError>;
