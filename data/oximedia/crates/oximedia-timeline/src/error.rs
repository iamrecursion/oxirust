//! Error types for timeline operations.

use thiserror::Error;

use crate::clip::ClipId;
use crate::track::TrackId;

/// Result type for timeline operations.
pub type TimelineResult<T> = Result<T, TimelineError>;

/// Errors that can occur during timeline operations.
#[derive(Error, Debug)]
pub enum TimelineError {
    /// Track not found.
    #[error("Track not found: {0}")]
    TrackNotFound(TrackId),

    /// Clip not found.
    #[error("Clip not found: {0}")]
    ClipNotFound(ClipId),

    /// Invalid time position.
    #[error("Invalid time position: {0}")]
    InvalidPosition(String),

    /// Invalid duration.
    #[error("Invalid duration: {0}")]
    InvalidDuration(String),

    /// Invalid speed.
    #[error("Invalid speed: {0} (must be between 0.25 and 4.0)")]
    InvalidSpeed(f64),

    /// Track is locked.
    #[error("Track {0} is locked")]
    TrackLocked(TrackId),

    /// Clip overlap detected.
    #[error("Clip overlap at position {0}")]
    ClipOverlap(i64),

    /// Insufficient space for operation.
    #[error("Insufficient space for operation")]
    InsufficientSpace,

    /// Invalid frame rate.
    #[error("Invalid frame rate: {0}/{1}")]
    InvalidFrameRate(i64, i64),

    /// Invalid sample rate.
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// Invalid transition duration.
    #[error("Invalid transition duration: {0}")]
    InvalidTransitionDuration(String),

    /// Effect not found.
    #[error("Effect not found: {0}")]
    EffectNotFound(String),

    /// Keyframe error.
    #[error("Keyframe error: {0}")]
    KeyframeError(String),

    /// Circular dependency detected.
    #[error("Circular dependency detected in nested sequences")]
    CircularDependency,

    /// Export error.
    #[error("Export error: {0}")]
    ExportError(String),

    /// Import error.
    #[error("Import error: {0}")]
    ImportError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Core error.
    #[error("Core error: {0}")]
    CoreError(#[from] oximedia_core::error::OxiError),

    /// EDL error.
    #[error("EDL error: {0}")]
    EdlError(String),

    /// AAF error.
    #[error("AAF error: {0}")]
    AafError(String),

    /// Playback error.
    #[error("Playback error: {0}")]
    PlaybackError(String),

    /// Cache error.
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Multi-camera error.
    #[error("Multi-camera error: {0}")]
    MultiCamError(String),

    /// Invalid timecode.
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}

impl TimelineError {
    /// Creates a new "other" error with a custom message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}
