//! Error types for multi-camera operations.

use thiserror::Error;

/// Result type for multi-camera operations
pub type Result<T> = std::result::Result<T, MultiCamError>;

/// Errors that can occur during multi-camera operations
#[derive(Debug, Error)]
pub enum MultiCamError {
    /// Synchronization failed
    #[error("Synchronization failed: {0}")]
    SyncFailed(String),

    /// Angle not found
    #[error("Camera angle {0} not found")]
    AngleNotFound(usize),

    /// Invalid angle count
    #[error("Invalid angle count: {0}")]
    InvalidAngleCount(usize),

    /// Invalid frame number
    #[error("Invalid frame number: {0}")]
    InvalidFrame(u64),

    /// Invalid timeline operation
    #[error("Invalid timeline operation: {0}")]
    InvalidOperation(String),

    /// No sync markers found
    #[error("No sync markers found")]
    NoSyncMarkers,

    /// Insufficient data for operation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Color matching failed
    #[error("Color matching failed: {0}")]
    ColorMatchFailed(String),

    /// Composition failed
    #[error("Composition failed: {0}")]
    CompositionFailed(String),

    /// Layout error
    #[error("Layout error: {0}")]
    LayoutError(String),

    /// Switching error
    #[error("Switching error: {0}")]
    SwitchingError(String),

    /// Scoring error
    #[error("Scoring error: {0}")]
    ScoringError(String),

    /// Drift detection failed
    #[error("Drift detection failed: {0}")]
    DriftDetectionFailed(String),

    /// Genlock simulation failed
    #[error("Genlock simulation failed: {0}")]
    GenlockFailed(String),

    /// Spatial alignment failed
    #[error("Spatial alignment failed: {0}")]
    SpatialAlignmentFailed(String),

    /// Metadata error
    #[error("Metadata error: {0}")]
    MetadataError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Audio sync error
    #[error("Audio sync error: {0}")]
    AudioSyncError(String),

    /// Timecode sync error
    #[error("Timecode sync error: {0}")]
    TimecodeSyncError(String),

    /// Visual sync error
    #[error("Visual sync error: {0}")]
    VisualSyncError(String),

    /// Core error
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),

    /// Align error
    #[error("Alignment error: {0}")]
    Align(#[from] oximedia_align::AlignError),

    /// Audio error
    #[error("Audio error: {0}")]
    Audio(#[from] oximedia_audio::AudioError),

    /// Timecode error
    #[error("Timecode error: {0}")]
    Timecode(String),
}
