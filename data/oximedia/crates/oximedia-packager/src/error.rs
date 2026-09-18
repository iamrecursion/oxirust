//! Error types for the packager.

use std::io;
use thiserror::Error;

/// Result type alias for packager operations.
pub type PackagerResult<T> = Result<T, PackagerError>;

/// Errors that can occur during packaging operations.
#[derive(Debug, Error)]
pub enum PackagerError {
    /// I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported codec for packaging.
    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    /// Invalid bitrate ladder configuration.
    #[error("Invalid bitrate ladder: {0}")]
    InvalidLadder(String),

    /// Segment generation failed.
    #[error("Segment generation failed: {0}")]
    SegmentFailed(String),

    /// Manifest generation failed.
    #[error("Manifest generation failed: {0}")]
    ManifestFailed(String),

    /// Encryption error.
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Invalid media source.
    #[error("Invalid media source: {0}")]
    InvalidSource(String),

    /// Missing required parameter.
    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    /// Keyframe alignment failed.
    #[error("Keyframe alignment failed: {0}")]
    AlignmentFailed(String),

    /// DRM preparation failed.
    #[error("DRM preparation failed: {0}")]
    DrmFailed(String),

    /// Cloud upload failed.
    #[error("Cloud upload failed: {0}")]
    UploadFailed(String),

    /// Generic packaging error.
    #[error("Packaging error: {0}")]
    PackagingError(String),

    /// Core library error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// XML serialization error.
    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),

    /// Time parsing error.
    #[error("Time error: {0}")]
    Time(String),
}

impl PackagerError {
    /// Create an invalid configuration error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create an unsupported codec error.
    pub fn unsupported_codec(msg: impl Into<String>) -> Self {
        Self::UnsupportedCodec(msg.into())
    }

    /// Create a segment failed error.
    pub fn segment_failed(msg: impl Into<String>) -> Self {
        Self::SegmentFailed(msg.into())
    }

    /// Create a manifest failed error.
    pub fn manifest_failed(msg: impl Into<String>) -> Self {
        Self::ManifestFailed(msg.into())
    }
}
