//! Error types for oximedia-conform.

use std::path::PathBuf;

/// Result type for conform operations.
pub type ConformResult<T> = Result<T, ConformError>;

/// Errors that can occur during media conforming operations.
#[derive(Debug, thiserror::Error)]
pub enum ConformError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// EDL parsing error.
    #[error("EDL error: {0}")]
    Edl(String),

    /// AAF parsing error.
    #[error("AAF error: {0}")]
    Aaf(String),

    /// XML parsing error.
    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    /// Database error.
    #[error("Database error: {0}")]
    Database(String),

    /// Missing source file.
    #[error("Missing source file: {0}")]
    MissingSource(PathBuf),

    /// No matching source found for clip.
    #[error("No matching source found for clip: {0}")]
    NoMatch(String),

    /// Ambiguous matches found.
    #[error("Ambiguous matches found for clip {clip}: {count} candidates")]
    AmbiguousMatch {
        /// Clip identifier.
        clip: String,
        /// Number of matching candidates.
        count: usize,
    },

    /// Invalid timecode.
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),

    /// Timecode mismatch.
    #[error("Timecode mismatch: expected {expected}, found {found}")]
    TimecodeMismatch {
        /// Expected timecode.
        expected: String,
        /// Found timecode.
        found: String,
    },

    /// Invalid frame rate.
    #[error("Invalid frame rate: {0}")]
    InvalidFrameRate(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Missing field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid range.
    #[error("Invalid range: {0}")]
    InvalidRange(String),

    /// Export error.
    #[error("Export error: {0}")]
    Export(String),

    /// Validation error.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Checksum mismatch.
    #[error("Checksum mismatch for {path}: expected {expected}, found {found}")]
    ChecksumMismatch {
        /// The file path.
        path: PathBuf,
        /// Expected checksum.
        expected: String,
        /// Found checksum.
        found: String,
    },

    /// Duration mismatch.
    #[error("Duration mismatch for {path}: expected {expected}s, found {found}s")]
    DurationMismatch {
        /// The file path.
        path: PathBuf,
        /// Expected duration.
        expected: f64,
        /// Found duration.
        found: f64,
    },

    /// Insufficient handles.
    #[error("Insufficient handles for clip {clip}: need {needed} frames, have {available} frames")]
    InsufficientHandles {
        /// Clip identifier.
        clip: String,
        /// Needed frames.
        needed: u64,
        /// Available frames.
        available: u64,
    },

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Async task error.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("Async task error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

impl From<String> for ConformError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for ConformError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}
