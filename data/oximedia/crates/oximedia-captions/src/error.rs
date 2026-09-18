//! Error types for caption processing

use std::io;

/// Result type for caption operations
pub type Result<T> = std::result::Result<T, CaptionError>;

/// Errors that can occur during caption processing
#[derive(Debug, thiserror::Error)]
pub enum CaptionError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Invalid timestamp
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Invalid caption ID
    #[error("Invalid caption ID: {0}")]
    InvalidCaptionId(String),

    /// Caption not found
    #[error("Caption not found: {0}")]
    CaptionNotFound(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Standards compliance error
    #[error("Standards compliance error: {0}")]
    Compliance(String),

    /// Encoding error
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Template error
    #[error("Template error: {0}")]
    Template(String),

    /// Export error
    #[error("Export error: {0}")]
    Export(String),

    /// Import error
    #[error("Import error: {0}")]
    Import(String),

    /// Translation error
    #[error("Translation error: {0}")]
    Translation(String),

    /// Rendering error
    #[error("Rendering error: {0}")]
    Rendering(String),

    /// Embedding error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Character limit exceeded
    #[error("Character limit exceeded: {actual} > {limit}")]
    CharacterLimitExceeded {
        /// Actual character count
        actual: usize,
        /// Maximum allowed
        limit: usize,
    },

    /// Reading speed exceeded
    #[error("Reading speed exceeded: {actual} WPM > {limit} WPM")]
    ReadingSpeedExceeded {
        /// Actual reading speed
        actual: u32,
        /// Maximum allowed
        limit: u32,
    },

    /// Caption too short
    #[error("Caption duration too short: {actual}ms < {minimum}ms")]
    DurationTooShort {
        /// Actual duration
        actual: u64,
        /// Minimum required
        minimum: u64,
    },

    /// Caption overlap detected
    #[error("Caption overlap detected at {timestamp}")]
    Overlap {
        /// Timestamp of overlap
        timestamp: String,
    },

    /// Gap too small
    #[error("Gap too small: {actual} frames < {minimum} frames")]
    GapTooSmall {
        /// Actual gap
        actual: u32,
        /// Minimum required
        minimum: u32,
    },

    /// Invalid color format
    #[error("Invalid color format: {0}")]
    InvalidColor(String),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// XML error
    #[error("XML error: {0}")]
    Xml(String),

    /// Feature not enabled
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

impl From<String> for CaptionError {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for CaptionError {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}
