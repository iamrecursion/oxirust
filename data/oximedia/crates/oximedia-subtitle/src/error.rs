//! Error types for subtitle processing.

use thiserror::Error;

/// Result type for subtitle operations.
pub type SubtitleResult<T> = Result<T, SubtitleError>;

/// Errors that can occur during subtitle processing.
#[derive(Debug, Error)]
pub enum SubtitleError {
    /// Parse error.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid subtitle format.
    #[error("Invalid subtitle format: {0}")]
    InvalidFormat(String),

    /// Invalid timestamp.
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Font loading error.
    #[error("Font loading error: {0}")]
    FontError(String),

    /// Text rendering error.
    #[error("Text rendering error: {0}")]
    RenderError(String),

    /// Invalid color specification.
    #[error("Invalid color: {0}")]
    InvalidColor(String),

    /// Invalid style parameter.
    #[error("Invalid style: {0}")]
    InvalidStyle(String),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(String),

    /// Invalid frame format for overlay.
    #[error("Invalid frame format: {0}")]
    InvalidFrameFormat(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for SubtitleError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<std::string::FromUtf8Error> for SubtitleError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        Self::ParseError(format!("UTF-8 decode error: {err}"))
    }
}
