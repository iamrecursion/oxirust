//! Error types for image I/O operations.

use std::io;

/// Result type for image operations.
pub type ImageResult<T> = Result<T, ImageError>;

/// Error types for image I/O.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Invalid or corrupted file format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Unsupported feature or format.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Invalid pattern syntax.
    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    /// Frame number not found in sequence.
    #[error("Frame {0} not found in sequence")]
    FrameNotFound(u32),

    /// Invalid frame range.
    #[error("Invalid frame range: {0}")]
    InvalidRange(String),

    /// Compression/decompression error.
    #[error("Compression error: {0}")]
    Compression(String),

    /// Invalid image dimensions.
    #[error("Invalid dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),

    /// Invalid pixel format or type.
    #[error("Invalid pixel format: {0}")]
    InvalidPixelFormat(String),

    /// Metadata parsing error.
    #[error("Metadata error: {0}")]
    Metadata(String),
}

impl ImageError {
    /// Creates an invalid format error.
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidFormat(msg.into())
    }

    /// Creates an unsupported feature error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Creates a compression error.
    pub fn compression(msg: impl Into<String>) -> Self {
        Self::Compression(msg.into())
    }
}
