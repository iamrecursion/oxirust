//! Audio codec errors.

use thiserror::Error;

/// Audio codec errors.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Invalid audio data.
    #[error("Invalid audio data: {0}")]
    InvalidData(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Need more data.
    #[error("Need more data")]
    NeedMoreData,

    /// Buffer too small.
    #[error("Buffer too small: need {needed}, have {have}")]
    BufferTooSmall {
        /// Required size.
        needed: usize,
        /// Available size.
        have: usize,
    },

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// End of stream.
    #[error("End of stream")]
    Eof,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(String),

    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

/// Result type for audio operations.
pub type AudioResult<T> = Result<T, AudioError>;
