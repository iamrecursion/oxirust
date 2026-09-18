//! Codec-specific error types.

use thiserror::Error;

/// Codec-specific errors.
#[derive(Debug, Error)]
pub enum CodecError {
    /// Invalid bitstream data.
    #[error("Invalid bitstream: {0}")]
    InvalidBitstream(String),

    /// Unsupported codec feature.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Decoder needs more data.
    #[error("Need more data")]
    NeedMoreData,

    /// Encoder buffer too small.
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

    /// Internal codec error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// End of stream.
    #[error("End of stream")]
    Eof,

    /// Decoder error.
    #[error("Decoder error: {0}")]
    DecoderError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),

    /// Invalid data.
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Result type for codec operations.
pub type CodecResult<T> = Result<T, CodecError>;
