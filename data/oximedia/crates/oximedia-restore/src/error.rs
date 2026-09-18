//! Audio restoration errors.

use thiserror::Error;

/// Audio restoration errors.
#[derive(Debug, Error)]
pub enum RestoreError {
    /// Invalid audio data.
    #[error("Invalid audio data: {0}")]
    InvalidData(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// FFT error.
    #[error("FFT error: {0}")]
    Fft(String),

    /// Not enough data.
    #[error("Not enough data: need {needed}, have {have}")]
    NotEnoughData {
        /// Required size.
        needed: usize,
        /// Available size.
        have: usize,
    },

    /// Audio error.
    #[error("Audio error: {0}")]
    Audio(#[from] oximedia_audio::AudioError),

    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

/// Result type for audio restoration operations.
pub type RestoreResult<T> = Result<T, RestoreError>;
