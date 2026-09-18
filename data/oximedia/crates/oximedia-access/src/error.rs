//! Error types for accessibility operations.

use thiserror::Error;

/// Result type for accessibility operations.
pub type AccessResult<T> = Result<T, AccessError>;

/// Errors that can occur during accessibility processing.
#[derive(Error, Debug)]
pub enum AccessError {
    /// Audio description generation failed.
    #[error("Audio description generation failed: {0}")]
    AudioDescriptionFailed(String),

    /// Caption generation failed.
    #[error("Caption generation failed: {0}")]
    CaptionFailed(String),

    /// Sign language overlay failed.
    #[error("Sign language overlay failed: {0}")]
    SignLanguageFailed(String),

    /// Transcript generation failed.
    #[error("Transcript generation failed: {0}")]
    TranscriptFailed(String),

    /// Translation failed.
    #[error("Translation failed: {0}")]
    TranslationFailed(String),

    /// Text-to-speech synthesis failed.
    #[error("Text-to-speech failed: {0}")]
    TtsFailed(String),

    /// Speech-to-text transcription failed.
    #[error("Speech-to-text failed: {0}")]
    SttFailed(String),

    /// Visual enhancement failed.
    #[error("Visual enhancement failed: {0}")]
    VisualEnhancementFailed(String),

    /// Audio enhancement failed.
    #[error("Audio enhancement failed: {0}")]
    AudioEnhancementFailed(String),

    /// Speed control operation failed.
    #[error("Speed control failed: {0}")]
    SpeedControlFailed(String),

    /// Compliance checking failed.
    #[error("Compliance check failed: {0}")]
    ComplianceFailed(String),

    /// Invalid timing information.
    #[error("Invalid timing: {0}")]
    InvalidTiming(String),

    /// Language not supported.
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Voice not available.
    #[error("Voice not available: {0}")]
    VoiceNotAvailable(String),

    /// Invalid format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Quality threshold not met.
    #[error("Quality threshold not met: {0}")]
    QualityThresholdNotMet(String),

    /// Synchronization error.
    #[error("Synchronization error: {0}")]
    SyncError(String),

    /// Audio processing error.
    #[error("Audio processing error: {0}")]
    AudioError(#[from] oximedia_audio::error::AudioError),

    /// Subtitle processing error.
    #[error("Subtitle processing error: {0}")]
    SubtitleError(#[from] oximedia_subtitle::error::SubtitleError),

    /// Graph processing error.
    #[error("Graph processing error: {0}")]
    GraphError(#[from] oximedia_graph::error::GraphError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}
