//! Error types for audio post-production operations.

use thiserror::Error;

/// Result type for audio post-production operations.
pub type AudioPostResult<T> = Result<T, AudioPostError>;

/// Errors that can occur during audio post-production operations.
#[derive(Error, Debug)]
pub enum AudioPostError {
    /// Invalid sample rate
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// Invalid channel count
    #[error("Invalid channel count: {0}")]
    InvalidChannelCount(usize),

    /// Invalid buffer size
    #[error("Invalid buffer size: {0}")]
    InvalidBufferSize(usize),

    /// Channel not found
    #[error("Channel not found: {0}")]
    ChannelNotFound(usize),

    /// Cue not found
    #[error("Cue not found: {0}")]
    CueNotFound(usize),

    /// Take not found
    #[error("Take not found: {0}")]
    TakeNotFound(usize),

    /// Invalid timecode
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),

    /// Invalid gain value
    #[error("Invalid gain value: {0} dB")]
    InvalidGain(f32),

    /// Invalid pan value
    #[error("Invalid pan value: {0} (must be -1.0 to 1.0)")]
    InvalidPan(f32),

    /// Invalid frequency value
    #[error("Invalid frequency: {0} Hz")]
    InvalidFrequency(f32),

    /// Invalid Q value
    #[error("Invalid Q value: {0}")]
    InvalidQ(f32),

    /// Invalid ratio value
    #[error("Invalid ratio: {0}")]
    InvalidRatio(f32),

    /// Invalid threshold value
    #[error("Invalid threshold: {0} dB")]
    InvalidThreshold(f32),

    /// Invalid attack time
    #[error("Invalid attack time: {0} ms")]
    InvalidAttack(f32),

    /// Invalid release time
    #[error("Invalid release time: {0} ms")]
    InvalidRelease(f32),

    /// Stem not found
    #[error("Stem not found: {0}")]
    StemNotFound(String),

    /// Invalid loudness target
    #[error("Invalid loudness target: {0} LUFS")]
    InvalidLoudnessTarget(f32),

    /// Automation point not found
    #[error("Automation point not found at time {0}")]
    AutomationPointNotFound(f64),

    /// Invalid automation mode
    #[error("Invalid automation mode: {0}")]
    InvalidAutomationMode(String),

    /// Effect not found
    #[error("Effect not found: {0}")]
    EffectNotFound(String),

    /// Invalid effect parameter
    #[error("Invalid effect parameter: {0}")]
    InvalidEffectParameter(String),

    /// File I/O error
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}
