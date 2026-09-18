//! Speech-to-text transcription.

pub mod accuracy;
pub mod language;
pub mod transcribe;

pub use accuracy::{AccuracyImprover, TranscriptionAccuracy};
pub use language::SttLanguageModel;
pub use transcribe::{ChunkTranscriptionResult, IncrementalSttProcessor, SpeechToText};

use serde::{Deserialize, Serialize};

/// STT configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    /// Language code.
    pub language: String,
    /// Enable speaker diarization.
    pub speaker_diarization: bool,
    /// Enable punctuation.
    pub enable_punctuation: bool,
    /// Enable word-level timestamps.
    pub word_timestamps: bool,
    /// Model quality.
    pub model: SttModel,
}

/// STT model quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SttModel {
    /// Fast model for real-time.
    Fast,
    /// Standard model.
    Standard,
    /// High-accuracy model.
    HighAccuracy,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            speaker_diarization: true,
            enable_punctuation: true,
            word_timestamps: true,
            model: SttModel::Standard,
        }
    }
}
