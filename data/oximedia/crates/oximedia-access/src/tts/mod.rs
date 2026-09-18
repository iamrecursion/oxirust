//! Text-to-speech synthesis.

pub mod prosody;
pub mod synthesize;
pub mod voice;

pub use prosody::{
    BreakStrength, EmphasisLevel, ProsodyConfig, ProsodyControl, SayAsInterpret, SsmlBuilder,
    SsmlElement,
};
pub use synthesize::TextToSpeech;
pub use voice::{Voice, VoiceGender, VoiceRegistry};

use serde::{Deserialize, Serialize};

/// TTS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Voice to use.
    pub voice: String,
    /// Speech rate (0.5 to 2.0).
    pub rate: f32,
    /// Pitch adjustment in semitones (-12 to 12).
    pub pitch: f32,
    /// Volume (0.0 to 1.0).
    pub volume: f32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            voice: "en-US-Neural".to_string(),
            rate: 1.0,
            pitch: 0.0,
            volume: 0.8,
            sample_rate: 24000,
        }
    }
}
