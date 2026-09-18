//! Transcript generation from audio.

use crate::error::AccessResult;
use crate::transcript::{Transcript, TranscriptEntry};
use oximedia_audio::frame::AudioBuffer;

/// Transcript generator.
pub struct TranscriptGenerator {
    language: String,
}

impl TranscriptGenerator {
    /// Create a new transcript generator.
    #[must_use]
    pub fn new(language: String) -> Self {
        Self { language }
    }

    /// Generate transcript from audio.
    ///
    /// Integration point for speech-to-text services.
    pub fn generate(&self, _audio: &AudioBuffer) -> AccessResult<Transcript> {
        // Placeholder: Call STT service
        // In production: AWS Transcribe, Google STT, Whisper, etc.

        Ok(Transcript::new())
    }

    /// Generate from caption data.
    #[must_use]
    pub fn from_captions(&self, captions: &[crate::caption::Caption]) -> Transcript {
        let mut transcript = Transcript::new();

        for caption in captions {
            let entry = TranscriptEntry::new(
                caption.start_time(),
                caption.end_time(),
                caption.text().to_string(),
            );
            transcript.add_entry(entry);
        }

        transcript
    }

    /// Get configured language.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }
}

impl Default for TranscriptGenerator {
    fn default() -> Self {
        Self::new("en".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = TranscriptGenerator::new("en".to_string());
        assert_eq!(generator.language(), "en");
    }
}
