//! Voice selection and management.

use serde::{Deserialize, Serialize};

/// Gender of the voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceGender {
    /// Male voice.
    Male,
    /// Female voice.
    Female,
    /// Neutral voice.
    Neutral,
}

/// A TTS voice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// Voice identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Language code.
    pub language: String,
    /// Gender.
    pub gender: VoiceGender,
    /// Whether this is a neural/premium voice.
    pub neural: bool,
}

impl Voice {
    /// Create a new voice.
    #[must_use]
    pub fn new(id: String, name: String, language: String, gender: VoiceGender) -> Self {
        Self {
            id,
            name,
            language,
            gender,
            neural: false,
        }
    }

    /// Mark as neural voice.
    #[must_use]
    pub const fn with_neural(mut self, neural: bool) -> Self {
        self.neural = neural;
        self
    }
}

/// Registry of available voices.
pub struct VoiceRegistry {
    voices: Vec<Voice>,
}

impl VoiceRegistry {
    /// Create a new voice registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            voices: Self::default_voices(),
        }
    }

    /// Get all voices.
    #[must_use]
    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    /// Find voice by ID.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&Voice> {
        self.voices.iter().find(|v| v.id == id)
    }

    /// Find voices by language.
    #[must_use]
    pub fn find_by_language(&self, language: &str) -> Vec<&Voice> {
        self.voices
            .iter()
            .filter(|v| v.language == language)
            .collect()
    }

    /// Find voices by gender.
    #[must_use]
    pub fn find_by_gender(&self, gender: VoiceGender) -> Vec<&Voice> {
        self.voices.iter().filter(|v| v.gender == gender).collect()
    }

    /// Add a voice to the registry.
    pub fn add_voice(&mut self, voice: Voice) {
        self.voices.push(voice);
    }

    fn default_voices() -> Vec<Voice> {
        vec![
            Voice::new(
                "en-US-Neural-Female".to_string(),
                "US English Female Neural".to_string(),
                "en".to_string(),
                VoiceGender::Female,
            )
            .with_neural(true),
            Voice::new(
                "en-US-Neural-Male".to_string(),
                "US English Male Neural".to_string(),
                "en".to_string(),
                VoiceGender::Male,
            )
            .with_neural(true),
            Voice::new(
                "es-ES-Neural-Female".to_string(),
                "Spanish Female Neural".to_string(),
                "es".to_string(),
                VoiceGender::Female,
            )
            .with_neural(true),
        ]
    }
}

impl Default for VoiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_creation() {
        let voice = Voice::new(
            "test-voice".to_string(),
            "Test Voice".to_string(),
            "en".to_string(),
            VoiceGender::Female,
        );

        assert_eq!(voice.id, "test-voice");
        assert_eq!(voice.gender, VoiceGender::Female);
    }

    #[test]
    fn test_registry() {
        let registry = VoiceRegistry::new();
        assert!(!registry.voices().is_empty());

        let en_voices = registry.find_by_language("en");
        assert!(!en_voices.is_empty());
    }

    #[test]
    fn test_find_by_gender() {
        let registry = VoiceRegistry::new();
        let female_voices = registry.find_by_gender(VoiceGender::Female);
        assert!(!female_voices.is_empty());
    }
}
