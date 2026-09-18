//! Subtitle translation.

pub mod language;
pub mod quality;
pub mod subtitle;

pub use language::{Language, LanguageDetector};
pub use quality::TranslationQualityChecker;
pub use subtitle::SubtitleTranslator;

use serde::{Deserialize, Serialize};

/// Translation service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Source language code.
    pub source_lang: String,
    /// Target language code.
    pub target_lang: String,
    /// Preserve timing.
    pub preserve_timing: bool,
    /// Maximum characters per line in target language.
    pub max_chars_per_line: usize,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            source_lang: "en".to_string(),
            target_lang: "es".to_string(),
            preserve_timing: true,
            max_chars_per_line: 42,
        }
    }
}
