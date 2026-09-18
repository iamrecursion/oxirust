//! Language detection and management.

use serde::{Deserialize, Serialize};

/// Supported languages for translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    /// English.
    English,
    /// Spanish.
    Spanish,
    /// French.
    French,
    /// German.
    German,
    /// Italian.
    Italian,
    /// Portuguese.
    Portuguese,
    /// Russian.
    Russian,
    /// Japanese.
    Japanese,
    /// Chinese (Simplified).
    ChineseSimplified,
    /// Chinese (Traditional).
    ChineseTraditional,
    /// Korean.
    Korean,
    /// Arabic.
    Arabic,
    /// Hindi.
    Hindi,
    /// Dutch.
    Dutch,
    /// Polish.
    Polish,
    /// Turkish.
    Turkish,
    /// Vietnamese.
    Vietnamese,
    /// Thai.
    Thai,
    /// Swedish.
    Swedish,
    /// Norwegian.
    Norwegian,
}

impl Language {
    /// Get ISO 639-1 language code.
    #[must_use]
    pub const fn code(&self) -> &str {
        match self {
            Self::English => "en",
            Self::Spanish => "es",
            Self::French => "fr",
            Self::German => "de",
            Self::Italian => "it",
            Self::Portuguese => "pt",
            Self::Russian => "ru",
            Self::Japanese => "ja",
            Self::ChineseSimplified => "zh-CN",
            Self::ChineseTraditional => "zh-TW",
            Self::Korean => "ko",
            Self::Arabic => "ar",
            Self::Hindi => "hi",
            Self::Dutch => "nl",
            Self::Polish => "pl",
            Self::Turkish => "tr",
            Self::Vietnamese => "vi",
            Self::Thai => "th",
            Self::Swedish => "sv",
            Self::Norwegian => "no",
        }
    }

    /// Get language name.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::English => "English",
            Self::Spanish => "Spanish",
            Self::French => "French",
            Self::German => "German",
            Self::Italian => "Italian",
            Self::Portuguese => "Portuguese",
            Self::Russian => "Russian",
            Self::Japanese => "Japanese",
            Self::ChineseSimplified => "Chinese (Simplified)",
            Self::ChineseTraditional => "Chinese (Traditional)",
            Self::Korean => "Korean",
            Self::Arabic => "Arabic",
            Self::Hindi => "Hindi",
            Self::Dutch => "Dutch",
            Self::Polish => "Polish",
            Self::Turkish => "Turkish",
            Self::Vietnamese => "Vietnamese",
            Self::Thai => "Thai",
            Self::Swedish => "Swedish",
            Self::Norwegian => "Norwegian",
        }
    }

    /// Parse from language code.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" => Some(Self::English),
            "es" => Some(Self::Spanish),
            "fr" => Some(Self::French),
            "de" => Some(Self::German),
            "it" => Some(Self::Italian),
            "pt" => Some(Self::Portuguese),
            "ru" => Some(Self::Russian),
            "ja" => Some(Self::Japanese),
            "zh-cn" | "zh" => Some(Self::ChineseSimplified),
            "zh-tw" => Some(Self::ChineseTraditional),
            "ko" => Some(Self::Korean),
            "ar" => Some(Self::Arabic),
            "hi" => Some(Self::Hindi),
            "nl" => Some(Self::Dutch),
            "pl" => Some(Self::Polish),
            "tr" => Some(Self::Turkish),
            "vi" => Some(Self::Vietnamese),
            "th" => Some(Self::Thai),
            "sv" => Some(Self::Swedish),
            "no" => Some(Self::Norwegian),
            _ => None,
        }
    }
}

/// Detects language from text.
pub struct LanguageDetector;

impl LanguageDetector {
    /// Detect language from text.
    ///
    /// Integration point for language detection services.
    #[must_use]
    pub fn detect(_text: &str) -> Option<Language> {
        // Placeholder: Call language detection service
        // In production: Google Cloud Translation, AWS Comprehend, etc.
        Some(Language::English)
    }

    /// Get confidence score for detection (0.0 to 1.0).
    #[must_use]
    pub fn detect_with_confidence(_text: &str) -> (Option<Language>, f32) {
        (Some(Language::English), 0.95)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_code() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::Spanish.code(), "es");
        assert_eq!(Language::Japanese.code(), "ja");
    }

    #[test]
    fn test_from_code() {
        assert_eq!(Language::from_code("en"), Some(Language::English));
        assert_eq!(Language::from_code("ES"), Some(Language::Spanish));
        assert_eq!(Language::from_code("invalid"), None);
    }

    #[test]
    fn test_language_name() {
        assert_eq!(Language::English.name(), "English");
        assert_eq!(Language::ChineseSimplified.name(), "Chinese (Simplified)");
    }
}
