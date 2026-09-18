#![allow(dead_code)]
//! Alternative text generation and management for media assets.
//!
//! Provides a system for creating, storing, and validating alt-text
//! descriptions for images, video thumbnails, and audio clips to
//! ensure accessibility compliance.

use std::collections::HashMap;

/// The kind of media that alt text describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// A still image or photograph.
    Image,
    /// A video thumbnail or poster frame.
    VideoThumbnail,
    /// A video clip.
    VideoClip,
    /// An audio clip.
    AudioClip,
    /// An animated graphic (GIF, APNG).
    Animation,
    /// An infographic or chart.
    Infographic,
}

impl MediaKind {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Image => "Image",
            Self::VideoThumbnail => "Video Thumbnail",
            Self::VideoClip => "Video Clip",
            Self::AudioClip => "Audio Clip",
            Self::Animation => "Animation",
            Self::Infographic => "Infographic",
        }
    }

    /// Return the recommended maximum character length for alt text.
    #[must_use]
    pub fn recommended_max_length(&self) -> usize {
        match self {
            Self::Image => 125,
            Self::VideoThumbnail => 100,
            Self::VideoClip => 250,
            Self::AudioClip => 200,
            Self::Animation => 150,
            Self::Infographic => 300,
        }
    }
}

/// The language of an alt text entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Language {
    /// BCP-47 language tag (e.g. "en-US", "ja-JP").
    pub tag: String,
}

impl Language {
    /// Create a new language from a BCP-47 tag.
    #[must_use]
    pub fn new(tag: impl Into<String>) -> Self {
        Self { tag: tag.into() }
    }

    /// English (US).
    #[must_use]
    pub fn en_us() -> Self {
        Self::new("en-US")
    }

    /// Japanese.
    #[must_use]
    pub fn ja_jp() -> Self {
        Self::new("ja-JP")
    }
}

/// Severity of an alt-text validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationSeverity {
    /// Informational note.
    Info,
    /// Minor issue, should be fixed.
    Warning,
    /// Serious issue, must be fixed for compliance.
    Error,
}

/// A single validation issue found in alt text.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: ValidationSeverity,
    /// Human-readable message.
    pub message: String,
}

impl ValidationIssue {
    /// Create a new validation issue.
    #[must_use]
    pub fn new(severity: ValidationSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
        }
    }
}

/// An alt-text entry for a single media asset in one language.
#[derive(Debug, Clone)]
pub struct AltTextEntry {
    /// The alt text content.
    pub text: String,
    /// Language of this entry.
    pub language: Language,
    /// Who authored this entry.
    pub author: Option<String>,
    /// Whether this entry has been reviewed/approved.
    pub approved: bool,
}

impl AltTextEntry {
    /// Create a new alt-text entry.
    #[must_use]
    pub fn new(text: impl Into<String>, language: Language) -> Self {
        Self {
            text: text.into(),
            language,
            author: None,
            approved: false,
        }
    }

    /// Set the author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Mark as approved.
    #[must_use]
    pub fn approved(mut self) -> Self {
        self.approved = true;
        self
    }

    /// Return the character count of the text.
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Return the word count of the text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Validate the entry for the given media kind.
    #[must_use]
    pub fn validate(&self, kind: MediaKind) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if self.text.is_empty() {
            issues.push(ValidationIssue::new(
                ValidationSeverity::Error,
                "Alt text must not be empty",
            ));
            return issues;
        }

        let max_len = kind.recommended_max_length();
        if self.char_count() > max_len {
            issues.push(ValidationIssue::new(
                ValidationSeverity::Warning,
                format!(
                    "Alt text exceeds recommended length of {} characters (got {})",
                    max_len,
                    self.char_count()
                ),
            ));
        }

        if self.char_count() < 5 {
            issues.push(ValidationIssue::new(
                ValidationSeverity::Warning,
                "Alt text is very short; consider adding more detail",
            ));
        }

        let lower = self.text.to_lowercase();
        let redundant_prefixes = ["image of", "picture of", "photo of", "graphic of"];
        for prefix in &redundant_prefixes {
            if lower.starts_with(prefix) {
                issues.push(ValidationIssue::new(
                    ValidationSeverity::Info,
                    format!("Alt text starts with redundant prefix \"{prefix}\""),
                ));
                break;
            }
        }

        issues
    }
}

/// A collection of alt-text entries for a single media asset, keyed by language.
#[derive(Debug, Clone)]
pub struct MediaAltText {
    /// The asset identifier.
    pub asset_id: String,
    /// The kind of media.
    pub kind: MediaKind,
    /// Entries keyed by language tag.
    entries: HashMap<String, AltTextEntry>,
}

impl MediaAltText {
    /// Create a new alt-text collection for an asset.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, kind: MediaKind) -> Self {
        Self {
            asset_id: asset_id.into(),
            kind,
            entries: HashMap::new(),
        }
    }

    /// Add or replace an alt-text entry for a language.
    pub fn set_entry(&mut self, entry: AltTextEntry) {
        self.entries.insert(entry.language.tag.clone(), entry);
    }

    /// Get the entry for a given language tag.
    #[must_use]
    pub fn get_entry(&self, lang: &str) -> Option<&AltTextEntry> {
        self.entries.get(lang)
    }

    /// Remove the entry for a given language tag.
    pub fn remove_entry(&mut self, lang: &str) -> Option<AltTextEntry> {
        self.entries.remove(lang)
    }

    /// Return the number of language entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return all language tags that have entries.
    #[must_use]
    pub fn languages(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Check whether all entries are approved.
    #[must_use]
    pub fn all_approved(&self) -> bool {
        !self.entries.is_empty() && self.entries.values().all(|e| e.approved)
    }

    /// Validate all entries and return issues grouped by language.
    #[must_use]
    pub fn validate_all(&self) -> HashMap<String, Vec<ValidationIssue>> {
        let mut results = HashMap::new();
        for (lang, entry) in &self.entries {
            let issues = entry.validate(self.kind);
            if !issues.is_empty() {
                results.insert(lang.clone(), issues);
            }
        }
        results
    }

    /// Check whether at least one entry exists and has no errors.
    #[must_use]
    pub fn has_valid_entry(&self) -> bool {
        self.entries.values().any(|e| {
            let issues = e.validate(self.kind);
            !issues
                .iter()
                .any(|i| i.severity == ValidationSeverity::Error)
        })
    }
}

/// A registry of alt texts for multiple assets.
#[derive(Debug, Clone, Default)]
pub struct AltTextRegistry {
    /// Assets keyed by asset ID.
    assets: HashMap<String, MediaAltText>,
}

impl AltTextRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }

    /// Register an asset.
    pub fn register(&mut self, alt_text: MediaAltText) {
        self.assets.insert(alt_text.asset_id.clone(), alt_text);
    }

    /// Look up an asset.
    #[must_use]
    pub fn get(&self, asset_id: &str) -> Option<&MediaAltText> {
        self.assets.get(asset_id)
    }

    /// Return the number of registered assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Check whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Return IDs of assets missing alt text for the given language.
    #[must_use]
    pub fn missing_for_language(&self, lang: &str) -> Vec<&str> {
        self.assets
            .iter()
            .filter(|(_, a)| a.get_entry(lang).is_none())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Return the coverage ratio for a given language (0.0..=1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage(&self, lang: &str) -> f64 {
        if self.assets.is_empty() {
            return 0.0;
        }
        let covered = self
            .assets
            .values()
            .filter(|a| a.get_entry(lang).is_some())
            .count();
        covered as f64 / self.assets.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_kind_labels() {
        assert_eq!(MediaKind::Image.label(), "Image");
        assert_eq!(MediaKind::VideoClip.label(), "Video Clip");
        assert_eq!(MediaKind::AudioClip.label(), "Audio Clip");
    }

    #[test]
    fn test_recommended_max_length() {
        assert_eq!(MediaKind::Image.recommended_max_length(), 125);
        assert_eq!(MediaKind::Infographic.recommended_max_length(), 300);
    }

    #[test]
    fn test_alt_text_entry_creation() {
        let entry = AltTextEntry::new("A sunset over mountains", Language::en_us())
            .with_author("alice")
            .approved();
        assert_eq!(entry.char_count(), 23);
        assert_eq!(entry.word_count(), 4);
        assert!(entry.approved);
        assert_eq!(entry.author, Some("alice".to_string()));
    }

    #[test]
    fn test_validate_empty_text() {
        let entry = AltTextEntry::new("", Language::en_us());
        let issues = entry.validate(MediaKind::Image);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, ValidationSeverity::Error);
    }

    #[test]
    fn test_validate_too_long() {
        let long_text = "a".repeat(200);
        let entry = AltTextEntry::new(long_text, Language::en_us());
        let issues = entry.validate(MediaKind::Image);
        assert!(issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Warning));
    }

    #[test]
    fn test_validate_redundant_prefix() {
        let entry = AltTextEntry::new("Image of a cat sitting on a mat", Language::en_us());
        let issues = entry.validate(MediaKind::Image);
        assert!(issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Info));
    }

    #[test]
    fn test_validate_short_text() {
        let entry = AltTextEntry::new("Cat", Language::en_us());
        let issues = entry.validate(MediaKind::Image);
        assert!(issues.iter().any(|i| i.message.contains("very short")));
    }

    #[test]
    fn test_media_alt_text_set_get() {
        let mut mat = MediaAltText::new("asset-1", MediaKind::Image);
        mat.set_entry(AltTextEntry::new("A dog", Language::en_us()));
        mat.set_entry(AltTextEntry::new("Inu desu", Language::ja_jp()));
        assert_eq!(mat.entry_count(), 2);
        assert!(mat.get_entry("en-US").is_some());
        assert!(mat.get_entry("ja-JP").is_some());
        assert!(mat.get_entry("fr-FR").is_none());
    }

    #[test]
    fn test_media_alt_text_remove() {
        let mut mat = MediaAltText::new("asset-1", MediaKind::Image);
        mat.set_entry(AltTextEntry::new("text", Language::en_us()));
        let removed = mat.remove_entry("en-US");
        assert!(removed.is_some());
        assert_eq!(mat.entry_count(), 0);
    }

    #[test]
    fn test_all_approved() {
        let mut mat = MediaAltText::new("a", MediaKind::Image);
        mat.set_entry(AltTextEntry::new("text", Language::en_us()).approved());
        assert!(mat.all_approved());
        mat.set_entry(AltTextEntry::new("texte", Language::new("fr-FR")));
        assert!(!mat.all_approved());
    }

    #[test]
    fn test_validate_all() {
        let mut mat = MediaAltText::new("a", MediaKind::Image);
        mat.set_entry(AltTextEntry::new("", Language::en_us()));
        mat.set_entry(AltTextEntry::new(
            "Good description of the scene",
            Language::ja_jp(),
        ));
        let issues = mat.validate_all();
        assert!(issues.contains_key("en-US"));
        assert!(!issues.contains_key("ja-JP"));
    }

    #[test]
    fn test_registry_coverage() {
        let mut reg = AltTextRegistry::new();
        let mut a = MediaAltText::new("a1", MediaKind::Image);
        a.set_entry(AltTextEntry::new("desc", Language::en_us()));
        reg.register(a);
        reg.register(MediaAltText::new("a2", MediaKind::Image));
        assert!((reg.coverage("en-US") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_registry_missing_for_language() {
        let mut reg = AltTextRegistry::new();
        let mut a = MediaAltText::new("a1", MediaKind::Image);
        a.set_entry(AltTextEntry::new("desc", Language::en_us()));
        reg.register(a);
        reg.register(MediaAltText::new("a2", MediaKind::Image));
        let missing = reg.missing_for_language("en-US");
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&"a2"));
    }

    #[test]
    fn test_has_valid_entry() {
        let mut mat = MediaAltText::new("a", MediaKind::Image);
        mat.set_entry(AltTextEntry::new("A proper description", Language::en_us()));
        assert!(mat.has_valid_entry());
    }

    #[test]
    fn test_language_constructors() {
        assert_eq!(Language::en_us().tag, "en-US");
        assert_eq!(Language::ja_jp().tag, "ja-JP");
    }
}
