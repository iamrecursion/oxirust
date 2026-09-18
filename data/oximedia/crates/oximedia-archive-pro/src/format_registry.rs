#![allow(dead_code)]

//! Format registry for tracking archival file formats.
//!
//! Provides a centralised registry of media file formats with metadata about
//! their preservation risk, supported tools, and recommended migration targets.

use std::collections::HashMap;
use std::fmt;

/// Risk level indicating how likely a format is to become obsolete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RiskLevel {
    /// Minimal risk — widely supported open format.
    Minimal,
    /// Low risk — well-supported, broad ecosystem.
    Low,
    /// Moderate risk — support is narrowing.
    Moderate,
    /// High risk — few tools still support it.
    High,
    /// Critical risk — format is effectively obsolete.
    Critical,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimal => write!(f, "Minimal"),
            Self::Low => write!(f, "Low"),
            Self::Moderate => write!(f, "Moderate"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Category of a file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatCategory {
    /// Video container or codec.
    Video,
    /// Audio container or codec.
    Audio,
    /// Still image.
    Image,
    /// Document.
    Document,
    /// Subtitle / caption.
    Subtitle,
    /// Generic / other.
    Other,
}

impl fmt::Display for FormatCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Image => write!(f, "Image"),
            Self::Document => write!(f, "Document"),
            Self::Subtitle => write!(f, "Subtitle"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// A single entry in the format registry.
#[derive(Debug, Clone)]
pub struct FormatEntry {
    /// Short identifier (e.g. "ffv1-mkv").
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// MIME type.
    pub mime_type: String,
    /// Common file extensions.
    pub extensions: Vec<String>,
    /// Category.
    pub category: FormatCategory,
    /// Obsolescence risk level.
    pub risk: RiskLevel,
    /// Whether the format is open / royalty-free.
    pub is_open: bool,
    /// Whether lossless encoding is supported.
    pub supports_lossless: bool,
    /// Recommended migration target format id (if any).
    pub migration_target: Option<String>,
    /// Year the format was first published.
    pub year_published: Option<u16>,
    /// Free-form notes.
    pub notes: String,
}

impl FormatEntry {
    /// Create a new format entry with required fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        mime_type: impl Into<String>,
        category: FormatCategory,
        risk: RiskLevel,
        is_open: bool,
        supports_lossless: bool,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            mime_type: mime_type.into(),
            extensions: Vec::new(),
            category,
            risk,
            is_open,
            supports_lossless,
            migration_target: None,
            year_published: None,
            notes: String::new(),
        }
    }

    /// Add a file extension.
    pub fn add_extension(&mut self, ext: impl Into<String>) {
        self.extensions.push(ext.into());
    }

    /// Set the recommended migration target.
    pub fn set_migration_target(&mut self, target: impl Into<String>) {
        self.migration_target = Some(target.into());
    }

    /// Set the year published.
    pub fn set_year_published(&mut self, year: u16) {
        self.year_published = Some(year);
    }

    /// Set notes.
    pub fn set_notes(&mut self, notes: impl Into<String>) {
        self.notes = notes.into();
    }

    /// Check whether this format needs migration attention.
    pub fn needs_migration(&self) -> bool {
        matches!(self.risk, RiskLevel::High | RiskLevel::Critical)
    }

    /// Calculate an approximate age in years from a given reference year.
    pub fn age_years(&self, reference_year: u16) -> Option<u16> {
        self.year_published
            .map(|y| reference_year.saturating_sub(y))
    }
}

/// A registry that holds multiple format entries indexed by id.
#[derive(Debug, Clone, Default)]
pub struct FormatRegistry {
    /// Map from format id to entry.
    entries: HashMap<String, FormatEntry>,
}

impl FormatRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a format entry. Overwrites if the id already exists.
    pub fn register(&mut self, entry: FormatEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Look up a format by id.
    pub fn get(&self, id: &str) -> Option<&FormatEntry> {
        self.entries.get(id)
    }

    /// Remove a format by id.
    pub fn remove(&mut self, id: &str) -> Option<FormatEntry> {
        self.entries.remove(id)
    }

    /// Total number of registered formats.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &FormatEntry)> {
        self.entries.iter()
    }

    /// Find all formats that match a given category.
    pub fn by_category(&self, category: FormatCategory) -> Vec<&FormatEntry> {
        self.entries
            .values()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Find all formats at a given risk level or higher.
    pub fn at_risk(&self, min_level: RiskLevel) -> Vec<&FormatEntry> {
        self.entries
            .values()
            .filter(|e| e.risk >= min_level)
            .collect()
    }

    /// Find all formats that need migration.
    pub fn needing_migration(&self) -> Vec<&FormatEntry> {
        self.entries
            .values()
            .filter(|e| e.needs_migration())
            .collect()
    }

    /// Find all open formats.
    pub fn open_formats(&self) -> Vec<&FormatEntry> {
        self.entries.values().filter(|e| e.is_open).collect()
    }

    /// Find formats by extension.
    pub fn by_extension(&self, ext: &str) -> Vec<&FormatEntry> {
        self.entries
            .values()
            .filter(|e| e.extensions.iter().any(|x| x == ext))
            .collect()
    }

    /// Build a registry pre-loaded with common archival formats.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();

        let mut ffv1 = FormatEntry::new(
            "ffv1-mkv",
            "FFV1 in Matroska",
            "video/x-matroska",
            FormatCategory::Video,
            RiskLevel::Minimal,
            true,
            true,
        );
        ffv1.add_extension("mkv");
        ffv1.set_year_published(2003);
        reg.register(ffv1);

        let mut flac = FormatEntry::new(
            "flac",
            "FLAC",
            "audio/flac",
            FormatCategory::Audio,
            RiskLevel::Minimal,
            true,
            true,
        );
        flac.add_extension("flac");
        flac.set_year_published(2001);
        reg.register(flac);

        let mut tiff = FormatEntry::new(
            "tiff",
            "TIFF",
            "image/tiff",
            FormatCategory::Image,
            RiskLevel::Low,
            true,
            true,
        );
        tiff.add_extension("tiff");
        tiff.add_extension("tif");
        tiff.set_year_published(1986);
        reg.register(tiff);

        let mut wav = FormatEntry::new(
            "wav",
            "WAV PCM",
            "audio/wav",
            FormatCategory::Audio,
            RiskLevel::Minimal,
            true,
            true,
        );
        wav.add_extension("wav");
        wav.set_year_published(1991);
        reg.register(wav);

        let mut avi = FormatEntry::new(
            "avi",
            "AVI",
            "video/x-msvideo",
            FormatCategory::Video,
            RiskLevel::Moderate,
            false,
            false,
        );
        avi.add_extension("avi");
        avi.set_year_published(1992);
        avi.set_migration_target("ffv1-mkv");
        reg.register(avi);

        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> FormatEntry {
        let mut e = FormatEntry::new(
            "test-fmt",
            "Test Format",
            "application/test",
            FormatCategory::Video,
            RiskLevel::Low,
            true,
            false,
        );
        e.add_extension("test");
        e
    }

    #[test]
    fn test_entry_creation() {
        let e = sample_entry();
        assert_eq!(e.id, "test-fmt");
        assert_eq!(e.category, FormatCategory::Video);
        assert_eq!(e.risk, RiskLevel::Low);
        assert!(e.is_open);
    }

    #[test]
    fn test_entry_needs_migration() {
        let mut e = sample_entry();
        assert!(!e.needs_migration());
        e.risk = RiskLevel::High;
        assert!(e.needs_migration());
        e.risk = RiskLevel::Critical;
        assert!(e.needs_migration());
    }

    #[test]
    fn test_entry_age() {
        let mut e = sample_entry();
        e.set_year_published(2000);
        assert_eq!(e.age_years(2026), Some(26));
        let e2 = sample_entry();
        assert_eq!(e2.age_years(2026), None);
    }

    #[test]
    fn test_entry_migration_target() {
        let mut e = sample_entry();
        assert!(e.migration_target.is_none());
        e.set_migration_target("ffv1-mkv");
        assert_eq!(e.migration_target.as_deref(), Some("ffv1-mkv"));
    }

    #[test]
    fn test_registry_empty() {
        let reg = FormatRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = FormatRegistry::new();
        reg.register(sample_entry());
        assert_eq!(reg.len(), 1);
        assert!(reg.get("test-fmt").is_some());
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = FormatRegistry::new();
        reg.register(sample_entry());
        let removed = reg.remove("test-fmt");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_by_category() {
        let mut reg = FormatRegistry::new();
        reg.register(sample_entry());
        let mut audio = FormatEntry::new(
            "aud",
            "Audio",
            "audio/test",
            FormatCategory::Audio,
            RiskLevel::Minimal,
            true,
            true,
        );
        audio.add_extension("aud");
        reg.register(audio);
        assert_eq!(reg.by_category(FormatCategory::Video).len(), 1);
        assert_eq!(reg.by_category(FormatCategory::Audio).len(), 1);
    }

    #[test]
    fn test_registry_at_risk() {
        let mut reg = FormatRegistry::new();
        let mut high = sample_entry();
        high.id = "high".into();
        high.risk = RiskLevel::High;
        reg.register(high);
        reg.register(sample_entry()); // Low
        assert_eq!(reg.at_risk(RiskLevel::High).len(), 1);
        assert_eq!(reg.at_risk(RiskLevel::Low).len(), 2);
    }

    #[test]
    fn test_registry_open_formats() {
        let mut reg = FormatRegistry::new();
        reg.register(sample_entry()); // is_open = true
        let mut closed = sample_entry();
        closed.id = "closed".into();
        closed.is_open = false;
        reg.register(closed);
        assert_eq!(reg.open_formats().len(), 1);
    }

    #[test]
    fn test_registry_by_extension() {
        let mut reg = FormatRegistry::new();
        reg.register(sample_entry()); // ext = "test"
        assert_eq!(reg.by_extension("test").len(), 1);
        assert_eq!(reg.by_extension("nope").len(), 0);
    }

    #[test]
    fn test_registry_defaults() {
        let reg = FormatRegistry::with_defaults();
        assert!(reg.len() >= 5);
        assert!(reg.get("ffv1-mkv").is_some());
        assert!(reg.get("flac").is_some());
        assert!(reg.get("wav").is_some());
    }

    #[test]
    fn test_risk_display() {
        assert_eq!(RiskLevel::Minimal.to_string(), "Minimal");
        assert_eq!(RiskLevel::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_category_display() {
        assert_eq!(FormatCategory::Video.to_string(), "Video");
        assert_eq!(FormatCategory::Audio.to_string(), "Audio");
    }

    #[test]
    fn test_registry_needing_migration() {
        let reg = FormatRegistry::with_defaults();
        // AVI has Moderate risk, not High/Critical, so needing_migration should be empty
        // for defaults. Let's add one with High risk.
        let mut reg2 = reg;
        let mut risky = sample_entry();
        risky.id = "risky-fmt".into();
        risky.risk = RiskLevel::Critical;
        reg2.register(risky);
        assert_eq!(reg2.needing_migration().len(), 1);
    }

    #[test]
    fn test_entry_notes() {
        let mut e = sample_entry();
        assert!(e.notes.is_empty());
        e.set_notes("Some note");
        assert_eq!(e.notes, "Some note");
    }

    #[test]
    fn test_risk_ordering() {
        assert!(RiskLevel::Minimal < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Moderate);
        assert!(RiskLevel::Moderate < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }
}
