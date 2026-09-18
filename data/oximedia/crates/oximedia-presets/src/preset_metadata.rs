//! Preset metadata, tagging, categorization, and index support.

#![allow(dead_code)]

use std::collections::HashMap;

/// High-level grouping for preset organization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PresetCategory {
    /// Streaming platform (YouTube, Vimeo, etc.).
    Platform,
    /// Broadcast delivery standard (ATSC, DVB, etc.).
    Broadcast,
    /// Archive or long-term preservation.
    Archive,
    /// Mobile device delivery.
    Mobile,
    /// Social-media short-form content.
    Social,
    /// Codec-specific tuning preset.
    Codec,
    /// Quality-tier preset (low / medium / high / best).
    Quality,
    /// User-defined custom preset.
    Custom,
    /// Streaming ABR ladder rung.
    Streaming,
}

impl PresetCategory {
    /// Return a human-readable label for the category.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Platform => "Platform",
            Self::Broadcast => "Broadcast",
            Self::Archive => "Archive",
            Self::Mobile => "Mobile",
            Self::Social => "Social",
            Self::Codec => "Codec",
            Self::Quality => "Quality",
            Self::Custom => "Custom",
            Self::Streaming => "Streaming",
        }
    }

    /// Return `true` if the category is related to a specific delivery platform.
    #[must_use]
    pub fn is_delivery_category(&self) -> bool {
        matches!(
            self,
            Self::Platform | Self::Broadcast | Self::Streaming | Self::Social | Self::Mobile
        )
    }
}

/// A descriptive tag attached to a preset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PresetTag {
    /// Tag key (e.g. "hdr", "hls", "vertical").
    pub key: String,
    /// Optional free-form value (e.g. "2160p").
    pub value: Option<String>,
    /// Whether this tag only applies to a specific platform.
    pub platform_specific: bool,
}

impl PresetTag {
    /// Create a simple key-only tag.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            platform_specific: false,
        }
    }

    /// Create a key-value tag.
    #[must_use]
    pub fn with_value(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: Some(value.into()),
            platform_specific: false,
        }
    }

    /// Mark this tag as platform-specific.
    #[must_use]
    pub fn platform_specific(mut self) -> Self {
        self.platform_specific = true;
        self
    }

    /// Return `true` if this tag is specific to a single platform.
    #[must_use]
    pub fn is_platform_specific(&self) -> bool {
        self.platform_specific
    }

    /// Return the display string `"key"` or `"key=value"`.
    #[must_use]
    pub fn display(&self) -> String {
        match &self.value {
            Some(v) => format!("{}={}", self.key, v),
            None => self.key.clone(),
        }
    }
}

/// Rich metadata record for a single preset.
#[derive(Debug, Clone)]
pub struct PresetMetadata {
    /// Unique machine-readable identifier.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Short description for UI tooltips.
    pub description: String,
    /// Organizational category.
    pub category: PresetCategory,
    /// Tags attached to this preset.
    pub tags: Vec<PresetTag>,
    /// Semantic version string (e.g. `"2.1.0"`).
    pub version: String,
    /// Minimum library version required to use this preset.
    pub min_lib_version: String,
    /// Whether this preset ships with OxiMedia (vs. user-created).
    pub built_in: bool,
    /// Whether the preset has been marked as deprecated.
    pub deprecated: bool,
    /// Deprecation notice message, if any.
    pub deprecation_notice: Option<String>,
}

impl PresetMetadata {
    /// Current semantic version used by built-in presets.
    pub const CURRENT_VERSION: &'static str = "2.0.0";

    /// Create a new metadata record with sensible defaults.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, category: PresetCategory) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            category,
            tags: Vec::new(),
            version: Self::CURRENT_VERSION.to_string(),
            min_lib_version: "1.0.0".to_string(),
            built_in: true,
            deprecated: false,
            deprecation_notice: None,
        }
    }

    /// Return `true` if this preset's version matches the current library version.
    #[must_use]
    pub fn is_current_version(&self) -> bool {
        self.version == Self::CURRENT_VERSION
    }

    /// Add a tag by moving it into the tag list.
    pub fn push_tag(&mut self, tag: PresetTag) {
        self.tags.push(tag);
    }

    /// Check whether a tag with the given key is present.
    #[must_use]
    pub fn has_tag(&self, key: &str) -> bool {
        self.tags.iter().any(|t| t.key == key)
    }

    /// Mark this preset as deprecated with an optional notice.
    pub fn deprecate(&mut self, notice: impl Into<String>) {
        self.deprecated = true;
        self.deprecation_notice = Some(notice.into());
    }
}

/// An indexed collection of `PresetMetadata` records.
#[derive(Debug, Default)]
pub struct PresetIndex {
    entries: HashMap<String, PresetMetadata>,
    /// Category -> list of preset IDs.
    category_map: HashMap<String, Vec<String>>,
}

impl PresetIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new preset (or overwrite an existing one with the same ID).
    pub fn register(&mut self, meta: PresetMetadata) {
        let key = meta.category.label().to_string();
        self.category_map
            .entry(key)
            .or_default()
            .push(meta.id.clone());
        self.entries.insert(meta.id.clone(), meta);
    }

    /// Look up a preset by its ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&PresetMetadata> {
        self.entries.get(id)
    }

    /// Find all presets belonging to a given category.
    #[must_use]
    pub fn find_by_category(&self, category: &PresetCategory) -> Vec<&PresetMetadata> {
        let key = category.label();
        self.category_map
            .get(key)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.entries.get(id.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all presets that carry a specific tag key.
    #[must_use]
    pub fn find_by_tag(&self, tag_key: &str) -> Vec<&PresetMetadata> {
        self.entries
            .values()
            .filter(|m| m.has_tag(tag_key))
            .collect()
    }

    /// Total number of registered presets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the index contains no presets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all preset IDs in the index.
    #[must_use]
    pub fn all_ids(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PresetCategory ---

    #[test]
    fn test_category_label_platform() {
        assert_eq!(PresetCategory::Platform.label(), "Platform");
    }

    #[test]
    fn test_category_label_broadcast() {
        assert_eq!(PresetCategory::Broadcast.label(), "Broadcast");
    }

    #[test]
    fn test_category_is_delivery() {
        assert!(PresetCategory::Platform.is_delivery_category());
        assert!(PresetCategory::Social.is_delivery_category());
        assert!(!PresetCategory::Archive.is_delivery_category());
        assert!(!PresetCategory::Codec.is_delivery_category());
    }

    // --- PresetTag ---

    #[test]
    fn test_tag_new() {
        let tag = PresetTag::new("hdr");
        assert_eq!(tag.key, "hdr");
        assert!(tag.value.is_none());
        assert!(!tag.is_platform_specific());
    }

    #[test]
    fn test_tag_with_value() {
        let tag = PresetTag::with_value("resolution", "2160p");
        assert_eq!(tag.key, "resolution");
        assert_eq!(tag.value.as_deref(), Some("2160p"));
        assert_eq!(tag.display(), "resolution=2160p");
    }

    #[test]
    fn test_tag_platform_specific() {
        let tag = PresetTag::new("shorts").platform_specific();
        assert!(tag.is_platform_specific());
    }

    #[test]
    fn test_tag_display_no_value() {
        let tag = PresetTag::new("hls");
        assert_eq!(tag.display(), "hls");
    }

    // --- PresetMetadata ---

    #[test]
    fn test_metadata_new_defaults() {
        let m = PresetMetadata::new("yt-1080", "YouTube 1080p", PresetCategory::Platform);
        assert_eq!(m.id, "yt-1080");
        assert!(m.built_in);
        assert!(!m.deprecated);
        assert!(m.deprecation_notice.is_none());
    }

    #[test]
    fn test_metadata_is_current_version() {
        let m = PresetMetadata::new("x", "X", PresetCategory::Custom);
        assert!(m.is_current_version());
    }

    #[test]
    fn test_metadata_not_current_version() {
        let mut m = PresetMetadata::new("x", "X", PresetCategory::Custom);
        m.version = "1.0.0".to_string();
        assert!(!m.is_current_version());
    }

    #[test]
    fn test_metadata_has_tag() {
        let mut m = PresetMetadata::new("x", "X", PresetCategory::Codec);
        m.push_tag(PresetTag::new("av1"));
        assert!(m.has_tag("av1"));
        assert!(!m.has_tag("hevc"));
    }

    #[test]
    fn test_metadata_deprecate() {
        let mut m = PresetMetadata::new("old", "Old Preset", PresetCategory::Quality);
        m.deprecate("Use new-preset instead");
        assert!(m.deprecated);
        assert_eq!(
            m.deprecation_notice.as_deref(),
            Some("Use new-preset instead")
        );
    }

    // --- PresetIndex ---

    #[test]
    fn test_index_register_and_get() {
        let mut idx = PresetIndex::new();
        let m = PresetMetadata::new("hls-360", "HLS 360p", PresetCategory::Streaming);
        idx.register(m);
        assert!(idx.get("hls-360").is_some());
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_index_find_by_category() {
        let mut idx = PresetIndex::new();
        idx.register(PresetMetadata::new("a", "A", PresetCategory::Archive));
        idx.register(PresetMetadata::new("b", "B", PresetCategory::Archive));
        idx.register(PresetMetadata::new("c", "C", PresetCategory::Mobile));
        let archive = idx.find_by_category(&PresetCategory::Archive);
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn test_index_find_by_tag() {
        let mut idx = PresetIndex::new();
        let mut m = PresetMetadata::new("tagged", "Tagged", PresetCategory::Codec);
        m.push_tag(PresetTag::new("hdr10"));
        idx.register(m);
        idx.register(PresetMetadata::new("plain", "Plain", PresetCategory::Codec));
        let found = idx.find_by_tag("hdr10");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "tagged");
    }

    #[test]
    fn test_index_is_empty() {
        let idx = PresetIndex::new();
        assert!(idx.is_empty());
    }

    #[test]
    fn test_index_all_ids_count() {
        let mut idx = PresetIndex::new();
        for i in 0..5 {
            idx.register(PresetMetadata::new(
                format!("p{i}"),
                format!("Preset {i}"),
                PresetCategory::Quality,
            ));
        }
        assert_eq!(idx.all_ids().len(), 5);
    }
}
