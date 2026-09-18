#![allow(dead_code)]
//! Scene-level metadata: tags, environmental descriptors, and keyword sets.

use std::collections::HashSet;

/// A set of string tags attached to a scene.
#[derive(Debug, Clone, Default)]
pub struct SceneTags {
    tags: HashSet<String>,
}

impl SceneTags {
    /// Create an empty `SceneTags` collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: HashSet::new(),
        }
    }

    /// Add a tag (case-insensitive normalisation: stored lowercase).
    pub fn add_tag(&mut self, tag: &str) {
        self.tags.insert(tag.to_lowercase());
    }

    /// Returns `true` if the tag exists (case-insensitive).
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_lowercase())
    }

    /// Number of tags.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Returns `true` if no tags are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Iterator over tags in an unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.tags.iter().map(String::as_str)
    }
}

// ---------------------------------------------------------------------------

/// High-level descriptive metadata for a scene.
#[derive(Debug, Clone)]
pub struct SceneMetadata {
    /// Whether the scene takes place outdoors.
    pub outdoor: bool,
    /// Whether the scene is lit by daylight (as opposed to artificial light / night).
    pub daylight: bool,
    /// General location description (e.g. "beach", "office").
    pub location: String,
    /// Attached tags.
    pub tags: SceneTags,
}

impl SceneMetadata {
    /// Create a new `SceneMetadata`.
    #[must_use]
    pub fn new(outdoor: bool, daylight: bool, location: impl Into<String>) -> Self {
        Self {
            outdoor,
            daylight,
            location: location.into(),
            tags: SceneTags::new(),
        }
    }

    /// Returns `true` if this is an outdoor scene.
    #[must_use]
    pub fn is_outdoor(&self) -> bool {
        self.outdoor
    }

    /// Returns `true` if this scene is daylit.
    #[must_use]
    pub fn is_daylight(&self) -> bool {
        self.daylight
    }
}

impl Default for SceneMetadata {
    fn default() -> Self {
        Self::new(false, true, "unknown")
    }
}

// ---------------------------------------------------------------------------

/// A keyword set that can be merged with other sets.
#[derive(Debug, Clone, Default)]
pub struct SceneKeywords {
    keywords: HashSet<String>,
}

impl SceneKeywords {
    /// Create an empty `SceneKeywords`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keywords: HashSet::new(),
        }
    }

    /// Add a keyword.
    pub fn add(&mut self, kw: &str) {
        self.keywords.insert(kw.to_lowercase());
    }

    /// Merge another `SceneKeywords` into this one (union).
    pub fn merge(&mut self, other: &SceneKeywords) {
        for kw in &other.keywords {
            self.keywords.insert(kw.clone());
        }
    }

    /// Returns `true` if `keyword` is present.
    #[must_use]
    pub fn contains(&self, keyword: &str) -> bool {
        self.keywords.contains(&keyword.to_lowercase())
    }

    /// Number of unique keywords.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keywords.len()
    }

    /// Returns `true` if there are no keywords.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keywords.is_empty()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SceneTags ---

    #[test]
    fn test_add_and_has_tag() {
        let mut tags = SceneTags::new();
        tags.add_tag("Action");
        assert!(tags.has_tag("action"));
        assert!(tags.has_tag("ACTION"));
    }

    #[test]
    fn test_has_tag_false() {
        let tags = SceneTags::new();
        assert!(!tags.has_tag("drama"));
    }

    #[test]
    fn test_len_empty() {
        let tags = SceneTags::new();
        assert_eq!(tags.len(), 0);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_len_after_adds() {
        let mut tags = SceneTags::new();
        tags.add_tag("a");
        tags.add_tag("b");
        tags.add_tag("A"); // duplicate after lowercasing
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_iter_yields_all_tags() {
        let mut tags = SceneTags::new();
        tags.add_tag("x");
        tags.add_tag("y");
        let collected: Vec<&str> = tags.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    // --- SceneMetadata ---

    #[test]
    fn test_is_outdoor_true() {
        let meta = SceneMetadata::new(true, true, "park");
        assert!(meta.is_outdoor());
    }

    #[test]
    fn test_is_outdoor_false() {
        let meta = SceneMetadata::new(false, true, "studio");
        assert!(!meta.is_outdoor());
    }

    #[test]
    fn test_is_daylight_true() {
        let meta = SceneMetadata::new(true, true, "beach");
        assert!(meta.is_daylight());
    }

    #[test]
    fn test_is_daylight_false() {
        let meta = SceneMetadata::new(false, false, "nightclub");
        assert!(!meta.is_daylight());
    }

    #[test]
    fn test_default_metadata() {
        let meta = SceneMetadata::default();
        assert!(!meta.is_outdoor());
        assert!(meta.is_daylight());
        assert_eq!(meta.location, "unknown");
    }

    #[test]
    fn test_metadata_tags_integration() {
        let mut meta = SceneMetadata::new(true, true, "forest");
        meta.tags.add_tag("nature");
        assert!(meta.tags.has_tag("nature"));
    }

    // --- SceneKeywords ---

    #[test]
    fn test_keywords_add_and_contains() {
        let mut kw = SceneKeywords::new();
        kw.add("Sunset");
        assert!(kw.contains("sunset"));
        assert!(kw.contains("SUNSET"));
    }

    #[test]
    fn test_keywords_merge() {
        let mut kw1 = SceneKeywords::new();
        kw1.add("alpha");

        let mut kw2 = SceneKeywords::new();
        kw2.add("beta");
        kw2.add("gamma");

        kw1.merge(&kw2);
        assert_eq!(kw1.len(), 3);
        assert!(kw1.contains("alpha"));
        assert!(kw1.contains("beta"));
        assert!(kw1.contains("gamma"));
    }

    #[test]
    fn test_keywords_merge_deduplicates() {
        let mut kw1 = SceneKeywords::new();
        kw1.add("shared");

        let mut kw2 = SceneKeywords::new();
        kw2.add("SHARED"); // same after lowercasing

        kw1.merge(&kw2);
        assert_eq!(kw1.len(), 1);
    }

    #[test]
    fn test_keywords_is_empty() {
        let kw = SceneKeywords::new();
        assert!(kw.is_empty());
    }
}
