//! Tag-based preset indexing and retrieval.
//!
//! Provides a fast inverted index that maps tag strings to preset IDs,
//! enabling efficient multi-tag filtering and tag-cloud generation.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// A single tag attached to a preset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PresetTag(String);

impl PresetTag {
    /// Create a new tag (normalised to lowercase, trimmed).
    #[must_use]
    pub fn new(s: &str) -> Self {
        Self(s.trim().to_lowercase())
    }

    /// Return the tag as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PresetTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for PresetTag {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Associates a preset ID with a set of tags.
#[derive(Debug, Clone)]
pub struct TaggedPreset {
    /// Preset identifier.
    pub id: String,
    /// Tags associated with this preset.
    pub tags: HashSet<PresetTag>,
}

impl TaggedPreset {
    /// Create a new tagged preset with no tags.
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            tags: HashSet::new(),
        }
    }

    /// Add a tag to this preset.
    pub fn add_tag(&mut self, tag: impl Into<PresetTag>) {
        self.tags.insert(tag.into());
    }

    /// Remove a tag from this preset.
    pub fn remove_tag(&mut self, tag: &PresetTag) {
        self.tags.remove(tag);
    }

    /// Returns `true` if the preset has the given tag.
    #[must_use]
    pub fn has_tag(&self, tag: &PresetTag) -> bool {
        self.tags.contains(tag)
    }

    /// Returns `true` if the preset matches ALL the supplied tags.
    #[must_use]
    pub fn matches_all(&self, tags: &[PresetTag]) -> bool {
        tags.iter().all(|t| self.tags.contains(t))
    }

    /// Returns `true` if the preset matches ANY of the supplied tags.
    #[must_use]
    pub fn matches_any(&self, tags: &[PresetTag]) -> bool {
        tags.iter().any(|t| self.tags.contains(t))
    }

    /// Number of tags attached.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }
}

/// Inverted index that maps tags to the set of preset IDs that carry them.
///
/// Enables O(tag-count) multi-tag intersection queries.
#[derive(Debug, Default)]
pub struct PresetTagIndex {
    /// tag → set of preset IDs
    index: HashMap<PresetTag, HashSet<String>>,
    /// preset id → tagged preset record
    presets: HashMap<String, TaggedPreset>,
}

impl PresetTagIndex {
    /// Create an empty tag index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a `TaggedPreset` in the index.
    ///
    /// Re-registering the same ID replaces the previous entry and updates
    /// the inverted index accordingly.
    pub fn register(&mut self, tagged: TaggedPreset) {
        // Remove stale index entries for this ID if it already existed.
        if let Some(old) = self.presets.get(&tagged.id) {
            for tag in &old.tags {
                if let Some(set) = self.index.get_mut(tag) {
                    set.remove(&tagged.id);
                }
            }
        }

        // Insert new index entries.
        for tag in &tagged.tags {
            self.index
                .entry(tag.clone())
                .or_default()
                .insert(tagged.id.clone());
        }

        self.presets.insert(tagged.id.clone(), tagged);
    }

    /// Remove a preset from the index entirely.
    pub fn deregister(&mut self, id: &str) {
        if let Some(old) = self.presets.remove(id) {
            for tag in &old.tags {
                if let Some(set) = self.index.get_mut(tag) {
                    set.remove(id);
                }
            }
        }
    }

    /// Return all preset IDs that carry the given tag.
    #[must_use]
    pub fn by_tag(&self, tag: &PresetTag) -> Vec<&str> {
        self.index
            .get(tag)
            .map(|set| set.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Return all preset IDs that carry ALL of the supplied tags (intersection).
    #[must_use]
    pub fn by_all_tags(&self, tags: &[PresetTag]) -> Vec<&str> {
        if tags.is_empty() {
            return self.presets.keys().map(String::as_str).collect();
        }

        // Start with the smallest set to minimise work.
        let mut sets: Vec<&HashSet<String>> =
            tags.iter().filter_map(|t| self.index.get(t)).collect();

        if sets.is_empty() {
            return Vec::new();
        }

        sets.sort_by_key(|s| s.len());
        let base = sets[0];

        base.iter()
            .filter(|id| sets[1..].iter().all(|s| s.contains(*id)))
            .map(String::as_str)
            .collect()
    }

    /// Return all preset IDs that carry ANY of the supplied tags (union).
    #[must_use]
    pub fn by_any_tag(&self, tags: &[PresetTag]) -> Vec<&str> {
        let mut seen: HashSet<&str> = HashSet::new();
        for tag in tags {
            if let Some(set) = self.index.get(tag) {
                for id in set {
                    seen.insert(id.as_str());
                }
            }
        }
        seen.into_iter().collect()
    }

    /// All tags known to the index, sorted lexicographically.
    #[must_use]
    pub fn all_tags(&self) -> Vec<&PresetTag> {
        let mut tags: Vec<&PresetTag> = self.index.keys().collect();
        tags.sort();
        tags
    }

    /// Number of presets registered.
    #[must_use]
    pub fn preset_count(&self) -> usize {
        self.presets.len()
    }

    /// Number of unique tags in the index.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.index.len()
    }

    /// Look up a registered tagged preset by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&TaggedPreset> {
        self.presets.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> PresetTagIndex {
        let mut idx = PresetTagIndex::new();

        let mut p1 = TaggedPreset::new("yt-1080p");
        p1.add_tag(PresetTag::new("platform"));
        p1.add_tag(PresetTag::new("hd"));
        p1.add_tag(PresetTag::new("youtube"));
        idx.register(p1);

        let mut p2 = TaggedPreset::new("yt-4k");
        p2.add_tag(PresetTag::new("platform"));
        p2.add_tag(PresetTag::new("uhd"));
        p2.add_tag(PresetTag::new("youtube"));
        idx.register(p2);

        let mut p3 = TaggedPreset::new("hls-720p");
        p3.add_tag(PresetTag::new("streaming"));
        p3.add_tag(PresetTag::new("hd"));
        p3.add_tag(PresetTag::new("hls"));
        idx.register(p3);

        idx
    }

    #[test]
    fn test_empty_index() {
        let idx = PresetTagIndex::new();
        assert_eq!(idx.preset_count(), 0);
        assert_eq!(idx.tag_count(), 0);
    }

    #[test]
    fn test_register_presets() {
        let idx = make_index();
        assert_eq!(idx.preset_count(), 3);
    }

    #[test]
    fn test_by_tag_returns_correct_ids() {
        let idx = make_index();
        let platform_tag = PresetTag::new("platform");
        let mut ids = idx.by_tag(&platform_tag);
        ids.sort();
        assert_eq!(ids, vec!["yt-1080p", "yt-4k"]);
    }

    #[test]
    fn test_by_tag_no_match() {
        let idx = make_index();
        let tag = PresetTag::new("nonexistent");
        assert!(idx.by_tag(&tag).is_empty());
    }

    #[test]
    fn test_by_all_tags_intersection() {
        let idx = make_index();
        let tags = vec![PresetTag::new("platform"), PresetTag::new("hd")];
        let result = idx.by_all_tags(&tags);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "yt-1080p");
    }

    #[test]
    fn test_by_any_tag_union() {
        let idx = make_index();
        let tags = vec![PresetTag::new("streaming"), PresetTag::new("uhd")];
        let mut result = idx.by_any_tag(&tags);
        result.sort();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"hls-720p"));
        assert!(result.contains(&"yt-4k"));
    }

    #[test]
    fn test_all_tags_sorted() {
        let idx = make_index();
        let tags = idx.all_tags();
        // Must be sorted
        for pair in tags.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn test_deregister_removes_from_index() {
        let mut idx = make_index();
        idx.deregister("yt-1080p");
        assert_eq!(idx.preset_count(), 2);
        let platform_tag = PresetTag::new("platform");
        let ids = idx.by_tag(&platform_tag);
        assert!(!ids.contains(&"yt-1080p"));
    }

    #[test]
    fn test_re_register_updates_index() {
        let mut idx = make_index();
        // Replace yt-1080p with a version that drops "hd" and adds "4k"
        let mut updated = TaggedPreset::new("yt-1080p");
        updated.add_tag(PresetTag::new("platform"));
        updated.add_tag(PresetTag::new("4k"));
        idx.register(updated);

        let hd_tag = PresetTag::new("hd");
        assert!(!idx.by_tag(&hd_tag).contains(&"yt-1080p"));

        let fourk_tag = PresetTag::new("4k");
        assert!(idx.by_tag(&fourk_tag).contains(&"yt-1080p"));
    }

    #[test]
    fn test_tagged_preset_matches_all() {
        let mut tp = TaggedPreset::new("p1");
        tp.add_tag(PresetTag::new("a"));
        tp.add_tag(PresetTag::new("b"));
        let all_ab = vec![PresetTag::new("a"), PresetTag::new("b")];
        assert!(tp.matches_all(&all_ab));
        let all_abc = vec![
            PresetTag::new("a"),
            PresetTag::new("b"),
            PresetTag::new("c"),
        ];
        assert!(!tp.matches_all(&all_abc));
    }

    #[test]
    fn test_tagged_preset_matches_any() {
        let mut tp = TaggedPreset::new("p1");
        tp.add_tag(PresetTag::new("x"));
        let any = vec![PresetTag::new("y"), PresetTag::new("x")];
        assert!(tp.matches_any(&any));
        let none = vec![PresetTag::new("z")];
        assert!(!tp.matches_any(&none));
    }

    #[test]
    fn test_preset_tag_normalisation() {
        let t1 = PresetTag::new("  Hello  ");
        let t2 = PresetTag::new("hello");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_preset_tag_display() {
        let t = PresetTag::new("HLS");
        assert_eq!(t.to_string(), "hls");
    }

    #[test]
    fn test_by_all_tags_empty_returns_all() {
        let idx = make_index();
        let result = idx.by_all_tags(&[]);
        assert_eq!(result.len(), 3);
    }
}
