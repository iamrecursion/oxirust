#![allow(dead_code)]
//! Tagging and labelling system for review items.
//!
//! Provides a flexible tagging mechanism that allows reviewers to classify
//! and filter review comments, sessions, and versions using colour-coded
//! tags with hierarchical namespacing.

use std::collections::{HashMap, HashSet};

/// A colour represented as RGBA components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagColor {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0-255).
    pub a: u8,
}

impl TagColor {
    /// Create a new opaque colour.
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a colour with alpha transparency.
    #[must_use]
    pub fn with_alpha(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to a hex string like `#RRGGBB`.
    #[must_use]
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Return a predefined red colour.
    #[must_use]
    pub fn red() -> Self {
        Self::new(220, 53, 69)
    }

    /// Return a predefined green colour.
    #[must_use]
    pub fn green() -> Self {
        Self::new(40, 167, 69)
    }

    /// Return a predefined blue colour.
    #[must_use]
    pub fn blue() -> Self {
        Self::new(0, 123, 255)
    }

    /// Return a predefined yellow colour.
    #[must_use]
    pub fn yellow() -> Self {
        Self::new(255, 193, 7)
    }

    /// Return a predefined grey colour.
    #[must_use]
    pub fn grey() -> Self {
        Self::new(108, 117, 125)
    }

    /// Compute the luminance of this colour (0.0..=1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn luminance(&self) -> f64 {
        let r = f64::from(self.r) / 255.0;
        let g = f64::from(self.g) / 255.0;
        let b = f64::from(self.b) / 255.0;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Determine whether dark text should be used on this background.
    #[must_use]
    pub fn needs_dark_text(&self) -> bool {
        self.luminance() > 0.5
    }
}

impl Default for TagColor {
    fn default() -> Self {
        Self::grey()
    }
}

/// A single tag that can be applied to review items.
#[derive(Debug, Clone)]
pub struct ReviewTag {
    /// Tag identifier (unique within a registry).
    pub id: String,
    /// Display label.
    pub label: String,
    /// Colour associated with this tag.
    pub color: TagColor,
    /// Optional namespace prefix for hierarchical grouping (e.g. "audio", "video").
    pub namespace: Option<String>,
    /// Description of what this tag signifies.
    pub description: String,
}

impl ReviewTag {
    /// Create a new review tag.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>, color: TagColor) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            color,
            namespace: None,
            description: String::new(),
        }
    }

    /// Set the namespace.
    #[must_use]
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Return the fully-qualified name (namespace:id).
    #[must_use]
    pub fn qualified_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{ns}:{}", self.id),
            None => self.id.clone(),
        }
    }
}

impl PartialEq for ReviewTag {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.namespace == other.namespace
    }
}

impl Eq for ReviewTag {}

impl std::hash::Hash for ReviewTag {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.namespace.hash(state);
    }
}

/// A collection of tags applied to a single item.
#[derive(Debug, Clone, Default)]
pub struct TagSet {
    /// The tag IDs currently applied.
    tags: HashSet<String>,
}

impl TagSet {
    /// Create an empty tag set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: HashSet::new(),
        }
    }

    /// Add a tag by ID.
    pub fn add(&mut self, tag_id: impl Into<String>) -> bool {
        self.tags.insert(tag_id.into())
    }

    /// Remove a tag by ID.
    pub fn remove(&mut self, tag_id: &str) -> bool {
        self.tags.remove(tag_id)
    }

    /// Check whether a tag is present.
    #[must_use]
    pub fn contains(&self, tag_id: &str) -> bool {
        self.tags.contains(tag_id)
    }

    /// Return the number of tags.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Check whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Return an iterator over tag IDs.
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.tags.iter()
    }

    /// Compute the intersection with another tag set.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            tags: self.tags.intersection(&other.tags).cloned().collect(),
        }
    }

    /// Compute the union with another tag set.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            tags: self.tags.union(&other.tags).cloned().collect(),
        }
    }

    /// Clear all tags.
    pub fn clear(&mut self) {
        self.tags.clear();
    }
}

/// A registry that stores all known tags and manages lookups.
#[derive(Debug, Clone, Default)]
pub struct TagRegistry {
    /// Tags indexed by their ID.
    tags: HashMap<String, ReviewTag>,
}

impl TagRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
        }
    }

    /// Register a new tag. Returns `false` if a tag with the same ID already exists.
    pub fn register(&mut self, tag: ReviewTag) -> bool {
        if self.tags.contains_key(&tag.id) {
            return false;
        }
        self.tags.insert(tag.id.clone(), tag);
        true
    }

    /// Remove a tag by ID.
    pub fn unregister(&mut self, tag_id: &str) -> Option<ReviewTag> {
        self.tags.remove(tag_id)
    }

    /// Look up a tag by ID.
    #[must_use]
    pub fn get(&self, tag_id: &str) -> Option<&ReviewTag> {
        self.tags.get(tag_id)
    }

    /// Return the total number of registered tags.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Check whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Return all tags in a given namespace.
    #[must_use]
    pub fn tags_in_namespace(&self, namespace: &str) -> Vec<&ReviewTag> {
        self.tags
            .values()
            .filter(|t| t.namespace.as_deref() == Some(namespace))
            .collect()
    }

    /// Search tags whose label contains the given query (case-insensitive).
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&ReviewTag> {
        let lower = query.to_lowercase();
        self.tags
            .values()
            .filter(|t| t.label.to_lowercase().contains(&lower))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_color_hex() {
        let c = TagColor::new(255, 128, 0);
        assert_eq!(c.to_hex(), "#FF8000");
    }

    #[test]
    fn test_tag_color_predefined() {
        assert_eq!(TagColor::red().to_hex(), "#DC3545");
        assert_eq!(TagColor::green().to_hex(), "#28A745");
        assert_eq!(TagColor::blue().to_hex(), "#007BFF");
    }

    #[test]
    fn test_tag_color_luminance() {
        let white = TagColor::new(255, 255, 255);
        assert!(white.luminance() > 0.9);
        let black = TagColor::new(0, 0, 0);
        assert!(black.luminance() < 0.01);
    }

    #[test]
    fn test_tag_color_needs_dark_text() {
        assert!(TagColor::yellow().needs_dark_text());
        assert!(!TagColor::new(0, 0, 0).needs_dark_text());
    }

    #[test]
    fn test_review_tag_creation() {
        let tag = ReviewTag::new("bug", "Bug", TagColor::red())
            .with_namespace("issue")
            .with_description("Indicates a bug");
        assert_eq!(tag.qualified_name(), "issue:bug");
        assert_eq!(tag.description, "Indicates a bug");
    }

    #[test]
    fn test_review_tag_no_namespace() {
        let tag = ReviewTag::new("note", "Note", TagColor::blue());
        assert_eq!(tag.qualified_name(), "note");
    }

    #[test]
    fn test_tag_set_add_remove() {
        let mut ts = TagSet::new();
        assert!(ts.add("a"));
        assert!(ts.add("b"));
        assert!(!ts.add("a")); // duplicate
        assert_eq!(ts.len(), 2);
        assert!(ts.remove("a"));
        assert_eq!(ts.len(), 1);
        assert!(!ts.contains("a"));
    }

    #[test]
    fn test_tag_set_intersection() {
        let mut a = TagSet::new();
        a.add("x");
        a.add("y");
        let mut b = TagSet::new();
        b.add("y");
        b.add("z");
        let inter = a.intersection(&b);
        assert_eq!(inter.len(), 1);
        assert!(inter.contains("y"));
    }

    #[test]
    fn test_tag_set_union() {
        let mut a = TagSet::new();
        a.add("x");
        let mut b = TagSet::new();
        b.add("y");
        let u = a.union(&b);
        assert_eq!(u.len(), 2);
        assert!(u.contains("x"));
        assert!(u.contains("y"));
    }

    #[test]
    fn test_tag_registry_register_lookup() {
        let mut reg = TagRegistry::new();
        let tag = ReviewTag::new("bug", "Bug", TagColor::red());
        assert!(reg.register(tag));
        assert!(!reg.register(ReviewTag::new("bug", "Bug2", TagColor::blue()))); // dup
        assert_eq!(reg.len(), 1);
        let found = reg.get("bug").expect("should succeed in test");
        assert_eq!(found.label, "Bug");
    }

    #[test]
    fn test_tag_registry_unregister() {
        let mut reg = TagRegistry::new();
        reg.register(ReviewTag::new("tmp", "Temporary", TagColor::grey()));
        let removed = reg.unregister("tmp");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_tag_registry_namespace_filter() {
        let mut reg = TagRegistry::new();
        reg.register(ReviewTag::new("a", "A", TagColor::red()).with_namespace("audio"));
        reg.register(ReviewTag::new("b", "B", TagColor::blue()).with_namespace("video"));
        reg.register(ReviewTag::new("c", "C", TagColor::green()).with_namespace("audio"));
        let audio_tags = reg.tags_in_namespace("audio");
        assert_eq!(audio_tags.len(), 2);
    }

    #[test]
    fn test_tag_registry_search() {
        let mut reg = TagRegistry::new();
        reg.register(ReviewTag::new("1", "Color Issue", TagColor::red()));
        reg.register(ReviewTag::new("2", "Audio Problem", TagColor::blue()));
        reg.register(ReviewTag::new("3", "Color Grading", TagColor::green()));
        let results = reg.search("color");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tag_set_clear() {
        let mut ts = TagSet::new();
        ts.add("a");
        ts.add("b");
        ts.clear();
        assert!(ts.is_empty());
    }

    #[test]
    fn test_tag_color_alpha() {
        let c = TagColor::with_alpha(100, 200, 50, 128);
        assert_eq!(c.a, 128);
    }
}
