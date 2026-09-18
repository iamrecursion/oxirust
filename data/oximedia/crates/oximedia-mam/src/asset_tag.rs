//! Asset tagging with scoped tags and an inverted index for fast lookups.
//!
//! Supports global and user-scoped tags, tag collections per asset,
//! and an `AssetTagIndex` for tag-to-asset and co-occurrence queries.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// The scope / visibility level of a tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TagScope {
    /// Tag is visible to all users in the system.
    Global,
    /// Tag is private to a specific user (identified by user ID string).
    User(String),
    /// Tag belongs to a named project workspace.
    Project(String),
}

impl TagScope {
    /// Returns `true` if this is a `Global` tag scope.
    pub fn is_global(&self) -> bool {
        matches!(self, TagScope::Global)
    }

    /// Returns a displayable scope label.
    pub fn label(&self) -> String {
        match self {
            TagScope::Global => "global".to_string(),
            TagScope::User(uid) => format!("user:{uid}"),
            TagScope::Project(proj) => format!("project:{proj}"),
        }
    }
}

/// A single tag that can be attached to assets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetTag {
    /// Tag name (must be non-empty, no leading/trailing whitespace).
    pub name: String,
    /// Scope of the tag.
    pub scope: TagScope,
}

impl AssetTag {
    /// Create a new tag.
    pub fn new(name: impl Into<String>, scope: TagScope) -> Self {
        Self {
            name: name.into(),
            scope,
        }
    }

    /// Create a global tag.
    pub fn global(name: impl Into<String>) -> Self {
        Self::new(name, TagScope::Global)
    }

    /// Returns `true` if the tag name is non-empty and contains no leading/trailing whitespace.
    pub fn is_valid(&self) -> bool {
        let trimmed = self.name.trim();
        !trimmed.is_empty() && trimmed == self.name
    }
}

/// Ordered collection of tags associated with a single asset.
#[derive(Debug, Default, Clone)]
pub struct TagCollection {
    tags: Vec<AssetTag>,
}

impl TagCollection {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tag if it is valid and not already present.
    /// Returns `true` if the tag was newly inserted.
    pub fn add(&mut self, tag: AssetTag) -> bool {
        if !tag.is_valid() {
            return false;
        }
        if self.tags.contains(&tag) {
            return false;
        }
        self.tags.push(tag);
        true
    }

    /// Remove a tag by value. Returns `true` if the tag was present.
    pub fn remove(&mut self, tag: &AssetTag) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns `true` if the collection contains the given tag.
    pub fn has(&self, tag: &AssetTag) -> bool {
        self.tags.contains(tag)
    }

    /// Number of tags in the collection.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Returns `true` if the collection has no tags.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Iterate over all tags.
    pub fn iter(&self) -> impl Iterator<Item = &AssetTag> {
        self.tags.iter()
    }
}

/// Inverted index mapping tag names to sets of asset IDs.
///
/// Asset IDs are represented as `u64` for simplicity.
#[derive(Debug, Default)]
pub struct AssetTagIndex {
    /// tag_name -> set of asset IDs
    index: HashMap<String, HashSet<u64>>,
    /// asset_id -> list of tags
    asset_tags: HashMap<u64, Vec<AssetTag>>,
}

impl AssetTagIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `asset_id` has been tagged with `tag`.
    /// Only valid tags (per `AssetTag::is_valid`) are indexed.
    pub fn tag_asset(&mut self, asset_id: u64, tag: AssetTag) {
        if !tag.is_valid() {
            return;
        }
        self.index
            .entry(tag.name.clone())
            .or_default()
            .insert(asset_id);
        self.asset_tags.entry(asset_id).or_default().push(tag);
    }

    /// Tag an asset with multiple tags at once.
    pub fn tag_assets(&mut self, asset_id: u64, tags: impl IntoIterator<Item = AssetTag>) {
        for tag in tags {
            self.tag_asset(asset_id, tag);
        }
    }

    /// Return all asset IDs that have `tag_name`.
    pub fn find_by_tag(&self, tag_name: &str) -> HashSet<u64> {
        self.index.get(tag_name).cloned().unwrap_or_default()
    }

    /// Return all tags attached to `asset_id`.
    pub fn tags_for_asset(&self, asset_id: u64) -> Vec<&AssetTag> {
        self.asset_tags
            .get(&asset_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Total number of unique tag names in the index.
    pub fn unique_tag_count(&self) -> usize {
        self.index.len()
    }

    /// Total number of indexed assets.
    pub fn asset_count(&self) -> usize {
        self.asset_tags.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_scope_is_global_true() {
        assert!(TagScope::Global.is_global());
    }

    #[test]
    fn test_tag_scope_is_global_false_user() {
        assert!(!TagScope::User("alice".to_string()).is_global());
    }

    #[test]
    fn test_tag_scope_is_global_false_project() {
        assert!(!TagScope::Project("proj1".to_string()).is_global());
    }

    #[test]
    fn test_tag_scope_label_global() {
        assert_eq!(TagScope::Global.label(), "global");
    }

    #[test]
    fn test_tag_scope_label_user() {
        assert_eq!(TagScope::User("bob".to_string()).label(), "user:bob");
    }

    #[test]
    fn test_tag_scope_label_project() {
        assert_eq!(
            TagScope::Project("alpha".to_string()).label(),
            "project:alpha"
        );
    }

    #[test]
    fn test_asset_tag_is_valid_ok() {
        let t = AssetTag::global("documentary");
        assert!(t.is_valid());
    }

    #[test]
    fn test_asset_tag_is_valid_empty() {
        let t = AssetTag::global("");
        assert!(!t.is_valid());
    }

    #[test]
    fn test_asset_tag_is_valid_leading_whitespace() {
        let t = AssetTag::global(" leadingspace");
        assert!(!t.is_valid());
    }

    #[test]
    fn test_tag_collection_add_and_has() {
        let mut col = TagCollection::new();
        let tag = AssetTag::global("sports");
        col.add(tag.clone());
        assert!(col.has(&tag));
    }

    #[test]
    fn test_tag_collection_add_duplicate() {
        let mut col = TagCollection::new();
        let tag = AssetTag::global("sports");
        assert!(col.add(tag.clone()));
        assert!(!col.add(tag));
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_tag_collection_add_invalid() {
        let mut col = TagCollection::new();
        let tag = AssetTag::global("  ");
        assert!(!col.add(tag));
        assert!(col.is_empty());
    }

    #[test]
    fn test_tag_collection_remove_existing() {
        let mut col = TagCollection::new();
        let tag = AssetTag::global("news");
        col.add(tag.clone());
        assert!(col.remove(&tag));
        assert!(col.is_empty());
    }

    #[test]
    fn test_tag_collection_remove_missing() {
        let mut col = TagCollection::new();
        let tag = AssetTag::global("ghost");
        assert!(!col.remove(&tag));
    }

    #[test]
    fn test_asset_tag_index_tag_asset_and_find() {
        let mut idx = AssetTagIndex::new();
        idx.tag_asset(1, AssetTag::global("nature"));
        idx.tag_asset(2, AssetTag::global("nature"));
        let assets = idx.find_by_tag("nature");
        assert!(assets.contains(&1));
        assert!(assets.contains(&2));
    }

    #[test]
    fn test_asset_tag_index_find_missing_tag() {
        let idx = AssetTagIndex::new();
        assert!(idx.find_by_tag("absent").is_empty());
    }

    #[test]
    fn test_asset_tag_index_tag_assets_bulk() {
        let mut idx = AssetTagIndex::new();
        idx.tag_assets(10, vec![AssetTag::global("a"), AssetTag::global("b")]);
        assert_eq!(idx.tags_for_asset(10).len(), 2);
    }

    #[test]
    fn test_asset_tag_index_unique_tag_count() {
        let mut idx = AssetTagIndex::new();
        idx.tag_asset(1, AssetTag::global("x"));
        idx.tag_asset(2, AssetTag::global("x"));
        idx.tag_asset(1, AssetTag::global("y"));
        assert_eq!(idx.unique_tag_count(), 2);
    }

    #[test]
    fn test_asset_tag_index_asset_count() {
        let mut idx = AssetTagIndex::new();
        idx.tag_asset(5, AssetTag::global("tag1"));
        idx.tag_asset(6, AssetTag::global("tag2"));
        assert_eq!(idx.asset_count(), 2);
    }

    #[test]
    fn test_asset_tag_index_invalid_tag_not_indexed() {
        let mut idx = AssetTagIndex::new();
        idx.tag_asset(1, AssetTag::global(""));
        assert_eq!(idx.unique_tag_count(), 0);
        assert_eq!(idx.asset_count(), 0);
    }
}
