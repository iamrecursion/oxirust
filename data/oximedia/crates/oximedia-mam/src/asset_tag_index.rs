//! Tag-based index for MAM assets.
//!
//! Provides fast lookups from tag strings to sets of asset IDs,
//! tag frequency counts, and co-occurrence analysis useful for
//! recommendation and auto-tagging features.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// A single tag-to-assets mapping entry.
#[derive(Debug, Clone)]
pub struct TagEntry {
    /// The normalised tag string (lowercase, trimmed).
    pub tag: String,
    /// IDs of all assets that carry this tag.
    pub asset_ids: HashSet<String>,
}

impl TagEntry {
    /// Creates a new empty tag entry.
    pub fn new(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            asset_ids: HashSet::new(),
        }
    }

    /// Returns the number of assets with this tag.
    pub fn frequency(&self) -> usize {
        self.asset_ids.len()
    }

    /// Returns `true` when `asset_id` is tagged with this tag.
    pub fn contains(&self, asset_id: &str) -> bool {
        self.asset_ids.contains(asset_id)
    }
}

/// Normalises a raw tag string (lowercase, trim whitespace).
fn normalise(tag: &str) -> String {
    tag.trim().to_lowercase()
}

/// An inverted index from tags to asset IDs.
#[derive(Debug, Default)]
pub struct AssetTagIndex {
    /// Map from normalised tag → entry.
    index: HashMap<String, TagEntry>,
    /// Map from asset_id → set of tags on that asset.
    asset_tags: HashMap<String, HashSet<String>>,
}

impl AssetTagIndex {
    /// Creates an empty tag index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Associates `tag` with `asset_id`.
    ///
    /// The tag is normalised before insertion.
    pub fn add_tag(&mut self, asset_id: impl Into<String>, tag: &str) {
        let norm = normalise(tag);
        if norm.is_empty() {
            return;
        }
        let asset_id: String = asset_id.into();
        self.index
            .entry(norm.clone())
            .or_insert_with(|| TagEntry::new(norm.clone()))
            .asset_ids
            .insert(asset_id.clone());

        self.asset_tags.entry(asset_id).or_default().insert(norm);
    }

    /// Removes the association between `tag` and `asset_id`.
    ///
    /// Returns `true` if the association existed.
    pub fn remove_tag(&mut self, asset_id: &str, tag: &str) -> bool {
        let norm = normalise(tag);
        let removed = if let Some(entry) = self.index.get_mut(&norm) {
            entry.asset_ids.remove(asset_id)
        } else {
            false
        };
        if let Some(tags) = self.asset_tags.get_mut(asset_id) {
            tags.remove(&norm);
        }
        removed
    }

    /// Removes all tag associations for `asset_id`.
    pub fn remove_asset(&mut self, asset_id: &str) {
        if let Some(tags) = self.asset_tags.remove(asset_id) {
            for tag in &tags {
                if let Some(entry) = self.index.get_mut(tag) {
                    entry.asset_ids.remove(asset_id);
                }
            }
        }
    }

    /// Returns all asset IDs that have the given tag.
    pub fn assets_for_tag(&self, tag: &str) -> Vec<String> {
        let norm = normalise(tag);
        self.index
            .get(&norm)
            .map(|e| e.asset_ids.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns all tags associated with an asset.
    pub fn tags_for_asset(&self, asset_id: &str) -> Vec<String> {
        self.asset_tags
            .get(asset_id)
            .map(|set| {
                let mut v: Vec<String> = set.iter().cloned().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Returns all asset IDs that have **all** of the given tags.
    pub fn assets_with_all_tags(&self, tags: &[&str]) -> Vec<String> {
        if tags.is_empty() {
            return Vec::new();
        }
        let sets: Vec<HashSet<String>> = tags
            .iter()
            .filter_map(|t| {
                let norm = normalise(t);
                self.index.get(&norm).map(|e| e.asset_ids.clone())
            })
            .collect();

        if sets.len() != tags.len() {
            // At least one tag had no assets; intersection is empty.
            return Vec::new();
        }

        let mut iter = sets.into_iter();
        let first = iter
            .next()
            .expect("invariant: sets non-empty (len == tags.len() checked above)");
        let intersection: HashSet<String> =
            iter.fold(first, |acc, s| acc.intersection(&s).cloned().collect());

        let mut result: Vec<String> = intersection.into_iter().collect();
        result.sort();
        result
    }

    /// Returns all asset IDs that have **any** of the given tags (union).
    pub fn assets_with_any_tag(&self, tags: &[&str]) -> Vec<String> {
        let mut union: HashSet<String> = HashSet::new();
        for tag in tags {
            let norm = normalise(tag);
            if let Some(entry) = self.index.get(&norm) {
                union.extend(entry.asset_ids.iter().cloned());
            }
        }
        let mut result: Vec<String> = union.into_iter().collect();
        result.sort();
        result
    }

    /// Returns all tags ordered by descending frequency.
    pub fn most_frequent_tags(&self, limit: usize) -> Vec<(&str, usize)> {
        let mut pairs: Vec<(&str, usize)> = self
            .index
            .values()
            .map(|e| (e.tag.as_str(), e.frequency()))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        pairs.truncate(limit);
        pairs
    }

    /// Returns tags that co-occur with `tag` on the same assets,
    /// sorted by co-occurrence frequency (descending).
    pub fn co_occurring_tags(&self, tag: &str, limit: usize) -> Vec<(String, usize)> {
        let norm = normalise(tag);
        let asset_ids = match self.index.get(&norm) {
            Some(e) => &e.asset_ids,
            None => return Vec::new(),
        };

        let mut counts: HashMap<String, usize> = HashMap::new();
        for asset_id in asset_ids {
            if let Some(tags) = self.asset_tags.get(asset_id) {
                for t in tags {
                    if t != &norm {
                        *counts.entry(t.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        result.truncate(limit);
        result
    }

    /// Returns the total number of distinct tags in the index.
    pub fn tag_count(&self) -> usize {
        self.index.len()
    }

    /// Returns the total number of distinct assets tracked.
    pub fn asset_count(&self) -> usize {
        self.asset_tags.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_index() -> AssetTagIndex {
        let mut idx = AssetTagIndex::new();
        idx.add_tag("asset-1", "news");
        idx.add_tag("asset-1", "uk");
        idx.add_tag("asset-2", "news");
        idx.add_tag("asset-2", "sports");
        idx.add_tag("asset-3", "sports");
        idx.add_tag("asset-3", "football");
        idx
    }

    #[test]
    fn test_add_tag_normalisation() {
        let mut idx = AssetTagIndex::new();
        idx.add_tag("a1", "  News  ");
        let assets = idx.assets_for_tag("news");
        assert!(assets.contains(&"a1".to_string()));
    }

    #[test]
    fn test_assets_for_tag_found() {
        let idx = populated_index();
        let assets = idx.assets_for_tag("news");
        assert_eq!(assets.len(), 2);
        assert!(assets.contains(&"asset-1".to_string()));
        assert!(assets.contains(&"asset-2".to_string()));
    }

    #[test]
    fn test_assets_for_tag_not_found() {
        let idx = populated_index();
        assert!(idx.assets_for_tag("cooking").is_empty());
    }

    #[test]
    fn test_tags_for_asset_sorted() {
        let idx = populated_index();
        let tags = idx.tags_for_asset("asset-1");
        assert_eq!(tags, vec!["news", "uk"]);
    }

    #[test]
    fn test_tags_for_asset_missing() {
        let idx = populated_index();
        assert!(idx.tags_for_asset("ghost").is_empty());
    }

    #[test]
    fn test_assets_with_all_tags_intersection() {
        let idx = populated_index();
        let assets = idx.assets_with_all_tags(&["news", "sports"]);
        assert_eq!(assets, vec!["asset-2".to_string()]);
    }

    #[test]
    fn test_assets_with_all_tags_no_match() {
        let idx = populated_index();
        let assets = idx.assets_with_all_tags(&["news", "football"]);
        assert!(assets.is_empty());
    }

    #[test]
    fn test_assets_with_all_tags_empty_input() {
        let idx = populated_index();
        assert!(idx.assets_with_all_tags(&[]).is_empty());
    }

    #[test]
    fn test_assets_with_any_tag_union() {
        let idx = populated_index();
        let assets = idx.assets_with_any_tag(&["uk", "football"]);
        assert_eq!(assets.len(), 2);
        assert!(assets.contains(&"asset-1".to_string()));
        assert!(assets.contains(&"asset-3".to_string()));
    }

    #[test]
    fn test_remove_tag() {
        let mut idx = populated_index();
        let removed = idx.remove_tag("asset-1", "news");
        assert!(removed);
        let assets = idx.assets_for_tag("news");
        assert!(!assets.contains(&"asset-1".to_string()));
    }

    #[test]
    fn test_remove_tag_not_present_returns_false() {
        let mut idx = populated_index();
        assert!(!idx.remove_tag("asset-1", "cooking"));
    }

    #[test]
    fn test_remove_asset_clears_all_tags() {
        let mut idx = populated_index();
        idx.remove_asset("asset-1");
        assert!(idx.assets_for_tag("news").iter().all(|id| id != "asset-1"));
        assert!(idx.assets_for_tag("uk").is_empty());
        assert_eq!(idx.asset_count(), 2);
    }

    #[test]
    fn test_most_frequent_tags() {
        let idx = populated_index();
        let freq = idx.most_frequent_tags(3);
        // "news" and "sports" both appear in 2 assets; others in 1
        assert!(freq[0].1 >= freq[1].1);
    }

    #[test]
    fn test_co_occurring_tags() {
        let idx = populated_index();
        let co = idx.co_occurring_tags("news", 5);
        // "uk" co-occurs with news on asset-1; "sports" co-occurs on asset-2
        let co_tags: Vec<&str> = co.iter().map(|(t, _)| t.as_str()).collect();
        assert!(co_tags.contains(&"sports") || co_tags.contains(&"uk"));
    }

    #[test]
    fn test_co_occurring_tags_unknown_tag() {
        let idx = populated_index();
        assert!(idx.co_occurring_tags("ghost", 5).is_empty());
    }

    #[test]
    fn test_tag_count() {
        let idx = populated_index();
        // news, uk, sports, football
        assert_eq!(idx.tag_count(), 4);
    }

    #[test]
    fn test_asset_count() {
        let idx = populated_index();
        assert_eq!(idx.asset_count(), 3);
    }
}
