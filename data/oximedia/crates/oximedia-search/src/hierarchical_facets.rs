//! Hierarchical facet index for tree-structured media classification.
//!
//! Unlike flat facets (where every value is independent), hierarchical facets
//! form a tree: e.g. `"genre/rock/indie"` is a three-level path where `"genre"`
//! is the root level, `"rock"` is a child of `"genre"`, and `"indie"` is a
//! child of `"rock"`.
//!
//! Counts propagate **bottom-up**: adding `"genre/rock/indie"` increments the
//! count of `"genre"`, `"genre/rock"`, and `"genre/rock/indie"` by one. This
//! means the count at any intermediate node equals the total number of entries
//! beneath (and including) that node.
//!
//! # Example
//!
//! ```
//! use oximedia_search::hierarchical_facets::HierarchicalFacetIndex;
//!
//! let mut index = HierarchicalFacetIndex::new();
//! index.add_entry("genre/rock/indie");
//! index.add_entry("genre/rock/classic");
//! index.add_entry("genre/jazz");
//!
//! assert_eq!(index.count_facets("genre"), 3);
//! assert_eq!(index.count_facets("genre/rock"), 2);
//! assert_eq!(index.count_facets("genre/jazz"), 1);
//!
//! let children = index.drill_down("genre");
//! assert_eq!(children.len(), 2); // "rock" and "jazz"
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Internal node
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the hierarchical facet tree.
///
/// `count` represents the total number of `add_entry` calls that passed
/// through (or ended at) this node.
#[derive(Debug, Clone, Default)]
pub struct HierarchicalFacetNode {
    /// This node's label (the last path component).
    pub name: String,
    /// Number of entries that include this node in their path.
    pub count: usize,
    /// Direct children, keyed by their name.
    pub children: HashMap<String, HierarchicalFacetNode>,
}

impl HierarchicalFacetNode {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: 0,
            children: HashMap::new(),
        }
    }

    /// Recursively insert a path slice, incrementing counts at every level.
    fn insert(&mut self, components: &[&str]) {
        if components.is_empty() {
            return;
        }

        let head = components[0];
        let tail = &components[1..];

        let child = self
            .children
            .entry(head.to_string())
            .or_insert_with(|| HierarchicalFacetNode::new(head));

        child.count += 1;
        child.insert(tail);
    }

    /// Navigate to the node at `components` relative to this node, returning
    /// an immutable reference if found.
    fn navigate(&self, components: &[&str]) -> Option<&HierarchicalFacetNode> {
        if components.is_empty() {
            return Some(self);
        }
        let child = self.children.get(components[0])?;
        child.navigate(&components[1..])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// An index that organises facet entries as a hierarchical tree.
///
/// Paths are slash-separated strings such as `"format/video/mp4"`.  Empty
/// path components (consecutive slashes) are silently skipped.
#[derive(Debug, Default)]
pub struct HierarchicalFacetIndex {
    /// Virtual root node — not itself a facet, only a container.
    root: HierarchicalFacetNode,
    /// Total number of successful [`add_entry`] calls.
    total: usize,
}

impl HierarchicalFacetIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: HierarchicalFacetNode {
                name: String::new(),
                count: 0,
                children: HashMap::new(),
            },
            total: 0,
        }
    }

    /// Add an entry for the given slash-separated `path`.
    ///
    /// Empty path strings or paths consisting solely of slashes are ignored.
    /// Each non-empty component increments the count from the root downward.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_search::hierarchical_facets::HierarchicalFacetIndex;
    ///
    /// let mut idx = HierarchicalFacetIndex::new();
    /// idx.add_entry("codec/video/av1");
    /// assert_eq!(idx.count_facets("codec"), 1);
    /// assert_eq!(idx.count_facets("codec/video"), 1);
    /// assert_eq!(idx.count_facets("codec/video/av1"), 1);
    /// ```
    pub fn add_entry(&mut self, path: &str) {
        let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if components.is_empty() {
            return;
        }
        self.root.insert(&components);
        self.total += 1;
    }

    /// Return the count at the node identified by `path`.
    ///
    /// Returns `0` if the path does not exist.
    #[must_use]
    pub fn count_facets(&self, path: &str) -> usize {
        let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if components.is_empty() {
            return 0;
        }
        self.root.navigate(&components).map_or(0, |n| n.count)
    }

    /// Return the immediate children of the node at `path` as `(name, count)`
    /// pairs, sorted by count descending (ties broken alphabetically).
    ///
    /// Returns an empty `Vec` if `path` does not exist or has no children.
    #[must_use]
    pub fn drill_down(&self, path: &str) -> Vec<(&str, usize)> {
        let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let node = if components.is_empty() {
            &self.root
        } else {
            match self.root.navigate(&components) {
                Some(n) => n,
                None => return Vec::new(),
            }
        };

        let mut children: Vec<(&str, usize)> = node
            .children
            .iter()
            .map(|(name, child)| (name.as_str(), child.count))
            .collect();

        // Sort by count descending, then name ascending for determinism.
        children.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        children
    }

    /// Return the top-level facets (children of the virtual root) as
    /// `(name, count)` pairs, sorted by count descending.
    #[must_use]
    pub fn top_level(&self) -> Vec<(&str, usize)> {
        let mut top: Vec<(&str, usize)> = self
            .root
            .children
            .iter()
            .map(|(name, node)| (name.as_str(), node.count))
            .collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        top
    }

    /// Return the total number of `add_entry` calls that added at least one
    /// non-empty path component.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.total
    }

    /// Return `true` if the path node exists in the index.
    #[must_use]
    pub fn path_exists(&self, path: &str) -> bool {
        let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if components.is_empty() {
            return false;
        }
        self.root.navigate(&components).is_some()
    }

    /// Return all leaf paths (paths with no children) under `root_path`,
    /// together with their counts. If `root_path` is `""`, the search starts
    /// from the virtual root.
    ///
    /// Useful for enumerating the most specific facet values.
    #[must_use]
    pub fn leaf_paths(&self, root_path: &str) -> Vec<(String, usize)> {
        let start_node = if root_path.is_empty() {
            &self.root
        } else {
            let components: Vec<&str> = root_path.split('/').filter(|s| !s.is_empty()).collect();
            match self.root.navigate(&components) {
                Some(n) => n,
                None => return Vec::new(),
            }
        };

        let mut leaves = Vec::new();
        Self::collect_leaves(start_node, root_path, &mut leaves);
        leaves
    }

    fn collect_leaves(node: &HierarchicalFacetNode, prefix: &str, out: &mut Vec<(String, usize)>) {
        if node.children.is_empty() {
            if !prefix.is_empty() && !node.name.is_empty() {
                out.push((prefix.to_string(), node.count));
            }
            return;
        }
        for (child_name, child_node) in &node.children {
            let child_path = if prefix.is_empty() {
                child_name.clone()
            } else {
                format!("{}/{}", prefix, child_name)
            };
            Self::collect_leaves(child_node, &child_path, out);
        }
    }

    /// Return the depth of the deepest path in the index.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        Self::node_depth(&self.root)
    }

    fn node_depth(node: &HierarchicalFacetNode) -> usize {
        if node.children.is_empty() {
            return 0;
        }
        1 + node
            .children
            .values()
            .map(Self::node_depth)
            .max()
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────

    fn sorted_names(v: Vec<(&str, usize)>) -> Vec<String> {
        let mut names: Vec<String> = v.iter().map(|(n, _)| n.to_string()).collect();
        names.sort();
        names
    }

    // ── Basic operations ──────────────────────────────────────────────────

    #[test]
    fn test_empty_index() {
        let idx = HierarchicalFacetIndex::new();
        assert_eq!(idx.total_entries(), 0);
        assert!(!idx.path_exists("genre"));
        assert_eq!(idx.count_facets("genre"), 0);
        assert!(idx.top_level().is_empty());
        assert!(idx.drill_down("genre").is_empty());
    }

    #[test]
    fn test_single_entry_single_level() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("genre");
        assert_eq!(idx.total_entries(), 1);
        assert!(idx.path_exists("genre"));
        assert_eq!(idx.count_facets("genre"), 1);
    }

    #[test]
    fn test_single_entry_multi_level() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("genre/rock/indie");
        assert_eq!(idx.total_entries(), 1);

        // Every level in the path should be incremented
        assert_eq!(idx.count_facets("genre"), 1);
        assert_eq!(idx.count_facets("genre/rock"), 1);
        assert_eq!(idx.count_facets("genre/rock/indie"), 1);

        assert!(idx.path_exists("genre"));
        assert!(idx.path_exists("genre/rock"));
        assert!(idx.path_exists("genre/rock/indie"));
    }

    #[test]
    fn test_multiple_entries_same_path() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("genre/rock");
        idx.add_entry("genre/rock");
        idx.add_entry("genre/rock");
        assert_eq!(idx.total_entries(), 3);
        assert_eq!(idx.count_facets("genre"), 3);
        assert_eq!(idx.count_facets("genre/rock"), 3);
    }

    #[test]
    fn test_sibling_paths_aggregate_at_parent() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("genre/rock/indie");
        idx.add_entry("genre/rock/classic");
        idx.add_entry("genre/jazz");

        // Parent "genre" sees all 3 entries
        assert_eq!(idx.count_facets("genre"), 3);
        // "rock" sees its 2 children
        assert_eq!(idx.count_facets("genre/rock"), 2);
        // Leaf counts
        assert_eq!(idx.count_facets("genre/rock/indie"), 1);
        assert_eq!(idx.count_facets("genre/rock/classic"), 1);
        assert_eq!(idx.count_facets("genre/jazz"), 1);
    }

    #[test]
    fn test_drill_down_basic() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("genre/rock/indie");
        idx.add_entry("genre/rock/classic");
        idx.add_entry("genre/rock/classic");
        idx.add_entry("genre/jazz");

        let children = idx.drill_down("genre");
        assert_eq!(children.len(), 2);

        // "rock" has count 3, "jazz" has count 1 — "rock" should come first
        assert_eq!(children[0].0, "rock");
        assert_eq!(children[0].1, 3);
        assert_eq!(children[1].0, "jazz");
        assert_eq!(children[1].1, 1);
    }

    #[test]
    fn test_drill_down_nonexistent_path() {
        let idx = HierarchicalFacetIndex::new();
        let result = idx.drill_down("nonexistent");
        assert!(result.is_empty());
    }

    #[test]
    fn test_drill_down_leaf_node() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("a/b/c");
        // "c" is a leaf — drill_down should return empty
        let result = idx.drill_down("a/b/c");
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_level_sorted_by_count_desc() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("format/video");
        idx.add_entry("format/video");
        idx.add_entry("format/audio");
        idx.add_entry("codec/av1");

        let top = idx.top_level();
        // "format" has 3 entries (2 video + 1 audio), "codec" has 1
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "format");
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, "codec");
        assert_eq!(top[1].1, 1);
    }

    #[test]
    fn test_top_level_tie_broken_alphabetically() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("z/sub");
        idx.add_entry("a/sub");
        idx.add_entry("m/sub");

        let top = idx.top_level();
        // All have count 1 — should be sorted alphabetically ascending
        assert_eq!(top.len(), 3);
        let names: Vec<&str> = top.iter().map(|x| x.0).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_path_exists_partial() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("a/b/c/d");
        assert!(idx.path_exists("a"));
        assert!(idx.path_exists("a/b"));
        assert!(idx.path_exists("a/b/c"));
        assert!(idx.path_exists("a/b/c/d"));
        assert!(!idx.path_exists("a/b/c/d/e"));
        assert!(!idx.path_exists("x"));
    }

    #[test]
    fn test_count_nonexistent_path_returns_zero() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("a/b");
        assert_eq!(idx.count_facets("a/b/c"), 0);
        assert_eq!(idx.count_facets("z"), 0);
    }

    #[test]
    fn test_empty_path_ignored() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry(""); // empty path — should be ignored
        idx.add_entry("///"); // only slashes — should be ignored
        assert_eq!(idx.total_entries(), 0);
        assert!(idx.top_level().is_empty());
    }

    #[test]
    fn test_deep_hierarchy() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("l1/l2/l3/l4/l5/l6");
        assert_eq!(idx.count_facets("l1"), 1);
        assert_eq!(idx.count_facets("l1/l2"), 1);
        assert_eq!(idx.count_facets("l1/l2/l3/l4/l5/l6"), 1);
        assert!(!idx.path_exists("l1/l2/l3/l4/l5/l6/l7"));
        assert_eq!(idx.max_depth(), 6);
    }

    #[test]
    fn test_max_depth_flat() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("a");
        idx.add_entry("b");
        idx.add_entry("c");
        assert_eq!(idx.max_depth(), 1);
    }

    #[test]
    fn test_max_depth_empty() {
        let idx = HierarchicalFacetIndex::new();
        assert_eq!(idx.max_depth(), 0);
    }

    #[test]
    fn test_leaf_paths() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("codec/video/av1");
        idx.add_entry("codec/video/vp9");
        idx.add_entry("codec/audio/opus");

        let mut leaves = idx.leaf_paths("codec");
        leaves.sort_by(|a, b| a.0.cmp(&b.0));

        let paths: Vec<&str> = leaves.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"codec/audio/opus"));
        assert!(paths.contains(&"codec/video/av1"));
        assert!(paths.contains(&"codec/video/vp9"));
        assert_eq!(leaves.len(), 3);
    }

    #[test]
    fn test_leaf_paths_nonexistent() {
        let idx = HierarchicalFacetIndex::new();
        assert!(idx.leaf_paths("nonexistent").is_empty());
    }

    #[test]
    fn test_drill_down_multiple_levels() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("format/video/av1");
        idx.add_entry("format/video/av1");
        idx.add_entry("format/video/vp9");
        idx.add_entry("format/audio/opus");
        idx.add_entry("format/audio/flac");
        idx.add_entry("format/audio/flac");

        // Top-level: "format" with count 6
        let top = idx.top_level();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "format");
        assert_eq!(top[0].1, 6);

        // Drill into "format": "video" (3) and "audio" (3)
        let format_children = idx.drill_down("format");
        assert_eq!(format_children.len(), 2);
        // Both have count 3, so order is alphabetical: "audio" before "video"
        let child_names: Vec<&str> = format_children.iter().map(|x| x.0).collect();
        assert!(child_names.contains(&"video"));
        assert!(child_names.contains(&"audio"));
        assert_eq!(format_children[0].1, 3);
        assert_eq!(format_children[1].1, 3);

        // Drill into "format/video": "av1" (2) and "vp9" (1)
        let video_children = idx.drill_down("format/video");
        assert_eq!(video_children.len(), 2);
        assert_eq!(video_children[0].0, "av1");
        assert_eq!(video_children[0].1, 2);
        assert_eq!(video_children[1].0, "vp9");
        assert_eq!(video_children[1].1, 1);
    }

    #[test]
    fn test_total_entries_tracks_all_adds() {
        let mut idx = HierarchicalFacetIndex::new();
        for i in 0..100 {
            idx.add_entry(&format!("cat/{}", i % 5));
        }
        assert_eq!(idx.total_entries(), 100);
        // Each of the 5 sub-paths should have 20 entries
        for i in 0..5 {
            assert_eq!(idx.count_facets(&format!("cat/{}", i)), 20);
        }
        // Root "cat" should have 100
        assert_eq!(idx.count_facets("cat"), 100);
    }

    #[test]
    fn test_facet_node_children_independence() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("a/x");
        idx.add_entry("b/x");

        // Both "a" and "b" have a child "x" — they're independent
        assert_eq!(idx.count_facets("a/x"), 1);
        assert_eq!(idx.count_facets("b/x"), 1);
        assert_eq!(idx.count_facets("a"), 1);
        assert_eq!(idx.count_facets("b"), 1);

        let names = sorted_names(idx.top_level());
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_drill_down_empty_string_path_returns_top_level() {
        let mut idx = HierarchicalFacetIndex::new();
        idx.add_entry("x/y");
        idx.add_entry("z/w");

        // drill_down("") returns top-level children (same as top_level())
        let drill = idx.drill_down("");
        let top = idx.top_level();
        assert_eq!(drill.len(), top.len());
    }
}
