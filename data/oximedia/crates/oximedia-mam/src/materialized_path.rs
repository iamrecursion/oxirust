//! Materialized-path model for efficient folder hierarchy queries.
//!
//! The materialized-path pattern stores each node's full ancestry in a single
//! string column (e.g. `/root/projects/2024/interviews/`).  This allows the
//! following operations to execute in O(n) string comparisons rather than
//! requiring recursive CTE queries:
//!
//! - **Subtree listing**: all nodes whose path *starts with* an ancestor path.
//! - **Ancestor listing**: all prefixes of a node's path.
//! - **Depth calculation**: count path separators.
//! - **Re-rooting (move subtree)**: replace the path prefix for every
//!   descendant atomically.
//!
//! # Design
//!
//! - `MaterializedNode` — a folder/category node with id, name, and path.
//! - `PathTree` — in-memory store that indexes nodes by their materialized
//!   path and supports the operations described above.
//! - `PathQuery` — declarative query builder for common path-based lookups.
//! - `PathStats` — aggregate statistics over the tree.

use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A single node in the materialized-path tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedNode {
    /// Unique stable identifier (e.g. a UUID or numeric id).
    pub id: String,
    /// Display name (single segment, no slashes).
    pub name: String,
    /// Full materialized path including leading and trailing separator.
    /// Example: `/root/projects/2024/interviews/`
    pub path: String,
    /// IDs of assets attached to this folder node.
    pub asset_ids: Vec<String>,
}

impl MaterializedNode {
    /// Build a `MaterializedNode` from a parent path and a segment name.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `name` contains the path separator (`/`).
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        parent_path: &str,
    ) -> Result<Self, PathError> {
        let name = name.into();
        if name.contains('/') {
            return Err(PathError::InvalidSegment(name));
        }
        let path = build_path(parent_path, &name);
        Ok(Self {
            id: id.into(),
            name,
            path,
            asset_ids: Vec::new(),
        })
    }

    /// Create a root node (depth 0).
    #[must_use]
    pub fn root(id: impl Into<String>, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: id.into(),
            path: format!("/{name}/"),
            name,
            asset_ids: Vec::new(),
        }
    }

    /// Compute the depth of this node (0 = root).
    ///
    /// Depth is the number of non-empty segments in the path.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.path
            .split('/')
            .filter(|s| !s.is_empty())
            .count()
            .saturating_sub(1)
    }

    /// Return the parent path for this node.
    ///
    /// For the root node (`/name/`) this returns `/`.
    #[must_use]
    pub fn parent_path(&self) -> String {
        let segments: Vec<&str> = self.path.split('/').filter(|s| !s.is_empty()).collect();
        if segments.len() <= 1 {
            return "/".to_string();
        }
        let parent_segments = &segments[..segments.len() - 1];
        format!("/{}/", parent_segments.join("/"))
    }

    /// Returns `true` if `potential_ancestor` is an ancestor of this node.
    #[must_use]
    pub fn is_descendant_of(&self, potential_ancestor: &MaterializedNode) -> bool {
        self.path.starts_with(&potential_ancestor.path) && self.path != potential_ancestor.path
    }

    /// Attach an asset id to this node.
    pub fn add_asset(&mut self, asset_id: impl Into<String>) {
        let id = asset_id.into();
        if !self.asset_ids.contains(&id) {
            self.asset_ids.push(id);
        }
    }

    /// Remove an asset id from this node.
    ///
    /// Returns `true` if the id was present and removed.
    pub fn remove_asset(&mut self, asset_id: &str) -> bool {
        if let Some(pos) = self.asset_ids.iter().position(|a| a == asset_id) {
            self.asset_ids.remove(pos);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Path utilities
// ---------------------------------------------------------------------------

/// Build a materialized path by appending `segment` to `parent_path`.
#[must_use]
fn build_path(parent_path: &str, segment: &str) -> String {
    let base = parent_path.trim_end_matches('/');
    format!("{base}/{segment}/")
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors arising from path-tree operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    /// A segment contains the `/` separator.
    #[error("invalid path segment (contains '/'): {0}")]
    InvalidSegment(String),

    /// A node with the given id already exists.
    #[error("duplicate node id: {0}")]
    DuplicateId(String),

    /// No node was found for the given id.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// The target parent does not exist.
    #[error("parent not found at path: {0}")]
    ParentNotFound(String),
}

// ---------------------------------------------------------------------------
// PathTree
// ---------------------------------------------------------------------------

/// In-memory materialized-path tree indexed by node id.
///
/// In addition to the primary `nodes` map (id → node), a secondary
/// `path_index` (BTreeMap: path → id) enables O(log n + k) range queries
/// for `descendants` and `children`, replacing the previous O(n) linear scans.
#[derive(Debug, Default)]
pub struct PathTree {
    /// All nodes, keyed by their id.
    nodes: HashMap<String, MaterializedNode>,
    /// Secondary index: materialized path → node id.
    ///
    /// All paths end with `/`.  Because [`BTreeMap`] stores keys in
    /// lexicographic order, a prefix scan (`range(prefix..)`) followed by
    /// `take_while(starts_with)` produces all descendants in O(log n + k).
    path_index: BTreeMap<String, String>,
}

impl PathTree {
    /// Create a new, empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Insert a pre-built node.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::DuplicateId`] if a node with the same id already
    /// exists.
    pub fn insert(&mut self, node: MaterializedNode) -> Result<(), PathError> {
        if self.nodes.contains_key(&node.id) {
            return Err(PathError::DuplicateId(node.id.clone()));
        }
        self.path_index.insert(node.path.clone(), node.id.clone());
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Insert a root node (depth 0).
    ///
    /// Convenience wrapper around [`insert`](Self::insert).
    ///
    /// # Errors
    ///
    /// Returns an error if a node with the same id already exists.
    pub fn insert_root(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<(), PathError> {
        let node = MaterializedNode::root(id, name);
        self.insert(node)
    }

    /// Insert a child node under the node identified by `parent_id`.
    ///
    /// # Errors
    ///
    /// Returns an error if `parent_id` does not exist, if `name` contains a
    /// `/`, or if a node with the given `child_id` already exists.
    pub fn insert_child(
        &mut self,
        parent_id: &str,
        child_id: impl Into<String>,
        child_name: impl Into<String>,
    ) -> Result<(), PathError> {
        let parent_path = self
            .nodes
            .get(parent_id)
            .map(|n| n.path.clone())
            .ok_or_else(|| PathError::NodeNotFound(parent_id.to_string()))?;
        let node = MaterializedNode::new(child_id, child_name, &parent_path)?;
        self.insert(node)
    }

    /// Remove a node and all its descendants from the tree.
    ///
    /// Returns the number of nodes removed.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NodeNotFound`] if the id does not exist.
    pub fn remove_subtree(&mut self, id: &str) -> Result<usize, PathError> {
        let root_path = self
            .nodes
            .get(id)
            .map(|n| n.path.clone())
            .ok_or_else(|| PathError::NodeNotFound(id.to_string()))?;

        let to_remove: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| {
                self.nodes
                    .get(*k)
                    .map(|n| n.path.starts_with(&root_path))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            if let Some(removed) = self.nodes.remove(key) {
                self.path_index.remove(&removed.path);
            }
        }
        Ok(count)
    }

    /// Move node `id` (and its entire subtree) so that it becomes a child of
    /// the node at `new_parent_id`.
    ///
    /// All descendant paths are updated atomically.
    ///
    /// # Errors
    ///
    /// Returns an error if either id is not found.
    pub fn move_node(&mut self, id: &str, new_parent_id: &str) -> Result<(), PathError> {
        let old_path = self
            .nodes
            .get(id)
            .map(|n| n.path.clone())
            .ok_or_else(|| PathError::NodeNotFound(id.to_string()))?;

        let new_parent_path = self
            .nodes
            .get(new_parent_id)
            .map(|n| n.path.clone())
            .ok_or_else(|| PathError::NodeNotFound(new_parent_id.to_string()))?;

        let node_name = self
            .nodes
            .get(id)
            .map(|n| n.name.clone())
            .ok_or_else(|| PathError::NodeNotFound(id.to_string()))?;

        let new_path = build_path(&new_parent_path, &node_name);

        // Collect all descendants (including the node itself)
        let affected: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| {
                self.nodes
                    .get(*k)
                    .map(|n| n.path.starts_with(&old_path))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        for key in &affected {
            if let Some(node) = self.nodes.get_mut(key) {
                // Remove the old path from the index before mutating
                self.path_index.remove(&node.path);
                node.path = node.path.replacen(&old_path, &new_path, 1);
                self.path_index.insert(node.path.clone(), node.id.clone());
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Look up a node by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&MaterializedNode> {
        self.nodes.get(id)
    }

    /// Return all direct children of the node with the given id.
    ///
    /// Uses the BTreeMap path index for an O(log n + k) range scan rather than
    /// an O(n) linear pass over all nodes.
    #[must_use]
    pub fn children(&self, parent_id: &str) -> Vec<&MaterializedNode> {
        let parent_path = match self.nodes.get(parent_id) {
            Some(n) => n.path.clone(),
            None => return Vec::new(),
        };
        self.path_index
            .range(parent_path.clone()..)
            .take_while(|(k, _)| k.starts_with(parent_path.as_str()))
            .filter(|(k, _)| {
                // Exclude the parent itself
                if k.as_str() == parent_path.as_str() {
                    return false;
                }
                // Keep only direct children: the suffix after parent_path must
                // contain exactly one non-empty segment (no additional '/').
                let suffix = &k[parent_path.len()..];
                !suffix.is_empty() && !suffix.trim_end_matches('/').contains('/')
            })
            .filter_map(|(_, nid)| self.nodes.get(nid))
            .collect()
    }

    /// Return all descendants of the node with `id` (excluding `id` itself).
    ///
    /// Uses the BTreeMap path index for an O(log n + k) range scan rather than
    /// an O(n) linear pass over all nodes.
    #[must_use]
    pub fn descendants(&self, id: &str) -> Vec<&MaterializedNode> {
        let path = match self.nodes.get(id) {
            Some(n) => n.path.clone(),
            None => return Vec::new(),
        };
        self.path_index
            .range(path.clone()..)
            .take_while(|(k, _)| k.starts_with(path.as_str()))
            // Exclude the node itself
            .filter(|(k, _)| k.as_str() != path.as_str())
            .filter_map(|(_, nid)| self.nodes.get(nid))
            .collect()
    }

    /// Return all ancestors of the node with `id` (root first, parent last),
    /// excluding the node itself.
    ///
    /// Uses the BTreeMap path index for O(d · log n) point lookups (where d is
    /// depth) rather than an O(n) linear scan over all nodes.
    #[must_use]
    pub fn ancestors(&self, id: &str) -> Vec<&MaterializedNode> {
        let path = match self.nodes.get(id) {
            Some(n) => n.path.clone(),
            None => return Vec::new(),
        };

        // Enumerate ancestor paths from shallowest to deepest (root first).
        // For path `/a/b/c/` the ancestors are `/a/` and `/a/b/`.
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut result = Vec::with_capacity(segments.len().saturating_sub(1));
        for len in 1..segments.len() {
            let ancestor_path = format!("/{}/", segments[..len].join("/"));
            if let Some(nid) = self.path_index.get(&ancestor_path) {
                if let Some(node) = self.nodes.get(nid) {
                    result.push(node);
                }
            }
        }
        // result is already root-first because we iterate from len=1 upwards
        result
    }

    /// Return all nodes at exactly `depth` levels below the root (depth 0).
    #[must_use]
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<&MaterializedNode> {
        self.nodes.values().filter(|n| n.depth() == depth).collect()
    }

    /// Return the total number of nodes in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tree contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return aggregate statistics over the tree.
    #[must_use]
    pub fn stats(&self) -> PathStats {
        let max_depth = self.nodes.values().map(|n| n.depth()).max().unwrap_or(0);
        let total_assets: usize = self.nodes.values().map(|n| n.asset_ids.len()).sum();
        PathStats {
            node_count: self.nodes.len(),
            max_depth,
            total_assets,
        }
    }
}

// ---------------------------------------------------------------------------
// PathQuery
// ---------------------------------------------------------------------------

/// Declarative query builder for path-based lookups.
///
/// Constructed via the fluent builder pattern and executed against a
/// [`PathTree`] with [`PathQuery::execute`].
#[derive(Debug, Default, Clone)]
pub struct PathQuery {
    /// Only include nodes whose path starts with this prefix.
    prefix_filter: Option<String>,
    /// Only include nodes at exactly this depth.
    exact_depth: Option<usize>,
    /// Only include nodes whose name contains this substring (case-insensitive).
    name_contains: Option<String>,
    /// Maximum number of results to return.
    limit: Option<usize>,
}

impl PathQuery {
    /// Create a new, empty query (matches all nodes).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter to nodes whose path starts with `prefix`.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix_filter = Some(prefix.into());
        self
    }

    /// Filter to nodes at exactly `depth`.
    #[must_use]
    pub fn at_depth(mut self, depth: usize) -> Self {
        self.exact_depth = Some(depth);
        self
    }

    /// Filter to nodes whose name contains `substring` (case-insensitive).
    #[must_use]
    pub fn name_contains(mut self, substring: impl Into<String>) -> Self {
        self.name_contains = Some(substring.into().to_lowercase());
        self
    }

    /// Limit the result set to `n` entries.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Execute the query against `tree`, returning matching node references
    /// sorted by path (lexicographic).
    #[must_use]
    pub fn execute<'t>(&self, tree: &'t PathTree) -> Vec<&'t MaterializedNode> {
        let mut results: Vec<&MaterializedNode> = tree
            .nodes
            .values()
            .filter(|n| {
                if let Some(prefix) = &self.prefix_filter {
                    if !n.path.starts_with(prefix.as_str()) {
                        return false;
                    }
                }
                if let Some(depth) = self.exact_depth {
                    if n.depth() != depth {
                        return false;
                    }
                }
                if let Some(sub) = &self.name_contains {
                    if !n.name.to_lowercase().contains(sub.as_str()) {
                        return false;
                    }
                }
                true
            })
            .collect();

        results.sort_by(|a, b| a.path.cmp(&b.path));

        if let Some(lim) = self.limit {
            results.truncate(lim);
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics over a [`PathTree`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathStats {
    /// Total number of nodes in the tree.
    pub node_count: usize,
    /// Depth of the deepest node.
    pub max_depth: usize,
    /// Total number of asset references across all nodes.
    pub total_assets: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree() -> PathTree {
        let mut tree = PathTree::new();
        tree.insert_root("root", "root").expect("ok");
        tree.insert_child("root", "projects", "Projects")
            .expect("ok");
        tree.insert_child("projects", "proj-2024", "2024")
            .expect("ok");
        tree.insert_child("proj-2024", "interviews", "Interviews")
            .expect("ok");
        tree.insert_child("proj-2024", "broll", "BRoll")
            .expect("ok");
        tree.insert_child("root", "archive", "Archive").expect("ok");
        tree.insert_child("archive", "arch-2023", "2023")
            .expect("ok");
        tree
    }

    #[test]
    fn test_insert_and_lookup() {
        let tree = build_tree();
        let node = tree.get("interviews");
        assert!(node.is_some());
        assert_eq!(node.expect("should be Some").name, "Interviews");
    }

    #[test]
    fn test_node_depth_root() {
        let node = MaterializedNode::root("r", "root");
        assert_eq!(node.depth(), 0);
    }

    #[test]
    fn test_node_depth_nested() {
        let tree = build_tree();
        let node = tree.get("interviews").expect("exists");
        // root(0) → Projects(1) → 2024(2) → Interviews(3)
        assert_eq!(node.depth(), 3);
    }

    #[test]
    fn test_children_direct_only() {
        let tree = build_tree();
        let children = tree.children("proj-2024");
        assert_eq!(children.len(), 2);
        let names: Vec<&str> = children.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"Interviews"));
        assert!(names.contains(&"BRoll"));
    }

    #[test]
    fn test_descendants_excludes_self() {
        let tree = build_tree();
        let descs = tree.descendants("projects");
        // 2024, Interviews, BRoll
        assert_eq!(descs.len(), 3);
        assert!(!descs.iter().any(|n| n.id == "projects"));
    }

    #[test]
    fn test_ancestors_root_first() {
        let tree = build_tree();
        let ancestors = tree.ancestors("interviews");
        // root, projects, proj-2024
        assert_eq!(ancestors.len(), 3);
        assert_eq!(ancestors[0].id, "root");
        assert_eq!(ancestors.last().expect("exists").id, "proj-2024");
    }

    #[test]
    fn test_is_descendant_of() {
        let tree = build_tree();
        let interviews = tree.get("interviews").expect("exists");
        let root = tree.get("root").expect("exists");
        assert!(interviews.is_descendant_of(root));
        assert!(!root.is_descendant_of(interviews));
    }

    #[test]
    fn test_remove_subtree() {
        let mut tree = build_tree();
        let removed = tree.remove_subtree("proj-2024").expect("ok");
        // Removes proj-2024, Interviews, BRoll
        assert_eq!(removed, 3);
        assert!(tree.get("interviews").is_none());
        assert!(tree.get("proj-2024").is_none());
    }

    #[test]
    fn test_move_node_updates_paths() {
        let mut tree = build_tree();
        // Move archive/2023 under projects
        tree.move_node("arch-2023", "projects").expect("ok");
        let moved = tree.get("arch-2023").expect("exists");
        assert!(moved.path.starts_with("/root/Projects/"));
    }

    #[test]
    fn test_duplicate_id_error() {
        let mut tree = PathTree::new();
        tree.insert_root("r", "root").expect("ok");
        let err = tree.insert_root("r", "root2");
        assert!(matches!(err, Err(PathError::DuplicateId(_))));
    }

    #[test]
    fn test_invalid_segment_with_slash() {
        let err = MaterializedNode::new("id", "foo/bar", "/root/");
        assert!(matches!(err, Err(PathError::InvalidSegment(_))));
    }

    #[test]
    fn test_nodes_at_depth() {
        let tree = build_tree();
        let depth_1 = tree.nodes_at_depth(1);
        // Projects, Archive
        assert_eq!(depth_1.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut tree = build_tree();
        if let Some(n) = tree.nodes.get_mut("interviews") {
            n.add_asset("asset-1");
        }
        let stats = tree.stats();
        assert_eq!(stats.node_count, 7);
        assert_eq!(stats.max_depth, 3);
        assert_eq!(stats.total_assets, 1);
    }

    #[test]
    fn test_path_query_at_depth() {
        let tree = build_tree();
        let results = PathQuery::new().at_depth(2).execute(&tree);
        // proj-2024, arch-2023
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_path_query_name_contains() {
        let tree = build_tree();
        // "archive" node has name "Archive" and "arch-2023" has name "2023"
        let results = PathQuery::new().name_contains("arch").execute(&tree);
        assert!(results.iter().any(|n| n.id == "archive"));
        // "2023" node name does not contain "arch", only the "archive" node does
        assert_eq!(results.len(), 1);
        // Search for "2023" should match the arch-2023 node by its display name
        let results2 = PathQuery::new().name_contains("2023").execute(&tree);
        assert!(results2.iter().any(|n| n.id == "arch-2023"));
    }

    #[test]
    fn test_path_query_limit() {
        let tree = build_tree();
        let results = PathQuery::new().limit(2).execute(&tree);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_add_remove_asset_on_node() {
        let mut node = MaterializedNode::root("r", "root");
        node.add_asset("a1");
        node.add_asset("a1"); // duplicate, should not be added again
        assert_eq!(node.asset_ids.len(), 1);
        assert!(node.remove_asset("a1"));
        assert!(node.asset_ids.is_empty());
        assert!(!node.remove_asset("a1")); // not present
    }

    #[test]
    fn test_parent_path_of_root() {
        let node = MaterializedNode::root("r", "root");
        assert_eq!(node.parent_path(), "/");
    }

    #[test]
    fn test_parent_path_of_child() {
        let node = MaterializedNode::new("id", "child", "/root/").expect("ok");
        assert_eq!(node.parent_path(), "/root/");
    }

    // -----------------------------------------------------------------------
    // BTreeMap path-prefix index tests (Wave 22, Slice 2)
    // -----------------------------------------------------------------------

    /// Build a PathTree with `count` leaf nodes arranged as a single root with
    /// `count` direct children.
    fn build_wide_tree(count: usize) -> PathTree {
        let mut tree = PathTree::new();
        tree.insert_root("root", "root").expect("ok");
        for i in 0..count {
            let child_id = format!("child-{i}");
            let child_name = format!("child{i}");
            tree.insert_child("root", &child_id, &child_name)
                .expect("ok");
        }
        tree
    }

    /// 1. Build 10k-node tree, call `descendants` on root, verify count.
    #[test]
    fn test_path_index_descendants_10k() {
        const N: usize = 10_000;
        let tree = build_wide_tree(N);
        let descs = tree.descendants("root");
        assert_eq!(
            descs.len(),
            N,
            "descendants count should match inserted nodes"
        );
        // Spot-check: first and last child should be present
        assert!(descs.iter().any(|n| n.id == "child-0"));
        assert!(descs.iter().any(|n| n.id == format!("child-{}", N - 1)));
        // The root itself must not appear
        assert!(!descs.iter().any(|n| n.id == "root"));
    }

    /// 2. Deep chain `/a/b/c/d/e/` — ancestors of "e" must be [a, b, c, d].
    #[test]
    fn test_path_index_ancestors_chain() {
        let mut tree = PathTree::new();
        tree.insert_root("a", "a").expect("ok");
        tree.insert_child("a", "b", "b").expect("ok");
        tree.insert_child("b", "c", "c").expect("ok");
        tree.insert_child("c", "d", "d").expect("ok");
        tree.insert_child("d", "e", "e").expect("ok");

        let ancestors = tree.ancestors("e");
        // Expect root-first order: a, b, c, d
        assert_eq!(ancestors.len(), 4);
        assert_eq!(ancestors[0].id, "a");
        assert_eq!(ancestors[1].id, "b");
        assert_eq!(ancestors[2].id, "c");
        assert_eq!(ancestors[3].id, "d");
    }

    /// 3. `children` must return only direct children, not grandchildren.
    #[test]
    fn test_path_index_children_single_level_only() {
        let mut tree = PathTree::new();
        tree.insert_root("root", "root").expect("ok");
        tree.insert_child("root", "child-a", "childA").expect("ok");
        tree.insert_child("root", "child-b", "childB").expect("ok");
        // Grandchildren — must NOT appear in children("root")
        tree.insert_child("child-a", "grandchild-1", "gc1")
            .expect("ok");
        tree.insert_child("child-a", "grandchild-2", "gc2")
            .expect("ok");

        let children = tree.children("root");
        assert_eq!(children.len(), 2, "only direct children expected");
        let ids: Vec<&str> = children.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"child-a"));
        assert!(ids.contains(&"child-b"));
        // Grandchildren must not appear
        assert!(!ids.contains(&"grandchild-1"));
        assert!(!ids.contains(&"grandchild-2"));
    }

    /// 4. After `move_node`, `descendants` and `children` reflect updated paths.
    #[test]
    fn test_path_index_move_node_updates_index() {
        let mut tree = PathTree::new();
        tree.insert_root("root", "root").expect("ok");
        tree.insert_child("root", "alpha", "alpha").expect("ok");
        tree.insert_child("root", "beta", "beta").expect("ok");
        // alpha has one child
        tree.insert_child("alpha", "child-of-alpha", "sub")
            .expect("ok");

        // Move "alpha" (with its subtree) under "beta"
        tree.move_node("alpha", "beta").expect("ok");

        // "root" should now only have "beta" as direct child
        let root_children = tree.children("root");
        assert_eq!(root_children.len(), 1);
        assert_eq!(root_children[0].id, "beta");

        // "beta" should have "alpha" as direct child
        let beta_children = tree.children("beta");
        assert_eq!(beta_children.len(), 1);
        assert_eq!(beta_children[0].id, "alpha");

        // descendants of "root" = beta + alpha + child-of-alpha
        let root_descs = tree.descendants("root");
        assert_eq!(root_descs.len(), 3);

        // The moved node's path must reflect the new location
        let alpha = tree.get("alpha").expect("exists");
        assert!(
            alpha.path.starts_with("/root/beta/"),
            "alpha should be under /root/beta/ but is at {}",
            alpha.path
        );

        // child-of-alpha also moved
        let sub = tree.get("child-of-alpha").expect("exists");
        assert!(
            sub.path.starts_with("/root/beta/alpha/"),
            "child-of-alpha should be under /root/beta/alpha/ but is at {}",
            sub.path
        );
    }

    /// 5. After `remove_subtree`, removed nodes no longer appear in descendants.
    #[test]
    fn test_path_index_remove_node_clears_index() {
        let mut tree = PathTree::new();
        tree.insert_root("root", "root").expect("ok");
        tree.insert_child("root", "keep", "keep").expect("ok");
        tree.insert_child("root", "prune", "prune").expect("ok");
        tree.insert_child("prune", "sub-prune", "subprune")
            .expect("ok");

        let removed = tree.remove_subtree("prune").expect("ok");
        assert_eq!(removed, 2, "prune + sub-prune should be removed");

        // Descendants of root must not include the pruned subtree
        let descs = tree.descendants("root");
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].id, "keep");

        // Verify path_index is clean: looking up descendants of the removed
        // node returns empty (not found).
        assert!(tree.get("prune").is_none());
        assert!(tree.get("sub-prune").is_none());
        let descs_of_pruned = tree.descendants("prune");
        assert!(descs_of_pruned.is_empty());
    }
}
