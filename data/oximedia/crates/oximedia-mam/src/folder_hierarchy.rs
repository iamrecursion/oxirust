//! Hierarchical folder tree for organising media assets.
//!
//! Provides `FolderNode` and `FolderTree` with path-based insertion
//! and lookup so assets can be organised in virtual folder structures.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single node in the folder hierarchy.
#[derive(Debug, Clone)]
pub struct FolderNode {
    /// Folder name (single path segment).
    pub name: String,
    /// IDs of assets directly contained in this folder.
    pub asset_ids: Vec<u64>,
    /// Child folders keyed by their name.
    pub children: HashMap<String, FolderNode>,
}

impl FolderNode {
    /// Create a new, empty folder node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            asset_ids: Vec::new(),
            children: HashMap::new(),
        }
    }

    /// Return the total number of assets in this node and all descendants.
    #[must_use]
    pub fn total_assets(&self) -> usize {
        let child_total: usize = self.children.values().map(FolderNode::total_assets).sum();
        self.asset_ids.len() + child_total
    }

    /// Return the depth of this subtree (0 = leaf).
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self
                .children
                .values()
                .map(FolderNode::depth)
                .max()
                .unwrap_or(0)
        }
    }
}

/// A virtual folder tree rooted at a single [`FolderNode`].
#[derive(Debug)]
pub struct FolderTree {
    root: FolderNode,
}

impl FolderTree {
    /// Create a new tree with the given root name.
    #[must_use]
    pub fn new(root_name: impl Into<String>) -> Self {
        Self {
            root: FolderNode::new(root_name),
        }
    }

    /// Insert a path (e.g. `"Projects/2024/Interviews"`) and, optionally,
    /// associate `asset_id` with the deepest folder.
    ///
    /// Missing intermediate folders are created automatically.
    pub fn insert_path(&mut self, path: &str, asset_id: Option<u64>) {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut current = &mut self.root;
        for segment in segments {
            current = current
                .children
                .entry(segment.to_string())
                .or_insert_with(|| FolderNode::new(segment));
        }

        if let Some(id) = asset_id {
            if !current.asset_ids.contains(&id) {
                current.asset_ids.push(id);
            }
        }
    }

    /// Find the node at `path`, returning `None` if any segment is missing.
    #[must_use]
    pub fn find(&self, path: &str) -> Option<&FolderNode> {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut current = &self.root;
        for segment in segments {
            current = current.children.get(segment)?;
        }
        Some(current)
    }

    /// Find the node at `path` mutably.
    #[must_use]
    pub fn find_mut(&mut self, path: &str) -> Option<&mut FolderNode> {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut current = &mut self.root;
        for segment in segments {
            current = current.children.get_mut(segment)?;
        }
        Some(current)
    }

    /// Return a reference to the root node.
    #[must_use]
    pub fn root(&self) -> &FolderNode {
        &self.root
    }

    /// Return the total number of assets in the entire tree.
    #[must_use]
    pub fn total_assets(&self) -> usize {
        self.root.total_assets()
    }

    /// Collect all paths present in the tree (depth-first).
    #[must_use]
    pub fn all_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        collect_paths(&self.root, String::new(), &mut paths);
        paths
    }

    /// Remove an asset id from the folder at `path`.
    ///
    /// Returns `true` if the id was found and removed.
    pub fn remove_asset(&mut self, path: &str, asset_id: u64) -> bool {
        if let Some(node) = self.find_mut(path) {
            if let Some(pos) = node.asset_ids.iter().position(|&id| id == asset_id) {
                node.asset_ids.remove(pos);
                return true;
            }
        }
        false
    }
}

/// Recursively collect folder paths into `out`.
fn collect_paths(node: &FolderNode, prefix: String, out: &mut Vec<String>) {
    let current_path = if prefix.is_empty() {
        node.name.clone()
    } else {
        format!("{}/{}", prefix, node.name)
    };
    out.push(current_path.clone());
    for child in node.children.values() {
        collect_paths(child, current_path.clone(), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> FolderTree {
        let mut tree = FolderTree::new("root");
        tree.insert_path("Projects/2024/Interviews", Some(1));
        tree.insert_path("Projects/2024/Interviews", Some(2));
        tree.insert_path("Projects/2024/BRoll", Some(3));
        tree.insert_path("Archive/2023", Some(4));
        tree
    }

    #[test]
    fn test_insert_and_find() {
        let tree = sample_tree();
        let node = tree.find("Projects/2024/Interviews");
        assert!(node.is_some());
        assert_eq!(node.expect("should succeed in test").asset_ids.len(), 2);
    }

    #[test]
    fn test_find_nonexistent_path() {
        let tree = sample_tree();
        assert!(tree.find("Does/Not/Exist").is_none());
    }

    #[test]
    fn test_total_assets_root() {
        let tree = sample_tree();
        assert_eq!(tree.total_assets(), 4);
    }

    #[test]
    fn test_total_assets_subtree() {
        let tree = sample_tree();
        let node = tree.find("Projects/2024").expect("should succeed in test");
        assert_eq!(node.total_assets(), 3);
    }

    #[test]
    fn test_node_depth_leaf() {
        let node = FolderNode::new("leaf");
        assert_eq!(node.depth(), 0);
    }

    #[test]
    fn test_node_depth_nested() {
        let tree = sample_tree();
        // root → Projects → 2024 → Interviews / BRoll  depth = 3
        assert_eq!(tree.root().depth(), 3);
    }

    #[test]
    fn test_insert_duplicate_asset_id_not_duplicated() {
        let mut tree = FolderTree::new("root");
        tree.insert_path("Folder", Some(99));
        tree.insert_path("Folder", Some(99));
        let node = tree.find("Folder").expect("should succeed in test");
        assert_eq!(node.asset_ids.len(), 1);
    }

    #[test]
    fn test_insert_path_no_asset() {
        let mut tree = FolderTree::new("root");
        tree.insert_path("Empty/Folder", None);
        let node = tree.find("Empty/Folder");
        assert!(node.is_some());
        assert!(node.expect("should succeed in test").asset_ids.is_empty());
    }

    #[test]
    fn test_all_paths_contains_expected() {
        let tree = sample_tree();
        let paths = tree.all_paths();
        // Should contain at least one path per inserted segment
        assert!(paths.iter().any(|p| p.contains("Interviews")));
        assert!(paths.iter().any(|p| p.contains("Archive")));
    }

    #[test]
    fn test_remove_asset_success() {
        let mut tree = sample_tree();
        let removed = tree.remove_asset("Projects/2024/Interviews", 1);
        assert!(removed);
        assert_eq!(
            tree.find("Projects/2024/Interviews")
                .expect("should succeed in test")
                .asset_ids
                .len(),
            1
        );
    }

    #[test]
    fn test_remove_asset_not_present() {
        let mut tree = sample_tree();
        let removed = tree.remove_asset("Projects/2024/BRoll", 999);
        assert!(!removed);
    }

    #[test]
    fn test_remove_asset_wrong_path() {
        let mut tree = sample_tree();
        let removed = tree.remove_asset("No/Such/Path", 1);
        assert!(!removed);
    }

    #[test]
    fn test_folder_node_new() {
        let node = FolderNode::new("testfolder");
        assert_eq!(node.name, "testfolder");
        assert!(node.asset_ids.is_empty());
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_tree_root_name() {
        let tree = FolderTree::new("library");
        assert_eq!(tree.root().name, "library");
    }
}
