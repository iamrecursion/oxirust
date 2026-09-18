//! Asset collection management
//!
//! Provides simple in-memory asset collection management for MAM workflows:
//! - Collection hierarchy (projects, series, seasons, reels, playlists)
//! - Tree-based organization with parent-child relationships
//! - Asset membership tracking

#![allow(dead_code)]

/// The type of a collection, determining its role in the asset hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionType {
    /// Top-level production project
    Project,
    /// A series of related episodes or content
    Series,
    /// A single season within a series
    Season,
    /// A physical or logical film reel
    Reel,
    /// A curated playlist of assets
    PlayList,
    /// A user-defined custom grouping
    Custom,
}

impl CollectionType {
    /// Returns `true` if this collection type supports child collections.
    #[must_use]
    pub fn is_hierarchical(&self) -> bool {
        matches!(
            self,
            CollectionType::Project | CollectionType::Series | CollectionType::Season
        )
    }
}

/// An asset collection node in the collection tree.
#[derive(Debug, Clone)]
pub struct Collection {
    /// Unique identifier for this collection
    pub id: u64,
    /// Human-readable display name
    pub name: String,
    /// The classification of this collection
    pub collection_type: CollectionType,
    /// Parent collection ID, or `None` if this is a root collection
    pub parent_id: Option<u64>,
    /// Ordered list of asset IDs belonging to this collection
    pub asset_ids: Vec<u64>,
}

impl Collection {
    /// Returns `true` if this collection has no parent (is a root node).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Returns the number of assets directly in this collection.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.asset_ids.len()
    }

    /// Returns `true` if the given `asset_id` is a member of this collection.
    #[must_use]
    pub fn contains(&self, asset_id: u64) -> bool {
        self.asset_ids.contains(&asset_id)
    }
}

/// An in-memory tree of collections supporting hierarchical organization.
#[derive(Debug, Default)]
pub struct CollectionTree {
    /// All collections stored in the tree
    pub collections: Vec<Collection>,
    /// Counter used to assign unique IDs
    pub next_id: u64,
}

impl CollectionTree {
    /// Create a new, empty collection tree.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new root-level collection and return its ID.
    pub fn create_root(&mut self, name: impl Into<String>, ct: CollectionType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.collections.push(Collection {
            id,
            name: name.into(),
            collection_type: ct,
            parent_id: None,
            asset_ids: Vec::new(),
        });
        id
    }

    /// Create a child collection under `parent_id`.
    ///
    /// Returns `Some(id)` on success, or `None` if the parent does not exist.
    pub fn create_child(
        &mut self,
        parent_id: u64,
        name: impl Into<String>,
        ct: CollectionType,
    ) -> Option<u64> {
        // Verify parent exists
        if self.collections.iter().all(|c| c.id != parent_id) {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.collections.push(Collection {
            id,
            name: name.into(),
            collection_type: ct,
            parent_id: Some(parent_id),
            asset_ids: Vec::new(),
        });
        Some(id)
    }

    /// Return references to all direct children of the given collection ID.
    #[must_use]
    pub fn children_of(&self, id: u64) -> Vec<&Collection> {
        self.collections
            .iter()
            .filter(|c| c.parent_id == Some(id))
            .collect()
    }

    /// Find a collection by its ID.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&Collection> {
        self.collections.iter().find(|c| c.id == id)
    }

    /// Add an asset to the specified collection.
    ///
    /// Does nothing if the collection does not exist or already contains the asset.
    pub fn add_asset(&mut self, collection_id: u64, asset_id: u64) {
        if let Some(col) = self.collections.iter_mut().find(|c| c.id == collection_id) {
            if !col.asset_ids.contains(&asset_id) {
                col.asset_ids.push(asset_id);
            }
        }
    }

    /// Returns the total number of collections in the tree.
    #[must_use]
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree() -> CollectionTree {
        CollectionTree::new()
    }

    #[test]
    fn test_collection_type_is_hierarchical_project() {
        assert!(CollectionType::Project.is_hierarchical());
    }

    #[test]
    fn test_collection_type_is_hierarchical_series() {
        assert!(CollectionType::Series.is_hierarchical());
    }

    #[test]
    fn test_collection_type_is_hierarchical_season() {
        assert!(CollectionType::Season.is_hierarchical());
    }

    #[test]
    fn test_collection_type_not_hierarchical_reel() {
        assert!(!CollectionType::Reel.is_hierarchical());
    }

    #[test]
    fn test_collection_type_not_hierarchical_playlist() {
        assert!(!CollectionType::PlayList.is_hierarchical());
    }

    #[test]
    fn test_collection_type_not_hierarchical_custom() {
        assert!(!CollectionType::Custom.is_hierarchical());
    }

    #[test]
    fn test_create_root_returns_id() {
        let mut tree = make_tree();
        let id = tree.create_root("My Project", CollectionType::Project);
        assert_eq!(id, 0);
        assert_eq!(tree.collection_count(), 1);
    }

    #[test]
    fn test_root_collection_is_root() {
        let mut tree = make_tree();
        let id = tree.create_root("Root", CollectionType::Project);
        let col = tree.find_by_id(id).expect("should succeed in test");
        assert!(col.is_root());
    }

    #[test]
    fn test_create_child_success() {
        let mut tree = make_tree();
        let root = tree.create_root("Project", CollectionType::Project);
        let child = tree.create_child(root, "Season 1", CollectionType::Season);
        assert!(child.is_some());
        let child_id = child.expect("should succeed in test");
        let col = tree.find_by_id(child_id).expect("should succeed in test");
        assert_eq!(col.parent_id, Some(root));
        assert!(!col.is_root());
    }

    #[test]
    fn test_create_child_invalid_parent() {
        let mut tree = make_tree();
        let result = tree.create_child(999, "Orphan", CollectionType::Custom);
        assert!(result.is_none());
    }

    #[test]
    fn test_children_of() {
        let mut tree = make_tree();
        let root = tree.create_root("Project", CollectionType::Project);
        let c1 = tree
            .create_child(root, "Season 1", CollectionType::Season)
            .expect("should succeed in test");
        let c2 = tree
            .create_child(root, "Season 2", CollectionType::Season)
            .expect("should succeed in test");
        let children = tree.children_of(root);
        let ids: Vec<u64> = children.iter().map(|c| c.id).collect();
        assert!(ids.contains(&c1));
        assert!(ids.contains(&c2));
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_add_asset_and_contains() {
        let mut tree = make_tree();
        let root = tree.create_root("Project", CollectionType::Project);
        tree.add_asset(root, 42);
        let col = tree.find_by_id(root).expect("should succeed in test");
        assert!(col.contains(42));
        assert_eq!(col.asset_count(), 1);
    }

    #[test]
    fn test_add_asset_no_duplicates() {
        let mut tree = make_tree();
        let root = tree.create_root("Project", CollectionType::Project);
        tree.add_asset(root, 42);
        tree.add_asset(root, 42);
        let col = tree.find_by_id(root).expect("should succeed in test");
        assert_eq!(col.asset_count(), 1);
    }

    #[test]
    fn test_add_asset_missing_collection() {
        let mut tree = make_tree();
        // Should not panic
        tree.add_asset(999, 1);
    }

    #[test]
    fn test_find_by_id_missing() {
        let tree = make_tree();
        assert!(tree.find_by_id(0).is_none());
    }

    #[test]
    fn test_collection_count_multiple() {
        let mut tree = make_tree();
        tree.create_root("A", CollectionType::Project);
        tree.create_root("B", CollectionType::Series);
        tree.create_root("C", CollectionType::Custom);
        assert_eq!(tree.collection_count(), 3);
    }

    #[test]
    fn test_children_of_leaf_empty() {
        let mut tree = make_tree();
        let id = tree.create_root("Reel", CollectionType::Reel);
        assert!(tree.children_of(id).is_empty());
    }
}
