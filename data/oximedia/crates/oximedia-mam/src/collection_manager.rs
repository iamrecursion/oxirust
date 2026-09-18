#![allow(dead_code)]
//! Collection management for MAM: typed collections, asset membership, and
//! a top-level [`CollectionManager`] registry.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CollectionType
// ---------------------------------------------------------------------------

/// Classifies the purpose and behaviour of a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectionType {
    /// A general-purpose folder-like container.
    Folder,
    /// A saved, dynamically evaluated search.
    SmartCollection,
    /// A sequential list (playlist, shot-list, …).
    Sequence,
    /// A thematic grouping with no nesting (flat).
    Album,
    /// A project-level container that may nest sub-collections.
    Project,
}

impl CollectionType {
    /// Returns `true` if this collection type may contain nested
    /// sub-collections.
    #[must_use]
    pub fn allows_nested(self) -> bool {
        matches!(self, Self::Folder | Self::Project)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Folder => "folder",
            Self::SmartCollection => "smart_collection",
            Self::Sequence => "sequence",
            Self::Album => "album",
            Self::Project => "project",
        }
    }
}

// ---------------------------------------------------------------------------
// AssetCollection
// ---------------------------------------------------------------------------

/// A named collection of asset IDs.
#[derive(Debug, Clone)]
pub struct AssetCollection {
    /// Unique identifier for this collection.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Type / purpose of the collection.
    pub collection_type: CollectionType,
    /// Ordered list of asset IDs.
    asset_ids: Vec<u64>,
}

impl AssetCollection {
    /// Create a new, empty collection.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, collection_type: CollectionType) -> Self {
        Self {
            id,
            name: name.into(),
            collection_type,
            asset_ids: Vec::new(),
        }
    }

    /// Add an asset ID to the collection.  Duplicate IDs are allowed (e.g.
    /// for sequenced playlists); call [`Self::contains`] first if uniqueness is
    /// required.
    pub fn add_asset(&mut self, asset_id: u64) {
        self.asset_ids.push(asset_id);
    }

    /// Remove the first occurrence of `asset_id`.  Returns `true` if an entry
    /// was removed, `false` if the ID was not present.
    pub fn remove_asset(&mut self, asset_id: u64) -> bool {
        if let Some(pos) = self.asset_ids.iter().position(|&id| id == asset_id) {
            self.asset_ids.remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns `true` if at least one occurrence of `asset_id` exists.
    #[must_use]
    pub fn contains(&self, asset_id: u64) -> bool {
        self.asset_ids.contains(&asset_id)
    }

    /// Total number of asset entries (may include duplicates for sequences).
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.asset_ids.len()
    }

    /// Iterate over asset IDs in order.
    pub fn asset_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.asset_ids.iter().copied()
    }
}

// ---------------------------------------------------------------------------
// CollectionManager
// ---------------------------------------------------------------------------

/// Central registry that creates, stores, and retrieves [`AssetCollection`]s.
#[derive(Debug, Default)]
pub struct CollectionManager {
    collections: HashMap<u64, AssetCollection>,
    next_id: u64,
}

impl CollectionManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new collection and register it.  Returns the assigned ID.
    pub fn create(&mut self, name: impl Into<String>, kind: CollectionType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let collection = AssetCollection::new(id, name, kind);
        self.collections.insert(id, collection);
        id
    }

    /// Find a collection by ID.  Returns `None` if not found.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&AssetCollection> {
        self.collections.get(&id)
    }

    /// Find a collection by ID (mutable borrow).
    pub fn find_mut(&mut self, id: u64) -> Option<&mut AssetCollection> {
        self.collections.get_mut(&id)
    }

    /// Delete a collection by ID.  Returns `true` if it existed.
    pub fn delete(&mut self, id: u64) -> bool {
        self.collections.remove(&id).is_some()
    }

    /// Total number of registered collections.
    #[must_use]
    pub fn count(&self) -> usize {
        self.collections.len()
    }

    /// Iterate over all collections.
    pub fn iter(&self) -> impl Iterator<Item = &AssetCollection> {
        self.collections.values()
    }

    /// Find all collections whose name contains `substr` (case-insensitive).
    #[must_use]
    pub fn find_by_name_substr(&self, substr: &str) -> Vec<&AssetCollection> {
        let lower = substr.to_lowercase();
        self.collections
            .values()
            .filter(|c| c.name.to_lowercase().contains(&lower))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_type_allows_nested_folder() {
        assert!(CollectionType::Folder.allows_nested());
    }

    #[test]
    fn test_collection_type_allows_nested_project() {
        assert!(CollectionType::Project.allows_nested());
    }

    #[test]
    fn test_collection_type_no_nested_album() {
        assert!(!CollectionType::Album.allows_nested());
    }

    #[test]
    fn test_collection_type_no_nested_smart() {
        assert!(!CollectionType::SmartCollection.allows_nested());
    }

    #[test]
    fn test_collection_type_no_nested_sequence() {
        assert!(!CollectionType::Sequence.allows_nested());
    }

    #[test]
    fn test_add_asset_and_contains() {
        let mut c = AssetCollection::new(1, "Test", CollectionType::Folder);
        assert!(!c.contains(42));
        c.add_asset(42);
        assert!(c.contains(42));
    }

    #[test]
    fn test_asset_count() {
        let mut c = AssetCollection::new(1, "Test", CollectionType::Folder);
        assert_eq!(c.asset_count(), 0);
        c.add_asset(1);
        c.add_asset(2);
        assert_eq!(c.asset_count(), 2);
    }

    #[test]
    fn test_remove_asset() {
        let mut c = AssetCollection::new(1, "Test", CollectionType::Folder);
        c.add_asset(10);
        assert!(c.remove_asset(10));
        assert!(!c.contains(10));
    }

    #[test]
    fn test_remove_absent_asset_returns_false() {
        let mut c = AssetCollection::new(1, "Test", CollectionType::Folder);
        assert!(!c.remove_asset(999));
    }

    #[test]
    fn test_manager_create_and_find() {
        let mut mgr = CollectionManager::new();
        let id = mgr.create("Rushes", CollectionType::Folder);
        let c = mgr.find(id).expect("should succeed in test");
        assert_eq!(c.name, "Rushes");
    }

    #[test]
    fn test_manager_count() {
        let mut mgr = CollectionManager::new();
        assert_eq!(mgr.count(), 0);
        mgr.create("A", CollectionType::Album);
        mgr.create("B", CollectionType::Sequence);
        assert_eq!(mgr.count(), 2);
    }

    #[test]
    fn test_manager_delete() {
        let mut mgr = CollectionManager::new();
        let id = mgr.create("X", CollectionType::Folder);
        assert!(mgr.delete(id));
        assert!(mgr.find(id).is_none());
        assert!(!mgr.delete(id));
    }

    #[test]
    fn test_manager_find_by_name_substr() {
        let mut mgr = CollectionManager::new();
        mgr.create("Rushes 2024", CollectionType::Folder);
        mgr.create("Interview Clips", CollectionType::Album);
        mgr.create("rushes backup", CollectionType::Folder);
        let results = mgr.find_by_name_substr("rushes");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_manager_find_mut_add_asset() {
        let mut mgr = CollectionManager::new();
        let id = mgr.create("MyCollection", CollectionType::Folder);
        if let Some(c) = mgr.find_mut(id) {
            c.add_asset(77);
        }
        assert!(mgr.find(id).expect("should succeed in test").contains(77));
    }

    #[test]
    fn test_collection_type_labels() {
        assert_eq!(CollectionType::Folder.label(), "folder");
        assert_eq!(CollectionType::SmartCollection.label(), "smart_collection");
        assert_eq!(CollectionType::Sequence.label(), "sequence");
        assert_eq!(CollectionType::Album.label(), "album");
        assert_eq!(CollectionType::Project.label(), "project");
    }
}
