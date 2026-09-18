//! Hierarchical folder organization.

use crate::clip::ClipId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Unique identifier for a folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FolderId(Uuid);

impl FolderId {
    /// Creates a new random folder ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a folder ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for FolderId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FolderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A hierarchical folder for organizing clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Folder {
    /// Unique identifier.
    pub id: FolderId,

    /// Folder name.
    pub name: String,

    /// Parent folder ID.
    pub parent_id: Option<FolderId>,

    /// Optional description.
    pub description: Option<String>,

    /// Clips in this folder.
    clip_ids: HashSet<ClipId>,

    /// Child folder IDs.
    child_folder_ids: HashSet<FolderId>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,
}

impl Folder {
    /// Creates a new root folder.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: FolderId::new(),
            name: name.into(),
            parent_id: None,
            description: None,
            clip_ids: HashSet::new(),
            child_folder_ids: HashSet::new(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Creates a new child folder.
    #[must_use]
    pub fn new_child(name: impl Into<String>, parent_id: FolderId) -> Self {
        let mut folder = Self::new(name);
        folder.parent_id = Some(parent_id);
        folder
    }

    /// Adds a clip to the folder.
    pub fn add_clip(&mut self, clip_id: ClipId) -> bool {
        if self.clip_ids.insert(clip_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Removes a clip from the folder.
    pub fn remove_clip(&mut self, clip_id: &ClipId) -> bool {
        if self.clip_ids.remove(clip_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Adds a child folder.
    pub fn add_child_folder(&mut self, folder_id: FolderId) -> bool {
        if self.child_folder_ids.insert(folder_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Removes a child folder.
    pub fn remove_child_folder(&mut self, folder_id: &FolderId) -> bool {
        if self.child_folder_ids.remove(folder_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Returns all clip IDs in the folder.
    #[must_use]
    pub fn clips(&self) -> Vec<ClipId> {
        self.clip_ids.iter().copied().collect()
    }

    /// Returns all child folder IDs.
    #[must_use]
    pub fn child_folders(&self) -> Vec<FolderId> {
        self.child_folder_ids.iter().copied().collect()
    }

    /// Returns the number of clips in the folder.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clip_ids.len()
    }

    /// Returns the number of child folders.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.child_folder_ids.len()
    }

    /// Checks if this is a root folder.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Sets the description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
        self.modified_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folder_creation() {
        let folder = Folder::new("Root Folder");
        assert_eq!(folder.name, "Root Folder");
        assert!(folder.is_root());
    }

    #[test]
    fn test_child_folder() {
        let parent = Folder::new("Parent");
        let child = Folder::new_child("Child", parent.id);
        assert_eq!(child.parent_id, Some(parent.id));
        assert!(!child.is_root());
    }

    #[test]
    fn test_folder_clips() {
        let mut folder = Folder::new("Test");
        let clip = ClipId::new();

        assert!(folder.add_clip(clip));
        assert_eq!(folder.clip_count(), 1);

        assert!(folder.remove_clip(&clip));
        assert_eq!(folder.clip_count(), 0);
    }

    #[test]
    fn test_child_folders() {
        let mut parent = Folder::new("Parent");
        let child = Folder::new_child("Child", parent.id);

        assert!(parent.add_child_folder(child.id));
        assert_eq!(parent.child_count(), 1);
    }
}
