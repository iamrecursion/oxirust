//! Collection management for clips.

use crate::clip::ClipId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Unique identifier for a collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionId(Uuid);

impl CollectionId {
    /// Creates a new random collection ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a collection ID from a UUID.
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

impl Default for CollectionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A collection of clips (manual grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Unique identifier.
    pub id: CollectionId,

    /// Collection name.
    pub name: String,

    /// Optional description.
    pub description: Option<String>,

    /// Clips in this collection.
    clip_ids: HashSet<ClipId>,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,

    /// Color tag (RGB hex).
    pub color: Option<String>,
}

impl Collection {
    /// Creates a new collection.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: CollectionId::new(),
            name: name.into(),
            description: None,
            clip_ids: HashSet::new(),
            created_at: now,
            modified_at: now,
            color: None,
        }
    }

    /// Adds a clip to the collection.
    pub fn add_clip(&mut self, clip_id: ClipId) -> bool {
        if self.clip_ids.insert(clip_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Removes a clip from the collection.
    pub fn remove_clip(&mut self, clip_id: &ClipId) -> bool {
        if self.clip_ids.remove(clip_id) {
            self.modified_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Checks if the collection contains a clip.
    #[must_use]
    pub fn contains(&self, clip_id: &ClipId) -> bool {
        self.clip_ids.contains(clip_id)
    }

    /// Returns all clip IDs in the collection.
    #[must_use]
    pub fn clips(&self) -> Vec<ClipId> {
        self.clip_ids.iter().copied().collect()
    }

    /// Returns the number of clips in the collection.
    #[must_use]
    pub fn count(&self) -> usize {
        self.clip_ids.len()
    }

    /// Checks if the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clip_ids.is_empty()
    }

    /// Sets the description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
        self.modified_at = Utc::now();
    }

    /// Sets the color tag.
    pub fn set_color(&mut self, color: impl Into<String>) {
        self.color = Some(color.into());
        self.modified_at = Utc::now();
    }

    /// Clears all clips from the collection.
    pub fn clear(&mut self) {
        self.clip_ids.clear();
        self.modified_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_creation() {
        let collection = Collection::new("Best Takes");
        assert_eq!(collection.name, "Best Takes");
        assert!(collection.is_empty());
    }

    #[test]
    fn test_collection_clips() {
        let mut collection = Collection::new("Test");
        let clip1 = ClipId::new();
        let clip2 = ClipId::new();

        assert!(collection.add_clip(clip1));
        assert!(!collection.add_clip(clip1)); // Duplicate
        assert!(collection.add_clip(clip2));

        assert_eq!(collection.count(), 2);
        assert!(collection.contains(&clip1));

        collection.clear();
        assert!(collection.is_empty());
    }
}
