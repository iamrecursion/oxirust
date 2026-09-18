//! Favorites management for clips.

use crate::clip::ClipId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Manages favorite clips.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Favorite {
    clip_ids: HashSet<ClipId>,
}

impl Favorite {
    /// Creates a new favorites manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clip_ids: HashSet::new(),
        }
    }

    /// Adds a clip to favorites.
    pub fn add(&mut self, clip_id: ClipId) -> bool {
        self.clip_ids.insert(clip_id)
    }

    /// Removes a clip from favorites.
    pub fn remove(&mut self, clip_id: &ClipId) -> bool {
        self.clip_ids.remove(clip_id)
    }

    /// Checks if a clip is in favorites.
    #[must_use]
    pub fn contains(&self, clip_id: &ClipId) -> bool {
        self.clip_ids.contains(clip_id)
    }

    /// Returns all favorite clip IDs.
    #[must_use]
    pub fn all(&self) -> Vec<ClipId> {
        self.clip_ids.iter().copied().collect()
    }

    /// Returns the number of favorites.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clip_ids.len()
    }

    /// Checks if there are no favorites.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clip_ids.is_empty()
    }

    /// Clears all favorites.
    pub fn clear(&mut self) {
        self.clip_ids.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_favorites() {
        let mut favorites = Favorite::new();
        let clip1 = ClipId::new();
        let clip2 = ClipId::new();

        assert!(favorites.add(clip1));
        assert!(!favorites.add(clip1)); // Already added

        assert!(favorites.contains(&clip1));
        assert!(!favorites.contains(&clip2));
        assert_eq!(favorites.len(), 1);

        assert!(favorites.remove(&clip1));
        assert!(!favorites.remove(&clip1)); // Already removed
        assert!(favorites.is_empty());
    }
}
