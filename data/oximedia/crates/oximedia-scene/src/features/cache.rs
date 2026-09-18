//! Feature caching for scene analysis.

use std::collections::HashMap;

/// Cache for extracted features to avoid redundant computation.
#[derive(Debug, Clone, Default)]
pub struct FeatureCache {
    entries: HashMap<u64, Vec<f32>>,
}

impl FeatureCache {
    /// Create a new empty feature cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert features for a given frame index.
    pub fn insert(&mut self, frame_id: u64, features: Vec<f32>) {
        self.entries.insert(frame_id, features);
    }

    /// Get cached features for a frame index.
    #[must_use]
    pub fn get(&self, frame_id: u64) -> Option<&Vec<f32>> {
        self.entries.get(&frame_id)
    }

    /// Check if features are cached for a frame index.
    #[must_use]
    pub fn contains(&self, frame_id: u64) -> bool {
        self.entries.contains_key(&frame_id)
    }

    /// Remove features for a frame index.
    pub fn remove(&mut self, frame_id: u64) -> Option<Vec<f32>> {
        self.entries.remove(&frame_id)
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new() {
        let cache = FeatureCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_get() {
        let mut cache = FeatureCache::new();
        cache.insert(42, vec![1.0, 2.0, 3.0]);
        assert!(cache.contains(42));
        assert_eq!(cache.get(42), Some(&vec![1.0, 2.0, 3.0]));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = FeatureCache::new();
        cache.insert(1, vec![1.0]);
        assert!(cache.remove(1).is_some());
        assert!(!cache.contains(1));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = FeatureCache::new();
        cache.insert(1, vec![1.0]);
        cache.insert(2, vec![2.0]);
        cache.clear();
        assert!(cache.is_empty());
    }
}
