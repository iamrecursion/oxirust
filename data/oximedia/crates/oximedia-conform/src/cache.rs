//! Caching for media metadata and match results.

use crate::types::{ClipMatch, MediaFile};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cache entry with expiration.
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    /// Cached value.
    value: T,
    /// Insertion time.
    inserted_at: Instant,
    /// Time to live.
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry.
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            inserted_at: Instant::now(),
            ttl,
        }
    }

    /// Check if the entry is expired.
    fn is_expired(&self) -> bool {
        self.inserted_at.elapsed() > self.ttl
    }
}

/// Media metadata cache.
pub struct MediaCache {
    /// Cache storage.
    cache: Arc<DashMap<PathBuf, CacheEntry<MediaFile>>>,
    /// Default TTL.
    default_ttl: Duration,
}

impl MediaCache {
    /// Create a new media cache with default TTL of 1 hour.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            default_ttl: Duration::from_secs(3600),
        }
    }

    /// Create a cache with a specific TTL.
    #[must_use]
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            default_ttl: ttl,
        }
    }

    /// Get a cached media file.
    #[must_use]
    pub fn get(&self, path: &PathBuf) -> Option<MediaFile> {
        if let Some(entry) = self.cache.get(path) {
            if !entry.is_expired() {
                return Some(entry.value.clone());
            }
            // Entry expired, remove it
            drop(entry);
            self.cache.remove(path);
        }
        None
    }

    /// Insert a media file into the cache.
    pub fn insert(&self, path: PathBuf, media: MediaFile) {
        let entry = CacheEntry::new(media, self.default_ttl);
        self.cache.insert(path, entry);
    }

    /// Insert with custom TTL.
    pub fn insert_with_ttl(&self, path: PathBuf, media: MediaFile, ttl: Duration) {
        let entry = CacheEntry::new(media, ttl);
        self.cache.insert(path, entry);
    }

    /// Remove a specific entry.
    #[must_use]
    pub fn remove(&self, path: &PathBuf) -> Option<MediaFile> {
        self.cache.remove(path).map(|(_, entry)| entry.value)
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Remove expired entries.
    pub fn cleanup_expired(&self) {
        self.cache.retain(|_, entry| !entry.is_expired());
    }

    /// Get cache size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for MediaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Match results cache.
pub struct MatchCache {
    /// Cache storage.
    cache: Arc<RwLock<Vec<ClipMatch>>>,
}

impl MatchCache {
    /// Create a new match cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Store match results.
    pub fn store(&self, matches: Vec<ClipMatch>) {
        *self.cache.write() = matches;
    }

    /// Get all matches.
    #[must_use]
    pub fn get_all(&self) -> Vec<ClipMatch> {
        self.cache.read().clone()
    }

    /// Find matches for a specific clip ID.
    #[must_use]
    pub fn find_by_clip_id(&self, clip_id: &str) -> Vec<ClipMatch> {
        self.cache
            .read()
            .iter()
            .filter(|m| m.clip.id == clip_id)
            .cloned()
            .collect()
    }

    /// Clear all matches.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Get match count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }
}

impl Default for MatchCache {
    fn default() -> Self {
        Self::new()
    }
}

/// LRU (Least Recently Used) cache for media files.
pub struct LruMediaCache {
    /// Maximum capacity.
    capacity: usize,
    /// Cache storage with access times.
    cache: Arc<DashMap<PathBuf, (MediaFile, Instant)>>,
}

impl LruMediaCache {
    /// Create a new LRU cache with specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: Arc::new(DashMap::new()),
        }
    }

    /// Get a cached media file.
    #[must_use]
    pub fn get(&self, path: &PathBuf) -> Option<MediaFile> {
        if let Some(mut entry) = self.cache.get_mut(path) {
            // Update access time
            entry.1 = Instant::now();
            Some(entry.0.clone())
        } else {
            None
        }
    }

    /// Insert a media file into the cache.
    pub fn insert(&self, path: PathBuf, media: MediaFile) {
        // Check capacity and evict if necessary
        if self.cache.len() >= self.capacity {
            self.evict_lru();
        }

        self.cache.insert(path, (media, Instant::now()));
    }

    /// Evict the least recently used entry.
    fn evict_lru(&self) {
        if let Some(entry) = self.cache.iter().min_by_key(|entry| entry.value().1) {
            let oldest_path = entry.key().clone();
            drop(entry);
            self.cache.remove(&oldest_path);
        }
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get cache size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Simple in-memory cache for generic key-value pairs.
pub struct SimpleCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Cache storage.
    cache: Arc<DashMap<K, V>>,
    /// Maximum size.
    max_size: usize,
}

impl<K, V> SimpleCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create a new simple cache.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            max_size,
        }
    }

    /// Get a value from the cache.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<V> {
        self.cache.get(key).map(|v| v.clone())
    }

    /// Insert a value into the cache.
    pub fn insert(&self, key: K, value: V) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(first_key) = self.cache.iter().next().map(|entry| entry.key().clone()) {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }

    /// Remove a value from the cache.
    pub fn remove(&self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|(_, v)| v)
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get cache size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_media_cache() {
        let cache = MediaCache::new();
        let path = PathBuf::from("/test/file.mov");
        let media = MediaFile::new(path.clone());

        cache.insert(path.clone(), media.clone());
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(&path);
        assert!(retrieved.is_some());

        let _ = cache.remove(&path);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_media_cache_expiration() {
        let cache = MediaCache::with_ttl(Duration::from_millis(100));
        let path = PathBuf::from("/test/file.mov");
        let media = MediaFile::new(path.clone());

        cache.insert(path.clone(), media);
        assert!(cache.get(&path).is_some());

        thread::sleep(Duration::from_millis(150));
        assert!(cache.get(&path).is_none());
    }

    #[test]
    fn test_match_cache() {
        let cache = MatchCache::new();
        assert_eq!(cache.len(), 0);

        cache.store(vec![]);
        assert_eq!(cache.len(), 0);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_cache() {
        let cache = LruMediaCache::new(2);
        let path1 = PathBuf::from("/test/file1.mov");
        let path2 = PathBuf::from("/test/file2.mov");
        let path3 = PathBuf::from("/test/file3.mov");

        cache.insert(path1.clone(), MediaFile::new(path1.clone()));
        cache.insert(path2.clone(), MediaFile::new(path2.clone()));
        assert_eq!(cache.len(), 2);

        // This should evict the least recently used
        cache.insert(path3.clone(), MediaFile::new(path3.clone()));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_simple_cache() {
        let cache = SimpleCache::new(3);
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");

        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert_eq!(cache.get(&"key3"), None);

        cache.remove(&"key1");
        assert_eq!(cache.get(&"key1"), None);
    }
}
