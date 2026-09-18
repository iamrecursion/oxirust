#![allow(dead_code)]
//! Search result caching with TTL-based invalidation for the search pipeline.
//!
//! Wraps the low-level `cache::SearchCache` with a high-level API specifically
//! designed for caching search results in the pipeline. Features:
//!
//! - Automatic cache key generation from query + filters + sort
//! - TTL-based expiry with configurable default TTL
//! - Index-version-based invalidation: bumping the version invalidates all entries
//! - Selective invalidation by tag (e.g., invalidate all results for a specific MIME type)
//! - Cache warming: pre-populate cache with results for frequent queries
//! - Statistics tracking (hits, misses, evictions, invalidations)

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::SearchResultItem;

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// A cache key derived from query parameters via FNV-1a hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResultCacheKey(u64);

impl ResultCacheKey {
    /// Compute a cache key from a query string.
    #[must_use]
    pub fn from_query(query: &str) -> Self {
        Self(fnv1a_hash(query.as_bytes()))
    }

    /// Compute a cache key from a query string + sort + offset + limit.
    #[must_use]
    pub fn from_params(query: &str, sort: &str, offset: usize, limit: usize) -> Self {
        let mut hasher_input = Vec::with_capacity(query.len() + sort.len() + 32);
        hasher_input.extend_from_slice(query.as_bytes());
        hasher_input.push(0xff);
        hasher_input.extend_from_slice(sort.as_bytes());
        hasher_input.push(0xff);
        hasher_input.extend_from_slice(&offset.to_le_bytes());
        hasher_input.extend_from_slice(&limit.to_le_bytes());
        Self(fnv1a_hash(&hasher_input))
    }

    /// Raw hash value.
    #[must_use]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// FNV-1a 64-bit hash function.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Cached result entry
// ---------------------------------------------------------------------------

/// A cached search result set with metadata.
#[derive(Debug, Clone)]
struct CachedResults {
    /// The cached results.
    results: Vec<SearchResultItem>,
    /// Total count (may differ from results.len() due to pagination).
    total: usize,
    /// When the entry was created.
    created_at: Instant,
    /// TTL for this entry.
    ttl: Duration,
    /// Index version at the time of caching.
    index_version: u64,
    /// Tags for selective invalidation.
    tags: HashSet<String>,
}

impl CachedResults {
    /// Check if this entry has expired.
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }

    /// Check if this entry's index version is stale.
    fn is_stale(&self, current_version: u64) -> bool {
        self.index_version < current_version
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Statistics for the result cache.
#[derive(Debug, Clone, Default)]
pub struct ResultCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted due to capacity.
    pub evictions: u64,
    /// Number of entries invalidated.
    pub invalidations: u64,
    /// Number of expired entries removed on access.
    pub expirations: u64,
}

impl ResultCacheStats {
    /// Hit rate as a fraction in [0.0, 1.0].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total lookups (hits + misses).
    #[must_use]
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }
}

// ---------------------------------------------------------------------------
// Result cache
// ---------------------------------------------------------------------------

/// High-level cache for search results with TTL and index-version invalidation.
#[derive(Debug)]
pub struct ResultCache {
    /// Cached entries.
    entries: HashMap<ResultCacheKey, CachedResults>,
    /// LRU order tracking.
    lru_order: VecDeque<ResultCacheKey>,
    /// Maximum number of cached result sets.
    capacity: usize,
    /// Default TTL for new entries.
    default_ttl: Duration,
    /// Current index version — bumping this invalidates all entries.
    index_version: u64,
    /// Statistics.
    stats: ResultCacheStats,
}

impl ResultCache {
    /// Create a new result cache with the given capacity and default TTL.
    #[must_use]
    pub fn new(capacity: usize, default_ttl_secs: u64) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            capacity: capacity.max(1),
            default_ttl: Duration::from_secs(default_ttl_secs),
            index_version: 0,
            stats: ResultCacheStats::default(),
        }
    }

    /// Look up cached results for the given key.
    ///
    /// Returns `None` on miss, expiry, or stale index version.
    pub fn get(&mut self, key: &ResultCacheKey) -> Option<(&[SearchResultItem], usize)> {
        // Check entry validity
        let should_remove = self.entries.get(key).map_or(false, |entry| {
            entry.is_expired() || entry.is_stale(self.index_version)
        });

        if should_remove {
            self.entries.remove(key);
            self.lru_order.retain(|k| k != key);
            if self
                .entries
                .get(key)
                .map_or(false, CachedResults::is_expired)
            {
                self.stats.expirations += 1;
            }
            self.stats.misses += 1;
            return None;
        }

        if !self.entries.contains_key(key) {
            self.stats.misses += 1;
            return None;
        }

        // Update LRU order
        self.lru_order.retain(|k| k != key);
        self.lru_order.push_back(*key);
        self.stats.hits += 1;

        self.entries
            .get(key)
            .map(|e| (e.results.as_slice(), e.total))
    }

    /// Insert results into the cache with the default TTL.
    pub fn insert(&mut self, key: ResultCacheKey, results: Vec<SearchResultItem>, total: usize) {
        self.insert_with_ttl(key, results, total, self.default_ttl, HashSet::new());
    }

    /// Insert results with a specific TTL and tags.
    pub fn insert_with_ttl(
        &mut self,
        key: ResultCacheKey,
        results: Vec<SearchResultItem>,
        total: usize,
        ttl: Duration,
        tags: HashSet<String>,
    ) {
        // Update existing
        if self.entries.contains_key(&key) {
            self.entries.insert(
                key,
                CachedResults {
                    results,
                    total,
                    created_at: Instant::now(),
                    ttl,
                    index_version: self.index_version,
                    tags,
                },
            );
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push_back(key);
            return;
        }

        // Evict LRU entries until under capacity
        while self.entries.len() >= self.capacity {
            if let Some(lru_key) = self.lru_order.pop_front() {
                self.entries.remove(&lru_key);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }

        self.entries.insert(
            key,
            CachedResults {
                results,
                total,
                created_at: Instant::now(),
                ttl,
                index_version: self.index_version,
                tags,
            },
        );
        self.lru_order.push_back(key);
    }

    /// Insert results with tags for selective invalidation.
    pub fn insert_tagged(
        &mut self,
        key: ResultCacheKey,
        results: Vec<SearchResultItem>,
        total: usize,
        tags: HashSet<String>,
    ) {
        self.insert_with_ttl(key, results, total, self.default_ttl, tags);
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &ResultCacheKey) {
        if self.entries.remove(key).is_some() {
            self.lru_order.retain(|k| k != key);
        }
    }

    /// Invalidate all entries that have a specific tag.
    ///
    /// Returns the number of entries invalidated.
    pub fn invalidate_by_tag(&mut self, tag: &str) -> usize {
        let keys_to_remove: Vec<ResultCacheKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.tags.contains(tag))
            .map(|(key, _)| *key)
            .collect();

        let count = keys_to_remove.len();
        for key in &keys_to_remove {
            self.entries.remove(key);
            self.lru_order.retain(|k| k != key);
        }
        self.stats.invalidations += count as u64;
        count
    }

    /// Bump the index version, effectively invalidating all cached entries.
    ///
    /// Entries are lazily removed on next access rather than eagerly cleared,
    /// so this operation is O(1).
    pub fn invalidate_all(&mut self) {
        self.index_version += 1;
    }

    /// Eagerly clear all entries.
    pub fn clear(&mut self) {
        let count = self.entries.len();
        self.entries.clear();
        self.lru_order.clear();
        self.stats.invalidations += count as u64;
    }

    /// Remove all expired entries eagerly.
    ///
    /// Returns the number of entries removed.
    pub fn purge_expired(&mut self) -> usize {
        let version = self.index_version;
        let expired_keys: Vec<ResultCacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired() || e.is_stale(version))
            .map(|(k, _)| *k)
            .collect();

        let count = expired_keys.len();
        for key in &expired_keys {
            self.entries.remove(key);
            self.lru_order.retain(|k| k != key);
        }
        self.stats.expirations += count as u64;
        count
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current index version.
    #[must_use]
    pub fn index_version(&self) -> u64 {
        self.index_version
    }

    /// Cache statistics.
    #[must_use]
    pub fn stats(&self) -> &ResultCacheStats {
        &self.stats
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = ResultCacheStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_results(n: usize) -> Vec<SearchResultItem> {
        (0..n)
            .map(|i| SearchResultItem {
                asset_id: Uuid::new_v4(),
                score: 1.0 - (i as f32 * 0.1),
                title: Some(format!("Result {i}")),
                description: None,
                file_path: format!("/media/file_{i}.mp4"),
                mime_type: Some("video/mp4".to_string()),
                duration_ms: Some(60_000),
                created_at: 1_700_000_000,
                modified_at: None,
                file_size: None,
                matched_fields: Vec::new(),
                thumbnail_url: None,
            })
            .collect()
    }

    // -- Key tests --

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = ResultCacheKey::from_query("video sunset");
        let k2 = ResultCacheKey::from_query("video sunset");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_queries() {
        let k1 = ResultCacheKey::from_query("video");
        let k2 = ResultCacheKey::from_query("audio");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_from_params() {
        let k1 = ResultCacheKey::from_params("video", "relevance", 0, 10);
        let k2 = ResultCacheKey::from_params("video", "relevance", 0, 10);
        assert_eq!(k1, k2);

        let k3 = ResultCacheKey::from_params("video", "relevance", 10, 10);
        assert_ne!(k1, k3);
    }

    // -- Basic cache operations --

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("test");
        let results = make_results(5);
        cache.insert(key, results.clone(), 50);

        let cached = cache.get(&key);
        assert!(cached.is_some());
        let (items, total) = cached.expect("should be cached");
        assert_eq!(items.len(), 5);
        assert_eq!(total, 50);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("nonexistent");
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_hit_stats() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("test");
        cache.insert(key, make_results(3), 3);
        let _ = cache.get(&key);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
        assert!((cache.stats().hit_rate() - 1.0).abs() < 1e-9);
    }

    // -- LRU eviction --

    #[test]
    fn test_lru_eviction() {
        let mut cache = ResultCache::new(3, 60);
        let k1 = ResultCacheKey::from_query("q1");
        let k2 = ResultCacheKey::from_query("q2");
        let k3 = ResultCacheKey::from_query("q3");
        let k4 = ResultCacheKey::from_query("q4");

        cache.insert(k1, make_results(1), 1);
        cache.insert(k2, make_results(1), 1);
        cache.insert(k3, make_results(1), 1);
        cache.insert(k4, make_results(1), 1); // evicts k1

        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k4).is_some());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut cache = ResultCache::new(3, 60);
        let k1 = ResultCacheKey::from_query("q1");
        let k2 = ResultCacheKey::from_query("q2");
        let k3 = ResultCacheKey::from_query("q3");
        let k4 = ResultCacheKey::from_query("q4");

        cache.insert(k1, make_results(1), 1);
        cache.insert(k2, make_results(1), 1);
        cache.insert(k3, make_results(1), 1);

        let _ = cache.get(&k1); // k1 is now most recent
        cache.insert(k4, make_results(1), 1); // should evict k2 (LRU)

        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none()); // evicted
    }

    // -- Index version invalidation --

    #[test]
    fn test_invalidate_all_via_version() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("test");
        cache.insert(key, make_results(3), 3);

        cache.invalidate_all(); // bump version

        // Entry is stale and should be treated as miss
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_new_entries_after_version_bump() {
        let mut cache = ResultCache::new(100, 60);
        cache.invalidate_all();

        let key = ResultCacheKey::from_query("test");
        cache.insert(key, make_results(2), 2);
        assert!(cache.get(&key).is_some()); // new entry at current version
    }

    // -- Tag-based invalidation --

    #[test]
    fn test_invalidate_by_tag() {
        let mut cache = ResultCache::new(100, 60);
        let k1 = ResultCacheKey::from_query("q1");
        let k2 = ResultCacheKey::from_query("q2");
        let k3 = ResultCacheKey::from_query("q3");

        let mut video_tags = HashSet::new();
        video_tags.insert("video".to_string());
        let mut audio_tags = HashSet::new();
        audio_tags.insert("audio".to_string());

        cache.insert_tagged(k1, make_results(1), 1, video_tags.clone());
        cache.insert_tagged(k2, make_results(1), 1, video_tags);
        cache.insert_tagged(k3, make_results(1), 1, audio_tags);

        let invalidated = cache.invalidate_by_tag("video");
        assert_eq!(invalidated, 2);
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_none());
        assert!(cache.get(&k3).is_some()); // audio tag, not invalidated
    }

    #[test]
    fn test_invalidate_by_tag_nonexistent() {
        let mut cache = ResultCache::new(100, 60);
        cache.insert(ResultCacheKey::from_query("q"), make_results(1), 1);
        let invalidated = cache.invalidate_by_tag("nonexistent");
        assert_eq!(invalidated, 0);
    }

    // -- Purge expired --

    #[test]
    fn test_purge_after_version_bump() {
        let mut cache = ResultCache::new(100, 60);
        cache.insert(ResultCacheKey::from_query("q1"), make_results(1), 1);
        cache.insert(ResultCacheKey::from_query("q2"), make_results(1), 1);

        cache.invalidate_all();
        let purged = cache.purge_expired();
        assert_eq!(purged, 2);
        assert!(cache.is_empty());
    }

    // -- Clear and remove --

    #[test]
    fn test_clear() {
        let mut cache = ResultCache::new(100, 60);
        cache.insert(ResultCacheKey::from_query("q1"), make_results(1), 1);
        cache.insert(ResultCacheKey::from_query("q2"), make_results(1), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().invalidations, 2);
    }

    #[test]
    fn test_remove() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("q");
        cache.insert(key, make_results(1), 1);
        cache.remove(&key);
        assert!(cache.get(&key).is_none());
    }

    // -- Stats --

    #[test]
    fn test_stats_total_lookups() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("q");
        cache.insert(key, make_results(1), 1);
        let _ = cache.get(&key);
        let _ = cache.get(&ResultCacheKey::from_query("miss"));
        assert_eq!(cache.stats().total_lookups(), 2);
    }

    #[test]
    fn test_reset_stats() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("q");
        cache.insert(key, make_results(1), 1);
        let _ = cache.get(&key);
        cache.reset_stats();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_capacity_and_length() {
        let cache = ResultCache::new(50, 60);
        assert_eq!(cache.capacity(), 50);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_index_version() {
        let mut cache = ResultCache::new(100, 60);
        assert_eq!(cache.index_version(), 0);
        cache.invalidate_all();
        assert_eq!(cache.index_version(), 1);
        cache.invalidate_all();
        assert_eq!(cache.index_version(), 2);
    }

    #[test]
    fn test_update_existing_entry() {
        let mut cache = ResultCache::new(100, 60);
        let key = ResultCacheKey::from_query("q");
        cache.insert(key, make_results(3), 30);
        cache.insert(key, make_results(5), 50);
        assert_eq!(cache.len(), 1);
        let (items, total) = cache.get(&key).expect("should exist");
        assert_eq!(items.len(), 5);
        assert_eq!(total, 50);
    }

    #[test]
    fn test_stats_hit_rate_no_lookups() {
        let stats = ResultCacheStats::default();
        assert!((stats.hit_rate()).abs() < 1e-9);
    }
}
