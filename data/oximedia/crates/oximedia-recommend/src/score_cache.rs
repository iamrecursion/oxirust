//! Score caching layer for recommendation systems.
//!
//! This module provides an in-memory cache for pre-computed recommendation
//! scores, supporting TTL-based expiration and LRU-like eviction to avoid
//! recomputing expensive similarity and ranking scores on every request.

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

/// A single cached score entry with expiration metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    /// The cached score value
    pub score: f64,
    /// Unix timestamp when this entry was created
    pub created_at: i64,
    /// Unix timestamp when this entry expires
    pub expires_at: i64,
    /// Number of times this entry was accessed
    pub access_count: u64,
    /// Unix timestamp of last access
    pub last_accessed: i64,
}

impl CacheEntry {
    /// Creates a new cache entry with the given TTL in seconds.
    #[must_use]
    pub fn new(score: f64, now: i64, ttl_secs: i64) -> Self {
        Self {
            score,
            created_at: now,
            expires_at: now + ttl_secs,
            access_count: 0,
            last_accessed: now,
        }
    }

    /// Returns true if this entry has expired at the given timestamp.
    #[must_use]
    pub fn is_expired(&self, now: i64) -> bool {
        now >= self.expires_at
    }

    /// Returns the remaining TTL in seconds, or 0 if expired.
    #[must_use]
    pub fn remaining_ttl(&self, now: i64) -> i64 {
        (self.expires_at - now).max(0)
    }

    /// Returns the age of this entry in seconds.
    #[must_use]
    pub fn age(&self, now: i64) -> i64 {
        now - self.created_at
    }

    /// Records an access, updating the access count and timestamp.
    pub fn record_access(&mut self, now: i64) {
        self.access_count += 1;
        self.last_accessed = now;
    }
}

/// Cache key combining user and content identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// User ID
    pub user_id: Uuid,
    /// Content ID
    pub content_id: Uuid,
}

impl CacheKey {
    /// Creates a new cache key.
    #[must_use]
    pub fn new(user_id: Uuid, content_id: Uuid) -> Self {
        Self {
            user_id,
            content_id,
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    /// Total number of cache lookups
    pub lookups: u64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries evicted
    pub evictions: u64,
    /// Number of expired entries removed
    pub expirations: u64,
}

impl CacheStats {
    /// Returns the hit rate as a fraction (0.0-1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}

/// An in-memory score cache with TTL-based expiration and capacity limits.
///
/// Supports `get_or_compute()` for transparent caching: if a score is
/// already cached and valid, it is returned; otherwise, the provided
/// compute function is called and its result is stored.
#[derive(Debug)]
pub struct ScoreCache {
    /// Cached entries
    entries: HashMap<CacheKey, CacheEntry>,
    /// Default TTL in seconds
    default_ttl_secs: i64,
    /// Maximum number of entries
    max_capacity: usize,
    /// Cache statistics
    stats: CacheStats,
}

impl ScoreCache {
    /// Creates a new score cache.
    ///
    /// # Arguments
    ///
    /// * `default_ttl_secs` - Default time-to-live for entries in seconds.
    /// * `max_capacity` - Maximum number of cached entries.
    #[must_use]
    pub fn new(default_ttl_secs: i64, max_capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl_secs,
            max_capacity,
            stats: CacheStats::default(),
        }
    }

    /// Looks up a cached score.
    ///
    /// Returns `Some(score)` if the entry exists and is not expired.
    /// Returns `None` on miss or expiration.
    pub fn get(&mut self, key: &CacheKey, now: i64) -> Option<f64> {
        self.stats.lookups += 1;

        let entry = if let Some(e) = self.entries.get_mut(key) {
            e
        } else {
            self.stats.misses += 1;
            return None;
        };
        if entry.is_expired(now) {
            self.stats.misses += 1;
            // Remove expired entry
            self.entries.remove(key);
            self.stats.expirations += 1;
            return None;
        }

        entry.record_access(now);
        self.stats.hits += 1;
        Some(entry.score)
    }

    /// Inserts a score into the cache.
    ///
    /// If the cache is at capacity, evicts the least-recently-accessed entry first.
    pub fn put(&mut self, key: CacheKey, score: f64, now: i64) {
        self.put_with_ttl(key, score, now, self.default_ttl_secs);
    }

    /// Inserts a score with a custom TTL.
    pub fn put_with_ttl(&mut self, key: CacheKey, score: f64, now: i64, ttl_secs: i64) {
        if self.entries.len() >= self.max_capacity && !self.entries.contains_key(&key) {
            self.evict_lru();
        }
        self.entries
            .insert(key, CacheEntry::new(score, now, ttl_secs));
    }

    /// Gets a cached score, or computes and caches it if missing/expired.
    ///
    /// The `compute` closure is only called on a cache miss.
    pub fn get_or_compute<F>(&mut self, key: CacheKey, now: i64, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        if let Some(score) = self.get(&key, now) {
            return score;
        }
        let score = compute();
        self.put(key, score, now);
        score
    }

    /// Removes all expired entries from the cache.
    pub fn evict_expired(&mut self, now: i64) {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired(now));
        let removed = before - self.entries.len();
        self.stats.expirations += removed as u64;
    }

    /// Evicts the least-recently-accessed entry.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| *k);

        if let Some(key) = lru_key {
            self.entries.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Invalidates (removes) a specific entry.
    pub fn invalidate(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Invalidates all entries for a given user.
    pub fn invalidate_user(&mut self, user_id: Uuid) {
        self.entries.retain(|k, _| k.user_id != user_id);
    }

    /// Invalidates all entries for a given content item.
    pub fn invalidate_content(&mut self, content_id: Uuid) {
        self.entries.retain(|k, _| k.content_id != content_id);
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a reference to cache statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Resets cache statistics to zero.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_cache_entry_creation() {
        let entry = CacheEntry::new(0.85, 1000, 60);
        assert!((entry.score - 0.85).abs() < f64::EPSILON);
        assert_eq!(entry.expires_at, 1060);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new(1.0, 1000, 60);
        assert!(!entry.is_expired(1050));
        assert!(entry.is_expired(1060));
        assert!(entry.is_expired(1100));
    }

    #[test]
    fn test_cache_entry_remaining_ttl() {
        let entry = CacheEntry::new(1.0, 1000, 60);
        assert_eq!(entry.remaining_ttl(1000), 60);
        assert_eq!(entry.remaining_ttl(1030), 30);
        assert_eq!(entry.remaining_ttl(1070), 0);
    }

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry::new(1.0, 1000, 60);
        assert_eq!(entry.age(1000), 0);
        assert_eq!(entry.age(1025), 25);
    }

    #[test]
    fn test_cache_entry_record_access() {
        let mut entry = CacheEntry::new(1.0, 1000, 60);
        entry.record_access(1010);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_accessed, 1010);
        entry.record_access(1020);
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        cache.put(key, 0.92, 1000);
        let val = cache.get(&key, 1001).expect("should succeed in test");
        assert!((val - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        assert!(cache.get(&key, 1000).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_expired_entry_returns_none() {
        let mut cache = ScoreCache::new(60, 100);
        let key = CacheKey::new(uid(), uid());
        cache.put(key, 0.5, 1000);
        assert!(cache.get(&key, 1061).is_none());
    }

    #[test]
    fn test_get_or_compute_miss() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        let val = cache.get_or_compute(key, 1000, || 0.77);
        assert!((val - 0.77).abs() < f64::EPSILON);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_get_or_compute_hit() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        cache.put(key, 0.5, 1000);
        let val = cache.get_or_compute(key, 1001, || 0.99);
        // Should return cached value, not compute
        assert!((val - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evict_expired() {
        let mut cache = ScoreCache::new(60, 100);
        let k1 = CacheKey::new(uid(), uid());
        let k2 = CacheKey::new(uid(), uid());
        cache.put(k1, 0.1, 1000);
        cache.put(k2, 0.2, 1050);
        cache.evict_expired(1061);
        // k1 expired (created at 1000, ttl 60 => expires 1060), k2 alive (expires 1110)
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut cache = ScoreCache::new(300, 2);
        let k1 = CacheKey::new(uid(), uid());
        let k2 = CacheKey::new(uid(), uid());
        let k3 = CacheKey::new(uid(), uid());
        cache.put(k1, 0.1, 1000);
        cache.put(k2, 0.2, 1001);
        // Access k1 so k2 becomes LRU
        let _ = cache.get(&k1, 1002);
        cache.put(k3, 0.3, 1003);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_invalidate_specific() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        cache.put(key, 0.5, 1000);
        assert!(cache.invalidate(&key));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_invalidate_user() {
        let mut cache = ScoreCache::new(300, 100);
        let u = uid();
        let k1 = CacheKey::new(u, uid());
        let k2 = CacheKey::new(u, uid());
        let k3 = CacheKey::new(uid(), uid());
        cache.put(k1, 0.1, 1000);
        cache.put(k2, 0.2, 1000);
        cache.put(k3, 0.3, 1000);
        cache.invalidate_user(u);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = ScoreCache::new(300, 100);
        let key = CacheKey::new(uid(), uid());
        cache.put(key, 0.5, 1000);
        let _ = cache.get(&key, 1001); // hit
        let _ = cache.get(&CacheKey::new(uid(), uid()), 1001); // miss
        assert!((cache.stats().hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_cache() {
        let mut cache = ScoreCache::new(300, 100);
        cache.put(CacheKey::new(uid(), uid()), 0.1, 1000);
        cache.put(CacheKey::new(uid(), uid()), 0.2, 1000);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
