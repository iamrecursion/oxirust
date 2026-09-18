#![allow(dead_code)]
//! Response-level caching with TTL and invalidation.
//!
//! Provides an in-memory key-value cache for HTTP responses (or arbitrary
//! byte payloads) with configurable TTL, capacity limits, and LRU eviction.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single cached entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached payload bytes.
    pub body: Vec<u8>,
    /// Content-Type header value.
    pub content_type: String,
    /// HTTP status code.
    pub status: u16,
    /// When this entry was created.
    created_at: Instant,
    /// Time-to-live.
    ttl: Duration,
    /// Number of times this entry has been served.
    pub hit_count: u64,
    /// Last access time (for LRU eviction).
    last_accessed: Instant,
}

impl CacheEntry {
    /// Creates a new cache entry.
    pub fn new(body: Vec<u8>, content_type: impl Into<String>, status: u16, ttl: Duration) -> Self {
        let now = Instant::now();
        Self {
            body,
            content_type: content_type.into(),
            status,
            created_at: now,
            ttl,
            hit_count: 0,
            last_accessed: now,
        }
    }

    /// Returns `true` if the entry has expired.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    /// Age of the entry.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Remaining TTL (zero if expired).
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.created_at.elapsed())
    }

    /// Size in bytes of the cached body.
    pub fn body_size(&self) -> usize {
        self.body.len()
    }

    /// Records a hit and updates last-access time.
    fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_accessed = Instant::now();
    }
}

/// Statistics for the response cache.
#[derive(Debug, Clone)]
pub struct ResponseCacheStats {
    /// Total number of lookups.
    pub lookups: u64,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Total entries evicted.
    pub evictions: u64,
    /// Total entries invalidated.
    pub invalidations: u64,
    /// Current number of entries.
    pub entries: usize,
    /// Current total bytes cached.
    pub bytes_cached: usize,
}

impl ResponseCacheStats {
    /// Creates zeroed stats.
    pub fn new() -> Self {
        Self {
            lookups: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            invalidations: 0,
            entries: 0,
            bytes_cached: 0,
        }
    }

    /// Hit ratio as a fraction.
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_ratio(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }

    /// Average entry size in bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_entry_size(&self) -> f64 {
        if self.entries == 0 {
            return 0.0;
        }
        self.bytes_cached as f64 / self.entries as f64
    }
}

impl Default for ResponseCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the response cache.
#[derive(Debug, Clone)]
pub struct ResponseCacheConfig {
    /// Maximum number of entries.
    pub max_entries: usize,
    /// Maximum total cached bytes.
    pub max_bytes: usize,
    /// Default TTL for entries that don't specify one.
    pub default_ttl: Duration,
    /// Whether to serve stale entries while revalidating.
    pub serve_stale: bool,
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: 64 * 1024 * 1024, // 64 MB
            default_ttl: Duration::from_secs(300),
            serve_stale: false,
        }
    }
}

/// An in-memory response cache.
pub struct ResponseCache {
    /// Configuration.
    config: ResponseCacheConfig,
    /// Cached entries keyed by request path / key.
    entries: HashMap<String, CacheEntry>,
    /// Accumulated statistics.
    stats: ResponseCacheStats,
    /// Total bytes currently cached.
    current_bytes: usize,
}

impl ResponseCache {
    /// Creates a new response cache.
    pub fn new(config: ResponseCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: ResponseCacheStats::new(),
            current_bytes: 0,
        }
    }

    /// Inserts an entry into the cache.
    /// Evicts LRU entries if capacity is exceeded.
    pub fn put(&mut self, key: impl Into<String>, entry: CacheEntry) {
        let key = key.into();
        let entry_size = entry.body_size();

        // Remove existing entry with same key first
        if let Some(old) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.body_size());
        }

        // Evict until we have space
        while self.entries.len() >= self.config.max_entries
            || self.current_bytes + entry_size > self.config.max_bytes
        {
            if !self.evict_lru() {
                break;
            }
        }

        self.current_bytes += entry_size;
        self.entries.insert(key, entry);
        self.update_stats_counts();
    }

    /// Looks up a cache entry. Returns `None` on miss or expiry.
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        self.stats.lookups += 1;

        // Check for existence and expiry without borrowing issues
        let expired = self.entries.get(key).map(|e| e.is_expired());

        match expired {
            Some(true) => {
                if !self.config.serve_stale {
                    self.remove(key);
                    self.stats.misses += 1;
                    return None;
                }
                // serve stale
                self.stats.hits += 1;
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.record_hit();
                }
                self.entries.get(key)
            }
            Some(false) => {
                self.stats.hits += 1;
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.record_hit();
                }
                self.entries.get(key)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Checks if a key exists and is not expired.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.get(key).is_some_and(|e| !e.is_expired())
    }

    /// Removes a specific key.
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.body_size());
            self.stats.invalidations += 1;
            self.update_stats_counts();
            true
        } else {
            false
        }
    }

    /// Invalidates all entries whose keys start with the given prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys_to_remove: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            self.remove(&key);
        }
        count
    }

    /// Removes all expired entries.
    pub fn purge_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for key in expired_keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.body_size());
                self.stats.evictions += 1;
            }
        }
        self.update_stats_counts();
        count
    }

    /// Clears the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
        self.update_stats_counts();
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &ResponseCacheStats {
        &self.stats
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    // ── Internal helpers ──

    fn evict_lru(&mut self) -> bool {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.body_size());
                self.stats.evictions += 1;
                self.update_stats_counts();
                return true;
            }
        }
        false
    }

    fn update_stats_counts(&mut self) {
        self.stats.entries = self.entries.len();
        self.stats.bytes_cached = self.current_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(body: &[u8], ttl_secs: u64) -> CacheEntry {
        CacheEntry::new(
            body.to_vec(),
            "text/plain",
            200,
            Duration::from_secs(ttl_secs),
        )
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/api/data", make_entry(b"hello", 60));
        let entry = cache.get("/api/data");
        assert!(entry.is_some());
        assert_eq!(entry.expect("should succeed in test").body, b"hello");
    }

    #[test]
    fn test_miss() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        assert!(cache.get("/missing").is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_hit_count_increments() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/x", make_entry(b"data", 60));
        cache.get("/x");
        cache.get("/x");
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn test_expired_entry_removed() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/exp", make_entry(b"data", 0)); // 0 second TTL
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get("/exp").is_none());
    }

    #[test]
    fn test_remove() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/rem", make_entry(b"data", 60));
        assert!(cache.remove("/rem"));
        assert!(!cache.contains("/rem"));
        assert_eq!(cache.stats().invalidations, 1);
    }

    #[test]
    fn test_invalidate_prefix() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/api/users/1", make_entry(b"u1", 60));
        cache.put("/api/users/2", make_entry(b"u2", 60));
        cache.put("/api/media/1", make_entry(b"m1", 60));
        let count = cache.invalidate_prefix("/api/users");
        assert_eq!(count, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_purge_expired() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/a", make_entry(b"a", 0));
        cache.put("/b", make_entry(b"b", 600));
        std::thread::sleep(Duration::from_millis(5));
        let purged = cache.purge_expired();
        assert_eq!(purged, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = ResponseCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = ResponseCache::new(config);
        cache.put("/a", make_entry(b"a", 60));
        std::thread::sleep(Duration::from_millis(1));
        cache.put("/b", make_entry(b"b", 60));
        std::thread::sleep(Duration::from_millis(1));
        cache.put("/c", make_entry(b"c", 60)); // should evict /a
        assert!(!cache.contains("/a"));
        assert!(cache.contains("/b"));
        assert!(cache.contains("/c"));
    }

    #[test]
    fn test_clear() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/a", make_entry(b"a", 60));
        cache.put("/b", make_entry(b"b", 60));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().bytes_cached, 0);
    }

    #[test]
    fn test_stats_hit_ratio() {
        let mut stats = ResponseCacheStats::new();
        stats.lookups = 10;
        stats.hits = 7;
        stats.misses = 3;
        assert!((stats.hit_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_entry_remaining_ttl() {
        let entry = make_entry(b"data", 600);
        assert!(entry.remaining_ttl() > Duration::from_secs(599));
    }

    #[test]
    fn test_contains_returns_false_for_expired() {
        let mut cache = ResponseCache::new(ResponseCacheConfig::default());
        cache.put("/x", make_entry(b"x", 0));
        std::thread::sleep(Duration::from_millis(5));
        assert!(!cache.contains("/x"));
    }

    #[test]
    fn test_default_config() {
        let cfg = ResponseCacheConfig::default();
        assert_eq!(cfg.max_entries, 1024);
        assert_eq!(cfg.max_bytes, 64 * 1024 * 1024);
        assert!(!cfg.serve_stale);
    }
}
