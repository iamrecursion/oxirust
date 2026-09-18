//! Search result caching with LRU eviction.
//!
//! Keys are derived from queries via FNV-1a hashing; values are wrapped in
//! [`CacheEntry`] to support TTL-based expiry.  [`SearchCache`] maintains a
//! `VecDeque` as a simple LRU order tracker.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ──────────────────────────────────────────────────────────────────────────────
// CacheKey
// ──────────────────────────────────────────────────────────────────────────────

/// An opaque key derived from a serialised query via FNV-1a hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(u64);

impl CacheKey {
    /// Computes a `CacheKey` from any byte sequence using FNV-1a 64-bit.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(hash)
    }

    /// Computes a `CacheKey` from a string slice.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }

    /// Returns the raw hash value.
    #[must_use]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheEntry
// ──────────────────────────────────────────────────────────────────────────────

/// A cached value together with its creation time and TTL.
pub struct CacheEntry<T> {
    /// The cached data.
    pub data: T,
    /// Instant at which this entry was created.
    pub created_at: Instant,
    /// Time-to-live in seconds.  `0` means "never expire".
    pub ttl_secs: u64,
}

impl<T> CacheEntry<T> {
    /// Creates a new entry.
    #[must_use]
    pub fn new(data: T, ttl_secs: u64) -> Self {
        Self {
            data,
            created_at: Instant::now(),
            ttl_secs,
        }
    }

    /// Returns `true` when the entry has outlived its TTL.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }
        self.created_at.elapsed() >= Duration::from_secs(self.ttl_secs)
    }

    /// Returns the age of this entry.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheStats
// ──────────────────────────────────────────────────────────────────────────────

/// Statistics for a [`SearchCache`] instance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of failed cache lookups (key absent or expired).
    pub misses: u64,
    /// Number of entries evicted due to capacity.
    pub evictions: u64,
}

impl CacheStats {
    /// Computes the hit-rate as a value in `[0, 1]`.  Returns `0.0` when no
    /// lookups have been performed yet.
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SearchCache
// ──────────────────────────────────────────────────────────────────────────────

/// An LRU cache for search results.
///
/// Internally a `HashMap<CacheKey, CacheEntry<T>>` stores the entries and a
/// `VecDeque<CacheKey>` tracks access order (most recently used at the back).
/// When the cache exceeds `capacity`, the least recently used entry (front of
/// the deque) is evicted.
pub struct SearchCache<T> {
    entries: HashMap<CacheKey, CacheEntry<T>>,
    order: VecDeque<CacheKey>,
    capacity: usize,
    stats: CacheStats,
    /// Default TTL applied to newly inserted entries.  `0` means no expiry.
    pub default_ttl_secs: u64,
}

impl<T> SearchCache<T> {
    /// Creates a new cache with the given maximum `capacity`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
            stats: CacheStats::default(),
            default_ttl_secs: 0,
        }
    }

    /// Creates a cache with a non-zero default TTL.
    #[must_use]
    pub fn with_ttl(capacity: usize, ttl_secs: u64) -> Self {
        let mut c = Self::new(capacity);
        c.default_ttl_secs = ttl_secs;
        c
    }

    /// Returns a reference to the cached data for `key`, or `None` if absent
    /// or expired.  Updates LRU order on a hit.
    pub fn get(&mut self, key: &CacheKey) -> Option<&T> {
        // Check existence and expiry without borrowing mutably for too long.
        let expired = self.entries.get(key).map_or(true, CacheEntry::is_expired);

        if expired {
            if self.entries.contains_key(key) {
                // Remove expired entry.
                self.entries.remove(key);
                self.order.retain(|k| k != key);
            }
            self.stats.misses += 1;
            return None;
        }

        // Move to back (most recently used).
        self.order.retain(|k| k != key);
        self.order.push_back(*key);
        self.stats.hits += 1;
        self.entries.get(key).map(|e| &e.data)
    }

    /// Inserts a value for `key` using `default_ttl_secs`.
    ///
    /// If the cache is at capacity the least recently used entry is evicted.
    pub fn insert(&mut self, key: CacheKey, data: T) {
        self.insert_with_ttl(key, data, self.default_ttl_secs);
    }

    /// Inserts a value for `key` with an explicit TTL (seconds).
    pub fn insert_with_ttl(&mut self, key: CacheKey, data: T, ttl_secs: u64) {
        // If the key already exists, update in-place (no capacity change).
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.entries.entry(key) {
            e.insert(CacheEntry::new(data, ttl_secs));
            self.order.retain(|k| k != &key);
            self.order.push_back(key);
            return;
        }

        // Evict LRU entries until below capacity.
        while self.entries.len() >= self.capacity {
            if let Some(lru_key) = self.order.pop_front() {
                self.entries.remove(&lru_key);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }

        self.entries.insert(key, CacheEntry::new(data, ttl_secs));
        self.order.push_back(key);
    }

    /// Removes the entry for `key` if it exists.
    pub fn remove(&mut self, key: &CacheKey) {
        self.entries.remove(key);
        self.order.retain(|k| k != key);
    }

    /// Returns the number of valid (non-expired) entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries and resets statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.stats = CacheStats::default();
    }

    /// Returns a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Maximum capacity of this cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CacheKey ──

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = CacheKey::from_str("hello world");
        let k2 = CacheKey::from_str("hello world");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_inputs() {
        let k1 = CacheKey::from_str("query_a");
        let k2 = CacheKey::from_str("query_b");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_from_bytes() {
        let k = CacheKey::from_bytes(b"test data");
        assert_ne!(k.raw(), 0);
    }

    // ── CacheEntry ──

    #[test]
    fn test_cache_entry_not_expired() {
        let e: CacheEntry<i32> = CacheEntry::new(42, 60);
        assert!(!e.is_expired());
    }

    #[test]
    fn test_cache_entry_zero_ttl_never_expires() {
        let e: CacheEntry<i32> = CacheEntry::new(42, 0);
        assert!(!e.is_expired());
    }

    // ── CacheStats ──

    #[test]
    fn test_cache_stats_hit_rate_zero_when_no_lookups() {
        let s = CacheStats::default();
        assert!((s.hit_rate() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cache_stats_hit_rate_calculation() {
        let s = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
        };
        assert!((s.hit_rate() - 0.75).abs() < f32::EPSILON);
    }

    // ── SearchCache ──

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache: SearchCache<String> = SearchCache::new(10);
        let key = CacheKey::from_str("q1");
        cache.insert(key, "result1".to_string());
        assert_eq!(cache.get(&key), Some(&"result1".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache: SearchCache<String> = SearchCache::new(10);
        let key = CacheKey::from_str("nonexistent");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_stats_after_hit_and_miss() {
        let mut cache: SearchCache<i32> = SearchCache::new(10);
        let k = CacheKey::from_str("k");
        cache.insert(k, 1);
        let _ = cache.get(&k); // hit
        let _ = cache.get(&CacheKey::from_str("nope")); // miss
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache: SearchCache<i32> = SearchCache::new(3);
        let k1 = CacheKey::from_str("k1");
        let k2 = CacheKey::from_str("k2");
        let k3 = CacheKey::from_str("k3");
        let k4 = CacheKey::from_str("k4");
        cache.insert(k1, 1);
        cache.insert(k2, 2);
        cache.insert(k3, 3);
        cache.insert(k4, 4); // should evict k1 (LRU)
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get(&k1).is_none(), "k1 should have been evicted");
        assert!(cache.get(&k4).is_some());
    }

    #[test]
    fn test_cache_lru_order_updated_on_get() {
        let mut cache: SearchCache<i32> = SearchCache::new(3);
        let k1 = CacheKey::from_str("k1");
        let k2 = CacheKey::from_str("k2");
        let k3 = CacheKey::from_str("k3");
        let k4 = CacheKey::from_str("k4");
        cache.insert(k1, 1);
        cache.insert(k2, 2);
        cache.insert(k3, 3);
        // Access k1, making k2 the LRU.
        let _ = cache.get(&k1);
        cache.insert(k4, 4); // should evict k2
        assert!(cache.get(&k2).is_none(), "k2 should be evicted");
        assert!(cache.get(&k1).is_some());
    }

    #[test]
    fn test_cache_update_existing_key() {
        let mut cache: SearchCache<i32> = SearchCache::new(3);
        let k = CacheKey::from_str("k");
        cache.insert(k, 1);
        cache.insert(k, 2); // update
        assert_eq!(cache.get(&k), Some(&2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: SearchCache<i32> = SearchCache::new(10);
        cache.insert(CacheKey::from_str("a"), 1);
        cache.insert(CacheKey::from_str("b"), 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache: SearchCache<i32> = SearchCache::new(10);
        let k = CacheKey::from_str("k");
        cache.insert(k, 42);
        cache.remove(&k);
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn test_cache_capacity() {
        let cache: SearchCache<i32> = SearchCache::new(5);
        assert_eq!(cache.capacity(), 5);
    }
}
