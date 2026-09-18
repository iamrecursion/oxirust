//! LRU cache for accelerated processing results.
//!
//! `AccelCache` is a fixed-capacity byte-budget cache that evicts the
//! least-recently-used entry when capacity is exceeded.  It tracks hit/miss
//! statistics and per-entry access counts.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single entry stored in the cache.
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The cached value.
    pub data: T,
    /// Approximate memory footprint in bytes.
    pub size_bytes: usize,
    /// Monotonic access timestamp (incremented counter, not wall clock).
    pub last_access: u64,
    /// Number of times this entry has been retrieved.
    pub hit_count: u32,
}

/// Aggregate cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total successful lookups.
    pub hits: u64,
    /// Total unsuccessful lookups.
    pub misses: u64,
    /// Total evictions performed.
    pub evictions: u64,
}

/// LRU cache backed by a `HashMap` with a byte-budget capacity limit.
pub struct AccelCache<T> {
    /// Maximum total byte usage.
    capacity_bytes: usize,
    /// Cached entries keyed by an opaque `u64` identifier.
    entries: HashMap<u64, CacheEntry<T>>,
    /// Monotonic clock counter used as a logical timestamp.
    clock: u64,
    /// Running statistics.
    stats: CacheStats,
    /// Current total byte usage.
    current_usage: usize,
}

impl<T: Clone> AccelCache<T> {
    /// Create a new cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            entries: HashMap::new(),
            clock: 0,
            stats: CacheStats::default(),
            current_usage: 0,
        }
    }

    /// Look up a value by key.
    ///
    /// Updates `last_access` and `hit_count` on a successful lookup, and
    /// increments [`CacheStats::hits`] or [`CacheStats::misses`] accordingly.
    pub fn get(&mut self, key: u64) -> Option<&T> {
        if !self.entries.contains_key(&key) {
            self.stats.misses += 1;
            return None;
        }
        self.clock += 1;
        let clock = self.clock;
        let entry = self
            .entries
            .get_mut(&key)
            .unwrap_or_else(|| unreachable!("key confirmed present via contains_key"));
        entry.last_access = clock;
        entry.hit_count += 1;
        self.stats.hits += 1;
        // Now re-borrow immutably; the mutable borrow above has ended.
        self.entries.get(&key).map(|e| &e.data)
    }

    /// Insert or replace an entry.
    ///
    /// If inserting would exceed the byte capacity, the least-recently-used
    /// entry is evicted first (potentially multiple times).
    pub fn insert(&mut self, key: u64, data: T, size: usize) {
        // Remove an existing entry with the same key so we don't double-count.
        if let Some(old) = self.entries.remove(&key) {
            self.current_usage = self.current_usage.saturating_sub(old.size_bytes);
        }

        // Evict until we have room (or the cache is empty).
        while self.current_usage + size > self.capacity_bytes && !self.entries.is_empty() {
            self.evict_lru();
        }

        self.clock += 1;
        self.current_usage += size;
        self.entries.insert(
            key,
            CacheEntry {
                data,
                size_bytes: size,
                last_access: self.clock,
                hit_count: 0,
            },
        );
    }

    /// Evict the least-recently-used entry.
    ///
    /// Does nothing if the cache is empty.
    pub fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(&k, _)| k)
            .unwrap_or_else(|| unreachable!("entries is non-empty (checked above)"));

        if let Some(evicted) = self.entries.remove(&lru_key) {
            self.current_usage = self.current_usage.saturating_sub(evicted.size_bytes);
            self.stats.evictions += 1;
        }
    }

    /// Return the cache hit rate as a value in [0.0, 1.0].
    ///
    /// Returns 0.0 if no lookups have been performed.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }

    /// Return the current total byte usage.
    #[must_use]
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Return the byte capacity.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Return the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_usage = 0;
    }

    /// Return `true` if the cache contains an entry for `key` (without
    /// updating access statistics).
    #[must_use]
    pub fn contains_key(&self, key: u64) -> bool {
        self.entries.contains_key(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_miss_on_empty() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        assert!(cache.get(42).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_insert_and_hit() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(1, 99u32, 4);
        let v = cache.get(1);
        assert_eq!(v, Some(&99u32));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_hit_rate_pure_hits() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(1, 1u32, 4);
        cache.get(1);
        cache.get(1);
        assert!((cache.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_hit_rate_zero_lookups() {
        let cache: AccelCache<u32> = AccelCache::new(1024);
        assert!((cache.hit_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_current_usage_tracking() {
        let mut cache: AccelCache<Vec<u8>> = AccelCache::new(1024);
        cache.insert(1, vec![0u8; 100], 100);
        cache.insert(2, vec![0u8; 200], 200);
        assert_eq!(cache.current_usage(), 300);
    }

    #[test]
    fn test_evict_lru_removes_oldest() {
        let mut cache: AccelCache<u32> = AccelCache::new(8);
        // Insert two 4-byte entries that exactly fill the cache.
        cache.insert(1, 10u32, 4);
        cache.insert(2, 20u32, 4);
        // Access key 1 to make key 2 the LRU.
        cache.get(1);
        // Insert a third entry; key 2 should be evicted.
        cache.insert(3, 30u32, 4);
        assert!(!cache.contains_key(2));
        assert!(cache.contains_key(1));
        assert!(cache.contains_key(3));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_evict_lru_on_empty_does_nothing() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.evict_lru(); // must not panic
        assert_eq!(cache.stats().evictions, 0);
    }

    #[test]
    fn test_replace_existing_key() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(1, 10u32, 4);
        cache.insert(1, 20u32, 4); // replace
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.current_usage(), 4);
        assert_eq!(cache.get(1), Some(&20u32));
    }

    #[test]
    fn test_clear_resets_usage() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(1, 1u32, 100);
        cache.clear();
        assert_eq!(cache.current_usage(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        assert!(cache.is_empty());
        cache.insert(1, 1u32, 1);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_hit_count_increments() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(7, 777u32, 4);
        cache.get(7);
        cache.get(7);
        let entry = &cache.entries[&7];
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    fn test_mixed_hit_rate() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(1, 10u32, 4);
        cache.get(1); // hit
        cache.get(2); // miss
        cache.get(1); // hit
        cache.get(3); // miss
                      // 2 hits, 2 misses → 0.5
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_snapshot_reflects_current_state() {
        let mut cache: AccelCache<u32> = AccelCache::new(1024);
        cache.insert(10, 100u32, 4);
        cache.get(10);
        cache.get(99);
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.evictions, 0);
    }
}
