//! Proxy cache management: LRU eviction, TTL-based staleness, and utilisation tracking.
//!
//! Provides a simple in-memory cache that tracks proxy files by path, access
//! time, and hit count.  Eviction strategies include LRU (least-recently used),
//! TTL (time-to-live), and LFU (least-frequently used).

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A single entry in the proxy cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    /// File-system path of the cached proxy.
    pub path: String,
    /// Size of the proxy file in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (milliseconds) of the most recent access.
    pub last_access_ms: u64,
    /// Number of times this entry has been accessed.
    pub hit_count: u32,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub fn new(path: impl Into<String>, size_bytes: u64, now_ms: u64) -> Self {
        Self {
            path: path.into(),
            size_bytes,
            last_access_ms: now_ms,
            hit_count: 1,
        }
    }

    /// Returns `true` when the entry has not been accessed within `ttl_ms` milliseconds.
    ///
    /// # Arguments
    /// * `now_ms` – current time in Unix milliseconds.
    /// * `ttl_ms` – time-to-live threshold in milliseconds.
    #[must_use]
    pub fn is_stale(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_access_ms) > ttl_ms
    }
}

// ---------------------------------------------------------------------------
// CachePolicy
// ---------------------------------------------------------------------------

/// Eviction policy for the proxy cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Evict the least-recently-used entry.
    Lru,
    /// Evict entries that have exceeded their time-to-live.
    Ttl,
    /// Evict the least-frequently-used entry.
    Lfu,
}

impl CachePolicy {
    /// Human-readable description of the policy.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Lru => "Least Recently Used (LRU): evict the entry not accessed longest",
            Self::Ttl => "Time To Live (TTL): evict entries older than a configured threshold",
            Self::Lfu => "Least Frequently Used (LFU): evict the entry with the fewest accesses",
        }
    }
}

// ---------------------------------------------------------------------------
// ProxyCache
// ---------------------------------------------------------------------------

/// In-memory proxy cache with configurable capacity.
#[derive(Debug)]
pub struct ProxyCache {
    /// All cached entries.
    pub entries: Vec<CacheEntry>,
    /// Maximum allowed total size of the cache in bytes.
    pub max_size_bytes: u64,
    /// Current total size of all entries in bytes.
    pub used_bytes: u64,
}

impl ProxyCache {
    /// Create a new, empty cache with the given maximum size.
    #[must_use]
    pub fn new(max_size_bytes: u64) -> Self {
        Self {
            entries: Vec::new(),
            max_size_bytes,
            used_bytes: 0,
        }
    }

    /// Add a new entry to the cache.
    ///
    /// If an entry with the same path already exists it is replaced and the
    /// used-byte count is updated accordingly.
    pub fn add(&mut self, path: &str, size: u64, now_ms: u64) {
        // Remove existing entry with the same path to avoid duplicates.
        if let Some(pos) = self.entries.iter().position(|e| e.path == path) {
            let old_size = self.entries[pos].size_bytes;
            self.entries.remove(pos);
            self.used_bytes = self.used_bytes.saturating_sub(old_size);
        }
        self.entries.push(CacheEntry::new(path, size, now_ms));
        self.used_bytes += size;
    }

    /// Update the access timestamp and hit count for an entry.
    ///
    /// Returns `true` if the entry was found and updated.
    pub fn touch(&mut self, path: &str, now_ms: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.path == path) {
            entry.last_access_ms = now_ms;
            entry.hit_count = entry.hit_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Remove the least-recently-used entry and return its path.
    ///
    /// Returns `None` if the cache is empty.
    pub fn evict_lru(&mut self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        // Find the index of the entry with the smallest `last_access_ms`.
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_access_ms)
            .map(|(i, _)| i)?;

        let removed = self.entries.remove(idx);
        self.used_bytes = self.used_bytes.saturating_sub(removed.size_bytes);
        Some(removed.path)
    }

    /// Remove all entries that are stale (last access older than `ttl_ms`).
    ///
    /// Returns the paths of all evicted entries.
    pub fn evict_stale(&mut self, now_ms: u64, ttl_ms: u64) -> Vec<String> {
        let mut evicted = Vec::new();
        self.entries.retain(|e| {
            if e.is_stale(now_ms, ttl_ms) {
                evicted.push(e.path.clone());
                false
            } else {
                true
            }
        });
        // Update used_bytes: subtract sizes of evicted entries.
        // We already removed them, so recalculate from remaining entries.
        self.used_bytes = self.entries.iter().map(|e| e.size_bytes).sum();
        evicted
    }

    /// Cache utilisation as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when `max_size_bytes` is 0.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_size_bytes == 0 {
            return 0.0;
        }
        (self.used_bytes as f64 / self.max_size_bytes as f64).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_is_stale_fresh() {
        let entry = CacheEntry::new("p.mp4", 1024, 1_000);
        // TTL 5 000 ms, now 2 000 ms → only 1 000 ms old → not stale
        assert!(!entry.is_stale(2_000, 5_000));
    }

    #[test]
    fn test_cache_entry_is_stale_expired() {
        let entry = CacheEntry::new("p.mp4", 1024, 0);
        // TTL 500 ms, now 1 000 ms → 1 000 ms old → stale
        assert!(entry.is_stale(1_000, 500));
    }

    #[test]
    fn test_cache_entry_is_stale_exactly_at_ttl() {
        let entry = CacheEntry::new("p.mp4", 1024, 0);
        // Exactly at TTL boundary → not stale (> rather than >=)
        assert!(!entry.is_stale(500, 500));
    }

    #[test]
    fn test_cache_policy_descriptions_non_empty() {
        assert!(!CachePolicy::Lru.description().is_empty());
        assert!(!CachePolicy::Ttl.description().is_empty());
        assert!(!CachePolicy::Lfu.description().is_empty());
    }

    #[test]
    fn test_proxy_cache_add_single() {
        let mut cache = ProxyCache::new(1_000_000);
        cache.add("a.mp4", 100, 1_000);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.used_bytes, 100);
    }

    #[test]
    fn test_proxy_cache_add_replaces_existing() {
        let mut cache = ProxyCache::new(1_000_000);
        cache.add("a.mp4", 100, 1_000);
        cache.add("a.mp4", 200, 2_000);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.used_bytes, 200);
    }

    #[test]
    fn test_proxy_cache_touch_updates_access() {
        let mut cache = ProxyCache::new(1_000_000);
        cache.add("a.mp4", 100, 1_000);
        let updated = cache.touch("a.mp4", 5_000);
        assert!(updated);
        assert_eq!(cache.entries[0].last_access_ms, 5_000);
        assert_eq!(cache.entries[0].hit_count, 2);
    }

    #[test]
    fn test_proxy_cache_touch_missing_returns_false() {
        let mut cache = ProxyCache::new(1_000_000);
        assert!(!cache.touch("missing.mp4", 1_000));
    }

    #[test]
    fn test_proxy_cache_evict_lru_empty() {
        let mut cache = ProxyCache::new(1_000_000);
        assert!(cache.evict_lru().is_none());
    }

    #[test]
    fn test_proxy_cache_evict_lru_removes_oldest() {
        let mut cache = ProxyCache::new(1_000_000);
        cache.add("old.mp4", 100, 1_000);
        cache.add("new.mp4", 100, 9_000);
        let evicted = cache.evict_lru();
        assert_eq!(evicted, Some("old.mp4".to_string()));
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_proxy_cache_evict_stale() {
        let mut cache = ProxyCache::new(1_000_000);
        cache.add("stale.mp4", 100, 0);
        cache.add("fresh.mp4", 200, 9_000);
        let evicted = cache.evict_stale(10_000, 5_000);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], "stale.mp4");
        assert_eq!(cache.used_bytes, 200);
    }

    #[test]
    fn test_proxy_cache_utilization() {
        let mut cache = ProxyCache::new(1_000);
        cache.add("a.mp4", 500, 0);
        let u = cache.utilization();
        assert!((u - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_proxy_cache_utilization_zero_max() {
        let cache = ProxyCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }
}

// ============================================================================
// Disk-Bounded LRU Cache Manager
// ============================================================================

/// Statistics snapshot for a `DiskBoundedCache`.
#[derive(Debug, Clone)]
pub struct DiskCacheStats {
    /// Number of entries currently in the cache.
    pub entry_count: usize,
    /// Total bytes occupied by cached proxies.
    pub used_bytes: u64,
    /// Maximum allowed bytes.
    pub max_bytes: u64,
    /// Cache utilisation fraction in `[0.0, 1.0]`.
    pub utilization: f64,
    /// Total number of LRU evictions performed since construction.
    pub eviction_count: u64,
    /// Total number of successful insertions.
    pub insertion_count: u64,
    /// Total number of cache hits (access to an existing entry).
    pub hit_count: u64,
    /// Total number of cache misses (lookup that found nothing).
    pub miss_count: u64,
}

/// A disk-space-bounded LRU proxy cache that automatically evicts entries
/// when the configured space limit would be exceeded.
///
/// Unlike the simpler `ProxyCache`, `DiskBoundedCache`:
/// * Enforces a hard disk-space ceiling on insertions.
/// * Uses LRU eviction automatically upon every insertion that would violate the limit.
/// * Maintains hit/miss/eviction counters for telemetry.
/// * Supports lookup with automatic LRU touch.
#[derive(Debug)]
pub struct DiskBoundedCache {
    /// Inner ordered list of entries; oldest-accessed first (front = LRU end).
    entries: std::collections::VecDeque<CacheEntry>,
    /// Maximum allowed total size in bytes.
    max_bytes: u64,
    /// Current total size in bytes.
    used_bytes: u64,
    /// Cumulative eviction count.
    eviction_count: u64,
    /// Cumulative insertion count.
    insertion_count: u64,
    /// Cumulative hit count.
    hit_count: u64,
    /// Cumulative miss count.
    miss_count: u64,
}

impl DiskBoundedCache {
    /// Create a new cache with a disk-space limit of `max_bytes`.
    ///
    /// # Errors
    ///
    /// Returns a descriptive string if `max_bytes` is zero.
    pub fn new(max_bytes: u64) -> Result<Self, String> {
        if max_bytes == 0 {
            return Err("DiskBoundedCache: max_bytes must be > 0".to_string());
        }
        Ok(Self {
            entries: std::collections::VecDeque::new(),
            max_bytes,
            used_bytes: 0,
            eviction_count: 0,
            insertion_count: 0,
            hit_count: 0,
            miss_count: 0,
        })
    }

    /// Insert a proxy entry identified by `path` with `size_bytes` and access time `now_ms`.
    ///
    /// If an entry with the same path already exists it is updated in-place (moved to the
    /// MRU end).  Before inserting, LRU entries are evicted until the new entry fits within
    /// the space limit.  If the single entry itself is larger than the limit, the insert
    /// is rejected and `false` is returned.
    ///
    /// Returns `true` when the entry was inserted/updated, `false` when it was rejected.
    pub fn insert(&mut self, path: &str, size_bytes: u64, now_ms: u64) -> bool {
        if size_bytes > self.max_bytes {
            return false;
        }

        // If the path already exists, remove the old entry first
        if let Some(pos) = self.entries.iter().position(|e| e.path == path) {
            let old_size = self.entries[pos].size_bytes;
            self.entries.remove(pos);
            self.used_bytes = self.used_bytes.saturating_sub(old_size);
        }

        // Evict LRU entries until there is room for the new entry
        while !self.entries.is_empty() && self.used_bytes + size_bytes > self.max_bytes {
            // Front of VecDeque is the LRU end
            if let Some(evicted) = self.entries.pop_front() {
                self.used_bytes = self.used_bytes.saturating_sub(evicted.size_bytes);
                self.eviction_count += 1;
            }
        }

        // Push the new entry to the MRU end (back)
        self.entries
            .push_back(CacheEntry::new(path, size_bytes, now_ms));
        self.used_bytes += size_bytes;
        self.insertion_count += 1;
        true
    }

    /// Look up a cache entry by path.
    ///
    /// On hit: moves the entry to the MRU position and updates access time.
    /// On miss: increments the miss counter.
    ///
    /// Returns a clone of the entry on hit, or `None` on miss.
    pub fn access(&mut self, path: &str, now_ms: u64) -> Option<CacheEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.path == path) {
            // Move to MRU end
            if let Some(mut entry) = self.entries.remove(pos) {
                entry.last_access_ms = now_ms;
                entry.hit_count = entry.hit_count.saturating_add(1);
                let clone = entry.clone();
                self.entries.push_back(entry);
                self.hit_count += 1;
                Some(clone)
            } else {
                // Position was found but remove returned None — should not happen
                self.miss_count += 1;
                None
            }
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Remove and return the path of the least-recently-used entry.
    ///
    /// Returns `None` if the cache is empty.
    pub fn evict_lru(&mut self) -> Option<String> {
        self.entries.pop_front().map(|e| {
            self.used_bytes = self.used_bytes.saturating_sub(e.size_bytes);
            self.eviction_count += 1;
            e.path
        })
    }

    /// Remove all entries, returning their paths.
    pub fn clear(&mut self) -> Vec<String> {
        let paths: Vec<String> = self.entries.iter().map(|e| e.path.clone()).collect();
        self.entries.clear();
        self.used_bytes = 0;
        self.eviction_count += paths.len() as u64;
        paths
    }

    /// Whether the cache contains an entry for `path`.
    pub fn contains(&self, path: &str) -> bool {
        self.entries.iter().any(|e| e.path == path)
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current total used bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Maximum allowed bytes.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Cache utilisation fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        (self.used_bytes as f64 / self.max_bytes as f64).min(1.0)
    }

    /// Collect and return a statistics snapshot.
    pub fn stats(&self) -> DiskCacheStats {
        DiskCacheStats {
            entry_count: self.entries.len(),
            used_bytes: self.used_bytes,
            max_bytes: self.max_bytes,
            utilization: self.utilization(),
            eviction_count: self.eviction_count,
            insertion_count: self.insertion_count,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
        }
    }
}

#[cfg(test)]
mod disk_bounded_tests {
    use super::*;

    fn make_cache(max_bytes: u64) -> DiskBoundedCache {
        DiskBoundedCache::new(max_bytes).expect("valid max_bytes")
    }

    #[test]
    fn test_new_rejects_zero_max() {
        assert!(DiskBoundedCache::new(0).is_err());
    }

    #[test]
    fn test_insert_single_entry() {
        let mut cache = make_cache(1_000);
        assert!(cache.insert("a.mp4", 100, 1_000));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.used_bytes(), 100);
    }

    #[test]
    fn test_insert_over_limit_rejected() {
        let mut cache = make_cache(500);
        assert!(!cache.insert("huge.mp4", 1_000, 1_000));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_eviction_on_insert() {
        let mut cache = make_cache(200);
        cache.insert("a.mp4", 100, 1_000);
        cache.insert("b.mp4", 100, 2_000);
        // Both fit; total = 200
        assert_eq!(cache.len(), 2);
        // Insert a third entry that requires eviction of LRU ("a.mp4")
        cache.insert("c.mp4", 100, 3_000);
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains("a.mp4"), "a.mp4 should have been evicted");
        assert!(cache.contains("b.mp4"));
        assert!(cache.contains("c.mp4"));
        assert_eq!(cache.stats().eviction_count, 1);
    }

    #[test]
    fn test_access_hit_moves_to_mru() {
        // max_bytes=200 so two 100-byte entries fill the cache completely.
        // Inserting a third entry must evict the LRU one.
        let mut cache = make_cache(200);
        cache.insert("a.mp4", 100, 1_000);
        cache.insert("b.mp4", 100, 2_000);
        // Access "a.mp4" → it becomes MRU; "b.mp4" becomes LRU
        let entry = cache.access("a.mp4", 5_000).expect("hit expected");
        assert_eq!(entry.path, "a.mp4");

        // Insert "c.mp4" → should evict "b.mp4" (now LRU), not "a.mp4"
        cache.insert("c.mp4", 100, 6_000);
        assert!(!cache.contains("b.mp4"), "b.mp4 should be evicted");
        assert!(cache.contains("a.mp4"), "a.mp4 is MRU, should survive");
    }

    #[test]
    fn test_access_miss_increments_counter() {
        let mut cache = make_cache(1_000);
        let result = cache.access("nonexistent.mp4", 1_000);
        assert!(result.is_none());
        assert_eq!(cache.stats().miss_count, 1);
    }

    #[test]
    fn test_hit_count_increments() {
        let mut cache = make_cache(1_000);
        cache.insert("a.mp4", 100, 1_000);
        cache.access("a.mp4", 2_000);
        cache.access("a.mp4", 3_000);
        assert_eq!(cache.stats().hit_count, 2);
    }

    #[test]
    fn test_update_existing_entry() {
        let mut cache = make_cache(500);
        cache.insert("a.mp4", 200, 1_000);
        // Update with new size
        cache.insert("a.mp4", 300, 2_000);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.used_bytes(), 300);
    }

    #[test]
    fn test_evict_lru_explicit() {
        let mut cache = make_cache(500);
        cache.insert("old.mp4", 100, 1_000);
        cache.insert("new.mp4", 100, 9_000);
        let evicted = cache.evict_lru();
        assert_eq!(evicted, Some("old.mp4".to_string()));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.used_bytes(), 100);
    }

    #[test]
    fn test_evict_lru_empty_returns_none() {
        let mut cache = make_cache(500);
        assert!(cache.evict_lru().is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = make_cache(1_000);
        cache.insert("a.mp4", 100, 1_000);
        cache.insert("b.mp4", 200, 2_000);
        let cleared = cache.clear();
        assert_eq!(cleared.len(), 2);
        assert!(cache.is_empty());
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn test_utilization() {
        let mut cache = make_cache(1_000);
        cache.insert("a.mp4", 500, 1_000);
        assert!((cache.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_fields() {
        let mut cache = make_cache(1_000);
        cache.insert("a.mp4", 100, 1_000);
        cache.insert("b.mp4", 100, 2_000);
        cache.access("a.mp4", 3_000);
        cache.access("missing.mp4", 4_000);
        let s = cache.stats();
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.insertion_count, 2);
        assert_eq!(s.hit_count, 1);
        assert_eq!(s.miss_count, 1);
    }

    #[test]
    fn test_rapid_create_evict_cycles() {
        // Stress: 1000 inserts into a small cache (holds ≤5 entries)
        let mut cache = make_cache(500);
        let mut total_evictions = 0u64;
        for i in 0..1_000u64 {
            let path = format!("proxy_{i}.mp4");
            cache.insert(&path, 100, i * 10);
            total_evictions = cache.stats().eviction_count;
        }
        // Should have evicted many entries but remain within limits
        assert!(cache.used_bytes() <= 500);
        assert!(
            total_evictions > 900,
            "expected many evictions, got {total_evictions}"
        );
    }

    #[test]
    fn test_multiple_evictions_per_insert() {
        // Insert 10 small entries, then one large one that forces many evictions
        let mut cache = make_cache(1_000);
        for i in 0..10u64 {
            cache.insert(&format!("{i}.mp4"), 100, i);
        }
        assert_eq!(cache.len(), 10);
        // Insert one 600-byte entry: must evict 6 old ones to make room
        cache.insert("big.mp4", 600, 100);
        assert!(cache.used_bytes() <= 1_000);
        assert!(cache.contains("big.mp4"));
    }
}
