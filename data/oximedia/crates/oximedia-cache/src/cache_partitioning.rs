//! Cache partitioning: isolate cache space per tenant, stream, or workload.
//!
//! [`PartitionedCache`] manages a collection of named [`CachePartition`]s,
//! each with an independent LRU policy and byte-level capacity budget.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A single cached value together with its metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Raw value bytes.
    pub data: Vec<u8>,
    /// Size in bytes; if `0` the `data.len()` is used.
    pub size_bytes: usize,
    /// Optional expiry wall-clock instant.
    pub expires_at: Option<Instant>,
    /// Logical priority tag (higher = more important to keep).
    pub priority: u32,
    /// Wall-clock time at which this entry was last read.
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
}

impl CacheEntry {
    /// Create a new `CacheEntry` with default metadata.
    pub fn new(data: Vec<u8>) -> Self {
        let size_bytes = data.len();
        Self {
            data,
            size_bytes,
            expires_at: None,
            priority: 0,
            last_accessed: Instant::now(),
            access_count: 0,
        }
    }

    /// Create a `CacheEntry` with an explicit byte-size hint, TTL, and priority.
    pub fn with_meta(
        data: Vec<u8>,
        size_bytes: usize,
        ttl: Option<Duration>,
        priority: u32,
    ) -> Self {
        let effective_size = if size_bytes == 0 {
            data.len()
        } else {
            size_bytes
        };
        Self {
            data,
            size_bytes: effective_size,
            expires_at: ttl.map(|d| Instant::now() + d),
            priority,
            last_accessed: Instant::now(),
            access_count: 0,
        }
    }

    /// Return `true` if this entry has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Instant::now() >= exp)
            .unwrap_or(false)
    }
}

// ── CachePartition ────────────────────────────────────────────────────────────

/// An isolated cache partition with its own byte-level capacity.
///
/// Internally uses an insertion-ordered key list (`Vec<String>`) for LRU
/// tracking, and a `HashMap` for O(1) value access.  Eviction walks from
/// the tail (oldest) towards the head (newest), skipping high-priority items.
pub struct CachePartition {
    /// Human-readable name for this partition.
    pub name: String,
    /// Maximum byte budget for this partition.
    pub max_bytes: usize,
    /// Key→entry store.
    entries: HashMap<String, CacheEntry>,
    /// LRU ordering: front = MRU, back = LRU.
    lru_order: Vec<String>,
    /// Currently used bytes.
    used_bytes: usize,
    /// Partition-level hit counter.
    hits: u64,
    /// Partition-level miss counter.
    misses: u64,
    /// Partition-level eviction counter.
    evictions: u64,
}

impl CachePartition {
    /// Create a new partition with the given name and byte capacity.
    pub fn new(name: impl Into<String>, max_bytes: usize) -> Self {
        Self {
            name: name.into(),
            max_bytes,
            entries: HashMap::new(),
            lru_order: Vec::new(),
            used_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Return the number of entries in this partition.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the partition is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return currently used bytes.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Return partition statistics.
    pub fn stats(&self) -> PartitionStats {
        PartitionStats {
            name: self.name.clone(),
            entry_count: self.entries.len(),
            used_bytes: self.used_bytes,
            max_bytes: self.max_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    /// Retrieve the entry for `key`.
    ///
    /// Returns `None` if the key is absent or its TTL has expired (in which
    /// case the entry is lazily removed).
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        // Lazy TTL eviction.
        let expired = self
            .entries
            .get(key)
            .map(|e| e.is_expired())
            .unwrap_or(false);
        if expired {
            self.remove_entry(key);
            self.misses += 1;
            return None;
        }

        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            self.hits += 1;
            // Promote to MRU head.
            self.lru_order.retain(|k| k != key);
            self.lru_order.insert(0, key.to_string());
            // SAFETY: we just confirmed the key exists.
            self.entries.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Peek at an entry without updating LRU order or access statistics.
    pub fn peek(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    /// Insert or update `(key, entry)` in this partition.
    ///
    /// If the entry does not fit even after evicting all lower-priority
    /// entries the insert is silently dropped.
    pub fn put(&mut self, key: String, entry: CacheEntry) {
        // Remove existing entry if present so we can replace it.
        if self.entries.contains_key(&key) {
            self.remove_entry(&key);
        }

        // Evict until there is enough room.
        while self.used_bytes + entry.size_bytes > self.max_bytes {
            if self.evict_one_lru().is_none() {
                break;
            }
        }

        // If still no room, drop the insert.
        if self.used_bytes + entry.size_bytes > self.max_bytes {
            return;
        }

        self.used_bytes += entry.size_bytes;
        self.lru_order.insert(0, key.clone());
        self.entries.insert(key, entry);
    }

    /// Remove `key` from this partition. Returns `true` if it was present.
    pub fn remove(&mut self, key: &str) -> bool {
        self.remove_entry(key)
    }

    /// Evict the least-recently-used entry.
    ///
    /// Returns the evicted key or `None` if the partition is empty.
    pub fn evict_one_lru(&mut self) -> Option<String> {
        // Walk from LRU end to find an evictable entry.
        let victim = self
            .lru_order
            .iter()
            .rev()
            .find(|k| {
                // Skip high-priority entries (priority > 0) as a simple heuristic.
                self.entries
                    .get(*k)
                    .map(|e| e.priority == 0)
                    .unwrap_or(false)
            })
            .cloned()
            .or_else(|| {
                // If all entries have priority > 0, fall back to true LRU.
                self.lru_order.last().cloned()
            });

        if let Some(key) = victim {
            self.remove_entry(&key);
            self.evictions += 1;
            Some(key)
        } else {
            None
        }
    }

    /// Evict entries from this partition until `bytes_to_free` bytes have been
    /// freed or the partition is empty.
    ///
    /// Returns the number of bytes actually freed.
    pub fn evict_bytes(&mut self, bytes_to_free: usize) -> usize {
        let start_used = self.used_bytes;
        while self.used_bytes + bytes_to_free > start_used {
            if self.used_bytes == 0 {
                break;
            }
            // Check if we've freed enough.
            if start_used.saturating_sub(self.used_bytes) >= bytes_to_free {
                break;
            }
            if self.evict_one_lru().is_none() {
                break;
            }
        }
        start_used.saturating_sub(self.used_bytes)
    }

    /// Purge all expired entries. Returns the count of entries removed.
    pub fn purge_expired(&mut self) -> usize {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for key in expired {
            self.remove_entry(&key);
        }
        count
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn remove_entry(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
            self.lru_order.retain(|k| k != key);
            true
        } else {
            false
        }
    }
}

// ── PartitionStats ────────────────────────────────────────────────────────────

/// Snapshot of per-partition statistics.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Partition name.
    pub name: String,
    /// Number of entries.
    pub entry_count: usize,
    /// Used bytes.
    pub used_bytes: usize,
    /// Maximum allowed bytes.
    pub max_bytes: usize,
    /// Cumulative hit count.
    pub hits: u64,
    /// Cumulative miss count.
    pub misses: u64,
    /// Cumulative eviction count.
    pub evictions: u64,
}

impl PartitionStats {
    /// Return the utilisation ratio `used_bytes / max_bytes`, clamped to
    /// `[0.0, 1.0]`.
    pub fn utilisation(&self) -> f64 {
        if self.max_bytes == 0 {
            return 1.0;
        }
        (self.used_bytes as f64 / self.max_bytes as f64).min(1.0)
    }
}

// ── PartitionedCache ──────────────────────────────────────────────────────────

/// A cache composed of multiple named [`CachePartition`]s.
///
/// Each partition is isolated: operations on one partition never evict entries
/// from another.  A configurable `default_partition` is used when callers
/// omit the partition name.
pub struct PartitionedCache {
    partitions: HashMap<String, CachePartition>,
    /// Name of the default partition used by `*_default` methods.
    pub default_partition: String,
}

impl PartitionedCache {
    /// Create a new `PartitionedCache` with a single default partition.
    pub fn new(default_partition: impl Into<String>, default_capacity_bytes: usize) -> Self {
        let name = default_partition.into();
        let mut partitions = HashMap::new();
        partitions.insert(
            name.clone(),
            CachePartition::new(name.clone(), default_capacity_bytes),
        );
        Self {
            partitions,
            default_partition: name,
        }
    }

    /// Add a new named partition.  If a partition with the same name already
    /// exists it is replaced.
    pub fn add_partition(&mut self, name: impl Into<String>, max_bytes: usize) {
        let n = name.into();
        self.partitions
            .insert(n.clone(), CachePartition::new(n, max_bytes));
    }

    /// Remove a partition by name.  Returns `true` if it existed.
    ///
    /// The default partition cannot be removed.
    pub fn remove_partition(&mut self, name: &str) -> bool {
        if name == self.default_partition {
            return false;
        }
        self.partitions.remove(name).is_some()
    }

    /// Return `true` if a partition with `name` exists.
    pub fn has_partition(&self, name: &str) -> bool {
        self.partitions.contains_key(name)
    }

    /// Return a list of all partition names.
    pub fn partition_names(&self) -> Vec<String> {
        self.partitions.keys().cloned().collect()
    }

    /// Get the entry for `key` from `partition`.
    ///
    /// Returns `None` if the partition does not exist, the key is absent, or
    /// the entry has expired.
    pub fn get(&mut self, partition: &str, key: &str) -> Option<&CacheEntry> {
        self.partitions.get_mut(partition)?.get(key)
    }

    /// Insert `(key, entry)` into `partition`.
    ///
    /// If `partition` does not exist the call is silently dropped.
    pub fn put(&mut self, partition: &str, key: String, entry: CacheEntry) {
        if let Some(p) = self.partitions.get_mut(partition) {
            p.put(key, entry);
        }
    }

    /// Remove `key` from `partition`.  Returns `true` if it was found.
    pub fn remove(&mut self, partition: &str, key: &str) -> bool {
        self.partitions
            .get_mut(partition)
            .map(|p| p.remove(key))
            .unwrap_or(false)
    }

    /// Evict `bytes` of data from `partition`.
    ///
    /// Returns the number of bytes freed.
    pub fn evict_from(&mut self, partition: &str, bytes: usize) -> usize {
        self.partitions
            .get_mut(partition)
            .map(|p| p.evict_bytes(bytes))
            .unwrap_or(0)
    }

    /// Return partition statistics for `partition`, or `None` if it does not
    /// exist.
    pub fn partition_stats(&self, partition: &str) -> Option<PartitionStats> {
        self.partitions.get(partition).map(|p| p.stats())
    }

    /// Return statistics for all partitions.
    pub fn all_stats(&self) -> Vec<PartitionStats> {
        self.partitions.values().map(|p| p.stats()).collect()
    }

    /// Total number of entries across all partitions.
    pub fn total_entries(&self) -> usize {
        self.partitions.values().map(|p| p.len()).sum()
    }

    /// Total bytes used across all partitions.
    pub fn total_used_bytes(&self) -> usize {
        self.partitions.values().map(|p| p.used_bytes()).sum()
    }

    /// Purge expired entries from all partitions. Returns total count removed.
    pub fn purge_all_expired(&mut self) -> usize {
        self.partitions
            .values_mut()
            .map(|p| p.purge_expired())
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(default_cap: usize) -> PartitionedCache {
        PartitionedCache::new("default", default_cap)
    }

    fn entry(data: &[u8]) -> CacheEntry {
        CacheEntry::new(data.to_vec())
    }

    fn entry_sized(data: &[u8], size: usize) -> CacheEntry {
        CacheEntry::with_meta(data.to_vec(), size, None, 0)
    }

    // 1. Basic put and get on default partition
    #[test]
    fn test_put_get_default() {
        let mut cache = make_cache(1024);
        cache.put("default", "key1".to_string(), entry(b"hello"));
        let e = cache.get("default", "key1").expect("entry should be found");
        assert_eq!(e.data, b"hello");
    }

    // 2. Miss on absent key returns None
    #[test]
    fn test_get_absent() {
        let mut cache = make_cache(1024);
        assert!(cache.get("default", "missing").is_none());
    }

    // 3. Put on non-existent partition is silently dropped
    #[test]
    fn test_put_nonexistent_partition_ignored() {
        let mut cache = make_cache(1024);
        cache.put("ghost", "k".to_string(), entry(b"v"));
        assert!(cache.get("ghost", "k").is_none());
    }

    // 4. add_partition creates isolated namespace
    #[test]
    fn test_add_partition_isolation() {
        let mut cache = make_cache(1024);
        cache.add_partition("tenant_a", 512);
        cache.add_partition("tenant_b", 512);
        cache.put("tenant_a", "shared".to_string(), entry(b"a-data"));
        cache.put("tenant_b", "shared".to_string(), entry(b"b-data"));
        let a = cache.get("tenant_a", "shared").expect("tenant_a entry");
        assert_eq!(a.data, b"a-data");
        let b = cache.get("tenant_b", "shared").expect("tenant_b entry");
        assert_eq!(b.data, b"b-data");
    }

    // 5. evict_from frees space in the target partition
    #[test]
    fn test_evict_from() {
        let mut cache = make_cache(10_000);
        cache.add_partition("stream", 1000);
        for i in 0..5u8 {
            cache.put("stream", format!("seg-{i}"), entry_sized(&[i; 100], 100));
        }
        let freed = cache.evict_from("stream", 150);
        assert!(freed > 0, "some bytes should be freed");
    }

    // 6. remove_partition removes it from the cache
    #[test]
    fn test_remove_partition() {
        let mut cache = make_cache(1024);
        cache.add_partition("temp", 256);
        assert!(cache.has_partition("temp"));
        let removed = cache.remove_partition("temp");
        assert!(removed);
        assert!(!cache.has_partition("temp"));
    }

    // 7. Default partition cannot be removed
    #[test]
    fn test_cannot_remove_default_partition() {
        let mut cache = make_cache(1024);
        let removed = cache.remove_partition("default");
        assert!(!removed);
        assert!(cache.has_partition("default"));
    }

    // 8. Partition stats track entries and bytes correctly
    #[test]
    fn test_partition_stats() {
        let mut cache = make_cache(10_000);
        cache.add_partition("analytics", 5000);
        for i in 0u8..10 {
            cache.put("analytics", format!("e-{i}"), entry_sized(&[i; 50], 50));
        }
        let stats = cache
            .partition_stats("analytics")
            .expect("stats should exist");
        assert_eq!(stats.entry_count, 10);
        assert_eq!(stats.used_bytes, 500);
    }

    // 9. LRU eviction respects partition boundary
    #[test]
    fn test_lru_eviction_within_partition() {
        let mut cache = make_cache(10_000);
        cache.add_partition("p1", 300); // fits 3 × 100-byte entries
        cache.put("p1", "a".to_string(), entry_sized(b"A", 100));
        cache.put("p1", "b".to_string(), entry_sized(b"B", 100));
        cache.put("p1", "c".to_string(), entry_sized(b"C", 100));
        // Access "a" to make "b" the LRU.
        cache.get("p1", "a");
        // Insert "d" → should evict "b".
        cache.put("p1", "d".to_string(), entry_sized(b"D", 100));
        assert!(cache.get("p1", "b").is_none(), "b should be evicted");
        assert!(cache.get("p1", "a").is_some());
        assert!(cache.get("p1", "d").is_some());
    }

    // 10. total_entries sums across partitions
    #[test]
    fn test_total_entries() {
        let mut cache = make_cache(10_000);
        cache.add_partition("x", 1000);
        cache.add_partition("y", 1000);
        cache.put("default", "d1".to_string(), entry(b"d"));
        cache.put("x", "x1".to_string(), entry(b"x"));
        cache.put("x", "x2".to_string(), entry(b"x"));
        cache.put("y", "y1".to_string(), entry(b"y"));
        assert_eq!(cache.total_entries(), 4);
    }

    // 11. purge_all_expired removes TTL-expired entries
    #[test]
    fn test_purge_all_expired() {
        let mut cache = make_cache(10_000);
        cache.add_partition("ttl_test", 1000);
        let expired_entry =
            CacheEntry::with_meta(b"expire".to_vec(), 6, Some(Duration::from_millis(0)), 0);
        let live_entry =
            CacheEntry::with_meta(b"live".to_vec(), 4, Some(Duration::from_secs(3600)), 0);
        cache.put("ttl_test", "expired".to_string(), expired_entry);
        cache.put("ttl_test", "live".to_string(), live_entry);
        std::thread::sleep(Duration::from_millis(2));
        let removed = cache.purge_all_expired();
        assert_eq!(removed, 1);
        assert!(cache.get("ttl_test", "expired").is_none());
        assert!(cache.get("ttl_test", "live").is_some());
    }

    // 12. partition_stats utilisation calculation
    #[test]
    fn test_partition_utilisation() {
        let mut cache = make_cache(10_000);
        cache.add_partition("util", 1000);
        for i in 0u8..5 {
            cache.put("util", format!("k{i}"), entry_sized(&[i; 100], 100));
        }
        let stats = cache.partition_stats("util").expect("should exist");
        let util = stats.utilisation();
        assert!(
            (util - 0.5).abs() < 1e-9,
            "expected 50% utilisation, got {util}"
        );
    }
}
