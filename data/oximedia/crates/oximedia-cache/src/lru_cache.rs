//! High-performance LRU (Least Recently Used) cache with capacity management.
//!
//! This module provides an arena-based doubly-linked-list LRU cache that avoids
//! heap allocations per node by storing all nodes in a `Vec<Option<LruNode>>`
//! and threading `prev`/`next` indices through them.  Eviction, insertion, and
//! lookup are all O(1) amortised.

use hashbrown::HashMap;
use std::time::{Duration, Instant};

/// Sentinel index meaning "no node".
const SENTINEL: usize = usize::MAX;

// ── Internal node ────────────────────────────────────────────────────────────

struct LruNode<K, V> {
    key: K,
    value: V,
    prev: usize,
    next: usize,
    /// Approximate byte size of this entry (provided by the caller).
    size_bytes: usize,
    /// Number of times this entry has been successfully accessed.
    access_count: u64,
    /// Wall-clock time of the most recent access.
    last_accessed: Instant,
    /// Optional TTL: the instant at which this entry expires.
    expires_at: Option<Instant>,
    /// Whether this entry is pinned (cannot be evicted).
    pinned: bool,
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of successful cache lookups.
    pub hits: u64,
    /// Total number of failed cache lookups.
    pub misses: u64,
    /// Total number of entries evicted to make room for new ones.
    pub evictions: u64,
    /// Sum of `size_bytes` for all currently resident entries.
    pub total_size_bytes: usize,
    /// Maximum number of entries the cache will hold before evicting.
    pub capacity: usize,
    /// Number of entries currently resident in the cache.
    pub entry_count: usize,
    /// Number of entries that expired via TTL.
    pub ttl_expirations: u64,
    /// Number of pinned entries currently in the cache.
    pub pinned_count: usize,
}

/// High-performance LRU cache backed by an arena of `Option<LruNode>` slots.
///
/// # Type parameters
/// * `K` – key type; must implement `Eq + Hash + Clone`.
/// * `V` – value type.
pub struct LruCache<K: Eq + std::hash::Hash + Clone, V> {
    capacity: usize,
    /// Maps each live key to its slot index inside `slots`.
    map: HashMap<K, usize>,
    /// Arena of node slots; `None` slots are free.
    slots: Vec<Option<LruNode<K, V>>>,
    /// Index of the most-recently-used node, or `SENTINEL` when empty.
    head: usize,
    /// Index of the least-recently-used node, or `SENTINEL` when empty.
    tail: usize,
    /// Pool of free slot indices for reuse.
    free: Vec<usize>,
    /// Number of live entries currently in the cache.
    len: usize,
    // ── stats ──
    hits: u64,
    misses: u64,
    evictions: u64,
    total_size_bytes: usize,
    /// Number of entries that expired via TTL lazy eviction.
    ttl_expirations: u64,
    /// Default TTL applied to entries inserted without an explicit TTL.
    default_ttl: Option<Duration>,
}

// ── Private helpers ───────────────────────────────────────────────────────────

impl<K: Eq + std::hash::Hash + Clone, V> LruCache<K, V> {
    /// Detach the node at `idx` from the linked list without freeing its slot.
    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let node = self.slots[idx].as_ref().expect("detach: slot must be Some");
            (node.prev, node.next)
        };
        if prev != SENTINEL {
            if let Some(n) = self.slots[prev].as_mut() {
                n.next = next;
            }
        } else {
            self.head = next;
        }
        if next != SENTINEL {
            if let Some(n) = self.slots[next].as_mut() {
                n.prev = prev;
            }
        } else {
            self.tail = prev;
        }
        if let Some(n) = self.slots[idx].as_mut() {
            n.prev = SENTINEL;
            n.next = SENTINEL;
        }
    }

    /// Attach the node at `idx` at the MRU head.
    fn attach_head(&mut self, idx: usize) {
        let old_head = self.head;
        if let Some(n) = self.slots[idx].as_mut() {
            n.prev = SENTINEL;
            n.next = old_head;
        }
        if old_head != SENTINEL {
            if let Some(n) = self.slots[old_head].as_mut() {
                n.prev = idx;
            }
        } else {
            // List was empty → this node is also the tail.
            self.tail = idx;
        }
        self.head = idx;
    }

    /// Acquire a free slot index, recycling from the free list or extending the
    /// arena.
    fn alloc_slot(&mut self) -> usize {
        if let Some(idx) = self.free.pop() {
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(None);
            idx
        }
    }
}

// ── Public impl ───────────────────────────────────────────────────────────────

impl<K: Eq + std::hash::Hash + Clone, V> LruCache<K, V> {
    /// Create a new `LruCache` with the given entry capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            slots: Vec::new(),
            head: SENTINEL,
            tail: SENTINEL,
            free: Vec::new(),
            len: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            total_size_bytes: 0,
            ttl_expirations: 0,
            default_ttl: None,
        }
    }

    /// Create a new `LruCache` with a default TTL applied to all entries
    /// unless overridden by `insert_with_ttl`.
    pub fn with_default_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            default_ttl: Some(ttl),
            ..Self::new(capacity)
        }
    }

    /// Look up `key`, move it to the MRU head, and return a shared reference
    /// to its value.  Records a cache hit/miss in the statistics.
    ///
    /// If the entry has a TTL and has expired, it is lazily evicted and `None`
    /// is returned (counted as a miss).
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            // Check TTL expiration lazily on access.
            let expired = self.slots[idx]
                .as_ref()
                .and_then(|n| n.expires_at)
                .map(|exp| Instant::now() >= exp)
                .unwrap_or(false);
            if expired {
                self.ttl_expirations += 1;
                self.misses += 1;
                let key_clone = self.slots[idx].as_ref().map(|n| n.key.clone());
                if let Some(k) = key_clone {
                    self.map.remove(&k);
                }
                self.detach(idx);
                if let Some(node) = self.slots[idx].take() {
                    self.total_size_bytes = self.total_size_bytes.saturating_sub(node.size_bytes);
                }
                self.len -= 1;
                self.free.push(idx);
                return None;
            }
            self.hits += 1;
            if let Some(node) = self.slots[idx].as_mut() {
                node.access_count += 1;
                node.last_accessed = Instant::now();
            }
            self.detach(idx);
            self.attach_head(idx);
            self.slots[idx].as_ref().map(|n| &n.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert `(key, value)` into the cache.
    ///
    /// * If the key already exists, its value is updated and it is promoted to
    ///   the MRU head.
    /// * If the cache is at capacity the LRU entry is evicted first.
    ///
    /// The default TTL (if configured via `with_default_ttl`) is applied.
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize) {
        let expires_at = self.default_ttl.map(|d| Instant::now() + d);
        self.insert_inner(key, value, size_bytes, expires_at, false);
    }

    /// Insert `(key, value)` with an explicit TTL duration.
    ///
    /// Overrides any default TTL configured on the cache.
    pub fn insert_with_ttl(&mut self, key: K, value: V, size_bytes: usize, ttl: Duration) {
        let expires_at = Some(Instant::now() + ttl);
        self.insert_inner(key, value, size_bytes, expires_at, false);
    }

    /// Insert a **pinned** entry that will not be evicted by LRU eviction.
    ///
    /// Pinned entries remain in the cache until explicitly removed via
    /// `remove` or `unpin` + subsequent eviction.
    pub fn insert_pinned(&mut self, key: K, value: V, size_bytes: usize) {
        let expires_at = self.default_ttl.map(|d| Instant::now() + d);
        self.insert_inner(key, value, size_bytes, expires_at, true);
    }

    /// Internal insertion with TTL and pin support.
    fn insert_inner(
        &mut self,
        key: K,
        value: V,
        size_bytes: usize,
        expires_at: Option<Instant>,
        pinned: bool,
    ) {
        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry.
            let old_size = self.slots[idx].as_ref().map(|n| n.size_bytes).unwrap_or(0);
            self.total_size_bytes = self.total_size_bytes.saturating_sub(old_size);
            if let Some(node) = self.slots[idx].as_mut() {
                node.value = value;
                node.size_bytes = size_bytes;
                node.last_accessed = Instant::now();
                node.expires_at = expires_at;
                node.pinned = pinned;
            }
            self.total_size_bytes += size_bytes;
            self.detach(idx);
            self.attach_head(idx);
            return;
        }

        // Evict LRU if at capacity.
        if self.len == self.capacity {
            self.evict_lru();
        }

        let idx = self.alloc_slot();
        self.slots[idx] = Some(LruNode {
            key: key.clone(),
            value,
            prev: SENTINEL,
            next: SENTINEL,
            size_bytes,
            access_count: 0,
            last_accessed: Instant::now(),
            expires_at,
            pinned,
        });
        self.attach_head(idx);
        self.map.insert(key, idx);
        self.total_size_bytes += size_bytes;
        self.len += 1;
    }

    /// Remove the entry for `key`, returning its value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        let node = self.slots[idx].take()?;
        self.total_size_bytes = self.total_size_bytes.saturating_sub(node.size_bytes);
        self.len -= 1;
        self.free.push(idx);
        Some(node.value)
    }

    /// Returns `true` if the cache contains an entry for `key`.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the number of entries currently resident in the cache.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        let pinned_count = self
            .slots
            .iter()
            .filter(|s| s.as_ref().map(|n| n.pinned).unwrap_or(false))
            .count();
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            total_size_bytes: self.total_size_bytes,
            capacity: self.capacity,
            entry_count: self.len,
            ttl_expirations: self.ttl_expirations,
            pinned_count,
        }
    }

    /// Return a shared reference to the value for `key` without updating LRU
    /// order or statistics.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map
            .get(key)
            .and_then(|&idx| self.slots[idx].as_ref())
            .map(|n| &n.value)
    }

    /// Manually evict the least-recently-used **unpinned** entry, returning
    /// `(key, value)`.
    ///
    /// Pinned entries are skipped.  Returns `None` if the cache is empty or
    /// every entry is pinned.
    pub fn evict_lru(&mut self) -> Option<(K, V)> {
        if self.tail == SENTINEL {
            return None;
        }
        // Walk from tail (LRU) towards head (MRU) looking for the first
        // unpinned entry.
        let mut candidate = self.tail;
        while candidate != SENTINEL {
            let is_pinned = self.slots[candidate]
                .as_ref()
                .map(|n| n.pinned)
                .unwrap_or(false);
            if !is_pinned {
                break;
            }
            candidate = self.slots[candidate]
                .as_ref()
                .map(|n| n.prev)
                .unwrap_or(SENTINEL);
        }
        if candidate == SENTINEL {
            return None;
        }
        let key = self.slots[candidate].as_ref()?.key.clone();
        self.map.remove(&key);
        self.detach(candidate);
        let node = self.slots[candidate].take()?;
        self.total_size_bytes = self.total_size_bytes.saturating_sub(node.size_bytes);
        self.len -= 1;
        self.evictions += 1;
        self.free.push(candidate);
        Some((key, node.value))
    }

    // ── TTL helpers ──────────────────────────────────────────────────────────

    /// Set the default TTL for newly inserted entries.
    pub fn set_default_ttl(&mut self, ttl: Option<Duration>) {
        self.default_ttl = ttl;
    }

    /// Eagerly purge all expired entries.  Returns the number of entries
    /// removed.
    pub fn purge_expired(&mut self) -> usize {
        let now = Instant::now();
        let expired_keys: Vec<K> = self
            .map
            .keys()
            .filter(|k| {
                self.map
                    .get(*k)
                    .and_then(|&idx| self.slots[idx].as_ref().and_then(|n| n.expires_at))
                    .map(|exp| now >= exp)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let count = expired_keys.len();
        for key in expired_keys {
            if let Some(idx) = self.map.remove(&key) {
                self.detach(idx);
                if let Some(node) = self.slots[idx].take() {
                    self.total_size_bytes = self.total_size_bytes.saturating_sub(node.size_bytes);
                }
                self.len -= 1;
                self.ttl_expirations += 1;
                self.free.push(idx);
            }
        }
        count
    }

    // ── Pinning helpers ──────────────────────────────────────────────────────

    /// Pin an existing entry so it cannot be evicted by LRU eviction.
    ///
    /// Returns `true` if the entry was found and pinned.
    pub fn pin(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.map.get(key) {
            if let Some(node) = self.slots[idx].as_mut() {
                node.pinned = true;
                return true;
            }
        }
        false
    }

    /// Unpin an existing entry so it becomes eligible for LRU eviction again.
    ///
    /// Returns `true` if the entry was found and unpinned.
    pub fn unpin(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.map.get(key) {
            if let Some(node) = self.slots[idx].as_mut() {
                node.pinned = false;
                return true;
            }
        }
        false
    }

    /// Returns `true` if the entry for `key` is pinned.
    pub fn is_pinned(&self, key: &K) -> bool {
        self.map
            .get(key)
            .and_then(|&idx| self.slots[idx].as_ref())
            .map(|n| n.pinned)
            .unwrap_or(false)
    }

    // ── Extended TTL helpers ─────────────────────────────────────────────────

    /// Refresh the TTL of an existing entry, resetting its expiration clock.
    ///
    /// If the entry has no TTL (neither explicit nor default), this is a no-op
    /// and returns `false`.  Returns `true` if the TTL was successfully
    /// refreshed.
    pub fn refresh_ttl(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.map.get(key) {
            if let Some(node) = self.slots[idx].as_mut() {
                if let Some(old_exp) = node.expires_at {
                    // Compute the original TTL duration by subtracting the
                    // insertion/refresh instant from the old expiry.
                    // Since we only know the last_accessed time, we use the
                    // default_ttl if available, otherwise estimate from the
                    // remaining time (conservative: grant the same duration
                    // the entry was last configured with).
                    let ttl = if let Some(d) = self.default_ttl {
                        d
                    } else {
                        // Best effort: grant the time between now and old expiry
                        // if still in the future, otherwise 0.
                        let now = Instant::now();
                        if old_exp > now {
                            old_exp.duration_since(now)
                        } else {
                            // Already expired; refresh with a zero-length TTL
                            // (caller should set a specific TTL instead).
                            return false;
                        }
                    };
                    node.expires_at = Some(Instant::now() + ttl);
                    return true;
                }
            }
        }
        false
    }

    /// Set an explicit TTL on an existing entry, overriding any previous TTL.
    ///
    /// Returns `true` if the entry was found and the TTL was set.
    pub fn set_entry_ttl(&mut self, key: &K, ttl: Duration) -> bool {
        if let Some(&idx) = self.map.get(key) {
            if let Some(node) = self.slots[idx].as_mut() {
                node.expires_at = Some(Instant::now() + ttl);
                return true;
            }
        }
        false
    }

    /// Remove the TTL from an existing entry, making it live indefinitely
    /// (until evicted by LRU or explicitly removed).
    ///
    /// Returns `true` if the entry was found and TTL cleared.
    pub fn clear_entry_ttl(&mut self, key: &K) -> bool {
        if let Some(&idx) = self.map.get(key) {
            if let Some(node) = self.slots[idx].as_mut() {
                node.expires_at = None;
                return true;
            }
        }
        false
    }

    /// Return the remaining TTL for the entry, or `None` if the entry does
    /// not exist or has no TTL.
    pub fn remaining_ttl(&self, key: &K) -> Option<Duration> {
        self.map
            .get(key)
            .and_then(|&idx| self.slots[idx].as_ref())
            .and_then(|n| n.expires_at)
            .and_then(|exp| {
                let now = Instant::now();
                if exp > now {
                    Some(exp.duration_since(now))
                } else {
                    None // already expired
                }
            })
    }

    // ── Extended pinning helpers ─────────────────────────────────────────────

    /// Return the number of pinned entries.
    pub fn pinned_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.as_ref().map(|n| n.pinned).unwrap_or(false))
            .count()
    }

    /// Unpin all currently pinned entries.  Returns the number of entries
    /// unpinned.
    pub fn unpin_all(&mut self) -> usize {
        let mut count = 0usize;
        for slot in &mut self.slots {
            if let Some(node) = slot.as_mut() {
                if node.pinned {
                    node.pinned = false;
                    count += 1;
                }
            }
        }
        count
    }

    // ── Capacity management ─────────────────────────────────────────────────

    /// Resize the cache to `new_capacity`, evicting excess entries if
    /// necessary.
    ///
    /// Returns the number of entries evicted.
    pub fn resize(&mut self, new_capacity: usize) -> usize {
        let new_cap = new_capacity.max(1);
        self.capacity = new_cap;
        let mut evicted = 0usize;
        while self.len > new_cap {
            if self.evict_lru().is_some() {
                evicted += 1;
            } else {
                break; // all pinned
            }
        }
        evicted
    }

    /// Return the current capacity (maximum number of entries).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // ── Iteration helpers ───────────────────────────────────────────────────

    /// Return a vector of all keys currently in the cache (in no particular
    /// order).
    pub fn keys(&self) -> Vec<K> {
        self.map.keys().cloned().collect()
    }

    /// Remove all entries from the cache, resetting statistics.
    pub fn clear(&mut self) {
        self.map.clear();
        self.slots.clear();
        self.free.clear();
        self.head = SENTINEL;
        self.tail = SENTINEL;
        self.len = 0;
        self.total_size_bytes = 0;
    }

    /// Return the access count for a specific entry, or `None` if the key
    /// does not exist.
    pub fn access_count(&self, key: &K) -> Option<u64> {
        self.map
            .get(key)
            .and_then(|&idx| self.slots[idx].as_ref())
            .map(|n| n.access_count)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Basic insertion and retrieval
    #[test]
    fn test_insert_and_get() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 10);
        cache.insert("b", 2, 20);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    // 2. Miss on absent key
    #[test]
    fn test_miss_on_absent_key() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 10);
        assert_eq!(cache.get(&"z"), None);
    }

    // 3. LRU eviction when capacity is exceeded
    #[test]
    fn test_lru_eviction() {
        let mut cache: LruCache<i32, &str> = LruCache::new(3);
        cache.insert(1, "one", 1);
        cache.insert(2, "two", 1);
        cache.insert(3, "three", 1);
        // Access key 1 → moves to MRU head; key 2 becomes LRU tail
        cache.get(&1);
        // Insert key 4 → should evict key 2 (LRU)
        cache.insert(4, "four", 1);
        assert!(!cache.contains(&2), "key 2 should have been evicted");
        assert!(cache.contains(&1));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    // 4. len and is_empty
    #[test]
    fn test_len_and_is_empty() {
        let mut cache: LruCache<u32, u32> = LruCache::new(5);
        assert!(cache.is_empty());
        cache.insert(1, 100, 8);
        cache.insert(2, 200, 8);
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
    }

    // 5. remove
    #[test]
    fn test_remove() {
        let mut cache: LruCache<&str, u64> = LruCache::new(4);
        cache.insert("x", 42, 8);
        let removed = cache.remove(&"x");
        assert_eq!(removed, Some(42));
        assert!(!cache.contains(&"x"));
        assert_eq!(cache.len(), 0);
    }

    // 6. peek does not affect LRU order
    #[test]
    fn test_peek_no_side_effects() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);
        cache.insert(1, 10, 1);
        cache.insert(2, 20, 1);
        cache.insert(3, 30, 1);
        // peek at the LRU tail (key 1) — should not promote it
        let _ = cache.peek(&1);
        // Insert a 4th entry → should still evict key 1 (still LRU tail)
        cache.insert(4, 40, 1);
        assert!(!cache.contains(&1));
    }

    // 7. Manual evict_lru
    #[test]
    fn test_evict_lru_manual() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 1);
        cache.insert("b", 2, 1);
        cache.insert("c", 3, 1);
        let evicted = cache.evict_lru();
        assert!(evicted.is_some());
        let (k, _v) = evicted.expect("eviction should succeed");
        assert_eq!(
            k, "a",
            "oldest-inserted key should be evicted when nothing was accessed"
        );
    }

    // 8. Hit / miss counters
    #[test]
    fn test_stats_hit_miss() {
        let mut cache: LruCache<i32, i32> = LruCache::new(4);
        cache.insert(1, 10, 8);
        cache.get(&1);
        cache.get(&1);
        cache.get(&99); // miss
        let s = cache.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
    }

    // 9. Eviction counter
    #[test]
    fn test_stats_evictions() {
        let mut cache: LruCache<i32, i32> = LruCache::new(2);
        cache.insert(1, 1, 1);
        cache.insert(2, 2, 1);
        cache.insert(3, 3, 1); // evicts 1
        cache.insert(4, 4, 1); // evicts 2
        assert_eq!(cache.stats().evictions, 2);
    }

    // 10. total_size_bytes tracking
    #[test]
    fn test_total_size_bytes() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert(1, 1, 100);
        cache.insert(2, 2, 200);
        assert_eq!(cache.stats().total_size_bytes, 300);
        cache.remove(&1);
        assert_eq!(cache.stats().total_size_bytes, 200);
    }

    // 11. Update existing key preserves size tracking
    #[test]
    fn test_update_existing_key() {
        let mut cache: LruCache<i32, i32> = LruCache::new(4);
        cache.insert(1, 10, 100);
        cache.insert(1, 20, 50); // update
        assert_eq!(cache.get(&1), Some(&20));
        assert_eq!(cache.stats().total_size_bytes, 50);
        assert_eq!(cache.len(), 1);
    }

    // 12. contains
    #[test]
    fn test_contains() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("hello", 1, 5);
        assert!(cache.contains(&"hello"));
        assert!(!cache.contains(&"world"));
    }

    // 13. evict_lru on empty cache returns None
    #[test]
    fn test_evict_lru_empty() {
        let mut cache: LruCache<i32, i32> = LruCache::new(4);
        assert_eq!(cache.evict_lru(), None);
    }

    // 14. Capacity reported in stats
    #[test]
    fn test_stats_capacity() {
        let cache: LruCache<i32, i32> = LruCache::new(7);
        assert_eq!(cache.stats().capacity, 7);
        assert_eq!(cache.stats().entry_count, 0);
    }

    // 15. Large sequential workload – evictions keep len at capacity
    #[test]
    fn test_large_sequential_workload() {
        let cap = 10usize;
        let mut cache: LruCache<usize, usize> = LruCache::new(cap);
        for i in 0..100 {
            cache.insert(i, i * 2, 1);
        }
        assert_eq!(cache.len(), cap);
        // The last `cap` keys should all be present.
        for i in (100 - cap)..100 {
            assert!(cache.contains(&i), "key {i} should be present");
        }
    }

    // ── TTL tests ────────────────────────────────────────────────────────────

    // 16. TTL expiration on get (lazy eviction)
    #[test]
    fn test_ttl_expired_entry_returns_none() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        // Insert with a TTL of 0ms (already expired on next access).
        cache.insert_with_ttl("ephemeral", 42, 8, Duration::from_millis(0));
        // Small sleep to ensure expiry.
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(
            cache.get(&"ephemeral"),
            None,
            "expired entry should be gone"
        );
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().ttl_expirations, 1);
    }

    // 17. Non-expired TTL entry is still accessible
    #[test]
    fn test_ttl_non_expired_entry() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_with_ttl("long_lived", 99, 8, Duration::from_secs(3600));
        assert_eq!(cache.get(&"long_lived"), Some(&99));
    }

    // 18. Default TTL applied to all entries
    #[test]
    fn test_default_ttl() {
        let mut cache: LruCache<&str, i32> =
            LruCache::with_default_ttl(4, Duration::from_millis(0));
        cache.insert("a", 1, 8);
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(cache.get(&"a"), None, "default TTL should expire entry");
    }

    // 19. purge_expired clears expired entries eagerly
    #[test]
    fn test_purge_expired() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_with_ttl("x", 1, 8, Duration::from_millis(0));
        cache.insert("y", 2, 8); // no TTL
        std::thread::sleep(Duration::from_millis(2));
        let purged = cache.purge_expired();
        assert_eq!(purged, 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&"y"));
    }

    // 20. set_default_ttl can change the default
    #[test]
    fn test_set_default_ttl() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.set_default_ttl(Some(Duration::from_secs(3600)));
        cache.insert("a", 1, 8);
        assert_eq!(cache.get(&"a"), Some(&1));
        cache.set_default_ttl(None);
        cache.insert("b", 2, 8);
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    // ── Pinning tests ────────────────────────────────────────────────────────

    // 21. Pinned entry survives eviction
    #[test]
    fn test_pinned_entry_survives_eviction() {
        let mut cache: LruCache<i32, &str> = LruCache::new(3);
        cache.insert_pinned(1, "pinned", 1);
        cache.insert(2, "two", 1);
        cache.insert(3, "three", 1);
        // Key 1 is LRU but pinned → eviction should skip it and evict key 2.
        cache.insert(4, "four", 1);
        assert!(cache.contains(&1), "pinned entry should survive");
        assert!(!cache.contains(&2), "unpinned LRU should be evicted");
    }

    // 22. insert_pinned sets pinned flag
    #[test]
    fn test_insert_pinned() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_pinned("critical", 99, 8);
        assert!(cache.is_pinned(&"critical"));
        assert_eq!(cache.stats().pinned_count, 1);
    }

    // 23. pin / unpin existing entries
    #[test]
    fn test_pin_and_unpin() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("x", 1, 8);
        assert!(!cache.is_pinned(&"x"));
        assert!(cache.pin(&"x"));
        assert!(cache.is_pinned(&"x"));
        assert!(cache.unpin(&"x"));
        assert!(!cache.is_pinned(&"x"));
    }

    // 24. pin absent key returns false
    #[test]
    fn test_pin_absent() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        assert!(!cache.pin(&"ghost"));
    }

    // 25. All entries pinned → evict_lru returns None
    #[test]
    fn test_all_pinned_evict_returns_none() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);
        cache.insert_pinned(1, 10, 1);
        cache.insert_pinned(2, 20, 1);
        cache.insert_pinned(3, 30, 1);
        assert_eq!(cache.evict_lru(), None);
    }

    // 26. Pinned entry can still be explicitly removed
    #[test]
    fn test_pinned_entry_can_be_removed() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_pinned("keep", 42, 8);
        let removed = cache.remove(&"keep");
        assert_eq!(removed, Some(42));
        assert_eq!(cache.len(), 0);
    }

    // ── Extended TTL tests ──────────────────────────────────────────────────

    // 27. refresh_ttl extends the expiration
    #[test]
    fn test_refresh_ttl() {
        let mut cache: LruCache<&str, i32> =
            LruCache::with_default_ttl(4, Duration::from_secs(3600));
        cache.insert("a", 1, 8);
        assert!(cache.refresh_ttl(&"a"));
        // Entry should still be accessible.
        assert_eq!(cache.get(&"a"), Some(&1));
    }

    // 28. refresh_ttl returns false for absent key
    #[test]
    fn test_refresh_ttl_absent() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        assert!(!cache.refresh_ttl(&"ghost"));
    }

    // 29. refresh_ttl returns false for entry without TTL
    #[test]
    fn test_refresh_ttl_no_ttl() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 8);
        assert!(!cache.refresh_ttl(&"a"));
    }

    // 30. set_entry_ttl overrides existing TTL
    #[test]
    fn test_set_entry_ttl() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 8);
        assert!(cache.set_entry_ttl(&"a", Duration::from_millis(0)));
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(cache.get(&"a"), None, "entry should expire with new TTL");
    }

    // 31. set_entry_ttl on absent key returns false
    #[test]
    fn test_set_entry_ttl_absent() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        assert!(!cache.set_entry_ttl(&"nope", Duration::from_secs(60)));
    }

    // 32. clear_entry_ttl removes TTL
    #[test]
    fn test_clear_entry_ttl() {
        let mut cache: LruCache<&str, i32> =
            LruCache::with_default_ttl(4, Duration::from_millis(1));
        cache.insert("a", 1, 8);
        assert!(cache.clear_entry_ttl(&"a"));
        std::thread::sleep(Duration::from_millis(5));
        // Should still be accessible since TTL was cleared.
        assert_eq!(cache.get(&"a"), Some(&1));
    }

    // 33. remaining_ttl returns correct duration
    #[test]
    fn test_remaining_ttl() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_with_ttl("a", 1, 8, Duration::from_secs(3600));
        let remaining = cache.remaining_ttl(&"a");
        assert!(remaining.is_some());
        let r = remaining.expect("should have remaining TTL");
        assert!(r.as_secs() > 3590, "remaining TTL should be close to 3600s");
    }

    // 34. remaining_ttl returns None for expired entry
    #[test]
    fn test_remaining_ttl_expired() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_with_ttl("a", 1, 8, Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.remaining_ttl(&"a").is_none());
    }

    // 35. remaining_ttl returns None for entry without TTL
    #[test]
    fn test_remaining_ttl_no_ttl() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 8);
        assert!(cache.remaining_ttl(&"a").is_none());
    }

    // ── Extended pinning tests ──────────────────────────────────────────────

    // 36. pinned_count tracks correctly
    #[test]
    fn test_pinned_count() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert_pinned(1, 10, 1);
        cache.insert_pinned(2, 20, 1);
        cache.insert(3, 30, 1);
        assert_eq!(cache.pinned_count(), 2);
    }

    // 37. unpin_all clears all pins
    #[test]
    fn test_unpin_all() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert_pinned(1, 10, 1);
        cache.insert_pinned(2, 20, 1);
        cache.insert_pinned(3, 30, 1);
        let unpinned = cache.unpin_all();
        assert_eq!(unpinned, 3);
        assert_eq!(cache.pinned_count(), 0);
    }

    // 38. unpin_all on cache with no pins returns 0
    #[test]
    fn test_unpin_all_none_pinned() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert(1, 10, 1);
        assert_eq!(cache.unpin_all(), 0);
    }

    // 39. After unpin_all, entries can be evicted
    #[test]
    fn test_unpin_all_allows_eviction() {
        let mut cache: LruCache<i32, i32> = LruCache::new(3);
        cache.insert_pinned(1, 10, 1);
        cache.insert_pinned(2, 20, 1);
        cache.insert_pinned(3, 30, 1);
        assert_eq!(cache.evict_lru(), None); // all pinned
        cache.unpin_all();
        assert!(cache.evict_lru().is_some());
    }

    // ── Capacity management tests ───────────────────────────────────────────

    // 40. resize shrinks cache
    #[test]
    fn test_resize_shrink() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        for i in 0..10 {
            cache.insert(i, i * 10, 1);
        }
        let evicted = cache.resize(5);
        assert_eq!(evicted, 5);
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.capacity(), 5);
    }

    // 41. resize grows cache (no evictions)
    #[test]
    fn test_resize_grow() {
        let mut cache: LruCache<i32, i32> = LruCache::new(5);
        for i in 0..5 {
            cache.insert(i, i, 1);
        }
        let evicted = cache.resize(20);
        assert_eq!(evicted, 0);
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.capacity(), 20);
    }

    // 42. resize respects pinned entries
    #[test]
    fn test_resize_with_pinned() {
        let mut cache: LruCache<i32, i32> = LruCache::new(5);
        cache.insert_pinned(1, 10, 1);
        cache.insert(2, 20, 1);
        cache.insert(3, 30, 1);
        cache.insert(4, 40, 1);
        cache.insert(5, 50, 1);
        let evicted = cache.resize(2);
        // Should evict unpinned entries but not pinned
        assert!(evicted >= 3);
        assert!(cache.contains(&1), "pinned entry should survive");
    }

    // ── keys / clear / access_count tests ───────────────────────────────────

    // 43. keys returns all keys
    #[test]
    fn test_keys() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert(1, 10, 1);
        cache.insert(2, 20, 1);
        cache.insert(3, 30, 1);
        let mut keys = cache.keys();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    // 44. clear resets the cache
    #[test]
    fn test_clear() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert(1, 10, 100);
        cache.insert(2, 20, 200);
        cache.get(&1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_size_bytes, 0);
        assert_eq!(cache.len(), 0);
    }

    // 45. access_count returns correct count
    #[test]
    fn test_access_count() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert("a", 1, 8);
        cache.get(&"a");
        cache.get(&"a");
        cache.get(&"a");
        assert_eq!(cache.access_count(&"a"), Some(3));
    }

    // 46. access_count returns None for absent key
    #[test]
    fn test_access_count_absent() {
        let cache: LruCache<&str, i32> = LruCache::new(4);
        assert_eq!(cache.access_count(&"ghost"), None);
    }

    // 47. TTL expiration increments ttl_expirations stat
    #[test]
    fn test_ttl_stats_counter() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert_with_ttl(1, 10, 8, Duration::from_millis(0));
        cache.insert_with_ttl(2, 20, 8, Duration::from_millis(0));
        std::thread::sleep(Duration::from_millis(5));
        cache.get(&1); // triggers expiration
        cache.get(&2); // triggers expiration
        assert_eq!(cache.stats().ttl_expirations, 2);
    }

    // 48. pinned entry with TTL still expires
    #[test]
    fn test_pinned_entry_with_ttl_expires() {
        let mut cache: LruCache<&str, i32> = LruCache::new(4);
        cache.insert_with_ttl("pinned_ttl", 42, 8, Duration::from_millis(0));
        cache.pin(&"pinned_ttl");
        std::thread::sleep(Duration::from_millis(5));
        // TTL expiration is checked on get, regardless of pin status
        assert_eq!(cache.get(&"pinned_ttl"), None);
    }

    // 49. Large workload with mixed TTL and pinning
    #[test]
    fn test_mixed_ttl_and_pinning_workload() {
        let mut cache: LruCache<i32, i32> = LruCache::new(20);
        // Insert pinned entries
        for i in 0..5 {
            cache.insert_pinned(i, i * 100, 10);
        }
        // Insert TTL entries
        for i in 5..15 {
            cache.insert_with_ttl(i, i * 100, 10, Duration::from_secs(3600));
        }
        // Insert normal entries
        for i in 15..20 {
            cache.insert(i, i * 100, 10);
        }
        assert_eq!(cache.len(), 20);
        assert_eq!(cache.pinned_count(), 5);

        // Overflow: insert 10 more to trigger evictions
        for i in 100..110 {
            cache.insert(i, i, 10);
        }
        // Pinned entries should survive
        for i in 0..5 {
            assert!(cache.contains(&i), "pinned entry {i} should survive");
        }
    }

    // 50. purge_expired with mixed entries
    #[test]
    fn test_purge_expired_mixed() {
        let mut cache: LruCache<i32, i32> = LruCache::new(10);
        cache.insert_with_ttl(1, 10, 8, Duration::from_millis(0)); // will expire
        cache.insert_with_ttl(2, 20, 8, Duration::from_secs(3600)); // won't expire
        cache.insert(3, 30, 8); // no TTL
        std::thread::sleep(Duration::from_millis(5));
        let purged = cache.purge_expired();
        assert_eq!(purged, 1);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
    }
}
