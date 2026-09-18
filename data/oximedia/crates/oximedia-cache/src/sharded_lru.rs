//! Sharded LRU cache for concurrent access with reduced lock contention.
//!
//! [`ShardedLruCache`] divides the key space into `num_shards` independent
//! [`LruCache`] instances.  Each cache operation acquires only the lock for
//! the shard that owns the key, keeping contention proportional to
//! `1 / num_shards` under uniform load.
//!
//! Shard assignment uses FNV-1a hashing of the serialised key, which is
//! computed without acquiring any lock.

use crate::lru_cache::LruCache;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

// ── FNV-1a hasher (stdlib-independent, no deps) ───────────────────────────────

const FNV_OFFSET: u64 = 0xcbf29ce484222325u64;
const FNV_PRIME: u64 = 0x00000100000001b3u64;

struct Fnv1aHasher(u64);

impl Fnv1aHasher {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

impl Hasher for Fnv1aHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }
    fn finish(&self) -> u64 {
        Fnv1aHasher::finish(self)
    }
}

fn fnv1a_hash<K: Hash>(key: &K) -> u64 {
    let mut hasher = Fnv1aHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

// ── ShardedLruCache ───────────────────────────────────────────────────────────

/// An LRU cache sharded across `num_shards` independent [`LruCache`] instances.
///
/// # Type parameters
/// * `K` – key type; must implement `Eq + Hash + Clone + Send`.
/// * `V` – value type; must implement `Clone + Send`.
///
/// # Sharding
///
/// The shard for a key is `fnv1a(key) % num_shards`.  This provides an
/// even distribution for uniformly distributed keys without requiring the key
/// type to implement any additional traits beyond `Hash`.
///
/// # Capacity
///
/// `capacity` is the *total* maximum entry count.  Each shard is given
/// `capacity / num_shards` (remainder distributed to the first `r` shards).
pub struct ShardedLruCache<K, V>
where
    K: Eq + Hash + Clone + Send + 'static,
    V: Clone + Send + 'static,
{
    shards: Vec<Mutex<LruCache<K, V>>>,
    num_shards: usize,
    capacity: usize,
}

impl<K, V> ShardedLruCache<K, V>
where
    K: Eq + Hash + Clone + Send + 'static,
    V: Clone + Send + 'static,
{
    /// Create a new `ShardedLruCache` with `num_shards` shards and total
    /// entry capacity `capacity`.
    ///
    /// # Panics
    ///
    /// Does not panic; `num_shards` is clamped to `[1, capacity]`.
    pub fn new(num_shards: usize, capacity: usize) -> Self {
        let num_shards = num_shards.clamp(1, capacity.max(1));
        let base = capacity / num_shards;
        let remainder = capacity % num_shards;

        let shards = (0..num_shards)
            .map(|i| {
                // Distribute the remainder among the first `remainder` shards.
                let shard_cap = if i < remainder { base + 1 } else { base };
                // Ensure each shard has at least capacity 1.
                Mutex::new(LruCache::new(shard_cap.max(1)))
            })
            .collect();

        Self {
            shards,
            num_shards,
            capacity,
        }
    }

    /// Determine which shard owns `key`.
    fn shard_index(&self, key: &K) -> usize {
        (fnv1a_hash(key) % self.num_shards as u64) as usize
    }

    /// Look up `key`, returning a cloned copy of its value if present.
    ///
    /// This acquires only the shard lock for `key`.
    pub fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        self.shards[idx]
            .lock()
            .ok()
            .and_then(|mut shard| shard.get(key).cloned())
    }

    /// Insert `(key, value)` with the given `size_bytes` hint.
    ///
    /// This acquires only the shard lock for `key`.
    pub fn put(&self, key: K, value: V, size_bytes: usize) {
        let idx = self.shard_index(&key);
        if let Ok(mut shard) = self.shards[idx].lock() {
            shard.insert(key, value, size_bytes);
        }
    }

    /// Return `true` if the cache contains `key`.
    pub fn contains(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        self.shards[idx]
            .lock()
            .map(|shard| shard.contains(key))
            .unwrap_or(false)
    }

    /// Remove `key` from the cache.  Returns the removed value if present.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        self.shards[idx]
            .lock()
            .ok()
            .and_then(|mut shard| shard.remove(key))
    }

    /// Return the total number of entries across all shards.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().map(|shard| shard.len()).unwrap_or(0))
            .sum()
    }

    /// Return `true` when all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the total configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Return the capacity of shard `idx`, or `0` if out of range.
    pub fn shard_capacity(&self, idx: usize) -> usize {
        self.shards
            .get(idx)
            .and_then(|s| s.lock().ok())
            .map(|shard| shard.capacity())
            .unwrap_or(0)
    }

    /// Return the entry count per shard as a vector.
    pub fn shard_lengths(&self) -> Vec<usize> {
        self.shards
            .iter()
            .map(|s| s.lock().map(|shard| shard.len()).unwrap_or(0))
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    // 1. Basic put and get
    #[test]
    fn test_put_and_get() {
        let cache: ShardedLruCache<String, i32> = ShardedLruCache::new(4, 100);
        cache.put("key_a".to_string(), 42, 8);
        assert_eq!(cache.get(&"key_a".to_string()), Some(42));
    }

    // 2. Miss on absent key returns None
    #[test]
    fn test_get_absent() {
        let cache: ShardedLruCache<String, i32> = ShardedLruCache::new(4, 100);
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    // 3. contains returns true for present key
    #[test]
    fn test_contains() {
        let cache: ShardedLruCache<u32, u32> = ShardedLruCache::new(8, 200);
        cache.put(99, 999, 4);
        assert!(cache.contains(&99));
        assert!(!cache.contains(&100));
    }

    // 4. len tracks total entries
    #[test]
    fn test_len() {
        let cache: ShardedLruCache<u32, u32> = ShardedLruCache::new(4, 100);
        assert_eq!(cache.len(), 0);
        cache.put(1, 10, 4);
        cache.put(2, 20, 4);
        cache.put(3, 30, 4);
        assert_eq!(cache.len(), 3);
    }

    // 5. is_empty
    #[test]
    fn test_is_empty() {
        let cache: ShardedLruCache<i32, i32> = ShardedLruCache::new(4, 50);
        assert!(cache.is_empty());
        cache.put(1, 1, 1);
        assert!(!cache.is_empty());
    }

    // 6. capacity returns total configured capacity
    #[test]
    fn test_capacity() {
        let cache: ShardedLruCache<i32, i32> = ShardedLruCache::new(4, 128);
        assert_eq!(cache.capacity(), 128);
    }

    // 7. num_shards is respected
    #[test]
    fn test_num_shards() {
        let cache: ShardedLruCache<i32, i32> = ShardedLruCache::new(8, 1000);
        assert_eq!(cache.num_shards(), 8);
    }

    // 8. remove returns value and deletes entry
    #[test]
    fn test_remove() {
        let cache: ShardedLruCache<String, String> = ShardedLruCache::new(4, 50);
        cache.put("k".to_string(), "v".to_string(), 2);
        let removed = cache.remove(&"k".to_string());
        assert_eq!(removed, Some("v".to_string()));
        assert!(!cache.contains(&"k".to_string()));
        assert_eq!(cache.len(), 0);
    }

    // 9. Entries distribute across shards
    #[test]
    fn test_distribution_across_shards() {
        let cache: ShardedLruCache<u32, u32> = ShardedLruCache::new(4, 400);
        for i in 0u32..200 {
            cache.put(i, i, 4);
        }
        let lengths = cache.shard_lengths();
        assert_eq!(lengths.len(), 4);
        // Each shard should have some entries (not all in one shard).
        let non_empty = lengths.iter().filter(|&&l| l > 0).count();
        assert!(
            non_empty >= 2,
            "entries should spread across at least 2 shards"
        );
    }

    // 10. LRU eviction within shard
    #[test]
    fn test_lru_eviction_within_shard() {
        // Use single shard for predictable behaviour.
        let cache: ShardedLruCache<u32, u32> = ShardedLruCache::new(1, 3);
        cache.put(1, 10, 1);
        cache.put(2, 20, 1);
        cache.put(3, 30, 1);
        // Access 1 to make it MRU; key 2 should be LRU.
        cache.get(&1);
        // Insert 4 → evicts 2.
        cache.put(4, 40, 1);
        assert!(!cache.contains(&2), "key 2 should be evicted");
        assert!(cache.contains(&1));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    // 11. Many keys fill cache to capacity
    #[test]
    fn test_fill_to_capacity() {
        let cap = 50usize;
        let shards = 4;
        let cache: ShardedLruCache<usize, usize> = ShardedLruCache::new(shards, cap);
        for i in 0..200 {
            cache.put(i, i, 1);
        }
        // Total len must not exceed capacity (some shards may have slightly
        // different capacities due to remainder distribution).
        assert!(
            cache.len() <= cap,
            "total len {} must not exceed capacity {}",
            cache.len(),
            cap
        );
    }

    // 12. Concurrent reads from multiple threads
    #[test]
    fn test_concurrent_reads() {
        let cache = Arc::new(ShardedLruCache::<u32, u32>::new(8, 1000));
        // Pre-populate.
        for i in 0u32..100 {
            cache.put(i, i * 2, 4);
        }
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for i in 0u32..100 {
                        let v = c.get(&i);
                        if let Some(val) = v {
                            assert_eq!(val, i * 2, "thread {t}: key {i} has wrong value");
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
    }

    // 13. Concurrent writes from multiple threads do not panic or corrupt
    #[test]
    fn test_concurrent_writes() {
        let cache = Arc::new(ShardedLruCache::<u32, u32>::new(8, 500));
        let threads: Vec<_> = (0u32..8)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for i in 0u32..100 {
                        c.put(t * 100 + i, t * 1000 + i, 4);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
        // Total entries must be ≤ 500.
        assert!(
            cache.len() <= 500,
            "total len {} must not exceed 500",
            cache.len()
        );
    }

    // 14. Mixed read-write from multiple threads produces no inconsistency
    #[test]
    fn test_concurrent_mixed_rw() {
        let cache = Arc::new(ShardedLruCache::<u32, u32>::new(4, 200));
        // Pre-populate stable keys.
        for i in 0u32..50 {
            cache.put(i, i, 1);
        }
        let threads: Vec<_> = (0u32..4)
            .map(|t| {
                let c = Arc::clone(&cache);
                thread::spawn(move || {
                    for i in 0u32..50 {
                        // Mix reads and writes.
                        c.get(&i);
                        c.put(1000 + t * 100 + i, t + i, 1);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread panicked");
        }
    }

    // 15. shard_capacity returns correct per-shard capacity
    #[test]
    fn test_shard_capacity() {
        let total = 10usize;
        let shards = 3usize;
        let cache: ShardedLruCache<i32, i32> = ShardedLruCache::new(shards, total);
        // 10 / 3 = 3 rem 1 → shard 0 has 4, shards 1-2 have 3.
        let c0 = cache.shard_capacity(0);
        let c1 = cache.shard_capacity(1);
        let c2 = cache.shard_capacity(2);
        // All shards combined should equal total capacity.
        assert_eq!(c0 + c1 + c2, total);
        // Each shard must have at least 1.
        assert!(c0 >= 1);
        assert!(c1 >= 1);
        assert!(c2 >= 1);
    }

    // 16. All inserted keys are retrievable (small cache, many distinct keys)
    #[test]
    fn test_all_keys_retrievable_within_capacity() {
        let cap = 20usize;
        let cache: ShardedLruCache<u32, u32> = ShardedLruCache::new(4, cap);
        let keys: Vec<u32> = (0..cap as u32).collect();
        for &k in &keys {
            cache.put(k, k * 10, 1);
        }
        let found: HashSet<u32> = keys
            .iter()
            .filter(|&&k| cache.contains(&k))
            .copied()
            .collect();
        // All keys fit within capacity, so all should be present.
        assert_eq!(found.len(), cap, "all {cap} keys should be in cache");
    }
}
