//! 2Q (two-queue) eviction policy — a scan-resistant alternative to LRU.
//!
//! The 2Q algorithm maintains three queues:
//!
//! | Queue | Purpose |
//! |-------|---------|
//! | **A1in** | FIFO buffer for recently inserted items (first access) |
//! | **A1out** | Ghost queue: tracks keys recently evicted from A1in |
//! | **Am** | LRU queue for items accessed at least twice (promoted from A1in or A1out hit) |
//!
//! On a cache miss:
//! - If the key is in A1out (ghost hit), it is admitted directly to Am.
//! - Otherwise it enters A1in.
//!
//! On a cache hit:
//! - If in Am, the entry is moved to the MRU position of Am.
//! - If in A1in, no promotion happens yet (FIFO within A1in).
//!
//! This prevents sequential scans from polluting the long-term cache (Am).
//!
//! # Reference
//!
//! Theodore Johnson and Dennis Shasha, "2Q: A Low Overhead High Performance
//! Buffer Management Replacement Algorithm," VLDB 1994.

use std::collections::{HashMap, HashSet, VecDeque};

// ── TwoQueueCache ───────────────────────────────────────────────────────────

/// A 2Q cache with configurable total capacity and A1in/A1out sizing.
///
/// # Type parameters
/// * `K` — key type (must be `Eq + Hash + Clone`).
/// * `V` — value type.
pub struct TwoQueueCache<K: Eq + std::hash::Hash + Clone, V> {
    /// Maximum total entries (Am + A1in).
    capacity: usize,
    /// Target size for A1in (the FIFO admission buffer), typically ~25% of
    /// capacity.
    a1in_capacity: usize,
    /// Target size for A1out (the ghost list), typically ~50% of capacity.
    a1out_capacity: usize,

    // ── A1in: FIFO buffer for first-access items ────────────────────────
    /// Maps keys in A1in to their values.
    a1in_map: HashMap<K, V>,
    /// FIFO order of keys in A1in (front = oldest).
    a1in_queue: VecDeque<K>,

    // ── Am: LRU buffer for frequently accessed items ────────────────────
    /// Maps keys in Am to their values.
    am_map: HashMap<K, V>,
    /// LRU order of keys in Am (front = LRU, back = MRU).
    am_queue: VecDeque<K>,

    // ── A1out: ghost list tracking recently evicted A1in keys ───────────
    /// FIFO queue of ghost keys.
    a1out_queue: VecDeque<K>,
    /// Set of keys in A1out for O(1) membership tests.
    a1out_set: HashSet<K>,

    // ── Stats ───────────────────────────────────────────────────────────
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Snapshot of 2Q cache statistics.
#[derive(Debug, Clone)]
pub struct TwoQueueStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Number of entries in A1in.
    pub a1in_len: usize,
    /// Number of entries in Am.
    pub am_len: usize,
    /// Number of ghost entries in A1out.
    pub a1out_len: usize,
    /// Total capacity.
    pub capacity: usize,
}

impl<K: Eq + std::hash::Hash + Clone, V> TwoQueueCache<K, V> {
    /// Create a new 2Q cache with the given capacity.
    ///
    /// A1in is sized to ~25% of capacity and A1out to ~50%.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(2);
        let a1in_cap = (cap / 4).max(1);
        let a1out_cap = (cap / 2).max(1);
        Self {
            capacity: cap,
            a1in_capacity: a1in_cap,
            a1out_capacity: a1out_cap,
            a1in_map: HashMap::new(),
            a1in_queue: VecDeque::new(),
            am_map: HashMap::new(),
            am_queue: VecDeque::new(),
            a1out_queue: VecDeque::new(),
            a1out_set: HashSet::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Create a 2Q cache with explicit A1in and A1out sizing.
    pub fn with_queue_sizes(capacity: usize, a1in_capacity: usize, a1out_capacity: usize) -> Self {
        let cap = capacity.max(2);
        Self {
            capacity: cap,
            a1in_capacity: a1in_capacity.max(1),
            a1out_capacity: a1out_capacity.max(1),
            a1in_map: HashMap::new(),
            a1in_queue: VecDeque::new(),
            am_map: HashMap::new(),
            am_queue: VecDeque::new(),
            a1out_queue: VecDeque::new(),
            a1out_set: HashSet::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up `key`.  On a hit in Am, promote to MRU.  On a hit in A1in,
    /// the entry stays in place (FIFO semantics).
    pub fn get(&mut self, key: &K) -> Option<&V> {
        // Check Am first (more likely to hit).
        if self.am_map.contains_key(key) {
            self.hits += 1;
            // Promote to MRU in Am.
            self.am_queue.retain(|k| k != key);
            self.am_queue.push_back(key.clone());
            return self.am_map.get(key);
        }
        // Check A1in.
        if self.a1in_map.contains_key(key) {
            self.hits += 1;
            // No promotion within A1in — FIFO semantics.
            return self.a1in_map.get(key);
        }
        self.misses += 1;
        None
    }

    /// Insert `(key, value)` into the cache.
    ///
    /// - If `key` is already in Am or A1in, its value is updated.
    /// - If `key` is in A1out (ghost hit), it is admitted to Am.
    /// - Otherwise it enters A1in.
    pub fn insert(&mut self, key: K, value: V) {
        // Already in Am → update value and promote.
        if self.am_map.contains_key(&key) {
            self.am_map.insert(key.clone(), value);
            self.am_queue.retain(|k| k != &key);
            self.am_queue.push_back(key);
            return;
        }
        // Already in A1in → update value (no position change).
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.a1in_map.entry(key.clone())
        {
            e.insert(value);
            return;
        }
        // Ghost hit in A1out → promote to Am.
        if self.a1out_set.contains(&key) {
            self.a1out_set.remove(&key);
            self.a1out_queue.retain(|k| k != &key);
            self.ensure_room_am();
            self.am_map.insert(key.clone(), value);
            self.am_queue.push_back(key);
            return;
        }
        // Brand new key → insert into A1in.
        self.ensure_room_a1in();
        self.a1in_map.insert(key.clone(), value);
        self.a1in_queue.push_back(key);
    }

    /// Remove `key` from the cache entirely (including ghost list).
    pub fn remove(&mut self, key: &K) -> Option<V> {
        // Check Am.
        if let Some(v) = self.am_map.remove(key) {
            self.am_queue.retain(|k| k != key);
            return Some(v);
        }
        // Check A1in.
        if let Some(v) = self.a1in_map.remove(key) {
            self.a1in_queue.retain(|k| k != key);
            return Some(v);
        }
        // Remove from ghost list too.
        self.a1out_set.remove(key);
        self.a1out_queue.retain(|k| k != key);
        None
    }

    /// Returns `true` if the cache contains `key` (not counting ghosts).
    pub fn contains(&self, key: &K) -> bool {
        self.am_map.contains_key(key) || self.a1in_map.contains_key(key)
    }

    /// Total number of live entries.
    pub fn len(&self) -> usize {
        self.am_map.len() + self.a1in_map.len()
    }

    /// Returns `true` when the cache has no live entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> TwoQueueStats {
        TwoQueueStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            a1in_len: self.a1in_map.len(),
            am_len: self.am_map.len(),
            a1out_len: self.a1out_set.len(),
            capacity: self.capacity,
        }
    }

    /// Peek at `key` without updating access metadata or LRU order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        if let Some(v) = self.am_map.get(key) {
            return Some(v);
        }
        self.a1in_map.get(key)
    }

    /// Return a mutable reference to the value for `key` without changing
    /// queue positions.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(v) = self.am_map.get_mut(key) {
            self.hits += 1;
            return Some(v);
        }
        if let Some(v) = self.a1in_map.get_mut(key) {
            self.hits += 1;
            return Some(v);
        }
        self.misses += 1;
        None
    }

    /// Remove all entries, ghost list included.
    pub fn clear(&mut self) {
        self.a1in_map.clear();
        self.a1in_queue.clear();
        self.am_map.clear();
        self.am_queue.clear();
        self.a1out_queue.clear();
        self.a1out_set.clear();
    }

    /// Return `true` if `key` is in the ghost list (A1out).
    pub fn is_ghost(&self, key: &K) -> bool {
        self.a1out_set.contains(key)
    }

    /// Return the number of entries in A1in.
    pub fn a1in_len(&self) -> usize {
        self.a1in_map.len()
    }

    /// Return the number of entries in Am.
    pub fn am_len(&self) -> usize {
        self.am_map.len()
    }

    /// Return the number of ghost entries in A1out.
    pub fn a1out_len(&self) -> usize {
        self.a1out_set.len()
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Make room in A1in by evicting the oldest entry to A1out (ghost list).
    fn ensure_room_a1in(&mut self) {
        // Also respect total capacity.
        while self.len() >= self.capacity {
            self.evict_from_am_or_a1in();
        }
        while self.a1in_map.len() >= self.a1in_capacity {
            self.evict_a1in_to_ghost();
        }
    }

    /// Make room in Am.
    fn ensure_room_am(&mut self) {
        while self.len() >= self.capacity {
            self.evict_from_am_or_a1in();
        }
    }

    /// Evict the oldest entry from A1in, moving its key to A1out.
    fn evict_a1in_to_ghost(&mut self) {
        if let Some(key) = self.a1in_queue.pop_front() {
            self.a1in_map.remove(&key);
            self.evictions += 1;
            // Add to ghost list.
            self.a1out_queue.push_back(key.clone());
            self.a1out_set.insert(key);
            // Trim ghost list if over capacity.
            while self.a1out_set.len() > self.a1out_capacity {
                if let Some(ghost) = self.a1out_queue.pop_front() {
                    self.a1out_set.remove(&ghost);
                }
            }
        }
    }

    /// Evict from A1in (FIFO) or Am (LRU) to make room for total capacity.
    ///
    /// Per the 2Q paper, we prefer evicting from A1in first (since those are
    /// one-access items) to protect the Am hot set.  This is what gives 2Q
    /// its scan-resistance property.
    fn evict_from_am_or_a1in(&mut self) {
        if !self.a1in_map.is_empty() {
            self.evict_a1in_to_ghost();
        } else if let Some(key) = self.am_queue.pop_front() {
            self.am_map.remove(&key);
            self.evictions += 1;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Basic insert and get
    #[test]
    fn test_insert_and_get() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    // 2. Miss returns None
    #[test]
    fn test_miss() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        assert_eq!(cache.get(&99), None);
    }

    // 3. Contains
    #[test]
    fn test_contains() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("hello", 1);
        assert!(cache.contains(&"hello"));
        assert!(!cache.contains(&"world"));
    }

    // 4. Len and is_empty
    #[test]
    fn test_len_and_is_empty() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        assert!(cache.is_empty());
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.len(), 2);
    }

    // 5. Remove
    #[test]
    fn test_remove() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("x", 42);
        let removed = cache.remove(&"x");
        assert_eq!(removed, Some(42));
        assert!(!cache.contains(&"x"));
    }

    // 6. Scan resistance: sequential scan does not pollute Am
    #[test]
    fn test_scan_resistance() {
        // capacity=8, A1in=2, A1out=8 (generous ghost list).
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(8, 2, 8);
        // Step 1: Insert 3 hot items → they enter A1in then flow to ghost
        // as more items push them out.
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30); // A1in is full (cap=2), key 1 evicted to ghost.
        cache.insert(4, 40); // key 2 evicted to ghost.
        cache.insert(5, 50); // key 3 evicted to ghost.
                             // Now keys 1,2,3 should be in A1out.
        assert!(cache.is_ghost(&1), "key 1 should be in ghost");
        assert!(cache.is_ghost(&2), "key 2 should be in ghost");
        assert!(cache.is_ghost(&3), "key 3 should be in ghost");
        // Step 2: Re-insert hot items → ghost hit → promoted to Am.
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        assert!(
            cache.am_len() >= 3,
            "Am should have 3 entries after ghost hits"
        );
        // Step 3: Flood with sequential scan.
        for i in 1000..1050 {
            cache.insert(i, i);
        }
        // Hot items in Am should survive the sequential scan.
        assert!(
            cache.am_len() >= 3,
            "Am should still have entries after scan (got {})",
            cache.am_len()
        );
        // Verify hot keys are still accessible.
        for &k in &[1, 2, 3] {
            assert!(cache.contains(&k), "hot key {k} should survive scan");
        }
    }

    // 7. Ghost hit promotes to Am
    #[test]
    fn test_ghost_hit_promotes_to_am() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 2);
        cache.insert(1, 10); // A1in
        cache.insert(2, 20); // evicts 1 to ghost (A1in cap=1)
                             // Now key 1 is in A1out.
        cache.insert(1, 100); // ghost hit → Am
        let s = cache.stats();
        assert_eq!(s.am_len, 1, "ghost hit should put key in Am");
        assert_eq!(cache.get(&1), Some(&100));
    }

    // 8. Stats counters
    #[test]
    fn test_stats() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        cache.insert(1, 10);
        cache.get(&1); // hit
        cache.get(&99); // miss
        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    // 9. Update existing key in A1in
    #[test]
    fn test_update_in_a1in() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("k", 1);
        cache.insert("k", 2);
        assert_eq!(cache.get(&"k"), Some(&2));
        assert_eq!(cache.len(), 1);
    }

    // 10. Update existing key in Am
    #[test]
    fn test_update_in_am() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 2);
        cache.insert(1, 10);
        cache.insert(2, 20); // evicts 1 to ghost
        cache.insert(1, 100); // ghost hit → Am
        cache.insert(1, 200); // update in Am
        assert_eq!(cache.get(&1), Some(&200));
    }

    // 11. Capacity is respected
    #[test]
    fn test_capacity() {
        let mut cache: TwoQueueCache<usize, usize> = TwoQueueCache::new(5);
        for i in 0..100 {
            cache.insert(i, i);
        }
        assert!(cache.len() <= 5, "len {} exceeds capacity 5", cache.len());
    }

    // 12. Remove absent key returns None
    #[test]
    fn test_remove_absent() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        assert_eq!(cache.remove(&42), None);
    }

    // 13. Custom queue sizes
    #[test]
    fn test_custom_queue_sizes() {
        let cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(100, 10, 20);
        let s = cache.stats();
        assert_eq!(s.capacity, 100);
    }

    // 14. Am LRU eviction under Am pressure
    #[test]
    fn test_am_lru_eviction() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        // Build up Am entries via ghost hits.
        for i in 0..10 {
            cache.insert(i, i);
        }
        // Re-insert early keys that are now in ghost.
        for i in 0..4 {
            cache.insert(i, i * 100);
        }
        // Am should have at most capacity entries total.
        assert!(cache.len() <= 4);
    }

    // 15. Hit in Am promotes to MRU
    #[test]
    fn test_am_hit_promotes_to_mru() {
        // capacity=6, A1in=1, A1out=6 (large ghost to keep evicted keys).
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(6, 1, 6);
        // Insert items — they enter A1in (cap=1), overflow to ghost.
        for i in 0..6 {
            cache.insert(i, i);
        }
        // Re-insert some items → ghost hit → promoted to Am.
        for i in 0..4 {
            cache.insert(i, i * 10);
        }
        assert!(cache.am_len() > 0, "Am should have entries");
        // Access key 0 to promote to MRU within Am.
        cache.get(&0);
        // Key 0 should still be present.
        let val = cache.get(&0);
        assert!(val.is_some(), "key 0 should be promoted to MRU");
    }

    // ── Enhanced 2Q tests ───────────────────────────────────────────────────

    // 16. peek does not update LRU order
    #[test]
    fn test_peek_no_side_effects() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("a", 1);
        let val = cache.peek(&"a");
        assert_eq!(val, Some(&1));
        // Stats should not change from peek
        let s = cache.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
    }

    // 17. peek absent key returns None
    #[test]
    fn test_peek_absent() {
        let cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        assert_eq!(cache.peek(&99), None);
    }

    // 18. get_mut allows in-place mutation
    #[test]
    fn test_get_mut() {
        let mut cache: TwoQueueCache<&str, i32> = TwoQueueCache::new(10);
        cache.insert("a", 1);
        if let Some(v) = cache.get_mut(&"a") {
            *v = 42;
        }
        assert_eq!(cache.get(&"a"), Some(&42));
    }

    // 19. get_mut absent key records miss
    #[test]
    fn test_get_mut_absent() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        assert!(cache.get_mut(&99).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    // 20. clear removes everything
    #[test]
    fn test_clear() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(10);
        for i in 0..10 {
            cache.insert(i, i);
        }
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.a1out_len(), 0);
    }

    // 21. is_ghost detects ghost entries
    #[test]
    fn test_is_ghost() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10); // A1in
        cache.insert(2, 20); // evicts 1 to ghost (A1in cap=1)
        assert!(cache.is_ghost(&1));
        assert!(!cache.is_ghost(&2));
        assert!(!cache.is_ghost(&99));
    }

    // 22. a1in_len / am_len / a1out_len
    #[test]
    fn test_queue_length_accessors() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10);
        assert_eq!(cache.a1in_len(), 1);
        assert_eq!(cache.am_len(), 0);
        cache.insert(2, 20); // evicts 1 to ghost
        assert_eq!(cache.a1in_len(), 1);
        assert_eq!(cache.a1out_len(), 1);
        // Ghost hit: re-insert 1 → Am
        cache.insert(1, 100);
        assert_eq!(cache.am_len(), 1);
        assert_eq!(cache.a1out_len(), 0);
    }

    // 23. Large sequential workload maintains capacity
    #[test]
    fn test_large_sequential_workload() {
        let cap = 20;
        let mut cache: TwoQueueCache<usize, usize> = TwoQueueCache::new(cap);
        for i in 0..1000 {
            cache.insert(i, i * 2);
        }
        assert!(
            cache.len() <= cap,
            "cache exceeds capacity: {}",
            cache.len()
        );
    }

    // 24. Ghost list respects a1out_capacity
    #[test]
    fn test_ghost_list_bounded() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 3);
        // Insert many items to fill ghost list
        for i in 0..20 {
            cache.insert(i, i);
        }
        assert!(cache.a1out_len() <= 3, "ghost list exceeds capacity");
    }

    // 25. peek on item in Am
    #[test]
    fn test_peek_am_item() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10);
        cache.insert(2, 20); // evicts 1 to ghost
        cache.insert(1, 100); // ghost hit → Am
        assert_eq!(cache.peek(&1), Some(&100));
    }

    // 26. Remove from Am
    #[test]
    fn test_remove_from_am() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10);
        cache.insert(2, 20); // evicts 1 to ghost
        cache.insert(1, 100); // ghost hit → Am
        let removed = cache.remove(&1);
        assert_eq!(removed, Some(100));
        assert!(!cache.contains(&1));
    }

    // 27. Remove also clears ghost entry
    #[test]
    fn test_remove_clears_ghost() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10);
        cache.insert(2, 20); // evicts 1 to ghost
        assert!(cache.is_ghost(&1));
        cache.remove(&1); // should also clear ghost
        assert!(!cache.is_ghost(&1));
    }

    // 28. Stats evictions counter
    #[test]
    fn test_stats_evictions() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::new(4);
        for i in 0..10 {
            cache.insert(i, i);
        }
        assert!(cache.stats().evictions > 0, "should have evictions");
    }

    // 29. Mixed Am and A1in access pattern
    #[test]
    fn test_mixed_access_pattern() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(6, 2, 4);
        // Insert items
        for i in 0..6 {
            cache.insert(i, i * 10);
        }
        // Some items flow to ghost, re-insert for Am promotion
        for i in 0..3 {
            cache.insert(i, i * 100);
        }
        // Access items in Am
        for i in 0..3 {
            let val = cache.get(&i);
            assert!(val.is_some(), "key {i} should be accessible");
        }
        assert!(cache.stats().hits >= 3);
    }

    // 30. get_mut on Am entry counts as hit
    #[test]
    fn test_get_mut_am_hit() {
        let mut cache: TwoQueueCache<i32, i32> = TwoQueueCache::with_queue_sizes(4, 1, 4);
        cache.insert(1, 10);
        cache.insert(2, 20); // evicts 1 to ghost
        cache.insert(1, 100); // ghost hit → Am
        if let Some(v) = cache.get_mut(&1) {
            *v = 999;
        }
        assert_eq!(cache.peek(&1), Some(&999));
        assert!(cache.stats().hits >= 1);
    }
}
