//! Write-behind (write-back) cache with dirty tracking and flush.
//!
//! This module provides a cache that batches writes and lazily flushes dirty
//! entries to a backing store.  Entries are marked *dirty* on insert/update;
//! the caller periodically calls [`WriteBehindCache::flush`] (or
//! `flush_if_needed`) to persist dirty entries.
//!
//! The backing store is abstracted via the [`BackingStore`] trait so the same
//! cache can sit in front of an in-memory map, a file, or a network service.

use std::collections::HashMap;
use std::time::Instant;

// ── BackingStore trait ──────────────────────────────────────────────────────

/// Abstraction over the origin data store that the cache sits in front of.
///
/// Implementations are expected to be synchronous.  For async I/O, the caller
/// should provide a blocking adapter.
pub trait BackingStore {
    /// The key type.
    type Key: Eq + std::hash::Hash + Clone;
    /// The value type.
    type Value: Clone;
    /// Error type returned by store operations.
    type Error: std::fmt::Debug;

    /// Write `(key, value)` to the backing store.
    fn write(&mut self, key: &Self::Key, value: &Self::Value) -> Result<(), Self::Error>;

    /// Read the value for `key` from the backing store (cache miss path).
    fn read(&self, key: &Self::Key) -> Result<Option<Self::Value>, Self::Error>;

    /// Delete `key` from the backing store.
    fn delete(&mut self, key: &Self::Key) -> Result<(), Self::Error>;
}

// ── Internal entry ──────────────────────────────────────────────────────────

struct CacheEntry<V> {
    value: V,
    dirty: bool,
    last_modified: Instant,
}

// ── WriteBehindCache ────────────────────────────────────────────────────────

/// Write-behind cache that defers writes to a [`BackingStore`] until
/// `flush` is called.
///
/// # Dirty tracking
///
/// Every `put` marks the entry as *dirty*.  `flush` iterates all dirty
/// entries, writes them to the backing store, and clears the dirty flag.
/// `flush_if_needed` only flushes when the dirty count exceeds a threshold.
///
/// # Eviction
///
/// Entries are evicted in insertion order (FIFO) when the cache exceeds
/// `capacity`.  **Dirty entries are flushed before eviction** so data is
/// never silently lost.
pub struct WriteBehindCache<S: BackingStore> {
    entries: HashMap<S::Key, CacheEntry<S::Value>>,
    /// Insertion order for FIFO eviction.
    order: Vec<S::Key>,
    capacity: usize,
    store: S,
    /// Number of entries currently marked dirty.
    dirty_count: usize,
    /// Total number of successful flush operations.
    total_flushes: u64,
    /// Total number of entries written to the store across all flushes.
    total_entries_flushed: u64,
}

/// Snapshot of write-behind cache statistics.
#[derive(Debug, Clone)]
pub struct WriteBehindStats {
    /// Number of entries currently in the cache.
    pub entry_count: usize,
    /// Number of dirty (unflushed) entries.
    pub dirty_count: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Total number of flush operations performed.
    pub total_flushes: u64,
    /// Total number of individual entries flushed to the store.
    pub total_entries_flushed: u64,
}

/// Errors that can occur during write-behind cache operations.
#[derive(Debug)]
pub enum WriteBehindError<E: std::fmt::Debug> {
    /// The backing store returned an error.
    StoreError(E),
}

impl<E: std::fmt::Debug> std::fmt::Display for WriteBehindError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StoreError(e) => write!(f, "backing store error: {e:?}"),
        }
    }
}

impl<S: BackingStore> WriteBehindCache<S> {
    /// Create a new write-behind cache with the given capacity and backing
    /// store.
    pub fn new(capacity: usize, store: S) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            capacity: capacity.max(1),
            store,
            dirty_count: 0,
            total_flushes: 0,
            total_entries_flushed: 0,
        }
    }

    /// Insert or update `(key, value)`.  The entry is marked dirty.
    ///
    /// If the cache is at capacity, the oldest entry is evicted (and flushed
    /// if dirty).
    pub fn put(&mut self, key: S::Key, value: S::Value) -> Result<(), WriteBehindError<S::Error>> {
        if self.entries.contains_key(&key) {
            // Update in place.
            if let Some(entry) = self.entries.get_mut(&key) {
                if !entry.dirty {
                    self.dirty_count += 1;
                }
                entry.value = value;
                entry.dirty = true;
                entry.last_modified = Instant::now();
            }
            return Ok(());
        }

        // Evict if at capacity.
        while self.entries.len() >= self.capacity {
            self.evict_oldest()?;
        }

        self.order.push(key.clone());
        self.entries.insert(
            key,
            CacheEntry {
                value,
                dirty: true,
                last_modified: Instant::now(),
            },
        );
        self.dirty_count += 1;
        Ok(())
    }

    /// Look up `key`.  On a cache miss, attempts to load from the backing
    /// store (read-through).
    pub fn get(&mut self, key: &S::Key) -> Result<Option<&S::Value>, WriteBehindError<S::Error>> {
        if self.entries.contains_key(key) {
            return Ok(self.entries.get(key).map(|e| &e.value));
        }
        // Read-through from backing store.
        let value = self.store.read(key).map_err(WriteBehindError::StoreError)?;
        if let Some(v) = value {
            // Cache the value (clean, not dirty).
            while self.entries.len() >= self.capacity {
                self.evict_oldest().map_err(|e| match e {
                    WriteBehindError::StoreError(se) => WriteBehindError::StoreError(se),
                })?;
            }
            self.order.push(key.clone());
            self.entries.insert(
                key.clone(),
                CacheEntry {
                    value: v,
                    dirty: false,
                    last_modified: Instant::now(),
                },
            );
            return Ok(self.entries.get(key).map(|e| &e.value));
        }
        Ok(None)
    }

    /// Remove `key` from the cache and the backing store.
    pub fn delete(&mut self, key: &S::Key) -> Result<bool, WriteBehindError<S::Error>> {
        if let Some(entry) = self.entries.remove(key) {
            self.order.retain(|k| k != key);
            if entry.dirty {
                self.dirty_count = self.dirty_count.saturating_sub(1);
            }
            self.store
                .delete(key)
                .map_err(WriteBehindError::StoreError)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Flush all dirty entries to the backing store.
    ///
    /// Returns the number of entries flushed.
    pub fn flush(&mut self) -> Result<usize, WriteBehindError<S::Error>> {
        let dirty_keys: Vec<S::Key> = self
            .entries
            .iter()
            .filter(|(_, e)| e.dirty)
            .map(|(k, _)| k.clone())
            .collect();
        let count = dirty_keys.len();
        for key in &dirty_keys {
            if let Some(entry) = self.entries.get(key) {
                self.store
                    .write(key, &entry.value)
                    .map_err(WriteBehindError::StoreError)?;
            }
            if let Some(entry) = self.entries.get_mut(key) {
                entry.dirty = false;
            }
        }
        self.dirty_count = 0;
        self.total_flushes += 1;
        self.total_entries_flushed += count as u64;
        Ok(count)
    }

    /// Flush only if the number of dirty entries exceeds `threshold`.
    pub fn flush_if_needed(
        &mut self,
        threshold: usize,
    ) -> Result<usize, WriteBehindError<S::Error>> {
        if self.dirty_count >= threshold {
            self.flush()
        } else {
            Ok(0)
        }
    }

    /// Return the number of dirty entries.
    pub fn dirty_count(&self) -> usize {
        self.dirty_count
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> WriteBehindStats {
        WriteBehindStats {
            entry_count: self.entries.len(),
            dirty_count: self.dirty_count,
            capacity: self.capacity,
            total_flushes: self.total_flushes,
            total_entries_flushed: self.total_entries_flushed,
        }
    }

    /// Return a shared reference to the backing store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Return a mutable reference to the backing store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Returns `true` if the entry for `key` is dirty.
    pub fn is_dirty(&self, key: &S::Key) -> bool {
        self.entries.get(key).map(|e| e.dirty).unwrap_or(false)
    }

    /// Returns `true` if the cache contains an entry for `key`.
    pub fn contains(&self, key: &S::Key) -> bool {
        self.entries.contains_key(key)
    }

    /// Return the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Flush only entries whose dirty age exceeds `max_age`.
    ///
    /// "Dirty age" is the time since the entry was last modified.  This
    /// allows the caller to implement time-based flush policies (e.g. flush
    /// all entries older than 5 seconds).
    ///
    /// Returns the number of entries flushed.
    pub fn flush_older_than(
        &mut self,
        max_age: std::time::Duration,
    ) -> Result<usize, WriteBehindError<S::Error>> {
        let now = Instant::now();
        let old_dirty_keys: Vec<S::Key> = self
            .entries
            .iter()
            .filter(|(_, e)| e.dirty && now.duration_since(e.last_modified) >= max_age)
            .map(|(k, _)| k.clone())
            .collect();
        let count = old_dirty_keys.len();
        for key in &old_dirty_keys {
            if let Some(entry) = self.entries.get(key) {
                self.store
                    .write(key, &entry.value)
                    .map_err(WriteBehindError::StoreError)?;
            }
            if let Some(entry) = self.entries.get_mut(key) {
                entry.dirty = false;
            }
        }
        self.dirty_count = self.dirty_count.saturating_sub(count);
        if count > 0 {
            self.total_flushes += 1;
            self.total_entries_flushed += count as u64;
        }
        Ok(count)
    }

    /// Return a list of all dirty keys.
    pub fn dirty_keys(&self) -> Vec<S::Key> {
        self.entries
            .iter()
            .filter(|(_, e)| e.dirty)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Mark an entry as clean without writing to the backing store.
    ///
    /// Useful when the caller knows the backing store is already up to date
    /// (e.g. after an external write).  Returns `true` if the entry was
    /// dirty and is now clean.
    pub fn mark_clean(&mut self, key: &S::Key) -> bool {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.dirty {
                entry.dirty = false;
                self.dirty_count = self.dirty_count.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Return the capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Evict the oldest entry (FIFO).  Flushes it if dirty.
    fn evict_oldest(&mut self) -> Result<(), WriteBehindError<S::Error>> {
        if self.order.is_empty() {
            return Ok(());
        }
        let key = self.order.remove(0);
        if let Some(entry) = self.entries.remove(&key) {
            if entry.dirty {
                self.store
                    .write(&key, &entry.value)
                    .map_err(WriteBehindError::StoreError)?;
                self.dirty_count = self.dirty_count.saturating_sub(1);
                self.total_entries_flushed += 1;
            }
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// In-memory backing store for testing.
    #[derive(Clone)]
    struct MemStore {
        data: Arc<Mutex<HashMap<String, String>>>,
    }

    impl MemStore {
        fn new() -> Self {
            Self {
                data: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn snapshot(&self) -> HashMap<String, String> {
            let guard = self.data.lock().unwrap_or_else(|p| p.into_inner());
            guard.clone()
        }
    }

    impl BackingStore for MemStore {
        type Key = String;
        type Value = String;
        type Error = String;

        fn write(&mut self, key: &String, value: &String) -> Result<(), String> {
            let mut guard = self.data.lock().unwrap_or_else(|p| p.into_inner());
            guard.insert(key.clone(), value.clone());
            Ok(())
        }

        fn read(&self, key: &String) -> Result<Option<String>, String> {
            let guard = self.data.lock().unwrap_or_else(|p| p.into_inner());
            Ok(guard.get(key).cloned())
        }

        fn delete(&mut self, key: &String) -> Result<(), String> {
            let mut guard = self.data.lock().unwrap_or_else(|p| p.into_inner());
            guard.remove(key);
            Ok(())
        }
    }

    // 1. Basic put and get
    #[test]
    fn test_put_and_get() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("k1".to_string(), "v1".to_string()).ok();
        let val = cache.get(&"k1".to_string()).ok().flatten();
        assert_eq!(val, Some(&"v1".to_string()));
    }

    // 2. Dirty tracking
    #[test]
    fn test_dirty_tracking() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        assert!(cache.is_dirty(&"a".to_string()));
        assert_eq!(cache.dirty_count(), 1);
    }

    // 3. Flush writes to store
    #[test]
    fn test_flush_writes_to_store() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store.clone());
        cache.put("x".to_string(), "42".to_string()).ok();
        let flushed = cache.flush().ok();
        assert_eq!(flushed, Some(1));
        assert!(!cache.is_dirty(&"x".to_string()));
        let snap = store.snapshot();
        assert_eq!(snap.get("x"), Some(&"42".to_string()));
    }

    // 4. Flush clears dirty count
    #[test]
    fn test_flush_clears_dirty() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        cache.flush().ok();
        assert_eq!(cache.dirty_count(), 0);
    }

    // 5. flush_if_needed respects threshold
    #[test]
    fn test_flush_if_needed() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        let flushed = cache.flush_if_needed(5).ok();
        assert_eq!(flushed, Some(0)); // threshold not met
        cache.put("b".to_string(), "2".to_string()).ok();
        cache.put("c".to_string(), "3".to_string()).ok();
        let flushed = cache.flush_if_needed(2).ok();
        assert_eq!(flushed, Some(3)); // now all 3 dirty entries flushed
    }

    // 6. Eviction flushes dirty entries
    #[test]
    fn test_eviction_flushes_dirty() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(2, store.clone());
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        // This should evict "a" and flush it.
        cache.put("c".to_string(), "3".to_string()).ok();
        let snap = store.snapshot();
        assert_eq!(snap.get("a"), Some(&"1".to_string()));
    }

    // 7. Delete removes from cache and store
    #[test]
    fn test_delete() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store.clone());
        cache.put("k".to_string(), "v".to_string()).ok();
        cache.flush().ok();
        let deleted = cache.delete(&"k".to_string()).ok();
        assert_eq!(deleted, Some(true));
        let snap = store.snapshot();
        assert!(!snap.contains_key("k"));
    }

    // 8. Read-through on miss
    #[test]
    fn test_read_through() {
        let store = MemStore::new();
        {
            let mut guard = store.data.lock().unwrap_or_else(|p| p.into_inner());
            guard.insert("pre".to_string(), "existing".to_string());
        }
        let mut cache = WriteBehindCache::new(10, store);
        let val = cache.get(&"pre".to_string()).ok().flatten();
        assert_eq!(val, Some(&"existing".to_string()));
        // Now it should be cached (clean).
        assert!(!cache.is_dirty(&"pre".to_string()));
    }

    // 9. Update marks entry dirty again after flush
    #[test]
    fn test_update_re_dirties() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.flush().ok();
        assert!(!cache.is_dirty(&"a".to_string()));
        cache.put("a".to_string(), "2".to_string()).ok();
        assert!(cache.is_dirty(&"a".to_string()));
    }

    // 10. Stats
    #[test]
    fn test_stats() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        cache.flush().ok();
        let s = cache.stats();
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.dirty_count, 0);
        assert_eq!(s.total_flushes, 1);
        assert_eq!(s.total_entries_flushed, 2);
    }

    // 11. Delete absent key
    #[test]
    fn test_delete_absent() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        let deleted = cache.delete(&"ghost".to_string()).ok();
        assert_eq!(deleted, Some(false));
    }

    // 12. Get absent key returns None
    #[test]
    fn test_get_absent() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        let val = cache.get(&"nope".to_string()).ok().flatten();
        assert!(val.is_none());
    }

    // ── Enhanced write-behind tests ─────────────────────────────────────────

    // 13. contains
    #[test]
    fn test_contains() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("x".to_string(), "val".to_string()).ok();
        assert!(cache.contains(&"x".to_string()));
        assert!(!cache.contains(&"y".to_string()));
    }

    // 14. len and is_empty
    #[test]
    fn test_len_and_is_empty() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
    }

    // 15. flush_older_than only flushes old entries
    #[test]
    fn test_flush_older_than() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store.clone());
        cache.put("old".to_string(), "old_val".to_string()).ok();
        // Sleep to age the entry
        std::thread::sleep(std::time::Duration::from_millis(50));
        cache.put("new".to_string(), "new_val".to_string()).ok();
        // Flush entries older than 30ms
        let flushed = cache
            .flush_older_than(std::time::Duration::from_millis(30))
            .ok();
        assert_eq!(flushed, Some(1));
        // "old" should be clean, "new" still dirty
        assert!(!cache.is_dirty(&"old".to_string()));
        assert!(cache.is_dirty(&"new".to_string()));
        // Store should have "old"
        let snap = store.snapshot();
        assert!(snap.contains_key("old"));
    }

    // 16. flush_older_than with zero duration flushes all dirty
    #[test]
    fn test_flush_older_than_zero() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        let flushed = cache
            .flush_older_than(std::time::Duration::from_millis(0))
            .ok();
        assert_eq!(flushed, Some(2));
        assert_eq!(cache.dirty_count(), 0);
    }

    // 17. dirty_keys returns correct set
    #[test]
    fn test_dirty_keys() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        cache.put("c".to_string(), "3".to_string()).ok();
        cache.flush().ok();
        // Re-dirty one entry
        cache.put("b".to_string(), "updated".to_string()).ok();
        let dirty = cache.dirty_keys();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0], "b");
    }

    // 18. mark_clean without store write
    #[test]
    fn test_mark_clean() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store.clone());
        cache.put("x".to_string(), "val".to_string()).ok();
        assert!(cache.is_dirty(&"x".to_string()));
        assert!(cache.mark_clean(&"x".to_string()));
        assert!(!cache.is_dirty(&"x".to_string()));
        assert_eq!(cache.dirty_count(), 0);
        // Store should NOT have the entry (mark_clean doesn't write)
        let snap = store.snapshot();
        assert!(!snap.contains_key("x"));
    }

    // 19. mark_clean on clean entry returns false
    #[test]
    fn test_mark_clean_already_clean() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.flush().ok();
        assert!(!cache.mark_clean(&"a".to_string()));
    }

    // 20. mark_clean on absent entry returns false
    #[test]
    fn test_mark_clean_absent() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        assert!(!cache.mark_clean(&"ghost".to_string()));
    }

    // 21. capacity getter
    #[test]
    fn test_capacity() {
        let store = MemStore::new();
        let cache: WriteBehindCache<MemStore> = WriteBehindCache::new(42, store);
        assert_eq!(cache.capacity(), 42);
    }

    // 22. Multiple flushes accumulate stats
    #[test]
    fn test_multiple_flushes_stats() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(10, store);
        cache.put("a".to_string(), "1".to_string()).ok();
        cache.flush().ok();
        cache.put("b".to_string(), "2".to_string()).ok();
        cache.flush().ok();
        let s = cache.stats();
        assert_eq!(s.total_flushes, 2);
        assert_eq!(s.total_entries_flushed, 2);
    }

    // 23. Eviction cascade: filling beyond capacity flushes all dirty entries
    #[test]
    fn test_eviction_cascade() {
        let store = MemStore::new();
        let mut cache = WriteBehindCache::new(3, store.clone());
        for i in 0..5 {
            cache.put(format!("k{i}"), format!("v{i}")).ok();
        }
        // At least the first 2 entries should have been evicted and flushed
        let snap = store.snapshot();
        assert!(snap.contains_key("k0"), "evicted k0 should be in store");
        assert!(snap.contains_key("k1"), "evicted k1 should be in store");
    }

    // 24. Read-through caches as clean
    #[test]
    fn test_read_through_is_clean() {
        let store = MemStore::new();
        {
            let mut guard = store.data.lock().unwrap_or_else(|p| p.into_inner());
            guard.insert("existing".to_string(), "value".to_string());
        }
        let mut cache = WriteBehindCache::new(10, store);
        cache.get(&"existing".to_string()).ok();
        assert!(!cache.is_dirty(&"existing".to_string()));
        assert_eq!(cache.dirty_count(), 0);
    }

    // 25. Store reference accessors
    #[test]
    fn test_store_accessors() {
        let store = MemStore::new();
        let cache = WriteBehindCache::new(10, store);
        let _store_ref = cache.store();
        // Just verify it compiles and doesn't panic
    }
}
