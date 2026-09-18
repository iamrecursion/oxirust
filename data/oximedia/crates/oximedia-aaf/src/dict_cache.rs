//! Dictionary cache for AAF processing
//!
//! Provides an LRU-like cache for AAF dictionary entries to avoid repeated
//! lookups in large AAF files. Caches class, property, type, and data
//! definition lookups keyed by their string names.

use crate::dictionary::{
    Auid, ClassDefinition, DataDefinition, PropertyDefinition, TypeDefinition,
};
use std::collections::{HashMap, VecDeque};

/// Default maximum number of entries in the cache.
const DEFAULT_CAPACITY: usize = 256;

/// A cached dictionary entry — one of the four AAF definition kinds.
#[derive(Debug, Clone)]
pub enum CachedEntry {
    /// A cached class definition
    Class(ClassDefinition),
    /// A cached property definition
    Property(PropertyDefinition),
    /// A cached type definition
    Type(TypeDefinition),
    /// A cached data definition
    DataDef(DataDefinition),
    /// A generic AUID cached by name (e.g. for quick AUID lookup)
    Auid {
        /// The AUID value
        auid: Auid,
        /// Human-readable name associated with this AUID
        name: String,
    },
}

impl CachedEntry {
    /// Return the name associated with this entry (for display / logging).
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Class(c) => &c.name,
            Self::Property(p) => &p.name,
            Self::Type(t) => &t.name,
            Self::DataDef(d) => &d.name,
            Self::Auid { name, .. } => name,
        }
    }

    /// Return the AUID associated with this entry, if any.
    #[must_use]
    pub fn auid(&self) -> Auid {
        match self {
            Self::Class(c) => c.auid,
            Self::Property(p) => p.auid,
            Self::Type(t) => t.auid,
            Self::DataDef(d) => d.auid,
            Self::Auid { auid, .. } => *auid,
        }
    }
}

/// LRU cache for AAF dictionary entries keyed by name string.
///
/// Maintains a bounded cache of recently used dictionary entries to avoid
/// repeated linear scans over the `Dictionary` maps during AAF parsing.
/// When the cache is full the least-recently-used entry is evicted.
///
/// # Example
///
/// ```rust
/// use oximedia_aaf::dict_cache::{DictCache, CachedEntry};
/// use oximedia_aaf::dictionary::{Auid, DataDefinition};
///
/// let mut cache = DictCache::new(64);
/// let entry = CachedEntry::DataDef(DataDefinition::new(Auid::PICTURE, "Picture"));
/// cache.insert("Picture".to_string(), entry);
///
/// assert!(cache.get("Picture").is_some());
/// ```
pub struct DictCache {
    /// Maximum number of entries before eviction
    capacity: usize,
    /// The cached entries, keyed by name
    store: HashMap<String, CachedEntry>,
    /// Ordered queue for LRU tracking (front = oldest)
    order: VecDeque<String>,
}

impl DictCache {
    /// Create a new cache with the given capacity.
    ///
    /// If `capacity` is 0 the cache behaves as if it always misses; every
    /// `insert` call is a no-op and `get` always returns `None`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            store: HashMap::with_capacity(capacity.min(DEFAULT_CAPACITY)),
            order: VecDeque::with_capacity(capacity.min(DEFAULT_CAPACITY)),
        }
    }

    /// Create a cache with the default capacity (`256` entries).
    #[must_use]
    pub fn default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    /// Look up an entry by its name key.
    ///
    /// Returns a reference to the `CachedEntry` if the name is present,
    /// moving that key to the most-recently-used position.
    #[must_use]
    pub fn get(&mut self, key: &str) -> Option<&CachedEntry> {
        if !self.store.contains_key(key) {
            return None;
        }

        // Promote key to back of LRU queue (most recently used)
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let promoted = self.order.remove(pos)?;
            self.order.push_back(promoted);
        }

        self.store.get(key)
    }

    /// Look up an entry by name without updating LRU order.
    ///
    /// Useful for read-only inspection where cache ordering should not change.
    #[must_use]
    pub fn peek(&self, key: &str) -> Option<&CachedEntry> {
        self.store.get(key)
    }

    /// Insert a new entry into the cache.
    ///
    /// If an entry with the same key already exists it is replaced (and the
    /// LRU order for that key is refreshed). If the cache is at capacity the
    /// least-recently-used entry is evicted first.
    pub fn insert(&mut self, key: String, value: CachedEntry) {
        if self.capacity == 0 {
            return;
        }

        if self.store.contains_key(&key) {
            // Update existing: replace value, refresh LRU position
            self.store.insert(key.clone(), value);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
            return;
        }

        // Evict LRU entry if at capacity
        if self.store.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.store.remove(&evicted);
            }
        }

        self.store.insert(key.clone(), value);
        self.order.push_back(key);
    }

    /// Remove an entry by key, returning it if present.
    pub fn remove(&mut self, key: &str) -> Option<CachedEntry> {
        if let Some(entry) = self.store.remove(key) {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
            Some(entry)
        } else {
            None
        }
    }

    /// Return the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return the configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.store.clear();
        self.order.clear();
    }

    /// Return `true` if the cache contains an entry for `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    /// Drain all entries as an iterator of `(String, CachedEntry)` pairs.
    pub fn drain(&mut self) -> impl Iterator<Item = (String, CachedEntry)> + '_ {
        self.order.clear();
        self.store.drain()
    }
}

impl std::fmt::Debug for DictCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DictCache")
            .field("capacity", &self.capacity)
            .field("len", &self.store.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::DataDefinition;

    fn make_entry(name: &str) -> CachedEntry {
        CachedEntry::DataDef(DataDefinition::new(Auid::null(), name))
    }

    #[test]
    fn test_new_cache_is_empty() {
        let cache = DictCache::new(16);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 16);
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = DictCache::new(16);
        cache.insert("Picture".to_string(), make_entry("Picture"));
        let entry = cache.get("Picture");
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.name()), Some("Picture"));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let mut cache = DictCache::new(16);
        assert!(cache.get("NoSuchKey").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = DictCache::new(3);
        cache.insert("a".to_string(), make_entry("a"));
        cache.insert("b".to_string(), make_entry("b"));
        cache.insert("c".to_string(), make_entry("c"));

        // "a" is the LRU; inserting "d" should evict "a"
        cache.insert("d".to_string(), make_entry("d"));
        assert!(!cache.contains("a"), "LRU entry 'a' should be evicted");
        assert!(cache.contains("b"));
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_get_promotes_lru() {
        let mut cache = DictCache::new(3);
        cache.insert("a".to_string(), make_entry("a"));
        cache.insert("b".to_string(), make_entry("b"));
        cache.insert("c".to_string(), make_entry("c"));

        // Access "a" to make it most recently used
        let _ = cache.get("a");

        // Now "b" should be the LRU; inserting "d" evicts "b"
        cache.insert("d".to_string(), make_entry("d"));
        assert!(cache.contains("a"), "'a' should survive (was promoted)");
        assert!(!cache.contains("b"), "'b' should be evicted");
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_insert_replaces_existing() {
        let mut cache = DictCache::new(16);
        cache.insert("Sound".to_string(), make_entry("Sound"));
        cache.insert("Sound".to_string(), make_entry("Sound-v2"));
        // Should still be only one entry
        assert_eq!(cache.len(), 1);
        let entry = cache.peek("Sound");
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.name()), Some("Sound-v2"));
    }

    #[test]
    fn test_remove() {
        let mut cache = DictCache::new(16);
        cache.insert("Timecode".to_string(), make_entry("Timecode"));
        let removed = cache.remove("Timecode");
        assert!(removed.is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_remove_missing_returns_none() {
        let mut cache = DictCache::new(16);
        assert!(cache.remove("ghost").is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = DictCache::new(16);
        cache.insert("a".to_string(), make_entry("a"));
        cache.insert("b".to_string(), make_entry("b"));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_zero_capacity_no_op() {
        let mut cache = DictCache::new(0);
        cache.insert("x".to_string(), make_entry("x"));
        assert!(cache.is_empty());
        assert!(cache.get("x").is_none());
    }

    #[test]
    fn test_cached_entry_class_name_and_auid() {
        use crate::dictionary::ClassDefinition;
        let class = ClassDefinition::new(Auid::CLASS_HEADER, "Header", None);
        let entry = CachedEntry::Class(class);
        assert_eq!(entry.name(), "Header");
        assert_eq!(entry.auid(), Auid::CLASS_HEADER);
    }

    #[test]
    fn test_cached_entry_auid_variant() {
        let entry = CachedEntry::Auid {
            auid: Auid::PICTURE,
            name: "Picture".to_string(),
        };
        assert_eq!(entry.name(), "Picture");
        assert_eq!(entry.auid(), Auid::PICTURE);
    }

    #[test]
    fn test_peek_does_not_affect_order() {
        let mut cache = DictCache::new(3);
        cache.insert("a".to_string(), make_entry("a"));
        cache.insert("b".to_string(), make_entry("b"));
        cache.insert("c".to_string(), make_entry("c"));

        // Peek "a" — this must NOT promote it in LRU order
        let _ = cache.peek("a");

        // Inserting "d" should still evict "a" (it remains LRU)
        cache.insert("d".to_string(), make_entry("d"));
        assert!(
            !cache.contains("a"),
            "'a' should be evicted (peek does not promote)"
        );
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_drain_empties_cache() {
        let mut cache = DictCache::new(16);
        cache.insert("a".to_string(), make_entry("a"));
        cache.insert("b".to_string(), make_entry("b"));
        let drained: Vec<_> = cache.drain().collect();
        assert_eq!(drained.len(), 2);
        assert!(cache.is_empty());
    }
}
