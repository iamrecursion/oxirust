//! WebAssembly bindings providing a simple in-memory LRU cache for browser use.
//!
//! Implements a standalone LRU (Least Recently Used) cache that is fully
//! compatible with WASM (no threads, no `Instant`, no file-system).
//! Keys are UTF-8 strings; values are opaque byte vectors.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone WASM-compatible LRU
// ---------------------------------------------------------------------------

/// A minimal doubly-linked-list LRU backed by a Vec arena.
///
/// Unlike `oximedia_cache::LruCache`, this implementation avoids `std::time::Instant`
/// (which is unavailable in the WASM browser environment) and any thread primitives.
struct Node {
    key: String,
    value: Vec<u8>,
    prev: usize,
    next: usize,
}

const SENTINEL: usize = usize::MAX;

struct InnerLru {
    capacity: usize,
    map: HashMap<String, usize>,
    slots: Vec<Node>,
    free: Vec<usize>,
    head: usize, // MRU end
    tail: usize, // LRU end
    len: usize,
}

impl InnerLru {
    fn new(capacity: usize) -> Self {
        InnerLru {
            capacity,
            map: HashMap::new(),
            slots: Vec::new(),
            free: Vec::new(),
            head: SENTINEL,
            tail: SENTINEL,
            len: 0,
        }
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let n = &self.slots[idx];
            (n.prev, n.next)
        };
        if prev != SENTINEL {
            self.slots[prev].next = next;
        } else {
            self.head = next;
        }
        if next != SENTINEL {
            self.slots[next].prev = prev;
        } else {
            self.tail = prev;
        }
        self.slots[idx].prev = SENTINEL;
        self.slots[idx].next = SENTINEL;
    }

    fn push_front(&mut self, idx: usize) {
        self.slots[idx].prev = SENTINEL;
        self.slots[idx].next = self.head;
        if self.head != SENTINEL {
            self.slots[self.head].prev = idx;
        } else {
            self.tail = idx;
        }
        self.head = idx;
    }

    fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        let &idx = self.map.get(key)?;
        self.detach(idx);
        self.push_front(idx);
        Some(self.slots[idx].value.clone())
    }

    fn put(&mut self, key: String, value: Vec<u8>) {
        if let Some(&idx) = self.map.get(&key) {
            self.slots[idx].value = value;
            self.detach(idx);
            self.push_front(idx);
            return;
        }

        // Evict LRU if at capacity.
        if self.len >= self.capacity && self.capacity > 0 {
            let lru = self.tail;
            if lru != SENTINEL {
                self.detach(lru);
                let evicted_key = self.slots[lru].key.clone();
                self.map.remove(&evicted_key);
                self.free.push(lru);
                self.len -= 1;
            }
        }

        // Allocate slot.
        let idx = if let Some(slot) = self.free.pop() {
            self.slots[slot] = Node {
                key: key.clone(),
                value,
                prev: SENTINEL,
                next: SENTINEL,
            };
            slot
        } else {
            let slot = self.slots.len();
            self.slots.push(Node {
                key: key.clone(),
                value,
                prev: SENTINEL,
                next: SENTINEL,
            });
            slot
        };

        self.map.insert(key, idx);
        self.push_front(idx);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }
}

// ---------------------------------------------------------------------------
// WasmLruCache — public WASM type
// ---------------------------------------------------------------------------

/// An in-memory LRU cache for browser-side use.
///
/// Stores arbitrary byte payloads keyed by UTF-8 strings.  Automatically evicts
/// the least-recently-used entry when `capacity` is reached.
///
/// # Example
///
/// ```javascript
/// const cache = new WasmLruCache(128);
/// cache.put("segment-0", new Uint8Array([1, 2, 3]));
/// const bytes = cache.get("segment-0"); // Uint8Array or undefined
/// ```
#[wasm_bindgen]
pub struct WasmLruCache {
    inner: InnerLru,
}

#[wasm_bindgen]
impl WasmLruCache {
    /// Create a new LRU cache with the given entry capacity.
    ///
    /// `capacity` must be ≥ 1.  If 0 is passed, the cache is created with
    /// capacity 1 to prevent degenerate behaviour.
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: u32) -> WasmLruCache {
        let cap = (capacity as usize).max(1);
        WasmLruCache {
            inner: InnerLru::new(cap),
        }
    }

    /// Insert or update a key-value pair.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted
    /// before the new entry is inserted.
    pub fn put(&mut self, key: &str, value: Vec<u8>) {
        self.inner.put(key.to_string(), value);
    }

    /// Retrieve the value for `key`, promoting it to most-recently-used.
    ///
    /// Returns `None` (undefined in JS) if the key is not present.
    pub fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        self.inner.get(key)
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Return `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.inner = InnerLru::new(self.inner.capacity);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_put_get() {
        let mut cache = WasmLruCache::new(4);
        cache.put("a", vec![1, 2, 3]);
        let val = cache.get("a");
        assert_eq!(val, Some(vec![1u8, 2, 3]), "should retrieve inserted value");
    }

    #[test]
    fn miss_returns_none() {
        let mut cache = WasmLruCache::new(4);
        assert!(cache.get("missing").is_none(), "missing key should be None");
    }

    #[test]
    fn len_and_is_empty() {
        let mut cache = WasmLruCache::new(8);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.put("x", vec![0]);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn evicts_lru_on_overflow() {
        let mut cache = WasmLruCache::new(2);
        cache.put("a", vec![1]);
        cache.put("b", vec![2]);
        // Access "a" to make "b" the LRU.
        cache.get("a");
        // Insert "c": "b" should be evicted.
        cache.put("c", vec![3]);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("b").is_none(), "b should have been evicted");
        assert!(cache.get("a").is_some(), "a should still be present");
        assert!(cache.get("c").is_some(), "c should be present");
    }

    #[test]
    fn update_existing_key() {
        let mut cache = WasmLruCache::new(4);
        cache.put("k", vec![1]);
        cache.put("k", vec![2, 3]);
        let val = cache.get("k");
        assert_eq!(val, Some(vec![2u8, 3]), "should return updated value");
        assert_eq!(cache.len(), 1, "updating should not grow the cache");
    }

    #[test]
    fn clear_empties_cache() {
        let mut cache = WasmLruCache::new(4);
        cache.put("a", vec![0]);
        cache.put("b", vec![1]);
        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.get("a").is_none());
    }
}
