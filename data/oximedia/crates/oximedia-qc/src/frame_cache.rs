//! Frame cache for amortising per-frame decoding cost across multiple QC rules.
//!
//! [`crate::frame_cache::FrameCache`] stores decoded frames keyed by `(file_id, frame_idx)` with a
//! bounded LRU eviction policy.  All entries are wrapped in [`std::sync::Arc`] so callers
//! share ownership without copying data.
//!
//! The cache is intentionally scoped to a single validation run — do **not**
//! store it in a `static` or across multiple calls to
//! [`crate::QualityControl::validate`].

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// A decoded video frame stored in the cache.
///
/// The frame data is stored as raw RGBA bytes in row-major order.
#[derive(Debug, Clone)]
pub struct CachedFrame {
    /// Width of the frame in pixels.
    pub width: u32,
    /// Height of the frame in pixels.
    pub height: u32,
    /// Raw pixel data (RGBA, row-major).
    pub data: Vec<u8>,
    /// Frame index (display order).
    pub frame_idx: u64,
}

impl CachedFrame {
    /// Creates a new cached frame.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<u8>, frame_idx: u64) -> Self {
        Self {
            width,
            height,
            data,
            frame_idx,
        }
    }
}

/// Key identifying a specific frame within a specific file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrameKey {
    /// Unique identifier for the source file (e.g. a stable hash of the path).
    pub file_id: u64,
    /// Zero-based frame index in display order.
    pub frame_idx: u64,
}

impl FrameKey {
    /// Creates a new frame key.
    #[must_use]
    pub fn new(file_id: u64, frame_idx: u64) -> Self {
        Self { file_id, frame_idx }
    }
}

/// Bounded LRU frame cache.
///
/// Holds up to `max_entries` decoded frames.  When the capacity is exceeded the
/// least-recently-used entry is evicted before the new one is inserted.
///
/// All stored frames are wrapped in [`Arc`] so callers share ownership and
/// cloning a reference is O(1).
pub struct FrameCache {
    /// The actual frame store.
    store: HashMap<FrameKey, Arc<CachedFrame>>,
    /// LRU queue: front = LRU (oldest), back = MRU (most recently used).
    lru_order: VecDeque<FrameKey>,
    /// Maximum number of entries before eviction.
    max_entries: usize,
    /// Total number of cache hits since construction.
    hit_count: u64,
    /// Total number of cache misses since construction.
    miss_count: u64,
}

impl FrameCache {
    /// Creates a new frame cache with the given maximum entry count.
    ///
    /// `max_entries` must be at least 1.  If 0 is passed it is clamped to 1.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.max(1);
        Self {
            store: HashMap::with_capacity(cap),
            lru_order: VecDeque::with_capacity(cap + 1),
            max_entries: cap,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Returns the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Returns the maximum number of entries allowed before eviction.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Returns the total number of cache hits since the cache was created.
    #[must_use]
    pub fn hit_count(&self) -> u64 {
        self.hit_count
    }

    /// Returns the total number of cache misses since the cache was created.
    #[must_use]
    pub fn miss_count(&self) -> u64 {
        self.miss_count
    }

    /// Looks up a frame by key, returning a shared reference if present.
    ///
    /// A hit promotes the entry to the MRU position in the LRU queue.
    pub fn get(&mut self, key: &FrameKey) -> Option<Arc<CachedFrame>> {
        if let Some(frame) = self.store.get(key) {
            self.hit_count += 1;
            // Promote to MRU: remove the key from its current LRU position
            // and push to the back.
            self.lru_order.retain(|k| k != key);
            self.lru_order.push_back(key.clone());
            Some(Arc::clone(frame))
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Inserts a frame into the cache.
    ///
    /// If the cache is at capacity, the LRU entry is evicted first.
    /// If the key already exists, the old value is replaced and the entry is
    /// promoted to MRU.
    pub fn insert(&mut self, key: FrameKey, frame: Arc<CachedFrame>) {
        if self.store.contains_key(&key) {
            // Replace in place and promote.
            self.lru_order.retain(|k| k != &key);
            self.store.insert(key.clone(), frame);
            self.lru_order.push_back(key);
            return;
        }

        // Evict LRU if at capacity.
        if self.store.len() >= self.max_entries {
            self.evict_lru();
        }

        self.store.insert(key.clone(), frame);
        self.lru_order.push_back(key);
    }

    /// Evicts the single least-recently-used entry from the cache.
    ///
    /// Does nothing if the cache is empty.
    pub fn evict_lru(&mut self) {
        if let Some(lru_key) = self.lru_order.pop_front() {
            self.store.remove(&lru_key);
        }
    }

    /// Fetches a frame, calling `decode_fn` on cache miss to produce a new frame.
    ///
    /// The decoded frame is inserted into the cache before being returned.
    ///
    /// # Errors
    ///
    /// Propagates any error returned by `decode_fn`.
    pub fn get_or_decode<E, F>(
        &mut self,
        key: FrameKey,
        decode_fn: F,
    ) -> Result<Arc<CachedFrame>, E>
    where
        F: FnOnce(&FrameKey) -> Result<CachedFrame, E>,
    {
        if let Some(cached) = self.get(&key) {
            return Ok(cached);
        }
        let frame = decode_fn(&key)?;
        let arc = Arc::new(frame);
        self.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.store.clear();
        self.lru_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(frame_idx: u64) -> Arc<CachedFrame> {
        Arc::new(CachedFrame::new(4, 4, vec![0u8; 4 * 4 * 4], frame_idx))
    }

    fn key(file_id: u64, frame_idx: u64) -> FrameKey {
        FrameKey::new(file_id, frame_idx)
    }

    // ── Basic operations ──────────────────────────────────────────────────────

    #[test]
    fn test_frame_cache_empty_on_creation() {
        let cache = FrameCache::new(16);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_frame_cache_insert_and_get() {
        let mut cache = FrameCache::new(16);
        let k = key(1, 0);
        cache.insert(k.clone(), make_frame(0));
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        let got = cache.get(&k);
        assert!(got.is_some(), "Should find inserted frame");
    }

    /// Access same (file, frame) twice; second access must be a hit.
    #[test]
    fn test_frame_cache_hit_count() {
        let mut cache = FrameCache::new(8);
        let k = key(42, 7);
        cache.insert(k.clone(), make_frame(7));

        // First access after insert — also a hit since get() bumps hit_count.
        let _ = cache.get(&k);
        assert_eq!(cache.hit_count(), 1, "First get should be a hit");

        // Second access
        let _ = cache.get(&k);
        assert_eq!(cache.hit_count(), 2, "Second get should also be a hit");

        // Miss on an absent key
        let _ = cache.get(&key(99, 99));
        assert_eq!(cache.miss_count(), 1, "Absent key should be a miss");
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    /// Fill to capacity + 1; verify the oldest (first inserted) entry is evicted.
    #[test]
    fn test_frame_cache_lru_eviction() {
        let max = 4;
        let mut cache = FrameCache::new(max);

        for i in 0..max {
            cache.insert(key(1, i as u64), make_frame(i as u64));
        }
        assert_eq!(cache.len(), max);

        // Insert one more — LRU (frame 0) should be evicted.
        cache.insert(key(1, max as u64), make_frame(max as u64));
        assert_eq!(
            cache.len(),
            max,
            "Cache should stay at max_entries after eviction"
        );

        // Frame 0 (LRU) must no longer be present.
        assert!(
            cache.get(&key(1, 0)).is_none(),
            "LRU entry (frame 0) should have been evicted"
        );

        // Most recently inserted frame must still be present.
        assert!(
            cache.get(&key(1, max as u64)).is_some(),
            "Newest frame should still be in cache"
        );
    }

    /// Accessing a frame in the middle promotes it, so it is not evicted next.
    #[test]
    fn test_lru_promotion_prevents_eviction() {
        let mut cache = FrameCache::new(3);
        cache.insert(key(0, 0), make_frame(0)); // oldest
        cache.insert(key(0, 1), make_frame(1));
        cache.insert(key(0, 2), make_frame(2));

        // Access frame 0 — promotes it to MRU.
        let _ = cache.get(&key(0, 0));

        // Insert frame 3 — should evict frame 1 (now the LRU), not frame 0.
        cache.insert(key(0, 3), make_frame(3));

        assert!(
            cache.get(&key(0, 0)).is_some(),
            "Promoted frame 0 should not be evicted"
        );
        assert!(
            cache.get(&key(0, 1)).is_none(),
            "Frame 1 should have been evicted as the new LRU"
        );
    }

    // ── get_or_decode ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_or_decode_calls_decoder_on_miss() {
        let mut cache = FrameCache::new(8);
        let k = key(5, 10);

        let mut decode_calls = 0u32;
        let frame = cache
            .get_or_decode(k.clone(), |_fk| {
                decode_calls += 1;
                Ok::<_, String>(CachedFrame::new(2, 2, vec![0u8; 16], 10))
            })
            .expect("get_or_decode should succeed");

        assert_eq!(decode_calls, 1, "Decoder should be called on miss");
        assert_eq!(frame.frame_idx, 10);

        // Second access — must be a hit, no decode call.
        let _ = cache
            .get_or_decode(k, |_| {
                decode_calls += 1;
                Ok::<_, String>(CachedFrame::new(2, 2, vec![0u8; 16], 10))
            })
            .expect("second get_or_decode should succeed");
        assert_eq!(decode_calls, 1, "Decoder must NOT be called on cache hit");
        assert_eq!(cache.hit_count(), 1);
    }

    #[test]
    fn test_get_or_decode_propagates_error() {
        let mut cache = FrameCache::new(8);
        let k = key(1, 1);
        let result = cache.get_or_decode(k, |_| Err("decode failed"));
        assert!(result.is_err());
    }
}
