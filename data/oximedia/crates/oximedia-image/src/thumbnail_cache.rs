//! Thumbnail cache with LRU eviction for image sequences.
//!
//! Provides a bounded in-memory cache for decoded thumbnails, with
//! configurable size limits and LRU (least-recently-used) eviction policy.

#![allow(dead_code)]

use std::collections::HashMap;

/// Standard thumbnail size presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThumbnailSize {
    /// 64×64 pixels – icon.
    Icon,
    /// 128×128 pixels – small.
    Small,
    /// 256×256 pixels – medium.
    Medium,
    /// 512×512 pixels – large.
    Large,
    /// Custom width × height.
    Custom(u32, u32),
}

impl ThumbnailSize {
    /// Return the (width, height) pair for this size.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Icon => (64, 64),
            Self::Small => (128, 128),
            Self::Medium => (256, 256),
            Self::Large => (512, 512),
            Self::Custom(w, h) => (*w, *h),
        }
    }

    /// Return the pixel area for this size.
    #[must_use]
    pub fn pixel_area(&self) -> u64 {
        let (w, h) = self.dimensions();
        u64::from(w) * u64::from(h)
    }
}

/// A single cached thumbnail entry.
#[derive(Clone, Debug)]
pub struct ThumbnailEntry {
    /// Key identifying the source frame (e.g. file path + frame number).
    pub key: String,
    /// Rendered size.
    pub size: ThumbnailSize,
    /// Raw RGBA pixel data (width × height × 4 bytes).
    pub data: Vec<u8>,
    /// Access counter – incremented on every cache hit.
    pub access_count: u64,
    /// Monotonic access timestamp used for LRU ordering.
    pub last_accessed: u64,
}

impl ThumbnailEntry {
    /// Create a new entry with the given key, size, and pixel data.
    #[must_use]
    pub fn new(key: impl Into<String>, size: ThumbnailSize, data: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            size,
            data,
            access_count: 0,
            last_accessed: 0,
        }
    }

    /// Return the memory footprint of the pixel data in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

/// Compound cache key combining asset path, frame index, and size.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    path: String,
    frame: u32,
    size: ThumbnailSize,
}

/// LRU thumbnail cache with a configurable byte budget.
///
/// # Example
/// ```
/// use oximedia_image::thumbnail_cache::{ThumbnailCache, ThumbnailSize};
///
/// let mut cache = ThumbnailCache::new(1024 * 1024); // 1 MiB
/// let pixels = vec![0u8; 64 * 64 * 4];
/// cache.insert("clip.dpx", 0, ThumbnailSize::Icon, pixels);
/// assert!(cache.get("clip.dpx", 0, ThumbnailSize::Icon).is_some());
/// ```
pub struct ThumbnailCache {
    entries: HashMap<CacheKey, ThumbnailEntry>,
    /// Maximum total bytes to hold in cache.
    capacity_bytes: usize,
    /// Current total bytes used.
    used_bytes: usize,
    /// Monotonic clock counter.
    clock: u64,
}

impl ThumbnailCache {
    /// Create a new cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity_bytes,
            used_bytes: 0,
            clock: 0,
        }
    }

    /// Return the configured byte capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    /// Return the current number of bytes occupied.
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Return the number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a thumbnail; updates `last_accessed` and `access_count` on hit.
    pub fn get(&mut self, path: &str, frame: u32, size: ThumbnailSize) -> Option<&ThumbnailEntry> {
        let key = CacheKey {
            path: path.to_owned(),
            frame,
            size,
        };
        self.clock += 1;
        let clock = self.clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed = clock;
            entry.access_count += 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Insert a thumbnail into the cache, evicting LRU entries as needed.
    ///
    /// Returns `false` if the single entry is larger than the total capacity.
    pub fn insert(&mut self, path: &str, frame: u32, size: ThumbnailSize, data: Vec<u8>) -> bool {
        let entry_bytes = data.len();
        if entry_bytes > self.capacity_bytes {
            return false;
        }

        let key = CacheKey {
            path: path.to_owned(),
            frame,
            size,
        };

        // Remove existing entry for this key if present.
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes -= old.byte_size();
        }

        // Evict until there is room.
        while self.used_bytes + entry_bytes > self.capacity_bytes {
            self.evict_lru();
        }

        self.clock += 1;
        let mut entry = ThumbnailEntry::new(path, size, data);
        entry.last_accessed = self.clock;

        self.used_bytes += entry_bytes;
        self.entries.insert(key, entry);
        true
    }

    /// Evict the single least-recently-used entry.
    ///
    /// Does nothing if the cache is empty.
    pub fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Find key with smallest `last_accessed`.
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(k) = lru_key {
            if let Some(removed) = self.entries.remove(&k) {
                self.used_bytes -= removed.byte_size();
            }
        }
    }

    /// Remove all entries and reset counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }

    /// Invalidate all entries whose path starts with `prefix`.
    pub fn invalidate_prefix(&mut self, prefix: &str) {
        let to_remove: Vec<CacheKey> = self
            .entries
            .keys()
            .filter(|k| k.path.starts_with(prefix))
            .cloned()
            .collect();
        for k in to_remove {
            if let Some(e) = self.entries.remove(&k) {
                self.used_bytes -= e.byte_size();
            }
        }
    }

    /// Return usage as a fraction in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn usage_ratio(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 1.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixels(size: ThumbnailSize) -> Vec<u8> {
        let (w, h) = size.dimensions();
        vec![128u8; (w * h * 4) as usize]
    }

    #[test]
    fn test_size_dimensions_icon() {
        assert_eq!(ThumbnailSize::Icon.dimensions(), (64, 64));
    }

    #[test]
    fn test_size_dimensions_custom() {
        assert_eq!(ThumbnailSize::Custom(320, 180).dimensions(), (320, 180));
    }

    #[test]
    fn test_size_pixel_area() {
        assert_eq!(ThumbnailSize::Small.pixel_area(), 128 * 128);
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        cache.insert(
            "a.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        assert!(cache.get("a.dpx", 0, ThumbnailSize::Icon).is_some());
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        assert!(cache.get("missing.dpx", 0, ThumbnailSize::Icon).is_none());
    }

    #[test]
    fn test_cache_len_and_used_bytes() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        let data = make_pixels(ThumbnailSize::Icon);
        let expected_bytes = data.len();
        cache.insert("b.dpx", 1, ThumbnailSize::Icon, data);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.used_bytes(), expected_bytes);
    }

    #[test]
    fn test_cache_lru_eviction() {
        // Capacity for exactly 2 Icon entries (64*64*4 = 16384 bytes each).
        let icon_bytes = (64 * 64 * 4) as usize;
        let mut cache = ThumbnailCache::new(icon_bytes * 2);
        cache.insert(
            "f1.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        cache.insert(
            "f2.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        // Access f1 so f2 becomes LRU.
        cache.get("f1.dpx", 0, ThumbnailSize::Icon);
        // Insert f3 – should evict f2.
        cache.insert(
            "f3.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        assert!(cache.get("f1.dpx", 0, ThumbnailSize::Icon).is_some());
        assert!(cache.get("f2.dpx", 0, ThumbnailSize::Icon).is_none());
        assert!(cache.get("f3.dpx", 0, ThumbnailSize::Icon).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        cache.insert(
            "c.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn test_cache_invalidate_prefix() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        cache.insert(
            "shot01/f001.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        cache.insert(
            "shot01/f002.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        cache.insert(
            "shot02/f001.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        cache.invalidate_prefix("shot01/");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_oversized_entry_rejected() {
        let mut cache = ThumbnailCache::new(100);
        let result = cache.insert(
            "big.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        assert!(!result);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_access_count_increments() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        cache.insert(
            "d.dpx",
            0,
            ThumbnailSize::Small,
            make_pixels(ThumbnailSize::Small),
        );
        cache.get("d.dpx", 0, ThumbnailSize::Small);
        cache.get("d.dpx", 0, ThumbnailSize::Small);
        let entry = cache
            .get("d.dpx", 0, ThumbnailSize::Small)
            .expect("should succeed in test");
        assert_eq!(entry.access_count, 3);
    }

    #[test]
    fn test_usage_ratio_empty() {
        let cache = ThumbnailCache::new(1024);
        assert!((cache.usage_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_full() {
        let icon_bytes = (64 * 64 * 4) as usize;
        let mut cache = ThumbnailCache::new(icon_bytes);
        cache.insert(
            "e.dpx",
            0,
            ThumbnailSize::Icon,
            make_pixels(ThumbnailSize::Icon),
        );
        assert!((cache.usage_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_reinsertion_updates_data() {
        let mut cache = ThumbnailCache::new(1024 * 1024);
        let data1 = vec![0u8; 64 * 64 * 4];
        let data2 = vec![255u8; 64 * 64 * 4];
        cache.insert("x.dpx", 0, ThumbnailSize::Icon, data1);
        cache.insert("x.dpx", 0, ThumbnailSize::Icon, data2);
        // Only one entry should exist.
        assert_eq!(cache.len(), 1);
        let entry = cache
            .get("x.dpx", 0, ThumbnailSize::Icon)
            .expect("should succeed in test");
        assert_eq!(entry.data[0], 255);
    }

    #[test]
    fn test_evict_lru_on_empty_cache_is_noop() {
        let mut cache = ThumbnailCache::new(1024);
        cache.evict_lru(); // should not panic
        assert!(cache.is_empty());
    }

    #[test]
    fn test_capacity_getter() {
        let cache = ThumbnailCache::new(8192);
        assert_eq!(cache.capacity(), 8192);
    }
}
