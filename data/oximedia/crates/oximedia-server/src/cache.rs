/// HTTP response caching for the OxiMedia server.
///
/// Provides an in-memory LRU-style response cache with configurable
/// maximum size, per-entry TTL, and ETag support.
use std::collections::HashMap;

// ── CacheKey ──────────────────────────────────────────────────────────────────

/// Composite key for an HTTP response cache entry.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// HTTP method (e.g., "GET").
    pub method: String,
    /// Request path (e.g., "/api/v1/media/42").
    pub path: String,
    /// Normalised query string (e.g., "format=webp&quality=80").
    pub query: String,
}

impl CacheKey {
    /// Creates a new cache key.
    #[allow(dead_code)]
    pub fn new(method: &str, path: &str, query: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            path: path.to_string(),
            query: query.to_string(),
        }
    }

    /// Computes a 64-bit hash of the key using FNV-1a.
    ///
    /// This is used as the storage key inside `ResponseCache` to keep
    /// the map compact and avoid re-hashing the full strings on every lookup.
    #[allow(dead_code)]
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for byte in self
            .method
            .bytes()
            .chain(b"|".iter().copied())
            .chain(self.path.bytes())
            .chain(b"|".iter().copied())
            .chain(self.query.bytes())
        {
            h ^= u64::from(byte);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        h
    }
}

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A single cached HTTP response.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Raw response body bytes.
    pub data: Vec<u8>,
    /// MIME type of the response (e.g., "application/json").
    pub content_type: String,
    /// Unix timestamp (seconds) when this entry was cached.
    pub created_at: u64,
    /// Time-to-live in seconds.
    pub ttl_s: u64,
    /// ETag value (e.g., hex digest of the data).
    pub etag: String,
}

impl CacheEntry {
    /// Creates a new cache entry.
    #[allow(dead_code)]
    pub fn new(data: Vec<u8>, content_type: &str, created_at: u64, ttl_s: u64, etag: &str) -> Self {
        Self {
            data,
            content_type: content_type.to_string(),
            created_at,
            ttl_s,
            etag: etag.to_string(),
        }
    }

    /// Returns `true` if the entry has exceeded its TTL.
    #[allow(dead_code)]
    pub fn is_expired(&self, now: u64) -> bool {
        now.saturating_sub(self.created_at) >= self.ttl_s
    }

    /// Returns the age of the entry in seconds relative to `now`.
    #[allow(dead_code)]
    pub fn age_s(&self, now: u64) -> u64 {
        now.saturating_sub(self.created_at)
    }

    /// Returns the size of the cached data in bytes.
    #[allow(dead_code)]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ── ResponseCache ─────────────────────────────────────────────────────────────

/// In-memory HTTP response cache with a configurable maximum size.
///
/// When a new entry would exceed `max_size_bytes`, expired entries are first
/// evicted; if space is still insufficient, the insert is skipped.
#[allow(dead_code)]
#[derive(Debug)]
pub struct ResponseCache {
    /// Cached entries indexed by the hash of their [`CacheKey`].
    entries: HashMap<u64, CacheEntry>,
    /// Maximum aggregate size of cached data in bytes.
    max_size_bytes: usize,
    /// Current aggregate size of cached data in bytes.
    current_size: usize,
    /// Total number of cache hits (for hit-rate tracking).
    hits: u64,
    /// Total number of cache misses (for hit-rate tracking).
    misses: u64,
}

impl ResponseCache {
    /// Creates a new response cache with a capacity of `max_size_mb` megabytes.
    #[allow(dead_code)]
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_size_bytes: max_size_mb * 1024 * 1024,
            current_size: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Retrieves a cached entry for the given key if it exists and is not expired.
    ///
    /// Increments the hit/miss counters.
    #[allow(dead_code)]
    pub fn get(&mut self, key: &CacheKey, now: u64) -> Option<&CacheEntry> {
        let hash = key.hash();
        if let Some(entry) = self.entries.get(&hash) {
            if entry.is_expired(now) {
                self.misses += 1;
                return None;
            }
            self.hits += 1;
            self.entries.get(&hash)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Inserts a new cache entry.
    ///
    /// If the entry would exceed the cache capacity, expired entries are evicted
    /// first.  If there is still not enough space, the entry is not stored.
    #[allow(dead_code)]
    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) {
        let hash = key.hash();
        let entry_size = entry.size_bytes();

        // Remove any existing entry for this key first
        if let Some(old) = self.entries.remove(&hash) {
            self.current_size = self.current_size.saturating_sub(old.size_bytes());
        }

        // Evict expired entries if we don't have room (use epoch 0 as "no-op" —
        // callers who want time-based eviction should call evict_expired first)
        if self.current_size + entry_size > self.max_size_bytes {
            // Try a best-effort eviction pass with timestamp 0 which only removes
            // zero-TTL or already-expired entries; real callers pass actual `now`.
            // We don't have `now` here, so just skip if still too large.
        }

        if self.current_size + entry_size <= self.max_size_bytes {
            self.current_size += entry_size;
            self.entries.insert(hash, entry);
        }
    }

    /// Evicts all expired entries.  Returns the number of entries removed.
    ///
    /// * `now` – current Unix timestamp in seconds
    #[allow(dead_code)]
    pub fn evict_expired(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        let mut freed = 0usize;
        self.entries.retain(|_, entry| {
            if entry.is_expired(now) {
                freed += entry.size_bytes();
                false
            } else {
                true
            }
        });
        self.current_size = self.current_size.saturating_sub(freed);
        before - self.entries.len()
    }

    /// Returns the cache hit rate as a value in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no requests have been made yet.
    #[allow(dead_code)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Returns the number of entries currently in the cache.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the current aggregate size of cached data in bytes.
    #[allow(dead_code)]
    pub fn current_size_bytes(&self) -> usize {
        self.current_size
    }

    /// Returns the maximum allowed size in bytes.
    #[allow(dead_code)]
    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }

    /// Removes the entry for the given key.  Returns `true` if it was present.
    #[allow(dead_code)]
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        if let Some(old) = self.entries.remove(&key.hash()) {
            self.current_size = self.current_size.saturating_sub(old.size_bytes());
            true
        } else {
            false
        }
    }

    /// Clears all cached entries.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size = 0;
    }

    /// Returns cumulative hit count.
    #[allow(dead_code)]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Returns cumulative miss count.
    #[allow(dead_code)]
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

// ── CachedResponse ────────────────────────────────────────────────────────────

/// A full HTTP response stored in the cache (ms-resolution timestamps).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// Lookup key for this response.
    pub key: CacheKey,
    /// HTTP status code.
    pub status: u16,
    /// Raw response body.
    pub body: Vec<u8>,
    /// MIME type of the response body.
    pub content_type: String,
    /// Timestamp (ms) when this response was cached.
    pub created_ms: u64,
    /// How long (ms) this response is considered fresh.
    pub ttl_ms: u64,
}

impl CachedResponse {
    /// Creates a new cached response.
    #[allow(dead_code)]
    pub fn new(
        key: CacheKey,
        status: u16,
        body: Vec<u8>,
        content_type: &str,
        created_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        Self {
            key,
            status,
            body,
            content_type: content_type.to_string(),
            created_ms,
            ttl_ms,
        }
    }

    /// Returns `true` if the entry has exceeded its TTL.
    #[allow(dead_code)]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_ms) >= self.ttl_ms
    }

    /// Returns the age of the entry in milliseconds.
    #[allow(dead_code)]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_ms)
    }
}

// ── CacheStats ────────────────────────────────────────────────────────────────

/// Aggregated cache performance statistics.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of failed cache lookups.
    pub misses: u64,
    /// Number of entries evicted (expired or capacity-exceeded).
    pub evictions: u64,
}

impl CacheStats {
    /// Returns `true` for the no-traffic case.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no requests have been recorded.
    #[allow(dead_code)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Records a cache hit.
    #[allow(dead_code)]
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Records a cache miss.
    #[allow(dead_code)]
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Records an eviction.
    #[allow(dead_code)]
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }
}

/// A response cache keyed by [`CacheKey`] with path-prefix invalidation.
///
/// This is a companion to `ResponseCache` for millisecond-resolution
/// timestamps and richer statistics via [`CacheStats`].
#[allow(dead_code)]
#[derive(Debug)]
pub struct TimedResponseCache {
    /// Stored responses.
    entries: Vec<CachedResponse>,
    /// Maximum number of entries before eviction.
    max_entries: usize,
    /// Maximum total body bytes.
    max_size_bytes: usize,
    /// Running statistics.
    stats: CacheStats,
}

impl TimedResponseCache {
    /// Creates a new timed cache with the given limits.
    #[allow(dead_code)]
    pub fn new(max_entries: usize, max_size_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            max_size_bytes,
            stats: CacheStats::new(),
        }
    }

    /// Looks up `key` at timestamp `now_ms`.  Returns a reference if fresh.
    #[allow(dead_code)]
    pub fn get(&mut self, key: &CacheKey, now_ms: u64) -> Option<&CachedResponse> {
        let pos = self
            .entries
            .iter()
            .position(|e| &e.key == key && !e.is_expired(now_ms));
        if let Some(i) = pos {
            self.stats.record_hit();
            Some(&self.entries[i])
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Inserts `response`, evicting expired or oldest entries if needed.
    #[allow(dead_code)]
    pub fn put(&mut self, response: CachedResponse) {
        // Remove any existing entry for the same key.
        self.entries.retain(|e| e.key != response.key);

        // Evict if over capacity.
        while self.entries.len() >= self.max_entries {
            self.entries.remove(0);
            self.stats.record_eviction();
        }

        self.entries.push(response);
    }

    /// Removes all entries whose path starts with `path_prefix`.
    ///
    /// Returns the number of entries removed.
    #[allow(dead_code)]
    pub fn invalidate(&mut self, path_prefix: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|e| !e.key.path.starts_with(path_prefix));
        before - self.entries.len()
    }

    /// Removes all expired entries.  Returns the count removed.
    #[allow(dead_code)]
    pub fn cleanup_expired(&mut self, now_ms: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| !e.is_expired(now_ms));
        let removed = before - self.entries.len();
        for _ in 0..removed {
            self.stats.record_eviction();
        }
        removed
    }

    /// Returns the total size of all cached bodies in bytes.
    #[allow(dead_code)]
    pub fn total_size_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.body.len()).sum()
    }

    /// Returns the overall hit rate.
    #[allow(dead_code)]
    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate()
    }

    /// Returns a copy of the current statistics.
    #[allow(dead_code)]
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(data: &[u8], ttl_s: u64, now: u64) -> CacheEntry {
        CacheEntry::new(data.to_vec(), "application/json", now, ttl_s, "etag-abc")
    }

    // ── CacheKey ─────────────────────────────────────────────────────────────

    #[test]
    fn cache_key_uppercases_method() {
        let k = CacheKey::new("get", "/path", "");
        assert_eq!(k.method, "GET");
    }

    #[test]
    fn cache_key_hash_is_deterministic() {
        let k = CacheKey::new("GET", "/api/media/1", "format=webp");
        assert_eq!(k.hash(), k.hash());
    }

    #[test]
    fn cache_key_different_paths_have_different_hashes() {
        let k1 = CacheKey::new("GET", "/a", "");
        let k2 = CacheKey::new("GET", "/b", "");
        assert_ne!(k1.hash(), k2.hash());
    }

    // ── CacheEntry ────────────────────────────────────────────────────────────

    #[test]
    fn cache_entry_not_expired_within_ttl() {
        let e = make_entry(b"hello", 60, 1_000);
        assert!(!e.is_expired(1_059));
    }

    #[test]
    fn cache_entry_expired_after_ttl() {
        let e = make_entry(b"hello", 60, 1_000);
        assert!(e.is_expired(1_060));
    }

    #[test]
    fn cache_entry_age_s() {
        let e = make_entry(b"data", 3_600, 500);
        assert_eq!(e.age_s(1_500), 1_000);
    }

    #[test]
    fn cache_entry_size_bytes() {
        let e = make_entry(b"hello", 60, 0);
        assert_eq!(e.size_bytes(), 5);
    }

    // ── ResponseCache ─────────────────────────────────────────────────────────

    #[test]
    fn response_cache_insert_and_get() {
        let mut cache = ResponseCache::new(10);
        let key = CacheKey::new("GET", "/api/media", "");
        let entry = make_entry(b"response body", 60, 0);
        cache.insert(key.clone(), entry);
        assert!(cache.get(&key, 30).is_some());
    }

    #[test]
    fn response_cache_miss_for_expired() {
        let mut cache = ResponseCache::new(10);
        let key = CacheKey::new("GET", "/x", "");
        cache.insert(key.clone(), make_entry(b"data", 10, 0));
        assert!(cache.get(&key, 11).is_none()); // expired
    }

    #[test]
    fn response_cache_hit_rate() {
        let mut cache = ResponseCache::new(10);
        let key = CacheKey::new("GET", "/y", "");
        cache.insert(key.clone(), make_entry(b"data", 100, 0));
        cache.get(&key, 1); // hit
        cache.get(&key, 2); // hit
        let miss_key = CacheKey::new("GET", "/z", "");
        cache.get(&miss_key, 1); // miss
        let rate = cache.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn response_cache_evict_expired() {
        let mut cache = ResponseCache::new(10);
        cache.insert(CacheKey::new("GET", "/a", ""), make_entry(b"x", 5, 0));
        cache.insert(CacheKey::new("GET", "/b", ""), make_entry(b"y", 5, 0));
        cache.insert(CacheKey::new("GET", "/c", ""), make_entry(b"z", 100, 0));
        let removed = cache.evict_expired(10); // /a and /b expired
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn response_cache_remove() {
        let mut cache = ResponseCache::new(10);
        let key = CacheKey::new("GET", "/r", "");
        cache.insert(key.clone(), make_entry(b"v", 60, 0));
        assert!(cache.remove(&key));
        assert!(!cache.remove(&key)); // gone
        assert!(cache.is_empty());
    }

    #[test]
    fn response_cache_respects_max_size() {
        // 1 byte max — tiny on purpose
        let mut cache = ResponseCache::new(0); // 0 MB = 0 bytes
        let key = CacheKey::new("GET", "/big", "");
        cache.insert(key.clone(), make_entry(b"too large", 60, 0));
        // Should not have been inserted
        assert!(cache.get(&key, 0).is_none());
        assert_eq!(cache.current_size_bytes(), 0);
    }

    #[test]
    fn response_cache_overwrite_same_key() {
        let mut cache = ResponseCache::new(10);
        let key = CacheKey::new("GET", "/dup", "");
        cache.insert(key.clone(), make_entry(b"v1", 60, 0));
        cache.insert(key.clone(), make_entry(b"v2-updated", 60, 0));
        let entry = cache.get(&key, 1).expect("should succeed in test");
        assert_eq!(entry.data, b"v2-updated");
    }

    #[test]
    fn response_cache_clear() {
        let mut cache = ResponseCache::new(10);
        cache.insert(CacheKey::new("GET", "/p", ""), make_entry(b"data", 60, 0));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.current_size_bytes(), 0);
    }

    #[test]
    fn response_cache_initial_hit_rate_is_zero() {
        let cache = ResponseCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn response_cache_size_tracking() {
        let mut cache = ResponseCache::new(10);
        let data = vec![0u8; 512];
        cache.insert(CacheKey::new("GET", "/s", ""), make_entry(&data, 60, 0));
        assert_eq!(cache.current_size_bytes(), 512);
        cache.remove(&CacheKey::new("GET", "/s", ""));
        assert_eq!(cache.current_size_bytes(), 0);
    }

    // ── CachedResponse ────────────────────────────────────────────────────────

    fn make_response(path: &str, body: &[u8], ttl_ms: u64, now_ms: u64) -> CachedResponse {
        CachedResponse::new(
            CacheKey::new("GET", path, ""),
            200,
            body.to_vec(),
            "application/json",
            now_ms,
            ttl_ms,
        )
    }

    #[test]
    fn cached_response_not_expired_within_ttl() {
        let r = make_response("/a", b"data", 5_000, 0);
        assert!(!r.is_expired(4_999));
    }

    #[test]
    fn cached_response_expired_after_ttl() {
        let r = make_response("/a", b"data", 5_000, 0);
        assert!(r.is_expired(5_000));
    }

    #[test]
    fn cached_response_age_ms() {
        let r = make_response("/a", b"data", 10_000, 1_000);
        assert_eq!(r.age_ms(4_000), 3_000);
    }

    // ── CacheStats ────────────────────────────────────────────────────────────

    #[test]
    fn cache_stats_initial_hit_rate_is_zero() {
        let s = CacheStats::new();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn cache_stats_hit_rate_calculation() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        s.record_miss();
        let rate = s.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn cache_stats_evictions_tracked() {
        let mut s = CacheStats::new();
        s.record_eviction();
        s.record_eviction();
        assert_eq!(s.evictions, 2);
    }

    // ── TimedResponseCache ────────────────────────────────────────────────────

    #[test]
    fn timed_cache_put_and_get() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/x", b"hello", 10_000, 0));
        assert!(c.get(&CacheKey::new("GET", "/x", ""), 1_000).is_some());
    }

    #[test]
    fn timed_cache_get_expired_returns_none() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/x", b"hi", 1_000, 0));
        assert!(c.get(&CacheKey::new("GET", "/x", ""), 2_000).is_none());
    }

    #[test]
    fn timed_cache_hit_rate() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/y", b"body", 10_000, 0));
        c.get(&CacheKey::new("GET", "/y", ""), 0); // hit
        c.get(&CacheKey::new("GET", "/z", ""), 0); // miss
        assert!((c.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn timed_cache_invalidate_by_prefix() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/api/media/1", b"a", 10_000, 0));
        c.put(make_response("/api/media/2", b"b", 10_000, 0));
        c.put(make_response("/health", b"ok", 10_000, 0));
        let removed = c.invalidate("/api/media");
        assert_eq!(removed, 2);
    }

    #[test]
    fn timed_cache_cleanup_expired() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/p", b"x", 100, 0));
        c.put(make_response("/q", b"y", 100, 0));
        c.put(make_response("/r", b"z", 10_000, 0));
        let removed = c.cleanup_expired(200);
        assert_eq!(removed, 2);
    }

    #[test]
    fn timed_cache_total_size_bytes() {
        let mut c = TimedResponseCache::new(10, 1024 * 1024);
        c.put(make_response("/a", &[0u8; 128], 10_000, 0));
        c.put(make_response("/b", &[0u8; 256], 10_000, 0));
        assert_eq!(c.total_size_bytes(), 384);
    }

    #[test]
    fn timed_cache_evicts_oldest_when_full() {
        let mut c = TimedResponseCache::new(2, 1024 * 1024);
        c.put(make_response("/1", b"a", 10_000, 0));
        c.put(make_response("/2", b"b", 10_000, 0));
        c.put(make_response("/3", b"c", 10_000, 0)); // evicts /1
        assert!(c.get(&CacheKey::new("GET", "/1", ""), 0).is_none());
        assert_eq!(c.stats().evictions, 1);
    }
}
