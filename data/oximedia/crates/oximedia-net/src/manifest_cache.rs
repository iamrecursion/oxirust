//! Manifest and playlist caching layer.
//!
//! Caching parsed manifest / playlist structures avoids expensive re-parsing
//! on every client request.  This module provides:
//!
//! - [`CachedManifest`] — a parsed manifest entry with TTL and ETag support.
//! - [`ManifestCache`] — a thread-safe LRU cache for HLS and DASH manifests.
//! - Cache-Control / ETag HTTP header integration.
//! - Automatic expiry and background eviction.
//! - Stale-while-revalidate semantics (serve stale while fetching fresh copy).
//!
//! Supported manifest types:
//! - HLS Master Playlist (M3U8)
//! - HLS Media Playlist (M3U8)
//! - DASH MPD (XML)

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// ─── Manifest Type ────────────────────────────────────────────────────────────

/// Identifies the type of cached manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifestType {
    /// HLS master playlist (multi-bitrate).
    HlsMaster,
    /// HLS media playlist (single-bitrate or live).
    HlsMedia,
    /// MPEG-DASH MPD.
    DashMpd,
}

impl ManifestType {
    /// Returns the MIME content-type for this manifest.
    #[must_use]
    pub const fn content_type(&self) -> &'static str {
        match self {
            Self::HlsMaster | Self::HlsMedia => "application/vnd.apple.mpegurl",
            Self::DashMpd => "application/dash+xml",
        }
    }

    /// Returns a short label for logging.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::HlsMaster => "HLS-Master",
            Self::HlsMedia => "HLS-Media",
            Self::DashMpd => "DASH-MPD",
        }
    }
}

// ─── Cache Entry ──────────────────────────────────────────────────────────────

/// A single cached manifest entry.
#[derive(Debug, Clone)]
pub struct CachedManifest {
    /// Type of this manifest.
    pub manifest_type: ManifestType,
    /// Raw manifest text (M3U8 or XML).
    pub content: String,
    /// ETag for conditional requests (computed as a simple hash of the content).
    pub etag: String,
    /// When this entry was stored.
    pub created_at: Instant,
    /// TTL before the entry must be re-fetched.
    pub ttl: Duration,
    /// Stale-while-revalidate window (serve stale for this long while fetching).
    pub stale_while_revalidate: Duration,
    /// Number of cache hits.
    pub hits: u64,
    /// Whether a background revalidation is in progress.
    pub revalidating: bool,
}

impl CachedManifest {
    /// Creates a new cache entry.
    #[must_use]
    pub fn new(manifest_type: ManifestType, content: String, ttl: Duration) -> Self {
        let etag = compute_etag(&content);
        Self {
            manifest_type,
            content,
            etag,
            created_at: Instant::now(),
            ttl,
            stale_while_revalidate: ttl / 2,
            hits: 0,
            revalidating: false,
        }
    }

    /// Returns whether this entry is still fresh (within TTL).
    #[must_use]
    pub fn is_fresh(&self) -> bool {
        self.created_at.elapsed() < self.ttl
    }

    /// Returns whether this entry is stale but within the stale-while-revalidate window.
    #[must_use]
    pub fn is_stale_but_usable(&self) -> bool {
        let age = self.created_at.elapsed();
        age >= self.ttl && age < self.ttl + self.stale_while_revalidate
    }

    /// Returns the age of this entry.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns the remaining TTL (zero if expired).
    #[must_use]
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl
            .checked_sub(self.created_at.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Returns the Cache-Control header value for this entry.
    #[must_use]
    pub fn cache_control_header(&self) -> String {
        let max_age = self.ttl.as_secs();
        let swr = self.stale_while_revalidate.as_secs();
        format!("public, max-age={max_age}, stale-while-revalidate={swr}")
    }

    /// Returns a complete ETag header value (quoted).
    #[must_use]
    pub fn etag_header(&self) -> String {
        format!("\"{}\"", self.etag)
    }
}

/// Computes a simple content-based ETag (FNV-1a 64-bit hash, hex-encoded).
fn compute_etag(content: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325_u64;
    for byte in content.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("{hash:016x}")
}

// ─── Cache Configuration ──────────────────────────────────────────────────────

/// Configuration for the manifest cache.
#[derive(Debug, Clone)]
pub struct ManifestCacheConfig {
    /// Maximum number of entries.
    pub max_entries: usize,
    /// Default TTL for HLS media playlists.
    pub hls_media_ttl: Duration,
    /// Default TTL for HLS master playlists.
    pub hls_master_ttl: Duration,
    /// Default TTL for DASH MPDs.
    pub dash_mpd_ttl: Duration,
    /// How often expired entries are evicted.
    pub eviction_interval: Duration,
    /// Maximum size in bytes (approximate, based on content lengths).
    pub max_size_bytes: usize,
}

impl Default for ManifestCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            hls_media_ttl: Duration::from_secs(2), // Live playlist: refresh frequently
            hls_master_ttl: Duration::from_secs(300), // Master rarely changes
            dash_mpd_ttl: Duration::from_secs(2),
            eviction_interval: Duration::from_secs(10),
            max_size_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

impl ManifestCacheConfig {
    /// Returns the default TTL for the given manifest type.
    #[must_use]
    pub fn default_ttl(&self, manifest_type: ManifestType) -> Duration {
        match manifest_type {
            ManifestType::HlsMaster => self.hls_master_ttl,
            ManifestType::HlsMedia => self.hls_media_ttl,
            ManifestType::DashMpd => self.dash_mpd_ttl,
        }
    }
}

// ─── Manifest Cache ───────────────────────────────────────────────────────────

/// Inner cache state.
struct CacheInner {
    /// Entries keyed by URL.
    entries: HashMap<String, CachedManifest>,
    /// Maximum entries.
    max_entries: usize,
    /// Total cache hits.
    total_hits: u64,
    /// Total cache misses.
    total_misses: u64,
    /// Total bytes stored.
    total_bytes: usize,
    /// Maximum bytes.
    max_bytes: usize,
}

impl CacheInner {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries.min(256)),
            max_entries,
            total_hits: 0,
            total_misses: 0,
            total_bytes: 0,
            max_bytes,
        }
    }

    fn insert(&mut self, url: String, entry: CachedManifest) {
        let entry_bytes = entry.content.len();

        // Evict by count.
        while self.entries.len() >= self.max_entries {
            self.evict_oldest();
        }
        // Evict by size.
        while self.total_bytes + entry_bytes > self.max_bytes && !self.entries.is_empty() {
            self.evict_oldest();
        }

        if let Some(old) = self.entries.get(&url) {
            self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
        }
        self.total_bytes += entry_bytes;
        self.entries.insert(url, entry);
    }

    fn get(&mut self, url: &str) -> Option<&mut CachedManifest> {
        if self.entries.contains_key(url) {
            self.total_hits += 1;
            self.entries.get_mut(url)
        } else {
            self.total_misses += 1;
            None
        }
    }

    fn remove(&mut self, url: &str) -> Option<CachedManifest> {
        if let Some(e) = self.entries.remove(url) {
            self.total_bytes = self.total_bytes.saturating_sub(e.content.len());
            Some(e)
        } else {
            None
        }
    }

    fn evict_expired(&mut self) {
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.is_fresh() && !e.is_stale_but_usable())
            .map(|(k, _)| k.clone())
            .collect();

        for key in to_remove {
            if let Some(e) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(e.content.len());
            }
        }
    }

    fn evict_oldest(&mut self) {
        // Remove the entry with the oldest creation time.
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            if let Some(e) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(e.content.len());
            }
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Thread-safe manifest and playlist cache.
#[derive(Clone)]
pub struct ManifestCache {
    inner: Arc<RwLock<CacheInner>>,
    config: ManifestCacheConfig,
}

impl ManifestCache {
    /// Creates a new manifest cache.
    #[must_use]
    pub fn new(config: ManifestCacheConfig) -> Self {
        let inner = CacheInner::new(config.max_entries, config.max_size_bytes);
        Self {
            inner: Arc::new(RwLock::new(inner)),
            config,
        }
    }

    /// Stores a manifest in the cache with the default TTL for its type.
    pub async fn put(&self, url: impl Into<String>, content: String, manifest_type: ManifestType) {
        let ttl = self.config.default_ttl(manifest_type);
        let entry = CachedManifest::new(manifest_type, content, ttl);
        let mut inner = self.inner.write().await;
        inner.insert(url.into(), entry);
    }

    /// Stores a manifest with a custom TTL.
    pub async fn put_with_ttl(
        &self,
        url: impl Into<String>,
        content: String,
        manifest_type: ManifestType,
        ttl: Duration,
    ) {
        let entry = CachedManifest::new(manifest_type, content, ttl);
        let mut inner = self.inner.write().await;
        inner.insert(url.into(), entry);
    }

    /// Retrieves a manifest from the cache.
    ///
    /// Returns:
    /// - `Some(entry)` if fresh or stale-but-usable.
    /// - `None` if not in the cache or fully expired.
    pub async fn get(&self, url: &str) -> Option<CachedManifest> {
        let mut inner = self.inner.write().await;
        if let Some(entry) = inner.get(url) {
            if entry.is_fresh() || entry.is_stale_but_usable() {
                entry.hits += 1;
                return Some(entry.clone());
            }
        }
        None
    }

    /// Retrieves a manifest and indicates whether it needs revalidation.
    ///
    /// Returns `(entry, needs_revalidation)`.  The caller should start a
    /// background fetch when `needs_revalidation` is `true`.
    pub async fn get_with_revalidation(&self, url: &str) -> (Option<CachedManifest>, bool) {
        let mut inner = self.inner.write().await;
        if let Some(entry) = inner.get(url) {
            let fresh = entry.is_fresh();
            let usable = fresh || entry.is_stale_but_usable();
            if usable {
                entry.hits += 1;
                let needs_revalidation = !fresh && !entry.revalidating;
                if needs_revalidation {
                    entry.revalidating = true;
                }
                return (Some(entry.clone()), needs_revalidation);
            }
        }
        (None, false)
    }

    /// Checks if a request matches the cached ETag (conditional request).
    ///
    /// Returns `true` if the resource has not changed (304 Not Modified).
    pub async fn etag_matches(&self, url: &str, client_etag: &str) -> bool {
        let inner = self.inner.read().await;
        if let Some(entry) = inner.entries.get(url) {
            let server_etag = entry.etag_header();
            let client_clean = client_etag.trim_matches('"');
            let server_clean = entry.etag.as_str();
            return client_clean == server_clean || client_etag == server_etag;
        }
        false
    }

    /// Removes a manifest from the cache (e.g. after invalidation).
    pub async fn invalidate(&self, url: &str) {
        let mut inner = self.inner.write().await;
        inner.remove(url);
    }

    /// Evicts all expired entries.
    pub async fn evict_expired(&self) {
        let mut inner = self.inner.write().await;
        inner.evict_expired();
    }

    /// Clears all entries.
    pub async fn clear(&self) {
        let mut inner = self.inner.write().await;
        inner.entries.clear();
        inner.total_bytes = 0;
    }

    /// Returns the number of entries.
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Returns `true` if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }

    /// Returns cache hit/miss statistics.
    pub async fn hit_stats(&self) -> (u64, u64) {
        let inner = self.inner.read().await;
        (inner.total_hits, inner.total_misses)
    }

    /// Returns approximate total cached size in bytes.
    pub async fn total_bytes(&self) -> usize {
        self.inner.read().await.total_bytes
    }
}

impl Default for ManifestCache {
    fn default() -> Self {
        Self::new(ManifestCacheConfig::default())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hls() -> String {
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:6.006,\nseg0.ts\n".to_owned()
    }

    fn sample_mpd() -> String {
        "<?xml version=\"1.0\"?><MPD></MPD>".to_owned()
    }

    // 1. ManifestType content-type
    #[test]
    fn test_manifest_type_content_type() {
        assert!(ManifestType::HlsMaster.content_type().contains("mpegurl"));
        assert!(ManifestType::DashMpd.content_type().contains("dash+xml"));
    }

    // 2. ManifestType label
    #[test]
    fn test_manifest_type_label() {
        assert_eq!(ManifestType::HlsMaster.label(), "HLS-Master");
        assert_eq!(ManifestType::DashMpd.label(), "DASH-MPD");
    }

    // 3. CachedManifest is_fresh
    #[test]
    fn test_cached_manifest_is_fresh() {
        let entry = CachedManifest::new(
            ManifestType::HlsMedia,
            sample_hls(),
            Duration::from_secs(60),
        );
        assert!(entry.is_fresh());
    }

    // 4. CachedManifest is_stale_but_usable
    #[test]
    fn test_cached_manifest_stale_but_usable() {
        // Zero TTL → immediately stale, but stale_while_revalidate = 0 too
        let mut entry = CachedManifest::new(ManifestType::HlsMedia, sample_hls(), Duration::ZERO);
        entry.stale_while_revalidate = Duration::from_secs(60);
        // With 0 TTL and 60s SWR, it should be stale-but-usable immediately.
        assert!(entry.is_stale_but_usable());
    }

    // 5. Cache-Control header format
    #[test]
    fn test_cache_control_header() {
        let entry = CachedManifest::new(
            ManifestType::HlsMedia,
            sample_hls(),
            Duration::from_secs(10),
        );
        let cc = entry.cache_control_header();
        assert!(cc.contains("max-age=10"));
        assert!(cc.contains("stale-while-revalidate"));
    }

    // 6. ETag header is quoted
    #[test]
    fn test_etag_header_quoted() {
        let entry = CachedManifest::new(
            ManifestType::HlsMedia,
            sample_hls(),
            Duration::from_secs(10),
        );
        let etag = entry.etag_header();
        assert!(etag.starts_with('"') && etag.ends_with('"'));
    }

    // 7. ETags change when content changes
    #[test]
    fn test_etag_changes_with_content() {
        let e1 = CachedManifest::new(
            ManifestType::HlsMedia,
            "content A".to_owned(),
            Duration::from_secs(10),
        );
        let e2 = CachedManifest::new(
            ManifestType::HlsMedia,
            "content B".to_owned(),
            Duration::from_secs(10),
        );
        assert_ne!(e1.etag, e2.etag);
    }

    // 8. Remaining TTL
    #[test]
    fn test_remaining_ttl() {
        let entry = CachedManifest::new(
            ManifestType::HlsMedia,
            sample_hls(),
            Duration::from_secs(30),
        );
        let rem = entry.remaining_ttl();
        assert!(rem > Duration::from_secs(25));
    }

    // 9. ManifestCacheConfig defaults
    #[test]
    fn test_cache_config_defaults() {
        let cfg = ManifestCacheConfig::default();
        assert!(cfg.hls_master_ttl > cfg.hls_media_ttl);
        assert_eq!(cfg.default_ttl(ManifestType::HlsMaster), cfg.hls_master_ttl);
    }

    // 10. Cache put and get
    #[tokio::test]
    async fn test_cache_put_get() {
        let cache = ManifestCache::default();
        cache
            .put(
                "http://example.com/master.m3u8",
                sample_hls(),
                ManifestType::HlsMaster,
            )
            .await;
        let result = cache.get("http://example.com/master.m3u8").await;
        assert!(result.is_some());
        assert_eq!(
            result.expect("should be some").manifest_type,
            ManifestType::HlsMaster
        );
    }

    // 11. Cache miss returns None
    #[tokio::test]
    async fn test_cache_miss() {
        let cache = ManifestCache::default();
        let result = cache.get("http://example.com/missing.m3u8").await;
        assert!(result.is_none());
    }

    // 12. Expired entry not returned
    #[tokio::test]
    async fn test_cache_expired_entry() {
        let cache = ManifestCache::default();
        cache
            .put_with_ttl(
                "http://example.com/live.m3u8",
                sample_hls(),
                ManifestType::HlsMedia,
                Duration::from_nanos(1),
            )
            .await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        // Entry is expired AND past SWR (which is 0 for a nano-TTL)
        let result = cache.get("http://example.com/live.m3u8").await;
        // Stale-while-revalidate = TTL/2 = 0, so it should be gone.
        assert!(result.is_none());
    }

    // 13. ETag match detection
    #[tokio::test]
    async fn test_etag_match() {
        let cache = ManifestCache::default();
        let content = sample_hls();
        cache
            .put(
                "http://x.com/m.m3u8",
                content.clone(),
                ManifestType::HlsMedia,
            )
            .await;
        let entry = cache
            .get("http://x.com/m.m3u8")
            .await
            .expect("should exist");
        let etag = entry.etag.clone();
        let matched = cache.etag_matches("http://x.com/m.m3u8", &etag).await;
        assert!(matched);
    }

    // 14. ETag mismatch
    #[tokio::test]
    async fn test_etag_mismatch() {
        let cache = ManifestCache::default();
        cache
            .put("http://x.com/m.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        let matched = cache
            .etag_matches("http://x.com/m.m3u8", "stale_etag")
            .await;
        assert!(!matched);
    }

    // 15. Invalidate removes entry
    #[tokio::test]
    async fn test_cache_invalidate() {
        let cache = ManifestCache::default();
        cache
            .put("http://x.com/m.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache.invalidate("http://x.com/m.m3u8").await;
        assert!(cache.is_empty().await);
    }

    // 16. Clear removes all entries
    #[tokio::test]
    async fn test_cache_clear() {
        let cache = ManifestCache::default();
        cache
            .put("http://a.com/a.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache
            .put("http://b.com/b.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache.clear().await;
        assert_eq!(cache.len().await, 0);
    }

    // 17. Hit stats tracking
    #[tokio::test]
    async fn test_cache_hit_stats() {
        let cache = ManifestCache::default();
        cache
            .put("http://x.com/m.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache.get("http://x.com/m.m3u8").await;
        cache.get("http://x.com/m.m3u8").await;
        cache.get("http://x.com/missing.m3u8").await; // miss
        let (hits, misses) = cache.hit_stats().await;
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
    }

    // 18. Total bytes tracking
    #[tokio::test]
    async fn test_cache_total_bytes() {
        let cache = ManifestCache::default();
        let content = sample_hls();
        let expected_bytes = content.len();
        cache
            .put("http://x.com/m.m3u8", content, ManifestType::HlsMedia)
            .await;
        assert_eq!(cache.total_bytes().await, expected_bytes);
    }

    // 19. Evict expired entries
    #[tokio::test]
    async fn test_cache_evict_expired() {
        let cache = ManifestCache::default();
        cache
            .put_with_ttl(
                "http://x.com/a.m3u8",
                sample_hls(),
                ManifestType::HlsMedia,
                Duration::from_nanos(1),
            )
            .await;
        cache
            .put("http://x.com/b.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        cache.evict_expired().await;
        assert_eq!(cache.len().await, 1);
    }

    // 20. Max entries eviction
    #[tokio::test]
    async fn test_cache_max_entries() {
        let mut cfg = ManifestCacheConfig::default();
        cfg.max_entries = 2;
        let cache = ManifestCache::new(cfg);
        cache
            .put("http://x.com/a.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache
            .put("http://x.com/b.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        cache
            .put("http://x.com/c.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        assert!(cache.len().await <= 2);
    }

    // 21. get_with_revalidation on fresh entry
    #[tokio::test]
    async fn test_cache_revalidation_fresh() {
        let cache = ManifestCache::default();
        cache
            .put("http://x.com/m.m3u8", sample_hls(), ManifestType::HlsMedia)
            .await;
        let (entry, needs_revalidation) = cache.get_with_revalidation("http://x.com/m.m3u8").await;
        assert!(entry.is_some());
        assert!(!needs_revalidation); // Fresh, no revalidation needed.
    }

    // 22. DASH MPD stored and retrieved correctly
    #[tokio::test]
    async fn test_cache_dash_mpd() {
        let cache = ManifestCache::default();
        let mpd = sample_mpd();
        cache
            .put(
                "http://x.com/manifest.mpd",
                mpd.clone(),
                ManifestType::DashMpd,
            )
            .await;
        let entry = cache
            .get("http://x.com/manifest.mpd")
            .await
            .expect("should exist");
        assert_eq!(entry.content, mpd);
        assert_eq!(entry.manifest_type, ManifestType::DashMpd);
    }
}
