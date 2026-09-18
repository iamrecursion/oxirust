//! Media proxy module for proxying media requests to external storage with caching.
//!
//! Provides a transparent caching proxy layer between clients and external media
//! storage backends (S3, GCS, Azure, HTTP origins). Supports cache-through reads,
//! range requests, and origin failover.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Supported external storage backends.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StorageBackend {
    /// HTTP/HTTPS origin server.
    Http { base_url: String },
    /// S3-compatible storage.
    S3 { bucket: String, region: String },
    /// Google Cloud Storage.
    Gcs { bucket: String },
    /// Azure Blob Storage.
    AzureBlob { container: String, account: String },
    /// Local filesystem.
    Local { base_path: String },
}

impl StorageBackend {
    /// Returns a label for the backend type.
    pub fn label(&self) -> &str {
        match self {
            Self::Http { .. } => "http",
            Self::S3 { .. } => "s3",
            Self::Gcs { .. } => "gcs",
            Self::AzureBlob { .. } => "azure-blob",
            Self::Local { .. } => "local",
        }
    }

    /// Constructs the full URL/path for a given key.
    pub fn resolve_path(&self, key: &str) -> String {
        match self {
            Self::Http { base_url } => format!("{}/{}", base_url.trim_end_matches('/'), key),
            Self::S3 { bucket, region } => format!("s3://{}.{}/{}", bucket, region, key),
            Self::Gcs { bucket } => format!("gs://{}/{}", bucket, key),
            Self::AzureBlob { container, account } => {
                format!("azure://{}/{}/{}", account, container, key)
            }
            Self::Local { base_path } => format!("{}/{}", base_path, key),
        }
    }
}

/// Configuration for the media proxy.
#[derive(Debug, Clone)]
pub struct MediaProxyConfig {
    /// Maximum cached content size per entry.
    pub max_cache_entry_size: u64,
    /// Total cache capacity.
    pub max_cache_size: u64,
    /// Default TTL for cached entries.
    pub cache_ttl: Duration,
    /// Timeout for upstream requests.
    pub upstream_timeout: Duration,
    /// Whether to cache 404 responses.
    pub cache_negative: bool,
    /// Negative cache TTL.
    pub negative_cache_ttl: Duration,
    /// Whether to support range requests.
    pub support_range_requests: bool,
    /// Maximum number of origins for failover.
    pub max_origins: usize,
}

impl Default for MediaProxyConfig {
    fn default() -> Self {
        Self {
            max_cache_entry_size: 256 * 1024 * 1024, // 256 MB
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10 GB
            cache_ttl: Duration::from_secs(3600),
            upstream_timeout: Duration::from_secs(30),
            cache_negative: true,
            negative_cache_ttl: Duration::from_secs(60),
            support_range_requests: true,
            max_origins: 5,
        }
    }
}

/// A cached proxy entry.
#[derive(Debug, Clone)]
pub struct ProxyCacheEntry {
    /// Content bytes (may be None for large files stored on disk).
    pub body: Option<Vec<u8>>,
    /// Content type.
    pub content_type: String,
    /// Content length.
    pub content_length: u64,
    /// Origin ETag.
    pub etag: Option<String>,
    /// Origin last-modified.
    pub last_modified: Option<String>,
    /// When the entry was cached.
    pub cached_at: Instant,
    /// TTL for this entry.
    pub ttl: Duration,
    /// Number of times served from cache.
    pub hit_count: u64,
    /// Whether this is a negative (404) cache entry.
    pub is_negative: bool,
}

impl ProxyCacheEntry {
    /// Whether the entry has expired.
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }

    /// Remaining TTL.
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.cached_at.elapsed())
    }

    /// Age of the entry.
    pub fn age(&self) -> Duration {
        self.cached_at.elapsed()
    }
}

/// Proxy request metadata.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    /// The requested key/path.
    pub key: String,
    /// Range request header value.
    pub range: Option<String>,
    /// If-None-Match header.
    pub if_none_match: Option<String>,
    /// If-Modified-Since header.
    pub if_modified_since: Option<String>,
    /// Client IP for logging.
    pub client_ip: Option<String>,
}

/// Proxy response.
#[derive(Debug, Clone)]
pub struct ProxyResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body.
    pub body: Vec<u8>,
    /// Content-Type header.
    pub content_type: String,
    /// Whether served from cache.
    pub from_cache: bool,
    /// Response headers.
    pub headers: HashMap<String, String>,
}

/// Statistics for the proxy.
#[derive(Debug, Clone, Default)]
pub struct ProxyStats {
    /// Total requests.
    pub total_requests: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses (fetched from origin).
    pub cache_misses: u64,
    /// Origin errors.
    pub origin_errors: u64,
    /// Bytes served from cache.
    pub cache_bytes_served: u64,
    /// Bytes fetched from origin.
    pub origin_bytes_fetched: u64,
    /// Current cache entry count.
    pub cache_entries: usize,
    /// Current cache size in bytes.
    pub cache_bytes: u64,
}

impl ProxyStats {
    /// Cache hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_requests as f64
    }

    /// Bandwidth savings ratio (cache bytes / total bytes).
    pub fn bandwidth_savings(&self) -> f64 {
        let total = self.cache_bytes_served + self.origin_bytes_fetched;
        if total == 0 {
            return 0.0;
        }
        self.cache_bytes_served as f64 / total as f64
    }
}

/// Origin health status.
#[derive(Debug, Clone)]
pub struct OriginHealth {
    /// Backend identifier.
    pub backend: StorageBackend,
    /// Whether the origin is healthy.
    pub healthy: bool,
    /// Last successful request time.
    pub last_success: Option<Instant>,
    /// Last failure time.
    pub last_failure: Option<Instant>,
    /// Consecutive failure count.
    pub consecutive_failures: u32,
    /// Average response time.
    pub avg_response_time_ms: f64,
}

/// The media proxy.
pub struct MediaProxy {
    config: MediaProxyConfig,
    /// Origins in priority order.
    origins: Vec<StorageBackend>,
    /// Origin health status.
    origin_health: HashMap<String, OriginHealth>,
    /// In-memory cache.
    cache: HashMap<String, ProxyCacheEntry>,
    /// Current total cache size.
    cache_size: u64,
    /// Statistics.
    stats: ProxyStats,
}

impl MediaProxy {
    /// Creates a new media proxy.
    pub fn new(config: MediaProxyConfig) -> Self {
        Self {
            config,
            origins: Vec::new(),
            origin_health: HashMap::new(),
            cache: HashMap::new(),
            cache_size: 0,
            stats: ProxyStats::default(),
        }
    }

    /// Adds an origin backend.
    pub fn add_origin(&mut self, backend: StorageBackend) -> bool {
        if self.origins.len() >= self.config.max_origins {
            return false;
        }
        let label = backend.label().to_string();
        self.origin_health.insert(
            label,
            OriginHealth {
                backend: backend.clone(),
                healthy: true,
                last_success: None,
                last_failure: None,
                consecutive_failures: 0,
                avg_response_time_ms: 0.0,
            },
        );
        self.origins.push(backend);
        true
    }

    /// Returns the number of configured origins.
    pub fn origin_count(&self) -> usize {
        self.origins.len()
    }

    /// Returns healthy origins.
    pub fn healthy_origins(&self) -> Vec<&StorageBackend> {
        self.origins
            .iter()
            .filter(|o| {
                self.origin_health
                    .get(o.label())
                    .map(|h| h.healthy)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Looks up a cached entry.
    pub fn cache_lookup(&mut self, key: &str) -> Option<&ProxyCacheEntry> {
        self.stats.total_requests += 1;

        let expired = self.cache.get(key).map(|e| e.is_expired());
        match expired {
            Some(true) => {
                self.evict_key(key);
                self.stats.cache_misses += 1;
                None
            }
            Some(false) => {
                self.stats.cache_hits += 1;
                if let Some(entry) = self.cache.get_mut(key) {
                    entry.hit_count += 1;
                    self.stats.cache_bytes_served += entry.content_length;
                }
                self.cache.get(key)
            }
            None => {
                self.stats.cache_misses += 1;
                None
            }
        }
    }

    /// Stores a response in the cache.
    pub fn cache_store(&mut self, key: &str, entry: ProxyCacheEntry) {
        let size = entry.content_length;

        // Don't cache if too large
        if size > self.config.max_cache_entry_size {
            return;
        }

        // Evict until we have space
        while self.cache_size + size > self.config.max_cache_size && !self.cache.is_empty() {
            self.evict_lru();
        }

        // Remove existing entry
        self.evict_key(key);

        self.cache_size += size;
        self.stats.cache_entries = self.cache.len() + 1;
        self.stats.cache_bytes = self.cache_size;
        self.cache.insert(key.to_string(), entry);
    }

    /// Invalidates a cached entry.
    pub fn cache_invalidate(&mut self, key: &str) -> bool {
        self.evict_key(key)
    }

    /// Invalidates all entries matching a prefix.
    pub fn cache_invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys: Vec<String> = self
            .cache
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = keys.len();
        for key in keys {
            self.evict_key(&key);
        }
        count
    }

    /// Purges expired entries.
    pub fn purge_expired(&mut self) -> usize {
        let expired: Vec<String> = self
            .cache
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for key in expired {
            self.evict_key(&key);
        }
        count
    }

    /// Records an origin success.
    pub fn record_origin_success(&mut self, backend_label: &str, response_time_ms: f64) {
        if let Some(health) = self.origin_health.get_mut(backend_label) {
            health.healthy = true;
            health.last_success = Some(Instant::now());
            health.consecutive_failures = 0;
            // Running average
            health.avg_response_time_ms =
                health.avg_response_time_ms * 0.9 + response_time_ms * 0.1;
        }
    }

    /// Records an origin failure.
    pub fn record_origin_failure(&mut self, backend_label: &str) {
        if let Some(health) = self.origin_health.get_mut(backend_label) {
            health.last_failure = Some(Instant::now());
            health.consecutive_failures += 1;
            if health.consecutive_failures >= 3 {
                health.healthy = false;
            }
        }
        self.stats.origin_errors += 1;
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &ProxyStats {
        &self.stats
    }

    /// Returns cache entry count.
    pub fn cache_entry_count(&self) -> usize {
        self.cache.len()
    }

    /// Returns total cache size in bytes.
    pub fn cache_total_bytes(&self) -> u64 {
        self.cache_size
    }

    fn evict_key(&mut self, key: &str) -> bool {
        if let Some(entry) = self.cache.remove(key) {
            self.cache_size = self.cache_size.saturating_sub(entry.content_length);
            self.stats.cache_entries = self.cache.len();
            self.stats.cache_bytes = self.cache_size;
            true
        } else {
            false
        }
    }

    fn evict_lru(&mut self) {
        let lru_key = self
            .cache
            .iter()
            .min_by_key(|(_, e)| e.cached_at)
            .map(|(k, _)| k.clone());
        if let Some(key) = lru_key {
            self.evict_key(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(size: u64, ttl_secs: u64) -> ProxyCacheEntry {
        ProxyCacheEntry {
            body: Some(vec![0u8; size as usize]),
            content_type: "video/mp4".to_string(),
            content_length: size,
            etag: None,
            last_modified: None,
            cached_at: Instant::now(),
            ttl: Duration::from_secs(ttl_secs),
            hit_count: 0,
            is_negative: false,
        }
    }

    // StorageBackend

    #[test]
    fn test_backend_labels() {
        assert_eq!(
            StorageBackend::Http {
                base_url: "x".into()
            }
            .label(),
            "http"
        );
        assert_eq!(
            StorageBackend::S3 {
                bucket: "b".into(),
                region: "r".into()
            }
            .label(),
            "s3"
        );
        assert_eq!(StorageBackend::Gcs { bucket: "b".into() }.label(), "gcs");
    }

    #[test]
    fn test_backend_resolve_path_http() {
        let b = StorageBackend::Http {
            base_url: "https://cdn.example.com".into(),
        };
        assert_eq!(
            b.resolve_path("media/file.mp4"),
            "https://cdn.example.com/media/file.mp4"
        );
    }

    #[test]
    fn test_backend_resolve_path_s3() {
        let b = StorageBackend::S3 {
            bucket: "my-bucket".into(),
            region: "us-east-1".into(),
        };
        assert!(b.resolve_path("key").contains("my-bucket"));
    }

    // ProxyCacheEntry

    #[test]
    fn test_entry_not_expired() {
        let entry = make_entry(100, 60);
        assert!(!entry.is_expired());
        assert!(entry.remaining_ttl() > Duration::ZERO);
    }

    #[test]
    fn test_entry_expired() {
        let entry = make_entry(100, 0);
        std::thread::sleep(Duration::from_millis(5));
        assert!(entry.is_expired());
    }

    // ProxyStats

    #[test]
    fn test_stats_hit_ratio() {
        let mut stats = ProxyStats::default();
        stats.total_requests = 10;
        stats.cache_hits = 7;
        assert!((stats.hit_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_stats_hit_ratio_zero() {
        let stats = ProxyStats::default();
        assert!((stats.hit_ratio()).abs() < 1e-9);
    }

    #[test]
    fn test_stats_bandwidth_savings() {
        let mut stats = ProxyStats::default();
        stats.cache_bytes_served = 700;
        stats.origin_bytes_fetched = 300;
        assert!((stats.bandwidth_savings() - 0.7).abs() < 1e-9);
    }

    // MediaProxy

    #[test]
    fn test_proxy_add_origin() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        assert!(proxy.add_origin(StorageBackend::Http {
            base_url: "https://cdn1.example.com".into()
        }));
        assert_eq!(proxy.origin_count(), 1);
    }

    #[test]
    fn test_proxy_max_origins() {
        let config = MediaProxyConfig {
            max_origins: 2,
            ..Default::default()
        };
        let mut proxy = MediaProxy::new(config);
        assert!(proxy.add_origin(StorageBackend::Http {
            base_url: "u1".into()
        }));
        assert!(proxy.add_origin(StorageBackend::Http {
            base_url: "u2".into()
        }));
        assert!(!proxy.add_origin(StorageBackend::Http {
            base_url: "u3".into()
        }));
    }

    #[test]
    fn test_proxy_cache_store_and_lookup() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("key1", make_entry(100, 60));
        let entry = proxy.cache_lookup("key1");
        assert!(entry.is_some());
    }

    #[test]
    fn test_proxy_cache_miss() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        assert!(proxy.cache_lookup("missing").is_none());
        assert_eq!(proxy.stats().cache_misses, 1);
    }

    #[test]
    fn test_proxy_cache_expired_entry() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("key1", make_entry(100, 0));
        std::thread::sleep(Duration::from_millis(5));
        assert!(proxy.cache_lookup("key1").is_none());
    }

    #[test]
    fn test_proxy_cache_invalidate() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("key1", make_entry(100, 60));
        assert!(proxy.cache_invalidate("key1"));
        assert_eq!(proxy.cache_entry_count(), 0);
    }

    #[test]
    fn test_proxy_cache_invalidate_prefix() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("media/1", make_entry(100, 60));
        proxy.cache_store("media/2", make_entry(100, 60));
        proxy.cache_store("other/1", make_entry(100, 60));
        let count = proxy.cache_invalidate_prefix("media/");
        assert_eq!(count, 2);
        assert_eq!(proxy.cache_entry_count(), 1);
    }

    #[test]
    fn test_proxy_purge_expired() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("expired", make_entry(100, 0));
        proxy.cache_store("valid", make_entry(100, 600));
        std::thread::sleep(Duration::from_millis(5));
        let purged = proxy.purge_expired();
        assert_eq!(purged, 1);
        assert_eq!(proxy.cache_entry_count(), 1);
    }

    #[test]
    fn test_proxy_too_large_entry_not_cached() {
        let config = MediaProxyConfig {
            max_cache_entry_size: 50,
            ..Default::default()
        };
        let mut proxy = MediaProxy::new(config);
        proxy.cache_store("big", make_entry(100, 60));
        assert_eq!(proxy.cache_entry_count(), 0);
    }

    #[test]
    fn test_proxy_origin_health_tracking() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.add_origin(StorageBackend::Http {
            base_url: "https://origin.example.com".into(),
        });

        proxy.record_origin_success("http", 50.0);
        let healthy = proxy.healthy_origins();
        assert_eq!(healthy.len(), 1);

        // 3 failures should mark unhealthy
        proxy.record_origin_failure("http");
        proxy.record_origin_failure("http");
        proxy.record_origin_failure("http");
        let healthy = proxy.healthy_origins();
        assert_eq!(healthy.len(), 0);
    }

    #[test]
    fn test_proxy_origin_recovery() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.add_origin(StorageBackend::Http {
            base_url: "https://origin.example.com".into(),
        });

        // Mark as unhealthy
        for _ in 0..3 {
            proxy.record_origin_failure("http");
        }
        assert_eq!(proxy.healthy_origins().len(), 0);

        // Recovery
        proxy.record_origin_success("http", 30.0);
        assert_eq!(proxy.healthy_origins().len(), 1);
    }

    #[test]
    fn test_proxy_cache_size_tracking() {
        let mut proxy = MediaProxy::new(MediaProxyConfig::default());
        proxy.cache_store("a", make_entry(100, 60));
        proxy.cache_store("b", make_entry(200, 60));
        assert_eq!(proxy.cache_total_bytes(), 300);
        proxy.cache_invalidate("a");
        assert_eq!(proxy.cache_total_bytes(), 200);
    }

    #[test]
    fn test_default_config() {
        let cfg = MediaProxyConfig::default();
        assert!(cfg.support_range_requests);
        assert!(cfg.cache_negative);
        assert_eq!(cfg.max_origins, 5);
    }
}
