//! ETags and conditional GET (If-None-Match) for media metadata endpoints.
//!
//! Provides ETag generation, storage, and conditional-GET validation.
//! Integrates with `ResponseCache` to avoid recomputing metadata when
//! clients send `If-None-Match` headers.

#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Strategy for generating ETags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtagStrategy {
    /// Strong ETag: content hash (SHA-256 truncated).
    Strong,
    /// Weak ETag: based on last-modified + size.
    Weak,
}

/// A computed ETag value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Etag {
    /// The ETag value (without surrounding quotes).
    pub value: String,
    /// Whether this is a weak ETag.
    pub weak: bool,
}

impl Etag {
    /// Creates a strong ETag from content bytes.
    pub fn from_content(content: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        // Use first 16 bytes (128 bits) for shorter ETags
        let value = hex::encode(&hash[..16]);
        Self { value, weak: false }
    }

    /// Creates a strong ETag from a string.
    pub fn from_string(s: &str) -> Self {
        Self::from_content(s.as_bytes())
    }

    /// Creates a weak ETag from last-modified timestamp and size.
    pub fn weak_from_metadata(last_modified: u64, size: u64) -> Self {
        let value = format!("{:x}-{:x}", last_modified, size);
        Self { value, weak: true }
    }

    /// Formats the ETag for use in HTTP headers.
    pub fn header_value(&self) -> String {
        if self.weak {
            format!("W/\"{}\"", self.value)
        } else {
            format!("\"{}\"", self.value)
        }
    }

    /// Parses an ETag from an HTTP header value.
    pub fn parse(header_value: &str) -> Option<Self> {
        let trimmed = header_value.trim();

        if let Some(rest) = trimmed.strip_prefix("W/") {
            let value = rest.trim_matches('"');
            if value.is_empty() {
                return None;
            }
            Some(Self {
                value: value.to_string(),
                weak: true,
            })
        } else {
            let value = trimmed.trim_matches('"');
            if value.is_empty() {
                return None;
            }
            Some(Self {
                value: value.to_string(),
                weak: false,
            })
        }
    }

    /// Checks whether this ETag matches another for conditional request purposes.
    ///
    /// Strong comparison: both must be strong and have the same value.
    /// Weak comparison: values must match (ignores weak/strong distinction).
    pub fn matches(&self, other: &Self, strong_comparison: bool) -> bool {
        if strong_comparison {
            !self.weak && !other.weak && self.value == other.value
        } else {
            self.value == other.value
        }
    }
}

/// Result of a conditional GET check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionalResult {
    /// The resource has not been modified — return 304.
    NotModified {
        /// The matching ETag.
        etag: Etag,
    },
    /// The resource has been modified or no ETag was provided — return full response.
    Modified,
    /// The `If-None-Match` header contained `*` (resource exists).
    Exists,
}

impl ConditionalResult {
    /// Returns `true` for `NotModified`.
    pub fn is_not_modified(&self) -> bool {
        matches!(self, Self::NotModified { .. })
    }

    /// Returns the HTTP status code.
    pub fn status_code(&self) -> u16 {
        match self {
            Self::NotModified { .. } => 304,
            Self::Modified | Self::Exists => 200,
        }
    }
}

/// A cached ETag entry with metadata.
#[derive(Debug, Clone)]
struct EtagEntry {
    /// The ETag value.
    etag: Etag,
    /// When the ETag was generated.
    created_at: Instant,
    /// Last time the ETag was validated.
    last_validated: Instant,
    /// TTL after which the ETag should be regenerated.
    ttl: Duration,
}

impl EtagEntry {
    fn is_stale(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Statistics for the ETag cache.
#[derive(Debug, Clone, Default)]
pub struct EtagCacheStats {
    /// Total conditional requests received.
    pub conditional_requests: u64,
    /// Requests that returned 304 Not Modified.
    pub not_modified: u64,
    /// Requests that returned full content.
    pub full_responses: u64,
    /// ETags generated.
    pub etags_generated: u64,
    /// ETags invalidated.
    pub etags_invalidated: u64,
}

impl EtagCacheStats {
    /// Not-modified ratio (bandwidth savings).
    pub fn not_modified_ratio(&self) -> f64 {
        if self.conditional_requests == 0 {
            return 0.0;
        }
        self.not_modified as f64 / self.conditional_requests as f64
    }
}

/// Configuration for the ETag cache.
#[derive(Debug, Clone)]
pub struct EtagCacheConfig {
    /// Default TTL for generated ETags.
    pub default_ttl: Duration,
    /// Maximum number of cached ETags.
    pub max_entries: usize,
    /// ETag generation strategy.
    pub strategy: EtagStrategy,
}

impl Default for EtagCacheConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(300),
            max_entries: 10_000,
            strategy: EtagStrategy::Strong,
        }
    }
}

/// ETag cache for media metadata endpoints.
pub struct EtagCache {
    config: EtagCacheConfig,
    /// resource_key -> EtagEntry
    entries: HashMap<String, EtagEntry>,
    stats: EtagCacheStats,
}

impl EtagCache {
    /// Creates a new ETag cache.
    pub fn new(config: EtagCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: EtagCacheStats::default(),
        }
    }

    /// Generates and caches an ETag for a resource.
    pub fn generate(&mut self, resource_key: &str, content: &[u8]) -> Etag {
        let etag = match self.config.strategy {
            EtagStrategy::Strong => Etag::from_content(content),
            EtagStrategy::Weak => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                Etag::weak_from_metadata(now, content.len() as u64)
            }
        };

        // Evict oldest if at capacity
        if self.entries.len() >= self.config.max_entries && !self.entries.contains_key(resource_key)
        {
            self.evict_oldest();
        }

        let now = Instant::now();
        self.entries.insert(
            resource_key.to_string(),
            EtagEntry {
                etag: etag.clone(),
                created_at: now,
                last_validated: now,
                ttl: self.config.default_ttl,
            },
        );

        self.stats.etags_generated += 1;
        etag
    }

    /// Looks up the current ETag for a resource.
    pub fn get_etag(&self, resource_key: &str) -> Option<&Etag> {
        self.entries
            .get(resource_key)
            .filter(|e| !e.is_stale())
            .map(|e| &e.etag)
    }

    /// Evaluates a conditional GET request.
    ///
    /// `if_none_match` is the value of the `If-None-Match` header.
    pub fn check_conditional(
        &mut self,
        resource_key: &str,
        if_none_match: &str,
    ) -> ConditionalResult {
        self.stats.conditional_requests += 1;

        // Handle `*` wildcard
        if if_none_match.trim() == "*" {
            if self.entries.contains_key(resource_key) {
                self.stats.not_modified += 1;
                return ConditionalResult::Exists;
            }
            self.stats.full_responses += 1;
            return ConditionalResult::Modified;
        }

        // Parse the client's ETags (could be comma-separated)
        let client_etags: Vec<Etag> = if_none_match
            .split(',')
            .filter_map(|s| Etag::parse(s.trim()))
            .collect();

        // Look up current ETag
        let current = match self.entries.get_mut(resource_key) {
            Some(entry) if !entry.is_stale() => {
                entry.last_validated = Instant::now();
                entry.etag.clone()
            }
            _ => {
                self.stats.full_responses += 1;
                return ConditionalResult::Modified;
            }
        };

        // Weak comparison for conditional GET (RFC 7232 Section 3.2)
        for client_etag in &client_etags {
            if current.matches(client_etag, false) {
                self.stats.not_modified += 1;
                return ConditionalResult::NotModified { etag: current };
            }
        }

        self.stats.full_responses += 1;
        ConditionalResult::Modified
    }

    /// Invalidates the ETag for a resource (e.g. after update).
    pub fn invalidate(&mut self, resource_key: &str) -> bool {
        if self.entries.remove(resource_key).is_some() {
            self.stats.etags_invalidated += 1;
            true
        } else {
            false
        }
    }

    /// Invalidates all ETags matching a prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = keys.len();
        for key in keys {
            self.entries.remove(&key);
            self.stats.etags_invalidated += 1;
        }
        count
    }

    /// Purges stale entries.
    pub fn purge_stale(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| !e.is_stale());
        before - self.entries.len()
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &EtagCacheStats {
        &self.stats
    }

    /// Returns the number of cached ETags.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&mut self) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone());
        if let Some(key) = oldest {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Etag

    #[test]
    fn test_etag_from_content() {
        let etag = Etag::from_content(b"hello world");
        assert!(!etag.weak);
        assert!(!etag.value.is_empty());
    }

    #[test]
    fn test_etag_from_content_deterministic() {
        let e1 = Etag::from_content(b"same content");
        let e2 = Etag::from_content(b"same content");
        assert_eq!(e1.value, e2.value);
    }

    #[test]
    fn test_etag_from_content_different() {
        let e1 = Etag::from_content(b"content a");
        let e2 = Etag::from_content(b"content b");
        assert_ne!(e1.value, e2.value);
    }

    #[test]
    fn test_etag_from_string() {
        let etag = Etag::from_string("test data");
        assert!(!etag.weak);
    }

    #[test]
    fn test_etag_weak_from_metadata() {
        let etag = Etag::weak_from_metadata(1234567890, 1024);
        assert!(etag.weak);
    }

    #[test]
    fn test_etag_header_value_strong() {
        let etag = Etag::from_content(b"test");
        let header = etag.header_value();
        assert!(header.starts_with('"'));
        assert!(header.ends_with('"'));
        assert!(!header.starts_with("W/"));
    }

    #[test]
    fn test_etag_header_value_weak() {
        let etag = Etag::weak_from_metadata(100, 200);
        let header = etag.header_value();
        assert!(header.starts_with("W/\""));
    }

    #[test]
    fn test_etag_parse_strong() {
        let etag = Etag::parse("\"abc123\"").expect("should parse");
        assert_eq!(etag.value, "abc123");
        assert!(!etag.weak);
    }

    #[test]
    fn test_etag_parse_weak() {
        let etag = Etag::parse("W/\"abc123\"").expect("should parse");
        assert_eq!(etag.value, "abc123");
        assert!(etag.weak);
    }

    #[test]
    fn test_etag_parse_empty() {
        assert!(Etag::parse("\"\"").is_none());
    }

    #[test]
    fn test_etag_matches_weak_comparison() {
        let e1 = Etag {
            value: "abc".to_string(),
            weak: true,
        };
        let e2 = Etag {
            value: "abc".to_string(),
            weak: false,
        };
        assert!(e1.matches(&e2, false)); // weak comparison
    }

    #[test]
    fn test_etag_matches_strong_comparison() {
        let e1 = Etag {
            value: "abc".to_string(),
            weak: true,
        };
        let e2 = Etag {
            value: "abc".to_string(),
            weak: false,
        };
        assert!(!e1.matches(&e2, true)); // strong fails: e1 is weak
    }

    #[test]
    fn test_etag_matches_strong_both_strong() {
        let e1 = Etag {
            value: "abc".to_string(),
            weak: false,
        };
        let e2 = Etag {
            value: "abc".to_string(),
            weak: false,
        };
        assert!(e1.matches(&e2, true));
    }

    // ConditionalResult

    #[test]
    fn test_conditional_not_modified() {
        let result = ConditionalResult::NotModified {
            etag: Etag::from_content(b"x"),
        };
        assert!(result.is_not_modified());
        assert_eq!(result.status_code(), 304);
    }

    #[test]
    fn test_conditional_modified() {
        let result = ConditionalResult::Modified;
        assert!(!result.is_not_modified());
        assert_eq!(result.status_code(), 200);
    }

    // EtagCacheStats

    #[test]
    fn test_stats_not_modified_ratio() {
        let mut stats = EtagCacheStats::default();
        stats.conditional_requests = 10;
        stats.not_modified = 7;
        assert!((stats.not_modified_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_stats_not_modified_ratio_zero() {
        let stats = EtagCacheStats::default();
        assert!((stats.not_modified_ratio()).abs() < 1e-9);
    }

    // EtagCache

    #[test]
    fn test_generate_and_get() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        let etag = cache.generate("/media/1/metadata", b"hello");
        let stored = cache.get_etag("/media/1/metadata");
        assert!(stored.is_some());
        assert_eq!(stored.expect("should exist").value, etag.value);
    }

    #[test]
    fn test_get_missing() {
        let cache = EtagCache::new(EtagCacheConfig::default());
        assert!(cache.get_etag("/missing").is_none());
    }

    #[test]
    fn test_check_conditional_not_modified() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        let etag = cache.generate("/media/1", b"data");
        let result = cache.check_conditional("/media/1", &etag.header_value());
        assert!(result.is_not_modified());
    }

    #[test]
    fn test_check_conditional_modified() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        cache.generate("/media/1", b"data");
        let result = cache.check_conditional("/media/1", "\"different-etag\"");
        assert!(!result.is_not_modified());
    }

    #[test]
    fn test_check_conditional_wildcard_exists() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        cache.generate("/media/1", b"data");
        let result = cache.check_conditional("/media/1", "*");
        assert_eq!(result, ConditionalResult::Exists);
    }

    #[test]
    fn test_check_conditional_wildcard_missing() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        let result = cache.check_conditional("/missing", "*");
        assert_eq!(result, ConditionalResult::Modified);
    }

    #[test]
    fn test_check_conditional_multiple_etags() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        let etag = cache.generate("/media/1", b"data");
        let header = format!("\"bad1\", {}, \"bad2\"", etag.header_value());
        let result = cache.check_conditional("/media/1", &header);
        assert!(result.is_not_modified());
    }

    #[test]
    fn test_invalidate() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        cache.generate("/media/1", b"data");
        assert!(cache.invalidate("/media/1"));
        assert!(cache.get_etag("/media/1").is_none());
        assert_eq!(cache.stats().etags_invalidated, 1);
    }

    #[test]
    fn test_invalidate_missing() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        assert!(!cache.invalidate("/missing"));
    }

    #[test]
    fn test_invalidate_prefix() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        cache.generate("/media/1/meta", b"a");
        cache.generate("/media/1/thumb", b"b");
        cache.generate("/media/2/meta", b"c");
        let count = cache.invalidate_prefix("/media/1");
        assert_eq!(count, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_purge_stale() {
        let config = EtagCacheConfig {
            default_ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let mut cache = EtagCache::new(config);
        cache.generate("/a", b"a");
        cache.generate("/b", b"b");
        std::thread::sleep(Duration::from_millis(5));
        let purged = cache.purge_stale();
        assert_eq!(purged, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_max_entries_eviction() {
        let config = EtagCacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = EtagCache::new(config);
        cache.generate("/a", b"a");
        std::thread::sleep(Duration::from_millis(1));
        cache.generate("/b", b"b");
        std::thread::sleep(Duration::from_millis(1));
        cache.generate("/c", b"c");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut cache = EtagCache::new(EtagCacheConfig::default());
        let etag = cache.generate("/media/1", b"data");
        assert_eq!(cache.stats().etags_generated, 1);
        cache.check_conditional("/media/1", &etag.header_value());
        assert_eq!(cache.stats().conditional_requests, 1);
        assert_eq!(cache.stats().not_modified, 1);
    }

    #[test]
    fn test_weak_etag_strategy() {
        let config = EtagCacheConfig {
            strategy: EtagStrategy::Weak,
            ..Default::default()
        };
        let mut cache = EtagCache::new(config);
        let etag = cache.generate("/media/1", b"data");
        assert!(etag.weak);
    }

    #[test]
    fn test_default_config() {
        let cfg = EtagCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert_eq!(cfg.strategy, EtagStrategy::Strong);
    }
}
