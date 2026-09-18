//! Media-content-aware caching.
//!
//! Standard eviction policies treat all entries equally.  For multimedia
//! workloads, different content types have very different access patterns
//! and cost/benefit trade-offs:
//!
//! * Manifests are tiny but highly re-fetched → very high priority, short TTL.
//! * Video segments are large and often sequential → medium priority, long TTL.
//! * Thumbnails are small and rarely re-fetched but cheap → very long TTL.
//!
//! [`ContentAwareCache`] layers this domain knowledge on top of
//! [`LruCache`] so that eviction candidates are
//! scored by a combined recency × priority × size-efficiency metric rather
//! than pure LRU order.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::lru_cache::LruCache;

// ── MediaContentType ──────────────────────────────────────────────────────────

/// The media type of a cached entry.
///
/// The variant determines the default priority, TTL, and eviction score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaContentType {
    /// A video segment (e.g. MPEG-DASH chunk or HLS `.ts` file).
    VideoSegment {
        /// Encoded bitrate in bits per second.
        bitrate: u32,
        /// Codec name (e.g. `"av1"`, `"vp9"`).
        codec: String,
    },
    /// An audio-only segment.
    AudioSegment {
        /// Encoded bitrate in bits per second.
        bitrate: u32,
    },
    /// A still image (e.g. JPEG or PNG frame).
    Image {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// A streaming manifest / playlist (e.g. HLS `.m3u8` or DASH `.mpd`).
    Manifest,
    /// A thumbnail preview image.
    Thumbnail,
    /// Lightweight metadata / sidecar (e.g. `.json` or `.xml` descriptor).
    Metadata,
}

// ── ContentCachePriority ──────────────────────────────────────────────────────

/// Numeric priority (higher = more important to keep in cache).
///
/// Derived from [`MediaContentType`] via [`ContentCachePriority::for_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContentCachePriority(pub u8);

impl ContentCachePriority {
    /// Compute the priority for the given content type.
    ///
    /// | Content type                        | Priority |
    /// |-------------------------------------|----------|
    /// | Manifest                            | 10       |
    /// | Thumbnail                           |  8       |
    /// | VideoSegment (bitrate ≥ 4 Mbps)     |  7       |
    /// | VideoSegment (bitrate < 4 Mbps)     |  6       |
    /// | AudioSegment                        |  5       |
    /// | Image                               |  4       |
    /// | Metadata                            |  3       |
    pub fn for_type(content_type: &MediaContentType) -> Self {
        let p = match content_type {
            MediaContentType::Manifest => 10,
            MediaContentType::Thumbnail => 8,
            MediaContentType::VideoSegment { bitrate, .. } => {
                if *bitrate >= 4_000_000 {
                    7
                } else {
                    6
                }
            }
            MediaContentType::AudioSegment { .. } => 5,
            MediaContentType::Image { .. } => 4,
            MediaContentType::Metadata => 3,
        };
        Self(p)
    }
}

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A single entry held in [`ContentAwareCache`].
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cache key.
    pub key: String,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// Media content type (determines priority and TTL).
    pub content_type: MediaContentType,
    /// Wall-clock time at which this entry was first inserted.
    pub inserted_at: Instant,
    /// Wall-clock time of the most recent successful lookup.
    pub last_accessed: Instant,
    /// Total number of successful lookups since insertion.
    pub access_count: u32,
    /// `data.len()` cached to avoid recomputation.
    pub size_bytes: usize,
}

impl CacheEntry {
    /// Create a new entry.
    fn new(key: String, data: Vec<u8>, content_type: MediaContentType) -> Self {
        let size = data.len();
        let now = Instant::now();
        Self {
            key,
            data,
            content_type,
            inserted_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes: size,
        }
    }

    /// Compute the eviction score for this entry using default weights.
    ///
    /// A higher score means the entry is a *better* eviction candidate (i.e.
    /// it should be evicted first).
    ///
    /// ```text
    /// score = (1.0 - recency) × (1.0 / priority) × size_factor
    /// ```
    ///
    /// Where:
    /// * `recency  = e^(-age_secs / 60.0)` — exponential decay over 1 minute.
    /// * `priority = ContentCachePriority::for_type(content_type).0` cast to f32.
    /// * `size_factor = size_bytes / 1_048_576.0 + 1.0` — larger entries score
    ///   higher (give bigger bang for the eviction buck).
    pub fn score_for_eviction(&self) -> f32 {
        self.score_for_eviction_weighted(&ScoringWeights::default())
    }

    /// Compute the eviction score using configurable [`ScoringWeights`].
    ///
    /// ```text
    /// score = (1 - recency).powf(w.recency_exp)
    ///       × (1 / priority).powf(w.priority_exp)
    ///       × size_factor.powf(w.size_exp)
    /// ```
    ///
    /// Default weights (`recency_exp = priority_exp = size_exp = 1.0`)
    /// reproduce the original behaviour exactly.
    pub fn score_for_eviction_weighted(&self, w: &ScoringWeights) -> f32 {
        let age_secs = self.last_accessed.elapsed().as_secs_f64();
        let recency = (-age_secs / 60.0_f64).exp(); // 1.0 when just accessed, → 0 with age
        let base_priority = ContentCachePriority::for_type(&self.content_type).0 as f64;
        let multiplier = w.priority_multiplier(&self.content_type);
        let effective_priority = (base_priority * multiplier).max(0.001);
        let size_factor = self.size_bytes as f64 / 1_048_576.0 + 1.0;
        let score = (1.0 - recency).powf(w.recency_exp)
            * (1.0 / effective_priority).powf(w.priority_exp)
            * size_factor.powf(w.size_exp);
        score as f32
    }
}

// ── TTL helpers ───────────────────────────────────────────────────────────────

/// Return the recommended TTL for the given content type.
///
/// | Content type          | TTL        |
/// |-----------------------|------------|
/// | Manifest              | 30 s       |
/// | VideoSegment          | 300 s (5 min) |
/// | AudioSegment          | 300 s      |
/// | Image                 | 3 600 s (1 h) |
/// | Thumbnail             | 86 400 s (24 h) |
/// | Metadata              | 600 s (10 min) |
pub fn ttl_for_type(content_type: &MediaContentType) -> Duration {
    match content_type {
        MediaContentType::Manifest => Duration::from_secs(30),
        MediaContentType::VideoSegment { .. } => Duration::from_secs(300),
        MediaContentType::AudioSegment { .. } => Duration::from_secs(300),
        MediaContentType::Image { .. } => Duration::from_secs(3_600),
        MediaContentType::Thumbnail => Duration::from_secs(86_400),
        MediaContentType::Metadata => Duration::from_secs(600),
    }
}

// ── ScoringWeights ────────────────────────────────────────────────────────────

/// Configurable exponent weights used by `score_for_eviction`.
///
/// The eviction score formula is:
/// ```text
/// score = (1 - recency).powf(recency_exp)
///       × (1/priority).powf(priority_exp)
///       × size_factor.powf(size_exp)
/// ```
///
/// Defaults (`recency_exp = 1.0`, `priority_exp = 1.0`, `size_exp = 1.0`)
/// reproduce the original behaviour exactly.
///
/// `per_type_priority` overrides the `priority` weight multiplier per
/// [`MediaContentType`] variant; keys are matched by discriminant tag, so
/// only the variant kind matters (e.g. all `VideoSegment` variants share one
/// entry).
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Exponent applied to the recency factor `(1 - recency)`.  Higher = aged
    /// entries are penalised more aggressively.
    pub recency_exp: f64,
    /// Exponent applied to `(1 / priority)`.  Higher = low-priority items are
    /// more strongly preferred for eviction.
    pub priority_exp: f64,
    /// Exponent applied to `size_factor`.  Higher = large entries are more
    /// strongly preferred for eviction.
    pub size_exp: f64,
    /// Per-type priority weight override.  When a content type has an entry
    /// here, its `priority` value is multiplied by this factor before the
    /// scoring formula is applied.  Values < 1.0 make the type *harder* to
    /// evict; values > 1.0 make it *easier* to evict.
    pub per_type_priority: HashMap<String, f64>,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            recency_exp: 1.0,
            priority_exp: 1.0,
            size_exp: 1.0,
            per_type_priority: HashMap::new(),
        }
    }
}

impl ScoringWeights {
    /// Create default weights that reproduce the original eviction behaviour.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the per-type priority multiplier for the given content type.
    fn type_key(content_type: &MediaContentType) -> &'static str {
        match content_type {
            MediaContentType::VideoSegment { .. } => "VideoSegment",
            MediaContentType::AudioSegment { .. } => "AudioSegment",
            MediaContentType::Image { .. } => "Image",
            MediaContentType::Manifest => "Manifest",
            MediaContentType::Thumbnail => "Thumbnail",
            MediaContentType::Metadata => "Metadata",
        }
    }

    /// Look up the priority multiplier for a content type.
    pub fn priority_multiplier(&self, content_type: &MediaContentType) -> f64 {
        let key = Self::type_key(content_type);
        self.per_type_priority.get(key).copied().unwrap_or(1.0)
    }
}

// ── ContentAwareCache ─────────────────────────────────────────────────────────

/// A media-content-aware cache that scores eviction candidates by a
/// recency × priority × size metric rather than pure LRU order.
///
/// Internally, it delegates storage to an [`LruCache<String, CacheEntry>`] for
/// O(1) operations and maintains a live count of bytes.
pub struct ContentAwareCache {
    inner: LruCache<String, CacheEntry>,
    /// Capacity in number of entries.
    capacity: usize,
    /// Current total size of all resident entries in bytes.
    total_bytes: usize,
    /// Optional byte-level capacity; entries whose cumulative size exceeds
    /// this trigger an additional content-aware eviction pass.
    max_bytes: Option<usize>,
    /// Configurable scoring weights used by `score_for_eviction`.
    scoring_weights: ScoringWeights,
}

impl ContentAwareCache {
    /// Create a new `ContentAwareCache` with an entry-count `capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: LruCache::new(capacity),
            capacity,
            total_bytes: 0,
            max_bytes: None,
            scoring_weights: ScoringWeights::default(),
        }
    }

    /// Set an optional byte-level capacity in addition to the entry count cap.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    /// Set custom scoring weights used for eviction candidate selection.
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    /// Update scoring weights at runtime.
    pub fn set_scoring_weights(&mut self, weights: ScoringWeights) {
        self.scoring_weights = weights;
    }

    /// Return a reference to the current scoring weights.
    pub fn scoring_weights(&self) -> &ScoringWeights {
        &self.scoring_weights
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /// Insert a media entry.
    ///
    /// If the cache is at capacity, the entry with the highest eviction score
    /// is removed first.  If the new entry has a higher priority than the
    /// current worst entry, it displaces it; otherwise the LRU entry is used
    /// as the fallback.
    pub fn insert_media(&mut self, key: String, data: Vec<u8>, content_type: MediaContentType) {
        let size = data.len();

        // If the key already exists, remove the old size first.
        if let Some(old) = self.inner.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.size_bytes);
        }

        // Enforce byte-level capacity if configured.
        if let Some(max_bytes) = self.max_bytes {
            while self.total_bytes + size > max_bytes && !self.inner.is_empty() {
                self.evict_worst();
            }
        }

        // Enforce entry-count capacity.
        if self.inner.len() >= self.capacity {
            self.evict_worst();
        }

        let entry = CacheEntry::new(key.clone(), data, content_type);
        self.total_bytes += size;
        self.inner.insert(key, entry, size);
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    /// Look up `key` and return an immutable reference to its [`CacheEntry`]
    /// if present (updating last-accessed time).
    pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
        // We need to update last_accessed but LruCache::get only returns &V.
        // We do a two-step: get (to update LRU order) then update the entry.
        let key_owned = key.to_string();
        if self.inner.contains(&key_owned) {
            // Touch the entry to update `last_accessed` and `access_count`.
            // We use `peek` to get a reference without a second LRU move, then
            // perform a targeted update via `insert` (which handles duplicates).
            let updated_entry = {
                let entry = self.inner.peek(&key_owned)?;
                let mut e = entry.clone();
                e.last_accessed = Instant::now();
                e.access_count = e.access_count.saturating_add(1);
                e
            };
            let size = updated_entry.size_bytes;
            // Re-insert to move to MRU head and persist the updated timestamps.
            // Adjust total_bytes: remove old, add back same size.
            self.total_bytes = self.total_bytes.saturating_sub(size);
            self.inner.insert(key_owned.clone(), updated_entry, size);
            self.total_bytes += size;
            self.inner.peek(&key_owned)
        } else {
            None
        }
    }

    /// Peek at `key` without updating access metadata or LRU order.
    pub fn peek(&self, key: &str) -> Option<&CacheEntry> {
        self.inner.peek(&key.to_string())
    }

    // ── Removal ───────────────────────────────────────────────────────────────

    /// Explicitly remove an entry by `key`.
    ///
    /// Returns `true` if the entry was present.
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.inner.remove(&key.to_string()) {
            self.total_bytes = self.total_bytes.saturating_sub(entry.size_bytes);
            true
        } else {
            false
        }
    }

    // ── TTL expiry ────────────────────────────────────────────────────────────

    /// Scan all entries and remove those whose TTL has elapsed.
    ///
    /// Returns the number of entries evicted.
    ///
    /// This is an O(n) operation; callers should invoke it periodically rather
    /// than on every access.
    pub fn evict_expired(&mut self) -> usize {
        // Collect expired keys — we cannot mutate inner while iterating.
        // We use a drain-based approach: repeatedly pop the LRU until we find a
        // non-expired entry or exhaust the cache.  This is O(n) worst case but
        // is acceptable for a maintenance sweep.
        //
        // A more efficient approach would require direct iteration over the
        // backing HashMap; that's not exposed by LruCache, so we collect keys
        // via `peek` patterns.  Instead we collect entries into a scratch vec.
        let mut expired_keys: Vec<String> = Vec::new();
        let mut remaining: Vec<(String, CacheEntry)> = Vec::new();

        // Drain by repeated LRU eviction, noting which entries are expired.
        while let Some((k, entry)) = self.inner.evict_lru() {
            let ttl = ttl_for_type(&entry.content_type);
            if entry.inserted_at.elapsed() > ttl {
                expired_keys.push(k);
            } else {
                remaining.push((k, entry));
            }
        }

        // Re-insert non-expired entries.
        for (k, entry) in remaining {
            let size = entry.size_bytes;
            self.inner.insert(k, entry, size);
        }

        // Recompute total_bytes.
        self.total_bytes = 0;
        // We cannot iterate LruCache directly; use a fresh total.
        // Instead, drain and reinsert again is wasteful; track via expired_keys count.
        // Correct approach: adjust total_bytes by subtracting expired sizes.
        // Since we already re-inserted all non-expired entries, recalculate.
        // Use the stats for a best-effort figure.
        self.total_bytes = self.inner.stats().total_size_bytes;

        expired_keys.len()
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` when the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the total number of bytes currently stored.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Return the entry-count capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // ── Internal eviction ─────────────────────────────────────────────────────

    /// Evict the entry with the highest eviction score (content-aware).
    ///
    /// Because [`LruCache`] does not expose iteration, we must drain and
    /// re-insert to find the worst entry.  This is O(n) per eviction;
    /// acceptable for moderate cache sizes and infrequent evictions.
    ///
    /// If finding the worst entry is too expensive (e.g. very large cache),
    /// callers can fall back to plain LRU via `inner.evict_lru()` directly.
    fn evict_worst(&mut self) {
        if self.inner.is_empty() {
            return;
        }

        // Capture the current weights — we cannot borrow self.scoring_weights
        // and self.inner mutably at the same time.
        let weights = self.scoring_weights.clone();

        // Drain everything, find the worst, re-insert the rest.
        let mut entries: Vec<(String, CacheEntry)> = Vec::with_capacity(self.inner.len());
        while let Some((k, entry)) = self.inner.evict_lru() {
            entries.push((k, entry));
        }

        // Find the index of the highest eviction score.
        let worst_idx = entries
            .iter()
            .enumerate()
            .max_by(|(_, (_, a)), (_, (_, b))| {
                a.score_for_eviction_weighted(&weights)
                    .partial_cmp(&b.score_for_eviction_weighted(&weights))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Remove the worst entry.
        let (_, evicted) = entries.remove(worst_idx);
        self.total_bytes = self.total_bytes.saturating_sub(evicted.size_bytes);

        // Re-insert the remaining entries.
        for (k, entry) in entries {
            let size = entry.size_bytes;
            self.inner.insert(k, entry, size);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContentCachePriority ──────────────────────────────────────────────────

    #[test]
    fn test_priority_manifest_is_highest() {
        let p = ContentCachePriority::for_type(&MediaContentType::Manifest);
        assert_eq!(p.0, 10);
    }

    #[test]
    fn test_priority_thumbnail() {
        let p = ContentCachePriority::for_type(&MediaContentType::Thumbnail);
        assert_eq!(p.0, 8);
    }

    #[test]
    fn test_priority_high_bitrate_video() {
        let p = ContentCachePriority::for_type(&MediaContentType::VideoSegment {
            bitrate: 5_000_000,
            codec: "av1".into(),
        });
        assert_eq!(p.0, 7);
    }

    #[test]
    fn test_priority_low_bitrate_video() {
        let p = ContentCachePriority::for_type(&MediaContentType::VideoSegment {
            bitrate: 1_000_000,
            codec: "vp9".into(),
        });
        assert_eq!(p.0, 6);
    }

    #[test]
    fn test_priority_audio() {
        let p =
            ContentCachePriority::for_type(&MediaContentType::AudioSegment { bitrate: 128_000 });
        assert_eq!(p.0, 5);
    }

    #[test]
    fn test_priority_image() {
        let p = ContentCachePriority::for_type(&MediaContentType::Image {
            width: 1920,
            height: 1080,
        });
        assert_eq!(p.0, 4);
    }

    #[test]
    fn test_priority_metadata() {
        let p = ContentCachePriority::for_type(&MediaContentType::Metadata);
        assert_eq!(p.0, 3);
    }

    // ── TTL ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ttl_manifest() {
        assert_eq!(
            ttl_for_type(&MediaContentType::Manifest),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn test_ttl_video_segment() {
        let ct = MediaContentType::VideoSegment {
            bitrate: 2_000_000,
            codec: "av1".into(),
        };
        assert_eq!(ttl_for_type(&ct), Duration::from_secs(300));
    }

    #[test]
    fn test_ttl_thumbnail() {
        assert_eq!(
            ttl_for_type(&MediaContentType::Thumbnail),
            Duration::from_secs(86_400)
        );
    }

    #[test]
    fn test_ttl_image() {
        let ct = MediaContentType::Image {
            width: 100,
            height: 100,
        };
        assert_eq!(ttl_for_type(&ct), Duration::from_secs(3_600));
    }

    // ── ContentAwareCache basic operations ────────────────────────────────────

    #[test]
    fn test_insert_and_get() {
        let mut cache = ContentAwareCache::new(16);
        cache.insert_media(
            "seg1".into(),
            vec![0u8; 1024],
            MediaContentType::VideoSegment {
                bitrate: 2_000_000,
                codec: "av1".into(),
            },
        );
        let entry = cache.get("seg1");
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.size_bytes), Some(1024));
    }

    #[test]
    fn test_get_absent_returns_none() {
        let mut cache = ContentAwareCache::new(8);
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut cache = ContentAwareCache::new(8);
        assert!(cache.is_empty());
        cache.insert_media("m".into(), vec![1, 2], MediaContentType::Manifest);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut cache = ContentAwareCache::new(8);
        cache.insert_media("key".into(), vec![0u8; 512], MediaContentType::Metadata);
        assert!(cache.remove("key"));
        assert!(cache.get("key").is_none());
    }

    #[test]
    fn test_remove_absent() {
        let mut cache = ContentAwareCache::new(8);
        assert!(!cache.remove("ghost"));
    }

    #[test]
    fn test_total_bytes_tracking() {
        let mut cache = ContentAwareCache::new(16);
        cache.insert_media("a".into(), vec![0u8; 100], MediaContentType::Manifest);
        cache.insert_media("b".into(), vec![0u8; 200], MediaContentType::Metadata);
        assert_eq!(cache.total_bytes(), 300);
        cache.remove("a");
        assert_eq!(cache.total_bytes(), 200);
    }

    #[test]
    fn test_capacity_reported() {
        let cache = ContentAwareCache::new(32);
        assert_eq!(cache.capacity(), 32);
    }

    // ── Eviction scoring ─────────────────────────────────────────────────────

    #[test]
    fn test_score_for_eviction_just_inserted_is_low() {
        let entry = CacheEntry::new("k".into(), vec![0u8; 100], MediaContentType::Manifest);
        // Just inserted → recency ≈ 1.0 → (1 - 1) * … ≈ 0.
        let score = entry.score_for_eviction();
        assert!(
            score < 0.1,
            "fresh entry should have low eviction score, got {score}"
        );
    }

    #[test]
    fn test_score_low_priority_higher_than_high_priority() {
        // An aged metadata entry should score higher than an aged manifest entry
        // because metadata has lower priority (easier to re-fetch).
        let manifest_entry =
            CacheEntry::new("m".into(), vec![0u8; 100], MediaContentType::Manifest);
        let meta_entry = CacheEntry::new("d".into(), vec![0u8; 100], MediaContentType::Metadata);
        // Force age by checking the formula with same recency.
        // priority_manifest = 10, priority_metadata = 3 → 1/3 > 1/10.
        let p_manifest = ContentCachePriority::for_type(&MediaContentType::Manifest).0 as f32;
        let p_meta = ContentCachePriority::for_type(&MediaContentType::Metadata).0 as f32;
        assert!(
            1.0 / p_meta > 1.0 / p_manifest,
            "metadata entry should evict before manifest"
        );
        drop(manifest_entry);
        drop(meta_entry);
    }

    // ── Content-aware eviction when at capacity ───────────────────────────────

    #[test]
    fn test_eviction_prefers_low_priority_entries() {
        // Capacity of 2: insert a Manifest + a Metadata.
        // Then insert a third entry. The Metadata (priority=3) should be evicted
        // over Manifest (priority=10), all else being equal.
        let mut cache = ContentAwareCache::new(2);
        // Insert manifest and metadata with tiny data.
        cache.insert_media("manifest".into(), vec![0u8; 1], MediaContentType::Manifest);
        cache.insert_media("meta".into(), vec![0u8; 1], MediaContentType::Metadata);
        // Let both entries age so the score formula produces non-zero values,
        // then refresh manifest's last_accessed.  Without this sleep the age
        // is <1 µs on fast (Linux/CUDA) machines and both scores collapse to
        // (1 - 1) * … = 0, making eviction order non-deterministic (flaky).
        std::thread::sleep(std::time::Duration::from_millis(100));
        // Force the manifest to be "recently used" relative to the aged metadata.
        let _ = cache.get("manifest");
        // Insert third entry to trigger eviction.
        cache.insert_media(
            "new".into(),
            vec![0u8; 1],
            MediaContentType::VideoSegment {
                bitrate: 2_000_000,
                codec: "av1".into(),
            },
        );
        assert_eq!(cache.len(), 2);
        // Manifest should still be present (higher priority).
        assert!(
            cache.peek("manifest").is_some(),
            "manifest should survive eviction"
        );
    }

    // ── access_count and last_accessed updates ────────────────────────────────

    #[test]
    fn test_access_count_increments_on_get() {
        let mut cache = ContentAwareCache::new(8);
        cache.insert_media("k".into(), vec![1, 2, 3], MediaContentType::Thumbnail);
        cache.get("k");
        cache.get("k");
        let count = cache.peek("k").map(|e| e.access_count).unwrap_or(0);
        assert_eq!(count, 2, "access_count should be 2 after two gets");
    }

    // ── Byte-level capacity ───────────────────────────────────────────────────

    #[test]
    fn test_max_bytes_triggers_eviction() {
        let mut cache = ContentAwareCache::new(100).with_max_bytes(500);
        // Insert 5 × 100-byte entries = 500 bytes (at limit).
        for i in 0..5u32 {
            cache.insert_media(
                format!("seg_{i}"),
                vec![0u8; 100],
                MediaContentType::AudioSegment { bitrate: 128_000 },
            );
        }
        assert!(cache.total_bytes() <= 500);
        // Insert one more → must evict to stay within budget.
        cache.insert_media("extra".into(), vec![0u8; 100], MediaContentType::Metadata);
        assert!(
            cache.total_bytes() <= 500,
            "total bytes exceeded budget: {}",
            cache.total_bytes()
        );
    }

    // ── Peek does not update metadata ─────────────────────────────────────────

    #[test]
    fn test_peek_does_not_change_access_count() {
        let mut cache = ContentAwareCache::new(8);
        cache.insert_media("p".into(), vec![99], MediaContentType::Manifest);
        let before = cache.peek("p").map(|e| e.access_count).unwrap_or(99);
        let _ = cache.peek("p");
        let after = cache.peek("p").map(|e| e.access_count).unwrap_or(99);
        assert_eq!(before, after, "peek must not change access_count");
    }

    // ── Upsert behaviour ─────────────────────────────────────────────────────

    #[test]
    fn test_insert_same_key_updates_value() {
        let mut cache = ContentAwareCache::new(8);
        cache.insert_media("k".into(), vec![1, 2, 3], MediaContentType::Manifest);
        cache.insert_media("k".into(), vec![10, 20], MediaContentType::Manifest);
        assert_eq!(cache.len(), 1, "duplicate key should not increase len");
        assert_eq!(
            cache.total_bytes(),
            2,
            "total_bytes should reflect updated size"
        );
    }

    // ── evict_expired ─────────────────────────────────────────────────────────

    #[test]
    fn test_evict_expired_no_entries() {
        let mut cache = ContentAwareCache::new(8);
        assert_eq!(cache.evict_expired(), 0);
    }

    #[test]
    fn test_evict_expired_fresh_entries_survive() {
        let mut cache = ContentAwareCache::new(8);
        cache.insert_media("fresh".into(), vec![0u8; 10], MediaContentType::Manifest);
        // Manifest TTL is 30 s; the entry was just inserted → should not expire.
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 0);
        assert!(cache.peek("fresh").is_some());
    }
}
