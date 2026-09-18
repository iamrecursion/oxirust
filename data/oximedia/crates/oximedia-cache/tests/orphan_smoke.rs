//! Smoke tests for newly-wired orphan modules in oximedia-cache.

// ─── adaptive ────────────────────────────────────────────────────────────────
#[test]
fn test_adaptive_policy_hit_rate_adjustment() {
    use oximedia_cache::adaptive::{AdaptiveConfig, AdaptivePolicy};

    let cfg = AdaptiveConfig {
        target_hit_rate: 0.80,
        tolerance: 0.05,
        adjustment_interval: 5,
        min_capacity: 8,
        max_capacity: 512,
        growth_factor: 1.5,
        shrink_factor: 0.75,
        ttl_extension: std::time::Duration::from_secs(10),
        ttl_reduction: std::time::Duration::from_secs(5),
        window_size: 20,
    };
    let mut policy = AdaptivePolicy::new(cfg).expect("valid config");
    // Record all hits — hit rate = 1.0 > target+tolerance → should shrink
    for _ in 0..5 {
        let _ = policy.record_hit();
    }
    // After threshold hits, rolling hit rate should be 1.0
    assert!(
        (policy.rolling_hit_rate() - 1.0).abs() < 1e-5,
        "all hits → 100% hit rate"
    );
}

// ─── admission_filter ────────────────────────────────────────────────────────
#[test]
fn test_admission_filter_threshold() {
    use oximedia_cache::admission_filter::{AdmissionConfig, AdmissionFilter};

    let cfg = AdmissionConfig {
        admission_threshold: 3.0,
        target_hit_rate: 0.80,
        decay_factor: 0.90,
        adjust_interval: 100,
        threshold_step: 0.5,
        min_threshold: 1.0,
        max_threshold: 20.0,
    };
    let mut filter = AdmissionFilter::new(cfg).expect("valid config");

    // Record 5 accesses to "key-a" — frequency should exceed threshold of 3.0
    for _ in 0..5 {
        filter.record_access("key-a");
    }
    assert!(
        filter.should_admit("key-a"),
        "key-a should be admitted after 5 accesses"
    );
    assert!(
        !filter.should_admit("cold-key"),
        "cold key should not be admitted"
    );
}

// ─── eviction ────────────────────────────────────────────────────────────────
#[test]
fn test_eviction_lru_policy() {
    use oximedia_cache::eviction::{create_eviction_fn, EvictionStrategy};

    let evict = create_eviction_fn(EvictionStrategy::Lru);
    let entries: Vec<(String, u64)> = vec![
        ("frame-001".to_string(), 1000),
        ("frame-002".to_string(), 500),
        ("frame-003".to_string(), 2000),
    ];
    let to_evict = evict(&entries);
    // LRU evicts the entry with smallest access_time (oldest)
    assert_eq!(to_evict, Some("frame-002".to_string()));
}

#[test]
fn test_eviction_fifo_policy() {
    use oximedia_cache::eviction::{create_eviction_fn, EvictionStrategy};

    let evict = create_eviction_fn(EvictionStrategy::Fifo);
    let entries: Vec<(String, u64)> =
        vec![("first".to_string(), 1000), ("second".to_string(), 2000)];
    let to_evict = evict(&entries);
    // FIFO evicts index 0 (first in the slice)
    assert_eq!(to_evict, Some("first".to_string()));
}

// ─── key_norm ────────────────────────────────────────────────────────────────
#[test]
fn test_key_norm_lowercase_and_sort_params() {
    use oximedia_cache::key_norm::normalize_cache_key;

    let key = normalize_cache_key("https://CDN.Example.com/Video/Clip.mp4/?b=2&a=1#frag");
    assert_eq!(key, "https://cdn.example.com/video/clip.mp4?a=1&b=2");
}

// ─── negative ────────────────────────────────────────────────────────────────
#[test]
fn test_negative_cache_ttl() {
    use oximedia_cache::negative::NegativeCache;

    let mut nc = NegativeCache::new(5_000); // 5-second TTL
    nc.insert_miss("missing-asset", 1_000_000);

    assert!(
        nc.is_known_miss("missing-asset", 1_002_000),
        "should be a known miss within TTL"
    );
    assert!(
        !nc.is_known_miss("missing-asset", 1_006_000),
        "should expire after TTL"
    );
}

// ─── segment_cache ───────────────────────────────────────────────────────────
#[test]
fn test_segment_cache_insert_and_get() {
    use oximedia_cache::segment_cache::{
        MediaSegment, SegmentCache, SegmentCacheConfig, SegmentRef,
    };

    let config = SegmentCacheConfig {
        max_segments: 32,
        max_bytes: 1024 * 1024,
        prefetch_ahead: 3,
        evict_played: true,
    };
    let mut cache = SegmentCache::new(config);

    let segment = MediaSegment {
        segment_id: "stream-01-0".to_string(),
        stream_id: "stream-01".to_string(),
        sequence: 0,
        duration_secs: 6.0,
        data: vec![0xAB; 512],
        content_type: "video/mp2t".to_string(),
    };
    let seg_ref = SegmentRef {
        stream_id: "stream-01".to_string(),
        sequence: 0,
    };
    cache.insert(segment).expect("insert should succeed");

    let retrieved = cache.get(&seg_ref);
    assert!(
        retrieved.is_some(),
        "segment should be retrievable after insert"
    );
    assert_eq!(retrieved.unwrap().data.len(), 512);
}

// ─── stats ───────────────────────────────────────────────────────────────────
#[test]
fn test_cache_stats_hit_rate() {
    use oximedia_cache::stats::CacheStats;

    let mut stats = CacheStats::new();
    stats.record_hit();
    stats.record_hit();
    stats.record_miss();
    assert!(
        (stats.hit_rate() - 2.0 / 3.0).abs() < 1e-6,
        "hit rate should be 2/3"
    );

    let json = stats.to_json();
    assert!(json.contains("\"hits\":2"), "JSON should contain hits:2");
    assert!(
        json.contains("\"misses\":1"),
        "JSON should contain misses:1"
    );
}

// ─── tier_compressor ─────────────────────────────────────────────────────────
#[test]
fn test_tier_compressor_roundtrip() {
    use oximedia_cache::tier_compressor::TierCompressor;

    let compressor = TierCompressor::new(1);
    let data: Vec<u8> = b"Hello, OxiMedia cache compress test! AAAAAAAAAAAAAAAAAAA".to_vec();
    let compressed = compressor.compress(&data).expect("compress should succeed");
    let decompressed = compressor
        .decompress(&compressed)
        .expect("decompress should succeed");
    assert_eq!(decompressed, data, "roundtrip should be lossless");
}

// ─── ttl_cache ───────────────────────────────────────────────────────────────
#[test]
fn test_ttl_cache_expiry() {
    use oximedia_cache::ttl_cache::TtlCache;

    let mut cache: TtlCache<&str, Vec<u8>> = TtlCache::new(64, 30);
    cache.insert("frame-001", vec![0u8; 128], 1_000);
    assert!(
        cache.get(&"frame-001", 1_020).is_some(),
        "within TTL should return value"
    );
    assert!(
        cache.get(&"frame-001", 1_031).is_none(),
        "past TTL should return None"
    );
}

// ─── weighted_cache ──────────────────────────────────────────────────────────
#[test]
fn test_weighted_cache_insert_and_eviction() {
    use oximedia_cache::weighted_cache::{CacheMediaType, WeightConfig, WeightedCache};

    let weights = WeightConfig::default();
    let mut cache = WeightedCache::new(3, weights);

    cache.insert(
        "a".to_string(),
        vec![1u8; 100],
        CacheMediaType::VideoSegment,
        1_u8,
    );
    cache.insert(
        "b".to_string(),
        vec![2u8; 100],
        CacheMediaType::AudioSegment,
        2_u8,
    );
    cache.insert(
        "c".to_string(),
        vec![3u8; 100],
        CacheMediaType::Metadata,
        3_u8,
    );
    // Insert 4th entry — should evict one
    cache.insert("d".to_string(), vec![4u8; 100], CacheMediaType::Image, 1_u8);

    assert!(cache.len() <= 3, "cache should not exceed capacity");
}

// ─── write_through ───────────────────────────────────────────────────────────
#[test]
fn test_write_through_cache_persist() {
    use oximedia_cache::write_through::{InMemoryStore, WriteThroughCache};

    let store: InMemoryStore<String, Vec<u8>> = InMemoryStore::new();
    let mut cache = WriteThroughCache::new(32, store);

    cache
        .put("key".to_string(), vec![1, 2, 3])
        .expect("put should succeed");
    let val = cache.get(&"key".to_string());
    assert!(val.is_some(), "value should be retrievable after put");
    assert_eq!(val.unwrap(), &[1u8, 2, 3]);
}
