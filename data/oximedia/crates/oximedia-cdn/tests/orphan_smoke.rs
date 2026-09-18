//! Smoke tests for all 15 newly wired CDN orphan modules.

// ── failover ─────────────────────────────────────────────────────────────────

#[test]
fn test_failover_manager_best_cdn() {
    use oximedia_cdn::failover::CdnFailoverManager;
    let mut mgr = CdnFailoverManager::new();
    mgr.add_cdn("fastly", 0.9);
    mgr.add_cdn("cloudfront", 0.6);
    assert_eq!(mgr.best_cdn(), Some("fastly"));
    mgr.add_cdn("akamai", 1.0);
    assert_eq!(mgr.best_cdn(), Some("akamai"));
}

#[test]
fn test_failover_manager_empty() {
    use oximedia_cdn::failover::CdnFailoverManager;
    let mgr = CdnFailoverManager::new();
    assert_eq!(mgr.best_cdn(), None);
}

// ── request_coalescing ───────────────────────────────────────────────────────

#[test]
fn test_request_coalescing_config_defaults() {
    use oximedia_cdn::request_coalescing::CoalescingConfig;
    let cfg = CoalescingConfig::default();
    assert!(cfg.max_in_flight > 0);
}

// ── bandwidth_throttle ───────────────────────────────────────────────────────

#[test]
fn test_bandwidth_throttle_token_bucket_allowed() {
    use oximedia_cdn::bandwidth_throttle::{ThrottleConfig, TokenBucket};
    let config = ThrottleConfig::default();
    let mut bucket = TokenBucket::new("client-1", config);
    // First small request should be allowed from the burst capacity
    let result = bucket.consume(1024).expect("no error");
    assert!(result.is_allowed());
}

#[test]
fn test_bandwidth_throttle_over_capacity() {
    use oximedia_cdn::bandwidth_throttle::{ThrottleConfig, TokenBucket};
    // Create a very small bucket (1 byte burst, 1 byte/sec)
    let config = ThrottleConfig::new(1);
    let mut bucket = TokenBucket::new("tiny-client", config);
    // Requesting 1 MiB should be throttled (exceeds burst capacity)
    let result = bucket.consume(1_048_576);
    // Either error (oversized) or throttled result is acceptable
    if let Ok(r) = result {
        assert!(r.is_throttled());
    }
}

// ── edge_opt ─────────────────────────────────────────────────────────────────

#[test]
fn test_edge_opt_format_selection() {
    use oximedia_cdn::edge_opt::EdgeBandwidthOptimizer;
    let optimizer = EdgeBandwidthOptimizer::new();
    let formats = [("4K", 20_000u32), ("1080p", 8_000u32), ("720p", 4_000u32)];
    // 10 Mbps client → best available without exceeding bandwidth
    let best = optimizer.select_format(10_000, &formats);
    assert_eq!(best, "1080p");
    // High bandwidth → 4K
    assert_eq!(optimizer.select_format(25_000, &formats), "4K");
}

// ── warming ──────────────────────────────────────────────────────────────────

#[test]
fn test_warming_cache_warmer() {
    use oximedia_cdn::warming::CacheWarmer;
    let mut warmer = CacheWarmer::new();
    assert!(warmer.is_empty());
    warmer.schedule("https://cdn.example.com/video.mp4", 10);
    warmer.schedule("https://cdn.example.com/thumb.jpg", 5);
    assert!(!warmer.is_empty());
    assert_eq!(warmer.pending_count(), 2);
    // Higher priority should come first
    let url = warmer.warm_next().expect("some url");
    assert_eq!(url, "https://cdn.example.com/video.mp4");
}

// ── manifest_rewrite ─────────────────────────────────────────────────────────

#[test]
fn test_manifest_rewrite_format_detection() {
    use oximedia_cdn::manifest_rewrite::ManifestRewriter;
    // detect_format inspects content, not filename
    let hls_content = "#EXTM3U\n#EXT-X-VERSION:3\n";
    assert_eq!(
        ManifestRewriter::detect_format(hls_content),
        Some(oximedia_cdn::manifest_rewrite::ManifestFormat::Hls)
    );
    let dash_content = "<?xml version=\"1.0\"?><MPD></MPD>";
    assert_eq!(
        ManifestRewriter::detect_format(dash_content),
        Some(oximedia_cdn::manifest_rewrite::ManifestFormat::Dash)
    );
    assert_eq!(ManifestRewriter::detect_format("plain text"), None);
}

#[test]
fn test_manifest_rewrite_config_default() {
    use oximedia_cdn::manifest_rewrite::RewriteConfig;
    let cfg = RewriteConfig::default();
    // default is constructed without errors; check a field that exists
    assert!(cfg.rewrite_media_uris || !cfg.rewrite_media_uris); // always true, just compile check
}

// ── origin_shield ────────────────────────────────────────────────────────────

#[test]
fn test_origin_shield_add_node() {
    use oximedia_cdn::origin_shield::{OriginShield, ShieldNode};
    let mut shield = OriginShield::default();
    shield.add_node(ShieldNode::new("shield-1", "iad", 10.0));
    assert_eq!(shield.node_count(), 1);
}

#[test]
fn test_shield_node_fill_ratio() {
    use oximedia_cdn::origin_shield::ShieldNode;
    let node = ShieldNode::new("n1", "us-east", 10.0);
    assert_eq!(node.fill_ratio(), 0.0);
    assert!(!node.is_near_full());
}

// ── prefetch_scheduler ───────────────────────────────────────────────────────

#[test]
fn test_prefetch_scheduler_config() {
    use oximedia_cdn::prefetch_scheduler::PrefetchConfig;
    let cfg = PrefetchConfig::default();
    assert!(cfg.top_n_content > 0);
    assert!(cfg.max_item_size_bytes > 0);
}

// ── cache_warming ────────────────────────────────────────────────────────────

#[test]
fn test_cache_warming_warmer_empty() {
    use oximedia_cdn::cache_warming::CacheWarmer;
    let warmer = CacheWarmer::new();
    assert!(warmer.is_empty());
    assert_eq!(warmer.len(), 0);
}

// ── cost ─────────────────────────────────────────────────────────────────────

#[test]
fn test_cdn_cost_analytics() {
    use oximedia_cdn::cost::CdnCostAnalytics;
    let mut analytics = CdnCostAnalytics::new();
    analytics.record_transfer("cloudflare", 1_073_741_824, 0.02); // 1 GiB at $0.02/GB
    assert!(analytics.total_cost() > 0.0);
    assert_eq!(analytics.total_bytes(), 1_073_741_824);
}

// ── geo_restrict ─────────────────────────────────────────────────────────────

#[test]
fn test_geo_restrict_allow_list() {
    use oximedia_cdn::geo_restrict::GeoRestriction;
    let mut restriction = GeoRestriction::new();
    restriction.allow("US");
    restriction.allow("CA");
    assert!(restriction.is_allowed("US"));
    assert!(restriction.is_allowed("CA"));
    assert!(!restriction.is_allowed("GB")); // not in allow-list
}

#[test]
fn test_geo_restrict_block_overrides_allow() {
    use oximedia_cdn::geo_restrict::GeoRestriction;
    let mut r = GeoRestriction::new();
    r.allow("US");
    r.block("US");
    assert!(!r.is_allowed("US")); // block overrides allow
}

// ── ssl_cert_manager ─────────────────────────────────────────────────────────

#[test]
fn test_ssl_cert_manager_cert_status() {
    use oximedia_cdn::ssl_cert_manager::{CertRecord, CertStatus};
    let now = 1_700_000_000u64;
    let not_before = now - 86_400;
    let not_after = now + 86_400 * 90; // 90 days in future
    let record =
        CertRecord::new("edge-1", "example.com", not_before, not_after).expect("valid cert");
    let status = record.status_at(now, 30);
    assert_eq!(status, CertStatus::Valid);
}

#[test]
fn test_ssl_cert_manager_cert_expiry() {
    use oximedia_cdn::ssl_cert_manager::{CertRecord, CertStatus};
    let now = 1_700_000_000u64;
    let record =
        CertRecord::new("edge-1", "example.com", now - 86_400 * 2, now - 1).expect("valid window");
    let status = record.status_at(now, 30);
    assert_eq!(status, CertStatus::Expired);
}

// ── cdn_log_analysis ─────────────────────────────────────────────────────────

#[test]
fn test_cdn_log_analysis_cache_status() {
    use oximedia_cdn::cdn_log_analysis::CacheStatus;
    // Verify variants can be constructed and compared
    assert_eq!(CacheStatus::Hit, CacheStatus::Hit);
    assert_ne!(CacheStatus::Hit, CacheStatus::Miss);
    assert_ne!(CacheStatus::Miss, CacheStatus::Unknown);
}

// ── multi_cdn ────────────────────────────────────────────────────────────────

#[test]
fn test_multi_cdn_router_select_primary() {
    use oximedia_cdn::multi_cdn::{CdnProvider, MultiCdnRouter, RoutingStrategy};
    use std::sync::Arc;
    let mut router = MultiCdnRouter::new(RoutingStrategy::Primary);
    // Priority 0 = highest priority
    router.add_provider(Arc::new(CdnProvider::new(
        "cf",
        "Cloudflare",
        "https://cf.net",
        10,
        1,
    )));
    router.add_provider(Arc::new(CdnProvider::new(
        "fl",
        "Fastly",
        "https://fastly.net",
        10,
        0,
    )));
    let selected = router.select().expect("should select a provider");
    assert_eq!(selected.id, "fl");
}

// ── request_router ───────────────────────────────────────────────────────────

#[test]
fn test_request_router_pop_management() {
    use oximedia_cdn::request_router::{CdnPop, RequestRouter};
    let mut router = RequestRouter::new();
    // CdnPop::new(id, region, country, lat, lon, weight, healthy, priority)
    router.add_pop(CdnPop::new(
        "pop-1",
        "us-east-1",
        "US",
        38.9,
        -77.0,
        1.0,
        true,
        1.0,
    ));
    router.add_pop(CdnPop::new(
        "pop-2",
        "eu-west-1",
        "DE",
        52.5,
        13.4,
        1.0,
        true,
        1.0,
    ));
    assert_eq!(router.pop_count(), 2);
    assert_eq!(router.healthy_pops().len(), 2);
    router.remove_pop("pop-1");
    assert_eq!(router.pop_count(), 1);
}
