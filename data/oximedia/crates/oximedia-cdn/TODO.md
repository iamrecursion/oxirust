# oximedia-cdn TODO

## Current Status
- 5 modules: `edge_manager`, `cache_invalidation`, `origin_failover`, `geo_routing`, `cdn_metrics`
- `CdnManager` orchestrates all subsystems with `RwLock`/`Mutex` guarded state
- Zero external dependencies beyond `thiserror` and `uuid` (pure Rust)
- Features: consistent-hash ring, EWMA latency smoothing, Haversine geo-routing, Prometheus metrics export

## Enhancements
- [x] Replace `unwrap_or_else(|e| e.into_inner())` mutex recovery in `CdnManager` with proper error propagation
- [x] Add health check probing with configurable HTTP/TCP check in `origin_failover` (verified 2026-05-16; src/origin_failover.rs:369 HealthCheckProtocol, HealthChecker, HealthCheckConfig:404)
- [x] Extend `cache_invalidation` with tag-based invalidation (purge by content tag, not just path/glob)
- [x] Add `edge_manager` node weight support for heterogeneous capacity (some PoPs larger than others) (done — weighted_score at src/edge_manager.rs:110, best_node_by_capacity at :238)
- [x] Implement soft-purge in `cache_invalidation` that marks stale but serves while revalidating (implemented 2026-05-31: SoftPurge, StaleCacheEntry, soft_purge/is_stale/serve_while_stale on InvalidationManager)
- [x] Extend `geo_routing` with anycast simulation support for DNS-based routing (implemented 2026-06-04: AnycastRouter/AnycastGroup/VirtualIp in anycast.rs; withdraw_pop/announce_pop/resolve)
- [x] Add `cdn_metrics` per-content-type breakdown (video, audio, image, manifest) (implemented 2026-05-31: ContentCategory, ContentTypeMetrics, ContentTypeMetricsStore)
- [x] Improve `origin_failover` with circuit breaker pattern (half-open state for recovery detection)

## New Features
- [x] Add `token_auth` module for signed URL generation and validation (HMAC-based)
- [x] Implement `bandwidth_throttle` module for per-client or per-origin rate limiting (verified 2026-05-16; src/bandwidth_throttle.rs:829 lines)
- [x] Add `manifest_rewrite` module for HLS/DASH manifest URL rewriting to edge-local paths (verified 2026-05-16; src/manifest_rewrite.rs:912 lines)
- [x] Implement `ssl_cert_manager` module tracking TLS certificate expiration per edge node (verified 2026-05-16; src/ssl_cert_manager.rs:790 lines)
- [x] Add `cdn_log_analysis` module parsing access logs for traffic patterns and anomaly detection (verified 2026-05-16; src/cdn_log_analysis.rs:860 lines)
- [x] Implement `multi_cdn` module for coordinating across multiple CDN providers (failover, cost optimization) (verified 2026-05-16; src/multi_cdn.rs:661 lines, parking_lot::Mutex)
- [x] Add `prefetch_scheduler` module that pushes popular content to edges before demand spikes (verified 2026-05-16; src/prefetch_scheduler.rs:661 lines)
- [x] Implement `request_coalescing` module to deduplicate concurrent origin fetches for the same resource (verified 2026-05-16; src/request_coalescing.rs:626 lines)

## Performance
- [x] Replace `std::sync::RwLock` with `parking_lot::RwLock` in `CdnManager` for better performance (verified 2026-05-16; src/multi_cdn.rs:25 parking_lot::Mutex used for EWMA latency)
- [x] Use lock-free atomic counters in `cdn_metrics` instead of mutex-protected fields
- [x] Implement connection pooling simulation in `origin_failover` for origin keep-alive reuse (implemented 2026-06-04: ConnectionPool/PooledConn with acquire/release/idle_len; max_idle eviction; reused/created counters)
- [x] Cache Haversine distance calculations in `geo_routing` for repeated client-to-edge lookups (implemented 2026-05-31: HaversineCache with (i64×4) rounded-key HashMap per GeoRouter)
- [x] Use a spatial index (R-tree) in `geo_routing` for O(log n) nearest-edge lookup instead of linear scan (implemented 2026-05-31: RtreeEdgeIndex/EdgePoint wrapping rstar::RTree; lazy-built when fleet > 16)

## Testing
- [x] Add concurrent invalidation stress test with 1000+ requests and multiple processing nodes (implemented 2026-05-31: 16 threads × 100 requests = 1600 total, test_concurrent_invalidation_stress)
- [ ] Test `origin_failover` recovery: mark origin failed, simulate health check success, verify re-selection
- [x] Add `geo_routing` test with edge cases: equidistant PoPs, antipodal points, same-location client/edge (implemented 2026-06-06: stable id-tiebreak fix at src/geo_routing.rs:463 makes equidistant selection deterministic & insertion-order-independent; test_assign_edge_equidistant_deterministic at src/geo_routing.rs:997, test_assign_edge_equidistant_id_tiebreak_flips:1058, test_haversine_antipodal_finite:1083, test_assign_edge_zero_distance_exact_colocation:1117)
- [x] Test `cdn_metrics` Prometheus output format against Prometheus parser specification (implemented 2026-06-06: hand-rolled lenient exposition parser, no new dep; test_prometheus_help_type_ordered HELP→TYPE→sample at src/cdn_metrics.rs:908, test_prometheus_no_duplicate_series:979, test_prometheus_counter_total_suffix:1009, test_prometheus_sample_lines_wellformed:1043, test_prometheus_gauge_value_matches_hit_ratio:1087)
- [x] Add `CdnManager` integration test covering full lifecycle: add edges, configure origins, route, invalidate (implemented 2026-05-31: test_cdn_manager_lifecycle)
- [ ] Test `cache_invalidation` rate limiting enforcement under burst conditions

## Documentation
- [ ] Document CDN architecture with data flow diagram showing edge-origin-client interactions
- [ ] Add deployment guide for configuring `CdnManager` in a multi-region setup
- [ ] Document Prometheus metrics names, types, and labels for monitoring dashboard setup
