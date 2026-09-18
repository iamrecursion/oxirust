//! OxiMedia CDN — edge management, cache invalidation, geographic routing,
//! origin failover, and metrics collection.
//!
//! `oximedia-cdn` provides five complementary subsystems for building
//! CDN-aware multimedia delivery pipelines:
//!
//! - [`edge_manager`] — track and score CDN edge PoPs, perform health updates,
//!   and select the optimal node for a request.
//! - [`cache_invalidation`] — queue and simulate cache-purge operations with
//!   priority scheduling, per-node rate limiting, and flexible glob scopes.
//! - [`origin_failover`] — manage upstream origin servers with automatic
//!   health tracking, EWMA latency smoothing, and multiple load-balancing
//!   strategies.
//! - [`geo_routing`] — route requests to the closest edge PoP using Haversine
//!   distance and model propagation latency.
//! - [`cdn_metrics`] — collect lock-free CDN counters and export them in
//!   Prometheus text format.
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_cdn::{CdnConfig, CdnManager};
//! use oximedia_cdn::edge_manager::{EdgeManager, EdgeNode};
//! use oximedia_cdn::cache_invalidation::{InvalidationQueue, InvalidationRequest, InvalidationScope};
//! use oximedia_cdn::origin_failover::{OriginPool, OriginServer, OriginStrategy};
//! use oximedia_cdn::geo_routing::{GeoRouter, GeoLocation};
//! use oximedia_cdn::cdn_metrics::MetricsRegistry;
//!
//! let config = CdnConfig::default();
//! let manager = CdnManager::new(config);
//!
//! // Add an edge node
//! if let Ok(mut edge) = manager.edge.write() {
//!     edge.add_node(EdgeNode::new("pop-iad", "cloudflare", "us-east-1", "iad.cdn.example.com"));
//! }
//!
//! // Record a cache hit
//! manager.metrics.record_hit("pop-iad", 1024);
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

/// Anycast / DNS-based virtual-IP routing simulation (multi-PoP VIP groups).
pub mod anycast;
/// Per-client and per-origin bandwidth throttle with token-bucket shaping.
pub mod bandwidth_throttle;
pub mod cache_invalidation;
/// Cache warming pipeline: crawl URL lists and populate edge nodes.
pub mod cache_warming;
/// CDN access-log ingestion, parsing, and aggregated analytics.
pub mod cdn_log_analysis;
pub mod cdn_metrics;
/// CDN cost model: per-request/per-byte billing estimation by provider.
pub mod cost;
pub mod edge_manager;
/// Edge-node optimisation: content replication scoring and eviction hints.
pub mod edge_opt;
/// Origin-level failover state machine with health-based promotion/demotion.
pub mod failover;
/// Geographic access restrictions: deny/allow rules per country/region.
pub mod geo_restrict;
pub mod geo_routing;
/// HLS/DASH manifest rewriting for CDN URL signing and domain substitution.
pub mod manifest_rewrite;
/// Multi-CDN orchestration: provider registry, traffic splitting, failover.
pub mod multi_cdn;
pub mod origin_failover;
/// Origin shield: designate a single PoP as the sole origin-facing proxy.
pub mod origin_shield;
/// Prefetch scheduler: time-based and popularity-driven prefetch queues.
pub mod prefetch_scheduler;
/// Request coalescing: collapse identical in-flight origin fetches into one.
pub mod request_coalescing;
/// Request routing engine: rule-based dispatch to edge nodes or origins.
pub mod request_router;
/// TLS/SSL certificate lifecycle management: provisioning, rotation, expiry.
pub mod ssl_cert_manager;
pub mod token_auth;
/// Proactive cache warming: schedule pre-fetches before anticipated demand.
pub mod warming;

// ─── Convenience re-exports ───────────────────────────────────────────────────

pub use cache_invalidation::{
    CacheEntry, CacheState, InvalidationError, InvalidationManager, InvalidationPriority,
    InvalidationQueue, InvalidationRequest, InvalidationResult, InvalidationScope,
    ManagedInvalidationRequest, ManagedInvalidationResult, SoftPurge, SoftPurgePolicy,
    StaleCacheEntry, TagIndex, TagInvalidationStore,
};
pub use cdn_metrics::{
    CdnMetrics, ContentCategory, ContentTypeMetrics, ContentTypeMetricsStore, EdgeMetrics,
    EdgeSnapshot, MetricSnapshot, MetricsRegistry,
};
pub use edge_manager::{EdgeFeature, EdgeManager, EdgeNode};
pub use geo_routing::{
    haversine_km, latency_from_km, EdgeNodeGeo, EdgeNodeId, EdgePoint, GeoLocation, GeoRouter,
    HaversineCache, Region, RtreeEdgeIndex,
};
pub use origin_failover::{
    CircuitBreaker, CircuitBreakerState, HealthCheckConfig, HealthCheckProbe, HealthCheckProtocol,
    HealthChecker, OriginError, OriginPool, OriginServer, OriginStrategy,
};
pub use token_auth::{
    SignedUrlClaims, SigningKey, TokenAuth, TokenConfig, TokenError, TokenSigner, TokenValidator,
};

// ─── CdnError ─────────────────────────────────────────────────────────────────

/// Errors that can arise from CDN manager operations.
#[derive(Debug, thiserror::Error)]
pub enum CdnError {
    /// A mutex or RwLock was poisoned.
    #[error("internal lock poisoned: {0}")]
    LockPoisoned(String),
    /// An invalidation subsystem error.
    #[error("invalidation error: {0}")]
    Invalidation(#[from] InvalidationError),
}

// ─── CdnConfig ───────────────────────────────────────────────────────────────

/// Top-level CDN configuration, aggregating settings for all subsystems.
#[derive(Debug, Clone)]
pub struct CdnConfig {
    /// Default origin selection strategy.
    pub origin_strategy: OriginStrategy,
    /// Maximum invalidations per node per minute (rate limiting).
    pub invalidation_rate_per_min: usize,
    /// Maximum pending invalidation requests in the queue.
    pub invalidation_queue_capacity: usize,
    /// Health-check interval in seconds.
    pub health_check_interval_secs: u64,
}

impl Default for CdnConfig {
    fn default() -> Self {
        Self {
            origin_strategy: OriginStrategy::Priority,
            invalidation_rate_per_min: 100,
            invalidation_queue_capacity: 10_000,
            health_check_interval_secs: 30,
        }
    }
}

impl CdnConfig {
    /// Create a new config with the given origin strategy.
    pub fn with_strategy(strategy: OriginStrategy) -> Self {
        Self {
            origin_strategy: strategy,
            ..Self::default()
        }
    }
}

// ─── CdnManager ──────────────────────────────────────────────────────────────

/// Orchestrates all CDN subsystems under a single handle.
///
/// Each subsystem is exposed as a public field so callers can use the
/// fine-grained APIs directly.  Shared mutable state is guarded by
/// `std::sync::RwLock` where needed.
pub struct CdnManager {
    /// Configuration snapshot used to create this manager.
    pub config: CdnConfig,
    /// Edge-node registry and selector.
    pub edge: std::sync::RwLock<EdgeManager>,
    /// Cache invalidation queue.
    pub invalidation: std::sync::Mutex<InvalidationQueue>,
    /// Origin server pool.
    pub origins: std::sync::Mutex<OriginPool>,
    /// Geographic router.
    pub geo: std::sync::RwLock<GeoRouter>,
    /// Metrics registry (global + per-edge).
    pub metrics: MetricsRegistry,
}

impl CdnManager {
    /// Create a new manager from `config`.
    pub fn new(config: CdnConfig) -> Self {
        let strategy = config.origin_strategy.clone();
        let cap = config.invalidation_queue_capacity;
        let rate = config.invalidation_rate_per_min;
        Self {
            config,
            edge: std::sync::RwLock::new(EdgeManager::new()),
            invalidation: std::sync::Mutex::new(InvalidationQueue::new(cap, rate)),
            origins: std::sync::Mutex::new(OriginPool::new(strategy)),
            geo: std::sync::RwLock::new(GeoRouter::new()),
            metrics: MetricsRegistry::new(),
        }
    }

    /// Submit a cache invalidation request to the queue.
    ///
    /// Returns an error if the queue is full or the lock is poisoned.
    pub fn submit_invalidation(&self, request: InvalidationRequest) -> Result<(), CdnError> {
        let mut q = self
            .invalidation
            .lock()
            .map_err(|e| CdnError::LockPoisoned(e.to_string()))?;
        q.submit(request)?;
        Ok(())
    }

    /// Process a batch of up to `batch_size` invalidations against `node_ids`.
    pub fn process_invalidations(
        &self,
        node_ids: &[&str],
        batch_size: usize,
    ) -> Result<Vec<InvalidationResult>, CdnError> {
        let mut q = self
            .invalidation
            .lock()
            .map_err(|e| CdnError::LockPoisoned(e.to_string()))?;
        Ok(q.process_batch(node_ids, batch_size))
    }

    /// Select the best origin server according to the configured strategy.
    pub fn select_origin(&self) -> Result<Option<std::sync::Arc<OriginServer>>, CdnError> {
        let pool = self
            .origins
            .lock()
            .map_err(|e| CdnError::LockPoisoned(e.to_string()))?;
        Ok(pool.select())
    }

    /// Assign the geographically closest edge node to `location`.
    ///
    /// Requires a write lock because [`GeoRouter::assign_edge`] lazily builds
    /// the R-tree index and populates the Haversine cache on first use.
    pub fn route(&self, location: &GeoLocation) -> Result<Option<EdgeNodeId>, CdnError> {
        let mut router = self
            .geo
            .write()
            .map_err(|e| CdnError::LockPoisoned(e.to_string()))?;
        Ok(router.assign_edge(location).cloned())
    }

    /// Render all metrics in Prometheus text format.
    pub fn prometheus_metrics(&self) -> String {
        self.metrics.to_prometheus()
    }
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::cache_invalidation::InvalidationScope;
    use crate::geo_routing::GeoLocation;
    use crate::origin_failover::OriginServer;

    // 1. CdnConfig::default values
    #[test]
    fn test_cdn_config_defaults() {
        let cfg = CdnConfig::default();
        assert_eq!(cfg.invalidation_rate_per_min, 100);
        assert_eq!(cfg.invalidation_queue_capacity, 10_000);
        assert_eq!(cfg.health_check_interval_secs, 30);
    }

    // 2. CdnManager creation
    #[test]
    fn test_cdn_manager_new() {
        let mgr = CdnManager::new(CdnConfig::default());
        // Edge manager starts empty
        assert_eq!(mgr.edge.read().expect("lock ok").nodes().len(), 0);
    }

    // 3. Add edge node via manager
    #[test]
    fn test_cdn_manager_add_edge_node() {
        let mgr = CdnManager::new(CdnConfig::default());
        mgr.edge.write().expect("lock ok").add_node(EdgeNode::new(
            "n1",
            "cf",
            "us-east-1",
            "n1.cdn.example.com",
        ));
        assert_eq!(mgr.edge.read().expect("lock ok").nodes().len(), 1);
    }

    // 4. submit_invalidation round-trip
    #[test]
    fn test_cdn_manager_invalidation_round_trip() {
        let mgr = CdnManager::new(CdnConfig::default());
        let req = InvalidationRequest::new(InvalidationScope::All, 10);
        mgr.submit_invalidation(req).expect("submit ok");
        let results = mgr
            .process_invalidations(&["node-1"], 10)
            .expect("process ok");
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    // 5. process_invalidations with no nodes returns 0 results
    #[test]
    fn test_cdn_manager_process_no_nodes() {
        let mgr = CdnManager::new(CdnConfig::default());
        let req = InvalidationRequest::new(InvalidationScope::All, 1);
        mgr.submit_invalidation(req).expect("ok");
        // No node IDs → nothing can be dispatched → request stays in queue.
        let results = mgr.process_invalidations(&[], 10).expect("process ok");
        assert_eq!(results.len(), 0);
    }

    // 6. Origin pool select returns None when empty
    #[test]
    fn test_cdn_manager_select_origin_empty() {
        let mgr = CdnManager::new(CdnConfig::default());
        assert!(mgr.select_origin().expect("lock ok").is_none());
    }

    // 7. Origin pool select returns a server after adding one
    #[test]
    fn test_cdn_manager_select_origin_after_add() {
        let mgr = CdnManager::new(CdnConfig::default());
        mgr.origins
            .lock()
            .expect("lock ok")
            .add_server(Arc::new(OriginServer::new("o1", "http://origin1", 1, 0)));
        let sel = mgr.select_origin().expect("lock ok").expect("origin");
        assert_eq!(sel.url, "http://origin1");
    }

    // 8. Geographic routing via CdnManager::route
    #[test]
    fn test_cdn_manager_route() {
        let mgr = CdnManager::new(CdnConfig::default());
        mgr.geo.write().expect("lock ok").add_node(EdgeNodeGeo::new(
            "eu-pop",
            GeoLocation::new(48.8566, 2.3522, "FR"), // Paris
        ));
        mgr.geo.write().expect("lock ok").add_node(EdgeNodeGeo::new(
            "us-pop",
            GeoLocation::new(40.7128, -74.0060, "US"), // New York
        ));
        let client = GeoLocation::new(51.5074, -0.1278, "GB"); // London
        let edge_id = mgr.route(&client).expect("lock ok").expect("edge");
        assert_eq!(edge_id.0, "eu-pop"); // Paris closer than New York from London
    }

    // 9. prometheus_metrics output is non-empty
    #[test]
    fn test_cdn_manager_prometheus_metrics() {
        let mgr = CdnManager::new(CdnConfig::default());
        mgr.metrics.record_hit("pop-1", 1000);
        let prom = mgr.prometheus_metrics();
        assert!(!prom.is_empty());
        assert!(prom.contains("cdn_requests_total"));
    }

    // 10. CdnConfig::with_strategy
    #[test]
    fn test_cdn_config_with_strategy() {
        let cfg = CdnConfig::with_strategy(OriginStrategy::WeightedRoundRobin);
        assert_eq!(cfg.origin_strategy, OriginStrategy::WeightedRoundRobin);
    }

    // 11. End-to-end: health check updates origin pool, then routing works
    #[test]
    fn test_e2e_health_check_and_routing() {
        let mgr = CdnManager::new(CdnConfig::default());

        // Register two origins, one that will fail
        let healthy_origin = Arc::new(OriginServer::new("h", "http://healthy", 1, 0));
        let flaky_origin = Arc::new(OriginServer::new("f", "http://flaky", 1, 1));
        {
            let mut pool = mgr.origins.lock().expect("lock ok");
            pool.add_server(Arc::clone(&healthy_origin));
            pool.add_server(Arc::clone(&flaky_origin));
        }

        // Simulate 3 failures on flaky
        for _ in 0..3 {
            flaky_origin.record_failure();
        }
        assert!(!flaky_origin.is_healthy());

        // Pool should now return the healthy origin (priority 0)
        let sel = mgr.select_origin().expect("lock ok").expect("origin");
        assert_eq!(sel.url, "http://healthy");
    }

    // 12. Invalidation queue capacity enforcement
    #[test]
    fn test_e2e_invalidation_queue_capacity() {
        let mut cfg = CdnConfig::default();
        cfg.invalidation_queue_capacity = 3;
        let mgr = CdnManager::new(cfg);
        for _ in 0..3 {
            mgr.submit_invalidation(InvalidationRequest::new(InvalidationScope::All, 1))
                .expect("ok");
        }
        let err = mgr
            .submit_invalidation(InvalidationRequest::new(InvalidationScope::All, 1))
            .unwrap_err();
        assert!(matches!(
            err,
            CdnError::Invalidation(InvalidationError::QueueFull(3))
        ));
    }

    // 13. Metrics aggregation across multiple nodes
    #[test]
    fn test_e2e_metrics_aggregation() {
        let mgr = CdnManager::new(CdnConfig::default());
        mgr.metrics.record_hit("pop-a", 1_073_741_824); // 1 GiB
        mgr.metrics.record_hit("pop-b", 1_073_741_824); // 1 GiB
        mgr.metrics.record_miss("pop-a", 100);
        let global = mgr.metrics.global.snapshot();
        assert_eq!(global.cache_hits, 2);
        assert!((global.total_bandwidth_gb() - 2.0).abs() < 1e-6);
    }

    // 14. HealthChecker integration with CdnManager origin pool
    #[test]
    fn test_e2e_health_checker() {
        let mut pool = OriginPool::new(OriginStrategy::Priority);
        let s1 = Arc::new(OriginServer::new("s1", "http://s1", 1, 0));
        pool.add_server(Arc::clone(&s1));
        let pool = Arc::new(pool);

        let checker = HealthChecker::new(Arc::clone(&pool), Duration::from_secs(60));
        assert!(checker.is_due());
        let probed = checker.check_now(|_| (true, 42.0));
        assert_eq!(probed, 1);
        assert!(s1.is_healthy());
        // EWMA should have been updated: 0.3*42 + 0.7*100 = 12.6 + 70 = 82.6
        let ewma = s1.ewma_ms();
        assert!((ewma - 82.6).abs() < 0.1, "ewma={ewma}");
    }

    // 15. GeoRouter haversine is internally consistent
    #[test]
    fn test_e2e_geo_latency_consistency() {
        let ny = GeoLocation::new(40.7128, -74.0060, "US");
        let london = GeoLocation::new(51.5074, -0.1278, "GB");
        let dist = haversine_km(ny.latitude, ny.longitude, london.latitude, london.longitude);
        let lat = latency_from_km(dist);
        let mut router = GeoRouter::new();
        let lat2 = router.latency_estimate_ms(&ny, &london);
        assert!((lat - lat2).abs() < 1e-9, "lat={lat} lat2={lat2}");
    }

    // 16. Concurrent invalidation stress test: 16 threads × 100 requests each
    #[test]
    fn test_concurrent_invalidation_stress() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let queue = Arc::new(Mutex::new(InvalidationQueue::new(100_000, 100_000)));
        let submitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let threads: Vec<_> = (0..16)
            .map(|_| {
                let q = Arc::clone(&queue);
                let counter = Arc::clone(&submitted);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let req = InvalidationRequest::new(InvalidationScope::All, 1);
                        if q.lock().expect("lock ok").submit(req).is_ok() {
                            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().expect("thread did not panic");
        }

        // All 1600 requests should have been submitted (queue capacity 100k)
        let total = submitted.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(total, 1600, "expected 1600 submissions, got {total}");

        // Process all and verify no panic
        let results = queue
            .lock()
            .expect("lock ok")
            .process_batch(&["stress-node"], 2000);
        assert_eq!(results.len(), 1600, "expected 1600 results processed");
    }

    // 17. CdnManager lifecycle integration test
    #[test]
    fn test_cdn_manager_lifecycle() {
        use crate::edge_manager::EdgeNode;

        let mgr = CdnManager::new(CdnConfig::default());

        // Add 3 edge nodes
        {
            let mut edge = mgr.edge.write().expect("edge lock ok");
            edge.add_node(EdgeNode::new(
                "node-1",
                "cf",
                "us-east-1",
                "n1.cdn.example.com",
            ));
            edge.add_node(EdgeNode::new(
                "node-2",
                "aws",
                "eu-west-1",
                "n2.cdn.example.com",
            ));
            edge.add_node(EdgeNode::new(
                "node-3",
                "gcp",
                "ap-east-1",
                "n3.cdn.example.com",
            ));
        }
        assert_eq!(mgr.edge.read().expect("read ok").nodes().len(), 3);

        // Configure 2 origins
        {
            let mut pool = mgr.origins.lock().expect("origins lock ok");
            pool.add_server(Arc::new(OriginServer::new(
                "o1",
                "http://origin1.example.com",
                1,
                0,
            )));
            pool.add_server(Arc::new(OriginServer::new(
                "o2",
                "http://origin2.example.com",
                1,
                1,
            )));
        }

        // Route 10 requests (geo routing)
        {
            let client = GeoLocation::new(40.7128, -74.0060, "US");
            mgr.geo
                .write()
                .expect("geo lock ok")
                .add_node(EdgeNodeGeo::new(
                    "us-pop",
                    GeoLocation::new(40.0, -74.0, "US"),
                ));
            mgr.geo
                .write()
                .expect("geo lock ok")
                .add_node(EdgeNodeGeo::new(
                    "eu-pop",
                    GeoLocation::new(51.5, 0.1, "GB"),
                ));
            for _ in 0..10 {
                let id = mgr.route(&client).expect("route ok");
                assert!(id.is_some(), "should route to an edge");
            }
        }

        // Invalidate 1 key
        let req =
            InvalidationRequest::new(InvalidationScope::Url("/media/hero.mp4".to_string()), 100);
        mgr.submit_invalidation(req).expect("submit ok");

        // Process the invalidation
        let results = mgr
            .process_invalidations(&["node-1", "node-2"], 10)
            .expect("process ok");
        assert_eq!(results.len(), 1, "one request should be processed");
        assert!(results[0].success, "invalidation should succeed");

        // Remaining requests should still route correctly
        let client2 = GeoLocation::new(51.5074, -0.1278, "GB");
        let id2 = mgr.route(&client2).expect("route ok");
        assert!(id2.is_some(), "should still route after invalidation");
    }
}
