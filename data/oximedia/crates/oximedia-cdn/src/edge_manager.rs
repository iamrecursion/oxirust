//! CDN edge node management and health monitoring.
//!
//! Provides [`EdgeNode`] and [`EdgeManager`] for tracking the state of CDN
//! edge PoPs, performing health checks, and selecting the best node for a
//! given region / feature-set combination.

/// A capability flag that an edge node may or may not support.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeFeature {
    /// HTTP/2 support.
    Http2,
    /// HTTP/3 (QUIC-based) support.
    Http3,
    /// TLS 1.3 support.
    Tls13,
    /// WebSocket proxying support.
    Websockets,
    /// QUIC transport support.
    Quic,
    /// Edge-Side Includes support.
    EdgeSideIncludes,
    /// On-the-fly image optimisation.
    ImageOptimization,
}

/// A single CDN edge node (Point of Presence).
#[derive(Debug, Clone)]
pub struct EdgeNode {
    /// Unique identifier for this node.
    pub id: String,
    /// Provider name: `"cloudflare"`, `"fastly"`, `"akamai"`, `"cloudfront"`, or `"custom"`.
    pub provider: String,
    /// AWS-style region identifier, e.g. `"us-east-1"`.
    pub region: String,
    /// Hostname of the edge node.
    pub hostname: String,
    /// Point-of-Presence identifier (data-centre code).
    pub pop: String,
    /// Nominal link capacity in Gbps.
    pub capacity_gbps: f32,
    /// Current load as a percentage (0 – 100).
    pub current_load_pct: f32,
    /// Most-recently observed latency in milliseconds.
    pub latency_ms: u32,
    /// Whether the node is currently available.
    pub available: bool,
    /// Feature flags supported by this node.
    pub features: Vec<EdgeFeature>,
}

impl EdgeNode {
    /// Create a new [`EdgeNode`] with sensible defaults.
    ///
    /// The node starts with zero load, 50 ms latency, marked as available,
    /// and no optional features.
    pub fn new(id: &str, provider: &str, region: &str, hostname: &str) -> Self {
        Self {
            id: id.to_string(),
            provider: provider.to_string(),
            region: region.to_string(),
            hostname: hostname.to_string(),
            pop: String::new(),
            capacity_gbps: 0.0,
            current_load_pct: 0.0,
            latency_ms: 50,
            available: true,
            features: Vec::new(),
        }
    }

    /// Compute a composite quality score in `[0, 1]`.
    ///
    /// When `capacity_gbps > 0`, capacity is factored in via a
    /// capacity-weighted adjustment:
    ///
    /// ```text
    /// base   = (1 - load/100) * 0.3
    ///        + clamp(1 - latency/1000, 0, 1) * 0.3
    ///        + (available ? 1 : 0) * 0.15
    ///        + capacity_factor * 0.25
    /// ```
    ///
    /// where `capacity_factor = clamp(capacity_gbps / 100.0, 0, 1)`.
    ///
    /// When `capacity_gbps == 0` (legacy / unset), the original formula is
    /// used for backward compatibility:
    ///
    /// ```text
    /// score = (1 - load/100) * 0.4
    ///       + clamp(1 - latency/1000, 0, 1) * 0.4
    ///       + (available ? 1 : 0) * 0.2
    /// ```
    pub fn score(&self) -> f32 {
        if self.capacity_gbps > 0.0 {
            self.weighted_score()
        } else {
            self.legacy_score()
        }
    }

    /// Legacy score formula (no capacity weighting).
    fn legacy_score(&self) -> f32 {
        let load_score = (1.0 - self.current_load_pct / 100.0).clamp(0.0, 1.0) * 0.4;
        let latency_score = (1.0 - self.latency_ms as f32 / 1000.0).max(0.0) * 0.4;
        let avail_score = if self.available { 1.0_f32 } else { 0.0_f32 } * 0.2;
        load_score + latency_score + avail_score
    }

    /// Capacity-weighted score formula for heterogeneous node weights.
    fn weighted_score(&self) -> f32 {
        let load_score = (1.0 - self.current_load_pct / 100.0).clamp(0.0, 1.0) * 0.3;
        let latency_score = (1.0 - self.latency_ms as f32 / 1000.0).max(0.0) * 0.3;
        let avail_score = if self.available { 1.0_f32 } else { 0.0_f32 } * 0.15;
        let capacity_factor = (self.capacity_gbps / 100.0).clamp(0.0, 1.0) * 0.25;
        load_score + latency_score + avail_score + capacity_factor
    }

    /// Compute effective capacity: `capacity_gbps * (1 - load/100)`.
    ///
    /// Returns `0.0` if the node is unavailable.
    pub fn effective_capacity_gbps(&self) -> f32 {
        if !self.available {
            return 0.0;
        }
        self.capacity_gbps * (1.0 - self.current_load_pct / 100.0).max(0.0)
    }

    /// Return `true` if the node supports the given [`EdgeFeature`].
    pub fn supports(&self, feature: &EdgeFeature) -> bool {
        self.features.contains(feature)
    }
}

/// Manages a collection of [`EdgeNode`]s and provides selection helpers.
pub struct EdgeManager {
    nodes: Vec<EdgeNode>,
    /// Desired health-check polling interval (seconds).  Not actively used in
    /// this pure-logic crate but exposed for callers that drive health checks.
    pub health_check_interval_secs: u64,
}

impl EdgeManager {
    /// Create an empty [`EdgeManager`] with a 30-second health-check interval.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            health_check_interval_secs: 30,
        }
    }

    /// Add a node to the pool.
    pub fn add_node(&mut self, node: EdgeNode) {
        self.nodes.push(node);
    }

    /// Remove a node by ID.  Returns `true` if a node was actually removed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        self.nodes.len() < before
    }

    /// Update health metrics for a node.  Returns `false` if no node with
    /// `id` was found.
    pub fn update_health(
        &mut self,
        id: &str,
        latency_ms: u32,
        load_pct: f32,
        available: bool,
    ) -> bool {
        for node in &mut self.nodes {
            if node.id == id {
                node.latency_ms = latency_ms;
                node.current_load_pct = load_pct.clamp(0.0, 100.0);
                node.available = available;
                return true;
            }
        }
        false
    }

    /// Return the highest-scoring available node in `region` that supports
    /// **all** of the requested `features`.
    pub fn best_node_for(&self, region: &str, features: &[EdgeFeature]) -> Option<&EdgeNode> {
        self.nodes
            .iter()
            .filter(|n| n.region == region && n.available && features.iter().all(|f| n.supports(f)))
            .max_by(|a, b| {
                a.score()
                    .partial_cmp(&b.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return all nodes whose region matches `region`.
    pub fn nodes_in_region(&self, region: &str) -> Vec<&EdgeNode> {
        self.nodes.iter().filter(|n| n.region == region).collect()
    }

    /// Average load percentage across all *available* nodes.  Returns `0.0`
    /// when there are no available nodes.
    pub fn global_load_pct(&self) -> f32 {
        let available: Vec<&EdgeNode> = self.nodes.iter().filter(|n| n.available).collect();
        if available.is_empty() {
            return 0.0;
        }
        let total: f32 = available.iter().map(|n| n.current_load_pct).sum();
        total / available.len() as f32
    }

    /// Return all nodes whose `current_load_pct` exceeds `threshold_pct`.
    pub fn overloaded_nodes(&self, threshold_pct: f32) -> Vec<&EdgeNode> {
        self.nodes
            .iter()
            .filter(|n| n.current_load_pct > threshold_pct)
            .collect()
    }

    /// Return alternative nodes (excluding `primary_id`) sorted by descending
    /// score — the recommended failover chain.
    pub fn failover_chain(&self, primary_id: &str) -> Vec<&EdgeNode> {
        let mut chain: Vec<&EdgeNode> = self.nodes.iter().filter(|n| n.id != primary_id).collect();
        chain.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        chain
    }

    /// Select the best node in `region` using capacity-weighted scoring.
    ///
    /// Unlike [`best_node_for`](EdgeManager::best_node_for), this method
    /// explicitly prioritises nodes with higher effective capacity (capacity
    /// multiplied by headroom). This is useful in heterogeneous deployments
    /// where edge nodes have different link speeds.
    pub fn best_node_by_capacity(
        &self,
        region: &str,
        features: &[EdgeFeature],
    ) -> Option<&EdgeNode> {
        self.nodes
            .iter()
            .filter(|n| n.region == region && n.available && features.iter().all(|f| n.supports(f)))
            .max_by(|a, b| {
                let a_eff = a.effective_capacity_gbps();
                let b_eff = b.effective_capacity_gbps();
                a_eff
                    .partial_cmp(&b_eff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Total effective capacity across all available nodes in `region`.
    pub fn total_capacity_gbps(&self, region: &str) -> f32 {
        self.nodes
            .iter()
            .filter(|n| n.region == region && n.available)
            .map(|n| n.effective_capacity_gbps())
            .sum()
    }

    /// Return all nodes sorted by effective capacity (descending).
    pub fn nodes_by_capacity(&self) -> Vec<&EdgeNode> {
        let mut sorted: Vec<&EdgeNode> = self.nodes.iter().filter(|n| n.available).collect();
        sorted.sort_by(|a, b| {
            b.effective_capacity_gbps()
                .partial_cmp(&a.effective_capacity_gbps())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> &[EdgeNode] {
        &self.nodes
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, region: &str, load: f32, latency: u32, available: bool) -> EdgeNode {
        let mut n = EdgeNode::new(id, "cloudflare", region, &format!("{id}.example.com"));
        n.current_load_pct = load;
        n.latency_ms = latency;
        n.available = available;
        n
    }

    // 1. Basic construction
    #[test]
    fn test_new_node_defaults() {
        let n = EdgeNode::new("node-1", "fastly", "us-east-1", "cdn1.example.com");
        assert_eq!(n.id, "node-1");
        assert_eq!(n.provider, "fastly");
        assert_eq!(n.region, "us-east-1");
        assert_eq!(n.hostname, "cdn1.example.com");
        assert!(n.available);
        assert_eq!(n.current_load_pct, 0.0);
        assert!(n.features.is_empty());
    }

    // 2. Score at zero load and low latency
    #[test]
    fn test_score_zero_load_low_latency() {
        let n = make_node("n", "r", 0.0, 0, true);
        // (1-0)*0.4 + (1-0)*0.4 + 1*0.2 = 1.0
        let s = n.score();
        assert!((s - 1.0).abs() < 1e-5, "score={s}");
    }

    // 3. Score at 100% load
    #[test]
    fn test_score_full_load() {
        let n = make_node("n", "r", 100.0, 0, true);
        // load_score=0, latency_score=0.4, avail_score=0.2 → 0.6
        let s = n.score();
        assert!((s - 0.6).abs() < 1e-5, "score={s}");
    }

    // 4. Score when unavailable
    #[test]
    fn test_score_unavailable() {
        let n = make_node("n", "r", 0.0, 0, false);
        // (1)*0.4 + (1)*0.4 + 0*0.2 = 0.8
        let s = n.score();
        assert!((s - 0.8).abs() < 1e-5, "score={s}");
    }

    // 5. Score clamps latency at 1000 ms
    #[test]
    fn test_score_high_latency_clamp() {
        let n = make_node("n", "r", 0.0, 2000, true);
        // latency_score = max(1 - 2000/1000, 0) * 0.4 = 0
        let s = n.score();
        // 0.4 + 0 + 0.2 = 0.6
        assert!((s - 0.6).abs() < 1e-5, "score={s}");
    }

    // 6. Feature support
    #[test]
    fn test_supports_feature() {
        let mut n = EdgeNode::new("n", "cf", "eu-west-1", "h");
        n.features.push(EdgeFeature::Http3);
        n.features.push(EdgeFeature::Tls13);
        assert!(n.supports(&EdgeFeature::Http3));
        assert!(n.supports(&EdgeFeature::Tls13));
        assert!(!n.supports(&EdgeFeature::Quic));
    }

    // 7. Add / remove nodes
    #[test]
    fn test_add_remove_node() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("n1", "r", 0.0, 50, true));
        assert_eq!(mgr.nodes().len(), 1);
        assert!(mgr.remove_node("n1"));
        assert_eq!(mgr.nodes().len(), 0);
        assert!(!mgr.remove_node("n1")); // already gone
    }

    // 8. update_health success path
    #[test]
    fn test_update_health_found() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("n1", "r", 10.0, 100, true));
        assert!(mgr.update_health("n1", 200, 60.0, false));
        let n = &mgr.nodes()[0];
        assert_eq!(n.latency_ms, 200);
        assert!((n.current_load_pct - 60.0).abs() < 1e-5);
        assert!(!n.available);
    }

    // 9. update_health unknown ID
    #[test]
    fn test_update_health_not_found() {
        let mut mgr = EdgeManager::new();
        assert!(!mgr.update_health("ghost", 100, 50.0, true));
    }

    // 10. best_node_for selects highest score in correct region
    #[test]
    fn test_best_node_for_region_score() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("a", "us-east-1", 50.0, 100, true));
        mgr.add_node(make_node("b", "us-east-1", 10.0, 20, true)); // better score
        mgr.add_node(make_node("c", "eu-west-1", 0.0, 5, true)); // wrong region
        let best = mgr
            .best_node_for("us-east-1", &[])
            .expect("should find a node");
        assert_eq!(best.id, "b");
    }

    // 11. best_node_for filters by feature
    #[test]
    fn test_best_node_for_filters_features() {
        let mut mgr = EdgeManager::new();
        let mut n1 = make_node("n1", "us-east-1", 0.0, 10, true);
        n1.features.push(EdgeFeature::Http2);
        let n2 = make_node("n2", "us-east-1", 0.0, 20, true); // no features
        mgr.add_node(n1);
        mgr.add_node(n2);
        // Requesting Http3: neither node has it → None
        assert!(mgr
            .best_node_for("us-east-1", &[EdgeFeature::Http3])
            .is_none());
        // Requesting Http2: only n1 qualifies
        let best = mgr
            .best_node_for("us-east-1", &[EdgeFeature::Http2])
            .expect("n1");
        assert_eq!(best.id, "n1");
    }

    // 12. nodes_in_region
    #[test]
    fn test_nodes_in_region() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("a", "us-east-1", 0.0, 50, true));
        mgr.add_node(make_node("b", "us-east-1", 0.0, 50, true));
        mgr.add_node(make_node("c", "ap-northeast-1", 0.0, 50, true));
        let r = mgr.nodes_in_region("us-east-1");
        assert_eq!(r.len(), 2);
    }

    // 13. global_load_pct
    #[test]
    fn test_global_load_pct() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("a", "r", 40.0, 50, true));
        mgr.add_node(make_node("b", "r", 60.0, 50, true));
        mgr.add_node(make_node("c", "r", 80.0, 50, false)); // unavailable — excluded
        let avg = mgr.global_load_pct();
        assert!((avg - 50.0).abs() < 1e-4, "avg={avg}");
    }

    // 14. overloaded_nodes
    #[test]
    fn test_overloaded_nodes() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("a", "r", 30.0, 50, true));
        mgr.add_node(make_node("b", "r", 75.0, 50, true));
        mgr.add_node(make_node("c", "r", 90.0, 50, true));
        let over = mgr.overloaded_nodes(70.0);
        assert_eq!(over.len(), 2);
        let ids: Vec<&str> = over.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    // 15. failover_chain excludes primary and sorts by descending score
    #[test]
    fn test_failover_chain() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_node("primary", "r", 50.0, 200, true));
        mgr.add_node(make_node("alt1", "r", 10.0, 20, true)); // best score
        mgr.add_node(make_node("alt2", "r", 80.0, 500, true)); // worst score
        let chain = mgr.failover_chain("primary");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].id, "alt1"); // highest score first
        assert!(!chain.iter().any(|n| n.id == "primary"));
    }

    // ── Capacity-aware scoring ────────────────────────────────────────────

    fn make_capacity_node(
        id: &str,
        region: &str,
        load: f32,
        latency: u32,
        available: bool,
        capacity_gbps: f32,
    ) -> EdgeNode {
        let mut n = make_node(id, region, load, latency, available);
        n.capacity_gbps = capacity_gbps;
        n
    }

    // 16. weighted_score includes capacity factor
    #[test]
    fn test_weighted_score_with_capacity() {
        let n = make_capacity_node("n", "r", 0.0, 0, true, 100.0);
        // capacity_factor = (100/100).clamp(0,1) * 0.25 = 0.25
        // load_score = (1-0)*0.3 = 0.3
        // latency_score = (1-0)*0.3 = 0.3
        // avail_score = 1*0.15 = 0.15
        // total = 1.0
        let s = n.score();
        assert!((s - 1.0).abs() < 1e-5, "score={s}");
    }

    // 17. weighted_score vs legacy_score when capacity=0
    #[test]
    fn test_legacy_score_when_no_capacity() {
        let n = make_node("n", "r", 0.0, 0, true);
        assert!((n.capacity_gbps - 0.0).abs() < 1e-5);
        let s = n.score();
        // Legacy formula: 0.4 + 0.4 + 0.2 = 1.0
        assert!((s - 1.0).abs() < 1e-5, "score={s}");
    }

    // 18. Higher capacity → higher score
    #[test]
    fn test_higher_capacity_higher_score() {
        let small = make_capacity_node("s", "r", 50.0, 100, true, 10.0);
        let big = make_capacity_node("b", "r", 50.0, 100, true, 100.0);
        assert!(
            big.score() > small.score(),
            "big={} small={}",
            big.score(),
            small.score()
        );
    }

    // 19. effective_capacity_gbps
    #[test]
    fn test_effective_capacity() {
        let n = make_capacity_node("n", "r", 50.0, 50, true, 40.0);
        let eff = n.effective_capacity_gbps();
        // 40 * (1 - 0.5) = 20
        assert!((eff - 20.0).abs() < 1e-4, "eff={eff}");
    }

    // 20. effective_capacity is 0 when unavailable
    #[test]
    fn test_effective_capacity_unavailable() {
        let n = make_capacity_node("n", "r", 0.0, 0, false, 100.0);
        assert!((n.effective_capacity_gbps() - 0.0).abs() < 1e-5);
    }

    // 21. best_node_by_capacity selects highest effective capacity
    #[test]
    fn test_best_node_by_capacity() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_capacity_node(
            "small",
            "us-east-1",
            10.0,
            50,
            true,
            10.0,
        ));
        mgr.add_node(make_capacity_node(
            "big",
            "us-east-1",
            10.0,
            50,
            true,
            100.0,
        ));
        mgr.add_node(make_capacity_node(
            "other",
            "eu-west-1",
            0.0,
            10,
            true,
            200.0,
        ));

        let best = mgr.best_node_by_capacity("us-east-1", &[]).expect("node");
        assert_eq!(best.id, "big");
    }

    // 22. best_node_by_capacity respects features
    #[test]
    fn test_best_node_by_capacity_with_features() {
        let mut mgr = EdgeManager::new();
        let mut n1 = make_capacity_node("n1", "r", 0.0, 50, true, 100.0);
        n1.features.push(EdgeFeature::Http3);
        let n2 = make_capacity_node("n2", "r", 0.0, 50, true, 200.0);
        // n2 has no Http3
        mgr.add_node(n1);
        mgr.add_node(n2);

        let best = mgr
            .best_node_by_capacity("r", &[EdgeFeature::Http3])
            .expect("n1");
        assert_eq!(best.id, "n1");
    }

    // 23. total_capacity_gbps
    #[test]
    fn test_total_capacity_gbps() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_capacity_node("a", "r", 0.0, 50, true, 40.0));
        mgr.add_node(make_capacity_node("b", "r", 50.0, 50, true, 40.0));
        mgr.add_node(make_capacity_node("c", "r", 0.0, 50, false, 100.0)); // unavailable
                                                                           // a: 40 * 1.0 = 40, b: 40 * 0.5 = 20, c: excluded
        let total = mgr.total_capacity_gbps("r");
        assert!((total - 60.0).abs() < 1e-4, "total={total}");
    }

    // 24. nodes_by_capacity returns sorted order
    #[test]
    fn test_nodes_by_capacity_sorted() {
        let mut mgr = EdgeManager::new();
        mgr.add_node(make_capacity_node("low", "r", 0.0, 50, true, 10.0));
        mgr.add_node(make_capacity_node("high", "r", 0.0, 50, true, 100.0));
        mgr.add_node(make_capacity_node("mid", "r", 0.0, 50, true, 50.0));
        let sorted = mgr.nodes_by_capacity();
        assert_eq!(sorted[0].id, "high");
        assert_eq!(sorted[1].id, "mid");
        assert_eq!(sorted[2].id, "low");
    }

    // 25. capacity_factor clamps at 100 Gbps
    #[test]
    fn test_capacity_factor_clamped() {
        let n1 = make_capacity_node("n1", "r", 0.0, 0, true, 100.0);
        let n2 = make_capacity_node("n2", "r", 0.0, 0, true, 500.0);
        // Both should have the same capacity_factor (clamped to 1.0)
        assert!((n1.score() - n2.score()).abs() < 1e-5);
    }
}
