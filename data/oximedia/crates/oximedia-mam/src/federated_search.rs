//! Federated search across multiple MAM instances.
//!
//! This module enables a coordinator node to fan a search query out to a set
//! of remote MAM peers, merge the results, and return a unified ranked result
//! set to the caller — all without requiring a shared database.
//!
//! Key features:
//! - Peer registry with health-check state
//! - Query serialisation / normalisation
//! - Result merging with configurable ranking strategies
//! - Deduplication of cross-instance hits by content checksum
//! - Per-peer timeout and error isolation (one failing peer does not block)
//! - Facet aggregation across peers

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Peer registry
// ---------------------------------------------------------------------------

/// Status of a remote MAM peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerStatus {
    /// Peer is reachable and healthy.
    Healthy,
    /// Peer was last seen healthy but has not been checked recently.
    Unknown,
    /// Peer failed its last health check.
    Unhealthy,
    /// Peer is intentionally excluded from federated queries.
    Disabled,
}

impl PeerStatus {
    /// Returns `true` if the peer should be included in federated searches.
    #[must_use]
    pub const fn is_queryable(&self) -> bool {
        matches!(self, Self::Healthy | Self::Unknown)
    }
}

/// Configuration for a remote MAM peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Unique peer id (assigned locally).
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Base URL of the remote MAM API.
    pub base_url: String,
    /// Optional API key / bearer token.
    pub api_key: Option<String>,
    /// Query timeout in milliseconds.
    pub timeout_ms: u64,
    /// Weight applied to results from this peer (higher = more trust).
    pub weight: f64,
    /// Geographical region tag for locality-aware routing.
    pub region: Option<String>,
}

impl PeerConfig {
    /// Create a new peer configuration.
    #[must_use]
    pub fn new(name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            base_url: base_url.into(),
            api_key: None,
            timeout_ms: 5_000,
            weight: 1.0,
            region: None,
        }
    }

    /// Builder: set API key.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Builder: set timeout.
    #[must_use]
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Builder: set peer weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Builder: set region.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }
}

/// Runtime record for a peer including health state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerRecord {
    pub config: PeerConfig,
    pub status: PeerStatus,
    pub last_checked_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    /// Average response time in ms (exponential moving average).
    pub avg_response_ms: f64,
    /// Consecutive error count.
    pub error_streak: u32,
}

impl PeerRecord {
    /// Create a new unknown-status record.
    #[must_use]
    pub fn new(config: PeerConfig) -> Self {
        Self {
            config,
            status: PeerStatus::Unknown,
            last_checked_at: None,
            last_error: None,
            avg_response_ms: 0.0,
            error_streak: 0,
        }
    }

    /// Record a successful health check / query response.
    pub fn record_success(&mut self, response_ms: u64) {
        self.status = PeerStatus::Healthy;
        self.last_checked_at = Some(Utc::now());
        self.last_error = None;
        self.error_streak = 0;
        // Exponential moving average with α=0.2
        self.avg_response_ms = 0.8 * self.avg_response_ms + 0.2 * response_ms as f64;
    }

    /// Record a failed health check / query.
    pub fn record_failure(&mut self, error: impl Into<String>) {
        self.last_checked_at = Some(Utc::now());
        self.last_error = Some(error.into());
        self.error_streak += 1;
        if self.error_streak >= 3 {
            self.status = PeerStatus::Unhealthy;
        }
    }

    /// Disable this peer.
    pub fn disable(&mut self) {
        self.status = PeerStatus::Disabled;
    }

    /// Enable this peer (reset to Unknown for next check).
    pub fn enable(&mut self) {
        self.status = PeerStatus::Unknown;
        self.error_streak = 0;
    }
}

// ---------------------------------------------------------------------------
// Federated query
// ---------------------------------------------------------------------------

/// Sort order for merged results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    Relevance,
    DateNewest,
    DateOldest,
    SizeAscending,
    SizeDescending,
    NameAscending,
}

/// A search query sent to each peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQuery {
    /// Free-text search string.
    pub query: String,
    /// Optional MIME-type filter (e.g. `"video/*"`).
    pub mime_filter: Option<String>,
    /// Optional date range filter (start).
    pub date_from: Option<DateTime<Utc>>,
    /// Optional date range filter (end).
    pub date_to: Option<DateTime<Utc>>,
    /// Tag filters (all must match).
    pub tags: Vec<String>,
    /// Desired sort order.
    pub sort: SortOrder,
    /// Maximum results per peer.
    pub limit_per_peer: usize,
    /// Deduplicate results by content checksum across peers.
    pub deduplicate: bool,
    /// Peers to exclude (by id).
    pub exclude_peers: Vec<Uuid>,
    /// Only query peers in these regions.
    pub region_filter: Option<Vec<String>>,
}

impl FederatedQuery {
    /// Create a simple text query.
    #[must_use]
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            mime_filter: None,
            date_from: None,
            date_to: None,
            tags: Vec::new(),
            sort: SortOrder::Relevance,
            limit_per_peer: 20,
            deduplicate: true,
            exclude_peers: Vec::new(),
            region_filter: None,
        }
    }

    /// Builder: set MIME filter.
    #[must_use]
    pub fn with_mime(mut self, mime: impl Into<String>) -> Self {
        self.mime_filter = Some(mime.into());
        self
    }

    /// Builder: add a tag filter.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Builder: set sort order.
    #[must_use]
    pub fn sorted_by(mut self, order: SortOrder) -> Self {
        self.sort = order;
        self
    }

    /// Builder: set limit per peer.
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit_per_peer = n;
        self
    }

    /// Builder: disable deduplication.
    #[must_use]
    pub fn no_dedup(mut self) -> Self {
        self.deduplicate = false;
        self
    }

    /// Builder: restrict to regions.
    #[must_use]
    pub fn in_regions(mut self, regions: Vec<String>) -> Self {
        self.region_filter = Some(regions);
        self
    }
}

// ---------------------------------------------------------------------------
// Search hit
// ---------------------------------------------------------------------------

/// A single search result from a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Asset id on the originating peer.
    pub asset_id: Uuid,
    /// Originating peer id.
    pub peer_id: Uuid,
    /// Originating peer name (for display).
    pub peer_name: String,
    /// Asset filename.
    pub filename: String,
    /// Asset title (if available).
    pub title: Option<String>,
    /// MIME type.
    pub mime_type: Option<String>,
    /// File size in bytes.
    pub size_bytes: Option<u64>,
    /// Creation timestamp.
    pub created_at: Option<DateTime<Utc>>,
    /// Relevance score from the originating peer (higher = more relevant).
    pub score: f64,
    /// Content checksum (e.g. SHA-256 hex) used for deduplication.
    pub checksum: Option<String>,
    /// Tags attached to the asset.
    pub tags: Vec<String>,
    /// Preview URL (proxy thumbnail).
    pub preview_url: Option<String>,
    /// Direct link to the asset detail page on the peer.
    pub detail_url: Option<String>,
}

impl SearchHit {
    /// Create a minimal search hit.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        peer_id: Uuid,
        peer_name: impl Into<String>,
        filename: impl Into<String>,
        score: f64,
    ) -> Self {
        Self {
            asset_id,
            peer_id,
            peer_name: peer_name.into(),
            filename: filename.into(),
            title: None,
            mime_type: None,
            size_bytes: None,
            created_at: None,
            score,
            checksum: None,
            tags: Vec::new(),
            preview_url: None,
            detail_url: None,
        }
    }

    /// Weighted score using the peer's configured weight.
    #[must_use]
    pub fn weighted_score(&self, weight: f64) -> f64 {
        self.score * weight
    }
}

// ---------------------------------------------------------------------------
// Peer response
// ---------------------------------------------------------------------------

/// Response from a single peer to a federated query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerQueryResponse {
    pub peer_id: Uuid,
    pub peer_name: String,
    pub hits: Vec<SearchHit>,
    /// Total matching assets on this peer (may exceed `hits.len()`).
    pub total_count: u64,
    /// How long the query took in milliseconds.
    pub latency_ms: u64,
    /// Whether this response was retrieved successfully.
    pub success: bool,
    /// Error message if `!success`.
    pub error: Option<String>,
    /// Facets returned by this peer.
    pub facets: HashMap<String, Vec<FacetBucket>>,
}

/// A facet bucket (e.g. mime_type="video/mp4", count=42).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetBucket {
    pub value: String,
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Merged results
// ---------------------------------------------------------------------------

/// Ranking strategy for merging hits from multiple peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingStrategy {
    /// Multiply each hit's score by its peer's weight, then sort descending.
    WeightedScore,
    /// Round-robin across peers (interleave).
    RoundRobin,
    /// Newest assets first (requires `created_at`).
    Newest,
    /// Peers with lower latency contribute results first.
    LowLatencyFirst,
}

/// The merged result of a federated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    pub hits: Vec<SearchHit>,
    /// Total across all peers (before dedup / limit).
    pub total_count: u64,
    /// Individual peer responses.
    pub peer_responses: Vec<PeerQueryResponse>,
    /// Number of duplicates removed (if dedup was enabled).
    pub duplicates_removed: usize,
    /// Aggregated facets from all peers.
    pub facets: HashMap<String, Vec<FacetBucket>>,
    /// When the federated query was completed.
    pub completed_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Result merger
// ---------------------------------------------------------------------------

/// Merges peer responses into a single ranked result set.
pub struct ResultMerger;

impl ResultMerger {
    /// Merge peer responses according to the strategy.
    ///
    /// `peer_weights`: peer_id → weight (1.0 if not present).
    #[must_use]
    pub fn merge(
        responses: Vec<PeerQueryResponse>,
        strategy: RankingStrategy,
        deduplicate: bool,
        peer_weights: &HashMap<Uuid, f64>,
        max_results: usize,
    ) -> FederatedResult {
        let total_count: u64 = responses
            .iter()
            .map(|r| r.total_count)
            .collect::<Vec<_>>()
            .iter()
            .sum();
        let facets = Self::aggregate_facets(&responses);

        let mut all_hits: Vec<SearchHit> = responses
            .iter()
            .flat_map(|r| r.hits.iter().cloned())
            .collect();

        // Deduplication by checksum
        let mut duplicates_removed = 0;
        if deduplicate {
            let mut seen_checksums = std::collections::HashSet::new();
            let before = all_hits.len();
            all_hits.retain(|h| {
                if let Some(ref ck) = h.checksum {
                    seen_checksums.insert(ck.clone())
                } else {
                    true
                }
            });
            duplicates_removed = before - all_hits.len();
        }

        // Sort / interleave based on strategy
        match strategy {
            RankingStrategy::WeightedScore => {
                all_hits.sort_by(|a, b| {
                    let wa = peer_weights.get(&a.peer_id).copied().unwrap_or(1.0);
                    let wb = peer_weights.get(&b.peer_id).copied().unwrap_or(1.0);
                    b.weighted_score(wb)
                        .partial_cmp(&a.weighted_score(wa))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            RankingStrategy::RoundRobin => {
                all_hits = Self::round_robin(all_hits, peer_weights);
            }
            RankingStrategy::Newest => {
                all_hits.sort_by(|a, b| {
                    let ta = a.created_at.unwrap_or(DateTime::UNIX_EPOCH);
                    let tb = b.created_at.unwrap_or(DateTime::UNIX_EPOCH);
                    tb.cmp(&ta)
                });
            }
            RankingStrategy::LowLatencyFirst => {
                // Build latency map
                let latency: HashMap<Uuid, u64> = responses
                    .iter()
                    .map(|r| (r.peer_id, r.latency_ms))
                    .collect();
                all_hits.sort_by_key(|h| latency.get(&h.peer_id).copied().unwrap_or(u64::MAX));
            }
        }

        let hits: Vec<SearchHit> = all_hits.into_iter().take(max_results).collect();

        FederatedResult {
            hits,
            total_count,
            peer_responses: responses,
            duplicates_removed,
            facets,
            completed_at: Utc::now(),
        }
    }

    fn round_robin(hits: Vec<SearchHit>, _weights: &HashMap<Uuid, f64>) -> Vec<SearchHit> {
        // Group by peer
        let mut by_peer: HashMap<Uuid, Vec<SearchHit>> = HashMap::new();
        for h in hits {
            by_peer.entry(h.peer_id).or_default().push(h);
        }
        let mut peer_queues: Vec<Vec<SearchHit>> = by_peer.into_values().collect();
        let mut result = Vec::new();
        let mut round = 0;
        loop {
            let mut any = false;
            for queue in &mut peer_queues {
                if round < queue.len() {
                    result.push(queue[round].clone());
                    any = true;
                }
            }
            if !any {
                break;
            }
            round += 1;
        }
        result
    }

    fn aggregate_facets(responses: &[PeerQueryResponse]) -> HashMap<String, Vec<FacetBucket>> {
        let mut agg: HashMap<String, HashMap<String, u64>> = HashMap::new();
        for resp in responses {
            for (facet, buckets) in &resp.facets {
                let entry = agg.entry(facet.clone()).or_default();
                for b in buckets {
                    *entry.entry(b.value.clone()).or_insert(0) += b.count;
                }
            }
        }
        agg.into_iter()
            .map(|(facet, buckets)| {
                let mut sorted: Vec<FacetBucket> = buckets
                    .into_iter()
                    .map(|(value, count)| FacetBucket { value, count })
                    .collect();
                sorted.sort_by(|a, b| b.count.cmp(&a.count));
                (facet, sorted)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Peer registry
// ---------------------------------------------------------------------------

/// Registry of known MAM peers.
#[derive(Debug)]
pub struct PeerRegistry {
    peers: HashMap<Uuid, PeerRecord>,
}

impl PeerRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Register a peer.
    pub fn register(&mut self, config: PeerConfig) -> Uuid {
        let id = config.id;
        self.peers.insert(id, PeerRecord::new(config));
        id
    }

    /// Remove a peer.
    pub fn deregister(&mut self, id: Uuid) {
        self.peers.remove(&id);
    }

    /// Get a peer record.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&PeerRecord> {
        self.peers.get(&id)
    }

    /// Get a mutable peer record.
    #[must_use]
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut PeerRecord> {
        self.peers.get_mut(&id)
    }

    /// All queryable peers.
    #[must_use]
    pub fn queryable_peers(&self) -> Vec<&PeerRecord> {
        self.peers
            .values()
            .filter(|p| p.status.is_queryable())
            .collect()
    }

    /// Peers filtered by a query's region and exclusion list.
    #[must_use]
    pub fn peers_for_query(&self, query: &FederatedQuery) -> Vec<&PeerRecord> {
        self.peers
            .values()
            .filter(|p| {
                if !p.status.is_queryable() {
                    return false;
                }
                if query.exclude_peers.contains(&p.config.id) {
                    return false;
                }
                if let Some(ref regions) = query.region_filter {
                    if let Some(ref region) = p.config.region {
                        if !regions.contains(region) {
                            return false;
                        }
                    } else {
                        // Peer has no region — exclude from region-filtered query
                        return false;
                    }
                }
                true
            })
            .collect()
    }

    /// Weight map: peer_id → weight.
    #[must_use]
    pub fn weight_map(&self) -> HashMap<Uuid, f64> {
        self.peers
            .values()
            .map(|p| (p.config.id, p.config.weight))
            .collect()
    }

    /// Total number of registered peers.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}

impl Default for PeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    fn make_peer(name: &str, url: &str) -> PeerConfig {
        PeerConfig::new(name, url)
    }

    fn make_hit(peer_id: Uuid, peer_name: &str, score: f64, checksum: Option<&str>) -> SearchHit {
        let mut h = SearchHit::new(uid(), peer_id, peer_name, "file.mp4", score);
        h.checksum = checksum.map(|s| s.to_string());
        h
    }

    fn success_response(peer_id: Uuid, name: &str, hits: Vec<SearchHit>) -> PeerQueryResponse {
        PeerQueryResponse {
            peer_id,
            peer_name: name.to_string(),
            total_count: hits.len() as u64,
            latency_ms: 100,
            success: true,
            error: None,
            facets: HashMap::new(),
            hits,
        }
    }

    // --- PeerStatus ---

    #[test]
    fn test_peer_status_queryable() {
        assert!(PeerStatus::Healthy.is_queryable());
        assert!(PeerStatus::Unknown.is_queryable());
        assert!(!PeerStatus::Unhealthy.is_queryable());
        assert!(!PeerStatus::Disabled.is_queryable());
    }

    // --- PeerConfig builder ---

    #[test]
    fn test_peer_config_builder() {
        let cfg = PeerConfig::new("Tokyo", "https://mam.example.jp")
            .with_api_key("secret")
            .with_timeout(2_000)
            .with_weight(1.5)
            .with_region("ap-northeast");
        assert_eq!(cfg.timeout_ms, 2_000);
        assert_eq!(cfg.weight, 1.5);
        assert_eq!(cfg.region.as_deref(), Some("ap-northeast"));
        assert!(cfg.api_key.is_some());
    }

    // --- PeerRecord health tracking ---

    #[test]
    fn test_peer_record_success_resets_streak() {
        let mut r = PeerRecord::new(make_peer("A", "http://a"));
        r.record_failure("timeout");
        r.record_failure("timeout");
        assert_eq!(r.error_streak, 2);
        r.record_success(150);
        assert_eq!(r.status, PeerStatus::Healthy);
        assert_eq!(r.error_streak, 0);
        assert!(r.avg_response_ms > 0.0);
    }

    #[test]
    fn test_peer_record_three_failures_marks_unhealthy() {
        let mut r = PeerRecord::new(make_peer("B", "http://b"));
        r.record_failure("e1");
        r.record_failure("e2");
        assert_eq!(r.status, PeerStatus::Unknown); // still Unknown
        r.record_failure("e3");
        assert_eq!(r.status, PeerStatus::Unhealthy);
    }

    #[test]
    fn test_peer_record_disable_enable() {
        let mut r = PeerRecord::new(make_peer("C", "http://c"));
        r.record_failure("e");
        r.record_failure("e");
        r.record_failure("e");
        assert_eq!(r.status, PeerStatus::Unhealthy);
        r.disable();
        assert_eq!(r.status, PeerStatus::Disabled);
        r.enable();
        assert_eq!(r.status, PeerStatus::Unknown);
        assert_eq!(r.error_streak, 0);
    }

    // --- FederatedQuery builder ---

    #[test]
    fn test_query_builder() {
        let q = FederatedQuery::text("nature documentary")
            .with_mime("video/*")
            .with_tag("nature")
            .sorted_by(SortOrder::DateNewest)
            .limit(50)
            .no_dedup()
            .in_regions(vec!["eu-west".to_string()]);

        assert_eq!(q.query, "nature documentary");
        assert_eq!(q.mime_filter.as_deref(), Some("video/*"));
        assert!(q.tags.contains(&"nature".to_string()));
        assert_eq!(q.sort, SortOrder::DateNewest);
        assert_eq!(q.limit_per_peer, 50);
        assert!(!q.deduplicate);
        assert!(q.region_filter.is_some());
    }

    // --- SearchHit ---

    #[test]
    fn test_hit_weighted_score() {
        let h = make_hit(uid(), "Peer A", 0.8, None);
        assert!((h.weighted_score(2.0) - 1.6).abs() < 1e-9);
    }

    // --- ResultMerger ---

    #[test]
    fn test_merge_weighted_score() {
        let p1 = uid();
        let p2 = uid();
        let r1 = success_response(p1, "P1", vec![make_hit(p1, "P1", 0.5, None)]);
        let r2 = success_response(p2, "P2", vec![make_hit(p2, "P2", 0.9, None)]);
        let weights: HashMap<Uuid, f64> = [(p1, 1.0), (p2, 1.0)].into_iter().collect();

        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::WeightedScore,
            false,
            &weights,
            100,
        );
        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.hits[0].peer_id, p2); // 0.9 > 0.5
    }

    #[test]
    fn test_merge_deduplication_by_checksum() {
        let p1 = uid();
        let p2 = uid();
        let r1 = success_response(p1, "P1", vec![make_hit(p1, "P1", 0.9, Some("abc123"))]);
        let r2 = success_response(p2, "P2", vec![make_hit(p2, "P2", 0.8, Some("abc123"))]);
        let weights = HashMap::new();

        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::WeightedScore,
            true,
            &weights,
            100,
        );
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.duplicates_removed, 1);
    }

    #[test]
    fn test_merge_no_deduplication() {
        let p1 = uid();
        let p2 = uid();
        let r1 = success_response(p1, "P1", vec![make_hit(p1, "P1", 0.9, Some("abc123"))]);
        let r2 = success_response(p2, "P2", vec![make_hit(p2, "P2", 0.8, Some("abc123"))]);
        let weights = HashMap::new();

        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::WeightedScore,
            false,
            &weights,
            100,
        );
        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.duplicates_removed, 0);
    }

    #[test]
    fn test_merge_round_robin() {
        let p1 = uid();
        let p2 = uid();
        let r1 = success_response(
            p1,
            "P1",
            vec![make_hit(p1, "P1", 1.0, None), make_hit(p1, "P1", 0.8, None)],
        );
        let r2 = success_response(p2, "P2", vec![make_hit(p2, "P2", 0.9, None)]);
        let weights = HashMap::new();
        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::RoundRobin,
            false,
            &weights,
            100,
        );
        assert_eq!(result.hits.len(), 3);
        // Round-robin alternates peers; first hit comes from whichever peer
        // happens to be first in the HashMap — just check total count
    }

    #[test]
    fn test_merge_newest_strategy() {
        let p1 = uid();
        let older = Utc::now() - chrono::Duration::days(30);
        let newer = Utc::now() - chrono::Duration::days(1);
        let mut h1 = make_hit(p1, "P1", 0.5, None);
        h1.created_at = Some(older);
        let mut h2 = make_hit(p1, "P1", 0.3, None);
        h2.created_at = Some(newer);
        let r = success_response(p1, "P1", vec![h1, h2]);
        let weights = HashMap::new();

        let result = ResultMerger::merge(vec![r], RankingStrategy::Newest, false, &weights, 100);
        // Newer should come first
        assert!(
            result.hits[0].created_at.expect("created_at should exist")
                > result.hits[1].created_at.expect("created_at should exist")
        );
    }

    #[test]
    fn test_merge_max_results_limit() {
        let p1 = uid();
        let hits: Vec<SearchHit> = (0..10)
            .map(|i| make_hit(p1, "P1", i as f64, None))
            .collect();
        let r = success_response(p1, "P1", hits);
        let weights = HashMap::new();

        let result =
            ResultMerger::merge(vec![r], RankingStrategy::WeightedScore, false, &weights, 5);
        assert_eq!(result.hits.len(), 5);
    }

    #[test]
    fn test_merge_total_count_sum() {
        let p1 = uid();
        let p2 = uid();
        let r1 = PeerQueryResponse {
            peer_id: p1,
            peer_name: "P1".to_string(),
            hits: vec![],
            total_count: 100,
            latency_ms: 50,
            success: true,
            error: None,
            facets: HashMap::new(),
        };
        let r2 = PeerQueryResponse {
            peer_id: p2,
            peer_name: "P2".to_string(),
            hits: vec![],
            total_count: 200,
            latency_ms: 80,
            success: true,
            error: None,
            facets: HashMap::new(),
        };
        let weights = HashMap::new();
        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::WeightedScore,
            false,
            &weights,
            100,
        );
        assert_eq!(result.total_count, 300);
    }

    #[test]
    fn test_facet_aggregation() {
        let p1 = uid();
        let p2 = uid();
        let mut facets1 = HashMap::new();
        facets1.insert(
            "mime_type".to_string(),
            vec![
                FacetBucket {
                    value: "video/mp4".to_string(),
                    count: 30,
                },
                FacetBucket {
                    value: "audio/wav".to_string(),
                    count: 10,
                },
            ],
        );
        let mut facets2 = HashMap::new();
        facets2.insert(
            "mime_type".to_string(),
            vec![FacetBucket {
                value: "video/mp4".to_string(),
                count: 20,
            }],
        );
        let r1 = PeerQueryResponse {
            peer_id: p1,
            peer_name: "P1".to_string(),
            hits: vec![],
            total_count: 40,
            latency_ms: 50,
            success: true,
            error: None,
            facets: facets1,
        };
        let r2 = PeerQueryResponse {
            peer_id: p2,
            peer_name: "P2".to_string(),
            hits: vec![],
            total_count: 20,
            latency_ms: 80,
            success: true,
            error: None,
            facets: facets2,
        };
        let weights = HashMap::new();
        let result = ResultMerger::merge(
            vec![r1, r2],
            RankingStrategy::WeightedScore,
            false,
            &weights,
            100,
        );

        let mime_facets = result.facets.get("mime_type").expect("mime_type facet");
        let mp4 = mime_facets
            .iter()
            .find(|b| b.value == "video/mp4")
            .expect("mp4 bucket");
        assert_eq!(mp4.count, 50); // 30 + 20
    }

    // --- PeerRegistry ---

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = PeerRegistry::new();
        reg.register(make_peer("A", "http://a"));
        reg.register(make_peer("B", "http://b"));
        assert_eq!(reg.peer_count(), 2);
    }

    #[test]
    fn test_registry_deregister() {
        let mut reg = PeerRegistry::new();
        let id = reg.register(make_peer("A", "http://a"));
        reg.deregister(id);
        assert_eq!(reg.peer_count(), 0);
    }

    #[test]
    fn test_registry_queryable_peers() {
        let mut reg = PeerRegistry::new();
        let id1 = reg.register(make_peer("A", "http://a"));
        reg.register(make_peer("B", "http://b"));
        if let Some(p) = reg.get_mut(id1) {
            p.disable();
        }
        let queryable = reg.queryable_peers();
        assert_eq!(queryable.len(), 1);
    }

    #[test]
    fn test_registry_peers_for_query_exclude() {
        let mut reg = PeerRegistry::new();
        let id1 = reg.register(make_peer("A", "http://a"));
        reg.register(make_peer("B", "http://b"));
        let mut q = FederatedQuery::text("test");
        q.exclude_peers.push(id1);
        let peers = reg.peers_for_query(&q);
        assert_eq!(peers.len(), 1);
        assert_ne!(peers[0].config.id, id1);
    }

    #[test]
    fn test_registry_peers_for_query_region() {
        let mut reg = PeerRegistry::new();
        reg.register(make_peer("A", "http://a").with_region("eu-west"));
        reg.register(make_peer("B", "http://b").with_region("us-east"));
        let q = FederatedQuery::text("test").in_regions(vec!["eu-west".to_string()]);
        let peers = reg.peers_for_query(&q);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].config.region.as_deref(), Some("eu-west"));
    }

    #[test]
    fn test_registry_weight_map() {
        let mut reg = PeerRegistry::new();
        reg.register(make_peer("A", "http://a").with_weight(2.0));
        reg.register(make_peer("B", "http://b").with_weight(0.5));
        let wm = reg.weight_map();
        let weights: Vec<f64> = wm.values().cloned().collect();
        assert!(weights.contains(&2.0));
        assert!(weights.contains(&0.5));
    }
}
