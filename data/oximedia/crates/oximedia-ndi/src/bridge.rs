//! NDI bridge mode for cross-subnet discovery and streaming.
//!
//! In NDI, mDNS-based discovery is confined to a single Layer-2 broadcast
//! domain.  An NDI *bridge* extends discovery and relaying across subnet
//! boundaries by:
//!
//! 1. **Source registration** — each subnet peer registers known NDI sources
//!    (name + address) with the bridge.
//! 2. **Route table** — the bridge maintains a registry that maps source names
//!    to their home subnets and reachable relay endpoints.
//! 3. **Relay selection** — when a receiver on subnet A asks for a source that
//!    lives on subnet B, the bridge selects the optimal relay path.
//!
//! This module is a pure data-model and logic layer; it does not open network
//! sockets itself (the sender/receiver modules own those).

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// SubnetId
// ---------------------------------------------------------------------------

/// Identifies a network subnet (CIDR prefix as a string, e.g. "192.168.1.0/24").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SubnetId(pub String);

impl SubnetId {
    /// Create a new subnet identifier from a CIDR string.
    pub fn new(cidr: impl Into<String>) -> Self {
        Self(cidr.into())
    }
}

impl std::fmt::Display for SubnetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// BridgeSourceEntry
// ---------------------------------------------------------------------------

/// A single NDI source entry known to the bridge.
#[derive(Debug, Clone)]
pub struct BridgeSourceEntry {
    /// Source name as advertised via mDNS.
    pub name: String,
    /// The actual endpoint (IP:port) of the NDI sender.
    pub endpoint: SocketAddr,
    /// Subnet this source lives in.
    pub subnet: SubnetId,
    /// When this entry was last refreshed.
    pub last_seen: Instant,
    /// Round-trip latency to this source (if probed).
    pub rtt_ms: Option<f64>,
}

impl BridgeSourceEntry {
    /// Create a new bridge source entry.
    pub fn new(name: String, endpoint: SocketAddr, subnet: SubnetId) -> Self {
        Self {
            name,
            endpoint,
            subnet,
            last_seen: Instant::now(),
            rtt_ms: None,
        }
    }

    /// Refresh the last-seen timestamp.
    pub fn touch(&mut self) {
        self.last_seen = Instant::now();
    }

    /// Returns `true` if the entry has not been refreshed within `ttl`.
    pub fn is_stale(&self, ttl: Duration) -> bool {
        self.last_seen.elapsed() > ttl
    }
}

// ---------------------------------------------------------------------------
// RelayEndpoint
// ---------------------------------------------------------------------------

/// A relay node that forwards NDI traffic between subnets.
#[derive(Debug, Clone)]
pub struct RelayEndpoint {
    /// Relay node identifier / hostname.
    pub id: String,
    /// Address reachable from the requesting subnet.
    pub address: SocketAddr,
    /// Subnets this relay can forward to.
    pub reachable_subnets: Vec<SubnetId>,
    /// Measured latency through this relay in milliseconds.
    pub latency_ms: f64,
}

impl RelayEndpoint {
    /// Create a new relay endpoint.
    pub fn new(
        id: impl Into<String>,
        address: SocketAddr,
        reachable_subnets: Vec<SubnetId>,
        latency_ms: f64,
    ) -> Self {
        Self {
            id: id.into(),
            address,
            reachable_subnets,
            latency_ms,
        }
    }

    /// Returns `true` if this relay can reach the specified subnet.
    pub fn can_reach(&self, subnet: &SubnetId) -> bool {
        self.reachable_subnets.contains(subnet)
    }
}

// ---------------------------------------------------------------------------
// BridgeRouteTable
// ---------------------------------------------------------------------------

/// Maintains a registry of cross-subnet NDI sources and relay endpoints.
///
/// Sources are keyed by their NDI source name.  Multiple sources can share the
/// same name (on different subnets) — queries return all candidates sorted by
/// latency.
#[derive(Debug, Default)]
pub struct BridgeRouteTable {
    /// Source entries, keyed by name.
    sources: HashMap<String, Vec<BridgeSourceEntry>>,
    /// Known relay endpoints.
    relays: Vec<RelayEndpoint>,
    /// Stale entry TTL.
    ttl: Duration,
}

impl BridgeRouteTable {
    /// Create a new bridge route table with the given entry TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            sources: HashMap::new(),
            relays: Vec::new(),
            ttl,
        }
    }

    /// Register or refresh an NDI source in the route table.
    pub fn register_source(&mut self, entry: BridgeSourceEntry) {
        let entries = self.sources.entry(entry.name.clone()).or_default();
        // Update existing entry from same subnet or add new
        if let Some(existing) = entries.iter_mut().find(|e| e.subnet == entry.subnet) {
            existing.endpoint = entry.endpoint;
            existing.rtt_ms = entry.rtt_ms;
            existing.touch();
        } else {
            entries.push(entry);
        }
    }

    /// Remove a source registration.
    pub fn remove_source(&mut self, name: &str, subnet: &SubnetId) {
        if let Some(entries) = self.sources.get_mut(name) {
            entries.retain(|e| &e.subnet != subnet);
            if entries.is_empty() {
                self.sources.remove(name);
            }
        }
    }

    /// Look up all known endpoints for a source name.
    ///
    /// Results are sorted by RTT (lowest first); entries without RTT are last.
    pub fn lookup(&self, name: &str) -> Vec<&BridgeSourceEntry> {
        let mut candidates: Vec<&BridgeSourceEntry> = self
            .sources
            .get(name)
            .map(|v| v.iter().collect())
            .unwrap_or_default();

        candidates.sort_by(|a, b| {
            let ra = a.rtt_ms.unwrap_or(f64::MAX);
            let rb = b.rtt_ms.unwrap_or(f64::MAX);
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Return all source names registered in the table.
    pub fn all_source_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    /// Evict stale entries (those not refreshed within `self.ttl`).
    pub fn evict_stale(&mut self) {
        let ttl = self.ttl;
        for entries in self.sources.values_mut() {
            entries.retain(|e| !e.is_stale(ttl));
        }
        self.sources.retain(|_, v| !v.is_empty());
    }

    /// Register a relay endpoint.
    pub fn register_relay(&mut self, relay: RelayEndpoint) {
        if let Some(existing) = self.relays.iter_mut().find(|r| r.id == relay.id) {
            *existing = relay;
        } else {
            self.relays.push(relay);
        }
    }

    /// Remove a relay by ID.
    pub fn remove_relay(&mut self, relay_id: &str) {
        self.relays.retain(|r| r.id != relay_id);
    }

    /// Find the best relay that can reach `target_subnet`, sorted by latency.
    pub fn best_relay_for(&self, target_subnet: &SubnetId) -> Option<&RelayEndpoint> {
        let mut candidates: Vec<&RelayEndpoint> = self
            .relays
            .iter()
            .filter(|r| r.can_reach(target_subnet))
            .collect();
        candidates.sort_by(|a, b| {
            a.latency_ms
                .partial_cmp(&b.latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.into_iter().next()
    }

    /// Total number of registered sources (across all subnets).
    pub fn source_count(&self) -> usize {
        self.sources.values().map(|v| v.len()).sum()
    }

    /// Total number of registered relay endpoints.
    pub fn relay_count(&self) -> usize {
        self.relays.len()
    }
}

// ---------------------------------------------------------------------------
// BridgeConfig
// ---------------------------------------------------------------------------

/// Configuration for an NDI bridge node.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// This bridge node's local address.
    pub local_address: IpAddr,
    /// Port used for bridge control/registration traffic.
    pub control_port: u16,
    /// TTL for source registrations.
    pub source_ttl: Duration,
    /// Maximum number of simultaneous relayed streams.
    pub max_relay_streams: usize,
    /// Whether to allow unauthenticated source registrations.
    pub allow_anonymous: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            local_address: "0.0.0.0".parse().expect("valid default IP"),
            control_port: 5990,
            source_ttl: Duration::from_secs(30),
            max_relay_streams: 64,
            allow_anonymous: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CrossSubnetPath — resolved route for a source request
// ---------------------------------------------------------------------------

/// Describes the resolved forwarding path to reach an NDI source from a
/// requester on a different subnet.
#[derive(Debug, Clone)]
pub struct CrossSubnetPath {
    /// Source name requested.
    pub source_name: String,
    /// Direct endpoint of the source (on its home subnet).
    pub source_endpoint: SocketAddr,
    /// Relay to use, if the source is not directly reachable.
    pub relay: Option<SocketAddr>,
    /// Estimated end-to-end latency in milliseconds.
    pub estimated_latency_ms: f64,
}

impl CrossSubnetPath {
    /// Returns `true` if the path goes through a relay.
    pub fn uses_relay(&self) -> bool {
        self.relay.is_some()
    }
}

/// Resolve the best cross-subnet path for `source_name` as seen from
/// `requester_subnet`.
///
/// Returns `None` if the source is not known or no path exists.
pub fn resolve_path(
    table: &BridgeRouteTable,
    source_name: &str,
    requester_subnet: &SubnetId,
) -> Option<CrossSubnetPath> {
    let candidates = table.lookup(source_name);

    // Try direct (same-subnet) first
    if let Some(direct) = candidates.iter().find(|e| &e.subnet == requester_subnet) {
        return Some(CrossSubnetPath {
            source_name: source_name.to_string(),
            source_endpoint: direct.endpoint,
            relay: None,
            estimated_latency_ms: direct.rtt_ms.unwrap_or(0.0),
        });
    }

    // Try via relay
    for candidate in &candidates {
        if let Some(relay) = table.best_relay_for(&candidate.subnet) {
            let relay_latency = relay.latency_ms;
            let source_latency = candidate.rtt_ms.unwrap_or(0.0);
            return Some(CrossSubnetPath {
                source_name: source_name.to_string(),
                source_endpoint: candidate.endpoint,
                relay: Some(relay.address),
                estimated_latency_ms: relay_latency + source_latency,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{}", port).parse().expect("valid addr")
    }

    fn subnet(s: &str) -> SubnetId {
        SubnetId::new(s)
    }

    fn make_source(name: &str, port: u16, s: &str) -> BridgeSourceEntry {
        let mut e = BridgeSourceEntry::new(name.to_string(), make_addr(port), subnet(s));
        e.rtt_ms = Some(10.0);
        e
    }

    #[test]
    fn test_subnet_id_display() {
        let s = SubnetId::new("10.0.0.0/24");
        assert_eq!(s.to_string(), "10.0.0.0/24");
    }

    #[test]
    fn test_register_and_lookup() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("CAM1", 5960, "192.168.1.0/24"));
        let results = table.lookup("CAM1");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "CAM1");
    }

    #[test]
    fn test_lookup_multiple_subnets() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        let mut e1 = make_source("CAM1", 5960, "10.0.0.0/24");
        e1.rtt_ms = Some(5.0);
        let mut e2 = make_source("CAM1", 5961, "10.1.0.0/24");
        e2.rtt_ms = Some(20.0);
        table.register_source(e1);
        table.register_source(e2);

        let results = table.lookup("CAM1");
        assert_eq!(results.len(), 2);
        // Sorted by RTT: 5ms first
        assert!(
            results[0].rtt_ms.expect("first result should have RTT")
                < results[1].rtt_ms.expect("second result should have RTT")
        );
    }

    #[test]
    fn test_remove_source() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("SRC", 5960, "10.0.0.0/24"));
        table.remove_source("SRC", &subnet("10.0.0.0/24"));
        assert!(table.lookup("SRC").is_empty());
    }

    #[test]
    fn test_register_relay() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        let relay =
            RelayEndpoint::new("relay1", make_addr(6000), vec![subnet("10.1.0.0/24")], 15.0);
        table.register_relay(relay);
        assert_eq!(table.relay_count(), 1);

        let best = table.best_relay_for(&subnet("10.1.0.0/24"));
        assert!(best.is_some());
        assert_eq!(
            best.expect("should find relay for matching subnet").id,
            "relay1"
        );
    }

    #[test]
    fn test_best_relay_sorts_by_latency() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        let sn = subnet("10.2.0.0/24");
        let relay_fast = RelayEndpoint::new("fast", make_addr(6001), vec![sn.clone()], 5.0);
        let relay_slow = RelayEndpoint::new("slow", make_addr(6002), vec![sn.clone()], 30.0);
        table.register_relay(relay_slow);
        table.register_relay(relay_fast);

        let best = table.best_relay_for(&sn).expect("should find relay");
        assert_eq!(best.id, "fast");
    }

    #[test]
    fn test_best_relay_no_match() {
        let table = BridgeRouteTable::new(Duration::from_secs(60));
        assert!(table.best_relay_for(&subnet("99.0.0.0/8")).is_none());
    }

    #[test]
    fn test_source_count() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("A", 5960, "10.0.0.0/24"));
        table.register_source(make_source("B", 5961, "10.0.0.0/24"));
        assert_eq!(table.source_count(), 2);
    }

    #[test]
    fn test_all_source_names() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("X", 5960, "10.0.0.0/24"));
        table.register_source(make_source("Y", 5961, "10.0.0.0/24"));
        let mut names = table.all_source_names();
        names.sort();
        assert_eq!(names, vec!["X", "Y"]);
    }

    #[test]
    fn test_resolve_path_direct() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("CAM", 5960, "10.0.0.0/24"));

        let path =
            resolve_path(&table, "CAM", &subnet("10.0.0.0/24")).expect("should resolve directly");
        assert!(!path.uses_relay());
        assert_eq!(path.source_name, "CAM");
    }

    #[test]
    fn test_resolve_path_via_relay() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        table.register_source(make_source("REMOTE", 5960, "192.168.2.0/24"));
        let relay = RelayEndpoint::new("r1", make_addr(7000), vec![subnet("192.168.2.0/24")], 8.0);
        table.register_relay(relay);

        let path = resolve_path(&table, "REMOTE", &subnet("10.0.0.0/24"))
            .expect("path via relay should resolve");
        assert!(path.uses_relay());
        assert_eq!(
            path.relay
                .expect("relay path should have relay address")
                .port(),
            7000
        );
    }

    #[test]
    fn test_resolve_path_not_found() {
        let table = BridgeRouteTable::new(Duration::from_secs(60));
        assert!(resolve_path(&table, "GHOST", &subnet("10.0.0.0/24")).is_none());
    }

    #[test]
    fn test_relay_endpoint_can_reach() {
        let relay = RelayEndpoint::new("r", make_addr(9000), vec![subnet("172.16.0.0/16")], 5.0);
        assert!(relay.can_reach(&subnet("172.16.0.0/16")));
        assert!(!relay.can_reach(&subnet("10.0.0.0/8")));
    }

    #[test]
    fn test_bridge_config_default() {
        let cfg = BridgeConfig::default();
        assert_eq!(cfg.control_port, 5990);
        assert_eq!(cfg.max_relay_streams, 64);
        assert!(cfg.allow_anonymous);
    }

    #[test]
    fn test_update_relay_replaces_existing() {
        let mut table = BridgeRouteTable::new(Duration::from_secs(60));
        let r1 = RelayEndpoint::new("r1", make_addr(6000), vec![], 100.0);
        let r2 = RelayEndpoint::new("r1", make_addr(6001), vec![], 5.0);
        table.register_relay(r1);
        table.register_relay(r2);
        assert_eq!(table.relay_count(), 1);
        // Updated latency should be 5.0
        assert!((table.relays[0].latency_ms - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_source_entry_stale() {
        let e = BridgeSourceEntry::new("S".to_string(), make_addr(5960), subnet("10.0.0.0/24"));
        assert!(!e.is_stale(Duration::from_secs(10)));
        // With a tiny TTL it should be stale almost immediately on slow machines —
        // we use 0ns to guarantee it.
        assert!(e.is_stale(Duration::ZERO));
    }
}
