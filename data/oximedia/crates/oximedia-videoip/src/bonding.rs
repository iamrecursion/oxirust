#![allow(dead_code)]
//! Network interface bonding and aggregation for video-over-IP streams.
//!
//! This module provides link aggregation capabilities for professional video
//! streaming, allowing multiple network interfaces to be combined for:
//!
//! - **Increased bandwidth** - Aggregate throughput across multiple links
//! - **Failover resilience** - Automatic failover when a link goes down
//! - **Load balancing** - Distribute packets across healthy links
//! - **Redundancy** - Send duplicate packets on separate paths for zero-loss delivery
//!
//! # Bonding Modes
//!
//! - `RoundRobin` - Simple packet-level distribution across all links
//! - `ActiveBackup` - One active link with automatic failover
//! - `WeightedBalance` - Distribute based on link capacity weights
//! - `Broadcast` - Send on all links for maximum redundancy

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// Bonding mode for link aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondingMode {
    /// Distribute packets across links in round-robin order.
    RoundRobin,
    /// One active link, failover to backup on failure.
    ActiveBackup,
    /// Distribute proportionally by link weight/capacity.
    WeightedBalance,
    /// Send all packets on all links (maximum redundancy).
    Broadcast,
}

impl std::fmt::Display for BondingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::RoundRobin => "round_robin",
            Self::ActiveBackup => "active_backup",
            Self::WeightedBalance => "weighted_balance",
            Self::Broadcast => "broadcast",
        };
        write!(f, "{label}")
    }
}

/// Health status of a network link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkHealth {
    /// Link is healthy and active.
    Healthy,
    /// Link is degraded (high loss or latency).
    Degraded,
    /// Link is down or unreachable.
    Down,
}

impl std::fmt::Display for LinkHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Down => "down",
        };
        write!(f, "{label}")
    }
}

/// Configuration for a single network link.
#[derive(Debug, Clone)]
pub struct LinkConfig {
    /// Unique identifier for this link.
    pub id: String,
    /// Local IP address bound to this link.
    pub local_addr: IpAddr,
    /// Capacity weight (higher = more traffic routed here).
    pub weight: u32,
    /// Maximum bandwidth in bits per second.
    pub max_bandwidth_bps: u64,
    /// Health check interval.
    pub health_check_interval: Duration,
    /// Number of consecutive failed health checks before marking as down.
    pub failure_threshold: u32,
}

/// Runtime state of a single link.
#[derive(Debug, Clone)]
pub struct LinkState {
    /// Link configuration.
    pub config: LinkConfig,
    /// Current health status.
    pub health: LinkHealth,
    /// Packets sent on this link.
    pub packets_sent: u64,
    /// Bytes sent on this link.
    pub bytes_sent: u64,
    /// Packets lost on this link.
    pub packets_lost: u64,
    /// Current estimated RTT.
    pub rtt: Duration,
    /// Last health check time.
    pub last_health_check: Option<Instant>,
    /// Consecutive health check failures.
    pub consecutive_failures: u32,
}

impl LinkState {
    /// Create a new link state from configuration.
    fn new(config: LinkConfig) -> Self {
        Self {
            config,
            health: LinkHealth::Healthy,
            packets_sent: 0,
            bytes_sent: 0,
            packets_lost: 0,
            rtt: Duration::from_millis(1),
            last_health_check: None,
            consecutive_failures: 0,
        }
    }

    /// Compute the loss rate for this link.
    #[allow(clippy::cast_precision_loss)]
    fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_lost as f64 / self.packets_sent as f64
    }
}

/// Configuration for the bonding group.
#[derive(Debug, Clone)]
pub struct BondingConfig {
    /// Bonding mode.
    pub mode: BondingMode,
    /// Loss rate threshold for marking a link as degraded (0.0-1.0).
    pub degraded_loss_threshold: f64,
    /// Loss rate threshold for marking a link as down (0.0-1.0).
    pub down_loss_threshold: f64,
    /// RTT threshold for marking a link as degraded.
    pub degraded_rtt_threshold: Duration,
    /// Minimum number of healthy links required.
    pub min_healthy_links: usize,
}

impl Default for BondingConfig {
    fn default() -> Self {
        Self {
            mode: BondingMode::WeightedBalance,
            degraded_loss_threshold: 0.01,
            down_loss_threshold: 0.10,
            degraded_rtt_threshold: Duration::from_millis(50),
            min_healthy_links: 1,
        }
    }
}

/// Routing decision for a packet.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// Link IDs to send this packet on.
    pub link_ids: Vec<String>,
    /// Whether this is a redundant transmission (same packet on multiple links).
    pub is_redundant: bool,
}

/// Statistics for the bonding group.
#[derive(Debug, Clone)]
pub struct BondingStats {
    /// Total links in the bond.
    pub total_links: usize,
    /// Number of healthy links.
    pub healthy_links: usize,
    /// Number of degraded links.
    pub degraded_links: usize,
    /// Number of down links.
    pub down_links: usize,
    /// Aggregate bandwidth available (sum of healthy link capacities).
    pub aggregate_bandwidth_bps: u64,
    /// Total packets routed.
    pub total_packets_routed: u64,
    /// Current bonding mode.
    pub mode: BondingMode,
}

/// Network interface bonding group.
pub struct BondingGroup {
    /// Configuration.
    config: BondingConfig,
    /// Link states by ID.
    links: HashMap<String, LinkState>,
    /// Ordered link IDs for round-robin.
    link_order: Vec<String>,
    /// Current round-robin index.
    rr_index: usize,
    /// Active link ID (for active-backup mode).
    active_link: Option<String>,
    /// Total packets routed.
    total_routed: u64,
}

impl BondingGroup {
    /// Create a new bonding group.
    #[must_use]
    pub fn new(config: BondingConfig) -> Self {
        Self {
            config,
            links: HashMap::new(),
            link_order: Vec::new(),
            rr_index: 0,
            active_link: None,
            total_routed: 0,
        }
    }

    /// Add a link to the bonding group.
    pub fn add_link(&mut self, config: LinkConfig) {
        let id = config.id.clone();
        self.links.insert(id.clone(), LinkState::new(config));
        self.link_order.push(id.clone());
        if self.active_link.is_none() {
            self.active_link = Some(id);
        }
    }

    /// Remove a link from the bonding group.
    pub fn remove_link(&mut self, link_id: &str) -> bool {
        if self.links.remove(link_id).is_some() {
            self.link_order.retain(|id| id != link_id);
            if self.active_link.as_deref() == Some(link_id) {
                self.active_link = self.find_healthy_link();
            }
            true
        } else {
            false
        }
    }

    /// Report link health metrics.
    pub fn report_link_health(
        &mut self,
        link_id: &str,
        rtt: Duration,
        packets_sent: u64,
        packets_lost: u64,
    ) {
        if let Some(link) = self.links.get_mut(link_id) {
            link.rtt = rtt;
            link.packets_sent += packets_sent;
            link.packets_lost += packets_lost;
            link.last_health_check = Some(Instant::now());

            // Update health status
            let loss = link.loss_rate();
            if loss >= self.config.down_loss_threshold {
                link.consecutive_failures += 1;
                if link.consecutive_failures >= link.config.failure_threshold {
                    link.health = LinkHealth::Down;
                }
            } else if loss >= self.config.degraded_loss_threshold
                || rtt > self.config.degraded_rtt_threshold
            {
                link.health = LinkHealth::Degraded;
                link.consecutive_failures = 0;
            } else {
                link.health = LinkHealth::Healthy;
                link.consecutive_failures = 0;
            }

            // Update active link if current one went down (active-backup mode)
            if self.config.mode == BondingMode::ActiveBackup {
                if let Some(active_id) = &self.active_link {
                    if let Some(active) = self.links.get(active_id) {
                        if active.health == LinkHealth::Down {
                            self.active_link = self.find_healthy_link();
                        }
                    }
                }
            }
        }
    }

    /// Find a healthy link (preferring the one with lowest RTT).
    fn find_healthy_link(&self) -> Option<String> {
        self.links
            .iter()
            .filter(|(_, state)| state.health == LinkHealth::Healthy)
            .min_by_key(|(_, state)| state.rtt)
            .map(|(id, _)| id.clone())
    }

    /// Route a packet according to the bonding policy.
    pub fn route_packet(&mut self, _packet_size: usize) -> RouteDecision {
        self.total_routed += 1;

        match self.config.mode {
            BondingMode::RoundRobin => self.route_round_robin(),
            BondingMode::ActiveBackup => self.route_active_backup(),
            BondingMode::WeightedBalance => self.route_weighted(),
            BondingMode::Broadcast => self.route_broadcast(),
        }
    }

    /// Round-robin routing.
    fn route_round_robin(&mut self) -> RouteDecision {
        let healthy: Vec<String> = self
            .link_order
            .iter()
            .filter(|id| {
                self.links
                    .get(*id)
                    .is_some_and(|s| s.health != LinkHealth::Down)
            })
            .cloned()
            .collect();

        if healthy.is_empty() {
            return RouteDecision {
                link_ids: Vec::new(),
                is_redundant: false,
            };
        }

        let idx = self.rr_index % healthy.len();
        self.rr_index = self.rr_index.wrapping_add(1);

        RouteDecision {
            link_ids: vec![healthy[idx].clone()],
            is_redundant: false,
        }
    }

    /// Active-backup routing.
    fn route_active_backup(&self) -> RouteDecision {
        let link_id = self
            .active_link
            .clone()
            .or_else(|| self.find_healthy_link());

        RouteDecision {
            link_ids: link_id.into_iter().collect(),
            is_redundant: false,
        }
    }

    /// Weighted balance routing.
    #[allow(clippy::cast_precision_loss)]
    fn route_weighted(&mut self) -> RouteDecision {
        let healthy: Vec<(String, u32)> = self
            .links
            .iter()
            .filter(|(_, state)| state.health != LinkHealth::Down)
            .map(|(id, state)| (id.clone(), state.config.weight))
            .collect();

        if healthy.is_empty() {
            return RouteDecision {
                link_ids: Vec::new(),
                is_redundant: false,
            };
        }

        let total_weight: u32 = healthy.iter().map(|(_, w)| w).sum();
        if total_weight == 0 {
            return self.route_round_robin();
        }

        // Use packet counter as deterministic selection
        let slot = (self.total_routed % u64::from(total_weight)) as u32;
        let mut cumulative = 0u32;
        for (id, weight) in &healthy {
            cumulative += weight;
            if slot < cumulative {
                return RouteDecision {
                    link_ids: vec![id.clone()],
                    is_redundant: false,
                };
            }
        }

        // Fallback to first healthy
        RouteDecision {
            link_ids: vec![healthy[0].0.clone()],
            is_redundant: false,
        }
    }

    /// Broadcast routing (send on all non-down links).
    fn route_broadcast(&self) -> RouteDecision {
        let ids: Vec<String> = self
            .links
            .iter()
            .filter(|(_, state)| state.health != LinkHealth::Down)
            .map(|(id, _)| id.clone())
            .collect();

        RouteDecision {
            link_ids: ids,
            is_redundant: true,
        }
    }

    /// Get bonding group statistics.
    #[must_use]
    pub fn stats(&self) -> BondingStats {
        let healthy = self
            .links
            .values()
            .filter(|s| s.health == LinkHealth::Healthy)
            .count();
        let degraded = self
            .links
            .values()
            .filter(|s| s.health == LinkHealth::Degraded)
            .count();
        let down = self
            .links
            .values()
            .filter(|s| s.health == LinkHealth::Down)
            .count();
        let agg_bw: u64 = self
            .links
            .values()
            .filter(|s| s.health != LinkHealth::Down)
            .map(|s| s.config.max_bandwidth_bps)
            .sum();

        BondingStats {
            total_links: self.links.len(),
            healthy_links: healthy,
            degraded_links: degraded,
            down_links: down,
            aggregate_bandwidth_bps: agg_bw,
            total_packets_routed: self.total_routed,
            mode: self.config.mode,
        }
    }

    /// Get the number of links.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Check if the minimum healthy links requirement is met.
    #[must_use]
    pub fn is_operational(&self) -> bool {
        let healthy = self
            .links
            .values()
            .filter(|s| s.health == LinkHealth::Healthy)
            .count();
        healthy >= self.config.min_healthy_links
    }
}

/// Helper to create a standard link configuration.
#[must_use]
pub fn make_link_config(id: &str, addr: IpAddr, bandwidth_mbps: u64, weight: u32) -> LinkConfig {
    LinkConfig {
        id: id.to_string(),
        local_addr: addr,
        weight,
        max_bandwidth_bps: bandwidth_mbps * 1_000_000,
        health_check_interval: Duration::from_secs(1),
        failure_threshold: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn test_addr(last_octet: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, last_octet))
    }

    fn make_test_bond(mode: BondingMode) -> BondingGroup {
        let config = BondingConfig {
            mode,
            ..Default::default()
        };
        let mut group = BondingGroup::new(config);
        group.add_link(make_link_config("eth0", test_addr(1), 1000, 3));
        group.add_link(make_link_config("eth1", test_addr(2), 1000, 1));
        group
    }

    #[test]
    fn test_add_remove_links() {
        let mut group = make_test_bond(BondingMode::RoundRobin);
        assert_eq!(group.link_count(), 2);
        group.remove_link("eth0");
        assert_eq!(group.link_count(), 1);
        assert!(!group.remove_link("nonexistent"));
    }

    #[test]
    fn test_round_robin_distribution() {
        let mut group = make_test_bond(BondingMode::RoundRobin);
        let d1 = group.route_packet(1000);
        let d2 = group.route_packet(1000);
        assert_eq!(d1.link_ids.len(), 1);
        assert_eq!(d2.link_ids.len(), 1);
        // Should cycle between the two links
        assert_ne!(d1.link_ids[0], d2.link_ids[0]);
    }

    #[test]
    fn test_active_backup_single_link() {
        let mut group = make_test_bond(BondingMode::ActiveBackup);
        let d1 = group.route_packet(1000);
        let d2 = group.route_packet(1000);
        assert_eq!(
            d1.link_ids, d2.link_ids,
            "active-backup should use same link"
        );
    }

    #[test]
    fn test_broadcast_all_links() {
        let mut group = make_test_bond(BondingMode::Broadcast);
        let decision = group.route_packet(1000);
        assert_eq!(decision.link_ids.len(), 2);
        assert!(decision.is_redundant);
    }

    #[test]
    fn test_weighted_balance_respects_weight() {
        let mut group = make_test_bond(BondingMode::WeightedBalance);
        let mut eth0_count = 0u64;
        let mut eth1_count = 0u64;

        for _ in 0..400 {
            let d = group.route_packet(1000);
            if d.link_ids.first().map(|s| s.as_str()) == Some("eth0") {
                eth0_count += 1;
            } else {
                eth1_count += 1;
            }
        }
        // eth0 has weight 3, eth1 has weight 1 => expect ~75%/25% split
        assert!(
            eth0_count > eth1_count,
            "eth0 (weight 3) should get more traffic: {} vs {}",
            eth0_count,
            eth1_count
        );
    }

    #[test]
    fn test_link_health_degrades() {
        let mut group = make_test_bond(BondingMode::RoundRobin);
        // Report high loss for eth0
        group.report_link_health("eth0", Duration::from_millis(5), 100, 5);
        let state = group.links.get("eth0").expect("should succeed in test");
        assert_eq!(state.health, LinkHealth::Degraded);
    }

    #[test]
    fn test_link_goes_down() {
        let mut group = make_test_bond(BondingMode::RoundRobin);
        // Report very high loss repeatedly
        for _ in 0..3 {
            group.report_link_health("eth0", Duration::from_millis(5), 100, 50);
        }
        let state = group.links.get("eth0").expect("should succeed in test");
        assert_eq!(state.health, LinkHealth::Down);
    }

    #[test]
    fn test_active_backup_failover() {
        let mut group = make_test_bond(BondingMode::ActiveBackup);
        let initial = group.active_link.clone().expect("should succeed in test");

        // Kill the active link
        for _ in 0..3 {
            group.report_link_health(&initial, Duration::from_millis(5), 100, 50);
        }

        let d = group.route_packet(1000);
        assert_eq!(d.link_ids.len(), 1);
        assert_ne!(d.link_ids[0], initial, "should failover to backup");
    }

    #[test]
    fn test_bonding_stats() {
        let mut group = make_test_bond(BondingMode::RoundRobin);
        group.route_packet(1000);
        group.route_packet(1000);
        let stats = group.stats();
        assert_eq!(stats.total_links, 2);
        assert_eq!(stats.healthy_links, 2);
        assert_eq!(stats.total_packets_routed, 2);
        assert_eq!(stats.aggregate_bandwidth_bps, 2_000_000_000);
    }

    #[test]
    fn test_is_operational() {
        let config = BondingConfig {
            mode: BondingMode::RoundRobin,
            min_healthy_links: 2,
            ..Default::default()
        };
        let mut group = BondingGroup::new(config);
        group.add_link(make_link_config("eth0", test_addr(1), 1000, 1));
        group.add_link(make_link_config("eth1", test_addr(2), 1000, 1));
        assert!(group.is_operational());

        // Kill one link
        for _ in 0..3 {
            group.report_link_health("eth0", Duration::from_millis(5), 100, 50);
        }
        assert!(!group.is_operational());
    }

    #[test]
    fn test_bonding_mode_display() {
        assert_eq!(format!("{}", BondingMode::RoundRobin), "round_robin");
        assert_eq!(format!("{}", BondingMode::ActiveBackup), "active_backup");
        assert_eq!(
            format!("{}", BondingMode::WeightedBalance),
            "weighted_balance"
        );
        assert_eq!(format!("{}", BondingMode::Broadcast), "broadcast");
    }

    #[test]
    fn test_link_health_display() {
        assert_eq!(format!("{}", LinkHealth::Healthy), "healthy");
        assert_eq!(format!("{}", LinkHealth::Degraded), "degraded");
        assert_eq!(format!("{}", LinkHealth::Down), "down");
    }

    #[test]
    fn test_empty_group_route() {
        let config = BondingConfig::default();
        let mut group = BondingGroup::new(config);
        let d = group.route_packet(1000);
        assert!(d.link_ids.is_empty());
    }
}
