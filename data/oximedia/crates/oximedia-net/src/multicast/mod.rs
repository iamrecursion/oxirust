//! IP multicast management.
//!
//! This module provides types and utilities for IP multicast group management,
//! including IGMPv1/v2/v3 and Source-Specific Multicast (SSM).

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

/// A multicast group address with optional source and TTL.
#[derive(Debug, Clone)]
pub struct MulticastGroup {
    /// Multicast group IP address (e.g. "239.255.0.1").
    pub address: String,
    /// UDP port for this group.
    pub port: u16,
    /// Optional source address for Source-Specific Multicast (IGMPv3).
    pub source: Option<String>,
    /// IP TTL / hop limit for multicast packets.
    pub ttl: u8,
}

impl MulticastGroup {
    /// Creates a new multicast group.
    #[must_use]
    pub fn new(address: impl Into<String>, port: u16) -> Self {
        Self {
            address: address.into(),
            port,
            source: None,
            ttl: 1,
        }
    }

    /// Sets the source address (enabling SSM mode).
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Sets the TTL.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: u8) -> Self {
        self.ttl = ttl;
        self
    }

    /// Returns `true` if the given IP address string falls within the
    /// IPv4 multicast range `224.0.0.0/4` (224.0.0.0 – 239.255.255.255).
    ///
    /// Only IPv4 multicast addresses are validated; IPv6 (`ff00::/8`) is not
    /// covered by this helper.
    #[must_use]
    pub fn is_valid_address(addr: &str) -> bool {
        // Parse as IPv4 dotted-decimal.
        let octets: Vec<&str> = addr.split('.').collect();
        if octets.len() != 4 {
            return false;
        }
        if let Ok(first) = octets[0].parse::<u8>() {
            // 224.x.x.x – 239.x.x.x  → first octet & 0xF0 == 0xE0
            return first >= 224 && first <= 239;
        }
        false
    }
}

/// IGMP version used when joining a multicast group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgmpVersion {
    /// IGMPv1 – basic join/leave.
    V1,
    /// IGMPv2 – adds leave group message.
    V2,
    /// IGMPv3 – adds source-specific multicast.
    V3,
}

/// A request to join a multicast group on a specific network interface.
#[derive(Debug, Clone)]
pub struct MulticastJoinRequest {
    /// The multicast group to join.
    pub group: MulticastGroup,
    /// Network interface name (e.g. "eth0").
    pub interface: String,
    /// IGMP version to use.
    pub igmp_version: IgmpVersion,
}

impl MulticastJoinRequest {
    /// Creates a new join request.
    #[must_use]
    pub fn new(
        group: MulticastGroup,
        interface: impl Into<String>,
        igmp_version: IgmpVersion,
    ) -> Self {
        Self {
            group,
            interface: interface.into(),
            igmp_version,
        }
    }
}

/// A Source-Specific Multicast (SSM) channel (RFC 4607).
///
/// Combines a source unicast IP with a multicast group address.
#[derive(Debug, Clone)]
pub struct SsmChannel {
    /// Source unicast IP address.
    pub source_ip: String,
    /// Multicast group IP address.
    pub group_ip: String,
    /// UDP port.
    pub port: u16,
}

impl SsmChannel {
    /// Creates a new SSM channel.
    #[must_use]
    pub fn new(source_ip: impl Into<String>, group_ip: impl Into<String>, port: u16) -> Self {
        Self {
            source_ip: source_ip.into(),
            group_ip: group_ip.into(),
            port,
        }
    }

    /// Returns `true` if the group address is a valid multicast address.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        MulticastGroup::is_valid_address(&self.group_ip)
    }
}

/// Per-group reception statistics.
#[derive(Debug, Clone, Default)]
pub struct MulticastStats {
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Number of times the group was joined.
    pub join_count: u32,
    /// Millisecond timestamp of the last received packet (from epoch).
    pub last_packet_ms: u64,
}

/// Tracks joined multicast groups and provides simple management.
#[derive(Debug, Default)]
pub struct MulticastManager {
    active: Vec<(MulticastGroup, MulticastStats)>,
}

impl MulticastManager {
    /// Creates a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Joins a multicast group.
    ///
    /// Returns `true` if the address is valid and the join was recorded.
    /// Returns `false` if the address is not a valid IPv4 multicast address.
    pub fn join(&mut self, group: MulticastGroup) -> bool {
        if !MulticastGroup::is_valid_address(&group.address) {
            return false;
        }

        // Update stats if already joined.
        if let Some((_g, stats)) = self
            .active
            .iter_mut()
            .find(|(g, _)| g.address == group.address && g.port == group.port)
        {
            stats.join_count += 1;
            return true;
        }

        let mut stats = MulticastStats::default();
        stats.join_count = 1;
        self.active.push((group, stats));
        true
    }

    /// Leaves a multicast group.
    pub fn leave(&mut self, group: &MulticastGroup) {
        self.active
            .retain(|(g, _)| !(g.address == group.address && g.port == group.port));
    }

    /// Returns references to all currently active groups.
    #[must_use]
    pub fn active_groups(&self) -> Vec<&MulticastGroup> {
        self.active.iter().map(|(g, _)| g).collect()
    }

    /// Returns the statistics for a group, if present.
    #[must_use]
    pub fn stats_for(&self, address: &str, port: u16) -> Option<&MulticastStats> {
        self.active
            .iter()
            .find(|(g, _)| g.address == address && g.port == port)
            .map(|(_, s)| s)
    }

    /// Records a received packet for a group (for testing / internal simulation).
    pub fn record_packet(&mut self, address: &str, port: u16, bytes: u64, now_ms: u64) {
        if let Some((_g, stats)) = self
            .active
            .iter_mut()
            .find(|(g, _)| g.address == address && g.port == port)
        {
            stats.packets_received += 1;
            stats.bytes_received += bytes;
            stats.last_packet_ms = now_ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_address_valid() {
        assert!(MulticastGroup::is_valid_address("224.0.0.1"));
        assert!(MulticastGroup::is_valid_address("239.255.255.255"));
        assert!(MulticastGroup::is_valid_address("232.1.2.3"));
    }

    #[test]
    fn test_is_valid_address_invalid() {
        assert!(!MulticastGroup::is_valid_address("192.168.1.1"));
        assert!(!MulticastGroup::is_valid_address("10.0.0.1"));
        assert!(!MulticastGroup::is_valid_address("255.255.255.255"));
        assert!(!MulticastGroup::is_valid_address("0.0.0.0"));
        assert!(!MulticastGroup::is_valid_address("not.an.ip"));
        assert!(!MulticastGroup::is_valid_address("223.255.255.255")); // just below range
    }

    #[test]
    fn test_multicast_group_new() {
        let g = MulticastGroup::new("239.255.0.1", 5004);
        assert_eq!(g.address, "239.255.0.1");
        assert_eq!(g.port, 5004);
        assert_eq!(g.ttl, 1);
        assert!(g.source.is_none());
    }

    #[test]
    fn test_multicast_group_with_source() {
        let g = MulticastGroup::new("239.255.0.1", 5004)
            .with_source("10.0.0.1")
            .with_ttl(8);
        assert_eq!(g.source, Some("10.0.0.1".to_string()));
        assert_eq!(g.ttl, 8);
    }

    #[test]
    fn test_igmp_version_variants() {
        assert_ne!(IgmpVersion::V1, IgmpVersion::V2);
        assert_ne!(IgmpVersion::V2, IgmpVersion::V3);
        assert_eq!(IgmpVersion::V3, IgmpVersion::V3);
    }

    #[test]
    fn test_multicast_join_request() {
        let group = MulticastGroup::new("239.1.0.1", 1234);
        let req = MulticastJoinRequest::new(group, "eth0", IgmpVersion::V3);
        assert_eq!(req.interface, "eth0");
        assert_eq!(req.igmp_version, IgmpVersion::V3);
    }

    #[test]
    fn test_ssm_channel_valid() {
        let ch = SsmChannel::new("10.0.0.1", "232.1.2.3", 5004);
        assert!(ch.is_valid());
    }

    #[test]
    fn test_ssm_channel_invalid_group() {
        let ch = SsmChannel::new("10.0.0.1", "192.168.1.1", 5004);
        assert!(!ch.is_valid());
    }

    #[test]
    fn test_multicast_manager_join_leave() {
        let mut mgr = MulticastManager::new();
        let g = MulticastGroup::new("239.1.0.1", 5004);
        assert!(mgr.join(g.clone()));
        assert_eq!(mgr.active_groups().len(), 1);

        mgr.leave(&g);
        assert_eq!(mgr.active_groups().len(), 0);
    }

    #[test]
    fn test_multicast_manager_reject_invalid() {
        let mut mgr = MulticastManager::new();
        let bad = MulticastGroup::new("10.0.0.1", 5004);
        assert!(!mgr.join(bad));
        assert_eq!(mgr.active_groups().len(), 0);
    }

    #[test]
    fn test_multicast_manager_double_join() {
        let mut mgr = MulticastManager::new();
        let g = MulticastGroup::new("239.1.0.1", 5004);
        mgr.join(g.clone());
        mgr.join(g.clone());
        // Should still be only one active group entry.
        assert_eq!(mgr.active_groups().len(), 1);
        // join_count should reflect two joins.
        let stats = mgr
            .stats_for("239.1.0.1", 5004)
            .expect("should succeed in test");
        assert_eq!(stats.join_count, 2);
    }

    #[test]
    fn test_multicast_stats_record_packet() {
        let mut mgr = MulticastManager::new();
        let g = MulticastGroup::new("239.1.0.1", 5004);
        mgr.join(g);

        mgr.record_packet("239.1.0.1", 5004, 1316, 1_000);
        mgr.record_packet("239.1.0.1", 5004, 1316, 2_000);

        let stats = mgr
            .stats_for("239.1.0.1", 5004)
            .expect("should succeed in test");
        assert_eq!(stats.packets_received, 2);
        assert_eq!(stats.bytes_received, 2632);
        assert_eq!(stats.last_packet_ms, 2_000);
    }
}
