//! IP multicast management for professional video transport.
//!
//! Supports Any-Source Multicast (ASM) and Source-Specific Multicast (SSM)
//! groups, IGMP version tracking, subscriber management, and bandwidth
//! accounting.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// MulticastGroup
// ---------------------------------------------------------------------------

/// An IP multicast group, optionally in SSM (Source-Specific Multicast) mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticastGroup {
    /// Multicast group address string (e.g. `"239.1.2.3"`).
    pub address: String,
    /// UDP port number.
    pub port: u16,
    /// Source address for SSM, if applicable (e.g. `"10.0.0.1"`).
    pub source_specific: Option<String>,
}

impl MulticastGroup {
    /// Create a new ASM (Any-Source Multicast) group.
    #[must_use]
    pub fn new(address: impl Into<String>, port: u16) -> Self {
        Self {
            address: address.into(),
            port,
            source_specific: None,
        }
    }

    /// Create a new SSM (Source-Specific Multicast) group.
    #[must_use]
    pub fn new_ssm(address: impl Into<String>, port: u16, source: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            port,
            source_specific: Some(source.into()),
        }
    }

    /// Returns `true` when this group uses Source-Specific Multicast.
    #[must_use]
    pub fn is_ssm(&self) -> bool {
        self.source_specific.is_some()
    }

    /// Returns `true` when the group address is in the 224.x.x.x – 239.x.x.x
    /// multicast range (first octet between 224 and 239 inclusive).
    #[must_use]
    pub fn is_valid_address(&self) -> bool {
        let first_octet: Option<u8> = self.address.split('.').next().and_then(|s| s.parse().ok());
        matches!(first_octet, Some(224..=239))
    }
}

// ---------------------------------------------------------------------------
// IgmpVersion
// ---------------------------------------------------------------------------

/// IGMP protocol version used by this session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgmpVersion {
    /// IGMP v1 — basic group membership only.
    V1,
    /// IGMP v2 — adds leave group messages.
    V2,
    /// IGMP v3 — adds source-specific multicast support.
    V3,
}

impl IgmpVersion {
    /// Returns `true` when this IGMP version supports SSM (`IGMPv3` only).
    #[must_use]
    pub fn supports_ssm(&self) -> bool {
        *self == Self::V3
    }
}

// ---------------------------------------------------------------------------
// MulticastStream
// ---------------------------------------------------------------------------

/// An active multicast video stream.
#[derive(Debug, Clone)]
pub struct MulticastStream {
    /// Unique stream identifier.
    pub id: u64,
    /// Multicast group this stream is sent to.
    pub group: MulticastGroup,
    /// Stream bandwidth in megabits per second.
    pub bandwidth_mbps: f32,
    /// IP addresses of current subscribers.
    pub subscribers: Vec<String>,
}

impl MulticastStream {
    /// Create a new multicast stream.
    #[must_use]
    pub fn new(id: u64, group: MulticastGroup, bandwidth_mbps: f32) -> Self {
        Self {
            id,
            group,
            bandwidth_mbps,
            subscribers: Vec::new(),
        }
    }

    /// Number of active subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Add a subscriber by IP address. No-op if already subscribed.
    pub fn add_subscriber(&mut self, ip: &str) {
        if !self.subscribers.iter().any(|s| s == ip) {
            self.subscribers.push(ip.to_string());
        }
    }

    /// Remove a subscriber by IP address.
    ///
    /// Returns `true` if the subscriber was found and removed.
    pub fn remove_subscriber(&mut self, ip: &str) -> bool {
        if let Some(pos) = self.subscribers.iter().position(|s| s == ip) {
            self.subscribers.remove(pos);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// MulticastRouter
// ---------------------------------------------------------------------------

/// Manages a collection of multicast video streams and their subscribers.
#[derive(Debug, Default)]
pub struct MulticastRouter {
    /// All managed streams.
    pub streams: Vec<MulticastStream>,
}

impl MulticastRouter {
    /// Create a new, empty multicast router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
        }
    }

    /// Register a multicast stream with the router.
    pub fn add_stream(&mut self, stream: MulticastStream) {
        self.streams.push(stream);
    }

    /// Subscribe `subscriber` IP to the stream with `stream_id`.
    ///
    /// Returns `true` on success, `false` if the stream ID was not found.
    pub fn join(&mut self, stream_id: u64, subscriber: &str) -> bool {
        if let Some(s) = self.streams.iter_mut().find(|s| s.id == stream_id) {
            s.add_subscriber(subscriber);
            true
        } else {
            false
        }
    }

    /// Unsubscribe `subscriber` IP from the stream with `stream_id`.
    ///
    /// Returns `true` when the subscriber was found and removed,
    /// `false` when the stream or subscriber was not found.
    pub fn leave(&mut self, stream_id: u64, subscriber: &str) -> bool {
        if let Some(s) = self.streams.iter_mut().find(|s| s.id == stream_id) {
            s.remove_subscriber(subscriber)
        } else {
            false
        }
    }

    /// Return references to all streams that `subscriber` is currently
    /// subscribed to.
    #[must_use]
    pub fn streams_for(&self, subscriber: &str) -> Vec<&MulticastStream> {
        self.streams
            .iter()
            .filter(|s| s.subscribers.iter().any(|ip| ip == subscriber))
            .collect()
    }

    /// Sum of bandwidth across all managed streams in Mbps.
    #[must_use]
    pub fn total_bandwidth_mbps(&self) -> f32 {
        self.streams.iter().map(|s| s.bandwidth_mbps).sum()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // MulticastGroup

    #[test]
    fn test_group_is_ssm_false() {
        let g = MulticastGroup::new("239.1.2.3", 5004);
        assert!(!g.is_ssm());
    }

    #[test]
    fn test_group_is_ssm_true() {
        let g = MulticastGroup::new_ssm("239.1.2.3", 5004, "10.0.0.1");
        assert!(g.is_ssm());
        assert_eq!(g.source_specific.as_deref(), Some("10.0.0.1"));
    }

    #[test]
    fn test_group_valid_address_224() {
        let g = MulticastGroup::new("224.0.0.1", 5000);
        assert!(g.is_valid_address());
    }

    #[test]
    fn test_group_valid_address_239() {
        let g = MulticastGroup::new("239.255.255.250", 5000);
        assert!(g.is_valid_address());
    }

    #[test]
    fn test_group_invalid_address_unicast() {
        let g = MulticastGroup::new("192.168.1.1", 5000);
        assert!(!g.is_valid_address());
    }

    #[test]
    fn test_group_invalid_address_broadcast() {
        let g = MulticastGroup::new("255.255.255.255", 5000);
        assert!(!g.is_valid_address());
    }

    // IgmpVersion

    #[test]
    fn test_igmp_v1_no_ssm() {
        assert!(!IgmpVersion::V1.supports_ssm());
    }

    #[test]
    fn test_igmp_v2_no_ssm() {
        assert!(!IgmpVersion::V2.supports_ssm());
    }

    #[test]
    fn test_igmp_v3_supports_ssm() {
        assert!(IgmpVersion::V3.supports_ssm());
    }

    // MulticastStream

    #[test]
    fn test_stream_add_subscriber() {
        let mut s = MulticastStream::new(1, MulticastGroup::new("239.0.0.1", 5000), 10.0);
        s.add_subscriber("10.0.0.1");
        assert_eq!(s.subscriber_count(), 1);
    }

    #[test]
    fn test_stream_add_subscriber_no_duplicates() {
        let mut s = MulticastStream::new(1, MulticastGroup::new("239.0.0.1", 5000), 10.0);
        s.add_subscriber("10.0.0.1");
        s.add_subscriber("10.0.0.1");
        assert_eq!(s.subscriber_count(), 1);
    }

    #[test]
    fn test_stream_remove_subscriber_found() {
        let mut s = MulticastStream::new(1, MulticastGroup::new("239.0.0.1", 5000), 10.0);
        s.add_subscriber("10.0.0.2");
        assert!(s.remove_subscriber("10.0.0.2"));
        assert_eq!(s.subscriber_count(), 0);
    }

    #[test]
    fn test_stream_remove_subscriber_not_found() {
        let mut s = MulticastStream::new(1, MulticastGroup::new("239.0.0.1", 5000), 10.0);
        assert!(!s.remove_subscriber("10.0.0.99"));
    }

    // MulticastRouter

    #[test]
    fn test_router_join_success() {
        let mut r = MulticastRouter::new();
        r.add_stream(MulticastStream::new(
            1,
            MulticastGroup::new("239.0.0.1", 5000),
            5.0,
        ));
        assert!(r.join(1, "10.0.0.1"));
        assert_eq!(r.streams[0].subscriber_count(), 1);
    }

    #[test]
    fn test_router_join_unknown_stream() {
        let mut r = MulticastRouter::new();
        assert!(!r.join(99, "10.0.0.1"));
    }

    #[test]
    fn test_router_leave_success() {
        let mut r = MulticastRouter::new();
        r.add_stream(MulticastStream::new(
            1,
            MulticastGroup::new("239.0.0.1", 5000),
            5.0,
        ));
        r.join(1, "10.0.0.1");
        assert!(r.leave(1, "10.0.0.1"));
        assert_eq!(r.streams[0].subscriber_count(), 0);
    }

    #[test]
    fn test_router_streams_for_subscriber() {
        let mut r = MulticastRouter::new();
        r.add_stream(MulticastStream::new(
            1,
            MulticastGroup::new("239.0.0.1", 5000),
            5.0,
        ));
        r.add_stream(MulticastStream::new(
            2,
            MulticastGroup::new("239.0.0.2", 5001),
            10.0,
        ));
        r.join(1, "10.0.0.1");
        r.join(2, "10.0.0.1");
        let streams = r.streams_for("10.0.0.1");
        assert_eq!(streams.len(), 2);
    }

    #[test]
    fn test_router_total_bandwidth() {
        let mut r = MulticastRouter::new();
        r.add_stream(MulticastStream::new(
            1,
            MulticastGroup::new("239.0.0.1", 5000),
            5.0,
        ));
        r.add_stream(MulticastStream::new(
            2,
            MulticastGroup::new("239.0.0.2", 5001),
            10.0,
        ));
        assert!((r.total_bandwidth_mbps() - 15.0).abs() < f32::EPSILON);
    }
}
