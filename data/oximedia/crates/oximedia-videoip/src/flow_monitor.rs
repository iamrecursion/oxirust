//! IP video flow monitoring (SMPTE ST 2110).
//!
//! Tracks per-flow RTP statistics including packet loss, bandwidth, and
//! staleness detection for professional media-over-IP deployments.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// FlowId
// ---------------------------------------------------------------------------

/// Five-tuple identifier for an RTP media flow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlowId {
    /// Source IP address.
    pub src_ip: String,
    /// Destination IP address.
    pub dst_ip: String,
    /// Source UDP port.
    pub src_port: u16,
    /// Destination UDP port.
    pub dst_port: u16,
    /// RTP Synchronisation Source identifier.
    pub ssrc: u32,
}

impl FlowId {
    /// Create a new flow identifier.
    #[must_use]
    pub fn new(
        src_ip: impl Into<String>,
        dst_ip: impl Into<String>,
        src_port: u16,
        dst_port: u16,
        ssrc: u32,
    ) -> Self {
        Self {
            src_ip: src_ip.into(),
            dst_ip: dst_ip.into(),
            src_port,
            dst_port,
            ssrc,
        }
    }

    /// Returns `true` when the destination address is an IP multicast address
    /// (first octet 224–239).
    #[must_use]
    pub fn is_multicast(&self) -> bool {
        let first: Option<u8> = self.dst_ip.split('.').next().and_then(|s| s.parse().ok());
        matches!(first, Some(224..=239))
    }
}

// ---------------------------------------------------------------------------
// FlowStats
// ---------------------------------------------------------------------------

/// Accumulated per-flow RTP statistics.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// Total RTP packets received.
    pub packets_received: u64,
    /// Total RTP packets lost (detected by sequence-number gaps).
    pub packets_lost: u64,
    /// Total payload bytes received.
    pub bytes_received: u64,
    /// Most recent RTP timestamp.
    pub last_rtp_ts: u32,
    /// Wall-clock time of the most recent received packet (ms since epoch).
    pub last_recv_ms: u64,
}

impl FlowStats {
    /// Packet loss as a percentage of all expected packets (0.0–100.0).
    ///
    /// Returns `0.0` when no packets have been expected yet.
    #[must_use]
    pub fn packet_loss_pct(&self) -> f64 {
        let total = self.packets_received + self.packets_lost;
        if total == 0 {
            return 0.0;
        }
        self.packets_lost as f64 / total as f64 * 100.0
    }

    /// Estimated average bandwidth in Mbps over `duration_ms` milliseconds.
    ///
    /// Returns `0.0` when `duration_ms` is zero.
    #[must_use]
    pub fn bandwidth_mbps(&self, duration_ms: u64) -> f64 {
        if duration_ms == 0 {
            return 0.0;
        }
        let bits = self.bytes_received as f64 * 8.0;
        let duration_s = duration_ms as f64 / 1_000.0;
        bits / duration_s / 1_000_000.0
    }

    /// Returns `true` when the flow has not received a packet within the
    /// staleness `timeout_ms` window.
    #[must_use]
    pub fn is_stale(&self, now_ms: u64, timeout_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_recv_ms) >= timeout_ms
    }
}

// ---------------------------------------------------------------------------
// FlowMonitor
// ---------------------------------------------------------------------------

/// Monitors all active ST 2110 media flows.
#[derive(Debug, Default)]
pub struct FlowMonitor {
    /// All tracked flows: `(FlowId, FlowStats)`.
    pub flows: Vec<(FlowId, FlowStats)>,
}

impl FlowMonitor {
    /// Create a new, empty flow monitor.
    #[must_use]
    pub fn new() -> Self {
        Self { flows: Vec::new() }
    }

    /// Record an update for the given flow.
    ///
    /// Creates the flow entry if it does not exist, otherwise accumulates
    /// packet and byte counters.
    pub fn update(&mut self, id: FlowId, bytes: u64, rtp_ts: u32, now_ms: u64, lost: u64) {
        if let Some((_, stats)) = self.flows.iter_mut().find(|(fid, _)| fid == &id) {
            stats.packets_received += 1;
            stats.packets_lost += lost;
            stats.bytes_received += bytes;
            stats.last_rtp_ts = rtp_ts;
            stats.last_recv_ms = now_ms;
        } else {
            self.flows.push((
                id,
                FlowStats {
                    packets_received: 1,
                    packets_lost: lost,
                    bytes_received: bytes,
                    last_rtp_ts: rtp_ts,
                    last_recv_ms: now_ms,
                },
            ));
        }
    }

    /// Return a reference to the statistics for `id`, or `None` if unknown.
    #[must_use]
    pub fn get_stats(&self, id: &FlowId) -> Option<&FlowStats> {
        self.flows
            .iter()
            .find(|(fid, _)| fid == id)
            .map(|(_, stats)| stats)
    }

    /// Estimated total bandwidth across all flows in Mbps over `duration_ms`.
    #[must_use]
    pub fn total_bandwidth_mbps(&self, duration_ms: u64) -> f64 {
        self.flows
            .iter()
            .map(|(_, s)| s.bandwidth_mbps(duration_ms))
            .sum()
    }

    /// Return references to flow IDs that have not received a packet within
    /// `timeout_ms` milliseconds of `now_ms`.
    #[must_use]
    pub fn stale_flows(&self, now_ms: u64, timeout_ms: u64) -> Vec<&FlowId> {
        self.flows
            .iter()
            .filter(|(_, s)| s.is_stale(now_ms, timeout_ms))
            .map(|(id, _)| id)
            .collect()
    }

    /// Remove all stale flows and return how many were removed.
    pub fn remove_stale(&mut self, now_ms: u64, timeout_ms: u64) -> usize {
        let before = self.flows.len();
        self.flows.retain(|(_, s)| !s.is_stale(now_ms, timeout_ms));
        before - self.flows.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(dst: &str) -> FlowId {
        FlowId::new("10.0.0.1", dst, 5000, 5000, 0xABCD_1234)
    }

    // FlowId

    #[test]
    fn test_flow_id_is_multicast_true() {
        let id = make_id("239.1.2.3");
        assert!(id.is_multicast());
    }

    #[test]
    fn test_flow_id_is_multicast_false_unicast() {
        let id = make_id("10.0.0.2");
        assert!(!id.is_multicast());
    }

    #[test]
    fn test_flow_id_is_multicast_boundary_224() {
        let id = make_id("224.0.0.1");
        assert!(id.is_multicast());
    }

    #[test]
    fn test_flow_id_is_multicast_boundary_239() {
        let id = make_id("239.255.255.250");
        assert!(id.is_multicast());
    }

    #[test]
    fn test_flow_id_is_multicast_false_240() {
        let id = make_id("240.0.0.1");
        assert!(!id.is_multicast());
    }

    // FlowStats

    #[test]
    fn test_flow_stats_packet_loss_pct_zero() {
        let s = FlowStats {
            packets_received: 1000,
            packets_lost: 0,
            bytes_received: 100_000,
            last_rtp_ts: 0,
            last_recv_ms: 1000,
        };
        assert_eq!(s.packet_loss_pct(), 0.0);
    }

    #[test]
    fn test_flow_stats_packet_loss_pct_ten() {
        let s = FlowStats {
            packets_received: 90,
            packets_lost: 10,
            bytes_received: 90_000,
            last_rtp_ts: 0,
            last_recv_ms: 1000,
        };
        assert!((s.packet_loss_pct() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_flow_stats_packet_loss_pct_empty() {
        let s = FlowStats {
            packets_received: 0,
            packets_lost: 0,
            bytes_received: 0,
            last_rtp_ts: 0,
            last_recv_ms: 0,
        };
        assert_eq!(s.packet_loss_pct(), 0.0);
    }

    #[test]
    fn test_flow_stats_bandwidth_mbps() {
        // 1_000_000 bytes in 1000 ms = 8 Mbps
        let s = FlowStats {
            packets_received: 100,
            packets_lost: 0,
            bytes_received: 1_000_000,
            last_rtp_ts: 0,
            last_recv_ms: 1000,
        };
        assert!((s.bandwidth_mbps(1000) - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_flow_stats_bandwidth_zero_duration() {
        let s = FlowStats {
            packets_received: 10,
            packets_lost: 0,
            bytes_received: 50_000,
            last_rtp_ts: 0,
            last_recv_ms: 0,
        };
        assert_eq!(s.bandwidth_mbps(0), 0.0);
    }

    #[test]
    fn test_flow_stats_is_stale_true() {
        let s = FlowStats {
            packets_received: 1,
            packets_lost: 0,
            bytes_received: 100,
            last_rtp_ts: 0,
            last_recv_ms: 1000,
        };
        assert!(s.is_stale(6001, 5000));
    }

    #[test]
    fn test_flow_stats_is_stale_false() {
        let s = FlowStats {
            packets_received: 1,
            packets_lost: 0,
            bytes_received: 100,
            last_rtp_ts: 0,
            last_recv_ms: 5000,
        };
        assert!(!s.is_stale(6000, 5000));
    }

    // FlowMonitor

    #[test]
    fn test_monitor_update_creates_flow() {
        let mut m = FlowMonitor::new();
        m.update(make_id("10.0.0.2"), 1316, 12345, 1000, 0);
        assert_eq!(m.flows.len(), 1);
    }

    #[test]
    fn test_monitor_update_accumulates() {
        let mut m = FlowMonitor::new();
        let id = make_id("10.0.0.2");
        m.update(id.clone(), 1000, 1, 1000, 0);
        m.update(id.clone(), 2000, 2, 2000, 1);
        let s = m.get_stats(&id).expect("should succeed in test");
        assert_eq!(s.packets_received, 2);
        assert_eq!(s.packets_lost, 1);
        assert_eq!(s.bytes_received, 3000);
    }

    #[test]
    fn test_monitor_get_stats_unknown() {
        let m = FlowMonitor::new();
        assert!(m.get_stats(&make_id("10.0.0.99")).is_none());
    }

    #[test]
    fn test_monitor_stale_flows() {
        let mut m = FlowMonitor::new();
        m.update(make_id("10.0.0.2"), 100, 0, 1000, 0);
        let stale = m.stale_flows(10_000, 5000);
        assert_eq!(stale.len(), 1);
    }

    #[test]
    fn test_monitor_remove_stale() {
        let mut m = FlowMonitor::new();
        m.update(make_id("10.0.0.2"), 100, 0, 1000, 0); // stale
        m.update(make_id("10.0.0.3"), 100, 0, 9000, 0); // fresh
        let removed = m.remove_stale(10_000, 5000);
        assert_eq!(removed, 1);
        assert_eq!(m.flows.len(), 1);
    }

    #[test]
    fn test_monitor_total_bandwidth() {
        let mut m = FlowMonitor::new();
        // two flows, 1 MB each, 1000 ms → each 8 Mbps → 16 Mbps total
        m.update(make_id("10.0.0.2"), 1_000_000, 0, 1000, 0);
        m.update(make_id("10.0.0.3"), 1_000_000, 0, 1000, 0);
        assert!((m.total_bandwidth_mbps(1000) - 16.0).abs() < 0.01);
    }
}
