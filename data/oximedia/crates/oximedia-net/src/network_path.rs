#![allow(dead_code)]
//! Network path discovery and latency estimation.
//!
//! Provides lightweight types for representing network hops and building
//! an estimated end-to-end path from probe results.

use std::net::IpAddr;

/// A single hop along a network path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathHop {
    /// Hop number (1-based).
    pub ttl: u8,
    /// IP address of the router at this hop, if known.
    pub address: Option<IpAddr>,
    /// Round-trip time to this hop in milliseconds, if measured.
    pub rtt_ms: Option<f64>,
}

impl PathHop {
    /// Creates a new `PathHop`.
    #[must_use]
    pub const fn new(ttl: u8, address: Option<IpAddr>, rtt_ms: Option<f64>) -> Self {
        Self {
            ttl,
            address,
            rtt_ms,
        }
    }

    /// Returns `true` if this hop is likely a gateway (TTL == 1).
    #[must_use]
    pub fn is_gateway(&self) -> bool {
        self.ttl == 1
    }

    /// Returns `true` when this hop's address is known.
    #[must_use]
    pub fn has_address(&self) -> bool {
        self.address.is_some()
    }

    /// Returns `true` when a round-trip measurement exists for this hop.
    #[must_use]
    pub fn has_rtt(&self) -> bool {
        self.rtt_ms.is_some()
    }
}

/// A complete network path consisting of ordered hops.
#[derive(Debug, Clone, Default)]
pub struct NetworkPath {
    hops: Vec<PathHop>,
}

impl NetworkPath {
    /// Creates an empty `NetworkPath`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a hop to the path.
    pub fn push(&mut self, hop: PathHop) {
        self.hops.push(hop);
    }

    /// Returns the number of hops.
    #[must_use]
    pub fn hop_count(&self) -> usize {
        self.hops.len()
    }

    /// Returns `true` when no hops have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hops.is_empty()
    }

    /// Provides a read-only slice of all hops.
    #[must_use]
    pub fn hops(&self) -> &[PathHop] {
        &self.hops
    }

    /// Estimates the total one-way latency in milliseconds by summing
    /// each hop's RTT and halving the total (RTT → OWD approximation).
    ///
    /// Hops with no RTT measurement are skipped.
    #[must_use]
    pub fn estimated_latency_ms(&self) -> f64 {
        let total_rtt: f64 = self.hops.iter().filter_map(|h| h.rtt_ms).sum();
        total_rtt / 2.0
    }

    /// Returns the hop with the highest RTT, if any.
    #[must_use]
    pub fn slowest_hop(&self) -> Option<&PathHop> {
        self.hops
            .iter()
            .filter(|h| h.rtt_ms.is_some())
            .max_by(|a, b| {
                a.rtt_ms
                    .unwrap_or(0.0)
                    .partial_cmp(&b.rtt_ms.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Status of a path probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStatus {
    /// Probe is in progress.
    InProgress,
    /// Probe completed successfully.
    Complete,
    /// Probe failed (e.g., timeout).
    Failed,
}

/// An active path probe that accumulates hops and produces a `NetworkPath`.
#[derive(Debug)]
pub struct PathProbe {
    destination: IpAddr,
    path: NetworkPath,
    status: ProbeStatus,
    max_hops: u8,
}

impl PathProbe {
    /// Creates a new probe targeting `destination` with a given hop limit.
    #[must_use]
    pub fn new(destination: IpAddr, max_hops: u8) -> Self {
        Self {
            destination,
            path: NetworkPath::new(),
            status: ProbeStatus::InProgress,
            max_hops,
        }
    }

    /// Returns the probe destination address.
    #[must_use]
    pub fn destination(&self) -> IpAddr {
        self.destination
    }

    /// Returns the current probe status.
    #[must_use]
    pub fn status(&self) -> ProbeStatus {
        self.status
    }

    /// Adds a hop to the path under construction.
    ///
    /// Once `max_hops` is reached the probe is automatically marked complete.
    pub fn add_hop(&mut self, hop: PathHop) {
        if self.status == ProbeStatus::InProgress {
            self.path.push(hop);
            if self.path.hop_count() as u8 >= self.max_hops {
                self.status = ProbeStatus::Complete;
            }
        }
    }

    /// Marks the probe as complete and returns the completed `NetworkPath`.
    pub fn complete(&mut self) -> &NetworkPath {
        self.status = ProbeStatus::Complete;
        &self.path
    }

    /// Marks the probe as failed.
    pub fn fail(&mut self) {
        self.status = ProbeStatus::Failed;
    }

    /// Returns a reference to the path accumulated so far.
    #[must_use]
    pub fn path(&self) -> &NetworkPath {
        &self.path
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ipv4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    // 1. Gateway detection via TTL == 1
    #[test]
    fn test_is_gateway_true() {
        let hop = PathHop::new(1, Some(ipv4(192, 168, 1, 1)), Some(1.0));
        assert!(hop.is_gateway());
    }

    // 2. Non-gateway hop
    #[test]
    fn test_is_gateway_false() {
        let hop = PathHop::new(3, None, None);
        assert!(!hop.is_gateway());
    }

    // 3. has_address
    #[test]
    fn test_has_address_true() {
        let hop = PathHop::new(2, Some(ipv4(10, 0, 0, 1)), None);
        assert!(hop.has_address());
    }

    // 4. has_address false when address is None
    #[test]
    fn test_has_address_false() {
        let hop = PathHop::new(2, None, None);
        assert!(!hop.has_address());
    }

    // 5. Empty path
    #[test]
    fn test_empty_path() {
        let p = NetworkPath::new();
        assert!(p.is_empty());
        assert_eq!(p.hop_count(), 0);
    }

    // 6. hop_count after pushes
    #[test]
    fn test_hop_count() {
        let mut p = NetworkPath::new();
        p.push(PathHop::new(1, None, Some(2.0)));
        p.push(PathHop::new(2, None, Some(5.0)));
        assert_eq!(p.hop_count(), 2);
    }

    // 7. estimated_latency sums and halves RTTs
    #[test]
    fn test_estimated_latency_ms() {
        let mut p = NetworkPath::new();
        p.push(PathHop::new(1, None, Some(10.0)));
        p.push(PathHop::new(2, None, Some(20.0)));
        // total RTT = 30 ms → OWD ≈ 15 ms
        assert!((p.estimated_latency_ms() - 15.0).abs() < 1e-9);
    }

    // 8. estimated_latency skips hops without RTT
    #[test]
    fn test_estimated_latency_skips_none_rtt() {
        let mut p = NetworkPath::new();
        p.push(PathHop::new(1, None, None));
        p.push(PathHop::new(2, None, Some(20.0)));
        assert!((p.estimated_latency_ms() - 10.0).abs() < 1e-9);
    }

    // 9. slowest_hop returns highest RTT
    #[test]
    fn test_slowest_hop() {
        let mut p = NetworkPath::new();
        p.push(PathHop::new(1, None, Some(5.0)));
        p.push(PathHop::new(2, None, Some(50.0)));
        p.push(PathHop::new(3, None, Some(15.0)));
        assert_eq!(p.slowest_hop().map(|h| h.ttl), Some(2));
    }

    // 10. slowest_hop returns None for empty path
    #[test]
    fn test_slowest_hop_empty() {
        let p = NetworkPath::new();
        assert!(p.slowest_hop().is_none());
    }

    // 11. PathProbe starts InProgress
    #[test]
    fn test_probe_initial_status() {
        let probe = PathProbe::new(ipv4(8, 8, 8, 8), 30);
        assert_eq!(probe.status(), ProbeStatus::InProgress);
    }

    // 12. add_hop accumulates hops
    #[test]
    fn test_probe_add_hop() {
        let mut probe = PathProbe::new(ipv4(8, 8, 8, 8), 30);
        probe.add_hop(PathHop::new(1, Some(ipv4(192, 168, 1, 1)), Some(1.5)));
        assert_eq!(probe.path().hop_count(), 1);
    }

    // 13. probe auto-completes at max_hops
    #[test]
    fn test_probe_auto_complete() {
        let mut probe = PathProbe::new(ipv4(8, 8, 8, 8), 2);
        probe.add_hop(PathHop::new(1, None, Some(1.0)));
        probe.add_hop(PathHop::new(2, None, Some(2.0)));
        assert_eq!(probe.status(), ProbeStatus::Complete);
    }

    // 14. complete() returns path and marks status
    #[test]
    fn test_probe_complete_returns_path() {
        let mut probe = PathProbe::new(ipv4(1, 2, 3, 4), 30);
        probe.add_hop(PathHop::new(1, None, Some(3.0)));
        let path = probe.complete();
        assert_eq!(path.hop_count(), 1);
        assert_eq!(probe.status(), ProbeStatus::Complete);
    }
}
