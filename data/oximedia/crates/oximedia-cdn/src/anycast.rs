//! Anycast / DNS-based virtual-IP routing simulation.
//!
//! Models multiple Points-of-Presence (PoPs) sharing one virtual IP address,
//! as used in real-world anycast deployments (e.g. Cloudflare, Fastly).
//!
//! # Overview
//!
//! - [`crate::anycast::VirtualIp`] — a string handle representing an anycast VIP.
//! - [`crate::anycast::AnycastGroup`] — one VIP mapped to a set of [`EdgeNodeGeo`] PoPs.
//! - [`crate::anycast::AnycastRouter`] — holds multiple groups and resolves a client
//!   [`GeoLocation`] to the globally nearest *active* PoP.
//!
//! Resolution algorithm:
//! 1. For every group, find the nearest *active* PoP using Haversine distance
//!    (via [`haversine_km`]) then convert to latency (via [`latency_from_km`]).
//! 2. Across all groups, return the `(vip, node_id, latency_ms)` triple with
//!    the minimal latency.

use crate::geo_routing::{haversine_km, latency_from_km, EdgeNodeGeo, EdgeNodeId, GeoLocation};

// ─── VirtualIp ────────────────────────────────────────────────────────────────

/// An anycast virtual IP address (or DNS name acting as a VIP).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VirtualIp(pub String);

impl VirtualIp {
    /// Create from any `Into<String>`.
    pub fn new(vip: impl Into<String>) -> Self {
        Self(vip.into())
    }
}

impl std::fmt::Display for VirtualIp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── AnycastGroup ─────────────────────────────────────────────────────────────

/// One anycast group: a virtual IP and the PoPs that announce it.
#[derive(Debug, Clone)]
pub struct AnycastGroup {
    /// The virtual IP shared by all PoPs in this group.
    pub vip: VirtualIp,
    /// PoPs that participate in this anycast group.
    pub pops: Vec<EdgeNodeGeo>,
}

impl AnycastGroup {
    /// Create a new group with no PoPs.
    pub fn new(vip: impl Into<String>) -> Self {
        Self {
            vip: VirtualIp::new(vip),
            pops: Vec::new(),
        }
    }

    /// Add a PoP to this group.
    pub fn add_pop(&mut self, pop: EdgeNodeGeo) {
        self.pops.push(pop);
    }
}

// ─── AnycastRouter ────────────────────────────────────────────────────────────

/// Routes a client to the globally nearest active PoP across all anycast groups.
///
/// Each group models one virtual IP that multiple PoPs announce; the router
/// selects the PoP with the lowest propagation latency from the client across
/// *all* registered groups.
#[derive(Debug, Default)]
pub struct AnycastRouter {
    groups: Vec<AnycastGroup>,
}

impl AnycastRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an anycast group.
    pub fn add_group(&mut self, group: AnycastGroup) {
        self.groups.push(group);
    }

    /// Remove the group with the given VIP string.
    ///
    /// No-op if no such group exists.
    pub fn remove_group(&mut self, vip: &str) {
        self.groups.retain(|g| g.vip.0 != vip);
    }

    /// Return the PoPs for `vip`, or `None` if the group is not registered.
    pub fn pops_in_group(&self, vip: &str) -> Option<&[EdgeNodeGeo]> {
        self.groups
            .iter()
            .find(|g| g.vip.0 == vip)
            .map(|g| g.pops.as_slice())
    }

    /// Withdraw a PoP from announcing the given VIP (set `active = false`).
    ///
    /// If the VIP or node is not found, this is a no-op.
    pub fn withdraw_pop(&mut self, vip: &str, node_id: &EdgeNodeId) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.vip.0 == vip) {
            if let Some(pop) = group.pops.iter_mut().find(|p| &p.id == node_id) {
                pop.active = false;
            }
        }
    }

    /// Announce a PoP for the given VIP (set `active = true`).
    ///
    /// If the VIP or node is not found, this is a no-op.
    pub fn announce_pop(&mut self, vip: &str, node_id: &EdgeNodeId) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.vip.0 == vip) {
            if let Some(pop) = group.pops.iter_mut().find(|p| &p.id == node_id) {
                pop.active = true;
            }
        }
    }

    /// Resolve the nearest active PoP across all groups for `client`.
    ///
    /// Returns `Some((vip, node_id, latency_ms))` for the globally closest
    /// active PoP, or `None` if every PoP in every group is withdrawn.
    ///
    /// Distance is computed with [`haversine_km`] and converted to milliseconds
    /// with [`latency_from_km`].
    pub fn resolve(&self, client: &GeoLocation) -> Option<(&VirtualIp, &EdgeNodeId, f64)> {
        let clat = client.latitude;
        let clon = client.longitude;

        let mut best: Option<(&VirtualIp, &EdgeNodeId, f64)> = None;

        for group in &self.groups {
            // Find the nearest active PoP in this group.
            let group_best = group
                .pops
                .iter()
                .filter(|p| p.active)
                .map(|p| {
                    let dist = haversine_km(clat, clon, p.location.latitude, p.location.longitude);
                    let latency = latency_from_km(dist);
                    (&group.vip, &p.id, latency)
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(candidate) = group_best {
                match best {
                    None => best = Some(candidate),
                    Some(ref current) if candidate.2 < current.2 => best = Some(candidate),
                    _ => {}
                }
            }
        }

        best
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_routing::GeoLocation;

    fn loc(lat: f64, lon: f64) -> GeoLocation {
        GeoLocation::new(lat, lon, "US")
    }

    fn pop(id: &str, lat: f64, lon: f64) -> EdgeNodeGeo {
        EdgeNodeGeo::new(id, loc(lat, lon))
    }

    // 1. resolve returns the nearest PoP from a single group
    #[test]
    fn resolve_nearest_pop() {
        let mut router = AnycastRouter::new();
        let mut group = AnycastGroup::new("1.2.3.4");
        // Client is at (0, 0); near-pop at (1, 1), mid-pop at (10, 10), far-pop at (50, 50)
        group.add_pop(pop("near", 1.0, 1.0));
        group.add_pop(pop("mid", 10.0, 10.0));
        group.add_pop(pop("far", 50.0, 50.0));
        router.add_group(group);

        let client = loc(0.0, 0.0);
        let (vip, node_id, _latency) = router.resolve(&client).expect("should resolve");
        assert_eq!(vip.0, "1.2.3.4");
        assert_eq!(node_id.0, "near");
    }

    // 2. withdrawn nearest pop causes fallback to second-nearest
    #[test]
    fn withdrawn_pop_excluded() {
        let mut router = AnycastRouter::new();
        let mut group = AnycastGroup::new("10.0.0.1");
        group.add_pop(pop("near", 1.0, 1.0));
        group.add_pop(pop("second", 10.0, 10.0));
        router.add_group(group);

        let near_id = EdgeNodeId::new("near");
        router.withdraw_pop("10.0.0.1", &near_id);

        let client = loc(0.0, 0.0);
        let (_vip, node_id, _) = router.resolve(&client).expect("should still resolve");
        assert_eq!(node_id.0, "second");
    }

    // 3. group with all pops withdrawn returns None
    #[test]
    fn empty_group_returns_none() {
        let mut router = AnycastRouter::new();
        let mut group = AnycastGroup::new("192.168.1.1");
        group.add_pop(pop("only", 5.0, 5.0));
        router.add_group(group);

        let only_id = EdgeNodeId::new("only");
        router.withdraw_pop("192.168.1.1", &only_id);

        let client = loc(0.0, 0.0);
        assert!(router.resolve(&client).is_none());
    }

    // 4. two groups — client closer to group B selects B's VIP
    #[test]
    fn two_groups_pick_correct_vip() {
        let mut router = AnycastRouter::new();

        // Group A: PoP at (50, 10) — far from (0, 0) client
        let mut group_a = AnycastGroup::new("vip-a");
        group_a.add_pop(pop("a-pop", 50.0, 10.0));
        router.add_group(group_a);

        // Group B: PoP at (2, 2) — near the (0, 0) client
        let mut group_b = AnycastGroup::new("vip-b");
        group_b.add_pop(pop("b-pop", 2.0, 2.0));
        router.add_group(group_b);

        let client = loc(0.0, 0.0);
        let (vip, node_id, _) = router.resolve(&client).expect("should resolve");
        assert_eq!(vip.0, "vip-b", "group B's VIP should win (closer PoP)");
        assert_eq!(node_id.0, "b-pop");
    }
}
