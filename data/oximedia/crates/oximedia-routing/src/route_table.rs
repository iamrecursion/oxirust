#![allow(dead_code)]
//! IP routing table with longest-prefix-match lookup.
//!
//! Provides [`RouteEntry`] for individual route records and [`RouteTable`]
//! for managing a collection of routes with longest-prefix-match semantics.

use std::net::{IpAddr, Ipv4Addr};

// ---------------------------------------------------------------------------
// Route entry
// ---------------------------------------------------------------------------

/// A single entry in a routing table.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteEntry {
    /// Destination network address.
    pub destination: Ipv4Addr,
    /// Prefix length (0–32).
    pub prefix_len: u8,
    /// Next-hop IP address.
    pub next_hop: IpAddr,
    /// Outgoing interface name.
    pub interface: String,
    /// Administrative metric (lower = preferred).
    pub metric: u32,
}

impl RouteEntry {
    /// Creates a new route entry.
    pub fn new(
        destination: Ipv4Addr,
        prefix_len: u8,
        next_hop: IpAddr,
        interface: impl Into<String>,
    ) -> Self {
        Self {
            destination,
            prefix_len: prefix_len.min(32),
            next_hop,
            interface: interface.into(),
            metric: 1,
        }
    }

    /// Sets the administrative metric.
    pub fn with_metric(mut self, metric: u32) -> Self {
        self.metric = metric;
        self
    }

    /// Returns the network mask as a `u32` bitmask.
    pub fn mask(&self) -> u32 {
        if self.prefix_len == 0 {
            0u32
        } else {
            u32::MAX << (32 - self.prefix_len)
        }
    }

    /// Returns true if `addr` falls within this route's network.
    pub fn contains(&self, addr: Ipv4Addr) -> bool {
        let dest = u32::from(self.destination);
        let target = u32::from(addr);
        let mask = self.mask();
        (dest & mask) == (target & mask)
    }
}

// ---------------------------------------------------------------------------
// RouteTable
// ---------------------------------------------------------------------------

/// A routing table supporting insertion, deletion, and longest-prefix-match.
#[derive(Debug, Default, Clone)]
pub struct RouteTable {
    entries: Vec<RouteEntry>,
}

impl RouteTable {
    /// Creates an empty routing table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a route entry.
    ///
    /// If an entry with the same destination and prefix length already exists,
    /// it is replaced.
    pub fn insert(&mut self, entry: RouteEntry) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|e| e.destination == entry.destination && e.prefix_len == entry.prefix_len)
        {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Removes a route by destination and prefix length.
    ///
    /// Returns `true` if a route was removed.
    pub fn remove(&mut self, destination: Ipv4Addr, prefix_len: u8) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|e| !(e.destination == destination && e.prefix_len == prefix_len));
        self.entries.len() < before
    }

    /// Returns the number of routes in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Performs a longest-prefix-match lookup for `addr`.
    ///
    /// Returns a reference to the best matching [`RouteEntry`], or `None` if
    /// no route matches.
    pub fn longest_prefix_match(&self, addr: Ipv4Addr) -> Option<&RouteEntry> {
        self.entries
            .iter()
            .filter(|e| e.contains(addr))
            .max_by_key(|e| (e.prefix_len, std::cmp::Reverse(e.metric)))
    }

    /// Returns all routes that match `addr` (may be multiple).
    pub fn all_matching(&self, addr: Ipv4Addr) -> Vec<&RouteEntry> {
        self.entries.iter().filter(|e| e.contains(addr)).collect()
    }

    /// Returns a reference to all stored entries.
    pub fn entries(&self) -> &[RouteEntry] {
        &self.entries
    }

    /// Removes all entries from the table.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn gw(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn test_route_entry_mask_slash24() {
        let entry = RouteEntry::new(
            Ipv4Addr::new(192, 168, 1, 0),
            24,
            gw(192, 168, 1, 1),
            "eth0",
        );
        assert_eq!(entry.mask(), 0xFFFF_FF00);
    }

    #[test]
    fn test_route_entry_mask_slash0() {
        let entry = RouteEntry::new(Ipv4Addr::new(0, 0, 0, 0), 0, gw(10, 0, 0, 1), "eth0");
        assert_eq!(entry.mask(), 0u32);
    }

    #[test]
    fn test_route_entry_mask_slash32() {
        let entry = RouteEntry::new(Ipv4Addr::new(10, 0, 0, 5), 32, gw(10, 0, 0, 1), "eth0");
        assert_eq!(entry.mask(), u32::MAX);
    }

    #[test]
    fn test_route_entry_contains_true() {
        let entry = RouteEntry::new(Ipv4Addr::new(10, 0, 0, 0), 8, gw(10, 0, 0, 1), "eth0");
        assert!(entry.contains(Ipv4Addr::new(10, 50, 1, 2)));
    }

    #[test]
    fn test_route_entry_contains_false() {
        let entry = RouteEntry::new(Ipv4Addr::new(10, 0, 0, 0), 8, gw(10, 0, 0, 1), "eth0");
        assert!(!entry.contains(Ipv4Addr::new(192, 168, 1, 1)));
    }

    #[test]
    fn test_insert_and_len() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_insert_replaces_existing() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        table.insert(
            RouteEntry::new(Ipv4Addr::new(10, 0, 0, 0), 8, gw(10, 0, 0, 2), "eth1").with_metric(5),
        );
        assert_eq!(table.len(), 1);
        assert_eq!(table.entries()[0].metric, 5);
    }

    #[test]
    fn test_remove_existing() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        let removed = table.remove(Ipv4Addr::new(10, 0, 0, 0), 8);
        assert!(removed);
        assert!(table.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut table = RouteTable::new();
        assert!(!table.remove(Ipv4Addr::new(1, 2, 3, 4), 24));
    }

    #[test]
    fn test_longest_prefix_match_selects_most_specific() {
        let mut table = RouteTable::new();
        // Default route /0
        table.insert(RouteEntry::new(
            Ipv4Addr::new(0, 0, 0, 0),
            0,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        // /8
        table.insert(RouteEntry::new(
            Ipv4Addr::new(192, 0, 0, 0),
            8,
            gw(10, 0, 0, 2),
            "eth1",
        ));
        // /24 — most specific for 192.168.1.x
        table.insert(RouteEntry::new(
            Ipv4Addr::new(192, 168, 1, 0),
            24,
            gw(10, 0, 0, 3),
            "eth2",
        ));

        let result = table.longest_prefix_match(Ipv4Addr::new(192, 168, 1, 55));
        assert!(result.is_some());
        assert_eq!(result.expect("should succeed in test").prefix_len, 24);
    }

    #[test]
    fn test_longest_prefix_match_default_route() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(0, 0, 0, 0),
            0,
            gw(172, 16, 0, 1),
            "eth0",
        ));
        let result = table.longest_prefix_match(Ipv4Addr::new(8, 8, 8, 8));
        assert!(result.is_some());
        assert_eq!(result.expect("should succeed in test").prefix_len, 0);
    }

    #[test]
    fn test_longest_prefix_match_no_match() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        // 192.168.x.x does not match 10.0.0.0/8
        let result = table.longest_prefix_match(Ipv4Addr::new(192, 168, 1, 1));
        assert!(result.is_none());
    }

    #[test]
    fn test_all_matching_returns_multiple() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(0, 0, 0, 0),
            0,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 2),
            "eth1",
        ));
        let matches = table.all_matching(Ipv4Addr::new(10, 1, 2, 3));
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut table = RouteTable::new();
        table.insert(RouteEntry::new(
            Ipv4Addr::new(10, 0, 0, 0),
            8,
            gw(10, 0, 0, 1),
            "eth0",
        ));
        table.clear();
        assert!(table.is_empty());
    }

    #[test]
    fn test_route_entry_with_metric() {
        let entry = RouteEntry::new(Ipv4Addr::new(10, 0, 0, 0), 8, gw(10, 0, 0, 1), "eth0")
            .with_metric(100);
        assert_eq!(entry.metric, 100);
    }
}
