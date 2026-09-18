//! Multicast group management for `VideoIP` streams.
//!
//! Tracks membership for IP multicast groups used in video distribution.

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::Ipv4Addr;

/// Represents a single multicast group and its member addresses.
#[derive(Debug, Clone)]
pub struct MulticastGroup {
    /// The multicast group address (224.0.0.0/4).
    address: Ipv4Addr,
    /// Set of member source addresses.
    members: Vec<Ipv4Addr>,
    /// Human-readable label for the group.
    label: String,
    /// Whether this group is currently active.
    active: bool,
}

impl MulticastGroup {
    /// Creates a new `MulticastGroup` with the given address and label.
    pub fn new(address: Ipv4Addr, label: impl Into<String>) -> Self {
        Self {
            address,
            members: Vec::new(),
            label: label.into(),
            active: true,
        }
    }

    /// Returns the multicast address for this group.
    #[must_use]
    pub fn address(&self) -> Ipv4Addr {
        self.address
    }

    /// Returns the label for this group.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns `true` if the group is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Deactivates the group.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Returns the current member count.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Adds `addr` as a member. Returns `true` if newly added.
    pub fn join(&mut self, addr: Ipv4Addr) -> bool {
        if self.members.contains(&addr) {
            return false;
        }
        self.members.push(addr);
        true
    }

    /// Removes `addr` from the group. Returns `true` if the member was present.
    pub fn leave(&mut self, addr: Ipv4Addr) -> bool {
        if let Some(pos) = self.members.iter().position(|&m| m == addr) {
            self.members.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns a slice of all member addresses.
    #[must_use]
    pub fn members(&self) -> &[Ipv4Addr] {
        &self.members
    }

    /// Returns `true` if `addr` is a member.
    #[must_use]
    pub fn contains(&self, addr: Ipv4Addr) -> bool {
        self.members.contains(&addr)
    }
}

/// Manages a collection of `MulticastGroup` instances.
#[derive(Debug, Clone, Default)]
pub struct GroupManager {
    groups: HashMap<Ipv4Addr, MulticastGroup>,
}

impl GroupManager {
    /// Creates an empty `GroupManager`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
        }
    }

    /// Creates and registers a new multicast group. Returns `false` if the
    /// address was already registered.
    pub fn create(&mut self, address: Ipv4Addr, label: impl Into<String>) -> bool {
        if self.groups.contains_key(&address) {
            return false;
        }
        self.groups
            .insert(address, MulticastGroup::new(address, label));
        true
    }

    /// Looks up a group by address, returning an immutable reference.
    #[must_use]
    pub fn find(&self, address: Ipv4Addr) -> Option<&MulticastGroup> {
        self.groups.get(&address)
    }

    /// Looks up a group by address, returning a mutable reference.
    pub fn find_mut(&mut self, address: Ipv4Addr) -> Option<&mut MulticastGroup> {
        self.groups.get_mut(&address)
    }

    /// Returns all currently active groups.
    #[must_use]
    pub fn active_groups(&self) -> Vec<&MulticastGroup> {
        self.groups.values().filter(|g| g.is_active()).collect()
    }

    /// Returns the total number of registered groups (active and inactive).
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns the total member count across all active groups.
    #[must_use]
    pub fn total_members(&self) -> usize {
        self.active_groups().iter().map(|g| g.member_count()).sum()
    }

    /// Removes a group by address. Returns the removed group if it existed.
    pub fn remove(&mut self, address: Ipv4Addr) -> Option<MulticastGroup> {
        self.groups.remove(&address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn addr(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr::new(a, b, c, d)
    }

    #[test]
    fn test_new_group_has_zero_members() {
        let g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        assert_eq!(g.member_count(), 0);
    }

    #[test]
    fn test_join_adds_member() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        assert!(g.join(addr(192, 168, 1, 10)));
        assert_eq!(g.member_count(), 1);
    }

    #[test]
    fn test_join_duplicate_returns_false() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        g.join(addr(192, 168, 1, 10));
        assert!(!g.join(addr(192, 168, 1, 10)));
        assert_eq!(g.member_count(), 1);
    }

    #[test]
    fn test_leave_removes_member() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        g.join(addr(10, 0, 0, 1));
        assert!(g.leave(addr(10, 0, 0, 1)));
        assert_eq!(g.member_count(), 0);
    }

    #[test]
    fn test_leave_absent_member_returns_false() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        assert!(!g.leave(addr(10, 0, 0, 99)));
    }

    #[test]
    fn test_contains() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        g.join(addr(10, 0, 0, 5));
        assert!(g.contains(addr(10, 0, 0, 5)));
        assert!(!g.contains(addr(10, 0, 0, 6)));
    }

    #[test]
    fn test_group_active_by_default() {
        let g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        assert!(g.is_active());
    }

    #[test]
    fn test_group_deactivate() {
        let mut g = MulticastGroup::new(addr(224, 0, 1, 1), "test");
        g.deactivate();
        assert!(!g.is_active());
    }

    #[test]
    fn test_manager_create_and_find() {
        let mut mgr = GroupManager::new();
        assert!(mgr.create(addr(224, 1, 0, 1), "grp-a"));
        assert!(mgr.find(addr(224, 1, 0, 1)).is_some());
    }

    #[test]
    fn test_manager_create_duplicate_returns_false() {
        let mut mgr = GroupManager::new();
        mgr.create(addr(224, 1, 0, 1), "grp-a");
        assert!(!mgr.create(addr(224, 1, 0, 1), "grp-a-dup"));
    }

    #[test]
    fn test_manager_active_groups() {
        let mut mgr = GroupManager::new();
        mgr.create(addr(224, 1, 0, 1), "a");
        mgr.create(addr(224, 1, 0, 2), "b");
        mgr.find_mut(addr(224, 1, 0, 2))
            .expect("should succeed in test")
            .deactivate();
        assert_eq!(mgr.active_groups().len(), 1);
    }

    #[test]
    fn test_manager_total_members() {
        let mut mgr = GroupManager::new();
        mgr.create(addr(224, 1, 0, 1), "a");
        mgr.find_mut(addr(224, 1, 0, 1))
            .expect("should succeed in test")
            .join(addr(10, 0, 0, 1));
        mgr.find_mut(addr(224, 1, 0, 1))
            .expect("should succeed in test")
            .join(addr(10, 0, 0, 2));
        assert_eq!(mgr.total_members(), 2);
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = GroupManager::new();
        mgr.create(addr(224, 1, 0, 1), "a");
        assert!(mgr.remove(addr(224, 1, 0, 1)).is_some());
        assert_eq!(mgr.group_count(), 0);
    }

    #[test]
    fn test_group_label() {
        let g = MulticastGroup::new(addr(224, 0, 2, 1), "video-feed");
        assert_eq!(g.label(), "video-feed");
    }
}
