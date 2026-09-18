//! Clock domain management: domain discovery, master election, and domain isolation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifier for a clock domain (PTP domain number, 0–127).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DomainId(pub u8);

impl DomainId {
    /// Default PTP domain.
    pub const DEFAULT: Self = Self(0);

    /// Create a new domain identifier (clamped to 0–127).
    #[must_use]
    pub fn new(id: u8) -> Self {
        Self(id.min(127))
    }

    /// Return the raw domain number.
    #[must_use]
    pub fn value(self) -> u8 {
        self.0
    }
}

/// Role of a clock within a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClockRole {
    /// Grandmaster / master clock providing the reference.
    Master,
    /// Slave clock synchronising to a master.
    Slave,
    /// Passive observer; does not participate in BMCA.
    Passive,
    /// Role not yet determined.
    Unknown,
}

/// A clock participant registered within a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMember {
    /// Unique clock identifier (64-bit EUI).
    pub clock_id: u64,
    /// Current role.
    pub role: ClockRole,
    /// Clock quality score (lower = better; follows PTP clock quality).
    pub quality: u8,
    /// Whether the member is currently reachable.
    pub reachable: bool,
}

impl DomainMember {
    /// Create a new domain member.
    #[must_use]
    pub fn new(clock_id: u64, quality: u8) -> Self {
        Self {
            clock_id,
            role: ClockRole::Unknown,
            quality,
            reachable: true,
        }
    }
}

/// Manages a single PTP/clock domain.
#[derive(Debug)]
pub struct ClockDomain {
    /// Domain identifier.
    pub id: DomainId,
    /// Members registered in this domain.
    members: HashMap<u64, DomainMember>,
    /// Current elected master (if any).
    master: Option<u64>,
    /// Whether the domain is isolated (no external sync).
    isolated: bool,
}

impl ClockDomain {
    /// Create a new empty clock domain.
    #[must_use]
    pub fn new(id: DomainId) -> Self {
        Self {
            id,
            members: HashMap::new(),
            master: None,
            isolated: false,
        }
    }

    /// Register a member in this domain.
    pub fn register(&mut self, member: DomainMember) {
        self.members.insert(member.clock_id, member);
    }

    /// Mark a member as unreachable.
    pub fn mark_unreachable(&mut self, clock_id: u64) {
        if let Some(m) = self.members.get_mut(&clock_id) {
            m.reachable = false;
        }
        // If the master becomes unreachable, trigger re-election.
        if self.master == Some(clock_id) {
            self.master = None;
            self.elect_master();
        }
    }

    /// Run the Best Master Clock Algorithm (BMCA) – simplified.
    ///
    /// The member with the lowest quality value among reachable members is elected.
    pub fn elect_master(&mut self) -> Option<u64> {
        let best = self
            .members
            .values()
            .filter(|m| m.reachable)
            .min_by_key(|m| m.quality)
            .map(|m| m.clock_id);

        if let Some(id) = best {
            // Demote previous master
            for m in self.members.values_mut() {
                m.role = if m.clock_id == id {
                    ClockRole::Master
                } else {
                    ClockRole::Slave
                };
            }
            self.master = Some(id);
        }
        self.master
    }

    /// Return the current master clock ID.
    #[must_use]
    pub fn master(&self) -> Option<u64> {
        self.master
    }

    /// Number of reachable members.
    #[must_use]
    pub fn reachable_count(&self) -> usize {
        self.members.values().filter(|m| m.reachable).count()
    }

    /// Isolate this domain from external synchronisation.
    pub fn isolate(&mut self) {
        self.isolated = true;
    }

    /// Re-enable external synchronisation.
    pub fn unisolate(&mut self) {
        self.isolated = false;
    }

    /// Whether the domain is currently isolated.
    #[must_use]
    pub fn is_isolated(&self) -> bool {
        self.isolated
    }

    /// Return a reference to a member by ID.
    #[must_use]
    pub fn member(&self, clock_id: u64) -> Option<&DomainMember> {
        self.members.get(&clock_id)
    }
}

/// Registry that manages multiple clock domains.
#[derive(Debug, Default)]
pub struct DomainRegistry {
    domains: HashMap<u8, ClockDomain>,
}

impl DomainRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a domain.
    pub fn get_or_create(&mut self, id: DomainId) -> &mut ClockDomain {
        self.domains
            .entry(id.0)
            .or_insert_with(|| ClockDomain::new(id))
    }

    /// Number of known domains.
    #[must_use]
    pub fn domain_count(&self) -> usize {
        self.domains.len()
    }

    /// Remove a domain from the registry.
    pub fn remove(&mut self, id: DomainId) -> Option<ClockDomain> {
        self.domains.remove(&id.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_domain() -> ClockDomain {
        let mut d = ClockDomain::new(DomainId::DEFAULT);
        d.register(DomainMember::new(1, 10));
        d.register(DomainMember::new(2, 20));
        d.register(DomainMember::new(3, 5));
        d
    }

    #[test]
    fn test_domain_id_default() {
        assert_eq!(DomainId::DEFAULT.value(), 0);
    }

    #[test]
    fn test_domain_id_clamped() {
        let id = DomainId::new(200);
        assert_eq!(id.value(), 127);
    }

    #[test]
    fn test_domain_id_valid() {
        let id = DomainId::new(10);
        assert_eq!(id.value(), 10);
    }

    #[test]
    fn test_elect_master_chooses_best_quality() {
        let mut d = make_domain();
        let master = d.elect_master();
        // Clock 3 has quality=5, the lowest
        assert_eq!(master, Some(3));
    }

    #[test]
    fn test_master_role_assigned() {
        let mut d = make_domain();
        d.elect_master();
        let m = d.member(3).expect("should succeed in test");
        assert_eq!(m.role, ClockRole::Master);
    }

    #[test]
    fn test_slave_role_assigned() {
        let mut d = make_domain();
        d.elect_master();
        let m = d.member(1).expect("should succeed in test");
        assert_eq!(m.role, ClockRole::Slave);
    }

    #[test]
    fn test_reachable_count() {
        let d = make_domain();
        assert_eq!(d.reachable_count(), 3);
    }

    #[test]
    fn test_mark_unreachable_reduces_count() {
        let mut d = make_domain();
        d.mark_unreachable(1);
        assert_eq!(d.reachable_count(), 2);
    }

    #[test]
    fn test_master_unreachable_triggers_reelection() {
        let mut d = make_domain();
        d.elect_master();
        d.mark_unreachable(3);
        // After clock 3 (quality 5) is gone, clock 1 (quality 10) should win
        assert_eq!(d.master(), Some(1));
    }

    #[test]
    fn test_isolate() {
        let mut d = ClockDomain::new(DomainId::DEFAULT);
        d.isolate();
        assert!(d.is_isolated());
    }

    #[test]
    fn test_unisolate() {
        let mut d = ClockDomain::new(DomainId::DEFAULT);
        d.isolate();
        d.unisolate();
        assert!(!d.is_isolated());
    }

    #[test]
    fn test_elect_master_empty_domain() {
        let mut d = ClockDomain::new(DomainId::DEFAULT);
        assert_eq!(d.elect_master(), None);
    }

    #[test]
    fn test_registry_new_is_empty() {
        let r = DomainRegistry::new();
        assert_eq!(r.domain_count(), 0);
    }

    #[test]
    fn test_registry_get_or_create() {
        let mut r = DomainRegistry::new();
        r.get_or_create(DomainId::new(1));
        r.get_or_create(DomainId::new(2));
        assert_eq!(r.domain_count(), 2);
    }

    #[test]
    fn test_registry_remove() {
        let mut r = DomainRegistry::new();
        r.get_or_create(DomainId::new(5));
        let removed = r.remove(DomainId::new(5));
        assert!(removed.is_some());
        assert_eq!(r.domain_count(), 0);
    }
}
