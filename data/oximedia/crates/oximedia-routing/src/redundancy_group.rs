#![allow(dead_code)]
//! Redundancy group management for failover routing paths.
//!
//! Groups multiple routing paths into redundancy sets with primary/secondary
//! roles, automatic failover detection, and manual override capabilities.

use std::collections::HashMap;

/// Unique identifier for a redundancy group.
pub type GroupId = u64;

/// Unique identifier for a path member within a group.
pub type MemberId = u64;

/// Role of a member within a redundancy group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberRole {
    /// Primary (active) path.
    Primary,
    /// Secondary (standby) path, ready for immediate failover.
    Secondary,
    /// Tertiary backup, activated only if both primary and secondary fail.
    Tertiary,
}

/// Health state of a path member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberHealth {
    /// Member is operating normally.
    Healthy,
    /// Member has intermittent errors.
    Degraded,
    /// Member is faulted and cannot carry traffic.
    Faulted,
    /// Health has not yet been determined.
    Unknown,
}

/// Failover strategy for the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverStrategy {
    /// Automatic failover on health change with no operator intervention.
    Automatic,
    /// Manual failover only — operator must confirm.
    Manual,
    /// Semi-automatic — failover triggered but operator notified.
    SemiAutomatic,
}

/// A single path member within a redundancy group.
#[derive(Debug, Clone)]
pub struct PathMember {
    /// Member identifier.
    pub id: MemberId,
    /// Human-readable label.
    pub label: String,
    /// Assigned role.
    pub role: MemberRole,
    /// Current health.
    pub health: MemberHealth,
    /// Whether this member is currently carrying traffic.
    pub active: bool,
    /// Priority (lower = preferred, 0 is highest).
    pub priority: u32,
    /// Latency in microseconds (measured).
    pub latency_us: u64,
}

impl PathMember {
    /// Create a new path member.
    pub fn new(id: MemberId, label: impl Into<String>, role: MemberRole) -> Self {
        Self {
            id,
            label: label.into(),
            role,
            health: MemberHealth::Unknown,
            active: false,
            priority: match role {
                MemberRole::Primary => 0,
                MemberRole::Secondary => 1,
                MemberRole::Tertiary => 2,
            },
            latency_us: 0,
        }
    }

    /// Whether this member can be activated (health is sufficient).
    pub fn is_available(&self) -> bool {
        matches!(self.health, MemberHealth::Healthy | MemberHealth::Degraded)
    }
}

/// Event recorded when a failover occurs.
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    /// Group in which failover happened.
    pub group_id: GroupId,
    /// Previous active member.
    pub from_member: MemberId,
    /// New active member.
    pub to_member: MemberId,
    /// Reason for the failover.
    pub reason: FailoverReason,
    /// Monotonic timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Reason a failover was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverReason {
    /// Primary member went down.
    HealthDegraded,
    /// Operator triggered manual failover.
    ManualSwitch,
    /// Scheduled maintenance switchover.
    Maintenance,
    /// Latency exceeded threshold.
    LatencyThreshold,
}

/// A redundancy group holding multiple path members.
#[derive(Debug)]
pub struct RedundancyGroup {
    /// Group identifier.
    pub id: GroupId,
    /// Human-readable name.
    pub name: String,
    /// Failover strategy.
    pub strategy: FailoverStrategy,
    /// Path members.
    members: Vec<PathMember>,
    /// Failover event log.
    events: Vec<FailoverEvent>,
    /// Maximum events to retain.
    max_events: usize,
    /// Internal clock for event timestamps.
    clock_us: u64,
}

impl RedundancyGroup {
    /// Create a new redundancy group.
    pub fn new(id: GroupId, name: impl Into<String>, strategy: FailoverStrategy) -> Self {
        Self {
            id,
            name: name.into(),
            strategy,
            members: Vec::new(),
            events: Vec::new(),
            max_events: 256,
            clock_us: 0,
        }
    }

    /// Add a member to the group.
    pub fn add_member(&mut self, member: PathMember) {
        self.members.push(member);
    }

    /// Number of members in the group.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get a member by ID.
    pub fn member(&self, id: MemberId) -> Option<&PathMember> {
        self.members.iter().find(|m| m.id == id)
    }

    /// Get a mutable member by ID.
    pub fn member_mut(&mut self, id: MemberId) -> Option<&mut PathMember> {
        self.members.iter_mut().find(|m| m.id == id)
    }

    /// Return the currently active member, if any.
    pub fn active_member(&self) -> Option<&PathMember> {
        self.members.iter().find(|m| m.active)
    }

    /// Return the ID of the currently active member.
    pub fn active_member_id(&self) -> Option<MemberId> {
        self.active_member().map(|m| m.id)
    }

    /// Activate a specific member and deactivate all others.
    pub fn activate(&mut self, member_id: MemberId) -> bool {
        let exists = self.members.iter().any(|m| m.id == member_id);
        if !exists {
            return false;
        }
        let prev_active = self.active_member_id();
        for m in &mut self.members {
            m.active = m.id == member_id;
        }
        if let Some(prev) = prev_active {
            if prev != member_id {
                self.record_event(prev, member_id, FailoverReason::ManualSwitch);
            }
        }
        true
    }

    /// Update the health of a member and trigger automatic failover if needed.
    pub fn update_health(&mut self, member_id: MemberId, health: MemberHealth) {
        if let Some(m) = self.members.iter_mut().find(|m| m.id == member_id) {
            m.health = health;
        }

        if self.strategy == FailoverStrategy::Manual {
            return;
        }

        // Check if the active member is still available
        let active_faulted = self.active_member().map_or(true, |m| !m.is_available());

        if active_faulted {
            self.failover_to_best();
        }
    }

    /// Attempt failover to the best available member.
    fn failover_to_best(&mut self) {
        let prev_active = self.active_member_id();

        // Sort candidates by priority, pick first available
        let mut candidates: Vec<(MemberId, u32)> = self
            .members
            .iter()
            .filter(|m| m.is_available())
            .map(|m| (m.id, m.priority))
            .collect();
        candidates.sort_by_key(|(_, p)| *p);

        if let Some((best_id, _)) = candidates.first() {
            let best_id = *best_id;
            for m in &mut self.members {
                m.active = m.id == best_id;
            }
            if let Some(prev) = prev_active {
                if prev != best_id {
                    self.record_event(prev, best_id, FailoverReason::HealthDegraded);
                }
            }
        }
    }

    /// Record a failover event.
    fn record_event(&mut self, from: MemberId, to: MemberId, reason: FailoverReason) {
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(FailoverEvent {
            group_id: self.id,
            from_member: from,
            to_member: to,
            reason,
            timestamp_us: self.clock_us,
        });
    }

    /// Get the failover event log.
    pub fn events(&self) -> &[FailoverEvent] {
        &self.events
    }

    /// Advance the internal clock.
    pub fn advance_clock(&mut self, delta_us: u64) {
        self.clock_us += delta_us;
    }

    /// Number of members currently available (healthy or degraded).
    pub fn available_count(&self) -> usize {
        self.members.iter().filter(|m| m.is_available()).count()
    }

    /// Whether all members are down.
    pub fn is_total_failure(&self) -> bool {
        self.available_count() == 0 && !self.members.is_empty()
    }
}

/// Manager for multiple redundancy groups.
#[derive(Debug, Default)]
pub struct RedundancyManager {
    /// All groups keyed by ID.
    groups: HashMap<GroupId, RedundancyGroup>,
    /// Next auto-assigned group ID.
    next_id: GroupId,
}

impl RedundancyManager {
    /// Create a new manager.
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a group and return its ID.
    pub fn add_group(&mut self, name: impl Into<String>, strategy: FailoverStrategy) -> GroupId {
        let id = self.next_id;
        self.next_id += 1;
        let group = RedundancyGroup::new(id, name, strategy);
        self.groups.insert(id, group);
        id
    }

    /// Get a group by ID.
    pub fn group(&self, id: GroupId) -> Option<&RedundancyGroup> {
        self.groups.get(&id)
    }

    /// Get a mutable group by ID.
    pub fn group_mut(&mut self, id: GroupId) -> Option<&mut RedundancyGroup> {
        self.groups.get_mut(&id)
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Return all groups that have a total failure (no available members).
    pub fn failed_groups(&self) -> Vec<GroupId> {
        self.groups
            .values()
            .filter(|g| g.is_total_failure())
            .map(|g| g.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_creation() {
        let m = PathMember::new(1, "Path-A", MemberRole::Primary);
        assert_eq!(m.id, 1);
        assert_eq!(m.role, MemberRole::Primary);
        assert_eq!(m.priority, 0);
        assert!(!m.active);
    }

    #[test]
    fn test_member_availability() {
        let mut m = PathMember::new(1, "Path-A", MemberRole::Primary);
        assert!(!m.is_available()); // Unknown health
        m.health = MemberHealth::Healthy;
        assert!(m.is_available());
        m.health = MemberHealth::Degraded;
        assert!(m.is_available());
        m.health = MemberHealth::Faulted;
        assert!(!m.is_available());
    }

    #[test]
    fn test_group_add_member() {
        let mut g = RedundancyGroup::new(1, "Group1", FailoverStrategy::Automatic);
        g.add_member(PathMember::new(1, "A", MemberRole::Primary));
        g.add_member(PathMember::new(2, "B", MemberRole::Secondary));
        assert_eq!(g.member_count(), 2);
    }

    #[test]
    fn test_group_activate() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        let mut m1 = PathMember::new(1, "A", MemberRole::Primary);
        m1.health = MemberHealth::Healthy;
        let mut m2 = PathMember::new(2, "B", MemberRole::Secondary);
        m2.health = MemberHealth::Healthy;
        g.add_member(m1);
        g.add_member(m2);

        assert!(g.activate(1));
        assert_eq!(g.active_member_id(), Some(1));

        assert!(g.activate(2));
        assert_eq!(g.active_member_id(), Some(2));
    }

    #[test]
    fn test_group_activate_invalid() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        assert!(!g.activate(999));
    }

    #[test]
    fn test_automatic_failover() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        let mut m1 = PathMember::new(1, "Primary", MemberRole::Primary);
        m1.health = MemberHealth::Healthy;
        m1.active = true;
        let mut m2 = PathMember::new(2, "Secondary", MemberRole::Secondary);
        m2.health = MemberHealth::Healthy;
        g.add_member(m1);
        g.add_member(m2);

        // Fault the primary
        g.update_health(1, MemberHealth::Faulted);
        // Should have failed over to secondary
        assert_eq!(g.active_member_id(), Some(2));
        assert_eq!(g.events().len(), 1);
        assert_eq!(g.events()[0].reason, FailoverReason::HealthDegraded);
    }

    #[test]
    fn test_manual_no_auto_failover() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Manual);
        let mut m1 = PathMember::new(1, "Primary", MemberRole::Primary);
        m1.health = MemberHealth::Healthy;
        m1.active = true;
        let mut m2 = PathMember::new(2, "Secondary", MemberRole::Secondary);
        m2.health = MemberHealth::Healthy;
        g.add_member(m1);
        g.add_member(m2);

        // Fault the primary — should NOT auto-failover
        g.update_health(1, MemberHealth::Faulted);
        assert_eq!(g.active_member_id(), Some(1)); // still primary
    }

    #[test]
    fn test_total_failure() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        let mut m1 = PathMember::new(1, "A", MemberRole::Primary);
        m1.health = MemberHealth::Faulted;
        let mut m2 = PathMember::new(2, "B", MemberRole::Secondary);
        m2.health = MemberHealth::Faulted;
        g.add_member(m1);
        g.add_member(m2);
        assert!(g.is_total_failure());
    }

    #[test]
    fn test_available_count() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        let mut m1 = PathMember::new(1, "A", MemberRole::Primary);
        m1.health = MemberHealth::Healthy;
        let mut m2 = PathMember::new(2, "B", MemberRole::Secondary);
        m2.health = MemberHealth::Faulted;
        g.add_member(m1);
        g.add_member(m2);
        assert_eq!(g.available_count(), 1);
    }

    #[test]
    fn test_manager_add_group() {
        let mut mgr = RedundancyManager::new();
        let id = mgr.add_group("TestGroup", FailoverStrategy::Automatic);
        assert_eq!(mgr.group_count(), 1);
        assert!(mgr.group(id).is_some());
    }

    #[test]
    fn test_manager_failed_groups() {
        let mut mgr = RedundancyManager::new();
        let id = mgr.add_group("Failing", FailoverStrategy::Automatic);
        let grp = mgr.group_mut(id).expect("should succeed in test");
        let mut m = PathMember::new(1, "A", MemberRole::Primary);
        m.health = MemberHealth::Faulted;
        grp.add_member(m);
        assert_eq!(mgr.failed_groups(), vec![id]);
    }

    #[test]
    fn test_failover_event_log_cap() {
        let mut g = RedundancyGroup::new(1, "G1", FailoverStrategy::Automatic);
        g.max_events = 2;
        let mut m1 = PathMember::new(1, "A", MemberRole::Primary);
        m1.health = MemberHealth::Healthy;
        m1.active = true;
        let mut m2 = PathMember::new(2, "B", MemberRole::Secondary);
        m2.health = MemberHealth::Healthy;
        g.add_member(m1);
        g.add_member(m2);

        // Generate 3 failover events
        g.update_health(1, MemberHealth::Faulted);
        g.update_health(1, MemberHealth::Healthy);
        g.update_health(2, MemberHealth::Faulted);
        // Only last 2 should be retained
        assert!(g.events().len() <= 2);
    }
}
