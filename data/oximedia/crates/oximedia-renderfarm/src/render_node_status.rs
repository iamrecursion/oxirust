#![allow(dead_code)]
//! Render node status tracking for the `OxiMedia` render farm.
//!
//! Provides a load-level enum, per-node status records with work-acceptance
//! logic, and a registry for fleet-wide status queries.

use std::collections::HashMap;

/// Qualitative load level of a render node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLoadLevel {
    /// Node is idle and ready for work.
    Idle,
    /// Node is lightly loaded (< 25 % utilisation).
    Light,
    /// Node is moderately loaded (25–74 %).
    Moderate,
    /// Node is heavily loaded (75–99 %).
    Heavy,
    /// Node is at capacity and cannot accept more work.
    Saturated,
}

impl NodeLoadLevel {
    /// Returns `true` when the node cannot take additional tasks.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        matches!(self, NodeLoadLevel::Heavy | NodeLoadLevel::Saturated)
    }

    /// Numeric utilisation midpoint (%) used for sorting/display.
    #[must_use]
    pub fn utilisation_midpoint(&self) -> u8 {
        match self {
            NodeLoadLevel::Idle => 0,
            NodeLoadLevel::Light => 12,
            NodeLoadLevel::Moderate => 50,
            NodeLoadLevel::Heavy => 87,
            NodeLoadLevel::Saturated => 100,
        }
    }

    /// Derive a `NodeLoadLevel` from a 0–100 utilisation percentage.
    #[must_use]
    pub fn from_utilisation(pct: u8) -> Self {
        match pct {
            0 => NodeLoadLevel::Idle,
            1..=24 => NodeLoadLevel::Light,
            25..=74 => NodeLoadLevel::Moderate,
            75..=99 => NodeLoadLevel::Heavy,
            _ => NodeLoadLevel::Saturated,
        }
    }
}

/// Status snapshot for a single render node.
#[derive(Debug, Clone)]
pub struct RenderNodeStatus {
    /// Unique node identifier.
    pub node_id: String,
    /// Human-readable hostname.
    pub hostname: String,
    /// Current load level.
    pub load_level: NodeLoadLevel,
    /// Number of tasks currently executing on this node.
    pub active_task_count: u32,
    /// Maximum concurrent tasks this node supports.
    pub max_tasks: u32,
    /// Whether the node is online and reachable.
    pub online: bool,
    /// Whether the node has been administratively disabled.
    pub disabled: bool,
    /// Last heartbeat timestamp (Unix epoch seconds).
    pub last_heartbeat_ts: u64,
}

impl RenderNodeStatus {
    /// Create a new node status record.
    #[must_use]
    pub fn new(node_id: &str, hostname: &str, max_tasks: u32) -> Self {
        Self {
            node_id: node_id.to_string(),
            hostname: hostname.to_string(),
            load_level: NodeLoadLevel::Idle,
            active_task_count: 0,
            max_tasks,
            online: true,
            disabled: false,
            last_heartbeat_ts: 0,
        }
    }

    /// Returns `true` when the node is online, not disabled, and not overloaded.
    #[must_use]
    pub fn is_accepting_work(&self) -> bool {
        self.online && !self.disabled && !self.load_level.is_overloaded()
    }

    /// Returns the number of task slots still available on this node.
    #[must_use]
    pub fn available_slots(&self) -> u32 {
        self.max_tasks.saturating_sub(self.active_task_count)
    }

    /// Utilisation percentage (0–100).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn utilisation_pct(&self) -> u8 {
        if self.max_tasks == 0 {
            return 100;
        }
        ((self.active_task_count as f32 / self.max_tasks as f32) * 100.0) as u8
    }

    /// Update heartbeat timestamp.
    pub fn record_heartbeat(&mut self, ts: u64) {
        self.last_heartbeat_ts = ts;
    }

    /// Increment active task count and recompute load level.
    pub fn assign_task(&mut self) {
        if self.active_task_count < self.max_tasks {
            self.active_task_count += 1;
            self.load_level = NodeLoadLevel::from_utilisation(self.utilisation_pct());
        }
    }

    /// Decrement active task count and recompute load level.
    pub fn complete_task(&mut self) {
        if self.active_task_count > 0 {
            self.active_task_count -= 1;
            self.load_level = NodeLoadLevel::from_utilisation(self.utilisation_pct());
        }
    }
}

/// Fleet-wide registry of render node statuses.
#[derive(Debug, Default)]
pub struct NodeStatusRegistry {
    nodes: HashMap<String, RenderNodeStatus>,
}

impl NodeStatusRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Insert or replace the status for a node.
    pub fn update(&mut self, status: RenderNodeStatus) {
        self.nodes.insert(status.node_id.clone(), status);
    }

    /// Return a reference to a specific node's status.
    #[must_use]
    pub fn get(&self, node_id: &str) -> Option<&RenderNodeStatus> {
        self.nodes.get(node_id)
    }

    /// Return all nodes that are currently accepting work.
    #[must_use]
    pub fn available_nodes(&self) -> Vec<&RenderNodeStatus> {
        self.nodes
            .values()
            .filter(|n| n.is_accepting_work())
            .collect()
    }

    /// Total number of nodes tracked (regardless of state).
    #[must_use]
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the node with the most available task slots, or `None` if all are busy.
    #[must_use]
    pub fn least_loaded_node(&self) -> Option<&RenderNodeStatus> {
        self.nodes
            .values()
            .filter(|n| n.is_accepting_work())
            .max_by_key(|n| n.available_slots())
    }

    /// Remove a node from the registry.
    pub fn remove(&mut self, node_id: &str) -> Option<RenderNodeStatus> {
        self.nodes.remove(node_id)
    }

    /// Count nodes that are online and available.
    #[must_use]
    pub fn online_count(&self) -> usize {
        self.nodes.values().filter(|n| n.online).count()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn idle_node(id: &str) -> RenderNodeStatus {
        RenderNodeStatus::new(id, &format!("{id}.farm.local"), 8)
    }

    #[test]
    fn test_load_level_idle_not_overloaded() {
        assert!(!NodeLoadLevel::Idle.is_overloaded());
    }

    #[test]
    fn test_load_level_heavy_is_overloaded() {
        assert!(NodeLoadLevel::Heavy.is_overloaded());
    }

    #[test]
    fn test_load_level_saturated_is_overloaded() {
        assert!(NodeLoadLevel::Saturated.is_overloaded());
    }

    #[test]
    fn test_load_level_moderate_not_overloaded() {
        assert!(!NodeLoadLevel::Moderate.is_overloaded());
    }

    #[test]
    fn test_load_level_from_utilisation_zero() {
        assert_eq!(NodeLoadLevel::from_utilisation(0), NodeLoadLevel::Idle);
    }

    #[test]
    fn test_load_level_from_utilisation_heavy() {
        assert_eq!(NodeLoadLevel::from_utilisation(80), NodeLoadLevel::Heavy);
    }

    #[test]
    fn test_load_level_from_utilisation_saturated() {
        assert_eq!(
            NodeLoadLevel::from_utilisation(100),
            NodeLoadLevel::Saturated
        );
    }

    #[test]
    fn test_node_accepting_work_when_idle() {
        let n = idle_node("node-01");
        assert!(n.is_accepting_work());
    }

    #[test]
    fn test_node_not_accepting_work_when_offline() {
        let mut n = idle_node("node-02");
        n.online = false;
        assert!(!n.is_accepting_work());
    }

    #[test]
    fn test_node_not_accepting_work_when_disabled() {
        let mut n = idle_node("node-03");
        n.disabled = true;
        assert!(!n.is_accepting_work());
    }

    #[test]
    fn test_node_available_slots_full() {
        let n = idle_node("node-04");
        assert_eq!(n.available_slots(), 8);
    }

    #[test]
    fn test_node_assign_and_complete_task() {
        let mut n = idle_node("node-05");
        n.assign_task();
        assert_eq!(n.active_task_count, 1);
        n.complete_task();
        assert_eq!(n.active_task_count, 0);
    }

    #[test]
    fn test_registry_empty_initially() {
        let r = NodeStatusRegistry::new();
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn test_registry_update_and_count() {
        let mut r = NodeStatusRegistry::new();
        r.update(idle_node("node-01"));
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn test_registry_available_nodes() {
        let mut r = NodeStatusRegistry::new();
        r.update(idle_node("node-01"));
        let mut n2 = idle_node("node-02");
        n2.online = false;
        r.update(n2);
        assert_eq!(r.available_nodes().len(), 1);
    }

    #[test]
    fn test_registry_least_loaded_node() {
        let mut r = NodeStatusRegistry::new();
        let mut busy = idle_node("node-busy");
        for _ in 0..7 {
            busy.assign_task();
        }
        r.update(busy);
        r.update(idle_node("node-free"));
        let best = r.least_loaded_node().expect("should succeed in test");
        assert_eq!(best.node_id, "node-free");
    }

    #[test]
    fn test_registry_remove() {
        let mut r = NodeStatusRegistry::new();
        r.update(idle_node("node-x"));
        let removed = r.remove("node-x");
        assert!(removed.is_some());
        assert_eq!(r.count(), 0);
    }
}
