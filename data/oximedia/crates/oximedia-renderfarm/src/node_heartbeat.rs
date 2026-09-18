#![allow(dead_code)]
//! Heartbeat monitoring for render farm nodes.
//!
//! Tracks liveness of worker nodes through periodic heartbeats, detects stale
//! or failed nodes, and provides health summaries for the farm.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique identifier for a render node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(String);

impl NodeId {
    /// Creates a new node identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Resource snapshot sent with each heartbeat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeResources {
    /// CPU utilization percentage (0.0–100.0).
    pub cpu_percent: f64,
    /// Memory utilization percentage (0.0–100.0).
    pub memory_percent: f64,
    /// GPU utilization percentage (0.0–100.0), if applicable.
    pub gpu_percent: f64,
    /// Available disk space in bytes.
    pub disk_free_bytes: u64,
    /// Current number of active render tasks on this node.
    pub active_tasks: u32,
}

impl NodeResources {
    /// Creates a new resource snapshot.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cpu_percent: f64,
        memory_percent: f64,
        gpu_percent: f64,
        disk_free_bytes: u64,
        active_tasks: u32,
    ) -> Self {
        Self {
            cpu_percent,
            memory_percent,
            gpu_percent,
            disk_free_bytes,
            active_tasks,
        }
    }

    /// Returns `true` if the node appears overloaded.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.cpu_percent > 95.0 || self.memory_percent > 95.0
    }

    /// Returns `true` if disk space is critically low (< 1 GB).
    #[must_use]
    pub fn disk_critical(&self) -> bool {
        self.disk_free_bytes < 1_073_741_824
    }
}

impl Default for NodeResources {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            gpu_percent: 0.0,
            disk_free_bytes: u64::MAX,
            active_tasks: 0,
        }
    }
}

/// The observed status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLiveness {
    /// Node is responding normally.
    Alive,
    /// Node missed one or more heartbeats but is within the warning window.
    Suspect,
    /// Node has exceeded the dead threshold — treat as failed.
    Dead,
}

/// Configuration for heartbeat monitoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeartbeatConfig {
    /// Expected interval between heartbeats.
    pub interval: Duration,
    /// After this many missed intervals, the node is suspect.
    pub suspect_after_missed: u32,
    /// After this many missed intervals, the node is declared dead.
    pub dead_after_missed: u32,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            suspect_after_missed: 2,
            dead_after_missed: 5,
        }
    }
}

/// Internal record for a single node.
#[derive(Debug, Clone)]
struct NodeRecord {
    /// Last time a heartbeat was received.
    last_seen: Instant,
    /// Latest resource snapshot.
    resources: NodeResources,
    /// Total number of heartbeats received.
    heartbeat_count: u64,
}

/// Central heartbeat tracker for all farm nodes.
#[derive(Debug)]
pub struct HeartbeatTracker {
    /// Configuration.
    config: HeartbeatConfig,
    /// Per-node records.
    nodes: HashMap<NodeId, NodeRecord>,
}

impl HeartbeatTracker {
    /// Creates a new tracker with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HeartbeatConfig::default(),
            nodes: HashMap::new(),
        }
    }

    /// Creates a new tracker with custom configuration.
    #[must_use]
    pub fn with_config(config: HeartbeatConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Records a heartbeat from a node, registering it if new.
    pub fn record_heartbeat(&mut self, node_id: NodeId, resources: NodeResources) {
        let record = self.nodes.entry(node_id).or_insert_with(|| NodeRecord {
            last_seen: Instant::now(),
            resources: NodeResources::default(),
            heartbeat_count: 0,
        });
        record.last_seen = Instant::now();
        record.resources = resources;
        record.heartbeat_count += 1;
    }

    /// Determines the liveness status of a node.
    #[must_use]
    pub fn liveness(&self, node_id: &NodeId) -> NodeLiveness {
        let Some(record) = self.nodes.get(node_id) else {
            return NodeLiveness::Dead;
        };

        let elapsed = record.last_seen.elapsed();
        let suspect_threshold = self.config.interval * self.config.suspect_after_missed;
        let dead_threshold = self.config.interval * self.config.dead_after_missed;

        if elapsed >= dead_threshold {
            NodeLiveness::Dead
        } else if elapsed >= suspect_threshold {
            NodeLiveness::Suspect
        } else {
            NodeLiveness::Alive
        }
    }

    /// Returns the latest resource snapshot for a node, if available.
    #[must_use]
    pub fn resources(&self, node_id: &NodeId) -> Option<&NodeResources> {
        self.nodes.get(node_id).map(|r| &r.resources)
    }

    /// Returns the total heartbeat count for a node.
    #[must_use]
    pub fn heartbeat_count(&self, node_id: &NodeId) -> u64 {
        self.nodes.get(node_id).map_or(0, |r| r.heartbeat_count)
    }

    /// Returns the number of tracked nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns a list of node IDs whose status is `Dead`.
    #[must_use]
    pub fn dead_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .keys()
            .filter(|id| self.liveness(id) == NodeLiveness::Dead)
            .cloned()
            .collect()
    }

    /// Returns a list of node IDs whose status is `Alive`.
    #[must_use]
    pub fn alive_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .keys()
            .filter(|id| self.liveness(id) == NodeLiveness::Alive)
            .cloned()
            .collect()
    }

    /// Removes a node from tracking.
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        self.nodes.remove(node_id).is_some()
    }

    /// Returns a farm health summary.
    #[must_use]
    pub fn health_summary(&self) -> HealthSummary {
        let mut alive = 0u32;
        let mut suspect = 0u32;
        let mut dead = 0u32;
        let mut overloaded = 0u32;
        let mut disk_critical = 0u32;

        for (id, record) in &self.nodes {
            match self.liveness(id) {
                NodeLiveness::Alive => alive += 1,
                NodeLiveness::Suspect => suspect += 1,
                NodeLiveness::Dead => dead += 1,
            }
            if record.resources.is_overloaded() {
                overloaded += 1;
            }
            if record.resources.disk_critical() {
                disk_critical += 1;
            }
        }

        HealthSummary {
            total: self.nodes.len() as u32,
            alive,
            suspect,
            dead,
            overloaded,
            disk_critical,
        }
    }
}

impl Default for HeartbeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of farm node health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthSummary {
    /// Total number of tracked nodes.
    pub total: u32,
    /// Number of alive nodes.
    pub alive: u32,
    /// Number of suspect nodes.
    pub suspect: u32,
    /// Number of dead nodes.
    pub dead: u32,
    /// Number of nodes reporting overload.
    pub overloaded: u32,
    /// Number of nodes with critically low disk.
    pub disk_critical: u32,
}

impl HealthSummary {
    /// Returns `true` if all nodes are alive and none overloaded.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.dead == 0 && self.suspect == 0 && self.overloaded == 0 && self.disk_critical == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resources() -> NodeResources {
        NodeResources::new(50.0, 60.0, 30.0, 100_000_000_000, 2)
    }

    #[test]
    fn test_node_id() {
        let id = NodeId::new("node-1");
        assert_eq!(id.as_str(), "node-1");
    }

    #[test]
    fn test_resources_not_overloaded() {
        let r = make_resources();
        assert!(!r.is_overloaded());
    }

    #[test]
    fn test_resources_overloaded() {
        let r = NodeResources::new(96.0, 50.0, 0.0, 1_000_000_000_000, 0);
        assert!(r.is_overloaded());
    }

    #[test]
    fn test_resources_disk_critical() {
        let r = NodeResources::new(10.0, 10.0, 0.0, 500_000_000, 0);
        assert!(r.disk_critical());
    }

    #[test]
    fn test_resources_disk_ok() {
        let r = NodeResources::new(10.0, 10.0, 0.0, 100_000_000_000, 0);
        assert!(!r.disk_critical());
    }

    #[test]
    fn test_tracker_new() {
        let t = HeartbeatTracker::new();
        assert_eq!(t.node_count(), 0);
    }

    #[test]
    fn test_record_heartbeat() {
        let mut t = HeartbeatTracker::new();
        let id = NodeId::new("node-1");
        t.record_heartbeat(id.clone(), make_resources());
        assert_eq!(t.node_count(), 1);
        assert_eq!(t.heartbeat_count(&id), 1);
    }

    #[test]
    fn test_liveness_alive() {
        let mut t = HeartbeatTracker::new();
        let id = NodeId::new("node-1");
        t.record_heartbeat(id.clone(), make_resources());
        assert_eq!(t.liveness(&id), NodeLiveness::Alive);
    }

    #[test]
    fn test_liveness_unknown_is_dead() {
        let t = HeartbeatTracker::new();
        let id = NodeId::new("no-such-node");
        assert_eq!(t.liveness(&id), NodeLiveness::Dead);
    }

    #[test]
    fn test_resources_lookup() {
        let mut t = HeartbeatTracker::new();
        let id = NodeId::new("node-1");
        let r = make_resources();
        t.record_heartbeat(id.clone(), r);
        let got = t.resources(&id).expect("should succeed in test");
        assert!((got.cpu_percent - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove_node() {
        let mut t = HeartbeatTracker::new();
        let id = NodeId::new("node-1");
        t.record_heartbeat(id.clone(), make_resources());
        assert!(t.remove_node(&id));
        assert_eq!(t.node_count(), 0);
    }

    #[test]
    fn test_alive_nodes_list() {
        let mut t = HeartbeatTracker::new();
        t.record_heartbeat(NodeId::new("a"), make_resources());
        t.record_heartbeat(NodeId::new("b"), make_resources());
        let alive = t.alive_nodes();
        assert_eq!(alive.len(), 2);
    }

    #[test]
    fn test_health_summary_healthy() {
        let mut t = HeartbeatTracker::new();
        t.record_heartbeat(NodeId::new("x"), make_resources());
        let s = t.health_summary();
        assert_eq!(s.total, 1);
        assert_eq!(s.alive, 1);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_heartbeat_config_default() {
        let c = HeartbeatConfig::default();
        assert_eq!(c.interval, Duration::from_secs(30));
        assert_eq!(c.suspect_after_missed, 2);
        assert_eq!(c.dead_after_missed, 5);
    }

    #[test]
    fn test_multiple_heartbeats() {
        let mut t = HeartbeatTracker::new();
        let id = NodeId::new("node-1");
        t.record_heartbeat(id.clone(), make_resources());
        t.record_heartbeat(id.clone(), make_resources());
        t.record_heartbeat(id.clone(), make_resources());
        assert_eq!(t.heartbeat_count(&id), 3);
    }
}
