//! Node heartbeat management for distributed cluster health monitoring.
//!
//! Tracks heartbeat intervals, detects failures via configurable timeouts,
//! and maintains per-node liveness state.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Liveness state of a node
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLiveness {
    /// Node is alive and sending heartbeats
    Alive,
    /// Node has not sent a heartbeat recently; suspected dead
    Suspected,
    /// Node confirmed dead (exceeded dead threshold)
    Dead,
}

/// Configuration for the heartbeat detector
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval at which nodes should send heartbeats
    pub heartbeat_interval: Duration,
    /// Timeout after which a node is suspected
    pub suspect_timeout: Duration,
    /// Timeout after which a node is declared dead
    pub dead_timeout: Duration,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(5),
            suspect_timeout: Duration::from_secs(15),
            dead_timeout: Duration::from_secs(30),
        }
    }
}

/// Per-node heartbeat record
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NodeHeartbeatRecord {
    /// Node identifier
    pub node_id: String,
    /// Time the node was first registered
    pub registered_at: Instant,
    /// Time of the last received heartbeat
    pub last_heartbeat: Instant,
    /// Number of missed consecutive heartbeats
    pub missed_count: u32,
    /// Current liveness state
    pub liveness: NodeLiveness,
}

impl NodeHeartbeatRecord {
    /// Create a new record; `last_heartbeat` is set to now.
    #[allow(dead_code)]
    pub fn new(node_id: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            node_id: node_id.into(),
            registered_at: now,
            last_heartbeat: now,
            missed_count: 0,
            liveness: NodeLiveness::Alive,
        }
    }

    /// Update the record on heartbeat receipt.
    #[allow(dead_code)]
    pub fn record_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
        self.missed_count = 0;
        self.liveness = NodeLiveness::Alive;
    }

    /// Elapsed time since last heartbeat.
    #[allow(dead_code)]
    #[must_use]
    pub fn elapsed_since_heartbeat(&self) -> Duration {
        self.last_heartbeat.elapsed()
    }
}

/// Central heartbeat tracker for all cluster nodes
#[allow(dead_code)]
pub struct HeartbeatTracker {
    config: HeartbeatConfig,
    records: HashMap<String, NodeHeartbeatRecord>,
}

impl HeartbeatTracker {
    /// Create a new tracker with the given configuration.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(config: HeartbeatConfig) -> Self {
        Self {
            config,
            records: HashMap::new(),
        }
    }

    /// Register a new node.
    #[allow(dead_code)]
    pub fn register(&mut self, node_id: impl Into<String>) {
        let id = node_id.into();
        self.records
            .entry(id.clone())
            .or_insert_with(|| NodeHeartbeatRecord::new(id));
    }

    /// Deregister a node.
    #[allow(dead_code)]
    pub fn deregister(&mut self, node_id: &str) {
        self.records.remove(node_id);
    }

    /// Record a heartbeat from a node.
    ///
    /// Returns `false` if the node is not registered.
    #[allow(dead_code)]
    pub fn heartbeat(&mut self, node_id: &str) -> bool {
        if let Some(record) = self.records.get_mut(node_id) {
            record.record_heartbeat();
            true
        } else {
            false
        }
    }

    /// Run a single check pass over all registered nodes and update liveness.
    ///
    /// Returns the list of node IDs whose state changed.
    #[allow(dead_code)]
    pub fn check(&mut self) -> Vec<(String, NodeLiveness)> {
        let suspect = self.config.suspect_timeout;
        let dead = self.config.dead_timeout;

        let mut changes = Vec::new();

        for record in self.records.values_mut() {
            let elapsed = record.elapsed_since_heartbeat();
            let new_liveness = if elapsed >= dead {
                NodeLiveness::Dead
            } else if elapsed >= suspect {
                NodeLiveness::Suspected
            } else {
                NodeLiveness::Alive
            };

            if new_liveness != record.liveness {
                if new_liveness != NodeLiveness::Alive {
                    record.missed_count += 1;
                }
                record.liveness = new_liveness;
                changes.push((record.node_id.clone(), new_liveness));
            }
        }

        changes
    }

    /// Return the current liveness of a node.
    #[allow(dead_code)]
    #[must_use]
    pub fn liveness(&self, node_id: &str) -> Option<NodeLiveness> {
        self.records.get(node_id).map(|r| r.liveness)
    }

    /// Return all alive nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn alive_nodes(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|r| r.liveness == NodeLiveness::Alive)
            .map(|r| r.node_id.as_str())
            .collect()
    }

    /// Return all dead nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn dead_nodes(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|r| r.liveness == NodeLiveness::Dead)
            .map(|r| r.node_id.as_str())
            .collect()
    }

    /// Return the number of registered nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tracker() -> HeartbeatTracker {
        HeartbeatTracker::new(HeartbeatConfig::default())
    }

    #[test]
    fn test_register_adds_node() {
        let mut tracker = default_tracker();
        tracker.register("node-0");
        assert_eq!(tracker.node_count(), 1);
    }

    #[test]
    fn test_deregister_removes_node() {
        let mut tracker = default_tracker();
        tracker.register("node-0");
        tracker.deregister("node-0");
        assert_eq!(tracker.node_count(), 0);
    }

    #[test]
    fn test_heartbeat_unknown_node_returns_false() {
        let mut tracker = default_tracker();
        assert!(!tracker.heartbeat("ghost"));
    }

    #[test]
    fn test_heartbeat_known_node_returns_true() {
        let mut tracker = default_tracker();
        tracker.register("node-1");
        assert!(tracker.heartbeat("node-1"));
    }

    #[test]
    fn test_initial_liveness_is_alive() {
        let mut tracker = default_tracker();
        tracker.register("node-2");
        assert_eq!(tracker.liveness("node-2"), Some(NodeLiveness::Alive));
    }

    #[test]
    fn test_liveness_unknown_node_is_none() {
        let tracker = default_tracker();
        assert!(tracker.liveness("missing").is_none());
    }

    #[test]
    fn test_alive_nodes_list() {
        let mut tracker = default_tracker();
        tracker.register("n0");
        tracker.register("n1");
        let alive = tracker.alive_nodes();
        assert_eq!(alive.len(), 2);
    }

    #[test]
    fn test_dead_nodes_initially_empty() {
        let mut tracker = default_tracker();
        tracker.register("n0");
        assert!(tracker.dead_nodes().is_empty());
    }

    #[test]
    fn test_node_heartbeat_record_new() {
        let record = NodeHeartbeatRecord::new("node-x");
        assert_eq!(record.liveness, NodeLiveness::Alive);
        assert_eq!(record.missed_count, 0);
    }

    #[test]
    fn test_record_heartbeat_resets_missed_count() {
        let mut record = NodeHeartbeatRecord::new("node-y");
        record.missed_count = 5;
        record.liveness = NodeLiveness::Suspected;
        record.record_heartbeat();
        assert_eq!(record.missed_count, 0);
        assert_eq!(record.liveness, NodeLiveness::Alive);
    }

    #[test]
    fn test_elapsed_since_heartbeat_is_small() {
        let record = NodeHeartbeatRecord::new("node-z");
        assert!(record.elapsed_since_heartbeat() < Duration::from_millis(100));
    }

    #[test]
    fn test_check_no_changes_for_fresh_nodes() {
        let mut tracker = default_tracker();
        tracker.register("n0");
        tracker.register("n1");
        let changes = tracker.check();
        assert!(changes.is_empty());
    }

    #[test]
    fn test_heartbeat_config_default_values() {
        let cfg = HeartbeatConfig::default();
        assert_eq!(cfg.heartbeat_interval, Duration::from_secs(5));
        assert_eq!(cfg.suspect_timeout, Duration::from_secs(15));
        assert_eq!(cfg.dead_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_suspect_before_dead_threshold() {
        let cfg = HeartbeatConfig {
            suspect_timeout: Duration::from_secs(5),
            dead_timeout: Duration::from_secs(30),
            ..HeartbeatConfig::default()
        };
        assert!(cfg.suspect_timeout < cfg.dead_timeout);
    }
}
