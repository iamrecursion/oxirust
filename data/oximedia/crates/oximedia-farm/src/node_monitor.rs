//! Real-time monitoring of individual farm nodes and pooled capacity tracking.
//!
//! Provides `FarmNode` with availability and utilisation queries, and
//! `FarmNodePool` for aggregate capacity management across the cluster.

#![allow(dead_code)]

use std::collections::HashMap;

/// Operational status of a single farm node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is reachable and accepting work.
    Online,
    /// Node is reachable but not accepting new tasks (draining).
    Draining,
    /// Node has exceeded its resource thresholds.
    Overloaded,
    /// Node is unreachable or has failed health checks.
    Offline,
    /// Node is in the process of starting up.
    Initialising,
}

impl NodeStatus {
    /// Returns `true` if the node can accept new tasks.
    #[must_use]
    pub fn is_available(self) -> bool {
        matches!(self, Self::Online)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Draining => "draining",
            Self::Overloaded => "overloaded",
            Self::Offline => "offline",
            Self::Initialising => "initialising",
        }
    }
}

/// Snapshot of a single farm node's state and resource usage.
#[derive(Debug, Clone)]
pub struct FarmNode {
    /// Unique node identifier.
    pub node_id: String,
    /// Current operational status.
    pub status: NodeStatus,
    /// Number of CPU cores available on this node.
    pub cpu_cores: u32,
    /// Currently used CPU cores.
    pub cpu_used: u32,
    /// Total RAM in megabytes.
    pub memory_mb: u64,
    /// Currently used RAM in megabytes.
    pub memory_used_mb: u64,
    /// Number of tasks currently running on this node.
    pub running_tasks: u32,
    /// Maximum concurrent tasks this node can handle.
    pub max_tasks: u32,
}

impl FarmNode {
    /// Create a new farm node.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        status: NodeStatus,
        cpu_cores: u32,
        memory_mb: u64,
        max_tasks: u32,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            status,
            cpu_cores,
            cpu_used: 0,
            memory_mb,
            memory_used_mb: 0,
            running_tasks: 0,
            max_tasks,
        }
    }

    /// Returns `true` if the node is online and has task capacity remaining.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.status.is_available() && self.running_tasks < self.max_tasks
    }

    /// CPU utilisation as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when the node has no CPU cores.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.cpu_cores == 0 {
            return 0.0;
        }
        f64::from(self.cpu_used) / f64::from(self.cpu_cores)
    }

    /// Memory utilisation as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when total memory is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_mb == 0 {
            return 0.0;
        }
        self.memory_used_mb as f64 / self.memory_mb as f64
    }

    /// Free task slots remaining on this node.
    #[must_use]
    pub fn free_slots(&self) -> u32 {
        self.max_tasks.saturating_sub(self.running_tasks)
    }
}

/// A pool of farm nodes providing aggregate capacity and availability queries.
#[derive(Debug, Clone, Default)]
pub struct FarmNodePool {
    nodes: HashMap<String, FarmNode>,
}

impl FarmNodePool {
    /// Create an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a node in the pool.
    pub fn add_node(&mut self, node: FarmNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node by ID, returning it if it was present.
    pub fn remove_node(&mut self, node_id: &str) -> Option<FarmNode> {
        self.nodes.remove(node_id)
    }

    /// Returns references to all nodes that are currently available.
    #[must_use]
    pub fn available_nodes(&self) -> Vec<&FarmNode> {
        self.nodes.values().filter(|n| n.is_available()).collect()
    }

    /// Total task capacity across all nodes in the pool.
    #[must_use]
    pub fn total_capacity(&self) -> u32 {
        self.nodes.values().map(|n| n.max_tasks).sum()
    }

    /// Total number of currently running tasks across all nodes.
    #[must_use]
    pub fn running_tasks(&self) -> u32 {
        self.nodes.values().map(|n| n.running_tasks).sum()
    }

    /// Number of nodes currently in the pool.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Retrieve a reference to a node by ID.
    #[must_use]
    pub fn get(&self, node_id: &str) -> Option<&FarmNode> {
        self.nodes.get(node_id)
    }

    /// Retrieve a mutable reference to a node by ID.
    pub fn get_mut(&mut self, node_id: &str) -> Option<&mut FarmNode> {
        self.nodes.get_mut(node_id)
    }

    /// Average CPU utilisation across all nodes (0.0 if pool is empty).
    #[must_use]
    pub fn average_utilization(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.nodes.values().map(FarmNode::utilization).sum();
        sum / self.nodes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn online_node(id: &str, cores: u32, max_tasks: u32) -> FarmNode {
        FarmNode::new(id, NodeStatus::Online, cores, 4096, max_tasks)
    }

    #[test]
    fn test_node_status_is_available_online() {
        assert!(NodeStatus::Online.is_available());
    }

    #[test]
    fn test_node_status_is_available_offline() {
        assert!(!NodeStatus::Offline.is_available());
    }

    #[test]
    fn test_node_status_is_available_draining() {
        assert!(!NodeStatus::Draining.is_available());
    }

    #[test]
    fn test_node_status_label() {
        assert_eq!(NodeStatus::Overloaded.label(), "overloaded");
        assert_eq!(NodeStatus::Initialising.label(), "initialising");
    }

    #[test]
    fn test_farm_node_is_available_with_capacity() {
        let node = online_node("n1", 8, 4);
        assert!(node.is_available());
    }

    #[test]
    fn test_farm_node_is_available_full() {
        let mut node = online_node("n1", 8, 2);
        node.running_tasks = 2;
        assert!(!node.is_available());
    }

    #[test]
    fn test_farm_node_utilization_zero_cores() {
        let node = FarmNode::new("n0", NodeStatus::Online, 0, 1024, 1);
        assert!((node.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_farm_node_utilization_partial() {
        let mut node = online_node("n1", 8, 4);
        node.cpu_used = 4;
        assert!((node.utilization() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_farm_node_free_slots() {
        let mut node = online_node("n1", 8, 10);
        node.running_tasks = 3;
        assert_eq!(node.free_slots(), 7);
    }

    #[test]
    fn test_pool_starts_empty() {
        let pool = FarmNodePool::new();
        assert_eq!(pool.node_count(), 0);
    }

    #[test]
    fn test_pool_add_node() {
        let mut pool = FarmNodePool::new();
        pool.add_node(online_node("n1", 8, 4));
        assert_eq!(pool.node_count(), 1);
    }

    #[test]
    fn test_pool_available_nodes() {
        let mut pool = FarmNodePool::new();
        pool.add_node(online_node("n1", 8, 4));
        let mut offline = online_node("n2", 4, 2);
        offline.status = NodeStatus::Offline;
        pool.add_node(offline);
        assert_eq!(pool.available_nodes().len(), 1);
    }

    #[test]
    fn test_pool_total_capacity() {
        let mut pool = FarmNodePool::new();
        pool.add_node(online_node("n1", 8, 6));
        pool.add_node(online_node("n2", 4, 4));
        assert_eq!(pool.total_capacity(), 10);
    }

    #[test]
    fn test_pool_remove_node() {
        let mut pool = FarmNodePool::new();
        pool.add_node(online_node("n1", 8, 4));
        let removed = pool.remove_node("n1");
        assert!(removed.is_some());
        assert_eq!(pool.node_count(), 0);
    }

    #[test]
    fn test_pool_average_utilization_empty() {
        let pool = FarmNodePool::new();
        assert!((pool.average_utilization() - 0.0).abs() < f64::EPSILON);
    }
}
