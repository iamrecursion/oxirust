//! Node health monitoring for distributed systems.
//!
//! This module provides health checking and status tracking for
//! worker nodes in the distributed encoding cluster. It supports
//! configurable check intervals, failure thresholds, and automatic
//! node status transitions.

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

/// Health status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    /// Node is healthy and accepting work
    Healthy,
    /// Node is responsive but showing warning signs
    Degraded,
    /// Node has not responded within expected interval
    Suspect,
    /// Node is confirmed unreachable
    Unreachable,
    /// Node is undergoing maintenance
    Maintenance,
    /// Node is draining (finishing current work, not accepting new)
    Draining,
}

impl HealthStatus {
    /// Returns true if the node can accept new work.
    #[must_use]
    pub fn can_accept_work(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Returns true if the node is considered alive (responsive).
    #[must_use]
    pub fn is_alive(&self) -> bool {
        !matches!(self, Self::Unreachable)
    }
}

/// Configuration for health checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthCheckConfig {
    /// Interval between health checks in seconds
    pub check_interval_secs: u64,
    /// Number of consecutive failures before marking unreachable
    pub failure_threshold: u32,
    /// Number of consecutive successes to recover from degraded
    pub recovery_threshold: u32,
    /// Timeout for a single health check in milliseconds
    pub timeout_ms: u64,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 10,
            failure_threshold: 3,
            recovery_threshold: 2,
            timeout_ms: 5000,
        }
    }
}

/// Health check result from probing a single node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeHealthCheck {
    /// Node that was checked
    pub node_id: Uuid,
    /// Whether the check succeeded
    pub success: bool,
    /// Response latency in milliseconds
    pub latency_ms: u64,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// Number of active tasks on the node
    pub active_tasks: u32,
    /// Unix timestamp of the check
    pub checked_at: i64,
    /// Optional error message if the check failed
    pub error: Option<String>,
}

impl NodeHealthCheck {
    /// Creates a successful health check result.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        node_id: Uuid,
        latency_ms: u64,
        cpu_utilization: f64,
        memory_utilization: f64,
        active_tasks: u32,
        checked_at: i64,
    ) -> Self {
        Self {
            node_id,
            success: true,
            latency_ms,
            cpu_utilization,
            memory_utilization,
            active_tasks,
            checked_at,
            error: None,
        }
    }

    /// Creates a failed health check result.
    #[must_use]
    pub fn failure(node_id: Uuid, error: &str, checked_at: i64) -> Self {
        Self {
            node_id,
            success: false,
            latency_ms: 0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            active_tasks: 0,
            checked_at,
            error: Some(error.to_string()),
        }
    }

    /// Returns true if the node appears overloaded.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        self.cpu_utilization > 0.9 || self.memory_utilization > 0.9
    }
}

/// Tracked state for a single node in the registry.
#[derive(Debug, Clone)]
struct NodeState {
    /// Current health status
    status: HealthStatus,
    /// Consecutive failure count
    consecutive_failures: u32,
    /// Consecutive success count
    consecutive_successes: u32,
    /// Last health check result
    last_check: Option<NodeHealthCheck>,
    /// Timestamp of the last status change
    status_changed_at: i64,
}

impl NodeState {
    fn new() -> Self {
        Self {
            status: HealthStatus::Healthy,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_check: None,
            status_changed_at: 0,
        }
    }
}

/// Registry tracking health state for all nodes in the cluster.
///
/// Processes health check results and automatically transitions
/// node statuses based on configurable thresholds.
#[derive(Debug)]
pub struct HealthRegistry {
    /// Configuration
    config: HealthCheckConfig,
    /// Per-node tracked state
    nodes: HashMap<Uuid, NodeState>,
}

impl HealthRegistry {
    /// Creates a new health registry with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HealthCheckConfig::default(),
            nodes: HashMap::new(),
        }
    }

    /// Creates a new health registry with custom configuration.
    #[must_use]
    pub fn with_config(config: HealthCheckConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Registers a new node in the registry.
    pub fn register_node(&mut self, node_id: Uuid) {
        self.nodes.entry(node_id).or_insert_with(NodeState::new);
    }

    /// Removes a node from the registry.
    pub fn unregister_node(&mut self, node_id: &Uuid) -> bool {
        self.nodes.remove(node_id).is_some()
    }

    /// Processes a health check result and updates node status.
    pub fn process_check(&mut self, check: NodeHealthCheck) {
        let node = self
            .nodes
            .entry(check.node_id)
            .or_insert_with(NodeState::new);

        if check.success {
            node.consecutive_failures = 0;
            node.consecutive_successes += 1;

            if check.is_overloaded() {
                node.consecutive_successes = 0;
                self.transition(check.node_id, HealthStatus::Degraded, check.checked_at);
            } else if node.consecutive_successes >= self.config.recovery_threshold {
                // Only recover if currently degraded or suspect
                if matches!(node.status, HealthStatus::Degraded | HealthStatus::Suspect) {
                    self.transition(check.node_id, HealthStatus::Healthy, check.checked_at);
                }
            }
        } else {
            node.consecutive_successes = 0;
            node.consecutive_failures += 1;

            if node.consecutive_failures >= self.config.failure_threshold {
                self.transition(check.node_id, HealthStatus::Unreachable, check.checked_at);
            } else if node.consecutive_failures >= 1 {
                self.transition(check.node_id, HealthStatus::Suspect, check.checked_at);
            }
        }

        // Update last check (re-borrow after transitions)
        if let Some(ns) = self.nodes.get_mut(&check.node_id) {
            ns.last_check = Some(check);
        }
    }

    /// Manually sets a node to maintenance mode.
    pub fn set_maintenance(&mut self, node_id: Uuid, now: i64) {
        self.register_node(node_id);
        self.transition(node_id, HealthStatus::Maintenance, now);
    }

    /// Manually sets a node to draining mode.
    pub fn set_draining(&mut self, node_id: Uuid, now: i64) {
        self.register_node(node_id);
        self.transition(node_id, HealthStatus::Draining, now);
    }

    /// Gets the current status of a node.
    #[must_use]
    pub fn get_status(&self, node_id: &Uuid) -> Option<HealthStatus> {
        self.nodes.get(node_id).map(|n| n.status)
    }

    /// Returns all nodes that can accept work.
    #[must_use]
    pub fn available_nodes(&self) -> Vec<Uuid> {
        self.nodes
            .iter()
            .filter(|(_, state)| state.status.can_accept_work())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns total number of registered nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns nodes in a specific status.
    #[must_use]
    pub fn nodes_with_status(&self, status: HealthStatus) -> Vec<Uuid> {
        self.nodes
            .iter()
            .filter(|(_, state)| state.status == status)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Transitions a node to a new status.
    fn transition(&mut self, node_id: Uuid, new_status: HealthStatus, now: i64) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            if node.status != new_status {
                node.status = new_status;
                node.status_changed_at = now;
            }
        }
    }
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_health_status_can_accept_work() {
        assert!(HealthStatus::Healthy.can_accept_work());
        assert!(HealthStatus::Degraded.can_accept_work());
        assert!(!HealthStatus::Unreachable.can_accept_work());
        assert!(!HealthStatus::Maintenance.can_accept_work());
        assert!(!HealthStatus::Draining.can_accept_work());
    }

    #[test]
    fn test_health_status_is_alive() {
        assert!(HealthStatus::Healthy.is_alive());
        assert!(HealthStatus::Degraded.is_alive());
        assert!(HealthStatus::Suspect.is_alive());
        assert!(!HealthStatus::Unreachable.is_alive());
    }

    #[test]
    fn test_health_check_config_defaults() {
        let cfg = HealthCheckConfig::default();
        assert_eq!(cfg.check_interval_secs, 10);
        assert_eq!(cfg.failure_threshold, 3);
        assert_eq!(cfg.recovery_threshold, 2);
    }

    #[test]
    fn test_node_health_check_success() {
        let id = nid();
        let check = NodeHealthCheck::success(id, 15, 0.5, 0.6, 2, 1000);
        assert!(check.success);
        assert_eq!(check.latency_ms, 15);
        assert!(!check.is_overloaded());
    }

    #[test]
    fn test_node_health_check_failure() {
        let id = nid();
        let check = NodeHealthCheck::failure(id, "timeout", 1000);
        assert!(!check.success);
        assert_eq!(check.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_overloaded_detection() {
        let check = NodeHealthCheck::success(nid(), 100, 0.95, 0.5, 5, 1000);
        assert!(check.is_overloaded());
        let check2 = NodeHealthCheck::success(nid(), 10, 0.5, 0.95, 1, 1000);
        assert!(check2.is_overloaded());
    }

    #[test]
    fn test_registry_register_and_status() {
        let mut reg = HealthRegistry::new();
        let id = nid();
        reg.register_node(id);
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Healthy));
        assert_eq!(reg.node_count(), 1);
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = HealthRegistry::new();
        let id = nid();
        reg.register_node(id);
        assert!(reg.unregister_node(&id));
        assert!(reg.get_status(&id).is_none());
    }

    #[test]
    fn test_registry_failure_escalation() {
        let config = HealthCheckConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_config(config);
        let id = nid();
        reg.register_node(id);

        // First failure => Suspect
        reg.process_check(NodeHealthCheck::failure(id, "err", 100));
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Suspect));

        // Second failure => Unreachable (threshold = 2)
        reg.process_check(NodeHealthCheck::failure(id, "err", 200));
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Unreachable));
    }

    #[test]
    fn test_registry_recovery() {
        let config = HealthCheckConfig {
            recovery_threshold: 2,
            ..Default::default()
        };
        let mut reg = HealthRegistry::with_config(config);
        let id = nid();
        reg.register_node(id);

        // Make it degraded
        reg.process_check(NodeHealthCheck::success(id, 10, 0.95, 0.5, 5, 100));
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Degraded));

        // First success — not enough to recover
        reg.process_check(NodeHealthCheck::success(id, 10, 0.5, 0.5, 2, 200));
        // Still degraded (need 2 consecutive successes)
        // Actually after the overloaded check sets it to Degraded, consecutive_successes
        // resets. Let's verify:
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Degraded));

        // Second success => recovery
        reg.process_check(NodeHealthCheck::success(id, 10, 0.5, 0.5, 2, 300));
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Healthy));
    }

    #[test]
    fn test_available_nodes() {
        let mut reg = HealthRegistry::new();
        let h = nid();
        let u = nid();
        reg.register_node(h);
        reg.register_node(u);
        reg.set_maintenance(u, 100);
        let available = reg.available_nodes();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0], h);
    }

    #[test]
    fn test_set_maintenance() {
        let mut reg = HealthRegistry::new();
        let id = nid();
        reg.set_maintenance(id, 100);
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Maintenance));
    }

    #[test]
    fn test_set_draining() {
        let mut reg = HealthRegistry::new();
        let id = nid();
        reg.set_draining(id, 100);
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Draining));
        assert!(!HealthStatus::Draining.can_accept_work());
    }

    #[test]
    fn test_nodes_with_status() {
        let mut reg = HealthRegistry::new();
        let a = nid();
        let b = nid();
        let c = nid();
        reg.register_node(a);
        reg.register_node(b);
        reg.register_node(c);
        reg.set_maintenance(c, 100);
        let healthy = reg.nodes_with_status(HealthStatus::Healthy);
        assert_eq!(healthy.len(), 2);
        let maint = reg.nodes_with_status(HealthStatus::Maintenance);
        assert_eq!(maint.len(), 1);
    }

    #[test]
    fn test_overloaded_marks_degraded() {
        let mut reg = HealthRegistry::new();
        let id = nid();
        reg.register_node(id);
        reg.process_check(NodeHealthCheck::success(id, 50, 0.95, 0.4, 10, 100));
        assert_eq!(reg.get_status(&id), Some(HealthStatus::Degraded));
    }
}
