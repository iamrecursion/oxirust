//! Node health monitoring for the render farm.
//!
//! Tracks per-node health metrics, heartbeat freshness, and overall
//! farm health status.

/// The operational health status of a farm node.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Node is fully operational.
    Healthy,
    /// Node is functioning but with reduced performance.
    Degraded {
        /// Human-readable reason for degradation.
        reason: String,
    },
    /// Node is not accepting work.
    Unavailable {
        /// Unix timestamp (seconds) when the node became unavailable.
        since: u64,
    },
    /// Node is offline for planned maintenance.
    Maintenance,
}

impl HealthStatus {
    /// Return `true` if the node can currently accept jobs.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded { .. })
    }
}

/// A snapshot of a farm node's health metrics.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// Unique node identifier.
    pub node_id: u64,
    /// Human-readable hostname.
    pub hostname: String,
    /// Current health status.
    pub status: HealthStatus,
    /// CPU temperature in degrees Celsius.
    pub cpu_temp_c: f64,
    /// Memory currently in use (GB).
    pub memory_used_gb: f64,
    /// Free disk space (GB).
    pub disk_free_gb: f64,
    /// Unix timestamp (seconds) of the last received heartbeat.
    pub last_heartbeat: u64,
    /// Total number of jobs successfully completed by this node.
    pub jobs_completed: u64,
    /// Total number of jobs that failed on this node.
    pub jobs_failed: u64,
}

impl NodeHealth {
    /// Create a new `NodeHealth` with sensible defaults.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(node_id: u64, hostname: &str) -> Self {
        Self {
            node_id,
            hostname: hostname.to_string(),
            status: HealthStatus::Healthy,
            cpu_temp_c: 0.0,
            memory_used_gb: 0.0,
            disk_free_gb: 0.0,
            last_heartbeat: 0,
            jobs_completed: 0,
            jobs_failed: 0,
        }
    }

    /// Return `true` if the node can accept new jobs.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.status.is_operational()
    }

    /// Fraction of jobs that failed: `jobs_failed / (jobs_completed + jobs_failed)`.
    /// Returns 0.0 when no jobs have been processed.
    #[allow(dead_code)]
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        let total = self.jobs_completed + self.jobs_failed;
        if total == 0 {
            return 0.0;
        }
        self.jobs_failed as f64 / total as f64
    }

    /// Seconds since the last heartbeat was received.
    #[allow(dead_code)]
    #[must_use]
    pub fn heartbeat_age_s(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_heartbeat)
    }

    /// Return `true` if the last heartbeat is older than `timeout_s` seconds.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_stale(&self, now: u64, timeout_s: u64) -> bool {
        self.heartbeat_age_s(now) > timeout_s
    }
}

/// Thresholds used to classify a node as unhealthy.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    /// Maximum acceptable CPU temperature (°C).
    pub max_cpu_temp: f64,
    /// Maximum acceptable memory utilisation percentage (0-100).
    pub max_memory_pct: f64,
    /// Minimum acceptable free disk space (GB).
    pub min_disk_gb: f64,
    /// Maximum acceptable heartbeat age in seconds.
    pub heartbeat_timeout_s: u64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            max_cpu_temp: 90.0,
            max_memory_pct: 95.0,
            min_disk_gb: 10.0,
            heartbeat_timeout_s: 120,
        }
    }
}

impl HealthThresholds {
    /// Create thresholds with explicit values.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(
        max_cpu_temp: f64,
        max_memory_pct: f64,
        min_disk_gb: f64,
        heartbeat_timeout_s: u64,
    ) -> Self {
        Self {
            max_cpu_temp,
            max_memory_pct,
            min_disk_gb,
            heartbeat_timeout_s,
        }
    }
}

/// Monitors the health of all farm nodes.
#[allow(dead_code)]
#[derive(Debug)]
pub struct HealthMonitor {
    /// Per-node health records (keyed by `node_id`).
    pub nodes: Vec<NodeHealth>,
    /// Thresholds used to classify node health.
    pub alert_thresholds: HealthThresholds,
}

impl HealthMonitor {
    /// Create a new monitor with the given alert thresholds.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(thresholds: HealthThresholds) -> Self {
        Self {
            nodes: Vec::new(),
            alert_thresholds: thresholds,
        }
    }

    /// Insert or replace the health record for a node.
    #[allow(dead_code)]
    pub fn update_node(&mut self, health: NodeHealth) {
        if let Some(existing) = self.nodes.iter_mut().find(|n| n.node_id == health.node_id) {
            *existing = health;
        } else {
            self.nodes.push(health);
        }
    }

    /// Return nodes whose status is not `Healthy` or whose metrics
    /// exceed the configured thresholds.
    #[allow(dead_code)]
    #[must_use]
    pub fn unhealthy_nodes(&self) -> Vec<&NodeHealth> {
        let t = &self.alert_thresholds;
        self.nodes
            .iter()
            .filter(|n| {
                !matches!(n.status, HealthStatus::Healthy)
                    || n.cpu_temp_c > t.max_cpu_temp
                    || n.disk_free_gb < t.min_disk_gb
            })
            .collect()
    }

    /// Return nodes whose last heartbeat is older than the configured timeout.
    #[allow(dead_code)]
    #[must_use]
    pub fn stale_nodes(&self, now: u64) -> Vec<&NodeHealth> {
        self.nodes
            .iter()
            .filter(|n| n.is_stale(now, self.alert_thresholds.heartbeat_timeout_s))
            .collect()
    }

    /// Return all nodes that are currently available to accept jobs.
    #[allow(dead_code)]
    #[must_use]
    pub fn available_nodes(&self) -> Vec<&NodeHealth> {
        self.nodes.iter().filter(|n| n.is_available()).collect()
    }

    /// Return the number of tracked nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_healthy_node(id: u64) -> NodeHealth {
        let mut n = NodeHealth::new(id, &format!("node-{id}"));
        n.cpu_temp_c = 55.0;
        n.memory_used_gb = 32.0;
        n.disk_free_gb = 500.0;
        n.last_heartbeat = 1_000_000;
        n
    }

    #[test]
    fn test_health_status_is_operational() {
        assert!(HealthStatus::Healthy.is_operational());
        assert!(HealthStatus::Degraded {
            reason: "hot".to_string()
        }
        .is_operational());
        assert!(!HealthStatus::Unavailable { since: 0 }.is_operational());
        assert!(!HealthStatus::Maintenance.is_operational());
    }

    #[test]
    fn test_node_is_available_healthy() {
        let n = make_healthy_node(1);
        assert!(n.is_available());
    }

    #[test]
    fn test_node_is_available_degraded() {
        let mut n = make_healthy_node(2);
        n.status = HealthStatus::Degraded {
            reason: "overheating".to_string(),
        };
        assert!(n.is_available());
    }

    #[test]
    fn test_node_not_available_unavailable() {
        let mut n = make_healthy_node(3);
        n.status = HealthStatus::Unavailable { since: 999 };
        assert!(!n.is_available());
    }

    #[test]
    fn test_failure_rate_no_jobs() {
        let n = make_healthy_node(1);
        assert_eq!(n.failure_rate(), 0.0);
    }

    #[test]
    fn test_failure_rate_some_failures() {
        let mut n = make_healthy_node(1);
        n.jobs_completed = 8;
        n.jobs_failed = 2;
        assert!((n.failure_rate() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_failure_rate_all_failures() {
        let mut n = make_healthy_node(1);
        n.jobs_completed = 0;
        n.jobs_failed = 5;
        assert!((n.failure_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_heartbeat_age() {
        let n = make_healthy_node(1); // last_heartbeat = 1_000_000
        assert_eq!(n.heartbeat_age_s(1_000_060), 60);
    }

    #[test]
    fn test_is_stale() {
        let n = make_healthy_node(1); // last_heartbeat = 1_000_000
        assert!(n.is_stale(1_000_200, 120));
        assert!(!n.is_stale(1_000_100, 120));
    }

    #[test]
    fn test_monitor_update_node() {
        let thresholds = HealthThresholds::default();
        let mut monitor = HealthMonitor::new(thresholds);
        monitor.update_node(make_healthy_node(1));
        monitor.update_node(make_healthy_node(2));
        assert_eq!(monitor.node_count(), 2);
        // Update existing
        monitor.update_node(make_healthy_node(1));
        assert_eq!(monitor.node_count(), 2);
    }

    #[test]
    fn test_monitor_unhealthy_nodes_cpu_temp() {
        let thresholds = HealthThresholds::new(90.0, 95.0, 10.0, 120);
        let mut monitor = HealthMonitor::new(thresholds);
        let mut hot = make_healthy_node(1);
        hot.cpu_temp_c = 95.0; // exceeds 90°C threshold
        monitor.update_node(hot);
        monitor.update_node(make_healthy_node(2));
        assert_eq!(monitor.unhealthy_nodes().len(), 1);
    }

    #[test]
    fn test_monitor_unhealthy_nodes_low_disk() {
        let thresholds = HealthThresholds::new(90.0, 95.0, 50.0, 120);
        let mut monitor = HealthMonitor::new(thresholds);
        let mut low_disk = make_healthy_node(1);
        low_disk.disk_free_gb = 5.0; // below 50 GB threshold
        monitor.update_node(low_disk);
        assert_eq!(monitor.unhealthy_nodes().len(), 1);
    }

    #[test]
    fn test_monitor_stale_nodes() {
        let thresholds = HealthThresholds::new(90.0, 95.0, 10.0, 60);
        let mut monitor = HealthMonitor::new(thresholds);
        let mut stale = make_healthy_node(1);
        stale.last_heartbeat = 1_000_000;
        monitor.update_node(stale);
        let mut fresh = make_healthy_node(2);
        fresh.last_heartbeat = 1_000_090; // only 30s ago
        monitor.update_node(fresh);
        let now = 1_000_120u64;
        let stale_list = monitor.stale_nodes(now);
        assert_eq!(stale_list.len(), 1);
        assert_eq!(stale_list[0].node_id, 1);
    }
}
