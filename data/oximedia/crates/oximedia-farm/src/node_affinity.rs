//! Node affinity rules for task placement in the render farm.
//!
//! Provides types for expressing hardware and network requirements for tasks
//! and for selecting suitable worker nodes based on those requirements.

#![allow(dead_code)]

/// A rule expressing a hard or soft requirement for task placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinityRule {
    /// The node must have a GPU available.
    RequireGpu,
    /// The node must have an FPGA available.
    RequireFpga,
    /// Prefer nodes with low network latency.
    PreferLowLatency,
    /// Prefer nodes with large amounts of RAM.
    PreferHighMemory,
    /// Prefer nodes with high storage bandwidth.
    PreferFastStorage,
    /// The task must run on the specific node with the given ID.
    RequireSpecificNode(u32),
}

impl AffinityRule {
    /// Return `true` if this rule is a hard requirement (must be satisfied).
    ///
    /// `Prefer*` variants are soft requirements; all `Require*` variants are hard.
    #[must_use]
    pub fn is_hard_requirement(&self) -> bool {
        matches!(
            self,
            Self::RequireGpu | Self::RequireFpga | Self::RequireSpecificNode(_)
        )
    }
}

/// Hardware and network capabilities of a single worker node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCapabilities {
    /// Unique node identifier.
    pub node_id: u32,
    /// Whether this node has a GPU.
    pub has_gpu: bool,
    /// Whether this node has an FPGA.
    pub has_fpga: bool,
    /// Total RAM in gigabytes.
    pub ram_gb: u32,
    /// Storage bandwidth in megabytes per second.
    pub storage_bandwidth_mbps: u32,
    /// Network latency in milliseconds (lower is better).
    pub network_latency_ms: u32,
}

impl NodeCapabilities {
    /// Create a new node capabilities descriptor.
    #[must_use]
    pub fn new(
        node_id: u32,
        has_gpu: bool,
        has_fpga: bool,
        ram_gb: u32,
        storage_bandwidth_mbps: u32,
        network_latency_ms: u32,
    ) -> Self {
        Self {
            node_id,
            has_gpu,
            has_fpga,
            ram_gb,
            storage_bandwidth_mbps,
            network_latency_ms,
        }
    }

    /// Return `true` if this node satisfies the given affinity rule.
    #[must_use]
    pub fn satisfies(&self, rule: &AffinityRule) -> bool {
        match rule {
            AffinityRule::RequireGpu => self.has_gpu,
            AffinityRule::RequireFpga => self.has_fpga,
            AffinityRule::PreferLowLatency => self.network_latency_ms <= 10,
            AffinityRule::PreferHighMemory => self.ram_gb >= 64,
            AffinityRule::PreferFastStorage => self.storage_bandwidth_mbps >= 1_000,
            AffinityRule::RequireSpecificNode(id) => self.node_id == *id,
        }
    }
}

/// Scheduler that places tasks on nodes according to affinity rules.
#[derive(Debug, Clone, Default)]
pub struct AffinityScheduler {
    /// Known node capabilities.
    pub capabilities: Vec<NodeCapabilities>,
}

impl AffinityScheduler {
    /// Create a new scheduler with no nodes registered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node with the scheduler.
    pub fn add_node(&mut self, cap: NodeCapabilities) {
        self.capabilities.push(cap);
    }

    /// Return the IDs of all nodes that satisfy **all** given rules.
    ///
    /// Hard-requirement rules must be satisfied.  Soft (`Prefer*`) rules are
    /// included in the filter as well to return the best candidates.
    #[must_use]
    pub fn find_suitable_nodes(&self, rules: &[AffinityRule]) -> Vec<u32> {
        self.capabilities
            .iter()
            .filter(|cap| rules.iter().all(|r| cap.satisfies(r)))
            .map(|cap| cap.node_id)
            .collect()
    }

    /// Return the ID of the first node that fully satisfies all given rules,
    /// or `None` if no such node exists.
    #[must_use]
    pub fn best_node_for(&self, rules: &[AffinityRule]) -> Option<u32> {
        self.capabilities
            .iter()
            .find(|cap| rules.iter().all(|r| cap.satisfies(r)))
            .map(|cap| cap.node_id)
    }

    /// Return the number of nodes registered with the scheduler.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.capabilities.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_node(id: u32) -> NodeCapabilities {
        NodeCapabilities::new(id, true, false, 32, 500, 5)
    }

    fn fpga_node(id: u32) -> NodeCapabilities {
        NodeCapabilities::new(id, false, true, 64, 2_000, 2)
    }

    fn cpu_only_node(id: u32) -> NodeCapabilities {
        NodeCapabilities::new(id, false, false, 16, 200, 20)
    }

    // --- AffinityRule tests ---

    #[test]
    fn test_require_gpu_is_hard() {
        assert!(AffinityRule::RequireGpu.is_hard_requirement());
    }

    #[test]
    fn test_require_fpga_is_hard() {
        assert!(AffinityRule::RequireFpga.is_hard_requirement());
    }

    #[test]
    fn test_require_specific_node_is_hard() {
        assert!(AffinityRule::RequireSpecificNode(42).is_hard_requirement());
    }

    #[test]
    fn test_prefer_low_latency_is_soft() {
        assert!(!AffinityRule::PreferLowLatency.is_hard_requirement());
    }

    #[test]
    fn test_prefer_high_memory_is_soft() {
        assert!(!AffinityRule::PreferHighMemory.is_hard_requirement());
    }

    #[test]
    fn test_prefer_fast_storage_is_soft() {
        assert!(!AffinityRule::PreferFastStorage.is_hard_requirement());
    }

    // --- NodeCapabilities::satisfies tests ---

    #[test]
    fn test_gpu_node_satisfies_require_gpu() {
        let node = gpu_node(1);
        assert!(node.satisfies(&AffinityRule::RequireGpu));
    }

    #[test]
    fn test_cpu_only_node_does_not_satisfy_require_gpu() {
        let node = cpu_only_node(2);
        assert!(!node.satisfies(&AffinityRule::RequireGpu));
    }

    #[test]
    fn test_fpga_node_satisfies_require_fpga() {
        let node = fpga_node(3);
        assert!(node.satisfies(&AffinityRule::RequireFpga));
    }

    #[test]
    fn test_node_satisfies_prefer_low_latency() {
        let node = NodeCapabilities::new(4, false, false, 8, 100, 5);
        assert!(node.satisfies(&AffinityRule::PreferLowLatency));
    }

    #[test]
    fn test_node_does_not_satisfy_prefer_low_latency_high_latency() {
        let node = cpu_only_node(5); // latency = 20
        assert!(!node.satisfies(&AffinityRule::PreferLowLatency));
    }

    #[test]
    fn test_node_satisfies_require_specific_node() {
        let node = cpu_only_node(7);
        assert!(node.satisfies(&AffinityRule::RequireSpecificNode(7)));
        assert!(!node.satisfies(&AffinityRule::RequireSpecificNode(8)));
    }

    // --- AffinityScheduler tests ---

    #[test]
    fn test_scheduler_starts_empty() {
        let scheduler = AffinityScheduler::new();
        assert_eq!(scheduler.node_count(), 0);
    }

    #[test]
    fn test_add_node_increments_count() {
        let mut scheduler = AffinityScheduler::new();
        scheduler.add_node(gpu_node(1));
        assert_eq!(scheduler.node_count(), 1);
    }

    #[test]
    fn test_find_suitable_nodes_for_gpu() {
        let mut scheduler = AffinityScheduler::new();
        scheduler.add_node(gpu_node(1));
        scheduler.add_node(cpu_only_node(2));
        scheduler.add_node(gpu_node(3));
        let nodes = scheduler.find_suitable_nodes(&[AffinityRule::RequireGpu]);
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&1));
        assert!(nodes.contains(&3));
    }

    #[test]
    fn test_find_suitable_nodes_empty_rules_returns_all() {
        let mut scheduler = AffinityScheduler::new();
        scheduler.add_node(gpu_node(1));
        scheduler.add_node(cpu_only_node(2));
        let nodes = scheduler.find_suitable_nodes(&[]);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_best_node_for_returns_first_match() {
        let mut scheduler = AffinityScheduler::new();
        scheduler.add_node(cpu_only_node(10));
        scheduler.add_node(gpu_node(20));
        scheduler.add_node(gpu_node(30));
        let best = scheduler.best_node_for(&[AffinityRule::RequireGpu]);
        assert_eq!(best, Some(20));
    }

    #[test]
    fn test_best_node_for_returns_none_when_no_match() {
        let mut scheduler = AffinityScheduler::new();
        scheduler.add_node(cpu_only_node(1));
        let best = scheduler.best_node_for(&[AffinityRule::RequireGpu]);
        assert!(best.is_none());
    }
}
