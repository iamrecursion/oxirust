#![allow(dead_code)]
//! Render node pool management for the render farm.
//!
//! Provides logical grouping of render nodes into pools based on hardware
//! capabilities, project affinity, or departmental ownership. Pools support
//! capacity tracking, automatic node selection, and usage statistics.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ─── Node Capability Tag ────────────────────────────────────────────────────

/// Hardware or software capability that a render node can advertise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityTag {
    /// Node has one or more GPUs suitable for GPU rendering.
    Gpu,
    /// Node supports NVIDIA CUDA compute.
    Cuda,
    /// Node supports `OptiX` ray-tracing.
    Optix,
    /// Node has high-bandwidth network (25 Gbps+).
    HighBandwidth,
    /// Node has NVMe-class local storage.
    FastStorage,
    /// Node has 64+ GB RAM.
    HighMemory,
    /// Node supports Docker / containerised workloads.
    Container,
    /// Node supports VFX composition rendering.
    Compositing,
}

impl fmt::Display for CapabilityTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Gpu => "GPU",
            Self::Cuda => "CUDA",
            Self::Optix => "OptiX",
            Self::HighBandwidth => "High Bandwidth",
            Self::FastStorage => "Fast Storage",
            Self::HighMemory => "High Memory",
            Self::Container => "Container",
            Self::Compositing => "Compositing",
        };
        f.write_str(label)
    }
}

// ─── Pool Tier ──────────────────────────────────────────────────────────────

/// Priority tier for a node pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PoolTier {
    /// Low-priority batch pool.
    Batch,
    /// Standard production pool.
    Standard,
    /// High-priority pool for critical deadlines.
    Priority,
    /// Reserved for emergency / rush jobs only.
    Emergency,
}

impl PoolTier {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Batch => "Batch",
            Self::Standard => "Standard",
            Self::Priority => "Priority",
            Self::Emergency => "Emergency",
        }
    }
}

impl fmt::Display for PoolTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ─── Node Info ──────────────────────────────────────────────────────────────

/// Lightweight descriptor for a render node in a pool.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Unique node identifier (hostname or UUID string).
    pub id: String,
    /// Advertised capabilities.
    pub capabilities: HashSet<CapabilityTag>,
    /// Number of logical CPU cores.
    pub cpu_cores: u32,
    /// Total RAM in megabytes.
    pub ram_mb: u64,
    /// Number of GPUs.
    pub gpu_count: u32,
    /// Whether the node is currently online and healthy.
    pub online: bool,
}

impl NodeInfo {
    /// Create a new node with basic info.
    #[must_use]
    pub fn new(id: impl Into<String>, cpu_cores: u32, ram_mb: u64) -> Self {
        Self {
            id: id.into(),
            capabilities: HashSet::new(),
            cpu_cores,
            ram_mb,
            gpu_count: 0,
            online: true,
        }
    }

    /// Builder: add a capability tag.
    #[must_use]
    pub fn with_capability(mut self, tag: CapabilityTag) -> Self {
        self.capabilities.insert(tag);
        self
    }

    /// Builder: set GPU count.
    #[must_use]
    pub fn with_gpus(mut self, count: u32) -> Self {
        self.gpu_count = count;
        if count > 0 {
            self.capabilities.insert(CapabilityTag::Gpu);
        }
        self
    }

    /// Returns `true` if the node has a specific capability.
    #[must_use]
    pub fn has_capability(&self, tag: CapabilityTag) -> bool {
        self.capabilities.contains(&tag)
    }
}

// ─── NodePool ───────────────────────────────────────────────────────────────

/// A logical grouping of render nodes.
#[derive(Debug, Clone)]
pub struct NodePool {
    /// Pool name (e.g., "gpu-farm", "compositing-nodes").
    pub name: String,
    /// Tier / priority level.
    pub tier: PoolTier,
    /// Required capabilities — only nodes with all of these may join.
    pub required_capabilities: HashSet<CapabilityTag>,
    /// Nodes currently in this pool.
    nodes: HashMap<String, NodeInfo>,
    /// Maximum number of nodes in this pool (0 = unlimited).
    pub max_nodes: usize,
}

impl NodePool {
    /// Create a new pool with the given name and tier.
    #[must_use]
    pub fn new(name: impl Into<String>, tier: PoolTier) -> Self {
        Self {
            name: name.into(),
            tier,
            required_capabilities: HashSet::new(),
            nodes: HashMap::new(),
            max_nodes: 0,
        }
    }

    /// Builder: require a capability for pool membership.
    #[must_use]
    pub fn require(mut self, tag: CapabilityTag) -> Self {
        self.required_capabilities.insert(tag);
        self
    }

    /// Builder: set the maximum node count.
    #[must_use]
    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }

    /// Returns `true` if the node meets all required capabilities.
    #[must_use]
    pub fn node_eligible(&self, node: &NodeInfo) -> bool {
        self.required_capabilities
            .iter()
            .all(|cap| node.capabilities.contains(cap))
    }

    /// Attempt to add a node to the pool.
    ///
    /// Returns `false` if the node is ineligible or the pool is full.
    pub fn add_node(&mut self, node: NodeInfo) -> bool {
        if !self.node_eligible(&node) {
            return false;
        }
        if self.max_nodes > 0 && self.nodes.len() >= self.max_nodes {
            return false;
        }
        self.nodes.insert(node.id.clone(), node);
        true
    }

    /// Remove a node by ID. Returns the removed node if found.
    pub fn remove_node(&mut self, id: &str) -> Option<NodeInfo> {
        self.nodes.remove(id)
    }

    /// Number of nodes in the pool.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of online nodes.
    #[must_use]
    pub fn online_count(&self) -> usize {
        self.nodes.values().filter(|n| n.online).count()
    }

    /// Total CPU cores across all online nodes.
    #[must_use]
    pub fn total_cpu_cores(&self) -> u64 {
        self.nodes
            .values()
            .filter(|n| n.online)
            .map(|n| u64::from(n.cpu_cores))
            .sum()
    }

    /// Total RAM (MB) across all online nodes.
    #[must_use]
    pub fn total_ram_mb(&self) -> u64 {
        self.nodes
            .values()
            .filter(|n| n.online)
            .map(|n| n.ram_mb)
            .sum()
    }

    /// Total GPU count across all online nodes.
    #[must_use]
    pub fn total_gpus(&self) -> u64 {
        self.nodes
            .values()
            .filter(|n| n.online)
            .map(|n| u64::from(n.gpu_count))
            .sum()
    }

    /// Get a reference to a node by ID.
    #[must_use]
    pub fn get_node(&self, id: &str) -> Option<&NodeInfo> {
        self.nodes.get(id)
    }

    /// List all node IDs.
    #[must_use]
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Select nodes matching an additional set of desired capabilities.
    #[must_use]
    pub fn select_nodes(&self, desired: &HashSet<CapabilityTag>) -> Vec<&NodeInfo> {
        self.nodes
            .values()
            .filter(|n| n.online && desired.iter().all(|c| n.capabilities.contains(c)))
            .collect()
    }

    /// Returns `true` if the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ─── Pool usage stats ───────────────────────────────────────────────────────

/// Summary statistics for a node pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Pool name.
    pub name: String,
    /// Total nodes.
    pub total_nodes: usize,
    /// Online nodes.
    pub online_nodes: usize,
    /// Total CPU cores (online only).
    pub cpu_cores: u64,
    /// Total RAM MB (online only).
    pub ram_mb: u64,
    /// Total GPUs (online only).
    pub gpus: u64,
}

impl PoolStats {
    /// Compute stats from a pool.
    #[must_use]
    pub fn from_pool(pool: &NodePool) -> Self {
        Self {
            name: pool.name.clone(),
            total_nodes: pool.node_count(),
            online_nodes: pool.online_count(),
            cpu_cores: pool.total_cpu_cores(),
            ram_mb: pool.total_ram_mb(),
            gpus: pool.total_gpus(),
        }
    }

    /// Utilisation ratio (online / total). Returns 0 if pool is empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn online_ratio(&self) -> f64 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        self.online_nodes as f64 / self.total_nodes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node(id: &str) -> NodeInfo {
        NodeInfo::new(id, 16, 32768)
            .with_gpus(2)
            .with_capability(CapabilityTag::Cuda)
    }

    #[test]
    fn test_node_info_new() {
        let n = NodeInfo::new("node-1", 8, 16384);
        assert_eq!(n.id, "node-1");
        assert_eq!(n.cpu_cores, 8);
        assert!(n.online);
    }

    #[test]
    fn test_node_with_gpus() {
        let n = NodeInfo::new("n", 4, 8192).with_gpus(4);
        assert_eq!(n.gpu_count, 4);
        assert!(n.has_capability(CapabilityTag::Gpu));
    }

    #[test]
    fn test_node_capability_check() {
        let n = sample_node("a");
        assert!(n.has_capability(CapabilityTag::Cuda));
        assert!(!n.has_capability(CapabilityTag::Optix));
    }

    #[test]
    fn test_capability_tag_display() {
        assert_eq!(format!("{}", CapabilityTag::Gpu), "GPU");
        assert_eq!(format!("{}", CapabilityTag::HighMemory), "High Memory");
    }

    #[test]
    fn test_pool_tier_ordering() {
        assert!(PoolTier::Batch < PoolTier::Standard);
        assert!(PoolTier::Standard < PoolTier::Priority);
        assert!(PoolTier::Priority < PoolTier::Emergency);
    }

    #[test]
    fn test_pool_tier_display() {
        assert_eq!(format!("{}", PoolTier::Emergency), "Emergency");
    }

    #[test]
    fn test_pool_add_eligible_node() {
        let mut pool = NodePool::new("gpu-farm", PoolTier::Standard).require(CapabilityTag::Gpu);
        let n = sample_node("node-1");
        assert!(pool.add_node(n));
        assert_eq!(pool.node_count(), 1);
    }

    #[test]
    fn test_pool_reject_ineligible_node() {
        let mut pool =
            NodePool::new("optix-farm", PoolTier::Standard).require(CapabilityTag::Optix);
        let n = sample_node("node-1"); // has GPU + CUDA, not OptiX
        assert!(!pool.add_node(n));
        assert_eq!(pool.node_count(), 0);
    }

    #[test]
    fn test_pool_max_nodes() {
        let mut pool = NodePool::new("small", PoolTier::Batch).with_max_nodes(1);
        assert!(pool.add_node(NodeInfo::new("a", 4, 8192)));
        assert!(!pool.add_node(NodeInfo::new("b", 4, 8192)));
    }

    #[test]
    fn test_pool_remove_node() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(NodeInfo::new("a", 4, 8192));
        assert!(pool.remove_node("a").is_some());
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_online_count() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        let mut n1 = NodeInfo::new("a", 4, 8192);
        n1.online = true;
        let mut n2 = NodeInfo::new("b", 4, 8192);
        n2.online = false;
        pool.add_node(n1);
        pool.add_node(n2);
        assert_eq!(pool.online_count(), 1);
    }

    #[test]
    fn test_pool_total_resources() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(NodeInfo::new("a", 16, 32768).with_gpus(2));
        pool.add_node(NodeInfo::new("b", 8, 16384).with_gpus(1));
        assert_eq!(pool.total_cpu_cores(), 24);
        assert_eq!(pool.total_ram_mb(), 49152);
        assert_eq!(pool.total_gpus(), 3);
    }

    #[test]
    fn test_pool_select_nodes() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(
            NodeInfo::new("a", 4, 8192)
                .with_gpus(1)
                .with_capability(CapabilityTag::Cuda),
        );
        pool.add_node(NodeInfo::new("b", 4, 8192));
        let mut desired = HashSet::new();
        desired.insert(CapabilityTag::Cuda);
        let selected = pool.select_nodes(&desired);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "a");
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(NodeInfo::new("a", 16, 32768).with_gpus(2));
        let stats = PoolStats::from_pool(&pool);
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.online_nodes, 1);
        assert!((stats.online_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pool_stats_empty() {
        let pool = NodePool::new("empty", PoolTier::Batch);
        let stats = PoolStats::from_pool(&pool);
        assert_eq!(stats.total_nodes, 0);
        assert!((stats.online_ratio() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pool_node_ids() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(NodeInfo::new("alpha", 4, 8192));
        pool.add_node(NodeInfo::new("beta", 4, 8192));
        let ids = pool.node_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"alpha".to_string()));
        assert!(ids.contains(&"beta".to_string()));
    }

    #[test]
    fn test_pool_get_node() {
        let mut pool = NodePool::new("test", PoolTier::Standard);
        pool.add_node(NodeInfo::new("x", 8, 16384));
        assert!(pool.get_node("x").is_some());
        assert!(pool.get_node("y").is_none());
    }
}
