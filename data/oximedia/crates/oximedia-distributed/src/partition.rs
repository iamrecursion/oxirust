//! Data partitioning for distributed encoding using consistent hashing.
//!
//! Provides consistent hashing for stable partition assignment,
//! partition rebalancing logic, and virtual node management.

use std::collections::BTreeMap;
use std::collections::HashMap;

/// A node in the distributed cluster
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterNode {
    /// Unique node identifier
    pub node_id: String,
    /// Node address (host:port)
    pub address: String,
    /// Weight for load distribution (1–100)
    pub weight: u32,
}

impl ClusterNode {
    /// Create a new cluster node
    #[allow(dead_code)]
    pub fn new(node_id: impl Into<String>, address: impl Into<String>, weight: u32) -> Self {
        Self {
            node_id: node_id.into(),
            address: address.into(),
            weight: weight.clamp(1, 100),
        }
    }
}

/// A partition assigned to a node
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition index
    pub index: u32,
    /// Owning node id
    pub owner: String,
    /// Replica node ids
    pub replicas: Vec<String>,
}

/// Consistent hash ring for stable partition assignment.
///
/// Virtual nodes (vnodes) per physical node are proportional to `weight`.
#[allow(dead_code)]
pub struct ConsistentHashRing {
    /// ring: hash -> `node_id`
    ring: BTreeMap<u64, String>,
    /// physical nodes
    nodes: HashMap<String, ClusterNode>,
    /// vnodes per unit weight
    vnodes_per_weight: u32,
}

impl ConsistentHashRing {
    /// Create a new ring with the given vnode density.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(vnodes_per_weight: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: HashMap::new(),
            vnodes_per_weight,
        }
    }

    /// Add a node to the ring.
    #[allow(dead_code)]
    pub fn add_node(&mut self, node: ClusterNode) {
        let vnodes = node.weight * self.vnodes_per_weight;
        for i in 0..vnodes {
            let key = format!("{}-{}", node.node_id, i);
            let hash = fnv1a_hash(key.as_bytes());
            self.ring.insert(hash, node.node_id.clone());
        }
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node from the ring.
    #[allow(dead_code)]
    pub fn remove_node(&mut self, node_id: &str) {
        if let Some(node) = self.nodes.remove(node_id) {
            let vnodes = node.weight * self.vnodes_per_weight;
            for i in 0..vnodes {
                let key = format!("{node_id}-{i}");
                let hash = fnv1a_hash(key.as_bytes());
                self.ring.remove(&hash);
            }
        }
    }

    /// Look up the responsible node for a given key.
    #[allow(dead_code)]
    #[must_use]
    pub fn get_node(&self, key: &[u8]) -> Option<&ClusterNode> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = fnv1a_hash(key);
        // clockwise lookup
        let node_id = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, v)| v.as_str())?;
        self.nodes.get(node_id)
    }

    /// Return the number of physical nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the total number of virtual nodes in the ring.
    #[allow(dead_code)]
    #[must_use]
    pub fn vnode_count(&self) -> usize {
        self.ring.len()
    }
}

/// FNV-1a 64-bit hash (no external deps).
#[allow(dead_code)]
fn fnv1a_hash(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Partition assignment table
#[allow(dead_code)]
pub struct PartitionTable {
    /// Total number of partitions
    pub partition_count: u32,
    /// Assignments: `partition_index` -> Partition
    assignments: Vec<Partition>,
}

impl PartitionTable {
    /// Create a new partition table and assign partitions to nodes.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(partition_count: u32, nodes: &[ClusterNode]) -> Self {
        let mut ring = ConsistentHashRing::new(10);
        for node in nodes {
            ring.add_node(node.clone());
        }

        let mut assignments = Vec::with_capacity(partition_count as usize);
        for i in 0..partition_count {
            let key = i.to_le_bytes();
            let owner = ring
                .get_node(&key)
                .map(|n| n.node_id.clone())
                .unwrap_or_default();
            assignments.push(Partition {
                index: i,
                owner,
                replicas: Vec::new(),
            });
        }

        Self {
            partition_count,
            assignments,
        }
    }

    /// Look up the owner of a partition.
    #[allow(dead_code)]
    #[must_use]
    pub fn owner_of(&self, partition_index: u32) -> Option<&str> {
        self.assignments
            .get(partition_index as usize)
            .map(|p| p.owner.as_str())
    }

    /// Return all partitions owned by a given node.
    #[allow(dead_code)]
    #[must_use]
    pub fn partitions_for_node<'a>(&'a self, node_id: &str) -> Vec<&'a Partition> {
        self.assignments
            .iter()
            .filter(|p| p.owner == node_id)
            .collect()
    }

    /// Rebalance by rebuilding the ring after node changes.
    #[allow(dead_code)]
    pub fn rebalance(&mut self, nodes: &[ClusterNode]) {
        let new = PartitionTable::new(self.partition_count, nodes);
        self.assignments = new.assignments;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes(n: usize) -> Vec<ClusterNode> {
        (0..n)
            .map(|i| ClusterNode::new(format!("node-{i}"), format!("127.0.0.1:{}", 7000 + i), 10))
            .collect()
    }

    #[test]
    fn test_cluster_node_weight_clamped() {
        let node = ClusterNode::new("n0", "127.0.0.1:7000", 200);
        assert_eq!(node.weight, 100);
        let node2 = ClusterNode::new("n1", "127.0.0.1:7001", 0);
        assert_eq!(node2.weight, 1);
    }

    #[test]
    fn test_add_node_increases_vnode_count() {
        let mut ring = ConsistentHashRing::new(10);
        assert_eq!(ring.vnode_count(), 0);
        ring.add_node(ClusterNode::new("n0", "127.0.0.1:7000", 1));
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_remove_node_decreases_vnode_count() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node(ClusterNode::new("n0", "127.0.0.1:7000", 1));
        ring.add_node(ClusterNode::new("n1", "127.0.0.1:7001", 1));
        ring.remove_node("n0");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.vnode_count(), 10);
    }

    #[test]
    fn test_get_node_empty_ring_returns_none() {
        let ring = ConsistentHashRing::new(10);
        assert!(ring.get_node(b"some_key").is_none());
    }

    #[test]
    fn test_get_node_returns_some() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node(ClusterNode::new("n0", "127.0.0.1:7000", 1));
        assert!(ring.get_node(b"video/segment/0001").is_some());
    }

    #[test]
    fn test_consistent_hashing_same_key_same_node() {
        let mut ring = ConsistentHashRing::new(20);
        for node in make_nodes(4) {
            ring.add_node(node);
        }
        let node1 = ring.get_node(b"job-abc").map(|n| n.node_id.clone());
        let node2 = ring.get_node(b"job-abc").map(|n| n.node_id.clone());
        assert_eq!(node1, node2);
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_different_inputs() {
        assert_ne!(fnv1a_hash(b"foo"), fnv1a_hash(b"bar"));
    }

    #[test]
    fn test_partition_table_all_partitions_assigned() {
        let nodes = make_nodes(3);
        let table = PartitionTable::new(64, &nodes);
        for i in 0..64 {
            assert!(!table.owner_of(i).unwrap_or("").is_empty());
        }
    }

    #[test]
    fn test_partition_table_owner_of_out_of_range() {
        let nodes = make_nodes(2);
        let table = PartitionTable::new(16, &nodes);
        assert!(table.owner_of(100).is_none());
    }

    #[test]
    fn test_partitions_for_node_coverage() {
        let nodes = make_nodes(4);
        let table = PartitionTable::new(64, &nodes);
        let total: usize = nodes
            .iter()
            .map(|n| table.partitions_for_node(&n.node_id).len())
            .sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn test_rebalance_after_node_removal() {
        let mut nodes = make_nodes(4);
        let mut table = PartitionTable::new(32, &nodes);
        nodes.remove(3);
        table.rebalance(&nodes);
        for i in 0..32 {
            let owner = table.owner_of(i).unwrap_or("");
            assert!(owner.starts_with("node-"));
            assert_ne!(owner, "node-3");
        }
    }

    #[test]
    fn test_partition_table_empty_nodes() {
        let table = PartitionTable::new(8, &[]);
        // With no nodes all owners are empty string
        for i in 0..8 {
            assert_eq!(table.owner_of(i), Some(""));
        }
    }
}
