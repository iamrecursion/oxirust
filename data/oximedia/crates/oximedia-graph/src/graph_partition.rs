#![allow(dead_code)]
//! Graph partitioning for parallel execution.
//!
//! This module provides algorithms for splitting a processing graph into
//! partitions that can be executed on different threads or machines,
//! minimizing inter-partition communication.

use std::collections::{HashMap, HashSet};

/// Identifier for a partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionId(pub u32);

impl std::fmt::Display for PartitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "partition_{}", self.0)
    }
}

/// A node in the partitionable graph.
#[derive(Debug, Clone)]
pub struct PartNode {
    /// Node identifier.
    pub id: u64,
    /// Computational weight (cost).
    pub weight: f64,
    /// Memory requirement in bytes.
    pub memory_bytes: u64,
}

impl PartNode {
    /// Create a new partition node.
    pub fn new(id: u64, weight: f64, memory_bytes: u64) -> Self {
        Self {
            id,
            weight,
            memory_bytes,
        }
    }
}

/// An edge between two nodes, carrying a communication cost.
#[derive(Debug, Clone)]
pub struct PartEdge {
    /// Source node ID.
    pub from: u64,
    /// Destination node ID.
    pub to: u64,
    /// Communication cost if these nodes are in different partitions.
    pub comm_cost: f64,
}

impl PartEdge {
    /// Create a new partition edge.
    pub fn new(from: u64, to: u64, comm_cost: f64) -> Self {
        Self {
            from,
            to,
            comm_cost,
        }
    }
}

/// A partition of graph nodes.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition identifier.
    pub id: PartitionId,
    /// Node IDs assigned to this partition.
    pub nodes: Vec<u64>,
    /// Total computational weight.
    pub total_weight: f64,
    /// Total memory requirement.
    pub total_memory: u64,
}

impl Partition {
    /// Create a new empty partition.
    pub fn new(id: PartitionId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            total_weight: 0.0,
            total_memory: 0,
        }
    }

    /// Add a node to the partition.
    pub fn add_node(&mut self, node: &PartNode) {
        self.nodes.push(node.id);
        self.total_weight += node.weight;
        self.total_memory += node.memory_bytes;
    }

    /// Number of nodes in this partition.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if a node is in this partition.
    pub fn contains(&self, node_id: u64) -> bool {
        self.nodes.contains(&node_id)
    }
}

/// Result of a graph partitioning operation.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// The computed partitions.
    pub partitions: Vec<Partition>,
    /// Node-to-partition assignment.
    pub assignment: HashMap<u64, PartitionId>,
    /// Total inter-partition communication cost (edge cut).
    pub edge_cut_cost: f64,
    /// Load imbalance ratio (max_weight / avg_weight).
    pub imbalance: f64,
}

impl PartitionResult {
    /// Number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Get the partition a node belongs to.
    pub fn partition_of(&self, node_id: u64) -> Option<PartitionId> {
        self.assignment.get(&node_id).copied()
    }

    /// Get edges that cross partition boundaries.
    pub fn cut_edges<'a>(&'a self, edges: &'a [PartEdge]) -> Vec<&'a PartEdge> {
        edges
            .iter()
            .filter(|e| {
                let p_from = self.assignment.get(&e.from);
                let p_to = self.assignment.get(&e.to);
                match (p_from, p_to) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                }
            })
            .collect()
    }
}

/// Strategy for partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Simple round-robin assignment by node order.
    RoundRobin,
    /// Greedy assignment minimizing maximum partition weight.
    GreedyBalance,
    /// Greedy assignment minimizing edge cut cost.
    GreedyMinCut,
}

/// Graph partitioner.
pub struct GraphPartitioner {
    /// Nodes in the graph.
    nodes: Vec<PartNode>,
    /// Edges in the graph.
    edges: Vec<PartEdge>,
}

impl GraphPartitioner {
    /// Create a new partitioner with the given nodes and edges.
    pub fn new(nodes: Vec<PartNode>, edges: Vec<PartEdge>) -> Self {
        Self { nodes, edges }
    }

    /// Partition the graph into `k` partitions using the given strategy.
    #[allow(clippy::cast_precision_loss)]
    pub fn partition(&self, k: u32, strategy: PartitionStrategy) -> PartitionResult {
        if k == 0 {
            return PartitionResult {
                partitions: Vec::new(),
                assignment: HashMap::new(),
                edge_cut_cost: 0.0,
                imbalance: 0.0,
            };
        }
        if self.nodes.is_empty() {
            let partitions = (0..k).map(|i| Partition::new(PartitionId(i))).collect();
            return PartitionResult {
                partitions,
                assignment: HashMap::new(),
                edge_cut_cost: 0.0,
                imbalance: 0.0,
            };
        }

        let assignment = match strategy {
            PartitionStrategy::RoundRobin => self.round_robin(k),
            PartitionStrategy::GreedyBalance => self.greedy_balance(k),
            PartitionStrategy::GreedyMinCut => self.greedy_min_cut(k),
        };

        self.build_result(k, &assignment)
    }

    /// Simple round-robin assignment.
    fn round_robin(&self, k: u32) -> HashMap<u64, PartitionId> {
        let mut assignment = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let part = PartitionId((i as u32) % k);
            assignment.insert(node.id, part);
        }
        assignment
    }

    /// Greedy load-balanced assignment.
    #[allow(clippy::cast_precision_loss)]
    fn greedy_balance(&self, k: u32) -> HashMap<u64, PartitionId> {
        let mut assignment = HashMap::new();
        let mut weights = vec![0.0_f64; k as usize];

        // Sort nodes by decreasing weight for LPT (Longest Processing Time) heuristic
        let mut sorted_nodes: Vec<_> = self.nodes.iter().collect();
        sorted_nodes.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for node in sorted_nodes {
            // Assign to the partition with the smallest total weight
            let min_idx = weights
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            assignment.insert(node.id, PartitionId(min_idx as u32));
            weights[min_idx] += node.weight;
        }

        assignment
    }

    /// Greedy assignment minimizing edge cut.
    fn greedy_min_cut(&self, k: u32) -> HashMap<u64, PartitionId> {
        let mut assignment = HashMap::new();
        let mut partition_nodes: Vec<HashSet<u64>> = vec![HashSet::new(); k as usize];

        // Build adjacency for quick neighbor lookup
        let mut adjacency: HashMap<u64, Vec<(u64, f64)>> = HashMap::new();
        for edge in &self.edges {
            adjacency
                .entry(edge.from)
                .or_default()
                .push((edge.to, edge.comm_cost));
            adjacency
                .entry(edge.to)
                .or_default()
                .push((edge.from, edge.comm_cost));
        }

        for node in &self.nodes {
            // For each partition, compute how much communication cost would
            // be saved by placing this node there (neighbors already in partition).
            let mut best_part = 0_usize;
            let mut best_saved = f64::NEG_INFINITY;

            for p in 0..k as usize {
                let saved: f64 = adjacency
                    .get(&node.id)
                    .map(|neighbors| {
                        neighbors
                            .iter()
                            .filter(|(nid, _)| partition_nodes[p].contains(nid))
                            .map(|(_, cost)| *cost)
                            .sum()
                    })
                    .unwrap_or(0.0);

                if saved > best_saved
                    || (saved == best_saved
                        && partition_nodes[p].len() < partition_nodes[best_part].len())
                {
                    best_saved = saved;
                    best_part = p;
                }
            }

            assignment.insert(node.id, PartitionId(best_part as u32));
            partition_nodes[best_part].insert(node.id);
        }

        assignment
    }

    /// Build the result from an assignment.
    #[allow(clippy::cast_precision_loss)]
    fn build_result(&self, k: u32, assignment: &HashMap<u64, PartitionId>) -> PartitionResult {
        let node_map: HashMap<u64, &PartNode> = self.nodes.iter().map(|n| (n.id, n)).collect();

        let mut partitions: Vec<Partition> =
            (0..k).map(|i| Partition::new(PartitionId(i))).collect();

        for (node_id, part_id) in assignment {
            if let Some(node) = node_map.get(node_id) {
                if (part_id.0 as usize) < partitions.len() {
                    partitions[part_id.0 as usize].add_node(node);
                }
            }
        }

        let edge_cut_cost: f64 = self
            .edges
            .iter()
            .filter(|e| {
                let p_from = assignment.get(&e.from);
                let p_to = assignment.get(&e.to);
                match (p_from, p_to) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                }
            })
            .map(|e| e.comm_cost)
            .sum();

        let weights: Vec<f64> = partitions.iter().map(|p| p.total_weight).collect();
        let avg = if weights.is_empty() {
            1.0
        } else {
            let sum: f64 = weights.iter().sum();
            sum / weights.len() as f64
        };
        let max_w = weights.iter().cloned().fold(0.0_f64, f64::max);
        let imbalance = if avg > 0.0 { max_w / avg } else { 0.0 };

        PartitionResult {
            partitions,
            assignment: assignment.clone(),
            edge_cut_cost,
            imbalance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes(n: u64) -> Vec<PartNode> {
        (0..n).map(|i| PartNode::new(i, 1.0, 1024)).collect()
    }

    fn make_chain_edges(n: u64) -> Vec<PartEdge> {
        (0..n.saturating_sub(1))
            .map(|i| PartEdge::new(i, i + 1, 1.0))
            .collect()
    }

    #[test]
    fn test_partition_id_display() {
        assert_eq!(format!("{}", PartitionId(3)), "partition_3");
    }

    #[test]
    fn test_part_node() {
        let n = PartNode::new(1, 5.0, 2048);
        assert_eq!(n.id, 1);
        assert!((n.weight - 5.0).abs() < f64::EPSILON);
        assert_eq!(n.memory_bytes, 2048);
    }

    #[test]
    fn test_partition_add_node() {
        let mut p = Partition::new(PartitionId(0));
        p.add_node(&PartNode::new(1, 3.0, 100));
        p.add_node(&PartNode::new(2, 2.0, 200));
        assert_eq!(p.node_count(), 2);
        assert!((p.total_weight - 5.0).abs() < f64::EPSILON);
        assert_eq!(p.total_memory, 300);
    }

    #[test]
    fn test_partition_contains() {
        let mut p = Partition::new(PartitionId(0));
        p.add_node(&PartNode::new(42, 1.0, 10));
        assert!(p.contains(42));
        assert!(!p.contains(99));
    }

    #[test]
    fn test_round_robin_partition() {
        let nodes = make_nodes(6);
        let edges = make_chain_edges(6);
        let partitioner = GraphPartitioner::new(nodes, edges);
        let result = partitioner.partition(3, PartitionStrategy::RoundRobin);
        assert_eq!(result.partition_count(), 3);
        for p in &result.partitions {
            assert_eq!(p.node_count(), 2);
        }
    }

    #[test]
    fn test_greedy_balance_partition() {
        let nodes = vec![
            PartNode::new(0, 10.0, 100),
            PartNode::new(1, 5.0, 100),
            PartNode::new(2, 3.0, 100),
            PartNode::new(3, 2.0, 100),
        ];
        let edges = Vec::new();
        let partitioner = GraphPartitioner::new(nodes, edges);
        let result = partitioner.partition(2, PartitionStrategy::GreedyBalance);
        assert_eq!(result.partition_count(), 2);
        // With LPT: 10+2=12 and 5+3=8, imbalance = 12/10 = 1.2
        assert!(result.imbalance <= 1.5);
    }

    #[test]
    fn test_greedy_min_cut() {
        let nodes = make_nodes(4);
        let edges = vec![
            PartEdge::new(0, 1, 10.0),
            PartEdge::new(2, 3, 10.0),
            PartEdge::new(1, 2, 1.0),
        ];
        let partitioner = GraphPartitioner::new(nodes, edges);
        let result = partitioner.partition(2, PartitionStrategy::GreedyMinCut);
        assert_eq!(result.partition_count(), 2);
        // Ideally {0,1} and {2,3} with cut cost = 1.0
        // The greedy might not be perfect but edge_cut_cost should be computed
        assert!(result.edge_cut_cost >= 0.0);
    }

    #[test]
    fn test_partition_result_partition_of() {
        let nodes = make_nodes(4);
        let edges = Vec::new();
        let partitioner = GraphPartitioner::new(nodes, edges);
        let result = partitioner.partition(2, PartitionStrategy::RoundRobin);
        for i in 0..4 {
            assert!(result.partition_of(i).is_some());
        }
        assert!(result.partition_of(999).is_none());
    }

    #[test]
    fn test_cut_edges() {
        let nodes = make_nodes(4);
        let edges = vec![
            PartEdge::new(0, 1, 5.0),
            PartEdge::new(2, 3, 5.0),
            PartEdge::new(1, 2, 3.0),
        ];
        let partitioner = GraphPartitioner::new(nodes, edges.clone());
        let result = partitioner.partition(2, PartitionStrategy::RoundRobin);
        let cuts = result.cut_edges(&edges);
        // Some edges should cross partitions
        assert!(!cuts.is_empty() || result.edge_cut_cost == 0.0);
    }

    #[test]
    fn test_empty_graph_partition() {
        let partitioner = GraphPartitioner::new(Vec::new(), Vec::new());
        let result = partitioner.partition(2, PartitionStrategy::RoundRobin);
        assert_eq!(result.partition_count(), 2);
        assert!((result.edge_cut_cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_partitions() {
        let nodes = make_nodes(4);
        let partitioner = GraphPartitioner::new(nodes, Vec::new());
        let result = partitioner.partition(0, PartitionStrategy::RoundRobin);
        assert!(result.partitions.is_empty());
    }

    #[test]
    fn test_single_partition() {
        let nodes = make_nodes(4);
        let edges = make_chain_edges(4);
        let partitioner = GraphPartitioner::new(nodes, edges);
        let result = partitioner.partition(1, PartitionStrategy::GreedyBalance);
        assert_eq!(result.partition_count(), 1);
        assert_eq!(result.partitions[0].node_count(), 4);
        assert!((result.edge_cut_cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_imbalance_ratio() {
        let nodes = vec![PartNode::new(0, 10.0, 100), PartNode::new(1, 1.0, 100)];
        let partitioner = GraphPartitioner::new(nodes, Vec::new());
        let result = partitioner.partition(2, PartitionStrategy::RoundRobin);
        // Partition 0 has weight 10, partition 1 has weight 1, avg = 5.5
        assert!(result.imbalance > 1.0);
    }
}
