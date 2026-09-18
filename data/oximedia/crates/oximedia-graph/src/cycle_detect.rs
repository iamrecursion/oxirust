#![allow(dead_code)]
//! Cycle detection algorithms for directed graphs.
//!
//! This module provides multiple cycle detection strategies including
//! DFS-based coloring, path-based, and incremental cycle detection
//! for maintaining DAG invariants when edges are added.

use std::collections::{HashMap, HashSet, VecDeque};

/// A node identifier for cycle detection graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CycleNodeId(
    /// Inner identifier value.
    pub usize,
);

impl std::fmt::Display for CycleNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CycleNode({})", self.0)
    }
}

/// Node coloring state used during DFS traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeColor {
    /// Node has not been visited.
    White,
    /// Node is currently being explored (on the DFS stack).
    Gray,
    /// Node exploration is complete.
    Black,
}

/// Represents a detected cycle as a sequence of node IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    /// The nodes forming the cycle, in traversal order.
    pub nodes: Vec<CycleNodeId>,
}

impl Cycle {
    /// Create a new cycle from a list of nodes.
    pub fn new(nodes: Vec<CycleNodeId>) -> Self {
        Self { nodes }
    }

    /// Return the length of the cycle (number of edges).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the cycle is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if a node is part of this cycle.
    pub fn contains(&self, id: CycleNodeId) -> bool {
        self.nodes.contains(&id)
    }

    /// Return a canonical form of the cycle (starting from the smallest node).
    pub fn canonical(&self) -> Self {
        if self.nodes.is_empty() {
            return self.clone();
        }
        let min_pos = self
            .nodes
            .iter()
            .enumerate()
            .min_by_key(|(_, n)| n.0)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut rotated = Vec::with_capacity(self.nodes.len());
        rotated.extend_from_slice(&self.nodes[min_pos..]);
        rotated.extend_from_slice(&self.nodes[..min_pos]);
        Self { nodes: rotated }
    }
}

impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Vec<String> = self.nodes.iter().map(|n| n.0.to_string()).collect();
        write!(f, "[{}]", ids.join(" -> "))
    }
}

/// Result of a cycle detection analysis.
#[derive(Debug, Clone)]
pub struct CycleDetectionResult {
    /// Whether the graph is acyclic.
    pub is_acyclic: bool,
    /// All cycles found (may be empty if only checking existence).
    pub cycles: Vec<Cycle>,
    /// Number of nodes visited during detection.
    pub nodes_visited: usize,
}

impl CycleDetectionResult {
    /// Create a result indicating no cycles.
    pub fn acyclic(nodes_visited: usize) -> Self {
        Self {
            is_acyclic: true,
            cycles: Vec::new(),
            nodes_visited,
        }
    }

    /// Create a result with detected cycles.
    pub fn with_cycles(cycles: Vec<Cycle>, nodes_visited: usize) -> Self {
        Self {
            is_acyclic: cycles.is_empty(),
            cycles,
            nodes_visited,
        }
    }

    /// Return the total number of cycles found.
    pub fn cycle_count(&self) -> usize {
        self.cycles.len()
    }
}

/// Directed graph for cycle detection.
pub struct CycleGraph {
    /// Adjacency list.
    adjacency: HashMap<CycleNodeId, HashSet<CycleNodeId>>,
}

impl CycleGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, id: CycleNodeId) {
        self.adjacency.entry(id).or_default();
    }

    /// Add a directed edge from `from` to `to`.
    pub fn add_edge(&mut self, from: CycleNodeId, to: CycleNodeId) {
        self.add_node(from);
        self.add_node(to);
        self.adjacency.entry(from).or_default().insert(to);
    }

    /// Remove a directed edge.
    pub fn remove_edge(&mut self, from: CycleNodeId, to: CycleNodeId) -> bool {
        self.adjacency.get_mut(&from).is_some_and(|s| s.remove(&to))
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|s| s.len()).sum()
    }

    /// Check if the graph has any cycles using DFS coloring.
    pub fn has_cycle(&self) -> bool {
        let mut colors: HashMap<CycleNodeId, NodeColor> = self
            .adjacency
            .keys()
            .map(|&k| (k, NodeColor::White))
            .collect();

        let mut nodes: Vec<CycleNodeId> = self.adjacency.keys().copied().collect();
        nodes.sort();

        for &node in &nodes {
            if colors[&node] == NodeColor::White && self.dfs_has_cycle(node, &mut colors) {
                return true;
            }
        }
        false
    }

    /// DFS helper for cycle detection.
    fn dfs_has_cycle(
        &self,
        node: CycleNodeId,
        colors: &mut HashMap<CycleNodeId, NodeColor>,
    ) -> bool {
        colors.insert(node, NodeColor::Gray);

        if let Some(successors) = self.adjacency.get(&node) {
            for &succ in successors {
                match colors.get(&succ) {
                    Some(NodeColor::Gray) => return true,
                    Some(NodeColor::White) => {
                        if self.dfs_has_cycle(succ, colors) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }

        colors.insert(node, NodeColor::Black);
        false
    }

    /// Detect all cycles in the graph using DFS.
    pub fn detect_all_cycles(&self) -> CycleDetectionResult {
        let mut visited: HashSet<CycleNodeId> = HashSet::new();
        let mut rec_stack: HashSet<CycleNodeId> = HashSet::new();
        let mut path: Vec<CycleNodeId> = Vec::new();
        let mut cycles: Vec<Cycle> = Vec::new();
        let mut nodes_visited = 0_usize;

        let mut nodes: Vec<CycleNodeId> = self.adjacency.keys().copied().collect();
        nodes.sort();

        for &node in &nodes {
            if !visited.contains(&node) {
                self.dfs_find_cycles(
                    node,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                    &mut nodes_visited,
                );
            }
        }

        CycleDetectionResult::with_cycles(cycles, nodes_visited)
    }

    /// DFS helper to find all cycles.
    fn dfs_find_cycles(
        &self,
        node: CycleNodeId,
        visited: &mut HashSet<CycleNodeId>,
        rec_stack: &mut HashSet<CycleNodeId>,
        path: &mut Vec<CycleNodeId>,
        cycles: &mut Vec<Cycle>,
        nodes_visited: &mut usize,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);
        *nodes_visited += 1;

        if let Some(successors) = self.adjacency.get(&node) {
            let mut sorted_succ: Vec<CycleNodeId> = successors.iter().copied().collect();
            sorted_succ.sort();
            for succ in sorted_succ {
                if rec_stack.contains(&succ) {
                    // Found a cycle - extract it from the path
                    if let Some(start_idx) = path.iter().position(|&n| n == succ) {
                        let cycle_nodes = path[start_idx..].to_vec();
                        cycles.push(Cycle::new(cycle_nodes));
                    }
                } else if !visited.contains(&succ) {
                    self.dfs_find_cycles(succ, visited, rec_stack, path, cycles, nodes_visited);
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
    }

    /// Check if adding an edge from `from` to `to` would create a cycle.
    /// This performs a BFS from `to` to see if it can reach `from`.
    pub fn would_create_cycle(&self, from: CycleNodeId, to: CycleNodeId) -> bool {
        if from == to {
            return true;
        }

        let mut visited: HashSet<CycleNodeId> = HashSet::new();
        let mut queue: VecDeque<CycleNodeId> = VecDeque::new();
        queue.push_back(to);

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true;
            }
            if visited.insert(current) {
                if let Some(successors) = self.adjacency.get(&current) {
                    for &succ in successors {
                        queue.push_back(succ);
                    }
                }
            }
        }

        false
    }

    /// Add an edge only if it would not create a cycle.
    /// Returns true if the edge was added, false if it would create a cycle.
    pub fn try_add_edge(&mut self, from: CycleNodeId, to: CycleNodeId) -> bool {
        if self.would_create_cycle(from, to) {
            return false;
        }
        self.add_edge(from, to);
        true
    }

    /// Return all nodes that are part of at least one cycle.
    pub fn cyclic_nodes(&self) -> HashSet<CycleNodeId> {
        let result = self.detect_all_cycles();
        let mut cyclic: HashSet<CycleNodeId> = HashSet::new();
        for cycle in &result.cycles {
            for &node in &cycle.nodes {
                cyclic.insert(node);
            }
        }
        cyclic
    }

    /// Check if a specific node is part of any cycle.
    pub fn is_in_cycle(&self, id: CycleNodeId) -> bool {
        self.cyclic_nodes().contains(&id)
    }
}

impl Default for CycleGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cn(id: usize) -> CycleNodeId {
        CycleNodeId(id)
    }

    #[test]
    fn test_empty_graph_no_cycles() {
        let graph = CycleGraph::new();
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_single_node_no_cycle() {
        let mut graph = CycleGraph::new();
        graph.add_node(cn(0));
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_dag_no_cycle() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        graph.add_edge(cn(0), cn(2));
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_simple_cycle() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        graph.add_edge(cn(2), cn(0));
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_self_loop() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(0));
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_detect_all_cycles() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(0));
        let result = graph.detect_all_cycles();
        assert!(!result.is_acyclic);
        assert!(!result.cycles.is_empty());
    }

    #[test]
    fn test_detect_acyclic() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        let result = graph.detect_all_cycles();
        assert!(result.is_acyclic);
        assert_eq!(result.cycle_count(), 0);
    }

    #[test]
    fn test_would_create_cycle() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        assert!(graph.would_create_cycle(cn(2), cn(0)));
        assert!(!graph.would_create_cycle(cn(0), cn(2)));
    }

    #[test]
    fn test_would_create_self_loop() {
        let graph = CycleGraph::new();
        assert!(graph.would_create_cycle(cn(0), cn(0)));
    }

    #[test]
    fn test_try_add_edge_success() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        assert!(graph.try_add_edge(cn(1), cn(2)));
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_try_add_edge_rejected() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        assert!(!graph.try_add_edge(cn(2), cn(0)));
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_cyclic_nodes() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(2));
        graph.add_edge(cn(2), cn(0));
        graph.add_edge(cn(3), cn(0)); // Node 3 is not in the cycle
        let cyclic = graph.cyclic_nodes();
        assert!(cyclic.contains(&cn(0)));
        assert!(cyclic.contains(&cn(1)));
        assert!(cyclic.contains(&cn(2)));
        assert!(!cyclic.contains(&cn(3)));
    }

    #[test]
    fn test_is_in_cycle() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        graph.add_edge(cn(1), cn(0));
        graph.add_node(cn(2));
        assert!(graph.is_in_cycle(cn(0)));
        assert!(!graph.is_in_cycle(cn(2)));
    }

    #[test]
    fn test_remove_edge() {
        let mut graph = CycleGraph::new();
        graph.add_edge(cn(0), cn(1));
        assert!(graph.remove_edge(cn(0), cn(1)));
        assert!(!graph.remove_edge(cn(0), cn(1)));
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_cycle_canonical() {
        let cycle = Cycle::new(vec![cn(2), cn(0), cn(1)]);
        let canonical = cycle.canonical();
        assert_eq!(canonical.nodes[0], cn(0));
    }

    #[test]
    fn test_cycle_display() {
        let cycle = Cycle::new(vec![cn(0), cn(1), cn(2)]);
        let display = format!("{cycle}");
        assert!(display.contains("0"));
        assert!(display.contains("2"));
    }

    #[test]
    fn test_cycle_contains() {
        let cycle = Cycle::new(vec![cn(0), cn(1), cn(2)]);
        assert!(cycle.contains(cn(1)));
        assert!(!cycle.contains(cn(5)));
    }

    #[test]
    fn test_node_color_states() {
        assert_eq!(NodeColor::White, NodeColor::White);
        assert_ne!(NodeColor::White, NodeColor::Gray);
        assert_ne!(NodeColor::Gray, NodeColor::Black);
    }

    #[test]
    fn test_cycle_detection_result_acyclic() {
        let result = CycleDetectionResult::acyclic(10);
        assert!(result.is_acyclic);
        assert_eq!(result.nodes_visited, 10);
        assert_eq!(result.cycle_count(), 0);
    }
}
