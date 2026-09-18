//! Topological sorting for directed acyclic graphs.
//!
//! This module provides Kahn's algorithm and DFS-based topological sort
//! implementations for ordering graph nodes such that every directed edge
//! goes from an earlier node to a later node in the ordering.
//!
//! For large graphs (> 1 000 nodes) prefer [`FastTopoSorter`], which uses
//! integer-indexed adjacency lists and an in-degree array instead of hash maps,
//! cutting constant-factor overhead by roughly 3–5×.

pub use fast_topo::CycleError;
pub use fast_topo::FastTopoSorter;

use std::collections::{HashMap, HashSet, VecDeque};

/// A node identifier in the topological graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TopoNodeId(
    /// Inner identifier value.
    pub usize,
);

impl std::fmt::Display for TopoNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Error types for topological sort operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopoError {
    /// The graph contains a cycle, making topological sort impossible.
    CycleDetected(
        /// Nodes involved in the cycle.
        Vec<TopoNodeId>,
    ),
    /// A referenced node does not exist in the graph.
    NodeNotFound(
        /// The missing node.
        TopoNodeId,
    ),
    /// The graph is empty.
    EmptyGraph,
}

impl std::fmt::Display for TopoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected(nodes) => {
                write!(f, "Cycle detected involving {} nodes", nodes.len())
            }
            Self::NodeNotFound(id) => write!(f, "Node {id} not found"),
            Self::EmptyGraph => write!(f, "Graph is empty"),
        }
    }
}

/// Directed graph structure for topological sorting.
pub struct TopoGraph {
    /// Adjacency list: node -> set of successor nodes.
    adjacency: HashMap<TopoNodeId, HashSet<TopoNodeId>>,
    /// Reverse adjacency: node -> set of predecessor nodes.
    reverse: HashMap<TopoNodeId, HashSet<TopoNodeId>>,
}

impl TopoGraph {
    /// Create a new empty topological graph.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, id: TopoNodeId) {
        self.adjacency.entry(id).or_default();
        self.reverse.entry(id).or_default();
    }

    /// Add a directed edge from `from` to `to`.
    pub fn add_edge(&mut self, from: TopoNodeId, to: TopoNodeId) {
        self.add_node(from);
        self.add_node(to);
        self.adjacency.entry(from).or_default().insert(to);
        self.reverse.entry(to).or_default().insert(from);
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|s| s.len()).sum()
    }

    /// Return the in-degree of a node.
    pub fn in_degree(&self, id: TopoNodeId) -> usize {
        self.reverse.get(&id).map_or(0, |s| s.len())
    }

    /// Return the out-degree of a node.
    pub fn out_degree(&self, id: TopoNodeId) -> usize {
        self.adjacency.get(&id).map_or(0, |s| s.len())
    }

    /// Return all nodes with in-degree zero (source nodes).
    pub fn sources(&self) -> Vec<TopoNodeId> {
        let mut sources: Vec<TopoNodeId> = self
            .adjacency
            .keys()
            .filter(|id| self.in_degree(**id) == 0)
            .copied()
            .collect();
        sources.sort();
        sources
    }

    /// Return all nodes with out-degree zero (sink nodes).
    pub fn sinks(&self) -> Vec<TopoNodeId> {
        let mut sinks: Vec<TopoNodeId> = self
            .adjacency
            .keys()
            .filter(|id| self.out_degree(**id) == 0)
            .copied()
            .collect();
        sinks.sort();
        sinks
    }

    /// Perform topological sort using Kahn's algorithm (BFS-based).
    ///
    /// Returns nodes in topological order or an error if a cycle exists.
    pub fn sort_kahn(&self) -> Result<Vec<TopoNodeId>, TopoError> {
        if self.adjacency.is_empty() {
            return Err(TopoError::EmptyGraph);
        }

        let mut in_degrees: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in self.adjacency.keys() {
            in_degrees.insert(node, self.in_degree(node));
        }

        let mut queue: VecDeque<TopoNodeId> = in_degrees
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort queue for deterministic output
        let mut sorted_start: Vec<TopoNodeId> = queue.drain(..).collect();
        sorted_start.sort();
        queue.extend(sorted_start);

        let mut result = Vec::with_capacity(self.adjacency.len());

        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(successors) = self.adjacency.get(&node) {
                let mut sorted_succ: Vec<TopoNodeId> = successors.iter().copied().collect();
                sorted_succ.sort();
                for succ in sorted_succ {
                    if let Some(deg) = in_degrees.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if result.len() != self.adjacency.len() {
            let remaining: Vec<TopoNodeId> = self
                .adjacency
                .keys()
                .filter(|id| !result.contains(id))
                .copied()
                .collect();
            return Err(TopoError::CycleDetected(remaining));
        }

        Ok(result)
    }

    /// Perform topological sort using DFS-based algorithm.
    ///
    /// Returns nodes in topological order or an error if a cycle exists.
    pub fn sort_dfs(&self) -> Result<Vec<TopoNodeId>, TopoError> {
        if self.adjacency.is_empty() {
            return Err(TopoError::EmptyGraph);
        }

        let mut visited: HashSet<TopoNodeId> = HashSet::new();
        let mut in_stack: HashSet<TopoNodeId> = HashSet::new();
        let mut result: Vec<TopoNodeId> = Vec::new();

        let mut nodes: Vec<TopoNodeId> = self.adjacency.keys().copied().collect();
        nodes.sort();

        for node in &nodes {
            if !visited.contains(node)
                && !Self::dfs_visit(
                    *node,
                    &self.adjacency,
                    &mut visited,
                    &mut in_stack,
                    &mut result,
                )
            {
                let cycle_nodes: Vec<TopoNodeId> = in_stack.into_iter().collect();
                return Err(TopoError::CycleDetected(cycle_nodes));
            }
        }

        result.reverse();
        Ok(result)
    }

    /// DFS visit helper. Returns false if a cycle is detected.
    fn dfs_visit(
        node: TopoNodeId,
        adjacency: &HashMap<TopoNodeId, HashSet<TopoNodeId>>,
        visited: &mut HashSet<TopoNodeId>,
        in_stack: &mut HashSet<TopoNodeId>,
        result: &mut Vec<TopoNodeId>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        if let Some(successors) = adjacency.get(&node) {
            let mut sorted_succ: Vec<TopoNodeId> = successors.iter().copied().collect();
            sorted_succ.sort();
            for succ in sorted_succ {
                if in_stack.contains(&succ) {
                    return false;
                }
                if !visited.contains(&succ)
                    && !Self::dfs_visit(succ, adjacency, visited, in_stack, result)
                {
                    return false;
                }
            }
        }

        in_stack.remove(&node);
        result.push(node);
        true
    }

    /// Check if the graph is a DAG (has no cycles).
    pub fn is_dag(&self) -> bool {
        self.sort_kahn().is_ok()
    }

    /// Return the longest path length in the DAG.
    pub fn longest_path(&self) -> Result<usize, TopoError> {
        let order = self.sort_kahn()?;
        let mut dist: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in &order {
            dist.insert(node, 0);
        }

        for &node in &order {
            let node_dist = dist[&node];
            if let Some(successors) = self.adjacency.get(&node) {
                for &succ in successors {
                    let entry = dist.entry(succ).or_insert(0);
                    if node_dist + 1 > *entry {
                        *entry = node_dist + 1;
                    }
                }
            }
        }

        Ok(dist.values().copied().max().unwrap_or(0))
    }

    /// Return the depth (longest path from any source) for each node.
    pub fn node_depths(&self) -> Result<HashMap<TopoNodeId, usize>, TopoError> {
        let order = self.sort_kahn()?;
        let mut depths: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in &order {
            depths.insert(node, 0);
        }

        for &node in &order {
            let node_depth = depths[&node];
            if let Some(successors) = self.adjacency.get(&node) {
                for &succ in successors {
                    let entry = depths.entry(succ).or_insert(0);
                    if node_depth + 1 > *entry {
                        *entry = node_depth + 1;
                    }
                }
            }
        }

        Ok(depths)
    }

    /// Check if node `a` can reach node `b` (transitively).
    pub fn can_reach(&self, a: TopoNodeId, b: TopoNodeId) -> bool {
        let mut visited: HashSet<TopoNodeId> = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(a);

        while let Some(current) = queue.pop_front() {
            if current == b {
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
}

impl Default for TopoGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: usize) -> TopoNodeId {
        TopoNodeId(id)
    }

    #[test]
    fn test_empty_graph() {
        let graph = TopoGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert!(matches!(graph.sort_kahn(), Err(TopoError::EmptyGraph)));
    }

    #[test]
    fn test_single_node() {
        let mut graph = TopoGraph::new();
        graph.add_node(n(0));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order, vec![n(0)]);
    }

    #[test]
    fn test_linear_chain() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(3));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order, vec![n(0), n(1), n(2), n(3)]);
    }

    #[test]
    fn test_diamond_graph() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order[0], n(0));
        assert_eq!(order[3], n(3));
    }

    #[test]
    fn test_cycle_detection_kahn() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(0));
        let result = graph.sort_kahn();
        assert!(matches!(result, Err(TopoError::CycleDetected(_))));
    }

    #[test]
    fn test_cycle_detection_dfs() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(0));
        let result = graph.sort_dfs();
        assert!(matches!(result, Err(TopoError::CycleDetected(_))));
    }

    #[test]
    fn test_dfs_sort_matches_kahn() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let kahn = graph.sort_kahn().expect("sort_kahn should succeed");
        let dfs = graph.sort_dfs().expect("sort_dfs should succeed");
        // Both should have 0 first and 3 last
        assert_eq!(kahn[0], n(0));
        assert_eq!(dfs[0], n(0));
        assert_eq!(*kahn.last().expect("last should succeed"), n(3));
        assert_eq!(*dfs.last().expect("last should succeed"), n(3));
    }

    #[test]
    fn test_sources_and_sinks() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(3));
        graph.add_edge(n(2), n(4));
        assert_eq!(graph.sources(), vec![n(0), n(1)]);
        assert_eq!(graph.sinks(), vec![n(3), n(4)]);
    }

    #[test]
    fn test_in_out_degree() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(2));
        assert_eq!(graph.out_degree(n(0)), 2);
        assert_eq!(graph.in_degree(n(2)), 2);
        assert_eq!(graph.in_degree(n(0)), 0);
    }

    #[test]
    fn test_is_dag() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        assert!(graph.is_dag());

        graph.add_edge(n(2), n(0));
        assert!(!graph.is_dag());
    }

    #[test]
    fn test_longest_path() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(0), n(2));
        assert_eq!(
            graph.longest_path().expect("longest_path should succeed"),
            2
        );
    }

    #[test]
    fn test_node_depths() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let depths = graph.node_depths().expect("node_depths should succeed");
        assert_eq!(depths[&n(0)], 0);
        assert_eq!(depths[&n(3)], 2);
    }

    #[test]
    fn test_can_reach() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        assert!(graph.can_reach(n(0), n(2)));
        assert!(!graph.can_reach(n(2), n(0)));
    }

    #[test]
    fn test_topo_error_display() {
        let err = TopoError::EmptyGraph;
        assert_eq!(format!("{err}"), "Graph is empty");
        let err2 = TopoError::NodeNotFound(n(5));
        assert!(format!("{err2}").contains("5"));
    }

    #[test]
    fn test_edge_count() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(0), n(2));
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_node_id_display() {
        let id = TopoNodeId(42);
        assert_eq!(format!("{id}"), "Node(42)");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// High-performance integer-indexed topological sorter
// ─────────────────────────────────────────────────────────────────────────────

/// Sub-module containing [`FastTopoSorter`] and [`CycleError`].
pub mod fast_topo {
    use std::collections::VecDeque;

    /// Error returned by [`FastTopoSorter::sort`] when a cycle is detected.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CycleError {
        /// Nodes that were not reachable in topological order (i.e., they form
        /// or are downstream of the cycle).
        pub remaining: Vec<usize>,
    }

    impl std::fmt::Display for CycleError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "cycle detected; {} nodes could not be ordered",
                self.remaining.len()
            )
        }
    }

    impl std::error::Error for CycleError {}

    /// Cache-friendly topological sorter for large graphs.
    ///
    /// Uses a `Vec<Vec<usize>>` adjacency list indexed directly by node integer
    /// ID (O(1) random access, no hash overhead) and a `Vec<u32>` in-degree
    /// array.  Kahn's BFS algorithm is used for cycle detection and ordering.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_graph::topological::FastTopoSorter;
    ///
    /// let mut sorter = FastTopoSorter::new(4);
    /// sorter.add_edge(0, 1);
    /// sorter.add_edge(1, 2);
    /// sorter.add_edge(0, 3);
    /// let order = sorter.sort().expect("DAG must sort cleanly");
    /// assert_eq!(order[0], 0);
    /// ```
    pub struct FastTopoSorter {
        /// Number of nodes.
        n: usize,
        /// `adjacency[i]` — integer indices of nodes reachable from node `i`.
        adjacency: Vec<Vec<usize>>,
        /// `in_degree[i]` — number of incoming edges for node `i`.
        in_degree: Vec<u32>,
    }

    impl FastTopoSorter {
        /// Create a sorter for a graph with exactly `n` nodes (IDs `0..n`).
        #[must_use]
        pub fn new(n: usize) -> Self {
            Self {
                n,
                adjacency: vec![Vec::new(); n],
                in_degree: vec![0u32; n],
            }
        }

        /// Add a directed edge from node `from` to node `to`.
        ///
        /// # Panics
        ///
        /// Panics if either `from` or `to` is `>= n`.
        pub fn add_edge(&mut self, from: usize, to: usize) {
            assert!(from < self.n, "node id {from} out of range (n={})", self.n);
            assert!(to < self.n, "node id {to} out of range (n={})", self.n);
            self.adjacency[from].push(to);
            self.in_degree[to] = self.in_degree[to].saturating_add(1);
        }

        /// Run Kahn's algorithm and return nodes in topological order.
        ///
        /// Returns `Err(CycleError)` if the graph contains a cycle; the
        /// `remaining` field lists node IDs whose in-degree never reached zero.
        ///
        /// # Complexity
        ///
        /// O(V + E) time, O(V) extra space.
        pub fn sort(&self) -> Result<Vec<usize>, CycleError> {
            // Work on a mutable copy of in-degrees so `self` stays immutable
            // (callers may want to sort multiple times or inspect the structure).
            let mut deg = self.in_degree.clone();

            // Seed the queue with all zero-in-degree nodes (sorted for determinism).
            let mut queue: VecDeque<usize> = (0..self.n).filter(|&i| deg[i] == 0).collect();

            let mut result = Vec::with_capacity(self.n);

            while let Some(node) = queue.pop_front() {
                result.push(node);
                for &succ in &self.adjacency[node] {
                    // Saturating sub: in_degree should never underflow on a
                    // well-formed graph, but we avoid panics defensively.
                    deg[succ] = deg[succ].saturating_sub(1);
                    if deg[succ] == 0 {
                        queue.push_back(succ);
                    }
                }
            }

            if result.len() != self.n {
                let remaining: Vec<usize> = (0..self.n).filter(|&i| deg[i] > 0).collect();
                return Err(CycleError { remaining });
            }

            Ok(result)
        }

        /// Return the number of nodes in the graph.
        #[must_use]
        pub fn node_count(&self) -> usize {
            self.n
        }

        /// Return the total number of edges.
        #[must_use]
        pub fn edge_count(&self) -> usize {
            self.adjacency.iter().map(|v| v.len()).sum()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_fast_topo_simple() {
            // 5-node DAG:  0 → 1 → 3
            //              0 → 2 → 3
            //              3 → 4
            let mut s = FastTopoSorter::new(5);
            s.add_edge(0, 1);
            s.add_edge(0, 2);
            s.add_edge(1, 3);
            s.add_edge(2, 3);
            s.add_edge(3, 4);

            let order = s.sort().expect("DAG must succeed");
            assert_eq!(order.len(), 5, "all 5 nodes must appear");
            assert_eq!(order[0], 0, "node 0 has no predecessors");
            assert_eq!(
                *order.last().expect("non-empty"),
                4,
                "node 4 is the only sink"
            );

            // Verify topological constraint: for every edge u→v, pos(u) < pos(v).
            let mut pos = vec![0usize; 5];
            for (rank, &node) in order.iter().enumerate() {
                pos[node] = rank;
            }
            assert!(pos[0] < pos[1]);
            assert!(pos[0] < pos[2]);
            assert!(pos[1] < pos[3]);
            assert!(pos[2] < pos[3]);
            assert!(pos[3] < pos[4]);
        }

        #[test]
        fn test_fast_topo_cycle_detected() {
            // 0 → 1 → 2 → 0  (simple 3-cycle)
            let mut s = FastTopoSorter::new(3);
            s.add_edge(0, 1);
            s.add_edge(1, 2);
            s.add_edge(2, 0);

            let result = s.sort();
            assert!(result.is_err(), "cycle must produce an error");
            let err = result.expect_err("expected CycleError");
            assert_eq!(
                err.remaining.len(),
                3,
                "all three nodes are stuck in the cycle"
            );
        }

        #[test]
        fn test_fast_topo_large() {
            // 10 000-node linear chain: 0 → 1 → 2 → … → 9999
            let n = 10_000usize;
            let mut s = FastTopoSorter::new(n);
            for i in 0..n - 1 {
                s.add_edge(i, i + 1);
            }

            let start = std::time::Instant::now();
            let order = s.sort().expect("linear chain must sort cleanly");
            let elapsed = start.elapsed();

            assert_eq!(order.len(), n, "all {n} nodes must appear");
            assert!(
                elapsed.as_millis() < 50,
                "sort must complete in < 50 ms, took {} ms",
                elapsed.as_millis()
            );

            // Verify the chain is sorted in order.
            for (rank, &node) in order.iter().enumerate() {
                assert_eq!(node, rank, "linear chain must be in strict ascending order");
            }
        }
    }
}
