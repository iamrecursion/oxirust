// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Task dependency graph for workflow execution planning.
//!
//! Models tasks as nodes and their dependencies as directed edges, and
//! exposes helpers for finding root tasks, successors, predecessors, and
//! the critical path through the graph.

/// A single task node in the dependency graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// Unique task identifier.
    pub id: u64,
    /// Human-readable task label.
    pub name: String,
    /// Estimated execution time in milliseconds.
    pub estimated_ms: u64,
    /// Symbolic resource requirements (e.g. `"gpu"`, `"network"`).
    pub resource_requirements: Vec<String>,
}

impl TaskNode {
    /// Create a new `TaskNode`.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, estimated_ms: u64) -> Self {
        Self {
            id,
            name: name.into(),
            estimated_ms,
            resource_requirements: Vec::new(),
        }
    }

    /// Add a resource requirement.
    #[must_use]
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource_requirements.push(resource.into());
        self
    }

    /// Total number of resource requirements.
    #[must_use]
    pub fn total_resource_count(&self) -> usize {
        self.resource_requirements.len()
    }
}

/// Relationship type between two task nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// The successor may only start after the predecessor completes.
    Sequential,
    /// The successor may start at the same time as the predecessor.
    Parallel,
    /// The successor runs only if the predecessor succeeds.
    ConditionalSuccess,
    /// The successor runs only if the predecessor fails.
    ConditionalFailure,
}

impl EdgeType {
    /// Returns `true` for conditional edge types.
    #[must_use]
    pub const fn is_conditional(self) -> bool {
        matches!(self, Self::ConditionalSuccess | Self::ConditionalFailure)
    }
}

/// A directed edge from one [`TaskNode`] to another.
#[derive(Debug, Clone)]
pub struct TaskEdge {
    /// Source node ID.
    pub from_id: u64,
    /// Destination node ID.
    pub to_id: u64,
    /// Relationship type.
    pub edge_type: EdgeType,
}

impl TaskEdge {
    /// Create a new `TaskEdge`.
    #[must_use]
    pub fn new(from_id: u64, to_id: u64, edge_type: EdgeType) -> Self {
        Self {
            from_id,
            to_id,
            edge_type,
        }
    }
}

/// A directed acyclic task dependency graph.
#[derive(Debug, Clone, Default)]
pub struct TaskGraph {
    /// All registered task nodes.
    pub nodes: Vec<TaskNode>,
    /// All directed edges.
    pub edges: Vec<TaskEdge>,
}

impl TaskGraph {
    /// Create an empty `TaskGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a task node.
    pub fn add_node(&mut self, node: TaskNode) {
        self.nodes.push(node);
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, edge: TaskEdge) {
        self.edges.push(edge);
    }

    /// Return the IDs of all nodes that are direct successors of `id`.
    #[must_use]
    pub fn successors(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|e| e.from_id == id)
            .map(|e| e.to_id)
            .collect()
    }

    /// Return the IDs of all nodes that are direct predecessors of `id`.
    #[must_use]
    pub fn predecessors(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|e| e.to_id == id)
            .map(|e| e.from_id)
            .collect()
    }

    /// Return IDs of nodes that have no incoming edges (roots of the DAG).
    #[must_use]
    pub fn root_nodes(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|n| self.predecessors(n.id).is_empty())
            .map(|n| n.id)
            .collect()
    }

    /// Compute the critical path length (ms) via dynamic programming.
    ///
    /// The critical path is the longest weighted path from any root node to
    /// any leaf node, where each node contributes its `estimated_ms`.
    ///
    /// Returns 0 if the graph has no nodes.
    #[must_use]
    pub fn critical_path_ms(&self) -> u64 {
        if self.nodes.is_empty() {
            return 0;
        }

        // Build a map from id → estimated_ms for quick lookup.
        let mut best: std::collections::HashMap<u64, u64> =
            self.nodes.iter().map(|n| (n.id, 0u64)).collect();

        // Topological sort (Kahn's algorithm).
        let mut in_degree: std::collections::HashMap<u64, usize> =
            self.nodes.iter().map(|n| (n.id, 0)).collect();

        for e in &self.edges {
            *in_degree.entry(e.to_id).or_insert(0) += 1;
        }

        let mut queue: std::collections::VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Initialise roots with their own cost.
        for &id in &queue {
            if let Some(node) = self.nodes.iter().find(|n| n.id == id) {
                best.insert(id, node.estimated_ms);
            }
        }

        while let Some(node_id) = queue.pop_front() {
            let current_best = best[&node_id];
            let node_cost = self
                .nodes
                .iter()
                .find(|n| n.id == node_id)
                .map_or(0, |n| n.estimated_ms);

            for succ_id in self.successors(node_id) {
                let candidate = current_best
                    + node_cost.saturating_sub(
                        // node cost already counted once in current_best
                        // recompute: current_best is the path cost *up to and including* node_id
                        // so for successor we add succ's own cost
                        node_cost, // effectively: current_best (which includes node_id cost) + succ cost
                    );
                // Simpler: best[succ] = max(best[succ], best[node] + succ_cost)
                let succ_cost = self
                    .nodes
                    .iter()
                    .find(|n| n.id == succ_id)
                    .map_or(0, |n| n.estimated_ms);

                let new_val = current_best + succ_cost;
                let entry = best.entry(succ_id).or_insert(0);
                if new_val > *entry {
                    *entry = new_val;
                }
                let _ = candidate; // suppress unused-variable lint

                // Decrement in-degree; enqueue when it reaches 0.
                if let Some(deg) = in_degree.get_mut(&succ_id) {
                    if *deg > 0 {
                        *deg -= 1;
                    }
                    if *deg == 0 {
                        queue.push_back(succ_id);
                    }
                }
            }
        }

        best.values().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TaskNode ---

    #[test]
    fn test_task_node_resource_count_empty() {
        let n = TaskNode::new(1, "encode", 5000);
        assert_eq!(n.total_resource_count(), 0);
    }

    #[test]
    fn test_task_node_resource_count_with_resources() {
        let n = TaskNode::new(1, "encode", 5000)
            .with_resource("gpu")
            .with_resource("network");
        assert_eq!(n.total_resource_count(), 2);
    }

    // --- EdgeType ---

    #[test]
    fn test_edge_type_is_conditional_false() {
        assert!(!EdgeType::Sequential.is_conditional());
        assert!(!EdgeType::Parallel.is_conditional());
    }

    #[test]
    fn test_edge_type_is_conditional_true() {
        assert!(EdgeType::ConditionalSuccess.is_conditional());
        assert!(EdgeType::ConditionalFailure.is_conditional());
    }

    // --- TaskGraph basic operations ---

    #[test]
    fn test_add_node_and_edge() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 100));
        g.add_node(TaskNode::new(2, "b", 200));
        g.add_edge(TaskEdge::new(1, 2, EdgeType::Sequential));
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn test_successors() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 100));
        g.add_node(TaskNode::new(2, "b", 100));
        g.add_node(TaskNode::new(3, "c", 100));
        g.add_edge(TaskEdge::new(1, 2, EdgeType::Sequential));
        g.add_edge(TaskEdge::new(1, 3, EdgeType::Parallel));
        let mut succs = g.successors(1);
        succs.sort_unstable();
        assert_eq!(succs, vec![2, 3]);
    }

    #[test]
    fn test_predecessors() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 100));
        g.add_node(TaskNode::new(2, "b", 100));
        g.add_node(TaskNode::new(3, "c", 100));
        g.add_edge(TaskEdge::new(1, 3, EdgeType::Sequential));
        g.add_edge(TaskEdge::new(2, 3, EdgeType::Sequential));
        let mut preds = g.predecessors(3);
        preds.sort_unstable();
        assert_eq!(preds, vec![1, 2]);
    }

    #[test]
    fn test_root_nodes_single() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "root", 0));
        g.add_node(TaskNode::new(2, "leaf", 0));
        g.add_edge(TaskEdge::new(1, 2, EdgeType::Sequential));
        assert_eq!(g.root_nodes(), vec![1]);
    }

    #[test]
    fn test_root_nodes_multiple() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 0));
        g.add_node(TaskNode::new(2, "b", 0));
        g.add_node(TaskNode::new(3, "c", 0));
        g.add_edge(TaskEdge::new(1, 3, EdgeType::Sequential));
        g.add_edge(TaskEdge::new(2, 3, EdgeType::Sequential));
        let mut roots = g.root_nodes();
        roots.sort_unstable();
        assert_eq!(roots, vec![1, 2]);
    }

    #[test]
    fn test_successors_empty() {
        let g = TaskGraph::new();
        assert!(g.successors(99).is_empty());
    }

    #[test]
    fn test_predecessors_empty() {
        let g = TaskGraph::new();
        assert!(g.predecessors(99).is_empty());
    }

    // --- critical_path_ms ---

    #[test]
    fn test_critical_path_empty_graph() {
        let g = TaskGraph::new();
        assert_eq!(g.critical_path_ms(), 0);
    }

    #[test]
    fn test_critical_path_single_node() {
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "only", 500));
        assert_eq!(g.critical_path_ms(), 500);
    }

    #[test]
    fn test_critical_path_linear_chain() {
        // 1 → 2 → 3 with costs 100, 200, 300 → critical = 600
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 100));
        g.add_node(TaskNode::new(2, "b", 200));
        g.add_node(TaskNode::new(3, "c", 300));
        g.add_edge(TaskEdge::new(1, 2, EdgeType::Sequential));
        g.add_edge(TaskEdge::new(2, 3, EdgeType::Sequential));
        assert_eq!(g.critical_path_ms(), 600);
    }

    #[test]
    fn test_critical_path_parallel_paths() {
        // Root 1(100) → 2(200); Root 3(500) → 2.
        // Critical path: 3(500) + 2(200) = 700
        let mut g = TaskGraph::new();
        g.add_node(TaskNode::new(1, "a", 100));
        g.add_node(TaskNode::new(2, "b", 200));
        g.add_node(TaskNode::new(3, "c", 500));
        g.add_edge(TaskEdge::new(1, 2, EdgeType::Sequential));
        g.add_edge(TaskEdge::new(3, 2, EdgeType::Sequential));
        assert_eq!(g.critical_path_ms(), 700);
    }
}
