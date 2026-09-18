//! Job graph visualisation: DOT language export and ASCII rendering.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

// ── Graph node ────────────────────────────────────────────────────────────────

/// Status of a job node for rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    /// Waiting for dependencies.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Terminated with an error.
    Failed,
    /// Removed from the graph.
    Cancelled,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        };
        f.write_str(s)
    }
}

/// A node in the visualisation graph.
#[derive(Clone, Debug)]
pub struct VizNode {
    /// Unique identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Current status.
    pub status: NodeStatus,
}

impl VizNode {
    /// Create a new node.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>, status: NodeStatus) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            status,
        }
    }
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// A directed acyclic graph of job nodes for visualisation purposes.
pub struct JobGraphViz {
    nodes: HashMap<String, VizNode>,
    /// Directed edges: (from_id, to_id).
    edges: Vec<(String, String)>,
}

impl JobGraphViz {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node; returns `false` if a node with the same ID already exists.
    pub fn add_node(&mut self, node: VizNode) -> bool {
        if self.nodes.contains_key(&node.id) {
            return false;
        }
        self.nodes.insert(node.id.clone(), node);
        true
    }

    /// Add a directed edge from `from` to `to`.
    ///
    /// Returns `false` if either node does not exist.
    pub fn add_edge(&mut self, from: &str, to: &str) -> bool {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return false;
        }
        self.edges.push((from.to_string(), to.to_string()));
        true
    }

    /// Update the status of a node.
    ///
    /// Returns `false` if no such node exists.
    pub fn set_status(&mut self, id: &str, status: NodeStatus) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.status = status;
            true
        } else {
            false
        }
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the graph contains a cycle (depth-first search).
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, to) in &self.edges {
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut on_stack: HashSet<&str> = HashSet::new();

        fn dfs<'a>(
            node: &'a str,
            adj: &HashMap<&'a str, Vec<&'a str>>,
            visited: &mut HashSet<&'a str>,
            on_stack: &mut HashSet<&'a str>,
        ) -> bool {
            visited.insert(node);
            on_stack.insert(node);
            if let Some(neighbours) = adj.get(node) {
                for &nb in neighbours {
                    if !visited.contains(nb) {
                        if dfs(nb, adj, visited, on_stack) {
                            return true;
                        }
                    } else if on_stack.contains(nb) {
                        return true;
                    }
                }
            }
            on_stack.remove(node);
            false
        }

        for id in self.nodes.keys() {
            if !visited.contains(id.as_str())
                && dfs(id.as_str(), &adjacency, &mut visited, &mut on_stack)
            {
                return true;
            }
        }
        false
    }

    /// Export the graph in Graphviz DOT format.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph jobs {\n");
        let _ = writeln!(out, "  rankdir=LR;");

        // Sort node IDs for deterministic output.
        let mut ids: Vec<&str> = self.nodes.keys().map(String::as_str).collect();
        ids.sort_unstable();

        for id in &ids {
            let node = &self.nodes[*id];
            let shape = match node.status {
                NodeStatus::Running => "doublecircle",
                NodeStatus::Completed => "box",
                NodeStatus::Failed => "diamond",
                NodeStatus::Cancelled => "plaintext",
                NodeStatus::Pending => "circle",
            };
            let _ = writeln!(
                out,
                "  \"{}\" [label=\"{}\", shape={}, status=\"{}\"];",
                node.id, node.label, shape, node.status
            );
        }

        for (from, to) in &self.edges {
            let _ = writeln!(out, "  \"{}\" -> \"{}\";", from, to);
        }

        out.push('}');
        out
    }

    /// Nodes with no incoming edges (potential roots / ready-to-run jobs).
    #[must_use]
    pub fn root_nodes(&self) -> Vec<&VizNode> {
        let targets: HashSet<&str> = self.edges.iter().map(|(_, t)| t.as_str()).collect();
        let mut roots: Vec<&VizNode> = self
            .nodes
            .values()
            .filter(|n| !targets.contains(n.id.as_str()))
            .collect();
        roots.sort_by_key(|n| n.id.as_str());
        roots
    }

    /// Nodes with no outgoing edges (leaf nodes).
    #[must_use]
    pub fn leaf_nodes(&self) -> Vec<&VizNode> {
        let sources: HashSet<&str> = self.edges.iter().map(|(s, _)| s.as_str()).collect();
        let mut leaves: Vec<&VizNode> = self
            .nodes
            .values()
            .filter(|n| !sources.contains(n.id.as_str()))
            .collect();
        leaves.sort_by_key(|n| n.id.as_str());
        leaves
    }
}

impl Default for JobGraphViz {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> JobGraphViz {
        let mut g = JobGraphViz::new();
        g.add_node(VizNode::new("a", "Step A", NodeStatus::Completed));
        g.add_node(VizNode::new("b", "Step B", NodeStatus::Running));
        g.add_node(VizNode::new("c", "Step C", NodeStatus::Pending));
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g
    }

    #[test]
    fn test_node_status_display() {
        assert_eq!(NodeStatus::Completed.to_string(), "completed");
        assert_eq!(NodeStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn test_add_node() {
        let mut g = JobGraphViz::new();
        assert!(g.add_node(VizNode::new("x", "X", NodeStatus::Pending)));
        assert!(!g.add_node(VizNode::new("x", "dup", NodeStatus::Running)));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_edge_success() {
        let mut g = JobGraphViz::new();
        g.add_node(VizNode::new("a", "A", NodeStatus::Pending));
        g.add_node(VizNode::new("b", "B", NodeStatus::Pending));
        assert!(g.add_edge("a", "b"));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_add_edge_missing_node() {
        let mut g = JobGraphViz::new();
        g.add_node(VizNode::new("a", "A", NodeStatus::Pending));
        assert!(!g.add_edge("a", "z"));
    }

    #[test]
    fn test_set_status() {
        let mut g = make_simple_graph();
        assert!(g.set_status("c", NodeStatus::Running));
        assert_eq!(g.nodes["c"].status, NodeStatus::Running);
    }

    #[test]
    fn test_set_status_missing() {
        let mut g = make_simple_graph();
        assert!(!g.set_status("zzz", NodeStatus::Completed));
    }

    #[test]
    fn test_no_cycle() {
        let g = make_simple_graph();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_has_cycle() {
        let mut g = JobGraphViz::new();
        g.add_node(VizNode::new("a", "A", NodeStatus::Pending));
        g.add_node(VizNode::new("b", "B", NodeStatus::Pending));
        g.add_edge("a", "b");
        g.add_edge("b", "a"); // cycle
        assert!(g.has_cycle());
    }

    #[test]
    fn test_to_dot_contains_nodes() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("digraph jobs"));
        assert!(dot.contains("\"a\""));
        assert!(dot.contains("\"b\""));
    }

    #[test]
    fn test_to_dot_contains_edges() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("\"a\" -> \"b\""));
        assert!(dot.contains("\"b\" -> \"c\""));
    }

    #[test]
    fn test_root_nodes() {
        let g = make_simple_graph();
        let roots = g.root_nodes();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "a");
    }

    #[test]
    fn test_leaf_nodes() {
        let g = make_simple_graph();
        let leaves = g.leaf_nodes();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].id, "c");
    }

    #[test]
    fn test_empty_graph_no_cycle() {
        let g = JobGraphViz::new();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_node_count_edge_count() {
        let g = make_simple_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }
}
