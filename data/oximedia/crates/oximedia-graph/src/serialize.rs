//! Graph serialization: DOT format output, adjacency list, and edge list export.

use std::collections::HashMap;

/// A simple graph representation for serialization purposes.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SerializableGraph {
    /// Nodes: id -> optional label.
    pub nodes: HashMap<usize, String>,
    /// Edges: (from, to) -> optional label.
    pub edges: Vec<(usize, usize, Option<String>)>,
    /// Whether the graph is directed.
    pub directed: bool,
    /// Optional graph name.
    pub name: Option<String>,
}

impl SerializableGraph {
    /// Create a new empty directed graph.
    #[allow(dead_code)]
    pub fn new_directed() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            directed: true,
            name: None,
        }
    }

    /// Create a new empty undirected graph.
    #[allow(dead_code)]
    pub fn new_undirected() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            directed: false,
            name: None,
        }
    }

    /// Add a node with an optional label.
    #[allow(dead_code)]
    pub fn add_node(&mut self, id: usize, label: impl Into<String>) {
        self.nodes.insert(id, label.into());
    }

    /// Add an edge between two nodes with an optional label.
    #[allow(dead_code)]
    pub fn add_edge(&mut self, from: usize, to: usize, label: Option<String>) {
        self.edges.push((from, to, label));
    }

    /// Serialize to DOT (Graphviz) format.
    #[allow(dead_code)]
    pub fn to_dot(&self) -> String {
        let graph_type = if self.directed { "digraph" } else { "graph" };
        let name = self.name.as_deref().unwrap_or("G");
        let edge_op = if self.directed { "->" } else { "--" };

        let mut out = String::new();
        out.push_str(&format!("{graph_type} {name} {{\n"));

        // Sort node ids for deterministic output
        let mut node_ids: Vec<usize> = self.nodes.keys().copied().collect();
        node_ids.sort_unstable();
        for id in node_ids {
            let label = &self.nodes[&id];
            if label.is_empty() {
                out.push_str(&format!("    {id};\n"));
            } else {
                out.push_str(&format!("    {id} [label=\"{label}\"];\n"));
            }
        }

        for (from, to, label) in &self.edges {
            if let Some(lbl) = label {
                out.push_str(&format!("    {from} {edge_op} {to} [label=\"{lbl}\"];\n"));
            } else {
                out.push_str(&format!("    {from} {edge_op} {to};\n"));
            }
        }

        out.push_str("}\n");
        out
    }

    /// Serialize to adjacency list format.
    /// Each line: `<node_id>: <neighbor1> <neighbor2> ...`
    #[allow(dead_code)]
    pub fn to_adjacency_list(&self) -> String {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &id in self.nodes.keys() {
            adj.entry(id).or_default();
        }
        for &(from, to, _) in &self.edges {
            adj.entry(from).or_default().push(to);
            if !self.directed {
                adj.entry(to).or_default().push(from);
            }
        }

        let mut node_ids: Vec<usize> = adj.keys().copied().collect();
        node_ids.sort_unstable();

        let mut out = String::new();
        for id in node_ids {
            let mut neighbors = adj[&id].clone();
            neighbors.sort_unstable();
            let neighbor_str = neighbors
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!("{id}: {neighbor_str}\n"));
        }
        out
    }

    /// Serialize to edge list format.
    /// Each line: `<from> <to>` or `<from> <to> <label>`.
    #[allow(dead_code)]
    pub fn to_edge_list(&self) -> String {
        let mut out = String::new();
        for &(from, to, ref label) in &self.edges {
            if let Some(lbl) = label {
                out.push_str(&format!("{from} {to} {lbl}\n"));
            } else {
                out.push_str(&format!("{from} {to}\n"));
            }
        }
        out
    }

    /// Returns the number of nodes.
    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    #[allow(dead_code)]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> SerializableGraph {
        let mut g = SerializableGraph::new_directed();
        g.add_node(0, "A");
        g.add_node(1, "B");
        g.add_node(2, "C");
        g.add_edge(0, 1, None);
        g.add_edge(1, 2, Some("edge".to_string()));
        g
    }

    #[test]
    fn test_node_count() {
        let g = make_simple_graph();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_edge_count() {
        let g = make_simple_graph();
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_dot_contains_digraph() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_dot_contains_nodes() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("label=\"A\""));
        assert!(dot.contains("label=\"B\""));
    }

    #[test]
    fn test_dot_contains_edge_op() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_dot_undirected_uses_double_dash() {
        let mut g = SerializableGraph::new_undirected();
        g.add_node(0, "X");
        g.add_node(1, "Y");
        g.add_edge(0, 1, None);
        let dot = g.to_dot();
        assert!(dot.contains("--"));
        assert!(!dot.contains("->"));
    }

    #[test]
    fn test_dot_labeled_edge() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("label=\"edge\""));
    }

    #[test]
    fn test_adjacency_list_contains_nodes() {
        let g = make_simple_graph();
        let al = g.to_adjacency_list();
        assert!(al.contains("0:"));
        assert!(al.contains("1:"));
        assert!(al.contains("2:"));
    }

    #[test]
    fn test_adjacency_list_has_edges() {
        let g = make_simple_graph();
        let al = g.to_adjacency_list();
        // Node 0 should have neighbor 1
        assert!(al.contains("0: 1"));
    }

    #[test]
    fn test_edge_list_format() {
        let g = make_simple_graph();
        let el = g.to_edge_list();
        assert!(el.contains("0 1"));
        assert!(el.contains("1 2 edge"));
    }

    #[test]
    fn test_empty_graph_dot() {
        let g = SerializableGraph::new_directed();
        let dot = g.to_dot();
        assert!(dot.contains("digraph G {"));
        assert!(dot.contains("}"));
    }

    #[test]
    fn test_graph_with_name() {
        let mut g = SerializableGraph::new_directed();
        g.name = Some("MyGraph".to_string());
        let dot = g.to_dot();
        assert!(dot.contains("digraph MyGraph {"));
    }

    #[test]
    fn test_adjacency_list_undirected_bidirectional() {
        let mut g = SerializableGraph::new_undirected();
        g.add_node(0, "A");
        g.add_node(1, "B");
        g.add_edge(0, 1, None);
        let al = g.to_adjacency_list();
        // Both 0->1 and 1->0 should appear
        assert!(al.contains("0: 1"));
        assert!(al.contains("1: 0"));
    }
}
