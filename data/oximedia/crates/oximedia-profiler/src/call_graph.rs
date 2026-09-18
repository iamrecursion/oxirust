//! Call graph profiling — nodes, edges, hot-path analysis.
#![allow(dead_code)]

use std::collections::HashMap;

/// A single node in the call graph.
#[derive(Debug, Clone)]
pub struct CallNode {
    /// Fully-qualified function name.
    pub name: String,
    /// Total inclusive time in nanoseconds.
    pub inclusive_ns: u64,
    /// Total exclusive (self) time in nanoseconds.
    pub exclusive_ns: u64,
    /// Number of calls.
    pub call_count: u64,
    /// Indices of child nodes.
    pub children: Vec<usize>,
}

impl CallNode {
    /// Create a new call node.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inclusive_ns: 0,
            exclusive_ns: 0,
            call_count: 0,
            children: Vec::new(),
        }
    }

    /// Returns `true` if this node has no children (leaf function).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Average call duration in nanoseconds.
    pub fn avg_ns(&self) -> u64 {
        self.inclusive_ns.checked_div(self.call_count).unwrap_or(0)
    }
}

/// A directed edge between two call-graph nodes.
#[derive(Debug, Clone, Copy)]
pub struct CallEdge {
    /// Index of the caller node.
    pub caller: usize,
    /// Index of the callee node.
    pub callee: usize,
    /// Number of times this specific call site was invoked.
    pub call_count: u64,
    /// Total time spent in the callee from this edge (ns).
    pub total_ns: u64,
}

impl CallEdge {
    /// Create a new call edge.
    pub fn new(caller: usize, callee: usize) -> Self {
        Self {
            caller,
            callee,
            call_count: 0,
            total_ns: 0,
        }
    }

    /// Number of times this call was made.
    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    /// Returns `true` if this edge was traversed at least once.
    pub fn is_active(&self) -> bool {
        self.call_count > 0
    }
}

/// A complete call graph for a profiling session.
#[derive(Debug, Default)]
pub struct CallGraph {
    nodes: Vec<CallNode>,
    edges: Vec<CallEdge>,
    name_index: HashMap<String, usize>,
}

impl CallGraph {
    /// Create an empty call graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or insert a node by name, returning its index.
    pub fn get_or_insert(&mut self, name: impl Into<String> + Clone) -> usize {
        let key = name.clone().into();
        if let Some(&idx) = self.name_index.get(&key) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(CallNode::new(name));
        self.name_index.insert(key, idx);
        idx
    }

    /// Record a call from `caller_name` to `callee_name` with given duration.
    pub fn add_call(
        &mut self,
        caller_name: impl Into<String> + Clone,
        callee_name: impl Into<String> + Clone,
        duration_ns: u64,
    ) {
        let caller_idx = self.get_or_insert(caller_name);
        let callee_idx = self.get_or_insert(callee_name);

        // Update node stats.
        self.nodes[callee_idx].call_count += 1;
        self.nodes[callee_idx].inclusive_ns += duration_ns;

        // Add child link if not already present.
        if !self.nodes[caller_idx].children.contains(&callee_idx) {
            self.nodes[caller_idx].children.push(callee_idx);
        }

        // Update or create edge.
        if let Some(edge) = self
            .edges
            .iter_mut()
            .find(|e| e.caller == caller_idx && e.callee == callee_idx)
        {
            edge.call_count += 1;
            edge.total_ns += duration_ns;
        } else {
            let mut edge = CallEdge::new(caller_idx, callee_idx);
            edge.call_count = 1;
            edge.total_ns = duration_ns;
            self.edges.push(edge);
        }
    }

    /// Return indices of root nodes (nodes that have no incoming edges).
    pub fn root_nodes(&self) -> Vec<usize> {
        let callees: std::collections::HashSet<usize> =
            self.edges.iter().map(|e| e.callee).collect();
        (0..self.nodes.len())
            .filter(|i| !callees.contains(i))
            .collect()
    }

    /// Return the indices of nodes forming the hottest call path (by inclusive time).
    /// The path starts from any root and greedily follows the child with the highest
    /// inclusive time until reaching a leaf.
    pub fn hot_path(&self) -> Vec<usize> {
        let roots = self.root_nodes();
        if roots.is_empty() {
            return Vec::new();
        }
        // Pick the root with the highest inclusive time.
        let start = *roots
            .iter()
            .max_by_key(|&&i| self.nodes[i].inclusive_ns)
            .expect("invariant: roots is non-empty after early return check");

        let mut path = vec![start];
        let mut current = start;
        loop {
            let node = &self.nodes[current];
            if node.children.is_empty() {
                break;
            }
            let next = *node
                .children
                .iter()
                .max_by_key(|&&i| self.nodes[i].inclusive_ns)
                .expect("invariant: children is non-empty (checked by loop guard)");
            path.push(next);
            current = next;
        }
        path
    }

    /// Return the node at `index`, if it exists.
    pub fn node(&self, index: usize) -> Option<&CallNode> {
        self.nodes.get(index)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Edges slice.
    pub fn edges(&self) -> &[CallEdge] {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_graph() -> CallGraph {
        let mut g = CallGraph::new();
        g.add_call("main", "encode", 1000);
        g.add_call("main", "decode", 500);
        g.add_call("encode", "compress", 800);
        g.add_call("compress", "huffman", 300);
        g
    }

    #[test]
    fn test_call_node_is_leaf() {
        let leaf = CallNode::new("leaf");
        assert!(leaf.is_leaf());
        let mut parent = CallNode::new("parent");
        parent.children.push(0);
        assert!(!parent.is_leaf());
    }

    #[test]
    fn test_call_node_avg_ns_zero_calls() {
        let node = CallNode::new("fn");
        assert_eq!(node.avg_ns(), 0);
    }

    #[test]
    fn test_call_node_avg_ns() {
        let mut node = CallNode::new("fn");
        node.call_count = 4;
        node.inclusive_ns = 200;
        assert_eq!(node.avg_ns(), 50);
    }

    #[test]
    fn test_call_edge_call_count() {
        let mut e = CallEdge::new(0, 1);
        assert_eq!(e.call_count(), 0);
        assert!(!e.is_active());
        e.call_count = 3;
        assert_eq!(e.call_count(), 3);
        assert!(e.is_active());
    }

    #[test]
    fn test_graph_node_insertion() {
        let mut g = CallGraph::new();
        let i = g.get_or_insert("alpha");
        let j = g.get_or_insert("alpha"); // same name -> same index
        assert_eq!(i, j);
        let k = g.get_or_insert("beta");
        assert_ne!(i, k);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_add_call_increments_stats() {
        let mut g = CallGraph::new();
        g.add_call("a", "b", 100);
        g.add_call("a", "b", 200);
        let b_idx = *g.name_index.get("b").expect("should succeed in test");
        let b = g.node(b_idx).expect("should succeed in test");
        assert_eq!(b.call_count, 2);
        assert_eq!(b.inclusive_ns, 300);
    }

    #[test]
    fn test_add_call_edge_count() {
        let g = build_simple_graph();
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_add_call_no_duplicate_edges() {
        let mut g = CallGraph::new();
        g.add_call("a", "b", 10);
        g.add_call("a", "b", 20);
        // Only one edge from a->b
        let edges: Vec<_> = g.edges().iter().filter(|e| e.caller == 0).collect();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].call_count, 2);
    }

    #[test]
    fn test_root_nodes_simple() {
        let g = build_simple_graph();
        let roots = g.root_nodes();
        assert_eq!(roots.len(), 1);
        let main_idx = *g.name_index.get("main").expect("should succeed in test");
        assert!(roots.contains(&main_idx));
    }

    #[test]
    fn test_root_nodes_disconnected() {
        let mut g = CallGraph::new();
        g.get_or_insert("orphan_a");
        g.get_or_insert("orphan_b");
        let roots = g.root_nodes();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_hot_path_non_empty() {
        let g = build_simple_graph();
        let path = g.hot_path();
        assert!(!path.is_empty());
        // Path must start with main
        let main_idx = *g.name_index.get("main").expect("should succeed in test");
        assert_eq!(path[0], main_idx);
    }

    #[test]
    fn test_hot_path_ends_at_leaf() {
        let g = build_simple_graph();
        let path = g.hot_path();
        let last = *path.last().expect("should succeed in test");
        assert!(g.node(last).expect("should succeed in test").is_leaf());
    }

    #[test]
    fn test_empty_graph_hot_path() {
        let g = CallGraph::new();
        assert!(g.hot_path().is_empty());
    }

    #[test]
    fn test_children_updated() {
        let mut g = CallGraph::new();
        g.add_call("root", "child1", 10);
        g.add_call("root", "child2", 20);
        let root_idx = *g.name_index.get("root").expect("should succeed in test");
        assert_eq!(
            g.node(root_idx)
                .expect("should succeed in test")
                .children
                .len(),
            2
        );
    }
}
