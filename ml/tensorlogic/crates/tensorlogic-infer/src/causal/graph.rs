//! Causal graph (DAG) structure and graph-theoretic queries.
//!
//! Defines [`CausalGraph`] plus its impl block: d-separation, ancestors,
//! descendants, and backdoor-path reachability primitives.

use std::collections::{HashMap, HashSet, VecDeque};

use super::error::CausalError;

// ---------------------------------------------------------------------------
// CausalGraph
// ---------------------------------------------------------------------------

/// A Directed Acyclic Graph (DAG) representing causal structure among variables.
///
/// Nodes are identified by string names; edges encode direct causal relationships
/// (parent → child). The graph enforces acyclicity lazily via [`CausalGraph::is_acyclic`].
#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub(super) nodes: Vec<String>,
    /// Directed edges stored as (parent_idx, child_idx) index pairs.
    pub(super) edges: Vec<(usize, usize)>,
}

impl CausalGraph {
    /// Create a new causal graph with the given variable names.
    pub fn new(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            edges: Vec::new(),
        }
    }

    /// Return the index of a node by name, or `None` if it does not exist.
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n == name)
    }

    /// Add a directed edge `parent → child`.
    ///
    /// Returns [`CausalError::NodeNotFound`] if either node is absent.
    /// Does not check for cycles — call [`CausalGraph::is_acyclic`] separately.
    pub fn add_edge(&mut self, parent: &str, child: &str) -> Result<(), CausalError> {
        let p = self
            .node_index(parent)
            .ok_or_else(|| CausalError::NodeNotFound(parent.to_string()))?;
        let c = self
            .node_index(child)
            .ok_or_else(|| CausalError::NodeNotFound(child.to_string()))?;
        self.edges.push((p, c));
        Ok(())
    }

    /// Return direct parents of `node`.
    pub fn parents_of(&self, node: &str) -> Vec<String> {
        match self.node_index(node) {
            None => vec![],
            Some(idx) => self
                .edges
                .iter()
                .filter(|&&(_, c)| c == idx)
                .map(|&(p, _)| self.nodes[p].clone())
                .collect(),
        }
    }

    /// Return direct children of `node`.
    pub fn children_of(&self, node: &str) -> Vec<String> {
        match self.node_index(node) {
            None => vec![],
            Some(idx) => self
                .edges
                .iter()
                .filter(|&&(p, _)| p == idx)
                .map(|&(_, c)| self.nodes[c].clone())
                .collect(),
        }
    }

    /// Return all ancestors of `node` (transitive parents), excluding the node itself.
    pub fn ancestors_of(&self, node: &str) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(start) = self.node_index(node) {
            queue.push_back(start);
        }
        while let Some(cur) = queue.pop_front() {
            for &(p, c) in &self.edges {
                if c == cur && !visited.contains(&p) {
                    visited.insert(p);
                    queue.push_back(p);
                }
            }
        }
        visited.into_iter().map(|i| self.nodes[i].clone()).collect()
    }

    /// Return all descendants of `node` (transitive children), excluding the node itself.
    pub fn descendants_of(&self, node: &str) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(start) = self.node_index(node) {
            queue.push_back(start);
        }
        while let Some(cur) = queue.pop_front() {
            for &(p, c) in &self.edges {
                if p == cur && !visited.contains(&c) {
                    visited.insert(c);
                    queue.push_back(c);
                }
            }
        }
        visited.into_iter().map(|i| self.nodes[i].clone()).collect()
    }

    /// Check whether the graph is acyclic using Kahn's BFS topological sort algorithm.
    ///
    /// Returns `true` if the graph is a valid DAG.
    pub fn is_acyclic(&self) -> bool {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for &(_, c) in &self.edges {
            in_degree[c] += 1;
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut processed = 0usize;
        while let Some(cur) = queue.pop_front() {
            processed += 1;
            for &(p, c) in &self.edges {
                if p == cur {
                    in_degree[c] -= 1;
                    if in_degree[c] == 0 {
                        queue.push_back(c);
                    }
                }
            }
        }
        processed == n
    }

    /// Return all node names.
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Test d-separation: is `x` d-separated from `y` given the observed set `observed`?
    ///
    /// Uses the Bayes-Ball algorithm on the moral graph / active path traversal.
    /// A path is *active* given `observed` if:
    /// - At every non-collider on the path, the node is NOT in `observed`.
    /// - At every collider, the collider OR one of its descendants IS in `observed`.
    pub fn d_separated(&self, x: &str, y: &str, observed: &[&str]) -> bool {
        let x_idx = match self.node_index(x) {
            Some(i) => i,
            None => return true,
        };
        let y_idx = match self.node_index(y) {
            Some(i) => i,
            None => return true,
        };
        if x_idx == y_idx {
            return false;
        }

        let obs_set: HashSet<usize> = observed
            .iter()
            .filter_map(|&name| self.node_index(name))
            .collect();

        // Pre-compute descendants of all observed nodes (needed for collider check).
        let mut obs_or_desc: HashSet<usize> = obs_set.clone();
        for &o in &obs_set {
            let node_name = &self.nodes[o].clone();
            for desc in self.descendants_of(node_name) {
                if let Some(di) = self.node_index(&desc) {
                    obs_or_desc.insert(di);
                }
            }
        }

        // State: (node_idx, arrived_via_child: bool)
        // arrived_via_child = true  → we arrived at this node from one of its children (going "up")
        // arrived_via_child = false → we arrived from a parent (going "down")
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        let mut queue: VecDeque<(usize, bool)> = VecDeque::new();

        // We can start from x going both up and down.
        queue.push_back((x_idx, true));
        queue.push_back((x_idx, false));

        while let Some((cur, via_child)) = queue.pop_front() {
            if !visited.insert((cur, via_child)) {
                continue;
            }
            if cur == y_idx {
                return false; // active path found → NOT d-separated
            }

            if via_child && !obs_set.contains(&cur) {
                // Traversing up (non-collider direction): pass through parents and children
                // go up to parents
                for &(p, c) in &self.edges {
                    if c == cur {
                        let state = (p, true);
                        if !visited.contains(&state) {
                            queue.push_back(state);
                        }
                    }
                }
                // go down to children
                for &(p, c) in &self.edges {
                    if p == cur {
                        let state = (c, false);
                        if !visited.contains(&state) {
                            queue.push_back(state);
                        }
                    }
                }
            }

            if !via_child {
                // Arriving from above (going down)
                if !obs_set.contains(&cur) {
                    // Non-collider going down: continue downward
                    for &(p, c) in &self.edges {
                        if p == cur {
                            let state = (c, false);
                            if !visited.contains(&state) {
                                queue.push_back(state);
                            }
                        }
                    }
                }
                // Collider activation: if cur (collider) or descendant is observed, go up
                if obs_or_desc.contains(&cur) {
                    for &(p, c) in &self.edges {
                        if c == cur {
                            let state = (p, true);
                            if !visited.contains(&state) {
                                queue.push_back(state);
                            }
                        }
                    }
                }
            }
        }

        true // no active path found → d-separated
    }

    /// Internal helper: collect all undirected (bidirectional) adjacency paths from `src` to `dst`
    /// that are *backdoor paths* (i.e. paths that enter `src` via a parent of `src`).
    /// Returns true if there exists at least one unblocked backdoor path given `adjustment_set`.
    pub(super) fn has_unblocked_backdoor_path(
        &self,
        src: usize,
        dst: usize,
        adjustment_set: &HashSet<usize>,
    ) -> bool {
        // A backdoor path from src to dst is an undirected path that starts by going
        // "upward" from src (i.e. first step is via a parent of src).
        // We block a path by conditioning on a non-collider on the path,
        // or by NOT conditioning on a collider / its descendant.
        //
        // We use a simplified reachability check:
        // A node Z blocks a path if it is a non-collider on the path AND Z is in adjustment_set,
        // or it is a collider not in adjustment_set and none of its descendants are.
        //
        // State: (current_node, previous_node, direction: true=going_up)
        // We only consider paths that leave src going upward (backdoor).

        // Compute descendants for collider check
        let mut desc_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        for i in 0..self.nodes.len() {
            let desc_names = self.descendants_of(&self.nodes[i].clone());
            let desc_idxs: HashSet<usize> = desc_names
                .iter()
                .filter_map(|n| self.node_index(n))
                .collect();
            desc_map.insert(i, desc_idxs);
        }

        let is_in_adj_or_desc = |node: usize| -> bool {
            if adjustment_set.contains(&node) {
                return true;
            }
            if let Some(descs) = desc_map.get(&node) {
                return descs.iter().any(|d| adjustment_set.contains(d));
            }
            false
        };

        // State: (current_node, prev_node, arrived_via_up: bool)
        let mut visited: HashSet<(usize, usize, bool)> = HashSet::new();
        let mut queue: VecDeque<(usize, usize, bool)> = VecDeque::new();

        // Only start on parents of src (backdoor = entering src from above)
        for &(p, c) in &self.edges {
            if c == src {
                // parent p of src: going up (from src to p)
                // The first step is upward. p is a non-collider relative to src→p.
                // Block if p is in adjustment_set
                if !adjustment_set.contains(&p) {
                    queue.push_back((p, src, true));
                }
            }
        }

        while let Some((cur, prev, going_up)) = queue.pop_front() {
            if !visited.insert((cur, prev, going_up)) {
                continue;
            }
            if cur == dst {
                return true;
            }

            // Explore neighbors
            // Build set of parents and children of cur
            let parents: Vec<usize> = self
                .edges
                .iter()
                .filter(|&&(_, c)| c == cur)
                .map(|&(p, _)| p)
                .collect();
            let children: Vec<usize> = self
                .edges
                .iter()
                .filter(|&&(p, _)| p == cur)
                .map(|&(_, c)| c)
                .collect();

            for &next in parents.iter().chain(children.iter()) {
                if next == prev {
                    continue;
                }
                // Determine if cur is a collider on the segment prev→cur→next
                // cur is a collider iff both prev and next are parents of cur
                let prev_is_parent_of_cur = parents.contains(&prev);
                let next_is_parent_of_cur = parents.contains(&next);
                let is_collider = prev_is_parent_of_cur && next_is_parent_of_cur;

                let blocked = if is_collider {
                    // Collider: blocked unless cur or its descendant is in adjustment set
                    !is_in_adj_or_desc(cur)
                } else {
                    // Non-collider: blocked if cur is in adjustment set
                    adjustment_set.contains(&cur)
                };

                if !blocked {
                    let next_going_up = parents.contains(&next);
                    let state = (next, cur, next_going_up);
                    if !visited.contains(&state) {
                        queue.push_back(state);
                    }
                }
            }
        }

        false
    }

    /// Check whether there is a directed path from `src` to `dst`.
    pub fn has_directed_path(&self, src: &str, dst: &str) -> bool {
        let src_idx = match self.node_index(src) {
            Some(i) => i,
            None => return false,
        };
        let dst_idx = match self.node_index(dst) {
            Some(i) => i,
            None => return false,
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(src_idx);
        while let Some(cur) = queue.pop_front() {
            if cur == dst_idx {
                return true;
            }
            if !visited.insert(cur) {
                continue;
            }
            for &(p, c) in &self.edges {
                if p == cur && !visited.contains(&c) {
                    queue.push_back(c);
                }
            }
        }
        false
    }
}
