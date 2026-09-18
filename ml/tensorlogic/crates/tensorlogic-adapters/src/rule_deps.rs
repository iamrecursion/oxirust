//! Rule dependency graph for TensorLogic.
//!
//! Builds a directed graph where rules depend on predicates and predicates
//! are defined by rules. Enables cycle detection, stratification, SCC
//! computation, and transitive dependency analysis.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::SymbolTable;

// ─────────────────────────────────────────────────────────────────────────────
// DepNode
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the dependency graph — either a named rule or a named predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DepNode {
    /// A rule identified by its name.
    Rule(String),
    /// A predicate identified by its name.
    Predicate(String),
}

impl DepNode {
    /// The name string inside the variant.
    pub fn name(&self) -> &str {
        match self {
            DepNode::Rule(n) | DepNode::Predicate(n) => n.as_str(),
        }
    }

    /// Returns `true` when this node is a rule.
    pub fn is_rule(&self) -> bool {
        matches!(self, DepNode::Rule(_))
    }

    /// Returns `true` when this node is a predicate.
    pub fn is_predicate(&self) -> bool {
        matches!(self, DepNode::Predicate(_))
    }
}

impl std::fmt::Display for DepNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepNode::Rule(n) => write!(f, "Rule({n})"),
            DepNode::Predicate(n) => write!(f, "Pred({n})"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DepEdge
// ─────────────────────────────────────────────────────────────────────────────

/// The semantics of a directed edge in the dependency graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepEdge {
    /// Rule uses the predicate positively (in head or positive body literal).
    Positive,
    /// Rule uses the predicate under negation.
    Negative,
    /// Rule *defines* (writes to) the predicate — i.e. the head.
    Defines,
}

impl std::fmt::Display for DepEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepEdge::Positive => write!(f, "+"),
            DepEdge::Negative => write!(f, "−"),
            DepEdge::Defines => write!(f, "def"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RuleDependencyGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Directed graph capturing dependencies between rules and predicates.
#[derive(Debug, Clone)]
pub struct RuleDependencyGraph {
    /// Adjacency list: node → list of (neighbour, edge_type).
    edges: HashMap<DepNode, Vec<(DepNode, DepEdge)>>,
    /// Full node set (includes nodes with no outgoing edges).
    nodes: HashSet<DepNode>,
}

impl Default for RuleDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleDependencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        RuleDependencyGraph {
            edges: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Insert a node (idempotent).
    pub fn add_node(&mut self, node: DepNode) {
        self.nodes.insert(node.clone());
        self.edges.entry(node).or_default();
    }

    /// Insert a directed edge from `from` to `to` with edge type `edge`.
    /// Both endpoints are automatically added as nodes.
    pub fn add_edge(&mut self, from: DepNode, to: DepNode, edge: DepEdge) {
        self.add_node(from.clone());
        self.add_node(to.clone());
        self.edges.entry(from).or_default().push((to, edge));
    }

    // ── Construction ──────────────────────────────────────────────────────────

    /// Build a dependency graph from a `SymbolTable`.
    ///
    /// Because `SymbolTable` stores predicates (not first-class rules with
    /// heads/bodies), this method treats each predicate as both a *defining*
    /// entity and a potential *dependency*.  For every predicate `p` it:
    ///
    /// 1. Adds `Predicate(p.name)` as a node.
    /// 2. Creates a synthetic `Rule("<p>_rule")` that defines `p`.
    /// 3. Adds a `Defines` edge from the rule node to the predicate node.
    /// 4. For each argument domain `d` of `p`: adds `Predicate(d)` and a
    ///    `Positive` edge from the rule to that domain predicate (modelling
    ///    that evaluating `p` requires its domain to be populated).
    pub fn from_symbol_table(table: &SymbolTable) -> Self {
        let mut graph = RuleDependencyGraph::new();

        for (pred_name, pred_info) in &table.predicates {
            let pred_node = DepNode::Predicate(pred_name.clone());
            let rule_node = DepNode::Rule(format!("{pred_name}_rule"));

            graph.add_edge(rule_node.clone(), pred_node, DepEdge::Defines);

            for domain_name in &pred_info.arg_domains {
                let domain_node = DepNode::Predicate(domain_name.clone());
                graph.add_edge(rule_node.clone(), domain_node, DepEdge::Positive);
            }
        }

        graph
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// All nodes in the graph.
    pub fn nodes(&self) -> &HashSet<DepNode> {
        &self.nodes
    }

    /// Nodes that `node` has outgoing edges to (successors / dependencies).
    pub fn successors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.edges
            .get(node)
            .map(|v| v.iter().map(|(n, _)| n).collect())
            .unwrap_or_default()
    }

    /// Nodes that have outgoing edges pointing to `node` (predecessors).
    pub fn predecessors(&self, node: &DepNode) -> Vec<&DepNode> {
        self.nodes
            .iter()
            .filter(|n| {
                self.edges
                    .get(n)
                    .map(|v| v.iter().any(|(t, _)| t == node))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    /// Returns `true` if the graph contains at least one directed cycle.
    pub fn has_cycle(&self) -> bool {
        let mut visited: HashSet<&DepNode> = HashSet::new();
        let mut in_stack: HashSet<&DepNode> = HashSet::new();

        for node in &self.nodes {
            if !visited.contains(node) && self.dfs_has_cycle(node, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_has_cycle<'a>(
        &'a self,
        node: &'a DepNode,
        visited: &mut HashSet<&'a DepNode>,
        in_stack: &mut HashSet<&'a DepNode>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        if let Some(neighbours) = self.edges.get(node) {
            for (next, _) in neighbours {
                if !visited.contains(next) {
                    if self.dfs_has_cycle(next, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(next) {
                    return true;
                }
            }
        }

        in_stack.remove(node);
        false
    }

    /// Return the set of nodes that participate in *any* cycle.
    pub fn find_cycle_nodes(&self) -> HashSet<&DepNode> {
        // A node participates in a cycle iff it belongs to an SCC of size > 1
        // OR has a self-loop.
        let sccs = self.strongly_connected_components();
        let mut result: HashSet<&DepNode> = HashSet::new();

        for scc in &sccs {
            if scc.len() > 1 {
                for node in scc {
                    if let Some(n) = self.nodes.get(node) {
                        result.insert(n);
                    }
                }
            } else if scc.len() == 1 {
                // Check self-loop
                let node = &scc[0];
                if let Some(neighbours) = self.edges.get(node) {
                    if neighbours.iter().any(|(t, _)| t == node) {
                        if let Some(n) = self.nodes.get(node) {
                            result.insert(n);
                        }
                    }
                }
            }
        }

        result
    }

    // ── Transitive dependencies ────────────────────────────────────────────────

    /// Compute all nodes reachable from `node` via BFS (all edge types).
    /// The starting node itself is *not* included in the result.
    pub fn transitive_deps(&self, node: &DepNode) -> HashSet<DepNode> {
        let mut visited: HashSet<DepNode> = HashSet::new();
        let mut queue: VecDeque<DepNode> = VecDeque::new();

        // Seed the queue with direct successors.
        if let Some(neighbours) = self.edges.get(node) {
            for (next, _) in neighbours {
                if !visited.contains(next) {
                    visited.insert(next.clone());
                    queue.push_back(next.clone());
                }
            }
        }

        while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = self.edges.get(&current) {
                for (next, _) in neighbours {
                    if !visited.contains(next) {
                        visited.insert(next.clone());
                        queue.push_back(next.clone());
                    }
                }
            }
        }

        visited
    }

    // ── Strongly Connected Components (Kosaraju's algorithm) ──────────────────

    /// Compute all strongly connected components.
    /// Each SCC is returned as a `Vec<DepNode>`; SCCs are in reverse topological
    /// order (i.e. the first SCC has no outgoing edges to later SCCs).
    pub fn strongly_connected_components(&self) -> Vec<Vec<DepNode>> {
        // ── Pass 1: DFS on original graph, record finish order ────────────────
        let mut visited: HashSet<&DepNode> = HashSet::new();
        let mut finish_stack: Vec<&DepNode> = Vec::new();

        for node in &self.nodes {
            if !visited.contains(node) {
                self.kosaraju_dfs_forward(node, &mut visited, &mut finish_stack);
            }
        }

        // ── Build transposed graph ────────────────────────────────────────────
        let transposed = self.transpose();

        // ── Pass 2: DFS on transposed graph in reverse finish order ───────────
        let mut visited2: HashSet<DepNode> = HashSet::new();
        let mut sccs: Vec<Vec<DepNode>> = Vec::new();

        for node in finish_stack.into_iter().rev() {
            if !visited2.contains(node) {
                let mut component: Vec<DepNode> = Vec::new();
                Self::kosaraju_dfs_backward(node, &transposed, &mut visited2, &mut component);
                sccs.push(component);
            }
        }

        sccs
    }

    fn kosaraju_dfs_forward<'a>(
        &'a self,
        node: &'a DepNode,
        visited: &mut HashSet<&'a DepNode>,
        finish_stack: &mut Vec<&'a DepNode>,
    ) {
        visited.insert(node);
        if let Some(neighbours) = self.edges.get(node) {
            for (next, _) in neighbours {
                if !visited.contains(next) {
                    self.kosaraju_dfs_forward(next, visited, finish_stack);
                }
            }
        }
        finish_stack.push(node);
    }

    fn kosaraju_dfs_backward(
        node: &DepNode,
        transposed: &HashMap<DepNode, Vec<DepNode>>,
        visited: &mut HashSet<DepNode>,
        component: &mut Vec<DepNode>,
    ) {
        visited.insert(node.clone());
        component.push(node.clone());

        if let Some(neighbours) = transposed.get(node) {
            for next in neighbours {
                if !visited.contains(next) {
                    Self::kosaraju_dfs_backward(next, transposed, visited, component);
                }
            }
        }
    }

    /// Build the transpose (reverse) of this graph.
    fn transpose(&self) -> HashMap<DepNode, Vec<DepNode>> {
        let mut trans: HashMap<DepNode, Vec<DepNode>> = HashMap::new();

        // Ensure every node appears (even without incoming edges).
        for node in &self.nodes {
            trans.entry(node.clone()).or_default();
        }

        for (from, neighbours) in &self.edges {
            for (to, _) in neighbours {
                trans.entry(to.clone()).or_default().push(from.clone());
            }
        }

        trans
    }

    // ── Stratification ────────────────────────────────────────────────────────

    /// Compute Datalog stratification layers.
    ///
    /// Returns `Ok(layers)` where layers are sorted by stratum index, or
    /// `Err(StratificationError::NegativeCycle{..})` when the graph is
    /// unstratifiable.
    pub fn stratify(&self) -> Result<Vec<StratificationLayer>, StratificationError> {
        // Assign every node an integer stratum starting at 0.
        let mut stratum: HashMap<DepNode, usize> =
            self.nodes.iter().map(|n| (n.clone(), 0_usize)).collect();

        // Iterative fixed-point propagation.
        let max_iters = self.nodes.len().saturating_add(1);
        let mut changed = true;
        let mut iter = 0_usize;

        while changed && iter < max_iters {
            changed = false;
            iter = iter.saturating_add(1);

            for (from, neighbours) in &self.edges {
                let s_from = *stratum.get(from).unwrap_or(&0);
                for (to, edge_kind) in neighbours {
                    let min_stratum = match edge_kind {
                        DepEdge::Positive | DepEdge::Defines => s_from,
                        DepEdge::Negative => s_from.saturating_add(1),
                    };
                    let current = stratum.entry(to.clone()).or_insert(0);
                    if min_stratum > *current {
                        *current = min_stratum;
                        changed = true;
                    }
                }
            }
        }

        // Detect negative cycles: a negative edge (u→v) where stratum[u] >=
        // stratum[v] after convergence indicates an unstratifiable graph.
        let mut cycle_nodes: Vec<String> = Vec::new();
        for (from, neighbours) in &self.edges {
            let s_from = *stratum.get(from).unwrap_or(&0);
            for (to, edge_kind) in neighbours {
                if *edge_kind == DepEdge::Negative {
                    let s_to = *stratum.get(to).unwrap_or(&0);
                    if s_from >= s_to {
                        cycle_nodes.push(from.name().to_owned());
                        cycle_nodes.push(to.name().to_owned());
                    }
                }
            }
        }

        if !cycle_nodes.is_empty() {
            cycle_nodes.sort();
            cycle_nodes.dedup();
            return Err(StratificationError::NegativeCycle {
                participating_nodes: cycle_nodes,
            });
        }

        // Group nodes by stratum.
        let mut layers_map: HashMap<usize, Vec<DepNode>> = HashMap::new();
        for (node, s) in &stratum {
            layers_map.entry(*s).or_default().push(node.clone());
        }

        // Determine which strata have at least one negative incoming edge.
        let mut negative_strata: HashSet<usize> = HashSet::new();
        for (from, neighbours) in &self.edges {
            let s_from = *stratum.get(from).unwrap_or(&0);
            for (to, edge_kind) in neighbours {
                if *edge_kind == DepEdge::Negative {
                    let s_to = *stratum.get(to).unwrap_or(&0);
                    // The target stratum is strictly higher due to the +1 rule.
                    if s_to > s_from {
                        negative_strata.insert(s_to);
                    }
                }
            }
        }

        let mut sorted_strata: Vec<usize> = layers_map.keys().copied().collect();
        sorted_strata.sort_unstable();

        let layers: Vec<StratificationLayer> = sorted_strata
            .into_iter()
            .map(|s| {
                let mut nodes = layers_map.remove(&s).unwrap_or_default();
                nodes.sort();
                StratificationLayer {
                    stratum: s,
                    nodes,
                    has_negation: negative_strata.contains(&s),
                }
            })
            .collect();

        Ok(layers)
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    /// Render as a human-readable ASCII adjacency list (for debugging).
    pub fn to_ascii(&self) -> String {
        let mut buf = String::new();
        let mut sorted_nodes: Vec<&DepNode> = self.nodes.iter().collect();
        sorted_nodes.sort();

        for node in sorted_nodes {
            buf.push_str(&format!("{node}"));
            let mut succs: Vec<String> = self
                .edges
                .get(node)
                .map(|v| v.iter().map(|(n, e)| format!("  →{n}[{e}]")).collect())
                .unwrap_or_default();
            succs.sort();

            if succs.is_empty() {
                buf.push_str(" (leaf)\n");
            } else {
                buf.push('\n');
                for s in succs {
                    buf.push_str(&s);
                    buf.push('\n');
                }
            }
        }

        buf
    }

    /// Render as Graphviz DOT format.
    pub fn to_dot(&self) -> String {
        let mut buf = String::from("digraph rule_deps {\n    rankdir=LR;\n");

        // Node declarations with shape hints.
        let mut sorted_nodes: Vec<&DepNode> = self.nodes.iter().collect();
        sorted_nodes.sort();

        for node in &sorted_nodes {
            let (shape, label) = match node {
                DepNode::Rule(n) => ("box", format!("Rule\\n{n}")),
                DepNode::Predicate(n) => ("ellipse", format!("Pred\\n{n}")),
            };
            let id = dot_id(node);
            buf.push_str(&format!("    {id} [label=\"{label}\" shape={shape}];\n"));
        }

        // Edge declarations.
        for from in &sorted_nodes {
            if let Some(neighbours) = self.edges.get(from) {
                let mut sorted_neighbours: Vec<&(DepNode, DepEdge)> = neighbours.iter().collect();
                sorted_neighbours.sort_by_key(|(n, _)| n);

                for (to, edge_kind) in sorted_neighbours {
                    let from_id = dot_id(from);
                    let to_id = dot_id(to);
                    let (style, label) = match edge_kind {
                        DepEdge::Positive => ("solid", "pos"),
                        DepEdge::Negative => ("dashed", "neg"),
                        DepEdge::Defines => ("bold", "def"),
                    };
                    buf.push_str(&format!(
                        "    {from_id} -> {to_id} [label=\"{label}\" style={style}];\n"
                    ));
                }
            }
        }

        buf.push('}');
        buf
    }
}

/// Sanitise a node name to a valid DOT identifier.
fn dot_id(node: &DepNode) -> String {
    let prefix = if node.is_rule() { "r_" } else { "p_" };
    let name: String = node
        .name()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{prefix}{name}")
}

// ─────────────────────────────────────────────────────────────────────────────
// StratificationLayer
// ─────────────────────────────────────────────────────────────────────────────

/// A set of nodes that can be evaluated at the same stratum.
#[derive(Debug, Clone)]
pub struct StratificationLayer {
    /// Zero-based stratum index (lower = evaluated first).
    pub stratum: usize,
    /// All nodes at this stratum (sorted for determinism).
    pub nodes: Vec<DepNode>,
    /// `true` when at least one incoming edge to this stratum is `Negative`.
    pub has_negation: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// StratificationError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the stratification algorithm.
#[derive(Debug, Clone)]
pub enum StratificationError {
    /// The graph contains a cycle involving at least one negative edge.
    NegativeCycle {
        /// Names of the nodes that participate in the negative cycle.
        participating_nodes: Vec<String>,
    },
    /// General stratification failure with a descriptive message.
    UnstratifiableGraph(String),
}

impl std::fmt::Display for StratificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StratificationError::NegativeCycle {
                participating_nodes,
            } => {
                write!(
                    f,
                    "Negative cycle detected involving nodes: [{}]",
                    participating_nodes.join(", ")
                )
            }
            StratificationError::UnstratifiableGraph(msg) => {
                write!(f, "Unstratifiable graph: {msg}")
            }
        }
    }
}

impl std::error::Error for StratificationError {}

// ─────────────────────────────────────────────────────────────────────────────
// DepGraphStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a `RuleDependencyGraph`.
#[derive(Debug, Clone)]
pub struct DepGraphStats {
    /// Total node count.
    pub num_nodes: usize,
    /// Total edge count.
    pub num_edges: usize,
    /// Number of rule nodes.
    pub num_rules: usize,
    /// Number of predicate nodes.
    pub num_predicates: usize,
    /// Whether the graph contains any directed cycle.
    pub has_cycles: bool,
    /// Number of strongly connected components.
    pub num_sccs: usize,
    /// Size of the largest SCC.
    pub max_scc_size: usize,
    /// Number of strata (`None` when the graph is not stratifiable).
    pub num_strata: Option<usize>,
    /// Length of the longest chain of dependencies (BFS diameter from any node).
    pub longest_dependency_chain: usize,
}

impl DepGraphStats {
    /// Compute statistics for the given graph.
    pub fn compute(graph: &RuleDependencyGraph) -> Self {
        let num_nodes = graph.node_count();
        let num_edges = graph.edge_count();
        let num_rules = graph.nodes.iter().filter(|n| n.is_rule()).count();
        let num_predicates = graph.nodes.iter().filter(|n| n.is_predicate()).count();
        let has_cycles = graph.has_cycle();

        let sccs = graph.strongly_connected_components();
        let num_sccs = sccs.len();
        let max_scc_size = sccs.iter().map(|s| s.len()).max().unwrap_or(0);

        let num_strata = match graph.stratify() {
            Ok(layers) => Some(layers.len()),
            Err(_) => None,
        };

        let longest_dependency_chain = compute_longest_chain(graph);

        DepGraphStats {
            num_nodes,
            num_edges,
            num_rules,
            num_predicates,
            has_cycles,
            num_sccs,
            max_scc_size,
            num_strata,
            longest_dependency_chain,
        }
    }
}

/// BFS-based longest chain length across all starting nodes.
fn compute_longest_chain(graph: &RuleDependencyGraph) -> usize {
    let mut max_len = 0_usize;

    for start in &graph.nodes {
        let mut dist: HashMap<&DepNode, usize> = HashMap::new();
        let mut queue: VecDeque<&DepNode> = VecDeque::new();
        dist.insert(start, 0);
        queue.push_back(start);

        while let Some(cur) = queue.pop_front() {
            let cur_dist = *dist.get(cur).unwrap_or(&0);
            if let Some(neighbours) = graph.edges.get(cur) {
                for (next, _) in neighbours {
                    if !dist.contains_key(next) {
                        dist.insert(next, cur_dist + 1);
                        queue.push_back(next);
                        if cur_dist + 1 > max_len {
                            max_len = cur_dist + 1;
                        }
                    }
                }
            }
        }
    }

    max_len
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn rule(n: &str) -> DepNode {
        DepNode::Rule(n.to_owned())
    }

    fn pred(n: &str) -> DepNode {
        DepNode::Predicate(n.to_owned())
    }

    // ── DepNode ────────────────────────────────────────────────────────────────

    #[test]
    fn test_dep_node_name() {
        assert_eq!(rule("foo").name(), "foo");
        assert_eq!(pred("bar").name(), "bar");
    }

    #[test]
    fn test_dep_node_is_rule_predicate() {
        let r = rule("r1");
        let p = pred("p1");
        assert!(r.is_rule());
        assert!(!r.is_predicate());
        assert!(p.is_predicate());
        assert!(!p.is_rule());
    }

    // ── Graph construction ────────────────────────────────────────────────────

    #[test]
    fn test_add_node_and_edge() {
        let mut g = RuleDependencyGraph::new();
        g.add_node(rule("r1"));
        g.add_node(pred("p1"));
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0);

        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);
        assert_eq!(g.edge_count(), 1);
        // add_edge should not duplicate nodes
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_successors() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);
        g.add_edge(rule("r1"), pred("p2"), DepEdge::Positive);

        let mut succs: Vec<&DepNode> = g.successors(&rule("r1"));
        succs.sort();
        assert_eq!(succs.len(), 2);
        assert!(succs.contains(&&pred("p1")));
        assert!(succs.contains(&&pred("p2")));
    }

    #[test]
    fn test_predecessors() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);
        g.add_edge(rule("r2"), pred("p1"), DepEdge::Positive);

        let preds = g.predecessors(&pred("p1"));
        assert_eq!(preds.len(), 2);
        assert!(preds.contains(&&rule("r1")));
        assert!(preds.contains(&&rule("r2")));
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    #[test]
    fn test_has_cycle_false() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);
        g.add_edge(pred("p1"), pred("p2"), DepEdge::Positive);
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_has_cycle_true() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("a"), DepEdge::Positive);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_find_cycle_nodes() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("a"), DepEdge::Positive);
        // pred("c") is outside the cycle
        g.add_edge(pred("c"), pred("a"), DepEdge::Positive);

        let cycle_nodes = g.find_cycle_nodes();
        assert!(cycle_nodes.contains(&pred("a")));
        assert!(cycle_nodes.contains(&pred("b")));
        assert!(!cycle_nodes.contains(&pred("c")));
    }

    // ── Transitive deps ───────────────────────────────────────────────────────

    #[test]
    fn test_transitive_deps_simple() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("c"), DepEdge::Positive);

        let deps = g.transitive_deps(&pred("a"));
        assert!(deps.contains(&pred("b")));
        assert!(deps.contains(&pred("c")));
        assert!(!deps.contains(&pred("a")));
    }

    #[test]
    fn test_transitive_deps_empty() {
        let mut g = RuleDependencyGraph::new();
        g.add_node(pred("leaf"));

        let deps = g.transitive_deps(&pred("leaf"));
        assert!(deps.is_empty());
    }

    // ── SCCs ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_scc_single_node() {
        let mut g = RuleDependencyGraph::new();
        g.add_node(pred("p1"));

        let sccs = g.strongly_connected_components();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 1);
    }

    #[test]
    fn test_scc_cycle() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("a"), DepEdge::Positive);

        let sccs = g.strongly_connected_components();
        // Should find exactly one SCC of size 2.
        let big: Vec<_> = sccs.iter().filter(|s| s.len() == 2).collect();
        assert_eq!(big.len(), 1);
        let scc = &big[0];
        assert!(scc.contains(&pred("a")));
        assert!(scc.contains(&pred("b")));
    }

    #[test]
    fn test_scc_dag() {
        let mut g = RuleDependencyGraph::new();
        // Pure DAG: a→b→c, no back-edges.
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("c"), DepEdge::Positive);

        let sccs = g.strongly_connected_components();
        // Every node is its own SCC.
        assert_eq!(sccs.len(), 3);
        assert!(sccs.iter().all(|s| s.len() == 1));
    }

    // ── Stratification ────────────────────────────────────────────────────────

    #[test]
    fn test_stratify_simple_dag() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("c"), DepEdge::Positive);

        let layers = g.stratify().expect("should stratify");
        // a, b, c must each be at stratum 0 because all edges are Positive
        // (stratum[v] = max(stratum[v], stratum[u]) — same stratum is fine).
        // The exact assignment: all at 0.
        let get_stratum = |name: &str| -> usize {
            layers
                .iter()
                .find(|l| l.nodes.contains(&pred(name)))
                .map(|l| l.stratum)
                .expect("node present")
        };
        // With only Positive edges the fixed-point keeps all at 0.
        assert_eq!(get_stratum("a"), 0);
        assert_eq!(get_stratum("b"), 0);
        assert_eq!(get_stratum("c"), 0);
    }

    #[test]
    fn test_stratify_with_negation() {
        let mut g = RuleDependencyGraph::new();
        // a -neg→ b: b must be at a higher stratum than a.
        g.add_edge(pred("a"), pred("b"), DepEdge::Negative);

        let layers = g.stratify().expect("should stratify");
        let stratum_a = layers
            .iter()
            .find(|l| l.nodes.contains(&pred("a")))
            .map(|l| l.stratum)
            .expect("a present");
        let stratum_b = layers
            .iter()
            .find(|l| l.nodes.contains(&pred("b")))
            .map(|l| l.stratum)
            .expect("b present");
        assert!(stratum_b > stratum_a);
    }

    #[test]
    fn test_stratify_negative_cycle_error() {
        let mut g = RuleDependencyGraph::new();
        // A -neg→ B -neg→ A  ⇒ unstratifiable.
        g.add_edge(pred("a"), pred("b"), DepEdge::Negative);
        g.add_edge(pred("b"), pred("a"), DepEdge::Negative);

        let result = g.stratify();
        assert!(
            matches!(result, Err(StratificationError::NegativeCycle { .. })),
            "expected NegativeCycle, got: {result:?}"
        );
    }

    // ── Stats ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_dep_graph_stats_basic() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);
        g.add_edge(rule("r1"), pred("p2"), DepEdge::Positive);

        let stats = DepGraphStats::compute(&g);
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_edges, 2);
        assert_eq!(stats.num_rules, 1);
        assert_eq!(stats.num_predicates, 2);
    }

    #[test]
    fn test_dep_graph_stats_has_cycles() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(pred("a"), pred("b"), DepEdge::Positive);
        g.add_edge(pred("b"), pred("a"), DepEdge::Positive);

        let stats = DepGraphStats::compute(&g);
        assert!(stats.has_cycles);
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    #[test]
    fn test_to_ascii_nonempty() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);

        let ascii = g.to_ascii();
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_to_dot_contains_digraph() {
        let mut g = RuleDependencyGraph::new();
        g.add_edge(rule("r1"), pred("p1"), DepEdge::Defines);

        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }
}
