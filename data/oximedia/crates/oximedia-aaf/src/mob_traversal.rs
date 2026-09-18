//! AAF mob graph traversal utilities
//!
//! Provides a directed-graph model of the AAF mob reference network, with
//! DFS / BFS traversal, cycle detection, depth computation, and statistical
//! summaries.
//!
//! The graph is built from `MobNode` descriptors and explicit edge pairs
//! `(from_mob_id, to_mob_id)`.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ─── MobType ─────────────────────────────────────────────────────────────────

/// The semantic type of a mob in the AAF reference graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MobType {
    /// Composition mob — top-level editorial sequence.
    Composition,
    /// Master mob — intermediate link between composition and source.
    Master,
    /// Source mob — references essence.
    Source,
    /// File source mob — directly references a file.
    FileSrc,
}

impl MobType {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Composition => "Composition",
            Self::Master => "Master",
            Self::Source => "Source",
            Self::FileSrc => "FileSrc",
        }
    }

    /// Returns `true` for leaf (source) mob types.
    #[must_use]
    pub fn is_source(&self) -> bool {
        matches!(self, Self::Source | Self::FileSrc)
    }
}

// ─── MobNode ─────────────────────────────────────────────────────────────────

/// A node in the mob reference graph.
#[derive(Debug, Clone)]
pub struct MobNode {
    /// Unique mob identifier.
    pub mob_id: u64,
    /// Mob type.
    pub mob_type: MobType,
    /// Human-readable mob name.
    pub name: String,
    /// Slot IDs contained by this mob.
    pub slots: Vec<u64>,
}

impl MobNode {
    /// Create a new mob node.
    #[must_use]
    pub fn new(mob_id: u64, mob_type: MobType, name: impl Into<String>) -> Self {
        Self {
            mob_id,
            mob_type,
            name: name.into(),
            slots: Vec::new(),
        }
    }

    /// Add a slot ID to this node.
    pub fn add_slot(&mut self, slot_id: u64) {
        self.slots.push(slot_id);
    }
}

// ─── MobGraph ────────────────────────────────────────────────────────────────

/// Directed graph of mob references.
///
/// - `mobs`: all nodes, indexed by `mob_id`
/// - `edges`: directed edges `from → Vec<to>`
#[derive(Debug)]
pub struct MobGraph {
    /// All mob nodes indexed by `mob_id`.
    pub mobs: HashMap<u64, MobNode>,
    /// Directed edges: `mob_id → [referenced_mob_ids]`.
    pub edges: HashMap<u64, Vec<u64>>,
}

impl MobGraph {
    /// Build a `MobGraph` from a slice of nodes and a slice of directed edges.
    ///
    /// Edges are `(from_mob_id, to_mob_id)` pairs.
    ///
    /// The mob index (`mobs` HashMap) is built in parallel using Rayon, which
    /// speeds up large compositions with thousands of mobs.  Edge construction
    /// remains sequential because `HashMap::entry` is not thread-safe and the
    /// number of edges is typically small relative to the node count.
    #[must_use]
    pub fn build(mobs: &[MobNode], references: &[(u64, u64)]) -> Self {
        // Parallel collect: each thread processes its partition and Rayon
        // merges the per-thread HashMaps into one via the `FromParallelIterator`
        // implementation on `HashMap`.
        let mob_map: HashMap<u64, MobNode> =
            mobs.par_iter().map(|n| (n.mob_id, n.clone())).collect();

        let mut edge_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (from, to) in references {
            edge_map.entry(*from).or_default().push(*to);
        }

        Self {
            mobs: mob_map,
            edges: edge_map,
        }
    }

    /// Iterator over direct successors of `mob_id`.
    #[must_use]
    pub fn successors(&self, mob_id: u64) -> &[u64] {
        self.edges.get(&mob_id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.mobs.len()
    }
}

// ─── Traversal ────────────────────────────────────────────────────────────────

/// Depth-first search traversal starting from `start_id`.
///
/// Returns mob IDs in DFS visit order (pre-order).  Each node is visited at
/// most once, so cycles do not cause infinite loops.
#[must_use]
pub fn traverse_dfs(graph: &MobGraph, start_id: u64) -> Vec<u64> {
    let mut visited: HashSet<u64> = HashSet::new();
    let mut order: Vec<u64> = Vec::new();
    dfs_recursive(graph, start_id, &mut visited, &mut order);
    order
}

fn dfs_recursive(graph: &MobGraph, node_id: u64, visited: &mut HashSet<u64>, order: &mut Vec<u64>) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);
    if graph.mobs.contains_key(&node_id) {
        order.push(node_id);
    }
    for &next in graph.successors(node_id) {
        dfs_recursive(graph, next, visited, order);
    }
}

/// Follow the reference chain from a composition mob and return all reachable
/// leaf (source) mob IDs.
///
/// A mob is considered a source leaf if its `mob_type` is `Source` or `FileSrc`.
#[must_use]
pub fn find_source_clips(graph: &MobGraph, composition_id: u64) -> Vec<u64> {
    let reachable = traverse_dfs(graph, composition_id);
    reachable
        .into_iter()
        .filter(|&id| {
            graph
                .mobs
                .get(&id)
                .map(|n| n.mob_type.is_source())
                .unwrap_or(false)
        })
        .collect()
}

/// Compute the transitive closure of all mobs reachable from `mob_id` via BFS.
///
/// Returns mob IDs in BFS order (level-by-level), excluding the start node.
#[must_use]
pub fn find_all_references(graph: &MobGraph, mob_id: u64) -> Vec<u64> {
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue = VecDeque::new();
    let mut order: Vec<u64> = Vec::new();

    visited.insert(mob_id);
    queue.push_back(mob_id);

    while let Some(current) = queue.pop_front() {
        for &next in graph.successors(current) {
            if visited.insert(next) {
                order.push(next);
                queue.push_back(next);
            }
        }
    }

    order
}

// ─── Cycle detection ─────────────────────────────────────────────────────────

/// Colour used during DFS cycle detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    /// Node not yet visited.
    White,
    /// Node is on the current DFS stack (in-progress).
    Gray,
    /// Node fully processed.
    Black,
}

/// Detect all simple cycles in the graph using DFS with gray/white/black colouring.
///
/// Returns a list of cycles; each cycle is a `Vec<u64>` of mob IDs forming
/// a closed path (the first element equals the last for readability, but the
/// implementation returns the cycle path without repeating the start node).
#[must_use]
pub fn detect_cycles(graph: &MobGraph) -> Vec<Vec<u64>> {
    let mut color: HashMap<u64, Color> = HashMap::new();
    let mut cycles: Vec<Vec<u64>> = Vec::new();
    let mut stack: Vec<u64> = Vec::new();

    for &start in graph.mobs.keys() {
        if *color.get(&start).unwrap_or(&Color::White) == Color::White {
            dfs_color(graph, start, &mut color, &mut stack, &mut cycles);
        }
    }

    cycles
}

fn dfs_color(
    graph: &MobGraph,
    node: u64,
    color: &mut HashMap<u64, Color>,
    stack: &mut Vec<u64>,
    cycles: &mut Vec<Vec<u64>>,
) {
    color.insert(node, Color::Gray);
    stack.push(node);

    for &next in graph.successors(node) {
        match color.get(&next).copied().unwrap_or(Color::White) {
            Color::White => {
                dfs_color(graph, next, color, stack, cycles);
            }
            Color::Gray => {
                // Back-edge found — extract the cycle
                if let Some(pos) = stack.iter().position(|&id| id == next) {
                    let cycle: Vec<u64> = stack[pos..].to_vec();
                    cycles.push(cycle);
                }
            }
            Color::Black => {}
        }
    }

    stack.pop();
    color.insert(node, Color::Black);
}

// ─── Depth computation ────────────────────────────────────────────────────────

/// Compute BFS-level depth of every reachable node from `root_id`.
///
/// The root node has depth 0.  Returns a map of `mob_id → depth`.
/// Unreachable nodes are not included.
#[must_use]
pub fn compute_mob_depth(graph: &MobGraph, root_id: u64) -> HashMap<u64, usize> {
    let mut depths: HashMap<u64, usize> = HashMap::new();
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();

    depths.insert(root_id, 0);
    queue.push_back((root_id, 0));

    while let Some((current, depth)) = queue.pop_front() {
        for &next in graph.successors(current) {
            if let std::collections::hash_map::Entry::Vacant(e) = depths.entry(next) {
                e.insert(depth + 1);
                queue.push_back((next, depth + 1));
            }
        }
    }

    depths
}

// ─── Statistics ───────────────────────────────────────────────────────────────

// ─── Topological Sort ─────────────────────────────────────────────────────────

/// Compute a topological ordering of the mob graph.
///
/// Returns `None` if the graph contains a cycle (topological sort is undefined
/// for cyclic graphs).
#[must_use]
pub fn topological_sort(graph: &MobGraph) -> Option<Vec<u64>> {
    let mut in_degree: HashMap<u64, usize> = HashMap::new();
    for &id in graph.mobs.keys() {
        in_degree.entry(id).or_insert(0);
    }
    for successors in graph.edges.values() {
        for &succ in successors {
            *in_degree.entry(succ).or_insert(0) += 1;
        }
    }

    let mut queue: VecDeque<u64> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut order = Vec::with_capacity(graph.mobs.len());

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &succ in graph.successors(node) {
            if let Some(deg) = in_degree.get_mut(&succ) {
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    queue.push_back(succ);
                }
            }
        }
    }

    if order.len() == graph.mobs.len() {
        Some(order)
    } else {
        None // cycle detected
    }
}

// ─── Reverse Edges ────────────────────────────────────────────────────────────

/// Compute the reverse edge map: for each mob, which mobs reference it.
#[must_use]
pub fn reverse_edges(graph: &MobGraph) -> HashMap<u64, Vec<u64>> {
    let mut rev: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&from, tos) in &graph.edges {
        for &to in tos {
            rev.entry(to).or_default().push(from);
        }
    }
    rev
}

/// Find all mobs that have no incoming edges (root mobs).
#[must_use]
pub fn find_roots(graph: &MobGraph) -> Vec<u64> {
    let rev = reverse_edges(graph);
    graph
        .mobs
        .keys()
        .filter(|id| {
            rev.get(id)
                .map(|parents| parents.is_empty())
                .unwrap_or(true)
        })
        .copied()
        .collect()
}

/// Find all mobs that have no outgoing edges (leaf mobs).
#[must_use]
pub fn find_leaves(graph: &MobGraph) -> Vec<u64> {
    graph
        .mobs
        .keys()
        .filter(|&id| graph.successors(*id).is_empty())
        .copied()
        .collect()
}

/// Count total edges in the graph.
#[must_use]
pub fn edge_count(graph: &MobGraph) -> usize {
    graph.edges.values().map(|v| v.len()).sum()
}

// ─── Path Finding ─────────────────────────────────────────────────────────────

/// Find a path from `start_id` to `end_id` via DFS.
///
/// Returns `None` if no path exists.
#[must_use]
pub fn find_path(graph: &MobGraph, start_id: u64, end_id: u64) -> Option<Vec<u64>> {
    if start_id == end_id {
        return Some(vec![start_id]);
    }

    let mut visited: HashSet<u64> = HashSet::new();
    let mut path: Vec<u64> = Vec::new();

    if dfs_find_path(graph, start_id, end_id, &mut visited, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn dfs_find_path(
    graph: &MobGraph,
    current: u64,
    target: u64,
    visited: &mut HashSet<u64>,
    path: &mut Vec<u64>,
) -> bool {
    if !visited.insert(current) {
        return false;
    }
    path.push(current);

    if current == target {
        return true;
    }

    for &next in graph.successors(current) {
        if dfs_find_path(graph, next, target, visited, path) {
            return true;
        }
    }

    path.pop();
    false
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Aggregate statistics about a mob graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobStatistics {
    /// Total number of mobs in the graph.
    pub total_mobs: usize,
    /// Number of leaf source / file-source mobs.
    pub source_clip_count: usize,
    /// Maximum DFS depth from any composition mob (0 for empty graphs).
    pub max_depth: usize,
    /// Whether the graph contains at least one cycle.
    pub has_cycles: bool,
    /// Number of composition mobs.
    pub composition_count: usize,
    /// Number of master mobs.
    pub master_count: usize,
    /// Total number of directed edges.
    pub total_edges: usize,
    /// Number of root nodes (no incoming edges).
    pub root_count: usize,
    /// Number of leaf nodes (no outgoing edges).
    pub leaf_count: usize,
}

/// Compute aggregate statistics for the entire mob graph.
#[must_use]
pub fn compute_statistics(graph: &MobGraph) -> MobStatistics {
    let total_mobs = graph.mobs.len();

    let source_clip_count = graph
        .mobs
        .values()
        .filter(|n| n.mob_type.is_source())
        .count();

    // Max depth: take the maximum over all composition mob roots
    let max_depth = graph
        .mobs
        .values()
        .filter(|n| n.mob_type == MobType::Composition)
        .map(|n| {
            let depths = compute_mob_depth(graph, n.mob_id);
            depths.values().copied().max().unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    let has_cycles = !detect_cycles(graph).is_empty();

    let composition_count = graph
        .mobs
        .values()
        .filter(|n| n.mob_type == MobType::Composition)
        .count();

    let master_count = graph
        .mobs
        .values()
        .filter(|n| n.mob_type == MobType::Master)
        .count();

    let total_edges = edge_count(graph);
    let root_count = find_roots(graph).len();
    let leaf_count = find_leaves(graph).len();

    MobStatistics {
        total_mobs,
        source_clip_count,
        max_depth,
        has_cycles,
        composition_count,
        master_count,
        total_edges,
        root_count,
        leaf_count,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ────────────────────────────────────────────────────

    fn comp(id: u64, name: &str) -> MobNode {
        MobNode::new(id, MobType::Composition, name)
    }
    fn master(id: u64, name: &str) -> MobNode {
        MobNode::new(id, MobType::Master, name)
    }
    fn source(id: u64, name: &str) -> MobNode {
        MobNode::new(id, MobType::Source, name)
    }
    fn filesrc(id: u64, name: &str) -> MobNode {
        MobNode::new(id, MobType::FileSrc, name)
    }

    /// Simple 3-level chain: Composition → Master → Source
    fn three_level_graph() -> MobGraph {
        let nodes = vec![comp(1, "C1"), master(2, "M1"), source(3, "S1")];
        let edges = vec![(1, 2), (2, 3)];
        MobGraph::build(&nodes, &edges)
    }

    // ── MobNode tests ─────────────────────────────────────────────────────

    #[test]
    fn test_mob_node_creation() {
        let node = MobNode::new(42, MobType::Composition, "MyComp");
        assert_eq!(node.mob_id, 42);
        assert_eq!(node.mob_type, MobType::Composition);
        assert_eq!(node.name, "MyComp");
        assert!(node.slots.is_empty());
    }

    #[test]
    fn test_mob_node_add_slot() {
        let mut node = MobNode::new(1, MobType::Master, "M");
        node.add_slot(10);
        node.add_slot(20);
        assert_eq!(node.slots, vec![10, 20]);
    }

    #[test]
    fn test_mob_type_label() {
        assert_eq!(MobType::Composition.label(), "Composition");
        assert_eq!(MobType::FileSrc.label(), "FileSrc");
    }

    #[test]
    fn test_mob_type_is_source() {
        assert!(MobType::Source.is_source());
        assert!(MobType::FileSrc.is_source());
        assert!(!MobType::Master.is_source());
        assert!(!MobType::Composition.is_source());
    }

    // ── MobGraph tests ────────────────────────────────────────────────────

    #[test]
    fn test_graph_build_node_count() {
        let g = three_level_graph();
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_graph_successors() {
        let g = three_level_graph();
        assert_eq!(g.successors(1), &[2]);
        assert_eq!(g.successors(2), &[3]);
        assert!(g.successors(3).is_empty());
    }

    #[test]
    fn test_graph_unknown_node_has_no_successors() {
        let g = three_level_graph();
        assert!(g.successors(999).is_empty());
    }

    // ── DFS traversal ─────────────────────────────────────────────────────

    #[test]
    fn test_dfs_linear_chain() {
        let g = three_level_graph();
        let order = traverse_dfs(&g, 1);
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn test_dfs_from_leaf() {
        let g = three_level_graph();
        let order = traverse_dfs(&g, 3);
        assert_eq!(order, vec![3]);
    }

    #[test]
    fn test_dfs_unknown_start() {
        let g = three_level_graph();
        let order = traverse_dfs(&g, 999);
        assert!(order.is_empty());
    }

    #[test]
    fn test_dfs_branching_graph() {
        let nodes = vec![
            comp(1, "C"),
            master(2, "M"),
            source(3, "S1"),
            source(4, "S2"),
        ];
        let edges = vec![(1, 2), (2, 3), (2, 4)];
        let g = MobGraph::build(&nodes, &edges);
        let order = traverse_dfs(&g, 1);
        assert!(order.contains(&1));
        assert!(order.contains(&3));
        assert!(order.contains(&4));
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_dfs_no_revisit_with_shared_target() {
        let nodes = vec![comp(1, "C1"), comp(2, "C2"), master(3, "M")];
        let edges = vec![(1, 3), (2, 3)];
        let g = MobGraph::build(&nodes, &edges);
        let order = traverse_dfs(&g, 1);
        // Node 3 should appear exactly once
        assert_eq!(order.iter().filter(|&&id| id == 3).count(), 1);
    }

    // ── find_source_clips ─────────────────────────────────────────────────

    #[test]
    fn test_find_source_clips_basic() {
        let g = three_level_graph();
        let sources = find_source_clips(&g, 1);
        assert_eq!(sources, vec![3]);
    }

    #[test]
    fn test_find_source_clips_multiple() {
        let nodes = vec![
            comp(1, "C"),
            master(2, "M1"),
            master(3, "M2"),
            filesrc(4, "F1"),
            source(5, "S1"),
        ];
        let edges = vec![(1, 2), (1, 3), (2, 4), (3, 5)];
        let g = MobGraph::build(&nodes, &edges);
        let mut sources = find_source_clips(&g, 1);
        sources.sort_unstable();
        assert_eq!(sources, vec![4, 5]);
    }

    #[test]
    fn test_find_source_clips_none_for_master_root() {
        let nodes = vec![master(1, "M"), master(2, "M2")];
        let edges = vec![(1, 2)];
        let g = MobGraph::build(&nodes, &edges);
        let sources = find_source_clips(&g, 1);
        assert!(sources.is_empty());
    }

    // ── find_all_references ───────────────────────────────────────────────

    #[test]
    fn test_find_all_references_linear() {
        let g = three_level_graph();
        let refs = find_all_references(&g, 1);
        assert_eq!(refs, vec![2, 3]);
    }

    #[test]
    fn test_find_all_references_no_children() {
        let g = three_level_graph();
        let refs = find_all_references(&g, 3);
        assert!(refs.is_empty());
    }

    #[test]
    fn test_find_all_references_excludes_start() {
        let g = three_level_graph();
        let refs = find_all_references(&g, 1);
        assert!(!refs.contains(&1));
    }

    // ── Cycle detection ───────────────────────────────────────────────────

    #[test]
    fn test_no_cycles_in_acyclic_graph() {
        let g = three_level_graph();
        let cycles = detect_cycles(&g);
        assert!(cycles.is_empty(), "Expected no cycles, got: {cycles:?}");
    }

    #[test]
    fn test_detects_simple_cycle() {
        let nodes = vec![comp(1, "A"), master(2, "B")];
        let edges = vec![(1, 2), (2, 1)];
        let g = MobGraph::build(&nodes, &edges);
        let cycles = detect_cycles(&g);
        assert!(!cycles.is_empty(), "Should detect cycle A→B→A");
    }

    #[test]
    fn test_detects_self_loop() {
        let nodes = vec![comp(1, "Self")];
        let edges = vec![(1, 1)];
        let g = MobGraph::build(&nodes, &edges);
        let cycles = detect_cycles(&g);
        assert!(!cycles.is_empty(), "Self-loop should be a cycle");
    }

    #[test]
    fn test_no_cycle_detected_for_shared_child() {
        // Diamond: C→M1, C→M2, M1→S, M2→S (no cycle)
        let nodes = vec![
            comp(1, "C"),
            master(2, "M1"),
            master(3, "M2"),
            source(4, "S"),
        ];
        let edges = vec![(1, 2), (1, 3), (2, 4), (3, 4)];
        let g = MobGraph::build(&nodes, &edges);
        let cycles = detect_cycles(&g);
        assert!(cycles.is_empty(), "Diamond graph has no cycles");
    }

    // ── compute_mob_depth ─────────────────────────────────────────────────

    #[test]
    fn test_depth_linear_chain() {
        let g = three_level_graph();
        let depths = compute_mob_depth(&g, 1);
        assert_eq!(depths[&1], 0);
        assert_eq!(depths[&2], 1);
        assert_eq!(depths[&3], 2);
    }

    #[test]
    fn test_depth_root_only() {
        let g = three_level_graph();
        let depths = compute_mob_depth(&g, 3);
        assert_eq!(depths[&3], 0);
        assert_eq!(depths.len(), 1);
    }

    #[test]
    fn test_depth_branching() {
        let nodes = vec![
            comp(1, "C"),
            master(2, "M"),
            source(3, "S1"),
            source(4, "S2"),
        ];
        let edges = vec![(1, 2), (2, 3), (2, 4)];
        let g = MobGraph::build(&nodes, &edges);
        let depths = compute_mob_depth(&g, 1);
        assert_eq!(depths[&1], 0);
        assert_eq!(depths[&2], 1);
        assert_eq!(depths[&3], 2);
        assert_eq!(depths[&4], 2);
    }

    #[test]
    fn test_depth_unreachable_nodes_excluded() {
        let nodes = vec![comp(1, "C"), master(2, "M"), source(99, "Isolated")];
        let edges = vec![(1, 2)];
        let g = MobGraph::build(&nodes, &edges);
        let depths = compute_mob_depth(&g, 1);
        assert!(
            !depths.contains_key(&99),
            "Unreachable node should not appear"
        );
    }

    // ── compute_statistics ────────────────────────────────────────────────

    #[test]
    fn test_statistics_basic() {
        let g = three_level_graph();
        let stats = compute_statistics(&g);
        assert_eq!(stats.total_mobs, 3);
        assert_eq!(stats.source_clip_count, 1);
        assert_eq!(stats.max_depth, 2);
        assert!(!stats.has_cycles);
    }

    #[test]
    fn test_statistics_empty_graph() {
        let g = MobGraph::build(&[], &[]);
        let stats = compute_statistics(&g);
        assert_eq!(stats.total_mobs, 0);
        assert_eq!(stats.source_clip_count, 0);
        assert_eq!(stats.max_depth, 0);
        assert!(!stats.has_cycles);
    }

    #[test]
    fn test_statistics_with_cycle() {
        let nodes = vec![comp(1, "A"), master(2, "B")];
        let edges = vec![(1, 2), (2, 1)];
        let g = MobGraph::build(&nodes, &edges);
        let stats = compute_statistics(&g);
        assert!(stats.has_cycles);
    }

    #[test]
    fn test_statistics_multiple_sources() {
        let nodes = vec![
            comp(1, "C"),
            master(2, "M"),
            source(3, "S1"),
            filesrc(4, "F1"),
            filesrc(5, "F2"),
        ];
        let edges = vec![(1, 2), (2, 3), (2, 4), (2, 5)];
        let g = MobGraph::build(&nodes, &edges);
        let stats = compute_statistics(&g);
        assert_eq!(stats.source_clip_count, 3);
        assert_eq!(stats.max_depth, 2);
        assert!(!stats.has_cycles);
        assert_eq!(stats.composition_count, 1);
        assert_eq!(stats.master_count, 1);
        assert_eq!(stats.total_edges, 4);
    }

    // ── Topological sort ────────────────────────────────────────────────

    #[test]
    fn test_topological_sort_linear() {
        let g = three_level_graph();
        let order = topological_sort(&g);
        assert!(order.is_some());
        let order = order.expect("topo sort");
        // Node 1 must appear before 2, and 2 before 3
        let pos1 = order.iter().position(|&x| x == 1).expect("1");
        let pos2 = order.iter().position(|&x| x == 2).expect("2");
        let pos3 = order.iter().position(|&x| x == 3).expect("3");
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_topological_sort_with_cycle() {
        let nodes = vec![comp(1, "A"), master(2, "B")];
        let edges = vec![(1, 2), (2, 1)];
        let g = MobGraph::build(&nodes, &edges);
        assert!(topological_sort(&g).is_none());
    }

    #[test]
    fn test_topological_sort_empty() {
        let g = MobGraph::build(&[], &[]);
        let order = topological_sort(&g);
        assert!(order.is_some());
        assert!(order.expect("empty").is_empty());
    }

    // ── Reverse edges / roots / leaves ──────────────────────────────────

    #[test]
    fn test_reverse_edges() {
        let g = three_level_graph();
        let rev = reverse_edges(&g);
        assert!(rev.get(&2).map(|v| v.contains(&1)).unwrap_or(false));
        assert!(rev.get(&3).map(|v| v.contains(&2)).unwrap_or(false));
    }

    #[test]
    fn test_find_roots() {
        let g = three_level_graph();
        let roots = find_roots(&g);
        assert_eq!(roots, vec![1]);
    }

    #[test]
    fn test_find_leaves() {
        let g = three_level_graph();
        let mut leaves = find_leaves(&g);
        leaves.sort_unstable();
        assert_eq!(leaves, vec![3]);
    }

    #[test]
    fn test_find_roots_multiple() {
        let nodes = vec![comp(1, "C1"), comp(2, "C2"), source(3, "S")];
        let edges = vec![(1, 3), (2, 3)];
        let g = MobGraph::build(&nodes, &edges);
        let mut roots = find_roots(&g);
        roots.sort_unstable();
        assert_eq!(roots, vec![1, 2]);
    }

    #[test]
    fn test_find_leaves_multiple() {
        let nodes = vec![comp(1, "C"), source(2, "S1"), source(3, "S2")];
        let edges = vec![(1, 2), (1, 3)];
        let g = MobGraph::build(&nodes, &edges);
        let mut leaves = find_leaves(&g);
        leaves.sort_unstable();
        assert_eq!(leaves, vec![2, 3]);
    }

    #[test]
    fn test_edge_count() {
        let g = three_level_graph();
        assert_eq!(edge_count(&g), 2);
    }

    // ── Path finding ────────────────────────────────────────────────────

    #[test]
    fn test_find_path_exists() {
        let g = three_level_graph();
        let path = find_path(&g, 1, 3);
        assert!(path.is_some());
        let path = path.expect("path");
        assert_eq!(path.first(), Some(&1));
        assert_eq!(path.last(), Some(&3));
    }

    #[test]
    fn test_find_path_self() {
        let g = three_level_graph();
        let path = find_path(&g, 2, 2);
        assert_eq!(path, Some(vec![2]));
    }

    #[test]
    fn test_find_path_no_path() {
        let g = three_level_graph();
        let path = find_path(&g, 3, 1); // reverse direction
        assert!(path.is_none());
    }

    #[test]
    fn test_find_path_branching() {
        let nodes = vec![
            comp(1, "C"),
            master(2, "M1"),
            master(3, "M2"),
            source(4, "S"),
        ];
        let edges = vec![(1, 2), (1, 3), (3, 4)];
        let g = MobGraph::build(&nodes, &edges);
        let path = find_path(&g, 1, 4);
        assert!(path.is_some());
        let path = path.expect("path to 4");
        assert_eq!(path.first(), Some(&1));
        assert_eq!(path.last(), Some(&4));
    }

    // ── Parallel build ──────────────────────────────────────────────────

    /// Verify that the parallel `MobGraph::build` produces the same node map
    /// and edge map as a reference sequential build (same inputs, both paths).
    #[test]
    fn test_mob_graph_parallel_build_matches_sequential() {
        // Build a moderately large graph so rayon uses multiple threads.
        let node_count = 500u64;
        let nodes: Vec<MobNode> = (0..node_count)
            .map(|i| {
                let mob_type = match i % 4 {
                    0 => MobType::Composition,
                    1 => MobType::Master,
                    2 => MobType::Source,
                    _ => MobType::FileSrc,
                };
                MobNode::new(i, mob_type, format!("mob_{i}"))
            })
            .collect();

        // Create a chain of edges: 0→1, 1→2, …, 498→499  plus some skips.
        let mut edges: Vec<(u64, u64)> = (0..node_count - 1).map(|i| (i, i + 1)).collect();
        // Add some fan-out edges for variety.
        edges.push((0, 50));
        edges.push((0, 100));
        edges.push((100, 200));
        edges.push((200, 400));

        // Build via the (now parallel) MobGraph::build.
        let parallel_graph = MobGraph::build(&nodes, &edges);

        // Build a reference sequential graph.
        let seq_mob_map: HashMap<u64, MobNode> =
            nodes.iter().map(|n| (n.mob_id, n.clone())).collect();
        let mut seq_edge_map: HashMap<u64, Vec<u64>> = HashMap::new();
        for (from, to) in &edges {
            seq_edge_map.entry(*from).or_default().push(*to);
        }

        // Node maps must contain identical keys and mob names.
        assert_eq!(
            parallel_graph.mobs.len(),
            seq_mob_map.len(),
            "node count mismatch"
        );
        for (id, node) in &seq_mob_map {
            let par_node = parallel_graph
                .mobs
                .get(id)
                .unwrap_or_else(|| panic!("mob_id {id} missing from parallel graph"));
            assert_eq!(par_node.name, node.name, "name mismatch for mob {id}");
            assert_eq!(
                par_node.mob_type, node.mob_type,
                "type mismatch for mob {id}"
            );
        }

        // Edge maps must be identical (same keys, same destination sets —
        // order within a destination Vec may differ due to parallelism).
        assert_eq!(
            parallel_graph.edges.len(),
            seq_edge_map.len(),
            "edge-map key count mismatch"
        );
        for (from, seq_tos) in &seq_edge_map {
            let par_tos = parallel_graph
                .edges
                .get(from)
                .unwrap_or_else(|| panic!("from={from} missing from parallel edge map"));
            let mut seq_sorted = seq_tos.clone();
            let mut par_sorted = par_tos.clone();
            seq_sorted.sort_unstable();
            par_sorted.sort_unstable();
            assert_eq!(
                seq_sorted, par_sorted,
                "edge destinations differ for from={from}"
            );
        }
    }

    // ── Statistics extended ─────────────────────────────────────────────

    #[test]
    fn test_statistics_root_and_leaf_counts() {
        let g = three_level_graph();
        let stats = compute_statistics(&g);
        assert_eq!(stats.root_count, 1);
        assert_eq!(stats.leaf_count, 1);
    }

    #[test]
    fn test_statistics_composition_and_master_counts() {
        let nodes = vec![
            comp(1, "C1"),
            comp(2, "C2"),
            master(3, "M1"),
            master(4, "M2"),
            source(5, "S"),
        ];
        let edges = vec![(1, 3), (2, 4), (3, 5), (4, 5)];
        let g = MobGraph::build(&nodes, &edges);
        let stats = compute_statistics(&g);
        assert_eq!(stats.composition_count, 2);
        assert_eq!(stats.master_count, 2);
        assert_eq!(stats.total_edges, 4);
        assert_eq!(stats.root_count, 2);
        assert_eq!(stats.leaf_count, 1);
    }
}
