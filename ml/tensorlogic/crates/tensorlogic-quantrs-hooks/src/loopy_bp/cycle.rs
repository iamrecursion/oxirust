//! Cycle detection and analysis for factor graphs.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::FactorGraph;

/// Summary of the cycle structure in a factor graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CycleAnalysis {
    /// True if the graph contains at least one cycle.
    pub has_cycles: bool,
    /// Approximate girth (length of shortest cycle), or `None` for trees/forests.
    pub girth: Option<usize>,
    /// Number of independent cycles (cycle rank = |E| - |V| + c).
    pub cycle_rank: usize,
    /// Whether the graph is a tree (connected and acyclic).
    pub is_tree: bool,
    /// Number of connected components.
    pub num_components: usize,
}

/// Detect cycles in a factor graph by treating it as a bipartite graph
/// (variable nodes ↔ factor nodes).
pub struct CycleDetector<'a> {
    graph: &'a FactorGraph,
}

impl<'a> CycleDetector<'a> {
    /// Create a new detector for the given factor graph.
    pub fn new(graph: &'a FactorGraph) -> Self {
        Self { graph }
    }

    /// Run cycle analysis.
    pub fn analyse(&self) -> CycleAnalysis {
        // Build the bipartite adjacency list.
        // Variable nodes are named "v:{name}", factor nodes "f:{id}".
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();

        for var_name in self.graph.variable_names() {
            let v_node = format!("v:{}", var_name);
            if let Some(factors) = self.graph.get_adjacent_factors(var_name) {
                for f_id in factors {
                    let f_node = format!("f:{}", f_id);
                    adj.entry(v_node.clone()).or_default().push(f_node.clone());
                    adj.entry(f_node).or_default().push(v_node.clone());
                }
            }
        }

        if adj.is_empty() {
            return CycleAnalysis {
                has_cycles: false,
                girth: None,
                cycle_rank: 0,
                is_tree: false,
                num_components: 0,
            };
        }

        let all_nodes: Vec<String> = adj.keys().cloned().collect();
        let num_nodes = all_nodes.len();

        // Count edges (each stored twice in adj, so divide by 2).
        let num_edges: usize = adj.values().map(|v| v.len()).sum::<usize>() / 2;

        let mut visited: HashSet<String> = HashSet::new();
        let mut num_components = 0usize;
        let mut has_cycles = false;
        let mut min_girth: Option<usize> = None;

        for start in &all_nodes {
            if visited.contains(start) {
                continue;
            }
            num_components += 1;

            // BFS from start — detect cycles by looking for cross-edges.
            // Also compute shortest cycle through each node.
            let mut depth: HashMap<String, usize> = HashMap::new();
            let mut parent: HashMap<String, Option<String>> = HashMap::new();
            let mut queue: VecDeque<String> = VecDeque::new();

            depth.insert(start.clone(), 0);
            parent.insert(start.clone(), None);
            queue.push_back(start.clone());
            visited.insert(start.clone());

            while let Some(cur) = queue.pop_front() {
                let cur_depth = depth.get(&cur).copied().unwrap_or(0);
                if let Some(neighbours) = adj.get(&cur) {
                    for nb in neighbours {
                        if !visited.contains(nb) {
                            visited.insert(nb.clone());
                            depth.insert(nb.clone(), cur_depth + 1);
                            parent.insert(nb.clone(), Some(cur.clone()));
                            queue.push_back(nb.clone());
                        } else {
                            // Cross edge — check if it's a back-edge (not to parent).
                            let is_parent = parent
                                .get(&cur)
                                .and_then(|p| p.as_ref())
                                .map(|p| p == nb)
                                .unwrap_or(false);
                            if !is_parent {
                                has_cycles = true;
                                // Cycle length ≈ depth[cur] + depth[nb] + 1
                                let cycle_len = cur_depth + depth.get(nb).copied().unwrap_or(0) + 1;
                                min_girth =
                                    Some(min_girth.map(|g| g.min(cycle_len)).unwrap_or(cycle_len));
                            }
                        }
                    }
                }
            }
        }

        // Cycle rank (circuit rank / cyclomatic number) = |E| - |V| + components.
        // Signed arithmetic is required: for trees |E| = |V| - 1 so |E| - |V| = -1,
        // giving cycle_rank = -1 + 1 = 0.
        let cycle_rank = ((num_edges as isize) - (num_nodes as isize) + (num_components as isize))
            .max(0) as usize;
        let is_tree = num_components == 1 && !has_cycles;

        CycleAnalysis {
            has_cycles,
            girth: min_girth,
            cycle_rank,
            is_tree,
            num_components,
        }
    }
}
