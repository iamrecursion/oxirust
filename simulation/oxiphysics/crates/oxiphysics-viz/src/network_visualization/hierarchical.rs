// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Layer assignment and crossing minimisation of the hierarchical (Sugiyama)
//! layout.
//!
//! # Layering
//!
//! Every edge points from its `source` to its `target`, whether or not it is
//! marked `directed`.  Edges that close a cycle are reversed for the layering
//! (the back edges of a depth-first search, which leaves an acyclic graph), and
//! every node then goes to the layer one below its lowest-placed predecessor
//! (longest-path layering), so all non-reversed edges point downwards.  Both
//! steps run in `O(V + E)` and the depth-first search is iterative, so a long
//! chain cannot exhaust the call stack.
//!
//! # Crossing minimisation
//!
//! The nodes of every layer are seeded in ascending id order.  Each pass then
//! sweeps down the layers, ordering every layer by the mean position of its
//! neighbours in the layer above (the barycentre heuristic), and back up,
//! ordering by the neighbours in the layer below.  Neighbours are neighbours in
//! the *adjacent* layer, whatever the direction of the edge between them, so
//! directed and undirected graphs are treated alike.  A node without such a
//! neighbour keeps its current slot.  The crossings of the whole drawing are
//! counted after every pass and the best ordering seen is returned.
//!
//! Edges that span more than one layer take no part in the sweeps and in the
//! crossing count (no virtual nodes are inserted for them).
//!
//! Nothing depends on the iteration order of a hash map: the same graph always
//! gives the same layers and the same order.

use std::collections::{HashMap, VecDeque};

use super::NetworkGraph;

/// Distinct node ids in order of first occurrence and their dense indices.
fn index_nodes(graph: &NetworkGraph) -> (Vec<usize>, HashMap<usize, usize>) {
    let mut ids: Vec<usize> = Vec::with_capacity(graph.nodes.len());
    let mut index: HashMap<usize, usize> = HashMap::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        index.entry(node.id).or_insert_with(|| {
            ids.push(node.id);
            ids.len() - 1
        });
    }
    (ids, index)
}

/// Longest-path layering: every node gets the length of the longest path that
/// ends at it, after the edges that close cycles have been reversed.
///
/// Edges with an end point that is not a node, and self-loops, are ignored.
pub(super) fn assign_layers(graph: &NetworkGraph) -> HashMap<usize, usize> {
    let (ids, index) = index_nodes(graph);
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for edge in &graph.edges {
        if let (Some(&source), Some(&target)) = (index.get(&edge.source), index.get(&edge.target))
            && source != target
        {
            successors[source].push(target);
        }
    }
    let acyclic = break_cycles(&successors);
    let layers = longest_path_layers(&acyclic);
    ids.into_iter().zip(layers).collect()
}

/// State of a node during the depth-first search of [`break_cycles`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Visit {
    Unvisited,
    OnPath,
    Done,
}

/// Successor lists of an acyclic graph obtained by reversing the back edges of
/// a depth-first search (visiting nodes and edges in index order).
///
/// Every kept edge ends at a node that finishes before its source does, and a
/// reversed back edge ends at a node that finished before the ancestor it now
/// starts from; ordering the nodes by decreasing finishing time is therefore a
/// topological order of the result.
fn break_cycles(successors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut visit = vec![Visit::Unvisited; successors.len()];
    let mut acyclic: Vec<Vec<usize>> = vec![Vec::new(); successors.len()];
    // The current path: a node and the position of its next edge to examine.
    let mut path: Vec<(usize, usize)> = Vec::new();
    for root in 0..successors.len() {
        if visit[root] != Visit::Unvisited {
            continue;
        }
        visit[root] = Visit::OnPath;
        path.push((root, 0));
        while let Some(&(node, next)) = path.last() {
            let Some(&target) = successors[node].get(next) else {
                visit[node] = Visit::Done;
                path.pop();
                continue;
            };
            if let Some(frame) = path.last_mut() {
                frame.1 += 1;
            }
            match visit[target] {
                Visit::Unvisited => {
                    acyclic[node].push(target);
                    visit[target] = Visit::OnPath;
                    path.push((target, 0));
                }
                Visit::OnPath => acyclic[target].push(node),
                Visit::Done => acyclic[node].push(target),
            }
        }
    }
    acyclic
}

/// Layer of every node of an acyclic graph: the longest path that ends at it
/// (Kahn's algorithm).
fn longest_path_layers(successors: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0_usize; successors.len()];
    for targets in successors {
        for &target in targets {
            in_degree[target] += 1;
        }
    }
    let mut layer = vec![0_usize; successors.len()];
    let mut ready: VecDeque<usize> = (0..successors.len())
        .filter(|&node| in_degree[node] == 0)
        .collect();
    while let Some(node) = ready.pop_front() {
        for &target in &successors[node] {
            layer[target] = layer[target].max(layer[node] + 1);
            in_degree[target] -= 1;
            if in_degree[target] == 0 {
                ready.push_back(target);
            }
        }
    }
    layer
}

/// Order the nodes of every layer to reduce edge crossings (see the module
/// documentation).  Returns one list of node ids per layer, indexed by layer;
/// layers without nodes are empty lists.
///
/// `layers` maps node ids to layers; edges with an end point that is not in
/// `layers` are ignored.
pub(super) fn minimize_crossings(
    graph: &NetworkGraph,
    layers: &HashMap<usize, usize>,
    passes: usize,
) -> Vec<Vec<usize>> {
    let mut entries: Vec<(usize, usize)> = layers.iter().map(|(&id, &layer)| (id, layer)).collect();
    entries.sort_unstable();
    let ids: Vec<usize> = entries.iter().map(|&(id, _)| id).collect();
    let level: Vec<usize> = entries.iter().map(|&(_, layer)| layer).collect();
    let Some(layer_count) = level
        .iter()
        .copied()
        .max()
        .and_then(|deepest| deepest.checked_add(1))
    else {
        return Vec::new();
    };
    let index: HashMap<usize, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Seed: ascending ids (the nodes are indexed in ascending id order).
    let mut order: Vec<Vec<usize>> = vec![Vec::new(); layer_count];
    for (node, &layer) in level.iter().enumerate() {
        order[layer].push(node);
    }
    let mut above: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    let mut below: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for edge in &graph.edges {
        let (Some(&a), Some(&b)) = (index.get(&edge.source), index.get(&edge.target)) else {
            continue;
        };
        let (upper, lower) = if level[a].checked_add(1) == Some(level[b]) {
            (a, b)
        } else if level[b].checked_add(1) == Some(level[a]) {
            (b, a)
        } else {
            continue;
        };
        below[upper].push(lower);
        above[lower].push(upper);
    }

    let mut position = vec![0_usize; ids.len()];
    refresh_positions(&order, &mut position);
    let mut best = order.clone();
    let mut best_crossings = count_crossings(&order, &below, &position);
    for _ in 0..passes {
        if best_crossings == 0 {
            break;
        }
        let before = order.clone();
        // Down: every layer follows the layer above it.
        for layer in order.iter_mut().skip(1) {
            reorder(layer, &above, &mut position);
        }
        // Up: every layer but the last follows the layer below it.
        for layer in order.iter_mut().rev().skip(1) {
            reorder(layer, &below, &mut position);
        }
        let crossings = count_crossings(&order, &below, &position);
        if crossings < best_crossings {
            best_crossings = crossings;
            best = order.clone();
        }
        // Another pass would repeat this one.
        if order == before {
            break;
        }
    }
    best.into_iter()
        .map(|layer| layer.into_iter().map(|node| ids[node]).collect())
        .collect()
}

/// Number of edge crossings of the drawing that places `ordered_layers` one
/// below the other, counting only edges between adjacent layers.
///
/// Parallel edges count individually; edges that share an end point never
/// cross.
pub(super) fn crossing_count(graph: &NetworkGraph, ordered_layers: &[Vec<usize>]) -> usize {
    let mut node_of: HashMap<usize, usize> = HashMap::new();
    let mut level: Vec<usize> = Vec::new();
    let mut order: Vec<Vec<usize>> = Vec::with_capacity(ordered_layers.len());
    for (layer, ids) in ordered_layers.iter().enumerate() {
        let mut nodes = Vec::with_capacity(ids.len());
        for &id in ids {
            node_of.entry(id).or_insert_with(|| {
                level.push(layer);
                level.len() - 1
            });
            if let Some(&node) = node_of.get(&id) {
                nodes.push(node);
            }
        }
        order.push(nodes);
    }
    let mut below: Vec<Vec<usize>> = vec![Vec::new(); level.len()];
    for edge in &graph.edges {
        if let (Some(&a), Some(&b)) = (node_of.get(&edge.source), node_of.get(&edge.target)) {
            if level[a].checked_add(1) == Some(level[b]) {
                below[a].push(b);
            } else if level[b].checked_add(1) == Some(level[a]) {
                below[b].push(a);
            }
        }
    }
    let mut position = vec![0_usize; level.len()];
    refresh_positions(&order, &mut position);
    count_crossings(&order, &below, &position)
}

/// Record the slot of every node within its layer.
fn refresh_positions(order: &[Vec<usize>], position: &mut [usize]) {
    for layer in order {
        for (slot, &node) in layer.iter().enumerate() {
            position[node] = slot;
        }
    }
}

/// Reorder `layer` by the barycentre of every node's `neighbours` (the nodes it
/// is linked to in the adjacent layer, whose slots are in `position`).
///
/// A node without neighbours has its current slot as its key, so it stays
/// where it is relative to the others; equal keys keep their current order.
fn reorder(layer: &mut [usize], neighbours: &[Vec<usize>], position: &mut [usize]) {
    let mut keyed: Vec<(f64, usize, usize)> = layer
        .iter()
        .enumerate()
        .map(|(slot, &node)| {
            let fixed = &neighbours[node];
            let key = if fixed.is_empty() {
                slot as f64
            } else {
                let total: f64 = fixed.iter().map(|&other| position[other] as f64).sum();
                total / fixed.len() as f64
            };
            (key, slot, node)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (slot, &(_, _, node)) in keyed.iter().enumerate() {
        layer[slot] = node;
        position[node] = slot;
    }
}

/// Crossings between every pair of adjacent layers; `below[n]` lists the
/// neighbours of `n` in the next layer.
///
/// Two edges `(a, b)` and `(c, d)` between layers `l` and `l + 1` cross exactly
/// when `a` is left of `c` and `b` is right of `d` (or the other way round), so
/// the crossings are the inversions of the sequence of target slots when the
/// edges are sorted by source slot.  A Fenwick tree counts them in
/// `O(E log V)`.
fn count_crossings(order: &[Vec<usize>], below: &[Vec<usize>], position: &[usize]) -> usize {
    let mut total = 0_usize;
    let mut targets: Vec<usize> = Vec::new();
    for pair in order.windows(2) {
        let (upper, lower) = (&pair[0], &pair[1]);
        let mut tree = vec![0_usize; lower.len() + 1];
        let mut seen = 0_usize;
        for &node in upper {
            targets.clear();
            targets.extend(below[node].iter().map(|&other| position[other]));
            // Edges from one node never cross each other.
            targets.sort_unstable();
            for &target in &targets {
                // Earlier edges that end strictly to the right of this one.
                total += seen - prefix_count(&tree, target + 1);
                add_one(&mut tree, target + 1);
                seen += 1;
            }
        }
    }
    total
}

/// Add one to the 1-based slot `slot` of a Fenwick tree.
fn add_one(tree: &mut [usize], mut slot: usize) {
    while slot < tree.len() {
        tree[slot] += 1;
        slot += slot & slot.wrapping_neg();
    }
}

/// Sum of the slots `1..=slot` of a Fenwick tree.
fn prefix_count(tree: &[usize], mut slot: usize) -> usize {
    let mut sum = 0;
    while slot > 0 {
        sum += tree.get(slot).copied().unwrap_or(0);
        slot -= slot & slot.wrapping_neg();
    }
    sum
}

#[cfg(test)]
mod tests;
