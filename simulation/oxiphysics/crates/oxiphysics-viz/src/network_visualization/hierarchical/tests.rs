// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the layering and the crossing minimisation of the
//! hierarchical layout.

use std::collections::HashMap;

use super::super::{NetworkEdge, NetworkGraph, NetworkNode};
use super::{assign_layers, break_cycles, crossing_count, minimize_crossings};

/// Graph with one node per id, inserted in the given order.
fn graph_of(ids: &[usize]) -> NetworkGraph {
    let mut graph = NetworkGraph::new();
    for &id in ids {
        graph.add_node(NetworkNode::new(id, format!("n{id}")));
    }
    graph
}

/// Add the directed edge `a -> b`.
fn arc(graph: &mut NetworkGraph, a: usize, b: usize) {
    graph.add_edge(NetworkEdge::directed(a, b, 1.0));
}

/// Deterministic LCG.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// Reference crossing count: every pair of edges between the same two adjacent
/// layers, compared directly.
fn brute_force_crossings(graph: &NetworkGraph, ordered: &[Vec<usize>]) -> usize {
    let mut slot: HashMap<usize, (usize, usize)> = HashMap::new();
    for (layer, ids) in ordered.iter().enumerate() {
        for (pos, &id) in ids.iter().enumerate() {
            slot.insert(id, (layer, pos));
        }
    }
    // Normalised to (upper slot, lower slot) per layer pair.
    let mut edges: Vec<(usize, usize, usize)> = Vec::new();
    for edge in &graph.edges {
        let (Some(&(la, pa)), Some(&(lb, pb))) = (slot.get(&edge.source), slot.get(&edge.target))
        else {
            continue;
        };
        if la + 1 == lb {
            edges.push((la, pa, pb));
        } else if lb + 1 == la {
            edges.push((lb, pb, pa));
        }
    }
    let mut crossings = 0;
    for (i, &(layer_a, upper_a, lower_a)) in edges.iter().enumerate() {
        for &(layer_b, upper_b, lower_b) in &edges[i + 1..] {
            if layer_a == layer_b
                && ((upper_a < upper_b && lower_a > lower_b)
                    || (upper_a > upper_b && lower_a < lower_b))
            {
                crossings += 1;
            }
        }
    }
    crossings
}

// ---------------------------------------------------------------------------
// Layer assignment
// ---------------------------------------------------------------------------

#[test]
fn layers_follow_the_longest_path() {
    let mut graph = graph_of(&[0, 1, 2, 3, 4]);
    arc(&mut graph, 0, 1);
    arc(&mut graph, 0, 2);
    arc(&mut graph, 1, 3);
    arc(&mut graph, 2, 3);
    arc(&mut graph, 0, 3); // a shortcut must not pull node 3 up
    let layers = assign_layers(&graph);
    let expected: HashMap<usize, usize> = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 0)]
        .into_iter()
        .collect();
    assert_eq!(layers, expected);
}

#[test]
fn layering_a_long_chain_takes_linear_time() {
    // The previous implementation scanned all edges for every node, i.e.
    // 10^10 edge visits for this chain.
    let n = 100_000;
    let mut graph = NetworkGraph::new();
    for id in 0..n {
        graph.add_node(NetworkNode::new(id, String::new()));
    }
    for id in 0..n - 1 {
        arc(&mut graph, id, id + 1);
    }
    let layers = assign_layers(&graph);
    assert_eq!(layers.len(), n);
    assert_eq!(layers[&0], 0);
    assert_eq!(layers[&(n / 2)], n / 2);
    assert_eq!(layers[&(n - 1)], n - 1);
}

#[test]
fn a_cycle_is_layered_by_reversing_one_of_its_edges() {
    // 0 -> 1 -> 2 -> 0: the previous code found no source and put every node
    // in layer 0.
    let mut graph = graph_of(&[0, 1, 2]);
    arc(&mut graph, 0, 1);
    arc(&mut graph, 1, 2);
    arc(&mut graph, 2, 0);
    let layers = assign_layers(&graph);
    let expected: HashMap<usize, usize> = [(0, 0), (1, 1), (2, 2)].into_iter().collect();
    assert_eq!(layers, expected);

    // A cycle hanging below a source, and a 2-cycle.
    let mut graph = graph_of(&[0, 1, 2, 3, 4]);
    arc(&mut graph, 0, 1);
    arc(&mut graph, 1, 2);
    arc(&mut graph, 2, 1);
    arc(&mut graph, 3, 4);
    arc(&mut graph, 4, 3);
    let layers = assign_layers(&graph);
    let expected: HashMap<usize, usize> = [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)]
        .into_iter()
        .collect();
    assert_eq!(layers, expected);
}

#[test]
fn every_edge_points_down_after_cycle_removal() {
    let mut rng = Lcg(5);
    for trial in 0..20 {
        let n = 30;
        let ids: Vec<usize> = (0..n).collect();
        let mut graph = graph_of(&ids);
        for _ in 0..70 {
            let a = rng.below(n);
            let b = rng.below(n);
            arc(&mut graph, a, b);
        }
        let layers = assign_layers(&graph);
        assert_eq!(layers.len(), n);
        // Each edge (a, b) with a != b either points down, or its reverse does
        // (it was a back edge) - never both ends in the same layer.
        for edge in &graph.edges {
            if edge.source != edge.target {
                assert_ne!(
                    layers[&edge.source], layers[&edge.target],
                    "trial {trial}: edge {}->{} is flat",
                    edge.source, edge.target
                );
            }
        }
        // The reversed graph is acyclic.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in &graph.edges {
            if edge.source != edge.target {
                successors[edge.source].push(edge.target);
            }
        }
        let acyclic = break_cycles(&successors);
        for (node, targets) in acyclic.iter().enumerate() {
            for &target in targets {
                assert!(layers[&node] < layers[&target], "trial {trial}");
            }
        }
        // Reversing an edge keeps the edge count.
        let before: usize = successors.iter().map(Vec::len).sum();
        let after: usize = acyclic.iter().map(Vec::len).sum();
        assert_eq!(before, after);
    }
}

#[test]
fn self_loops_dangling_edges_and_duplicate_nodes_are_harmless() {
    let mut graph = graph_of(&[7, 8, 8, 9]);
    arc(&mut graph, 7, 7);
    arc(&mut graph, 7, 8);
    arc(&mut graph, 8, 404);
    arc(&mut graph, 505, 9);
    arc(&mut graph, 8, 9);
    let layers = assign_layers(&graph);
    let expected: HashMap<usize, usize> = [(7, 0), (8, 1), (9, 2)].into_iter().collect();
    assert_eq!(layers, expected);
    assert!(assign_layers(&NetworkGraph::new()).is_empty());
}

#[test]
fn undirected_edges_are_layered_by_their_stored_orientation() {
    let mut graph = graph_of(&[0, 1, 2]);
    graph.add_edge(NetworkEdge::undirected(0, 1));
    graph.add_edge(NetworkEdge::undirected(1, 2));
    let layers = assign_layers(&graph);
    let expected: HashMap<usize, usize> = [(0, 0), (1, 1), (2, 2)].into_iter().collect();
    assert_eq!(layers, expected);
}

// ---------------------------------------------------------------------------
// Crossing count
// ---------------------------------------------------------------------------

#[test]
fn crossing_count_of_small_drawings() {
    // 0 -> 3 and 1 -> 2 cross when 2 is left of 3 ...
    let mut graph = graph_of(&[0, 1, 2, 3]);
    arc(&mut graph, 0, 3);
    arc(&mut graph, 1, 2);
    assert_eq!(crossing_count(&graph, &[vec![0, 1], vec![2, 3]]), 1);
    assert_eq!(crossing_count(&graph, &[vec![0, 1], vec![3, 2]]), 0);
    assert_eq!(crossing_count(&graph, &[vec![1, 0], vec![2, 3]]), 0);
    // Parallel edges count individually.
    arc(&mut graph, 0, 3);
    assert_eq!(crossing_count(&graph, &[vec![0, 1], vec![2, 3]]), 2);
    // Edges that share an end point never cross.
    let mut fan = graph_of(&[0, 1, 2]);
    arc(&mut fan, 0, 1);
    arc(&mut fan, 0, 2);
    assert_eq!(crossing_count(&fan, &[vec![0], vec![1, 2]]), 0);
    // Edges spanning two layers are not counted.
    let mut long = graph_of(&[0, 1, 2, 3, 4]);
    arc(&mut long, 0, 4);
    arc(&mut long, 1, 2);
    arc(&mut long, 3, 2);
    assert_eq!(crossing_count(&long, &[vec![0, 1], vec![3, 2], vec![4]]), 0);
    // No layers, no crossings.
    assert_eq!(crossing_count(&graph, &[]), 0);
}

#[test]
fn crossing_count_matches_the_brute_force_count() {
    let mut rng = Lcg(77);
    for trial in 0..25 {
        let layer_sizes = [5, 7, 6, 4];
        let mut ordered: Vec<Vec<usize>> = Vec::new();
        let mut next_id = 0;
        for &size in &layer_sizes {
            ordered.push((next_id..next_id + size).collect());
            next_id += size;
        }
        let ids: Vec<usize> = (0..next_id).collect();
        let mut graph = graph_of(&ids);
        for pair in 0..layer_sizes.len() - 1 {
            for _ in 0..12 {
                let a = ordered[pair][rng.below(layer_sizes[pair])];
                let b = ordered[pair + 1][rng.below(layer_sizes[pair + 1])];
                if rng.below(2) == 0 {
                    arc(&mut graph, a, b);
                } else {
                    graph.add_edge(NetworkEdge::undirected(b, a));
                }
            }
        }
        // Shuffle each layer a bit.
        for layer in &mut ordered {
            for i in (1..layer.len()).rev() {
                layer.swap(i, rng.below(i + 1));
            }
        }
        assert_eq!(
            crossing_count(&graph, &ordered),
            brute_force_crossings(&graph, &ordered),
            "trial {trial}"
        );
    }
}

// ---------------------------------------------------------------------------
// Crossing minimisation
// ---------------------------------------------------------------------------

#[test]
fn a_directed_matching_is_untangled() {
    // 0 -> 3 and 1 -> 2: the seed order [0, 1] / [2, 3] crosses.
    let mut graph = graph_of(&[0, 1, 2, 3]);
    arc(&mut graph, 0, 3);
    arc(&mut graph, 1, 2);
    let layers = assign_layers(&graph);
    let ordered = minimize_crossings(&graph, &layers, 4);
    assert_eq!(ordered, vec![vec![0, 1], vec![3, 2]]);
    assert_eq!(crossing_count(&graph, &ordered), 0);
}

#[test]
fn an_undirected_matching_is_untangled_as_well() {
    let mut graph = graph_of(&[0, 1, 2, 3]);
    graph.add_edge(NetworkEdge::undirected(0, 3));
    graph.add_edge(NetworkEdge::undirected(2, 1));
    let layers = assign_layers(&graph);
    // 2 -> 1 puts node 2 above node 1.
    let ordered = minimize_crossings(&graph, &layers, 4);
    assert_eq!(crossing_count(&graph, &ordered), 0);
}

#[test]
fn k22_keeps_its_one_unavoidable_crossing() {
    // Any two-layer drawing of K(2,2) has exactly one crossing.
    let mut graph = graph_of(&[0, 1, 2, 3]);
    for source in [0, 1] {
        for target in [2, 3] {
            arc(&mut graph, source, target);
        }
    }
    let layers = assign_layers(&graph);
    for passes in [0, 1, 4, 50] {
        let ordered = minimize_crossings(&graph, &layers, passes);
        assert_eq!(crossing_count(&graph, &ordered), 1, "passes {passes}");
        assert_eq!(ordered.len(), 2);
    }
}

#[test]
fn zero_passes_return_the_seed_order_by_id() {
    let mut graph = graph_of(&[5, 3, 9, 1, 7]);
    arc(&mut graph, 5, 1);
    arc(&mut graph, 3, 9);
    arc(&mut graph, 9, 7);
    let layers = assign_layers(&graph);
    let ordered = minimize_crossings(&graph, &layers, 0);
    // Layer 0: sources 3 and 5; layer 1: 1 and 9; layer 2: 7.
    assert_eq!(ordered, vec![vec![3, 5], vec![1, 9], vec![7]]);
}

#[test]
fn empty_layer_maps_and_graphs_give_no_layers() {
    let graph = graph_of(&[1, 2]);
    assert!(minimize_crossings(&graph, &HashMap::new(), 4).is_empty());
    assert!(minimize_crossings(&NetworkGraph::new(), &HashMap::new(), 4).is_empty());
}

#[test]
fn the_result_is_never_worse_than_the_seed_and_usually_better() {
    let mut rng = Lcg(2024);
    let mut improved = 0;
    for trial in 0..20 {
        let layer_sizes = [6, 8, 8, 6];
        let mut layers: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (layer, &size) in layer_sizes.iter().enumerate() {
            let ids: Vec<usize> = (next_id..next_id + size).collect();
            for &id in &ids {
                layers.insert(id, layer);
            }
            members.push(ids);
            next_id += size;
        }
        let all: Vec<usize> = (0..next_id).collect();
        let mut graph = graph_of(&all);
        for pair in 0..layer_sizes.len() - 1 {
            for &upper in &members[pair] {
                for _ in 0..2 {
                    let lower = members[pair + 1][rng.below(layer_sizes[pair + 1])];
                    arc(&mut graph, upper, lower);
                }
            }
        }
        let seed = minimize_crossings(&graph, &layers, 0);
        let ordered = minimize_crossings(&graph, &layers, 8);
        let seed_crossings = crossing_count(&graph, &seed);
        let crossings = crossing_count(&graph, &ordered);
        assert!(crossings <= seed_crossings, "trial {trial}");
        assert_eq!(crossings, brute_force_crossings(&graph, &ordered));
        if crossings < seed_crossings {
            improved += 1;
        }
        // The result is a permutation of the seed within every layer.
        for (layer, ids) in ordered.iter().enumerate() {
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, members[layer], "trial {trial} layer {layer}");
        }
    }
    assert!(improved >= 15, "only {improved} of 20 graphs improved");
}

#[test]
fn the_order_does_not_depend_on_how_the_graph_was_built() {
    let edges = [
        (0, 6),
        (1, 5),
        (2, 4),
        (3, 7),
        (1, 7),
        (0, 4),
        (6, 8),
        (5, 9),
    ];
    let ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let build = |node_order: &[usize], edge_order: &[usize]| {
        let mut graph = graph_of(node_order);
        for &i in edge_order {
            arc(&mut graph, edges[i].0, edges[i].1);
        }
        graph
    };
    let forward: Vec<usize> = (0..edges.len()).collect();
    let backward: Vec<usize> = (0..edges.len()).rev().collect();
    let reversed_ids: Vec<usize> = ids.iter().rev().copied().collect();
    let reference_graph = build(&ids, &forward);
    let reference_layers = assign_layers(&reference_graph);
    let reference = minimize_crossings(&reference_graph, &reference_layers, 6);
    for _ in 0..10 {
        // Fresh hash maps iterate in a different order every time.
        let graph = build(&reversed_ids, &backward);
        let layers = assign_layers(&graph);
        assert_eq!(layers, reference_layers);
        assert_eq!(minimize_crossings(&graph, &layers, 6), reference);
    }
}

#[test]
fn isolated_nodes_keep_their_slot_between_the_others() {
    // Layer 1 is [3, 4, 5] with 3 linked to node 2 and 5 linked to node 0; node
    // 4 has no neighbour.  The linked nodes swap and the isolated one, which
    // keeps its slot as its sort key, ends up between them.
    let mut graph = graph_of(&[0, 1, 2, 3, 4, 5]);
    arc(&mut graph, 0, 5);
    arc(&mut graph, 2, 3);
    let mut layers: HashMap<usize, usize> = HashMap::new();
    for (id, layer) in [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)] {
        layers.insert(id, layer);
    }
    let seed = minimize_crossings(&graph, &layers, 0);
    assert_eq!(crossing_count(&graph, &seed), 1);
    let ordered = minimize_crossings(&graph, &layers, 4);
    assert_eq!(ordered, vec![vec![0, 1, 2], vec![5, 4, 3]]);
    assert_eq!(crossing_count(&graph, &ordered), 0);
}

#[test]
fn layers_may_have_gaps() {
    let mut graph = graph_of(&[0, 1, 2]);
    arc(&mut graph, 0, 1);
    arc(&mut graph, 1, 2);
    let layers: HashMap<usize, usize> = [(0, 0), (1, 2), (2, 3)].into_iter().collect();
    let ordered = minimize_crossings(&graph, &layers, 3);
    assert_eq!(ordered, vec![vec![0], vec![], vec![1], vec![2]]);
}
