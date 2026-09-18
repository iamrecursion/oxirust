// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the label-propagation kernel.

use super::super::test_support::{Lcg, clique_edges, graph_from_edges, planted_partition};
use super::super::{WeightedGraph, densify};
use super::propagate;

/// Dense labels of `propagate` with the node index as the node id.
fn dense_labels(graph: &WeightedGraph, max_iter: usize) -> Vec<usize> {
    let ids: Vec<usize> = (0..graph.len()).collect();
    let mut labels = propagate(&ids, graph, max_iter);
    densify(&mut labels);
    labels
}

#[test]
fn disconnected_cliques_each_converge_to_one_label() {
    let mut edges = clique_edges(&[0, 1, 2, 3, 4], 1.0);
    edges.extend(clique_edges(&[5, 6, 7, 8, 9], 1.0));
    let graph = graph_from_edges(10, &edges);
    let labels = dense_labels(&graph, 20);
    assert_eq!(labels, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
}

#[test]
fn nodes_without_neighbours_keep_their_own_label() {
    let graph = graph_from_edges(4, &[(1, 2, 1.0)]);
    let labels = dense_labels(&graph, 10);
    assert_eq!(labels[0], 0);
    assert_eq!(labels[1], labels[2]);
    assert_ne!(labels[3], labels[0]);
    assert_ne!(labels[3], labels[1]);
    // A self-loop is no neighbour either.
    let mut looped = WeightedGraph::with_nodes(2);
    looped.add_edge(0, 0, 5.0);
    assert_eq!(dense_labels(&looped, 10), vec![0, 1]);
}

#[test]
fn no_sweeps_leave_every_node_alone() {
    let graph = graph_from_edges(5, &clique_edges(&[0, 1, 2, 3, 4], 1.0));
    assert_eq!(dense_labels(&graph, 0), vec![0, 1, 2, 3, 4]);
}

#[test]
fn a_node_follows_the_heavier_side_not_the_larger_one() {
    // Node 3 touches node 0 with weight 3 and the pair {1, 2} with weight 1
    // each.  1 and 2 hold on to each other (weight 10).  By neighbour count
    // node 3 joins {1, 2}; by weight it stays with node 0.
    let graph = graph_from_edges(4, &[(1, 2, 10.0), (3, 0, 3.0), (3, 1, 1.0), (3, 2, 1.0)]);
    let labels = dense_labels(&graph, 20);
    assert_eq!(labels[0], labels[3]);
    assert_eq!(labels[1], labels[2]);
    assert_ne!(labels[0], labels[1]);
}

#[test]
fn ties_go_to_the_smallest_label_whatever_the_edge_order() {
    // Node 3 has the smallest id, so it is visited first and sees the labels
    // of nodes 0, 1 and 2 -- the ranks 1, 2 and 3 of their ids -- with equal
    // weight.  It must take label 1, however the adjacency list is ordered.
    let ids = [1, 2, 3, 0];
    let forward = graph_from_edges(4, &[(3, 0, 1.0), (3, 1, 1.0), (3, 2, 1.0)]);
    let backward = graph_from_edges(4, &[(3, 2, 1.0), (3, 1, 1.0), (3, 0, 1.0)]);
    for graph in [&forward, &backward] {
        let labels = propagate(&ids, graph, 1);
        assert_eq!(labels, vec![1, 1, 1, 1]);
    }
}

#[test]
fn weights_that_differ_only_by_rounding_still_tie() {
    // Node 2 (smallest id, visited first) sees label 1 with weight 0.3 and
    // label 2 with weight 0.1 + 0.2, which is one unit in the last place more.
    // They are the same weight, so the smaller label 1 has to win.
    assert_ne!(0.1_f64 + 0.2, 0.3);
    let graph = graph_from_edges(3, &[(2, 0, 0.3), (2, 1, 0.1), (2, 1, 0.2)]);
    let labels = propagate(&[1, 2, 0], &graph, 1);
    assert_eq!(labels, vec![1, 1, 1]);
}

#[test]
fn the_outcome_depends_on_the_ids_not_on_the_node_indices() {
    // The same random weighted graph twice: once with node i at index i, once
    // with the indices permuted (node i at index perm[i]).  Visiting order and
    // labels follow the ids, so every id must end with the same label.
    let mut rng = Lcg(11);
    let n = 14;
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for _ in 0..40 {
        let a = rng.below(n);
        let b = rng.below(n);
        if a != b {
            edges.push((a, b, 1.0 + rng.below(3) as f64));
        }
    }
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let straight = graph_from_edges(n, &edges);
    let permuted_edges: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|&(a, b, w)| (perm[a], perm[b], w))
        .collect();
    let permuted = graph_from_edges(n, &permuted_edges);
    let straight_ids: Vec<usize> = (0..n).collect();
    let mut permuted_ids = vec![0; n];
    for (id, &index) in perm.iter().enumerate() {
        permuted_ids[index] = id;
    }
    let expected = propagate(&straight_ids, &straight, 50);
    let found = propagate(&permuted_ids, &permuted, 50);
    for id in 0..n {
        assert_eq!(found[perm[id]], expected[id], "id {id}");
    }
}

#[test]
fn results_are_deterministic_and_labels_stay_in_range() {
    let graph = planted_partition(4, 25, 3, 2, 99);
    let ids: Vec<usize> = (0..graph.len()).collect();
    let first = propagate(&ids, &graph, 30);
    let second = propagate(&ids, &graph, 30);
    assert_eq!(first, second);
    assert!(first.iter().all(|&label| label < graph.len()));
}

#[test]
fn extra_sweeps_do_not_change_a_converged_labelling() {
    let mut edges = clique_edges(&[0, 1, 2, 3, 4], 1.0);
    edges.extend(clique_edges(&[5, 6, 7, 8, 9], 1.0));
    let graph = graph_from_edges(10, &edges);
    let ids: Vec<usize> = (0..10).collect();
    // The first sweep settles everything and the second confirms it.
    let settled = propagate(&ids, &graph, 2);
    assert_eq!(propagate(&ids, &graph, 10_000), settled);
}

#[test]
fn groups_without_bridges_never_share_a_label() {
    // Disconnected dense groups cannot exchange labels.
    let graph = planted_partition(5, 20, 6, 0, 7);
    let labels = dense_labels(&graph, 50);
    for group in 0..5 {
        let members = &labels[group * 20..(group + 1) * 20];
        for other in (0..5).filter(|&other| other != group) {
            let rivals = &labels[other * 20..(other + 1) * 20];
            assert!(members.iter().all(|label| !rivals.contains(label)));
        }
    }
}
