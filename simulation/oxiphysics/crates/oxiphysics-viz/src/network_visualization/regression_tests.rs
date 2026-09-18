// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the audit findings on the network visualisation:
//!
//! * F5072 - `dfs` recursed once per node of a path.
//! * F5074 - edges to unknown nodes made the layout and the metrics panic.
//! * F5075 - `barnes_hut_repulsion` dropped nearby cells and `apply` never
//!   used it.
//! * F5076 / F5077 - the Sugiyama crossing minimisation did nothing on directed
//!   graphs and depended on hash-map order.
//! * F5080 - `density` halved the density of undirected graphs.
//! * F5087 - `NetworkAnimation::frame_at` replayed events in insertion order.
//!
//! The community-detection findings (F5081, F5082, F5086) have their tests in
//! `tests/network_community.rs` and next to the kernels in `community/`.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::{
    CommunityDetection, ForceDirectedConfig, ForceDirectedLayout, HierarchicalLayout,
    NetworkAnimation, NetworkEdge, NetworkEvent, NetworkGraph, NetworkMetrics, NetworkNode,
};
use crate::error::Error;

/// Graph with one node per id, inserted in the given order.
fn graph_of(ids: &[usize]) -> NetworkGraph {
    let mut graph = NetworkGraph::new();
    for &id in ids {
        graph.add_node(NetworkNode::new(id, format!("n{id}")));
    }
    graph
}

/// Deterministic LCG.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        ((self.next_u64() >> 33) as usize) % bound
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// F5072 - iterative DFS
// ---------------------------------------------------------------------------

#[test]
fn dfs_visits_a_million_node_path_without_overflowing_the_stack() {
    // `dfs` only reads the adjacency lists, so fill them directly: a path
    // 0 - 1 - ... - (n - 1).  The recursive version needed one stack frame per
    // node and aborted long before 10^6 (the test threads have 2 MiB).
    let n = 1_000_000;
    let mut graph = NetworkGraph::new();
    graph.adjacency.reserve(n);
    for i in 0..n {
        let mut adjacent = Vec::with_capacity(2);
        if i > 0 {
            adjacent.push((i - 1, 1.0));
        }
        if i + 1 < n {
            adjacent.push((i + 1, 1.0));
        }
        graph.adjacency.insert(i, adjacent);
    }
    let order = graph.dfs(0);
    assert_eq!(order.len(), n);
    assert!(order.iter().copied().eq(0..n));
    // From the middle the first neighbour (towards 0) is followed first.
    let from_middle = graph.dfs(n / 2);
    assert_eq!(from_middle.len(), n);
    assert_eq!(from_middle[1], n / 2 - 1);
}

#[test]
fn dfs_keeps_the_preorder_of_the_recursive_search() {
    // Tree: 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5}.
    let mut tree = graph_of(&[0, 1, 2, 3, 4, 5]);
    for (a, b) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)] {
        tree.add_edge(NetworkEdge::undirected(a, b));
    }
    assert_eq!(tree.dfs(0), vec![0, 1, 3, 4, 2, 5]);

    // Diamond 0-1, 0-2, 1-3, 2-3: 3 is reached through 1 and not through 2.
    let mut diamond = graph_of(&[0, 1, 2, 3]);
    for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        diamond.add_edge(NetworkEdge::undirected(a, b));
    }
    assert_eq!(diamond.dfs(0), vec![0, 1, 3, 2]);

    // Directed edges are followed in their direction only, cycles terminate.
    let mut cycle = graph_of(&[0, 1, 2, 3]);
    for (a, b) in [(0, 1), (1, 2), (2, 0), (2, 3)] {
        cycle.add_edge(NetworkEdge::directed(a, b, 1.0));
    }
    assert_eq!(cycle.dfs(1), vec![1, 2, 0, 3]);
    assert_eq!(cycle.dfs(3), vec![3]);

    // A start that is not a node is reported alone.
    assert_eq!(tree.dfs(42), vec![42]);
}

#[test]
fn dfs_and_bfs_agree_on_the_reachable_set() {
    let mut rng = Lcg(9);
    let n = 200;
    let ids: Vec<usize> = (0..n).collect();
    let mut graph = graph_of(&ids);
    for _ in 0..260 {
        let a = rng.below(n);
        let b = rng.below(n);
        graph.add_edge(NetworkEdge::directed(a, b, 1.0));
    }
    for start in [0, 17, 199] {
        let dfs = graph.dfs(start);
        let bfs = graph.bfs(start);
        let dfs_set: HashSet<usize> = dfs.iter().copied().collect();
        let bfs_set: HashSet<usize> = bfs.iter().copied().collect();
        assert_eq!(dfs.len(), dfs_set.len(), "dfs visits a node twice");
        assert_eq!(dfs_set, bfs_set);
        assert_eq!(dfs[0], start);
    }
}

// ---------------------------------------------------------------------------
// F5074 - edges to unknown nodes
// ---------------------------------------------------------------------------

#[test]
fn try_add_edge_checks_both_end_points() {
    let mut graph = graph_of(&[0, 1]);
    assert!(graph.try_add_edge(NetworkEdge::undirected(0, 1)).is_ok());
    assert_eq!(graph.edge_count(), 1);
    assert!(graph.has_edge(0, 1) && graph.has_edge(1, 0));

    let err = graph
        .try_add_edge(NetworkEdge::undirected(0, 99))
        .expect_err("node 99 does not exist");
    assert!(matches!(&err, Error::NotFound { name } if name == "node 99"));
    let err = graph
        .try_add_edge(NetworkEdge::directed(98, 1, 2.0))
        .expect_err("node 98 does not exist");
    assert!(matches!(&err, Error::NotFound { name } if name == "node 98"));
    // The source is reported first when both are missing.
    let err = graph
        .try_add_edge(NetworkEdge::undirected(7, 8))
        .expect_err("neither exists");
    assert!(matches!(&err, Error::NotFound { name } if name == "node 7"));

    // Nothing was recorded for the refused edges.
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.adjacency.len(), 2);
    assert!(!graph.has_edge(0, 99) && !graph.has_edge(99, 0));
    assert_eq!(graph.degree(0), 1);
}

#[test]
fn try_add_edge_and_add_edge_record_an_edge_identically() {
    let mut strict = graph_of(&[3, 4]);
    let mut lenient = graph_of(&[3, 4]);
    let edge = NetworkEdge::directed(3, 4, 2.5).with_label("link");
    strict.try_add_edge(edge.clone()).expect("both are nodes");
    lenient.add_edge(edge);
    assert_eq!(strict.edges.len(), lenient.edges.len());
    assert_eq!(strict.adjacency, lenient.adjacency);
    assert_eq!(strict.edges[0].label.as_deref(), Some("link"));
}

#[test]
fn add_edge_still_accepts_edges_whose_nodes_come_later() {
    let mut graph = NetworkGraph::new();
    graph.add_edge(NetworkEdge::undirected(0, 1));
    graph.add_node(NetworkNode::new(0, "a"));
    graph.add_node(NetworkNode::new(1, "b"));
    assert!(graph.has_edge(0, 1) && graph.has_edge(1, 0));
    assert_eq!(NetworkMetrics::new(graph.clone()).density(), 1.0);
    let result = CommunityDetection::new(graph).label_propagation(10);
    assert_eq!(result.num_communities(), 1);
}

/// One node with edges to nodes that do not exist, in every flavour.
fn dangling_single_node() -> NetworkGraph {
    let mut graph = graph_of(&[0]);
    graph.add_edge(NetworkEdge::undirected(0, 99));
    graph.add_edge(NetworkEdge::directed(0, 98, 2.0));
    graph.add_edge(NetworkEdge::directed(97, 0, 3.0));
    graph.add_edge(NetworkEdge::undirected(96, 95));
    graph
}

#[test]
fn layouts_survive_edges_to_unknown_nodes() {
    let mut graph = dangling_single_node();
    ForceDirectedLayout::new().apply(&mut graph);
    assert!(graph.nodes[0].position.iter().all(|c| c.is_finite()));
    HierarchicalLayout::new().apply(&mut graph);
    assert_eq!(graph.nodes[0].position[1], 0.0);
}

#[test]
fn dangling_edges_exert_no_force() {
    let mut clean = graph_of(&[0, 1, 2, 3]);
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        clean.add_edge(NetworkEdge::undirected(a, b));
    }
    let mut noisy = clean.clone();
    noisy.add_edge(NetworkEdge::undirected(2, 77));
    noisy.add_edge(NetworkEdge::directed(66, 0, 1.0));
    noisy.add_edge(NetworkEdge::undirected(3, 3)); // a self-loop
    let layout = ForceDirectedLayout::new();
    layout.apply(&mut clean);
    layout.apply(&mut noisy);
    for (a, b) in clean.nodes.iter().zip(&noisy.nodes) {
        assert_eq!(a.position, b.position, "node {}", a.id);
    }
}

#[test]
fn metrics_ignore_edges_to_unknown_nodes() {
    let mut single = NetworkMetrics::new(dangling_single_node());
    let degrees = single.degree_distribution();
    assert_eq!(degrees.total_nodes, 1);
    assert_eq!(degrees.counts, HashMap::from([(0, 1)]));
    assert_eq!(degrees.mean_degree(), 0.0);
    assert_eq!(single.local_clustering_coefficient(0), 0.0);
    assert_eq!(single.global_clustering_coefficient(), 0.0);
    assert_eq!(single.betweenness_centrality(), HashMap::from([(0, 0.0)]));
    let ranks = single.pagerank(0.85, 100, 1e-12);
    assert_eq!(ranks.len(), 1);
    assert!((ranks[&0] - 1.0).abs() < 1e-12);
    assert_eq!(single.density(), 0.0);
    assert_eq!(single.average_shortest_path_length(), Some(0.0));
    single = NetworkMetrics::new(NetworkGraph::new());
    assert!(single.betweenness_centrality().is_empty());
}

#[test]
fn a_noisy_graph_has_the_metrics_of_its_clean_core() {
    let mut clean = graph_of(&[0, 1, 2, 3]);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (1, 3)] {
        clean.add_edge(NetworkEdge::undirected(a, b));
    }
    clean.add_edge(NetworkEdge::directed(3, 0, 1.0));
    let mut noisy = clean.clone();
    noisy.add_edge(NetworkEdge::undirected(1, 50));
    noisy.add_edge(NetworkEdge::directed(51, 2, 1.0));
    noisy.add_edge(NetworkEdge::directed(0, 52, 1.0));
    let clean = NetworkMetrics::new(clean);
    let noisy = NetworkMetrics::new(noisy);
    assert_eq!(
        clean.degree_distribution().counts,
        noisy.degree_distribution().counts
    );
    for id in 0..4 {
        assert_eq!(
            clean.local_clustering_coefficient(id),
            noisy.local_clustering_coefficient(id)
        );
    }
    assert_eq!(
        clean.betweenness_centrality(),
        noisy.betweenness_centrality()
    );
    assert_eq!(
        clean.pagerank(0.85, 200, 1e-12),
        noisy.pagerank(0.85, 200, 1e-12)
    );
    assert_eq!(clean.density(), noisy.density());
    assert_eq!(
        clean.average_shortest_path_length(),
        noisy.average_shortest_path_length()
    );
}

#[test]
fn community_detection_reports_only_real_nodes() {
    let detector = CommunityDetection::new(dangling_single_node());
    let expected = HashMap::from([(0, 0)]);
    for result in [
        detector.label_propagation(10),
        detector.greedy_modularity(),
        detector.spectral_clustering(2),
    ] {
        assert_eq!(result.membership, expected);
        assert_eq!(result.modularity, 0.0);
    }
    assert_eq!(detector.modularity(&expected), 0.0);
}

#[test]
fn betweenness_matches_the_textbook_values() {
    // Path 0 - 1 - 2 - 3 - 4: every pair is counted in both directions, so the
    // raw scores are 2 * (pairs through the node) and the normalisation is
    // 1 / ((n - 1)(n - 2)) = 1 / 12.  Node 1 lies on 3 pairs, node 2 on 4.
    let mut path = graph_of(&[0, 1, 2, 3, 4]);
    for i in 0..4 {
        path.add_edge(NetworkEdge::undirected(i, i + 1));
    }
    let bc = NetworkMetrics::new(path).betweenness_centrality();
    assert!(bc[&0].abs() < 1e-12 && bc[&4].abs() < 1e-12);
    assert!((bc[&1] - 6.0 / 12.0).abs() < 1e-12);
    assert!((bc[&2] - 8.0 / 12.0).abs() < 1e-12);
    assert!((bc[&3] - 6.0 / 12.0).abs() < 1e-12);

    // Star: the hub lies on every pair of leaves (here 6 pairs, 12 ordered).
    let mut star = graph_of(&[10, 1, 2, 3, 4]);
    for leaf in [1, 2, 3, 4] {
        star.add_edge(NetworkEdge::undirected(10, leaf));
    }
    let bc = NetworkMetrics::new(star).betweenness_centrality();
    assert!((bc[&10] - 1.0).abs() < 1e-12);
    assert!(bc[&1].abs() < 1e-12);

    // Directed path 0 -> 1 -> 2: only the ordered pair (0, 2) passes node 1.
    let mut directed = graph_of(&[0, 1, 2]);
    directed.add_edge(NetworkEdge::directed(0, 1, 1.0));
    directed.add_edge(NetworkEdge::directed(1, 2, 1.0));
    let bc = NetworkMetrics::new(directed).betweenness_centrality();
    assert!((bc[&1] - 1.0 / 2.0).abs() < 1e-12);
    assert!(bc[&0].abs() < 1e-12 && bc[&2].abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// F5075 - Barnes-Hut repulsion
// ---------------------------------------------------------------------------

/// Graph with random positions (ids `0..count`) and no edges.
fn scattered_graph(count: usize, seed: u64, extent: f64) -> NetworkGraph {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut graph = NetworkGraph::new();
    for id in 0..count {
        let x = (2.0 * rng.unit() - 1.0) * extent;
        let y = (2.0 * rng.unit() - 1.0) * extent;
        graph.add_node(NetworkNode::new(id, String::new()).with_position(x, y));
    }
    graph
}

/// Reference repulsion `repulsion * k^2 / d` along the line between every pair,
/// computed with plain double loops over the ids.
fn naive_repulsion(
    graph: &NetworkGraph,
    layout: &ForceDirectedLayout,
) -> BTreeMap<usize, [f64; 2]> {
    let k = layout.config.optimal_distance;
    let mut forces = BTreeMap::new();
    for a in &graph.nodes {
        let mut force = [0.0_f64; 2];
        for b in &graph.nodes {
            if a.id == b.id {
                continue;
            }
            let dx = a.position[0] - b.position[0];
            let dy = a.position[1] - b.position[1];
            let dist = dx.hypot(dy).max(0.01);
            let magnitude = layout.config.repulsion * k * k / dist;
            force[0] += magnitude * dx / dist;
            force[1] += magnitude * dy / dist;
        }
        forces.insert(a.id, force);
    }
    forces
}

fn layout_with_theta(theta: f64) -> ForceDirectedLayout {
    ForceDirectedLayout::with_config(ForceDirectedConfig {
        barnes_hut_theta: theta,
        ..ForceDirectedConfig::default()
    })
}

#[test]
fn barnes_hut_with_theta_zero_equals_the_exact_repulsion() {
    let graph = scattered_graph(50, 5, 100.0);
    let layout = layout_with_theta(0.0);
    let exact = naive_repulsion(&graph, &layout);
    let tree = layout.barnes_hut_repulsion(&graph);
    assert_eq!(tree.len(), 50);
    for (id, expected) in &exact {
        let got = tree[id];
        assert!(
            (got[0] - expected[0]).abs() < 1e-9 && (got[1] - expected[1]).abs() < 1e-9,
            "node {id}: {got:?} vs {expected:?}"
        );
    }
}

#[test]
fn barnes_hut_with_the_default_theta_is_within_five_percent() {
    for (count, seed) in [(50, 6_u64), (500, 7)] {
        let graph = scattered_graph(count, seed, 150.0);
        let layout = ForceDirectedLayout::new();
        assert_eq!(layout.config.barnes_hut_theta, 0.5);
        let exact = naive_repulsion(&graph, &layout);
        let tree = layout.barnes_hut_repulsion(&graph);
        let (mut error, mut norm) = (0.0_f64, 0.0_f64);
        for (id, e) in &exact {
            let t = tree[id];
            error += (t[0] - e[0]).powi(2) + (t[1] - e[1]).powi(2);
            norm += e[0].powi(2) + e[1].powi(2);
        }
        let relative = (error / norm).sqrt();
        assert!(relative < 0.05, "{count} nodes: relative error {relative}");
    }
}

#[test]
fn barnes_hut_does_not_drop_nearby_nodes() {
    // Two nodes 10 apart.  The previous grid with cells of 2k / theta = 200
    // put both in one multi-node cell and gave neither any repulsion.
    let mut graph = graph_of(&[0, 1]);
    graph.nodes[0].position = [0.0, 0.0];
    graph.nodes[1].position = [10.0, 0.0];
    let forces = ForceDirectedLayout::new().barnes_hut_repulsion(&graph);
    // k^2 / d = 2500 / 10 along the line, pointing apart.
    assert!((forces[&0][0] + 250.0).abs() < 1e-9);
    assert!((forces[&1][0] - 250.0).abs() < 1e-9);
    assert!(forces[&0][1].abs() < 1e-12 && forces[&1][1].abs() < 1e-12);
}

#[test]
fn barnes_hut_handles_nodes_at_the_same_position() {
    // A fresh graph has all nodes at the origin.
    let graph = graph_of(&[0, 1, 2]);
    let forces = ForceDirectedLayout::new().barnes_hut_repulsion(&graph);
    assert_eq!(forces.len(), 3);
    assert!(forces.values().all(|f| *f == [0.0, 0.0]));
}

#[test]
fn apply_switches_to_the_quadtree_above_256_nodes() {
    // One iteration without edges and with a temperature that never clamps:
    // every node then moves by exactly its repulsive force, so the exact and
    // the approximate layout can be compared force by force.  (After several
    // iterations of a symmetric start the tiny force errors would be amplified
    // by cancellation and the comparison would say little.)
    let config = |theta: f64| ForceDirectedConfig {
        iterations: 1,
        temperature: 1.0e9,
        barnes_hut_theta: theta,
        ..ForceDirectedConfig::default()
    };
    let displacements = |count: usize, theta: f64| -> Vec<[f64; 2]> {
        let mut graph = scattered_graph(count, 77, 400.0);
        let before: Vec<[f64; 2]> = graph.nodes.iter().map(|node| node.position).collect();
        ForceDirectedLayout::with_config(config(theta)).apply(&mut graph);
        graph
            .nodes
            .iter()
            .zip(&before)
            .map(|(node, start)| [node.position[0] - start[0], node.position[1] - start[1]])
            .collect()
    };
    // 256 nodes: the exact sum whatever theta is, so the results are identical.
    assert_eq!(displacements(256, 0.0), displacements(256, 0.5));
    // The exact path moves every node by the reference repulsion.
    let layout = layout_with_theta(0.0);
    let graph = scattered_graph(257, 77, 400.0);
    let reference = naive_repulsion(&graph, &layout);
    let exact = displacements(257, 0.0);
    for (node, moved) in graph.nodes.iter().zip(&exact) {
        let expected = reference[&node.id];
        assert!((moved[0] - expected[0]).abs() < 1e-9, "node {}", node.id);
        assert!((moved[1] - expected[1]).abs() < 1e-9, "node {}", node.id);
    }
    // 257 nodes: the approximation kicks in.  It differs from the exact sum,
    // but only by a fraction of a percent.
    let approximate = displacements(257, 0.5);
    assert_ne!(exact, approximate, "the quadtree path was not taken");
    let (mut error, mut norm) = (0.0_f64, 0.0_f64);
    for (a, e) in approximate.iter().zip(&exact) {
        assert!(a.iter().all(|c| c.is_finite()));
        error += (a[0] - e[0]).powi(2) + (a[1] - e[1]).powi(2);
        norm += e[0].powi(2) + e[1].powi(2);
    }
    let relative = (error / norm).sqrt();
    assert!(relative < 0.05, "relative error {relative}");
}

#[test]
fn a_large_layout_spreads_the_nodes_out() {
    // 600 nodes in a random tree: the layout has to run through the quadtree
    // and still separate the nodes (no two at the same place).
    let mut rng = Lcg(31);
    let n = 600;
    let mut graph = graph_of(&(0..n).collect::<Vec<_>>());
    for node in 1..n {
        graph.add_edge(NetworkEdge::undirected(node, rng.below(node)));
    }
    ForceDirectedLayout::new().apply(&mut graph);
    let mut cells: HashSet<(i64, i64)> = HashSet::new();
    for node in &graph.nodes {
        assert!(node.position.iter().all(|c| c.is_finite()));
        cells.insert((
            node.position[0].floor() as i64,
            node.position[1].floor() as i64,
        ));
    }
    assert!(
        cells.len() > 550,
        "only {} distinct unit cells",
        cells.len()
    );
}

// ---------------------------------------------------------------------------
// F5076 / F5077 - Sugiyama
// ---------------------------------------------------------------------------

#[test]
fn hierarchical_layout_untangles_a_directed_matching() {
    // 0 -> 3 and 1 -> 2: with the nodes of each layer in id order the edges
    // cross; after the layout 3 stands left of 2.
    let mut graph = graph_of(&[0, 1, 2, 3]);
    graph.add_edge(NetworkEdge::directed(0, 3, 1.0));
    graph.add_edge(NetworkEdge::directed(1, 2, 1.0));
    let layout = HierarchicalLayout::new();
    let layers = layout.assign_layers(&graph);
    let seed = HierarchicalLayout::with_config(super::HierarchicalConfig {
        crossing_passes: 0,
        ..super::HierarchicalConfig::default()
    })
    .minimize_crossings(&graph, &layers);
    assert_eq!(layout.count_crossings(&graph, &seed), 1);
    let ordered = layout.minimize_crossings(&graph, &layers);
    assert_eq!(ordered, vec![vec![0, 1], vec![3, 2]]);
    assert_eq!(layout.count_crossings(&graph, &ordered), 0);

    layout.apply(&mut graph);
    let x = |id: usize| graph.node(id).map(|node| node.position[0]);
    assert!(x(0) < x(1));
    assert!(x(3) < x(2));
    let y = |id: usize| graph.node(id).map(|node| node.position[1]);
    assert_eq!(y(0), y(1));
    assert_eq!(y(2), y(3));
    assert!(y(0) < y(2));
}

#[test]
fn directed_k22_cannot_be_drawn_without_its_one_crossing() {
    let mut graph = graph_of(&[0, 1, 2, 3]);
    for source in [0, 1] {
        for target in [2, 3] {
            graph.add_edge(NetworkEdge::directed(source, target, 1.0));
        }
    }
    let layout = HierarchicalLayout::new();
    let layers = layout.assign_layers(&graph);
    let ordered = layout.minimize_crossings(&graph, &layers);
    assert_eq!(layout.count_crossings(&graph, &ordered), 1);
}

#[test]
fn hierarchical_layout_is_repeatable_whatever_the_hash_order() {
    // Fresh hash maps iterate in a different order every time, and the nodes
    // and edges are inserted in a different order in every other copy.
    let edges = [
        (0, 5),
        (1, 4),
        (2, 6),
        (3, 7),
        (0, 7),
        (4, 8),
        (5, 9),
        (6, 8),
        (7, 9),
    ];
    let build = |reverse: bool| {
        let mut ids: Vec<usize> = (0..10).collect();
        let mut order: Vec<usize> = (0..edges.len()).collect();
        if reverse {
            ids.reverse();
            order.reverse();
        }
        let mut graph = graph_of(&ids);
        for i in order {
            graph.add_edge(NetworkEdge::directed(edges[i].0, edges[i].1, 1.0));
        }
        graph
    };
    let positions = |graph: &NetworkGraph| -> BTreeMap<usize, [u64; 2]> {
        graph
            .nodes
            .iter()
            .map(|node| {
                (
                    node.id,
                    [node.position[0].to_bits(), node.position[1].to_bits()],
                )
            })
            .collect()
    };
    let mut reference = build(false);
    HierarchicalLayout::new().apply(&mut reference);
    let reference = positions(&reference);
    for round in 0..20 {
        let mut graph = build(round % 2 == 1);
        HierarchicalLayout::new().apply(&mut graph);
        assert_eq!(positions(&graph), reference, "round {round}");
    }
}

#[test]
fn crossing_minimisation_reduces_the_crossings_of_a_random_dag() {
    let mut rng = Lcg(404);
    let layer_sizes = [8, 10, 10, 8];
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut next = 0;
    for &size in &layer_sizes {
        members.push((next..next + size).collect());
        next += size;
    }
    let mut graph = graph_of(&(0..next).collect::<Vec<_>>());
    for pair in 0..layer_sizes.len() - 1 {
        for &upper in &members[pair] {
            for _ in 0..2 {
                let lower = members[pair + 1][rng.below(layer_sizes[pair + 1])];
                graph.add_edge(NetworkEdge::directed(upper, lower, 1.0));
            }
        }
    }
    let layout = HierarchicalLayout::new();
    let layers = layout.assign_layers(&graph);
    assert_eq!(layers.len(), graph.node_count());
    let seed = HierarchicalLayout::with_config(super::HierarchicalConfig {
        crossing_passes: 0,
        ..super::HierarchicalConfig::default()
    })
    .minimize_crossings(&graph, &layers);
    let ordered = layout.minimize_crossings(&graph, &layers);
    let before = layout.count_crossings(&graph, &seed);
    let after = layout.count_crossings(&graph, &ordered);
    assert!(after < before, "{after} crossings after, {before} before");
}

#[test]
fn hierarchical_layout_handles_cycles_and_undirected_graphs() {
    // A directed cycle is layered instead of collapsing into one layer.
    let mut cycle = graph_of(&[0, 1, 2]);
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        cycle.add_edge(NetworkEdge::directed(a, b, 1.0));
    }
    HierarchicalLayout::new().apply(&mut cycle);
    let ys: Vec<f64> = (0..3)
        .filter_map(|id| cycle.node(id).map(|node| node.position[1]))
        .collect();
    assert_eq!(ys, vec![0.0, 100.0, 200.0]);

    let mut chain = graph_of(&[0, 1, 2, 3]);
    for i in 0..3 {
        chain.add_edge(NetworkEdge::undirected(i, i + 1));
    }
    HierarchicalLayout::new().apply(&mut chain);
    let ys: Vec<f64> = (0..4)
        .filter_map(|id| chain.node(id).map(|node| node.position[1]))
        .collect();
    assert_eq!(ys, vec![0.0, 100.0, 200.0, 300.0]);
}

#[test]
fn hierarchical_layout_of_a_long_chain_takes_linear_time() {
    // Layer assignment used to scan every edge for every node, and placing the
    // nodes looked every id up in the node list: both quadratic.  With 200_000
    // nodes this would take minutes.
    let n = 200_000;
    let mut graph = NetworkGraph::new();
    for id in 0..n {
        graph.add_node(NetworkNode::new(id, String::new()));
    }
    for id in 0..n - 1 {
        graph.add_edge(NetworkEdge::directed(id, id + 1, 1.0));
    }
    HierarchicalLayout::new().apply(&mut graph);
    for id in [0, 1, n / 2, n - 1] {
        let y = graph.nodes[id].position[1];
        assert_eq!(y, id as f64 * 100.0, "node {id}");
    }
}

// ---------------------------------------------------------------------------
// F5080 - density
// ---------------------------------------------------------------------------

#[test]
fn density_of_complete_and_cyclic_graphs() {
    let mut triangle = graph_of(&[0, 1, 2]);
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        triangle.add_edge(NetworkEdge::undirected(a, b));
    }
    assert!((NetworkMetrics::new(triangle).density() - 1.0).abs() < 1e-12);

    let mut cycle = graph_of(&[0, 1, 2]);
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        cycle.add_edge(NetworkEdge::directed(a, b, 1.0));
    }
    assert!((NetworkMetrics::new(cycle).density() - 0.5).abs() < 1e-12);

    // Complete directed graph: every ordered pair.
    let mut complete = graph_of(&[0, 1, 2, 3]);
    for a in 0..4 {
        for b in 0..4 {
            if a != b {
                complete.add_edge(NetworkEdge::directed(a, b, 1.0));
            }
        }
    }
    assert!((NetworkMetrics::new(complete).density() - 1.0).abs() < 1e-12);

    // 4 nodes, one undirected edge: 2 arcs of 12.
    let mut sparse = graph_of(&[0, 1, 2, 3]);
    sparse.add_edge(NetworkEdge::undirected(0, 1));
    assert!((NetworkMetrics::new(sparse).density() - 2.0 / 12.0).abs() < 1e-12);
}

#[test]
fn density_stays_within_zero_and_one() {
    // Parallel edges, both directions of one pair, an undirected edge on top
    // of a directed one, self-loops, edges to unknown nodes.
    let mut graph = graph_of(&[0, 1, 2]);
    graph.add_edge(NetworkEdge::undirected(0, 1));
    graph.add_edge(NetworkEdge::undirected(0, 1));
    graph.add_edge(NetworkEdge::undirected(1, 0));
    graph.add_edge(NetworkEdge::directed(0, 1, 1.0));
    graph.add_edge(NetworkEdge::directed(1, 2, 1.0));
    graph.add_edge(NetworkEdge::undirected(2, 2));
    graph.add_edge(NetworkEdge::undirected(0, 9));
    // Arcs: 0->1, 1->0, 1->2 out of 3 * 2 = 6.
    let density = NetworkMetrics::new(graph).density();
    assert!((density - 0.5).abs() < 1e-12, "{density}");
}

#[test]
fn density_of_tiny_graphs_is_zero() {
    assert_eq!(NetworkMetrics::new(NetworkGraph::new()).density(), 0.0);
    assert_eq!(NetworkMetrics::new(graph_of(&[4])).density(), 0.0);
    assert_eq!(NetworkMetrics::new(graph_of(&[4, 5])).density(), 0.0);
    // A node listed twice is one node.
    let mut twice = graph_of(&[4, 4]);
    twice.add_edge(NetworkEdge::undirected(4, 4));
    assert_eq!(NetworkMetrics::new(twice).density(), 0.0);
}

// ---------------------------------------------------------------------------
// F5087 - NetworkAnimation
// ---------------------------------------------------------------------------

fn animation() -> NetworkAnimation {
    NetworkAnimation::new(graph_of(&[0, 1, 2]), 30.0, 30.0)
}

fn mv(node_id: usize, target: [f64; 2], start_time: f64, duration: f64) -> NetworkEvent {
    NetworkEvent::NodeMove {
        node_id,
        target_position: target,
        start_time,
        duration,
    }
}

#[test]
fn events_are_applied_in_time_order_not_insertion_order() {
    let mut anim = animation();
    // Added in the wrong order on purpose.
    anim.add_event(NetworkEvent::NodeDisappear {
        node_id: 0,
        time: 2.0,
    });
    anim.add_event(NetworkEvent::NodeAppear {
        node_id: 0,
        time: 1.0,
    });
    assert!(!anim.frame_at(0.5).active_nodes.contains(&0));
    assert!(anim.frame_at(1.5).active_nodes.contains(&0));
    assert!(!anim.frame_at(3.0).active_nodes.contains(&0));
    // The same for edges.
    anim.add_event(NetworkEvent::EdgeDisappear {
        source: 0,
        target: 1,
        time: 6.0,
    });
    anim.add_event(NetworkEvent::EdgeAppear {
        source: 0,
        target: 1,
        time: 5.0,
    });
    assert!(anim.frame_at(5.5).active_edges.contains(&(0, 1)));
    assert!(anim.frame_at(7.0).active_edges.is_empty());
}

#[test]
fn simultaneous_events_keep_their_insertion_order() {
    let appear = NetworkEvent::NodeAppear {
        node_id: 2,
        time: 4.0,
    };
    let disappear = NetworkEvent::NodeDisappear {
        node_id: 2,
        time: 4.0,
    };
    let mut gone = animation();
    gone.add_event(appear.clone());
    gone.add_event(disappear.clone());
    assert!(gone.frame_at(4.0).active_nodes.is_empty());
    let mut present = animation();
    present.add_event(disappear);
    present.add_event(appear);
    assert!(present.frame_at(4.0).active_nodes.contains(&2));
}

#[test]
fn an_event_happens_from_its_time_on() {
    let mut anim = animation();
    anim.add_event(NetworkEvent::NodeAppear {
        node_id: 1,
        time: 2.0,
    });
    assert!(!anim.frame_at(1.999).active_nodes.contains(&1));
    assert!(anim.frame_at(2.0).active_nodes.contains(&1));
    assert!(anim.frame_at(1.0e9).active_nodes.contains(&1));
    // An event at a time that is not a number never happens.
    anim.add_event(NetworkEvent::NodeAppear {
        node_id: 2,
        time: f64::NAN,
    });
    assert!(!anim.frame_at(1.0e9).active_nodes.contains(&2));
    assert!(!anim.frame_at(f64::INFINITY).active_nodes.contains(&2));
}

#[test]
fn active_edges_are_listed_once_in_order_of_appearance() {
    let appear = |source: usize, target: usize, time: f64| NetworkEvent::EdgeAppear {
        source,
        target,
        time,
    };
    let mut anim = animation();
    // (0, 1) appears again after (1, 2); `Vec::dedup` only merges neighbours.
    anim.add_event(appear(0, 1, 1.0));
    anim.add_event(appear(1, 2, 2.0));
    anim.add_event(appear(0, 1, 3.0));
    assert_eq!(anim.frame_at(4.0).active_edges, vec![(0, 1), (1, 2)]);
    // Disappearing removes it, appearing again puts it at the end.
    anim.add_event(NetworkEvent::EdgeDisappear {
        source: 0,
        target: 1,
        time: 5.0,
    });
    assert_eq!(anim.frame_at(6.0).active_edges, vec![(1, 2)]);
    anim.add_event(appear(0, 1, 7.0));
    assert_eq!(anim.frame_at(8.0).active_edges, vec![(1, 2), (0, 1)]);
    // The pair has to match exactly.
    anim.add_event(NetworkEvent::EdgeDisappear {
        source: 2,
        target: 1,
        time: 9.0,
    });
    assert_eq!(anim.frame_at(10.0).active_edges, vec![(1, 2), (0, 1)]);
}

#[test]
fn chained_moves_are_continuous_whatever_the_insertion_order() {
    for reversed in [false, true] {
        let mut anim = animation();
        let (first, second) = (
            mv(0, [10.0, 0.0], 0.0, 10.0),
            mv(0, [10.0, 10.0], 10.0, 10.0),
        );
        if reversed {
            anim.add_event(second);
            anim.add_event(first);
        } else {
            anim.add_event(first);
            anim.add_event(second);
        }
        let at = |t: f64| anim.frame_at(t).node_positions[&0];
        assert_eq!(at(0.0), [0.0, 0.0]);
        assert_eq!(at(5.0), [5.0, 0.0]);
        // The second move starts where the first one ended (it used to snap
        // back to the base position).
        assert_eq!(at(10.0), [10.0, 0.0]);
        assert_eq!(at(15.0), [10.0, 5.0]);
        assert_eq!(at(20.0), [10.0, 10.0]);
        assert_eq!(at(25.0), [10.0, 10.0]);
        // No jump anywhere near the hand-over.
        let before = at(10.0 - 1e-6);
        let after = at(10.0 + 1e-6);
        assert!((before[0] - after[0]).abs() < 1e-4 && (before[1] - after[1]).abs() < 1e-4);
    }
}

#[test]
fn overlapping_moves_blend_without_a_jump() {
    let mut anim = animation();
    anim.add_event(mv(0, [10.0, 0.0], 0.0, 10.0));
    anim.add_event(mv(0, [10.0, 10.0], 5.0, 10.0));
    let at = |t: f64| anim.frame_at(t).node_positions[&0];
    // The second move begins at the position the first one has reached.
    assert_eq!(at(5.0), [5.0, 0.0]);
    let just_after = at(5.0 + 1e-6);
    assert!((just_after[0] - 5.0).abs() < 1e-4 && just_after[1].abs() < 1e-4);
    // 7.5: first move 75 % of the way, second move 25 %.
    let mid = at(7.5);
    assert!((mid[0] - 8.125).abs() < 1e-12 && (mid[1] - 2.5).abs() < 1e-12);
    assert_eq!(at(15.0), [10.0, 10.0]);
}

#[test]
fn moves_of_no_duration_jump_and_other_nodes_stay_put() {
    let mut anim = NetworkAnimation::new(
        {
            let mut graph = graph_of(&[0, 1]);
            graph.nodes[1].position = [3.0, 4.0];
            graph
        },
        10.0,
        30.0,
    );
    anim.add_event(mv(0, [7.0, 7.0], 2.0, 0.0));
    anim.add_event(mv(0, [8.0, 8.0], 4.0, -1.0));
    anim.add_event(mv(0, [9.0, 9.0], 6.0, f64::NAN));
    let frame = |t: f64| anim.frame_at(t).node_positions;
    assert_eq!(frame(1.999)[&0], [0.0, 0.0]);
    assert_eq!(frame(2.0)[&0], [7.0, 7.0]);
    assert_eq!(frame(4.0)[&0], [8.0, 8.0]);
    assert_eq!(frame(6.0)[&0], [9.0, 9.0]);
    for t in [0.0, 3.0, 8.0] {
        assert_eq!(frame(t)[&1], [3.0, 4.0]);
    }
    // A move of a node that is not in the base graph is ignored.
    anim.add_event(mv(55, [1.0, 1.0], 0.0, 1.0));
    assert!(!anim.frame_at(5.0).node_positions.contains_key(&55));
}

#[test]
fn sampled_frames_equal_frames_at_the_same_times() {
    let mut anim = animation();
    anim.add_event(NetworkEvent::EdgeAppear {
        source: 1,
        target: 2,
        time: 12.0,
    });
    anim.add_event(mv(2, [4.0, 4.0], 20.0, 5.0));
    anim.add_event(mv(2, [-4.0, 8.0], 5.0, 10.0));
    anim.add_event(NetworkEvent::NodeAppear {
        node_id: 1,
        time: 9.0,
    });
    let frames = anim.sample_frames(7);
    assert_eq!(frames.len(), 7);
    for (i, frame) in frames.iter().enumerate() {
        let t = 30.0 * i as f64 / 6.0;
        let direct = anim.frame_at(t);
        assert_eq!(frame.time, direct.time);
        assert_eq!(frame.active_nodes, direct.active_nodes);
        assert_eq!(frame.active_edges, direct.active_edges);
        assert_eq!(frame.node_positions, direct.node_positions);
    }
    // One frame is the frame at time 0.
    let single = anim.sample_frames(1);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].time, 0.0);
}

#[test]
fn interpolation_follows_the_ordered_timeline() {
    let mut anim = animation();
    anim.add_event(mv(0, [10.0, 10.0], 10.0, 10.0));
    anim.add_event(mv(0, [10.0, 0.0], 0.0, 10.0));
    // Frames at 10 and 20 are [10, 0] and [10, 10]; halfway is [10, 5].
    let halfway = anim.interpolate_position(0, 10.0, 20.0, 0.5);
    assert_eq!(halfway, [10.0, 5.0]);
    // A node that does not exist falls back to the origin.
    assert_eq!(anim.interpolate_position(9, 0.0, 1.0, 0.5), [0.0, 0.0]);
}

#[test]
fn every_event_reports_its_time() {
    let events = [
        (
            NetworkEvent::NodeAppear {
                node_id: 0,
                time: 1.5,
            },
            1.5,
        ),
        (
            NetworkEvent::NodeDisappear {
                node_id: 0,
                time: 2.5,
            },
            2.5,
        ),
        (
            NetworkEvent::EdgeAppear {
                source: 0,
                target: 1,
                time: 3.5,
            },
            3.5,
        ),
        (
            NetworkEvent::EdgeDisappear {
                source: 0,
                target: 1,
                time: 4.5,
            },
            4.5,
        ),
        (mv(0, [0.0, 0.0], 5.5, 1.0), 5.5),
    ];
    for (event, time) in events {
        assert_eq!(event.time(), time);
    }
}

// ---------------------------------------------------------------------------
// Label propagation on the shared community model
// ---------------------------------------------------------------------------

#[test]
fn label_propagation_uses_weights_and_ignores_directions() {
    // 0 and 3 are tied by a heavy edge, 1 and 2 by a heavier one; node 3 also
    // touches 1 and 2 weakly.  Directions must not matter.
    let mut graph = graph_of(&[0, 1, 2, 3]);
    let mut link = |a: usize, b: usize, weight: f64, directed: bool| {
        let mut edge = if directed {
            NetworkEdge::directed(a, b, weight)
        } else {
            NetworkEdge::undirected(a, b)
        };
        edge.weight = weight;
        graph.add_edge(edge);
    };
    link(1, 2, 10.0, true);
    link(3, 0, 3.0, true);
    link(1, 3, 1.0, false);
    link(2, 3, 1.0, false);
    let result = CommunityDetection::new(graph.clone()).label_propagation(20);
    assert_eq!(result.membership[&0], result.membership[&3]);
    assert_eq!(result.membership[&1], result.membership[&2]);
    assert_ne!(result.membership[&0], result.membership[&1]);
    let again = CommunityDetection::new(graph).modularity(&result.membership);
    assert!((result.modularity - again).abs() < 1e-12);
}

#[test]
fn label_propagation_does_not_depend_on_the_insertion_order() {
    let edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (3, 5),
        (6, 7),
        (7, 8),
        (6, 8),
    ];
    let build = |ids: &[usize], order: &[usize]| {
        let mut graph = graph_of(ids);
        for &i in order {
            graph.add_edge(NetworkEdge::undirected(edges[i].0, edges[i].1));
        }
        graph
    };
    let ids: Vec<usize> = (0..9).collect();
    let forward: Vec<usize> = (0..edges.len()).collect();
    let reversed_ids: Vec<usize> = ids.iter().rev().copied().collect();
    let backward: Vec<usize> = forward.iter().rev().copied().collect();
    let first = CommunityDetection::new(build(&ids, &forward)).label_propagation(30);
    let second = CommunityDetection::new(build(&reversed_ids, &backward)).label_propagation(30);
    // Same partition of the ids (the numbering depends on the node order).
    let partition = |membership: &HashMap<usize, usize>| {
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (&id, &label) in membership {
            groups.entry(label).or_default().push(id);
        }
        let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
        for group in &mut groups {
            group.sort_unstable();
        }
        groups.sort();
        groups
    };
    assert_eq!(partition(&first.membership), partition(&second.membership));
    assert!((first.modularity - second.modularity).abs() < 1e-12);
}

#[test]
fn label_propagation_separates_disconnected_cliques() {
    let mut graph = graph_of(&(0..9).collect::<Vec<_>>());
    for group in [[0, 1, 2], [3, 4, 5], [6, 7, 8]] {
        for (i, &a) in group.iter().enumerate() {
            for &b in &group[i + 1..] {
                graph.add_edge(NetworkEdge::undirected(a, b));
            }
        }
    }
    let result = CommunityDetection::new(graph).label_propagation(20);
    assert_eq!(result.num_communities(), 3);
    for group in [[0, 1, 2], [3, 4, 5], [6, 7, 8]] {
        assert_eq!(result.membership[&group[0]], result.membership[&group[1]]);
        assert_eq!(result.membership[&group[1]], result.membership[&group[2]]);
    }
    // Q = 3 * (3/9 - (6/18)^2) = 2/3.
    assert!((result.modularity - 2.0 / 3.0).abs() < 1e-12);
    // Labels are numbered in order of first appearance.
    assert_eq!(result.membership[&0], 0);
    assert_eq!(result.membership[&3], 1);
    assert_eq!(result.membership[&6], 2);
}
