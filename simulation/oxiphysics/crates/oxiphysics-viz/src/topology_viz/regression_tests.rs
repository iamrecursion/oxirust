// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Reeb graph, contour tree and Morse-Smale
//! manifold constructions.

use super::*;
use std::f64::consts::PI;

/// Mesh as (vertices with scalar, triangles), as taken by
/// [`ReebGraph::from_triangle_mesh`].
type ScalarMesh = (Vec<(f64, f64, f64, f64)>, Vec<(usize, usize, usize)>);

/// Triangulated torus (closed surface) on an `nu × nv` parameter grid with
/// scalar = tilted x coordinate (the "standing torus" height function).
fn torus_mesh(nu: usize, nv: usize) -> ScalarMesh {
    let (big_r, small_r) = (2.0, 0.7);
    let mut vertices = Vec::new();
    for i in 0..nu {
        let u = 2.0 * PI * i as f64 / nu as f64;
        for j in 0..nv {
            let v = 2.0 * PI * j as f64 / nv as f64;
            let x = (big_r + small_r * v.cos()) * u.cos();
            let y = (big_r + small_r * v.cos()) * u.sin();
            let z = small_r * v.sin();
            vertices.push((x, y, z, x + 1e-3 * y + 1e-4 * z));
        }
    }
    let idx = |i: usize, j: usize| (i % nu) * nv + (j % nv);
    let mut triangles = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let a = idx(i, j);
            let b = idx(i + 1, j);
            let c = idx(i + 1, j + 1);
            let d = idx(i, j + 1);
            triangles.push((a, b, c));
            triangles.push((a, c, d));
        }
    }
    (vertices, triangles)
}

/// Polar disk mesh (centre + `rings` rings of `sectors` vertices) carrying a
/// two-peak terrain that vanishes exactly on the boundary circle.
fn two_peak_disk(rings: usize, sectors: usize) -> ScalarMesh {
    let height = |x: f64, y: f64| {
        let r2 = x * x + y * y;
        let bump = |cx: f64| (-((x - cx).powi(2) + y * y) / (2.0 * 0.25 * 0.25)).exp();
        (bump(-0.45) + bump(0.45)) * (1.0 - r2)
    };
    let mut vertices = vec![(0.0, 0.0, 0.0, height(0.0, 0.0))];
    for ring in 1..=rings {
        let radius = ring as f64 / rings as f64;
        for s in 0..sectors {
            let theta = 2.0 * PI * s as f64 / sectors as f64;
            let (x, y) = (radius * theta.cos(), radius * theta.sin());
            let f = if ring == rings { 0.0 } else { height(x, y) };
            vertices.push((x, y, 0.0, f));
        }
    }
    let ring_idx = |ring: usize, s: usize| 1 + (ring - 1) * sectors + (s % sectors);
    let mut triangles = Vec::new();
    for s in 0..sectors {
        triangles.push((0, ring_idx(1, s), ring_idx(1, s + 1)));
    }
    for ring in 1..rings {
        for s in 0..sectors {
            let a = ring_idx(ring, s);
            let b = ring_idx(ring, s + 1);
            let c = ring_idx(ring + 1, s + 1);
            let d = ring_idx(ring + 1, s);
            triangles.push((a, d, c));
            triangles.push((a, c, b));
        }
    }
    (vertices, triangles)
}

/// Vertex adjacency (1-skeleton) of a triangle list.
fn mesh_adjacency(n: usize, triangles: &[(usize, usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(a, b, c) in triangles {
        for (p, q) in [(a, b), (b, c), (a, c)] {
            adj[p].push(q);
            adj[q].push(p);
        }
    }
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }
    adj
}

fn count_type(rg: &ReebGraph, ct: CriticalPointType) -> usize {
    rg.nodes_of_type(ct).len()
}

// --- Reeb graph (F5221) ---

#[test]
fn reeb_torus_height_function_has_one_loop() {
    let (vertices, triangles) = torus_mesh(24, 12);
    let rg = ReebGraph::from_triangle_mesh(&vertices, &triangles);
    assert_eq!(rg.betti_1(), 1, "torus Reeb graph must contain one loop");
    assert_eq!(rg.connected_components(), 1);
    assert_eq!(count_type(&rg, CriticalPointType::Minimum), 1);
    assert_eq!(count_type(&rg, CriticalPointType::Maximum), 1);
    assert_eq!(count_type(&rg, CriticalPointType::Saddle), 2);
    assert_eq!(rg.node_count(), 4);
    assert_eq!(rg.edge_count(), 4);
    // The split saddle has one arc below and two above.
    let saddles = rg.nodes_of_type(CriticalPointType::Saddle);
    for s in saddles {
        assert_eq!(rg.degree(s.id), 3);
    }
}

#[test]
fn reeb_two_peak_terrain_has_two_maxima_and_one_saddle() {
    let (vertices, triangles) = two_peak_disk(16, 48);
    let rg = ReebGraph::from_triangle_mesh(&vertices, &triangles);
    assert_eq!(count_type(&rg, CriticalPointType::Minimum), 1);
    assert_eq!(count_type(&rg, CriticalPointType::Maximum), 2);
    assert_eq!(count_type(&rg, CriticalPointType::Saddle), 1);
    assert_eq!(rg.edge_count(), 3);
    assert_eq!(rg.betti_1(), 0);
    // Both maxima sit near the bump centres (±0.45, 0).
    let mut max_x: Vec<f64> = rg
        .nodes_of_type(CriticalPointType::Maximum)
        .iter()
        .filter_map(|nd| nd.position.map(|p| p[0]))
        .collect();
    max_x.sort_by(f64::total_cmp);
    assert_eq!(max_x.len(), 2);
    assert!(max_x[0] < -0.2 && max_x[1] > 0.2, "maxima at {max_x:?}");
}

#[test]
fn reeb_graph_matches_contour_tree_on_simply_connected_mesh() {
    let (vertices, triangles) = two_peak_disk(10, 32);
    let rg = ReebGraph::from_triangle_mesh(&vertices, &triangles);
    let values: Vec<f64> = vertices.iter().map(|v| v.3).collect();
    let adj = mesh_adjacency(vertices.len(), &triangles);
    let ct = ContourTree::build(&values, &adj);
    assert_eq!(rg.node_count(), ct.node_count());
    assert_eq!(rg.edge_count(), ct.arc_count());
    let mut rg_vals: Vec<f64> = rg.nodes.iter().map(|nd| nd.value).collect();
    let mut ct_vals: Vec<f64> = ct.nodes.iter().map(|nd| nd.value).collect();
    rg_vals.sort_by(f64::total_cmp);
    ct_vals.sort_by(f64::total_cmp);
    assert_eq!(rg_vals, ct_vals);
}

#[test]
fn reeb_graph_skips_out_of_range_triangles() {
    let vertices = vec![
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
    ];
    let triangles = vec![(0, 1, 2), (0, 1, 7), (9, 9, 9)];
    let rg = ReebGraph::from_triangle_mesh(&vertices, &triangles);
    assert_eq!(rg.node_count(), 2);
    assert_eq!(rg.edge_count(), 1);
    assert_eq!(rg.edges[0].persistence, 2.0);
}

#[test]
fn reeb_graph_isolated_vertices_are_components() {
    let vertices = vec![
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (5.0, 5.0, 5.0, 0.5),
    ];
    let rg = ReebGraph::from_triangle_mesh(&vertices, &[(0, 1, 2)]);
    assert_eq!(rg.connected_components(), 2);
    assert_eq!(rg.node_count(), 3);
}

// --- Morse-Smale manifolds (F5222) ---

#[test]
fn ascending_manifolds_map_to_terminal_maxima() {
    let values = [0.0, 2.0, 1.0, 3.0, 0.0];
    let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
    let mut ms = MorseSmaleComplex::new();
    // Critical-point ids must not influence the vertex assignment.
    ms.add_critical_point([0.0; 3], 0.0, 0);
    ms.add_critical_point([0.0; 3], 3.0, 1);
    ms.compute_ascending_manifolds(&values, &adj);
    let asc: Vec<usize> = (0..5).map(|v| ms.ascending_manifold[&v]).collect();
    assert_eq!(asc, vec![1, 1, 3, 3, 3]);
    ms.compute_descending_manifolds(&values, &adj);
    let desc: Vec<usize> = (0..5).map(|v| ms.descending_manifold[&v]).collect();
    assert_eq!(desc, vec![0, 0, 2, 4, 4]);
}

#[test]
fn manifolds_follow_steepest_neighbour_and_ignore_bad_indices() {
    // Star: centre 0 (value 1) with leaves of values 2, 5, 0, 3.
    let values = [1.0, 2.0, 5.0, 0.0, 3.0];
    let adj = vec![vec![1, 2, 3, 4, 42], vec![0], vec![0], vec![0], vec![0]];
    let mut ms = MorseSmaleComplex::new();
    ms.compute_ascending_manifolds(&values, &adj);
    assert_eq!(ms.ascending_manifold[&0], 2);
    // Leaf 1 has only a lower neighbour: it is a local maximum itself.
    assert_eq!(ms.ascending_manifold[&1], 1);
    assert_eq!(ms.ascending_manifold[&3], 2);
    assert_eq!(ms.ascending_manifold[&4], 4);
    assert_eq!(ms.ascending_manifold.len(), 5);
    ms.compute_descending_manifolds(&values, &adj);
    assert_eq!(ms.descending_manifold[&0], 3);
    assert_eq!(ms.descending_manifold[&2], 3);
    assert_eq!(ms.descending_manifold[&3], 3);
}

#[test]
fn manifolds_break_plateau_ties_by_index() {
    // Flat plateau: ties broken by index, so the highest index is the max.
    let values = [1.0, 1.0, 1.0];
    let adj = vec![vec![1], vec![0, 2], vec![1]];
    let mut ms = MorseSmaleComplex::new();
    ms.compute_ascending_manifolds(&values, &adj);
    let asc: Vec<usize> = (0..3).map(|v| ms.ascending_manifold[&v]).collect();
    assert_eq!(asc, vec![2, 2, 2]);
    ms.compute_descending_manifolds(&values, &adj);
    let desc: Vec<usize> = (0..3).map(|v| ms.descending_manifold[&v]).collect();
    assert_eq!(desc, vec![0, 0, 0]);
}

// --- Contour tree (F5223) ---

fn arc_vertex_pairs(ct: &ContourTree) -> Vec<(usize, usize)> {
    let mut pairs: Vec<(usize, usize)> = ct
        .arcs
        .iter()
        .map(|a| (ct.nodes[a.lower].vertex, ct.nodes[a.upper].vertex))
        .collect();
    pairs.sort_unstable();
    pairs
}

#[test]
fn contour_tree_of_zigzag_path_is_the_path() {
    // Values along a path graph: the contour tree of a function on a path is
    // the path itself (not the value-sorted chain 0-4-2-1-3).
    let values = [0.0, 3.0, 1.0, 4.0, 0.0];
    let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
    let ct = ContourTree::build(&values, &adj);
    assert_eq!(ct.node_count(), 5);
    assert_eq!(ct.arc_count(), 4);
    assert_eq!(arc_vertex_pairs(&ct), vec![(0, 1), (2, 1), (2, 3), (4, 3)]);
    let leaf_vertices: Vec<usize> = ct.leaves().iter().map(|&i| ct.nodes[i].vertex).collect();
    assert_eq!(leaf_vertices, vec![0, 4]);
    for a in &ct.arcs {
        assert!(ct.nodes[a.lower].value <= ct.nodes[a.upper].value);
    }
}

#[test]
fn contour_tree_counts_regular_vertices() {
    // Y-shaped tree: 0(0) - 1(1) - 2(2) = junction, branches 2-3(3)-4(5)
    // and 2-5(4).  Vertex 2 is a split saddle; 1 and 3 are regular.
    let values = [0.0, 1.0, 2.0, 3.0, 5.0, 4.0];
    let adj = vec![
        vec![1],
        vec![0, 2],
        vec![1, 3, 5],
        vec![2, 4],
        vec![3],
        vec![2],
    ];
    let ct = ContourTree::build(&values, &adj);
    assert_eq!(ct.node_count(), 4);
    assert_eq!(ct.arc_count(), 3);
    assert_eq!(arc_vertex_pairs(&ct), vec![(0, 2), (2, 4), (2, 5)]);
    let regular_total: usize = ct.arcs.iter().map(|a| a.regular_count).sum();
    assert_eq!(regular_total, 2);
    let to_four = ct
        .arcs
        .iter()
        .find(|a| ct.nodes[a.upper].vertex == 4)
        .expect("arc to vertex 4");
    assert_eq!(to_four.regular_count, 1);
}

#[test]
fn contour_tree_two_peak_grid() {
    let (vertices, triangles) = two_peak_disk(12, 40);
    let values: Vec<f64> = vertices.iter().map(|v| v.3).collect();
    let adj = mesh_adjacency(vertices.len(), &triangles);
    let ct = ContourTree::build(&values, &adj);
    let count = |t: CriticalPointType| ct.nodes.iter().filter(|n| n.critical_type == t).count();
    assert_eq!(count(CriticalPointType::Minimum), 1);
    assert_eq!(count(CriticalPointType::Maximum), 2);
    assert_eq!(count(CriticalPointType::Saddle), 1);
    assert_eq!(ct.arc_count(), 3);
    assert_eq!(ct.leaves().len(), 3);
    let regular_total: usize = ct.arcs.iter().map(|a| a.regular_count).sum();
    assert_eq!(regular_total + ct.node_count(), vertices.len());
}

#[test]
fn contour_tree_rejects_mismatched_adjacency_and_bad_indices() {
    let ct = ContourTree::build(&[0.0, 1.0], &[vec![1]]);
    assert_eq!(ct.node_count(), 0);
    // Out-of-range neighbours and self loops are ignored; one-sided adjacency
    // is symmetrised.
    let ct = ContourTree::build(&[0.0, 1.0, 2.0], &[vec![1, 0, 99], vec![2], vec![]]);
    assert_eq!(ct.node_count(), 2);
    assert_eq!(ct.arc_count(), 1);
    assert_eq!(ct.arcs[0].regular_count, 1);
}
