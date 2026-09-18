// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of [`super::StraightSkeleton`]: the skeleton arcs and the mitred
//! offset rings of polygons with a known straight skeleton (squares,
//! rectangles, regular polygons, an L, a plus, a dumbbell), invariants of the
//! skeleton of random polygons in general position, the offsets against a
//! half-plane-intersection oracle for convex polygons, and invalid input.

use super::StraightSkeleton;
use crate::error::Error;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

type P2 = [f64; 2];

const TOL: f64 = 1e-9;

fn dist(a: P2, b: P2) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn signed_area(ring: &[P2]) -> f64 {
    let n = ring.len();
    0.5 * (0..n)
        .map(|i| cross(ring[i], ring[(i + 1) % n]))
        .sum::<f64>()
}

fn perimeter(ring: &[P2]) -> f64 {
    let n = ring.len();
    (0..n).map(|i| dist(ring[i], ring[(i + 1) % n])).sum()
}

/// Deterministic xorshift64* generator.
struct XorShift(u64);

impl XorShift {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn rectangle(w: f64, h: f64) -> Vec<P2> {
    vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

fn l_shape() -> Vec<P2> {
    vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ]
}

/// Plus sign with arms of width 2 reaching 5 from the centre.
fn plus_shape() -> Vec<P2> {
    vec![
        [5.0, -1.0],
        [5.0, 1.0],
        [1.0, 1.0],
        [1.0, 5.0],
        [-1.0, 5.0],
        [-1.0, 1.0],
        [-5.0, 1.0],
        [-5.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -5.0],
        [1.0, -5.0],
        [1.0, -1.0],
    ]
}

/// Two 4 x 4 squares joined by a neck of width 0.4 (it collapses at 0.2).
fn dumbbell() -> Vec<P2> {
    vec![
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.8],
        [6.0, 1.8],
        [6.0, 0.0],
        [10.0, 0.0],
        [10.0, 4.0],
        [6.0, 4.0],
        [6.0, 2.2],
        [4.0, 2.2],
        [4.0, 4.0],
        [0.0, 4.0],
    ]
}

fn regular_polygon(n: usize, radius: f64) -> Vec<P2> {
    (0..n)
        .map(|i| {
            let a = 2.0 * PI * i as f64 / n as f64;
            [radius * a.cos(), radius * a.sin()]
        })
        .collect()
}

/// Assert that `found` and `expected` are the same set of segments, in either
/// orientation, up to `TOL`.
fn assert_same_arcs(found: &[(P2, P2)], expected: &[(P2, P2)]) {
    assert_eq!(
        found.len(),
        expected.len(),
        "arc count: found {found:?}, expected {expected:?}"
    );
    let mut used = vec![false; found.len()];
    for &(a, b) in expected {
        let hit = found.iter().enumerate().position(|(i, &(c, d))| {
            !used[i]
                && ((dist(a, c) < TOL && dist(b, d) < TOL)
                    || (dist(a, d) < TOL && dist(b, c) < TOL))
        });
        match hit {
            Some(i) => used[i] = true,
            None => panic!("missing arc {a:?}-{b:?} in {found:?}"),
        }
    }
}

fn node_index(graph: &super::StraightSkeletonGraph, at: P2) -> usize {
    graph
        .nodes
        .iter()
        .position(|n| dist(n.position, at) < TOL)
        .unwrap_or_else(|| panic!("no skeleton node at {at:?}"))
}

// ── Skeleton arcs of polygons with a known skeleton ─────────────────────────

#[test]
fn square_skeleton_has_four_arcs_meeting_at_the_centre() {
    let sk = StraightSkeleton::new(rectangle(1.0, 1.0));
    let c = [0.5, 0.5];
    assert_same_arcs(
        &sk.skeleton_edges(),
        &[
            ([0.0, 0.0], c),
            ([1.0, 0.0], c),
            ([1.0, 1.0], c),
            ([0.0, 1.0], c),
        ],
    );
    let graph = sk.skeleton_graph().expect("square");
    assert_eq!(graph.nodes.len(), 5);
    let centre = node_index(&graph, c);
    assert!((graph.nodes[centre].time - 0.5).abs() < TOL);
    assert_eq!(graph.degrees()[centre], 4);
}

#[test]
fn rectangle_skeleton_has_four_corner_arcs_and_a_ridge() {
    // 2 x 1: the corners meet at (0.5, 0.5) and (1.5, 0.5), joined by a ridge
    // of length 1 (the old centroid spokes ended at (1, 0.5)).
    let sk = StraightSkeleton::new(rectangle(2.0, 1.0));
    let (l, r) = ([0.5, 0.5], [1.5, 0.5]);
    assert_same_arcs(
        &sk.skeleton_edges(),
        &[
            ([0.0, 0.0], l),
            ([0.0, 1.0], l),
            ([2.0, 0.0], r),
            ([2.0, 1.0], r),
            (l, r),
        ],
    );
    let graph = sk.skeleton_graph().expect("rectangle");
    for at in [l, r] {
        let i = node_index(&graph, at);
        assert!((graph.nodes[i].time - 0.5).abs() < TOL);
        assert_eq!(graph.degrees()[i], 3);
    }
    // The same for a clockwise polygon and for either side being longer.
    let reversed: Vec<P2> = rectangle(2.0, 1.0).into_iter().rev().collect();
    assert_eq!(StraightSkeleton::new(reversed).skeleton_edges().len(), 5);
    let tall = StraightSkeleton::new(rectangle(1.0, 3.0)).skeleton_edges();
    assert_same_arcs(
        &tall,
        &[
            ([0.0, 0.0], [0.5, 0.5]),
            ([1.0, 0.0], [0.5, 0.5]),
            ([0.0, 3.0], [0.5, 2.5]),
            ([1.0, 3.0], [0.5, 2.5]),
            ([0.5, 0.5], [0.5, 2.5]),
        ],
    );
}

#[test]
fn collinear_vertex_gets_its_own_arc_to_the_ridge() {
    // A vertex in the middle of the bottom edge of a 2 x 1 rectangle: its
    // bisector arc runs straight up to the ridge.
    let poly = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
    let graph = StraightSkeleton::new(poly)
        .skeleton_graph()
        .expect("polygon");
    assert_same_arcs(
        &graph.edges(),
        &[
            ([0.0, 0.0], [0.5, 0.5]),
            ([0.0, 1.0], [0.5, 0.5]),
            ([2.0, 0.0], [1.5, 0.5]),
            ([2.0, 1.0], [1.5, 0.5]),
            ([0.5, 0.5], [1.0, 0.5]),
            ([1.0, 0.5], [1.5, 0.5]),
            ([1.0, 0.0], [1.0, 0.5]),
        ],
    );
    assert!(graph.nodes[node_index(&graph, [1.0, 0.5])].time > 0.499);
}

#[test]
fn equilateral_triangle_skeleton_meets_at_the_incentre() {
    let tri = regular_polygon(3, 1.0);
    let graph = StraightSkeleton::new(tri.clone())
        .skeleton_graph()
        .expect("triangle");
    assert_eq!(graph.arcs.len(), 3);
    let centre = node_index(&graph, [0.0, 0.0]);
    assert_eq!(graph.degrees()[centre], 3);
    // Inradius of a triangle with circumradius 1 is 1/2.
    assert!((graph.nodes[centre].time - 0.5).abs() < TOL);
}

#[test]
fn regular_polygons_collapse_to_one_point_at_their_inradius() {
    for n in 3..=14 {
        let radius = 2.5;
        let graph = StraightSkeleton::new(regular_polygon(n, radius))
            .skeleton_graph()
            .expect("regular polygon");
        assert_eq!(graph.arcs.len(), n, "n = {n}");
        assert_eq!(graph.nodes.len(), n + 1, "n = {n}");
        let centre = node_index(&graph, [0.0, 0.0]);
        assert_eq!(graph.degrees()[centre], n, "n = {n}");
        let inradius = radius * (PI / n as f64).cos();
        assert!(
            (graph.nodes[centre].time - inradius).abs() < 1e-8,
            "n = {n}: {} vs {inradius}",
            graph.nodes[centre].time
        );
    }
}

#[test]
fn l_shape_skeleton_has_eight_arcs() {
    let graph = StraightSkeleton::new(l_shape())
        .skeleton_graph()
        .expect("L shape");
    let (a, b, c) = ([0.5, 0.5], [1.5, 0.5], [0.5, 1.5]);
    assert_same_arcs(
        &graph.edges(),
        &[
            ([0.0, 0.0], a),
            ([1.0, 1.0], a),
            ([2.0, 0.0], b),
            ([2.0, 1.0], b),
            ([1.0, 2.0], c),
            ([0.0, 2.0], c),
            (a, b),
            (a, c),
        ],
    );
    let degrees = graph.degrees();
    assert_eq!(degrees[node_index(&graph, a)], 4);
    assert_eq!(degrees[node_index(&graph, b)], 3);
    assert_eq!(degrees[node_index(&graph, c)], 3);
    for at in [a, b, c] {
        assert!((graph.nodes[node_index(&graph, at)].time - 0.5).abs() < TOL);
    }
}

#[test]
fn plus_shape_skeleton_has_a_node_of_degree_eight_at_the_centre() {
    let graph = StraightSkeleton::new(plus_shape())
        .skeleton_graph()
        .expect("plus");
    // 8 cap corners + 4 reflex corners + 4 arm ridges.
    assert_eq!(graph.arcs.len(), 16);
    let centre = node_index(&graph, [0.0, 0.0]);
    assert_eq!(graph.degrees()[centre], 8);
    assert!((graph.nodes[centre].time - 1.0).abs() < TOL);
    for tip in [[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]] {
        let i = node_index(&graph, tip);
        assert_eq!(graph.degrees()[i], 3, "{tip:?}");
        assert!((graph.nodes[i].time - 1.0).abs() < TOL);
    }
}

#[test]
fn dumbbell_neck_collapses_into_a_ridge_between_the_squares() {
    let poly = dumbbell();
    let graph = StraightSkeleton::new(poly)
        .skeleton_graph()
        .expect("dumbbell");
    // The reflex corners meet at the neck mouths at time 0.2 and the neck's
    // ridge joins the mouths.
    let (left, right) = ([3.8, 2.0], [6.2, 2.0]);
    let (li, ri) = (node_index(&graph, left), node_index(&graph, right));
    assert!((graph.nodes[li].time - 0.2).abs() < TOL);
    assert!((graph.nodes[ri].time - 0.2).abs() < TOL);
    assert!(
        graph
            .arcs
            .iter()
            .any(|a| (a[0] == li && a[1] == ri) || (a[0] == ri && a[1] == li))
    );
    // Both squares are bodies of a general shape: the centre of each is a
    // skeleton node at its inradius 2.
    for centre in [[2.0, 2.0], [8.0, 2.0]] {
        let i = graph
            .nodes
            .iter()
            .position(|n| dist(n.position, centre) < 1e-6)
            .unwrap_or_else(|| panic!("no node near {centre:?}"));
        assert!((graph.nodes[i].time - 2.0).abs() < 1e-6);
    }
}

#[test]
fn skeleton_does_not_depend_on_orientation_or_starting_vertex() {
    let base = l_shape();
    let reference = StraightSkeleton::new(base.clone()).skeleton_edges();
    for shift in 0..base.len() {
        let mut rotated = base.clone();
        rotated.rotate_left(shift);
        assert_same_arcs(
            &StraightSkeleton::new(rotated.clone()).skeleton_edges(),
            &reference,
        );
        rotated.reverse();
        assert_same_arcs(&StraightSkeleton::new(rotated).skeleton_edges(), &reference);
    }
    // A repeated closing vertex is ignored.
    let mut closed = base.clone();
    closed.push(base[0]);
    assert_same_arcs(&StraightSkeleton::new(closed).skeleton_edges(), &reference);
}

#[test]
fn skeleton_is_translation_and_scale_covariant() {
    let base = l_shape();
    let reference = StraightSkeleton::new(base.clone())
        .skeleton_graph()
        .expect("base");
    for (scale, shift) in [
        (1e-6, [3e-6, -2e-6]),
        (1.0, [1000.0, -2000.0]),
        (1e5, [-4e5, 1e5]),
    ] {
        let moved: Vec<P2> = base
            .iter()
            .map(|p| [p[0] * scale + shift[0], p[1] * scale + shift[1]])
            .collect();
        let graph = StraightSkeleton::new(moved)
            .skeleton_graph()
            .expect("moved");
        assert_eq!(graph.arcs.len(), reference.arcs.len(), "scale {scale}");
        assert_eq!(graph.nodes.len(), reference.nodes.len(), "scale {scale}");
        for base_node in &reference.nodes {
            let expected = [
                base_node.position[0] * scale + shift[0],
                base_node.position[1] * scale + shift[1],
            ];
            let found = graph
                .nodes
                .iter()
                .find(|n| dist(n.position, expected) < 1e-7 * scale)
                .unwrap_or_else(|| panic!("scale {scale}: no node near {expected:?}"));
            assert!((found.time - base_node.time * scale).abs() < 1e-7 * scale);
        }
    }
}

// ── Random polygons in general position ─────────────────────────────────────

/// Star-shaped polygon around the origin with `n` vertices.
fn random_star(rng: &mut XorShift, n: usize) -> Vec<P2> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64 + rng.range(-0.3, 0.3)) / n as f64;
            let radius = rng.range(0.3, 1.0);
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect()
}

fn point_in_polygon(p: P2, poly: &[P2]) -> bool {
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
        {
            inside = !inside;
        }
    }
    inside
}

fn distance_to_boundary(p: P2, poly: &[P2]) -> f64 {
    let n = poly.len();
    (0..n)
        .map(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            let ab = sub(b, a);
            let t = (dot(sub(p, a), ab) / dot(ab, ab)).clamp(0.0, 1.0);
            dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
        })
        .fold(f64::INFINITY, f64::min)
}

/// Whether the segments `a-b` and `c-d` cross, and every end point is farther
/// than `margin` from the other segment (so that arcs that merely touch within
/// the numerical resolution do not count).
fn segments_cross_beyond(a: P2, b: P2, c: P2, d: P2, margin: f64) -> bool {
    let near = |p: P2, s0: P2, s1: P2| {
        let e = sub(s1, s0);
        let t = (dot(sub(p, s0), e) / dot(e, e)).clamp(0.0, 1.0);
        dist(p, [s0[0] + t * e[0], s0[1] + t * e[1]]) <= margin
    };
    segments_properly_cross(a, b, c, d)
        && !(near(a, c, d) || near(b, c, d) || near(c, a, b) || near(d, a, b))
}

fn segments_properly_cross(a: P2, b: P2, c: P2, d: P2) -> bool {
    let d1 = cross(sub(b, a), sub(c, a));
    let d2 = cross(sub(b, a), sub(d, a));
    let d3 = cross(sub(d, c), sub(a, c));
    let d4 = cross(sub(d, c), sub(b, c));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

#[test]
fn random_polygons_give_a_tree_with_the_expected_counts() {
    let mut rng = XorShift(0x1234_5678_9ABC_DEF1);
    for round in 0..60 {
        let n = 5 + round % 20;
        let poly = random_star(&mut rng, n);
        let ccw: Vec<P2> = if signed_area(&poly) > 0.0 {
            poly.clone()
        } else {
            poly.iter().rev().copied().collect()
        };
        let graph = StraightSkeleton::new(poly.clone())
            .skeleton_graph()
            .unwrap_or_else(|e| panic!("round {round}: {e}"));

        // General position: n leaf arcs plus n - 3 inner arcs, n - 2 inner
        // nodes of degree three.
        assert_eq!(graph.arcs.len(), 2 * n - 3, "round {round} (n = {n})");
        assert_eq!(graph.nodes.len(), 2 * n - 2, "round {round}");
        let degrees = graph.degrees();
        assert_eq!(
            degrees.iter().filter(|&&d| d == 1).count(),
            n,
            "round {round}"
        );
        assert_eq!(
            degrees.iter().filter(|&&d| d == 3).count(),
            n - 2,
            "round {round}"
        );
        assert!(degrees.iter().all(|&d| d == 1 || d == 3), "{degrees:?}");

        // Connected: a union-find over the arcs leaves one component.
        let mut parent: Vec<usize> = (0..graph.nodes.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for arc in &graph.arcs {
            let (ra, rb) = (find(&mut parent, arc[0]), find(&mut parent, arc[1]));
            assert_ne!(ra, rb, "round {round}: the skeleton has a cycle");
            parent[ra] = rb;
        }

        // Leaves are the polygon vertices at time 0; the inner nodes lie
        // strictly inside, at a positive time not exceeding their distance to
        // the boundary.
        for (i, node) in graph.nodes.iter().enumerate() {
            if degrees[i] == 1 {
                assert_eq!(node.time, 0.0);
                assert!(ccw.iter().any(|&v| dist(v, node.position) < TOL));
            } else {
                assert!(node.time > 0.0);
                assert!(
                    point_in_polygon(node.position, &ccw),
                    "round {round}: {:?}",
                    node.position
                );
                let clearance = distance_to_boundary(node.position, &ccw);
                assert!(
                    clearance >= node.time - 1e-9,
                    "round {round}: node {:?} time {} clearance {clearance}",
                    node.position,
                    node.time
                );
            }
        }

        // Each inner node is at offset time from three or more edge lines.
        for (i, node) in graph
            .nodes
            .iter()
            .enumerate()
            .filter(|&(i, _)| degrees[i] == 3)
        {
            let lines_at_time = (0..n)
                .filter(|&k| {
                    let (a, b) = (ccw[k], ccw[(k + 1) % n]);
                    let e = sub(b, a);
                    let len = dot(e, e).sqrt();
                    let normal = [-e[1] / len, e[0] / len];
                    (dot(normal, sub(node.position, a)) - node.time).abs() < 1e-8
                })
                .count();
            assert!(
                lines_at_time >= 3,
                "round {round}: node {i} touches {lines_at_time} lines"
            );
        }

        // Arcs of a planar graph do not cross.
        let edges = graph.edges();
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                assert!(
                    !segments_properly_cross(edges[i].0, edges[i].1, edges[j].0, edges[j].1),
                    "round {round}: arcs {i} and {j} cross"
                );
            }
        }

        // The arc of a polygon vertex is its bisector: the vertex moves with
        // velocity (n1 + n2) / (1 + n1 . n2) until its first event.
        for (k, &vertex) in ccw.iter().enumerate() {
            let prev = ccw[(k + n - 1) % n];
            let next = ccw[(k + 1) % n];
            let unit = |a: P2, b: P2| {
                let e = sub(b, a);
                let len = dot(e, e).sqrt();
                [-e[1] / len, e[0] / len]
            };
            let (n1, n2) = (unit(prev, vertex), unit(vertex, next));
            let velocity = {
                let denom = 1.0 + dot(n1, n2);
                [(n1[0] + n2[0]) / denom, (n1[1] + n2[1]) / denom]
            };
            let leaf = graph
                .nodes
                .iter()
                .position(|node| node.time == 0.0 && dist(node.position, vertex) < TOL)
                .unwrap_or_else(|| panic!("no leaf at {vertex:?}"));
            let arc = graph
                .arcs
                .iter()
                .find(|a| a[0] == leaf || a[1] == leaf)
                .expect("leaf arc");
            let other = if arc[0] == leaf { arc[1] } else { arc[0] };
            let end = &graph.nodes[other];
            let expected = [
                vertex[0] + end.time * velocity[0],
                vertex[1] + end.time * velocity[1],
            ];
            assert!(
                dist(end.position, expected) < 1e-8,
                "round {round}: vertex {k} arc ends at {:?}, bisector gives {expected:?}",
                end.position
            );
        }
    }
}

// ── Offsets ─────────────────────────────────────────────────────────────────

fn assert_ring_eq(ring: &[P2], expected: &[P2]) {
    assert_eq!(ring.len(), expected.len(), "{ring:?} vs {expected:?}");
    for (i, (a, b)) in ring.iter().zip(expected.iter()).enumerate() {
        assert!(dist(*a, *b) < TOL, "vertex {i}: {a:?} vs {b:?}");
    }
}

#[test]
fn unit_square_offsets_in_by_the_distance() {
    let sk = StraightSkeleton::new(rectangle(1.0, 1.0));
    let rings = sk.compute_offsets(0.1);
    assert_eq!(rings.len(), 1);
    assert_ring_eq(&rings[0], &[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]);
    // Clockwise input: same square, vertices in the input order.
    let cw: Vec<P2> = rectangle(1.0, 1.0).into_iter().rev().collect();
    let rings = StraightSkeleton::new(cw.clone()).compute_offsets(0.1);
    assert_eq!(rings.len(), 1);
    let expected: Vec<P2> = cw
        .iter()
        .map(|p| {
            [
                if p[0] < 0.5 { 0.1 } else { 0.9 },
                if p[1] < 0.5 { 0.1 } else { 0.9 },
            ]
        })
        .collect();
    assert_ring_eq(&rings[0], &expected);
    assert!(signed_area(&rings[0]) < 0.0, "orientation is kept");
}

#[test]
fn collinear_vertex_stays_on_the_offset_edge() {
    let poly = vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let rings = StraightSkeleton::new(poly).compute_offsets(0.1);
    assert_eq!(rings.len(), 1);
    assert_ring_eq(
        &rings[0],
        &[[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    );
}

#[test]
fn equilateral_triangle_corners_move_twice_the_distance() {
    // Interior angle 60 degrees: the miter is d / sin(30 degrees) = 2 d.
    let tri = regular_polygon(3, 1.0);
    let d = 0.1;
    let rings = StraightSkeleton::new(tri.clone()).compute_offsets(d);
    assert_eq!(rings.len(), 1);
    for (corner, moved) in tri.iter().zip(rings[0].iter()) {
        assert!(
            (dist(*corner, *moved) - 2.0 * d).abs() < TOL,
            "{corner:?} -> {moved:?}"
        );
        // Towards the incentre.
        let towards = [-corner[0], -corner[1]];
        assert!(dot(sub(*moved, *corner), towards) > 0.0);
    }
}

#[test]
fn right_angle_and_reflex_corners_use_the_mitre_of_the_edge_lines() {
    let rings = StraightSkeleton::new(l_shape()).compute_offsets(0.2);
    assert_eq!(rings.len(), 1);
    assert_ring_eq(
        &rings[0],
        &[
            [0.2, 0.2],
            [1.8, 0.2],
            [1.8, 0.8],
            [0.8, 0.8],
            [0.8, 1.8],
            [0.2, 1.8],
        ],
    );
    let rings = StraightSkeleton::new(l_shape()).compute_offsets(0.4);
    assert_ring_eq(
        &rings[0],
        &[
            [0.4, 0.4],
            [1.6, 0.4],
            [1.6, 0.6],
            [0.6, 0.6],
            [0.6, 1.6],
            [0.4, 1.6],
        ],
    );
}

#[test]
fn offsets_vanish_once_the_polygon_has_collapsed() {
    let square = StraightSkeleton::new(rectangle(1.0, 1.0));
    assert!(square.compute_offsets(0.5).is_empty());
    assert!(square.compute_offsets(0.6).is_empty());
    assert!(square.compute_offsets(1e6).is_empty());
    let rect = StraightSkeleton::new(rectangle(2.0, 1.0));
    assert_eq!(rect.compute_offsets(0.49).len(), 1);
    assert_ring_eq(
        &rect.compute_offsets(0.3)[0],
        &[[0.3, 0.3], [1.7, 0.3], [1.7, 0.7], [0.3, 0.7]],
    );
    assert!(rect.compute_offsets(0.5).is_empty());
    assert!(
        StraightSkeleton::new(l_shape())
            .compute_offsets(0.6)
            .is_empty()
    );
    // The offset ring of a shape is not simply scaled: a 4 x 1 strip with a
    // vertex in the middle keeps it until the strip is gone.
    let strip = StraightSkeleton::new(vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.0],
        [0.0, 1.0],
    ]);
    assert_eq!(strip.compute_offsets(0.2)[0].len(), 5);
    assert!(strip.compute_offsets(0.5).is_empty());
}

#[test]
fn zero_offset_returns_the_polygon_and_negative_offsets_grow_it() {
    let sq = rectangle(1.0, 1.0);
    let sk = StraightSkeleton::new(sq.clone());
    let same = sk.compute_offsets(0.0);
    assert_eq!(same.len(), 1);
    assert_ring_eq(&same[0], &sq);
    let grown = sk.compute_offsets(-0.1);
    assert_eq!(grown.len(), 1);
    assert_ring_eq(
        &grown[0],
        &[[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]],
    );
    // Outward corners of a convex polygon are not rounded: a triangle's
    // corner moves out by twice the distance.
    let tri = regular_polygon(3, 1.0);
    let rings = StraightSkeleton::new(tri.clone()).compute_offsets(-0.1);
    for (corner, moved) in tri.iter().zip(rings[0].iter()) {
        assert!((dist(*corner, *moved) - 0.2).abs() < TOL);
        assert!(dot(sub(*moved, *corner), *corner) > 0.0);
    }
}

#[test]
fn outward_offset_closes_a_narrow_mouth_and_leaves_a_hole() {
    // A 10 x 10 square with a bottle-shaped pocket cut from the right side:
    // a 4 x 4 room behind a mouth of width 1.
    let bottle: Vec<P2> = vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 4.5],
        [6.0, 4.5],
        [6.0, 3.0],
        [2.0, 3.0],
        [2.0, 7.0],
        [6.0, 7.0],
        [6.0, 5.5],
        [10.0, 5.5],
        [10.0, 10.0],
        [0.0, 10.0],
    ];
    let sk = StraightSkeleton::new(bottle);
    // Before the mouth closes (at 0.5) there is a single ring: the room is
    // still connected to the outside.
    let early = sk.compute_offsets(-0.3);
    assert_eq!(early.len(), 1);
    assert_eq!(early[0].len(), 12);
    assert!(signed_area(&early[0]) > 0.0);
    // Afterwards the mouth has filled up: the outer boundary is the square
    // grown by 0.6 and the room, shrunk by 0.6 on every side, is a hole with
    // the opposite orientation.
    let late = sk.compute_offsets(-0.6);
    assert_eq!(late.len(), 2, "{late:?}");
    let outer = late
        .iter()
        .find(|r| signed_area(r) > 0.0)
        .expect("outer ring");
    let hole = late
        .iter()
        .find(|r| signed_area(r) < 0.0)
        .expect("hole ring");
    assert!(
        (signed_area(outer) - 11.2 * 11.2).abs() < 1e-9,
        "{}",
        signed_area(outer)
    );
    assert!(
        (signed_area(hole) + 2.8 * 2.8).abs() < 1e-9,
        "{}",
        signed_area(hole)
    );
    for p in hole.iter() {
        assert!(
            p[0] > 2.6 - TOL && p[0] < 5.4 + TOL && p[1] > 3.6 - TOL && p[1] < 6.4 + TOL,
            "{p:?}"
        );
    }
    // The hole closes at 2 (half the room's width); the outer ring stays.
    let very_late = sk.compute_offsets(-2.5);
    assert_eq!(very_late.len(), 1);
    assert!((signed_area(&very_late[0]) - 15.0 * 15.0).abs() < 1e-9);
}

#[test]
fn dumbbell_wavefront_splits_into_two_rings() {
    let poly = dumbbell();
    let sk = StraightSkeleton::new(poly);
    // Before the neck collapses (at 0.2) there is a single ring.
    let before = sk.compute_offsets(0.1);
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].len(), 12);
    // Afterwards two squares shrunk by 0.3 remain; each keeps one extra
    // vertex on the wall where the neck was attached.
    let after = sk.compute_offsets(0.3);
    assert_eq!(after.len(), 2, "{after:?}");
    for ring in &after {
        assert_eq!(ring.len(), 5);
        assert!(
            (signed_area(ring) - 3.4 * 3.4).abs() < 1e-9,
            "{}",
            signed_area(ring)
        );
        assert!((perimeter(ring) - 4.0 * 3.4).abs() < 1e-9);
    }
    let xs: Vec<f64> = after
        .iter()
        .map(|r| r.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min))
        .collect();
    assert!(xs.iter().any(|&x| (x - 0.3).abs() < TOL));
    assert!(xs.iter().any(|&x| (x - 6.3).abs() < TOL));
    // Both squares vanish together at their inradius 2.
    assert_eq!(sk.compute_offsets(1.99).len(), 2);
    assert!(sk.compute_offsets(2.0).is_empty());
}

// ── Offsets of convex polygons against a half-plane oracle ──────────────────

/// Intersection of the half-planes `normal . (x - a) >= d` of the edges of the
/// counter-clockwise convex polygon `poly`, by clipping a large square.
fn half_plane_offset(poly: &[P2], d: f64) -> Vec<P2> {
    let mut region: Vec<P2> = vec![
        [-100.0, -100.0],
        [100.0, -100.0],
        [100.0, 100.0],
        [-100.0, 100.0],
    ];
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let e = sub(b, a);
        let len = dot(e, e).sqrt();
        let normal = [-e[1] / len, e[0] / len];
        let signed = |p: P2| dot(normal, sub(p, a)) - d;
        let mut clipped: Vec<P2> = Vec::new();
        for j in 0..region.len() {
            let (p, q) = (region[j], region[(j + 1) % region.len()]);
            let (sp, sq) = (signed(p), signed(q));
            if sp >= 0.0 {
                clipped.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        region = clipped;
        if region.is_empty() {
            break;
        }
    }
    region
}

/// Random convex polygon: points on an ellipse with jittered angles.
fn random_convex(rng: &mut XorShift, n: usize) -> Vec<P2> {
    let (rx, ry) = (rng.range(0.6, 1.6), rng.range(0.6, 1.6));
    (0..n)
        .map(|i| {
            let a = 2.0 * PI * (i as f64 + rng.range(-0.35, 0.35)) / n as f64;
            [rx * a.cos(), ry * a.sin()]
        })
        .collect()
}

#[test]
fn convex_offsets_match_the_half_plane_intersection() {
    let mut rng = XorShift(0xC0FF_EE00_DEAD_BEEF);
    for round in 0..40 {
        let n = 3 + round % 9;
        let poly = random_convex(&mut rng, n);
        let sk = StraightSkeleton::new(poly.clone());
        let inradius_bound = distance_to_boundary([0.0, 0.0], &poly);
        for fraction in [0.05, 0.3, 0.6, 0.9, 1.2, 3.0] {
            let d = fraction * inradius_bound;
            let rings = sk.compute_offsets(d);
            let oracle = half_plane_offset(&poly, d);
            let oracle_area = if oracle.len() >= 3 {
                signed_area(&oracle)
            } else {
                0.0
            };
            if oracle_area < 1e-9 {
                assert!(rings.is_empty(), "round {round}, d = {d}: {rings:?}");
                continue;
            }
            assert_eq!(rings.len(), 1, "round {round}, d = {d}");
            assert!(
                (signed_area(&rings[0]) - oracle_area).abs() < 1e-9,
                "round {round}, d = {d}: area {} vs oracle {oracle_area}",
                signed_area(&rings[0])
            );
            assert_eq!(rings[0].len(), oracle.len(), "round {round}, d = {d}");
            for &v in &rings[0] {
                assert!(
                    oracle.iter().any(|&o| dist(o, v) < 1e-8),
                    "round {round}, d = {d}: vertex {v:?} not in the oracle"
                );
            }
        }
    }
}

#[test]
fn convex_outward_offsets_move_every_edge_line_out() {
    let mut rng = XorShift(0xFEED_FACE_0BAD_CAFE);
    for round in 0..20 {
        let n = 3 + round % 8;
        let poly = random_convex(&mut rng, n);
        let sk = StraightSkeleton::new(poly.clone());
        for d in [0.05, 0.4, 2.0] {
            let rings = sk.compute_offsets(-d);
            assert_eq!(rings.len(), 1, "round {round}, d = {d}");
            let ring = &rings[0];
            assert_eq!(ring.len(), n);
            for i in 0..n {
                // Consecutive offset vertices lie on the edge line moved out by d.
                let (a, b) = (poly[i], poly[(i + 1) % n]);
                let e = sub(b, a);
                let len = dot(e, e).sqrt();
                let normal = [-e[1] / len, e[0] / len];
                for p in [ring[i], ring[(i + 1) % n]] {
                    assert!(
                        (dot(normal, sub(p, a)) + d).abs() < 1e-8,
                        "round {round}, d = {d}, edge {i}"
                    );
                }
            }
        }
    }
}

// ── The wavefront conserves its area flux ───────────────────────────────────

/// Total area and perimeter of the offset rings at `t` (hole rings count
/// negative) and the sorted vertex counts of the rings.
fn ring_statistics(sk: &StraightSkeleton, t: f64) -> (f64, f64, Vec<usize>) {
    let rings = sk.compute_offsets(t);
    let mut counts: Vec<usize> = rings.iter().map(|r| r.len()).collect();
    counts.sort_unstable();
    (
        rings.iter().map(|r| signed_area(r)).sum(),
        rings.iter().map(|r| perimeter(r)).sum(),
        counts,
    )
}

#[test]
fn area_decreases_by_the_perimeter_between_events() {
    // dA/dt = -P(t) because every wavefront edge moves at unit speed. Between
    // two events the perimeter is linear in t, so the trapezoid rule is exact;
    // an interval whose rings change structure contains an event (where the
    // perimeter can jump, e.g. when a strip of zero width collapses) and is
    // skipped. A wavefront that crosses itself or loses a ring breaks the
    // balance everywhere after the mistake.
    let mut rng = XorShift(0xABCD_EF01_2345_6789);
    let mut shapes: Vec<Vec<P2>> = vec![l_shape(), plus_shape(), dumbbell()];
    for round in 0..12 {
        shapes.push(random_star(&mut rng, 6 + round * 2));
    }
    for (index, poly) in shapes.iter().enumerate() {
        let sk = StraightSkeleton::new(poly.clone());
        let graph = sk.skeleton_graph().expect("skeleton");
        let t_max = graph.nodes.iter().fold(0.0_f64, |m, node| m.max(node.time));
        let steps = 300;
        let dt = t_max / steps as f64;
        let delta = 1e-7 * t_max;
        let mut tested = 0;
        for k in 0..steps {
            let (t1, t2) = (k as f64 * dt + delta, (k + 1) as f64 * dt - delta);
            let (a1, p1, c1) = ring_statistics(&sk, t1);
            let (a2, p2, c2) = ring_statistics(&sk, t2);
            if c1 != c2 || c1.is_empty() {
                continue;
            }
            let predicted = -0.5 * (p1 + p2) * (t2 - t1);
            assert!(
                (a2 - a1 - predicted).abs() < 1e-9 * (1.0 + p1 + p2),
                "shape {index}, t in [{t1}, {t2}]: area changed by {}, perimeter flux {predicted}",
                a2 - a1
            );
            tested += 1;
        }
        assert!(
            tested * 4 >= steps * 3,
            "shape {index}: only {tested} of {steps} intervals were event-free"
        );
        // Past the last event nothing is left.
        assert!(
            sk.compute_offsets(t_max * 1.0001 + 1e-6).is_empty(),
            "shape {index}"
        );
    }
}

// ── Rectilinear polygons: every event is degenerate, with an exact oracle ───

type Cells = HashSet<(i32, i32)>;

/// Counter-clockwise outer boundary of a union of unit cells, with only the
/// corners or with every grid point of the boundary (collinear vertices);
/// `None` if the union has holes, pinch points or several components.
fn trace_boundary(cells: &Cells, corners_only: bool) -> Option<Vec<P2>> {
    let mut next: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
    for &(i, j) in cells {
        let edges = [
            ((i, j - 1), (i, j), (i + 1, j)),
            ((i + 1, j), (i + 1, j), (i + 1, j + 1)),
            ((i, j + 1), (i + 1, j + 1), (i, j + 1)),
            ((i - 1, j), (i, j + 1), (i, j)),
        ];
        for (neighbour, from, to) in edges {
            if !cells.contains(&neighbour) && next.insert(from, to).is_some() {
                return None;
            }
        }
    }
    let start = *next.keys().min()?;
    let mut vertices = vec![start];
    let mut current = *next.get(&start)?;
    while current != start {
        vertices.push(current);
        if vertices.len() > next.len() {
            return None;
        }
        current = *next.get(&current)?;
    }
    if vertices.len() != next.len() {
        return None;
    }
    let n = vertices.len();
    let corners = (0..n)
        .filter(|&k| {
            let (a, b, c) = (
                vertices[(k + n - 1) % n],
                vertices[k],
                vertices[(k + 1) % n],
            );
            !corners_only || (b.0 - a.0, b.1 - a.1) != (c.0 - b.0, c.1 - b.1)
        })
        .map(|k| [vertices[k].0 as f64, vertices[k].1 as f64])
        .collect();
    Some(corners)
}

/// Area of the union of the axis-aligned rectangles `[x0, x1] x [y0, y1]`
/// restricted to the cells for which `keep(centre)` holds (coordinate
/// compression).
fn rectangle_union_area(rects: &[[f64; 4]], keep: impl Fn(P2) -> bool) -> f64 {
    let mut xs: Vec<f64> = rects.iter().flat_map(|r| [r[0], r[2]]).collect();
    let mut ys: Vec<f64> = rects.iter().flat_map(|r| [r[1], r[3]]).collect();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut area = 0.0;
    for wx in xs.windows(2) {
        for wy in ys.windows(2) {
            let centre = [0.5 * (wx[0] + wx[1]), 0.5 * (wy[0] + wy[1])];
            if rects.iter().any(|r| {
                centre[0] > r[0] && centre[0] < r[2] && centre[1] > r[1] && centre[1] < r[3]
            }) && keep(centre)
            {
                area += (wx[1] - wx[0]) * (wy[1] - wy[0]);
            }
        }
    }
    area
}

/// Area of the points `x` of the union of `cells` whose square `[x - t, x + t]^2`
/// lies in the union: the L-infinity erosion, which is exactly the inward
/// straight-skeleton offset of a rectilinear polygon.
fn eroded_area(cells: &Cells, t: f64) -> f64 {
    let (min_i, max_i) = (
        cells.iter().map(|c| c.0).min().unwrap_or(0) - 1,
        cells.iter().map(|c| c.0).max().unwrap_or(0) + 1,
    );
    let (min_j, max_j) = (
        cells.iter().map(|c| c.1).min().unwrap_or(0) - 1,
        cells.iter().map(|c| c.1).max().unwrap_or(0) + 1,
    );
    // Every point within t of a complement cell is lost: the open squares of
    // half-width t + 1/2 around the complement cell centres.
    let half = t + 0.5;
    let blockers: Vec<P2> = (min_i..=max_i)
        .flat_map(|i| (min_j..=max_j).map(move |j| (i, j)))
        .filter(|c| !cells.contains(c))
        .map(|(i, j)| [i as f64 + 0.5, j as f64 + 0.5])
        .collect();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for &(i, j) in cells {
        xs.extend([i as f64, i as f64 + 1.0]);
        ys.extend([j as f64, j as f64 + 1.0]);
    }
    for b in &blockers {
        xs.extend([b[0] - half, b[0] + half]);
        ys.extend([b[1] - half, b[1] + half]);
    }
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    ys.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let mut area = 0.0;
    for wx in xs.windows(2) {
        for wy in ys.windows(2) {
            let c = [0.5 * (wx[0] + wx[1]), 0.5 * (wy[0] + wy[1])];
            let inside = cells.contains(&(c[0].floor() as i32, c[1].floor() as i32));
            let blocked = blockers
                .iter()
                .any(|b| (c[0] - b[0]).abs() < half && (c[1] - b[1]).abs() < half);
            if inside && !blocked {
                area += (wx[1] - wx[0]) * (wy[1] - wy[0]);
            }
        }
    }
    area
}

/// Area of the L-infinity dilation of the union of `cells` by `t`.
fn dilated_area(cells: &Cells, t: f64) -> f64 {
    let rects: Vec<[f64; 4]> = cells
        .iter()
        .map(|&(i, j)| {
            [
                i as f64 - t,
                j as f64 - t,
                i as f64 + 1.0 + t,
                j as f64 + 1.0 + t,
            ]
        })
        .collect();
    rectangle_union_area(&rects, |_| true)
}

/// Random polygons made of grid cells, grown from a seed cell, rotated by
/// `angle` (rotation does not change areas, but destroys the exactness of
/// coordinates and edge directions).
fn random_rectilinear_polygons(
    rng: &mut XorShift,
    count: usize,
    corners_only: bool,
    angle: f64,
) -> Vec<(Cells, Vec<P2>)> {
    let mut found = Vec::new();
    for _ in 0..4000 {
        if found.len() == count {
            break;
        }
        let target = 3 + (rng.next_f64() * 22.0) as usize;
        let mut cells: Cells = HashSet::new();
        cells.insert((2, 2));
        let mut guard = 0;
        while cells.len() < target && guard < 500 {
            guard += 1;
            let pool: Vec<(i32, i32)> = cells.iter().copied().collect();
            let (i, j) = pool[(rng.next_f64() * pool.len() as f64) as usize % pool.len()];
            let (di, dj) = [(1, 0), (-1, 0), (0, 1), (0, -1)][(rng.next_f64() * 4.0) as usize % 4];
            let candidate = (i + di, j + dj);
            if (0..6).contains(&candidate.0) && (0..6).contains(&candidate.1) {
                cells.insert(candidate);
            }
        }
        if let Some(polygon) = trace_boundary(&cells, corners_only)
            && polygon.len() >= 4
        {
            let (sin, cos) = angle.sin_cos();
            let rotated = polygon
                .iter()
                .map(|p| [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]])
                .collect();
            found.push((cells, rotated));
        }
    }
    found
}

/// Whether to keep only the corners of the grid polygons, and the rotation.
const RECTILINEAR_VARIANTS: [(bool, f64); 4] =
    [(true, 0.0), (false, 0.0), (true, 0.6), (false, 2.3)];

#[test]
fn rectilinear_inward_offsets_match_the_l_infinity_erosion() {
    // Every rectilinear polygon is a stress test of simultaneous events:
    // parallel edges collapse along their whole length, several reflex
    // vertices reach one point, strips vanish at once. The offset area at
    // any distance, including the event times (multiples of 1/2), must
    // equal the area of the erosion by a square.
    for (variant, &(corners_only, angle)) in RECTILINEAR_VARIANTS.iter().enumerate() {
        let mut rng = XorShift(0x0DDB_A11C_0FFE_E001 + variant as u64);
        let shapes = random_rectilinear_polygons(&mut rng, 25, corners_only, angle);
        assert!(shapes.len() >= 20, "only {} shapes generated", shapes.len());
        for (index, (cells, polygon)) in shapes.iter().enumerate() {
            let sk = StraightSkeleton::new(polygon.clone());
            for t in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
                let rings = sk
                    .try_compute_offsets(t)
                    .unwrap_or_else(|e| panic!("variant {variant}, shape {index}, t = {t}: {e}"));
                let area: f64 = rings.iter().map(|r| signed_area(r)).sum();
                let expected = eroded_area(cells, t);
                assert!(
                    (area - expected).abs() < 1e-8 * (1.0 + expected),
                    "variant {variant}, shape {index} {polygon:?}, t = {t}: \
                     offset area {area}, erosion {expected}"
                );
                assert!(rings.iter().all(|r| signed_area(r) > 0.0));
            }
        }
    }
}

#[test]
fn rectilinear_outward_offsets_match_the_l_infinity_dilation() {
    for (variant, &(corners_only, angle)) in RECTILINEAR_VARIANTS.iter().enumerate() {
        let mut rng = XorShift(0xFACE_B00C_1234_5678 + variant as u64);
        let shapes = random_rectilinear_polygons(&mut rng, 25, corners_only, angle);
        assert!(shapes.len() >= 20, "only {} shapes generated", shapes.len());
        for (index, (cells, polygon)) in shapes.iter().enumerate() {
            let sk = StraightSkeleton::new(polygon.clone());
            for t in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0] {
                let rings = sk
                    .try_compute_offsets(-t)
                    .unwrap_or_else(|e| panic!("variant {variant}, shape {index}, t = -{t}: {e}"));
                let area: f64 = rings.iter().map(|r| signed_area(r)).sum();
                let expected = dilated_area(cells, t);
                assert!(
                    (area - expected).abs() < 1e-8 * (1.0 + expected),
                    "variant {variant}, shape {index} {polygon:?}, t = -{t}: \
                     offset area {area}, dilation {expected}"
                );
                // One outer ring; any other ring is a hole of opposite orientation.
                assert_eq!(rings.iter().filter(|r| signed_area(r) > 0.0).count(), 1);
            }
        }
    }
}

#[test]
fn rectilinear_skeletons_are_planar_trees_inside_the_polygon() {
    for (variant, &(corners_only, angle)) in RECTILINEAR_VARIANTS.iter().enumerate() {
        let mut rng = XorShift(0x5EC7_100A_BADC_0DE0 + variant as u64);
        let shapes = random_rectilinear_polygons(&mut rng, 25, corners_only, angle);
        assert!(shapes.len() >= 20, "only {} shapes generated", shapes.len());
        for (index, (cells, polygon)) in shapes.iter().enumerate() {
            let graph = StraightSkeleton::new(polygon.clone())
                .skeleton_graph()
                .unwrap_or_else(|e| panic!("variant {variant}, shape {index} {polygon:?}: {e}"));
            let degrees = graph.degrees();
            // Every polygon vertex is a leaf; every other node joins three or
            // more arcs.
            let leaves: Vec<usize> = (0..graph.nodes.len())
                .filter(|&i| graph.nodes[i].time == 0.0)
                .collect();
            assert_eq!(
                leaves.len(),
                polygon.len(),
                "variant {variant}, shape {index}"
            );
            for &leaf in &leaves {
                assert_eq!(degrees[leaf], 1, "variant {variant}, shape {index}");
                assert!(
                    polygon
                        .iter()
                        .any(|&v| dist(v, graph.nodes[leaf].position) < TOL)
                );
            }
            // Undo the rotation to look the nodes up in the cell set.
            let (sin, cos) = (-angle).sin_cos();
            for (i, node) in graph.nodes.iter().enumerate() {
                if node.time == 0.0 {
                    continue;
                }
                assert!(
                    degrees[i] >= 3,
                    "variant {variant}, shape {index}: node {:?} has degree {}",
                    node.position,
                    degrees[i]
                );
                let p = node.position;
                let upright = [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]];
                // The node lies in the polygon, at least `time` (in the
                // L-infinity sense) away from the complement: the erosion
                // by `time - 1e-7` still contains it.
                assert!(
                    cells.contains(&(upright[0].floor() as i32, upright[1].floor() as i32)),
                    "variant {variant}, shape {index}: node {p:?} is outside the polygon"
                );
            }
            // A skeleton is a tree.
            assert_eq!(
                graph.arcs.len() + 1,
                graph.nodes.len(),
                "variant {variant}, shape {index}: not a tree"
            );
            let edges = graph.edges();
            for i in 0..edges.len() {
                for j in (i + 1)..edges.len() {
                    assert!(
                        !segments_properly_cross(edges[i].0, edges[i].1, edges[j].0, edges[j].1),
                        "variant {variant}, shape {index}: arcs {i} and {j} cross"
                    );
                }
            }
        }
    }
}

#[test]
fn nearly_degenerate_rectilinear_polygons_stay_consistent() {
    // Perturbing the vertices of a rectilinear polygon splits its massively
    // simultaneous events into nearly simultaneous ones, from far below the
    // merge tolerance up to clearly separated events. Whatever the amplitude
    // the wavefront must stay consistent: the skeleton is a tree, the offsets
    // follow the unperturbed erosion up to the size of the perturbation.
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let shapes = random_rectilinear_polygons(&mut rng, 20, true, 0.0);
    assert!(shapes.len() >= 15);
    for amplitude in [1e-12, 1e-10, 1e-8, 1e-7, 1e-5, 1e-3] {
        // Arcs closer than this to each other's end points only touch.
        let margin = 1e-6_f64.max(100.0 * amplitude);
        for (index, (cells, polygon)) in shapes.iter().enumerate() {
            let perturbed: Vec<P2> = polygon
                .iter()
                .map(|p| {
                    [
                        p[0] + rng.range(-amplitude, amplitude),
                        p[1] + rng.range(-amplitude, amplitude),
                    ]
                })
                .collect();
            let sk = StraightSkeleton::new(perturbed.clone());
            let graph = sk.skeleton_graph().unwrap_or_else(|e| {
                panic!("amplitude {amplitude}, shape {index} {perturbed:?}: {e}")
            });
            assert_eq!(
                graph.arcs.len() + 1,
                graph.nodes.len(),
                "amplitude {amplitude}, shape {index}: not a tree"
            );
            let edges = graph.edges();
            for i in 0..edges.len() {
                for j in (i + 1)..edges.len() {
                    assert!(
                        !segments_cross_beyond(
                            edges[i].0, edges[i].1, edges[j].0, edges[j].1, margin
                        ),
                        "amplitude {amplitude}, shape {index}: arcs {i} {:?} and {j} {:?} cross",
                        edges[i],
                        edges[j]
                    );
                }
            }
            for t in [0.25, 0.5, 1.0, 1.5] {
                let rings = sk.compute_offsets(t);
                let area: f64 = rings.iter().map(|r| signed_area(r)).sum();
                let expected = eroded_area(cells, t);
                assert!(
                    (area - expected).abs() < 300.0 * amplitude + 1e-9,
                    "amplitude {amplitude}, shape {index}, t = {t}: area {area} vs erosion {expected}"
                );
            }
        }
    }
}

// ── Invalid input ───────────────────────────────────────────────────────────

#[test]
fn polygons_without_a_skeleton_are_reported() {
    let bow_tie = vec![[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]];
    let sk = StraightSkeleton::new(bow_tie);
    assert!(sk.skeleton_edges().is_empty());
    assert!(sk.compute_offsets(0.1).is_empty());
    assert!(matches!(
        sk.skeleton_graph(),
        Err(Error::DegenerateGeometry { .. })
    ));

    let touching = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 0.0], [0.0, 4.0]];
    assert!(matches!(
        StraightSkeleton::new(touching).skeleton_graph(),
        Err(Error::DegenerateGeometry { .. })
    ));

    assert!(matches!(
        StraightSkeleton::new(vec![[0.0, 0.0], [1.0, 0.0]]).skeleton_graph(),
        Err(Error::TooFewPoints {
            required: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        StraightSkeleton::new(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
            .skeleton_graph(),
        Err(Error::TooFewPoints { .. }) | Err(Error::DegenerateGeometry { .. })
    ));
    assert!(matches!(
        StraightSkeleton::new(vec![[0.0, 0.0], [1.0, f64::NAN], [1.0, 1.0]]).skeleton_graph(),
        Err(Error::DegenerateGeometry { .. })
    ));
    // Collinear points enclose no area.
    assert!(matches!(
        StraightSkeleton::new(vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            .skeleton_graph(),
        Err(Error::DegenerateGeometry { .. })
    ));
    // A zero-width spike: the polygon doubles back along itself.
    let spike = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 3.0]];
    assert!(StraightSkeleton::new(spike).skeleton_graph().is_err());
    // Non-finite offset distance.
    let sq = StraightSkeleton::new(rectangle(1.0, 1.0));
    assert!(matches!(
        sq.try_compute_offsets(f64::NAN),
        Err(Error::OutOfRange { .. })
    ));
    assert!(sq.compute_offsets(f64::INFINITY).is_empty());
}

#[test]
fn random_input_terminates_with_a_result_or_an_error() {
    // Random point soups are mostly not simple polygons; whatever the answer,
    // it must come quickly and without panicking.
    let mut rng = XorShift(0x5EED_5EED_5EED_5EED);
    let mut accepted = 0;
    for _ in 0..300 {
        let n = 3 + (rng.next_f64() * 9.0) as usize;
        let poly: Vec<P2> = (0..n)
            .map(|_| [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)])
            .collect();
        let sk = StraightSkeleton::new(poly);
        if sk.skeleton_graph().is_ok() {
            accepted += 1;
            let _ = sk.compute_offsets(0.05);
            let _ = sk.compute_offsets(-0.05);
        }
    }
    assert!(
        accepted > 10,
        "only {accepted} of 300 random polygons were simple"
    );
}

#[test]
fn many_vertex_polygons_complete() {
    // A 200-vertex star with many reflex vertices, and a circle sampled at
    // 500 points (all events nearly simultaneous).
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    let star = random_star(&mut rng, 200);
    let graph = StraightSkeleton::new(star).skeleton_graph().expect("star");
    assert_eq!(graph.arcs.len(), 2 * 200 - 3);
    let circle = regular_polygon(500, 1.0);
    let graph = StraightSkeleton::new(circle)
        .skeleton_graph()
        .expect("circle");
    assert_eq!(graph.arcs.len(), 500);
    let centre = node_index(&graph, [0.0, 0.0]);
    assert!((graph.nodes[centre].time - (PI / 500.0).cos()).abs() < 1e-8);
}
