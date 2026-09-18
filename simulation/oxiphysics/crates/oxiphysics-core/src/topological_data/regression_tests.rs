// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Vietoris–Rips H₁ persistence and Betti numbers.
//!
//! The previous H₁ routine invented birth and death values (the previous
//! spanning-tree edge and the midpoint to `max_r`) and never built triangles;
//! the previous Betti routine returned β₂ = number of triangles and ignored
//! triangles for β₁.  Expected values below were derived by hand or with an
//! independent ℤ/2 reduction.

use super::*;
use std::f64::consts::{PI, SQRT_2};

const TOL: f64 = 1e-9;

/// Small xorshift64 generator for reproducible test clouds.
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_f64(&mut self) -> f64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn unit_square() -> Vec<[f64; 2]> {
    vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// Regular `n`-gon inscribed in the unit circle.
fn regular_polygon(n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n as f64;
            [angle.cos(), angle.sin()]
        })
        .collect()
}

/// `n` points on a unit circle with radial noise of amplitude 0.1.
fn noisy_ring(n: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = XorShift::new(seed);
    (0..n)
        .map(|k| {
            let angle = 2.0 * PI * k as f64 / n as f64;
            let radius = 1.0 + 0.1 * (rng.next_f64() - 0.5);
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect()
}

/// `n` points scattered uniformly in a `side × side` square.
fn scattered_cloud(n: usize, seed: u64, side: f64) -> Vec<[f64; 2]> {
    let mut rng = XorShift::new(seed);
    (0..n)
        .map(|_| [rng.next_f64() * side, rng.next_f64() * side])
        .collect()
}

fn translated(points: &[[f64; 2]], dx: f64, dy: f64) -> Vec<[f64; 2]> {
    points.iter().map(|p| [p[0] + dx, p[1] + dy]).collect()
}

fn assert_bars(actual: &[PersistentInterval], expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len(), "bars: {actual:?}");
    for (bar, &(birth, death)) in actual.iter().zip(expected) {
        assert_eq!(bar.dimension, 1);
        assert!((bar.birth - birth).abs() < TOL, "{bar:?} vs birth {birth}");
        if death.is_infinite() {
            assert!(bar.is_essential(), "{bar:?} should be essential");
        } else {
            assert!((bar.death - death).abs() < TOL, "{bar:?} vs death {death}");
        }
    }
}

fn betti(points: &[[f64; 2]], epsilon: f64) -> (usize, usize, usize) {
    let b = VietorisRips::new(points.to_vec(), epsilon).betti_numbers();
    (b.b0, b.b1, b.b2)
}

// ── H₁ persistence ───────────────────────────────────────────────────────────

#[test]
fn h1_of_the_unit_square_is_one_bar_from_one_to_sqrt_two() {
    // The four sides close a loop at 1; at sqrt(2) the diagonals and all four
    // triangles fill it.  The loops born with each diagonal are filled at the
    // same scale (zero persistence) and are not reported.
    assert_bars(&vietoris_rips_h1(&unit_square(), 2.0), &[(1.0, SQRT_2)]);
}

#[test]
fn h1_of_the_unit_square_below_the_diagonal_scale_is_essential() {
    assert_bars(
        &vietoris_rips_h1(&unit_square(), 1.2),
        &[(1.0, f64::INFINITY)],
    );
}

#[test]
fn h1_loop_is_born_at_the_closing_edge_not_the_previous_tree_edge() {
    // Sides 1, 1, 1.077 (p3-p0) and 1.4 (p2-p3): the spanning tree is completed
    // by 1.077, and the loop only closes with the 1.4 edge.  The old code
    // reported the birth 1.077.  It is filled at sqrt(2) by the diagonal p0-p2.
    let kite = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.4, 1.0]];
    assert_bars(&vietoris_rips_h1(&kite, 3.0), &[(1.4, SQRT_2)]);
}

#[test]
fn h1_of_a_filled_triangle_is_empty() {
    let triangle = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
    assert!(vietoris_rips_h1(&triangle, 10.0).is_empty());
    // Below the scale of the long edges there is no cycle at all.
    assert!(vietoris_rips_h1(&triangle, 1.05).is_empty());
}

#[test]
fn h1_of_degenerate_inputs_is_empty() {
    assert!(vietoris_rips_h1(&[], 1.0).is_empty());
    assert!(vietoris_rips_h1(&[[0.0, 0.0]], 1.0).is_empty());
    assert!(vietoris_rips_h1(&[[0.0, 0.0], [1.0, 0.0]], 5.0).is_empty());
    // A negative scale admits no edge.
    assert!(vietoris_rips_h1(&unit_square(), -1.0).is_empty());
}

#[test]
fn h1_of_a_regular_hexagon_dies_at_the_short_diagonal() {
    let hexagon = regular_polygon(6);
    // Side 1, short diagonal sqrt(3), long diagonal 2.
    assert_bars(&vietoris_rips_h1(&hexagon, 1.8), &[(1.0, 3.0_f64.sqrt())]);
    assert_bars(&vietoris_rips_h1(&hexagon, 1.1), &[(1.0, f64::INFINITY)]);
}

#[test]
fn h1_of_a_regular_pentagon_spans_side_to_diagonal() {
    let side = 2.0 * (PI / 5.0).sin();
    let diagonal = 2.0 * (2.0 * PI / 5.0).sin();
    let pentagon = regular_polygon(5);
    assert_bars(&vietoris_rips_h1(&pentagon, 2.5), &[(side, diagonal)]);
}

#[test]
fn h1_of_two_separated_squares_has_two_equal_bars() {
    let mut cloud = unit_square();
    cloud.extend(translated(&unit_square(), 10.0, 0.0));
    assert_bars(
        &vietoris_rips_h1(&cloud, 2.0),
        &[(1.0, SQRT_2), (1.0, SQRT_2)],
    );
}

#[test]
fn h1_of_a_noisy_ring_matches_an_independent_reduction() {
    // Reference values from an independent ℤ/2 reduction of the same filtration.
    let ring = noisy_ring(16, 1);
    assert_bars(
        &vietoris_rips_h1(&ring, 1.9),
        &[(0.404_154_695_957_366_6, 1.774_821_587_008_568_8)],
    );
    assert_bars(
        &vietoris_rips_h1(&ring, 0.6),
        &[(0.404_154_695_957_366_6, f64::INFINITY)],
    );
    assert!(vietoris_rips_h1(&ring, 0.4).is_empty());
}

#[test]
fn h1_edge_cap_truncates_the_filtration() {
    let square = unit_square();
    // The two shortest edges cannot contain a cycle; the four unit sides form
    // an unfilled one; all six edges (and four triangles) fill it.
    assert!(rips_h1_bars(&square, 2.0, 3).is_empty());
    assert_bars(&rips_h1_bars(&square, 2.0, 4), &[(1.0, f64::INFINITY)]);
    assert_bars(&rips_h1_bars(&square, 2.0, 6), &[(1.0, SQRT_2)]);
    assert_bars(&rips_h1_bars(&square, 2.0, 1_000), &[(1.0, SQRT_2)]);
}

#[test]
fn h1_edge_cap_only_builds_triangles_whose_edges_survive() {
    // With five of the six edges of K4 kept, only the two triangles that avoid
    // the dropped edge exist; the loop through the dropped edge stays open.
    let kite = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.4, 1.0]];
    let filtration = rips_two_skeleton_filtration(&kite, 3.0, 5);
    let edges = filtration.iter().filter(|(_, s)| s.len() == 2).count();
    let triangles = filtration.iter().filter(|(_, s)| s.len() == 3).count();
    assert_eq!(edges, 5);
    assert_eq!(triangles, 2);
    // Faces precede cofaces in the filtration order.
    let position = |simplex: &[usize]| filtration.iter().position(|(_, s)| s == simplex);
    for (index, (_, simplex)) in filtration.iter().enumerate() {
        if simplex.len() < 2 {
            continue;
        }
        for drop in 0..simplex.len() {
            let mut face = simplex.clone();
            face.remove(drop);
            let face_index = position(&face).expect("every face is in the filtration");
            assert!(face_index < index, "face {face:?} after {simplex:?}");
        }
    }
}

#[test]
fn h1_bars_are_sorted_bounded_and_agree_with_betti_one() {
    // The number of essential bars at scale r is β₁ of the 2-skeleton at r.
    for (seed, n, side, max_r) in [
        (1u64, 14usize, 2.0, 0.9),
        (2, 18, 3.0, 1.0),
        (3, 20, 3.0, 0.8),
        (4, 24, 4.0, 1.1),
        (5, 16, 2.5, 0.7),
    ] {
        let cloud = scattered_cloud(n, seed, side);
        let bars = vietoris_rips_h1(&cloud, max_r);
        for bar in &bars {
            assert_eq!(bar.dimension, 1);
            assert!(bar.birth > 0.0 && bar.birth <= bar.death, "{bar:?}");
            assert!(bar.birth <= max_r && (bar.is_essential() || bar.death <= max_r));
        }
        for pair in bars.windows(2) {
            assert!(pair[0].birth <= pair[1].birth);
        }
        let essential = bars.iter().filter(|b| b.is_essential()).count();
        let vr = VietorisRips::new(cloud.clone(), max_r);
        assert_eq!(essential, vr.betti_numbers().b1, "seed {seed}");
        assert_eq!(vr.persistent_h1(), bars, "seed {seed}");
    }
}

// ── Betti numbers ────────────────────────────────────────────────────────────

#[test]
fn betti_numbers_of_a_filled_triangle() {
    // The old formulas reported (1, 1, 1): β₂ was the triangle count.
    let triangle = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
    assert_eq!(betti(&triangle, 2.0), (1, 0, 0));
}

#[test]
fn betti_numbers_of_a_four_cycle_without_triangles() {
    assert_eq!(betti(&unit_square(), 1.2), (1, 1, 0));
}

#[test]
fn betti_numbers_of_four_mutually_close_points_count_the_hollow_tetrahedron() {
    // All six edges and four triangles, no 3-simplex: the 2-skeleton of a
    // tetrahedron is a 2-sphere, so β₂ = 1 and β₁ = 0.
    assert_eq!(betti(&unit_square(), 1.5), (1, 0, 1));
}

#[test]
fn betti_numbers_of_the_octahedral_hexagon() {
    let hexagon = regular_polygon(6);
    // Sides only: a 6-cycle.
    assert_eq!(betti(&hexagon, 1.1), (1, 1, 0));
    // Sides and short diagonals: the octahedron surface, a genuine 2-sphere
    // without any 4-clique to fill it.
    assert_eq!(betti(&hexagon, 1.8), (1, 0, 1));
    // Everything connected: the 2-skeleton of the 5-simplex has β₂ = C(5,3).
    assert_eq!(betti(&hexagon, 2.1), (1, 0, 10));
}

#[test]
fn betti_numbers_of_polygons_and_disjoint_unions() {
    assert_eq!(betti(&regular_polygon(5), 1.2), (1, 1, 0));
    let mut two_squares = unit_square();
    two_squares.extend(translated(&unit_square(), 10.0, 0.0));
    assert_eq!(betti(&two_squares, 1.2), (2, 2, 0));
    assert_eq!(betti(&two_squares, 1.5), (2, 0, 2));
}

#[test]
fn betti_numbers_of_paths_points_and_collinear_triples() {
    assert_eq!(betti(&[[0.0, 0.0]], 1.0), (1, 0, 0));
    let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    assert_eq!(betti(&line, 1.5), (1, 0, 0));
    assert_eq!(betti(&line, 2.5), (1, 0, 0));
}

#[test]
fn betti_numbers_of_a_noisy_ring_match_an_independent_reduction() {
    let ring = noisy_ring(16, 1);
    assert_eq!(betti(&ring, 0.4), (2, 0, 0));
    assert_eq!(betti(&ring, 0.6), (1, 1, 0));
    assert_eq!(betti(&ring, 1.4), (1, 1, 24));
    assert_eq!(betti(&ring, 1.9), (1, 0, 212));
}

#[test]
fn betti_numbers_satisfy_the_euler_poincare_relation() {
    for seed in 1..=12u64 {
        let cloud = scattered_cloud(16, seed, 3.0);
        for epsilon in [0.4, 0.8, 1.2, 1.6] {
            let vr = VietorisRips::new(cloud.clone(), epsilon);
            let b = vr.betti_numbers();
            let chi = b.b0 as i64 - b.b1 as i64 + b.b2 as i64;
            assert_eq!(chi, vr.euler_characteristic(), "seed {seed}, eps {epsilon}");
            assert_eq!(euler_characteristic(&b), vr.euler_characteristic());
            // rank ∂₁ = V − β₀ and β₁ both fit into the E edges; β₂ fits into the F triangles.
            assert!(b.b2 <= vr.n_triangles());
            assert!(b.b1 + vr.n_vertices() - b.b0 <= vr.n_edges());
        }
    }
}

#[test]
fn z2_column_rank_handles_dependent_and_empty_columns() {
    assert_eq!(z2_column_rank(vec![]), 0);
    assert_eq!(z2_column_rank(vec![vec![], vec![]]), 0);
    // {0,1} + {1,2} = {0,2}: rank 2 out of three columns.
    assert_eq!(z2_column_rank(vec![vec![0, 1], vec![1, 2], vec![0, 2]]), 2);
    assert_eq!(z2_column_rank(vec![vec![0], vec![1], vec![2]]), 3);
    // Duplicated columns cancel over ℤ/2.
    assert_eq!(z2_column_rank(vec![vec![0, 3], vec![0, 3]]), 1);
}
