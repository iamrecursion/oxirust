// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`flip_edges_2d`] (Lawson edge flipping).
//!
//! The flip used to build its triangles from the index order of the shared edge
//! and to test the in-circle determinant with an orientation that depended on the
//! same order, so non-Delaunay edges were never fixed while Delaunay ones could be
//! flipped.  These tests pin the result on exact configurations and on a Delaunay
//! property check of a jittered grid.

use super::*;

/// Twice the signed area of the triangle (positive = counter-clockwise).
fn signed_area(positions: &[[f64; 3]], tri: [usize; 3]) -> f64 {
    let [a, b, c] = tri.map(|i| positions[i]);
    (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
}

/// Rhombus `(0,0), (1,-0.2), (2,0), (1,0.2)`: the diagonal `0-2` is the long one
/// and violates the Delaunay property, the short diagonal `1-3` satisfies it.
fn rhombus() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, -0.2, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
    ]
}

/// Sorted vertex triples of a triangle list, for order-independent comparison.
fn vertex_sets(triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut sets: Vec<[usize; 3]> = triangles
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    sets.sort_unstable();
    sets
}

#[test]
fn non_delaunay_diagonal_is_flipped_for_either_triangle_order() {
    let positions = rhombus();
    for initial in [[[0usize, 1, 2], [0, 2, 3]], [[0, 2, 3], [0, 1, 2]]] {
        let mut triangles = initial;
        let flips = flip_edges_2d(&mut triangles, &positions);
        assert_eq!(flips, 1, "triangles {initial:?}");
        assert_eq!(vertex_sets(&triangles), vec![[0, 1, 3], [1, 2, 3]]);
        // Both new triangles contain the new diagonal 1-3 and stay
        // counter-clockwise like the triangles they replace.
        for t in &triangles {
            assert!(t.contains(&1) && t.contains(&3));
            assert!(signed_area(&positions, *t) > 0.0, "triangle {t:?}");
        }
        let total: f64 = triangles.iter().map(|&t| signed_area(&positions, t)).sum();
        assert!((total - 0.8).abs() < 1e-12, "area changed: {total}");
    }
}

#[test]
fn clockwise_triangles_are_flipped_just_the_same_and_stay_clockwise() {
    let positions = rhombus();
    let mut triangles = [[0usize, 2, 1], [0, 3, 2]];
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 1);
    assert_eq!(vertex_sets(&triangles), vec![[0, 1, 3], [1, 2, 3]]);
    for t in &triangles {
        assert!(signed_area(&positions, *t) < 0.0, "triangle {t:?}");
    }
}

#[test]
fn a_neighbour_with_the_opposite_winding_keeps_its_own_winding() {
    let positions = rhombus();
    // The second triangle is wound the other way round than the first.
    let mut triangles = [[0usize, 1, 2], [0, 3, 2]];
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 1);
    assert_eq!(vertex_sets(&triangles), vec![[0, 1, 3], [1, 2, 3]]);
    assert!(signed_area(&positions, triangles[0]) > 0.0);
    assert!(signed_area(&positions, triangles[1]) < 0.0);
}

#[test]
fn delaunay_triangulations_are_left_alone() {
    let positions = rhombus();
    let mut triangles = [[0usize, 1, 3], [1, 2, 3]];
    let before = triangles;
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 0);
    assert_eq!(triangles, before);
}

#[test]
fn a_flip_is_not_undone_by_a_second_call() {
    let positions = rhombus();
    let mut triangles = [[0usize, 1, 2], [0, 2, 3]];
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 1);
    let flipped = triangles;
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 0);
    assert_eq!(triangles, flipped);
}

#[test]
fn cocircular_points_are_never_flipped() {
    // A unit square and a square at inexact coordinates are cocircular up to
    // round-off; neither diagonal may be preferred.
    for (scale, offset) in [(1.0, [0.0, 0.0]), (0.1, [0.3, 0.7]), (3.7e-3, [12.3, -4.6])] {
        let positions: Vec<[f64; 3]> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| [offset[0] + scale * p[0], offset[1] + scale * p[1], 0.0])
            .collect();
        for initial in [[[0usize, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]] {
            let mut triangles = initial;
            assert_eq!(
                flip_edges_2d(&mut triangles, &positions),
                0,
                "scale {scale}"
            );
            assert_eq!(triangles, initial);
        }
    }
}

#[test]
fn a_regular_grid_with_inexact_coordinates_is_not_disturbed() {
    let cells = 6usize;
    let stride = cells + 1;
    let mut positions = Vec::new();
    for j in 0..stride {
        for i in 0..stride {
            positions.push([0.3 + 0.1 * i as f64, 0.7 + 0.1 * j as f64, 0.0]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            let v00 = j * stride + i;
            triangles.push([v00, v00 + 1, v00 + stride + 1]);
            triangles.push([v00, v00 + stride + 1, v00 + stride]);
        }
    }
    let before = triangles.clone();
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 0);
    assert_eq!(triangles, before);
}

#[test]
fn a_non_convex_quadrilateral_is_never_flipped() {
    // The quadrilateral 0-1-2-3 has a reflex corner at 2, so the other diagonal 1-3
    // lies outside of it; flipping would fold the triangles over each other.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, -1.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let mut triangles = [[0usize, 1, 2], [0, 2, 3]];
    let before = triangles;
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 0);
    assert_eq!(triangles, before);
}

#[test]
fn degenerate_and_invalid_input_is_left_untouched() {
    let positions = rhombus();
    // Repeated vertex, out-of-range vertex and collinear points.
    let mut with_repeat = [[0usize, 0, 2], [0, 2, 3]];
    let before = with_repeat;
    assert_eq!(flip_edges_2d(&mut with_repeat, &positions), 0);
    assert_eq!(with_repeat, before);

    let mut out_of_range = [[0usize, 1, 2], [0, 2, 9]];
    let before = out_of_range;
    assert_eq!(flip_edges_2d(&mut out_of_range, &positions), 0);
    assert_eq!(out_of_range, before);

    let collinear = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let mut flat = [[0usize, 1, 3], [1, 2, 3]];
    let before = flat;
    assert_eq!(flip_edges_2d(&mut flat, &collinear), 0);
    assert_eq!(flat, before);

    assert_eq!(flip_edges_2d(&mut [], &positions), 0);
}

/// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Whether `d` lies strictly inside the circumcircle of the triangle `tri`
/// (evaluated in the orientation of the triangle), with a small tolerance.
fn strictly_inside_circumcircle(positions: &[[f64; 3]], tri: [usize; 3], d: [f64; 3]) -> bool {
    let orientation = signed_area(positions, tri).signum();
    let [a, b, c] = tri.map(|i| positions[i]);
    let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
    let (bx, by) = (b[0] - d[0], b[1] - d[1]);
    let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    orientation * det > 1e-9
}

#[test]
fn flipping_a_jittered_grid_yields_a_delaunay_triangulation() {
    let cells = 7usize;
    let stride = cells + 1;
    let mut rng = Lcg(0x1357_9BDF_2468_ACE0);
    let mut positions = Vec::new();
    for j in 0..stride {
        for i in 0..stride {
            positions.push([
                i as f64 + 0.4 * (rng.next_unit() - 0.5),
                j as f64 + 0.4 * (rng.next_unit() - 0.5),
                0.0,
            ]);
        }
    }
    // Every cell is split along a random diagonal; the jitter keeps the cells
    // convex, so this is a valid triangulation that is far from Delaunay.
    let mut triangles = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            let v00 = j * stride + i;
            let (v10, v01, v11) = (v00 + 1, v00 + stride, v00 + stride + 1);
            if rng.next_unit() < 0.5 {
                triangles.push([v00, v10, v11]);
                triangles.push([v00, v11, v01]);
            } else {
                triangles.push([v00, v10, v01]);
                triangles.push([v10, v11, v01]);
            }
        }
    }
    let area_before: f64 = triangles.iter().map(|&t| signed_area(&positions, t)).sum();
    let count_before = triangles.len();

    let flips = flip_edges_2d(&mut triangles, &positions);
    assert!(flips > 0, "a random triangulation needs flips");
    assert_eq!(triangles.len(), count_before);

    // Winding and covered area are preserved.
    for t in &triangles {
        assert!(
            signed_area(&positions, *t) > 0.0,
            "triangle {t:?} flipped over"
        );
    }
    let area_after: f64 = triangles.iter().map(|&t| signed_area(&positions, t)).sum();
    assert!((area_before - area_after).abs() < 1e-9);

    // Empty-circumcircle property: no vertex lies strictly inside any circle.
    for t in &triangles {
        for (v, p) in positions.iter().enumerate() {
            if t.contains(&v) {
                continue;
            }
            assert!(
                !strictly_inside_circumcircle(&positions, *t, *p),
                "vertex {v} lies inside the circumcircle of {t:?}"
            );
        }
    }
    // Another pass finds nothing left to do.
    assert_eq!(flip_edges_2d(&mut triangles, &positions), 0);
}

#[test]
fn the_flip_sequence_is_deterministic() {
    let positions = rhombus();
    let run = || {
        let mut triangles = [[0usize, 1, 2], [0, 2, 3]];
        let flips = flip_edges_2d(&mut triangles, &positions);
        (flips, triangles)
    };
    let first = run();
    for _ in 0..20 {
        assert_eq!(run(), first);
    }
}

#[test]
fn random_triangle_soups_never_break_the_flip_loop() {
    let mut rng = Lcg(0xFEED_FACE_0BAD_5EED);
    for case in 0..400usize {
        let vertex_count = 1 + (rng.next_unit() * 12.0) as usize;
        let positions: Vec<[f64; 3]> = (0..vertex_count)
            .map(|_| {
                [
                    rng.next_unit() * 4.0 - 2.0,
                    rng.next_unit() * 4.0 - 2.0,
                    0.0,
                ]
            })
            .collect();
        // Every fifth case references vertices that do not exist.
        let index_range = vertex_count + if case % 5 == 0 { 2 } else { 0 };
        let triangle_count = (rng.next_unit() * 24.0) as usize;
        let mut triangles: Vec<[usize; 3]> = (0..triangle_count)
            .map(|_| {
                [
                    (rng.next_unit() * index_range as f64) as usize,
                    (rng.next_unit() * index_range as f64) as usize,
                    (rng.next_unit() * index_range as f64) as usize,
                ]
            })
            .collect();
        let before = triangles.clone();
        let flips = flip_edges_2d(&mut triangles, &positions);
        assert_eq!(triangles.len(), before.len(), "case {case}");
        assert!(flips <= 4 * before.len() * before.len() + 64, "case {case}");
        if before.iter().flatten().any(|&v| v >= vertex_count) {
            assert_eq!(flips, 0, "case {case}");
            assert_eq!(triangles, before, "case {case}");
        }
        // A flip only exchanges the diagonal of two triangles: the multiset of
        // vertices of the whole triangle list is unchanged by pairs of flips, but
        // every triangle still has three distinct vertices if it had before.
        for (after, original) in triangles.iter().zip(before.iter()) {
            let proper = |t: &[usize; 3]| t[0] != t[1] && t[1] != t[2] && t[0] != t[2];
            assert_eq!(proper(after), proper(original), "case {case}");
        }
    }
}
