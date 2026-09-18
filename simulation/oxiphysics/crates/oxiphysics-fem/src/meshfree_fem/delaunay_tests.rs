// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the Delaunay background cells (F1235): the cells must tile the convex
//! hull exactly and satisfy the empty-circumcircle property, which the previous
//! "nearest neighbours" routine did neither.

use std::collections::HashSet;

use oxiphysics_core::exact_predicates::{Orientation, incircle, orient2d};

use super::functions::Point2;
use super::*;

/// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn random_points(n: usize, seed: u64) -> Vec<Point2> {
    let mut state = seed;
    (0..n)
        .map(|_| [lcg_unit(&mut state), lcg_unit(&mut state)])
        .collect()
}

fn cross(o: Point2, a: Point2, b: Point2) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Strict convex hull (Andrew's monotone chain), counter-clockwise.
fn convex_hull(points: &[Point2]) -> Vec<Point2> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let mut lower: Vec<Point2> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point2> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Area of the convex hull, measured from its first vertex (exact for the
/// translation by a large offset).
fn hull_area(points: &[Point2]) -> f64 {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return 0.0;
    }
    0.5 * (1..hull.len() - 1)
        .map(|i| cross(hull[0], hull[i], hull[i + 1]))
        .sum::<f64>()
}

/// Number of distinct points on the boundary of the hull (vertices and points in the
/// relative interior of a hull edge).
fn hull_boundary_count(points: &[Point2]) -> usize {
    let hull = convex_hull(points);
    let mut distinct = points.to_vec();
    distinct.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    distinct.dedup();
    distinct
        .iter()
        .filter(|&&p| {
            (0..hull.len()).any(|i| {
                let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
                if p == a || p == b {
                    return true;
                }
                orient2d(a, b, p) == Orientation::Degenerate
                    && p[0] >= a[0].min(b[0])
                    && p[0] <= a[0].max(b[0])
                    && p[1] >= a[1].min(b[1])
                    && p[1] <= a[1].max(b[1])
            })
        })
        .count()
}

/// Check every property of a Delaunay triangulation of `points` and return the sum of
/// the triangle areas.
fn check_delaunay(points: &[Point2], triangles: &[[usize; 3]]) -> f64 {
    let mut directed_edges = HashSet::new();
    let mut area = 0.0;
    for &[a, b, c] in triangles {
        let (pa, pb, pc) = (points[a], points[b], points[c]);
        assert!(
            a != b && b != c && a != c,
            "degenerate triangle {a} {b} {c}"
        );
        assert_eq!(
            orient2d(pa, pb, pc),
            Orientation::Positive,
            "triangle {a} {b} {c} is not counter-clockwise"
        );
        for edge in [(a, b), (b, c), (c, a)] {
            assert!(
                directed_edges.insert(edge),
                "directed edge {edge:?} is used twice: the triangles overlap"
            );
        }
        area += 0.5 * cross(pa, pb, pc);
        // empty circumcircle: no point strictly inside (cocircular points are allowed)
        for (index, &p) in points.iter().enumerate() {
            if p == pa || p == pb || p == pc {
                continue;
            }
            assert_ne!(
                incircle(pa, pb, pc, p),
                Orientation::Positive,
                "point {index} lies inside the circumcircle of triangle {a} {b} {c}"
            );
        }
    }
    area
}

#[test]
fn fifty_random_points_tile_the_convex_hull_with_empty_circumcircles() {
    // The finding's acceptance test: Σ area = hull area to 1e-12 and every triangle
    // passes the empty-circumcircle check.
    let points = random_points(50, 0x5EED_0001);
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    let expected = hull_area(&points);
    assert!(
        (area - expected).abs() < 1e-12,
        "Σ area {area}, hull {expected}"
    );
    // Euler: a triangulation of n points with h on the hull boundary has 2n − 2 − h
    // triangles
    let h = hull_boundary_count(&points);
    assert_eq!(triangles.len(), 2 * 50 - 2 - h);
    // every point is a vertex
    let used: HashSet<usize> = triangles.iter().flatten().copied().collect();
    assert_eq!(used.len(), 50);

    // the same through the cell API
    let cells = generate_delaunay_cells(&points);
    assert_eq!(cells.len(), triangles.len());
    let cell_area: f64 = cells.iter().map(TriCell::area).sum();
    assert!(
        (cell_area - expected).abs() < 1e-12,
        "{cell_area} vs {expected}"
    );
}

#[test]
fn the_cells_are_never_missing_at_the_hull_for_many_inputs() {
    // A finite "super triangle" loses hull slivers; the exact ghost-vertex
    // construction must not, whatever the point distribution.
    for (seed, n) in [(1u64, 5usize), (2, 9), (3, 17), (4, 40), (5, 120), (6, 400)] {
        let points = random_points(n, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let triangles = delaunay_triangulation(&points);
        let area = check_delaunay(&points, &triangles);
        assert!(
            (area - hull_area(&points)).abs() < 1e-12,
            "seed {seed}: Σ area {area}, hull {}",
            hull_area(&points)
        );
        assert_eq!(
            triangles.len(),
            2 * n - 2 - hull_boundary_count(&points),
            "seed {seed}"
        );
    }
}

#[test]
fn a_regular_grid_with_cocircular_quadruples_is_triangulated_validly() {
    // Every unit square has four cocircular corners: ties must not corrupt the mesh.
    let mut points = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            points.push([i as f64, j as f64]);
        }
    }
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    assert!((area - 25.0).abs() < 1e-12, "area {area}");
    assert_eq!(triangles.len(), 50);
    assert_eq!(hull_boundary_count(&points), 20);
}

#[test]
fn points_on_the_hull_boundary_are_kept() {
    // A 2 x 2 square with the four edge midpoints and the centre: the midpoints are
    // collinear hull points that a naive construction drops or turns into slivers.
    let points = vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    assert!((area - 4.0).abs() < 1e-14, "area {area}");
    assert_eq!(triangles.len(), 2 * 9 - 2 - 8);
    let used: HashSet<usize> = triangles.iter().flatten().copied().collect();
    assert_eq!(used.len(), 9, "all nine points are vertices");
}

#[test]
fn cocircular_points_give_a_triangulation_of_the_polygon() {
    let n = 40;
    let points: Vec<Point2> = (0..n)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [3.0 * angle.cos(), 3.0 * angle.sin()]
        })
        .collect();
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    assert_eq!(triangles.len(), n - 2);
    assert!((area - hull_area(&points)).abs() < 1e-12, "{area}");
}

#[test]
fn sorted_columns_and_collinear_starts_are_handled() {
    // The first points in lexicographic order are collinear (a vertical column): the
    // seed triangle needs a point from the second column, and the column points are
    // inserted afterwards on the line.
    let mut points = Vec::new();
    for j in 0..7 {
        points.push([0.0, j as f64 * 0.5]);
    }
    for (i, x) in [1.0, 2.5, 3.0].iter().enumerate() {
        for j in 0..4 {
            points.push([*x, j as f64 * 0.8 + 0.1 * i as f64]);
        }
    }
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    assert!((area - hull_area(&points)).abs() < 1e-12, "{area}");
    assert_eq!(
        triangles.len(),
        2 * points.len() - 2 - hull_boundary_count(&points)
    );
}

#[test]
fn a_large_coordinate_offset_does_not_change_the_triangulation() {
    // Orientation and in-circle decisions are exact, so a translation by 1e9 only
    // changes the rounding of the coordinates themselves.
    let base = random_points(80, 77);
    let shifted: Vec<Point2> = base.iter().map(|p| [p[0] + 1.0e9, p[1] - 1.0e9]).collect();
    let triangles = delaunay_triangulation(&shifted);
    let area = check_delaunay(&shifted, &triangles);
    assert!((area - hull_area(&shifted)).abs() < 1e-12, "{area}");
    assert_eq!(triangles.len(), 2 * 80 - 2 - hull_boundary_count(&shifted));
}

#[test]
fn degenerate_inputs_give_no_cells() {
    assert!(delaunay_triangulation(&[]).is_empty());
    assert!(delaunay_triangulation(&[[0.0, 0.0]]).is_empty());
    assert!(delaunay_triangulation(&[[0.0, 0.0], [1.0, 1.0]]).is_empty());
    // all collinear: axis-aligned, diagonal and oblique
    let line: Vec<Point2> = (0..20).map(|i| [i as f64, 2.0 * i as f64 + 1.0]).collect();
    assert!(delaunay_triangulation(&line).is_empty());
    let horizontal: Vec<Point2> = (0..20).map(|i| [i as f64 * 0.1, 3.0]).collect();
    assert!(generate_delaunay_cells(&horizontal).is_empty());
    // the same point repeated
    assert!(delaunay_triangulation(&[[1.0, 2.0]; 7]).is_empty());
}

#[test]
fn non_finite_and_repeated_points_are_ignored() {
    let clean = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.4]];
    let reference = delaunay_triangulation(&clean);
    let reference_cells = generate_delaunay_cells(&clean);

    let mut noisy = clean.clone();
    noisy.push([f64::NAN, 0.5]);
    noisy.push([0.5, f64::INFINITY]);
    noisy.push([-0.0, 0.0]); // the same point as the first one
    noisy.push([1.0, 0.0]);
    let triangles = delaunay_triangulation(&noisy);
    assert_eq!(triangles.len(), reference.len());
    // the triangles only refer to the first occurrences, and cover the same area
    assert!(triangles.iter().flatten().all(|&i| i < clean.len()));
    let area = check_delaunay(&noisy[..clean.len()], &triangles);
    assert!((area - 1.0).abs() < 1e-14, "{area}");
    let noisy_cells = generate_delaunay_cells(&noisy);
    assert_eq!(noisy_cells.len(), reference_cells.len());
}

#[test]
fn generated_cells_reference_the_given_node_positions() {
    let nodes = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.87], [0.5, 0.3]];
    let triangles = delaunay_triangulation(&nodes);
    let cells = generate_delaunay_cells(&nodes);
    assert_eq!(cells.len(), triangles.len());
    for (cell, triangle) in cells.iter().zip(triangles.iter()) {
        for (vertex, &index) in cell.vertices.iter().zip(triangle.iter()) {
            assert_eq!(*vertex, nodes[index]);
        }
        assert!(cell.area() > 0.0);
    }
    // the interior node splits the triangle into three cells
    assert_eq!(cells.len(), 3);
    let area: f64 = cells.iter().map(TriCell::area).sum();
    assert!((area - 0.5 * 0.87).abs() < 1e-14, "{area}");
}

#[test]
fn delaunay_cells_of_a_meshfree_cloud_tile_the_domain() {
    // The cloud of a regular 9 x 9 meshfree discretisation of the unit square.
    let nodes: Vec<Point2> = generate_regular_nodes(0.0, 0.0, 1.0, 1.0, 9, 9, 2.5)
        .iter()
        .map(|node| node.pos)
        .collect();
    let cells = generate_delaunay_cells(&nodes);
    assert_eq!(cells.len(), 2 * 8 * 8);
    let area: f64 = cells.iter().map(TriCell::area).sum();
    assert!((area - 1.0).abs() < 1e-12, "{area}");
}

#[test]
fn thousands_of_points_are_triangulated_quickly_and_correctly() {
    // Larger input: also exercises the visibility walk over long distances.
    let points = random_points(2500, 0xC0FFEE);
    let triangles = delaunay_triangulation(&points);
    let area = check_delaunay(&points, &triangles);
    assert!((area - hull_area(&points)).abs() < 1e-11, "{area}");
    assert_eq!(triangles.len(), 2 * 2500 - 2 - hull_boundary_count(&points));
}
