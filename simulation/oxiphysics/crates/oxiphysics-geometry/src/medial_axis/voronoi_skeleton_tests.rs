// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of [`super::Voronoi2dSkeleton`] and [`super::voronoi_medial_axis_2d`]:
//! the output must be the Voronoi vertices (circumcentres of empty circles),
//! not the circumcentre of every point triplet.

use super::{Voronoi2dSkeleton, is_inside_polygon, point_to_polygon_dist, voronoi_medial_axis_2d};

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
}

fn random_points(rng: &mut XorShift, count: usize) -> Vec<[f64; 2]> {
    (0..count)
        .map(|_| [rng.next_f64(), rng.next_f64()])
        .collect()
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Circumcentre of three points, `None` when they are collinear.
fn circumcentre(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<[f64; 2]> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let a2 = a[0] * a[0] + a[1] * a[1];
    let b2 = b[0] * b[0] + b[1] * b[1];
    let c2 = c[0] * c[0] + c[1] * c[1];
    Some([
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ])
}

/// Brute-force Voronoi vertices inside `bbox`: circumcentres of the point
/// triplets whose circumcircle contains no other point (O(n^4)).
fn brute_force_vertices(points: &[[f64; 2]], bbox: [f64; 4]) -> Vec<[f64; 2]> {
    let n = points.len();
    let mut out: Vec<[f64; 2]> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let Some(c) = circumcentre(points[i], points[j], points[k]) else {
                    continue;
                };
                let r = dist(c, points[i]);
                let empty = points
                    .iter()
                    .enumerate()
                    .all(|(m, &q)| m == i || m == j || m == k || dist(c, q) >= r - 1e-9);
                let inside =
                    c[0] >= bbox[0] && c[1] >= bbox[1] && c[0] <= bbox[2] && c[1] <= bbox[3];
                if empty && inside && !out.iter().any(|&o| dist(o, c) < 1e-7) {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn assert_same_point_sets(found: &[[f64; 2]], expected: &[[f64; 2]], context: &str) {
    for &f in found {
        assert!(
            expected.iter().any(|&e| dist(e, f) < 1e-7),
            "{context}: spurious vertex {f:?}"
        );
    }
    for &e in expected {
        assert!(
            found.iter().any(|&f| dist(e, f) < 1e-7),
            "{context}: missing vertex {e:?}"
        );
    }
    assert_eq!(found.len(), expected.len(), "{context}: duplicate vertices");
}

#[test]
fn square_corners_have_exactly_one_voronoi_vertex() {
    let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let bbox = [-0.5, -0.5, 1.5, 1.5];
    let vertices = Voronoi2dSkeleton::compute(&corners, bbox);
    assert_eq!(vertices.len(), 1, "{vertices:?}");
    assert!(dist(vertices[0], [0.5, 0.5]) < 1e-12);

    // Input order, repeated points and non-finite points do not matter.
    let noisy = [
        [1.0, 1.0],
        [f64::NAN, 0.3],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [f64::INFINITY, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];
    assert_eq!(Voronoi2dSkeleton::compute(&noisy, bbox), vertices);
}

#[test]
fn bbox_clips_the_voronoi_vertices_inclusively() {
    let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    assert!(Voronoi2dSkeleton::compute(&corners, [0.0, 0.0, 0.4, 0.4]).is_empty());
    assert!(Voronoi2dSkeleton::compute(&corners, [0.6, 0.0, 2.0, 2.0]).is_empty());
    // The bounding box is closed: a vertex on its boundary is kept.
    let on_edge = Voronoi2dSkeleton::compute(&corners, [0.5, 0.5, 0.5, 0.5]);
    assert_eq!(on_edge.len(), 1);
}

#[test]
fn degenerate_point_sets_have_no_voronoi_vertices() {
    let bbox = [-10.0, -10.0, 10.0, 10.0];
    assert!(Voronoi2dSkeleton::compute(&[], bbox).is_empty());
    assert!(Voronoi2dSkeleton::compute(&[[0.0, 0.0]], bbox).is_empty());
    // Three input points but only two distinct ones.
    assert!(Voronoi2dSkeleton::compute(&[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], bbox).is_empty());
    // Collinear points: all Voronoi edges are parallel lines.
    let line: Vec<[f64; 2]> = (0..6).map(|i| [i as f64, 2.0 * i as f64]).collect();
    assert!(Voronoi2dSkeleton::compute(&line, bbox).is_empty());
}

#[test]
fn single_triangle_has_its_circumcentre_as_only_vertex() {
    let pts = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
    let vertices = Voronoi2dSkeleton::compute(&pts, [-10.0, -10.0, 10.0, 10.0]);
    assert_eq!(vertices.len(), 1);
    assert!(dist(vertices[0], [0.5, 0.375]) < 1e-12, "{vertices:?}");
}

#[test]
fn random_points_yield_only_empty_circle_vertices() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let n = 200;
    let points = random_points(&mut rng, n);
    let vertices = Voronoi2dSkeleton::compute(&points, [-10.0, -10.0, 10.0, 10.0]);

    // At most one vertex per Delaunay triangle, and there are at most 2n of
    // those; the old triplet enumeration produced ~1.3 million candidates.
    assert!(!vertices.is_empty());
    assert!(vertices.len() <= 2 * n, "{} vertices", vertices.len());

    for (idx, &v) in vertices.iter().enumerate() {
        let mut distances: Vec<f64> = points.iter().map(|&p| dist(v, p)).collect();
        distances.sort_by(f64::total_cmp);
        // Equidistant from its three nearest points ...
        assert!(
            distances[2] - distances[0] < 1e-9,
            "vertex {idx} {v:?}: nearest distances {:?}",
            &distances[..4]
        );
        // ... and no point lies strictly inside the circle.
        assert!(distances[3] >= distances[2] - 1e-9, "vertex {idx} {v:?}");
        for &w in &vertices[..idx] {
            assert!(dist(v, w) > 1e-9, "vertex {v:?} reported twice");
        }
    }
}

#[test]
fn random_points_match_a_brute_force_delaunay_oracle() {
    for seed in [1_u64, 7, 42, 2024, 99_991] {
        let mut rng = XorShift(0xD1B5_4A32_D192_ED03 ^ seed.wrapping_mul(0x9E37_79B9));
        let points = random_points(&mut rng, 30);
        // The bounding box of the points: every vertex inside it is found.
        let bbox = [0.0, 0.0, 1.0, 1.0];
        let found = Voronoi2dSkeleton::compute(&points, bbox);
        let expected = brute_force_vertices(&points, bbox);
        assert!(!expected.is_empty());
        assert_same_point_sets(&found, &expected, &format!("seed {seed}"));
    }
}

#[test]
fn grid_points_with_many_cocircular_quadruples_give_the_cell_centres() {
    // A 5x4 grid: every unit cell has four cocircular corners.
    let mut points = Vec::new();
    for ix in 0..5 {
        for iy in 0..4 {
            points.push([ix as f64, iy as f64]);
        }
    }
    let found = Voronoi2dSkeleton::compute(&points, [0.0, 0.0, 4.0, 3.0]);
    let mut expected = Vec::new();
    for ix in 0..4 {
        for iy in 0..3 {
            expected.push([ix as f64 + 0.5, iy as f64 + 0.5]);
        }
    }
    assert_same_point_sets(&found, &expected, "grid");
}

/// Closed outline of the rectangle `[0, w] x [0, h]` sampled every `step`,
/// counter-clockwise starting at the origin.
fn rectangle_outline(w: f64, h: f64, step: f64) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    let nx = (w / step).round() as usize;
    let ny = (h / step).round() as usize;
    for i in 0..nx {
        pts.push([i as f64 * step, 0.0]);
    }
    for j in 0..ny {
        pts.push([w, j as f64 * step]);
    }
    for i in 0..nx {
        pts.push([w - i as f64 * step, h]);
    }
    for j in 0..ny {
        pts.push([0.0, h - j as f64 * step]);
    }
    pts
}

fn dist_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]))
        .clamp(0.0, 1.0);
    dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

#[test]
fn medial_axis_of_a_square_is_its_centre_with_the_inscribed_radius() {
    let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let bbox = [-0.5, -0.5, 1.5, 1.5];
    let axis = voronoi_medial_axis_2d(&corners, bbox, 0.0);
    assert_eq!(axis.len(), 1, "{axis:?}");
    assert!(dist(axis[0].0, [0.5, 0.5]) < 1e-12);
    assert!((axis[0].1 - 0.5).abs() < 1e-12);
    // A minimum radius above the inscribed radius removes it.
    assert!(voronoi_medial_axis_2d(&corners, bbox, 0.51).is_empty());
}

#[test]
fn medial_axis_of_a_dense_rectangle_lies_on_the_rectangle_skeleton() {
    let outline = rectangle_outline(4.0, 2.0, 0.1);
    let bbox = [-1.0, -1.0, 5.0, 3.0];
    let axis = voronoi_medial_axis_2d(&outline, bbox, 0.2);
    assert!(axis.len() > 20, "{} vertices", axis.len());

    // The medial axis of a 4 x 2 rectangle: four corner bisectors and the
    // ridge y = 1 between x = 1 and x = 3.
    let segments = [
        ([0.0, 0.0], [1.0, 1.0]),
        ([0.0, 2.0], [1.0, 1.0]),
        ([4.0, 0.0], [3.0, 1.0]),
        ([4.0, 2.0], [3.0, 1.0]),
        ([1.0, 1.0], [3.0, 1.0]),
    ];
    for &(v, r) in &axis {
        assert!(is_inside_polygon(v, &outline), "{v:?} is outside the shape");
        assert!((r - point_to_polygon_dist(v, &outline)).abs() < 1e-12);
        assert!(r >= 0.2);
        let d = segments
            .iter()
            .map(|&(a, b)| dist_to_segment(v, a, b))
            .fold(f64::INFINITY, f64::min);
        assert!(d < 0.06, "vertex {v:?} (radius {r}) is {d} from the axis");
    }

    // The ridge is covered: a vertex with the full half-width as radius sits
    // on y = 1 for every probed x.
    for x in [1.5, 2.0, 2.5] {
        let hit = axis
            .iter()
            .any(|&(v, r)| (v[0] - x).abs() < 0.06 && (v[1] - 1.0).abs() < 0.02 && r > 0.98);
        assert!(hit, "no ridge vertex near x = {x}");
    }
}

#[test]
fn medial_axis_skips_voronoi_vertices_outside_the_shape() {
    // An L-shaped outline: the Voronoi vertices of its samples also exist in
    // the notch outside the shape, which must not be reported.
    let mut outline = Vec::new();
    let corners = [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.0],
        [1.0, 1.0],
        [1.0, 4.0],
        [0.0, 4.0],
    ];
    for i in 0..corners.len() {
        let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
        let steps = (dist(a, b) / 0.1).round() as usize;
        for s in 0..steps {
            let t = s as f64 / steps as f64;
            outline.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    let bbox = [-1.0, -1.0, 5.0, 5.0];
    let everything = Voronoi2dSkeleton::compute(&outline, bbox);
    let axis = voronoi_medial_axis_2d(&outline, bbox, 0.0);
    assert!(
        axis.len() < everything.len(),
        "the notch has Voronoi vertices"
    );
    assert!(axis.iter().all(|&(v, _)| is_inside_polygon(v, &outline)));
    assert!(
        everything
            .iter()
            .any(|&v| !is_inside_polygon(v, &outline) && dist(v, [2.0, 2.0]) < 3.0)
    );
}
