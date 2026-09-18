// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the shared Quickhull core.
//!
//! Three kinds of checks: the structural guarantees of the surface (closed,
//! consistently wound, outward, encloses the input), agreement with a plain
//! insertion-order hull built here as an independent oracle, and the internals
//! (seed selection, planes, the horizon loop test, the fallback for a point whose
//! visible region is unusable).

use super::*;
use crate::convex_hull::ConvexHull3D;
use crate::quickhull::ConvexHull3DVec;
use oxiphysics_core::exact_predicates::{Orientation, orient3d};
use oxiphysics_core::math::Vec3;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;

// ── helpers ───────────────────────────────────────────────────────────────

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }
}

fn shuffled<T: Clone>(items: &[T], rng: &mut Rng) -> Vec<T> {
    let mut out = items.to_vec();
    for i in (1..out.len()).rev() {
        let j = rng.below(i + 1);
        out.swap(i, j);
    }
    out
}

fn uniform_cube(rng: &mut Rng, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ]
        })
        .collect()
}

fn uniform_ball(rng: &mut Rng, n: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        if dot(p, p) <= 1.0 {
            out.push(p);
        }
    }
    out
}

/// Points on the unit sphere (all of them are hull vertices).
fn sphere_surface(rng: &mut Rng, n: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let length = dot(p, p).sqrt();
        if length > 0.1 && length <= 1.0 {
            out.push([p[0] / length, p[1] / length, p[2] / length]);
        }
    }
    out
}

/// Fibonacci lattice on the unit sphere.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).sqrt();
            let angle = golden * i as f64;
            [r * angle.cos(), y, r * angle.sin()]
        })
        .collect()
}

/// A cylinder: 12 angles, three heights (the middle ring lies on the side
/// faces).  Many coplanar, cocircular and collinear points.
fn cylinder(radius: f64, half_height: f64) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for k in 0..12 {
        let angle = 2.0 * PI * k as f64 / 12.0;
        for z in [-half_height, 0.0, half_height] {
            out.push([radius * angle.cos(), radius * angle.sin(), z]);
        }
    }
    out
}

fn lattice(nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                out.push([x as f64, y as f64, z as f64]);
            }
        }
    }
    out
}

fn rotated(points: &[[f64; 3]], axis: [f64; 3], angle: f64) -> Vec<[f64; 3]> {
    let length = dot(axis, axis).sqrt();
    let k = [axis[0] / length, axis[1] / length, axis[2] / length];
    let (s, c) = angle.sin_cos();
    points
        .iter()
        .map(|&p| {
            // Rodrigues' rotation formula.
            let kxp = cross(k, p);
            let kdp = dot(k, p);
            [
                p[0] * c + kxp[0] * s + k[0] * kdp * (1.0 - c),
                p[1] * c + kxp[1] * s + k[1] * kdp * (1.0 - c),
                p[2] * c + kxp[2] * s + k[2] * kdp * (1.0 - c),
            ]
        })
        .collect()
}

fn longest_side(points: &[[f64; 3]]) -> f64 {
    cloud_extent(points).map_or(0.0, |extent| extent.longest_side)
}

fn tolerances(points: &[[f64; 3]]) -> HullTolerances {
    let extent = longest_side(points);
    HullTolerances {
        visible_distance: 1e-10 * extent,
        seed_distance: 1e-10 * extent,
    }
}

/// Volume enclosed by a closed outward triangle surface.
fn volume(vertices: &[[f64; 3]], faces: &[[usize; 3]]) -> f64 {
    let n = vertices.len() as f64;
    let centre = vertices.iter().fold([0.0; 3], |sum, v| {
        [sum[0] + v[0], sum[1] + v[1], sum[2] + v[2]]
    });
    let centre = [centre[0] / n, centre[1] / n, centre[2] / n];
    let six: f64 = faces
        .iter()
        .map(|f| {
            let a = sub(vertices[f[0]], centre);
            let b = sub(vertices[f[1]], centre);
            let c = sub(vertices[f[2]], centre);
            dot(a, cross(b, c))
        })
        .sum();
    six / 6.0
}

/// The surface must be a closed, consistently wound 2-manifold (every directed
/// edge once and matched by its reverse, Euler characteristic 2) without
/// zero-area faces, with every face looking away from the centre, and no input
/// point (all of them, or a sample of `sample` of them) may lie in front of a
/// face plane by more than `tolerance_factor * 1e-10 * extent`.
fn assert_valid_surface(
    points: &[[f64; 3]],
    surface: &HullSurface,
    sample: Option<usize>,
    tolerance_factor: f64,
    label: &str,
) {
    assert_closed_manifold(surface, label);
    if surface.vertices.len() <= 600 {
        let tolerance = tolerance_factor.max(2.0) * 1e-10 * longest_side(points);
        assert_convex_within(surface, tolerance, label);
    }
    assert_outward_and_enclosing(points, surface, sample, tolerance_factor, label);
}

/// Every directed edge occurs once and is matched by its reverse, every vertex is
/// used, and the Euler characteristic is 2.
fn assert_closed_manifold(surface: &HullSurface, label: &str) {
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for face in &surface.faces {
        for k in 0..3 {
            *directed.entry((face[k], face[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    for (&(a, b), &count) in &directed {
        assert_eq!(count, 1, "{label}: edge ({a},{b}) used {count} times");
        assert!(
            directed.contains_key(&(b, a)),
            "{label}: edge ({a},{b}) has no opposite edge"
        );
    }
    let used: BTreeSet<usize> = surface.faces.iter().flatten().copied().collect();
    assert_eq!(used.len(), surface.vertices.len(), "{label}: unused vertex");
    let (v, e, f) = (
        surface.vertices.len() as i64,
        (directed.len() / 2) as i64,
        surface.faces.len() as i64,
    );
    assert_eq!(
        v - e + f,
        2,
        "{label}: Euler characteristic V={v} E={e} F={f}"
    );
}

/// No vertex of the surface lies more than `tolerance` in front of the plane of a
/// face: the surface is convex up to the tolerance.  Thin faces (whose rounded
/// plane is unreliable, see [`THIN_FACE_QUALITY`]) are skipped.
fn assert_convex_within(surface: &HullSurface, tolerance: f64, label: &str) {
    for face in &surface.faces {
        let a = surface.vertices[face[0]];
        let b = surface.vertices[face[1]];
        let c = surface.vertices[face[2]];
        let (normal, offset, quality) = plane_through(a, b, c);
        if quality < THIN_FACE_QUALITY {
            continue;
        }
        for (i, &v) in surface.vertices.iter().enumerate() {
            let height = dot(normal, v) - offset;
            assert!(
                height <= tolerance,
                "{label}: vertex {i} lies {height} in front of face {face:?}"
            );
        }
    }
}

/// No face has zero area or looks towards the centre, and no input point (all of
/// them or a sample) lies in front of a face plane by more than the tolerance.
fn assert_outward_and_enclosing(
    points: &[[f64; 3]],
    surface: &HullSurface,
    sample: Option<usize>,
    tolerance_factor: f64,
    label: &str,
) {
    let n = surface.vertices.len() as f64;
    let centre = surface.vertices.iter().fold([0.0; 3], |sum, p| {
        [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]]
    });
    let centre = [centre[0] / n, centre[1] / n, centre[2] / n];
    let tolerance = tolerance_factor * 1e-10 * longest_side(points);
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let probes: Vec<[f64; 3]> = match sample {
        None => points.to_vec(),
        Some(count) => (0..count)
            .map(|_| points[rng.below(points.len())])
            .collect(),
    };
    for face in &surface.faces {
        let a = surface.vertices[face[0]];
        let normal = cross(
            sub(surface.vertices[face[1]], a),
            sub(surface.vertices[face[2]], a),
        );
        let length = dot(normal, normal).sqrt();
        assert!(length > 0.0, "{label}: zero-area face {face:?}");
        let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
        assert!(
            dot(unit, sub(centre, a)) < 0.0,
            "{label}: face {face:?} looks inward"
        );
        for &p in &probes {
            assert!(
                dot(unit, sub(p, a)) <= tolerance,
                "{label}: point {p:?} lies {} in front of face {face:?}",
                dot(unit, sub(p, a))
            );
        }
    }
}

fn build(points: &[[f64; 3]]) -> HullSurface {
    build_hull(points, tolerances(points)).expect("the cloud spans a volume")
}

/// Indices (into `points`) of the vertices of `surface`; the points must be
/// pairwise distinct.
fn vertex_indices(points: &[[f64; 3]], surface: &HullSurface) -> BTreeSet<usize> {
    surface
        .vertices
        .iter()
        .map(|v| {
            points
                .iter()
                .position(|p| p == v)
                .expect("vertex is an input point")
        })
        .collect()
}

/// Hull of the points by inserting them in input order and testing each against
/// every face: the simple algorithm the core replaces, used as an oracle.
fn reference_triangles(points: &[[f64; 3]], tolerances: HullTolerances) -> Vec<[usize; 3]> {
    let (low, high) = bounding_box(points).expect("finite points");
    let side = sub(high, low);
    let axis = if side[0] >= side[1] && side[0] >= side[2] {
        0
    } else if side[1] >= side[2] {
        1
    } else {
        2
    };
    let seed = seed_tetrahedron(points, axis, tolerances.seed_distance).expect("seed");
    let inside = seed.iter().fold([0.0; 3], |sum, &i| {
        [
            sum[0] + 0.25 * points[i][0],
            sum[1] + 0.25 * points[i][1],
            sum[2] + 0.25 * points[i][2],
        ]
    });
    let distance = |triangle: &[usize; 3], p: [f64; 3]| {
        let (normal, offset, _) = plane_through(
            points[triangle[0]],
            points[triangle[1]],
            points[triangle[2]],
        );
        dot(normal, p) - offset
    };
    let [s0, s1, s2, s3] = seed;
    let mut faces: Vec<[usize; 3]> = Vec::new();
    for mut triangle in [[s0, s1, s2], [s0, s3, s1], [s0, s2, s3], [s1, s3, s2]] {
        if distance(&triangle, inside) > 0.0 {
            triangle.swap(1, 2);
        }
        faces.push(triangle);
    }
    for (i, &p) in points.iter().enumerate() {
        if seed.contains(&i) {
            continue;
        }
        let visible: Vec<bool> = faces
            .iter()
            .map(|triangle| distance(triangle, p) > tolerances.visible_distance)
            .collect();
        if !visible.contains(&true) {
            continue;
        }
        let visible_edges: HashSet<(usize, usize)> = faces
            .iter()
            .zip(&visible)
            .filter(|&(_, &seen)| seen)
            .flat_map(|(t, _)| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .collect();
        let horizon: Vec<(usize, usize)> = visible_edges
            .iter()
            .copied()
            .filter(|&(a, b)| !visible_edges.contains(&(b, a)))
            .collect();
        faces = faces
            .into_iter()
            .zip(visible)
            .filter(|(_, seen)| !seen)
            .map(|(t, _)| t)
            .collect();
        faces.extend(horizon.into_iter().map(|(a, b)| [a, b, i]));
    }
    faces
}

fn reference_vertices(triangles: &[[usize; 3]]) -> BTreeSet<usize> {
    triangles.iter().flatten().copied().collect()
}

fn reference_volume(points: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    volume(points, triangles)
}

// ── structural guarantees ─────────────────────────────────────────────────

#[test]
fn cube_corners_in_every_order_give_the_exact_cube() {
    let mut rng = Rng(11);
    let corners = lattice(2, 2, 2);
    for round in 0..40 {
        let points = shuffled(&corners, &mut rng);
        let surface = build(&points);
        let label = format!("cube round {round}");
        assert_valid_surface(&points, &surface, None, 1.0, &label);
        assert_eq!(surface.vertices.len(), 8, "{label}");
        assert_eq!(surface.faces.len(), 12, "{label}");
        let want = 1.0;
        assert!(
            (volume(&surface.vertices, &surface.faces) - want).abs() < 1e-12,
            "{label}"
        );
    }
}

#[test]
fn lattices_have_only_their_corners_as_vertices_in_every_order() {
    // Every face of the lattice carries many coplanar points and every edge many
    // collinear ones; exactly tied candidates are resolved lexicographically, so
    // only the eight corners ever become vertices, in whatever order the points
    // arrive.
    let mut rng = Rng(12);
    for &(nx, ny, nz) in &[(3, 3, 3), (4, 4, 4), (5, 2, 3), (2, 6, 4)] {
        let mut corners: BTreeSet<[i64; 3]> = BTreeSet::new();
        for &x in &[0, nx - 1] {
            for &y in &[0, ny - 1] {
                for &z in &[0, nz - 1] {
                    corners.insert([x as i64, y as i64, z as i64]);
                }
            }
        }
        let base = lattice(nx, ny, nz);
        for round in 0..25 {
            let points = shuffled(&base, &mut rng);
            let surface = build(&points);
            let label = format!("lattice {nx}x{ny}x{nz} round {round}");
            assert_valid_surface(&points, &surface, None, 1.0, &label);
            let vertices: BTreeSet<[i64; 3]> = surface
                .vertices
                .iter()
                .map(|v| [v[0] as i64, v[1] as i64, v[2] as i64])
                .collect();
            assert_eq!(vertices, corners, "{label}");
            assert_eq!(surface.faces.len(), 12, "{label}");
            let want = ((nx - 1) * (ny - 1) * (nz - 1)) as f64;
            assert!(
                (volume(&surface.vertices, &surface.faces) - want).abs() < 1e-10,
                "{label}"
            );
        }
    }
}

#[test]
fn cylinder_with_collinear_and_cocircular_points_keeps_every_rim_vertex() {
    // 12 angles at three heights: the two rims are 24 corners, the middle ring
    // lies in the side faces.  A point inside a flat face or on an edge may stay
    // a vertex of the triangulation, but every rim point must be one and nothing
    // else may be.
    let mut rng = Rng(13);
    let base = cylinder(1.0, 2.0);
    for round in 0..25 {
        let points = shuffled(&base, &mut rng);
        let surface = build(&points);
        let label = format!("cylinder round {round}");
        assert_valid_surface(&points, &surface, None, 1.0, &label);
        let rim = surface
            .vertices
            .iter()
            .filter(|v| v[2].abs() == 2.0)
            .count();
        assert_eq!(rim, 24, "{label}: every rim point is a corner");
        assert!(
            surface
                .vertices
                .iter()
                .all(|v| v[2].abs() == 2.0 || v[2] == 0.0),
            "{label}"
        );
        // Regular 12-gon of circumradius 1 (area 6 sin 30 degrees) times height 4.
        let want = 6.0 * (2.0 * PI / 12.0).sin() * 4.0;
        assert!(
            (volume(&surface.vertices, &surface.faces) - want).abs() < 1e-12,
            "{label}"
        );
    }
}

#[test]
fn rotated_lattice_with_jittered_duplicates_is_a_valid_hull() {
    // Coplanar only up to rounding error, and every point repeated with a 1e-14
    // jitter: signs of distances to "coplanar" planes are noise.
    let mut rng = Rng(14);
    let base = rotated(&lattice(3, 3, 3), [1.0, 2.0, 3.0], 0.7);
    for round in 0..30 {
        let mut points = base.clone();
        for p in &base {
            points.push([
                p[0] + 1e-14 * (rng.unit() - 0.5),
                p[1] + 1e-14 * (rng.unit() - 0.5),
                p[2] + 1e-14 * (rng.unit() - 0.5),
            ]);
        }
        let points = shuffled(&points, &mut rng);
        let surface = build(&points);
        let label = format!("rotated lattice round {round}");
        assert_valid_surface(&points, &surface, None, 1.0, &label);
        assert!(
            (volume(&surface.vertices, &surface.faces) - 8.0).abs() < 1e-8,
            "{label}"
        );
    }
}

#[test]
fn noise_at_the_scale_of_the_tolerance_keeps_the_surface_closed() {
    // Points of a lattice displaced by amounts around the visibility tolerance
    // (1e-10 of the extent): which of them stick out is arbitrary and the surface
    // may contain slivers whose orientation is rounding noise, but it must stay a
    // closed manifold that encloses the volume of the lattice.
    let mut rng = Rng(15);
    let base = lattice(4, 4, 4);
    for &noise in &[3e-11, 1e-10, 3e-10, 1e-9] {
        for round in 0..12 {
            let points: Vec<[f64; 3]> = shuffled(&base, &mut rng)
                .iter()
                .map(|p| {
                    [
                        p[0] + noise * 3.0 * (rng.unit() - 0.5),
                        p[1] + noise * 3.0 * (rng.unit() - 0.5),
                        p[2] + noise * 3.0 * (rng.unit() - 0.5),
                    ]
                })
                .collect();
            let surface = build(&points);
            let label = format!("noise {noise} round {round}");
            assert_closed_manifold(&surface, &label);
            assert_convex_within(&surface, 2.0 * 1e-10 * longest_side(&points), &label);
            let enclosed = volume(&surface.vertices, &surface.faces);
            assert!((enclosed - 27.0).abs() < 1e-6, "{label}: volume {enclosed}");
        }
    }
}

#[test]
fn flat_slabs_needles_and_clusters_are_valid_hulls() {
    let mut rng = Rng(16);
    for &thickness in &[1e-1, 1e-3, 1e-5] {
        let slab: Vec<[f64; 3]> = (0..200)
            .map(|_| {
                [
                    rng.range(-3.0, 3.0),
                    rng.range(-1.0, 1.0),
                    rng.range(0.0, thickness),
                ]
            })
            .collect();
        let surface = build(&slab);
        assert_valid_surface(&slab, &surface, None, 1.0, &format!("slab {thickness}"));
    }
    let needle: Vec<[f64; 3]> = (0..120)
        .map(|_| {
            [
                rng.range(0.0, 10.0),
                rng.range(0.0, 1e-3),
                rng.range(0.0, 1e-3),
            ]
        })
        .collect();
    let surface = build(&needle);
    assert_valid_surface(&needle, &surface, None, 1.0, "needle");
    // Three tight clusters far apart, with exact duplicates.
    let centres = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 40.0, 30.0]];
    let mut clusters = Vec::new();
    for centre in centres {
        for _ in 0..30 {
            clusters.push([
                centre[0] + rng.range(-1e-3, 1e-3),
                centre[1] + rng.range(-1e-3, 1e-3),
                centre[2] + rng.range(-1e-3, 1e-3),
            ]);
        }
    }
    let duplicates = clusters.clone();
    clusters.extend(duplicates);
    let clusters = shuffled(&clusters, &mut rng);
    let surface = build(&clusters);
    assert_valid_surface(&clusters, &surface, None, 1.0, "clusters");
}

#[test]
fn big_spheres_have_every_point_as_a_vertex() {
    // All points of the Fibonacci lattice and of a random sample are corners:
    // the hull is a triangulated sphere with n vertices and 2n - 4 faces.  The
    // insertion-order algorithm would test every point against about 2n faces,
    // billions of tests at this size.
    let mut rng = Rng(17);
    for points in [fibonacci_sphere(20_000), sphere_surface(&mut rng, 20_000)] {
        let n = points.len();
        let surface = build(&points);
        assert_eq!(surface.vertices.len(), n);
        assert_eq!(surface.faces.len(), 2 * n - 4);
        assert_valid_surface(&points, &surface, Some(150), 1.0, "sphere");
        let ball = 4.0 / 3.0 * PI;
        let v = volume(&surface.vertices, &surface.faces);
        assert!(v < ball && v > 0.999 * ball, "volume {v}");
    }
}

// ── agreement with the insertion-order oracle ─────────────────────────────

#[test]
fn generic_clouds_match_the_insertion_order_hull() {
    let mut rng = Rng(21);
    for &n in &[8_usize, 30, 120, 400] {
        for kind in 0..4 {
            let points = match kind {
                0 => uniform_cube(&mut rng, n),
                1 => uniform_ball(&mut rng, n),
                2 => sphere_surface(&mut rng, n),
                _ => sphere_surface(&mut rng, n)
                    .iter()
                    .map(|p| [5.0 * p[0], p[1], 0.2 * p[2]])
                    .collect(),
            };
            let label = format!("kind {kind}, n {n}");
            let surface = build(&points);
            assert_valid_surface(&points, &surface, None, 1.0, &label);
            let reference = reference_triangles(&points, tolerances(&points));
            let expected = reference_vertices(&reference);
            assert_eq!(vertex_indices(&points, &surface), expected, "{label}");
            let want = reference_volume(&points, &reference);
            let got = volume(&surface.vertices, &surface.faces);
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "{label}: volume {got} vs {want}"
            );
            // Generic clouds have simplicial hulls, so even the face count agrees.
            assert_eq!(surface.faces.len(), reference.len(), "{label}");
        }
    }
}

#[test]
fn degenerate_clouds_match_the_insertion_order_volume() {
    let mut rng = Rng(22);
    let rotated_lattice = rotated(&lattice(4, 3, 3), [0.3, -1.0, 2.0], 1.1);
    let cases: Vec<(&str, Vec<[f64; 3]>)> = vec![
        ("lattice", shuffled(&lattice(4, 4, 4), &mut rng)),
        ("rotated lattice", shuffled(&rotated_lattice, &mut rng)),
        ("cylinder", shuffled(&cylinder(1.5, 1.0), &mut rng)),
    ];
    for (label, points) in cases {
        let surface = build(&points);
        let reference = reference_triangles(&points, tolerances(&points));
        let want = reference_volume(&points, &reference);
        let got = volume(&surface.vertices, &surface.faces);
        assert!(
            (got - want).abs() <= 1e-9 * want,
            "{label}: volume {got} vs {want}"
        );
    }
}

#[test]
fn hull_does_not_depend_on_position_scale_or_order() {
    let mut rng = Rng(23);
    let cloud = uniform_ball(&mut rng, 200);
    let reference = vertex_indices(&cloud, &build(&cloud));
    assert!(reference.len() > 20);
    // Moved far from the origin and scaled: the same points are corners.
    for &(shift, factor) in &[(0.0, 1.0), (1e6, 1.0), (-3e2, 1e-1), (10.0, 1e4)] {
        let moved: Vec<[f64; 3]> = cloud
            .iter()
            .map(|p| [shift + factor * p[0], shift - factor * p[1], factor * p[2]])
            .collect();
        let surface = build(&moved);
        assert_valid_surface(
            &moved,
            &surface,
            None,
            1.0,
            &format!("shift {shift} scale {factor}"),
        );
        assert_eq!(
            vertex_indices(&moved, &surface),
            reference,
            "shift {shift} scale {factor}"
        );
    }
    // Any order of the input gives the same corners.
    for round in 0..10 {
        let order: Vec<usize> = shuffled(&(0..cloud.len()).collect::<Vec<_>>(), &mut rng);
        let permuted: Vec<[f64; 3]> = order.iter().map(|&i| cloud[i]).collect();
        let surface = build(&permuted);
        let corners: BTreeSet<usize> = vertex_indices(&permuted, &surface)
            .into_iter()
            .map(|i| order[i])
            .collect();
        assert_eq!(corners, reference, "round {round}");
    }
}

#[test]
fn both_public_builders_share_the_core() {
    let mut rng = Rng(24);
    for &n in &[10_usize, 60, 300] {
        let cloud = uniform_ball(&mut rng, n);
        let arrays = ConvexHull3D::build(&cloud).expect("ConvexHull3D");
        let vectors: Vec<Vec3> = cloud.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
        let nalgebra = ConvexHull3DVec::build(&vectors).expect("ConvexHull3DVec");
        assert_eq!(arrays.faces, nalgebra.faces, "n {n}");
        assert_eq!(arrays.vertices.len(), nalgebra.vertices.len(), "n {n}");
        for (a, b) in arrays.vertices.iter().zip(&nalgebra.vertices) {
            assert_eq!(*a, [b.x, b.y, b.z], "n {n}");
        }
        assert!((arrays.volume() - nalgebra.volume()).abs() <= 1e-12 * arrays.volume());
    }
}

#[test]
fn stress_random_families_never_break_the_surface() {
    let mut rng = Rng(25);
    for round in 0..300 {
        let n = 5 + rng.below(120);
        let family = rng.below(6);
        let mut points = match family {
            0 => uniform_cube(&mut rng, n),
            1 => uniform_ball(&mut rng, n),
            2 => sphere_surface(&mut rng, n),
            3 => {
                // Points on a few planes and lines: a quantised cloud.
                (0..n)
                    .map(|_| {
                        [
                            rng.below(4) as f64,
                            rng.below(4) as f64,
                            rng.below(3) as f64,
                        ]
                    })
                    .collect()
            }
            4 => {
                // Flat-ish and elongated.
                (0..n)
                    .map(|_| {
                        [
                            rng.range(-4.0, 4.0),
                            rng.range(-0.5, 0.5),
                            rng.range(0.0, 1e-4),
                        ]
                    })
                    .collect()
            }
            _ => {
                // Points on a coarse spherical grid: many cocircular rings.
                (0..n)
                    .map(|_| {
                        let ring = rng.below(5) as f64 / 4.0 * PI;
                        let angle = rng.below(8) as f64 / 8.0 * 2.0 * PI;
                        [
                            ring.sin() * angle.cos(),
                            ring.cos(),
                            ring.sin() * angle.sin(),
                        ]
                    })
                    .collect()
            }
        };
        if rng.below(3) == 0 {
            points = rotated(
                &points,
                [rng.unit() + 0.1, rng.unit(), rng.unit()],
                rng.range(0.0, 3.0),
            );
        }
        let points = shuffled(&points, &mut rng);
        match build_hull(&points, tolerances(&points)) {
            Some(surface) => assert_valid_surface(
                &points,
                &surface,
                None,
                20.0,
                &format!("round {round}, family {family}, n {n}"),
            ),
            None => {
                // No hull is only acceptable for a cloud that is flat within the
                // seed tolerance: no point may be farther from the best plane.
                let extent = longest_side(&points);
                let spread = (0..3)
                    .map(|a| {
                        let lo = points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min);
                        let hi = points
                            .iter()
                            .map(|p| p[a])
                            .fold(f64::NEG_INFINITY, f64::max);
                        hi - lo
                    })
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    spread <= 1e-3 * extent || family == 3,
                    "round {round}, family {family}, n {n}: no hull for a solid cloud"
                );
            }
        }
    }
}

// ── degenerate input ──────────────────────────────────────────────────────

#[test]
fn degenerate_and_invalid_input_has_no_hull() {
    let t = |points: &[[f64; 3]]| {
        build_hull(
            points,
            HullTolerances {
                visible_distance: 1e-10,
                seed_distance: 1e-10,
            },
        )
    };
    assert!(t(&[]).is_none());
    assert!(t(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).is_none());
    assert!(t(&[[1.0, 2.0, 3.0]; 6]).is_none());
    let line: Vec<[f64; 3]> = (0..8)
        .map(|i| [i as f64, 2.0 * i as f64, -(i as f64)])
        .collect();
    assert!(t(&line).is_none());
    let mut rng = Rng(31);
    let plane: Vec<[f64; 3]> = (0..30)
        .map(|_| {
            let (u, v) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
            [u + v, u - v, 0.5 * u + 2.0 * v]
        })
        .collect();
    assert!(t(&plane).is_none());
    let tetra = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(t(&tetra).is_some());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut points = tetra.to_vec();
        points.push([0.2, bad, 0.2]);
        assert!(t(&points).is_none(), "{bad}");
    }
    assert!(cloud_extent(&[]).is_none());
    assert!(cloud_extent(&[[1.0; 3]; 3]).is_none());
    assert!(cloud_extent(&[[f64::NAN; 3]]).is_none());
    let extent = cloud_extent(&[[0.0; 3], [3.0, 4.0, 0.0]]).expect("extent");
    assert_eq!(extent.longest_side, 4.0);
    assert!((extent.diagonal - 5.0).abs() < 1e-15);
}

#[test]
fn single_tetrahedron_and_octahedron_are_exact() {
    let tetra = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let surface = build(&tetra);
    assert_valid_surface(&tetra, &surface, None, 1.0, "tetrahedron");
    assert_eq!(surface.faces.len(), 4);
    assert!((volume(&surface.vertices, &surface.faces) - 1.0 / 6.0).abs() < 1e-15);
    let octahedron = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let surface = build(&octahedron);
    assert_valid_surface(&octahedron, &surface, None, 1.0, "octahedron");
    assert_eq!(surface.faces.len(), 8);
    assert!((volume(&surface.vertices, &surface.faces) - 4.0 / 3.0).abs() < 1e-14);
}

// ── internals ─────────────────────────────────────────────────────────────

#[test]
fn seed_comes_from_extreme_points_whatever_the_order() {
    // The 3 x 3 x 3 lattice: the seed is the space diagonal from the lexicographic
    // minimum to the lexicographic maximum, then the lexicographically largest of
    // the six equally distant corners, then the largest of the points farthest
    // from that plane: all corners.
    let mut rng = Rng(41);
    let base = lattice(3, 3, 3);
    let mut seeds: BTreeSet<Vec<[i64; 3]>> = BTreeSet::new();
    for _ in 0..30 {
        let points = shuffled(&base, &mut rng);
        let seed = seed_tetrahedron(&points, 0, 1e-10).expect("seed");
        let corners: Vec<[i64; 3]> = seed
            .iter()
            .map(|&i| {
                [
                    points[i][0] as i64,
                    points[i][1] as i64,
                    points[i][2] as i64,
                ]
            })
            .collect();
        for corner in &corners {
            assert!(corner.iter().all(|&c| c == 0 || c == 2), "{corner:?}");
        }
        seeds.insert(corners);
    }
    assert_eq!(seeds.len(), 1, "the seed depends on the order: {seeds:?}");
    let seed = seeds.into_iter().next().expect("one seed");
    assert_eq!(seed, vec![[0, 0, 0], [2, 2, 2], [2, 2, 0], [2, 0, 2]]);
}

#[test]
fn plane_through_is_the_same_for_every_corner_order_and_accurate_when_skinny() {
    let a = [0.3, -1.2, 2.0];
    let b = [4.0, 0.5, -1.0];
    let c = [1.0, 3.0, 0.7];
    let (normal, offset, quality) = plane_through(a, b, c);
    assert!((dot(normal, normal) - 1.0).abs() < 1e-15);
    assert!(quality > 0.1 && quality <= 1.0, "quality {quality}");
    for corners in [[a, b, c], [b, c, a], [c, a, b]] {
        let (n, d, q) = plane_through(corners[0], corners[1], corners[2]);
        assert!((q - quality).abs() < 1e-14);
        for (got, want) in n.iter().zip(normal.iter()) {
            assert!((got - want).abs() < 1e-15);
        }
        assert!((d - offset).abs() < 1e-14);
    }
    // The reversed winding flips the plane.
    let (flipped, flipped_offset, _) = plane_through(a, c, b);
    for (got, want) in flipped.iter().zip(normal.iter()) {
        assert!((got + want).abs() < 1e-15);
    }
    assert!((flipped_offset + offset).abs() < 1e-14);
    // All three corners lie on the plane.
    for corner in [a, b, c] {
        assert!((dot(normal, corner) - offset).abs() < 1e-14);
    }
    // A sliver of length 1e3 and width 1e-6 lying in the plane z = 5: the normal
    // is +z (counter-clockwise seen from above).
    let (sliver, sliver_offset, sliver_quality) =
        plane_through([0.0, 0.0, 5.0], [1e3, 1e-6, 5.0], [0.0, 1e-6, 5.0]);
    assert!((sliver[2].abs() - 1.0).abs() < 1e-12, "normal {sliver:?}");
    assert!((sliver_offset / sliver[2] - 5.0).abs() < 1e-9);
    // Twice the area (1e-3) over the squared longest edge (1e6): 1e-9, far below
    // the thin limit.
    assert!(
        (sliver_quality - 1e-9).abs() < 1e-12,
        "quality {sliver_quality}"
    );
    assert!(sliver_quality < THIN_FACE_QUALITY);
    // A triangle without area has no plane and quality zero.
    let (none, _, quality) = plane_through([0.0; 3], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    assert_eq!(none, [0.0; 3]);
    assert_eq!(quality, 0.0);
    let (none, _, quality) = plane_through([1.0; 3], [1.0; 3], [1.0; 3]);
    assert_eq!(none, [0.0; 3]);
    assert_eq!(quality, 0.0);
}

/// The corners of the unit tetrahedron `(0,0,0), (1,0,0), (0,1,0), (0,0,1)`
/// followed by `extra` points: the first four are the seed of the hull the
/// tests build.
fn tetrahedron_with(extra: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut points = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    points.extend_from_slice(extra);
    points
}

#[test]
fn horizon_loop_test_accepts_a_loop_and_rejects_everything_else() {
    let points = tetrahedron_with(&[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]);
    let mut hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let edge = |from: usize, to: usize| HorizonEdge {
        from,
        to,
        outer_face: 0,
        outer_edge: 0,
    };
    let mut check = |edges: &[(usize, usize)]| {
        hull.epoch += 1;
        hull.horizon = edges.iter().map(|&(from, to)| edge(from, to)).collect();
        hull.horizon_is_single_loop()
    };
    // A triangle and a square are fine, whichever edge comes first.
    assert!(check(&[(0, 1), (1, 2), (2, 0)]));
    assert!(check(&[(2, 0), (0, 1), (1, 2)]));
    assert!(check(&[(3, 4), (1, 3), (4, 5), (5, 1)]));
    // Too short, open, two separate loops, a figure eight (a vertex used twice).
    assert!(!check(&[]));
    assert!(!check(&[(0, 1), (1, 0)]));
    assert!(!check(&[(0, 1), (1, 2), (2, 3)]));
    assert!(!check(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]));
    assert!(!check(&[(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]));
    // A chain that runs into a loop it does not belong to.
    assert!(!check(&[(0, 1), (1, 2), (2, 3), (3, 2)]));
    // The state of one call does not leak into the next.
    assert!(check(&[(0, 1), (1, 2), (2, 0)]));
}

/// Faces of the seed tetrahedron whose outward normal is `direction`.
fn face_with_normal(hull: &Quickhull<'_>, direction: [f64; 3]) -> usize {
    let length = dot(direction, direction).sqrt();
    (0..hull.faces.len())
        .find(|&f| dot(hull.faces[f].normal, direction) > 0.999 * length)
        .expect("the tetrahedron has a face with this normal")
}

#[test]
fn visible_region_covers_exactly_the_faces_that_see_the_point() {
    // Points 4, 5 and 6 are only probes for the visibility search.
    let points = tetrahedron_with(&[[1.0, 1.0, 1.0], [2.0, 2.0, -0.5], [-1.0, -1.0, -1.0]]);
    let mut hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let slanted = face_with_normal(&hull, [1.0, 1.0, 1.0]);
    let bottom = face_with_normal(&hull, [0.0, 0.0, -1.0]);
    let back = face_with_normal(&hull, [-1.0, 0.0, 0.0]);

    // In front of the slanted face only: one visible face, a horizon of three.
    assert!(hull.find_visible_region(slanted, 4, 1e-12));
    assert_eq!(hull.visible, vec![slanted]);
    assert_eq!(hull.horizon.len(), 3);

    // In front of the slanted and the bottom face: two adjacent faces, a
    // horizon of four edges around them.
    assert!(hull.find_visible_region(slanted, 5, 1e-12));
    let visible: BTreeSet<usize> = hull.visible.iter().copied().collect();
    assert_eq!(visible, BTreeSet::from([slanted, bottom]));
    assert_eq!(hull.horizon.len(), 4);
    // The horizon edges are edges of visible faces whose partner is not visible.
    for edge in &hull.horizon {
        assert!(!visible.contains(&edge.outer_face));
        let outer = &hull.faces[edge.outer_face];
        assert_eq!(outer.vertices[edge.outer_edge], edge.to);
        assert_eq!(outer.vertices[(edge.outer_edge + 1) % 3], edge.from);
    }

    // Behind the origin vertex the three coordinate faces see the point and
    // the horizon is the boundary of the slanted face.
    assert!(hull.find_visible_region(bottom, 6, 1e-12));
    assert_eq!(hull.visible.len(), 3);
    assert!(!hull.visible.contains(&slanted));
    assert!(hull.visible.contains(&back));
    assert_eq!(hull.horizon.len(), 3);
}

#[test]
fn visibility_is_decided_exactly_not_by_rounded_distances() {
    // Points above the bottom face (its outward normal is -z) of the unit
    // tetrahedron: 1e-300 below it is strictly in front, exactly on it and
    // 1e-300 above it are not, although none is anywhere near the tolerance.
    let points = tetrahedron_with(&[
        [0.25, 0.25, -1e-300],
        [0.25, 0.25, 0.0],
        [0.25, 0.25, 1e-300],
    ]);
    let hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let bottom = face_with_normal(&hull, [0.0, 0.0, -1.0]);
    assert!(hull.sees(bottom, 4));
    assert!(!hull.sees(bottom, 5));
    assert!(!hull.sees(bottom, 6));
    // The rounded distance of the first one is far below the tolerance, so it is
    // not added to any outside set.
    assert!(hull.faces.iter().all(|face| face.outside.is_empty()));
    // The seed faces are wound outward by the exact predicate, whatever the
    // order of the seed corners: the opposite corner is always behind a face.
    for face in &hull.faces {
        let [a, b, c] = face.vertices;
        let opposite = (0..4)
            .find(|i| ![a, b, c].contains(i))
            .expect("fourth corner");
        assert_eq!(
            orient3d(points[a], points[b], points[c], points[opposite]),
            Orientation::Positive
        );
    }
}

#[test]
fn a_point_without_a_usable_visible_region_is_dropped_not_added() {
    // Two points in front of the slanted face; the nearer one is (0.25, 0.2, 1.9).
    let points = tetrahedron_with(&[[0.2, 0.2, 2.0], [0.25, 0.2, 1.9]]);
    let mut hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let owner = face_with_normal(&hull, [1.0, 1.0, 1.0]);
    assert_eq!(hull.faces[owner].outside.len(), 2);
    assert_eq!(hull.faces[owner].furthest, 4, "point 4 is the farthest");
    // Break the reverse edge of one neighbour: the horizon search then fails at
    // every tolerance.
    let neighbour = hull.faces[owner].neighbors[0];
    hull.faces[neighbour].vertices = [usize::MAX - 1, usize::MAX - 2, usize::MAX - 3];
    let before = hull.faces.iter().filter(|f| f.alive).count();
    let mut pending = Vec::new();
    hull.expand(owner, &mut pending);
    // Nothing was added: the same faces are alive, the point is gone from the
    // outside set, and the rest of the set is still there.
    assert_eq!(hull.faces.iter().filter(|f| f.alive).count(), before);
    assert_eq!(hull.faces[owner].outside, vec![5]);
    assert_eq!(hull.faces[owner].furthest, 5);
    assert_eq!(pending, vec![owner]);
}

#[test]
fn dropping_the_furthest_point_promotes_the_next_one() {
    // Three points on the normal of the slanted face through its centroid, at
    // heights 0.5, 2.0 and 1.0 above it.
    let centroid = 1.0 / 3.0;
    let direction = 1.0 / 3.0_f64.sqrt();
    let at = |height: f64| {
        [
            centroid + height * direction,
            centroid + height * direction,
            centroid + height * direction,
        ]
    };
    let points = tetrahedron_with(&[at(0.5), at(2.0), at(1.0)]);
    let mut hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let f = face_with_normal(&hull, [1.0, 1.0, 1.0]);
    assert_eq!(hull.faces[f].outside.len(), 3);
    assert_eq!(hull.faces[f].furthest, 5);
    assert!((hull.faces[f].furthest_distance - 2.0).abs() < 1e-12);
    hull.drop_furthest(f);
    assert_eq!(hull.faces[f].furthest, 6);
    assert_eq!(hull.faces[f].outside.len(), 2);
    hull.drop_furthest(f);
    assert_eq!(hull.faces[f].furthest, 4);
    hull.drop_furthest(f);
    assert!(hull.faces[f].outside.is_empty());
    assert_eq!(hull.faces[f].furthest, usize::MAX);
}

#[test]
fn exactly_tied_points_are_resolved_lexicographically() {
    // Two points at the same distance 1 below the bottom face (its normal is
    // exactly -z, so the distances tie exactly): the lexicographically larger
    // one is the farthest, whichever was assigned first.
    let small = [0.25, 0.25, -1.0];
    let large = [0.5, 0.25, -1.0];
    for extra in [[small, large], [large, small]] {
        let points = tetrahedron_with(&extra);
        let hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
        let f = face_with_normal(&hull, [0.0, 0.0, -1.0]);
        assert_eq!(hull.faces[f].outside.len(), 2);
        assert_eq!(hull.faces[f].furthest_distance, 1.0);
        let winner = hull.faces[f].furthest;
        assert_eq!(points[winner], large, "furthest {:?}", points[winner]);
    }
}

#[test]
fn a_much_larger_neighbour_that_the_new_face_would_make_reflex_joins_the_region() {
    // Tolerance 1e-6: a face is removed by a point only if the point is more than
    // 1e-8 in front of it.  The point (0.5, -5e-3, -9.9e-9) is 5e-3 in front of
    // the face y = 0 of the unit tetrahedron but only 9.9e-9 in front of the
    // bottom face z = 0, which the point therefore does not remove by itself.  The
    // new face on their common edge is small (area 2.5e-3), so the far corner
    // (0, 1, 0) of the large bottom face would lie 2e-6, more than the tolerance,
    // in front of it: the bottom face has to be removed as well.
    let points = tetrahedron_with(&[[0.5, -5e-3, -9.9e-9]]);
    let mut hull = Quickhull::new(&points, &points, 1e-6, [0, 1, 2, 3]).expect("hull");
    let front = face_with_normal(&hull, [0.0, -1.0, 0.0]);
    let bottom = face_with_normal(&hull, [0.0, 0.0, -1.0]);
    let threshold = 1e-8;
    assert!(hull.removed_by(front, 4, threshold));
    assert!(!hull.removed_by(bottom, 4, threshold));
    assert!(hull.find_visible_region(front, 4, threshold));
    assert!(hull.visible.contains(&front));
    assert!(
        hull.visible.contains(&bottom),
        "the reflex neighbour was not added: {:?}",
        hull.visible
    );
    assert_eq!(hull.horizon.len(), 4);
    // With the point further out (0.4 in front of the side face) the new face on
    // the common edge is large (area 0.2), the far corner of the bottom face would
    // only be 2.5e-8 in front of it, and the bottom face stays.
    let points = tetrahedron_with(&[[0.5, -0.4, -9.9e-9]]);
    let mut hull = Quickhull::new(&points, &points, 1e-6, [0, 1, 2, 3]).expect("hull");
    let front = face_with_normal(&hull, [0.0, -1.0, 0.0]);
    let bottom = face_with_normal(&hull, [0.0, 0.0, -1.0]);
    assert!(hull.find_visible_region(front, 4, threshold));
    assert!(!hull.visible.contains(&bottom), "{:?}", hull.visible);
}

#[test]
fn thin_faces_are_flagged_and_the_rest_are_not() {
    // The face (a, b, c) of this tetrahedron is a sliver: c is 1e-5 from the
    // segment ab.
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1e-5, 0.0],
        [0.5, 0.5, 0.5],
    ];
    let hull = Quickhull::new(&points, &points, 1e-10, [0, 1, 2, 3]).expect("hull");
    let thin: Vec<[usize; 3]> = hull
        .faces
        .iter()
        .filter(|face| face.thin)
        .map(|face| {
            let mut vertices = face.vertices;
            vertices.sort_unstable();
            vertices
        })
        .collect();
    assert_eq!(thin, vec![[0, 1, 2]]);
}
