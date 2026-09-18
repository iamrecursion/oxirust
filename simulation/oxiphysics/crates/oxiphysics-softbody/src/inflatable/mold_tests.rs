// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`BlowMolding::mold_collision`] and for the
//! [`MoldSurface`] it is built on.
//!
//! `mold_collision` used to be an associated function without `&self` whose
//! body was `let _ = shell;`: it could not even read the mold it was meant to
//! enforce, and a shell was never held back.

use super::mold_surface::{MoldSurface, SurfaceHit, TriangleFeature, closest_point_on_triangle};
use super::*;
use oxiphysics_geometry::boolean_ops::winding_number_inside;

/// SplitMix64: a small deterministic generator of floats in `[0, 1)`.
struct SplitMix(u64);

impl SplitMix {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// A float in `[low, high)`.
    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    fn point(&mut self, low: f64, high: f64) -> [f64; 3] {
        [
            self.range(low, high),
            self.range(low, high),
            self.range(low, high),
        ]
    }
}

/// A faceted sphere of the given radius as a mold, outward normals.
fn sphere_mold(panels: usize, radius: f64) -> BlowMolding {
    let mesh = balloon_model(panels, radius);
    BlowMolding {
        mold_vertices: mesh.vertices,
        mold_tris: mesh.tris,
    }
}

/// The same mold with every triangle wound the other way.
fn reversed(mold: &BlowMolding) -> BlowMolding {
    let mut flipped = mold.clone();
    for tri in &mut flipped.mold_tris {
        tri.swap(1, 2);
    }
    flipped
}

/// The axis-aligned cube `[-half, half]^3` with outward normals.
fn cube_mold(half: f64) -> BlowMolding {
    let vertices: Vec<[f64; 3]> = (0..8_usize)
        .map(|index| {
            let coordinate = |bit: usize| if (index >> bit) & 1 == 1 { half } else { -half };
            [coordinate(0), coordinate(1), coordinate(2)]
        })
        .collect();
    let mut tris = Vec::new();
    for axis in 0..3_usize {
        for high in [false, true] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let corner = |du: usize, dv: usize| (usize::from(high) << axis) | (du << u) | (dv << v);
            let quad = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
            for [a, b, c] in [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]] {
                let normal = vec_cross(
                    vec_sub(vertices[b], vertices[a]),
                    vec_sub(vertices[c], vertices[a]),
                );
                let wanted = if high { 1.0 } else { -1.0 };
                if normal[axis] * wanted > 0.0 {
                    tris.push([a, b, c]);
                } else {
                    tris.push([a, c, b]);
                }
            }
        }
    }
    BlowMolding {
        mold_vertices: vertices,
        mold_tris: tris,
    }
}

/// A thin convex wedge: an isosceles triangle with a 20 degree apex at the
/// origin, 10 long, extruded along `z` from -1 to 1. Its apex edge is so sharp
/// that the two face normals next to it are 160 degrees apart.
fn wedge_mold() -> BlowMolding {
    let half_width = 10.0 * 10.0_f64.to_radians().tan();
    let vertices = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [10.0, -half_width, -1.0],
        [10.0, -half_width, 1.0],
        [10.0, half_width, -1.0],
        [10.0, half_width, 1.0],
    ];
    let centre = vertices
        .iter()
        .fold([0.0; 3], |sum, v| vec_add(sum, vec_scale(*v, 1.0 / 6.0)));
    let drafts = [
        [0, 4, 5],
        [0, 5, 1],
        [0, 2, 3],
        [0, 3, 1],
        [2, 4, 5],
        [2, 5, 3],
        [0, 2, 4],
        [1, 3, 5],
    ];
    // The solid is convex, so a face points outward when its normal points
    // away from the centre of the vertices.
    let tris = drafts
        .iter()
        .map(|&[a, b, c]| {
            let normal = vec_cross(
                vec_sub(vertices[b], vertices[a]),
                vec_sub(vertices[c], vertices[a]),
            );
            if vec_dot(normal, vec_sub(vertices[a], centre)) > 0.0 {
                [a, b, c]
            } else {
                [a, c, b]
            }
        })
        .collect();
    BlowMolding {
        mold_vertices: vertices,
        mold_tris: tris,
    }
}

/// A star-shaped but strongly non-convex closed mold: the unit balloon with a
/// smooth radial bump pattern.
fn bumpy_mold() -> BlowMolding {
    let mesh = balloon_model(10, 1.0);
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| {
            let radius = vec_len(*v);
            let polar = (v[1] / radius).acos();
            let azimuth = v[2].atan2(v[0]);
            let scale = 1.0 + 0.35 * (3.0 * polar).sin() * (2.0 * azimuth).cos();
            vec_scale(*v, scale)
        })
        .collect();
    BlowMolding {
        mold_vertices: vertices,
        mold_tris: mesh.tris,
    }
}

/// The mold as a triangle soup: three private vertices per triangle.
fn soup(mold: &BlowMolding) -> BlowMolding {
    let mut vertices = Vec::new();
    let mut tris = Vec::new();
    for tri in &mold.mold_tris {
        let base = vertices.len();
        for &index in tri {
            vertices.push(mold.mold_vertices[index]);
        }
        tris.push([base, base + 1, base + 2]);
    }
    BlowMolding {
        mold_vertices: vertices,
        mold_tris: tris,
    }
}

/// A shell that is just a cloud of vertices (no faces) at rest.
fn cloud(vertices: Vec<[f64; 3]>) -> InflatableShell {
    let count = vertices.len();
    InflatableShell {
        vertices,
        velocities: vec![[0.0; 3]; count],
        masses: vec![1.0; count],
        tris: Vec::new(),
        target_volume: 1.0,
        pressure: 0.0,
        k_shell: 1.0,
        edge_rest: Vec::new(),
    }
}

/// Distance to, and position of, the closest point of the mold by brute force.
fn brute_force_closest(mold: &BlowMolding, p: [f64; 3]) -> ([f64; 3], f64) {
    let mut best = ([0.0; 3], f64::INFINITY);
    for tri in &mold.mold_tris {
        let [a, b, c] = [
            mold.mold_vertices[tri[0]],
            mold.mold_vertices[tri[1]],
            mold.mold_vertices[tri[2]],
        ];
        let (q, _) = closest_point_on_triangle(p, a, b, c);
        let offset = vec_sub(p, q);
        let distance_sq = vec_dot(offset, offset);
        if distance_sq < best.1 {
            best = (q, distance_sq);
        }
    }
    best
}

/// Squared distance from the query point `p` to the surface point of `hit`.
fn gap_sq(p: [f64; 3], hit: &SurfaceHit) -> f64 {
    let offset = vec_sub(p, hit.point);
    vec_dot(offset, offset)
}

fn assert_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64, what: &str) {
    let error = vec_len(vec_sub(actual, expected));
    assert!(
        error <= tolerance,
        "{what}: {actual:?} differs from {expected:?} by {error}"
    );
}

#[test]
fn an_empty_mold_leaves_the_shell_untouched() {
    let mut shell = balloon_model(3, 1.0);
    shell.velocities[2] = [1.0, 2.0, 3.0];
    let vertices = shell.vertices.clone();
    let velocities = shell.velocities.clone();
    BlowMolding::new().mold_collision(&mut shell);
    assert_eq!(shell.vertices, vertices);
    assert_eq!(shell.velocities, velocities);
}

#[test]
fn a_mold_without_a_usable_triangle_leaves_the_shell_untouched() {
    let mold = BlowMolding {
        mold_vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [f64::NAN, 0.0, 1.0],
        ],
        // Collinear, out of range, a repeated vertex and a NaN corner.
        mold_tris: vec![[0, 1, 2], [0, 1, 9], [0, 0, 1], [0, 1, 3]],
    };
    let mut shell = cloud(vec![[5.0, 5.0, 5.0], [0.5, 1.0, 0.0]]);
    let vertices = shell.vertices.clone();
    mold.mold_collision(&mut shell);
    assert_eq!(shell.vertices, vertices);
}

#[test]
fn a_sphere_mold_holds_back_a_larger_balloon() {
    let mold = sphere_mold(8, 0.8);
    let mut shell = balloon_model(4, 1.0);
    mold.mold_collision(&mut shell);
    for (index, vertex) in shell.vertices.iter().enumerate() {
        let radius = vec_len(*vertex);
        assert!(
            radius <= 0.8 + 1e-9,
            "vertex {index} at {vertex:?} stays outside the mold (radius {radius})"
        );
        // The mold is the faceted sphere through radius 0.8: the facets lie
        // inside it, but not much further than the mesh resolution allows.
        assert!(
            radius > 0.7,
            "vertex {index} was pushed too far in: {radius}"
        );
    }
}

#[test]
fn the_projection_is_the_closest_point_of_the_mold() {
    let mold = sphere_mold(8, 0.8);
    let original = balloon_model(4, 1.0);
    let mut shell = original.clone();
    mold.mold_collision(&mut shell);
    for (before, after) in original.vertices.iter().zip(shell.vertices.iter()) {
        let (closest, _) = brute_force_closest(&mold, *before);
        assert_close(*after, closest, 1e-12, "projected vertex");
    }
}

#[test]
fn vertices_inside_the_mold_do_not_move() {
    let mold = sphere_mold(8, 0.8);
    let mut shell = balloon_model(4, 0.5);
    shell.velocities[3] = [4.0, -2.0, 1.0];
    let vertices = shell.vertices.clone();
    let velocities = shell.velocities.clone();
    mold.mold_collision(&mut shell);
    assert_eq!(shell.vertices, vertices);
    assert_eq!(shell.velocities, velocities);
}

#[test]
fn only_the_vertices_outside_the_mold_move() {
    let mold = sphere_mold(8, 0.8);
    let mut shell = cloud(vec![
        [0.3, 0.1, -0.2],
        [1.5, 0.0, 0.0],
        [0.0, -0.4, 0.4],
        [0.0, 0.0, -2.0],
    ]);
    let before = shell.vertices.clone();
    mold.mold_collision(&mut shell);
    assert_eq!(shell.vertices[0], before[0]);
    assert_eq!(shell.vertices[2], before[2]);
    assert!(vec_len(shell.vertices[1]) <= 0.8 + 1e-9);
    assert!(vec_len(shell.vertices[1]) > 0.7);
    assert!(vec_len(shell.vertices[3]) <= 0.8 + 1e-9);
    assert!(vec_len(shell.vertices[3]) > 0.7);
}

#[test]
fn the_outward_velocity_is_removed_and_the_rest_is_kept() {
    let mold = cube_mold(1.0);
    let mut shell = cloud(vec![
        [2.0, 0.3, -0.2],
        [1.5, 0.3, -0.2],
        [0.5, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ]);
    shell.velocities = vec![
        [3.0, 1.0, 2.0],
        [-3.0, 1.0, 2.0],
        [5.0, 0.0, 0.0],
        [0.0, 4.0, -1.0],
    ];
    mold.mold_collision(&mut shell);

    assert_close(shell.vertices[0], [1.0, 0.3, -0.2], 1e-12, "face contact");
    assert_close(
        shell.velocities[0],
        [0.0, 1.0, 2.0],
        1e-12,
        "the normal velocity goes, the tangential velocity stays",
    );
    assert_close(shell.vertices[1], [1.0, 0.3, -0.2], 1e-12, "inward mover");
    assert_close(
        shell.velocities[1],
        [-3.0, 1.0, 2.0],
        1e-12,
        "a velocity pointing into the mold is kept",
    );
    assert_close(shell.vertices[2], [0.5, 0.0, 0.0], 0.0, "inside vertex");
    assert_close(shell.velocities[2], [5.0, 0.0, 0.0], 0.0, "inside velocity");
    assert_close(shell.vertices[3], [0.0, 1.0, 0.0], 1e-12, "top contact");
    assert_close(shell.velocities[3], [0.0, 0.0, -1.0], 1e-12, "top velocity");
}

#[test]
fn a_cube_mold_projects_onto_faces_edges_and_corners() {
    let mold = cube_mold(1.0);
    let cases = [
        ([2.0, 0.3, -0.2], [1.0, 0.3, -0.2], "face"),
        ([-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], "opposite face"),
        ([2.0, 2.0, 0.5], [1.0, 1.0, 0.5], "edge"),
        ([-2.0, 0.25, -3.0], [-1.0, 0.25, -1.0], "second edge"),
        ([2.0, 3.0, 4.0], [1.0, 1.0, 1.0], "corner"),
        ([-5.0, -2.0, 9.0], [-1.0, -1.0, 1.0], "second corner"),
        ([0.99, 0.0, 0.0], [0.99, 0.0, 0.0], "just inside"),
        ([1.0, 0.2, 0.3], [1.0, 0.2, 0.3], "exactly on the surface"),
        (
            [0.99, 0.99, 0.99],
            [0.99, 0.99, 0.99],
            "just inside a corner",
        ),
    ];
    let mut shell = cloud(cases.iter().map(|case| case.0).collect());
    mold.mold_collision(&mut shell);
    for (vertex, case) in shell.vertices.iter().zip(cases.iter()) {
        assert_close(*vertex, case.1, 1e-12, case.2);
    }
}

#[test]
fn the_closest_feature_is_reported_for_every_region_of_a_triangle() {
    let a = [0.0, 0.0, 0.0];
    let b = [2.0, 0.0, 0.0];
    let c = [0.0, 2.0, 0.0];
    let cases = [
        (
            [-1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0],
            TriangleFeature::Vertex(0),
        ),
        (
            [3.0, -1.0, 0.5],
            [2.0, 0.0, 0.0],
            TriangleFeature::Vertex(1),
        ),
        (
            [-1.0, 3.0, 0.5],
            [0.0, 2.0, 0.0],
            TriangleFeature::Vertex(2),
        ),
        ([1.0, -1.0, 0.7], [1.0, 0.0, 0.0], TriangleFeature::Edge(0)),
        ([2.0, 2.0, 0.7], [1.0, 1.0, 0.0], TriangleFeature::Edge(1)),
        ([-1.0, 1.0, 0.7], [0.0, 1.0, 0.0], TriangleFeature::Edge(2)),
        ([0.5, 0.5, 0.7], [0.5, 0.5, 0.0], TriangleFeature::Face),
    ];
    for (p, expected_point, expected_feature) in cases {
        let (point, feature) = closest_point_on_triangle(p, a, b, c);
        assert_eq!(feature, expected_feature, "feature for {p:?}");
        assert_close(point, expected_point, 1e-12, "closest point");
    }
}

#[test]
fn the_winding_of_a_closed_mold_does_not_matter() {
    let mut rng = SplitMix(7);
    // The sample boxes put a fair share of the points on either side of the mold.
    let molds = [
        ("sphere", sphere_mold(8, 0.8), 1.0),
        ("cube", cube_mold(1.0), 1.5),
    ];
    for (name, mold, reach) in molds {
        let points: Vec<[f64; 3]> = (0..300).map(|_| rng.point(-reach, reach)).collect();
        let flipped = reversed(&mold);
        let mut outward = cloud(points.clone());
        let mut inward = cloud(points.clone());
        outward.velocities = points.iter().map(|p| vec_scale(*p, 0.5)).collect();
        inward.velocities = outward.velocities.clone();
        mold.mold_collision(&mut outward);
        flipped.mold_collision(&mut inward);
        let mut moved = 0;
        for (index, point) in points.iter().enumerate() {
            assert_close(inward.vertices[index], outward.vertices[index], 1e-12, name);
            assert_close(
                inward.velocities[index],
                outward.velocities[index],
                1e-12,
                name,
            );
            if vec_len(vec_sub(outward.vertices[index], *point)) > 0.0 {
                moved += 1;
            }
        }
        assert!(
            moved > 50 && moved < points.len() - 50,
            "{name}: the test points should straddle the mold, {moved} of {} moved",
            points.len()
        );
    }
}

#[test]
fn the_classification_of_a_non_convex_mold_matches_the_winding_number() {
    let mold = bumpy_mold();
    // The mold is not convex: some chord between two of its vertices leaves it.
    let mut leaves = false;
    for (i, a) in mold.mold_vertices.iter().enumerate() {
        for b in mold.mold_vertices.iter().skip(i + 1) {
            let midpoint = vec_scale(vec_add(*a, *b), 0.5);
            if !winding_number_inside(midpoint, &mold.mold_vertices, &mold.mold_tris) {
                leaves = true;
            }
        }
    }
    assert!(leaves, "the bumpy mold should be non-convex");

    let mut rng = SplitMix(2024);
    let mut inside = 0;
    let mut outside = 0;
    let surface = MoldSurface::new(&mold.mold_vertices, &mold.mold_tris).expect("usable mold");
    for _ in 0..4000 {
        let p = rng.point(-1.8, 1.8);
        let hit = surface.closest(p).expect("finite point");
        if gap_sq(p, &hit) < 1e-8 {
            // Too close to the surface for the winding number to be decisive.
            continue;
        }
        let truth_inside = winding_number_inside(p, &mold.mold_vertices, &mold.mold_tris);
        let mut position = p;
        let moved = surface.constrain(&mut position, None);
        assert_eq!(
            moved, !truth_inside,
            "point {p:?}: the winding number says inside = {truth_inside}"
        );
        if truth_inside {
            inside += 1;
            assert_eq!(position, p);
        } else {
            outside += 1;
        }
    }
    assert!(
        inside > 200 && outside > 200,
        "inside {inside}, outside {outside}"
    );
}

#[test]
fn a_sharp_edge_is_classified_with_its_pseudo_normal() {
    // Beyond the apex edge of the wedge the closest surface point is on that
    // edge for every direction within 80 degrees of the -x axis. The face
    // normals on either side of the edge are 160 degrees apart, so the normal
    // of one face alone calls a point near the far side of that cone inside.
    let mold = wedge_mold();
    assert!(
        MoldSurface::new(&mold.mold_vertices, &mold.mold_tris).is_some(),
        "the wedge is a usable mold"
    );
    let mut rng = SplitMix(808);
    let mut points = Vec::new();
    for _ in 0..400 {
        let angle = (rng.range(-75.0, 75.0)).to_radians();
        let distance = rng.range(0.05, 0.5);
        let z = rng.range(-0.8, 0.8);
        points.push([-distance * angle.cos(), distance * angle.sin(), z]);
    }
    let mut shell = cloud(points.clone());
    mold.mold_collision(&mut shell);
    for (point, projected) in points.iter().zip(shell.vertices.iter()) {
        assert_close(
            *projected,
            [0.0, 0.0, point[2]],
            1e-12,
            "projection onto the apex edge",
        );
    }

    // The mirror image: points inside the wedge near the apex stay put.
    let inside: Vec<[f64; 3]> = (1..40)
        .map(|k| [0.3 + 0.2 * k as f64 / 2.0, 0.0, 0.0])
        .collect();
    let mut shell = cloud(inside.clone());
    mold.mold_collision(&mut shell);
    assert_eq!(shell.vertices, inside);
}

#[test]
fn a_triangle_soup_with_duplicated_vertices_classifies_like_the_indexed_mesh() {
    let indexed = bumpy_mold();
    let split = soup(&indexed);
    assert_eq!(split.mold_vertices.len(), 3 * indexed.mold_tris.len());
    let mut rng = SplitMix(99);
    for _ in 0..1500 {
        let p = rng.point(-1.8, 1.8);
        let mut from_indexed = cloud(vec![p]);
        let mut from_soup = cloud(vec![p]);
        indexed.mold_collision(&mut from_indexed);
        split.mold_collision(&mut from_soup);
        assert_close(
            from_soup.vertices[0],
            from_indexed.vertices[0],
            1e-12,
            "soup",
        );
    }
}

#[test]
fn the_hierarchy_finds_the_same_closest_point_as_a_brute_force_search() {
    let mut rng = SplitMix(31337);
    let mut vertices = Vec::new();
    let mut tris = Vec::new();
    for _ in 0..400 {
        let centre = rng.point(-3.0, 3.0);
        let base = vertices.len();
        for _ in 0..3 {
            let offset = rng.point(-0.4, 0.4);
            vertices.push(vec_add(centre, offset));
        }
        tris.push([base, base + 1, base + 2]);
    }
    let mold = BlowMolding {
        mold_vertices: vertices,
        mold_tris: tris,
    };
    let surface = MoldSurface::new(&mold.mold_vertices, &mold.mold_tris).expect("usable mold");
    for _ in 0..600 {
        let p = rng.point(-4.0, 4.0);
        let hit = surface.closest(p).expect("finite point");
        let (_, best_sq) = brute_force_closest(&mold, p);
        let found_sq = gap_sq(p, &hit);
        assert!(
            (found_sq - best_sq).abs() <= 1e-12 * best_sq.max(1e-12),
            "{p:?}: hierarchy {found_sq} vs brute force {best_sq}"
        );
    }
}

#[test]
fn projecting_twice_changes_nothing() {
    let mold = bumpy_mold();
    let mut rng = SplitMix(5);
    let mut shell = cloud((0..500).map(|_| rng.point(-2.0, 2.0)).collect());
    shell.velocities = (0..500).map(|_| rng.point(-1.0, 1.0)).collect();
    mold.mold_collision(&mut shell);
    let vertices = shell.vertices.clone();
    let velocities = shell.velocities.clone();
    mold.mold_collision(&mut shell);
    for index in 0..vertices.len() {
        assert_close(shell.vertices[index], vertices[index], 1e-12, "position");
        assert_close(
            shell.velocities[index],
            velocities[index],
            1e-12,
            "velocity",
        );
    }
}

#[test]
fn an_open_mold_is_a_one_sided_barrier() {
    let plate = BlowMolding {
        mold_vertices: vec![
            [-5.0, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
        ],
        mold_tris: vec![[0, 1, 2], [0, 2, 3]],
    };
    let mut shell = cloud(vec![[0.2, 0.3, 0.5], [0.2, 0.3, -0.5]]);
    shell.velocities = vec![[1.0, 0.0, 2.0], [1.0, 0.0, -2.0]];
    plate.mold_collision(&mut shell);
    // The normal of the plate is +z: the front vertex is outside, the back one
    // is not.
    assert_close(shell.vertices[0], [0.2, 0.3, 0.0], 1e-12, "front vertex");
    assert_close(
        shell.velocities[0],
        [1.0, 0.0, 0.0],
        1e-12,
        "front velocity",
    );
    assert_close(shell.vertices[1], [0.2, 0.3, -0.5], 0.0, "back vertex");
    assert_close(shell.velocities[1], [1.0, 0.0, -2.0], 0.0, "back velocity");

    // Wound the other way, the plate holds back the other side.
    let flipped = reversed(&plate);
    let mut shell = cloud(vec![[0.2, 0.3, 0.5], [0.2, 0.3, -0.5]]);
    flipped.mold_collision(&mut shell);
    assert_close(shell.vertices[0], [0.2, 0.3, 0.5], 0.0, "front vertex");
    assert_close(shell.vertices[1], [0.2, 0.3, 0.0], 1e-12, "back vertex");
}

#[test]
fn non_finite_vertices_and_missing_velocities_are_tolerated() {
    let mold = cube_mold(1.0);
    let mut shell = cloud(vec![
        [f64::NAN, 0.0, 0.0],
        [f64::INFINITY, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]);
    // No velocity for the last vertex.
    shell.velocities.truncate(2);
    mold.mold_collision(&mut shell);
    assert!(shell.vertices[0][0].is_nan());
    assert!(shell.vertices[1][0].is_infinite());
    assert_close(shell.vertices[2], [1.0, 0.0, 0.0], 1e-12, "last vertex");
}

#[test]
fn an_inflating_balloon_is_held_inside_a_mold_and_fills_it() {
    let mold = sphere_mold(8, 0.7);
    let surface = MoldSurface::new(&mold.mold_vertices, &mold.mold_tris).expect("usable mold");
    let mut shell = balloon_model(4, 0.4);
    let start = shell.compute_enclosed_volume();
    let mold_volume = balloon_model(8, 0.7).compute_enclosed_volume();
    // Ask for far more than the mold can hold, with a membrane that is soft
    // compared with the pressure, so that the balloon is driven into the mold.
    shell.target_volume = 3.0 * mold_volume;
    for step in 0..400 {
        shell.step(1.0 / 600.0, 2000.0, 0.001);
        mold.mold_collision(&mut shell);
        for (index, vertex) in shell.vertices.iter().enumerate() {
            assert!(
                vec_len(*vertex) <= 0.7 + 1e-9,
                "step {step}: vertex {index} at {vertex:?} escaped the mold"
            );
        }
    }
    let volume = shell.compute_enclosed_volume();
    assert!(
        shell.signed_enclosed_volume() > 0.0,
        "the balloon turned inside out"
    );
    assert!(
        volume > 2.0 * start,
        "the balloon should have inflated: {start} -> {volume}"
    );
    assert!(
        volume <= mold_volume * (1.0 + 1e-9),
        "the balloon {volume} cannot be larger than the mold {mold_volume}"
    );
    let touching = shell
        .vertices
        .iter()
        .filter(|v| {
            surface
                .closest(**v)
                .is_some_and(|hit| gap_sq(**v, &hit) < 1e-18)
        })
        .count();
    assert!(
        touching * 2 > shell.vertices.len(),
        "most vertices should press against the mold, {touching} of {}",
        shell.vertices.len()
    );
}
