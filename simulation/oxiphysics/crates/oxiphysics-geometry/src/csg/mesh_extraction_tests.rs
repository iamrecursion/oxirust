// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`extract_mesh_simple`]: the marching-cubes surface
//! extraction of an implicit surface (watertight, welded, outward-wound).

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::functions::{dot, length, sub};
use super::{
    CsgDifference, CsgUnion, ImplicitSurface, SdfBox, SdfSphere, SdfTorus, extract_mesh_simple,
};

type Vertices = Vec<[f64; 3]>;
type Triangles = Vec<[usize; 3]>;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A field defined by a closure.
struct FnField<F>(F);

impl<F> ImplicitSurface for FnField<F>
where
    F: Fn([f64; 3]) -> f64 + Send + Sync,
{
    fn sdf(&self, p: [f64; 3]) -> f64 {
        (self.0)(p)
    }
    fn gradient(&self, _p: [f64; 3]) -> [f64; 3] {
        [0.0, 0.0, 1.0]
    }
}

/// Counts the number of field evaluations.
struct CountingField<'a> {
    inner: &'a dyn ImplicitSurface,
    calls: AtomicUsize,
}

impl ImplicitSurface for CountingField<'_> {
    fn sdf(&self, p: [f64; 3]) -> f64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.sdf(p)
    }
    fn gradient(&self, p: [f64; 3]) -> [f64; 3] {
        self.inner.gradient(p)
    }
}

/// How often each directed edge occurs in the triangle list.
fn directed_edge_counts(triangles: &[[usize; 3]]) -> HashMap<(usize, usize), usize> {
    let mut counts = HashMap::new();
    for tri in triangles {
        for k in 0..3 {
            *counts.entry((tri[k], tri[(k + 1) % 3])).or_insert(0) += 1;
        }
    }
    counts
}

/// Asserts that the mesh is a closed, consistently oriented 2-manifold surface
/// (every directed edge used once and matched by its reverse, no unreferenced
/// vertex) and returns its Euler characteristic `V - E + F`.
fn closed_euler_characteristic(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    label: &str,
) -> i64 {
    let counts = directed_edge_counts(triangles);
    for (&(a, b), &count) in &counts {
        assert_eq!(
            count, 1,
            "{label}: directed edge ({a},{b}) used {count} times"
        );
        assert_eq!(
            counts.get(&(b, a)).copied(),
            Some(1),
            "{label}: edge ({a},{b}) has no opposite edge"
        );
    }
    let referenced: HashSet<usize> = triangles.iter().flatten().copied().collect();
    assert_eq!(
        referenced.len(),
        vertices.len(),
        "{label}: unreferenced vertices"
    );
    assert!(
        referenced.iter().all(|&v| v < vertices.len()),
        "{label}: index out of range"
    );
    let edges = (counts.len() / 2) as i64;
    vertices.len() as i64 - edges + triangles.len() as i64
}

fn triangle_corners(vertices: &[[f64; 3]], tri: &[usize; 3]) -> [[f64; 3]; 3] {
    [vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]]
}

fn surface_area(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|tri| {
            let [a, b, c] = triangle_corners(vertices, tri);
            0.5 * length(cross(sub(b, a), sub(c, a)))
        })
        .sum()
}

/// Signed volume from the divergence theorem (positive for outward winding).
fn enclosed_volume(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|tri| {
            let [a, b, c] = triangle_corners(vertices, tri);
            dot(a, cross(b, c))
        })
        .sum::<f64>()
        / 6.0
}

fn assert_close(got: f64, want: f64, relative: f64, what: &str) {
    assert!(
        (got - want).abs() <= relative * want.abs(),
        "{what}: {got} differs from {want} by more than {:.1}%",
        relative * 100.0
    );
}

fn sphere_mesh(resolution: usize) -> (Vertices, Triangles) {
    let sphere = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
    extract_mesh_simple(&sphere, [-1.5; 3], [1.5; 3], resolution)
}

#[test]
fn unit_sphere_is_a_closed_oriented_manifold_with_the_right_area() {
    let (vertices, triangles) = sphere_mesh(40);
    assert!(triangles.len() > 2000, "{} triangles", triangles.len());
    // Every edge is shared by exactly two triangles and the surface is a sphere.
    let euler = closed_euler_characteristic(&vertices, &triangles, "sphere");
    assert_eq!(euler, 2, "Euler characteristic of the sphere mesh");
    assert_close(surface_area(&vertices, &triangles), 4.0 * PI, 0.02, "area");
    assert_close(
        enclosed_volume(&vertices, &triangles),
        4.0 / 3.0 * PI,
        0.02,
        "volume",
    );
}

#[test]
fn unit_sphere_triangles_face_outward_and_vertices_lie_on_the_surface() {
    let (vertices, triangles) = sphere_mesh(40);
    for &p in &vertices {
        let radius = length(p);
        assert!(
            (radius - 1.0).abs() < 0.01,
            "vertex {p:?} at radius {radius}"
        );
    }
    for tri in &triangles {
        let [a, b, c] = triangle_corners(&vertices, tri);
        let normal = cross(sub(b, a), sub(c, a));
        assert!(length(normal) > 0.0, "zero-area triangle {tri:?}");
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        assert!(
            dot(normal, centroid) > 0.0,
            "triangle {tri:?} faces the inside of the sphere"
        );
    }
}

#[test]
fn unit_sphere_mesh_is_not_a_stack_of_horizontal_slices() {
    let (vertices, triangles) = sphere_mesh(40);
    // A horizontal triangle has a normal along z; on a sphere only the polar caps do.
    let tilted = triangles
        .iter()
        .filter(|tri| {
            let [a, b, c] = triangle_corners(&vertices, tri);
            let normal = cross(sub(b, a), sub(c, a));
            normal[2].abs() < 0.8 * length(normal)
        })
        .count();
    assert!(
        tilted * 2 > triangles.len(),
        "{tilted} of {} triangles are tilted",
        triangles.len()
    );
}

#[test]
fn vertices_are_welded_one_per_crossed_grid_edge() {
    let (vertices, triangles) = sphere_mesh(24);
    let distinct: HashSet<[u64; 3]> = vertices
        .iter()
        .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
        .collect();
    assert_eq!(
        distinct.len(),
        vertices.len(),
        "duplicated vertex positions"
    );
    // A closed triangulated surface has three half-edges per triangle and two
    // triangles per edge, so V = F / 2 + 2 for a sphere.
    assert_eq!(triangles.len(), 2 * (vertices.len() - 2));
}

#[test]
fn field_is_sampled_once_per_grid_node() {
    let sphere = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
    let counting = CountingField {
        inner: &sphere,
        calls: AtomicUsize::new(0),
    };
    let (_, triangles) = extract_mesh_simple(&counting, [-1.5; 3], [1.5; 3], 12);
    assert!(!triangles.is_empty());
    assert_eq!(counting.calls.load(Ordering::Relaxed), 13 * 13 * 13);
}

#[test]
fn box_mesh_is_closed_and_matches_the_box_volume() {
    let half = 0.83;
    let cube = SdfBox::new([0.0, 0.0, 0.0], [half, half, half]);
    let (vertices, triangles) = extract_mesh_simple(&cube, [-1.5; 3], [1.5; 3], 30);
    let euler = closed_euler_characteristic(&vertices, &triangles, "box");
    assert_eq!(euler, 2);
    let side = 2.0 * half;
    assert_close(
        enclosed_volume(&vertices, &triangles),
        side * side * side,
        0.03,
        "volume",
    );
    assert_close(
        surface_area(&vertices, &triangles),
        6.0 * side * side,
        0.03,
        "area",
    );
    for p in &vertices {
        assert!(
            p.iter().all(|c| c.abs() <= half + 1e-9),
            "vertex {p:?} outside the box"
        );
    }
}

#[test]
fn thick_shell_has_two_closed_boundary_components() {
    let shell = CsgDifference::new(
        Box::new(SdfSphere::new([0.0, 0.0, 0.0], 1.2)),
        Box::new(SdfSphere::new([0.0, 0.0, 0.0], 0.7)),
    );
    let (vertices, triangles) = extract_mesh_simple(&shell, [-1.5; 3], [1.5; 3], 40);
    // Two disjoint spheres: V - E + F = 2 + 2.
    let euler = closed_euler_characteristic(&vertices, &triangles, "shell");
    assert_eq!(euler, 4);
    assert_close(
        surface_area(&vertices, &triangles),
        4.0 * PI * (1.2 * 1.2 + 0.7 * 0.7),
        0.02,
        "area",
    );
    // Outward winding: the outer sphere contributes volume, the inner one
    // (whose outward side points into the cavity) subtracts it.
    assert_close(
        enclosed_volume(&vertices, &triangles),
        4.0 / 3.0 * PI * (1.2_f64.powi(3) - 0.7_f64.powi(3)),
        0.02,
        "volume",
    );
}

#[test]
fn torus_mesh_is_closed_with_genus_one() {
    let torus = SdfTorus::new(2.0, 0.5);
    // The ring lies in the xz-plane, so the box is flat along y and the grid
    // cells are not cubes.
    let (vertices, triangles) =
        extract_mesh_simple(&torus, [-3.0, -1.0, -3.0], [3.0, 1.0, 3.0], 60);
    let euler = closed_euler_characteristic(&vertices, &triangles, "torus");
    assert_eq!(euler, 0, "Euler characteristic of a torus");
    assert_close(
        surface_area(&vertices, &triangles),
        4.0 * PI * PI * 2.0 * 0.5,
        0.03,
        "area",
    );
    assert_close(
        enclosed_volume(&vertices, &triangles),
        2.0 * PI * PI * 2.0 * 0.25,
        0.03,
        "volume",
    );
}

#[test]
fn anisotropic_box_uses_the_requested_bounds() {
    let sphere = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
    let min = [-2.0, -1.5, -1.2];
    let max = [2.0, 1.5, 1.2];
    let (vertices, triangles) = extract_mesh_simple(&sphere, min, max, 48);
    let euler = closed_euler_characteristic(&vertices, &triangles, "anisotropic");
    assert_eq!(euler, 2);
    assert_close(surface_area(&vertices, &triangles), 4.0 * PI, 0.02, "area");
    for p in &vertices {
        for k in 0..3 {
            assert!(
                (min[k]..=max[k]).contains(&p[k]),
                "vertex {p:?} outside the bounds"
            );
        }
    }
}

#[test]
fn surface_cut_by_the_box_is_open_only_at_the_box_face() {
    let sphere = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
    let cut = -0.4;
    let (vertices, triangles) =
        extract_mesh_simple(&sphere, [cut, -1.5, -1.5], [1.5, 1.5, 1.5], 38);
    assert!(!triangles.is_empty());
    let counts = directed_edge_counts(&triangles);
    let mut open_edges = 0;
    for (&(a, b), &count) in &counts {
        assert_eq!(count, 1, "directed edge ({a},{b}) used {count} times");
        if !counts.contains_key(&(b, a)) {
            open_edges += 1;
            // An unmatched edge lies in the clipping plane x = cut.
            assert_eq!(vertices[a][0], cut, "open edge ({a},{b}) off the plane");
            assert_eq!(vertices[b][0], cut, "open edge ({a},{b}) off the plane");
        }
    }
    assert!(open_edges > 10, "the clipped surface must have a rim");
    // The rim is the circle of radius sqrt(1 - 0.4²) in the plane x = -0.4.
    let rim_radius = (1.0_f64 - cut * cut).sqrt();
    let rim_starts: Vec<usize> = counts
        .keys()
        .filter(|&&(a, b)| !counts.contains_key(&(b, a)))
        .map(|&(a, _)| a)
        .collect();
    assert_eq!(rim_starts.len(), open_edges);
    for a in rim_starts {
        let p = vertices[a];
        let r = (p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((r - rim_radius).abs() < 0.02, "rim vertex {p:?} at {r}");
    }
}

#[test]
fn tiny_resolutions_fall_back_to_two_cells_per_axis() {
    let (reference_vertices, reference_triangles) = sphere_mesh(2);
    // Only the centre node of the 3x3x3 grid is inside: an octahedron.
    assert_eq!(reference_vertices.len(), 6);
    assert_eq!(reference_triangles.len(), 8);
    assert_eq!(
        closed_euler_characteristic(&reference_vertices, &reference_triangles, "octahedron"),
        2
    );
    for resolution in [0, 1] {
        let (vertices, triangles) = sphere_mesh(resolution);
        assert_eq!(vertices, reference_vertices, "resolution {resolution}");
        assert_eq!(triangles, reference_triangles, "resolution {resolution}");
    }
}

#[test]
fn surface_that_misses_the_box_gives_an_empty_mesh() {
    let far = SdfSphere::new([10.0, 0.0, 0.0], 1.0);
    let (vertices, triangles) = extract_mesh_simple(&far, [-1.0; 3], [1.0; 3], 16);
    assert!(vertices.is_empty() && triangles.is_empty());
    // The box lies entirely inside the sphere: no node changes sign.
    let big = SdfSphere::new([0.0, 0.0, 0.0], 50.0);
    let (vertices, triangles) = extract_mesh_simple(&big, [-1.0; 3], [1.0; 3], 16);
    assert!(vertices.is_empty() && triangles.is_empty());
}

#[test]
fn invalid_bounds_give_an_empty_mesh_and_reversed_bounds_are_swapped() {
    let sphere = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
    let empty = |min: [f64; 3], max: [f64; 3]| {
        let (vertices, triangles) = extract_mesh_simple(&sphere, min, max, 16);
        vertices.is_empty() && triangles.is_empty()
    };
    assert!(empty([f64::NAN, -1.5, -1.5], [1.5; 3]));
    assert!(empty([-1.5; 3], [1.5, f64::INFINITY, 1.5]));
    assert!(empty([-1.5, f64::NEG_INFINITY, -1.5], [1.5; 3]));
    // No extent along z.
    assert!(empty([-1.5, -1.5, 0.0], [1.5, 1.5, 0.0]));
    // The extent overflows to infinity.
    assert!(empty([-f64::MAX, -1.5, -1.5], [f64::MAX, 1.5, 1.5]));

    let expected = extract_mesh_simple(&sphere, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], 20);
    assert!(!expected.1.is_empty());
    let reversed = extract_mesh_simple(&sphere, [1.5, 1.5, 1.5], [-1.5, -1.5, -1.5], 20);
    assert_eq!(reversed, expected);
    let mixed = extract_mesh_simple(&sphere, [1.5, -1.5, 1.5], [-1.5, 1.5, -1.5], 20);
    assert_eq!(mixed, expected);
}

#[test]
fn non_finite_field_values_never_produce_non_finite_vertices() {
    let with_nan = FnField(|p: [f64; 3]| {
        if p[0] > 0.9 {
            f64::NAN
        } else {
            length(p) - 1.0
        }
    });
    let with_infinities = FnField(|p: [f64; 3]| {
        if p[0] > 0.9 {
            f64::INFINITY
        } else if p[0] < -0.9 {
            f64::NEG_INFINITY
        } else {
            length(p) - 1.0
        }
    });
    for (label, field) in [
        ("nan", &with_nan as &dyn ImplicitSurface),
        ("inf", &with_infinities as &dyn ImplicitSurface),
    ] {
        let (vertices, triangles) = extract_mesh_simple(field, [-1.5; 3], [1.5; 3], 24);
        assert!(!triangles.is_empty(), "{label}");
        for p in &vertices {
            assert!(p.iter().all(|c| c.is_finite()), "{label}: vertex {p:?}");
        }
        for tri in &triangles {
            assert!(tri.iter().all(|&v| v < vertices.len()), "{label}");
        }
    }
}

#[test]
fn extraction_is_deterministic() {
    assert_eq!(sphere_mesh(20), sphere_mesh(20));
}

/// Deterministic pseudo-random value in `[0, 1)` for the counter `x` (SplitMix64).
fn hash01(x: u64) -> f64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// A union of `count` random overlapping spheres inside the cube `[-1.9, 1.9]^3`.
fn blob(seed: u64, count: u64) -> Box<dyn ImplicitSurface> {
    let mut combined: Option<Box<dyn ImplicitSurface>> = None;
    for k in 0..count {
        let value = |salt: u64| hash01(seed * 10_000 + k * 16 + salt);
        let centre = [
            (value(0) - 0.5) * 2.4,
            (value(1) - 0.5) * 2.4,
            (value(2) - 0.5) * 2.4,
        ];
        let sphere: Box<dyn ImplicitSurface> =
            Box::new(SdfSphere::new(centre, 0.35 + 0.35 * value(3)));
        combined = Some(match combined {
            None => sphere,
            Some(rest) => Box::new(CsgUnion::new(rest, sphere)),
        });
    }
    combined.unwrap_or_else(|| Box::new(SdfSphere::new([0.0; 3], 1.0)))
}

#[test]
fn unions_of_random_spheres_give_closed_oriented_surfaces() {
    // Overlapping blobs create saddles, tunnels and the ambiguous cube
    // configurations of marching cubes; the welded mesh must stay watertight
    // and consistently oriented, and refining the grid must converge to the
    // same solid.
    for seed in 0..6 {
        let field = blob(seed, 7);
        let label = format!("blob {seed}");
        let (vertices, triangles) = extract_mesh_simple(field.as_ref(), [-2.5; 3], [2.5; 3], 48);
        let euler = closed_euler_characteristic(&vertices, &triangles, &label);
        assert_eq!(euler % 2, 0, "{label}: Euler characteristic {euler}");
        let volume = enclosed_volume(&vertices, &triangles);
        assert!(volume > 0.0, "{label}: volume {volume}");

        let (fine_vertices, fine_triangles) =
            extract_mesh_simple(field.as_ref(), [-2.5; 3], [2.5; 3], 96);
        let fine_euler = closed_euler_characteristic(&fine_vertices, &fine_triangles, &label);
        assert_eq!(
            fine_euler % 2,
            0,
            "{label}: fine Euler characteristic {fine_euler}"
        );
        assert_close(
            volume,
            enclosed_volume(&fine_vertices, &fine_triangles),
            0.05,
            &format!("{label}: volume"),
        );
        assert_close(
            surface_area(&vertices, &triangles),
            surface_area(&fine_vertices, &fine_triangles),
            0.05,
            &format!("{label}: area"),
        );
    }
}
