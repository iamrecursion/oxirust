// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of [`super::shape_diameter`] on meshes with a known thickness: a slab,
//! a capped cylinder and a sphere, plus orientation, scale and input-validation
//! behaviour.

use super::shape_diameter;
use crate::mesh_boolean::SimpleMesh;
use crate::test_meshes::icosphere;
use std::f64::consts::{FRAC_PI_2, PI};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Signed volume enclosed by the mesh (positive for outward orientation).
fn signed_volume(mesh: &SimpleMesh) -> f64 {
    mesh.triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (
                mesh.vertices[t[0]],
                mesh.vertices[t[1]],
                mesh.vertices[t[2]],
            );
            dot(a, cross(b, c)) / 6.0
        })
        .sum()
}

/// Axis-aligned box with outward, counter-clockwise triangles.
fn box_mesh(lo: V3, hi: V3) -> SimpleMesh {
    let corner = |i: usize| -> V3 {
        [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ]
    };
    let vertices: Vec<V3> = (0..8).map(corner).collect();
    let centre = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let quads: [[usize; 4]; 6] = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ];
    let mut triangles = Vec::new();
    for quad in quads {
        let [a, b, c, d] = quad;
        let normal = cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a]));
        let face_centre = [
            0.25 * (vertices[a][0] + vertices[b][0] + vertices[c][0] + vertices[d][0]),
            0.25 * (vertices[a][1] + vertices[b][1] + vertices[c][1] + vertices[d][1]),
            0.25 * (vertices[a][2] + vertices[b][2] + vertices[c][2] + vertices[d][2]),
        ];
        if dot(normal, sub(face_centre, centre)) >= 0.0 {
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        } else {
            triangles.push([a, c, b]);
            triangles.push([a, d, c]);
        }
    }
    SimpleMesh::from_data(vertices, triangles)
}

/// Capped cylinder around the z axis from `z = 0` to `z = height` with outward,
/// counter-clockwise triangles.
fn cylinder_mesh(radius: f64, height: f64, segments: usize) -> SimpleMesh {
    let mut vertices: Vec<V3> = Vec::new();
    for i in 0..segments {
        let angle = 2.0 * PI * i as f64 / segments as f64;
        vertices.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
    }
    for i in 0..segments {
        let angle = 2.0 * PI * i as f64 / segments as f64;
        vertices.push([radius * angle.cos(), radius * angle.sin(), height]);
    }
    let bottom_centre = vertices.len();
    vertices.push([0.0, 0.0, 0.0]);
    let top_centre = vertices.len();
    vertices.push([0.0, 0.0, height]);
    let mut triangles = Vec::new();
    for i in 0..segments {
        let j = (i + 1) % segments;
        triangles.push([i, j, segments + j]);
        triangles.push([i, segments + j, segments + i]);
        triangles.push([top_centre, segments + i, segments + j]);
        triangles.push([bottom_centre, j, i]);
    }
    SimpleMesh::from_data(vertices, triangles)
}

fn flipped(mesh: &SimpleMesh) -> SimpleMesh {
    SimpleMesh::from_data(
        mesh.vertices.clone(),
        mesh.triangles.iter().map(|t| [t[0], t[2], t[1]]).collect(),
    )
}

/// 10 x 10 slab of thickness 2 between `z = 0` and `z = 2`.
fn slab() -> SimpleMesh {
    box_mesh([-5.0, -5.0, 0.0], [5.0, 5.0, 2.0])
}

#[test]
fn test_mesh_builders_are_outward_oriented() {
    assert!((signed_volume(&slab()) - 200.0).abs() < 1e-9);
    let cylinder = cylinder_mesh(1.0, 10.0, 96);
    let expected = 0.5 * 96.0 * (2.0 * PI / 96.0).sin() * 10.0;
    assert!((signed_volume(&cylinder) - expected).abs() < 1e-9);
    assert!(signed_volume(&flipped(&slab())) < 0.0);
}

#[test]
fn slab_thickness_is_read_along_the_normal() {
    let mesh = slab();
    // Top face, a point on a triangle diagonal and a generic point, bottom face.
    let queries: [(V3, V3); 4] = [
        ([1.3, -0.7, 2.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]),
        ([0.4, 0.4, 0.0], [0.0, 0.0, -1.0]),
        ([-3.0, 2.0, 2.0], [0.0, 0.0, 7.5]),
    ];
    for (p, n) in queries {
        for rays in [1, 2, 30] {
            let d = shape_diameter(&mesh, p, n, rays, 0.0).expect("slab hit");
            assert!((d - 2.0).abs() < 1e-6, "p = {p:?}, {rays} rays: {d}");
        }
    }
    // Through the side faces the slab is 10 wide.
    let d = shape_diameter(&mesh, [5.0, 0.1, 1.0], [1.0, 0.0, 0.0], 8, 0.0).expect("side hit");
    assert!((d - 10.0).abs() < 1e-6, "{d}");
}

#[test]
fn slab_median_over_a_cone_matches_the_analytic_value() {
    // Oblique rays of a 60 degree cone travel 2 / cos(theta) through the
    // slab; cos(theta) is uniform on [1/2, 1], so its median is 3/4.
    let mesh = slab();
    let d =
        shape_diameter(&mesh, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 2000, PI / 3.0).expect("slab cone");
    assert!((d - 2.0 / 0.75).abs() < 0.01 * (2.0 / 0.75), "{d}");
    // A wider cone can only lengthen the typical ray.
    let narrow = shape_diameter(&mesh, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 400, 0.2).expect("narrow");
    assert!(narrow > 2.0 && narrow < d, "{narrow} vs {d}");
}

#[test]
fn cylinder_diameter_is_twice_the_radius() {
    let (radius, segments) = (1.0, 96);
    let mesh = cylinder_mesh(radius, 10.0 * radius, segments);
    // Centre of the facet between the vertex columns 0 and 1.
    let facet_angle = PI / segments as f64;
    let wall = radius * facet_angle.cos();
    let p = [
        wall * facet_angle.cos(),
        wall * facet_angle.sin(),
        5.0 * radius,
    ];
    let n = [facet_angle.cos(), facet_angle.sin(), 0.0];
    let axis_ray = shape_diameter(&mesh, p, n, 1, 0.0).expect("axis ray");
    assert!(
        (axis_ray - 2.0 * wall).abs() < 1e-6,
        "opposite facet is {} away, got {axis_ray}",
        2.0 * wall
    );
    for half_angle in [PI / 6.0, PI / 4.0] {
        let d = shape_diameter(&mesh, p, n, 400, half_angle).expect("cone");
        assert!(
            (d - 2.0 * radius).abs() < 0.1 * 2.0 * radius,
            "half angle {half_angle}: {d}"
        );
    }
}

#[test]
fn open_tube_without_caps_still_measures_its_diameter() {
    // Only the side wall of the cylinder: the rays leave the tube through the
    // opposite wall, so the missing caps do not matter.
    let (radius, segments) = (2.0, 64);
    let capped = cylinder_mesh(radius, 10.0 * radius, segments);
    let tube = SimpleMesh::from_data(
        capped.vertices.clone(),
        capped
            .triangles
            .iter()
            .copied()
            .filter(|t| t[0] < 2 * segments && t[1] < 2 * segments && t[2] < 2 * segments)
            .collect(),
    );
    assert_eq!(tube.triangles.len(), 2 * segments);
    let facet_angle = PI / segments as f64;
    let n = [facet_angle.cos(), facet_angle.sin(), 0.0];
    let wall = radius * facet_angle.cos();
    let p = [wall * n[0], wall * n[1], 10.0];
    let d = shape_diameter(&tube, p, n, 1, 0.0).expect("tube axis ray");
    assert!((d - 2.0 * wall).abs() < 1e-6, "{d}");
    let d = shape_diameter(&tube, p, n, 300, PI / 4.0).expect("tube cone");
    assert!((d - 2.0 * radius).abs() < 0.1 * 2.0 * radius, "{d}");
}

#[test]
fn cylinder_thickness_scales_with_radius() {
    let facet_angle = PI / 64.0;
    let n = [facet_angle.cos(), facet_angle.sin(), 0.0];
    for radius in [0.25, 1.0, 4.0] {
        let mesh = cylinder_mesh(radius, 10.0 * radius, 64);
        let wall = radius * facet_angle.cos();
        let p = [wall * n[0], wall * n[1], 5.0 * radius];
        let d = shape_diameter(&mesh, p, n, 200, PI / 4.0).expect("cylinder");
        assert!((d / radius - 2.0).abs() < 0.2, "radius {radius}: {d}");
    }
}

#[test]
fn sphere_diameter_through_the_antipodal_vertex() {
    let (vertices, triangles) = icosphere(3);
    let mesh = SimpleMesh::from_data(vertices.clone(), triangles);
    for &p in vertices.iter().take(5) {
        let d = shape_diameter(&mesh, p, p, 3, 0.0).expect("diameter");
        assert!((d - 2.0).abs() < 1e-6, "{d}");
    }
    // Chord length 2 R cos(theta), cos(theta) uniform on [cos(a), 1]: the
    // median is R (1 + cos(a)).
    let half_angle = PI / 6.0;
    let expected = 1.0 + half_angle.cos();
    let d = shape_diameter(&mesh, vertices[0], vertices[0], 500, half_angle).expect("cone");
    assert!((d - expected).abs() < 0.03, "{d} vs {expected}");
}

#[test]
fn result_is_invariant_under_rotation_of_the_mesh_and_query() {
    let rotate = |v: V3| -> V3 {
        // Rotation by 0.7 rad about the axis (1, 2, 3)/sqrt(14) (Rodrigues).
        let k = {
            let l = 14.0_f64.sqrt();
            [1.0 / l, 2.0 / l, 3.0 / l]
        };
        let (s, c) = 0.7_f64.sin_cos();
        let kxv = cross(k, v);
        let kdv = dot(k, v);
        [
            v[0] * c + kxv[0] * s + k[0] * kdv * (1.0 - c),
            v[1] * c + kxv[1] * s + k[1] * kdv * (1.0 - c),
            v[2] * c + kxv[2] * s + k[2] * kdv * (1.0 - c),
        ]
    };
    let mesh = slab();
    let rotated = SimpleMesh::from_data(
        mesh.vertices.iter().map(|&v| rotate(v)).collect(),
        mesh.triangles.clone(),
    );
    let p = rotate([1.0, 2.0, 2.0]);
    let n = rotate([0.0, 0.0, 1.0]);
    let d = shape_diameter(&rotated, p, n, 12, 0.0).expect("rotated slab");
    assert!((d - 2.0).abs() < 1e-6, "{d}");
    let cone_original = shape_diameter(&mesh, [1.0, 2.0, 2.0], [0.0, 0.0, 1.0], 300, 0.6);
    let cone_rotated = shape_diameter(&rotated, p, n, 300, 0.6);
    let (a, b) = (
        cone_original.expect("original"),
        cone_rotated.expect("rotated"),
    );
    assert!((a - b).abs() < 1e-6, "{a} vs {b}");
}

#[test]
fn result_does_not_depend_on_the_mesh_scale() {
    // Absolute ray-triangle tolerances would reject every triangle of a
    // micrometre-sized mesh.
    for scale in [1e-7, 1.0, 1e7] {
        let mesh = box_mesh(
            [-5.0 * scale, -5.0 * scale, 0.0],
            [5.0 * scale, 5.0 * scale, 2.0 * scale],
        );
        let d = shape_diameter(&mesh, [0.0, 0.0, 2.0 * scale], [0.0, 0.0, 1.0], 5, 0.0)
            .expect("scaled slab");
        assert!((d / scale - 2.0).abs() < 1e-6, "scale {scale}: {d}");
    }
}

#[test]
fn inward_wound_meshes_have_no_valid_rays() {
    // Rays only count when they leave the solid through an outward-facing
    // triangle; on an inside-out mesh every ray enters instead.
    let inverted = flipped(&slab());
    assert_eq!(
        shape_diameter(&inverted, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 20, 0.0),
        None
    );
    assert_eq!(
        shape_diameter(&inverted, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 20, 0.5),
        None
    );
}

#[test]
fn rays_that_miss_or_have_nothing_to_hit_give_none() {
    let mesh = slab();
    // Far away from the slab, the rays pass beside it.
    assert_eq!(
        shape_diameter(&mesh, [100.0, 100.0, 100.0], [1.0, 0.0, 0.0], 10, 0.0),
        None
    );
    // A normal that points into the solid sends the rays outwards.
    assert_eq!(
        shape_diameter(&mesh, [0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 10, 0.0),
        None
    );
    // A single triangle is not a solid.
    let lone = SimpleMesh::from_data(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        vec![[0, 1, 2]],
    );
    assert_eq!(
        shape_diameter(&lone, [0.2, 0.2, 0.0], [0.0, 0.0, 1.0], 10, 0.0),
        None
    );
}

#[test]
fn invalid_input_gives_none() {
    let mesh = slab();
    let (p, n) = ([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]);
    assert!(shape_diameter(&mesh, p, n, 4, 0.0).is_some());
    assert_eq!(shape_diameter(&mesh, p, n, 0, 0.0), None);
    assert_eq!(shape_diameter(&mesh, p, n, 4, -0.1), None);
    assert_eq!(shape_diameter(&mesh, p, n, 4, f64::NAN), None);
    assert_eq!(shape_diameter(&mesh, p, n, 4, FRAC_PI_2), None);
    assert_eq!(shape_diameter(&mesh, p, n, 4, f64::INFINITY), None);
    assert_eq!(shape_diameter(&mesh, p, [0.0; 3], 4, 0.0), None);
    assert_eq!(shape_diameter(&mesh, p, [f64::NAN, 0.0, 1.0], 4, 0.0), None);
    assert_eq!(shape_diameter(&mesh, [f64::NAN, 0.0, 2.0], n, 4, 0.0), None);
    assert_eq!(
        shape_diameter(&SimpleMesh::new(), p, n, 4, 0.0),
        None,
        "empty mesh"
    );
}

#[test]
fn malformed_triangles_are_ignored() {
    let mut mesh = slab();
    let clean = shape_diameter(&mesh, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 6, 0.0);
    let nan_vertex = mesh.vertices.len();
    mesh.vertices.push([f64::NAN, 0.0, 1.0]);
    mesh.triangles.push([0, 1, 9999]); // index out of range
    mesh.triangles.push([0, 1, nan_vertex]); // non-finite coordinate
    mesh.triangles.push([0, 0, 1]); // repeated index, zero area
    let dirty = shape_diameter(&mesh, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 6, 0.0);
    assert!(clean.is_some());
    assert_eq!(clean, dirty);
}

#[test]
fn result_is_deterministic() {
    let mesh = cylinder_mesh(1.0, 10.0, 48);
    let p = [1.0, 0.0, 5.0];
    let n = [1.0, 0.0, 0.0];
    let first = shape_diameter(&mesh, p, n, 100, 0.8);
    let second = shape_diameter(&mesh, p, n, 100, 0.8);
    assert!(first.is_some());
    assert_eq!(first, second);
}
