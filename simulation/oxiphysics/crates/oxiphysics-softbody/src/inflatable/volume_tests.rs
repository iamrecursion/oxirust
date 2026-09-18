// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the face winding of the generated balloons and for
//! the volume constraint.
//!
//! * [`InflatableBody::sphere`] and [`balloon_model`] used to wind their
//!   triangles clockwise seen from outside (signed volume < 0) although the
//!   faces are documented as counter-clockwise.
//! * [`apply_volume_constraint`] combined the *unsigned* volume `|V|` with the
//!   gradient of the *signed* volume. On such a mesh the "inflating" step
//!   shrank the body, collapsed it through zero volume and left it inside
//!   out.

use super::*;
use std::f64::consts::PI;

/// The same mesh with every face wound the other way (inward normals).
fn reversed(body: &InflatableBody) -> InflatableBody {
    let mut flipped = body.clone();
    for face in &mut flipped.faces {
        face.indices.swap(1, 2);
    }
    flipped
}

/// `normal . centroid` of a triangle: positive for the outward faces of a
/// convex body that contains the origin.
fn outwardness(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let normal = vec_cross(vec_sub(b, a), vec_sub(c, a));
    let centroid = vec_scale(vec_add(vec_add(a, b), c), 1.0 / 3.0);
    vec_dot(normal, centroid)
}

#[test]
fn sphere_faces_point_outward_and_enclose_a_positive_volume() {
    for subdivisions in 0..=5_u32 {
        for radius in [0.5, 1.0, 2.5] {
            let body = InflatableBody::sphere(radius, subdivisions, 1.0);
            let signed = body.signed_total_volume();
            assert!(
                signed > 0.0,
                "sphere({radius}, {subdivisions}): signed volume {signed}"
            );
            assert!((body.total_volume() - signed).abs() < 1e-12 * signed);
            // Inscribed polyhedron: smaller than the ball it is inscribed in.
            assert!(signed < 4.0 / 3.0 * PI * radius.powi(3));
            for (f, face) in body.faces.iter().enumerate() {
                let [i0, i1, i2] = face.indices;
                let w = outwardness(
                    body.vertices[i0].pos,
                    body.vertices[i1].pos,
                    body.vertices[i2].pos,
                );
                assert!(
                    w > 0.0,
                    "sphere({radius}, {subdivisions}) face {f} {:?} points inward ({w})",
                    face.indices
                );
            }
        }
    }
    let unit = InflatableBody::sphere(1.0, 3, 1.0).signed_total_volume();
    let double = InflatableBody::sphere(2.0, 3, 1.0).signed_total_volume();
    assert!((double - 8.0 * unit).abs() < 1e-12 * double);
}

#[test]
fn sphere_mesh_is_a_closed_manifold() {
    // Every undirected edge is shared by exactly two faces and, with a
    // consistent winding, traversed once in each direction.
    for subdivisions in 0..=4_u32 {
        let body = InflatableBody::sphere(1.0, subdivisions, 1.0);
        let mut directed = std::collections::HashMap::<(usize, usize), usize>::new();
        for face in &body.faces {
            let [i0, i1, i2] = face.indices;
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                *directed.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &count) in &directed {
            assert_eq!(count, 1, "directed edge ({a}, {b}) used {count} times");
            assert_eq!(
                directed.get(&(b, a)),
                Some(&1),
                "edge ({a}, {b}) has no opposite twin: inconsistent winding"
            );
        }
        assert_eq!(directed.len(), 2 * body.edges.len());
    }
}

#[test]
fn balloon_model_faces_point_outward_and_enclose_a_positive_volume() {
    for panels in [2_usize, 3, 4, 6, 9] {
        let radius = 1.5;
        let shell = balloon_model(panels, radius);
        let mut six_volume = 0.0;
        for (t, tri) in shell.tris.iter().enumerate() {
            let a = shell.vertices[tri[0]];
            let b = shell.vertices[tri[1]];
            let c = shell.vertices[tri[2]];
            six_volume += vec_dot(a, vec_cross(b, c));
            assert!(
                outwardness(a, b, c) > 0.0,
                "balloon_model({panels}) triangle {t} {tri:?} points inward"
            );
        }
        let signed = six_volume / 6.0;
        assert!(
            signed > 0.0,
            "balloon_model({panels}): signed volume {signed}"
        );
        assert!(signed < 4.0 / 3.0 * PI * radius.powi(3));
        assert!((signed - shell.compute_enclosed_volume()).abs() < 1e-12 * signed);
    }
}

#[test]
fn balloon_model_pressure_pushes_every_vertex_outward() {
    let mut shell = balloon_model(4, 1.0);
    shell.pressure = 250.0;
    for (i, vertex) in shell.vertices.iter().enumerate() {
        let force = shell.pressure_force(i);
        assert!(
            vec_dot(force, *vertex) > 0.0,
            "vertex {i} at {vertex:?} is pushed inward by {force:?}"
        );
    }
}

#[test]
fn signed_total_volume_matches_a_tetrahedron_and_flips_with_the_winding() {
    let verts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let outward: Vec<InflatableFace> = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        .iter()
        .map(|&indices| InflatableFace { indices })
        .collect();
    let inward: Vec<InflatableFace> = outward
        .iter()
        .map(|f| InflatableFace {
            indices: [f.indices[0], f.indices[2], f.indices[1]],
        })
        .collect();
    assert!((signed_total_volume(&verts, &outward) - 1.0 / 6.0).abs() < 1e-15);
    assert!((signed_total_volume(&verts, &inward) + 1.0 / 6.0).abs() < 1e-15);
    assert!((total_volume(&verts, &outward) - 1.0 / 6.0).abs() < 1e-15);
    assert!((total_volume(&verts, &inward) - 1.0 / 6.0).abs() < 1e-15);
}

#[test]
fn volume_gradient_is_the_gradient_of_the_signed_volume() {
    let mut body = InflatableBody::sphere(1.0, 2, 1.0);
    for (i, vertex) in body.vertices.iter_mut().enumerate() {
        let s = i as f64;
        vertex.pos[0] += 0.05 * (1.3 * s).sin();
        vertex.pos[1] += 0.05 * (0.7 * s).cos();
        vertex.pos[2] += 0.05 * (2.1 * s).sin();
    }
    let gradient = volume_constraint_gradient(&body);
    let positions: Vec<[f64; 3]> = body.vertices.iter().map(|v| v.pos).collect();
    let h = 1e-6;
    for (vertex, g) in gradient.iter().enumerate() {
        for (axis, analytic) in g.iter().enumerate() {
            let mut plus = positions.clone();
            plus[vertex][axis] += h;
            let mut minus = positions.clone();
            minus[vertex][axis] -= h;
            let numeric = (signed_total_volume(&plus, &body.faces)
                - signed_total_volume(&minus, &body.faces))
                / (2.0 * h);
            assert!(
                (analytic - numeric).abs() < 1e-8,
                "vertex {vertex} axis {axis}: {analytic} vs {numeric}"
            );
        }
    }
    // The volume is homogeneous of degree three: sum x_i . grad_i = 3 V.
    let euler: f64 = positions
        .iter()
        .zip(gradient.iter())
        .map(|(x, g)| vec_dot(*x, *g))
        .sum();
    let signed = body.signed_total_volume();
    assert!(
        (euler - 3.0 * signed).abs() < 1e-10,
        "{euler} vs {}",
        3.0 * signed
    );
    // Reversing the winding negates the volume and its gradient.
    let flipped = volume_constraint_gradient(&reversed(&body));
    for (g, h) in gradient.iter().zip(flipped.iter()) {
        for axis in 0..3 {
            assert!((g[axis] + h[axis]).abs() < 1e-12);
        }
    }
}

#[test]
fn one_constraint_step_inflates_the_sphere_without_turning_it_inside_out() {
    for flipped in [false, true] {
        let mut body = InflatableBody::sphere(1.0, 1, 1.0);
        if flipped {
            body = reversed(&body);
        }
        let orientation = body.signed_total_volume().signum();
        assert_eq!(orientation, if flipped { -1.0 } else { 1.0 });
        let v0 = body.total_volume();
        let target = 1.1 * v0;
        apply_volume_constraint(&mut body, target, 1e-6, 0.01);
        let v1 = body.total_volume();
        assert!(
            v1 > v0,
            "flipped = {flipped}: the volume must increase, {v0} -> {v1}"
        );
        assert!(
            (v1 - target).abs() < 0.01 * target,
            "flipped = {flipped}: one step should land near the target, {v1} vs {target}"
        );
        assert_eq!(
            body.signed_total_volume().signum(),
            orientation,
            "flipped = {flipped}: the body must not turn inside out"
        );
        assert!(
            body.vertices[0].pos[1] > 0.0,
            "flipped = {flipped}: the north pole crossed the equator: {:?}",
            body.vertices[0].pos
        );
    }
}

#[test]
fn volume_constraint_converges_to_the_target_for_either_winding() {
    for flipped in [false, true] {
        for factor in [0.8, 1.1, 1.5] {
            let mut body = InflatableBody::sphere(1.0, 1, 1.0);
            if flipped {
                body = reversed(&body);
            }
            let orientation = body.signed_total_volume().signum();
            let target = factor * body.total_volume();
            for iteration in 0..20 {
                apply_volume_constraint(&mut body, target, 1e-6, 0.01);
                assert_eq!(
                    body.signed_total_volume().signum(),
                    orientation,
                    "flipped = {flipped}, x{factor}, iteration {iteration}: inverted"
                );
            }
            let volume = body.total_volume();
            assert!(
                (volume - target).abs() < 0.01 * target,
                "flipped = {flipped}, x{factor}: volume {volume} vs target {target}"
            );
            for vertex in &body.vertices {
                let radius = vec_len(vertex.pos);
                assert!(
                    (0.5..1.6).contains(&radius),
                    "flipped = {flipped}, x{factor}: vertex at radius {radius}"
                );
            }
        }
    }
}

#[test]
fn volume_constraint_at_the_target_does_not_move_the_body() {
    let mut body = InflatableBody::sphere(1.0, 2, 1.0);
    let before: Vec<[f64; 3]> = body.vertices.iter().map(|v| v.pos).collect();
    let target = body.total_volume();
    apply_volume_constraint(&mut body, target, 1e-6, 0.01);
    for (vertex, old) in body.vertices.iter().zip(before.iter()) {
        assert!(vec_len(vec_sub(vertex.pos, *old)) < 1e-12);
    }
}
