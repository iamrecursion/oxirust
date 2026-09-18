// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for contacts at zero distance.
//!
//! A vertex that lies exactly on a face, or two edges that cross, are the deepest
//! contacts there are. The vertex-face and edge-edge tests used to discard every
//! candidate closer than `1e-10`, because the direction between the closest
//! points is undefined there, and so let cloth authored coplanar with a collider
//! pass straight through it. They now fall back to the face normal and to the
//! common perpendicular of the edges and report the full thickness as depth.

use super::functions::{
    detect_edge_edge_contacts, detect_self_collision, detect_vertex_face_contacts,
    edge_edge_contact, make_quad_mesh, vertex_face_contact,
};
use super::*;

const THICKNESS: f64 = 0.01;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn assert_unit(normal: [f64; 3]) {
    assert!(
        (length(normal) - 1.0).abs() < 1e-12,
        "normal {normal:?} is not a unit vector"
    );
}

fn assert_close(actual: [f64; 3], expected: [f64; 3], tol: f64) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() <= tol,
            "{actual:?} differs from {expected:?}"
        );
    }
}

fn collision_config() -> DeformCollisionConfig {
    DeformCollisionConfig {
        thickness: THICKNESS,
        edge_edge: true,
        self_collision: false,
        adjacency_skip: false,
    }
}

/// Triangle in the plane z = 0 whose winding normal is +z.
const A: [f64; 3] = [0.0, 0.0, 0.0];
const B: [f64; 3] = [2.0, 0.0, 0.0];
const C: [f64; 3] = [0.0, 2.0, 0.0];

// ── vertex_face_contact ──────────────────────────────────────────────────────

#[test]
fn vertex_at_the_face_centroid_is_a_full_depth_contact() {
    let centroid = [2.0 / 3.0, 2.0 / 3.0, 0.0];
    let contact = vertex_face_contact(centroid, A, B, C, THICKNESS, 7, 9)
        .expect("a vertex on the face must produce a contact");
    assert_eq!(contact.depth, THICKNESS);
    assert_close(contact.normal, [0.0, 0.0, 1.0], 1e-15);
    assert_unit(contact.normal);
    assert_eq!(contact.contact_type, DeformContactType::VertexFace);
    assert_eq!((contact.feature_a, contact.feature_b), (7, 9));
    assert_close(contact.position, centroid, 1e-15);
    for weight in contact.bary {
        assert!((weight - 1.0 / 3.0).abs() < 1e-12, "{:?}", contact.bary);
    }
}

#[test]
fn the_face_normal_follows_the_winding_of_the_triangle() {
    let centroid = [2.0 / 3.0, 2.0 / 3.0, 0.0];
    let flipped = vertex_face_contact(centroid, A, C, B, THICKNESS, 0, 0)
        .expect("a vertex on the face must produce a contact");
    assert_close(flipped.normal, [0.0, 0.0, -1.0], 1e-15);
    assert_eq!(flipped.depth, THICKNESS);
}

#[test]
fn vertices_on_an_edge_or_a_corner_of_the_face_are_contacts_too() {
    for point in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.5, 0.0], A, B, C] {
        let contact = vertex_face_contact(point, A, B, C, THICKNESS, 0, 0)
            .unwrap_or_else(|| panic!("no contact for the boundary point {point:?}"));
        assert_eq!(contact.depth, THICKNESS, "{point:?}");
        assert_close(contact.normal, [0.0, 0.0, 1.0], 1e-15);
    }
}

#[test]
fn a_vertex_a_hair_above_the_face_takes_the_same_branch() {
    // Closer than the 1e-10 geometric tolerance: still the deepest contact.
    let contact = vertex_face_contact([0.5, 0.5, 1e-12], A, B, C, THICKNESS, 0, 0)
        .expect("contact within the tolerance");
    assert_eq!(contact.depth, THICKNESS);
    assert_close(contact.normal, [0.0, 0.0, 1.0], 1e-15);
}

#[test]
fn depth_is_continuous_with_the_distance_away_from_the_face() {
    // Above the tolerance the normal comes from the offset and the depth is the
    // remaining thickness, as before.
    let near = vertex_face_contact([0.5, 0.5, 1e-6], A, B, C, THICKNESS, 0, 0).expect("contact");
    assert!((near.depth - (THICKNESS - 1e-6)).abs() < 1e-15, "{near:?}");
    assert_close(near.normal, [0.0, 0.0, 1.0], 1e-12);
    // On the back side the offset decides the side: the normal flips.
    let behind =
        vertex_face_contact([0.5, 0.5, -0.004], A, B, C, THICKNESS, 0, 0).expect("contact");
    assert!((behind.depth - 0.006).abs() < 1e-15, "{behind:?}");
    assert_close(behind.normal, [0.0, 0.0, -1.0], 1e-12);
    // Beyond the thickness there is no contact.
    assert!(vertex_face_contact([0.5, 0.5, 0.02], A, B, C, THICKNESS, 0, 0).is_none());
}

#[test]
fn zero_thickness_has_no_contact_zone() {
    assert!(vertex_face_contact([0.5, 0.5, 0.0], A, B, C, 0.0, 0, 0).is_none());
}

#[test]
fn a_vertex_on_a_degenerate_face_gets_a_unit_normal_perpendicular_to_it() {
    // Collinear triangle along the x axis.
    let contact = vertex_face_contact(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        THICKNESS,
        0,
        0,
    )
    .expect("a vertex on the degenerate face is a contact");
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    assert!(contact.normal[0].abs() < 1e-12, "{:?}", contact.normal);
    // A triangle collapsed to a point still yields a valid unit normal.
    let point = [3.0, 3.0, 3.0];
    let collapsed = vertex_face_contact(point, point, point, point, THICKNESS, 0, 0)
        .expect("a vertex at the collapsed face is a contact");
    assert_unit(collapsed.normal);
    assert_eq!(collapsed.depth, THICKNESS);
}

// ── edge_edge_contact ────────────────────────────────────────────────────────

#[test]
fn crossing_edges_are_a_full_depth_contact() {
    let contact = edge_edge_contact(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        THICKNESS,
        3,
        4,
    )
    .expect("crossing edges must produce a contact");
    assert_eq!(contact.depth, THICKNESS);
    // (p1 - p0) x (q1 - q0) = (2, 0, 0) x (0, 2, 0) points along +z.
    assert_close(contact.normal, [0.0, 0.0, 1.0], 1e-15);
    assert_unit(contact.normal);
    assert_eq!(contact.contact_type, DeformContactType::EdgeEdge);
    assert_eq!((contact.feature_a, contact.feature_b), (3, 4));
    assert!((contact.bary[0] - 0.5).abs() < 1e-12 && (contact.bary[1] - 0.5).abs() < 1e-12);
    assert_close(contact.position, [0.0, 0.0, 0.0], 1e-15);
}

#[test]
fn the_normal_of_crossing_edges_is_perpendicular_to_both_edges() {
    let p0 = [0.2, -1.0, 0.4];
    let p1 = [0.2, 1.0, 0.4];
    let q0 = [-0.7, 0.3, 0.4];
    let q1 = [1.1, 0.3, 0.4 + 1e-14];
    let contact = edge_edge_contact(p0, p1, q0, q1, THICKNESS, 0, 0).expect("contact");
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    let d1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let d2 = [q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]];
    assert!(dot(contact.normal, d1).abs() < 1e-9);
    assert!(dot(contact.normal, d2).abs() < 1e-9);
}

#[test]
fn swapping_the_edges_flips_the_normal_of_crossing_edges() {
    let p0 = [-1.0, 0.0, 0.0];
    let p1 = [1.0, 0.0, 0.0];
    let q0 = [0.0, -1.0, 0.0];
    let q1 = [0.0, 1.0, 0.0];
    let ab = edge_edge_contact(p0, p1, q0, q1, THICKNESS, 0, 0).expect("contact");
    let ba = edge_edge_contact(q0, q1, p0, p1, THICKNESS, 0, 0).expect("contact");
    assert_close(ab.normal, [0.0, 0.0, 1.0], 1e-15);
    assert_close(ba.normal, [0.0, 0.0, -1.0], 1e-15);
}

#[test]
fn overlapping_parallel_edges_fall_back_to_a_perpendicular_of_the_edge() {
    let p0 = [0.0, 0.0, 0.0];
    let p1 = [2.0, 0.0, 0.0];
    let q0 = [1.0, 0.0, 0.0];
    let q1 = [3.0, 0.0, 0.0];
    let contact = edge_edge_contact(p0, p1, q0, q1, THICKNESS, 0, 0).expect("contact");
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    assert!(contact.normal[0].abs() < 1e-12, "{:?}", contact.normal);
}

#[test]
fn a_degenerate_edge_on_another_edge_still_has_a_unit_normal() {
    // The first edge has collapsed to the point (0, 0, 0), which lies on the second.
    let origin = [0.0, 0.0, 0.0];
    let contact = edge_edge_contact(
        origin,
        origin,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        THICKNESS,
        0,
        0,
    )
    .expect("contact");
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    assert!(contact.normal[0].abs() < 1e-12, "{:?}", contact.normal);
    // Two coincident points: any unit normal will do, but it must be one.
    let both = edge_edge_contact(origin, origin, origin, origin, THICKNESS, 0, 0).expect("contact");
    assert_unit(both.normal);
    assert_eq!(both.depth, THICKNESS);
}

#[test]
fn edges_that_pass_close_but_do_not_touch_are_unchanged() {
    let p0 = [0.0, 0.0, -1.0];
    let p1 = [0.0, 0.0, 1.0];
    let q0 = [-1.0, 0.004, 0.0];
    let q1 = [1.0, 0.004, 0.0];
    let contact = edge_edge_contact(p0, p1, q0, q1, THICKNESS, 0, 0).expect("contact");
    assert!((contact.depth - 0.006).abs() < 1e-15, "{contact:?}");
    // The normal points from edge q to edge p: down, here.
    assert_close(contact.normal, [0.0, -1.0, 0.0], 1e-12);
    let far = [-1.0, 0.5, 0.0];
    assert!(edge_edge_contact(p0, p1, far, [1.0, 0.5, 0.0], THICKNESS, 0, 0).is_none());
}

// ── the mesh detectors ───────────────────────────────────────────────────────

fn sheet(vertices: [[f64; 3]; 3]) -> DeformableMesh {
    DeformableMesh::new(vertices.to_vec(), vec![TriFace::new(0, 1, 2)], vec![1.0; 3])
}

#[test]
fn a_vertex_lying_exactly_on_a_collider_face_is_reported_by_the_mesh_detector() {
    // Cloth authored coplanar with the collider: its first vertex lies exactly in
    // the interior of the collider's triangle.
    let cloth = sheet([[0.5, 0.0, 0.5], [5.0, 3.0, 5.0], [6.0, 3.0, 5.0]]);
    let collider = sheet([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]);
    let bvh_cloth = DeformBvh::build(&cloth);
    let bvh_collider = DeformBvh::build(&collider);
    let contacts = detect_vertex_face_contacts(
        &cloth,
        &collider,
        &bvh_cloth,
        &bvh_collider,
        &collision_config(),
    );
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    let contact = &contacts[0];
    assert_eq!(contact.contact_type, DeformContactType::VertexFace);
    assert_eq!((contact.feature_a, contact.feature_b), (0, 0));
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    // The collider winding (b - a) x (c - a) = (2, 0, 0) x (0, 0, 2) = (0, -4, 0).
    assert_close(contact.normal, [0.0, -1.0, 0.0], 1e-15);
}

#[test]
fn crossing_edges_of_two_sheets_are_reported_by_the_mesh_detector() {
    // Both triangles lie in the plane y = 0; the edge of A along x crosses the
    // edge of B along z at the origin, and no vertex is within the thickness of
    // the other triangle's plane region only through that crossing.
    let sheet_a = sheet([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -3.0]]);
    let sheet_b = sheet([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [-4.0, 0.0, 0.0]]);
    let bvh_a = DeformBvh::build(&sheet_a);
    let bvh_b = DeformBvh::build(&sheet_b);
    let contacts =
        detect_edge_edge_contacts(&sheet_a, &sheet_b, &bvh_a, &bvh_b, &collision_config());
    let crossing = contacts
        .iter()
        .find(|c| (c.feature_a, c.feature_b) == (0, 0))
        .expect("the crossing edges must produce a contact");
    assert_eq!(crossing.depth, THICKNESS);
    assert_unit(crossing.normal);
    // (1 - -1, 0, 0) x (0, 0, 2) = (0, -4, 0): along -y.
    assert_close(crossing.normal, [0.0, -1.0, 0.0], 1e-15);
    assert_close(crossing.position, [0.0, 0.0, 0.0], 1e-15);
}

// ── self-collision ───────────────────────────────────────────────────────────

fn self_collision_config(adjacency_skip: bool) -> DeformCollisionConfig {
    DeformCollisionConfig {
        thickness: THICKNESS,
        edge_edge: true,
        self_collision: true,
        adjacency_skip,
    }
}

#[test]
fn the_corners_of_a_face_are_never_self_contacts() {
    // The two triangles of a quad share the vertices 0 and 2, which lie on both
    // faces by construction. They are not collisions, with or without the
    // adjacency skip.
    let quad = make_quad_mesh([0.0; 3], 1.0);
    let bvh = DeformBvh::build(&quad);
    for adjacency_skip in [false, true] {
        let config = self_collision_config(adjacency_skip);
        let contacts = detect_self_collision(&quad, &bvh, &config);
        assert!(
            contacts.is_empty(),
            "adjacency_skip = {adjacency_skip}: {contacts:?}"
        );
    }
    let mut hash = SpatialHash::new(0.5);
    hash.build_from_mesh(&quad);
    let hashed = hash.detect_self_collision_hashed(&quad, &self_collision_config(false));
    assert!(hashed.is_empty(), "{hashed:?}");
}

#[test]
fn a_vertex_of_another_face_exactly_on_a_face_is_a_self_contact() {
    // Triangle 1 hangs a vertex exactly into the interior of triangle 0, to which
    // it is not connected.
    let positions = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.5, 0.0, 0.5],
        [0.5, 3.0, 0.5],
        [0.8, 3.0, 0.5],
    ];
    let mesh = DeformableMesh::new(
        positions,
        vec![TriFace::new(0, 1, 2), TriFace::new(3, 4, 5)],
        vec![1.0; 6],
    );
    let bvh = DeformBvh::build(&mesh);
    let contacts = detect_self_collision(&mesh, &bvh, &self_collision_config(true));
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    let contact = &contacts[0];
    assert_eq!(contact.contact_type, DeformContactType::VertexFace);
    assert_eq!((contact.feature_a, contact.feature_b), (3, 0));
    assert_eq!(contact.depth, THICKNESS);
    assert_unit(contact.normal);
    // The winding of triangle 0: (2, 0, 0) x (0, 0, 2) = (0, -4, 0).
    assert_close(contact.normal, [0.0, -1.0, 0.0], 1e-15);
}
