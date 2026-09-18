// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the contact conventions, the contact response and the
//! swept continuous collision detection of deformable meshes.

use super::functions::{
    closest_points_edge_edge, detect_deformable_box, detect_deformable_plane,
    detect_deformable_sphere, detect_edge_edge_contacts, detect_self_collision,
    detect_vertex_face_contacts, make_quad_mesh,
};
use super::*;

type Vec3 = [f64; 3];

const THICKNESS: f64 = 0.02;

fn close(a: Vec3, b: Vec3, tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

fn config(thickness: f64) -> DeformCollisionConfig {
    DeformCollisionConfig {
        thickness,
        edge_edge: true,
        self_collision: true,
        adjacency_skip: false,
    }
}

fn triangle(a: Vec3, b: Vec3, c: Vec3) -> DeformableMesh {
    DeformableMesh::new(vec![a, b, c], vec![TriFace::new(0, 1, 2)], vec![1.0; 3])
}

/// `mesh` moved by `offset`.
fn translated(mesh: &DeformableMesh, offset: Vec3) -> DeformableMesh {
    let mut moved = mesh.clone();
    for p in &mut moved.positions {
        for k in 0..3 {
            p[k] += offset[k];
        }
    }
    moved
}

fn penalty_params() -> PenaltyParams {
    PenaltyParams {
        stiffness: 1000.0,
        damping: 0.0,
        friction: 0.0,
    }
}

fn full_step() -> ConstraintParams {
    ConstraintParams {
        iterations: 1,
        relaxation: 1.0,
        friction: 0.0,
    }
}

/// Contacts of mesh A and mesh B with the detectors' own conventions.
fn detect_pair(mesh_a: &DeformableMesh, mesh_b: &DeformableMesh) -> Vec<DeformContact> {
    let bvh_a = DeformBvh::build(mesh_a);
    let bvh_b = DeformBvh::build(mesh_b);
    detect_vertex_face_contacts(mesh_a, mesh_b, &bvh_a, &bvh_b, &config(THICKNESS))
}

// ---------------------------------------------------------------------------
// F0097: contacts of the vertices of mesh B
// ---------------------------------------------------------------------------

/// A unit quad in the plane y = 0 and a triangle whose first vertex hangs 0.005
/// above face 0 of the quad (inside the contact thickness) while the other two
/// are far away. Returns `(quad, triangle)`.
fn quad_and_hovering_triangle() -> (DeformableMesh, DeformableMesh) {
    let quad = make_quad_mesh([0.0; 3], 1.0);
    let hovering = triangle([0.7, 0.005, 0.3], [0.7, 1.0, 0.35], [0.75, 1.0, 0.3]);
    (quad, hovering)
}

#[test]
fn b_vertex_touching_an_a_face_is_a_face_vertex_contact() {
    let (mesh_a, mesh_b) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    let c = &contacts[0];
    assert_eq!(c.contact_type, DeformContactType::FaceVertex);
    assert_eq!(c.feature_a, 0, "feature_a is the face of mesh A");
    assert_eq!(c.feature_b, 0, "feature_b is the vertex of mesh B");
    assert!((c.depth - 0.015).abs() < 1e-12, "depth {}", c.depth);
    // Normals point from B to A: here B hovers above A.
    assert!(close(c.normal, [0.0, -1.0, 0.0], 1e-12), "{:?}", c.normal);
    assert!(
        close(c.position, [0.7, 0.0, 0.3], 1e-12),
        "{:?}",
        c.position
    );
}

#[test]
fn a_vertex_touching_a_b_face_stays_a_vertex_face_contact() {
    let (mesh_b, mesh_a) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    let c = &contacts[0];
    assert_eq!(c.contact_type, DeformContactType::VertexFace);
    assert_eq!(c.feature_a, 0, "feature_a is the vertex of mesh A");
    assert_eq!(c.feature_b, 0, "feature_b is the face of mesh B");
    // A hovers above B, so the normal from B to A points up.
    assert!(close(c.normal, [0.0, 1.0, 0.0], 1e-12), "{:?}", c.normal);
}

#[test]
fn penalty_force_on_a_b_vertex_leaves_mesh_a_unchanged() {
    let (mut mesh_a, mut mesh_b) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);
    let params = penalty_params();

    apply_penalty_forces(&mut mesh_a, &contacts, &params, 0.01);
    assert!(
        mesh_a.velocities.iter().all(|v| *v == [0.0; 3]),
        "the face-vertex contact must not move any vertex of mesh A: {:?}",
        mesh_a.velocities
    );

    // The A-side call must not interpret the B vertex index on mesh B either.
    apply_penalty_forces(&mut mesh_b, &contacts, &params, 0.01);
    assert!(mesh_b.velocities.iter().all(|v| *v == [0.0; 3]));

    apply_penalty_forces_b(&mut mesh_b, &contacts, &params, 0.01);
    // stiffness * depth = 15 N, pushing the B vertex away from A (upward).
    assert!(
        close(mesh_b.velocities[0], [0.0, 0.15, 0.0], 1e-12),
        "{:?}",
        mesh_b.velocities[0]
    );
    assert_eq!(mesh_b.velocities[1], [0.0; 3]);
    assert_eq!(mesh_b.velocities[2], [0.0; 3]);
    // Mesh A is still at rest after the whole exchange.
    assert!(mesh_a.velocities.iter().all(|v| *v == [0.0; 3]));
}

#[test]
fn penalty_force_on_an_a_vertex_pushes_along_the_normal() {
    let (mut mesh_b, mut mesh_a) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);
    apply_penalty_forces(&mut mesh_a, &contacts, &penalty_params(), 0.01);
    assert!(close(mesh_a.velocities[0], [0.0, 0.15, 0.0], 1e-12));
    assert_eq!(mesh_a.velocities[1], [0.0; 3]);
    apply_penalty_forces_b(&mut mesh_b, &contacts, &penalty_params(), 0.01);
    assert!(
        mesh_b.velocities.iter().all(|v| *v == [0.0; 3]),
        "a vertex-face contact moves no vertex of mesh B"
    );
}

#[test]
fn pinned_vertices_get_no_penalty_velocity() {
    let (mesh_a, mut mesh_b) = quad_and_hovering_triangle();
    mesh_b.inv_mass[0] = 0.0;
    let contacts = detect_pair(&mesh_a, &mesh_b);
    assert_eq!(contacts.len(), 1);
    apply_penalty_forces_b(&mut mesh_b, &contacts, &penalty_params(), 0.01);
    assert_eq!(mesh_b.velocities[0], [0.0; 3]);
}

#[test]
fn constraints_for_a_b_vertex_act_on_mesh_b_only() {
    let (mesh_a, mut mesh_b) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);

    assert!(generate_constraints(&mesh_a, &contacts).is_empty());
    let constraints = generate_constraints_b(&mesh_b, &contacts);
    assert_eq!(constraints.len(), 1, "{constraints:?}");
    let c = &constraints[0];
    assert_eq!(c.vertex, 0);
    // One thickness above the surface of A (y = 0), not mirrored into it.
    assert!(
        close(c.target, [0.7, THICKNESS, 0.3], 1e-12),
        "{:?}",
        c.target
    );
    assert!(close(c.normal, [0.0, 1.0, 0.0], 1e-12));
    assert!((c.depth - 0.015).abs() < 1e-12);

    let untouched = mesh_b.positions.clone();
    solve_constraints(&mut mesh_b, &constraints, &full_step());
    assert!(close(mesh_b.positions[0], [0.7, THICKNESS, 0.3], 1e-12));
    assert_eq!(mesh_b.positions[1], untouched[1]);
    assert_eq!(mesh_b.positions[2], untouched[2]);
}

#[test]
fn constraints_for_an_a_vertex_target_one_thickness_above_the_b_face() {
    let (mesh_b, mesh_a) = quad_and_hovering_triangle();
    let contacts = detect_pair(&mesh_a, &mesh_b);
    let constraints = generate_constraints(&mesh_a, &contacts);
    assert_eq!(constraints.len(), 1);
    assert_eq!(constraints[0].vertex, 0);
    assert!(close(constraints[0].target, [0.7, THICKNESS, 0.3], 1e-12));
    assert!(generate_constraints_b(&mesh_b, &contacts).is_empty());
}

/// Triangle A in the plane z = 0 whose edge a0-a1 runs along x at y = 0, and
/// triangle B in the plane x = 0.25 whose edge b0-b1 runs along z at y = 0.005:
/// the two edges cross 0.005 apart and no vertex is near the other triangle.
fn crossing_edge_triangles() -> (DeformableMesh, DeformableMesh) {
    let mesh_a = triangle([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -5.0, 0.0]);
    let mesh_b = triangle([0.25, 0.005, -1.0], [0.25, 0.005, 1.0], [0.25, 5.0, 0.0]);
    (mesh_a, mesh_b)
}

fn detect_edges(mesh_a: &DeformableMesh, mesh_b: &DeformableMesh) -> Vec<DeformContact> {
    let bvh_a = DeformBvh::build(mesh_a);
    let bvh_b = DeformBvh::build(mesh_b);
    detect_edge_edge_contacts(mesh_a, mesh_b, &bvh_a, &bvh_b, &config(THICKNESS))
}

#[test]
fn edge_edge_contact_records_the_edge_ids_and_parameters() {
    let (mesh_a, mesh_b) = crossing_edge_triangles();
    assert!(
        detect_pair(&mesh_a, &mesh_b).is_empty(),
        "no vertex contact"
    );
    let contacts = detect_edges(&mesh_a, &mesh_b);
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    let c = &contacts[0];
    assert_eq!(c.contact_type, DeformContactType::EdgeEdge);
    assert_eq!((c.feature_a, c.feature_b), (0, 0));
    // The closest points are x = 0.25 on edge A (s = 0.625 of -1 .. 1) and
    // z = 0 on edge B (t = 0.5 of -1 .. 1).
    assert!((c.bary[0] - 0.625).abs() < 1e-12, "s = {}", c.bary[0]);
    assert!((c.bary[1] - 0.5).abs() < 1e-12, "t = {}", c.bary[1]);
    assert_eq!(c.bary[2], 0.0);
    assert!(close(c.normal, [0.0, -1.0, 0.0], 1e-12));
    assert!((c.depth - 0.015).abs() < 1e-12);
}

#[test]
fn edge_edge_penalty_force_is_shared_by_the_contact_parameter() {
    let (mut mesh_a, mut mesh_b) = crossing_edge_triangles();
    let contacts = detect_edges(&mesh_a, &mesh_b);
    let params = penalty_params();

    apply_penalty_forces(&mut mesh_a, &contacts, &params, 0.01);
    // 15 N pushes edge A down (along the normal), shared 0.375 : 0.625.
    assert!(close(mesh_a.velocities[0], [0.0, -0.05625, 0.0], 1e-12));
    assert!(close(mesh_a.velocities[1], [0.0, -0.09375, 0.0], 1e-12));
    assert_eq!(
        mesh_a.velocities[2], [0.0; 3],
        "off-edge vertex is untouched"
    );
    assert!(mesh_b.velocities.iter().all(|v| *v == [0.0; 3]));

    apply_penalty_forces_b(&mut mesh_b, &contacts, &params, 0.01);
    // Edge B is pushed up, shared 0.5 : 0.5.
    assert!(close(mesh_b.velocities[0], [0.0, 0.075, 0.0], 1e-12));
    assert!(close(mesh_b.velocities[1], [0.0, 0.075, 0.0], 1e-12));
    assert_eq!(mesh_b.velocities[2], [0.0; 3]);
    // The net force on each edge equals the contact force.
    let net_a: f64 = mesh_a.velocities.iter().map(|v| v[1]).sum();
    assert!((net_a + 15.0 * 0.01).abs() < 1e-12);
}

#[test]
fn edge_edge_constraints_separate_the_edges_by_exactly_the_depth() {
    let (mut mesh_a, mut mesh_b) = crossing_edge_triangles();
    let contacts = detect_edges(&mesh_a, &mesh_b);

    let constraints_a = generate_constraints(&mesh_a, &contacts);
    let constraints_b = generate_constraints_b(&mesh_b, &contacts);
    assert_eq!(constraints_a.len(), 2, "{constraints_a:?}");
    assert_eq!(constraints_b.len(), 2, "{constraints_b:?}");
    assert!(constraints_a.iter().all(|c| c.vertex < 2));
    assert!(constraints_b.iter().all(|c| c.vertex < 2));
    // Edge A moves down, edge B up.
    assert!(constraints_a.iter().all(|c| c.normal[1] < -0.999));
    assert!(constraints_b.iter().all(|c| c.normal[1] > 0.999));

    solve_constraints(&mut mesh_a, &constraints_a, &full_step());
    solve_constraints(&mut mesh_b, &constraints_b, &full_step());

    // The edges were 0.005 apart; they now sit one thickness apart.
    let (_, _, _, _, dist_sq) = closest_points_edge_edge(
        mesh_a.positions[0],
        mesh_a.positions[1],
        mesh_b.positions[0],
        mesh_b.positions[1],
    );
    // (The edge of A is tilted by the unequal vertex shares, which makes the
    // distance of the closest points differ from the vertical gap by ~1e-8.)
    assert!(
        (dist_sq.sqrt() - THICKNESS).abs() < 1e-6,
        "distance {}",
        dist_sq.sqrt()
    );
    // Vertices off the edge did not move.
    assert_eq!(mesh_a.positions[2], [0.0, -5.0, 0.0]);
    assert_eq!(mesh_b.positions[2], [0.25, 5.0, 0.0]);
}

#[test]
fn edge_edge_constraints_give_a_pinned_vertex_no_share() {
    let (mut mesh_a, mesh_b) = crossing_edge_triangles();
    mesh_a.inv_mass[0] = 0.0;
    let contacts = detect_edges(&mesh_a, &mesh_b);
    let constraints = generate_constraints(&mesh_a, &contacts);
    assert_eq!(constraints.len(), 1, "{constraints:?}");
    assert_eq!(constraints[0].vertex, 1);
    // The free vertex (weight 0.625) alone has to move the contact point by the
    // edge's share 0.0075.
    assert!((0.625 * constraints[0].depth - 0.0075).abs() < 1e-12);

    mesh_a.inv_mass[1] = 0.0;
    assert!(generate_constraints(&mesh_a, &contacts).is_empty());
}

#[test]
fn self_collision_edge_contacts_use_both_sides_of_one_mesh() {
    // Both triangles in one mesh: vertices 0-2 are the first, 3-5 the second.
    let (mesh_a, mesh_b) = crossing_edge_triangles();
    let mut positions = mesh_a.positions.clone();
    positions.extend(mesh_b.positions.iter().copied());
    let mut mesh = DeformableMesh::new(
        positions,
        vec![TriFace::new(0, 1, 2), TriFace::new(3, 4, 5)],
        vec![1.0; 6],
    );
    let bvh = DeformBvh::build(&mesh);
    let contacts = detect_self_collision(&mesh, &bvh, &config(THICKNESS));
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    assert_eq!(contacts[0].contact_type, DeformContactType::EdgeEdge);
    assert_eq!((contacts[0].feature_a, contacts[0].feature_b), (0, 3));

    let constraints = [
        generate_constraints(&mesh, &contacts),
        generate_constraints_b(&mesh, &contacts),
    ]
    .concat();
    let mut moved: Vec<usize> = constraints.iter().map(|c| c.vertex).collect();
    moved.sort_unstable();
    assert_eq!(moved, vec![0, 1, 3, 4]);

    apply_penalty_forces(&mut mesh, &contacts, &penalty_params(), 0.01);
    apply_penalty_forces_b(&mut mesh, &contacts, &penalty_params(), 0.01);
    assert!(mesh.velocities[0][1] < 0.0 && mesh.velocities[1][1] < 0.0);
    assert!(mesh.velocities[3][1] > 0.0 && mesh.velocities[4][1] > 0.0);
    assert_eq!(mesh.velocities[2], [0.0; 3]);
    assert_eq!(mesh.velocities[5], [0.0; 3]);
}

#[test]
fn contacts_with_out_of_range_features_are_ignored() {
    let mut mesh = make_quad_mesh([0.0; 3], 1.0);
    let bogus = [
        DeformContact::new(
            [0.0; 3],
            [0.0, 1.0, 0.0],
            0.1,
            [0.0; 3],
            DeformContactType::VertexFace,
            99,
            0,
        ),
        DeformContact::new(
            [0.0; 3],
            [0.0, 1.0, 0.0],
            0.1,
            [0.0; 3],
            DeformContactType::FaceVertex,
            0,
            99,
        ),
        DeformContact::new(
            [0.0; 3],
            [0.0, 1.0, 0.0],
            0.1,
            [0.5, 0.5, 0.0],
            DeformContactType::EdgeEdge,
            300,
            300,
        ),
        DeformContact::new(
            [0.0; 3],
            [0.0, 1.0, 0.0],
            0.1,
            [f64::NAN, f64::NAN, 0.0],
            DeformContactType::FaceRigid,
            0,
            0,
        ),
    ];
    apply_penalty_forces(&mut mesh, &bogus, &penalty_params(), 0.01);
    apply_penalty_forces_b(&mut mesh, &bogus, &penalty_params(), 0.01);
    assert!(mesh.velocities.iter().all(|v| *v == [0.0; 3]));
    assert!(generate_constraints(&mesh, &bogus).is_empty());
    assert!(generate_constraints_b(&mesh, &bogus).is_empty());
}

#[test]
fn a_nan_edge_parameter_does_not_poison_the_velocities() {
    let mut mesh = make_quad_mesh([0.0; 3], 1.0);
    let contact = DeformContact::new(
        [0.0; 3],
        [0.0, 1.0, 0.0],
        0.1,
        [f64::NAN, f64::NAN, 0.0],
        DeformContactType::EdgeEdge,
        0,
        0,
    );
    apply_penalty_forces(&mut mesh, &[contact], &penalty_params(), 0.01);
    assert!(
        mesh.velocities
            .iter()
            .all(|v| v.iter().all(|c| c.is_finite()))
    );
}

// ---------------------------------------------------------------------------
// F0099: constraint targets
// ---------------------------------------------------------------------------

#[test]
fn plane_constraint_target_is_the_plane_plus_thickness() {
    // Every vertex 0.05 below the plane, thickness 0.01.
    let mut mesh = make_quad_mesh([0.0, -0.05, 0.0], 1.0);
    let plane = RigidPlane {
        normal: [0.0, 1.0, 0.0],
        offset: 0.0,
    };
    let contacts = detect_deformable_plane(&mesh, &plane, 0.01);
    assert_eq!(contacts.len(), 4);
    let constraints = generate_constraints(&mesh, &contacts);
    assert_eq!(constraints.len(), 4);
    for c in &constraints {
        assert!(
            (c.target[1] - 0.01).abs() < 1e-12,
            "target {:?} should be the plane plus the thickness",
            c.target
        );
        // x and z do not change.
        assert_eq!(c.target[0], mesh.positions[c.vertex][0]);
        assert_eq!(c.target[2], mesh.positions[c.vertex][2]);
    }
    solve_constraints(&mut mesh, &constraints, &full_step());
    for p in &mesh.positions {
        assert!((p[1] - 0.01).abs() < 1e-12, "{p:?}");
    }
}

#[test]
fn plane_constraint_for_a_vertex_inside_the_thickness_shell() {
    // 0.004 above the plane: still within the 0.01 shell.
    let mesh = make_quad_mesh([0.0, 0.004, 0.0], 1.0);
    let plane = RigidPlane {
        normal: [0.0, 1.0, 0.0],
        offset: 0.0,
    };
    let contacts = detect_deformable_plane(&mesh, &plane, 0.01);
    let constraints = generate_constraints(&mesh, &contacts);
    assert_eq!(constraints.len(), 4);
    assert!(
        constraints
            .iter()
            .all(|c| (c.target[1] - 0.01).abs() < 1e-12)
    );
}

#[test]
fn sphere_constraint_target_is_radius_plus_thickness_from_the_centre() {
    let mesh = triangle([0.7, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]);
    let sphere = RigidSphere {
        centre: [0.0; 3],
        radius: 1.0,
    };
    let contacts = detect_deformable_sphere(&mesh, &sphere, 0.05);
    assert_eq!(contacts.len(), 1);
    let constraints = generate_constraints(&mesh, &contacts);
    assert_eq!(constraints.len(), 1);
    assert_eq!(constraints[0].vertex, 0);
    // Radius 1 + thickness 0.05 along +x (not the 1.35 of surface + depth).
    assert!(
        close(constraints[0].target, [1.05, 0.0, 0.0], 1e-12),
        "{:?}",
        constraints[0].target
    );
}

#[test]
fn box_constraint_targets_are_thickness_outside_the_face() {
    let rigid_box = RigidBox {
        min: [-1.0; 3],
        max: [1.0; 3],
    };
    // One vertex just outside the +x face (within the shell), one inside it, one
    // far away.
    let mesh = triangle([1.005, 0.0, 0.0], [0.9, 0.0, 0.0], [5.0, 5.0, 5.0]);
    let contacts = detect_deformable_box(&mesh, &rigid_box, THICKNESS);
    assert_eq!(contacts.len(), 2, "{contacts:?}");
    let constraints = generate_constraints(&mesh, &contacts);
    assert_eq!(constraints.len(), 2);
    for c in &constraints {
        assert!(
            close(c.target, [1.0 + THICKNESS, 0.0, 0.0], 1e-12),
            "vertex {} targets {:?}",
            c.vertex,
            c.target
        );
    }
}

#[test]
fn constraints_of_rigid_contacts_resolve_the_whole_penetration() {
    // Detect, generate and solve: the vertices end up exactly a thickness from
    // the plane, with the normal and the depth of the constraint consistent.
    let mesh = make_quad_mesh([0.0, -0.3, 0.0], 1.0);
    let plane = RigidPlane {
        normal: [0.0, 1.0, 0.0],
        offset: 0.2,
    };
    let contacts = detect_deformable_plane(&mesh, &plane, 0.05);
    for c in generate_constraints(&mesh, &contacts) {
        assert!((v3_len_for_test(c.normal) - 1.0).abs() < 1e-12);
        let moved = (c.target[1] - mesh.positions[c.vertex][1]) * c.normal[1];
        assert!((moved - c.depth).abs() < 1e-12);
        assert!((c.target[1] - (0.2 + 0.05)).abs() < 1e-12);
    }
}

fn v3_len_for_test(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---------------------------------------------------------------------------
// F0098: swept broad phase and continuous collision detection
// ---------------------------------------------------------------------------

const CCD_THICKNESS: f64 = 0.01;

/// A large static quad B in the plane y = 0 (x, z in -1 .. 1).
fn big_sheet() -> DeformableMesh {
    make_quad_mesh([-1.0, 0.0, -1.0], 2.0)
}

/// A small quad 0.2 wide with its corner at (x, y, z).
fn small_sheet(x: f64, y: f64, z: f64) -> DeformableMesh {
    make_quad_mesh([x, y, z], 0.2)
}

#[test]
fn swept_bvh_encloses_start_and_end_of_the_step() {
    let prev = make_quad_mesh([0.0, 1.0, 0.0], 1.0);
    let curr = make_quad_mesh([0.0, -1.0, 0.0], 1.0);
    let bvh = DeformBvh::build_swept(&prev, &curr, 0.1);
    let root = &bvh.nodes[0].aabb;
    assert!(close(root.min, [-0.1, -1.1, -0.1], 1e-12), "{:?}", root.min);
    assert!(close(root.max, [1.1, 1.1, 1.1], 1e-12), "{:?}", root.max);
    // Every face box contains the face at both ends of the step.
    for node in bvh.nodes.iter().filter(|n| !n.faces.is_empty()) {
        for &face in &node.faces {
            for mesh in [&prev, &curr] {
                let f = &mesh.faces[face];
                for v in [f.v0, f.v1, f.v2] {
                    let p = mesh.positions[v];
                    assert!((0..3).all(|k| node.aabb.min[k] <= p[k] && p[k] <= node.aabb.max[k]));
                }
            }
        }
    }
}

#[test]
fn single_configuration_bvh_misses_a_sheet_that_passes_through() {
    let target = big_sheet();
    let prev = small_sheet(-0.1, 1.0, -0.1);
    let curr = small_sheet(-0.1, -1.0, -0.1);
    let target_bvh = DeformBvh::build(&target);
    // At the end of the step the small sheet is on the far side of the target and
    // far from it, so the boxes of a single configuration do not overlap ...
    assert!(
        DeformBvh::build(&curr)
            .query_overlap_pairs(&target_bvh)
            .is_empty()
    );
    // ... but the swept boxes do.
    let swept = DeformBvh::build_swept(&prev, &curr, CCD_THICKNESS);
    assert!(!swept.query_overlap_pairs(&target_bvh).is_empty());
}

#[test]
fn refit_swept_follows_the_new_motion() {
    let prev = make_quad_mesh([0.0, 1.0, 0.0], 1.0);
    let curr = make_quad_mesh([0.0, -1.0, 0.0], 1.0);
    let mut bvh = DeformBvh::build_swept(&prev, &curr, 0.1);
    let longer = make_quad_mesh([0.0, -3.0, 0.0], 1.0);
    bvh.refit_swept(&prev, &longer, 0.1);
    assert!((bvh.nodes[0].aabb.min[1] - (-3.1)).abs() < 1e-12);
    assert!((bvh.nodes[0].aabb.max[1] - 1.1).abs() < 1e-12);
    // A previous mesh that lacks vertices counts those vertices as unmoved.
    let partial_prev = triangle([0.0, 5.0, 0.0], [1.0, 5.0, 0.0], [1.0, 5.0, 1.0]);
    let swept = DeformBvh::build_swept(&partial_prev, &curr, 0.0);
    assert!((swept.nodes[0].aabb.max[1] - 5.0).abs() < 1e-12);
    assert!((swept.nodes[0].aabb.min[1] - (-1.0)).abs() < 1e-12);
}

#[test]
fn a_sheet_moving_fully_through_another_reports_a_vertex_face_hit() {
    // Mesh A is the small sheet; it crosses face 0 of the large target sheet
    // (z < x) away from the diagonal edge, so only its vertices meet the face.
    let target = big_sheet();
    let prev = small_sheet(0.3, 1.0, -0.5);
    let curr = small_sheet(0.3, -1.0, -0.5);
    let hit = ccd_deformable_deformable(&prev, &curr, &target, &target, CCD_THICKNESS)
        .expect("the sheet crosses the other one within the step");
    assert_eq!(hit.contact_type, DeformContactType::VertexFace);
    assert!((hit.toi - 0.5).abs() < 1e-6, "toi {}", hit.toi);
    assert!(hit.feature_a < 4, "vertex of the moving sheet (mesh A)");
    assert_eq!(hit.feature_b, 0, "face of the target sheet (mesh B)");
    // Mesh A comes from above mesh B, so the normal from B to A points up.
    assert!(close(hit.normal, [0.0, 1.0, 0.0], 1e-9), "{:?}", hit.normal);
    assert!(hit.position[1].abs() < 1e-6, "{:?}", hit.position);
}

#[test]
fn a_small_b_sheet_passing_through_a_large_a_sheet_is_a_face_vertex_hit() {
    // Mesh A is the large static sheet; mesh B is the small sheet that moves
    // through the interior of face 0 of A. None of A's vertices or edges is near
    // it, so only the test of B's vertices against A's faces can see the hit.
    let target = big_sheet();
    let prev = small_sheet(0.3, 1.0, -0.5);
    let curr = small_sheet(0.3, -1.0, -0.5);
    let hit = ccd_deformable_deformable(&target, &target, &prev, &curr, CCD_THICKNESS)
        .expect("a vertex of mesh B passes through a face of mesh A");
    assert_eq!(hit.contact_type, DeformContactType::FaceVertex);
    assert!((hit.toi - 0.5).abs() < 1e-6, "toi {}", hit.toi);
    assert_eq!(hit.feature_a, 0, "face of mesh A");
    assert!(hit.feature_b < 4, "vertex of mesh B");
    // Mesh B comes from above mesh A, so the normal from B to A points down.
    assert!(
        close(hit.normal, [0.0, -1.0, 0.0], 1e-9),
        "{:?}",
        hit.normal
    );
}

/// Two triangles whose edges cross while moving through each other: no vertex
/// of either comes near the other triangle, only edge b0-b1 crosses edge a0-a1.
fn edge_crossing_pair() -> (DeformableMesh, DeformableMesh, DeformableMesh) {
    let mesh_a = triangle([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -5.0, 0.0]);
    let prev_b = triangle([0.25, 1.0, -1.0], [0.25, 1.0, 1.0], [0.25, 6.0, 0.0]);
    let curr_b = translated(&prev_b, [0.0, -2.0, 0.0]);
    (mesh_a, prev_b, curr_b)
}

#[test]
fn crossing_edges_are_detected_when_no_vertex_meets_a_face() {
    let (mesh_a, prev_b, curr_b) = edge_crossing_pair();
    let hit = ccd_deformable_deformable(&mesh_a, &mesh_a, &prev_b, &curr_b, CCD_THICKNESS)
        .expect("the edges cross within the step");
    assert_eq!(hit.contact_type, DeformContactType::EdgeEdge);
    assert_eq!((hit.feature_a, hit.feature_b), (0, 0));
    assert!((hit.toi - 0.5).abs() < 1e-6, "toi {}", hit.toi);
    // B comes from above A: the normal from B to A points down.
    assert!(
        close(hit.normal, [0.0, -1.0, 0.0], 1e-9),
        "{:?}",
        hit.normal
    );
    assert!(
        close(hit.position, [0.25, 0.0, 0.0], 1e-6),
        "{:?}",
        hit.position
    );
}

#[test]
fn sheets_that_stay_apart_do_not_hit() {
    let target = big_sheet();
    let prev = small_sheet(-0.1, 1.0, -0.1);
    let curr = small_sheet(0.4, 0.5, 0.2);
    assert!(ccd_deformable_deformable(&prev, &curr, &target, &target, CCD_THICKNESS).is_none());
    // Moving past the edge of the target instead of through it.
    let beside = small_sheet(3.0, -1.0, -0.1);
    assert!(ccd_deformable_deformable(&prev, &beside, &target, &target, CCD_THICKNESS).is_none());
}

#[test]
fn prebuilt_swept_bvhs_give_the_same_result() {
    let target = big_sheet();
    let prev = small_sheet(0.3, 1.0, -0.5);
    let curr = small_sheet(0.3, -1.0, -0.5);
    let bvh_a = DeformBvh::build_swept(&prev, &curr, CCD_THICKNESS);
    let bvh_b = DeformBvh::build_swept(&target, &target, CCD_THICKNESS);
    let direct =
        ccd_deformable_deformable(&prev, &curr, &target, &target, CCD_THICKNESS).expect("hit");
    let reused = ccd_deformable_deformable_swept(
        &prev,
        &curr,
        &target,
        &target,
        &bvh_a,
        &bvh_b,
        CCD_THICKNESS,
    )
    .expect("hit");
    assert_eq!(direct.toi, reused.toi);
    assert_eq!(direct.contact_type, reused.contact_type);
    assert_eq!(
        (direct.feature_a, direct.feature_b),
        (reused.feature_a, reused.feature_b)
    );
    assert_eq!(direct.normal, reused.normal);
}

/// Deterministic xorshift generator for the randomized comparison.
struct TestRng(u64);

impl TestRng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn point(&mut self, extent: f64) -> Vec3 {
        [
            self.range(-extent, extent),
            self.range(-extent, extent),
            self.range(-extent, extent),
        ]
    }

    fn index(&mut self, bound: usize) -> usize {
        ((self.unit() * bound as f64) as usize).min(bound - 1)
    }
}

/// A random mesh of three to five triangles over six vertices.
fn random_mesh(rng: &mut TestRng, centre: Vec3) -> DeformableMesh {
    let positions: Vec<Vec3> = (0..6)
        .map(|_| {
            let p = rng.point(0.6);
            [centre[0] + p[0], centre[1] + p[1], centre[2] + p[2]]
        })
        .collect();
    let mut faces = Vec::new();
    let count = 3 + rng.index(3);
    while faces.len() < count {
        let (a, b, c) = (rng.index(6), rng.index(6), rng.index(6));
        if a != b && b != c && a != c {
            faces.push(TriFace::new(a, b, c));
        }
    }
    DeformableMesh::new(positions, faces, vec![1.0; 6])
}

/// The mesh with every vertex displaced by a random vector.
fn random_step(rng: &mut TestRng, mesh: &DeformableMesh, extent: f64) -> DeformableMesh {
    let mut next = mesh.clone();
    for p in &mut next.positions {
        let d = rng.point(extent);
        for k in 0..3 {
            p[k] += d[k];
        }
    }
    next
}

/// Earliest time of impact over every face pair, without a broad phase.
fn brute_force_toi(
    a_prev: &DeformableMesh,
    a_curr: &DeformableMesh,
    b_prev: &DeformableMesh,
    b_curr: &DeformableMesh,
    thickness: f64,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut take = |toi: Option<f64>| {
        if let Some(t) = toi {
            best = Some(best.map_or(t, |b| b.min(t)));
        }
    };
    let vertex_face =
        |vp: Vec3, vc: Vec3, prev: &DeformableMesh, curr: &DeformableMesh, f: &TriFace| {
            let ids = [f.v0, f.v1, f.v2];
            ccd_vertex_face_analytic(
                vp,
                vc,
                prev.positions[ids[0]],
                curr.positions[ids[0]],
                prev.positions[ids[1]],
                curr.positions[ids[1]],
                prev.positions[ids[2]],
                curr.positions[ids[2]],
                thickness,
            )
        };
    for fa in &a_curr.faces {
        for fb in &b_curr.faces {
            for v in [fa.v0, fa.v1, fa.v2] {
                take(vertex_face(
                    a_prev.positions[v],
                    a_curr.positions[v],
                    b_prev,
                    b_curr,
                    fb,
                ));
            }
            for v in [fb.v0, fb.v1, fb.v2] {
                take(vertex_face(
                    b_prev.positions[v],
                    b_curr.positions[v],
                    a_prev,
                    a_curr,
                    fa,
                ));
            }
            for (p0, p1) in fa.edges() {
                for (q0, q1) in fb.edges() {
                    take(ccd_edge_edge_analytic(
                        a_prev.positions[p0],
                        a_curr.positions[p0],
                        a_prev.positions[p1],
                        a_curr.positions[p1],
                        b_prev.positions[q0],
                        b_curr.positions[q0],
                        b_prev.positions[q1],
                        b_curr.positions[q1],
                        thickness,
                    ));
                }
            }
        }
    }
    best
}

#[test]
fn swept_broad_phase_never_prunes_a_hit() {
    let mut rng = TestRng(0x2545_f491_4f6c_dd1d);
    let (mut hits, mut misses) = (0, 0);
    for case in 0..80 {
        let a_prev = random_mesh(&mut rng, [0.0; 3]);
        let a_curr = random_step(&mut rng, &a_prev, 0.8);
        let offset = [
            rng.range(-1.6, 1.6),
            rng.range(-1.6, 1.6),
            rng.range(-0.8, 0.8),
        ];
        let b_prev = random_mesh(&mut rng, offset);
        let b_curr = random_step(&mut rng, &b_prev, 0.8);
        let thickness = 0.03;
        let expected = brute_force_toi(&a_prev, &a_curr, &b_prev, &b_curr, thickness);
        let actual = ccd_deformable_deformable(&a_prev, &a_curr, &b_prev, &b_curr, thickness);
        match (expected, actual) {
            (None, None) => misses += 1,
            (Some(t), Some(hit)) => {
                hits += 1;
                assert!(
                    (hit.toi - t).abs() < 1e-12,
                    "case {case}: toi {} != brute force {t}",
                    hit.toi
                );
                assert!((0.0..=1.0).contains(&hit.toi));
                assert!(
                    (v3_len_for_test(hit.normal) - 1.0).abs() < 1e-9,
                    "{:?}",
                    hit.normal
                );
            }
            (expected, actual) => panic!(
                "case {case}: broad phase disagrees with brute force: {expected:?} vs {:?}",
                actual.map(|h| h.toi)
            ),
        }
    }
    assert!(hits >= 10 && misses >= 10, "hits {hits}, misses {misses}");
}

#[test]
fn zero_distance_is_never_reported_as_a_nonunit_normal() {
    // A vertex that starts exactly on the other sheet: the shell test fires at
    // t = 0 and the normal still comes out as a unit vector on the right side.
    let target = big_sheet();
    let prev = small_sheet(0.3, 0.0, -0.5);
    let curr = small_sheet(0.3, 1.0, -0.5);
    let hit = ccd_deformable_deformable(&target, &target, &prev, &curr, CCD_THICKNESS)
        .expect("already touching at the start");
    assert_eq!(hit.toi, 0.0);
    assert!((v3_len_for_test(hit.normal) - 1.0).abs() < 1e-9);
    assert!(hit.normal[1].abs() > 0.999);
}
