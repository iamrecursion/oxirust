// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the batch rigid body simulation: orientation helpers,
//! impulses, broadphase, contact manifolds and the sequential-impulse solver.

use super::*;

const EPS: f32 = 1e-5;

fn approx_eq3(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < EPS && (a[1] - b[1]).abs() < EPS && (a[2] - b[2]).abs() < EPS
}

fn approx_eq4(a: [f32; 4], b: [f32; 4]) -> bool {
    (a[0] - b[0]).abs() < EPS
        && (a[1] - b[1]).abs() < EPS
        && (a[2] - b[2]).abs() < EPS
        && (a[3] - b[3]).abs() < EPS
}

fn quat_norm(q: [f32; 4]) -> f32 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

// ── quat_rotate ────────────────────────────────────────────────────────

#[test]
fn test_quat_rotate_identity() {
    let q = [0.0, 0.0, 0.0, 1.0_f32];
    let v = [1.0, 2.0, 3.0_f32];
    let r = quat_rotate(q, v);
    assert!(approx_eq3(r, v), "identity quat should not rotate: {r:?}");
}

#[test]
fn test_quat_rotate_180_about_z() {
    // 180° about Z: q = [0, 0, 1, 0]
    let q = [0.0, 0.0, 1.0_f32, 0.0];
    let v = [1.0, 0.0, 0.0_f32];
    let r = quat_rotate(q, v);
    assert!(approx_eq3(r, [-1.0, 0.0, 0.0]), "180 Z rotate: {r:?}");
}

#[test]
fn test_quat_rotate_90_about_y() {
    // 90° about Y: q = [0, sin45, 0, cos45]
    let half = std::f32::consts::FRAC_PI_4;
    let q = [0.0, half.sin(), 0.0, half.cos()];
    let v = [1.0, 0.0, 0.0_f32];
    let r = quat_rotate(q, v);
    // Should become ~[0, 0, -1]
    assert!((r[0]).abs() < EPS, "x should be ~0: {}", r[0]);
    assert!((r[1]).abs() < EPS, "y should be ~0: {}", r[1]);
    assert!((r[2] + 1.0).abs() < EPS, "z should be ~-1: {}", r[2]);
}

#[test]
fn test_quat_rotate_preserves_length() {
    let half = std::f32::consts::FRAC_PI_6;
    let q = quat_normalize([half.sin(), 0.0, 0.0, half.cos()]);
    let v = [3.0, 4.0, 0.0_f32];
    let r = quat_rotate(q, v);
    let len_v = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let len_r = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    assert!((len_v - len_r).abs() < EPS, "rotation must preserve length");
}

// ── quat_mul ───────────────────────────────────────────────────────────

#[test]
fn test_quat_mul_identity() {
    let id = [0.0, 0.0, 0.0, 1.0_f32];
    let q = [0.1, 0.2, 0.3_f32, 0.9];
    let q = quat_normalize(q);
    assert!(approx_eq4(quat_mul(id, q), q));
    assert!(approx_eq4(quat_mul(q, id), q));
}

#[test]
fn test_quat_mul_unit_norm() {
    let a = quat_normalize([1.0, 0.0, 0.0_f32, 1.0]);
    let b = quat_normalize([0.0, 1.0, 0.0_f32, 1.0]);
    let c = quat_mul(a, b);
    assert!(
        (quat_norm(c) - 1.0).abs() < EPS,
        "product must be unit: {}",
        quat_norm(c)
    );
}

#[test]
fn test_quat_mul_double_rotation() {
    // Two 90° rotations about Z => 180° about Z
    let half = std::f32::consts::FRAC_PI_4;
    let q90 = [0.0, 0.0, half.sin(), half.cos()];
    let q180 = quat_mul(q90, q90);
    let v = [1.0, 0.0, 0.0_f32];
    let r = quat_rotate(q180, v);
    assert!((r[0] + 1.0).abs() < EPS, "should be [-1,0,0]: {r:?}");
}

// ── integrate_orientation ──────────────────────────────────────────────

#[test]
fn test_integrate_orientation_no_rotation() {
    let q = [0.0, 0.0, 0.0, 1.0_f32];
    let q2 = integrate_orientation(q, [0.0; 3], 0.01);
    assert!((quat_norm(q2) - 1.0).abs() < EPS);
    assert!(approx_eq4(q2, q));
}

#[test]
fn test_integrate_orientation_stays_unit() {
    let q = [0.0, 0.0, 0.0, 1.0_f32];
    let omega = [0.0, 0.0, 1.0_f32]; // 1 rad/s about Z
    let mut q_cur = q;
    for _ in 0..100 {
        q_cur = integrate_orientation(q_cur, omega, 0.01);
    }
    assert!((quat_norm(q_cur) - 1.0).abs() < 1e-4);
}

#[test]
fn test_integrate_orientation_direction() {
    // Small rotation about X: after many steps orientation should shift
    let q = [0.0, 0.0, 0.0, 1.0_f32];
    let omega = [1.0, 0.0, 0.0_f32];
    let q2 = integrate_orientation(q, omega, 0.1);
    // x component should now be non-zero
    assert!(
        q2[0].abs() > 1e-4,
        "qx should be > 0 after rotation: {}",
        q2[0]
    );
}

// ── quat_normalize ─────────────────────────────────────────────────────

#[test]
fn test_quat_normalize_unit() {
    let q = [1.0_f32, 0.0, 0.0, 0.0];
    assert!(approx_eq4(quat_normalize(q), q));
}

#[test]
fn test_quat_normalize_zero_returns_identity() {
    let q = quat_normalize([0.0; 4]);
    assert!(approx_eq4(q, [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn test_quat_normalize_scales() {
    let q = [2.0_f32, 0.0, 0.0, 0.0];
    let n = quat_normalize(q);
    assert!((n[0] - 1.0).abs() < EPS);
}

// ── GpuRigidBody ──────────────────────────────────────────────────────

#[test]
fn test_rigid_body_new_defaults() {
    let b = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    assert_eq!(b.position, [0.0; 3]);
    assert_eq!(b.velocity, [0.0; 3]);
    assert_eq!(b.orientation, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(b.angular_velocity, [0.0; 3]);
    assert!((b.mass - 1.0).abs() < EPS);
}

#[test]
fn test_rigid_body_inv_inertia_diagonal() {
    let b = GpuRigidBody::new(1.0, 2.0, 4.0, 8.0);
    assert!((b.inv_inertia[0] - 0.5).abs() < EPS, "ixx");
    assert!((b.inv_inertia[4] - 0.25).abs() < EPS, "iyy");
    assert!((b.inv_inertia[8] - 0.125).abs() < EPS, "izz");
}

// ── GpuRigidBodyBatch::integrate_all ──────────────────────────────────

#[test]
fn test_batch_gravity_integration() {
    let mut batch = GpuRigidBodyBatch::new();
    batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    batch.integrate_all(1.0, [0.0, -9.81, 0.0]);
    assert!((batch.bodies[0].velocity[1] + 9.81).abs() < EPS);
    assert!((batch.bodies[0].position[1] + 9.81).abs() < EPS);
}

#[test]
fn test_batch_no_gravity_no_motion() {
    let mut batch = GpuRigidBodyBatch::new();
    batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    batch.integrate_all(1.0, [0.0; 3]);
    assert_eq!(batch.bodies[0].position, [0.0; 3]);
    assert_eq!(batch.bodies[0].velocity, [0.0; 3]);
}

#[test]
fn test_batch_multiple_bodies() {
    let mut batch = GpuRigidBodyBatch::new();
    for _ in 0..5 {
        batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    }
    batch.integrate_all(0.5, [0.0, -9.81, 0.0]);
    for b in &batch.bodies {
        assert!((b.velocity[1] + 9.81 * 0.5).abs() < EPS);
    }
}

#[test]
fn test_batch_add_returns_index() {
    let mut batch = GpuRigidBodyBatch::new();
    let i0 = batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    let i1 = batch.add(GpuRigidBody::new(2.0, 1.0, 1.0, 1.0));
    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
}

// ── GpuRigidBodyBatch::apply_impulse ──────────────────────────────────

#[test]
fn test_apply_impulse_linear() {
    let mut batch = GpuRigidBodyBatch::new();
    batch.add(GpuRigidBody::new(2.0, 1.0, 1.0, 1.0));
    // impulse [2,0,0] on mass 2 => dv = [1,0,0]
    batch.apply_impulse(0, [2.0, 0.0, 0.0], [0.0; 3]);
    assert!((batch.bodies[0].velocity[0] - 1.0).abs() < EPS);
}

#[test]
fn test_apply_impulse_angular() {
    let mut batch = GpuRigidBodyBatch::new();
    let mut b = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b.position = [0.0; 3];
    batch.add(b);
    // Apply Z-axis impulse at offset on X axis → angular velocity change
    batch.apply_impulse(0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(batch.bodies[0].angular_velocity[2].abs() > 1e-5);
}

#[test]
fn test_apply_impulse_zero_mass() {
    let mut batch = GpuRigidBodyBatch::new();
    batch.add(GpuRigidBody::new(0.0, 1.0, 1.0, 1.0));
    batch.apply_impulse(0, [100.0, 0.0, 0.0], [0.0; 3]);
    assert_eq!(batch.bodies[0].velocity, [0.0; 3]);
}

// ── GpuBroadphase ─────────────────────────────────────────────────────

#[test]
fn test_broadphase_no_bodies() {
    let bp = GpuBroadphase::new();
    assert!(bp.compute_pairs_sap().is_empty());
}

#[test]
fn test_broadphase_two_overlapping() {
    let mut bp = GpuBroadphase::new();
    let mut b1 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b1.position = [0.0; 3];
    let mut b2 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b2.position = [0.5, 0.0, 0.0];
    bp.add_body(b1, 1.0);
    bp.add_body(b2, 1.0);
    let pairs = bp.compute_pairs_sap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_broadphase_two_separated() {
    let mut bp = GpuBroadphase::new();
    let mut b1 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b1.position = [0.0; 3];
    let mut b2 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b2.position = [100.0, 0.0, 0.0];
    bp.add_body(b1, 0.5);
    bp.add_body(b2, 0.5);
    let pairs = bp.compute_pairs_sap();
    assert!(pairs.is_empty());
}

#[test]
fn test_broadphase_three_bodies_two_pairs() {
    let mut bp = GpuBroadphase::new();
    for x in [0.0_f32, 1.0, 2.0] {
        let mut b = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
        b.position = [x, 0.0, 0.0];
        bp.add_body(b, 0.8);
    }
    let pairs = bp.compute_pairs_sap();
    // 0-1 and 1-2 overlap, 0-2 may or may not
    assert!(pairs.len() >= 2, "expected >= 2 pairs, got {}", pairs.len());
}

#[test]
fn test_broadphase_pair_indices_valid() {
    let mut bp = GpuBroadphase::new();
    let mut b1 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b1.position = [0.0; 3];
    let mut b2 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b2.position = [0.3, 0.0, 0.0];
    bp.add_body(b1, 0.5);
    bp.add_body(b2, 0.5);
    let pairs = bp.compute_pairs_sap();
    assert_eq!(pairs.len(), 1);
    let p = &pairs[0];
    assert!(p.body_a < 2 && p.body_b < 2 && p.body_a != p.body_b);
}

// ── ContactManifoldGpu ────────────────────────────────────────────────

#[test]
fn test_manifold_empty() {
    let m = ContactManifoldGpu::new(0, 1);
    assert_eq!(m.contact_count(), 0);
}

#[test]
fn test_manifold_add_contact() {
    let mut m = ContactManifoldGpu::new(0, 1);
    m.add_contact([0.0; 3], [0.0, 1.0, 0.0], 0.01);
    assert_eq!(m.contact_count(), 1);
    assert!((m.penetrations[0] - 0.01).abs() < EPS);
}

#[test]
fn test_manifold_multiple_contacts() {
    let mut m = ContactManifoldGpu::new(0, 1);
    for i in 0..4 {
        m.add_contact([i as f32, 0.0, 0.0], [0.0, 1.0, 0.0], 0.01 * i as f32);
    }
    assert_eq!(m.contact_count(), 4);
}

// ── GpuConstraintSolver ───────────────────────────────────────────────

#[test]
fn test_solver_no_manifolds() {
    let solver = GpuConstraintSolver::new();
    let mut bodies = vec![GpuRigidBody::new(1.0, 1.0, 1.0, 1.0)];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    assert_eq!(bodies[0].velocity, [0.0; 3]);
}

#[test]
fn test_solver_separates_colliding_bodies() {
    // Body 0 (left, moving right) hits body 1 (right, moving left).  The
    // contact normal points from B to A, i.e. along -x.
    let mut b1 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b1.velocity = [1.0, 0.0, 0.0];
    let mut b2 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b2.velocity = [-1.0, 0.0, 0.0];
    b2.position = [0.5, 0.0, 0.0];

    let mut solver = GpuConstraintSolver::new();
    let pair = BroadphasePairGpu {
        body_a: 0,
        body_b: 1,
        aabb_a_center: [0.0; 3],
        aabb_a_half: [0.5; 3],
        aabb_b_center: [0.5; 3],
        aabb_b_half: [0.5; 3],
    };
    let mut manifold = ContactManifoldGpu::new(0, 1);
    manifold.add_contact([0.25, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.01);
    solver.add_manifold(pair, manifold);

    let mut bodies = vec![b1, b2];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);

    // Approach speed 2, default restitution 0.3: the bodies leave each
    // other at 0.6 and momentum is conserved (equal masses).
    let separation = bodies[1].velocity[0] - bodies[0].velocity[0];
    assert!(
        (separation - 0.6).abs() < 1e-5,
        "separation speed {separation}"
    );
    assert!((bodies[0].velocity[0] + bodies[1].velocity[0]).abs() < 1e-5);
}

#[test]
fn test_solver_static_body_kinematic() {
    // body_b has zero mass => should not move
    let mut b1 = GpuRigidBody::new(1.0, 1.0, 1.0, 1.0);
    b1.velocity = [0.0, -1.0, 0.0];
    let b2 = GpuRigidBody::new(0.0, 1.0, 1.0, 1.0); // static

    let mut solver = GpuConstraintSolver::new();
    let pair = BroadphasePairGpu {
        body_a: 0,
        body_b: 1,
        aabb_a_center: [0.0; 3],
        aabb_a_half: [0.5; 3],
        aabb_b_center: [0.0, -0.9, 0.0],
        aabb_b_half: [0.5; 3],
    };
    let mut manifold = ContactManifoldGpu::new(0, 1);
    manifold.add_contact([0.0, -0.5, 0.0], [0.0, 1.0, 0.0], 0.1);
    solver.add_manifold(pair, manifold);

    let mut bodies = vec![b1, b2];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);

    assert_eq!(bodies[1].velocity, [0.0; 3], "static body must not move");
    assert!(
        bodies[0].velocity[1] >= -EPS,
        "dynamic body should bounce up"
    );
}

// ── World-space inverse inertia (apply_impulse) ───────────────────────

/// Quaternion `[x, y, z, w]` for a rotation of `angle` about the unit `axis`.
fn axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let (s, c) = (0.5 * angle).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

fn conjugate(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn assert_close3(actual: [f32; 3], expected: [f32; 3], tol: f32, what: &str) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() <= tol,
            "{what}: {actual:?} vs {expected:?}"
        );
    }
}

/// Rotating the body frame by 90 degrees about z swaps the roles of the x
/// and y principal axes: a world-x torque is resisted by `Iyy`.
#[test]
fn test_apply_impulse_rotates_inverse_inertia_into_world_frame() {
    let mut batch = GpuRigidBodyBatch::new();
    // Ixx = 1, Iyy = 4, Izz = 1; rotated by 90 degrees about z.
    let mut body = GpuRigidBody::new(1.0, 1.0, 4.0, 1.0);
    body.orientation = axis_angle([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
    batch.add(body);
    // Lever arm +z, force +y: torque impulse r x J = (0,0,1) x (0,1,0) =
    // (-1, 0, 0), a unit world-x torque (negative sense).
    assert!(batch.apply_impulse(0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]));
    let omega = batch.bodies[0].angular_velocity;
    // World x is the body y axis: delta_omega_x = tau / Iyy = -1/4.  The
    // body-space tensor applied directly would give -1.
    assert_close3(omega, [-0.25, 0.0, 0.0], 1e-6, "world-x torque");
    // The linear part is untouched by the inertia.
    assert_close3(batch.bodies[0].velocity, [0.0, 1.0, 0.0], 1e-6, "linear");
}

#[test]
fn test_apply_impulse_world_inertia_for_a_quarter_turn_about_x() {
    let mut batch = GpuRigidBodyBatch::new();
    // Ixx = 1, Iyy = 4, Izz = 9, rotated by 90 degrees about x: body y ->
    // world z, body z -> world -y.
    let mut body = GpuRigidBody::new(1.0, 1.0, 4.0, 9.0);
    body.orientation = axis_angle([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2);
    batch.add(body);
    // World-y torque (0, 1, 0) from r = (1,0,0)... use r x J: r = (0,0,1),
    // J = (-1,0,0) gives r x J = (0*0 - 1*0, 1*(-1) - 0*0, 0) = (0,-1,0).
    batch.apply_impulse(0, [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert_close3(
        batch.bodies[0].angular_velocity,
        [0.0, -1.0 / 9.0, 0.0],
        1e-6,
        "world-y torque is resisted by Izz",
    );
    // World-z torque: r = (1,0,0), J = (0,1,0) -> (0,0,1) is the body y axis.
    let mut batch = GpuRigidBodyBatch::new();
    let mut body = GpuRigidBody::new(1.0, 1.0, 4.0, 9.0);
    body.orientation = axis_angle([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2);
    batch.add(body);
    batch.apply_impulse(0, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    assert_close3(
        batch.bodies[0].angular_velocity,
        [0.0, 0.0, 0.25],
        1e-6,
        "world-z torque is resisted by Iyy",
    );
}

#[test]
fn test_world_inv_inertia_matches_body_frame_round_trip() {
    // A full symmetric inverse-inertia tensor and assorted orientations: the
    // world matrix must equal "rotate into the body frame, apply, rotate
    // back".
    let inv_inertia = [2.0, 0.3, -0.2, 0.3, 0.5, 0.1, -0.2, 0.1, 1.25_f32];
    let orientations = [
        quat_normalize([0.1, 0.2, 0.3, 0.9]),
        axis_angle([0.0, 1.0, 0.0], 1.0),
        axis_angle([0.6, 0.0, 0.8], 2.5),
        [0.0, 0.0, 0.0, 1.0],
    ];
    for q in orientations {
        let mut body = GpuRigidBody::new(2.0, 1.0, 1.0, 1.0);
        body.orientation = q;
        body.inv_inertia = inv_inertia;
        let world = world_inv_inertia(&body);
        for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.7, 0.2_f32]] {
            let via_body = quat_rotate(q, apply_mat3(inv_inertia, quat_rotate(conjugate(q), v)));
            assert_close3(apply_mat3(world, v), via_body, 1e-5, "round trip");
        }
        // Symmetric, with the same trace as the body-space tensor.
        assert!((world[1] - world[3]).abs() < 1e-5);
        assert!((world[2] - world[6]).abs() < 1e-5);
        assert!((world[5] - world[7]).abs() < 1e-5);
        let trace = world[0] + world[4] + world[8];
        assert!((trace - 3.75).abs() < 1e-4, "trace {trace}");
    }
}

#[test]
fn test_world_inv_inertia_identity_orientation_and_scaled_quaternion() {
    let mut body = GpuRigidBody::new(1.0, 1.0, 4.0, 9.0);
    let body_space = body.inv_inertia;
    for (world, body_value) in world_inv_inertia(&body).iter().zip(body_space.iter()) {
        assert!((world - body_value).abs() < 1e-6);
    }
    // A non-unit quaternion describes the same rotation.
    let unit = axis_angle([0.0, 0.0, 1.0], 0.7);
    body.orientation = unit;
    let reference = world_inv_inertia(&body);
    body.orientation = [unit[0] * 3.0, unit[1] * 3.0, unit[2] * 3.0, unit[3] * 3.0];
    for (scaled, expected) in world_inv_inertia(&body).iter().zip(reference.iter()) {
        assert!((scaled - expected).abs() < 1e-5);
    }
}

#[test]
fn test_apply_impulse_bad_index_changes_nothing() {
    let mut batch = GpuRigidBodyBatch::new();
    batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    assert!(!batch.apply_impulse(1, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]));
    assert!(!batch.apply_impulse(usize::MAX, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]));
    assert_eq!(batch.bodies[0].velocity, [0.0; 3]);
    assert_eq!(batch.bodies[0].angular_velocity, [0.0; 3]);
    let mut empty = GpuRigidBodyBatch::new();
    assert!(!empty.apply_impulse(0, [1.0, 0.0, 0.0], [0.0; 3]));
}

#[test]
fn test_apply_impulse_static_body_ignores_linear_and_angular_impulse() {
    let mut batch = GpuRigidBodyBatch::new();
    // Mass zero marks the body static even though its tensor is non-zero.
    batch.add(GpuRigidBody::new(0.0, 1.0, 1.0, 1.0));
    assert!(batch.apply_impulse(0, [100.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
    assert_eq!(batch.bodies[0].velocity, [0.0; 3]);
    assert_eq!(batch.bodies[0].angular_velocity, [0.0; 3]);
}

#[test]
fn test_integrate_all_does_not_accelerate_static_bodies() {
    let mut batch = GpuRigidBodyBatch::new();
    let mut platform = GpuRigidBody::new(0.0, 1.0, 1.0, 1.0);
    platform.velocity = [1.0, 0.0, 0.0];
    batch.add(platform);
    batch.add(GpuRigidBody::new(1.0, 1.0, 1.0, 1.0));
    batch.integrate_all(0.5, [0.0, -10.0, 0.0]);
    // The static platform keeps moving kinematically but does not fall.
    assert_close3(
        batch.bodies[0].velocity,
        [1.0, 0.0, 0.0],
        1e-6,
        "platform v",
    );
    assert_close3(
        batch.bodies[0].position,
        [0.5, 0.0, 0.0],
        1e-6,
        "platform p",
    );
    assert!((batch.bodies[1].velocity[1] + 5.0).abs() < 1e-5);
}

// ── Sequential impulse solver: helpers ────────────────────────────────

/// A dynamic body with isotropic inertia.
fn ball(mass: f32, inertia: f32, position: [f32; 3], velocity: [f32; 3]) -> GpuRigidBody {
    let mut body = GpuRigidBody::new(mass, inertia, inertia, inertia);
    body.position = position;
    body.velocity = velocity;
    body
}

/// A static body (mass zero) at `position`.
fn wall(position: [f32; 3]) -> GpuRigidBody {
    let mut body = GpuRigidBody::new(0.0, 1.0, 1.0, 1.0);
    body.position = position;
    body
}

fn pair_of(body_a: usize, body_b: usize) -> BroadphasePairGpu {
    BroadphasePairGpu {
        body_a,
        body_b,
        aabb_a_center: [0.0; 3],
        aabb_a_half: [0.5; 3],
        aabb_b_center: [0.0; 3],
        aabb_b_half: [0.5; 3],
    }
}

/// A solver with the given material constants and one single-contact
/// manifold between `body_a` and `body_b` (normal from B to A).
fn contact_solver(
    restitution: f32,
    friction: f32,
    contacts: &[([f32; 3], [f32; 3], f32)],
    body_a: usize,
    body_b: usize,
) -> GpuConstraintSolver {
    let mut solver = GpuConstraintSolver::new();
    solver.restitution = restitution;
    solver.friction = friction;
    solver.baumgarte = 0.0;
    let mut manifold = ContactManifoldGpu::new(body_a, body_b);
    for &(point, normal, penetration) in contacts {
        manifold.add_contact(point, normal, penetration);
    }
    solver.add_manifold(pair_of(body_a, body_b), manifold);
    solver
}

/// Velocity of the material point of `body` at world position `point`.
fn point_velocity_of(body: &GpuRigidBody, point: [f32; 3]) -> [f32; 3] {
    let r = sub3f(point, body.position);
    add3f(body.velocity, cross3f(body.angular_velocity, r))
}

// ── Sequential impulse solver ─────────────────────────────────────────

#[test]
fn test_solver_default_constants() {
    let solver = GpuConstraintSolver::new();
    assert_eq!(solver.restitution, 0.3);
    assert_eq!(solver.friction, 0.5);
    assert_eq!(solver.baumgarte, 0.2);
    assert_eq!(solver.slop, 0.01);
    assert!(solver.pairs.is_empty() && solver.manifolds.is_empty());
    let defaulted = GpuConstraintSolver::default();
    assert_eq!(defaulted.restitution, solver.restitution);
}

/// Head-on collision of two equal masses; the contact normal points from B
/// to A.  Post-solve relative normal velocity is `-e` times the pre-solve one.
#[test]
fn test_solver_restitution_coefficient_is_honoured() {
    for (restitution, expected_a, expected_b) in
        [(0.0_f32, 0.0, 0.0), (0.5, 0.5, -0.5), (1.0, 1.0, -1.0)]
    {
        let a = ball(1.0, 1.0, [0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        let b = ball(1.0, 1.0, [-0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let solver = contact_solver(restitution, 0.0, &[([0.0; 3], [1.0, 0.0, 0.0], 0.0)], 0, 1);
        let mut bodies = vec![a, b];
        solver.solve_sequential_impulse(&mut bodies, 0.01, 8);
        assert!(
            (bodies[0].velocity[0] - expected_a).abs() < 1e-5
                && (bodies[1].velocity[0] - expected_b).abs() < 1e-5,
            "e = {restitution}: {:?} {:?}",
            bodies[0].velocity,
            bodies[1].velocity
        );
        // Pre-solve approach speed 2: post-solve separation e * 2.
        let separation = bodies[0].velocity[0] - bodies[1].velocity[0];
        assert!((separation - 2.0 * restitution).abs() < 1e-5);
        // Equal masses, no external force: momentum is conserved.
        assert!((bodies[0].velocity[0] + bodies[1].velocity[0]).abs() < 1e-5);
    }
}

/// The restitution comes from the approach speed before the solve, not from
/// the (changing) velocity of later iterations: the result is independent
/// of the iteration count.
#[test]
fn test_solver_restitution_is_independent_of_iteration_count() {
    let mut results = Vec::new();
    for iterations in [1, 2, 5, 40] {
        let a = ball(2.0, 1.0, [0.5, 0.0, 0.0], [-3.0, 0.0, 0.0]);
        let b = ball(1.0, 1.0, [-0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let solver = contact_solver(0.6, 0.0, &[([0.0; 3], [1.0, 0.0, 0.0], 0.0)], 0, 1);
        let mut bodies = vec![a, b];
        solver.solve_sequential_impulse(&mut bodies, 0.01, iterations);
        results.push((bodies[0].velocity[0], bodies[1].velocity[0]));
    }
    for result in &results {
        assert!((result.0 - results[0].0).abs() < 1e-5);
        assert!((result.1 - results[0].1).abs() < 1e-5);
    }
    // Approach speed 4, e = 0.6: separation 2.4 and momentum 2*(-3)+1 = -5.
    let (va, vb) = results[0];
    assert!(((va - vb) - 2.4).abs() < 1e-5);
    assert!((2.0 * va + vb + 5.0).abs() < 1e-4);
}

#[test]
fn test_solver_zero_mass_pair_stays_finite() {
    // Both bodies are static: the old solver divided by 1/ma + 1/mb = 0.
    let mut a = wall([0.5, 0.0, 0.0]);
    a.velocity = [-1.0, 0.0, 0.0];
    let mut b = wall([-0.5, 0.0, 0.0]);
    b.velocity = [1.0, 0.0, 0.0];
    let solver = contact_solver(0.5, 0.5, &[([0.0; 3], [1.0, 0.0, 0.0], 0.2)], 0, 1);
    let mut bodies = vec![a, b];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    assert_eq!(bodies[0].velocity, [-1.0, 0.0, 0.0]);
    assert_eq!(bodies[1].velocity, [1.0, 0.0, 0.0]);
    assert_eq!(bodies[0].angular_velocity, [0.0; 3]);
    assert_eq!(bodies[1].angular_velocity, [0.0; 3]);
}

#[test]
fn test_solver_baumgarte_bias_separates_a_resting_penetration() {
    // Body A rests on a static floor with 0.05 penetration: the pushout
    // speed is beta/dt * (pen - slop) = 0.2/0.01 * 0.04 = 0.8 m/s.
    let run = |penetration: f32, dt: f32| {
        let a = ball(1.0, 1.0, [0.0; 3], [0.0; 3]);
        let mut solver = contact_solver(
            0.3,
            0.0,
            &[([0.0, -0.5, 0.0], [0.0, 1.0, 0.0], penetration)],
            0,
            1,
        );
        solver.baumgarte = 0.2;
        solver.slop = 0.01;
        let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
        solver.solve_sequential_impulse(&mut bodies, dt, 10);
        bodies[0].velocity
    };
    let pushed = run(0.05, 0.01);
    assert_close3(pushed, [0.0, 0.8, 0.0], 1e-5, "bias pushout");
    // Within the slop, nothing happens.
    assert_eq!(run(0.01, 0.01), [0.0; 3]);
    assert_eq!(run(0.003, 0.01), [0.0; 3]);
    assert_eq!(run(-0.2, 0.01), [0.0; 3]);
    // Without a usable time step the bias is disabled rather than infinite.
    for dt in [0.0, -0.01, f32::NAN, f32::INFINITY] {
        assert_eq!(run(0.05, dt), [0.0; 3], "dt = {dt}");
    }
    // A smaller step corrects faster: the speed scales with 1/dt.
    assert_close3(run(0.05, 0.02), [0.0, 0.4, 0.0], 1e-5, "dt = 0.02");
}

#[test]
fn test_solver_bias_and_restitution_are_not_summed() {
    // Approach speed 10, e = 0.3 -> bounce 3.0; bias = 0.2/0.01 * 0.01 = 0.2.
    // The target is the larger of the two, not their sum.
    let a = ball(1.0, 1.0, [0.0; 3], [0.0, -10.0, 0.0]);
    let mut solver = contact_solver(0.3, 0.0, &[([0.0, -0.5, 0.0], [0.0, 1.0, 0.0], 0.02)], 0, 1);
    solver.baumgarte = 0.2;
    solver.slop = 0.01;
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    assert_close3(bodies[0].velocity, [0.0, 3.0, 0.0], 1e-4, "max, not sum");
}

/// Contact off the centre of mass: the normal impulse also spins the body.
/// A unit ball (m = 1, I = 1) falling at 1 m/s hits a static floor at
/// lever arm 1: J = 0.5, v_y = -0.5, omega_z = 0.5, and the contact point
/// itself comes to rest (e = 0).
#[test]
fn test_solver_off_center_contact_induces_spin() {
    let a = ball(1.0, 1.0, [0.0; 3], [0.0, -1.0, 0.0]);
    let solver = contact_solver(0.0, 0.0, &[([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([1.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    assert_close3(bodies[0].velocity, [0.0, -0.5, 0.0], 1e-5, "linear");
    assert_close3(bodies[0].angular_velocity, [0.0, 0.0, 0.5], 1e-5, "spin");
    let contact = point_velocity_of(&bodies[0], [1.0, 0.0, 0.0]);
    assert_close3(contact, [0.0; 3], 1e-5, "contact point rests");
}

/// Off-centre contact against a static body whose tensor is non-zero: the
/// static body must not pick up any linear or angular velocity.
#[test]
fn test_solver_static_body_keeps_zero_angular_velocity() {
    let a = ball(1.0, 1.0, [0.0; 3], [0.4, -1.0, 0.2]);
    // Wall centre well away from the contact point (lever arm (1, 1, 0)).
    let solver = contact_solver(0.3, 0.8, &[([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.05)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 12);
    assert_eq!(bodies[1].velocity, [0.0; 3]);
    assert_eq!(bodies[1].angular_velocity, [0.0; 3]);
    assert!(
        bodies[0].angular_velocity.iter().any(|w| w.abs() > 1e-3),
        "the dynamic body must spin: {:?}",
        bodies[0].angular_velocity
    );
}

/// Ball sliding on a floor: contact at the centre of mass (no friction
/// torque), approach speed 1, tangential speed 2.  Coulomb: the tangential
/// speed drops by exactly `mu * (m * 1)`.
#[test]
fn test_solver_friction_decelerates_sliding() {
    let a = ball(1.0, 1.0, [0.0; 3], [2.0, -1.0, 0.0]);
    let solver = contact_solver(0.0, 0.3, &[([0.0; 3], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    assert_close3(bodies[0].velocity, [1.7, 0.0, 0.0], 1e-5, "sliding");
    assert_eq!(bodies[0].angular_velocity, [0.0; 3]);
}

/// Friction opposes sliding but never reverses it: with a large coefficient
/// the tangential speed stops at zero.
#[test]
fn test_solver_friction_never_reverses_sliding() {
    for friction in [1.0_f32, 5.0, 100.0] {
        let a = ball(1.0, 1.0, [0.0; 3], [0.5, -1.0, 0.0]);
        let solver = contact_solver(0.0, friction, &[([0.0; 3], [0.0, 1.0, 0.0], 0.0)], 0, 1);
        let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
        solver.solve_sequential_impulse(&mut bodies, 0.01, 20);
        assert!(
            bodies[0].velocity[0].abs() < 1e-5,
            "mu = {friction}: tangential speed {}",
            bodies[0].velocity[0]
        );
        assert!(bodies[0].velocity[0] >= -1e-5, "friction reversed sliding");
    }
    // Zero friction leaves the tangential speed alone.
    let a = ball(1.0, 1.0, [0.0; 3], [0.5, -1.0, 0.0]);
    let solver = contact_solver(0.0, 0.0, &[([0.0; 3], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 20);
    assert!((bodies[0].velocity[0] - 0.5).abs() < 1e-6);
}

/// The friction limit is a cone: sliding at (3, 4) in the contact plane
/// loses `mu * lambda_n = 0.5` along the sliding direction (0.6, 0.8), not
/// 0.5 per axis (which a pyramid would remove).
#[test]
fn test_solver_friction_limit_is_a_cone() {
    let a = ball(1.0, 1.0, [0.0; 3], [3.0, -1.0, 4.0]);
    let solver = contact_solver(0.0, 0.5, &[([0.0; 3], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 40);
    assert_close3(
        bodies[0].velocity,
        [3.0 - 0.3, 0.0, 4.0 - 0.4],
        2e-4,
        "cone friction",
    );
}

/// A homogeneous ball (I = 2/5 m r²) that slides on the floor turns into
/// rolling: the classic 5/7 factor.  Tangential speed 1, rolling speed
/// 5/7, spin -5/7 (r = 1), and the contact point does not slip.
#[test]
fn test_solver_sliding_ball_reaches_rolling_velocity() {
    let a = ball(1.0, 0.4, [0.0; 3], [1.0, -1.0, 0.0]);
    let solver = contact_solver(0.0, 1.0, &[([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -2.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 20);
    let five_sevenths = 5.0 / 7.0;
    assert_close3(
        bodies[0].velocity,
        [five_sevenths, 0.0, 0.0],
        1e-4,
        "rolling speed",
    );
    assert_close3(
        bodies[0].angular_velocity,
        [0.0, 0.0, -five_sevenths],
        1e-4,
        "rolling spin",
    );
    let contact = point_velocity_of(&bodies[0], [0.0, -1.0, 0.0]);
    assert_close3(contact, [0.0; 3], 1e-4, "no slip at the contact point");
}

/// With a small friction coefficient the impulse is clamped at `mu * lambda_n`
/// and the ball keeps slipping: v = 1 - 0.1, spin = -0.1 / I = -0.25.
#[test]
fn test_solver_friction_impulse_is_clamped_by_the_normal_impulse() {
    let a = ball(1.0, 0.4, [0.0; 3], [1.0, -1.0, 0.0]);
    let solver = contact_solver(0.0, 0.1, &[([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.0)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -2.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 20);
    assert_close3(bodies[0].velocity, [0.9, 0.0, 0.0], 1e-4, "slipping speed");
    assert_close3(
        bodies[0].angular_velocity,
        [0.0, 0.0, -0.25],
        1e-4,
        "slipping spin",
    );
}

/// Two dynamic bodies with different masses, inertias and orientations hit
/// at three off-centre contact points, with restitution and friction: the
/// solver only exchanges equal and opposite impulses, so linear and angular
/// momentum are conserved.
#[test]
fn test_solver_conserves_linear_and_angular_momentum() {
    let mut a = GpuRigidBody::new(2.0, 1.0, 3.0, 0.5);
    a.position = [0.0, 0.0, 0.0];
    a.velocity = [0.3, -1.2, 0.4];
    a.angular_velocity = [0.5, -0.2, 0.9];
    a.orientation = axis_angle([0.0, 0.6, 0.8], 0.9);
    let mut b = GpuRigidBody::new(0.75, 0.4, 0.4, 1.6);
    b.position = [0.4, -1.1, 0.2];
    b.velocity = [-0.5, 0.8, 0.1];
    b.angular_velocity = [-0.3, 0.6, -0.4];
    b.orientation = axis_angle([1.0, 0.0, 0.0], 0.4);
    let contacts = [
        ([0.3, -0.6, 0.1], normalize3f([0.1, 1.0, 0.05]), 0.02),
        ([0.5, -0.55, -0.1], normalize3f([-0.05, 1.0, 0.1]), 0.03),
        ([0.1, -0.5, 0.3], normalize3f([0.0, 1.0, 0.0]), 0.0),
    ];
    let solver = contact_solver(0.4, 0.6, &contacts, 0, 1);

    let momentum = |bodies: &[GpuRigidBody]| {
        let mut linear = [0.0_f32; 3];
        let mut angular = [0.0_f32; 3];
        for body in bodies {
            let mass = body.mass;
            linear = add3f(linear, scale3f(body.velocity, mass));
            // Body-space diagonal inertia, rotated into the world frame.
            let rotation = rotation_matrix(body.orientation);
            let inertia = [
                1.0 / body.inv_inertia[0],
                0.0,
                0.0,
                0.0,
                1.0 / body.inv_inertia[4],
                0.0,
                0.0,
                0.0,
                1.0 / body.inv_inertia[8],
            ];
            let world = mat3_mul(mat3_mul(rotation, inertia), mat3_transpose(rotation));
            let spin = apply_mat3(world, body.angular_velocity);
            let orbit = scale3f(cross3f(body.position, body.velocity), mass);
            angular = add3f(angular, add3f(orbit, spin));
        }
        (linear, angular)
    };

    let mut bodies = vec![a, b];
    let (linear_before, angular_before) = momentum(&bodies);
    solver.solve_sequential_impulse(&mut bodies, 0.016, 25);
    let (linear_after, angular_after) = momentum(&bodies);
    assert_close3(linear_after, linear_before, 1e-4, "linear momentum");
    assert_close3(angular_after, angular_before, 1e-4, "angular momentum");
    // And the solve did something: the bodies were approaching.
    assert!(
        (bodies[0].velocity[1] - (-1.2)).abs() > 0.05,
        "the solver did not act: {:?}",
        bodies[0].velocity
    );
}

/// Two stacked balls on a static floor, all falling at 2 m/s: Gauss-Seidel
/// converges to a resting stack.
#[test]
fn test_solver_resting_stack_converges() {
    let lower = ball(1.0, 1.0, [0.0, 0.0, 0.0], [0.0, -2.0, 0.0]);
    let upper = ball(1.0, 1.0, [0.0, 1.0, 0.0], [0.0, -2.0, 0.0]);
    let floor = wall([0.0, -1.0, 0.0]);
    let mut solver = GpuConstraintSolver::new();
    solver.restitution = 0.0;
    solver.baumgarte = 0.0;
    let mut lower_floor = ContactManifoldGpu::new(0, 2);
    lower_floor.add_contact([0.0, -0.5, 0.0], [0.0, 1.0, 0.0], 0.0);
    solver.add_manifold(pair_of(0, 2), lower_floor);
    let mut upper_lower = ContactManifoldGpu::new(1, 0);
    upper_lower.add_contact([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], 0.0);
    solver.add_manifold(pair_of(1, 0), upper_lower);
    let mut bodies = vec![lower, upper, floor];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 40);
    assert!(
        bodies[0].velocity[1].abs() < 1e-4,
        "{:?}",
        bodies[0].velocity
    );
    assert!(
        bodies[1].velocity[1].abs() < 1e-4,
        "{:?}",
        bodies[1].velocity
    );
    assert_eq!(bodies[2].velocity, [0.0; 3]);
}

#[test]
fn test_solver_skips_invalid_manifolds_without_panicking() {
    let mut solver = GpuConstraintSolver::new();
    solver.restitution = 0.0;
    solver.baumgarte = 0.0;
    // Out-of-range body indices, in either slot, and a body paired with
    // itself.
    for (a, b) in [(0, 7), (9, 0), (usize::MAX, 0), (1, 1)] {
        let mut manifold = ContactManifoldGpu::new(a, b);
        manifold.add_contact([0.0; 3], [0.0, 1.0, 0.0], 0.1);
        solver.add_manifold(pair_of(0, 1), manifold);
    }
    // Mismatched array lengths: only the complete contact is used.
    let mut ragged = ContactManifoldGpu::new(0, 1);
    ragged.add_contact([0.0, -0.5, 0.0], [0.0, 1.0, 0.0], 0.0);
    ragged.contact_points.push([0.0; 3]);
    ragged.contact_points.push([0.0; 3]);
    solver.add_manifold(pair_of(0, 1), ragged);
    // Degenerate and non-finite normals.
    let mut broken = ContactManifoldGpu::new(0, 1);
    broken.add_contact([0.0; 3], [0.0; 3], 0.1);
    broken.add_contact([0.0; 3], [f32::NAN, 1.0, 0.0], 0.1);
    broken.add_contact([0.0; 3], [f32::INFINITY, 0.0, 0.0], 0.1);
    broken.add_contact([f32::NAN; 3], [0.0, 1.0, 0.0], 0.1);
    solver.add_manifold(pair_of(0, 1), broken);

    let a = ball(1.0, 1.0, [0.0; 3], [0.0, -1.0, 0.0]);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
    // The one valid contact stopped the body; nothing went non-finite.
    assert_close3(bodies[0].velocity, [0.0; 3], 1e-5, "valid contact solved");
    assert!(bodies[0].angular_velocity.iter().all(|w| w.is_finite()));
    assert_eq!(bodies[1].velocity, [0.0; 3]);
}

#[test]
fn test_solver_sanitises_material_constants() {
    let run = |restitution: f32, friction: f32, baumgarte: f32, slop: f32| {
        let a = ball(1.0, 1.0, [0.0; 3], [1.0, -2.0, 0.0]);
        let mut solver = contact_solver(
            restitution,
            friction,
            &[([0.0; 3], [0.0, 1.0, 0.0], 0.05)],
            0,
            1,
        );
        solver.baumgarte = baumgarte;
        solver.slop = slop;
        let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
        solver.solve_sequential_impulse(&mut bodies, 0.01, 10);
        bodies[0].velocity
    };
    // Restitution above one behaves like one; below zero like zero.
    assert_eq!(run(7.0, 0.0, 0.0, 0.0), run(1.0, 0.0, 0.0, 0.0));
    assert_eq!(run(-3.0, 0.0, 0.0, 0.0), run(0.0, 0.0, 0.0, 0.0));
    // NaN or negative friction, baumgarte and slop are treated as zero.
    assert_eq!(run(0.0, f32::NAN, 0.0, 0.0), run(0.0, 0.0, 0.0, 0.0));
    assert_eq!(run(0.0, -2.0, 0.0, 0.0), run(0.0, 0.0, 0.0, 0.0));
    assert_eq!(run(0.0, 0.0, f32::NAN, 0.0), run(0.0, 0.0, 0.0, 0.0));
    assert_eq!(run(0.0, 0.0, -1.0, 0.0), run(0.0, 0.0, 0.0, 0.0));
    assert_eq!(run(0.0, 0.0, 0.2, f32::NAN), run(0.0, 0.0, 0.2, 0.0));
    for velocity in [
        run(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
        run(f32::INFINITY, f32::INFINITY, 1e30, -1.0),
    ] {
        assert!(velocity.iter().all(|v| v.is_finite()), "{velocity:?}");
    }
}

#[test]
fn test_solver_zero_iterations_changes_nothing() {
    let a = ball(1.0, 1.0, [0.0; 3], [1.0, -1.0, 0.0]);
    let solver = contact_solver(0.5, 0.5, &[([0.0; 3], [0.0, 1.0, 0.0], 0.1)], 0, 1);
    let mut bodies = vec![a, wall([0.0, -1.0, 0.0])];
    solver.solve_sequential_impulse(&mut bodies, 0.01, 0);
    assert_eq!(bodies[0].velocity, [1.0, -1.0, 0.0]);
}

// ── Internal helpers ──────────────────────────────────────────────────

#[test]
fn test_cross3f() {
    let i = [1.0_f32, 0.0, 0.0];
    let j = [0.0_f32, 1.0, 0.0];
    let k = cross3f(i, j);
    assert!(approx_eq3(k, [0.0, 0.0, 1.0]));
}

#[test]
fn test_dot3f() {
    assert!((dot3f([1.0, 2.0, 3.0_f32], [4.0, 5.0, 6.0]) - 32.0).abs() < EPS);
}

#[test]
fn test_apply_mat3_identity() {
    let id = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let v = [3.0_f32, 4.0, 5.0];
    assert!(approx_eq3(apply_mat3(id, v), v));
}

#[test]
fn test_apply_mat3_scale() {
    let m = [2.0_f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let v = [1.0_f32, 1.0, 1.0];
    assert!(approx_eq3(apply_mat3(m, v), [2.0, 3.0, 4.0]));
}
