// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the multibody dynamics module.

use super::*;

const EPS: f64 = 1e-10;

// ── SpatialVec ────────────────────────────────────────────────────────────

#[test]
fn test_spatial_vec_zero() {
    let v = SpatialVec::zero();
    assert!(v.norm() < EPS);
}

#[test]
fn test_spatial_vec_add() {
    let a = SpatialVec::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let b = SpatialVec::new([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    let c = a.add(&b);
    assert!((c.angular[0] - 1.0).abs() < EPS);
    assert!((c.angular[1] - 1.0).abs() < EPS);
    assert!((c.linear[0] - 1.0).abs() < EPS);
    assert!((c.linear[1] - 1.0).abs() < EPS);
}

#[test]
fn test_spatial_vec_scale() {
    let v = SpatialVec::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let s = v.scale(2.0);
    assert!((s.angular[0] - 2.0).abs() < EPS);
    assert!((s.linear[2] - 12.0).abs() < EPS);
}

#[test]
fn test_spatial_vec_dot_orthogonal() {
    let a = SpatialVec::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let b = SpatialVec::new([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
    assert!(a.dot(&b).abs() < EPS);
}

#[test]
fn test_spatial_vec_dot_self() {
    let v = SpatialVec::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((v.dot(&v) - 2.0).abs() < EPS);
}

#[test]
fn test_spatial_vec_norm() {
    let v = SpatialVec::new([3.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
    assert!((v.norm() - 5.0).abs() < EPS);
}

// ── SpatialTransform ──────────────────────────────────────────────────────

#[test]
fn test_spatial_transform_identity_motion() {
    let t = SpatialTransform::identity();
    let v = SpatialVec::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let tv = t.apply_motion(&v);
    assert!((tv.angular[0] - 1.0).abs() < EPS);
    assert!((tv.linear[0] - 4.0).abs() < EPS);
}

#[test]
fn test_spatial_transform_translation_motion() {
    let t = SpatialTransform::translation([1.0, 0.0, 0.0]);
    // v = (ω=[0,0,1], v=[0,0,0])
    let v = SpatialVec::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    let tv = t.apply_motion(&v);
    // ω' = R*ω = ω (identity), v' = R*(v - p×ω) = -(p×ω)
    // p×ω = (1,0,0)×(0,0,1) = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)
    // v' = -(0,-1,0) = (0,1,0)
    assert!(
        (tv.linear[1] - 1.0).abs() < 1e-8,
        "tv.linear={:?}",
        tv.linear
    );
}

#[test]
fn test_spatial_transform_inverse_compose_identity() {
    let t = SpatialTransform::translation([1.0, 2.0, 3.0]);
    let t_inv = t.inverse();
    let composed = t.compose(&t_inv);
    // Should be close to identity
    for i in 0..3 {
        assert!(composed.pos[i].abs() < 1e-8, "pos[{i}]={}", composed.pos[i]);
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (composed.rot[i][j] - expected).abs() < 1e-8,
                "rot[{i}][{j}]={}",
                composed.rot[i][j]
            );
        }
    }
}

// ── RigidBodyInertia ──────────────────────────────────────────────────────

#[test]
fn test_sphere_inertia_isotropic() {
    let inertia = RigidBodyInertia::sphere(1.0, 1.0);
    let expected_i = 2.0 / 5.0;
    assert!((inertia.inertia_com[0][0] - expected_i).abs() < EPS);
    assert!((inertia.inertia_com[1][1] - expected_i).abs() < EPS);
    assert!((inertia.inertia_com[2][2] - expected_i).abs() < EPS);
}

#[test]
fn test_sphere_spatial_inertia_nonzero_diag() {
    let inertia = RigidBodyInertia::sphere(2.0, 0.5);
    let si = inertia.to_spatial();
    // Upper left diagonal: I + m*c×c×^T (c=0, so just I_rot)
    let i = 2.0 / 5.0 * 2.0 * 0.5 * 0.5;
    assert!((si.a[0][0] - i).abs() < EPS);
}

#[test]
fn test_point_mass_inertia() {
    let pm = RigidBodyInertia::point_mass(3.0);
    assert!((pm.mass - 3.0).abs() < EPS);
    for i in 0..3 {
        for j in 0..3 {
            assert!(pm.inertia_com[i][j].abs() < EPS);
        }
    }
}

// ── JointAxis ─────────────────────────────────────────────────────────────

#[test]
fn test_revolute_joint_dof() {
    let j = JointAxis::revolute([0.0, 0.0, 1.0]);
    assert_eq!(j.dof(), 1);
}

#[test]
fn test_prismatic_joint_dof() {
    let j = JointAxis::prismatic([1.0, 0.0, 0.0]);
    assert_eq!(j.dof(), 1);
}

#[test]
fn test_revolute_joint_transform_zero() {
    let j = JointAxis::revolute([0.0, 0.0, 1.0]);
    let t = j.joint_transform(0.0);
    // Zero rotation = identity
    for i in 0..3 {
        for k in 0..3 {
            let expected = if i == k { 1.0 } else { 0.0 };
            assert!((t.rot[i][k] - expected).abs() < 1e-8);
        }
    }
}

#[test]
fn test_revolute_joint_transform_90deg() {
    let j = JointAxis::revolute([0.0, 0.0, 1.0]);
    let t = j.joint_transform(std::f64::consts::FRAC_PI_2);
    // Should rotate X axis to Y axis
    // col 0 of R should be (0, 1, 0)
    assert!((t.rot[0][0]).abs() < 1e-8); // R[0][0] = cos(90°) = 0
    assert!((t.rot[1][0] - 1.0).abs() < 1e-8); // R[1][0] = sin(90°) = 1
}

#[test]
fn test_prismatic_joint_transform() {
    let j = JointAxis::prismatic([1.0, 0.0, 0.0]);
    let t = j.joint_transform(2.5);
    assert!((t.pos[0] - 2.5).abs() < EPS);
    assert!(t.pos[1].abs() < EPS);
    assert!(t.pos[2].abs() < EPS);
}

// ── ArticulatedBody ───────────────────────────────────────────────────────

#[test]
fn test_build_serial_chain() {
    let body = build_serial_chain(3, 1.0, 1.0, [0.0, -9.81, 0.0]);
    assert_eq!(body.num_links(), 3);
    assert_eq!(body.num_dof(), 3);
}

#[test]
fn test_forward_kinematics_zero_q() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0, -9.81, 0.0]);
    body.forward_kinematics();
    // With q=0, transforms should be sequential translations
    // Link 0 world pos ≈ (1, 0, 0)
    let p0 = body.links[0].x_world.pos;
    assert!((p0[0] - 1.0).abs() < 1e-6, "link 0 x={}", p0[0]);
}

#[test]
fn test_rnea_zero_qdd_no_gravity() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0, 0.0, 0.0]);
    body.forward_kinematics();
    let tau = body.rnea(&[0.0, 0.0], Some([0.0; 3]));
    // With zero qd and zero qdd and zero gravity: tau should be near zero
    for (i, &t) in tau.iter().enumerate() {
        assert!(t.abs() < 1e-4, "tau[{i}]={t}");
    }
}

#[test]
fn test_aba_zero_tau_gives_gravity_acc() {
    // A link whose centre of mass lies on the joint axis feels no gravity torque.
    let mut on_axis = build_serial_chain(1, 1.0, 1.0, [0.0, -9.81, 0.0]);
    let qdd = on_axis.aba(&[0.0]);
    assert_eq!(qdd.len(), 1);
    assert!(qdd[0].abs() < 1e-12, "qdd={}", qdd[0]);

    // A pendulum with the centre of mass 0.5 m along x from the joint, released
    // horizontally: qdd = -m g lc / (Izz + m lc^2) about +z.
    let (mass, lc, izz, g) = (2.0, 0.5, 0.3, 9.81);
    let inertia = RigidBodyInertia::new(
        mass,
        [lc, 0.0, 0.0],
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, izz]],
    );
    let link = Link::new(
        "pendulum",
        -1,
        inertia,
        JointAxis::revolute([0.0, 0.0, 1.0]),
        SpatialTransform::identity(),
    );
    let mut pendulum = ArticulatedBody::new([0.0, -g, 0.0]);
    pendulum.add_link(link);
    let qdd = pendulum.aba(&[0.0]);
    let expected = -mass * g * lc / (izz + mass * lc * lc);
    assert!(
        (qdd[0] - expected).abs() < 1e-10,
        "qdd={} expected={expected}",
        qdd[0]
    );
}

#[test]
fn test_crba_diagonal_positive() {
    let mut body = build_serial_chain(3, 1.0, 1.0, [0.0, 0.0, 0.0]);
    body.forward_kinematics();
    let h = body.crba();
    // Diagonal of CRBA should be positive
    for (i, row) in h.iter().enumerate() {
        assert!(row[i] > 0.0, "H[{i}][{i}]={}", row[i]);
    }
}

#[test]
fn test_crba_symmetric() {
    let mut body = build_serial_chain(3, 1.0, 1.0, [0.0, 0.0, 0.0]);
    body.forward_kinematics();
    let h = body.crba();
    for (i, row) in h.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                (val - h[j][i]).abs() < 1e-8,
                "H not symmetric at ({i},{j}): {val} vs {}",
                h[j][i]
            );
        }
    }
}

// ── Baumgarte ─────────────────────────────────────────────────────────────

#[test]
fn test_baumgarte_zero_errors() {
    let params = BaumgarteParams::default();
    let corr = baumgarte_stabilization(0.0, 0.0, &params, 0.01);
    assert!(corr.abs() < EPS);
}

#[test]
fn test_baumgarte_position_error_drives_correction() {
    let params = BaumgarteParams::default();
    let corr = baumgarte_stabilization(0.1, 0.0, &params, 0.01);
    // Correction should be negative (restoring) for positive position error
    assert!(corr < 0.0, "correction={corr}");
}

// ── Operational space ─────────────────────────────────────────────────────

#[test]
fn test_operational_space_inertia_identity_jacobian() {
    // J = [[1, 0]], H_inv = [1]
    let j = vec![vec![1.0, 0.0]];
    let h_inv = vec![1.0, 1.0];
    let lambda = operational_space_inertia(&j, &h_inv);
    // Lambda = (J H^{-1} J^T)^{-1} = (1*1*1)^{-1} = 1
    assert!((lambda[0][0] - 1.0).abs() < 1e-8, "lambda={}", lambda[0][0]);
}

// ── Loop constraint ───────────────────────────────────────────────────────

#[test]
fn test_loop_constraint_zero_error() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0, 0.0, 0.0]);
    body.forward_kinematics();
    // Get actual relative position
    let pa = body.links[0].x_world.pos;
    let pb = body.links[1].x_world.pos;
    let target = sub3(pb, pa);
    let lc = LoopConstraint::new(0, 1, target, 1.0);
    let err = lc.position_error(&body);
    let norm = len3(err);
    assert!(norm < 1e-8, "loop constraint error={norm}");
}

// ── Cable-driven ──────────────────────────────────────────────────────────

#[test]
fn test_cable_segment_creation() {
    let cable = CableSegment::new(0, [0.0; 3], 1, [0.0; 3], 1.0, 1000.0, 10.0);
    assert!((cable.rest_length - 1.0).abs() < EPS);
    assert!((cable.stiffness - 1000.0).abs() < EPS);
}

#[test]
fn test_cable_zero_force_at_rest_length() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0; 3]);
    body.forward_kinematics();
    // Place cable between links at rest length = distance between origins
    let pa = body.links[0].x_world.pos;
    let pb = body.links[1].x_world.pos;
    let dist = len3(sub3(pb, pa));
    let cable = CableSegment::new(0, [0.0; 3], 1, [0.0; 3], dist, 1000.0, 0.0);
    let (fa, fb) = cable.compute_force(&body);
    // At rest length: zero force
    assert!(fa.norm() < 1e-8, "fa={}", fa.norm());
    assert!(fb.norm() < 1e-8, "fb={}", fb.norm());
}

// ── Hybrid dynamics ───────────────────────────────────────────────────────

#[test]
fn test_hybrid_dynamics_all_forward() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0, 0.0, 0.0]);
    body.forward_kinematics();
    let modes = vec![HybridMode::ForwardDynamics; 2];
    let result = hybrid_dynamics(&mut body, &modes).expect("positive definite inertia");
    assert_eq!(result.len(), 2);
}

// ── Pseudo-inverse / null-space ───────────────────────────────────────────

#[test]
fn test_pseudo_inverse_fat_1x2() {
    // J = [[1, 0]], J+ = [[1], [0]] (min-norm pseudo-inverse)
    let j = vec![vec![1.0, 0.0]];
    let jp = pseudo_inverse_fat(&j);
    assert_eq!(jp.len(), 2);
    assert!((jp[0][0] - 1.0).abs() < 1e-8, "jp[0][0]={}", jp[0][0]);
    assert!(jp[1][0].abs() < 1e-8, "jp[1][0]={}", jp[1][0]);
}

#[test]
fn test_null_space_projector_1x2() {
    let j = vec![vec![1.0, 0.0]];
    let jp = pseudo_inverse_fat(&j);
    let n_proj = null_space_projector(&jp, &j);
    // N = I - J+ J = I - [[1],[0]]*[[1,0]] = [[0,0],[0,1]]
    assert!(n_proj[0][0].abs() < 1e-8);
    assert!((n_proj[1][1] - 1.0).abs() < 1e-8);
}

// ── Integration ───────────────────────────────────────────────────────────

#[test]
fn test_euler_integrate_zero_tau() {
    let mut body = build_serial_chain(1, 1.0, 1.0, [0.0; 3]);
    let q0 = body.links[0].q;
    let qd0 = body.links[0].qd;
    euler_integrate(&mut body, 0.01);
    // With zero tau and zero gravity, qd unchanged (or very small), q unchanged
    let q1 = body.links[0].q;
    let qd1 = body.links[0].qd;
    // No gravity: acceleration should be zero → q and qd unchanged
    assert!((q1 - q0).abs() < 1e-6, "q changed: {q1} vs {q0}");
    assert!((qd1 - qd0).abs() < 1e-6, "qd changed: {qd1} vs {qd0}");
}

#[test]
fn test_symplectic_euler_integrate() {
    let mut body = build_serial_chain(1, 1.0, 1.0, [0.0; 3]);
    body.links[0].tau = 1.0; // apply torque
    let qd0 = body.links[0].qd;
    symplectic_euler_integrate(&mut body, 0.01);
    let qd1 = body.links[0].qd;
    // Velocity should have increased due to torque
    assert!(qd1 > qd0, "qd should increase under positive torque");
}

#[test]
fn test_set_and_get_q() {
    let mut body = build_serial_chain(3, 1.0, 1.0, [0.0; 3]);
    body.set_q(&[0.1, 0.2, 0.3]);
    let q = body.get_q();
    assert!((q[0] - 0.1).abs() < EPS);
    assert!((q[1] - 0.2).abs() < EPS);
    assert!((q[2] - 0.3).abs() < EPS);
}

#[test]
fn test_set_and_get_qd() {
    let mut body = build_serial_chain(2, 1.0, 1.0, [0.0; 3]);
    body.set_qd(&[1.5, -0.5]);
    let qd = body.get_qd();
    assert!((qd[0] - 1.5).abs() < EPS);
    assert!((qd[1] + 0.5).abs() < EPS);
}

#[test]
fn test_cable_driven_system_creation() {
    let body = build_serial_chain(2, 1.0, 1.0, [0.0; 3]);
    let mut cds = CableDrivenSystem::new(body);
    let cable = CableSegment::new(0, [0.0; 3], 1, [0.0; 3], 1.0, 1000.0, 0.0);
    cds.add_cable(cable);
    assert_eq!(cds.cables.len(), 1);
}

#[test]
fn test_project_constraint_zero_error() {
    let q = 1.5;
    let q_new = project_constraint(q, 0.0, 1.0);
    assert!((q_new - q).abs() < EPS);
}

#[test]
fn test_project_constraint_nonzero_error() {
    let q = 1.0;
    let err = 0.1;
    let gain = 0.5;
    let q_new = project_constraint(q, err, gain);
    assert!((q_new - (q - gain * err)).abs() < EPS);
}

#[test]
fn test_spatial_mat_identity_mul_vec() {
    let i = SpatialMat::identity();
    let v = SpatialVec::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let iv = i.mul_vec(&v);
    for k in 0..3 {
        assert!((iv.angular[k] - v.angular[k]).abs() < EPS);
        assert!((iv.linear[k] - v.linear[k]).abs() < EPS);
    }
}

#[test]
fn test_skew_matrix_cross_product() {
    let v = [1.0, 2.0, 3.0];
    let sv = skew(v);
    let u = [4.0, 5.0, 6.0];
    let sv_u = mat3_vec(sv, u);
    let expected = cross3(v, u);
    for k in 0..3 {
        assert!(
            (sv_u[k] - expected[k]).abs() < EPS,
            "k={k}: sv_u={} expected={}",
            sv_u[k],
            expected[k]
        );
    }
}
