// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the multibody dynamics (audit findings F0415, F0416,
//! F0420, F0421).
//!
//! The dynamics are validated against references that do not share code with
//! the algorithms under test: closed-form planar-arm dynamics, finite
//! differences of the forward-kinematics poses and of the Lagrangian,
//! the kinetic energy of the link velocities, and the identities that tie the
//! recursive algorithms together (`aba ∘ rnea = id`, `H e_j = rnea(e_j)`,
//! `τ = H q̈ + C`).

use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// ── helpers ──────────────────────────────────────────────────────────────────

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn rotation_about(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    JointAxis::revolute(axis).joint_transform(angle).rot
}

fn random_vec3(rng: &mut StdRng, range: f64) -> [f64; 3] {
    [
        rng.random_range(-range..range),
        rng.random_range(-range..range),
        rng.random_range(-range..range),
    ]
}

fn random_unit(rng: &mut StdRng) -> [f64; 3] {
    loop {
        let v = random_vec3(rng, 1.0);
        if len3(v) > 0.2 {
            return norm3(v);
        }
    }
}

/// A rigid body with a random mass, centre of mass and a physical inertia
/// tensor (principal moments in [0.1, 0.2) satisfy the triangle inequality,
/// rotated into a random frame).
fn random_inertia(rng: &mut StdRng) -> RigidBodyInertia {
    let principal = [
        [rng.random_range(0.1..0.2), 0.0, 0.0],
        [0.0, rng.random_range(0.1..0.2), 0.0],
        [0.0, 0.0, rng.random_range(0.1..0.2)],
    ];
    let r = rotation_about(random_unit(rng), rng.random_range(-3.0..3.0));
    let tensor = mat3_mul(mat3_mul(r, principal), mat3_transpose(r));
    RigidBodyInertia::new(rng.random_range(0.5..2.0), random_vec3(rng, 0.4), tensor)
}

/// A random tree of revolute / prismatic joints in a random pose and motion
/// state, subject to a random external wrench on every link.
fn random_tree(rng: &mut StdRng, n_links: usize, branching: bool) -> ArticulatedBody {
    let mut body = ArticulatedBody::new([0.3, -9.81, 0.5]);
    for i in 0..n_links {
        let parent = if i == 0 {
            -1
        } else if branching {
            rng.random_range(0..i) as i64
        } else {
            (i - 1) as i64
        };
        let axis = random_unit(rng);
        let joint = if rng.random_range(0.0..1.0) < 0.7 {
            JointAxis::revolute(axis)
        } else {
            JointAxis::prismatic(axis)
        };
        let placement = SpatialTransform::new(
            rotation_about(random_unit(rng), rng.random_range(-3.0..3.0)),
            random_vec3(rng, 1.0),
        );
        let mut link = Link::new(
            format!("link_{i}"),
            parent,
            random_inertia(rng),
            joint,
            placement,
        );
        link.q = rng.random_range(-1.5..1.5);
        link.qd = rng.random_range(-1.5..1.5);
        link.f_ext = SpatialVec::new(random_vec3(rng, 2.0), random_vec3(rng, 2.0));
        body.add_link(link);
    }
    body
}

fn clear_external_wrenches(body: &mut ArticulatedBody) {
    for link in &mut body.links {
        link.f_ext = SpatialVec::zero();
    }
}

fn random_vector(rng: &mut StdRng, n: usize, range: f64) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-range..range)).collect()
}

/// Solve `H x = b` for a symmetric positive-definite `H` given as rows.
fn spd_solve(h: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let flat: Vec<f64> = h.iter().flatten().copied().collect();
    let factor = cholesky_factor(&flat, n).expect("joint-space inertia must be positive definite");
    cholesky_solve(&factor, n, b, 1)
}

fn mat_vec(h: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    h.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

// ── spatial algebra ──────────────────────────────────────────────────────────

#[test]
fn spatial_transform_follows_point_kinematics() {
    // Frame B sits at (1, 0, 0) in A, rotated by 90 degrees about z: B's x axis
    // is A's y axis.
    let r = rotation_about([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
    let placement = SpatialTransform::new(r, [1.0, 0.0, 0.0]);

    // A twist rotating about A's origin with unit rate about z: the point at B's
    // origin moves with velocity omega x p = (0, 1, 0) in A, which is B's +x.
    let twist_a = SpatialVec::new([0.0, 0.0, 1.0], [0.0; 3]);
    let twist_b = placement.apply_motion(&twist_a);
    let expected_b = SpatialVec::new([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    assert!(
        twist_b.sub(&expected_b).norm() < 1e-12,
        "twist_b={twist_b:?}"
    );

    // A pure force along B's x axis through B's origin: along A's y axis, with a
    // moment about A's origin of p x f = (0, 0, 1).
    let wrench_b = SpatialVec::new([0.0; 3], [1.0, 0.0, 0.0]);
    let wrench_a = placement.apply_force(&wrench_b);
    let expected_a = SpatialVec::new([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    assert!(
        wrench_a.sub(&expected_a).norm() < 1e-12,
        "wrench_a={wrench_a:?}"
    );
}

#[test]
fn spatial_transform_matrices_agree_and_conserve_power() {
    let mut rng = StdRng::seed_from_u64(21);
    let placement = SpatialTransform::new(
        rotation_about(random_unit(&mut rng), 1.1),
        random_vec3(&mut rng, 1.0),
    );
    let motion = SpatialVec::new(random_vec3(&mut rng, 1.0), random_vec3(&mut rng, 1.0));
    let wrench_b = SpatialVec::new(random_vec3(&mut rng, 1.0), random_vec3(&mut rng, 1.0));

    let by_matrix = placement.motion_matrix().mul_vec(&motion);
    assert!(by_matrix.sub(&placement.apply_motion(&motion)).norm() < 1e-12);
    let by_matrix = placement.force_matrix().mul_vec(&wrench_b);
    assert!(by_matrix.sub(&placement.apply_force(&wrench_b)).norm() < 1e-12);

    // The force transform is the transpose of the motion transform, so the
    // power v . f is the same in both frames.
    let power_b = placement.apply_motion(&motion).dot(&wrench_b);
    let power_a = motion.dot(&placement.apply_force(&wrench_b));
    assert!((power_a - power_b).abs() < 1e-12, "{power_a} vs {power_b}");

    // The transform composes like the placements do.
    let other = SpatialTransform::new(
        rotation_about(random_unit(&mut rng), -0.4),
        random_vec3(&mut rng, 1.0),
    );
    let chained = placement.compose(&other);
    let stepwise = other.apply_motion(&placement.apply_motion(&motion));
    assert!(chained.apply_motion(&motion).sub(&stepwise).norm() < 1e-12);
    // ... and the inverse undoes it.
    let round_trip = placement
        .inverse()
        .apply_motion(&placement.apply_motion(&motion));
    assert!(round_trip.sub(&motion).norm() < 1e-12);
}

#[test]
fn inertia_transform_matches_rigid_body_inertia_in_the_new_frame() {
    let mut rng = StdRng::seed_from_u64(22);
    let body_inertia = random_inertia(&mut rng);
    let r = rotation_about(random_unit(&mut rng), 0.9);
    let p = random_vec3(&mut rng, 1.0);
    let placement = SpatialTransform::new(r, p);

    // The same body described directly in frame A: centre of mass R c + p and
    // rotational inertia R Ic R^T about it.
    let in_a = RigidBodyInertia::new(
        body_inertia.mass,
        add3(mat3_vec(r, body_inertia.com), p),
        mat3_mul(mat3_mul(r, body_inertia.inertia_com), mat3_transpose(r)),
    )
    .to_spatial();
    let transformed = placement.inertia_to_parent(&body_inertia.to_spatial());

    for (got, want) in [
        (transformed.a, in_a.a),
        (transformed.b, in_a.b),
        (transformed.c, in_a.c),
        (transformed.d, in_a.d),
    ] {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (got[i][j] - want[i][j]).abs() < 1e-12,
                    "block mismatch at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn spatial_outer_product_is_rank_one() {
    let u = SpatialVec::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let v = SpatialVec::new([-1.0, 0.5, 2.0], [0.0, 3.0, -2.0]);
    let m = SpatialMat::outer(&u, &v);
    let w = SpatialVec::new([0.3, -0.7, 1.1], [2.0, -1.0, 0.4]);
    // (u v^T) w = u (v . w)
    let got = m.mul_vec(&w);
    let want = u.scale(v.dot(&w));
    assert!(got.sub(&want).norm() < 1e-12);
}

// ── kinematics / dynamics consistency ────────────────────────────────────────

fn twist_from_pose_difference(
    before: &SpatialTransform,
    now: &SpatialTransform,
    after: &SpatialTransform,
    dt: f64,
) -> SpatialVec {
    let mut d_rot = [[0.0; 3]; 3];
    for ((d_row, after_row), before_row) in d_rot.iter_mut().zip(&after.rot).zip(&before.rot) {
        for ((d, a), b) in d_row.iter_mut().zip(after_row).zip(before_row) {
            *d = (a - b) / (2.0 * dt);
        }
    }
    let omega_hat = mat3_mul(mat3_transpose(now.rot), d_rot);
    let omega = [omega_hat[2][1], omega_hat[0][2], omega_hat[1][0]];
    let d_pos = scale3(sub3(after.pos, before.pos), 1.0 / (2.0 * dt));
    SpatialVec::new(omega, mat3_vec(mat3_transpose(now.rot), d_pos))
}

#[test]
fn forward_kinematics_twists_match_finite_differences_of_the_poses() {
    let mut rng = StdRng::seed_from_u64(31);
    let mut body = random_tree(&mut rng, 5, true);
    let q0 = body.get_q();
    let qd = body.get_qd();
    let dt = 1e-6;
    let pose_at = |t: f64| -> Vec<SpatialTransform> {
        let mut shifted = body.clone();
        let q: Vec<f64> = q0.iter().zip(&qd).map(|(q, v)| q + t * v).collect();
        shifted.set_q(&q);
        shifted.forward_kinematics();
        shifted.links.iter().map(|l| l.x_world).collect()
    };
    let (before, now, after) = (pose_at(-dt), pose_at(0.0), pose_at(dt));
    body.forward_kinematics();
    for (i, link) in body.links.iter().enumerate() {
        let numeric = twist_from_pose_difference(&before[i], &now[i], &after[i], dt);
        let err = numeric.sub(&link.body_vel).norm();
        assert!(
            err < 1e-6,
            "link {i}: twist differs from the pose derivative by {err}"
        );
    }
}

#[test]
fn kinetic_energy_from_crba_equals_sum_of_link_energies() {
    for seed in 0..6 {
        let mut rng = StdRng::seed_from_u64(40 + seed);
        let mut body = random_tree(&mut rng, 6, seed.is_multiple_of(2));
        let h = body.crba();
        let qd = body.get_qd();
        let h_qd = mat_vec(&h, &qd);
        let from_h = 0.5 * qd.iter().zip(&h_qd).map(|(a, b)| a * b).sum::<f64>();
        body.forward_kinematics();
        let from_links: f64 = body
            .links
            .iter()
            .map(|l| 0.5 * l.body_vel.dot(&l.inertia.to_spatial().mul_vec(&l.body_vel)))
            .sum();
        assert!(
            (from_h - from_links).abs() < 1e-9 * (1.0 + from_links.abs()),
            "seed {seed}: {from_h} vs {from_links}"
        );
    }
}

// ── CRBA (F0416) ─────────────────────────────────────────────────────────────

/// Two-link planar arm in the x-y plane (joints about z); gravity along -y.
struct PlanarArm {
    m1: f64,
    m2: f64,
    l1: f64,
    lc1: f64,
    lc2: f64,
    i1: f64,
    i2: f64,
    g: f64,
}

impl PlanarArm {
    fn sample() -> Self {
        Self {
            m1: 1.3,
            m2: 0.7,
            l1: 0.9,
            lc1: 0.4,
            lc2: 0.35,
            i1: 0.05,
            i2: 0.03,
            g: 9.81,
        }
    }

    fn build(&self, q: [f64; 2], qd: [f64; 2]) -> ArticulatedBody {
        let axis = JointAxis::revolute([0.0, 0.0, 1.0]);
        let mut link1 = Link::new(
            "upper",
            -1,
            RigidBodyInertia::new(
                self.m1,
                [self.lc1, 0.0, 0.0],
                [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, self.i1]],
            ),
            axis,
            SpatialTransform::identity(),
        );
        let mut link2 = Link::new(
            "lower",
            0,
            RigidBodyInertia::new(
                self.m2,
                [self.lc2, 0.0, 0.0],
                [[0.015, 0.0, 0.0], [0.0, 0.025, 0.0], [0.0, 0.0, self.i2]],
            ),
            axis,
            SpatialTransform::translation([self.l1, 0.0, 0.0]),
        );
        link1.q = q[0];
        link2.q = q[1];
        link1.qd = qd[0];
        link2.qd = qd[1];
        let mut body = ArticulatedBody::new([0.0, -self.g, 0.0]);
        body.add_link(link1);
        body.add_link(link2);
        body
    }

    fn inertia_matrix(&self, q2: f64) -> [[f64; 2]; 2] {
        let h11 = self.i1
            + self.i2
            + self.m1 * self.lc1 * self.lc1
            + self.m2
                * (self.l1 * self.l1 + self.lc2 * self.lc2 + 2.0 * self.l1 * self.lc2 * q2.cos());
        let h12 = self.i2 + self.m2 * (self.lc2 * self.lc2 + self.l1 * self.lc2 * q2.cos());
        let h22 = self.i2 + self.m2 * self.lc2 * self.lc2;
        [[h11, h12], [h12, h22]]
    }

    /// `tau = H qdd + C(q, qd) qd + G(q)` (Spong, *Robot Modeling and Control*).
    fn torques(&self, q: [f64; 2], qd: [f64; 2], qdd: [f64; 2]) -> [f64; 2] {
        let h = self.inertia_matrix(q[1]);
        let hh = -self.m2 * self.l1 * self.lc2 * q[1].sin();
        let coriolis = [
            hh * (2.0 * qd[0] * qd[1] + qd[1] * qd[1]),
            -hh * qd[0] * qd[0],
        ];
        let q12 = q[0] + q[1];
        let gravity = [
            (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * q[0].cos()
                + self.m2 * self.lc2 * self.g * q12.cos(),
            self.m2 * self.lc2 * self.g * q12.cos(),
        ];
        [
            h[0][0] * qdd[0] + h[0][1] * qdd[1] + coriolis[0] + gravity[0],
            h[1][0] * qdd[0] + h[1][1] * qdd[1] + coriolis[1] + gravity[1],
        ]
    }
}

#[test]
fn crba_matches_the_analytic_planar_arm_inertia_matrix() {
    let arm = PlanarArm::sample();
    for q2 in [-2.0, -0.9, 0.0, 0.7, 2.4] {
        let mut body = arm.build([0.35, q2], [0.0; 2]);
        let h = body.crba();
        let expected = arm.inertia_matrix(q2);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (h[i][j] - expected[i][j]).abs() < 1e-12,
                    "q2={q2}: H[{i}][{j}]={} expected {}",
                    h[i][j],
                    expected[i][j]
                );
            }
        }
    }
}

#[test]
fn crba_columns_equal_rnea_with_unit_accelerations() {
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(50 + seed);
        let n = 3 + (seed % 4) as usize;
        let mut body = random_tree(&mut rng, n, seed.is_multiple_of(2));
        // H is configuration dependent only: no motion, gravity or wrenches.
        clear_external_wrenches(&mut body);
        body.set_qd(&vec![0.0; n]);
        let h = body.crba();
        for j in 0..n {
            let mut unit = vec![0.0; n];
            unit[j] = 1.0;
            let column = body.rnea(&unit, Some([0.0; 3]));
            for i in 0..n {
                assert!(
                    (h[i][j] - column[i]).abs() < 1e-10,
                    "seed {seed}: H[{i}][{j}]={} but rnea gives {}",
                    h[i][j],
                    column[i]
                );
                assert!((h[i][j] - h[j][i]).abs() < 1e-12, "H must be symmetric");
            }
        }
        // A physical system has a positive-definite joint-space inertia.
        let flat: Vec<f64> = h.iter().flatten().copied().collect();
        assert!(
            cholesky_factor(&flat, n).is_some(),
            "seed {seed}: H not SPD"
        );
    }
}

// ── RNEA ─────────────────────────────────────────────────────────────────────

fn offset_pendulum(placement: SpatialTransform) -> ArticulatedBody {
    let inertia = RigidBodyInertia::new(
        2.0,
        [0.5, 0.0, 0.0],
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]],
    );
    let link = Link::new(
        "pendulum",
        -1,
        inertia,
        JointAxis::revolute([0.0, 0.0, 1.0]),
        placement,
    );
    let mut body = ArticulatedBody::new([0.0, -9.81, 0.0]);
    body.add_link(link);
    body
}

#[test]
fn gravity_torque_of_a_pendulum_follows_the_joint_angle() {
    // tau_hold = m g lc cos(q) for a link of length direction rotated by q.
    let (mass, lc, izz, g) = (2.0, 0.5, 0.3, 9.81);
    let placements = [
        SpatialTransform::identity(),
        SpatialTransform::translation([0.3, 0.2, -0.1]),
    ];
    for placement in placements {
        for q in [0.0, 0.7, std::f64::consts::FRAC_PI_2, -1.1, 2.5] {
            let mut body = offset_pendulum(placement);
            body.links[0].q = q;
            let hold = body.rnea(&[0.0], None)[0];
            let expected = mass * g * lc * q.cos();
            assert!(
                (hold - expected).abs() < 1e-10,
                "q={q}: {hold} vs {expected}"
            );

            // Released from rest the pendulum accelerates by -tau_hold / I_total.
            let qdd = body.aba(&[0.0])[0];
            let expected_qdd = -expected / (izz + mass * lc * lc);
            assert!(
                (qdd - expected_qdd).abs() < 1e-10,
                "q={q}: {qdd} vs {expected_qdd}"
            );
        }
    }
}

#[test]
fn rnea_matches_the_analytic_planar_arm_dynamics() {
    let arm = PlanarArm::sample();
    let states = [
        ([0.4, -0.9], [0.7, -1.1], [0.3, 0.8]),
        ([-1.2, 2.0], [-0.5, 0.9], [1.3, -0.2]),
        ([2.7, 0.3], [1.9, 2.2], [-0.6, 0.5]),
    ];
    for (q, qd, qdd) in states {
        let mut body = arm.build(q, qd);
        let tau = body.rnea(&qdd, None);
        let expected = arm.torques(q, qd, qdd);
        assert!(
            (tau[0] - expected[0]).abs() < 1e-10 && (tau[1] - expected[1]).abs() < 1e-10,
            "q={q:?}: rnea {tau:?} vs textbook {expected:?}"
        );
    }
}

fn potential_energy(body: &ArticulatedBody) -> f64 {
    let mut posed = body.clone();
    posed.forward_kinematics();
    posed
        .links
        .iter()
        .map(|l| {
            let com_world = add3(mat3_vec(l.x_world.rot, l.inertia.com), l.x_world.pos);
            -l.inertia.mass * dot3(posed.gravity, com_world)
        })
        .sum()
}

fn inertia_at(body: &ArticulatedBody, q: &[f64]) -> Vec<Vec<f64>> {
    let mut posed = body.clone();
    posed.set_q(q);
    posed.crba()
}

#[test]
fn rnea_satisfies_the_euler_lagrange_equations() {
    // tau_k = sum_j H_kj qdd_j + sum_ij (dH_kj/dq_i - dH_ij/dq_k / 2) qd_i qd_j + dV/dq_k,
    // with H from CRBA and V from the forward-kinematics poses: the check shares
    // neither the velocity recursion nor the force recursion with RNEA.
    for seed in 0..4 {
        let mut rng = StdRng::seed_from_u64(60 + seed);
        let n = 4;
        let mut body = random_tree(&mut rng, n, seed.is_multiple_of(2));
        clear_external_wrenches(&mut body);
        let q0 = body.get_q();
        let qd = body.get_qd();
        let qdd = random_vector(&mut rng, n, 2.0);
        let tau = body.rnea(&qdd, None);

        let eps = 1e-6;
        let mut d_h: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
        let mut grad_v = vec![0.0; n];
        for i in 0..n {
            let (mut plus, mut minus) = (q0.clone(), q0.clone());
            plus[i] += eps;
            minus[i] -= eps;
            let (hp, hm) = (inertia_at(&body, &plus), inertia_at(&body, &minus));
            d_h.push(
                hp.iter()
                    .zip(&hm)
                    .map(|(rp, rm)| {
                        rp.iter()
                            .zip(rm)
                            .map(|(a, b)| (a - b) / (2.0 * eps))
                            .collect()
                    })
                    .collect(),
            );
            let mut bp = body.clone();
            bp.set_q(&plus);
            let mut bm = body.clone();
            bm.set_q(&minus);
            grad_v[i] = (potential_energy(&bp) - potential_energy(&bm)) / (2.0 * eps);
        }
        let h = inertia_at(&body, &q0);
        let mut expected = mat_vec(&h, &qdd);
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    expected[k] +=
                        d_h[i][k][j] * qd[i] * qd[j] - 0.5 * d_h[k][i][j] * qd[i] * qd[j];
                }
            }
            expected[k] += grad_v[k];
        }
        let err = max_abs_diff(&tau, &expected);
        assert!(
            err < 1e-5,
            "seed {seed}: Lagrange vs RNEA differ by {err} (tau {tau:?})"
        );
    }
}

#[test]
fn rnea_accumulates_the_forces_of_all_children() {
    // Two children on one parent: the parent's joint torque must carry both
    // subtrees.  With the parent held at rest and no gravity, the torque
    // required for a given qdd is the sum of what each child would need alone.
    let mut rng = StdRng::seed_from_u64(70);
    let mut body = random_tree(&mut rng, 3, false);
    // Re-attach link 2 to link 0 (siblings 1 and 2).
    body.links[2].parent = 0;
    clear_external_wrenches(&mut body);
    body.set_qd(&[0.0; 3]);
    body.gravity = [0.0; 3];
    let qdd = [0.0, 1.0, 1.0];
    let both = body.rnea(&qdd, None)[0];
    let only_1 = body.rnea(&[0.0, 1.0, 0.0], None)[0];
    let only_2 = body.rnea(&[0.0, 0.0, 1.0], None)[0];
    assert!(
        (both - (only_1 + only_2)).abs() < 1e-12,
        "{both} vs {only_1}+{only_2}"
    );
    assert!(
        only_1.abs() > 1e-6 && only_2.abs() > 1e-6,
        "both children must load the parent"
    );
}

// ── ABA (F0415) ──────────────────────────────────────────────────────────────

#[test]
fn aba_inverts_rnea_on_a_random_three_link_chain() {
    for seed in 0..10 {
        let mut rng = StdRng::seed_from_u64(80 + seed);
        let mut body = random_tree(&mut rng, 3, false);
        let qdd = random_vector(&mut rng, 3, 2.0);
        let tau = body.rnea(&qdd, None);
        let recovered = body.aba(&tau);
        let err = max_abs_diff(&qdd, &recovered);
        assert!(err < 1e-9, "seed {seed}: aba(rnea(qdd)) differs by {err}");
        // The accelerations are stored on the links too.
        for (link, expected) in body.links.iter().zip(&recovered) {
            assert!((link.qdd - expected).abs() < 1e-15);
        }
    }
}

#[test]
fn aba_inverts_rnea_on_branched_trees_with_external_wrenches() {
    for seed in 0..20 {
        let mut rng = StdRng::seed_from_u64(100 + seed);
        let n = 3 + (seed % 6) as usize;
        let mut body = random_tree(&mut rng, n, true);
        let qdd = random_vector(&mut rng, n, 2.0);
        let tau = body.rnea(&qdd, None);
        let recovered = body.aba(&tau);
        let err = max_abs_diff(&qdd, &recovered);
        assert!(
            err < 1e-9,
            "seed {seed} ({n} links): aba(rnea(qdd)) differs by {err}"
        );
    }
}

#[test]
fn aba_equals_forward_dynamics_assembled_from_h_and_c() {
    for seed in 0..8 {
        let mut rng = StdRng::seed_from_u64(130 + seed);
        let n = 4 + (seed % 3) as usize;
        let mut body = random_tree(&mut rng, n, seed.is_multiple_of(2));
        let tau = random_vector(&mut rng, n, 3.0);
        let h = body.crba();
        let bias = body.rnea(&vec![0.0; n], None);
        let rhs: Vec<f64> = tau.iter().zip(&bias).map(|(t, c)| t - c).collect();
        let expected = spd_solve(&h, &rhs);
        let qdd = body.aba(&tau);
        let err = max_abs_diff(&qdd, &expected);
        assert!(
            err < 1e-8,
            "seed {seed}: ABA differs from H^-1 (tau - C) by {err}"
        );
    }
}

#[test]
fn aba_responds_to_velocity_and_external_wrench() {
    // Same state, but with and without velocity / wrench: the accelerations
    // must differ (the old recursion ignored both).
    let mut rng = StdRng::seed_from_u64(150);
    let mut moving = random_tree(&mut rng, 4, false);
    let tau = vec![0.0; 4];
    let with_everything = moving.aba(&tau);

    let mut at_rest = moving.clone();
    at_rest.set_qd(&[0.0; 4]);
    let without_velocity = at_rest.aba(&tau);
    assert!(max_abs_diff(&with_everything, &without_velocity) > 1e-3);

    let mut unloaded = moving.clone();
    clear_external_wrenches(&mut unloaded);
    let without_wrench = unloaded.aba(&tau);
    assert!(max_abs_diff(&with_everything, &without_wrench) > 1e-3);
}

#[test]
fn aba_propagates_articulated_inertia_to_the_parent() {
    // A free-swinging two-link planar arm released from rest at q = 0
    // (horizontal): the upper joint's acceleration depends on the lower link.
    let arm = PlanarArm::sample();
    let mut body = arm.build([0.0, 0.0], [0.0; 2]);
    let qdd = body.aba(&[0.0, 0.0]);
    let h = arm.inertia_matrix(0.0);
    let g = arm.torques([0.0, 0.0], [0.0; 2], [0.0; 2]);
    // Solve H qdd = -G for the 2x2 case.
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let expected = [
        (-g[0] * h[1][1] + g[1] * h[0][1]) / det,
        (-g[1] * h[0][0] + g[0] * h[1][0]) / det,
    ];
    assert!(
        (qdd[0] - expected[0]).abs() < 1e-10 && (qdd[1] - expected[1]).abs() < 1e-10,
        "qdd {qdd:?} vs {expected:?}"
    );
    // The articulated inertia at the upper joint is the Schur complement
    // H11 - H12^2 / H22: the lower link's freedom is projected out, but its
    // inertia is part of what the upper joint has to move.
    let schur = h[0][0] - h[0][1] * h[0][1] / h[1][1];
    assert!(
        (body.links[0].d_aba - schur).abs() < 1e-10,
        "d_aba={} expected {schur}",
        body.links[0].d_aba
    );
    assert!(schur > arm.i1 + arm.m1 * arm.lc1 * arm.lc1);
}

#[test]
fn symplectic_euler_with_aba_conserves_energy() {
    let arm = PlanarArm::sample();
    let mut body = arm.build([0.5, -0.3], [0.2, 0.1]);
    let energy = |b: &ArticulatedBody| -> f64 {
        let mut posed = b.clone();
        let h = posed.crba();
        let qd = posed.get_qd();
        let h_qd = mat_vec(&h, &qd);
        0.5 * qd.iter().zip(&h_qd).map(|(a, b)| a * b).sum::<f64>() + potential_energy(b)
    };
    let e0 = energy(&body);
    for _ in 0..5000 {
        symplectic_euler_integrate(&mut body, 1e-4);
    }
    let e1 = energy(&body);
    let drift = (e1 - e0).abs();
    assert!(drift < 2e-3, "energy drifted by {drift} from {e0}");
    // The system actually moved.
    assert!((body.links[0].q - 0.5).abs() > 1e-2);
}

// ── fixed joints and locked subtrees ─────────────────────────────────────────

fn fixed_joint() -> JointAxis {
    JointAxis {
        joint_type: JointType::Fixed,
        axis: [0.0; 3],
        s_matrix: SpatialVec::zero(),
    }
}

#[test]
fn a_fixed_joint_is_welded_to_its_parent() {
    // A heavy block welded to the world carries a pendulum: the pendulum
    // behaves exactly as if it were attached to the world directly.
    let reference = offset_pendulum(SpatialTransform::identity());
    let mut reference_body = reference.clone();
    let expected_qdd = reference_body.aba(&[0.0])[0];

    let block = Link::new(
        "block",
        -1,
        RigidBodyInertia::new(
            25.0,
            [0.1, 0.2, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        ),
        fixed_joint(),
        SpatialTransform::identity(),
    );
    let mut pendulum_link = reference.links[0].clone();
    pendulum_link.parent = 0;
    let mut chain = ArticulatedBody::new(reference.gravity);
    chain.add_link(block);
    chain.add_link(pendulum_link);

    let qdd = chain.aba(&[0.0, 0.0]);
    assert_eq!(qdd[0], 0.0, "a fixed joint never accelerates");
    assert!(
        (qdd[1] - expected_qdd).abs() < 1e-10,
        "{} vs {expected_qdd}",
        qdd[1]
    );

    // Inverse dynamics agrees: no torque is attributed to the weld.
    let tau = chain.rnea(&[0.0, expected_qdd], None);
    assert!(tau[0].abs() < 1e-12 && tau[1].abs() < 1e-9, "tau={tau:?}");

    // Hybrid dynamics leaves the fixed joint alone as well.
    chain.links[0].tau = 0.0;
    chain.links[1].tau = 0.0;
    let modes = [HybridMode::ForwardDynamics; 2];
    let hybrid = hybrid_dynamics(&mut chain, &modes).expect("the pendulum joint has inertia");
    assert_eq!(hybrid[0], 0.0);
    assert!((hybrid[1] - expected_qdd).abs() < 1e-9);
}

// ── hybrid dynamics (F0420) ──────────────────────────────────────────────────

fn tree_with_inputs(seed: u64, n: usize) -> (ArticulatedBody, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut body = random_tree(&mut rng, n, seed.is_multiple_of(2));
    for link in &mut body.links {
        link.tau = rng.random_range(-3.0..3.0);
        link.qdd = rng.random_range(-3.0..3.0);
    }
    (body, rng)
}

#[test]
fn hybrid_all_forward_equals_aba() {
    for seed in 0..6 {
        let n = 3 + (seed % 4) as usize;
        let (mut body, _) = tree_with_inputs(160 + seed, n);
        let tau: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
        let mut reference = body.clone();
        let expected = reference.aba(&tau);
        let modes = vec![HybridMode::ForwardDynamics; n];
        let result = hybrid_dynamics(&mut body, &modes).expect("positive definite");
        let err = max_abs_diff(&result, &expected);
        assert!(err < 1e-8, "seed {seed}: hybrid vs aba differ by {err}");
        for (link, value) in body.links.iter().zip(&result) {
            assert!((link.qdd - value).abs() < 1e-15, "qdd is written back");
        }
    }
}

#[test]
fn hybrid_all_inverse_equals_rnea() {
    for seed in 0..6 {
        let n = 3 + (seed % 4) as usize;
        let (mut body, _) = tree_with_inputs(170 + seed, n);
        let qdd: Vec<f64> = body.links.iter().map(|l| l.qdd).collect();
        let mut reference = body.clone();
        let expected = reference.rnea(&qdd, None);
        let modes = vec![HybridMode::InverseDynamics; n];
        let result = hybrid_dynamics(&mut body, &modes).expect("no forward joints to solve");
        let err = max_abs_diff(&result, &expected);
        assert!(err < 1e-9, "seed {seed}: hybrid vs rnea differ by {err}");
        for (link, value) in body.links.iter().zip(&result) {
            assert!((link.tau - value).abs() < 1e-15, "tau is written back");
        }
    }
}

#[test]
fn hybrid_mixed_modes_satisfy_the_equation_of_motion() {
    for seed in 0..10 {
        let n = 4 + (seed % 4) as usize;
        let (mut body, mut rng) = tree_with_inputs(180 + seed, n);
        // Random assignment, but at least one joint of each kind.
        let mut modes: Vec<HybridMode> = (0..n)
            .map(|_| {
                if rng.random_range(0.0..1.0) < 0.5 {
                    HybridMode::ForwardDynamics
                } else {
                    HybridMode::InverseDynamics
                }
            })
            .collect();
        modes[0] = HybridMode::ForwardDynamics;
        modes[n - 1] = HybridMode::InverseDynamics;

        let tau_in: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
        let qdd_in: Vec<f64> = body.links.iter().map(|l| l.qdd).collect();
        let mut reference = body.clone();
        let h = reference.crba();
        let bias = reference.rnea(&vec![0.0; n], None);

        let result = hybrid_dynamics(&mut body, &modes).expect("positive definite");

        // Reassemble the complete (tau, qdd) pair from inputs and outputs.
        let mut tau_full = tau_in.clone();
        let mut qdd_full = qdd_in.clone();
        for i in 0..n {
            match modes[i] {
                HybridMode::ForwardDynamics => qdd_full[i] = result[i],
                HybridMode::InverseDynamics => tau_full[i] = result[i],
            }
        }
        // Prescribed values are untouched ...
        for i in 0..n {
            match modes[i] {
                HybridMode::ForwardDynamics => assert_eq!(body.links[i].tau, tau_in[i]),
                HybridMode::InverseDynamics => assert_eq!(body.links[i].qdd, qdd_in[i]),
            }
        }
        // ... and tau = H qdd + C holds for every joint.
        let h_qdd = mat_vec(&h, &qdd_full);
        let residual: Vec<f64> = (0..n).map(|i| h_qdd[i] + bias[i] - tau_full[i]).collect();
        let worst = residual.iter().fold(0.0_f64, |m, r| m.max(r.abs()));
        assert!(worst < 1e-8, "seed {seed}: H qdd + C - tau = {worst}");
        // The links hold the complete solution.
        let link_qdd: Vec<f64> = body.links.iter().map(|l| l.qdd).collect();
        let link_tau: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
        assert!(max_abs_diff(&link_qdd, &qdd_full) < 1e-12);
        assert!(max_abs_diff(&link_tau, &tau_full) < 1e-12);
        // RNEA on the full qdd reproduces the full tau.
        let check = reference.rnea(&qdd_full, None);
        assert!(max_abs_diff(&check, &tau_full) < 1e-8);
    }
}

#[test]
fn hybrid_defaults_missing_modes_to_forward_dynamics() {
    let (mut body, _) = tree_with_inputs(190, 3);
    let tau: Vec<f64> = body.links.iter().map(|l| l.tau).collect();
    let mut reference = body.clone();
    let expected = reference.aba(&tau);
    let result = hybrid_dynamics(&mut body, &[]).expect("positive definite");
    assert!(max_abs_diff(&result, &expected) < 1e-8);
}

fn coaxial_pair(first_is_massless: bool) -> ArticulatedBody {
    let axis = JointAxis::revolute([0.0, 0.0, 1.0]);
    let heavy = RigidBodyInertia::new(
        1.0,
        [0.4, 0.0, 0.0],
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
    );
    let first_inertia = if first_is_massless {
        RigidBodyInertia::point_mass(0.0)
    } else {
        heavy
    };
    let mut body = ArticulatedBody::new([0.0, -9.81, 0.0]);
    body.add_link(Link::new(
        "first",
        -1,
        first_inertia,
        axis,
        SpatialTransform::identity(),
    ));
    body.add_link(Link::new(
        "second",
        0,
        heavy,
        axis,
        SpatialTransform::identity(),
    ));
    body
}

#[test]
fn hybrid_reports_a_singular_forward_block() {
    // Both joints turn the same body about the same axis: H = I [[1,1],[1,1]]
    // is singular, so two forward joints are under-determined.
    let mut body = coaxial_pair(true);
    let modes = [HybridMode::ForwardDynamics; 2];
    let err = hybrid_dynamics(&mut body, &modes).expect_err("H_FF is singular");
    assert!(matches!(err, Error::SingularMatrix { .. }), "got {err:?}");

    // Prescribing the first joint's acceleration makes the problem well posed.
    let mut body = coaxial_pair(true);
    body.links[0].qdd = 0.5;
    body.links[1].tau = 0.0;
    let modes = [HybridMode::InverseDynamics, HybridMode::ForwardDynamics];
    let result = hybrid_dynamics(&mut body, &modes).expect("the forward block is regular");
    assert!(result.iter().all(|v| v.is_finite()));
    // Both joints drive one body, so the inverse joint's torque balances it.
    let mut reference = body.clone();
    let tau = reference.rnea(&[0.5, result[1]], None);
    assert!(
        (tau[1]).abs() < 1e-9,
        "the forward joint carries no torque: {tau:?}"
    );
    assert!((tau[0] - result[0]).abs() < 1e-9);
}

// ── operational space and redundancy (F0421) ─────────────────────────────────

fn random_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|_| random_vector(rng, cols, 1.0)).collect()
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let inner = b.len();
    let cols = b.first().map_or(0, Vec::len);
    a.iter()
        .map(|row| {
            (0..cols)
                .map(|c| (0..inner).map(|k| row[k] * b[k][c]).sum())
                .collect()
        })
        .collect()
}

fn max_deviation_from_identity(m: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0_f64;
    for (i, row) in m.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((v - target).abs());
        }
    }
    worst
}

#[test]
fn operational_space_inertia_inverts_for_a_three_dimensional_task() {
    let j = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let lambda = operational_space_inertia(&j, &[0.5, 0.5, 0.5]);
    for (i, row) in lambda.iter().enumerate() {
        for (k, v) in row.iter().enumerate() {
            let expected = if i == k { 2.0 } else { 0.0 };
            assert!((v - expected).abs() < 1e-12, "lambda[{i}][{k}]={v}");
        }
    }
}

#[test]
fn operational_space_inertia_is_the_inverse_of_j_hinv_jt_for_any_task_dimension() {
    let mut rng = StdRng::seed_from_u64(200);
    for (m, n) in [(1, 4), (2, 5), (3, 6), (4, 7), (6, 8)] {
        let j = random_matrix(&mut rng, m, n);
        let h_inv: Vec<f64> = (0..n).map(|_| rng.random_range(0.2..3.0)).collect();
        let lambda = operational_space_inertia(&j, &h_inv);
        // M = J H^-1 J^T
        let scaled: Vec<Vec<f64>> = (0..n)
            .map(|l| (0..m).map(|i| h_inv[l] * j[i][l]).collect())
            .collect();
        let m_matrix = matmul(&j, &scaled);
        let product = matmul(&lambda, &m_matrix);
        let dev = max_deviation_from_identity(&product);
        assert!(
            dev < 1e-9,
            "{m}x{n}: Lambda * (J H^-1 J^T) deviates from I by {dev}"
        );
    }
}

#[test]
fn operational_space_inertia_stays_finite_at_a_singularity() {
    // Two identical task rows: J H^-1 J^T = [[1,1],[1,1]] is singular.
    let j = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let lambda = operational_space_inertia(&j, &[1.0, 1.0, 1.0]);
    assert_eq!(lambda.len(), 2);
    for row in &lambda {
        assert!(row.iter().all(|v| v.is_finite()));
    }
    assert!((lambda[0][1] - lambda[1][0]).abs() < 1e-3 * lambda[0][0].abs().max(1.0));
    // The damped inverse is not the (wrong) un-inverted matrix of the old code.
    assert!(
        lambda[0][0] > 10.0,
        "damped inverse of a singular matrix is large"
    );
}

#[test]
fn pseudo_inverse_is_a_right_inverse_for_wide_matrices() {
    let mut rng = StdRng::seed_from_u64(210);
    for (m, n) in [(2, 5), (3, 6), (4, 7), (6, 9)] {
        let j = random_matrix(&mut rng, m, n);
        let j_pinv = pseudo_inverse_fat(&j);
        assert_eq!(j_pinv.len(), n);
        assert!(j_pinv.iter().all(|row| row.len() == m));
        let product = matmul(&j, &j_pinv);
        let dev = max_deviation_from_identity(&product);
        assert!(dev < 1e-10, "{m}x{n}: J J+ deviates from I by {dev}");

        // N = I - J+ J projects onto the null space: J N = 0, N N = N.
        let null = null_space_projector(&j_pinv, &j);
        let j_null = matmul(&j, &null);
        let worst = j_null.iter().flatten().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(worst < 1e-10, "{m}x{n}: J N = {worst}");
        let null_sq = matmul(&null, &null);
        let idem = null
            .iter()
            .flatten()
            .zip(null_sq.iter().flatten())
            .fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(idem < 1e-10, "{m}x{n}: N is not idempotent ({idem})");
    }
}

#[test]
fn pseudo_inverse_of_a_rank_deficient_matrix_is_bounded() {
    // Rank 1: both rows are parallel.
    let j = vec![vec![1.0, 2.0, 0.0, 1.0], vec![2.0, 4.0, 0.0, 2.0]];
    let j_pinv = pseudo_inverse_fat(&j);
    assert_eq!(j_pinv.len(), 4);
    assert!(j_pinv.iter().flatten().all(|v| v.is_finite()));
    // J J+ J = J holds for the damped inverse up to the damping.
    let back = matmul(&matmul(&j, &j_pinv), &j);
    for (row_back, row) in back.iter().zip(&j) {
        for (b, a) in row_back.iter().zip(row) {
            assert!((b - a).abs() < 1e-6, "J J+ J = {b}, J = {a}");
        }
    }
}

#[test]
fn dynamically_consistent_inverse_satisfies_j_jbar_identity() {
    let mut rng = StdRng::seed_from_u64(220);
    let (m, n) = (3, 6);
    let j = random_matrix(&mut rng, m, n);
    let h_inv: Vec<f64> = (0..n).map(|_| rng.random_range(0.2..3.0)).collect();
    let lambda = operational_space_inertia(&j, &h_inv);
    let j_bar = dynamically_consistent_jacobian_pinv(&j, &h_inv, &lambda);
    assert_eq!(j_bar.len(), n);
    let product = matmul(&j, &j_bar);
    let dev = max_deviation_from_identity(&product);
    assert!(dev < 1e-9, "J Jbar deviates from I by {dev}");
}

#[test]
fn invert_dense_handles_regular_singular_and_malformed_input() {
    let a = vec![
        vec![4.0, 7.0, 2.0],
        vec![3.0, 6.0, 1.0],
        vec![2.0, 5.0, 3.0],
    ];
    let inv = invert_dense(&a).expect("regular matrix");
    let dev = max_deviation_from_identity(&matmul(&a, &inv));
    assert!(dev < 1e-12, "A A^-1 deviates from I by {dev}");

    // A zero leading entry needs a row exchange.
    let swapped = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    assert_eq!(invert_dense(&swapped), Some(swapped.clone()));

    assert!(
        invert_dense(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none(),
        "singular"
    );
    assert!(invert_dense(&[vec![0.0]]).is_none());
    assert!(invert_dense(&[vec![1.0, 2.0]]).is_none(), "not square");
    assert!(invert_dense(&[vec![f64::NAN]]).is_none());
    assert_eq!(invert_dense(&[]), Some(Vec::new()));

    // A large-scale matrix is judged relative to its size.
    let big = vec![vec![1.0e9, 2.0e9], vec![3.0e9, 5.0e9]];
    let big_inv = invert_dense(&big).expect("regular at any scale");
    assert!(max_deviation_from_identity(&matmul(&big, &big_inv)) < 1e-9);
}

#[test]
fn damped_inverse_regularises_singular_matrices_of_any_scale() {
    for scale in [1.0, 1.0e-3, 1.0e6, 1.0e10] {
        let singular = vec![
            vec![scale, scale, 0.0],
            vec![scale, scale, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        assert!(invert_dense(&singular).is_none());
        let inv = invert_or_damped(&singular);
        assert!(inv.iter().flatten().all(|v| v.is_finite()), "scale {scale}");
        // The regular part (the 3rd diagonal entry is a null direction, the
        // first two share one) is inverted on its range: M M^+ M ~ M.
        let back = matmul(&matmul(&singular, &inv), &singular);
        for (row_back, row) in back.iter().zip(&singular) {
            for (b, a) in row_back.iter().zip(row) {
                assert!((b - a).abs() <= 1e-6 * scale, "scale {scale}: {b} vs {a}");
            }
        }
    }
    let nan = invert_or_damped(&[vec![f64::NAN, 0.0], vec![0.0, 1.0]]);
    assert!(
        nan.iter().flatten().all(|v| v.is_nan()),
        "NaN input must not look valid"
    );
}
