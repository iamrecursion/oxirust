//! Regression tests for the spatial-algebra multibody dynamics (RNEA, CRBA,
//! ABA) behind `MultibodySystem`, `MultibodyChain` and `FeatherstoneABA`.

use super::functions::rot_from_axis_angle;
use super::types::{
    Body, ChainElement, FeatherstoneABA, FeatherstoneAba, Joint, JointDof, JointSpaceInertia,
    JointType, MultiBodyLink, MultibodyChain, MultibodySystem, RecursiveNewtonEuler, RigidLink,
};
use std::f64::consts::PI;

const G: f64 = 9.81;

/// Deterministic xorshift64* generator for reproducible random models.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
    fn vec3(&mut self, lo: f64, hi: f64) -> [f64; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
    fn unit(&mut self) -> [f64; 3] {
        loop {
            let v = self.vec3(-1.0, 1.0);
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n > 0.2 {
                return [v[0] / n, v[1] / n, v[2] / n];
            }
        }
    }
}

fn assert_vec_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

fn mat_vec(h: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| h[i * n + j] * x[j]).sum())
        .collect()
}

fn link(mass: f64, inertia: [f64; 3], com: [f64; 3], joint_offset: [f64; 3]) -> RigidLink {
    RigidLink {
        name: "link".to_string(),
        mass,
        inertia,
        com_offset: com,
        joint_offset,
    }
}

// ── FeatherstoneABA (Body-based) ────────────────────────────────────────────

fn body_pendulum(m: f64, izz: f64, l: f64) -> FeatherstoneABA {
    let base = Body::new(0, 1.0, 1.0, 1.0, 1.0);
    let bob = Body::new(1, m, 0.3, 0.2, izz);
    FeatherstoneABA::new(vec![base, bob], vec![[0.0, 0.0, 1.0]], [0.0, -G, 0.0])
        .with_geometry(vec![[0.0; 3]], vec![[0.0, -l, 0.0]])
        .expect("geometry")
}

#[test]
fn body_aba_pendulum_matches_analytic() {
    let (m, izz, l) = (2.0, 0.05, 0.7);
    let mut aba = body_pendulum(m, izz, l);
    for &theta in &[0.0, 0.3, -1.1, 2.5] {
        for &omega in &[0.0, 1.3] {
            aba.q[0] = theta;
            let qdd = aba.forward_dynamics(&[omega]).expect("ABA");
            let expected = -m * G * l * theta.sin() / (izz + m * l * l);
            assert!(
                (qdd[0] - expected).abs() < 1e-9,
                "theta={theta} omega={omega}: {} vs {expected}",
                qdd[0]
            );
        }
    }
}

#[test]
fn body_aba_gravity_torque_is_lever_arm_not_force() {
    // A body whose centre of mass sits on the joint axis feels no gravity
    // torque, whatever its mass: q̈ = τ / Izz exactly.
    let mut aba = body_pendulum(5.0, 0.2, 0.0);
    aba.q[0] = 0.9;
    aba.applied_torques[0] = 0.4;
    let qdd = aba.forward_dynamics(&[0.0]).expect("ABA");
    assert!((qdd[0] - 0.4 / 0.2).abs() < 1e-12, "qdd = {}", qdd[0]);
}

#[test]
fn body_aba_step_updates_pose_from_kinematics() {
    let (m, izz, l) = (1.5, 0.02, 0.5);
    let mut aba = body_pendulum(m, izz, l);
    aba.q[0] = 0.6;
    let mut qd = vec![0.0];
    let dt = 1e-3;
    aba.step(dt, &mut qd).expect("step");
    let qdd0 = -m * G * l * 0.6_f64.sin() / (izz + m * l * l);
    assert!((qd[0] - qdd0 * dt).abs() < 1e-12);
    assert!((aba.q[0] - (0.6 + qd[0] * dt)).abs() < 1e-15);
    let q = aba.q[0];
    let body = &aba.bodies[1];
    let expected_pos = [l * q.sin(), -l * q.cos(), 0.0];
    assert_vec_close(&body.position, &expected_pos, 1e-12, "CoM position");
    let expected_quat = [(q / 2.0).cos(), 0.0, 0.0, (q / 2.0).sin()];
    assert_vec_close(&body.orientation, &expected_quat, 1e-12, "orientation");
    assert_vec_close(&body.angular_velocity, &[0.0, 0.0, qd[0]], 1e-12, "omega");
    let expected_vel = [l * q.cos() * qd[0], l * q.sin() * qd[0], 0.0];
    assert_vec_close(&body.velocity, &expected_vel, 1e-12, "CoM velocity");
}

#[test]
fn body_aba_double_pendulum_couples_joints() {
    // Point masses at the link tips: classic double pendulum (angles from +x).
    let (m1, m2, l1, l2) = (1.2, 0.8, 0.9, 0.6);
    let base = Body::new(0, 1.0, 1.0, 1.0, 1.0);
    let b1 = Body::new(1, m1, 0.0, 0.0, 0.0);
    let b2 = Body::new(2, m2, 0.0, 0.0, 0.0);
    let mut aba = FeatherstoneABA::new(
        vec![base, b1, b2],
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [0.0, -G, 0.0],
    )
    .with_geometry(
        vec![[0.0; 3], [l1, 0.0, 0.0]],
        vec![[l1, 0.0, 0.0], [l2, 0.0, 0.0]],
    )
    .expect("geometry");
    let (q1, q2, w1, w2) = (0.4, -0.7, 0.9, -1.4);
    aba.q = vec![q1, q2];
    aba.applied_torques = vec![0.3, -0.2];
    let qdd = aba.forward_dynamics(&[w1, w2]).expect("ABA");
    let (h, c, g) = double_pendulum_terms(m1, m2, l1, l2, [q1, q2], [w1, w2]);
    let rhs = [0.3 - c[0] - g[0], -0.2 - c[1] - g[1]];
    let det = h[0] * h[3] - h[1] * h[2];
    let expected = [
        (h[3] * rhs[0] - h[1] * rhs[1]) / det,
        (h[0] * rhs[1] - h[2] * rhs[0]) / det,
    ];
    assert_vec_close(&qdd, &expected, 1e-9, "double pendulum qdd");
}

/// Closed-form `H` (row-major 2×2), `C q̇` and `G` of a planar point-mass
/// double pendulum with gravity along −y and angles measured from +x.
fn double_pendulum_terms(
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
    q: [f64; 2],
    qd: [f64; 2],
) -> ([f64; 4], [f64; 2], [f64; 2]) {
    let c2 = q[1].cos();
    let s2 = q[1].sin();
    let h11 = m1 * l1 * l1 + m2 * (l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * c2);
    let h12 = m2 * (l2 * l2 + l1 * l2 * c2);
    let h22 = m2 * l2 * l2;
    let c = [
        -m2 * l1 * l2 * s2 * (2.0 * qd[0] * qd[1] + qd[1] * qd[1]),
        m2 * l1 * l2 * s2 * qd[0] * qd[0],
    ];
    let g = [
        G * ((m1 + m2) * l1 * q[0].cos() + m2 * l2 * (q[0] + q[1]).cos()),
        G * m2 * l2 * (q[0] + q[1]).cos(),
    ];
    ([h11, h12, h12, h22], c, g)
}

// ── MultibodyChain algorithms ───────────────────────────────────────────────

fn chain_double_pendulum(m1: f64, m2: f64, l1: f64, l2: f64) -> MultibodyChain {
    let mut chain = MultibodyChain::new();
    let z = JointDof::Revolute {
        axis: [0.0, 0.0, 1.0],
        limits: (-PI, PI),
    };
    chain.add_element(link(m1, [0.0; 3], [l1, 0.0, 0.0], [0.0; 3]), z.clone());
    chain.add_element(link(m2, [0.0; 3], [l2, 0.0, 0.0], [l1, 0.0, 0.0]), z);
    chain
}

#[test]
fn chain_pendulum_matches_analytic() {
    let (m, ic, l) = (3.0, 0.04, 0.45);
    let mut chain = MultibodyChain::new();
    chain.add_element(
        link(m, [0.1, 0.1, ic], [0.0, -l, 0.0], [0.0; 3]),
        JointDof::Revolute {
            axis: [0.0, 0.0, 1.0],
            limits: (-PI, PI),
        },
    );
    for &theta in &[0.2, 1.0, -2.0] {
        chain.q[0] = theta;
        let mut aba = FeatherstoneAba::new(chain.clone(), [0.0, -G, 0.0]);
        let qdd = aba.forward_dynamics(&[0.0]).expect("ABA");
        let expected = -m * G * l * theta.sin() / (ic + m * l * l);
        assert!((qdd[0] - expected).abs() < 1e-9, "{} vs {expected}", qdd[0]);
        assert_vec_close(&aba.chain.qdd, &qdd, 0.0, "stored qdd");
        let h = JointSpaceInertia::new(chain.clone())
            .compute()
            .expect("CRBA");
        assert!((h[0] - (ic + m * l * l)).abs() < 1e-12);
        let tau = RecursiveNewtonEuler::new(chain.clone(), [0.0, -G, 0.0])
            .inverse_dynamics()
            .expect("RNEA");
        assert!((tau[0] - m * G * l * theta.sin()).abs() < 1e-12);
    }
}

#[test]
fn chain_double_pendulum_matches_closed_form() {
    let (m1, m2, l1, l2) = (1.1, 0.7, 0.8, 0.5);
    let mut chain = chain_double_pendulum(m1, m2, l1, l2);
    let q = [0.6, 1.1];
    let qd = [-0.8, 1.7];
    chain.q = q.to_vec();
    chain.qd = qd.to_vec();
    let (h_ref, c_ref, g_ref) = double_pendulum_terms(m1, m2, l1, l2, q, qd);
    let h = JointSpaceInertia::new(chain.clone())
        .compute()
        .expect("CRBA");
    assert_vec_close(&h, &h_ref, 1e-12, "H");
    // RNEA with q̈ = 0 and gravity = C q̇ + G.
    let bias = RecursiveNewtonEuler::new(chain.clone(), [0.0, -G, 0.0])
        .inverse_dynamics()
        .expect("RNEA");
    let bias_ref = [c_ref[0] + g_ref[0], c_ref[1] + g_ref[1]];
    assert_vec_close(&bias, &bias_ref, 1e-12, "C qd + G");
    // Gravity-free RNEA isolates the Coriolis/centrifugal terms.
    let coriolis = RecursiveNewtonEuler::new(chain.clone(), [0.0; 3])
        .inverse_dynamics()
        .expect("RNEA");
    assert_vec_close(&coriolis, &c_ref, 1e-12, "C qd");
}

fn random_chain(rng: &mut Rng, with_spherical: bool) -> MultibodyChain {
    let mut chain = MultibodyChain::new();
    for k in 0..4 {
        let joint = if with_spherical && k == 1 {
            JointDof::Spherical
        } else if k == 2 {
            JointDof::Prismatic {
                axis: rng.unit(),
                limits: (-1.0, 1.0),
            }
        } else {
            JointDof::Revolute {
                axis: rng.unit(),
                limits: (-PI, PI),
            }
        };
        let l = link(
            rng.range(0.5, 3.0),
            rng.vec3(0.01, 0.2),
            rng.vec3(-0.4, 0.4),
            rng.vec3(-0.5, 0.5),
        );
        chain.add_element(l, joint);
    }
    let n = chain.total_dof();
    chain.q = (0..n).map(|_| rng.range(-1.2, 1.2)).collect();
    chain.qd = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
    chain
}

#[test]
fn chain_aba_crba_rnea_agree_on_random_chains() {
    let gravity = [0.4, -G, 1.3];
    for seed in 1..=8u64 {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xA24B_AED4_963E_E407));
        let chain = random_chain(&mut rng, seed % 2 == 0);
        let n = chain.total_dof();
        let tau: Vec<f64> = (0..n).map(|_| rng.range(-5.0, 5.0)).collect();

        let mut aba = FeatherstoneAba::new(chain.clone(), gravity);
        let qdd_aba = aba.forward_dynamics(&tau).expect("ABA");

        let h = JointSpaceInertia::new(chain.clone())
            .compute()
            .expect("CRBA");
        for i in 0..n {
            for j in 0..n {
                assert!((h[i * n + j] - h[j * n + i]).abs() < 1e-12, "H symmetric");
            }
        }
        let mut biased = chain.clone();
        biased.qdd = vec![0.0; n];
        let bias = RecursiveNewtonEuler::new(biased, gravity)
            .inverse_dynamics()
            .expect("RNEA");
        // ABA q̈ must solve H q̈ = τ − bias.
        let lhs = mat_vec(&h, n, &qdd_aba);
        let rhs: Vec<f64> = tau.iter().zip(bias.iter()).map(|(t, b)| t - b).collect();
        assert_vec_close(&lhs, &rhs, 1e-9, "H qdd_aba vs tau - bias");

        // RNEA(q, q̇, ABA(τ)) == τ.
        let mut accel = chain.clone();
        accel.qdd = qdd_aba.clone();
        let tau_back = RecursiveNewtonEuler::new(accel, gravity)
            .inverse_dynamics()
            .expect("RNEA");
        assert_vec_close(&tau_back, &tau, 1e-9, "RNEA(ABA(tau))");
    }
}

#[test]
fn chain_massless_subtree_is_reported() {
    let mut chain = MultibodyChain::new();
    chain.add_element(
        link(0.0, [0.0; 3], [0.0; 3], [0.0; 3]),
        JointDof::Revolute {
            axis: [0.0, 0.0, 1.0],
            limits: (-PI, PI),
        },
    );
    let mut aba = FeatherstoneAba::new(chain, [0.0, -G, 0.0]);
    assert!(aba.forward_dynamics(&[1.0]).is_err());
}

#[test]
fn chain_length_mismatch_is_reported() {
    let chain = MultibodyChain::default_6dof_arm();
    let mut aba = FeatherstoneAba::new(chain, [0.0, -G, 0.0]);
    assert!(aba.forward_dynamics(&[0.0; 3]).is_err());
}

#[test]
fn chain_element_struct_is_used_in_order() {
    // Elements are dynamically coupled: moving the second joint changes the
    // inertia seen by the first one (H00 depends on q1).
    let mut chain = chain_double_pendulum(1.0, 1.0, 1.0, 1.0);
    let first: &ChainElement = &chain.elements[0];
    assert!((first.link.mass - 1.0).abs() < 1e-15);
    chain.q = vec![0.0, 0.0];
    let h_straight = JointSpaceInertia::new(chain.clone())
        .compute()
        .expect("CRBA");
    chain.q = vec![0.0, PI / 2.0];
    let h_bent = JointSpaceInertia::new(chain).compute().expect("CRBA");
    assert!((h_straight[0] - 5.0).abs() < 1e-12);
    assert!((h_bent[0] - 3.0).abs() < 1e-12);
}

// ── MultibodySystem ─────────────────────────────────────────────────────────

#[test]
fn system_pendulum_mass_matrix_and_gravity() {
    let (m, ic, l) = (2.5, 0.03, 0.6);
    let mut sys = MultibodySystem::new();
    let mut bob = MultiBodyLink::new(m, 0.1, 0.1, ic, [0.0, -l, 0.0]);
    bob.parent_joint = Some(Joint::new_revolute(0, 1, -PI, PI));
    sys.add_link(bob);
    for &q in &[0.0, 0.7, -2.2] {
        sys.positions[0] = q;
        sys.velocities[0] = 0.0;
        let mm = sys.mass_matrix().expect("M");
        assert!((mm[0][0] - (ic + m * l * l)).abs() < 1e-12);
        let gv = sys.gravity_vector().expect("G");
        assert!((gv[0] - m * G * l * q.sin()).abs() < 1e-12);
        let qdd = sys.forward_dynamics(&[0.0]).expect("FD");
        let expected = -m * G * l * q.sin() / (ic + m * l * l);
        assert!((qdd[0] - expected).abs() < 1e-9);
        let fk = sys.forward_kinematics().expect("FK");
        assert_vec_close(&fk[0], &[l * q.sin(), -l * q.cos(), 0.0], 1e-12, "CoM");
    }
}

fn joint_with_frames(joint_type: JointType, parent: usize, child: usize, rng: &mut Rng) -> Joint {
    let mut joint = Joint::new_revolute(parent, child, -PI, PI);
    joint.joint_type = joint_type;
    joint.local_frame_parent = rot_from_axis_angle(rng.unit(), rng.range(-1.0, 1.0));
    joint.local_frame_child = rot_from_axis_angle(rng.unit(), rng.range(-1.0, 1.0));
    joint
}

/// Branched tree with every supported joint type.
fn random_system(rng: &mut Rng) -> MultibodySystem {
    let mut sys = MultibodySystem::new();
    sys.gravity = [0.3, -G, -0.6];
    sys.add_link(
        MultiBodyLink::new(4.0, 0.3, 0.3, 0.3, [0.0; 3]).with_joint_anchor([0.0, 0.2, 0.0]),
    );
    let layout = [
        (JointType::Revolute, 0usize),
        (JointType::Spherical, 1),
        (JointType::Prismatic, 0),
        (JointType::Universal, 3),
        (JointType::Planar, 1),
        (JointType::Cylindrical, 4),
    ];
    for (k, (jt, parent)) in layout.iter().enumerate() {
        let child = k + 1;
        let d = rng.vec3(0.02, 0.2);
        let off = rng.vec3(-0.03, 0.03);
        let mut l = MultiBodyLink::new(rng.range(0.5, 2.5), d[0], d[1], d[2], rng.vec3(-0.3, 0.3))
            .with_joint_anchor(rng.vec3(-0.4, 0.4));
        l.inertia[3] = off[0] * d[0].min(d[1]);
        l.inertia[4] = off[1] * d[0].min(d[2]);
        l.inertia[5] = off[2] * d[1].min(d[2]);
        l.parent_joint = Some(joint_with_frames(*jt, *parent, child, rng));
        sys.add_link(l);
    }
    let n = sys.n_dof();
    sys.positions = (0..n).map(|_| rng.range(-0.8, 0.8)).collect();
    sys.velocities = (0..n).map(|_| rng.range(-1.5, 1.5)).collect();
    sys
}

#[test]
fn system_inverse_of_forward_is_identity_on_random_trees() {
    for seed in 1..=6u64 {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ seed.wrapping_mul(0x9E37_79B9));
        let sys = random_system(&mut rng);
        let n = sys.n_dof();
        assert_eq!(n, 1 + 3 + 1 + 2 + 3 + 2);
        let tau: Vec<f64> = (0..n).map(|_| rng.range(-4.0, 4.0)).collect();
        let qdd = sys.forward_dynamics(&tau).expect("FD");
        let tau_back = sys
            .inverse_dynamics(&sys.positions, &sys.velocities, &qdd)
            .expect("ID");
        assert_vec_close(&tau_back, &tau, 1e-9, "ID(FD(tau))");
        // Decomposition: τ = M q̈ + C q̇ + G.
        let mm = sys.mass_matrix().expect("M");
        let c = sys.coriolis_vector().expect("C");
        let g = sys.gravity_vector().expect("G");
        for i in 0..n {
            let mq: f64 = (0..n).map(|j| mm[i][j] * qdd[j]).sum();
            assert!((mq + c[i] + g[i] - tau[i]).abs() < 1e-9 * (1.0 + tau[i].abs()));
            for (j, row_j) in mm.iter().enumerate() {
                assert!((mm[i][j] - row_j[i]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn system_jacobian_matches_finite_difference() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let sys = random_system(&mut rng);
    let h = 1e-6;
    for link_idx in 0..sys.n_links() {
        let j = sys.jacobian(link_idx).expect("J");
        for col in 0..sys.n_dof() {
            let mut plus = sys.clone();
            let mut minus = sys.clone();
            plus.positions[col] += h;
            minus.positions[col] -= h;
            let fp = plus.forward_kinematics().expect("FK");
            let fm = minus.forward_kinematics().expect("FK");
            for (row, j_row) in j.iter().enumerate() {
                let fd = (fp[link_idx][row] - fm[link_idx][row]) / (2.0 * h);
                assert!(
                    (j_row[col] - fd).abs() < 1e-7,
                    "J[{row}][{col}] of link {link_idx}: {} vs {fd}",
                    j_row[col]
                );
            }
        }
    }
}

fn total_energy(sys: &MultibodySystem) -> f64 {
    let mm = sys.mass_matrix().expect("M");
    let v = &sys.velocities;
    let kinetic: f64 = mm
        .iter()
        .zip(v.iter())
        .map(|(row, vi)| 0.5 * vi * row.iter().zip(v.iter()).map(|(m, vj)| m * vj).sum::<f64>())
        .sum();
    let coms = sys.forward_kinematics().expect("FK");
    let potential: f64 = sys
        .links
        .iter()
        .zip(coms.iter())
        .map(|(l, c)| {
            -l.mass * (sys.gravity[0] * c[0] + sys.gravity[1] * c[1] + sys.gravity[2] * c[2])
        })
        .sum();
    kinetic + potential
}

#[test]
fn system_unforced_motion_conserves_energy() {
    // RK4 on the exact dynamics: energy drift must be at integrator level,
    // which only holds if M, C and G are mutually consistent with the FK.
    let mut rng = Rng(0x0BAD_5EED_1234_4321);
    let mut sys = random_system(&mut rng);
    let n = sys.n_dof();
    let zero = vec![0.0; n];
    let e0 = total_energy(&sys);
    let dt = 1e-3;
    let deriv = |s: &MultibodySystem, q: &[f64], v: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut tmp = s.clone();
        tmp.positions = q.to_vec();
        tmp.velocities = v.to_vec();
        let a = tmp.forward_dynamics(&zero).expect("FD");
        (v.to_vec(), a)
    };
    for _ in 0..500 {
        let q0 = sys.positions.clone();
        let v0 = sys.velocities.clone();
        let (k1q, k1v) = deriv(&sys, &q0, &v0);
        let shift = |base: &[f64], k: &[f64], s: f64| -> Vec<f64> {
            base.iter().zip(k.iter()).map(|(b, d)| b + s * d).collect()
        };
        let (k2q, k2v) = deriv(
            &sys,
            &shift(&q0, &k1q, dt / 2.0),
            &shift(&v0, &k1v, dt / 2.0),
        );
        let (k3q, k3v) = deriv(
            &sys,
            &shift(&q0, &k2q, dt / 2.0),
            &shift(&v0, &k2v, dt / 2.0),
        );
        let (k4q, k4v) = deriv(&sys, &shift(&q0, &k3q, dt), &shift(&v0, &k3v, dt));
        for i in 0..n {
            sys.positions[i] = q0[i] + dt / 6.0 * (k1q[i] + 2.0 * k2q[i] + 2.0 * k3q[i] + k4q[i]);
            sys.velocities[i] = v0[i] + dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
        }
    }
    let e1 = total_energy(&sys);
    assert!(
        (e1 - e0).abs() < 1e-6 * (1.0 + e0.abs()),
        "energy drift {e0} -> {e1}"
    );
}

#[test]
fn system_integrate_euler_is_semi_implicit() {
    let (m, ic, l) = (1.0, 0.01, 0.4);
    let mut sys = MultibodySystem::new();
    let mut bob = MultiBodyLink::new(m, 0.1, 0.1, ic, [0.0, -l, 0.0]);
    bob.parent_joint = Some(Joint::new_revolute(0, 1, -PI, PI));
    sys.add_link(bob);
    sys.positions[0] = 0.5;
    let dt = 0.01;
    let qdd = -m * G * l * 0.5_f64.sin() / (ic + m * l * l);
    sys.integrate_euler(&[0.0], dt).expect("integrate");
    assert!((sys.velocities[0] - qdd * dt).abs() < 1e-12);
    assert!((sys.positions[0] - (0.5 + qdd * dt * dt)).abs() < 1e-12);
    assert!((sys.accelerations[0] - qdd).abs() < 1e-12);
}

#[test]
fn system_rejects_wrong_torque_length() {
    let mut sys = MultibodySystem::new();
    let mut bob = MultiBodyLink::new(1.0, 0.1, 0.1, 0.1, [0.0, -0.2, 0.0]);
    bob.parent_joint = Some(Joint::new_revolute(0, 1, -PI, PI));
    sys.add_link(bob);
    assert!(sys.forward_dynamics(&[0.0, 1.0]).is_err());
    assert!(sys.jacobian(3).is_err());
}
