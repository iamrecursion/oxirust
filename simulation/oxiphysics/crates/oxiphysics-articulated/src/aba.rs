// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Articulated Body Algorithm (ABA) for forward dynamics.
//!
//! Given joint positions `q`, velocities `q̇`, and applied torques `τ`,
//! computes joint accelerations `q̈` that would result from those inputs.
//!
//! Reference: Featherstone 2008 "Rigid Body Dynamics Algorithms", Algorithm 7.4.
//!
//! # Algorithm overview
//!
//! **Pass 1 (outward):** Compute velocity terms and transform bodies.
//! For each body `i` (root to leaves):
//! - Compute `X_J`, compose full transform `X = X_T ∘ X_J`.
//! - Propagate velocity: `v[i] = X · v[parent] + S · q̇_i`.
//! - Compute bias acceleration: `c[i] = v[i] × (S · q̇_i)  +  c_J`.
//!
//! **Pass 2 (inward):** Compute articulated inertias and bias forces.
//! For each body `i` (leaves to root):
//! - Initialise articulated inertia: `I_A[i] = I[i]`.
//! - For each child `j`: back-project child's articulated inertia to parent.
//! - Compute Articulated inertia Schur complement update:
//!   ```text
//!   U = I_A[child] · S
//!   D = S^T · U     (scalar DOF-×-DOF matrix, scalar for 1-DOF joints)
//!   I_A[parent] += X^* · (I_A[child] − U · D^{−1} · U^T) · X^{*T}
//!   p_A[parent] += X^* · (p_A[child] + I_A[child] · c_child + U · D^{−1} · (τ_child − S^T · p_A[child]))
//!   ```
//!
//! **Pass 3 (outward):** Compute accelerations.
//! For each body `i` (root to leaves):
//! - `a_hat = X · a[parent] + c[i]`
//! - `q̈_i = D^{−1} · (τ_i − S^T · (p_A[i] + I_A[i] · a_hat))`
//! - `a[i] = a_hat + S · q̈_i`

use crate::{
    error::ArticulatedError,
    model::ArticulatedModel,
    spatial::{SpatialInertia6x6, SpatialTransform, SpatialVec},
};

// ─── Internal per-body workspace ────────────────────────────────────────────

/// Maximum number of DoF of a single joint (a free-floating joint has 6).
const MAX_JOINT_DOF: usize = 6;

/// Relative pivot tolerance used to declare a joint-space articulated inertia
/// `D = Sᵀ·I_A·S` singular, measured against the scale `max|I_A|·max|S_k|²`.
const SINGULAR_REL_TOL: f64 = 1e-12;

/// Per-body workspace for ABA intermediate values.
///
/// Fixed-size arrays only: filling it never allocates.
#[derive(Debug, Clone, Copy)]
struct AbaBody {
    /// Full joint-to-parent transform (X_T ∘ X_J).
    x_full: SpatialTransform,
    /// Motion-subspace columns `S` at the current configuration (first `dof`
    /// entries are valid). Evaluated once in pass 1 and reused by passes 2-3.
    s: [SpatialVec; MAX_JOINT_DOF],
    /// Number of DoF of this body's joint.
    dof: usize,
    /// Body spatial velocity (in body frame).
    v: SpatialVec,
    /// Bias acceleration `c[i] = v × v_J + c_J` (no parent contribution yet).
    c: SpatialVec,
    /// Articulated inertia (6×6, starts as body's own inertia, then updated).
    i_a: SpatialInertia6x6,
    /// Articulated bias force (6-vector, starts as v ×* (I·v), then updated).
    p_a: SpatialVec,
    /// `U = I_A · S` (first `dof` columns valid).
    u: [SpatialVec; MAX_JOINT_DOF],
    /// `D⁻¹` with `D = Sᵀ·U`, row-major `dof×dof` in the first `dof²` slots.
    d_inv: [f64; MAX_JOINT_DOF * MAX_JOINT_DOF],
    /// `u_i = τ_i − Sᵀ · p_A` (bias-corrected torque, first `dof` valid).
    tau_bias: [f64; MAX_JOINT_DOF],
    /// Body acceleration (accumulated in pass 3).
    a: SpatialVec,
}

impl AbaBody {
    const EMPTY: Self = Self {
        x_full: SpatialTransform::IDENTITY,
        s: [SpatialVec::ZERO; MAX_JOINT_DOF],
        dof: 0,
        v: SpatialVec::ZERO,
        c: SpatialVec::ZERO,
        i_a: SpatialInertia6x6 {
            data: [[0.0; 6]; 6],
        },
        p_a: SpatialVec::ZERO,
        u: [SpatialVec::ZERO; MAX_JOINT_DOF],
        d_inv: [0.0; MAX_JOINT_DOF * MAX_JOINT_DOF],
        tau_bias: [0.0; MAX_JOINT_DOF],
        a: SpatialVec::ZERO,
    };
}

/// Reusable scratch memory for [`aba_with_workspace`].
///
/// Holding one workspace across simulation steps makes forward dynamics
/// allocation-free: the per-body buffers are sized on first use (or by
/// [`AbaWorkspace::new`]) and only grow if the model gains bodies.
#[derive(Debug, Clone, Default)]
pub struct AbaWorkspace {
    bodies: Vec<AbaBody>,
}

impl AbaWorkspace {
    /// Create a workspace pre-sized for `model`.
    pub fn new(model: &ArticulatedModel) -> Self {
        Self {
            bodies: vec![AbaBody::EMPTY; model.num_bodies()],
        }
    }

    /// Make room for `n` bodies (never shrinks the allocation).
    fn prepare(&mut self, n: usize) {
        if self.bodies.len() < n {
            self.bodies.resize(n, AbaBody::EMPTY);
        }
    }
}

/// Compute joint accelerations via ABA (forward dynamics).
///
/// # Arguments
/// - `model`: Articulated-body model (kinematic tree, topological order).
/// - `q`: Joint configuration; length must equal `model.total_nq()`.
/// - `q_dot`: Joint velocities; length must equal `model.total_dof()`.
/// - `tau`: Applied joint torques; length must equal `model.total_dof()`.
///
/// # Returns
/// Joint accelerations `q̈` (length = `model.total_dof()`).
///
/// # Panics
/// Panics if an input has the wrong length, or if a body's joint-space
/// articulated inertia `D = Sᵀ·I_A·S` is singular (e.g. a massless,
/// inertia-free leaf link on a revolute joint), where `q̈` is undefined. Use
/// [`try_aba`] (or [`aba_with_workspace`]) to receive these as errors.
pub fn aba(model: &ArticulatedModel, q: &[f64], q_dot: &[f64], tau: &[f64]) -> Vec<f64> {
    match try_aba(model, q, q_dot, tau) {
        Ok(q_ddot) => q_ddot,
        Err(err) => panic!("aba: {err}"),
    }
}

/// Fallible ABA forward dynamics: like [`aba`] but reports invalid input and
/// singular joint inertias as errors instead of panicking.
///
/// Allocates a fresh [`AbaWorkspace`]; use [`aba_with_workspace`] in a
/// simulation loop to reuse one.
///
/// # Errors
/// - [`ArticulatedError::DimensionMismatch`] if `q`, `q_dot` or `tau` has the
///   wrong length, or a joint's motion subspace does not have [`Joint::dof`]
///   columns.
/// - [`ArticulatedError::SingularJointInertia`] if a body's joint-space
///   articulated inertia `D = Sᵀ·I_A·S` is singular (pivot below
///   `1e-12·max|I_A|·max|S_k|²`).
///
/// [`Joint::dof`]: crate::joint::Joint::dof
pub fn try_aba(
    model: &ArticulatedModel,
    q: &[f64],
    q_dot: &[f64],
    tau: &[f64],
) -> Result<Vec<f64>, ArticulatedError> {
    let mut ws = AbaWorkspace::new(model);
    let mut q_ddot = vec![0.0f64; model.total_dof()];
    aba_with_workspace(model, &mut ws, q, q_dot, tau, &mut q_ddot)?;
    Ok(q_ddot)
}

/// Allocation-free ABA forward dynamics into a caller-provided buffer.
///
/// Writes `q̈` (length `model.total_dof()`) into `q_ddot`, using `ws` for all
/// intermediate per-body quantities. The motion subspace of each joint is
/// evaluated once per call (via [`Joint::motion_subspace_into`]) and reused by
/// all three passes.
///
/// # Errors
/// - [`ArticulatedError::DimensionMismatch`] if `q` (≠ `total_nq`), `q_dot`,
///   `tau` or `q_ddot` (≠ `total_dof`) has the wrong length, or a joint's
///   motion subspace does not have [`Joint::dof`] columns.
/// - [`ArticulatedError::SingularJointInertia`] if a body's joint-space
///   articulated inertia is singular. `q_ddot` is unspecified on error.
///
/// [`Joint::dof`]: crate::joint::Joint::dof
/// [`Joint::motion_subspace_into`]: crate::joint::Joint::motion_subspace_into
pub fn aba_with_workspace(
    model: &ArticulatedModel,
    ws: &mut AbaWorkspace,
    q: &[f64],
    q_dot: &[f64],
    tau: &[f64],
    q_ddot: &mut [f64],
) -> Result<(), ArticulatedError> {
    let n = model.num_bodies();
    check_len("q", model.total_nq(), q.len())?;
    check_len("q_dot", model.total_dof(), q_dot.len())?;
    check_len("tau", model.total_dof(), tau.len())?;
    check_len("q_ddot", model.total_dof(), q_ddot.len())?;

    ws.prepare(n);
    let wb = &mut ws.bodies[..n];

    let a_gravity = model.gravity;

    // ── Pass 1: Outward — velocities and bias accelerations ──────────────────

    for i in 0..n {
        let body = &model.bodies[i];
        let joint = &model.joints[i];
        let qi = model.q_slice(q, i);
        let q_dot_i = model.v_slice(q_dot, i);

        let x_j = joint.transform(qi);
        let x_full = body.parent_transform.compose(&x_j);

        let dof = joint.dof();
        let mut s = [SpatialVec::ZERO; MAX_JOINT_DOF];
        let cols = joint.motion_subspace_into(qi, &mut s);
        if cols != dof {
            return Err(ArticulatedError::DimensionMismatch {
                what: "joint motion subspace columns",
                expected: dof,
                got: cols,
            });
        }

        // Joint velocity v_J = S · q̇_i
        let v_j = subspace_velocity(&s[..dof], q_dot_i);

        // Coriolis c_J
        let c_j = joint.coriolis(qi, q_dot_i);

        let v_parent = match body.parent_id {
            Some(pid) => wb[pid].v,
            None => SpatialVec::ZERO,
        };

        // Body velocity: v[i] = X * v_parent + v_J
        let v = x_full.apply_velocity(&v_parent) + v_j;

        // Bias force: p_A[i] = v[i] ×* (I[i] * v[i])
        let iv = body.inertia.mul_vec(&v);

        let w = &mut wb[i];
        w.x_full = x_full;
        w.s = s;
        w.dof = dof;
        w.v = v;
        // Bias acceleration: c[i] = v[i] × v_J + c_J
        w.c = v.cross_motion(&v_j) + c_j;
        // Initialise articulated inertia with body's own inertia
        w.i_a = SpatialInertia6x6::from_packed(&body.inertia);
        w.p_a = v.cross_force(&iv);
    }

    // ── Pass 2: Inward — articulated inertias and bias forces ────────────────

    for i in (0..n).rev() {
        let body = &model.bodies[i];
        let tau_i = model.v_slice(tau, i);
        let w = wb[i];
        let dof = w.dof;
        let s = &w.s[..dof];

        // Inertia and bias force handed to the parent (Featherstone eq. 7.42):
        // I_a = I_A − U·D⁻¹·Uᵀ and p_a = p_A + I_a·c + U·D⁻¹·u.
        let (i_a_hat, p_contribution) = if dof > 0 {
            // U_k = I_A[i] · S_k  (for each column k of motion subspace)
            let mut u = [SpatialVec::ZERO; MAX_JOINT_DOF];
            for (uk, sk) in u.iter_mut().zip(s.iter()) {
                *uk = w.i_a.mul_vec6(sk);
            }

            // D[k,l] = S_k^T · U_l  (DOF×DOF matrix)
            let mut d = [0.0f64; MAX_JOINT_DOF * MAX_JOINT_DOF];
            for k in 0..dof {
                for l in 0..dof {
                    d[k * dof + l] = s[k].dot(&u[l]);
                }
            }

            let scale = inertia_scale(&w.i_a) * subspace_scale(s);
            let d_inv = invert_dof_matrix(&d, dof, SINGULAR_REL_TOL * scale)
                .ok_or(ArticulatedError::SingularJointInertia { body: i })?;

            // u_k = τ_k − S_k^T · p_A[i]
            let mut tau_bias = [0.0f64; MAX_JOINT_DOF];
            for (k, tb) in tau_bias.iter_mut().take(dof).enumerate() {
                *tb = tau_i[k] - s[k].dot(&w.p_a);
            }

            // Store for pass 3 (fixed-size copies, no allocation)
            wb[i].u = u;
            wb[i].d_inv = d_inv;
            wb[i].tau_bias = tau_bias;

            if body.parent_id.is_none() {
                continue;
            }

            // I_A_hat = I_A[i] - Σ_{k,l} U_k · D_inv[k,l] · U_l^T
            let mut i_a_hat = w.i_a;
            for k in 0..dof {
                for l in 0..dof {
                    i_a_hat = i_a_hat.sub_rank1_pair(&u[k], &u[l], d_inv[k * dof + l]);
                }
            }

            // U · D⁻¹ · u (uses the Schur complement, not the full I_A).
            let mut u_d_inv_tau = SpatialVec::ZERO;
            for k in 0..dof {
                let mut coeff = 0.0;
                for l in 0..dof {
                    coeff += d_inv[k * dof + l] * tau_bias[l];
                }
                u_d_inv_tau = u_d_inv_tau + u[k].scale(coeff);
            }
            let p = w.p_a + i_a_hat.mul_vec6(&w.c) + u_d_inv_tau;
            (i_a_hat, p)
        } else {
            // Fixed joint: propagate inertia and bias directly (no DOF reduction)
            (w.i_a, w.p_a + w.i_a.mul_vec6(&w.c))
        };

        if let Some(pid) = body.parent_id {
            // x_full is the pose of body i in its parent; with the motion
            // transform X = (i)X_(parent) built from it,
            // I_parent = Xᵀ · I_hat · X (computed by transform_to_parent).
            let i_a_parent = i_a_hat.transform_to_parent(&w.x_full);
            let p_in_parent = w.x_full.apply_force(&p_contribution);
            wb[pid].i_a = wb[pid].i_a.add(&i_a_parent);
            wb[pid].p_a = wb[pid].p_a + p_in_parent;
        }
    }

    // ── Pass 3: Outward — accelerations ──────────────────────────────────────

    for i in 0..n {
        let body = &model.bodies[i];
        let w = wb[i];
        let dof = w.dof;

        // a_hat = X * a_parent + c[i]
        let a_parent = match body.parent_id {
            Some(pid) => wb[pid].a,
            None => a_gravity, // a_0 = [0; −g_vec]: fictitious base acceleration
        };
        let a_hat = w.x_full.apply_velocity(&a_parent) + w.c;

        if dof > 0 {
            // Featherstone eq. 7.56: q̈_i = D_i^{-1} · (u_i − U_i^T · a_hat)
            //
            // where u_i = τ_i − S_i^T · p_A[i] (= tau_bias, stored in pass 2) and
            // U_i = I_A[i] · S_i (= u). Because I_A is symmetric,
            // U_iᵀ·a_hat = S_iᵀ·(I_A·a_hat). tau_bias already contains the
            // −Sᵀ·p_A term; it must not be subtracted again here.
            let mut rhs = [0.0f64; MAX_JOINT_DOF];
            for (l, r) in rhs.iter_mut().take(dof).enumerate() {
                *r = w.tau_bias[l] - w.u[l].dot(&a_hat);
            }

            let start = model.dof_start(i);
            let mut q_ddot_i = [0.0f64; MAX_JOINT_DOF];
            for k in 0..dof {
                let mut acc = 0.0;
                for (l, rhs_l) in rhs.iter().take(dof).enumerate() {
                    acc += w.d_inv[k * dof + l] * rhs_l;
                }
                q_ddot_i[k] = acc;
                q_ddot[start + k] = acc;
            }

            // a[i] = a_hat + S · q̈_i
            wb[i].a = a_hat + subspace_velocity(&w.s[..dof], &q_ddot_i[..dof]);
        } else {
            // Fixed joint: acceleration propagates without DOF contribution
            wb[i].a = a_hat;
        }
    }

    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// `Err(DimensionMismatch)` unless `got == expected`.
fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), ArticulatedError> {
    if got == expected {
        Ok(())
    } else {
        Err(ArticulatedError::DimensionMismatch {
            what,
            expected,
            got,
        })
    }
}

fn subspace_velocity(s: &[SpatialVec], v: &[f64]) -> SpatialVec {
    let mut result = SpatialVec::ZERO;
    for (sk, &vk) in s.iter().zip(v.iter()) {
        result = result + sk.scale(vk);
    }
    result
}

/// Largest absolute entry of a 6×6 inertia (the scale of `I_A`).
fn inertia_scale(i_a: &SpatialInertia6x6) -> f64 {
    i_a.data
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |acc, x| acc.max(x.abs()))
}

/// Largest squared Euclidean norm of the motion-subspace columns.
fn subspace_scale(s: &[SpatialVec]) -> f64 {
    s.iter().fold(0.0f64, |acc, sk| acc.max(sk.dot(sk)))
}

/// Invert a small `dof×dof` matrix (row-major, `dof ≤ 6`).
///
/// For 1×1: reciprocal. For larger: Gauss-Jordan elimination with partial
/// pivoting. Returns `None` when a pivot is not strictly larger than `tol`
/// in magnitude (singular or numerically singular matrix, or NaN entries).
fn invert_dof_matrix(
    d: &[f64; MAX_JOINT_DOF * MAX_JOINT_DOF],
    dof: usize,
    tol: f64,
) -> Option<[f64; MAX_JOINT_DOF * MAX_JOINT_DOF]> {
    let mut inv = [0.0f64; MAX_JOINT_DOF * MAX_JOINT_DOF];
    match dof {
        0 => Some(inv),
        1 => {
            let d00 = d[0];
            if d00.is_nan() || d00.abs() <= tol {
                return None;
            }
            inv[0] = 1.0 / d00;
            Some(inv)
        }
        n => {
            let mut a = *d;
            for i in 0..n {
                inv[i * n + i] = 1.0;
            }
            for col in 0..n {
                // Find pivot
                let mut pivot_row = col;
                let mut pivot_val = a[col * n + col].abs();
                for row in (col + 1)..n {
                    let v = a[row * n + col].abs();
                    if v > pivot_val {
                        pivot_val = v;
                        pivot_row = row;
                    }
                }
                if pivot_val.is_nan() || pivot_val <= tol {
                    return None;
                }
                if pivot_row != col {
                    for j in 0..n {
                        a.swap(col * n + j, pivot_row * n + j);
                        inv.swap(col * n + j, pivot_row * n + j);
                    }
                }
                let scale = 1.0 / a[col * n + col];
                for j in 0..n {
                    a[col * n + j] *= scale;
                    inv[col * n + j] *= scale;
                }
                for row in 0..n {
                    if row == col {
                        continue;
                    }
                    let factor = a[row * n + col];
                    for j in 0..n {
                        let a_col_j = a[col * n + j];
                        let inv_col_j = inv[col * n + j];
                        a[row * n + j] -= factor * a_col_j;
                        inv[row * n + j] -= factor * inv_col_j;
                    }
                }
            }
            Some(inv)
        }
    }
}

// ─── Extension for SpatialInertia6x6 needed by ABA ───────────────────────────

impl SpatialInertia6x6 {
    /// Subtract outer product of two *different* spatial vectors (for ABA):
    /// `I -= u ⊗ w * scale`
    pub(crate) fn sub_rank1_pair(&self, u: &SpatialVec, w: &SpatialVec, scale: f64) -> Self {
        let uu = [
            u.angular[0],
            u.angular[1],
            u.angular[2],
            u.linear[0],
            u.linear[1],
            u.linear[2],
        ];
        let ww = [
            w.angular[0],
            w.angular[1],
            w.angular[2],
            w.linear[0],
            w.linear[1],
            w.linear[2],
        ];
        let mut d = self.data;
        for i in 0..6 {
            for j in 0..6 {
                d[i][j] -= uu[i] * ww[j] * scale;
            }
        }
        Self { data: d }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        body::RigidBody,
        joint::RevoluteJoint,
        model::ArticulatedModel,
        spatial::{SpatialInertia, SpatialTransform},
    };

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn build_pendulum(m: f64, l: f64, g: f64) -> ArticulatedModel {
        let mut model = ArticulatedModel::new([0.0, -g, 0.0]);
        let joint = Box::new(RevoluteJoint::new([0.0, 0.0, 1.0]));
        let com = [0.0, -l, 0.0];
        let inertia = SpatialInertia::from_com(m, com, [[0.0; 3]; 3]);
        let body = RigidBody::new("link1", inertia, None, SpatialTransform::IDENTITY);
        model.add_body(body, joint);
        model
    }

    #[test]
    fn test_aba_pendulum_horizontal() {
        // Single pendulum at q=π/2 (horizontal), zero velocity, no torque.
        // Expected q_ddot = -g/L (for a point mass: θ̈ = -g/L·sin(π/2) = -g/L).
        let (m, l, g) = (1.0, 1.0, 9.81);
        let model = build_pendulum(m, l, g);

        let q_ddot = aba(&model, &[std::f64::consts::FRAC_PI_2], &[0.0], &[0.0]);
        let expected = -g / l;
        assert!(
            approx_eq(q_ddot[0], expected, 1e-6),
            "Pendulum ABA at horizontal: expected q_ddot={expected:.6}, got {:.6}",
            q_ddot[0]
        );
    }

    /// Mixed tree: free-floating base, spherical, universal, revolute,
    /// fixed and helical joints, with a branch.
    fn mixed_model() -> ArticulatedModel {
        use crate::joint::{
            FixedJoint, FreeFloatingJoint, HelicalJoint, SphericalJoint, UniversalJoint,
        };
        let mut model = ArticulatedModel::new([0.0, 0.0, -9.81]);
        let inertia = |m: f64, c: [f64; 3]| {
            SpatialInertia::from_com(
                m,
                c,
                [[0.05, 0.002, 0.0], [0.002, 0.04, 0.001], [0.0, 0.001, 0.03]],
            )
        };
        model.add_body(
            RigidBody::new(
                "base",
                inertia(3.0, [0.0, 0.0, 0.1]),
                None,
                SpatialTransform::IDENTITY,
            ),
            Box::new(FreeFloatingJoint),
        );
        model.add_body(
            RigidBody::new(
                "ball",
                inertia(1.2, [0.0, 0.1, 0.2]),
                Some(0),
                SpatialTransform::from_translation([0.1, 0.0, 0.3]),
            ),
            Box::new(SphericalJoint),
        );
        model.add_body(
            RigidBody::new(
                "cardan",
                inertia(0.8, [0.05, 0.0, 0.2]),
                Some(1),
                SpatialTransform::from_translation([0.0, 0.0, 0.4]),
            ),
            Box::new(UniversalJoint::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])),
        );
        model.add_body(
            RigidBody::new(
                "hinge",
                inertia(0.6, [0.0, 0.0, 0.15]),
                Some(0),
                SpatialTransform::from_translation([-0.2, 0.1, 0.0])
                    .compose(&SpatialTransform::from_axis_angle([0.0, 1.0, 0.0], 0.3)),
            ),
            Box::new(RevoluteJoint::new([0.0, 1.0, 0.0])),
        );
        model.add_body(
            RigidBody::new(
                "tool",
                inertia(0.2, [0.0, 0.0, 0.05]),
                Some(3),
                SpatialTransform::from_translation([0.0, 0.0, 0.3]),
            ),
            Box::new(FixedJoint),
        );
        model.add_body(
            RigidBody::new(
                "screw",
                inertia(0.3, [0.02, 0.0, 0.1]),
                Some(4),
                SpatialTransform::from_translation([0.0, 0.05, 0.1]),
            ),
            Box::new(HelicalJoint::new([0.0, 0.0, 1.0], 0.01)),
        );
        model
    }

    fn mixed_state(model: &ArticulatedModel) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nv = model.total_dof();
        let v0: Vec<f64> = (0..nv)
            .map(|k| 0.3 * (0.7 * k as f64 + 0.2).sin())
            .collect();
        let q = model.integrate(&model.neutral_configuration(), &v0, 1.0);
        let qd: Vec<f64> = (0..nv)
            .map(|k| 0.9 * (1.3 * k as f64 + 0.5).cos())
            .collect();
        let tau: Vec<f64> = (0..nv)
            .map(|k| 2.0 * (0.4 * k as f64 - 0.3).sin())
            .collect();
        (q, qd, tau)
    }

    /// ABA agrees with the dense solution `q̈ = M⁻¹·(τ − RNEA(q, q̇, 0))` on a
    /// branched tree with every joint type.
    #[test]
    fn test_aba_matches_crba_rnea_solution_on_mixed_tree() {
        let model = mixed_model();
        let (q, qd, tau) = mixed_state(&model);
        let nv = model.total_dof();
        let bias = crate::rnea::rnea(&model, &q, &qd, &vec![0.0; nv]);
        let m = crate::crba::compute_mass_matrix_crba(&model, &q);
        let q_ddot = aba(&model, &q, &qd, &tau);
        // Residual M·q̈ − (τ − bias) must vanish.
        for r in 0..nv {
            let lhs: f64 = (0..nv).map(|c| m[r][c] * q_ddot[c]).sum();
            let rhs = tau[r] - bias[r];
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "row {r}: M·q̈ = {lhs}, τ − bias = {rhs}"
            );
        }
    }

    /// A reused workspace gives bit-identical results to `aba`, across calls
    /// and across models of different sizes.
    #[test]
    fn test_aba_with_workspace_matches_aba_and_reuses_buffers() {
        let small = build_pendulum(1.0, 1.0, 9.81);
        let big = mixed_model();
        let mut ws = AbaWorkspace::new(&small);
        let (q, qd, tau) = mixed_state(&big);
        let expected = aba(&big, &q, &qd, &tau);
        let mut out = vec![0.0; big.total_dof()];
        for _ in 0..3 {
            aba_with_workspace(&big, &mut ws, &q, &qd, &tau, &mut out).expect("valid model");
            assert_eq!(out, expected);
        }
        let mut out_small = [0.0];
        aba_with_workspace(&small, &mut ws, &[0.4], &[0.1], &[0.0], &mut out_small)
            .expect("valid model");
        assert_eq!(out_small.to_vec(), aba(&small, &[0.4], &[0.1], &[0.0]));
    }

    /// Wrong input lengths are reported, not panicked on.
    #[test]
    fn test_try_aba_dimension_mismatch() {
        let model = build_pendulum(1.0, 1.0, 9.81);
        assert!(matches!(
            try_aba(&model, &[0.0, 0.0], &[0.0], &[0.0]),
            Err(ArticulatedError::DimensionMismatch { what: "q", .. })
        ));
        assert!(matches!(
            try_aba(&model, &[0.0], &[], &[0.0]),
            Err(ArticulatedError::DimensionMismatch { what: "q_dot", .. })
        ));
        assert!(matches!(
            try_aba(&model, &[0.0], &[0.0], &[0.0, 1.0]),
            Err(ArticulatedError::DimensionMismatch { what: "tau", .. })
        ));
        let mut ws = AbaWorkspace::default();
        let mut out = [0.0; 2];
        assert!(matches!(
            aba_with_workspace(&model, &mut ws, &[0.0], &[0.0], &[0.0], &mut out),
            Err(ArticulatedError::DimensionMismatch { what: "q_ddot", .. })
        ));
    }

    /// Two-link chain whose leaf is massless and inertia-free.
    fn chain_with_massless_leaf() -> ArticulatedModel {
        let mut model = build_pendulum(1.0, 1.0, 9.81);
        let massless = SpatialInertia::from_com(0.0, [0.0; 3], [[0.0; 3]; 3]);
        model.add_body(
            RigidBody::new(
                "massless",
                massless,
                Some(0),
                SpatialTransform::from_translation([0.0, -1.0, 0.0]),
            ),
            Box::new(RevoluteJoint::new([0.0, 0.0, 1.0])),
        );
        model
    }

    /// A revolute leaf with zero mass and zero inertia has `D = 0`: q̈ is
    /// undefined and must be reported instead of silently returning 0.
    #[test]
    fn test_try_aba_singular_joint_inertia() {
        let model = chain_with_massless_leaf();
        assert_eq!(
            try_aba(&model, &[0.3, 0.2], &[0.0, 0.0], &[0.0, 0.0]),
            Err(ArticulatedError::SingularJointInertia { body: 1 })
        );
        // A point mass on the joint axis: D = 0 relative to a non-zero I_A.
        let mut on_axis = ArticulatedModel::new([0.0, -9.81, 0.0]);
        on_axis.add_body(
            RigidBody::new(
                "on_axis",
                SpatialInertia::from_com(2.0, [0.0, 0.0, 0.5], [[0.0; 3]; 3]),
                None,
                SpatialTransform::IDENTITY,
            ),
            Box::new(RevoluteJoint::new([0.0, 0.0, 1.0])),
        );
        assert_eq!(
            try_aba(&on_axis, &[0.0], &[0.0], &[0.0]),
            Err(ArticulatedError::SingularJointInertia { body: 0 })
        );
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn test_aba_panics_on_singular_joint_inertia() {
        let model = chain_with_massless_leaf();
        let _ = aba(&model, &[0.3, 0.2], &[0.0, 0.0], &[0.0, 0.0]);
    }

    /// The singularity test is relative: a very light (but non-degenerate)
    /// pendulum still yields the mass-independent θ̈ = −g/L.
    #[test]
    fn test_aba_tiny_mass_is_not_singular() {
        let model = build_pendulum(1e-9, 1.0, 9.81);
        let q_ddot =
            try_aba(&model, &[std::f64::consts::FRAC_PI_2], &[0.0], &[0.0]).expect("non-singular");
        assert!(approx_eq(q_ddot[0], -9.81, 1e-6), "got {}", q_ddot[0]);
    }

    #[test]
    fn test_invert_dof_matrix_rejects_singular_and_nan() {
        let mut d = [0.0f64; MAX_JOINT_DOF * MAX_JOINT_DOF];
        // Rank-2 3×3 matrix.
        d[..9].copy_from_slice(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);
        assert!(invert_dof_matrix(&d, 3, 1e-12).is_none());
        d[0] = f64::NAN;
        assert!(invert_dof_matrix(&d, 1, 0.0).is_none());
        // Well-conditioned 2×2: [[2, 1], [1, 3]]⁻¹ = [[0.6, −0.2], [−0.2, 0.4]].
        d[..4].copy_from_slice(&[2.0, 1.0, 1.0, 3.0]);
        let inv = invert_dof_matrix(&d, 2, 1e-12).expect("invertible");
        for (got, want) in inv[..4].iter().zip([0.6, -0.2, -0.2, 0.4]) {
            assert!(approx_eq(*got, want, 1e-15));
        }
    }

    #[test]
    fn test_aba_hanging_no_torque() {
        // At q=0 (hanging), no torque. Expected q_ddot=0 (stable equilibrium).
        let (m, l, g) = (1.0, 1.0, 9.81);
        let model = build_pendulum(m, l, g);

        let q_ddot = aba(&model, &[0.0], &[0.0], &[0.0]);
        assert!(
            approx_eq(q_ddot[0], 0.0, 1e-8),
            "At q=0 (hanging), q_ddot should be 0, got {:.8}",
            q_ddot[0]
        );
    }
}
