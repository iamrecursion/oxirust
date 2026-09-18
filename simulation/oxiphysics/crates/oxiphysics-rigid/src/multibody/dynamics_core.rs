//! World-frame spatial-algebra core shared by the multibody dynamics types.
//!
//! Every spatial quantity in this module is expressed in the world frame and
//! taken about the world origin, using the angular-first convention: a motion
//! vector is `[ω; v_O]` and a force vector is `[n_O; f]`.  Working in a single
//! frame removes all Plücker transforms from the recursions — the motion
//! subspace of a joint is written directly in world coordinates and its time
//! derivative is `v_i ×ₘ S_i`.
//!
//! The model is a tree of bodies, each connected to its parent by a joint with
//! at most one degree of freedom (revolute, prismatic or fixed).  Multi-DoF
//! joints are expanded by the callers into chains of massless 1-DoF bodies,
//! which is an exact representation of Euler-angle (spherical, universal,
//! free) and planar/cylindrical parametrisations.
//!
//! Algorithms provided (Featherstone, *Rigid Body Dynamics Algorithms*, 2008):
//! * forward kinematics with world-frame motion subspaces and spatial inertias,
//! * the recursive Newton–Euler algorithm (RNEA, inverse dynamics),
//! * the composite rigid-body algorithm (CRBA, joint-space inertia `H(q)`),
//! * the three-pass articulated-body algorithm (ABA, forward dynamics),
//! * point Jacobians and a Cholesky solver for `H q̈ = b`.

use super::functions::{
    Mat3, mat3_identity, mat3_mul, mat3_transpose, mat3_vec_mul, rot_from_axis_angle, vec3_add,
    vec3_cross, vec3_dot, vec3_norm, vec3_scale,
};
use crate::error::{Error, Result};

/// Spatial 6-vector `[angular; linear]`.
pub(super) type Vec6 = [f64; 6];
/// Spatial 6×6 matrix (row-major).
pub(super) type Mat6 = [[f64; 6]; 6];

/// Relative tolerance below which an articulated pivot `D = Sᵀ Iᴬ S` is
/// treated as singular (massless sub-tree or gimbal lock).
const PIVOT_REL_TOL: f64 = 1e-12;

/// Kind of the single-DoF joint that connects a core body to its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CoreJointKind {
    /// Rigid connection, no degree of freedom.
    Fixed,
    /// Rotation about `axis` through the joint-frame origin.
    Revolute,
    /// Translation along `axis`.
    Prismatic,
}

/// One body of the core model.
///
/// Frame chain: `body = parent_body ∘ (pre_rotation, pre_translation) ∘
/// joint_motion(q) ∘ (post_rotation, 0)`.
#[derive(Debug, Clone)]
pub(super) struct CoreBody {
    /// Parent core body (`None` = attached to the fixed base frame).
    pub(super) parent: Option<usize>,
    /// Rotation of the joint frame relative to the parent body frame.
    pub(super) pre_rotation: Mat3,
    /// Joint-frame origin in the parent body frame (m).
    pub(super) pre_translation: [f64; 3],
    /// Joint kind.
    pub(super) kind: CoreJointKind,
    /// Joint axis in the joint frame (normalised by [`CoreModel::push`]).
    pub(super) axis: [f64; 3],
    /// Generalised-coordinate index (assigned by [`CoreModel::push`]).
    pub(super) dof: Option<usize>,
    /// Rotation of the body frame relative to the moved joint frame.
    pub(super) post_rotation: Mat3,
    /// Body mass (kg).
    pub(super) mass: f64,
    /// Centre of mass in the body frame (m).
    pub(super) com: [f64; 3],
    /// Rotational inertia about the centre of mass, body frame (kg·m²).
    pub(super) inertia: Mat3,
}

impl CoreBody {
    /// A massless body connected by `kind` about/along `axis`.
    pub(super) fn massless(
        parent: Option<usize>,
        pre_rotation: Mat3,
        pre_translation: [f64; 3],
        kind: CoreJointKind,
        axis: [f64; 3],
    ) -> Self {
        Self {
            parent,
            pre_rotation,
            pre_translation,
            kind,
            axis,
            dof: None,
            post_rotation: mat3_identity(),
            mass: 0.0,
            com: [0.0; 3],
            inertia: [[0.0; 3]; 3],
        }
    }

    /// Attach mass properties and the post-joint body rotation.
    pub(super) fn with_mass(
        mut self,
        mass: f64,
        com: [f64; 3],
        inertia: Mat3,
        post_rotation: Mat3,
    ) -> Self {
        self.mass = mass;
        self.com = com;
        self.inertia = inertia;
        self.post_rotation = post_rotation;
        self
    }
}

/// A tree of single-DoF bodies attached to a fixed base frame.
#[derive(Debug, Clone)]
pub(super) struct CoreModel {
    bodies: Vec<CoreBody>,
    n_dof: usize,
    base_rotation: Mat3,
    base_translation: [f64; 3],
}

/// World-frame kinematic quantities of every core body for one configuration.
#[derive(Debug, Clone)]
pub(super) struct CoreKinematics {
    /// Body-frame rotation (body → world).
    pub(super) rotation: Vec<Mat3>,
    /// Body-frame origin (m).
    pub(super) origin: Vec<[f64; 3]>,
    /// Centre of mass (m).
    pub(super) com: Vec<[f64; 3]>,
    /// Joint motion subspace `S_i` (zero for fixed joints).
    pub(super) motion: Vec<Vec6>,
    /// Spatial inertia about the world origin.
    pub(super) inertia: Vec<Mat6>,
}

fn articulated_error(reason: String) -> Error {
    Error::ArticulatedBodyError { reason }
}

impl CoreModel {
    /// Create an empty model whose base frame has the given world pose.
    pub(super) fn new(base_rotation: Mat3, base_translation: [f64; 3]) -> Self {
        Self {
            bodies: Vec::new(),
            n_dof: 0,
            base_rotation,
            base_translation,
        }
    }

    /// Append a body; returns its index.  Assigns the next generalised
    /// coordinate to non-fixed joints and normalises the joint axis.
    pub(super) fn push(&mut self, mut body: CoreBody) -> Result<usize> {
        let index = self.bodies.len();
        if let Some(p) = body.parent
            && p >= index
        {
            return Err(articulated_error(format!(
                "body {index} references parent {p}, bodies must be ordered parent-before-child"
            )));
        }
        if !(body.mass.is_finite() && body.mass >= 0.0) {
            return Err(articulated_error(format!(
                "body {index} has invalid mass {} (must be finite and non-negative)",
                body.mass
            )));
        }
        if body.kind == CoreJointKind::Fixed {
            body.dof = None;
        } else {
            let norm = vec3_norm(body.axis);
            if !(norm.is_finite() && norm > 1e-12) {
                return Err(Error::SpatialAlgebraError {
                    reason: format!("joint axis of body {index} has zero length"),
                });
            }
            body.axis = vec3_scale(body.axis, 1.0 / norm);
            body.dof = Some(self.n_dof);
            self.n_dof += 1;
        }
        self.bodies.push(body);
        Ok(index)
    }

    /// Number of generalised coordinates.
    pub(super) fn n_dof(&self) -> usize {
        self.n_dof
    }

    fn check_len(&self, values: &[f64], name: &str) -> Result<()> {
        if values.len() == self.n_dof {
            Ok(())
        } else {
            Err(articulated_error(format!(
                "{name} has length {} but the model has {} degrees of freedom",
                values.len(),
                self.n_dof
            )))
        }
    }

    fn joint_value(body: &CoreBody, values: &[f64]) -> f64 {
        body.dof.and_then(|d| values.get(d).copied()).unwrap_or(0.0)
    }

    /// Forward kinematics for configuration `q`.
    pub(super) fn kinematics(&self, q: &[f64]) -> Result<CoreKinematics> {
        self.check_len(q, "q")?;
        let n = self.bodies.len();
        let mut kin = CoreKinematics {
            rotation: Vec::with_capacity(n),
            origin: Vec::with_capacity(n),
            com: Vec::with_capacity(n),
            motion: Vec::with_capacity(n),
            inertia: Vec::with_capacity(n),
        };
        for body in &self.bodies {
            let (parent_rot, parent_pos) = match body.parent {
                Some(p) => (kin.rotation[p], kin.origin[p]),
                None => (self.base_rotation, self.base_translation),
            };
            let joint_rot = mat3_mul(parent_rot, body.pre_rotation);
            let joint_pos = vec3_add(parent_pos, mat3_vec_mul(parent_rot, body.pre_translation));
            let axis_world = mat3_vec_mul(joint_rot, body.axis);
            let qi = Self::joint_value(body, q);
            let (moved_rot, moved_pos, motion) = match body.kind {
                CoreJointKind::Fixed => (joint_rot, joint_pos, [0.0; 6]),
                CoreJointKind::Revolute => {
                    let lin = vec3_cross(joint_pos, axis_world);
                    (
                        mat3_mul(joint_rot, rot_from_axis_angle(body.axis, qi)),
                        joint_pos,
                        [
                            axis_world[0],
                            axis_world[1],
                            axis_world[2],
                            lin[0],
                            lin[1],
                            lin[2],
                        ],
                    )
                }
                CoreJointKind::Prismatic => (
                    joint_rot,
                    vec3_add(joint_pos, vec3_scale(axis_world, qi)),
                    [0.0, 0.0, 0.0, axis_world[0], axis_world[1], axis_world[2]],
                ),
            };
            let rot = mat3_mul(moved_rot, body.post_rotation);
            let com = vec3_add(moved_pos, mat3_vec_mul(rot, body.com));
            let ic_world = mat3_mul(mat3_mul(rot, body.inertia), mat3_transpose(rot));
            kin.inertia.push(spatial_inertia(body.mass, com, ic_world));
            kin.rotation.push(rot);
            kin.origin.push(moved_pos);
            kin.com.push(com);
            kin.motion.push(motion);
        }
        Ok(kin)
    }

    /// Spatial velocity of every body for joint velocities `qd`.
    pub(super) fn velocities(&self, kin: &CoreKinematics, qd: &[f64]) -> Result<Vec<Vec6>> {
        self.check_len(qd, "qd")?;
        let mut v: Vec<Vec6> = Vec::with_capacity(self.bodies.len());
        for (i, body) in self.bodies.iter().enumerate() {
            let vp = body.parent.map(|p| v[p]).unwrap_or([0.0; 6]);
            v.push(add6(vp, scale6(kin.motion[i], Self::joint_value(body, qd))));
        }
        Ok(v)
    }

    /// Recursive Newton–Euler inverse dynamics: `τ = H(q) q̈ + C(q, q̇) q̇ + G(q)`.
    ///
    /// Gravity enters as a fictitious base acceleration `a₀ = [0; −g]`.
    pub(super) fn rnea(
        &self,
        kin: &CoreKinematics,
        qd: &[f64],
        qdd: &[f64],
        gravity: [f64; 3],
    ) -> Result<Vec<f64>> {
        self.check_len(qd, "qd")?;
        self.check_len(qdd, "qdd")?;
        let n = self.bodies.len();
        let a0 = base_acceleration(gravity);
        let mut v: Vec<Vec6> = Vec::with_capacity(n);
        let mut a: Vec<Vec6> = Vec::with_capacity(n);
        let mut f: Vec<Vec6> = Vec::with_capacity(n);
        for (i, body) in self.bodies.iter().enumerate() {
            let (vp, ap) = match body.parent {
                Some(p) => (v[p], a[p]),
                None => ([0.0; 6], a0),
            };
            let vj = scale6(kin.motion[i], Self::joint_value(body, qd));
            let vi = add6(vp, vj);
            let ai = add6(
                add6(ap, scale6(kin.motion[i], Self::joint_value(body, qdd))),
                cross_motion(vi, vj),
            );
            let inertia = &kin.inertia[i];
            let fi = add6(
                mat6_vec(inertia, ai),
                cross_force(vi, mat6_vec(inertia, vi)),
            );
            v.push(vi);
            a.push(ai);
            f.push(fi);
        }
        let mut tau = vec![0.0; self.n_dof];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            if let Some(d) = body.dof {
                tau[d] = dot6(kin.motion[i], f[i]);
            }
            if let Some(p) = body.parent {
                let fi = f[i];
                f[p] = add6(f[p], fi);
            }
        }
        Ok(tau)
    }

    /// Composite rigid-body algorithm: joint-space inertia matrix `H(q)`.
    pub(super) fn crba(&self, kin: &CoreKinematics) -> Vec<Vec<f64>> {
        let n = self.bodies.len();
        let mut composite = kin.inertia.clone();
        for i in (0..n).rev() {
            if let Some(p) = self.bodies[i].parent {
                let ci = composite[i];
                composite[p] = add66(&composite[p], &ci);
            }
        }
        let mut h = vec![vec![0.0; self.n_dof]; self.n_dof];
        for (i, (body, ci)) in self.bodies.iter().zip(composite.iter()).enumerate() {
            let Some(di) = body.dof else {
                continue;
            };
            let force = mat6_vec(ci, kin.motion[i]);
            h[di][di] = dot6(kin.motion[i], force);
            let mut ancestor = body.parent;
            while let Some(j) = ancestor {
                if let Some(dj) = self.bodies[j].dof {
                    let hij = dot6(kin.motion[j], force);
                    h[di][dj] = hij;
                    h[dj][di] = hij;
                }
                ancestor = self.bodies[j].parent;
            }
        }
        h
    }

    /// Three-pass articulated-body algorithm: joint accelerations `q̈` for
    /// joint torques/forces `tau`.
    pub(super) fn aba(
        &self,
        kin: &CoreKinematics,
        qd: &[f64],
        tau: &[f64],
        gravity: [f64; 3],
    ) -> Result<Vec<f64>> {
        self.check_len(tau, "tau")?;
        let n = self.bodies.len();
        let v = self.velocities(kin, qd)?;
        let mut art_inertia = kin.inertia.clone();
        let mut bias: Vec<Vec6> = (0..n)
            .map(|i| cross_force(v[i], mat6_vec(&kin.inertia[i], v[i])))
            .collect();
        let coriolis: Vec<Vec6> = (0..n)
            .map(|i| {
                let vj = scale6(kin.motion[i], Self::joint_value(&self.bodies[i], qd));
                cross_motion(v[i], vj)
            })
            .collect();
        let mut u_col = vec![[0.0; 6]; n];
        let mut pivot = vec![0.0; n];
        let mut u_scalar = vec![0.0; n];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let ia = art_inertia[i];
            let (ia_pass, pa_pass) = if let Some(d) = body.dof {
                let s = kin.motion[i];
                let u = mat6_vec(&ia, s);
                let di = dot6(s, u);
                let scale = ia
                    .iter()
                    .flat_map(|row| row.iter())
                    .fold(0.0_f64, |acc, x| acc.max(x.abs()));
                let regular = di.is_finite() && di > PIVOT_REL_TOL * scale;
                if !regular {
                    return Err(Error::ForwardDynamicsFailure {
                        body_index: i,
                        reason: format!(
                            "articulated inertia about the joint axis is singular (D = {di:.3e}); \
                             the sub-tree is massless or the joint parametrisation is at a singularity"
                        ),
                    });
                }
                let ui = tau.get(d).copied().unwrap_or(0.0) - dot6(s, bias[i]);
                u_col[i] = u;
                pivot[i] = di;
                u_scalar[i] = ui;
                let mut ia_child = ia;
                for (r, row) in ia_child.iter_mut().enumerate() {
                    for (c, val) in row.iter_mut().enumerate() {
                        *val -= u[r] * u[c] / di;
                    }
                }
                let pa_child = add6(
                    add6(bias[i], mat6_vec(&ia_child, coriolis[i])),
                    scale6(u, ui / di),
                );
                (ia_child, pa_child)
            } else {
                (ia, add6(bias[i], mat6_vec(&ia, coriolis[i])))
            };
            if let Some(p) = body.parent {
                art_inertia[p] = add66(&art_inertia[p], &ia_pass);
                bias[p] = add6(bias[p], pa_pass);
            }
        }
        let a0 = base_acceleration(gravity);
        let mut acc: Vec<Vec6> = Vec::with_capacity(n);
        let mut qdd = vec![0.0; self.n_dof];
        for (i, body) in self.bodies.iter().enumerate() {
            let ap = body.parent.map(|p| acc[p]).unwrap_or(a0);
            let a_pre = add6(ap, coriolis[i]);
            if let Some(d) = body.dof {
                let qddi = (u_scalar[i] - dot6(u_col[i], a_pre)) / pivot[i];
                qdd[d] = qddi;
                acc.push(add6(a_pre, scale6(kin.motion[i], qddi)));
            } else {
                acc.push(a_pre);
            }
        }
        Ok(qdd)
    }

    /// Linear-velocity Jacobian (3 × n_dof, row-major) of a point rigidly
    /// attached to `body`, given in world coordinates.
    pub(super) fn point_jacobian(
        &self,
        kin: &CoreKinematics,
        body: usize,
        point: [f64; 3],
    ) -> Result<Vec<Vec<f64>>> {
        if body >= self.bodies.len() {
            return Err(articulated_error(format!(
                "body index {body} out of range ({} bodies)",
                self.bodies.len()
            )));
        }
        let mut j = vec![vec![0.0; self.n_dof]; 3];
        let mut current = Some(body);
        while let Some(b) = current {
            if let Some(d) = self.bodies[b].dof {
                let s = kin.motion[b];
                let lin = vec3_add([s[3], s[4], s[5]], vec3_cross([s[0], s[1], s[2]], point));
                for (row, val) in j.iter_mut().zip(lin.iter()) {
                    row[d] = *val;
                }
            }
            current = self.bodies[b].parent;
        }
        Ok(j)
    }
}

/// Linear velocity of a world point `p` moving with spatial velocity `v`.
pub(super) fn point_velocity(v: Vec6, p: [f64; 3]) -> [f64; 3] {
    vec3_add([v[3], v[4], v[5]], vec3_cross([v[0], v[1], v[2]], p))
}

fn base_acceleration(gravity: [f64; 3]) -> Vec6 {
    [0.0, 0.0, 0.0, -gravity[0], -gravity[1], -gravity[2]]
}

/// Skew-symmetric cross-product matrix `[c]×`.
fn skew(c: [f64; 3]) -> Mat3 {
    [[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]]
}

/// Spatial inertia about the world origin of a body with mass `m`, world
/// centre of mass `c` and world-frame rotational inertia `ic` about `c`:
/// `[[Ic + m [c]×[c]×ᵀ, m [c]×], [m [c]×ᵀ, m 1]]`.
pub(super) fn spatial_inertia(m: f64, c: [f64; 3], ic: Mat3) -> Mat6 {
    let cx = skew(c);
    let cc = vec3_dot(c, c);
    let mut out = [[0.0; 6]; 6];
    for r in 0..3 {
        for k in 0..3 {
            let delta = if r == k { 1.0 } else { 0.0 };
            out[r][k] = ic[r][k] + m * (cc * delta - c[r] * c[k]);
            out[r][k + 3] = m * cx[r][k];
            out[r + 3][k] = m * cx[k][r];
            out[r + 3][k + 3] = m * delta;
        }
    }
    out
}

fn mat6_vec(m: &Mat6, v: Vec6) -> Vec6 {
    let mut out = [0.0; 6];
    for (o, row) in out.iter_mut().zip(m.iter()) {
        *o = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    }
    out
}

fn add66(a: &Mat6, b: &Mat6) -> Mat6 {
    let mut out = *a;
    for (row_o, row_b) in out.iter_mut().zip(b.iter()) {
        for (o, x) in row_o.iter_mut().zip(row_b.iter()) {
            *o += x;
        }
    }
    out
}

fn add6(a: Vec6, b: Vec6) -> Vec6 {
    let mut out = a;
    for (o, x) in out.iter_mut().zip(b.iter()) {
        *o += x;
    }
    out
}

fn scale6(a: Vec6, s: f64) -> Vec6 {
    a.map(|x| x * s)
}

fn dot6(a: Vec6, b: Vec6) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Spatial motion cross product `v ×ₘ u`.
fn cross_motion(v: Vec6, u: Vec6) -> Vec6 {
    let w = [v[0], v[1], v[2]];
    let v0 = [v[3], v[4], v[5]];
    let uw = [u[0], u[1], u[2]];
    let uv = [u[3], u[4], u[5]];
    let top = vec3_cross(w, uw);
    let bot = vec3_add(vec3_cross(w, uv), vec3_cross(v0, uw));
    [top[0], top[1], top[2], bot[0], bot[1], bot[2]]
}

/// Spatial force cross product `v ×_f f`.
fn cross_force(v: Vec6, f: Vec6) -> Vec6 {
    let w = [v[0], v[1], v[2]];
    let v0 = [v[3], v[4], v[5]];
    let fn_ = [f[0], f[1], f[2]];
    let fl = [f[3], f[4], f[5]];
    let top = vec3_add(vec3_cross(w, fn_), vec3_cross(v0, fl));
    let bot = vec3_cross(w, fl);
    [top[0], top[1], top[2], bot[0], bot[1], bot[2]]
}

/// Solve `H x = b` for a symmetric positive-definite `H` by Cholesky
/// factorisation.
pub(super) fn cholesky_solve(h: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if h.len() != n || h.iter().any(|row| row.len() != n) {
        return Err(articulated_error(format!(
            "matrix/vector dimension mismatch in Cholesky solve (n = {n})"
        )));
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = l[i]
                .iter()
                .zip(l[j].iter())
                .take(j)
                .map(|(a, b)| a * b)
                .sum();
            let sum = h[i][j] - dot;
            if i == j {
                let positive = sum.is_finite() && sum > 0.0;
                if !positive {
                    return Err(Error::ForwardDynamicsFailure {
                        body_index: i,
                        reason: format!(
                            "joint-space inertia matrix is not positive definite (pivot {sum:.3e})"
                        ),
                    });
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let dot: f64 = l[i].iter().zip(y.iter()).take(i).map(|(a, b)| a * b).sum();
        y[i] = (b[i] - dot) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let dot: f64 = ((i + 1)..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (y[i] - dot) / l[i][i];
    }
    Ok(x)
}

/// Rotation matrix of a unit quaternion `[w, x, y, z]` (normalised here).
pub(super) fn quat_to_mat3(q: [f64; 4]) -> Result<Mat3> {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !(norm.is_finite() && norm > 1e-12) {
        return Err(Error::SpatialAlgebraError {
            reason: "orientation quaternion has zero length".to_string(),
        });
    }
    let [w, x, y, z] = q.map(|c| c / norm);
    Ok([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}

/// Unit quaternion `[w, x, y, z]` (with `w ≥ 0`) of a rotation matrix
/// (Shepperd's method).
pub(super) fn mat3_to_quat(m: Mat3) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
    q.map(|c| sign * c / norm)
}

/// Symmetric 3×3 inertia tensor from a diagonal.
pub(super) fn diag_inertia(d: [f64; 3]) -> Mat3 {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}
