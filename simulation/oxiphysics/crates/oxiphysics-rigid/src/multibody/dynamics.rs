//! Rigid multibody dynamics for the `multibody` model types.
//!
//! [`MultibodySystem`], [`MultibodyChain`] (through [`RecursiveNewtonEuler`],
//! [`JointSpaceInertia`] and [`FeatherstoneAba`]) and [`FeatherstoneABA`] are
//! each translated into the shared world-frame spatial-algebra core
//! (`dynamics_core`) and evaluated with the exact RNEA / CRBA / ABA
//! recursions.
//!
//! Joint conventions:
//! * [`MultibodySystem`] — joint frame = parent link frame ∘
//!   `Trans(joint_anchor)` ∘ `local_frame_parent`; revolute/prismatic joints act
//!   about/along the joint-frame **z** axis; the link frame is the moved joint
//!   frame ∘ `local_frame_childᵀ`.  Multi-DoF joint types are parametrised as
//!   successive 1-DoF motions in the joint frame: spherical = Rx·Ry·Rz,
//!   universal = Rx·Ry, planar = Tx·Ty·Rz, cylindrical = Rz·Tz (coordinates in
//!   that order).
//! * [`MultibodyChain`] — link `k` frame = link `k − 1` frame ∘
//!   `Trans(joint_offset)` ∘ joint motion; revolute/prismatic axes are given in
//!   the parent frame; spherical = Rx·Ry·Rz, free = Tx·Ty·Tz·Rx·Ry·Rz.

use super::dynamics_core::{
    CoreBody, CoreJointKind, CoreModel, cholesky_solve, diag_inertia, mat3_to_quat, point_velocity,
    quat_to_mat3,
};
use super::functions::{Mat3, mat3_identity, mat3_transpose};
use super::types::{
    FeatherstoneABA, FeatherstoneAba, JointDof, JointSpaceInertia, JointType, MultibodyChain,
    MultibodySystem, RecursiveNewtonEuler,
};
use crate::error::{Error, Result};

const X_AXIS: [f64; 3] = [1.0, 0.0, 0.0];
const Y_AXIS: [f64; 3] = [0.0, 1.0, 0.0];
const Z_AXIS: [f64; 3] = [0.0, 0.0, 1.0];

/// One joint expanded into successive 1-DoF motions, plus the mass
/// properties of the link it carries.
struct JointSequence<'a> {
    parent: Option<usize>,
    pre_rotation: Mat3,
    pre_translation: [f64; 3],
    steps: &'a [(CoreJointKind, [f64; 3])],
    mass: f64,
    com: [f64; 3],
    inertia: Mat3,
    post_rotation: Mat3,
}

impl JointSequence<'_> {
    /// Push the sequence into `model`; the final body carries the link's mass
    /// properties.  Returns the index of that body.
    fn push_into(&self, model: &mut CoreModel) -> Result<usize> {
        let fixed_step = [(CoreJointKind::Fixed, Z_AXIS)];
        let steps = if self.steps.is_empty() {
            &fixed_step[..]
        } else {
            self.steps
        };
        let mut current_parent = self.parent;
        let mut last = 0usize;
        for (k, (kind, axis)) in steps.iter().enumerate() {
            let (rot, trans) = if k == 0 {
                (self.pre_rotation, self.pre_translation)
            } else {
                (mat3_identity(), [0.0; 3])
            };
            let mut body = CoreBody::massless(current_parent, rot, trans, *kind, *axis);
            if k + 1 == steps.len() {
                body = body.with_mass(self.mass, self.com, self.inertia, self.post_rotation);
            }
            last = model.push(body)?;
            current_parent = Some(last);
        }
        Ok(last)
    }
}

fn joint_type_steps(joint_type: JointType) -> Vec<(CoreJointKind, [f64; 3])> {
    use CoreJointKind::{Prismatic, Revolute};
    match joint_type {
        JointType::Revolute => vec![(Revolute, Z_AXIS)],
        JointType::Prismatic => vec![(Prismatic, Z_AXIS)],
        JointType::Spherical => vec![(Revolute, X_AXIS), (Revolute, Y_AXIS), (Revolute, Z_AXIS)],
        JointType::Universal => vec![(Revolute, X_AXIS), (Revolute, Y_AXIS)],
        JointType::Fixed => Vec::new(),
        JointType::Planar => vec![(Prismatic, X_AXIS), (Prismatic, Y_AXIS), (Revolute, Z_AXIS)],
        JointType::Cylindrical => vec![(Revolute, Z_AXIS), (Prismatic, Z_AXIS)],
    }
}

fn joint_dof_steps(joint: &JointDof) -> Vec<(CoreJointKind, [f64; 3])> {
    use CoreJointKind::{Prismatic, Revolute};
    match joint {
        JointDof::Revolute { axis, .. } => vec![(Revolute, *axis)],
        JointDof::Prismatic { axis, .. } => vec![(Prismatic, *axis)],
        JointDof::Spherical => vec![(Revolute, X_AXIS), (Revolute, Y_AXIS), (Revolute, Z_AXIS)],
        JointDof::Free => vec![
            (Prismatic, X_AXIS),
            (Prismatic, Y_AXIS),
            (Prismatic, Z_AXIS),
            (Revolute, X_AXIS),
            (Revolute, Y_AXIS),
            (Revolute, Z_AXIS),
        ],
        JointDof::Fixed => Vec::new(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultibodySystem
// ─────────────────────────────────────────────────────────────────────────────

impl MultibodySystem {
    /// Build the core model; returns it with the core index of every link.
    fn core_model(&self) -> Result<(CoreModel, Vec<usize>)> {
        let mut model = CoreModel::new(mat3_identity(), [0.0; 3]);
        let mut link_body: Vec<usize> = Vec::with_capacity(self.links.len());
        for (i, link) in self.links.iter().enumerate() {
            let inertia = link.inertia_tensor();
            let index = match &link.parent_joint {
                None => JointSequence {
                    parent: None,
                    pre_rotation: mat3_identity(),
                    pre_translation: link.joint_anchor,
                    steps: &[],
                    mass: link.mass,
                    com: link.local_com,
                    inertia,
                    post_rotation: mat3_identity(),
                }
                .push_into(&mut model)?,
                Some(joint) => {
                    let parent = if joint.parent_body < i {
                        link_body.get(joint.parent_body).copied()
                    } else {
                        None
                    };
                    JointSequence {
                        parent,
                        pre_rotation: joint.local_frame_parent,
                        pre_translation: link.joint_anchor,
                        steps: &joint_type_steps(joint.joint_type),
                        mass: link.mass,
                        com: link.local_com,
                        inertia,
                        post_rotation: mat3_transpose(joint.local_frame_child),
                    }
                    .push_into(&mut model)?
                }
            };
            link_body.push(index);
        }
        if model.n_dof() != self.n_dof() {
            return Err(Error::ArticulatedBodyError {
                reason: format!(
                    "joint DoF layout ({}) does not match the stored coordinates ({})",
                    model.n_dof(),
                    self.n_dof()
                ),
            });
        }
        Ok((model, link_body))
    }

    /// Forward kinematics: world-space centre of mass of every link at the
    /// stored `positions`.
    pub fn forward_kinematics(&self) -> Result<Vec<[f64; 3]>> {
        let (model, link_body) = self.core_model()?;
        let kin = model.kinematics(&self.positions)?;
        Ok(link_body.iter().map(|&b| kin.com[b]).collect())
    }

    /// Linear-velocity Jacobian (3 × `n_dof`, row-major) of the centre of mass
    /// of link `link_idx` at the stored `positions`.
    pub fn jacobian(&self, link_idx: usize) -> Result<Vec<Vec<f64>>> {
        let (model, link_body) = self.core_model()?;
        let body = *link_body
            .get(link_idx)
            .ok_or_else(|| Error::ArticulatedBodyError {
                reason: format!(
                    "link index {link_idx} out of range ({} links)",
                    self.links.len()
                ),
            })?;
        let kin = model.kinematics(&self.positions)?;
        model.point_jacobian(&kin, body, kin.com[body])
    }

    /// Joint-space mass matrix `M(q)` (CRBA) at the stored `positions`.
    pub fn mass_matrix(&self) -> Result<Vec<Vec<f64>>> {
        let (model, _) = self.core_model()?;
        let kin = model.kinematics(&self.positions)?;
        Ok(model.crba(&kin))
    }

    /// Coriolis and centrifugal generalised forces `C(q, q̇)·q̇` at the stored
    /// state, computed as `RNEA(q, q̇, 0)` without gravity.
    pub fn coriolis_vector(&self) -> Result<Vec<f64>> {
        let (model, _) = self.core_model()?;
        let kin = model.kinematics(&self.positions)?;
        let zeros = vec![0.0; model.n_dof()];
        model.rnea(&kin, &self.velocities, &zeros, [0.0; 3])
    }

    /// Generalised gravity forces `G(q)` (torque needed to hold the pose),
    /// computed as `RNEA(q, 0, 0)` with `self.gravity`.
    pub fn gravity_vector(&self) -> Result<Vec<f64>> {
        let (model, _) = self.core_model()?;
        let kin = model.kinematics(&self.positions)?;
        let zeros = vec![0.0; model.n_dof()];
        model.rnea(&kin, &zeros, &zeros, self.gravity)
    }

    /// Forward dynamics at the stored state: solves
    /// `M(q) q̈ = τ − C(q, q̇) q̇ − G(q)` by Cholesky factorisation.
    pub fn forward_dynamics(&self, tau: &[f64]) -> Result<Vec<f64>> {
        let (model, _) = self.core_model()?;
        if tau.len() != model.n_dof() {
            return Err(Error::ArticulatedBodyError {
                reason: format!(
                    "tau has length {} but the system has {} degrees of freedom",
                    tau.len(),
                    model.n_dof()
                ),
            });
        }
        let kin = model.kinematics(&self.positions)?;
        let zeros = vec![0.0; model.n_dof()];
        let bias = model.rnea(&kin, &self.velocities, &zeros, self.gravity)?;
        let rhs: Vec<f64> = tau.iter().zip(bias.iter()).map(|(t, b)| t - b).collect();
        cholesky_solve(&model.crba(&kin), &rhs)
    }

    /// Inverse dynamics (recursive Newton–Euler): joint torques realising the
    /// accelerations `ddq` at configuration `q` and velocity `dq`.
    pub fn inverse_dynamics(&self, q: &[f64], dq: &[f64], ddq: &[f64]) -> Result<Vec<f64>> {
        let (model, _) = self.core_model()?;
        let kin = model.kinematics(q)?;
        model.rnea(&kin, dq, ddq, self.gravity)
    }

    /// Integrate one step with semi-implicit Euler given torques `tau` and
    /// time step `dt` (s).
    pub fn integrate_euler(&mut self, tau: &[f64], dt: f64) -> Result<()> {
        let qdd = self.forward_dynamics(tau)?;
        for (vel_i, (pos_i, qdd_i)) in self
            .velocities
            .iter_mut()
            .zip(self.positions.iter_mut().zip(qdd.iter()))
        {
            *vel_i += qdd_i * dt;
            *pos_i += *vel_i * dt;
        }
        self.accelerations = qdd;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultibodyChain-based algorithms
// ─────────────────────────────────────────────────────────────────────────────

impl MultibodyChain {
    /// Build the core model of this serial chain (fixed base at the origin).
    fn core_model(&self) -> Result<CoreModel> {
        let mut model = CoreModel::new(mat3_identity(), [0.0; 3]);
        let mut parent = None;
        for elem in &self.elements {
            let last = JointSequence {
                parent,
                pre_rotation: mat3_identity(),
                pre_translation: elem.link.joint_offset,
                steps: &joint_dof_steps(&elem.joint),
                mass: elem.link.mass,
                com: elem.link.com_offset,
                inertia: diag_inertia(elem.link.inertia),
                post_rotation: mat3_identity(),
            }
            .push_into(&mut model)?;
            parent = Some(last);
        }
        Ok(model)
    }
}

impl RecursiveNewtonEuler {
    /// Inverse dynamics: joint torques for the chain's `q`, `qd`, `qdd` under
    /// `self.gravity` (N·m or N per DoF).
    pub fn inverse_dynamics(&self) -> Result<Vec<f64>> {
        let model = self.chain.core_model()?;
        let kin = model.kinematics(&self.chain.q)?;
        model.rnea(&kin, &self.chain.qd, &self.chain.qdd, self.gravity)
    }
}

impl JointSpaceInertia {
    /// Joint-space inertia matrix `H(q)` at the chain's `q` (CRBA), as a flat
    /// row-major `n × n` vector.
    pub fn compute(&self) -> Result<Vec<f64>> {
        let model = self.chain.core_model()?;
        let kin = model.kinematics(&self.chain.q)?;
        Ok(model.crba(&kin).into_iter().flatten().collect())
    }
    /// Diagonal of `H(q)`.
    pub fn diagonal(&self) -> Result<Vec<f64>> {
        let n = self.chain.total_dof();
        let h = self.compute()?;
        Ok((0..n).map(|i| h[i * n + i]).collect())
    }
}

impl FeatherstoneAba {
    /// Forward dynamics (articulated-body algorithm): joint accelerations for
    /// the applied joint `torques` at the chain's `q`, `qd` under
    /// `self.gravity`.  The result is also stored in `chain.qdd`.
    pub fn forward_dynamics(&mut self, torques: &[f64]) -> Result<Vec<f64>> {
        let model = self.chain.core_model()?;
        let kin = model.kinematics(&self.chain.q)?;
        let qdd = model.aba(&kin, &self.chain.qd, torques, self.gravity)?;
        self.chain.qdd = qdd.clone();
        Ok(qdd)
    }
    /// Integrate one time step `dt` (s) using semi-implicit Euler.
    pub fn integrate(&mut self, torques: &[f64], dt: f64) -> Result<()> {
        let qdd = self.forward_dynamics(torques)?;
        for ((qd, q), a) in self
            .chain
            .qd
            .iter_mut()
            .zip(self.chain.q.iter_mut())
            .zip(qdd.iter())
        {
            *qd += a * dt;
            *q += *qd * dt;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatherstoneABA (Body-based serial chain)
// ─────────────────────────────────────────────────────────────────────────────

impl FeatherstoneABA {
    /// Set joint locations and child centre-of-mass offsets (one per joint).
    pub fn with_geometry(
        mut self,
        joint_offsets: Vec<[f64; 3]>,
        com_offsets: Vec<[f64; 3]>,
    ) -> Result<Self> {
        let n = self.n_joints();
        if joint_offsets.len() != n || com_offsets.len() != n {
            return Err(Error::ArticulatedBodyError {
                reason: format!(
                    "expected {n} joint and CoM offsets, got {} and {}",
                    joint_offsets.len(),
                    com_offsets.len()
                ),
            });
        }
        self.joint_offsets = joint_offsets;
        self.com_offsets = com_offsets;
        Ok(self)
    }

    /// Number of joints (`bodies.len() − 1`).
    pub fn n_joints(&self) -> usize {
        self.bodies.len().saturating_sub(1)
    }

    fn core_model(&self) -> Result<CoreModel> {
        let n = self.n_joints();
        let Some(base) = self.bodies.first() else {
            return Ok(CoreModel::new(mat3_identity(), [0.0; 3]));
        };
        if self.axes.len() != n || self.joint_offsets.len() != n || self.com_offsets.len() != n {
            return Err(Error::ArticulatedBodyError {
                reason: format!(
                    "{n} joints need {n} axes, joint offsets and CoM offsets (got {}, {}, {})",
                    self.axes.len(),
                    self.joint_offsets.len(),
                    self.com_offsets.len()
                ),
            });
        }
        let mut model = CoreModel::new(quat_to_mat3(base.orientation)?, base.position);
        let mut parent = None;
        for (k, body) in self.bodies.iter().skip(1).enumerate() {
            let core = CoreBody::massless(
                parent,
                mat3_identity(),
                self.joint_offsets[k],
                CoreJointKind::Revolute,
                self.axes[k],
            )
            .with_mass(
                body.mass,
                self.com_offsets[k],
                body.inertia_matrix(),
                mat3_identity(),
            );
            parent = Some(model.push(core)?);
        }
        Ok(model)
    }

    /// Forward dynamics: joint accelerations for joint velocities `qd`, the
    /// stored joint angles `q`, `applied_torques` and `gravity`.
    pub fn forward_dynamics(&self, qd: &[f64]) -> Result<Vec<f64>> {
        let model = self.core_model()?;
        let kin = model.kinematics(&self.q)?;
        model.aba(&kin, qd, &self.applied_torques, self.gravity)
    }

    /// Integrate one time step with semi-implicit Euler (`q̇ += q̈ dt`,
    /// `q += q̇ dt`), then update every body's world position (centre of
    /// mass), orientation, linear and angular velocity from forward
    /// kinematics.  `qd` is resized to the number of joints if necessary.
    pub fn step(&mut self, dt: f64, qd: &mut Vec<f64>) -> Result<()> {
        let n_joints = self.n_joints();
        qd.resize(n_joints, 0.0);
        let qdd = self.forward_dynamics(qd)?;
        for ((v, q), a) in qd.iter_mut().zip(self.q.iter_mut()).zip(qdd.iter()) {
            *v += a * dt;
            *q += *v * dt;
        }
        self.sync_bodies(qd)
    }

    /// Write the world-frame state implied by `q` and `qd` into
    /// `bodies[1..]`.
    pub fn sync_bodies(&mut self, qd: &[f64]) -> Result<()> {
        let model = self.core_model()?;
        let kin = model.kinematics(&self.q)?;
        let twists = model.velocities(&kin, qd)?;
        for (k, body) in self.bodies.iter_mut().skip(1).enumerate() {
            let twist = twists[k];
            body.position = kin.com[k];
            body.orientation = mat3_to_quat(kin.rotation[k]);
            body.angular_velocity = [twist[0], twist[1], twist[2]];
            body.velocity = point_velocity(twist, kin.com[k]);
        }
        Ok(())
    }
}
