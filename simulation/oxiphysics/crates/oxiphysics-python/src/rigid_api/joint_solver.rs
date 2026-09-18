// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Position-based joint projection for the `rigid` Python world.
//!
//! The engine's [`PhysicsWorld::step`](oxiphysics::rigid::world::PhysicsWorld::step)
//! integrates the bodies and resolves the collider contacts. Afterwards this
//! module projects the joint constraints on the integrated poses (Müller et
//! al., "Detailed Rigid Body Simulation with Extended Position Based
//! Dynamics", 2020):
//!
//! * positional rows move the two world anchors `x + R(q)·anchor` towards each
//!   other. The rows of one joint are solved as a block with the coupled
//!   effective inverse mass `K = Σ (m⁻¹ I₃ − [r]× I⁻¹ [r]×)` (restricted to the
//!   constrained directions), so the correction is split between translation
//!   and rotation of both bodies and the linearised error vanishes in one
//!   iteration even for strongly anisotropic bodies;
//! * angular rows rotate the bodies with `K = I_a⁻¹ + I_b⁻¹`;
//! * after all iterations the velocities are made consistent with the
//!   projected poses (`v += Δx / h`, `ω += Δθ / h`), which is exactly the
//!   PBD velocity update `v = (x − x_prev) / h`.
//!
//! Static and kinematic bodies have zero inverse mass and inertia and are
//! never written.

use super::joints::PyJointType;
use oxiphysics::core::BodyHandle;
use oxiphysics::core::math::{Mat3, Quat, Real, Vec3};
use oxiphysics::rigid::{BodyState, BodyType, RigidBody, RigidBodySet};
use std::collections::HashMap;

/// Corrections below this magnitude (m or rad) are skipped.
const MIN_CORRECTION: Real = 1e-12;
/// Generalised inverse masses below this are treated as immovable.
const MIN_INV_MASS: Real = 1e-15;

/// Reasons a joint cannot be attached to its bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JointFrameError {
    /// Hinge / slider axis is zero or not finite.
    Axis,
    /// An anchor is not finite.
    Anchor,
    /// Distance-joint target is negative or not finite.
    TargetDistance,
    /// Stiffness is outside `[0, 1]` or not finite.
    Stiffness,
}

impl std::fmt::Display for JointFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Axis => "joint axis must be a finite, non-zero vector",
            Self::Anchor => "joint anchors must be finite",
            Self::TargetDistance => "target_distance must be finite and >= 0",
            Self::Stiffness => "stiffness must be finite and within [0, 1]",
        };
        f.write_str(msg)
    }
}

/// A joint expressed in the local frames of its two engine bodies.
#[derive(Debug, Clone)]
pub(crate) struct JointFrame {
    /// Engine handle of body A.
    pub(crate) body_a: BodyHandle,
    /// Engine handle of body B.
    pub(crate) body_b: BodyHandle,
    /// Joint type.
    pub(crate) kind: PyJointType,
    /// Anchor on body A in A's local frame.
    pub(crate) local_anchor_a: Vec3,
    /// Anchor on body B in B's local frame.
    pub(crate) local_anchor_b: Vec3,
    /// Joint axis in A's local frame (unit length).
    pub(crate) local_axis_a: Vec3,
    /// Joint axis in B's local frame (unit length).
    pub(crate) local_axis_b: Vec3,
    /// Relative rotation `q_a⁻¹ · q_b` captured when the joint was attached.
    pub(crate) rest_rotation: Quat,
    /// Target anchor distance (distance joints).
    pub(crate) target_distance: Real,
    /// Fraction of the remaining error removed per solver iteration.
    pub(crate) stiffness: Real,
    /// Whether the joint participates in the solve.
    pub(crate) enabled: bool,
}

/// World-space description of a joint at the moment it is attached.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointSpec {
    /// Joint type.
    pub(crate) kind: PyJointType,
    /// World-space anchor on body A.
    pub(crate) anchor_a: [Real; 3],
    /// World-space anchor on body B.
    pub(crate) anchor_b: [Real; 3],
    /// World-space hinge / slider axis.
    pub(crate) axis: [Real; 3],
    /// Target anchor distance (distance joints).
    pub(crate) target_distance: Real,
    /// Per-iteration stiffness in `[0, 1]`.
    pub(crate) stiffness: Real,
    /// Whether the joint is active.
    pub(crate) enabled: bool,
}

fn vec3(a: [Real; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn finite3(a: [Real; 3]) -> bool {
    a.iter().all(|v| v.is_finite())
}

impl JointFrame {
    /// Express `spec` in the current local frames of bodies `a` and `b`.
    pub(crate) fn capture(
        spec: &JointSpec,
        handle_a: BodyHandle,
        a: &RigidBody,
        handle_b: BodyHandle,
        b: &RigidBody,
    ) -> Result<Self, JointFrameError> {
        if !finite3(spec.anchor_a) || !finite3(spec.anchor_b) {
            return Err(JointFrameError::Anchor);
        }
        if !spec.stiffness.is_finite() || !(0.0..=1.0).contains(&spec.stiffness) {
            return Err(JointFrameError::Stiffness);
        }
        if spec.kind == PyJointType::Distance
            && (!spec.target_distance.is_finite() || spec.target_distance < 0.0)
        {
            return Err(JointFrameError::TargetDistance);
        }
        let needs_axis = matches!(spec.kind, PyJointType::Hinge | PyJointType::Slider);
        let axis = vec3(spec.axis);
        let axis_len = axis.norm();
        let world_axis = if axis_len.is_finite() && axis_len > 1e-12 {
            axis / axis_len
        } else if needs_axis {
            return Err(JointFrameError::Axis);
        } else {
            Vec3::y()
        };
        let qa = a.transform.rotation;
        let qb = b.transform.rotation;
        let qa_inv = qa.inverse();
        let qb_inv = qb.inverse();
        Ok(Self {
            body_a: handle_a,
            body_b: handle_b,
            kind: spec.kind,
            local_anchor_a: qa_inv * (vec3(spec.anchor_a) - a.transform.position),
            local_anchor_b: qb_inv * (vec3(spec.anchor_b) - b.transform.position),
            local_axis_a: qa_inv * world_axis,
            local_axis_b: qb_inv * world_axis,
            rest_rotation: qa_inv * qb,
            target_distance: spec.target_distance,
            stiffness: spec.stiffness,
            enabled: spec.enabled,
        })
    }
}

/// Mass properties and pose of one body as seen by the joint solver.
struct Pose {
    position: Vec3,
    rotation: Quat,
    inv_mass: Real,
    inv_inertia: Mat3,
}

fn pose(bodies: &RigidBodySet, h: BodyHandle) -> Option<Pose> {
    let b = bodies.get(h)?;
    let dynamic = b.body_type == BodyType::Dynamic;
    Some(Pose {
        position: b.transform.position,
        rotation: b.transform.rotation,
        inv_mass: if dynamic { b.inverse_mass } else { 0.0 },
        inv_inertia: if dynamic {
            b.world_inverse_inertia
        } else {
            Mat3::zeros()
        },
    })
}

/// Translate and rotate a dynamic body (no-op for any other body type).
fn displace(bodies: &mut RigidBodySet, h: BodyHandle, dx: Vec3, dtheta: Vec3) {
    if let Some(b) = bodies.get_mut(h)
        && b.body_type == BodyType::Dynamic
    {
        b.transform.position += dx;
        if dtheta.norm_squared() > 0.0 {
            b.transform.rotation = Quat::from_scaled_axis(dtheta) * b.transform.rotation;
            b.update_world_inertia();
        }
    }
}

/// Directions in which a constraint row acts.
#[derive(Debug, Clone, Copy)]
enum Subspace {
    /// All three directions.
    Full,
    /// The plane spanned by two orthonormal vectors.
    Plane(Vec3, Vec3),
    /// A single unit direction.
    Line(Vec3),
}

impl Subspace {
    /// Plane perpendicular to the unit vector `axis`.
    fn perpendicular_to(axis: &Vec3) -> Self {
        let reference = if axis.x.abs() < 0.9 {
            Vec3::x()
        } else {
            Vec3::y()
        };
        let t1 = axis.cross(&reference).normalize();
        let t2 = axis.cross(&t1);
        Self::Plane(t1, t2)
    }

    /// Solve `K p = e` for `p` restricted to this subspace (`p = B (BᵀKB)⁻¹ Bᵀ e`).
    ///
    /// Returns `None` when the restricted effective-mass matrix is singular
    /// (both bodies immovable in these directions).
    fn solve(&self, k: &Mat3, e: &Vec3) -> Option<Vec3> {
        match self {
            Self::Full => {
                if k.trace() < MIN_INV_MASS {
                    return None;
                }
                k.try_inverse().map(|inv| inv * e)
            }
            Self::Plane(t1, t2) => {
                let k11 = t1.dot(&(k * t1));
                let k12 = t1.dot(&(k * t2));
                let k21 = t2.dot(&(k * t1));
                let k22 = t2.dot(&(k * t2));
                let det = k11 * k22 - k12 * k21;
                if det.abs() < MIN_INV_MASS * MIN_INV_MASS || (k11 + k22) < MIN_INV_MASS {
                    return None;
                }
                let e1 = t1.dot(e);
                let e2 = t2.dot(e);
                let l1 = (k22 * e1 - k12 * e2) / det;
                let l2 = (k11 * e2 - k21 * e1) / det;
                Some(t1 * l1 + t2 * l2)
            }
            Self::Line(n) => {
                let w = n.dot(&(k * n));
                if w < MIN_INV_MASS {
                    return None;
                }
                Some(n * (n.dot(e) / w))
            }
        }
    }
}

/// Skew-symmetric matrix `[r]×` with `[r]× v = r × v`.
fn cross_matrix(r: &Vec3) -> Mat3 {
    Mat3::new(0.0, -r.z, r.y, r.z, 0.0, -r.x, -r.y, r.x, 0.0)
}

/// Point effective inverse-mass matrix `K = m⁻¹ I₃ − [r]× I⁻¹ [r]×` of a body
/// at the lever arm `r`: an impulse `p` at the point moves it by `K p`.
fn point_inverse_mass(body: &Pose, r: &Vec3) -> Mat3 {
    let rx = cross_matrix(r);
    Mat3::identity() * body.inv_mass - rx * body.inv_inertia * rx
}

/// Remove the world-space anchor separation `error = p_a − p_b` within
/// `subspace` (block solve with the coupled linear + angular effective mass).
fn project_positional(
    bodies: &mut RigidBodySet,
    frame: &JointFrame,
    a: &Pose,
    b: &Pose,
    error: Vec3,
    subspace: Subspace,
) {
    if error.norm() < MIN_CORRECTION {
        return;
    }
    let r_a = a.rotation * frame.local_anchor_a;
    let r_b = b.rotation * frame.local_anchor_b;
    let k = point_inverse_mass(a, &r_a) + point_inverse_mass(b, &r_b);
    let Some(p) = subspace.solve(&k, &error) else {
        return;
    };
    displace(
        bodies,
        frame.body_a,
        -p * a.inv_mass,
        -(a.inv_inertia * r_a.cross(&p)),
    );
    displace(
        bodies,
        frame.body_b,
        p * b.inv_mass,
        b.inv_inertia * r_b.cross(&p),
    );
}

/// Remove the relative rotation `dphi` (world-space rotation vector by which
/// body B deviates from its target orientation relative to body A) within
/// `subspace`.
fn project_angular(
    bodies: &mut RigidBodySet,
    frame: &JointFrame,
    a: &Pose,
    b: &Pose,
    dphi: Vec3,
    subspace: Subspace,
) {
    if dphi.norm() < MIN_CORRECTION {
        return;
    }
    let k = a.inv_inertia + b.inv_inertia;
    let Some(p) = subspace.solve(&k, &dphi) else {
        return;
    };
    displace(bodies, frame.body_a, Vec3::zeros(), a.inv_inertia * p);
    displace(bodies, frame.body_b, Vec3::zeros(), -(b.inv_inertia * p));
}

/// Rotation vector taking unit vector `from` onto unit vector `to`.
fn alignment_rotation(from: &Vec3, to: &Vec3) -> Vec3 {
    let cross = from.cross(to);
    let sin = cross.norm();
    let cos = from.dot(to);
    if sin < MIN_CORRECTION {
        return Vec3::zeros();
    }
    cross * (sin.atan2(cos) / sin)
}

/// Rotation vector by which B deviates from `q_a · rest_rotation`.
fn orientation_error(frame: &JointFrame, a: &Pose, b: &Pose) -> Vec3 {
    let target = a.rotation * frame.rest_rotation;
    (b.rotation * target.inverse()).scaled_axis()
}

/// World anchor separation `p_a − p_b`.
fn anchor_error(frame: &JointFrame, a: &Pose, b: &Pose) -> Vec3 {
    let pa = a.position + a.rotation * frame.local_anchor_a;
    let pb = b.position + b.rotation * frame.local_anchor_b;
    pa - pb
}

/// One projection pass of a single joint.
fn project_joint(bodies: &mut RigidBodySet, frame: &JointFrame) {
    let s = frame.stiffness;
    // Angular rows first, so the positional rows see the corrected rotations.
    if !matches!(frame.kind, PyJointType::BallSocket | PyJointType::Distance) {
        let (Some(a), Some(b)) = (pose(bodies, frame.body_a), pose(bodies, frame.body_b)) else {
            return;
        };
        let (dphi, subspace) = if frame.kind == PyJointType::Hinge {
            let axis_a = a.rotation * frame.local_axis_a;
            let axis_b = b.rotation * frame.local_axis_b;
            (
                alignment_rotation(&axis_a, &axis_b),
                Subspace::perpendicular_to(&axis_a),
            )
        } else {
            (orientation_error(frame, &a, &b), Subspace::Full)
        };
        project_angular(bodies, frame, &a, &b, dphi * s, subspace);
    }
    let (Some(a), Some(b)) = (pose(bodies, frame.body_a), pose(bodies, frame.body_b)) else {
        return;
    };
    let d = anchor_error(frame, &a, &b);
    let (error, subspace) = match frame.kind {
        PyJointType::BallSocket | PyJointType::Hinge | PyJointType::Fixed => (d, Subspace::Full),
        PyJointType::Slider => {
            let axis = a.rotation * frame.local_axis_a;
            (d, Subspace::perpendicular_to(&axis))
        }
        PyJointType::Distance => {
            let len = d.norm();
            if len < MIN_CORRECTION {
                // Coincident anchors: the separation direction is undefined.
                return;
            }
            let n = d / len;
            (n * (len - frame.target_distance), Subspace::Line(n))
        }
    };
    project_positional(bodies, frame, &a, &b, error * s, subspace);
}

/// Whether a body moves under the joint solve and is currently active.
fn is_active_dynamic(b: &RigidBody) -> bool {
    b.body_type == BodyType::Dynamic && b.state != BodyState::Sleeping
}

/// Project all enabled joints `iterations` times and update the velocities of
/// the moved bodies for a step of length `h`.
///
/// A sleeping body linked to an awake dynamic body is woken first; joints
/// whose bodies are both asleep (or both non-dynamic) are skipped.
pub(crate) fn solve_joints(
    bodies: &mut RigidBodySet,
    frames: &[&JointFrame],
    h: Real,
    iterations: u32,
) {
    if h <= 0.0 || !h.is_finite() || iterations == 0 {
        return;
    }
    let mut active: Vec<&JointFrame> = Vec::with_capacity(frames.len());
    for frame in frames.iter().copied().filter(|f| f.enabled) {
        let a_awake = bodies.get(frame.body_a).map(is_active_dynamic);
        let b_awake = bodies.get(frame.body_b).map(is_active_dynamic);
        let (Some(a_awake), Some(b_awake)) = (a_awake, b_awake) else {
            continue;
        };
        if !a_awake && !b_awake {
            continue;
        }
        for h_body in [frame.body_a, frame.body_b] {
            if let Some(body) = bodies.get_mut(h_body)
                && body.body_type == BodyType::Dynamic
                && body.state == BodyState::Sleeping
            {
                body.wake_up();
            }
        }
        active.push(frame);
    }
    if active.is_empty() {
        return;
    }
    // Poses before the projection, for the velocity update.
    let mut before: HashMap<BodyHandle, (Vec3, Quat)> = HashMap::new();
    for frame in &active {
        for h_body in [frame.body_a, frame.body_b] {
            if let Some(b) = bodies.get(h_body)
                && b.body_type == BodyType::Dynamic
            {
                before
                    .entry(h_body)
                    .or_insert((b.transform.position, b.transform.rotation));
            }
        }
    }
    for _ in 0..iterations {
        for frame in &active {
            project_joint(bodies, frame);
        }
    }
    let inv_h = 1.0 / h;
    for (h_body, (x0, q0)) in before {
        if let Some(b) = bodies.get_mut(h_body) {
            let dx = b.transform.position - x0;
            let dtheta = (b.transform.rotation * q0.inverse()).scaled_axis();
            b.velocity += dx * inv_h;
            b.angular_velocity += dtheta * inv_h;
        }
    }
}

/// World-space anchor separation `|p_a − p_b|` of a joint (diagnostics).
pub(crate) fn anchor_gap(bodies: &RigidBodySet, frame: &JointFrame) -> Option<Real> {
    let a = pose(bodies, frame.body_a)?;
    let b = pose(bodies, frame.body_b)?;
    Some(anchor_error(frame, &a, &b).norm())
}
