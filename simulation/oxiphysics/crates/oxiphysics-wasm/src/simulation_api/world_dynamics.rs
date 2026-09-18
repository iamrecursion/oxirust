// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `WasmWorld` mass properties and joint enforcement.
//!
//! Joints are enforced with position-based projection after the bodies have
//! been integrated: every iteration moves and rotates the two bodies of each
//! joint along the constraint gradient, weighted by their generalised inverse
//! masses `1/m + (r x n)^T I^-1 (r x n)`, and the corrections are turned into
//! velocities (`v += dx / dt`, `w += dtheta / dt`) once all iterations ran.

use std::collections::BTreeMap;

use super::types::{WasmColliderShape, WasmRigidBody};
use super::types_3::{WasmPhysicsEvent, WasmWorld};
use super::types_4::{WasmBodyType, WasmJoint, WasmJointType, WasmRigidBodyHandle};
use super::vec_math::{
    M3, M3_ZERO, Q4, V3, m_rotated_diag, m_vec, q_apply_rotation_vector, q_conj, q_mul, q_rotate,
    q_to_rotation_vector, v_add, v_cross, v_dot, v_length, v_madd, v_neg, v_scale, v_sub,
};

/// Constraint error below which the joint iterations stop early.
const JOINT_TOLERANCE: f64 = 1e-12;

/// Volume and body-frame principal inertia per unit mass of a collider shape
/// centred on the body origin, or `None` for shapes without a volume.
fn unit_inertia(shape: &WasmColliderShape) -> Option<(f64, V3)> {
    match shape {
        WasmColliderShape::Sphere { radius } => {
            let r = radius.abs();
            let i = 0.4 * r * r;
            Some((4.0 / 3.0 * std::f64::consts::PI * r * r * r, [i, i, i]))
        }
        WasmColliderShape::Box { half_extents } => {
            let [hx, hy, hz] = half_extents.map(f64::abs);
            Some((
                8.0 * hx * hy * hz,
                [
                    (hy * hy + hz * hz) / 3.0,
                    (hx * hx + hz * hz) / 3.0,
                    (hx * hx + hy * hy) / 3.0,
                ],
            ))
        }
        WasmColliderShape::Cylinder {
            half_height,
            radius,
        } => {
            let (h, r) = (2.0 * half_height.abs(), radius.abs());
            let side = (3.0 * r * r + h * h) / 12.0;
            Some((std::f64::consts::PI * r * r * h, [side, 0.5 * r * r, side]))
        }
        WasmColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let (h, r) = (2.0 * half_height.abs(), radius.abs());
            let pi = std::f64::consts::PI;
            let v_cyl = pi * r * r * h;
            let v_caps = 4.0 / 3.0 * pi * r * r * r;
            let volume = v_cyl + v_caps;
            if volume <= 0.0 {
                return None;
            }
            let (fc, fs) = (v_cyl / volume, v_caps / volume);
            let axial = fc * 0.5 * r * r + fs * 0.4 * r * r;
            let side = fc * (h * h / 12.0 + r * r / 4.0)
                + fs * (0.4 * r * r + h * h / 4.0 + 3.0 * h * r / 8.0);
            Some((volume, [side, axial, side]))
        }
        WasmColliderShape::Cone {
            half_height,
            radius,
        } => {
            // Solid cone about its centroid (axis Y).
            let (h, r) = (2.0 * half_height.abs(), radius.abs());
            let axial = 0.3 * r * r;
            let side = 3.0 / 20.0 * r * r + 3.0 / 80.0 * h * h;
            Some((std::f64::consts::PI * r * r * h / 3.0, [side, axial, side]))
        }
        WasmColliderShape::Plane { .. } | WasmColliderShape::ConvexHull { .. } => None,
    }
}

/// World-space inverse inertia of a body (zero for non-dynamic bodies).
pub(super) fn world_inv_inertia(body: &WasmRigidBody) -> M3 {
    if body.body_type == WasmBodyType::Dynamic {
        m_rotated_diag(body.rotation, body.inv_inertia)
    } else {
        M3_ZERO
    }
}

/// Inverse mass used by the solvers (zero for non-dynamic bodies).
pub(super) fn solver_inv_mass(body: &WasmRigidBody) -> f64 {
    if body.body_type == WasmBodyType::Dynamic {
        body.inv_mass
    } else {
        0.0
    }
}

impl WasmWorld {
    /// Recompute the rotational inertia of a dynamic body from its colliders.
    ///
    /// The mass is distributed over the volumetric colliders in proportion to
    /// their volumes (all colliders are centred on the body origin).  A body
    /// without volumetric colliders keeps the inertia of a unit-radius solid
    /// sphere (`0.4 m`).
    pub(super) fn refresh_inertia(&mut self, handle: WasmRigidBodyHandle) {
        let parts: Vec<(f64, V3)> = self
            .colliders
            .values()
            .filter(|c| c.body == Some(handle))
            .filter_map(|c| unit_inertia(&c.shape))
            .filter(|(volume, _)| volume.is_finite() && *volume > 0.0)
            .collect();
        let Some(body) = self.bodies.get_mut(&handle.0) else {
            return;
        };
        if body.body_type != WasmBodyType::Dynamic {
            return;
        }
        let total_volume: f64 = parts.iter().map(|(v, _)| v).sum();
        let inertia = if total_volume > 0.0 {
            let mut sum = [0.0; 3];
            for (volume, unit) in &parts {
                let mass = body.mass * volume / total_volume;
                sum = v_madd(sum, *unit, mass);
            }
            sum
        } else {
            [0.4 * body.mass; 3]
        };
        body.inv_inertia = inertia.map(|i| if i > 1e-15 { 1.0 / i } else { 0.0 });
    }

    /// Enforce all joints by position projection (see the module docs).
    ///
    /// Runs `config.position_iters` (at least one) iterations over the joints
    /// in handle order and converts the corrections into velocities.  Sleeping
    /// bodies that a joint moves are woken up.
    pub(super) fn solve_joints(&mut self, dt: f64) {
        if self.joints.is_empty() || !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let mut start: BTreeMap<u32, (V3, Q4)> = BTreeMap::new();
        for joint in self.joints.values() {
            for id in [joint.body_a.0, joint.body_b.0] {
                if let Some(body) = self.bodies.get(&id) {
                    start.entry(id).or_insert((body.position, body.rotation));
                }
            }
        }
        let iterations = self.config.position_iters.max(1);
        for _ in 0..iterations {
            let mut max_error = 0.0_f64;
            for joint in self.joints.values() {
                max_error = max_error.max(project_joint(&mut self.bodies, joint));
            }
            if max_error < JOINT_TOLERANCE {
                break;
            }
        }
        let inv_dt = 1.0 / dt;
        for (id, (x0, q0)) in start {
            let Some(body) = self.bodies.get_mut(&id) else {
                continue;
            };
            if body.body_type != WasmBodyType::Dynamic {
                continue;
            }
            let dx = v_sub(body.position, x0);
            let dtheta = q_to_rotation_vector(q_mul(body.rotation, q_conj(q0)));
            if v_length(dx) <= 1e-15 && v_length(dtheta) <= 1e-15 {
                continue;
            }
            body.linear_vel = v_madd(body.linear_vel, dx, inv_dt);
            body.angular_vel = v_madd(body.angular_vel, dtheta, inv_dt);
            if body.sleeping {
                body.sleeping = false;
                body.sleep_time = 0.0;
                self.events.push(WasmPhysicsEvent::BodyWoke {
                    body: WasmRigidBodyHandle(id),
                });
            }
        }
    }
}

/// Pose and mass properties of one joint body.
struct JointBody {
    position: V3,
    rotation: Q4,
    inv_mass: f64,
    inv_inertia: M3,
}

impl JointBody {
    fn of(body: &WasmRigidBody) -> Self {
        Self {
            position: body.position,
            rotation: body.rotation,
            inv_mass: solver_inv_mass(body),
            inv_inertia: world_inv_inertia(body),
        }
    }

    /// Generalised inverse mass for moving the point at lever `r` along `n`.
    fn weight(&self, r: V3, n: V3) -> f64 {
        let rn = v_cross(r, n);
        self.inv_mass + v_dot(rn, m_vec(&self.inv_inertia, rn))
    }
}

/// Project one joint once; returns the magnitude of the corrected error.
fn project_joint(bodies: &mut BTreeMap<u32, WasmRigidBody>, joint: &WasmJoint) -> f64 {
    let (Some(a), Some(b)) = (bodies.get(&joint.body_a.0), bodies.get(&joint.body_b.0)) else {
        return 0.0;
    };
    let (a, b) = (JointBody::of(a), JointBody::of(b));
    let r_a = q_rotate(a.rotation, joint.anchor_a);
    let r_b = q_rotate(b.rotation, joint.anchor_b);
    let delta = v_sub(v_add(b.position, r_b), v_add(a.position, r_a));
    let length = v_length(delta);
    if length <= 1e-12 {
        // Coincident anchors: satisfied for a ball joint, and the direction of
        // a distance correction is undefined.
        return 0.0;
    }
    let n = v_scale(delta, 1.0 / length);
    let error = match joint.joint_type {
        WasmJointType::Ball => length,
        WasmJointType::Distance => {
            let (lower, upper) = if joint.limits_enabled {
                (joint.lower_limit, joint.upper_limit)
            } else {
                (joint.lower_limit, joint.lower_limit)
            };
            if length > upper {
                length - upper
            } else if length < lower {
                length - lower
            } else {
                0.0
            }
        }
        // Only ball and distance joints can be created on a `WasmWorld`
        // (`add_ball_joint` / `add_distance_joint`); no other type is ever
        // stored in its joint table.
        WasmJointType::Fixed
        | WasmJointType::Revolute
        | WasmJointType::Prismatic
        | WasmJointType::Generic6Dof
        | WasmJointType::Spring => return 0.0,
    };
    if error.abs() <= JOINT_TOLERANCE {
        return 0.0;
    }
    let w = a.weight(r_a, n) + b.weight(r_b, n);
    if !(w.is_finite() && w > 1e-15) {
        return 0.0;
    }
    // Positional impulse that pulls A toward B (and B toward A).
    let impulse = v_scale(n, error / w);
    if let Some(body) = bodies.get_mut(&joint.body_a.0) {
        apply_positional_impulse(body, &a, r_a, impulse);
    }
    if let Some(body) = bodies.get_mut(&joint.body_b.0) {
        apply_positional_impulse(body, &b, r_b, v_neg(impulse));
    }
    error.abs()
}

/// Move and rotate a body by the positional impulse `p` applied at lever `r`.
fn apply_positional_impulse(body: &mut WasmRigidBody, props: &JointBody, r: V3, p: V3) {
    if props.inv_mass <= 0.0 {
        return;
    }
    body.position = v_madd(body.position, p, props.inv_mass);
    let dtheta = m_vec(&props.inv_inertia, v_cross(r, p));
    body.rotation = q_apply_rotation_vector(body.rotation, dtheta);
}
