// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Crate-internal access to engine internals for the other WASM bridges.
//!
//! Two facilities live here:
//!
//! * [`JointBodyState`] together with
//!   [`WasmPhysicsEngine::joint_body_state`] and
//!   [`WasmPhysicsEngine::write_joint_body_velocity`] give the joint solver of
//!   [`crate::constraint_bridge`] the mass properties, pose and velocities it
//!   needs to build and solve Jacobian rows against the engine's bodies.
//! * [`collide_primitives`] and [`raycast_primitive`] expose the engine's
//!   narrow phase and ray casting (sphere, box, capsule, cylinder and plane,
//!   including SAT box-box) to [`crate::simulation_api::WasmWorld`], so both
//!   worlds share one tested set of geometric kernels.

use super::WasmPhysicsEngine;
use super::geom::{MAT3_IDENTITY, MAT3_ZERO, Mat3, madd, quat_normalized, quat_to_mat3};
use super::narrowphase::collide;
use super::raycast::ray_shape;
use super::shapes::{Shape, WorldShape};
use super::validation::require_finite3;
use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Joint solver access
// ---------------------------------------------------------------------------

/// Snapshot of the state of one engine body as seen by a joint solver.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointBodyState {
    /// World-space centre of mass.
    pub position: [f64; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub rotation: [f64; 4],
    /// Linear velocity.
    pub velocity: [f64; 3],
    /// Angular velocity (world frame).
    pub angular_velocity: [f64; 3],
    /// Inverse mass (`0` for static, kinematic and massless bodies).
    pub inv_mass: f64,
    /// World-space inverse inertia tensor (zero for non-dynamic bodies and
    /// for bodies without a volumetric collider).
    pub inv_inertia_world: [[f64; 3]; 3],
    /// Linear velocity change that the next fixed substep of
    /// [`WasmPhysicsEngine::step`] adds from gravity and accumulated forces
    /// (zero for non-dynamic and for sleeping bodies without pending forces).
    pub external_velocity_change: [f64; 3],
}

impl WasmPhysicsEngine {
    /// State of a live body for a joint solver, or `None` for a dead or stale
    /// handle.
    ///
    /// `dt` is the step the caller is about to integrate with; it scales the
    /// predicted external velocity change.
    pub(crate) fn joint_body_state(&self, handle: u32, dt: f64) -> Option<JointBodyState> {
        let body = self.get_body(handle)?;
        let dynamic = body.is_dynamic();
        let inv_mass = if dynamic { body.inv_mass } else { 0.0 };
        let inv_inertia_world: Mat3 = if dynamic && body.has_inertia {
            body.world_inv_inertia()
        } else {
            MAT3_ZERO
        };
        // `integrate` skips sleeping bodies unless a force or torque wakes
        // them, so only those bodies receive the external velocity change.
        let forces_pending = body
            .force
            .iter()
            .chain(body.torque.iter())
            .any(|c| *c != 0.0);
        let integrated = dynamic && (!body.sleeping || forces_pending);
        let external_velocity_change = if integrated {
            let gs = body.gravity_scale;
            let acceleration = madd(
                [
                    self.gravity[0] * gs,
                    self.gravity[1] * gs,
                    self.gravity[2] * gs,
                ],
                body.force,
                body.inv_mass,
            );
            [
                acceleration[0] * dt,
                acceleration[1] * dt,
                acceleration[2] * dt,
            ]
        } else {
            [0.0; 3]
        };
        Some(JointBodyState {
            position: body.position,
            rotation: body.rotation,
            velocity: body.velocity,
            angular_velocity: body.angular_velocity,
            inv_mass,
            inv_inertia_world,
            external_velocity_change,
        })
    }

    /// Overwrite the linear and angular velocity of a body after a joint
    /// solve and wake it.
    ///
    /// # Errors
    ///
    /// As [`WasmPhysicsEngine::set_velocity`]: an invalid handle or a
    /// non-finite component.
    pub(crate) fn write_joint_body_velocity(
        &mut self,
        handle: u32,
        velocity: [f64; 3],
        angular_velocity: [f64; 3],
    ) -> Result<()> {
        self.set_velocity(handle, velocity[0], velocity[1], velocity[2])?;
        self.set_angular_velocity(
            handle,
            angular_velocity[0],
            angular_velocity[1],
            angular_velocity[2],
        )
    }
}

impl WasmPhysicsEngine {
    /// Overwrite the pose of a body after a joint position projection.
    ///
    /// The quaternion is normalised; the sleep state is left unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidHandle`] for a dead or stale handle,
    /// [`Error::InvalidParameter`] for a non-finite position or a degenerate
    /// quaternion (the body is left untouched).
    pub(crate) fn write_joint_body_pose(
        &mut self,
        handle: u32,
        position: [f64; 3],
        rotation: [f64; 4],
    ) -> Result<()> {
        let body = self.get_body_mut(handle)?;
        require_finite3("position", position)?;
        let rotation = quat_normalized(rotation).ok_or_else(|| Error::InvalidParameter {
            name: "rotation".to_string(),
            message: "quaternion must be finite and non-zero".to_string(),
        })?;
        body.position = position;
        body.rotation = rotation;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared geometric kernels
// ---------------------------------------------------------------------------

/// Geometry of a primitive collider in its own frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PrimitiveShape {
    /// Sphere centred on the origin.
    Sphere {
        /// Radius.
        radius: f64,
    },
    /// Box centred on the origin.
    Box {
        /// Half-extents along the local axes.
        half_extents: [f64; 3],
    },
    /// Capsule around the local Y axis.
    Capsule {
        /// Radius of the caps and the cylindrical part.
        radius: f64,
        /// Half the length of the cylindrical part.
        half_height: f64,
    },
    /// Solid cylinder around the local Y axis.
    Cylinder {
        /// Radius.
        radius: f64,
        /// Half the cylinder length.
        half_height: f64,
    },
    /// World-space half-space `normal . x <= offset` (the pose is ignored).
    ///
    /// The normal does not have to be normalised; `normal . x = offset`
    /// describes the same plane after normalisation.
    Plane {
        /// Outward plane normal.
        normal: [f64; 3],
        /// Plane offset along `normal`.
        offset: f64,
    },
}

/// A primitive placed in the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PrimitivePose {
    /// Shape in its local frame.
    pub shape: PrimitiveShape,
    /// World position of the shape origin.
    pub position: [f64; 3],
    /// Orientation quaternion `[x, y, z, w]` (normalised internally).
    pub rotation: [f64; 4],
}

/// A contact between two primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PrimitiveContact {
    /// Unit normal pointing from primitive B toward primitive A.
    pub normal: [f64; 3],
    /// Penetration depth (positive).
    pub depth: f64,
    /// Deepest point of A, on A's surface.
    pub point_a: [f64; 3],
    /// Matching point on B's surface.
    pub point_b: [f64; 3],
}

/// Place a primitive into the world; `None` for a degenerate plane normal.
fn place(pose: &PrimitivePose) -> Option<WorldShape> {
    let shape = match pose.shape {
        PrimitiveShape::Sphere { radius } => Shape::Sphere { radius },
        PrimitiveShape::Box { half_extents } => Shape::Box { half_extents },
        PrimitiveShape::Capsule {
            radius,
            half_height,
        } => Shape::Capsule {
            radius,
            half_height,
        },
        PrimitiveShape::Cylinder {
            radius,
            half_height,
        } => Shape::Cylinder {
            radius,
            half_height,
        },
        PrimitiveShape::Plane { normal, offset } => {
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if !len.is_finite() || len < 1e-12 || !offset.is_finite() {
                return None;
            }
            let unit = [normal[0] / len, normal[1] / len, normal[2] / len];
            // Planes are world-space: identity pose, unchanged normal/offset.
            return Some(WorldShape::place(
                Shape::Plane {
                    normal: unit,
                    offset: offset / len,
                },
                [0.0; 3],
                &MAT3_IDENTITY,
                [0.0; 3],
                &MAT3_IDENTITY,
            ));
        }
    };
    let rotation = quat_to_mat3(quat_normalized(pose.rotation).unwrap_or([0.0, 0.0, 0.0, 1.0]));
    Some(WorldShape::place(
        shape,
        pose.position,
        &rotation,
        [0.0; 3],
        &MAT3_IDENTITY,
    ))
}

/// Contact between two placed primitives (normal from `b` toward `a`).
///
/// Uses the engine narrow phase: closed-form sphere / capsule / plane tests,
/// SAT with face clipping for box-box and GJK/EPA for the remaining convex
/// pairs.  Plane-plane pairs never collide.
pub(crate) fn collide_primitives(a: &PrimitivePose, b: &PrimitivePose) -> Option<PrimitiveContact> {
    let shape_a = place(a)?;
    let shape_b = place(b)?;
    collide(&shape_a, &shape_b).map(|c| PrimitiveContact {
        normal: c.normal,
        depth: c.depth,
        point_a: c.point_a,
        point_b: c.point_b,
    })
}

/// Cast a ray against a placed primitive.
///
/// `dir` must be a unit vector.  Returns the hit distance and the outward
/// surface normal.  Solids report the entry point (or the exit point for a
/// ray starting inside); planes are hit only by rays travelling against
/// their normal.
pub(crate) fn raycast_primitive(
    origin: [f64; 3],
    dir: [f64; 3],
    pose: &PrimitivePose,
) -> Option<(f64, [f64; 3])> {
    let shape = place(pose)?;
    ray_shape(origin, dir, &shape).map(|hit| (hit.distance, hit.normal))
}
