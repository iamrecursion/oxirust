// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Solving the constraints of a [`PyPhysicsWorld`] with the engine's joints.
//!
//! The world keeps its bodies in a representation of its own; the physics of a
//! constraint is not repeated here but delegated to `oxiphysics::constraints`.
//! Every step the constrained bodies are copied into a scratch
//! [`RigidBodySet`] of the engine, each [`PyConstraint`] becomes the matching
//! joint
//!
//! | [`ConstraintType`] | engine joint |
//! |---|---|
//! | `Distance` | `DistanceJoint` (equal lower and upper bound) |
//! | `PointToPoint` | `BallJoint` |
//! | `Hinge` | `RevoluteJoint` (with the reference rotation of the constraint) |
//! | `Slider` | `PrismaticJoint` |
//!
//! and the joints are solved in this order:
//!
//! 1. `prepare` (lever arms and axes from the current poses),
//! 2. `solver_iterations` (at least one) Gauss-Seidel sweeps of the velocity
//!    rows, each relaxed by the stiffness of its constraint
//!    (`solve_velocity_relaxed`): the relative velocity of the anchors, the
//!    relative angular velocity about the locked axes and the lever arms and
//!    inertias of the bodies are all part of the engine's rows,
//! 3. `post_solve`,
//! 4. [`POSITION_PASSES_PER_ITERATION`] position passes per solver iteration,
//!    which move the bodies back toward the joint's geometry. The engine's
//!    joints remove a fixed fraction (10 %) of the position error per pass.
//!
//! Afterwards the velocities, angular velocities and poses of the movable
//! bodies are copied back. Static bodies, kinematic bodies and bodies of
//! infinite mass take part as immovable bodies (a kinematic body with its
//! velocity). A sleeping body is woken first when its constraint partner is
//! awake and not static, so that the pair is simulated together (a body tied to
//! a kinematic body therefore never sleeps); a constraint between bodies that
//! all sleep, or none of which can move, is skipped.

use super::{ConstraintFrame, ConstraintType, PyConstraint, quat_from_xyzw};
use crate::world_api::{InternalBody, PyPhysicsWorld, approximate_inertia};
use oxiphysics::constraints::{
    BallJoint, Constraint, DistanceJoint, PrismaticJoint, RevoluteJoint,
};
use oxiphysics::core::BodyHandle;
use oxiphysics::core::math::{Mat3, Vec3};
use oxiphysics::rigid::{RigidBody, RigidBodySet};
use std::collections::HashMap;

/// Position passes per velocity iteration (`solver_iterations`).
///
/// A position pass of the engine's joints removes 10 % of the position error,
/// and the bodies are integrated before the constraints are solved, so the
/// error of a step (the chord a body travels away from a constraint) has to be
/// removed again within the same step. Two passes per iteration (16 passes at
/// the default of 8 iterations) keep a pendulum of length 1 m at `dt = 1/240 s`
/// within a fraction of a millimetre of its length; one pass per iteration
/// leaves about three times that error.
const POSITION_PASSES_PER_ITERATION: usize = 2;

/// The engine copies of the bodies a solve works on.
#[derive(Default)]
struct Scratch {
    /// The engine bodies.
    bodies: RigidBodySet,
    /// Engine handle of every world body copied so far.
    handles: HashMap<u32, BodyHandle>,
    /// World handles in the order the bodies were copied.
    order: Vec<u32>,
}

impl Scratch {
    /// Engine handle of the world body `handle`, copying the body on first use.
    /// `None` if the world has no such body.
    fn engine_handle(&mut self, world: &PyPhysicsWorld, handle: u32) -> Option<BodyHandle> {
        if let Some(&known) = self.handles.get(&handle) {
            return Some(known);
        }
        let engine = self.bodies.insert(engine_body(world.get_body(handle)?));
        self.handles.insert(handle, engine);
        self.order.push(handle);
        Some(engine)
    }
}

/// The state the solve hands back for one body.
struct BodyState {
    /// Position of the centre of mass.
    position: [f64; 3],
    /// Linear velocity.
    velocity: [f64; 3],
    /// Angular velocity.
    angular_velocity: [f64; 3],
    /// Orientation `[x, y, z, w]`.
    orientation: [f64; 4],
}

impl BodyState {
    /// The state of an engine body, `None` if a component is not finite.
    fn from_engine(body: &RigidBody) -> Option<Self> {
        let array = |v: &Vec3| [v.x, v.y, v.z];
        let rotation = body.transform.rotation.quaternion();
        let state = Self {
            position: array(&body.transform.position),
            velocity: array(&body.velocity),
            angular_velocity: array(&body.angular_velocity),
            orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
        };
        let finite = state
            .position
            .iter()
            .chain(&state.velocity)
            .chain(&state.angular_velocity)
            .chain(&state.orientation)
            .all(|c| c.is_finite());
        finite.then_some(state)
    }

    /// Store the state in a world body.
    fn apply_to(&self, body: &mut InternalBody) {
        body.position = self.position;
        body.velocity = self.velocity;
        body.angular_velocity = self.angular_velocity;
        body.orientation = self.orientation;
    }
}

/// The engine's copy of a world body.
///
/// A dynamic body gets the isotropic inertia the world integrates its rotation
/// with ([`approximate_inertia`]); static bodies, kinematic bodies and bodies
/// without a finite inverse mass are immovable, the latter two keep their
/// velocity so that the anchors they carry move.
fn engine_body(body: &InternalBody) -> RigidBody {
    let mut engine = if body.is_static {
        RigidBody::new_static()
    } else if body.inv_mass() > 0.0 {
        let mut dynamic = RigidBody::new(body.mass);
        dynamic.local_inertia = Mat3::identity() * approximate_inertia(body);
        dynamic
    } else {
        RigidBody::new_kinematic()
    };
    engine.transform.position = Vec3::from(body.position);
    engine.transform.rotation = quat_from_xyzw(body.orientation);
    if !body.is_static {
        engine.velocity = Vec3::from(body.velocity);
        engine.angular_velocity = Vec3::from(body.angular_velocity);
    }
    engine.update_world_inertia();
    engine
}

/// The engine joint of a constraint between the engine bodies `a` and `b`.
fn engine_joint(
    constraint: &PyConstraint,
    frame: &ConstraintFrame,
    a: BodyHandle,
    b: BodyHandle,
) -> Box<dyn Constraint> {
    let anchor_a = Vec3::from(constraint.anchor_a);
    let anchor_b = Vec3::from(constraint.anchor_b);
    let axis = Vec3::from(frame.local_axis);
    match constraint.constraint_type {
        ConstraintType::Distance => Box::new(DistanceJoint::fixed_distance(
            a,
            b,
            anchor_a,
            anchor_b,
            constraint.target_distance,
        )),
        ConstraintType::PointToPoint => Box::new(BallJoint::new(a, b, anchor_a, anchor_b)),
        ConstraintType::Hinge => Box::new(
            RevoluteJoint::new(a, b, anchor_a, anchor_b, axis)
                .with_reference_rotation(quat_from_xyzw(frame.reference_rotation)),
        ),
        ConstraintType::Slider => Box::new(PrismaticJoint::new(a, b, anchor_a, anchor_b, axis)),
    }
}

/// Whether a body can set its constraint partner in motion: it is awake and not
/// static. A kinematic body is moved from outside the simulation (and never
/// falls asleep), so it always counts.
fn drives(body: &InternalBody) -> bool {
    !body.is_static && !body.is_sleeping
}

/// Indices of the constraints that have to be solved: enabled, stiff, attached to
/// two existing bodies, at least one of which can move and at least one of which
/// drives the pair.
fn active_constraints(world: &PyPhysicsWorld) -> Vec<usize> {
    world
        .constraints
        .iter()
        .enumerate()
        .filter_map(|(index, constraint)| {
            if !constraint.enabled || constraint.stiffness.is_nan() || constraint.stiffness <= 0.0 {
                return None;
            }
            let (a, b) = world.constraint_bodies(constraint)?;
            let movable = a.inv_mass() > 0.0 || b.inv_mass() > 0.0;
            (movable && (drives(a) || drives(b))).then_some(index)
        })
        .collect()
}

/// Wake the sleeping dynamic bodies of the given constraints; returns whether
/// any body woke up.
fn wake_sleeping_bodies(world: &mut PyPhysicsWorld, active: &[usize]) -> bool {
    let handles: Vec<u32> = active
        .iter()
        .filter_map(|&index| world.constraints.get(index))
        .flat_map(|constraint| [constraint.body_a, constraint.body_b])
        .collect();
    let mut woke = false;
    for handle in handles {
        if let Some(body) = world.get_body_mut(handle)
            && body.is_sleeping
            && body.is_dynamic()
        {
            body.is_sleeping = false;
            body.sleep_timer = 0.0;
            woke = true;
        }
    }
    woke
}

/// The constraints to solve, after waking every sleeping body that an awake
/// partner drags along. A body that wakes up can drive further constraints
/// (a chain of sleeping bodies), so the set is recomputed until nothing wakes.
fn active_constraints_after_waking(world: &mut PyPhysicsWorld) -> Vec<usize> {
    loop {
        let active = active_constraints(world);
        if !wake_sleeping_bodies(world, &active) {
            return active;
        }
    }
}

/// Solve the constraints of `world` for a step of `dt` seconds (see the module
/// documentation). Does nothing for a non-positive or non-finite `dt`.
pub(super) fn resolve(world: &mut PyPhysicsWorld, dt: f64) {
    if !(dt.is_finite() && dt > 0.0) {
        return;
    }
    let active = active_constraints_after_waking(world);
    if active.is_empty() {
        return;
    }

    let mut scratch = Scratch::default();
    let mut joints: Vec<(Box<dyn Constraint>, f64)> = Vec::with_capacity(active.len());
    for constraint in active.iter().filter_map(|&i| world.constraints.get(i)) {
        let Some(frame) = constraint.frame.as_ref() else {
            continue;
        };
        let (Some(a), Some(b)) = (
            scratch.engine_handle(world, constraint.body_a),
            scratch.engine_handle(world, constraint.body_b),
        ) else {
            continue;
        };
        joints.push((engine_joint(constraint, frame, a, b), constraint.stiffness));
    }

    let iterations = world.config.solver_iterations.max(1) as usize;
    for (joint, _) in &mut joints {
        joint.prepare(&scratch.bodies, dt);
    }
    for _ in 0..iterations {
        for (joint, stiffness) in &mut joints {
            joint.solve_velocity_relaxed(&mut scratch.bodies, dt, *stiffness);
        }
    }
    for (joint, _) in &mut joints {
        joint.post_solve();
    }
    for _ in 0..iterations * POSITION_PASSES_PER_ITERATION {
        for (joint, _) in &mut joints {
            joint.solve_position(&mut scratch.bodies, dt);
        }
    }

    for &handle in &scratch.order {
        let Some(engine) = scratch
            .handles
            .get(&handle)
            .and_then(|&engine_handle| scratch.bodies.get(engine_handle))
        else {
            continue;
        };
        if engine.inverse_mass <= 0.0 {
            continue;
        }
        if let Some(state) = BodyState::from_engine(engine)
            && let Some(body) = world.get_body_mut(handle)
        {
            state.apply_to(body);
        }
    }
}
