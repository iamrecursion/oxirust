// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Constraint builders and constraint resolution.
//!
//! A [`PyConstraint`] joins two bodies of a [`PyPhysicsWorld`]: at a fixed
//! distance, at a common point (ball-socket), around a common axis (hinge) or
//! along a common axis (slider). The world keeps the constraints and solves them
//! in every [`step`](PyPhysicsWorld::step) with the joints and the
//! constraint-solver hooks of the engine (`oxiphysics::constraints`): the
//! velocities, the anchors' lever arms and the bodies' angular state take part
//! in the solve, so the constrained bodies keep the joint's geometry while they
//! move (the `solver` submodule describes the solve in detail).
//!
//! Positions are integrated before the constraints are solved, and the solve
//! projects the velocities onto the constraint manifold; like every
//! first-order projection this removes a little kinetic energy per step (the
//! relative loss is about `(v dt / L)^2` for a pendulum of length `L`), which
//! `step_substeps` reduces.
//!
//! # Frames
//!
//! * [`PyConstraint::distance`] takes its two anchors in the **local frames** of
//!   the bodies.
//! * [`PyConstraint::point_to_point`], [`PyConstraint::hinge`] and
//!   [`PyConstraint::slider`] take a **world-space pivot** (and a world-space
//!   axis). [`PyPhysicsWorld::add_constraint`] converts them into the frames of
//!   the two bodies from the poses the bodies have at that moment, and keeps the
//!   relative orientation of the bodies at that moment as the joint's reference
//!   (a hinge's angle is measured from it, a slider keeps it).

use super::{InternalBody, PyPhysicsWorld};
use oxiphysics::core::math::{Quat, Vec3};
use pyo3::prelude::*;

mod solver;
#[cfg(test)]
mod tests;

// ===========================================================================
// Constraint Builders
// ===========================================================================

/// Reasons [`PyPhysicsWorld::add_constraint`] rejects a constraint.
///
/// An unknown body surfaces in Python as an `IndexError`, every other error as
/// a `ValueError`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ConstraintError {
    /// A body handle of the constraint does not name a body of the world.
    #[error("the constraint refers to body {handle}, which does not exist")]
    UnknownBody {
        /// The handle without a body.
        handle: u32,
    },
    /// Both ends of the constraint are the same body.
    #[error("a constraint joins two different bodies, but both ends are body {handle}")]
    SameBody {
        /// The body used twice.
        handle: u32,
    },
    /// A number that has to be finite is NaN or infinite.
    #[error("{what} must be finite, got {value}")]
    NotFinite {
        /// What the value describes.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// The axis of a hinge or a slider is the zero vector.
    #[error("the axis of a {kind} constraint must not be the zero vector")]
    ZeroAxis {
        /// The kind of constraint.
        kind: &'static str,
    },
    /// The target distance is negative.
    #[error("the target distance must not be negative, got {value}")]
    NegativeDistance {
        /// The offending value.
        value: f64,
    },
    /// The stiffness is outside `[0, 1]`.
    #[error("the stiffness must lie in [0, 1], got {value}")]
    StiffnessOutOfRange {
        /// The offending value.
        value: f64,
    },
}

impl From<ConstraintError> for PyErr {
    fn from(err: ConstraintError) -> PyErr {
        let message = err.to_string();
        match err {
            ConstraintError::UnknownBody { .. } => pyo3::exceptions::PyIndexError::new_err(message),
            _ => pyo3::exceptions::PyValueError::new_err(message),
        }
    }
}

/// Type of constraint between two bodies.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Distance constraint: the two anchor points keep a fixed distance (the
    /// engine's `DistanceJoint` with equal lower and upper bound).
    Distance,
    /// Point-to-point (ball-socket): the two anchor points coincide, the bodies
    /// rotate freely about it (the engine's `BallJoint`).
    PointToPoint,
    /// Hinge: point-to-point, and the bodies only rotate relative to each other
    /// about the hinge axis (the engine's `RevoluteJoint`).
    Hinge,
    /// Slider: the bodies translate relative to each other along the slider
    /// axis only and do not rotate relative to each other (the engine's
    /// `PrismaticJoint`; the rotation lock acts on the velocities, so a torque
    /// can leave a small residual rotation).
    Slider,
}

/// The frames of a constraint that follow from the poses of its two bodies at
/// the moment it is added to a world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintFrame {
    /// Unit hinge or slider axis in the local frame of body A.
    pub local_axis: [f64; 3],
    /// Relative orientation `q_a⁻¹ q_b` of the two bodies when the constraint was
    /// added, as `[x, y, z, w]`: the hinge angle is measured from it and a
    /// slider keeps it.
    pub reference_rotation: [f64; 4],
    /// Generation counters of the two body slots. A constraint whose body was
    /// removed (and whose slot may now hold another body) no longer matches
    /// them and stays inert instead of acting on a stranger.
    pub generations: [u32; 2],
}

/// A constraint between two rigid bodies.
///
/// Constraints are stored in the world ([`PyPhysicsWorld::add_constraint`]) and
/// solved during each step with the engine's joints. See the module
/// documentation for the frames; the other parameters are
///
/// * `stiffness` in `[0, 1]`: the relaxation factor of every velocity sweep of
///   the solver. At `1` a sweep solves the constraint exactly; below `1` it
///   removes that fraction of the remaining violation, so after the world's
///   `solver_iterations` sweeps a fraction `(1 - stiffness)^solver_iterations`
///   of the violating velocity is left. `0` switches the constraint off.
/// * `enabled`: a disabled constraint is kept but not solved.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyConstraint {
    /// Unique handle (assigned by the world when the constraint is added).
    #[pyo3(get)]
    pub handle: u32,
    /// Type of constraint.
    #[pyo3(get)]
    pub constraint_type: ConstraintType,
    /// Handle of the first body.
    #[pyo3(get)]
    pub body_a: u32,
    /// Handle of the second body.
    #[pyo3(get)]
    pub body_b: u32,
    /// Anchor point on body A in the local frame of A.
    ///
    /// For a constraint built from a world-space pivot this is computed when the
    /// constraint is added to a world.
    pub anchor_a: [f64; 3],
    /// Anchor point on body B in the local frame of B (see `anchor_a`).
    pub anchor_b: [f64; 3],
    /// Target distance between the anchors (for Distance constraints).
    #[pyo3(get)]
    pub target_distance: f64,
    /// Hinge or slider axis in world space.
    pub axis: [f64; 3],
    /// Constraint stiffness (0..1, 1 = rigid), see the type documentation.
    pub stiffness: f64,
    /// Whether the constraint is currently active.
    #[pyo3(get)]
    pub enabled: bool,
    /// World-space pivot of a point-to-point, hinge or slider constraint, the
    /// point at which both bodies are joined when the constraint is added. When
    /// it is `None` the local anchors `anchor_a` and `anchor_b` are used as
    /// given.
    pub pivot: Option<[f64; 3]>,
    /// The frames fixed when the constraint was added to a world; `None` for a
    /// constraint that has not been added to one.
    pub frame: Option<ConstraintFrame>,
}

/// Stiffness of a newly built constraint: every velocity sweep removes half of
/// the remaining violation (see [`PyConstraint`]).
const DEFAULT_STIFFNESS: f64 = 0.5;

#[pymethods]
impl PyConstraint {
    /// Create a distance constraint: the point `anchor_a` of body A (in A's
    /// local frame) and the point `anchor_b` of body B (in B's local frame) keep
    /// the distance `distance`.
    #[staticmethod]
    pub fn distance(
        handle: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: [f64; 3],
        anchor_b: [f64; 3],
        distance: f64,
    ) -> Self {
        Self {
            handle,
            constraint_type: ConstraintType::Distance,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            target_distance: distance,
            axis: [0.0, 1.0, 0.0],
            stiffness: DEFAULT_STIFFNESS,
            enabled: true,
            pivot: None,
            frame: None,
        }
    }

    /// Create a point-to-point constraint (ball-socket joint) that joins the two
    /// bodies at the world-space point `pivot`.
    #[staticmethod]
    pub fn point_to_point(handle: u32, body_a: u32, body_b: u32, pivot: [f64; 3]) -> Self {
        Self::pivoted(
            handle,
            ConstraintType::PointToPoint,
            body_a,
            body_b,
            pivot,
            [0.0, 1.0, 0.0],
        )
    }

    /// Create a hinge constraint: the bodies are joined at the world-space point
    /// `pivot` and only rotate relative to each other about the world-space
    /// direction `axis`.
    #[staticmethod]
    pub fn hinge(handle: u32, body_a: u32, body_b: u32, pivot: [f64; 3], axis: [f64; 3]) -> Self {
        Self::pivoted(handle, ConstraintType::Hinge, body_a, body_b, pivot, axis)
    }

    /// Create a slider constraint: body B translates relative to body A along the
    /// world-space direction `axis` through the world-space point `pivot` and
    /// does not rotate relative to A.
    #[staticmethod]
    pub fn slider(handle: u32, body_a: u32, body_b: u32, pivot: [f64; 3], axis: [f64; 3]) -> Self {
        Self::pivoted(handle, ConstraintType::Slider, body_a, body_b, pivot, axis)
    }

    /// Enable or disable the constraint.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Constraint stiffness in `[0, 1]` (see the type documentation).
    #[getter]
    pub fn stiffness(&self) -> f64 {
        self.stiffness
    }

    /// Set the stiffness; raises `ValueError` outside `[0, 1]` or for NaN.
    #[setter]
    pub fn set_stiffness(&mut self, stiffness: f64) -> Result<(), ConstraintError> {
        check_stiffness(stiffness)?;
        self.stiffness = stiffness;
        Ok(())
    }
}

impl PyConstraint {
    /// A constraint joined at a world-space pivot (point-to-point, hinge, slider).
    fn pivoted(
        handle: u32,
        constraint_type: ConstraintType,
        body_a: u32,
        body_b: u32,
        pivot: [f64; 3],
        axis: [f64; 3],
    ) -> Self {
        Self {
            handle,
            constraint_type,
            body_a,
            body_b,
            anchor_a: [0.0; 3],
            anchor_b: [0.0; 3],
            target_distance: 0.0,
            axis,
            stiffness: DEFAULT_STIFFNESS,
            enabled: true,
            pivot: Some(pivot),
            frame: None,
        }
    }

    /// Set constraint stiffness and return `self` (builder pattern, Rust-only).
    pub fn with_stiffness(mut self, s: f64) -> Self {
        self.stiffness = s.clamp(0.0, 1.0);
        self
    }

    /// Join the bodies at the given **local** anchor points instead of a
    /// world-space pivot, and return `self` (builder pattern, Rust-only).
    pub fn with_local_anchors(mut self, anchor_a: [f64; 3], anchor_b: [f64; 3]) -> Self {
        self.anchor_a = anchor_a;
        self.anchor_b = anchor_b;
        self.pivot = None;
        self
    }

    /// Check the numbers of the constraint.
    fn validate(&self) -> Result<(), ConstraintError> {
        let finite = |what: &'static str, values: &[f64]| -> Result<(), ConstraintError> {
            match values.iter().find(|v| !v.is_finite()) {
                Some(&value) => Err(ConstraintError::NotFinite { what, value }),
                None => Ok(()),
            }
        };
        finite("an anchor coordinate", &self.anchor_a)?;
        finite("an anchor coordinate", &self.anchor_b)?;
        finite("an axis component", &self.axis)?;
        finite("the target distance", &[self.target_distance])?;
        if let Some(pivot) = &self.pivot {
            finite("a pivot coordinate", pivot)?;
        }
        if self.target_distance < 0.0 {
            return Err(ConstraintError::NegativeDistance {
                value: self.target_distance,
            });
        }
        check_stiffness(self.stiffness)
    }

    /// A copy of the constraint with its frames fixed from the current poses of
    /// the two bodies; this is the constraint the world stores.
    fn attached(
        &self,
        body_a: &InternalBody,
        body_b: &InternalBody,
        generations: [u32; 2],
    ) -> Result<Self, ConstraintError> {
        self.validate()?;
        let rotation_a = quat_from_xyzw(body_a.orientation);
        let rotation_b = quat_from_xyzw(body_b.orientation);
        let mut attached = self.clone();
        if let Some(pivot) = self.pivot {
            let pivot = Vec3::from(pivot);
            attached.anchor_a = to_array(
                &rotation_a.inverse_transform_vector(&(pivot - Vec3::from(body_a.position))),
            );
            attached.anchor_b = to_array(
                &rotation_b.inverse_transform_vector(&(pivot - Vec3::from(body_b.position))),
            );
        }
        let local_axis = match self.constraint_type {
            ConstraintType::Hinge | ConstraintType::Slider => {
                let axis = Vec3::from(self.axis);
                let length = axis.norm();
                if length < MIN_AXIS_LENGTH {
                    let kind = if self.constraint_type == ConstraintType::Hinge {
                        "hinge"
                    } else {
                        "slider"
                    };
                    return Err(ConstraintError::ZeroAxis { kind });
                }
                rotation_a.inverse_transform_vector(&(axis / length))
            }
            ConstraintType::Distance | ConstraintType::PointToPoint => Vec3::y(),
        };
        let reference = rotation_a.inverse() * rotation_b;
        let reference = reference.quaternion();
        attached.frame = Some(ConstraintFrame {
            local_axis: to_array(&local_axis),
            reference_rotation: [reference.i, reference.j, reference.k, reference.w],
            generations,
        });
        Ok(attached)
    }
}

/// Shortest hinge or slider axis that still has a direction.
const MIN_AXIS_LENGTH: f64 = 1.0e-12;

/// `Ok` if `stiffness` lies in `[0, 1]`.
fn check_stiffness(stiffness: f64) -> Result<(), ConstraintError> {
    if stiffness.is_finite() && (0.0..=1.0).contains(&stiffness) {
        Ok(())
    } else if stiffness.is_nan() {
        Err(ConstraintError::NotFinite {
            what: "the stiffness",
            value: stiffness,
        })
    } else {
        Err(ConstraintError::StiffnessOutOfRange { value: stiffness })
    }
}

/// Components of a vector as an array.
fn to_array(v: &Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// The unit quaternion of an `[x, y, z, w]` array (the layout of the world's
/// orientations). A degenerate array stands for the identity.
fn quat_from_xyzw(q: [f64; 4]) -> Quat {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !(norm.is_finite() && norm > 1.0e-12) {
        return Quat::identity();
    }
    Quat::from_quaternion(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))
}

/// Extension: constraint storage and resolution on `PyPhysicsWorld`.
#[pymethods]
impl PyPhysicsWorld {
    /// Add a constraint to the world. Returns the constraint handle.
    ///
    /// The world stores a copy whose frames are fixed from the poses the two
    /// bodies have now: a world-space pivot becomes an anchor in the frame of
    /// each body, a world-space axis becomes an axis in the frame of body A, and
    /// the relative orientation of the bodies becomes the reference of a hinge
    /// angle or the orientation a slider keeps. The constraint is not moved
    /// into place: bodies that violate it when it is added are pulled toward it
    /// by the following steps.
    ///
    /// Note: constraint handles are auto-incremented independently of body handles.
    ///
    /// # Errors
    ///
    /// `IndexError` if a body handle is unknown; `ValueError` if both ends are
    /// the same body, a number is not finite, the target distance is negative,
    /// the stiffness is outside `[0, 1]` or a hinge or slider has a zero axis.
    pub fn add_constraint(&mut self, constraint: PyConstraint) -> Result<u32, ConstraintError> {
        if constraint.body_a == constraint.body_b {
            return Err(ConstraintError::SameBody {
                handle: constraint.body_a,
            });
        }
        let mut attached = {
            let (body_a, generation_a) = self.body_and_generation(constraint.body_a).ok_or(
                ConstraintError::UnknownBody {
                    handle: constraint.body_a,
                },
            )?;
            let (body_b, generation_b) = self.body_and_generation(constraint.body_b).ok_or(
                ConstraintError::UnknownBody {
                    handle: constraint.body_b,
                },
            )?;
            constraint.attached(body_a, body_b, [generation_a, generation_b])?
        };
        let handle = self.next_constraint_handle();
        attached.handle = handle;
        self.constraints.push(attached);
        Ok(handle)
    }

    /// Remove a constraint by handle. Returns `true` if found.
    pub fn remove_constraint(&mut self, handle: u32) -> bool {
        let before = self.constraints.len();
        self.constraints.retain(|c| c.handle != handle);
        self.constraints.len() < before
    }

    /// Return the number of constraints in the world.
    ///
    /// A constraint whose body was removed stays in the world (inert) until it
    /// is removed with `remove_constraint`.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}

impl PyPhysicsWorld {
    /// The body in slot `handle` and the generation of the slot, `None` if the
    /// slot holds no body.
    fn body_and_generation(&self, handle: u32) -> Option<(&InternalBody, u32)> {
        let slot = self.slots.get(handle as usize)?;
        Some((slot.body.as_ref()?, slot.generation))
    }

    /// The two bodies of an added constraint, `None` if the constraint was not
    /// added to a world or one of its bodies was removed since (even if another
    /// body now occupies the slot).
    fn constraint_bodies(
        &self,
        constraint: &PyConstraint,
    ) -> Option<(&InternalBody, &InternalBody)> {
        let frame = constraint.frame.as_ref()?;
        let (body_a, generation_a) = self.body_and_generation(constraint.body_a)?;
        let (body_b, generation_b) = self.body_and_generation(constraint.body_b)?;
        (frame.generations == [generation_a, generation_b]).then_some((body_a, body_b))
    }

    /// Solve the enabled constraints for a step of `dt` seconds.
    ///
    /// Called internally by `step()` after the bodies were integrated and the
    /// contacts resolved; the engine's joints correct the velocities (including
    /// the angular ones) and the poses of the constrained bodies, see the
    /// `solver` module.
    pub(super) fn resolve_constraints(&mut self, dt: f64) {
        solver::resolve(self, dt);
    }

    /// Compute the next available constraint handle.
    fn next_constraint_handle(&self) -> u32 {
        self.constraints
            .iter()
            .map(|c| c.handle)
            .max()
            .map(|h| h + 1)
            .unwrap_or(0)
    }
}
