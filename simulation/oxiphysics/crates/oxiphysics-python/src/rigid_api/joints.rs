// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Joint types, joint sets and the pipeline configuration for the `rigid`
//! Python bindings. The world and the step executor live in
//! [`super::world`].

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyJointType
// ---------------------------------------------------------------------------

/// Type of joint between two rigid bodies.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyJointType {
    /// Ball-socket: rotation in all directions, no translation.
    BallSocket = 0,
    /// Hinge: rotation around one axis only.
    Hinge = 1,
    /// Slider: translation along one axis only.
    Slider = 2,
    /// Fixed: no relative motion.
    Fixed = 3,
    /// Distance: maintains a fixed or bounded distance.
    Distance = 4,
}

// ---------------------------------------------------------------------------
// PyJoint
// ---------------------------------------------------------------------------

/// A constraint joint between two rigid bodies.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyJoint {
    /// Unique joint handle.
    #[pyo3(get)]
    pub handle: u32,
    /// Type of joint.
    #[pyo3(get)]
    pub joint_type: PyJointType,
    /// Handle of body A.
    #[pyo3(get)]
    pub body_a: u32,
    /// Handle of body B.
    #[pyo3(get)]
    pub body_b: u32,
    /// Anchor on body A in world space, evaluated when the joint is added to
    /// a `PyWorld` (the world then keeps it fixed in body A's frame).
    #[pyo3(get, set)]
    pub anchor_a: [f64; 3],
    /// Anchor on body B in world space, evaluated when the joint is added to
    /// a `PyWorld` (the world then keeps it fixed in body B's frame).
    #[pyo3(get, set)]
    pub anchor_b: [f64; 3],
    /// Hinge / slider axis in world space at the time the joint is added.
    #[pyo3(get, set)]
    pub axis: [f64; 3],
    /// Target anchor-to-anchor distance for Distance joints.
    #[pyo3(get, set)]
    pub target_distance: f64,
    /// Joint stiffness in `[0, 1]`: the fraction of the remaining constraint
    /// error removed per joint-solver iteration (`1` = rigid projection).
    #[pyo3(get, set)]
    pub stiffness: f64,
    /// Whether the joint is currently active.
    #[pyo3(get, set)]
    pub enabled: bool,
}

#[pymethods]
impl PyJoint {
    /// Create a ball-socket joint at the given world-space pivot point.
    #[new]
    pub fn new(handle: u32, body_a: u32, body_b: u32, pivot: [f64; 3]) -> Self {
        Self {
            handle,
            joint_type: PyJointType::BallSocket,
            body_a,
            body_b,
            anchor_a: pivot,
            anchor_b: pivot,
            axis: [0.0, 1.0, 0.0],
            target_distance: 0.0,
            stiffness: 0.8,
            enabled: true,
        }
    }

    /// Create a hinge joint at a world-space pivot rotating about a
    /// world-space axis.
    #[staticmethod]
    pub fn hinge(handle: u32, body_a: u32, body_b: u32, pivot: [f64; 3], axis: [f64; 3]) -> Self {
        Self {
            handle,
            joint_type: PyJointType::Hinge,
            body_a,
            body_b,
            anchor_a: pivot,
            anchor_b: pivot,
            axis,
            target_distance: 0.0,
            stiffness: 0.8,
            enabled: true,
        }
    }

    /// Create a distance joint between two world-space anchors.
    #[staticmethod]
    pub fn distance(
        handle: u32,
        body_a: u32,
        body_b: u32,
        anchor_a: [f64; 3],
        anchor_b: [f64; 3],
        target: f64,
    ) -> Self {
        Self {
            handle,
            joint_type: PyJointType::Distance,
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            axis: [0.0, 1.0, 0.0],
            target_distance: target,
            stiffness: 0.8,
            enabled: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyJoint(handle={}, type={:?}, a={}, b={})",
            self.handle, self.joint_type, self.body_a, self.body_b
        )
    }
}

// ---------------------------------------------------------------------------
// PyJointSet
// ---------------------------------------------------------------------------

/// A collection of joints.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct PyJointSet {
    joints: Vec<PyJoint>,
    next_handle: u32,
}

#[pymethods]
impl PyJointSet {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a joint to the set. Returns its handle.
    pub fn add(&mut self, mut joint: PyJoint) -> u32 {
        let h = self.next_handle;
        joint.handle = h;
        self.joints.push(joint);
        self.next_handle += 1;
        h
    }

    /// Remove a joint by handle. Returns `True` if found.
    pub fn remove(&mut self, handle: u32) -> bool {
        let before = self.joints.len();
        self.joints.retain(|j| j.handle != handle);
        self.joints.len() < before
    }

    /// Get a joint by handle, or `None`.
    pub fn get(&self, handle: u32) -> Option<PyJoint> {
        self.joints.iter().find(|j| j.handle == handle).cloned()
    }

    /// Return all joints.
    pub fn all_joints(&self) -> Vec<PyJoint> {
        self.joints.clone()
    }

    /// Number of joints.
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// Whether the joint set is empty.
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("PyJointSet(len={})", self.joints.len())
    }
}

impl PyJointSet {
    /// Build a set from existing joints and the next handle to assign.
    pub(crate) fn from_parts(joints: Vec<PyJoint>, next_handle: u32) -> Self {
        Self {
            joints,
            next_handle,
        }
    }
}

// ---------------------------------------------------------------------------
// PyPipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the physics pipeline.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPipelineConfig {
    /// Gravity vector `[gx, gy, gz]`.
    #[pyo3(get, set)]
    pub gravity: [f64; 3],
    /// Default number of substeps of `PyWorld.step(dt)`.
    #[pyo3(get, set)]
    pub max_substeps: u32,
    /// Enable sleeping.
    #[pyo3(get, set)]
    pub sleep_enabled: bool,
    /// Linear sleep threshold.
    #[pyo3(get, set)]
    pub linear_sleep_threshold: f64,
    /// Angular sleep threshold.
    #[pyo3(get, set)]
    pub angular_sleep_threshold: f64,
    /// Baumgarte factor of the contact position correction (`[0, 1]`).
    #[pyo3(get, set)]
    pub baumgarte_factor: f64,
    /// Joint-projection (Gauss–Seidel) iterations per substep. More iterations
    /// tighten long joint chains.
    #[pyo3(get, set)]
    pub joint_iterations: u32,
}

#[pymethods]
impl PyPipelineConfig {
    #[new]
    #[pyo3(signature = (gravity = [0.0, -9.81, 0.0]))]
    pub fn new(gravity: [f64; 3]) -> Self {
        Self {
            gravity,
            max_substeps: 1,
            sleep_enabled: true,
            linear_sleep_threshold: 0.01,
            angular_sleep_threshold: 0.01,
            baumgarte_factor: 0.2,
            joint_iterations: 20,
        }
    }

    fn __repr__(&self) -> String {
        format!("PyPipelineConfig(gravity={:?})", self.gravity)
    }
}

impl Default for PyPipelineConfig {
    fn default() -> Self {
        Self::new([0.0, -9.81, 0.0])
    }
}
