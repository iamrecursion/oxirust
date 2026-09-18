// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `rigid.PyWorld` and `rigid.PyPhysicsPipeline`: a Python-facing wrapper
//! around the engine's collider-driven [`PhysicsWorld`].
//!
//! The engine owns all body state. Each (sub)step runs the full engine
//! pipeline (gravity and force integration, sweep-and-prune broadphase,
//! narrowphase manifolds, sequential-impulse contact solver with friction and
//! restitution, position integration, position correction, sleeping) and then
//! projects the joints with [`solve_joints`].

use super::bodies::{PyCollider, PyContact, PyRigidBody, PyRigidBodySet, PyShapeType};
use super::joint_solver::{JointFrame, JointSpec, anchor_gap, solve_joints};
use super::joints::{PyJoint, PyJointSet, PyPipelineConfig};
use oxiphysics::core::math::{Mat3, Quat, Real, Vec3};
use oxiphysics::core::{BodyHandle, ColliderHandle, Transform};
use oxiphysics::geometry::{BoxShape, Capsule, Shape, Sphere};
use oxiphysics::rigid::world::PhysicsWorld;
use oxiphysics::rigid::{BodyState, Collider, RigidBody};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Engine conversion helpers
// ---------------------------------------------------------------------------

fn vec3(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn arr3(v: &Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn finite3(a: [f64; 3]) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// `[x, y, z, w]` → unit quaternion (normalised); rejects zero / non-finite.
pub(crate) fn quat_from_xyzw(q: [f64; 4]) -> PyResult<Quat> {
    if !q.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err("orientation must be finite"));
    }
    let raw = nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]);
    if raw.norm() < 1e-12 {
        return Err(PyValueError::new_err(
            "orientation quaternion must be non-zero",
        ));
    }
    Ok(Quat::from_quaternion(raw))
}

/// Unit quaternion → `[x, y, z, w]`.
pub(crate) fn quat_to_xyzw(q: &Quat) -> [f64; 4] {
    let inner = q.quaternion();
    [inner.i, inner.j, inner.k, inner.w]
}

/// Engine shape for a Python collider.
fn collider_shape(c: &PyCollider) -> PyResult<Arc<dyn Shape>> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    match c.shape {
        PyShapeType::Sphere => {
            if !positive(c.radius) {
                return Err(PyValueError::new_err("sphere radius must be > 0"));
            }
            Ok(Arc::new(Sphere::new(c.radius)))
        }
        PyShapeType::Box => {
            if !c.half_extents.iter().all(|&h| positive(h)) {
                return Err(PyValueError::new_err("box half-extents must all be > 0"));
            }
            Ok(Arc::new(BoxShape::new(vec3(c.half_extents))))
        }
        PyShapeType::Capsule => {
            let half_height = c.half_extents[1];
            if !positive(c.radius) || !half_height.is_finite() || half_height < 0.0 {
                return Err(PyValueError::new_err(
                    "capsule needs radius > 0 and half_height (half_extents[1]) >= 0",
                ));
            }
            Ok(Arc::new(Capsule::new(c.radius, half_height)))
        }
        PyShapeType::ConvexHull | PyShapeType::TriMesh => Err(PyValueError::new_err(format!(
            "{:?} colliders carry no vertex data in the rigid API; use Sphere, Box or Capsule",
            c.shape
        ))),
    }
}

/// Engine body plus its colliders, built from a Python body description.
struct EngineBody {
    body: RigidBody,
    colliders: Vec<Collider>,
}

/// Build the engine body for `desc`.
///
/// The body position is its centre of mass. A dynamic body's inertia tensor is
/// assembled from its non-sensor colliders: the mass is distributed over them
/// in proportion to their volumes and each collider's tensor is shifted to the
/// body origin with the parallel-axis theorem. A dynamic body without solid
/// colliders keeps the engine's default tensor `m·I₃`.
fn build_engine_body(desc: &PyRigidBody) -> PyResult<EngineBody> {
    if !finite3(desc.position) || !finite3(desc.velocity) || !finite3(desc.angular_velocity) {
        return Err(PyValueError::new_err(
            "position, velocity and angular_velocity must be finite",
        ));
    }
    let rotation = quat_from_xyzw(desc.orientation)?;
    let damping_ok = |d: f64| d.is_finite() && d >= 0.0;
    if !damping_ok(desc.linear_damping) || !damping_ok(desc.angular_damping) {
        return Err(PyValueError::new_err("damping coefficients must be >= 0"));
    }
    let mut shapes: Vec<(Arc<dyn Shape>, &PyCollider)> = Vec::with_capacity(desc.colliders.len());
    for c in &desc.colliders {
        if !finite3(c.offset) {
            return Err(PyValueError::new_err("collider offset must be finite"));
        }
        if !c.friction.is_finite() || c.friction < 0.0 {
            return Err(PyValueError::new_err("collider friction must be >= 0"));
        }
        if !c.restitution.is_finite() || !(0.0..=1.0).contains(&c.restitution) {
            return Err(PyValueError::new_err(
                "collider restitution must be within [0, 1]",
            ));
        }
        shapes.push((collider_shape(c)?, c));
    }

    let mut body = if desc.is_static {
        RigidBody::new_static()
    } else if desc.is_kinematic {
        RigidBody::new_kinematic()
    } else {
        if !desc.mass.is_finite() || desc.mass <= 0.0 {
            return Err(PyValueError::new_err(
                "a dynamic body needs a finite mass > 0 (use static_body() or is_kinematic)",
            ));
        }
        let mut b = RigidBody::new(desc.mass);
        if let Some(inertia) = composite_inertia(desc.mass, &shapes) {
            b.local_inertia = inertia;
        }
        b
    };
    body.transform.position = vec3(desc.position);
    body.transform.rotation = rotation;
    if !desc.is_static {
        body.velocity = vec3(desc.velocity);
        body.angular_velocity = vec3(desc.angular_velocity);
    }
    body.linear_damping = desc.linear_damping;
    body.angular_damping = desc.angular_damping;
    body.update_world_inertia();

    let colliders = shapes
        .into_iter()
        .map(|(shape, c)| {
            let collider = Collider::new(shape)
                .with_friction(c.friction)
                .with_restitution(c.restitution)
                .with_local_transform(Transform::from_position(vec3(c.offset)));
            if c.is_sensor {
                collider.as_sensor()
            } else {
                collider
            }
        })
        .collect();
    Ok(EngineBody { body, colliders })
}

/// Inertia tensor about the body origin of `mass` spread over the solid
/// colliders by volume, or `None` when there is no solid collider.
fn composite_inertia(mass: Real, shapes: &[(Arc<dyn Shape>, &PyCollider)]) -> Option<Mat3> {
    let solid: Vec<(&Arc<dyn Shape>, &PyCollider, Real)> = shapes
        .iter()
        .filter(|(_, c)| !c.is_sensor)
        .map(|(s, c)| (s, *c, s.volume()))
        .filter(|(_, _, v)| v.is_finite() && *v > 0.0)
        .collect();
    let total: Real = solid.iter().map(|(_, _, v)| v).sum();
    if solid.is_empty() || total <= 0.0 {
        return None;
    }
    let mut inertia = Mat3::zeros();
    for (shape, c, volume) in solid {
        let m_i = mass * volume / total;
        let r = vec3(c.offset);
        inertia += shape.inertia_tensor(m_i)
            + (Mat3::identity() * r.norm_squared() - r * r.transpose()) * m_i;
    }
    Some(inertia)
}

// ---------------------------------------------------------------------------
// PyWorld
// ---------------------------------------------------------------------------

/// Bookkeeping for one body of a [`PyWorld`].
struct BodyEntry {
    /// Engine body handle.
    engine: BodyHandle,
    /// Engine handles of the body's colliders.
    colliders: Vec<ColliderHandle>,
    /// Description the body was added with; the dynamic state (pose and
    /// velocities) is always read back from the engine.
    desc: PyRigidBody,
}

/// Bookkeeping for one joint of a [`PyWorld`].
struct JointEntry {
    joint: PyJoint,
    frame: JointFrame,
}

/// A rigid-body world backed by the engine's collider-driven `PhysicsWorld`.
///
/// Bodies and joints are added with [`add_body`](PyWorld::add_body) /
/// [`add_joint`](PyWorld::add_joint), which return the handles used by every
/// other method. The `bodies` and `joints` attributes are read-only
/// *snapshots*: modifying the returned sets does not change the world — use
/// `set_body`, `apply_impulse`, `remove_body`, `set_joint_enabled`, ... instead.
#[pyclass]
pub struct PyWorld {
    engine: PhysicsWorld,
    bodies: BTreeMap<u32, BodyEntry>,
    engine_to_py: HashMap<BodyHandle, u32>,
    joints: BTreeMap<u32, JointEntry>,
    next_body: u32,
    next_joint: u32,
    /// Engine sleep delay (seconds) used while sleeping is enabled.
    sleep_delay: f64,
    /// Pipeline configuration (gravity, sleeping, solver settings). Applied to
    /// the engine at the start of every substep. Reading it returns a copy, so
    /// change it by assigning a modified object back:
    /// `cfg = world.config; cfg.gravity = [0, 0, 0]; world.config = cfg`.
    #[pyo3(get, set)]
    pub config: PyPipelineConfig,
    contacts: Vec<PyContact>,
}

impl std::fmt::Debug for PyWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyWorld")
            .field("bodies", &self.bodies.len())
            .field("joints", &self.joints.len())
            .field("time", &self.engine.time)
            .field("config", &self.config)
            .finish()
    }
}

#[pymethods]
impl PyWorld {
    #[new]
    #[pyo3(signature = (gravity = [0.0, -9.81, 0.0]))]
    pub fn new(gravity: [f64; 3]) -> Self {
        let engine = PhysicsWorld::new(gravity, 1.0 / 60.0);
        let sleep_delay = engine.time_before_sleep;
        Self {
            engine,
            bodies: BTreeMap::new(),
            engine_to_py: HashMap::new(),
            joints: BTreeMap::new(),
            next_body: 0,
            next_joint: 0,
            sleep_delay,
            config: PyPipelineConfig::new(gravity),
            contacts: Vec::new(),
        }
    }

    /// Add a body (with its colliders) and return its handle.
    ///
    /// The `handle` attribute of `body` is ignored; the world assigns one.
    /// Raises `ValueError` for invalid input (non-positive mass of a dynamic
    /// body, non-finite state, unsupported or degenerate collider shapes).
    pub fn add_body(&mut self, body: PyRigidBody) -> PyResult<u32> {
        let built = build_engine_body(&body)?;
        let handle = self.next_body;
        self.next_body = self
            .next_body
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("body handle space exhausted"))?;
        let engine_handle = self.engine.add_rigid_body(built.body);
        let colliders = built
            .colliders
            .into_iter()
            .map(|c| self.engine.add_collider(c, Some(engine_handle)))
            .collect();
        let mut desc = body;
        desc.handle = handle;
        self.engine_to_py.insert(engine_handle, handle);
        self.bodies.insert(
            handle,
            BodyEntry {
                engine: engine_handle,
                colliders,
                desc,
            },
        );
        Ok(handle)
    }

    /// Remove a body, its colliders and every joint attached to it.
    ///
    /// Returns `True` if the body existed.
    pub fn remove_body(&mut self, handle: u32) -> bool {
        let Some(entry) = self.bodies.remove(&handle) else {
            return false;
        };
        for ch in entry.colliders {
            self.engine.colliders.remove(ch);
        }
        self.engine.remove_rigid_body(entry.engine);
        self.engine_to_py.remove(&entry.engine);
        self.joints
            .retain(|_, j| j.joint.body_a != handle && j.joint.body_b != handle);
        self.contacts
            .retain(|c| c.body_a != handle && c.body_b != handle);
        true
    }

    /// Current state of a body, or `None` for an unknown handle.
    ///
    /// The returned object is a copy; pass it to `set_body` to apply changes.
    pub fn get_body(&self, handle: u32) -> Option<PyRigidBody> {
        self.bodies.get(&handle).map(|e| self.snapshot(e))
    }

    /// Replace the state, mass, damping and colliders of the body with handle
    /// `body.handle` (as obtained from `get_body`). Joints keep their
    /// body-local anchors.
    ///
    /// Returns `False` if no such body exists; raises `ValueError` for invalid
    /// input.
    pub fn set_body(&mut self, body: PyRigidBody) -> PyResult<bool> {
        let handle = body.handle;
        let Some(entry) = self.bodies.get_mut(&handle) else {
            return Ok(false);
        };
        let built = build_engine_body(&body)?;
        let Some(slot) = self.engine.get_body_mut(entry.engine) else {
            return Ok(false);
        };
        *slot = built.body;
        for ch in entry.colliders.drain(..) {
            self.engine.colliders.remove(ch);
        }
        for c in built.colliders {
            let ch = self.engine.add_collider(c, Some(entry.engine));
            entry.colliders.push(ch);
        }
        entry.desc = body;
        Ok(true)
    }

    /// Apply a linear impulse (N·s) to a dynamic body's centre of mass and
    /// wake it. Returns `False` for an unknown handle.
    pub fn apply_impulse(&mut self, handle: u32, impulse: [f64; 3]) -> PyResult<bool> {
        if !finite3(impulse) {
            return Err(PyValueError::new_err("impulse must be finite"));
        }
        let Some(entry) = self.bodies.get(&handle) else {
            return Ok(false);
        };
        let Some(body) = self.engine.get_body_mut(entry.engine) else {
            return Ok(false);
        };
        body.wake_up();
        body.apply_impulse(vec3(impulse));
        Ok(true)
    }

    /// Add a joint and return its handle.
    ///
    /// `anchor_a`, `anchor_b` and `axis` are interpreted in world space at the
    /// moment the joint is added and are stored in the bodies' local frames.
    /// The `handle` attribute of `joint` is ignored. Raises `ValueError` for
    /// unknown or identical bodies and invalid joint parameters.
    pub fn add_joint(&mut self, joint: PyJoint) -> PyResult<u32> {
        if joint.body_a == joint.body_b {
            return Err(PyValueError::new_err("a joint needs two different bodies"));
        }
        let (Some(ea), Some(eb)) = (
            self.bodies.get(&joint.body_a),
            self.bodies.get(&joint.body_b),
        ) else {
            return Err(PyValueError::new_err(format!(
                "unknown body handle in joint ({}, {})",
                joint.body_a, joint.body_b
            )));
        };
        let (Some(a), Some(b)) = (
            self.engine.get_body(ea.engine),
            self.engine.get_body(eb.engine),
        ) else {
            return Err(PyValueError::new_err("joint body missing from the engine"));
        };
        let spec = JointSpec {
            kind: joint.joint_type,
            anchor_a: joint.anchor_a,
            anchor_b: joint.anchor_b,
            axis: joint.axis,
            target_distance: joint.target_distance,
            stiffness: joint.stiffness,
            enabled: joint.enabled,
        };
        let frame = JointFrame::capture(&spec, ea.engine, a, eb.engine, b)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let handle = self.next_joint;
        self.next_joint = self
            .next_joint
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("joint handle space exhausted"))?;
        let mut joint = joint;
        joint.handle = handle;
        self.joints.insert(handle, JointEntry { joint, frame });
        Ok(handle)
    }

    /// Remove a joint. Returns `True` if it existed.
    pub fn remove_joint(&mut self, handle: u32) -> bool {
        self.joints.remove(&handle).is_some()
    }

    /// The joint with the given handle (a copy), or `None`.
    pub fn get_joint(&self, handle: u32) -> Option<PyJoint> {
        self.joints.get(&handle).map(|j| j.joint.clone())
    }

    /// Enable or disable a joint. Returns `False` for an unknown handle.
    pub fn set_joint_enabled(&mut self, handle: u32, enabled: bool) -> bool {
        match self.joints.get_mut(&handle) {
            Some(j) => {
                j.joint.enabled = enabled;
                j.frame.enabled = enabled;
                true
            }
            None => false,
        }
    }

    /// Current world-space distance between the two anchors of a joint, or
    /// `None` for an unknown handle. Zero for a perfectly satisfied
    /// ball-socket, hinge or fixed joint.
    pub fn joint_anchor_gap(&self, handle: u32) -> Option<f64> {
        let j = self.joints.get(&handle)?;
        anchor_gap(&self.engine.bodies, &j.frame)
    }

    /// Advance the world by `dt` seconds in `substeps` equal substeps
    /// (default: `config.max_substeps`). Returns the number of substeps run.
    #[pyo3(signature = (dt, substeps = None))]
    pub fn step(&mut self, dt: f64, substeps: Option<u32>) -> PyResult<u32> {
        let n = substeps.unwrap_or(self.config.max_substeps);
        self.advance(dt, n)
    }

    /// Contacts of the most recent substep.
    pub fn get_contacts(&self) -> Vec<PyContact> {
        self.contacts.clone()
    }

    /// Number of bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of joints.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    /// Number of bodies currently asleep.
    pub fn sleeping_body_count(&self) -> usize {
        self.engine.sleeping_body_count()
    }

    /// Accumulated simulation time in seconds.
    pub fn elapsed_time(&self) -> f64 {
        self.engine.time
    }

    /// Accumulated simulation time in seconds.
    #[getter]
    pub fn time(&self) -> f64 {
        self.engine.time
    }

    /// Read-only snapshot of all bodies (modifying it does not affect the
    /// world).
    #[getter]
    pub fn bodies(&self) -> PyRigidBodySet {
        PyRigidBodySet {
            bodies: self.bodies.values().map(|e| self.snapshot(e)).collect(),
            next_handle: self.next_body,
        }
    }

    /// Read-only snapshot of all joints (modifying it does not affect the
    /// world).
    #[getter]
    pub fn joints(&self) -> PyJointSet {
        PyJointSet::from_parts(
            self.joints.values().map(|j| j.joint.clone()).collect(),
            self.next_joint,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWorld(bodies={}, joints={}, time={:.3})",
            self.bodies.len(),
            self.joints.len(),
            self.engine.time
        )
    }
}

impl PyWorld {
    /// Python-side view of a body: its description with the engine state.
    fn snapshot(&self, entry: &BodyEntry) -> PyRigidBody {
        let mut out = entry.desc.clone();
        if let Some(b) = self.engine.get_body(entry.engine) {
            out.position = arr3(&b.transform.position);
            out.velocity = arr3(&b.velocity);
            out.orientation = quat_to_xyzw(&b.transform.rotation);
            out.angular_velocity = arr3(&b.angular_velocity);
            out.linear_damping = b.linear_damping;
            out.angular_damping = b.angular_damping;
        }
        out
    }

    /// Copy the pipeline configuration into the engine for a substep `h`.
    fn sync_config(&mut self, h: f64) {
        let cfg = &self.config;
        self.engine.gravity = cfg.gravity;
        self.engine.dt = h;
        self.engine.linear_sleep_threshold = cfg.linear_sleep_threshold.max(0.0);
        self.engine.angular_sleep_threshold = cfg.angular_sleep_threshold.max(0.0);
        self.engine.time_before_sleep = if cfg.sleep_enabled {
            self.sleep_delay
        } else {
            f64::INFINITY
        };
        self.engine.solver_config.baumgarte_factor = cfg.baumgarte_factor.clamp(0.0, 1.0);
        let never_sleep = !cfg.sleep_enabled;
        for entry in self.bodies.values() {
            if !never_sleep && entry.desc.can_sleep {
                continue;
            }
            if let Some(b) = self.engine.get_body_mut(entry.engine) {
                b.sleep_timer = 0.0;
                if b.state == BodyState::Sleeping {
                    b.wake_up();
                }
            }
        }
    }

    /// Advance by `dt` in `substeps` substeps; returns the substeps run.
    pub(crate) fn advance(&mut self, dt: f64, substeps: u32) -> PyResult<u32> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be finite and > 0"));
        }
        if !finite3(self.config.gravity) {
            return Err(PyValueError::new_err("config.gravity must be finite"));
        }
        let cfg = &self.config;
        if !cfg.baumgarte_factor.is_finite()
            || !cfg.linear_sleep_threshold.is_finite()
            || !cfg.angular_sleep_threshold.is_finite()
        {
            return Err(PyValueError::new_err(
                "config.baumgarte_factor and the sleep thresholds must be finite",
            ));
        }
        let n = substeps.max(1);
        let h = dt / f64::from(n);
        let iterations = self.config.joint_iterations;
        for _ in 0..n {
            self.sync_config(h);
            self.engine.step();
            let frames: Vec<&JointFrame> = self.joints.values().map(|j| &j.frame).collect();
            solve_joints(&mut self.engine.bodies, &frames, h, iterations);
        }
        self.contacts = self.collect_contacts();
        Ok(n)
    }

    /// Engine contacts of the last substep with Python body handles.
    fn collect_contacts(&self) -> Vec<PyContact> {
        self.engine
            .last_contacts
            .iter()
            .filter_map(|cp| {
                let a = *self.engine_to_py.get(&cp.body_a)?;
                let b = *self.engine_to_py.get(&cp.body_b)?;
                // A contact the solver did not act on received no impulse.
                let impulse = self.engine.contact_normal_impulse(cp).unwrap_or(0.0);
                Some(PyContact::new(
                    a,
                    b,
                    cp.contact_point,
                    cp.normal,
                    cp.depth,
                    impulse,
                ))
            })
            .collect()
    }

    /// Engine view of the bodies (tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn engine(&self) -> &PhysicsWorld {
        &self.engine
    }

    /// Number of engine colliders (tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn collider_count(&self) -> usize {
        self.engine.colliders.len()
    }
}

// ---------------------------------------------------------------------------
// PyPhysicsPipeline
// ---------------------------------------------------------------------------

/// A step executor for `PyWorld` that counts the substeps it ran.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct PyPhysicsPipeline {
    /// Number of substeps executed.
    #[pyo3(get)]
    pub step_count: u64,
}

#[pymethods]
impl PyPhysicsPipeline {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Step the world forward by `dt` seconds using `substeps` substeps
    /// (at least one).
    ///
    /// Every substep runs the engine pipeline (forces, broadphase,
    /// narrowphase, contact solver, integration, sleeping) followed by the
    /// joint projection; see `PyWorld`.
    pub fn step(&mut self, world: &mut PyWorld, dt: f64, substeps: u32) -> PyResult<()> {
        let n = world.advance(dt, substeps)?;
        self.step_count = self.step_count.saturating_add(u64::from(n));
        Ok(())
    }

    /// Return total number of substeps executed.
    pub fn total_steps(&self) -> u64 {
        self.step_count
    }

    fn __repr__(&self) -> String {
        format!("PyPhysicsPipeline(steps={})", self.step_count)
    }
}
