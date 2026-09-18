// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `WasmConstraintSolver`: a projected Gauss-Seidel (sequential impulse)
//! joint solver acting on the bodies of a [`WasmPhysicsEngine`].
//!
//! Every joint is turned into Jacobian rows `J = [lin_a, ang_a, lin_b, ang_b]`
//! acting on the linear and angular velocities of its two bodies:
//!
//! | joint       | rows                                                      |
//! |-------------|-----------------------------------------------------------|
//! | `ball`      | 3 linear (anchor points coincide)                         |
//! | `revolute`  | 3 linear + 2 angular (axes stay aligned) + motor + limit  |
//! | `prismatic` | 2 linear (perpendicular to the axis) + 3 angular + motor + limit |
//! | `fixed`     | 3 linear + 3 angular                                      |
//! | `spring`    | 1 soft row along the anchor line + length limit           |
//!
//! Each row is solved with the effective mass `1 / (J M^-1 J^T)` built from
//! the body masses and world-space inverse inertia tensors; positional drift
//! is removed with Baumgarte stabilisation.  Accumulated impulses are clamped
//! (motors to `motor_max_force * dt`, limits to one sign) and optionally
//! warm-started from the previous solve.  Sweeps stop early once the total
//! impulse change of a sweep drops below `1e-9`; the number of sweeps really
//! performed is reported in `iterations_used`.
//!
//! The spring row is the implicit (unconditionally stable) soft constraint
//! for the force `F = -k (L - L0) - c dL/dt`.
//!
//! ## Usage
//!
//! Call [`WasmConstraintSolver::solve`] right before
//! [`WasmPhysicsEngine::step`] with the same `dt` (one fixed engine substep),
//! or use [`WasmConstraintSolver::step`] which does both.  The solver works on
//! the velocities the next substep will produce from gravity and the
//! accumulated forces, so the joints hold under those loads without lagging
//! one step behind.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use super::{WasmJointConfig, WasmJointHandle, WasmJointState, WasmMotorTarget};
use crate::engine::WasmPhysicsEngine;
use crate::error::{Error, Result};
use crate::simulation_api::vec_math::{
    M3, M3_ZERO, Q_IDENTITY, Q4, V3, m_vec, q_apply_rotation_vector, q_conj, q_mul, q_rotate,
    q_to_rotation_vector, v_add, v_cross, v_dot, v_length, v_madd, v_neg, v_normalized, v_sub,
    v_tangent_basis, wrap_angle,
};
use crate::wasm_helpers::{err_to_jsvalue, to_js_value};

/// Body id that stands for the static world frame in
/// [`WasmJointConfig::body_a`] / [`WasmJointConfig::body_b`].
///
/// Anchors and axes given for the world are world-space coordinates.
pub const WORLD_BODY: u64 = u32::MAX as u64;

/// Default Baumgarte stabilisation factor.
pub const DEFAULT_BAUMGARTE: f64 = 0.2;

/// Default number of position-projection passes.
pub const DEFAULT_POSITION_ITERATIONS: u32 = 3;

/// Rigid-row violation below which no position projection is needed.
const POSITION_TOLERANCE: f64 = 1e-10;

/// Total impulse change of one sweep below which the solver stops early.
const CONVERGENCE_TOLERANCE: f64 = 1e-9;

/// Number of row slots per joint (warm-start storage).
const MAX_ROWS: usize = 8;
/// First angular slot.
const SLOT_ANGULAR: usize = 3;
/// Slot of the motor row.
const SLOT_MOTOR: usize = 6;
/// Slot of the limit row.
const SLOT_LIMIT: usize = 7;

fn default_baumgarte() -> f64 {
    DEFAULT_BAUMGARTE
}

fn default_position_iterations() -> u32 {
    DEFAULT_POSITION_ITERATIONS
}

/// Per-joint data that persists between solves.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct JointCache {
    /// Relative orientation `conj(q_a) * q_b` captured at the first solve;
    /// the zero of revolute angles and the target of angular locks.
    reference: Option<Q4>,
    /// Accumulated impulse of each row slot from the last solve.
    lambdas: [f64; MAX_ROWS],
}

/// Constraint solver managing a collection of joints between engine bodies.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConstraintSolver {
    /// Maximum number of Gauss-Seidel sweeps per solve.
    pub iterations: u32,
    /// Whether to warm-start from the previous solve's accumulated impulses.
    pub warm_start: bool,
    /// Baumgarte stabilisation factor in `[0, 1]` (fraction of the positional
    /// error removed per step).
    #[serde(default = "default_baumgarte")]
    pub baumgarte: f64,
    /// Number of nonlinear position-projection passes run by
    /// [`stabilize_positions`](Self::stabilize_positions) (and `step`).
    #[serde(default = "default_position_iterations")]
    pub position_iterations: u32,
    /// All joints currently managed by this solver.
    pub(crate) joints: Vec<(WasmJointHandle, WasmJointConfig)>,
    /// Per-joint state from the last solve.
    pub(crate) joint_states: Vec<WasmJointState>,
    /// Per-joint reference frames and warm-start impulses.
    #[serde(default)]
    pub(crate) caches: Vec<JointCache>,
    /// Next handle to assign.
    next_handle: u32,
    /// Predicted positional constraint error after the last solve: the sum of
    /// `|C + dt * J v|` over all rigid rows (m and rad).
    pub last_error: f64,
    /// Number of sweeps actually performed in the last solve.
    pub iterations_used: u32,
}

/// The joint types understood by the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JointKind {
    Ball,
    Fixed,
    Revolute,
    Prismatic,
    Spring,
}

fn joint_kind(cfg: &WasmJointConfig) -> Result<JointKind> {
    match cfg.joint_type.as_str() {
        "ball" => Ok(JointKind::Ball),
        "fixed" => Ok(JointKind::Fixed),
        "revolute" => Ok(JointKind::Revolute),
        "prismatic" => Ok(JointKind::Prismatic),
        "spring" => Ok(JointKind::Spring),
        other => Err(Error::InvalidParameter {
            name: "joint_type".to_string(),
            message: format!(
                "unknown joint type '{other}' (expected revolute, prismatic, ball, fixed or spring)"
            ),
        }),
    }
}

/// A body taking part in the solve.
#[derive(Debug, Clone)]
struct SolverBody {
    /// Engine handle (`None` for the world frame).
    handle: Option<u32>,
    position: V3,
    rotation: Q4,
    /// Velocity the solver works on (includes the predicted external change).
    velocity: V3,
    angular_velocity: V3,
    inv_mass: f64,
    inv_inertia: M3,
    /// Predicted velocity change of the next engine substep.
    external: V3,
    velocity_start: V3,
    angular_start: V3,
}

impl SolverBody {
    fn world() -> Self {
        Self {
            handle: None,
            position: [0.0; 3],
            rotation: Q_IDENTITY,
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            inv_mass: 0.0,
            inv_inertia: M3_ZERO,
            external: [0.0; 3],
            velocity_start: [0.0; 3],
            angular_start: [0.0; 3],
        }
    }
}

/// Resolve a body id to an index in `bodies`, loading it from the engine on
/// first use.
fn body_index(
    bodies: &mut Vec<SolverBody>,
    engine: &WasmPhysicsEngine,
    id: u64,
    field: &str,
    dt: f64,
) -> Result<usize> {
    if id == WORLD_BODY {
        if let Some(i) = bodies.iter().position(|b| b.handle.is_none()) {
            return Ok(i);
        }
        bodies.push(SolverBody::world());
        return Ok(bodies.len() - 1);
    }
    let handle = u32::try_from(id).map_err(|_| Error::InvalidParameter {
        name: field.to_string(),
        message: format!("body id {id} does not fit an engine handle (u32)"),
    })?;
    if let Some(i) = bodies.iter().position(|b| b.handle == Some(handle)) {
        return Ok(i);
    }
    let state = engine
        .joint_body_state(handle, dt)
        .ok_or(Error::BodyNotFound(handle))?;
    let velocity = v_add(state.velocity, state.external_velocity_change);
    bodies.push(SolverBody {
        handle: Some(handle),
        position: state.position,
        rotation: state.rotation,
        velocity,
        angular_velocity: state.angular_velocity,
        inv_mass: state.inv_mass,
        inv_inertia: state.inv_inertia_world,
        external: state.external_velocity_change,
        velocity_start: velocity,
        angular_start: state.angular_velocity,
    });
    Ok(bodies.len() - 1)
}

/// What a row constrains (for force reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowRole {
    /// Rigid linear row (reaction force).
    Linear,
    /// Rigid angular row (reaction torque).
    Angular,
    /// Motor drive.
    Motor,
    /// Linear limit or spring row.
    LinearSoft,
    /// Angular limit row.
    AngularLimit,
}

/// Jacobian of one scalar constraint.
#[derive(Debug, Clone, Copy)]
struct Jacobian {
    lin_a: V3,
    ang_a: V3,
    lin_b: V3,
    ang_b: V3,
}

impl Jacobian {
    /// Relative velocity of the anchor points along `n`; `lever_a` /
    /// `lever_b` are the lever arms from the body centres.
    fn linear(n: V3, lever_a: V3, lever_b: V3) -> Self {
        Self {
            lin_a: v_neg(n),
            ang_a: v_neg(v_cross(lever_a, n)),
            lin_b: n,
            ang_b: v_cross(lever_b, n),
        }
    }

    /// Relative angular velocity about `u`.
    fn angular(u: V3) -> Self {
        Self {
            lin_a: [0.0; 3],
            ang_a: v_neg(u),
            lin_b: [0.0; 3],
            ang_b: u,
        }
    }

    fn negated(self) -> Self {
        Self {
            lin_a: v_neg(self.lin_a),
            ang_a: v_neg(self.ang_a),
            lin_b: v_neg(self.lin_b),
            ang_b: v_neg(self.ang_b),
        }
    }

    fn velocity(&self, a: &SolverBody, b: &SolverBody) -> f64 {
        self.apply_to(
            (a.velocity, a.angular_velocity),
            (b.velocity, b.angular_velocity),
        )
    }

    /// `J . (lin_a, ang_a, lin_b, ang_b)` for arbitrary generalised vectors
    /// (velocities or displacements) of the two bodies.
    fn apply_to(&self, a: (V3, V3), b: (V3, V3)) -> f64 {
        v_dot(self.lin_a, a.0)
            + v_dot(self.ang_a, a.1)
            + v_dot(self.lin_b, b.0)
            + v_dot(self.ang_b, b.1)
    }

    /// `J M^-1 J^T`.
    fn inverse_mass(&self, a: &SolverBody, b: &SolverBody) -> f64 {
        a.inv_mass * v_dot(self.lin_a, self.lin_a)
            + v_dot(self.ang_a, m_vec(&a.inv_inertia, self.ang_a))
            + b.inv_mass * v_dot(self.lin_b, self.lin_b)
            + v_dot(self.ang_b, m_vec(&b.inv_inertia, self.ang_b))
    }
}

/// One scalar constraint row.
#[derive(Debug, Clone)]
struct Row {
    joint: usize,
    slot: usize,
    role: RowRole,
    ia: usize,
    ib: usize,
    jac: Jacobian,
    /// Effective mass `1 / (J M^-1 J^T + gamma)`.
    mass: f64,
    /// Velocity bias: the row drives `J v + bias + gamma * lambda` to zero.
    bias: f64,
    /// Softness (spring rows); zero for rigid rows.
    gamma: f64,
    lo: f64,
    hi: f64,
    lambda: f64,
    /// Positional error `C` (only meaningful when `rigid`).
    error: f64,
    /// Whether the row is a rigid position constraint counted in `last_error`.
    rigid: bool,
    /// Whether the row is a one-sided limit (`C >= 0` is admissible).
    unilateral: bool,
    /// Whether the row is a rigid equality solved together with the other
    /// equality rows of its joint (block solve).
    block: bool,
}

impl Row {
    /// Positional violation of the row (zero for a satisfied limit).
    fn violation(&self) -> f64 {
        if self.unilateral {
            self.error.min(0.0)
        } else {
            self.error
        }
    }
}

/// Parameters of a row before its effective mass is known.
#[derive(Debug, Clone, Copy)]
struct RowSpec {
    slot: usize,
    role: RowRole,
    jac: Jacobian,
    bias: f64,
    gamma: f64,
    lo: f64,
    hi: f64,
    error: f64,
    rigid: bool,
    block: bool,
}

/// Measured joint coordinate for state reporting.
#[derive(Debug, Clone, Copy)]
struct JointMeasure {
    position: f64,
    /// Jacobian whose `J v` is the coordinate rate, if the joint has one.
    rate: Option<Jacobian>,
    ia: usize,
    ib: usize,
    limit_active: bool,
}

/// Row construction context of one joint.
struct RowBuilder<'a> {
    rows: &'a mut Vec<Row>,
    bodies: &'a [SolverBody],
    joint: usize,
    ia: usize,
    ib: usize,
    beta_over_dt: f64,
    inv_dt: f64,
}

impl RowBuilder<'_> {
    /// Push a row; rows between two immovable bodies are dropped.
    fn push(&mut self, spec: RowSpec) {
        let a = &self.bodies[self.ia];
        let b = &self.bodies[self.ib];
        let k = spec.jac.inverse_mass(a, b) + spec.gamma;
        if !k.is_finite() || k <= 1e-15 {
            return;
        }
        self.rows.push(Row {
            joint: self.joint,
            slot: spec.slot,
            role: spec.role,
            ia: self.ia,
            ib: self.ib,
            jac: spec.jac,
            mass: 1.0 / k,
            bias: spec.bias,
            gamma: spec.gamma,
            lo: spec.lo,
            hi: spec.hi,
            lambda: 0.0,
            error: spec.error,
            rigid: spec.rigid,
            unilateral: spec.rigid && spec.lo >= 0.0,
            block: spec.block,
        });
    }

    /// Rigid equality row with Baumgarte bias for positional error `c`.
    fn equality(&mut self, slot: usize, role: RowRole, jac: Jacobian, c: f64) {
        self.push(RowSpec {
            slot,
            role,
            jac,
            bias: self.beta_over_dt * c,
            gamma: 0.0,
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
            error: c,
            rigid: true,
            block: true,
        });
    }

    /// One-sided limit row: keeps `c >= 0` for the coordinate whose rate is
    /// `jac . v` (`c` is negative when the limit is violated).
    ///
    /// The row is speculative: while the limit is not reached (`c > 0`) it
    /// lets the coordinate approach the limit by exactly `c` this step, so a
    /// fast joint stops at the limit instead of overshooting and bouncing
    /// back; a violated limit is pushed out with the Baumgarte factor.
    fn limit(&mut self, role: RowRole, jac: Jacobian, c: f64) {
        let bias = if c > 0.0 {
            c * self.inv_dt
        } else {
            self.beta_over_dt * c
        };
        self.push(RowSpec {
            slot: SLOT_LIMIT,
            role,
            jac,
            bias,
            gamma: 0.0,
            lo: 0.0,
            hi: f64::INFINITY,
            error: c,
            rigid: true,
            block: false,
        });
    }

    /// Limit row for the bound of `range = (lower, upper)` nearest to
    /// `coordinate` (whose rate is `jac . v`).  Returns whether the limit is
    /// reached or violated.
    fn range_limit(
        &mut self,
        role: RowRole,
        jac: Jacobian,
        coordinate: f64,
        range: (f64, f64),
    ) -> Result<bool> {
        let (lower, upper) = range;
        if lower.is_nan() || upper.is_nan() || lower > upper {
            return Err(Error::InvalidParameter {
                name: "lower_limit".to_string(),
                message: format!("limit range [{lower}, {upper}] is empty"),
            });
        }
        let (jac, c) = if coordinate < 0.5 * (lower + upper) || (upper.is_infinite() && upper > 0.0)
        {
            (jac, coordinate - lower)
        } else {
            (jac.negated(), upper - coordinate)
        };
        if !c.is_finite() {
            // Unbounded on the near side: nothing to enforce.
            return Ok(false);
        }
        self.limit(role, jac, c);
        Ok(c <= 0.0)
    }

    /// Motor row driving `jac . v` toward `target_velocity` with an impulse
    /// bounded by `max_impulse`.
    fn motor(&mut self, jac: Jacobian, target_velocity: f64, max_impulse: f64) {
        self.push(RowSpec {
            slot: SLOT_MOTOR,
            role: RowRole::Motor,
            jac,
            bias: -target_velocity,
            gamma: 0.0,
            lo: -max_impulse,
            hi: max_impulse,
            error: 0.0,
            rigid: false,
            block: false,
        });
    }
}

/// Unit world-space direction of a body-local axis.
fn world_axis(body: &SolverBody, local: V3, field: &str) -> Result<(V3, V3)> {
    let local_unit = v_normalized(local).ok_or_else(|| Error::InvalidParameter {
        name: field.to_string(),
        message: "joint axis must be a non-zero finite vector".to_string(),
    })?;
    Ok((local_unit, q_rotate(body.rotation, local_unit)))
}

/// Rotation of body B relative to body A since the reference was captured,
/// as a rotation vector in A's local frame and its quaternion.
fn relative_rotation(a: &SolverBody, b: &SolverBody, reference: Q4) -> (V3, Q4) {
    let q_ab = q_mul(q_conj(a.rotation), b.rotation);
    let q_delta = q_mul(q_ab, q_conj(reference));
    (q_to_rotation_vector(q_delta), q_delta)
}

/// Twist angle of `q_delta` about the unit local axis, in `(-pi, pi]`.
fn twist_angle(q_delta: Q4, axis_local: V3) -> f64 {
    let s = v_dot([q_delta[0], q_delta[1], q_delta[2]], axis_local);
    wrap_angle(2.0 * s.atan2(q_delta[3]))
}

/// Build the rows of one joint and return its measured coordinate.
fn build_joint_rows(
    kind: JointKind,
    cfg: &WasmJointConfig,
    cache: &mut JointCache,
    builder: &mut RowBuilder<'_>,
    dt: f64,
) -> Result<JointMeasure> {
    let a = builder.bodies[builder.ia].clone();
    let b = builder.bodies[builder.ib].clone();
    let r_a = q_rotate(a.rotation, cfg.anchor_a);
    let r_b = q_rotate(b.rotation, cfg.anchor_b);
    let p_a = v_add(a.position, r_a);
    let p_b = v_add(b.position, r_b);
    let d = v_sub(p_b, p_a);
    let reference = *cache
        .reference
        .get_or_insert_with(|| q_mul(q_conj(a.rotation), b.rotation));
    let max_impulse = if cfg.motor_max_force.is_finite() {
        cfg.motor_max_force.max(0.0) * dt
    } else {
        f64::INFINITY
    };
    let mut measure = JointMeasure {
        position: 0.0,
        rate: None,
        ia: builder.ia,
        ib: builder.ib,
        limit_active: false,
    };

    let point_rows = |builder: &mut RowBuilder<'_>| {
        for (axis_index, axis) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .into_iter()
            .enumerate()
        {
            builder.equality(
                axis_index,
                RowRole::Linear,
                Jacobian::linear(axis, r_a, r_b),
                d[axis_index],
            );
        }
    };
    let angular_lock_rows = |builder: &mut RowBuilder<'_>| {
        let (err_local, _) = relative_rotation(&a, &b, reference);
        let err_world = q_rotate(a.rotation, err_local);
        for (i, axis) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .into_iter()
            .enumerate()
        {
            builder.equality(
                SLOT_ANGULAR + i,
                RowRole::Angular,
                Jacobian::angular(axis),
                err_world[i],
            );
        }
    };

    match kind {
        JointKind::Ball => point_rows(builder),
        JointKind::Fixed => {
            point_rows(builder);
            angular_lock_rows(builder);
        }
        JointKind::Revolute => {
            point_rows(builder);
            let (axis_local, axis_a) = world_axis(&a, cfg.axis_a, "axis_a")?;
            let (_, axis_b) = world_axis(&b, cfg.axis_b, "axis_b")?;
            // Small rotation that would carry axis_a onto axis_b.
            let misalignment = v_cross(axis_a, axis_b);
            let (t1, t2) = v_tangent_basis(axis_a);
            builder.equality(
                SLOT_ANGULAR,
                RowRole::Angular,
                Jacobian::angular(t1),
                v_dot(misalignment, t1),
            );
            builder.equality(
                SLOT_ANGULAR + 1,
                RowRole::Angular,
                Jacobian::angular(t2),
                v_dot(misalignment, t2),
            );
            let (_, q_delta) = relative_rotation(&a, &b, reference);
            let angle = twist_angle(q_delta, axis_local);
            let axis_jac = Jacobian::angular(axis_a);
            measure.position = angle;
            measure.rate = Some(axis_jac);
            if cfg.motor_enabled {
                let target = match cfg.motor_target_position {
                    Some(p) => wrap_angle(p - angle) / dt,
                    None => cfg.motor_target_velocity,
                };
                builder.motor(axis_jac, target, max_impulse);
            }
            if cfg.limits_enabled {
                measure.limit_active = builder.range_limit(
                    RowRole::AngularLimit,
                    axis_jac,
                    angle,
                    (cfg.lower_limit, cfg.upper_limit),
                )?;
            }
        }
        JointKind::Prismatic => {
            let (_, axis) = world_axis(&a, cfg.axis_a, "axis_a")?;
            let (t1, t2) = v_tangent_basis(axis);
            // Body A's lever reaches the anchor of B so that the rotation of
            // A's axis is accounted for.
            let lever_a = v_add(r_a, d);
            builder.equality(
                0,
                RowRole::Linear,
                Jacobian::linear(t1, lever_a, r_b),
                v_dot(d, t1),
            );
            builder.equality(
                1,
                RowRole::Linear,
                Jacobian::linear(t2, lever_a, r_b),
                v_dot(d, t2),
            );
            angular_lock_rows(builder);
            let translation = v_dot(d, axis);
            let axis_jac = Jacobian::linear(axis, lever_a, r_b);
            measure.position = translation;
            measure.rate = Some(axis_jac);
            if cfg.motor_enabled {
                let target = match cfg.motor_target_position {
                    Some(p) => (p - translation) / dt,
                    None => cfg.motor_target_velocity,
                };
                builder.motor(axis_jac, target, max_impulse);
            }
            if cfg.limits_enabled {
                measure.limit_active = builder.range_limit(
                    RowRole::LinearSoft,
                    axis_jac,
                    translation,
                    (cfg.lower_limit, cfg.upper_limit),
                )?;
            }
        }
        JointKind::Spring => {
            let length = v_length(d);
            measure.position = length;
            if let Some(n) = v_normalized(d) {
                let jac = Jacobian::linear(n, r_a, r_b);
                measure.rate = Some(jac);
                let k = cfg.spring_stiffness.max(0.0);
                let c = cfg.spring_damping.max(0.0);
                let softness = dt * (c + dt * k);
                if softness > 0.0 && softness.is_finite() {
                    // Implicit spring-damper as a soft constraint.
                    let gamma = 1.0 / softness;
                    let bias = (length - cfg.spring_rest_length) * dt * k * gamma;
                    builder.push(RowSpec {
                        slot: 0,
                        role: RowRole::LinearSoft,
                        jac,
                        bias,
                        gamma,
                        lo: f64::NEG_INFINITY,
                        hi: f64::INFINITY,
                        error: 0.0,
                        rigid: false,
                        block: false,
                    });
                }
                if cfg.limits_enabled {
                    measure.limit_active = builder.range_limit(
                        RowRole::LinearSoft,
                        jac,
                        length,
                        (cfg.lower_limit, cfg.upper_limit),
                    )?;
                }
            }
        }
    }
    Ok(measure)
}

/// Apply the (pseudo-)impulse `delta` of a row to the generalised
/// velocities or displacements `(linear, angular)` of its bodies:
/// `gen += M^-1 J^T delta`.
fn apply_row(bodies: &[SolverBody], generalized: &mut [(V3, V3)], row: &Row, delta: f64) {
    let (a, b) = (&bodies[row.ia], &bodies[row.ib]);
    let ga = &mut generalized[row.ia];
    ga.0 = v_madd(ga.0, row.jac.lin_a, a.inv_mass * delta);
    ga.1 = v_madd(ga.1, m_vec(&a.inv_inertia, row.jac.ang_a), delta);
    let gb = &mut generalized[row.ib];
    gb.0 = v_madd(gb.0, row.jac.lin_b, b.inv_mass * delta);
    gb.1 = v_madd(gb.1, m_vec(&b.inv_inertia, row.jac.ang_b), delta);
}

/// Largest number of equality rows solved as one block (fixed joint).
const MAX_BLOCK: usize = 6;

/// A unit of the Gauss-Seidel sweep: one scalar row or a block of the
/// coupled equality rows of a joint, solved exactly with `K^-1`.
#[derive(Debug, Clone)]
enum Unit {
    Single(usize),
    Block {
        start: usize,
        len: usize,
        k_inv: Box<[[f64; MAX_BLOCK]; MAX_BLOCK]>,
    },
}

/// Coupling `J_i M^-1 J_j^T` of two rows of the same joint.
fn coupling(ri: &Row, rj: &Row, bodies: &[SolverBody]) -> f64 {
    let (a, b) = (&bodies[ri.ia], &bodies[ri.ib]);
    a.inv_mass * v_dot(ri.jac.lin_a, rj.jac.lin_a)
        + v_dot(ri.jac.ang_a, m_vec(&a.inv_inertia, rj.jac.ang_a))
        + b.inv_mass * v_dot(ri.jac.lin_b, rj.jac.lin_b)
        + v_dot(ri.jac.ang_b, m_vec(&b.inv_inertia, rj.jac.ang_b))
}

/// Inverse of the block matrix `K` of `rows` (Gauss-Jordan with partial
/// pivoting); `None` when `K` is (numerically) singular.
fn block_inverse(rows: &[Row], bodies: &[SolverBody]) -> Option<[[f64; MAX_BLOCK]; MAX_BLOCK]> {
    let n = rows.len();
    let mut k = [[0.0; MAX_BLOCK]; MAX_BLOCK];
    let mut inv = [[0.0; MAX_BLOCK]; MAX_BLOCK];
    let mut scale = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            k[i][j] = coupling(&rows[i], &rows[j], bodies);
        }
        inv[i][i] = 1.0;
        scale = scale.max(k[i][i].abs());
    }
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&x, &y| k[x][col].abs().total_cmp(&k[y][col].abs()))?;
        if k[pivot_row][col].abs() <= 1e-10 * scale {
            return None;
        }
        k.swap(col, pivot_row);
        inv.swap(col, pivot_row);
        let p = 1.0 / k[col][col];
        for c in 0..n {
            k[col][c] *= p;
            inv[col][c] *= p;
        }
        for r in 0..n {
            if r != col {
                let f = k[r][col];
                if f != 0.0 {
                    for c in 0..n {
                        k[r][c] -= f * k[col][c];
                        inv[r][c] -= f * inv[col][c];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Group the rows into sweep units: consecutive block rows of one joint are
/// solved together; blocks with a singular `K` fall back to scalar rows.
fn build_units(rows: &[Row], bodies: &[SolverBody]) -> Vec<Unit> {
    let mut units = Vec::with_capacity(rows.len());
    let mut i = 0;
    while i < rows.len() {
        if !rows[i].block {
            units.push(Unit::Single(i));
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < rows.len()
            && end - i < MAX_BLOCK
            && rows[end].block
            && rows[end].joint == rows[i].joint
        {
            end += 1;
        }
        let len = end - i;
        match (len > 1)
            .then(|| block_inverse(&rows[i..end], bodies))
            .flatten()
        {
            Some(k_inv) => units.push(Unit::Block {
                start: i,
                len,
                k_inv: Box::new(k_inv),
            }),
            None => units.extend((i..end).map(Unit::Single)),
        }
        i = end;
    }
    units
}

/// Projected Gauss-Seidel: up to `max_sweeps` sweeps driving
/// `J gen + bias + gamma * lambda` to zero for every row (within the impulse
/// bounds).  Returns the number of sweeps performed.
fn run_sweeps(
    rows: &mut [Row],
    units: &[Unit],
    bodies: &[SolverBody],
    generalized: &mut [(V3, V3)],
    max_sweeps: u32,
) -> u32 {
    let mut sweeps = 0;
    for _ in 0..max_sweeps {
        let mut total_change = 0.0;
        for unit in units {
            match unit {
                Unit::Single(index) => {
                    let row = &mut rows[*index];
                    let jv = row.jac.apply_to(generalized[row.ia], generalized[row.ib]);
                    let delta = -row.mass * (jv + row.bias + row.gamma * row.lambda);
                    let new_lambda = (row.lambda + delta).clamp(row.lo, row.hi);
                    let applied = new_lambda - row.lambda;
                    row.lambda = new_lambda;
                    if applied != 0.0 {
                        apply_row(bodies, generalized, row, applied);
                        total_change += applied.abs();
                    }
                }
                Unit::Block { start, len, k_inv } => {
                    let block = &mut rows[*start..*start + *len];
                    let mut rhs = [0.0; MAX_BLOCK];
                    for (r, row) in rhs.iter_mut().zip(block.iter()) {
                        *r = -(row.jac.apply_to(generalized[row.ia], generalized[row.ib])
                            + row.bias);
                    }
                    for (i, row) in block.iter_mut().enumerate() {
                        let delta: f64 = (0..*len).map(|j| k_inv[i][j] * rhs[j]).sum();
                        if delta != 0.0 {
                            row.lambda += delta;
                            apply_row(bodies, generalized, row, delta);
                            total_change += delta.abs();
                        }
                    }
                }
            }
        }
        sweeps += 1;
        if total_change < CONVERGENCE_TOLERANCE {
            break;
        }
    }
    sweeps
}

/// Bodies, rows and joint measurements of one linearisation.
struct Gathered {
    bodies: Vec<SolverBody>,
    rows: Vec<Row>,
    measures: Vec<JointMeasure>,
}

/// Change of a vector large enough to be written back.
fn changed(a: V3, b: V3) -> bool {
    v_length(v_sub(a, b)) > 1e-12
}

impl WasmConstraintSolver {
    /// Create a new solver with `iterations` sweeps and optional warm-starting.
    pub fn new(iterations: u32, warm_start: bool) -> Self {
        WasmConstraintSolver {
            iterations,
            warm_start,
            baumgarte: DEFAULT_BAUMGARTE,
            position_iterations: DEFAULT_POSITION_ITERATIONS,
            joints: Vec::new(),
            joint_states: Vec::new(),
            caches: Vec::new(),
            next_handle: 1,
            last_error: 0.0,
            iterations_used: 0,
        }
    }

    /// Add a joint and return its handle.
    ///
    /// The relative orientation of the two bodies at the first solve becomes
    /// the joint's reference (zero revolute angle, locked orientation of
    /// fixed and prismatic joints).
    pub fn add_joint(&mut self, config: WasmJointConfig) -> WasmJointHandle {
        let handle = WasmJointHandle::new(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1).max(1);
        self.joints.push((handle, config));
        self.joint_states.push(WasmJointState::new());
        self.caches.push(JointCache::default());
        handle
    }

    /// Remove a joint by handle. Returns `true` if found and removed.
    pub fn remove_joint(&mut self, handle: WasmJointHandle) -> bool {
        if let Some(pos) = self.joints.iter().position(|(h, _)| *h == handle) {
            self.joints.remove(pos);
            self.joint_states.remove(pos);
            if pos < self.caches.len() {
                self.caches.remove(pos);
            }
            true
        } else {
            false
        }
    }

    /// Load the bodies of all joints from `engine` and build the joint rows
    /// for a step of `dt` (captures missing joint references).
    fn gather(&mut self, engine: &WasmPhysicsEngine, dt: f64) -> Result<Gathered> {
        self.caches
            .resize_with(self.joints.len(), JointCache::default);
        self.joint_states
            .resize_with(self.joints.len(), WasmJointState::new);
        let beta = if self.baumgarte.is_finite() {
            self.baumgarte.clamp(0.0, 1.0)
        } else {
            DEFAULT_BAUMGARTE
        };
        let mut bodies: Vec<SolverBody> = Vec::new();
        let mut rows: Vec<Row> = Vec::new();
        let mut measures: Vec<JointMeasure> = Vec::with_capacity(self.joints.len());
        for (j, (_, cfg)) in self.joints.iter().enumerate() {
            let kind = joint_kind(cfg)?;
            let ia = body_index(&mut bodies, engine, cfg.body_a, "body_a", dt)?;
            let ib = body_index(&mut bodies, engine, cfg.body_b, "body_b", dt)?;
            if ia == ib {
                return Err(Error::InvalidParameter {
                    name: "body_b".to_string(),
                    message: "a joint must connect two different bodies".to_string(),
                });
            }
            let mut builder = RowBuilder {
                rows: &mut rows,
                bodies: &bodies,
                joint: j,
                ia,
                ib,
                beta_over_dt: beta / dt,
                inv_dt: 1.0 / dt,
            };
            let measure = build_joint_rows(kind, cfg, &mut self.caches[j], &mut builder, dt)?;
            measures.push(measure);
        }
        Ok(Gathered {
            bodies,
            rows,
            measures,
        })
    }

    /// Project the body poses of `engine` onto the joint manifold
    /// (nonlinear Gauss-Seidel position stabilisation).
    ///
    /// Each of the `position_iterations` passes re-linearises the rigid joint
    /// rows (point, axis and orientation locks and violated limits) at the
    /// current poses, solves for positional pseudo-impulses with up to
    /// `iterations` sweeps and moves / rotates the dynamic bodies by
    /// `M^-1 J^T lambda`.  Velocities are left untouched.  Returns the
    /// remaining positional error (sum over rigid rows, m and rad).
    ///
    /// [`step`](Self::step) calls this after integrating; call it yourself
    /// after [`WasmPhysicsEngine::step`] when using [`solve`](Self::solve)
    /// directly.
    ///
    /// # Errors
    ///
    /// As [`solve`](Self::solve).  On error no body is modified by the
    /// failing pass.
    pub fn stabilize_positions(&mut self, engine: &mut WasmPhysicsEngine) -> Result<f64> {
        let dt = engine.config().fixed_dt;
        for _ in 0..self.position_iterations {
            let Gathered { bodies, rows, .. } = self.gather(engine, dt)?;
            let mut rigid: Vec<Row> = rows.into_iter().filter(|r| r.rigid).collect();
            if rigid
                .iter()
                .all(|r| r.violation().abs() < POSITION_TOLERANCE)
            {
                break;
            }
            // Position-level problem: J d = -C (limits only push).
            for row in &mut rigid {
                row.bias = row.error;
                row.gamma = 0.0;
                row.lambda = 0.0;
            }
            let units = build_units(&rigid, &bodies);
            let mut displacements = vec![([0.0; 3], [0.0; 3]); bodies.len()];
            run_sweeps(
                &mut rigid,
                &units,
                &bodies,
                &mut displacements,
                self.iterations.max(1),
            );
            for (body, (dx, dtheta)) in bodies.iter().zip(&displacements) {
                let Some(handle) = body.handle else {
                    continue;
                };
                if body.inv_mass <= 0.0 || !(changed(*dx, [0.0; 3]) || changed(*dtheta, [0.0; 3])) {
                    continue;
                }
                engine.write_joint_body_pose(
                    handle,
                    v_add(body.position, *dx),
                    q_apply_rotation_vector(body.rotation, *dtheta),
                )?;
            }
        }
        let Gathered { rows, .. } = self.gather(engine, dt)?;
        Ok(rows
            .iter()
            .filter(|r| r.rigid)
            .map(|r| r.violation().abs())
            .sum())
    }

    /// Solve all joints against the bodies of `engine` for a step of `dt`.
    ///
    /// Runs up to `iterations` projected Gauss-Seidel sweeps over the joint
    /// rows and writes the corrected linear and angular velocities back to
    /// the dynamic bodies.  Returns the number of sweeps performed (also
    /// stored in `iterations_used`).  Body ids are engine handles; use
    /// [`WORLD_BODY`] to attach a joint to the static world.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidTimeStep`] if `dt` is not positive and finite;
    /// * [`Error::BodyNotFound`] if a joint refers to a dead body;
    /// * [`Error::InvalidParameter`] for an unknown joint type, a joint whose
    ///   two bodies are the same, a body id outside the `u32` range or a zero
    ///   joint axis.
    ///
    /// On error no body is modified.
    pub fn solve(&mut self, engine: &mut WasmPhysicsEngine, dt: f64) -> Result<u32> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::InvalidTimeStep(dt));
        }
        let Gathered {
            mut bodies,
            mut rows,
            measures,
        } = self.gather(engine, dt)?;

        // 2. Warm start.
        for row in &mut rows {
            row.lambda = if self.warm_start {
                self.caches[row.joint].lambdas[row.slot].clamp(row.lo, row.hi)
            } else {
                0.0
            };
        }
        let mut velocities: Vec<(V3, V3)> = bodies
            .iter()
            .map(|b| (b.velocity, b.angular_velocity))
            .collect();
        if self.warm_start {
            for row in &rows {
                if row.lambda != 0.0 {
                    apply_row(&bodies, &mut velocities, row, row.lambda);
                }
            }
        }

        // 3. Projected Gauss-Seidel sweeps.
        let units = build_units(&rows, &bodies);
        let sweeps = run_sweeps(&mut rows, &units, &bodies, &mut velocities, self.iterations);
        for (body, (v, w)) in bodies.iter_mut().zip(&velocities) {
            body.velocity = *v;
            body.angular_velocity = *w;
        }

        // 4. Write velocities back to the engine.
        for body in &bodies {
            let Some(handle) = body.handle else {
                continue;
            };
            if body.inv_mass <= 0.0 {
                continue;
            }
            if changed(body.velocity, body.velocity_start)
                || changed(body.angular_velocity, body.angular_start)
            {
                engine.write_joint_body_velocity(
                    handle,
                    v_sub(body.velocity, body.external),
                    body.angular_velocity,
                )?;
            }
        }

        // 5. Report states, residual error and warm-start impulses.
        let mut lin_sq = vec![0.0; self.joints.len()];
        let mut ang_sq = vec![0.0; self.joints.len()];
        let mut lambda_sq = vec![0.0; self.joints.len()];
        let mut motor = vec![0.0; self.joints.len()];
        let mut violation_sq = vec![0.0; self.joints.len()];
        let mut limit_pushing = vec![false; self.joints.len()];
        for cache in &mut self.caches {
            cache.lambdas = [0.0; MAX_ROWS];
        }
        let mut last_error = 0.0;
        for row in &rows {
            let j = row.joint;
            self.caches[j].lambdas[row.slot] = row.lambda;
            lambda_sq[j] += row.lambda * row.lambda;
            match row.role {
                RowRole::Linear | RowRole::LinearSoft => lin_sq[j] += row.lambda * row.lambda,
                RowRole::Angular | RowRole::AngularLimit => ang_sq[j] += row.lambda * row.lambda,
                RowRole::Motor => motor[j] = row.lambda,
            }
            if row.role == RowRole::Motor {
                continue;
            }
            if row.slot == SLOT_LIMIT && row.lambda > 0.0 {
                limit_pushing[j] = true;
            }
            if row.rigid {
                violation_sq[j] += row.violation() * row.violation();
                let jv = row.jac.velocity(&bodies[row.ia], &bodies[row.ib]);
                let predicted = row.error + dt * jv;
                // A one-sided limit that will be satisfied carries no error.
                if !(row.unilateral && predicted >= 0.0) {
                    last_error += predicted.abs();
                }
            }
        }
        for (j, measure) in measures.iter().enumerate() {
            let state = &mut self.joint_states[j];
            state.position = measure.position;
            state.velocity = measure
                .rate
                .map(|jac| jac.velocity(&bodies[measure.ia], &bodies[measure.ib]))
                .unwrap_or(0.0);
            state.motor_force = motor[j] / dt;
            state.constraint_force = lin_sq[j].sqrt() / dt;
            state.constraint_torque = ang_sq[j].sqrt() / dt;
            state.limit_active = measure.limit_active || limit_pushing[j];
            state.violation = violation_sq[j].sqrt();
            state.lambda = lambda_sq[j].sqrt();
        }
        self.last_error = last_error;
        self.iterations_used = sweeps;
        Ok(sweeps)
    }

    /// Solve the joints, advance `engine` by `dt` and stabilise the joint
    /// positions (`solve`, `engine.step`, `stabilize_positions`).
    ///
    /// Returns the number of velocity sweeps performed.
    ///
    /// # Errors
    ///
    /// Any error of [`solve`](Self::solve), [`WasmPhysicsEngine::step`] or
    /// [`stabilize_positions`](Self::stabilize_positions).
    pub fn step(&mut self, engine: &mut WasmPhysicsEngine, dt: f64) -> Result<u32> {
        let sweeps = self.solve(engine, dt)?;
        engine.step(dt)?;
        self.stabilize_positions(engine)?;
        Ok(sweeps)
    }

    /// Get a reference to the state of a joint, if it exists.
    pub fn get_joint_state(&self, handle: WasmJointHandle) -> Option<&WasmJointState> {
        self.joints
            .iter()
            .position(|(h, _)| *h == handle)
            .and_then(|i| self.joint_states.get(i))
    }

    /// Get the configuration of a joint, if it exists.
    pub fn get_joint_config(&self, handle: WasmJointHandle) -> Option<&WasmJointConfig> {
        self.joints
            .iter()
            .find(|(h, _)| *h == handle)
            .map(|(_, cfg)| cfg)
    }

    /// Set the motor command of a joint.
    ///
    /// * `TargetVelocity(v)` enables the motor in velocity mode
    ///   (`motor_target_velocity = v`, `motor_target_position = None`);
    /// * `TargetPosition(p)` enables the motor in position mode
    ///   (`motor_target_position = Some(p)`), driving the joint coordinate to
    ///   `p` with the velocity `(p - position) / dt`; the stored target
    ///   velocity is left untouched;
    /// * `MaxTorque(t)` sets the motor force/torque cap `motor_max_force`.
    ///
    /// # Errors
    ///
    /// [`Error::ConstraintNotFound`] for an unknown handle,
    /// [`Error::InvalidParameter`] for a non-finite target or a negative /
    /// `NaN` torque cap.
    pub fn set_motor_target(
        &mut self,
        handle: WasmJointHandle,
        target: WasmMotorTarget,
    ) -> Result<()> {
        let cfg = self
            .joints
            .iter_mut()
            .find(|(h, _)| *h == handle)
            .map(|(_, cfg)| cfg)
            .ok_or(Error::ConstraintNotFound(handle.raw()))?;
        match target {
            WasmMotorTarget::TargetVelocity(v) => {
                if !v.is_finite() {
                    return Err(Error::InvalidParameter {
                        name: "motor_target_velocity".to_string(),
                        message: format!("must be finite, got {v}"),
                    });
                }
                cfg.motor_enabled = true;
                cfg.motor_target_velocity = v;
                cfg.motor_target_position = None;
            }
            WasmMotorTarget::TargetPosition(p) => {
                if !p.is_finite() {
                    return Err(Error::InvalidParameter {
                        name: "motor_target_position".to_string(),
                        message: format!("must be finite, got {p}"),
                    });
                }
                cfg.motor_enabled = true;
                cfg.motor_target_position = Some(p);
            }
            WasmMotorTarget::MaxTorque(t) => {
                if t.is_nan() || t < 0.0 {
                    return Err(Error::InvalidParameter {
                        name: "motor_max_force".to_string(),
                        message: format!("must be >= 0, got {t}"),
                    });
                }
                cfg.motor_max_force = t;
            }
        }
        Ok(())
    }

    /// Number of joints currently managed.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
}

#[wasm_bindgen]
impl WasmConstraintSolver {
    /// Construct a solver with the given iteration count and warm-start flag.
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(iterations: u32, warm_start: bool) -> WasmConstraintSolver {
        WasmConstraintSolver::new(iterations, warm_start)
    }

    /// Add a joint and return the assigned raw handle id.
    #[wasm_bindgen(js_name = "add_joint")]
    pub fn add_joint_js(&mut self, config: WasmJointConfig) -> u32 {
        self.add_joint(config).raw()
    }

    /// Remove a joint by raw handle id; returns `true` when removed.
    #[wasm_bindgen(js_name = "remove_joint")]
    pub fn remove_joint_js(&mut self, handle_raw: u32) -> bool {
        self.remove_joint(WasmJointHandle::new(handle_raw))
    }

    /// Solve the joints against `engine` for a step of `dt` seconds; returns
    /// the number of sweeps performed.
    #[wasm_bindgen(js_name = "solve")]
    pub fn solve_js(
        &mut self,
        engine: &mut WasmPhysicsEngine,
        dt: f64,
    ) -> std::result::Result<u32, JsValue> {
        self.solve(engine, dt).map_err(err_to_jsvalue)
    }

    /// Solve the joints and advance `engine` by `dt` seconds.
    #[wasm_bindgen(js_name = "step")]
    pub fn step_js(
        &mut self,
        engine: &mut WasmPhysicsEngine,
        dt: f64,
    ) -> std::result::Result<u32, JsValue> {
        self.step(engine, dt).map_err(err_to_jsvalue)
    }

    /// Joint state for the given raw handle as a structured `JsValue`.
    ///
    /// Returns `JsValue::NULL` when the handle is unknown.
    #[wasm_bindgen(js_name = "get_joint_state")]
    pub fn get_joint_state_js(&self, handle_raw: u32) -> std::result::Result<JsValue, JsValue> {
        match self.get_joint_state(WasmJointHandle::new(handle_raw)) {
            Some(state) => to_js_value(state),
            None => Ok(JsValue::NULL),
        }
    }

    /// Drive a joint motor to a target velocity (enables the motor).
    #[wasm_bindgen(js_name = "set_motor_target_velocity")]
    pub fn set_motor_target_velocity_js(
        &mut self,
        handle_raw: u32,
        value: f64,
    ) -> std::result::Result<(), JsValue> {
        self.set_motor_target(
            WasmJointHandle::new(handle_raw),
            WasmMotorTarget::TargetVelocity(value),
        )
        .map_err(err_to_jsvalue)
    }

    /// Drive a joint motor to a target position (enables the motor).
    #[wasm_bindgen(js_name = "set_motor_target_position")]
    pub fn set_motor_target_position_js(
        &mut self,
        handle_raw: u32,
        value: f64,
    ) -> std::result::Result<(), JsValue> {
        self.set_motor_target(
            WasmJointHandle::new(handle_raw),
            WasmMotorTarget::TargetPosition(value),
        )
        .map_err(err_to_jsvalue)
    }

    /// Cap the motor torque/force for a joint by raw handle.
    #[wasm_bindgen(js_name = "set_motor_max_torque")]
    pub fn set_motor_max_torque_js(
        &mut self,
        handle_raw: u32,
        value: f64,
    ) -> std::result::Result<(), JsValue> {
        self.set_motor_target(
            WasmJointHandle::new(handle_raw),
            WasmMotorTarget::MaxTorque(value),
        )
        .map_err(err_to_jsvalue)
    }

    /// Number of joints currently managed.
    #[wasm_bindgen(js_name = "joint_count")]
    pub fn joint_count_js(&self) -> usize {
        self.joint_count()
    }

    /// Number of (handle, config) pairs currently stored — JS-friendly count.
    #[wasm_bindgen(js_name = "joints_count")]
    pub fn joints_count(&self) -> usize {
        self.joints.len()
    }

    /// Raw handle id of the joint at index `i`, if any.
    #[wasm_bindgen(js_name = "joint_handle_at")]
    pub fn joint_handle_at(&self, i: usize) -> Option<u32> {
        self.joints.get(i).map(|(h, _)| h.raw())
    }

    /// Cloned configuration of the joint at index `i`, if any.
    #[wasm_bindgen(js_name = "joint_config_at")]
    pub fn joint_config_at(&self, i: usize) -> Option<WasmJointConfig> {
        self.joints.get(i).map(|(_, cfg)| cfg.clone())
    }

    /// Cloned state of the joint at index `i`, if any.
    #[wasm_bindgen(js_name = "joint_state_at")]
    pub fn joint_state_at(&self, i: usize) -> Option<WasmJointState> {
        self.joint_states.get(i).cloned()
    }

    /// Serialise the entire solver state to a JSON string.
    #[wasm_bindgen(js_name = "to_json")]
    pub fn to_json_js(&self) -> std::result::Result<String, JsValue> {
        serde_json::to_string(self).map_err(err_to_jsvalue)
    }
}
