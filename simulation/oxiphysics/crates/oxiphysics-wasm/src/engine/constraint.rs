// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `EngineConstraintSolver` — iterative constraint solver for the WASM engine.
//!
//! The solver works on velocities with sequential impulses (projected
//! Gauss-Seidel).  Every constraint is a *row*: a body pair `(A, B)`, a unit
//! direction `n` and a target for the relative normal velocity
//!
//! ```text
//! v_rel = (v_A - v_B) . n
//! ```
//!
//! An impulse `lambda * n` on `A` together with `-lambda * n` on `B` changes
//! `v_rel` by `lambda * (w_A + w_B)`, where `w` is the inverse mass of a body
//! that responds to impulses (zero for the world, static bodies and kinematic
//! bodies).  The impulse increment that moves a row onto its target velocity is
//! therefore
//!
//! ```text
//! d_lambda = (target - v_rel) / (w_A + w_B)
//! ```
//!
//! and the engine receives the raw impulse `d_lambda * n` (it divides by the
//! mass itself).  Bilateral rows leave the accumulated impulse unrestricted;
//! unilateral (contact) rows clamp it to be non-negative, so they can push
//! bodies apart but never pull them together.

use super::WasmPhysicsEngine;
use super::geom::{Vec3, dot, normalized, sub};
use crate::error::{Error, Result};

/// Handle that stands for the static world instead of a body.
const WORLD: u32 = u32::MAX;

/// A single velocity constraint between two bodies (or one body and the world).
///
/// The row drives the relative normal velocity `(v_a - v_b) . normal` towards
/// [`bias`](Self::bias): exactly onto it for a bilateral row, or no lower than
/// it for a [`unilateral`](Self::unilateral) (contact) row.
#[derive(Debug, Clone)]
pub struct WasmConstraint {
    /// Handle of the first body (or `u32::MAX` for the static world).
    pub body_a: u32,
    /// Handle of the second body (or `u32::MAX` for the static world).
    pub body_b: u32,
    /// Direction along which the relative velocity is constrained (the solver
    /// normalises it).
    pub normal: [f64; 3],
    /// Target relative normal velocity in m/s.  Contact rows take the larger
    /// of the separation speed requested by the Baumgarte stabilisation
    /// (penetration depth beyond the slop, scaled by `baumgarte / dt`) and the
    /// rebound speed requested by restitution; explicit constraints use the
    /// value passed to [`EngineConstraintSolver::add_constraint`].
    pub bias: f64,
    /// Impulse accumulated along `normal` by the current solve (it starts at
    /// zero for every call of [`EngineConstraintSolver::solve`]).
    pub lambda: f64,
    /// `true` for one-sided (contact) rows, whose accumulated impulse is
    /// clamped to be non-negative; `false` for bilateral rows, which may pull
    /// as well as push.
    pub unilateral: bool,
}

/// Contact data for the constraint solver.
#[derive(Debug, Clone)]
pub struct ContactData {
    /// First body handle (or `u32::MAX` for the static world).
    pub body_a: u32,
    /// Second body handle (or `u32::MAX` for the static world).
    pub body_b: u32,
    /// Contact normal (pointing from B to A); the solver normalises it.
    pub normal: [f64; 3],
    /// Penetration depth.
    pub depth: f64,
    /// Combined restitution, in `[0, 1]`.
    pub restitution: f64,
}

/// A simple iterative (Gauss-Seidel) constraint solver.
///
/// Works in conjunction with a `WasmPhysicsEngine` by accepting contact data,
/// building constraint rows, and iterating to convergence.
///
/// The solver is velocity based (sequential impulses).  A row drives the
/// relative normal velocity `v_rel = (v_A - v_B) . n` of two bodies onto a
/// target by applying `d_lambda = (target - v_rel) / (w_A + w_B)` as an
/// impulse along `n` on A and `-n` on B, where `w` is the inverse mass of a
/// body that can move (zero for the world, static and kinematic bodies).
/// Contact rows additionally keep their accumulated impulse non-negative.
#[derive(Debug, Clone)]
pub struct EngineConstraintSolver {
    constraints: Vec<WasmConstraint>,
    contacts: Vec<ContactData>,
    /// Number of Gauss-Seidel iterations per solve call.
    pub(crate) iterations: u32,
    /// Baumgarte stabilisation factor (typical: 0.1–0.3).
    baumgarte: f64,
    /// Allowed penetration slop before correction (metres).
    slop: f64,
}

impl EngineConstraintSolver {
    /// Create a new constraint solver.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            contacts: Vec::new(),
            iterations: 10,
            baumgarte: 0.2,
            slop: 0.001,
        }
    }

    /// Set the number of solver iterations.
    pub fn set_iterations(&mut self, n: u32) {
        self.iterations = n.max(1);
    }

    /// Set the Baumgarte stabilisation factor, clamped to `[0, 1]`.
    ///
    /// A NaN factor is stored as is and reported by [`solve`](Self::solve).
    pub fn set_baumgarte(&mut self, b: f64) {
        self.baumgarte = b.clamp(0.0, 1.0);
    }

    /// Add a contact for the next solve call.
    ///
    /// `(nx, ny, nz)` points from body B to body A (it is normalised by
    /// [`solve`](Self::solve)), `depth` is the penetration depth in metres and
    /// `restitution` the combined coefficient of restitution in `[0, 1]`: an
    /// approaching pair leaves the contact with `restitution` times its closing
    /// speed (at any closing speed, so resting contacts should use `0`).  Use
    /// `u32::MAX` as a body handle for the static world.  The contact becomes
    /// a unilateral row: it can push the bodies apart, never pull them
    /// together.  The arguments are checked when the contact is solved.
    pub fn add_contact(
        &mut self,
        body_a: u32,
        body_b: u32,
        nx: f64,
        ny: f64,
        nz: f64,
        depth: f64,
        restitution: f64,
    ) {
        self.contacts.push(ContactData {
            body_a,
            body_b,
            normal: [nx, ny, nz],
            depth,
            restitution,
        });
    }

    /// Clear all accumulated contacts.
    pub fn clear_contacts(&mut self) {
        self.contacts.clear();
    }

    /// Add an explicit bilateral constraint.
    ///
    /// The relative velocity `(v_a - v_b) . n` along the direction
    /// `(nx, ny, nz)` (normalised by [`solve`](Self::solve)) is driven onto
    /// `bias` in m/s, so a `bias` of zero makes the two bodies move together
    /// along `n`.  Use `u32::MAX` as a body handle for the static world.  The
    /// arguments are checked when the constraint is solved.
    pub fn add_constraint(
        &mut self,
        body_a: u32,
        body_b: u32,
        nx: f64,
        ny: f64,
        nz: f64,
        bias: f64,
    ) {
        self.constraints.push(WasmConstraint {
            body_a,
            body_b,
            normal: [nx, ny, nz],
            bias,
            lambda: 0.0,
            unilateral: false,
        });
    }

    /// Clear bilateral constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Solve the queued contacts and constraints against `engine`.
    ///
    /// Runs the configured number of Gauss-Seidel sweeps (over the contact
    /// rows first, then over the explicit constraints) and changes the linear
    /// velocities of the dynamic bodies involved.  Positions are not touched:
    /// a penetrating contact is corrected through a separation velocity of
    /// `baumgarte / dt * (depth - slop)`, and an approaching contact bounces
    /// off with `restitution * closing_speed`; the larger of the two is the
    /// contact's target velocity.
    ///
    /// The world handle `u32::MAX`, static bodies and kinematic bodies act as
    /// infinitely heavy: they take part in the relative velocity but never
    /// receive an impulse, and a row whose bodies are all of that kind has
    /// nothing to move.
    ///
    /// Returns the number of rows (queued contacts plus explicit constraints)
    /// that were solved.
    ///
    /// # Errors
    ///
    /// Every row is validated before the first impulse is applied, so the
    /// following errors leave the engine untouched:
    ///
    /// * [`Error::InvalidTimeStep`] - `dt` is not finite and positive (or so
    ///   small that the Baumgarte separation speed overflows);
    /// * [`Error::InvalidHandle`] - a body handle is neither `u32::MAX` nor a
    ///   live body of `engine` (never allocated, removed or stale);
    /// * [`Error::SelfCollision`] - both handles of a row are the same;
    /// * [`Error::InvalidParameter`] - a normal is zero or not finite, a depth
    ///   or bias is not finite, or the Baumgarte factor is NaN;
    /// * [`Error::InvalidRestitution`] - a restitution is outside `[0, 1]`.
    ///
    /// An error reported by [`WasmPhysicsEngine::apply_impulse`] while the rows
    /// are being solved (a non-finite impulse) is propagated as well.
    pub fn solve(&mut self, engine: &mut WasmPhysicsEngine, dt: f64) -> Result<u32> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::InvalidTimeStep(dt));
        }
        let mut rows = self.prepare_rows(engine, dt)?;
        for _ in 0..self.iterations {
            for row in &mut rows {
                solve_row(engine, row)?;
            }
        }
        Ok(rows.len() as u32)
    }

    /// Validate the queued contacts and constraints and turn them into rows
    /// (contacts first, then the explicit constraints).
    fn prepare_rows(&self, engine: &WasmPhysicsEngine, dt: f64) -> Result<Vec<SolverRow>> {
        if self.baumgarte.is_nan() {
            return Err(Error::invalid_param("baumgarte", "must not be NaN"));
        }
        let mut rows = Vec::with_capacity(self.contacts.len() + self.constraints.len());
        for contact in &self.contacts {
            let normal = checked_direction(engine, contact.body_a, contact.body_b, contact.normal)?;
            if !contact.depth.is_finite() {
                return Err(Error::invalid_param("depth", "must be finite"));
            }
            if !(0.0..=1.0).contains(&contact.restitution) {
                return Err(Error::InvalidRestitution(contact.restitution));
            }
            // Speed at which Baumgarte stabilisation wants to separate the
            // bodies, and the speed at which Newton's restitution law wants
            // them to leave an impact; the bodies must satisfy both.
            let penetration = (contact.depth - self.slop).max(0.0);
            let separation_speed = self.baumgarte * penetration / dt;
            if !separation_speed.is_finite() {
                return Err(Error::InvalidTimeStep(dt));
            }
            let closing_speed =
                (-relative_normal_velocity(engine, contact.body_a, contact.body_b, normal))
                    .max(0.0);
            rows.push(SolverRow::new(
                engine,
                WasmConstraint {
                    body_a: contact.body_a,
                    body_b: contact.body_b,
                    normal,
                    bias: separation_speed.max(contact.restitution * closing_speed),
                    lambda: 0.0,
                    unilateral: true,
                },
            ));
        }
        for constraint in &self.constraints {
            let normal = checked_direction(
                engine,
                constraint.body_a,
                constraint.body_b,
                constraint.normal,
            )?;
            if !constraint.bias.is_finite() {
                return Err(Error::invalid_param("bias", "must be finite"));
            }
            rows.push(SolverRow::new(
                engine,
                WasmConstraint {
                    normal,
                    lambda: 0.0,
                    ..constraint.clone()
                },
            ));
        }
        Ok(rows)
    }

    /// Number of contacts currently queued.
    pub fn contact_count(&self) -> u32 {
        self.contacts.len() as u32
    }

    /// Number of explicit constraints.
    pub fn constraint_count(&self) -> u32 {
        self.constraints.len() as u32
    }
}

impl Default for EngineConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// A constraint row together with the quantities that stay fixed during one
/// solve.
struct SolverRow {
    constraint: WasmConstraint,
    /// Inverse mass with which body A responds to impulses.
    inv_mass_a: f64,
    /// Inverse mass with which body B responds to impulses.
    inv_mass_b: f64,
}

impl SolverRow {
    fn new(engine: &WasmPhysicsEngine, constraint: WasmConstraint) -> Self {
        let inv_mass_a = response_inv_mass(engine, constraint.body_a);
        let inv_mass_b = response_inv_mass(engine, constraint.body_b);
        Self {
            constraint,
            inv_mass_a,
            inv_mass_b,
        }
    }
}

/// Inverse mass with which a body responds to a constraint impulse: the
/// engine's inverse mass for a dynamic body, zero for the world and for static
/// and kinematic bodies (the engine's own contact resolution treats them the
/// same way, and the engine would otherwise move a kinematic body that was
/// given a mass).
fn response_inv_mass(engine: &WasmPhysicsEngine, handle: u32) -> f64 {
    if handle != WORLD && engine.body_is_dynamic(handle) {
        engine.body_inv_mass(handle)
    } else {
        0.0
    }
}

/// Relative velocity `(v_a - v_b) . normal` of two bodies (the world has
/// velocity zero).
fn relative_normal_velocity(
    engine: &WasmPhysicsEngine,
    body_a: u32,
    body_b: u32,
    normal: Vec3,
) -> f64 {
    dot(
        sub(engine.get_velocity(body_a), engine.get_velocity(body_b)),
        normal,
    )
}

/// Check the body pair of a row and return its unit direction.
fn checked_direction(
    engine: &WasmPhysicsEngine,
    body_a: u32,
    body_b: u32,
    normal: Vec3,
) -> Result<Vec3> {
    for handle in [body_a, body_b] {
        if handle != WORLD && !engine.contains_body(handle) {
            return Err(Error::InvalidHandle(handle));
        }
    }
    if body_a == body_b {
        return Err(Error::SelfCollision(body_a, body_b));
    }
    normalized(normal)
        .ok_or_else(|| Error::invalid_param("normal", "must be a finite, non-zero vector"))
}

/// One Gauss-Seidel update: apply the impulse increment that moves the
/// relative normal velocity of `row` onto its target.
fn solve_row(engine: &mut WasmPhysicsEngine, row: &mut SolverRow) -> Result<()> {
    let inv_mass_sum = row.inv_mass_a + row.inv_mass_b;
    if inv_mass_sum <= 0.0 {
        // Neither body can respond to an impulse.
        return Ok(());
    }
    let constraint = &mut row.constraint;
    let relative_velocity = relative_normal_velocity(
        engine,
        constraint.body_a,
        constraint.body_b,
        constraint.normal,
    );
    // A unit impulse along the row changes the relative velocity by
    // `inv_mass_sum`, so this increment lands exactly on the target.
    let lambda_old = constraint.lambda;
    let mut lambda = lambda_old + (constraint.bias - relative_velocity) / inv_mass_sum;
    if constraint.unilateral {
        // Accumulated clamp: a contact may take back an earlier push, but its
        // total impulse can never pull the bodies together.
        lambda = lambda.max(0.0);
    }
    let d_lambda = lambda - lambda_old;
    constraint.lambda = lambda;
    if d_lambda == 0.0 {
        return Ok(());
    }
    // The engine divides by the mass itself, so it gets the raw impulse.
    let [nx, ny, nz] = constraint.normal;
    if row.inv_mass_a > 0.0 {
        engine.apply_impulse(
            constraint.body_a,
            d_lambda * nx,
            d_lambda * ny,
            d_lambda * nz,
        )?;
    }
    if row.inv_mass_b > 0.0 {
        engine.apply_impulse(
            constraint.body_b,
            -d_lambda * nx,
            -d_lambda * ny,
            -d_lambda * nz,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RigidBodyConfig;

    const DT: f64 = 1.0 / 60.0;
    const TOLERANCE: f64 = 1e-12;

    fn space() -> WasmPhysicsEngine {
        WasmPhysicsEngine::new(0.0, 0.0, 0.0)
    }

    /// Dynamic body at the origin with the given mass and velocity.
    fn moving_body(engine: &mut WasmPhysicsEngine, mass: f64, velocity: [f64; 3]) -> u32 {
        let handle = engine
            .add_dynamic_body(mass, 0.0, 0.0, 0.0)
            .expect("dynamic body");
        engine
            .set_velocity(handle, velocity[0], velocity[1], velocity[2])
            .expect("velocity");
        handle
    }

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "{what}: expected {expected}, got {actual}"
        );
    }

    fn assert_velocity(engine: &WasmPhysicsEngine, handle: u32, expected: [f64; 3], what: &str) {
        let actual = engine.get_velocity(handle);
        for axis in 0..3 {
            assert_close(actual[axis], expected[axis], &format!("{what}[{axis}]"));
        }
    }

    /// Total linear momentum along `axis` of `(handle, mass)` pairs.
    fn momentum(engine: &WasmPhysicsEngine, bodies: &[(u32, f64)], axis: usize) -> f64 {
        bodies
            .iter()
            .map(|&(handle, mass)| mass * engine.get_velocity(handle)[axis])
            .sum()
    }

    /// Name of the parameter an `InvalidParameter` error complains about.
    fn invalid_parameter(result: Result<u32>) -> String {
        match result {
            Err(Error::InvalidParameter { name, .. }) => name,
            other => panic!("expected an invalid parameter, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Bilateral rows
    // -----------------------------------------------------------------------

    #[test]
    fn two_equal_masses_equalise_in_one_iteration() {
        let mut engine = space();
        let a = moving_body(&mut engine, 2.0, [1.0, 0.0, 0.0]);
        let b = moving_body(&mut engine, 2.0, [-1.0, 0.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        solver.set_iterations(1);
        solver.add_constraint(a, b, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(solver.solve(&mut engine, DT), Ok(1));
        assert_velocity(&engine, a, [0.0; 3], "a");
        assert_velocity(&engine, b, [0.0; 3], "b");
    }

    #[test]
    fn unequal_masses_meet_at_the_centre_of_mass_velocity() {
        let mut engine = space();
        let a = moving_body(&mut engine, 2.0, [1.0, 0.0, 0.0]);
        let b = moving_body(&mut engine, 6.0, [-1.0, 0.0, 0.0]);
        let bodies = [(a, 2.0), (b, 6.0)];
        let momentum_before = momentum(&engine, &bodies, 0);
        let mut solver = EngineConstraintSolver::new();
        solver.set_iterations(1);
        solver.add_constraint(a, b, 1.0, 0.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        // (2 * 1 + 6 * -1) / (2 + 6) = -0.5 for both bodies.
        assert_velocity(&engine, a, [-0.5, 0.0, 0.0], "a");
        assert_velocity(&engine, b, [-0.5, 0.0, 0.0], "b");
        assert_close(
            momentum(&engine, &bodies, 0),
            momentum_before,
            "linear momentum",
        );
    }

    #[test]
    fn only_the_constrained_direction_is_touched() {
        let mut engine = space();
        let a = moving_body(&mut engine, 1.0, [3.0, 1.0, 0.0]);
        let b = moving_body(&mut engine, 1.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(a, b, 0.0, 1.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, a, [3.0, 0.5, 0.0], "a");
        assert_velocity(&engine, b, [0.0, 0.5, 0.0], "b");
    }

    #[test]
    fn non_unit_normals_are_normalised() {
        let solve_with = |normal_length: f64| {
            let mut engine = space();
            let a = moving_body(&mut engine, 2.0, [0.0, 1.0, 0.0]);
            let b = moving_body(&mut engine, 6.0, [0.0, -1.0, 0.0]);
            let mut solver = EngineConstraintSolver::new();
            solver.add_constraint(a, b, 0.0, normal_length, 0.0, 0.0);
            solver.solve(&mut engine, DT).expect("solve");
            (engine.get_velocity(a), engine.get_velocity(b))
        };
        let (unit_a, unit_b) = solve_with(1.0);
        let (long_a, long_b) = solve_with(2.0);
        let (short_a, short_b) = solve_with(0.25);
        for axis in 0..3 {
            assert_close(long_a[axis], unit_a[axis], "length 2, body a");
            assert_close(long_b[axis], unit_b[axis], "length 2, body b");
            assert_close(short_a[axis], unit_a[axis], "length 0.25, body a");
            assert_close(short_b[axis], unit_b[axis], "length 0.25, body b");
        }
        assert_close(unit_a[1], -0.5, "common velocity");
    }

    #[test]
    fn bilateral_bias_is_the_target_relative_velocity() {
        let mut engine = space();
        let body = moving_body(&mut engine, 4.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(body, u32::MAX, 0.0, 1.0, 0.0, 2.5);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, body, [0.0, 2.5, 0.0], "driven body");
    }

    #[test]
    fn solving_twice_changes_nothing_once_the_rows_are_satisfied() {
        let mut engine = space();
        let a = moving_body(&mut engine, 2.0, [1.0, 0.0, 0.0]);
        let b = moving_body(&mut engine, 2.0, [-1.0, 0.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(a, b, 1.0, 0.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("first solve");
        let after_first = (engine.get_velocity(a), engine.get_velocity(b));
        solver.solve(&mut engine, DT).expect("second solve");
        assert_eq!(
            after_first,
            (engine.get_velocity(a), engine.get_velocity(b))
        );
    }

    // -----------------------------------------------------------------------
    // Contact rows
    // -----------------------------------------------------------------------

    #[test]
    fn resting_penetrating_body_gains_the_baumgarte_separation_velocity() {
        let mut engine = space();
        let body = moving_body(&mut engine, 3.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        // The body lies above the world, so the normal (world -> body) is +y.
        solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.05, 0.0);
        assert_eq!(solver.solve(&mut engine, DT), Ok(1));
        // baumgarte (0.2) / dt * (depth - slop (0.001)); the mass cancels.
        let expected = 0.2 / DT * (0.05 - 0.001);
        assert!(expected > 0.5);
        assert_velocity(&engine, body, [0.0, expected, 0.0], "body");
        assert_eq!(engine.get_position(body), [0.0; 3], "positions stay put");
    }

    #[test]
    fn world_as_body_a_pushes_the_body_away_just_the_same() {
        let mut engine = space();
        let body = moving_body(&mut engine, 0.5, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        // Mirror image: normal from body B (the body) to body A (the world).
        solver.add_contact(u32::MAX, body, 0.0, -1.0, 0.0, 0.05, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, body, [0.0, 0.2 / DT * (0.05 - 0.001), 0.0], "body");
    }

    #[test]
    fn penetration_within_the_slop_adds_no_velocity() {
        let mut engine = space();
        let body = moving_body(&mut engine, 1.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.0005, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_eq!(engine.get_velocity(body), [0.0; 3]);
    }

    #[test]
    fn baumgarte_factor_scales_the_separation_velocity() {
        for (factor, effective) in [(0.5, 0.5), (7.0, 1.0), (-1.0, 0.0)] {
            let mut engine = space();
            let body = moving_body(&mut engine, 1.0, [0.0; 3]);
            let mut solver = EngineConstraintSolver::new();
            solver.set_baumgarte(factor);
            solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.05, 0.0);
            solver.solve(&mut engine, DT).expect("solve");
            assert_velocity(
                &engine,
                body,
                [0.0, effective / DT * (0.05 - 0.001), 0.0],
                &format!("baumgarte {factor}"),
            );
        }
    }

    #[test]
    fn inelastic_contact_stops_an_approaching_body() {
        let mut engine = space();
        let body = moving_body(&mut engine, 2.0, [0.0, -4.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, body, [0.0; 3], "body");
    }

    #[test]
    fn restitution_bounces_an_approaching_body() {
        let mut engine = space();
        let body = moving_body(&mut engine, 2.0, [0.0, -4.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.0, 0.5);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, body, [0.0, 2.0, 0.0], "body");
    }

    #[test]
    fn restitution_conserves_momentum_between_two_bodies() {
        let mut engine = space();
        // The normal points from B to A, so A lies on the +x side of B.
        let a = moving_body(&mut engine, 1.0, [-1.0, 0.0, 0.0]);
        let b = moving_body(&mut engine, 3.0, [1.0, 0.0, 0.0]);
        let bodies = [(a, 1.0), (b, 3.0)];
        let momentum_before = momentum(&engine, &bodies, 0);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(a, b, 1.0, 0.0, 0.0, 0.0, 0.5);
        solver.solve(&mut engine, DT).expect("solve");
        // The closing speed of 2 m/s becomes a 1 m/s recession.
        assert_velocity(&engine, a, [1.25, 0.0, 0.0], "a");
        assert_velocity(&engine, b, [0.25, 0.0, 0.0], "b");
        assert_close(momentum(&engine, &bodies, 0), momentum_before, "momentum");
    }

    #[test]
    fn the_larger_of_bounce_and_separation_speed_wins() {
        let separation = 0.2 / DT * (0.05 - 0.001);
        for (restitution, expected) in [(0.5, 1.0), (0.1, separation), (0.0, separation)] {
            let mut engine = space();
            let body = moving_body(&mut engine, 1.0, [0.0, -2.0, 0.0]);
            let mut solver = EngineConstraintSolver::new();
            solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.05, restitution);
            solver.solve(&mut engine, DT).expect("solve");
            assert_velocity(
                &engine,
                body,
                [0.0, expected, 0.0],
                &format!("restitution {restitution}"),
            );
        }
    }

    #[test]
    fn a_contact_never_pulls_bodies_together_but_a_constraint_does() {
        let separating_pair = |engine: &mut WasmPhysicsEngine| {
            let a = moving_body(engine, 2.0, [1.0, 0.0, 0.0]);
            let b = moving_body(engine, 2.0, [-1.0, 0.0, 0.0]);
            (a, b)
        };

        let mut engine = space();
        let (a, b) = separating_pair(&mut engine);
        let mut contact = EngineConstraintSolver::new();
        contact.add_contact(a, b, 1.0, 0.0, 0.0, 0.0, 0.0);
        contact.solve(&mut engine, DT).expect("contact solve");
        assert_eq!(engine.get_velocity(a), [1.0, 0.0, 0.0]);
        assert_eq!(engine.get_velocity(b), [-1.0, 0.0, 0.0]);

        let mut engine = space();
        let (a, b) = separating_pair(&mut engine);
        let mut bilateral = EngineConstraintSolver::new();
        bilateral.add_constraint(a, b, 1.0, 0.0, 0.0, 0.0);
        bilateral.solve(&mut engine, DT).expect("constraint solve");
        assert_velocity(&engine, a, [0.0; 3], "a");
        assert_velocity(&engine, b, [0.0; 3], "b");
    }

    #[test]
    fn a_contact_takes_back_a_push_that_a_later_row_made_unnecessary() {
        // P falls onto the floor while Q rushes into P from below.  The floor
        // first stops P (impulse 2), then Q's contact lifts P away from the
        // floor, so on the next sweep the floor row must give its impulse back
        // (accumulated clamp) instead of keeping the momentum it injected.
        let run = |iterations: u32| {
            let mut engine = space();
            let p = moving_body(&mut engine, 1.0, [0.0, -2.0, 0.0]);
            let q = moving_body(&mut engine, 1.0, [0.0, 4.0, 0.0]);
            let mut solver = EngineConstraintSolver::new();
            solver.set_iterations(iterations);
            solver.add_contact(p, u32::MAX, 0.0, 1.0, 0.0, 0.0, 0.0);
            solver.add_contact(p, q, 0.0, 1.0, 0.0, 0.0, 0.0);
            solver.solve(&mut engine, DT).expect("solve");
            (engine.get_velocity(p)[1], engine.get_velocity(q)[1])
        };
        let (p1, q1) = run(1);
        assert_close(p1, 2.0, "P after one sweep");
        assert_close(q1, 2.0, "Q after one sweep");
        // Without the floor's impulse the total momentum stays at 1 * -2 + 1 * 4.
        for iterations in [2, 3, 10] {
            let (p, q) = run(iterations);
            assert_close(p, 1.0, "P");
            assert_close(q, 1.0, "Q");
        }
    }

    #[test]
    fn a_stack_of_bodies_comes_to_rest_on_the_world() {
        let mut engine = space();
        let masses = [1.0, 2.0, 4.0];
        let bodies: Vec<u32> = masses
            .iter()
            .map(|&mass| moving_body(&mut engine, mass, [0.0, -1.0, 0.0]))
            .collect();
        let mut solver = EngineConstraintSolver::new();
        solver.set_iterations(300);
        // Each body rests on the one below it; the lowest on the world.
        solver.add_contact(bodies[0], u32::MAX, 0.0, 1.0, 0.0, 0.0, 0.0);
        solver.add_contact(bodies[1], bodies[0], 0.0, 1.0, 0.0, 0.0, 0.0);
        solver.add_contact(bodies[2], bodies[1], 0.0, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(solver.solve(&mut engine, DT), Ok(3));
        for (&body, mass) in bodies.iter().zip(masses) {
            let vy = engine.get_velocity(body)[1];
            assert!(vy.abs() < 1e-9, "body of {mass} kg still moves at {vy} m/s");
        }
    }

    // -----------------------------------------------------------------------
    // Bodies that do not respond to impulses
    // -----------------------------------------------------------------------

    #[test]
    fn static_bodies_never_receive_impulses() {
        let mut engine = space();
        let wall = engine.add_static_body(0.0, 0.0, 0.0).expect("static body");
        let ball = moving_body(&mut engine, 1.0, [3.0, 0.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        // The ball (body A) approaches the wall (body B) from the -x side.
        solver.add_contact(ball, wall, -1.0, 0.0, 0.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, ball, [0.0; 3], "ball");
        assert_eq!(engine.get_velocity(wall), [0.0; 3]);
    }

    #[test]
    fn kinematic_bodies_are_not_pushed_even_when_they_carry_a_mass() {
        let mut engine = space();
        let platform = engine
            .add_rigid_body(&RigidBodyConfig {
                mass: 5.0,
                is_kinematic: true,
                linear_velocity: [0.0, 2.0, 0.0],
                ..RigidBodyConfig::default()
            })
            .expect("kinematic body");
        assert!(engine.body_inv_mass(platform) > 0.0);
        assert!(!engine.body_is_dynamic(platform));
        let rider = moving_body(&mut engine, 1.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(rider, platform, 0.0, 1.0, 0.0, 0.0);
        solver.solve(&mut engine, DT).expect("solve");
        assert_velocity(&engine, rider, [0.0, 2.0, 0.0], "rider");
        assert_eq!(engine.get_velocity(platform), [0.0, 2.0, 0.0]);
    }

    #[test]
    fn rows_between_immovable_bodies_are_counted_but_change_nothing() {
        let mut engine = space();
        let wall = engine.add_static_body(0.0, 0.0, 0.0).expect("static body");
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(wall, u32::MAX, 0.0, 1.0, 0.0, 0.5, 0.5);
        solver.add_constraint(wall, u32::MAX, 0.0, 1.0, 0.0, 3.0);
        assert_eq!(solver.solve(&mut engine, DT), Ok(2));
        assert_eq!(engine.get_velocity(wall), [0.0; 3]);
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn solving_without_rows_is_a_no_op() {
        let mut engine = space();
        let body = moving_body(&mut engine, 1.0, [1.0, 2.0, 3.0]);
        let mut solver = EngineConstraintSolver::new();
        assert_eq!(solver.solve(&mut engine, DT), Ok(0));
        assert_eq!(engine.get_velocity(body), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_row_count_covers_contacts_and_constraints() {
        let mut engine = space();
        let a = moving_body(&mut engine, 1.0, [0.0; 3]);
        let b = moving_body(&mut engine, 1.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(a, b, 1.0, 0.0, 0.0, 0.01, 0.0);
        solver.add_contact(a, u32::MAX, 0.0, 1.0, 0.0, 0.01, 0.0);
        solver.add_constraint(a, b, 0.0, 0.0, 1.0, 0.0);
        assert_eq!((solver.contact_count(), solver.constraint_count()), (2, 1));
        assert_eq!(solver.solve(&mut engine, DT), Ok(3));
        solver.clear_contacts();
        assert_eq!(solver.solve(&mut engine, DT), Ok(1));
        solver.clear_constraints();
        assert_eq!(solver.solve(&mut engine, DT), Ok(0));
    }

    #[test]
    fn invalid_time_steps_are_rejected_before_anything_moves() {
        let mut engine = space();
        let body = moving_body(&mut engine, 1.0, [0.0, -1.0, 0.0]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(body, u32::MAX, 0.0, 1.0, 0.0, 0.05, 0.0);
        for dt in [0.0, -DT, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    solver.solve(&mut engine, dt),
                    Err(Error::InvalidTimeStep(_))
                ),
                "dt = {dt}"
            );
        }
        // A step so small that the separation speed overflows.
        assert!(matches!(
            solver.solve(&mut engine, 5e-324),
            Err(Error::InvalidTimeStep(_))
        ));
        assert_eq!(engine.get_velocity(body), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn unknown_and_stale_handles_are_rejected_without_side_effects() {
        let mut engine = space();
        let a = moving_body(&mut engine, 1.0, [1.0, 0.0, 0.0]);
        let b = moving_body(&mut engine, 1.0, [-1.0, 0.0, 0.0]);
        let removed = moving_body(&mut engine, 1.0, [0.0; 3]);
        engine.remove_body(removed).expect("remove body");

        for bad in [removed, 777] {
            // The valid row comes first: nothing may have been applied when
            // the invalid one is reported.
            let mut solver = EngineConstraintSolver::new();
            solver.add_constraint(a, b, 1.0, 0.0, 0.0, 0.0);
            solver.add_contact(a, bad, 1.0, 0.0, 0.0, 0.01, 0.0);
            assert_eq!(
                solver.solve(&mut engine, DT),
                Err(Error::InvalidHandle(bad))
            );
            assert_eq!(engine.get_velocity(a), [1.0, 0.0, 0.0]);
            assert_eq!(engine.get_velocity(b), [-1.0, 0.0, 0.0]);

            let mut solver = EngineConstraintSolver::new();
            solver.add_constraint(bad, a, 1.0, 0.0, 0.0, 0.0);
            assert_eq!(
                solver.solve(&mut engine, DT),
                Err(Error::InvalidHandle(bad))
            );
        }
    }

    #[test]
    fn a_row_needs_two_different_bodies() {
        let mut engine = space();
        let a = moving_body(&mut engine, 1.0, [0.0; 3]);
        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(a, a, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(
            solver.solve(&mut engine, DT),
            Err(Error::SelfCollision(a, a))
        );

        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(u32::MAX, u32::MAX, 0.0, 1.0, 0.0, 0.1, 0.0);
        assert_eq!(
            solver.solve(&mut engine, DT),
            Err(Error::SelfCollision(u32::MAX, u32::MAX))
        );
    }

    #[test]
    fn degenerate_normals_are_rejected() {
        for normal in [
            [0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 1.0],
            [f64::INFINITY, 0.0, 0.0],
            [1e-200, 0.0, 0.0],
        ] {
            let mut engine = space();
            let a = moving_body(&mut engine, 1.0, [0.0; 3]);
            let mut contact = EngineConstraintSolver::new();
            contact.add_contact(a, u32::MAX, normal[0], normal[1], normal[2], 0.1, 0.0);
            assert_eq!(
                invalid_parameter(contact.solve(&mut engine, DT)),
                "normal",
                "contact normal {normal:?}"
            );
            let mut constraint = EngineConstraintSolver::new();
            constraint.add_constraint(a, u32::MAX, normal[0], normal[1], normal[2], 0.0);
            assert_eq!(
                invalid_parameter(constraint.solve(&mut engine, DT)),
                "normal",
                "constraint normal {normal:?}"
            );
        }
    }

    #[test]
    fn non_finite_numbers_and_bad_restitution_are_rejected() {
        let mut engine = space();
        let a = moving_body(&mut engine, 1.0, [0.0; 3]);

        let mut solver = EngineConstraintSolver::new();
        solver.add_contact(a, u32::MAX, 0.0, 1.0, 0.0, f64::NAN, 0.0);
        assert_eq!(invalid_parameter(solver.solve(&mut engine, DT)), "depth");

        let mut solver = EngineConstraintSolver::new();
        solver.add_constraint(a, u32::MAX, 0.0, 1.0, 0.0, f64::INFINITY);
        assert_eq!(invalid_parameter(solver.solve(&mut engine, DT)), "bias");

        let mut solver = EngineConstraintSolver::new();
        solver.set_baumgarte(f64::NAN);
        solver.add_contact(a, u32::MAX, 0.0, 1.0, 0.0, 0.1, 0.0);
        assert_eq!(
            invalid_parameter(solver.solve(&mut engine, DT)),
            "baumgarte"
        );

        for restitution in [1.5, -0.1, f64::NAN] {
            let mut solver = EngineConstraintSolver::new();
            solver.add_contact(a, u32::MAX, 0.0, 1.0, 0.0, 0.1, restitution);
            assert!(
                matches!(
                    solver.solve(&mut engine, DT),
                    Err(Error::InvalidRestitution(_))
                ),
                "restitution {restitution}"
            );
        }
        assert_eq!(engine.get_velocity(a), [0.0; 3]);
    }
}
