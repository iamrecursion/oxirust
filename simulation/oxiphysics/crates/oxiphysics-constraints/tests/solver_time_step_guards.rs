//! Regression tests for the time-step guards of the constraint solvers.
//!
//! A paused simulation passes `dt = 0.0` to its solvers (a broken clock passes
//! NaN or a negative step).  The Baumgarte and XPBD terms of the constraints
//! scale with `1 / dt`, so such a step used to turn velocities into NaN.  Every
//! solver entry point now skips a step whose `dt` is not positive and finite:
//! nothing is prepared, solved or integrated and the returned statistics say
//! that no iteration ran.  The `try_*` entry points report the same condition
//! as an error.

use std::cell::Cell;
use std::rc::Rc;

use oxiphysics_constraints::pgs_solver::{
    ConstraintBlock, PgsState, SolverStats, SorPgsSolver, solve_block_pgs, try_solve_block_pgs,
};
use oxiphysics_constraints::tgs_solver::{AdaptiveTgsSolver, TgsWarmStartState};
use oxiphysics_constraints::{Constraint, Error, PgsSolver, SixDofConstraint, TgsSolver};
use oxiphysics_core::BodyHandle;
use oxiphysics_core::math::Vec3;
use oxiphysics_rigid::{RigidBody, RigidBodySet};

const DT: f64 = 1.0 / 60.0;

/// The time steps no solver can use.
const BAD_STEPS: [f64; 5] = [0.0, -DT, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

// ── Probe constraint ─────────────────────────────────────────────────────────

#[derive(Default)]
struct Calls {
    prepare: Cell<usize>,
    velocity: Cell<usize>,
    position: Cell<usize>,
    post_solve: Cell<usize>,
}

impl Calls {
    fn total(&self) -> usize {
        self.prepare.get() + self.velocity.get() + self.position.get() + self.post_solve.get()
    }
}

/// A constraint that counts its calls and speeds its body up by 1 m/s in every
/// velocity sweep (and never converges).
struct Probe {
    body: BodyHandle,
    calls: Rc<Calls>,
}

impl Constraint for Probe {
    fn prepare(&mut self, _bodies: &RigidBodySet, _dt: f64) {
        self.calls.prepare.set(self.calls.prepare.get() + 1);
    }
    fn solve_velocity(&mut self, bodies: &mut RigidBodySet, _dt: f64) {
        self.calls.velocity.set(self.calls.velocity.get() + 1);
        if let Some(body) = bodies.get_mut(self.body) {
            body.velocity += Vec3::new(1.0, 0.0, 0.0);
        }
    }
    fn solve_position(&mut self, _bodies: &mut RigidBodySet, _dt: f64) {
        self.calls.position.set(self.calls.position.get() + 1);
    }
    fn body_handles(&self) -> Vec<BodyHandle> {
        vec![self.body]
    }
    fn post_solve(&mut self) {
        self.calls.post_solve.set(self.calls.post_solve.get() + 1);
    }
    fn residual(&self) -> f64 {
        1.0
    }
}

struct Scene {
    bodies: RigidBodySet,
    body: BodyHandle,
    constraints: Vec<Box<dyn Constraint>>,
    calls: Rc<Calls>,
}

fn scene() -> Scene {
    let mut bodies = RigidBodySet::new();
    let mut body = RigidBody::new(1.0);
    body.velocity = Vec3::new(2.0, 0.0, 0.0);
    let body = bodies.insert(body);
    let calls = Rc::new(Calls::default());
    let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(Probe {
        body,
        calls: Rc::clone(&calls),
    })];
    Scene {
        bodies,
        body,
        constraints,
        calls,
    }
}

impl Scene {
    fn velocity(&self) -> Vec3 {
        self.bodies
            .get(self.body)
            .map(|b| b.velocity)
            .unwrap_or_default()
    }

    fn position(&self) -> Vec3 {
        self.bodies
            .get(self.body)
            .map(|b| b.transform.position)
            .unwrap_or_default()
    }

    /// Nothing was prepared, solved or integrated.
    fn assert_untouched(&self, what: &str) {
        assert_eq!(self.calls.total(), 0, "{what}: the constraint was called");
        assert_eq!(self.velocity(), Vec3::new(2.0, 0.0, 0.0), "{what}");
        assert_eq!(self.position(), Vec3::zeros(), "{what}");
    }
}

fn assert_not_run(stats: SolverStats, what: &str) {
    assert_eq!(stats.iterations_used, 0, "{what}");
    assert_eq!(stats.residual, f64::MAX, "{what}");
    assert!(!stats.converged, "{what}");
}

fn assert_invalid_dt(error: &Error, dt: f64, what: &str) {
    match error {
        Error::InvalidTimestep { dt: reported } => {
            assert!(
                (reported.is_nan() && dt.is_nan()) || *reported == dt,
                "{what}: reported dt {reported}, passed {dt}"
            );
        }
        other => panic!("{what}: expected InvalidTimestep, got {other:?}"),
    }
}

// ── PGS, SOR-PGS and block PGS ───────────────────────────────────────────────

#[test]
fn pgs_solvers_skip_a_step_with_an_invalid_time_step() {
    let pgs = PgsSolver::new(4, 2);
    let sor = SorPgsSolver::new(4, 2, 1.3);
    for dt in BAD_STEPS {
        let what = format!("PgsSolver::solve, dt = {dt}");
        let mut s = scene();
        assert_not_run(pgs.solve(&mut s.constraints, &mut s.bodies, dt), &what);
        s.assert_untouched(&what);

        let what = format!("SorPgsSolver::solve, dt = {dt}");
        let mut s = scene();
        assert_not_run(sor.solve(&mut s.constraints, &mut s.bodies, dt), &what);
        s.assert_untouched(&what);

        let what = format!("solve_block_pgs, dt = {dt}");
        let mut s = scene();
        let blocks = [ConstraintBlock::new(vec![0], "probe")];
        let stats = solve_block_pgs(&mut s.constraints, &blocks, &mut s.bodies, dt, 4, 2);
        assert_not_run(stats, &what);
        s.assert_untouched(&what);
    }
}

#[test]
fn warm_started_pgs_solves_leave_their_state_alone_for_an_invalid_time_step() {
    let pgs = PgsSolver::new(4, 2);
    let sor = SorPgsSolver::new(4, 2, 1.3);
    for dt in BAD_STEPS {
        for use_sor in [false, true] {
            let what = format!("solve_warm (sor = {use_sor}), dt = {dt}");
            let mut s = scene();
            let mut state = PgsState::new(1);
            state.lambdas[0] = 3.0;
            state.cached_impulses[0] = [1.0, 2.0, 3.0];
            state.row_impulses[0] = vec![4.0, 5.0];
            state.prev_dt = 0.02;
            let stats = if use_sor {
                sor.solve_warm(&mut s.constraints, &mut s.bodies, dt, &mut state)
            } else {
                pgs.solve_warm(&mut s.constraints, &mut s.bodies, dt, &mut state)
            };
            assert_not_run(stats, &what);
            s.assert_untouched(&what);
            // Scaling the state by `dt / prev_dt` would have zeroed or poisoned it.
            assert_eq!(state.lambdas, vec![3.0], "{what}");
            assert_eq!(state.cached_impulses, vec![[1.0, 2.0, 3.0]], "{what}");
            assert_eq!(state.row_impulses, vec![vec![4.0, 5.0]], "{what}");
            assert_eq!(state.prev_dt, 0.02, "{what}");
        }
    }
}

#[test]
fn a_valid_time_step_still_runs_every_phase() {
    let pgs = PgsSolver::new(4, 2);
    let mut s = scene();
    let stats = pgs.solve(&mut s.constraints, &mut s.bodies, DT);
    // The probe never converges, so all four sweeps run.
    assert_eq!(stats.iterations_used, 4);
    assert!(!stats.converged);
    assert_eq!(s.calls.prepare.get(), 1);
    assert_eq!(s.calls.velocity.get(), 4);
    assert_eq!(s.calls.position.get(), 2);
    assert_eq!(s.calls.post_solve.get(), 1);
    assert_eq!(s.velocity(), Vec3::new(6.0, 0.0, 0.0));

    let mut s = scene();
    let blocks = [ConstraintBlock::new(vec![0], "probe")];
    let stats = solve_block_pgs(&mut s.constraints, &blocks, &mut s.bodies, DT, 3, 2);
    assert_eq!(stats.iterations_used, 3);
    assert_eq!(s.calls.velocity.get(), 6);
}

#[test]
fn try_solve_reports_an_invalid_time_step_as_an_error() {
    let pgs = PgsSolver::new(4, 2);
    let sor = SorPgsSolver::new(4, 2, 1.3);
    for dt in BAD_STEPS {
        let mut s = scene();
        let blocks = [ConstraintBlock::new(vec![0], "probe")];
        let mut state = PgsState::new(1);

        let error = pgs
            .try_solve(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("PgsSolver::try_solve");
        assert_invalid_dt(&error, dt, "PgsSolver::try_solve");
        let error = pgs
            .try_solve_warm(&mut s.constraints, &mut s.bodies, dt, &mut state)
            .expect_err("PgsSolver::try_solve_warm");
        assert_invalid_dt(&error, dt, "PgsSolver::try_solve_warm");
        let error = sor
            .try_solve(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("SorPgsSolver::try_solve");
        assert_invalid_dt(&error, dt, "SorPgsSolver::try_solve");
        let error = sor
            .try_solve_warm(&mut s.constraints, &mut s.bodies, dt, &mut state)
            .expect_err("SorPgsSolver::try_solve_warm");
        assert_invalid_dt(&error, dt, "SorPgsSolver::try_solve_warm");
        let error = try_solve_block_pgs(&mut s.constraints, &blocks, &mut s.bodies, dt, 4, 2)
            .expect_err("try_solve_block_pgs");
        assert_invalid_dt(&error, dt, "try_solve_block_pgs");
        s.assert_untouched(&format!("try_solve*, dt = {dt}"));
    }

    // A valid step gives exactly what the infallible entry points give.
    let mut s = scene();
    let stats = pgs
        .try_solve(&mut s.constraints, &mut s.bodies, DT)
        .expect("a valid step");
    assert_eq!(stats.iterations_used, 4);
    assert_eq!(s.velocity(), Vec3::new(6.0, 0.0, 0.0));
    let mut s = scene();
    let mut state = PgsState::new(1);
    let stats = sor
        .try_solve_warm(&mut s.constraints, &mut s.bodies, DT, &mut state)
        .expect("a valid step");
    assert_eq!(stats.iterations_used, 4);
    let mut s = scene();
    let blocks = [ConstraintBlock::new(vec![0], "probe")];
    let stats = try_solve_block_pgs(&mut s.constraints, &blocks, &mut s.bodies, DT, 2, 1)
        .expect("a valid step");
    assert_eq!(stats.iterations_used, 2);
}

// ── TGS ───────────────────────────────────────────────────────────────────────

#[test]
fn tgs_try_entry_points_report_invalid_time_steps_and_missing_substeps() {
    let tgs = TgsSolver::new(2, 3, Vec3::new(0.0, -9.81, 0.0));
    for dt in BAD_STEPS {
        let mut s = scene();
        let mut state = TgsWarmStartState::new(1);
        let handle = s.body;
        let mut island = scene();

        let error = tgs
            .try_step(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("try_step");
        assert_invalid_dt(&error, dt, "try_step");
        let error = tgs
            .try_solve(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("try_solve");
        assert_invalid_dt(&error, dt, "try_solve");
        let error = tgs
            .try_step_warm(&mut s.constraints, &mut s.bodies, dt, &mut state)
            .expect_err("try_step_warm");
        assert_invalid_dt(&error, dt, "try_step_warm");
        let error = tgs
            .try_step_with_diagnostics(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("try_step_with_diagnostics");
        assert_invalid_dt(&error, dt, "try_step_with_diagnostics");
        let error = tgs
            .try_solve_island(&[handle], &mut island.constraints, &mut island.bodies, dt)
            .expect_err("try_solve_island");
        assert_invalid_dt(&error, dt, "try_solve_island");
        let error = AdaptiveTgsSolver::new(3, Vec3::zeros(), 0.01)
            .try_step(&mut s.constraints, &mut s.bodies, dt)
            .expect_err("AdaptiveTgsSolver::try_step");
        assert_invalid_dt(&error, dt, "AdaptiveTgsSolver::try_step");

        s.assert_untouched(&format!("try_*, dt = {dt}"));
        island.assert_untouched(&format!("try_solve_island, dt = {dt}"));
        assert_eq!(state.age, 0);
    }

    // No substeps is a configuration error, not a silent no-op.
    let no_substeps = TgsSolver::new(0, 3, Vec3::zeros());
    let mut s = scene();
    match no_substeps.try_step(&mut s.constraints, &mut s.bodies, DT) {
        Err(Error::ConfigurationError {
            parameter, value, ..
        }) => {
            assert_eq!(parameter, "substeps");
            assert_eq!(value, 0.0);
        }
        other => panic!("expected a ConfigurationError, got {other:?}"),
    }
    assert!(
        no_substeps
            .try_solve_island(&[s.body], &mut s.constraints, &mut s.bodies, DT)
            .is_err()
    );
    s.assert_untouched("zero substeps");

    // A valid step gives what the infallible entry points give.
    let mut s = scene();
    let stats = tgs
        .try_step(&mut s.constraints, &mut s.bodies, DT)
        .expect("a valid step");
    assert_eq!(stats.iterations_used, 6);
    assert_eq!(s.calls.prepare.get(), 2);
    let mut s = scene();
    let (stats, diagnostics) = tgs
        .try_step_with_diagnostics(&mut s.constraints, &mut s.bodies, DT)
        .expect("a valid step");
    assert_eq!(stats.iterations_used, 6);
    assert_eq!(diagnostics.len(), 2);
    let mut s = scene();
    let mut state = TgsWarmStartState::new(1);
    tgs.try_step_warm(&mut s.constraints, &mut s.bodies, DT, &mut state)
        .expect("a valid step");
    assert_eq!(state.age, 1);
    let mut s = scene();
    let stats = AdaptiveTgsSolver::new(3, Vec3::zeros(), 0.01)
        .try_step(&mut s.constraints, &mut s.bodies, DT)
        .expect("a valid step");
    assert!(stats.iterations_used > 0);
}

#[test]
fn tgs_step_warm_leaves_its_state_alone_when_the_step_does_nothing() {
    let setups = [
        ("dt = 0", TgsSolver::new(2, 3, Vec3::zeros()), 0.0),
        ("dt = NaN", TgsSolver::new(2, 3, Vec3::zeros()), f64::NAN),
        ("dt < 0", TgsSolver::new(2, 3, Vec3::zeros()), -DT),
        ("no substeps", TgsSolver::new(0, 3, Vec3::zeros()), DT),
    ];
    for (what, tgs, dt) in setups {
        let mut s = scene();
        let mut state = TgsWarmStartState::new(2);
        state.normal_impulses = vec![1.5, 2.5];
        state.friction_impulses = vec![[0.5, 0.25], [0.125, 0.0625]];
        state.prev_dt = 0.02;
        state.age = 3;
        let stats = tgs.step_warm(&mut s.constraints, &mut s.bodies, dt, &mut state);
        assert_not_run(stats, what);
        s.assert_untouched(what);
        // `dt / prev_dt` would have scaled the impulses by zero or NaN and
        // `prev_dt` would have become unusable for the next frame.
        assert_eq!(state.normal_impulses, vec![1.5, 2.5], "{what}");
        assert_eq!(
            state.friction_impulses,
            vec![[0.5, 0.25], [0.125, 0.0625]],
            "{what}"
        );
        assert_eq!(state.prev_dt, 0.02, "{what}");
        assert_eq!(state.age, 3, "{what}");
    }

    // A usable step still scales the state by the time-step ratio and ages it.
    let tgs = TgsSolver::new(2, 3, Vec3::zeros());
    let mut s = scene();
    let mut state = TgsWarmStartState::new(1);
    state.normal_impulses = vec![4.0];
    state.prev_dt = 2.0 * DT;
    tgs.step_warm(&mut s.constraints, &mut s.bodies, DT, &mut state);
    assert_eq!(state.normal_impulses, vec![2.0]);
    assert_eq!(state.prev_dt, DT);
    assert_eq!(state.age, 1);
}

// ── Six degrees of freedom ───────────────────────────────────────────────────

/// Two dynamic bodies that are offset and rotating against each other, welded
/// together.
fn welded_pair() -> (RigidBodySet, SixDofConstraint, [BodyHandle; 2]) {
    let mut bodies = RigidBodySet::new();
    let mut a = RigidBody::new(1.0);
    a.velocity = Vec3::new(1.0, 0.0, 0.0);
    a.angular_velocity = Vec3::new(0.0, 0.5, 0.0);
    let mut b = RigidBody::new(2.0);
    b.transform.position = Vec3::new(0.3, 0.0, 0.0);
    b.velocity = Vec3::new(-1.0, 0.5, 0.0);
    b.angular_velocity = Vec3::new(0.2, 0.0, 0.0);
    let ha = bodies.insert(a);
    let hb = bodies.insert(b);
    (bodies, SixDofConstraint::weld(ha, hb), [ha, hb])
}

#[test]
fn a_paused_weld_keeps_every_velocity_finite() {
    for dt in BAD_STEPS {
        let (mut bodies, mut weld, handles) = welded_pair();
        weld.prepare(&bodies, dt);
        weld.solve_velocity(&mut bodies, dt);
        weld.solve_position(&mut bodies, dt);
        for handle in handles {
            let body = bodies.get(handle).expect("the body exists");
            for value in [
                body.velocity.x,
                body.velocity.y,
                body.velocity.z,
                body.angular_velocity.x,
                body.angular_velocity.y,
                body.angular_velocity.z,
                body.transform.position.x,
                body.transform.position.y,
                body.transform.position.z,
            ] {
                assert!(value.is_finite(), "dt = {dt}: {body:?}");
            }
        }
    }
}

#[test]
fn solvers_skip_a_paused_weld_entirely() {
    let pgs = PgsSolver::new(8, 2);
    let tgs = TgsSolver::new(2, 8, Vec3::zeros());
    for dt in [0.0, f64::NAN] {
        for use_tgs in [false, true] {
            let (mut bodies, weld, handles) = welded_pair();
            let before: Vec<_> = handles
                .iter()
                .map(|&h| {
                    let b = bodies.get(h).expect("the body exists");
                    (b.velocity, b.angular_velocity, b.transform.position)
                })
                .collect();
            let mut constraints: Vec<Box<dyn Constraint>> = vec![Box::new(weld)];
            let stats = if use_tgs {
                tgs.step(&mut constraints, &mut bodies, dt)
            } else {
                pgs.solve(&mut constraints, &mut bodies, dt)
            };
            assert_not_run(stats, &format!("tgs = {use_tgs}, dt = {dt}"));
            for (handle, expected) in handles.iter().zip(before) {
                let b = bodies.get(*handle).expect("the body exists");
                assert_eq!(
                    (b.velocity, b.angular_velocity, b.transform.position),
                    expected
                );
            }
        }
    }
}
