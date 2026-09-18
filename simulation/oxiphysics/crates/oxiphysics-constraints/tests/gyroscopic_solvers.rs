// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The implicit gyroscopic stage of the integration loops inside the constraint solvers:
//! [`TgsSolver`] (`step`, `step_with_diagnostics`, `solve_island`, [`AdaptiveTgsSolver`]) and
//! [`CcdConstraintSolver`].
//!
//! The angular velocity of a [`RigidBody`] is a world-frame quantity. A torque-free body that is
//! not isotropic cannot keep it constant: the conserved quantity is the world-frame angular
//! momentum `L = R I Rᵀ ω`. These solvers used to integrate the orientation with the angular
//! velocity of the start of every (sub-)step, so the *vector* `L` of an asymmetric body drifted by
//! almost 10 % in ten seconds even though its magnitude `|L|` is conserved to rounding. Every
//! conservation check below therefore compares the vector, never the magnitude.

use oxiphysics_constraints::tgs_solver::AdaptiveTgsSolver;
use oxiphysics_constraints::{CcdConstraint, CcdConstraintSolver, Constraint, TgsSolver};
use oxiphysics_core::BodyHandle;
use oxiphysics_core::math::{Mat3, Vec3};
use oxiphysics_rigid::gyroscopic::gyroscopic_implicit_step_world;
use oxiphysics_rigid::{RigidBody, RigidBodySet};

/// Principal moments of a 0.2 x 0.4 x 0.3 m box of unit mass. They are all different, so a free
/// body tumbles instead of spinning at a constant rate.
const BOX_MOMENTS: [f64; 3] = [
    0.020_833_333_333_333_332,
    0.010_833_333_333_333_334,
    0.016_666_666_666_666_666,
];

const DT: f64 = 0.01;
const STEPS: usize = 1000;

fn box_inertia() -> Mat3 {
    Mat3::from_diagonal(&Vec3::new(BOX_MOMENTS[0], BOX_MOMENTS[1], BOX_MOMENTS[2]))
}

/// Spin about no principal axis, close to the intermediate one.
fn initial_spin() -> Vec3 {
    Vec3::new(1.0, 0.1, 0.0)
}

/// Undamped, gravity-free unit-mass box with the given world-frame angular velocity.
fn box_body(omega: Vec3) -> RigidBody {
    let mut body = RigidBody::new(1.0);
    body.local_inertia = box_inertia();
    body.update_world_inertia();
    body.angular_velocity = omega;
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body.gravity_scale = 0.0;
    body
}

/// World-frame angular momentum `R I Rᵀ ω`, written out so that it does not depend on the helpers
/// the integrator itself uses.
fn world_angular_momentum(body: &RigidBody) -> Vec3 {
    let rotation = *body.transform.rotation.to_rotation_matrix().matrix();
    rotation * body.local_inertia * rotation.transpose() * body.angular_velocity
}

fn rotational_energy(body: &RigidBody) -> f64 {
    0.5 * world_angular_momentum(body).dot(&body.angular_velocity)
}

fn body_of(bodies: &RigidBodySet, handle: BodyHandle) -> &RigidBody {
    bodies.get(handle).expect("the body must exist")
}

/// Largest relative drift of the angular-momentum vector and the range of the rotational energy
/// of one body over a run.
struct Conservation {
    initial_momentum: Vec3,
    initial_energy: f64,
    max_vector_drift: f64,
    max_energy_ratio: f64,
    min_energy_ratio: f64,
}

impl Conservation {
    fn new(body: &RigidBody) -> Self {
        Self {
            initial_momentum: world_angular_momentum(body),
            initial_energy: rotational_energy(body),
            max_vector_drift: 0.0,
            max_energy_ratio: 1.0,
            min_energy_ratio: 1.0,
        }
    }

    fn record(&mut self, body: &RigidBody) {
        let drift = (world_angular_momentum(body) - self.initial_momentum).norm()
            / self.initial_momentum.norm();
        self.max_vector_drift = self.max_vector_drift.max(drift);
        let ratio = rotational_energy(body) / self.initial_energy;
        self.max_energy_ratio = self.max_energy_ratio.max(ratio);
        self.min_energy_ratio = self.min_energy_ratio.min(ratio);
    }

    /// The first-order backward-Euler step drifts by 4.7e-4 over ten seconds at a step of 0.01 s,
    /// is slightly dissipative and never gains energy.
    fn assert_conserved(&self, what: &str) {
        assert!(
            self.max_vector_drift < 2.0e-3,
            "{what}: the angular momentum vector drifted by {:.3e}",
            self.max_vector_drift
        );
        assert!(
            self.max_energy_ratio <= 1.0 + 1e-12,
            "{what}: the rotational energy grew to {} of its initial value",
            self.max_energy_ratio
        );
        assert!(
            self.min_energy_ratio > 0.999,
            "{what}: the rotational energy fell to {} of its initial value",
            self.min_energy_ratio
        );
    }
}

/// A set holding one asymmetric box, and its handle.
fn tumbling_box() -> (RigidBodySet, BodyHandle) {
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(box_body(initial_spin()));
    (bodies, handle)
}

fn no_constraints() -> Vec<Box<dyn Constraint>> {
    Vec::new()
}

/// The state a bit-compatibility check compares.
fn motion_state(body: &RigidBody) -> (Vec3, Vec3, Vec3, [f64; 4]) {
    let q = body.transform.rotation.quaternion();
    (
        body.transform.position,
        body.velocity,
        body.angular_velocity,
        [q.i, q.j, q.k, q.w],
    )
}

// ---------------------------------------------------------------------------
// TgsSolver
// ---------------------------------------------------------------------------

#[test]
fn tgs_step_conserves_the_angular_momentum_vector_of_an_asymmetric_body() {
    let solver = TgsSolver::new(4, 4, Vec3::zeros());
    assert!(solver.use_implicit_gyroscopic);
    let (mut bodies, handle) = tumbling_box();
    let mut constraints = no_constraints();
    let mut conservation = Conservation::new(body_of(&bodies, handle));
    let mut max_omega_change = 0.0_f64;
    for _ in 0..STEPS {
        solver.step(&mut constraints, &mut bodies, DT);
        let body = body_of(&bodies, handle);
        conservation.record(body);
        max_omega_change = max_omega_change.max((body.angular_velocity - initial_spin()).norm());
    }
    conservation.assert_conserved("TgsSolver::step");
    // The world-frame angular velocity genuinely changed: the body tumbles.
    assert!(
        max_omega_change > 1e-2,
        "omega never left its initial value (max change {max_omega_change:.3e})"
    );
}

#[test]
fn tgs_step_with_diagnostics_and_the_adaptive_solver_conserve_it_too() {
    let solver = TgsSolver::new(2, 4, Vec3::zeros());
    let (mut bodies, handle) = tumbling_box();
    let mut constraints = no_constraints();
    let mut conservation = Conservation::new(body_of(&bodies, handle));
    for _ in 0..STEPS {
        let (_, diagnostics) = solver.step_with_diagnostics(&mut constraints, &mut bodies, DT);
        assert_eq!(diagnostics.len(), 2);
        conservation.record(body_of(&bodies, handle));
    }
    conservation.assert_conserved("TgsSolver::step_with_diagnostics");

    let adaptive = AdaptiveTgsSolver::new(4, Vec3::zeros(), 0.05);
    assert!(adaptive.base.use_implicit_gyroscopic);
    let (mut bodies, handle) = tumbling_box();
    let mut conservation = Conservation::new(body_of(&bodies, handle));
    for _ in 0..STEPS {
        adaptive.step(&mut constraints, &mut bodies, DT);
        conservation.record(body_of(&bodies, handle));
    }
    conservation.assert_conserved("AdaptiveTgsSolver::step");
}

#[test]
fn tgs_stage_runs_in_every_substep() {
    // Backward Euler is first order, so the drift shrinks with the substep: eight substeps drift
    // about eight times less than one. A stage that ran once per step would not improve.
    let drift_with = |substeps: usize| {
        let solver = TgsSolver::new(substeps, 4, Vec3::zeros());
        let (mut bodies, handle) = tumbling_box();
        let mut constraints = no_constraints();
        let mut conservation = Conservation::new(body_of(&bodies, handle));
        for _ in 0..STEPS {
            solver.step(&mut constraints, &mut bodies, DT);
            conservation.record(body_of(&bodies, handle));
        }
        conservation.max_vector_drift
    };
    let coarse = drift_with(1);
    let fine = drift_with(8);
    assert!(
        fine < coarse / 4.0,
        "1 substep drifts by {coarse:.3e}, 8 substeps by {fine:.3e}"
    );
}

#[test]
fn tgs_with_the_stage_switched_off_keeps_the_world_frame_omega_constant() {
    // The pre-fix behaviour, still available on request: omega stays constant, |L| is conserved to
    // rounding and the vector L drifts by almost 10 %.
    let solver = TgsSolver::new(4, 4, Vec3::zeros()).with_implicit_gyroscopic(false);
    assert!(!solver.use_implicit_gyroscopic);
    let (mut bodies, handle) = tumbling_box();
    let mut constraints = no_constraints();
    let initial_l = world_angular_momentum(body_of(&bodies, handle));
    let mut max_vector_drift = 0.0_f64;
    let mut max_magnitude_drift = 0.0_f64;
    for _ in 0..STEPS {
        solver.step(&mut constraints, &mut bodies, DT);
        let body = body_of(&bodies, handle);
        assert_eq!(body.angular_velocity, initial_spin());
        let l = world_angular_momentum(body);
        max_vector_drift = max_vector_drift.max((l - initial_l).norm() / initial_l.norm());
        max_magnitude_drift =
            max_magnitude_drift.max((l.norm() - initial_l.norm()).abs() / initial_l.norm());
    }
    assert!(
        max_magnitude_drift < 1e-12,
        "magnitude drift {max_magnitude_drift:.3e}"
    );
    assert!(
        max_vector_drift > 0.05,
        "vector drift {max_vector_drift:.3e}"
    );
}

#[test]
fn every_tgs_constructor_enables_the_stage() {
    assert!(TgsSolver::new(4, 4, Vec3::zeros()).use_implicit_gyroscopic);
    assert!(TgsSolver::default().use_implicit_gyroscopic);
    assert!(TgsSolver::with_options(2, 4, 1, Vec3::zeros(), 0.2, true).use_implicit_gyroscopic);
    assert!(
        !TgsSolver::default()
            .with_implicit_gyroscopic(false)
            .use_implicit_gyroscopic
    );
    assert!(
        TgsSolver::default()
            .with_implicit_gyroscopic(false)
            .with_implicit_gyroscopic(true)
            .use_implicit_gyroscopic
    );
}

#[test]
fn tgs_island_solve_corrects_only_the_bodies_of_the_island() {
    let solver = TgsSolver::new(2, 4, Vec3::zeros());
    let mut bodies = RigidBodySet::new();
    let inside = bodies.insert(box_body(initial_spin()));
    let outside = bodies.insert(box_body(initial_spin()));
    let outside_before = motion_state(body_of(&bodies, outside));
    let mut constraints = no_constraints();
    let mut conservation = Conservation::new(body_of(&bodies, inside));
    for _ in 0..STEPS {
        solver.solve_island(&[inside], &mut constraints, &mut bodies, DT);
        conservation.record(body_of(&bodies, inside));
    }
    conservation.assert_conserved("TgsSolver::solve_island");
    // The island solve does not touch a body that is not part of the island, not even to rotate it.
    assert_eq!(motion_state(body_of(&bodies, outside)), outside_before);
}

#[test]
fn tgs_leaves_isotropic_bodies_bit_identical() {
    let spin = Vec3::new(0.3, -0.2, 0.5);
    let run = |enabled: bool| {
        let solver = TgsSolver::new(4, 4, Vec3::zeros()).with_implicit_gyroscopic(enabled);
        let mut sphere = RigidBody::new(2.0);
        sphere.angular_velocity = spin;
        sphere.velocity = Vec3::new(1.0, 2.0, -3.0);
        let mut bodies = RigidBodySet::new();
        let handle = bodies.insert(sphere);
        let mut constraints = no_constraints();
        for _ in 0..200 {
            solver.step(&mut constraints, &mut bodies, DT);
        }
        motion_state(body_of(&bodies, handle))
    };
    let with_stage = run(true);
    assert_eq!(with_stage, run(false));
    assert_eq!(with_stage.2, spin);
}

/// A constraint that overwrites the angular velocity of its body in every velocity sweep, as a
/// motor or a limit may.
struct SetAngularVelocity {
    handle: BodyHandle,
    target: Vec3,
}

impl Constraint for SetAngularVelocity {
    fn prepare(&mut self, _bodies: &RigidBodySet, _dt: f64) {}

    fn solve_velocity(&mut self, bodies: &mut RigidBodySet, _dt: f64) {
        if let Some(body) = bodies.get_mut(self.handle) {
            body.angular_velocity = self.target;
        }
    }

    fn solve_position(&mut self, _bodies: &mut RigidBodySet, _dt: f64) {}

    fn body_handles(&self) -> Vec<BodyHandle> {
        vec![self.handle]
    }
}

#[test]
fn tgs_stage_runs_after_the_constraint_solve_and_before_the_orientation_update() {
    let target = Vec3::new(0.2, 1.5, -0.7);
    let run = |enabled: bool| {
        let (mut bodies, handle) = tumbling_box();
        let mut constraints: Vec<Box<dyn Constraint>> =
            vec![Box::new(SetAngularVelocity { handle, target })];
        let solver = TgsSolver::new(1, 2, Vec3::zeros()).with_implicit_gyroscopic(enabled);
        solver.step(&mut constraints, &mut bodies, DT);
        body_of(&bodies, handle).clone()
    };
    // The body starts at the identity orientation, so its world inertia is the box tensor.
    let corrected = gyroscopic_implicit_step_world(target, &box_inertia(), DT);
    assert!(
        (corrected - target).norm() > 1e-4,
        "the target must be a spin the stage changes"
    );
    let with_stage = run(true);
    assert_eq!(with_stage.angular_velocity, corrected);
    assert_eq!(run(false).angular_velocity, target);
    // The orientation was advanced with the corrected angular velocity.
    let mut reference = box_body(corrected);
    reference.integrate_velocity(DT);
    assert_eq!(with_stage.transform.rotation, reference.transform.rotation);
}

// ---------------------------------------------------------------------------
// CcdConstraintSolver
// ---------------------------------------------------------------------------

/// A CCD constraint at `toi` that never produces an impulse: the tumbling body sits at rest at
/// the contact point, so the relative normal velocity there is zero and the solver leaves the
/// bodies alone. It only splits the step at `toi`.
fn inert_ccd_constraint(a: BodyHandle, b: BodyHandle, toi: f64) -> CcdConstraint {
    CcdConstraint::new(a, b, toi, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros(), 0.0, 0.0)
}

fn tumbling_box_with_static_partner() -> (RigidBodySet, BodyHandle, BodyHandle) {
    let (mut bodies, handle) = tumbling_box();
    let partner = bodies.insert(RigidBody::new_static());
    (bodies, handle, partner)
}

#[test]
fn ccd_solver_conserves_the_angular_momentum_vector_in_one_piece_and_in_three() {
    let solver = CcdConstraintSolver::new(4, Vec3::zeros());
    assert!(solver.use_implicit_gyroscopic);
    for toi_events in [&[][..], &[0.3, 0.7][..]] {
        let (mut bodies, handle, partner) = tumbling_box_with_static_partner();
        let constraints: Vec<CcdConstraint> = toi_events
            .iter()
            .map(|&toi| inert_ccd_constraint(handle, partner, toi))
            .collect();
        let mut conservation = Conservation::new(body_of(&bodies, handle));
        let mut max_omega_change = 0.0_f64;
        for _ in 0..STEPS {
            solver.solve(&constraints, &mut bodies, DT);
            let body = body_of(&bodies, handle);
            conservation.record(body);
            max_omega_change =
                max_omega_change.max((body.angular_velocity - initial_spin()).norm());
        }
        conservation.assert_conserved(&format!(
            "CcdConstraintSolver::solve with {} impacts",
            toi_events.len()
        ));
        assert!(
            max_omega_change > 1e-2,
            "omega never left its initial value with {} impacts",
            toi_events.len()
        );
    }
}

#[test]
fn ccd_solver_with_the_stage_switched_off_keeps_the_world_frame_omega_constant() {
    let solver = CcdConstraintSolver::new(4, Vec3::zeros()).with_implicit_gyroscopic(false);
    assert!(!solver.use_implicit_gyroscopic);
    let (mut bodies, handle, partner) = tumbling_box_with_static_partner();
    let constraints = [inert_ccd_constraint(handle, partner, 0.5)];
    let initial_l = world_angular_momentum(body_of(&bodies, handle));
    let mut max_vector_drift = 0.0_f64;
    let mut max_magnitude_drift = 0.0_f64;
    for _ in 0..STEPS {
        solver.solve(&constraints, &mut bodies, DT);
        let body = body_of(&bodies, handle);
        assert_eq!(body.angular_velocity, initial_spin());
        let l = world_angular_momentum(body);
        max_vector_drift = max_vector_drift.max((l - initial_l).norm() / initial_l.norm());
        max_magnitude_drift =
            max_magnitude_drift.max((l.norm() - initial_l.norm()).abs() / initial_l.norm());
    }
    assert!(
        max_magnitude_drift < 1e-12,
        "magnitude drift {max_magnitude_drift:.3e}"
    );
    assert!(
        max_vector_drift > 0.05,
        "vector drift {max_vector_drift:.3e}"
    );
}

#[test]
fn ccd_solver_stage_runs_in_every_integration_piece() {
    // Each stretch between two impacts is integrated on its own, so a step split into ten pieces
    // drifts much less than the same step integrated in one piece.
    let solver = CcdConstraintSolver::new(4, Vec3::zeros());
    let drift_with = |toi_events: &[f64]| {
        let (mut bodies, handle, partner) = tumbling_box_with_static_partner();
        let constraints: Vec<CcdConstraint> = toi_events
            .iter()
            .map(|&toi| inert_ccd_constraint(handle, partner, toi))
            .collect();
        let mut conservation = Conservation::new(body_of(&bodies, handle));
        for _ in 0..STEPS {
            solver.solve(&constraints, &mut bodies, DT);
            conservation.record(body_of(&bodies, handle));
        }
        conservation.max_vector_drift
    };
    let one_piece = drift_with(&[]);
    let ten_pieces = drift_with(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    assert!(
        ten_pieces < one_piece / 4.0,
        "one piece drifts by {one_piece:.3e}, ten pieces by {ten_pieces:.3e}"
    );
}

#[test]
fn ccd_solver_leaves_isotropic_bodies_bit_identical() {
    let spin = Vec3::new(0.3, -0.2, 0.5);
    let run = |enabled: bool| {
        let solver = CcdConstraintSolver::new(4, Vec3::new(0.0, -9.81, 0.0))
            .with_implicit_gyroscopic(enabled);
        let mut sphere = RigidBody::new(2.0);
        sphere.angular_velocity = spin;
        sphere.transform.position = Vec3::new(0.0, 100.0, 0.0);
        let mut bodies = RigidBodySet::new();
        let handle = bodies.insert(sphere);
        for _ in 0..200 {
            solver.solve(&[], &mut bodies, DT);
        }
        motion_state(body_of(&bodies, handle))
    };
    let with_stage = run(true);
    assert_eq!(with_stage, run(false));
    assert_eq!(with_stage.2, spin);
}

#[test]
fn the_default_ccd_solver_enables_the_stage() {
    assert!(CcdConstraintSolver::default().use_implicit_gyroscopic);
    assert!(
        !CcdConstraintSolver::default()
            .with_implicit_gyroscopic(false)
            .use_implicit_gyroscopic
    );
}
