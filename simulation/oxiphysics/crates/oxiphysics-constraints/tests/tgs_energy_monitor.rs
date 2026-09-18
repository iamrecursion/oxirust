//! Regression tests for the TGS energy-gain monitor.
//!
//! `TgsSolver::energy_monitoring` used to compare a body's speed with a
//! threshold times itself (`v² > 1.05 · 1.05 · v²`), a condition that is never
//! true, so enabling it silently did nothing.  The monitor now measures the
//! total kinetic energy of the dynamic bodies (translational and rotational)
//! right after the forces were integrated, lets the constraint phase run and
//! scales all dynamic velocities down by a common factor if the energy grew by
//! more than 5 %.

use oxiphysics_constraints::{Constraint, ContactConstraint, TgsSolver};
use oxiphysics_core::BodyHandle;
use oxiphysics_core::math::{Quat, Vec3};
use oxiphysics_rigid::{RigidBody, RigidBodySet};

const DT: f64 = 1.0 / 60.0;
const THRESHOLD: f64 = 1.05;

/// A constraint that adds a fixed velocity to its bodies in every velocity
/// sweep: an artificial source of energy.  It never converges, so the solver
/// runs all of its sweeps.
struct EnergyInjector {
    bodies: Vec<BodyHandle>,
    linear: Vec3,
    angular: Vec3,
}

impl Constraint for EnergyInjector {
    fn prepare(&mut self, _bodies: &RigidBodySet, _dt: f64) {}

    fn solve_velocity(&mut self, bodies: &mut RigidBodySet, _dt: f64) {
        for &handle in &self.bodies {
            if let Some(body) = bodies.get_mut(handle) {
                body.velocity += self.linear;
                body.angular_velocity += self.angular;
            }
        }
    }

    fn solve_position(&mut self, _bodies: &mut RigidBodySet, _dt: f64) {}

    fn body_handles(&self) -> Vec<BodyHandle> {
        self.bodies.clone()
    }

    fn residual(&self) -> f64 {
        1.0
    }
}

fn injector(bodies: &[BodyHandle], linear: Vec3, angular: Vec3) -> Vec<Box<dyn Constraint>> {
    vec![Box::new(EnergyInjector {
        bodies: bodies.to_vec(),
        linear,
        angular,
    })]
}

/// One substep, four velocity sweeps, no position passes, no gravity.
fn solver(energy_monitoring: bool) -> TgsSolver {
    TgsSolver::with_options(1, 4, 0, Vec3::zeros(), 0.2, energy_monitoring)
}

fn dynamic_body(mass: f64, velocity: Vec3, angular_velocity: Vec3) -> RigidBody {
    let mut body = RigidBody::new(mass);
    body.velocity = velocity;
    body.angular_velocity = angular_velocity;
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body
}

fn velocity_of(bodies: &RigidBodySet, handle: BodyHandle) -> Vec3 {
    bodies.get(handle).map(|b| b.velocity).unwrap_or_default()
}

fn angular_velocity_of(bodies: &RigidBodySet, handle: BodyHandle) -> Vec3 {
    bodies
        .get(handle)
        .map(|b| b.angular_velocity)
        .unwrap_or_default()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual}"
    );
}

// ── The monitor limits the energy a constraint phase creates ─────────────────

#[test]
fn energy_gain_is_limited_to_five_percent_of_the_reference() {
    let build = || {
        let mut bodies = RigidBodySet::new();
        let handle = bodies.insert(dynamic_body(2.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros()));
        let constraints = injector(&[handle], Vec3::new(0.0, 1.0, 0.0), Vec3::zeros());
        (bodies, handle, constraints)
    };

    // Reference: with the monitor off the injector multiplies the speed by 5.
    let (mut bodies, handle, mut constraints) = build();
    solver(false).step(&mut constraints, &mut bodies, DT);
    assert_eq!(velocity_of(&bodies, handle), Vec3::new(0.0, 5.0, 0.0));
    assert_close(
        solver(false).total_kinetic_energy(&bodies),
        25.0,
        1e-12,
        "KE",
    );

    // With the monitor the energy ends at exactly 105 % of the 1 J it began with.
    let (mut bodies, handle, mut constraints) = build();
    let monitored = solver(true);
    let reference = monitored.total_kinetic_energy(&bodies);
    assert_close(reference, 1.0, 1e-12, "reference energy");
    monitored.step(&mut constraints, &mut bodies, DT);
    let energy = monitored.total_kinetic_energy(&bodies);
    assert!(
        energy <= THRESHOLD * reference * (1.0 + 1e-12),
        "energy {energy} exceeds {}",
        THRESHOLD * reference
    );
    assert_close(energy, THRESHOLD * reference, 1e-9, "damped energy");
    // The direction of the motion is kept: the velocity is only scaled.
    let v = velocity_of(&bodies, handle);
    assert_close(v.y, THRESHOLD.sqrt(), 1e-12, "scaled speed");
    assert_eq!((v.x, v.z), (0.0, 0.0));
}

#[test]
fn a_gain_within_the_threshold_is_left_alone() {
    // Four sweeps add 0.002 m/s each to a 1 m/s body: 1.008 m/s, which is 0.8 %
    // more speed and 1.6 % more energy, below the 5 % allowance.
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros()));
    let mut constraints = injector(&[handle], Vec3::new(0.0, 0.002, 0.0), Vec3::zeros());
    solver(true).step(&mut constraints, &mut bodies, DT);
    assert_close(velocity_of(&bodies, handle).y, 1.008, 1e-12, "velocity");
}

#[test]
fn rotational_energy_counts_towards_the_gain() {
    let build = || {
        let mut bodies = RigidBodySet::new();
        // Mass 2 gives the isotropic inertia 2 kg·m²: 1 rad/s is 1 J.
        let handle = bodies.insert(dynamic_body(2.0, Vec3::zeros(), Vec3::new(0.0, 0.0, 1.0)));
        let constraints = injector(&[handle], Vec3::zeros(), Vec3::new(0.0, 0.0, 1.0));
        (bodies, handle, constraints)
    };
    let (mut bodies, handle, mut constraints) = build();
    solver(false).step(&mut constraints, &mut bodies, DT);
    assert_eq!(
        angular_velocity_of(&bodies, handle),
        Vec3::new(0.0, 0.0, 5.0)
    );

    let (mut bodies, handle, mut constraints) = build();
    let monitored = solver(true);
    let reference = monitored.total_kinetic_energy(&bodies);
    assert_close(reference, 1.0, 1e-12, "rotational reference energy");
    monitored.step(&mut constraints, &mut bodies, DT);
    let energy = monitored.total_kinetic_energy(&bodies);
    assert_close(energy, THRESHOLD * reference, 1e-9, "damped energy");
    assert_close(
        angular_velocity_of(&bodies, handle).z,
        THRESHOLD.sqrt(),
        1e-12,
        "scaled spin",
    );
}

#[test]
fn translation_and_rotation_are_scaled_by_the_same_factor() {
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(
        1.0,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ));
    let mut constraints = injector(
        &[handle],
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let monitored = solver(true);
    let reference = monitored.total_kinetic_energy(&bodies);
    monitored.step(&mut constraints, &mut bodies, DT);
    let v = velocity_of(&bodies, handle);
    let w = angular_velocity_of(&bodies, handle);
    // Both were 5 times larger than at the start before the common scale.
    assert_close(v.x / w.y, 1.0, 1e-12, "ratio of the scaled velocities");
    assert_close(
        monitored.total_kinetic_energy(&bodies),
        THRESHOLD * reference,
        1e-9,
        "damped energy",
    );
}

#[test]
fn the_reference_is_taken_after_gravity_so_free_fall_is_not_damped() {
    // Gravity adds energy legitimately: the monitor must not fight it.
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let solver = TgsSolver::with_options(4, 4, 0, gravity, 0.2, true);
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(1.0, Vec3::zeros(), Vec3::zeros()));
    let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();
    solver.step(&mut constraints, &mut bodies, DT);
    assert_close(
        velocity_of(&bodies, handle).y,
        -9.81 * DT,
        1e-12,
        "free fall",
    );
}

// ── Energy that is only passed on is never touched ───────────────────────────

/// A moving body of 1 kg hits a resting one of 1 kg head on, elastically.
fn elastic_collision() -> (RigidBodySet, [BodyHandle; 2], Vec<Box<dyn Constraint>>) {
    let mut bodies = RigidBodySet::new();
    let mut resting = dynamic_body(1.0, Vec3::zeros(), Vec3::zeros());
    resting.transform.position = Vec3::new(0.0, 0.0, 0.0);
    let mut moving = dynamic_body(1.0, Vec3::new(-2.0, 0.0, 0.0), Vec3::zeros());
    moving.transform.position = Vec3::new(1.0, 0.0, 0.0);
    let resting = bodies.insert(resting);
    let moving = bodies.insert(moving);
    let contact = ContactConstraint::new(
        moving,
        resting,
        Vec3::new(1.0, 0.0, 0.0),
        0.0,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.5, 0.0, 0.0),
        1.0,
        0.0,
    );
    (bodies, [moving, resting], vec![Box::new(contact)])
}

#[test]
fn a_body_that_receives_energy_from_another_one_is_not_damped() {
    // The resting body gains 2 J, the moving one loses them: a per-body gain
    // check would have zeroed the receiver.
    let mut results = Vec::new();
    for monitoring in [false, true] {
        let (mut bodies, [moving, resting], mut constraints) = elastic_collision();
        let tgs = TgsSolver::with_options(1, 8, 0, Vec3::zeros(), 0.2, monitoring);
        let before = tgs.total_kinetic_energy(&bodies);
        tgs.step(&mut constraints, &mut bodies, DT);
        let after = tgs.total_kinetic_energy(&bodies);
        assert_close(before, 2.0, 1e-12, "energy before");
        assert_close(
            after,
            2.0,
            1e-9,
            "an elastic collision conserves the energy",
        );
        results.push((velocity_of(&bodies, moving), velocity_of(&bodies, resting)));
    }
    let (moving, resting) = results[1];
    assert_close(moving.x, 0.0, 1e-9, "the moving body stopped");
    assert_close(
        resting.x,
        -2.0,
        1e-9,
        "the resting body took over the motion",
    );
    // Switching the monitor on changes nothing at all.
    assert_eq!(results[0], results[1]);
}

// ── Scope: islands, static and kinematic bodies, duplicates ──────────────────

#[test]
fn an_island_solve_measures_and_scales_only_its_own_bodies() {
    let mut bodies = RigidBodySet::new();
    let inside = bodies.insert(dynamic_body(1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros()));
    // A fast body outside the island: its 450 J must neither hide the gain of
    // the island nor be scaled.
    let outside = bodies.insert(dynamic_body(1.0, Vec3::new(30.0, 0.0, 0.0), Vec3::zeros()));
    let mut constraints = injector(&[inside], Vec3::new(0.0, 1.0, 0.0), Vec3::zeros());
    let tgs = solver(true);
    tgs.solve_island(&[inside], &mut constraints, &mut bodies, DT);

    assert_eq!(velocity_of(&bodies, outside), Vec3::new(30.0, 0.0, 0.0));
    let v = velocity_of(&bodies, inside);
    // The island started with 0.5 J and is held at 105 % of it.
    assert_close(
        0.5 * v.norm_squared(),
        THRESHOLD * 0.5,
        1e-9,
        "island energy",
    );
}

#[test]
fn static_and_kinematic_bodies_are_never_scaled() {
    let mut bodies = RigidBodySet::new();
    let dynamic = bodies.insert(dynamic_body(1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros()));
    let mut kinematic = RigidBody::new_kinematic();
    kinematic.velocity = Vec3::new(3.0, 0.0, 0.0);
    kinematic.angular_velocity = Vec3::new(0.0, 0.0, 2.0);
    let kinematic = bodies.insert(kinematic);
    let mut fixed = RigidBody::new_static();
    fixed.velocity = Vec3::new(0.0, 7.0, 0.0);
    let fixed = bodies.insert(fixed);
    let mut constraints = injector(&[dynamic], Vec3::new(0.0, 1.0, 0.0), Vec3::zeros());
    let tgs = solver(true);
    // Only the dynamic body counts towards the energy.
    assert_close(tgs.total_kinetic_energy(&bodies), 0.5, 1e-12, "energy");
    tgs.step(&mut constraints, &mut bodies, DT);
    assert_eq!(velocity_of(&bodies, kinematic), Vec3::new(3.0, 0.0, 0.0));
    assert_eq!(
        angular_velocity_of(&bodies, kinematic),
        Vec3::new(0.0, 0.0, 2.0)
    );
    assert_eq!(velocity_of(&bodies, fixed), Vec3::new(0.0, 7.0, 0.0));
    // The 1 kg body started with 0.5 J and is held at 105 % of it: 0.5 v² = 0.525.
    assert_close(
        velocity_of(&bodies, dynamic).y,
        THRESHOLD.sqrt(),
        1e-9,
        "dynamic body scaled to 105 % of its energy",
    );
}

#[test]
fn a_body_listed_twice_in_an_island_is_integrated_once() {
    let gravity = Vec3::new(0.0, -10.0, 0.0);
    for monitoring in [false, true] {
        let tgs = TgsSolver::with_options(1, 2, 0, gravity, 0.2, monitoring);
        let mut bodies = RigidBodySet::new();
        let handle = bodies.insert(dynamic_body(1.0, Vec3::zeros(), Vec3::zeros()));
        let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();
        tgs.solve_island(
            &[handle, handle, handle],
            &mut constraints,
            &mut bodies,
            0.1,
        );
        assert_close(
            velocity_of(&bodies, handle).y,
            -1.0,
            1e-12,
            "one integration of gravity, not three",
        );
    }
}

// ── Degenerate energies ──────────────────────────────────────────────────────

#[test]
fn energy_created_from_rest_is_removed_entirely() {
    // Nothing moves after the force step (the reference energy is zero), so
    // the 5 % allowance is zero as well: whatever the constraint phase makes up
    // is removed.
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(1.0, Vec3::zeros(), Vec3::zeros()));
    let mut constraints = injector(
        &[handle],
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    solver(true).step(&mut constraints, &mut bodies, DT);
    assert_eq!(velocity_of(&bodies, handle), Vec3::zeros());
    assert_eq!(angular_velocity_of(&bodies, handle), Vec3::zeros());

    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(1.0, Vec3::zeros(), Vec3::zeros()));
    let mut constraints = injector(&[handle], Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
    solver(false).step(&mut constraints, &mut bodies, DT);
    assert_eq!(velocity_of(&bodies, handle), Vec3::new(4.0, 0.0, 0.0));
}

#[test]
fn non_finite_energies_are_left_alone() {
    let mut bodies = RigidBodySet::new();
    let handle = bodies.insert(dynamic_body(
        1.0,
        Vec3::new(f64::NAN, 0.0, 0.0),
        Vec3::zeros(),
    ));
    let mut constraints = injector(&[handle], Vec3::new(0.0, 1.0, 0.0), Vec3::zeros());
    // Must neither panic nor loop; the NaN stays where it is.
    solver(true).step(&mut constraints, &mut bodies, DT);
    assert!(velocity_of(&bodies, handle).x.is_nan());
}

// ── The kinetic energy itself ────────────────────────────────────────────────

#[test]
fn total_kinetic_energy_includes_the_world_frame_rotational_energy() {
    let mut bodies = RigidBodySet::new();
    let mut body = RigidBody::new(2.0);
    body.velocity = Vec3::new(3.0, 4.0, 0.0);
    body.angular_velocity = Vec3::new(1.0, 0.0, 0.0);
    // The inertia about the local x axis is 1, about y it is 2: turning the
    // body by 90° about z makes the world-frame inertia about x equal to 2.
    body.local_inertia = oxiphysics_core::math::Mat3::from_diagonal(&Vec3::new(1.0, 2.0, 3.0));
    body.transform.rotation =
        Quat::from_scaled_axis(Vec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
    body.update_world_inertia();
    bodies.insert(body);
    bodies.insert(RigidBody::new_static());
    let tgs = TgsSolver::default();
    // 0.5 * 2 * 25 + 0.5 * 2 * 1
    assert_close(tgs.total_kinetic_energy(&bodies), 26.0, 1e-12, "energy");
}

#[test]
fn diagnostics_report_the_rotational_energy() {
    let mut bodies = RigidBodySet::new();
    bodies.insert(dynamic_body(1.0, Vec3::zeros(), Vec3::new(0.0, 0.0, 2.0)));
    let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();
    let tgs = TgsSolver::with_options(2, 1, 0, Vec3::zeros(), 0.2, false);
    let (_, diagnostics) = tgs.step_with_diagnostics(&mut constraints, &mut bodies, DT);
    assert_eq!(diagnostics.len(), 2);
    for substep in &diagnostics {
        assert_close(substep.ke_start, 2.0, 1e-12, "ke_start");
        assert_close(substep.ke_end, 2.0, 1e-12, "ke_end");
    }
}
