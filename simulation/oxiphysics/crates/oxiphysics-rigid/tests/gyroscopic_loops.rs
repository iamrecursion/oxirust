// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The implicit gyroscopic stage in the single-body and split-step integration loops of the
//! rigid crate: [`RigidBodySet::step_positions`], `Steppable for RigidBody` and
//! [`VariableMassBody::step`].
//!
//! The angular velocity of a [`RigidBody`] is a world-frame quantity. A torque-free body that is
//! not isotropic cannot keep it constant: the conserved quantity is the world-frame angular
//! momentum `L = R I Rᵀ ω`. These loops used to integrate the orientation with the angular
//! velocity of the start of the step (`integrate_forces` followed by `integrate_velocity`),
//! so the *vector* `L` of an asymmetric body drifted by almost 10 % in ten seconds even though
//! its magnitude `|L|` is conserved to rounding. Every conservation check below therefore compares
//! the vector, never the magnitude.

use oxiphysics_core::math::{Mat3, Vec3};
use oxiphysics_core::{Steppable, TimeStep};
use oxiphysics_rigid::{BodyState, RigidBody, RigidBodySet, VariableMassBody};

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

/// Largest relative drift of the angular-momentum vector and the range of the rotational energy
/// of one body over a run.
struct Conservation {
    initial_momentum: Vec3,
    initial_energy: f64,
    max_vector_drift: f64,
    min_energy_ratio: f64,
    max_energy_ratio: f64,
}

impl Conservation {
    fn new(body: &RigidBody) -> Self {
        Self {
            initial_momentum: world_angular_momentum(body),
            initial_energy: rotational_energy(body),
            max_vector_drift: 0.0,
            min_energy_ratio: 1.0,
            max_energy_ratio: 1.0,
        }
    }

    fn record(&mut self, body: &RigidBody) {
        let drift = (world_angular_momentum(body) - self.initial_momentum).norm()
            / self.initial_momentum.norm();
        self.max_vector_drift = self.max_vector_drift.max(drift);
        let ratio = rotational_energy(body) / self.initial_energy;
        self.min_energy_ratio = self.min_energy_ratio.min(ratio);
        self.max_energy_ratio = self.max_energy_ratio.max(ratio);
    }

    /// The first-order backward-Euler step drifts by 4.7e-4 over ten seconds at this step size,
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
// RigidBodySet::step_positions
// ---------------------------------------------------------------------------

/// One frame of the documented split-step sequence on `set`.
fn split_step(set: &mut RigidBodySet) {
    set.apply_gravity([0.0, 0.0, 0.0], DT);
    set.step_velocities(DT);
    set.step_positions(DT);
    set.clear_forces();
}

#[test]
fn step_positions_conserves_the_angular_momentum_vector_of_an_asymmetric_body() {
    let mut set = RigidBodySet::new();
    let handle = set.insert(box_body(initial_spin()));
    let mut conservation = Conservation::new(set.get(handle).expect("inserted body must exist"));
    let mut max_omega_change = 0.0_f64;
    for _ in 0..STEPS {
        split_step(&mut set);
        let body = set.get(handle).expect("inserted body must exist");
        conservation.record(body);
        max_omega_change = max_omega_change.max((body.angular_velocity - initial_spin()).norm());
    }
    conservation.assert_conserved("RigidBodySet::step_positions");
    // The world-frame angular velocity genuinely changed: the body tumbles.
    assert!(
        max_omega_change > 1e-2,
        "omega never left its initial value (max change {max_omega_change:.3e})"
    );
}

#[test]
fn step_positions_without_the_stage_would_drift_by_ten_percent() {
    // The pre-fix loop, written out: it is the reference that makes the test above meaningful.
    let mut body = box_body(initial_spin());
    let mut conservation = Conservation::new(&body);
    for _ in 0..STEPS {
        body.integrate_forces(DT, &Vec3::zeros());
        body.integrate_velocity(DT);
        conservation.record(&body);
    }
    assert_eq!(body.angular_velocity, initial_spin());
    assert!(
        conservation.max_vector_drift > 0.05,
        "vector drift {:.3e}",
        conservation.max_vector_drift
    );
}

#[test]
fn step_positions_leaves_isotropic_bodies_bit_identical() {
    let make = || {
        let mut sphere = RigidBody::new(2.0);
        sphere.angular_velocity = Vec3::new(0.3, -0.2, 0.5);
        sphere.velocity = Vec3::new(1.0, 2.0, -3.0);
        sphere
    };
    let mut with_stage = RigidBodySet::new();
    let a = with_stage.insert(make());
    let mut without_stage = RigidBodySet::new();
    let b = without_stage.insert(make());
    for _ in 0..200 {
        with_stage.step_positions(DT);
        for (_, body) in without_stage.iter_mut() {
            body.integrate_velocity(DT);
        }
    }
    let with_stage = motion_state(with_stage.get(a).expect("body must exist"));
    let without_stage = motion_state(without_stage.get(b).expect("body must exist"));
    assert_eq!(with_stage, without_stage);
    assert_eq!(with_stage.2, Vec3::new(0.3, -0.2, 0.5));
}

#[test]
fn step_positions_corrects_only_dynamic_awake_bodies() {
    let omega = initial_spin();
    let mut fixed = RigidBody::new_static();
    fixed.local_inertia = box_inertia();
    fixed.angular_velocity = omega;
    let mut kinematic = RigidBody::new_kinematic();
    kinematic.local_inertia = box_inertia();
    kinematic.angular_velocity = omega;
    let mut asleep = box_body(omega);
    asleep.state = BodyState::Sleeping;
    let awake = box_body(omega);

    let mut set = RigidBodySet::new();
    let fixed = set.insert(fixed);
    let kinematic = set.insert(kinematic);
    let asleep = set.insert(asleep);
    let awake = set.insert(awake);
    set.step_positions(DT);

    for handle in [fixed, kinematic, asleep] {
        assert_eq!(
            set.get(handle).expect("body must exist").angular_velocity,
            omega,
            "a static, kinematic or sleeping body must keep its angular velocity"
        );
    }
    // A kinematic body still moves with its angular velocity; the other two stand still.
    let turned = |handle| set.get(handle).expect("body must exist").transform.rotation;
    assert_ne!(turned(kinematic), oxiphysics_core::math::Quat::identity());
    assert_eq!(turned(fixed), oxiphysics_core::math::Quat::identity());
    assert_eq!(turned(asleep), oxiphysics_core::math::Quat::identity());
    // An awake dynamic body is corrected: delta omega = -dt I^-1 (omega x I omega), about 6e-4.
    let correction = (set.get(awake).expect("body must exist").angular_velocity - omega).norm();
    assert!(
        (5.0e-4..7.0e-4).contains(&correction),
        "correction {correction:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Steppable for RigidBody
// ---------------------------------------------------------------------------

#[test]
fn steppable_step_conserves_the_angular_momentum_vector_of_an_asymmetric_body() {
    // `Steppable::step` applies standard gravity, which `box_body` switches off per body.
    let mut body = box_body(initial_spin());
    let time_step = TimeStep::new(DT);
    let mut conservation = Conservation::new(&body);
    let mut max_omega_change = 0.0_f64;
    for _ in 0..STEPS {
        body.step(&time_step);
        conservation.record(&body);
        max_omega_change = max_omega_change.max((body.angular_velocity - initial_spin()).norm());
    }
    conservation.assert_conserved("Steppable::step");
    assert!(
        max_omega_change > 1e-2,
        "omega never left its initial value (max change {max_omega_change:.3e})"
    );
    assert_eq!(body.transform.position, Vec3::zeros());
}

#[test]
fn steppable_step_leaves_isotropic_bodies_bit_identical() {
    let make = || {
        let mut sphere = RigidBody::new(2.0);
        sphere.angular_velocity = Vec3::new(0.3, -0.2, 0.5);
        sphere.transform.position = Vec3::new(0.0, 10.0, 0.0);
        sphere
    };
    let mut stepped = make();
    let mut manual = make();
    let time_step = TimeStep::new(DT);
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    for _ in 0..200 {
        stepped.step(&time_step);
        manual.integrate_forces(DT, &gravity);
        manual.integrate_velocity(DT);
    }
    assert_eq!(motion_state(&stepped), motion_state(&manual));
    assert_eq!(stepped.angular_velocity, Vec3::new(0.3, -0.2, 0.5));
}

// ---------------------------------------------------------------------------
// VariableMassBody::step
// ---------------------------------------------------------------------------

/// A variable-mass body around an asymmetric box. Its mass changes during a burn but its inertia
/// tensor does not (`VariableMassBody` only updates the mass), so the angular momentum of the
/// tensor in use is still conserved by the thrust, which acts at the centre of mass.
fn variable_mass_box(throttle: f64) -> VariableMassBody {
    let mut vehicle = VariableMassBody::new(1.0, 0.5, 300.0, 10.0, [0.0, 0.0, 1.0]);
    vehicle.throttle = throttle;
    vehicle.body = box_body(initial_spin());
    vehicle
}

#[test]
fn variable_mass_body_step_conserves_the_angular_momentum_vector() {
    for throttle in [0.0, 1.0] {
        let mut vehicle = variable_mass_box(throttle);
        let mut conservation = Conservation::new(&vehicle.body);
        let mut max_omega_change = 0.0_f64;
        for _ in 0..STEPS {
            vehicle.step(DT, &[0.0, 0.0, 0.0]);
            conservation.record(&vehicle.body);
            max_omega_change =
                max_omega_change.max((vehicle.body.angular_velocity - initial_spin()).norm());
        }
        conservation.assert_conserved(&format!("VariableMassBody::step at throttle {throttle}"));
        assert!(
            max_omega_change > 1e-2,
            "omega never left its initial value at throttle {throttle}"
        );
        if throttle > 0.0 {
            assert!(
                vehicle.consumed_mass > 0.0,
                "the burn must have consumed propellant"
            );
        }
    }
}

#[test]
fn variable_mass_body_step_leaves_isotropic_bodies_bit_identical() {
    let spin = Vec3::new(0.3, -0.2, 0.5);
    let mut vehicle = VariableMassBody::new(2.0, 0.5, 300.0, 10.0, [0.0, 1.0, 0.0]);
    vehicle.throttle = 1.0;
    vehicle.body.angular_velocity = spin;
    // Thrust and gravity only move the centre of mass, so the orientation of an isotropic body
    // is that of the bare orientation integrator at constant angular velocity.
    let mut reference = RigidBody::new(2.0);
    reference.angular_velocity = spin;
    for _ in 0..200 {
        vehicle.step(DT, &[0.0, -9.81, 0.0]);
        reference.integrate_velocity(DT);
    }
    assert!(
        vehicle.consumed_mass > 0.0,
        "the burn must have consumed propellant"
    );
    assert_eq!(vehicle.body.angular_velocity, spin);
    assert_eq!(
        vehicle.body.transform.rotation,
        reference.transform.rotation
    );
}
