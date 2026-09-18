// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! A raycast vehicle in free flight and the gyroscopic stage of the chassis integration.
//!
//! [`RaycastVehicle::update`] only accumulates force and torque on the chassis; the caller's loop
//! integrates it. The angular velocity of a [`RigidBody`] is a world-frame quantity, and a chassis
//! (roll, pitch and yaw moments of inertia all different) is not free to keep it constant: in the
//! air, after a jump or a roll-over, it conserves its world-frame angular momentum
//! `L = R I Rᵀ ω`. A loop that runs `integrate_forces` and `integrate_velocity` only makes the
//! *vector* `L` drift by tens of percent in a few seconds even though its magnitude `|L|` is
//! conserved to rounding, so every conservation check below compares the vector, never the
//! magnitude. The loops of this crate run `integrate_gyroscopic` in between; the documentation of
//! [`RaycastVehicle::update`] prescribes that order.

use oxiphysics_core::math::{Mat3, Vec3};
use oxiphysics_rigid::RigidBody;
use oxiphysics_vehicle::{LinearSuspension, LinearTire, RaycastVehicle, flat_ground_query};

const DT: f64 = 0.005;
const STEPS: usize = 1000;
const GRAVITY: f64 = 9.81;

/// Roll, pitch and yaw moments of inertia of a mid-size passenger car (kg m^2).
const CAR_MOMENTS: [f64; 3] = [500.0, 2500.0, 2800.0];

/// A 1.5 t chassis with the inertia tensor of a car, 1 km above the ground, spinning about all
/// three axes at once.
fn airborne_chassis() -> RigidBody {
    let mut chassis = RigidBody::new(1500.0);
    chassis.local_inertia =
        Mat3::from_diagonal(&Vec3::new(CAR_MOMENTS[0], CAR_MOMENTS[1], CAR_MOMENTS[2]));
    chassis.update_world_inertia();
    chassis.transform.position = Vec3::new(0.0, 1000.0, 0.0);
    chassis.angular_velocity = Vec3::new(1.0, 0.3, 0.1);
    chassis.linear_damping = 0.0;
    chassis.angular_damping = 0.0;
    chassis
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

/// One step of the documented chassis loop: `update` accumulates the vehicle forces, then the
/// chassis is integrated. `gyroscopic` selects whether the gyroscopic stage runs in between.
/// Returns the torque `update` accumulated on the chassis (the integration clears it).
fn step(vehicle: &mut RaycastVehicle, chassis: &mut RigidBody, gyroscopic: bool) -> Vec3 {
    let gravity = Vec3::new(0.0, -GRAVITY, 0.0);
    let ground = flat_ground_query(0.0);
    chassis.force_accumulator = Vec3::zeros();
    chassis.torque_accumulator = Vec3::zeros();
    vehicle.update(
        DT,
        chassis,
        &gravity,
        &ground,
        &LinearSuspension,
        &LinearTire::default(),
    );
    let torque = chassis.torque_accumulator;
    chassis.integrate_forces(DT, &gravity);
    if gyroscopic {
        chassis.integrate_gyroscopic(DT);
    }
    chassis.integrate_velocity(DT);
    torque
}

#[test]
fn an_airborne_chassis_conserves_the_angular_momentum_vector() {
    let mut vehicle = RaycastVehicle::new_default_4wheel();
    vehicle.set_inputs(0.0, 0.0, 0.0);
    let mut chassis = airborne_chassis();
    let initial_momentum = world_angular_momentum(&chassis);
    let initial_energy = rotational_energy(&chassis);
    let mut max_vector_drift = 0.0_f64;
    let mut max_energy_ratio = 1.0_f64;
    let mut min_energy_ratio = 1.0_f64;
    let mut max_omega_change = 0.0_f64;
    for _ in 0..STEPS {
        let torque = step(&mut vehicle, &mut chassis, true);
        assert!(
            !vehicle.any_wheel_in_contact(),
            "the chassis must stay in the air"
        );
        // In the air `update` applies only the aerodynamic drag, at the centre of mass: no torque.
        assert_eq!(torque, Vec3::zeros());
        let momentum = world_angular_momentum(&chassis);
        max_vector_drift =
            max_vector_drift.max((momentum - initial_momentum).norm() / initial_momentum.norm());
        let ratio = rotational_energy(&chassis) / initial_energy;
        max_energy_ratio = max_energy_ratio.max(ratio);
        min_energy_ratio = min_energy_ratio.min(ratio);
        max_omega_change =
            max_omega_change.max((chassis.angular_velocity - Vec3::new(1.0, 0.3, 0.1)).norm());
    }
    assert!(
        max_vector_drift < 2.0e-2,
        "the angular momentum vector drifted by {max_vector_drift:.3e}"
    );
    // The implicit step is slightly dissipative, never energy-gaining.
    assert!(
        max_energy_ratio <= 1.0 + 1e-12,
        "the rotational energy grew to {max_energy_ratio} of its initial value"
    );
    assert!(
        min_energy_ratio > 0.98,
        "the rotational energy fell to {min_energy_ratio} of its initial value"
    );
    // The world-frame angular velocity genuinely changed: the chassis tumbles.
    assert!(
        max_omega_change > 0.1,
        "omega never left its initial value (max change {max_omega_change:.3e})"
    );
}

#[test]
fn without_the_gyroscopic_stage_only_the_angular_momentum_vector_drifts() {
    // The pre-fix chassis loop: omega stays constant, |L| is conserved to rounding and the
    // vector L rotates away from its initial direction.
    let mut vehicle = RaycastVehicle::new_default_4wheel();
    vehicle.set_inputs(0.0, 0.0, 0.0);
    let mut chassis = airborne_chassis();
    let initial_momentum = world_angular_momentum(&chassis);
    let mut max_vector_drift = 0.0_f64;
    let mut max_magnitude_drift = 0.0_f64;
    for _ in 0..STEPS {
        step(&mut vehicle, &mut chassis, false);
        assert_eq!(chassis.angular_velocity, Vec3::new(1.0, 0.3, 0.1));
        let momentum = world_angular_momentum(&chassis);
        max_vector_drift =
            max_vector_drift.max((momentum - initial_momentum).norm() / initial_momentum.norm());
        max_magnitude_drift = max_magnitude_drift
            .max((momentum.norm() - initial_momentum.norm()).abs() / initial_momentum.norm());
    }
    assert!(
        max_magnitude_drift < 1e-12,
        "|L| drifts by {max_magnitude_drift:.3e}"
    );
    assert!(
        max_vector_drift > 0.05,
        "L drifts by only {max_vector_drift:.3e}"
    );
}

#[test]
fn an_isotropic_chassis_on_the_ground_is_not_affected_by_the_stage() {
    // The default chassis of `RigidBody::new` is isotropic, and so are the chassis of the other
    // tests of this crate: running the stage in their loops changes no bit of their motion.
    let run = |gyroscopic: bool| {
        let mut vehicle = RaycastVehicle::new_default_4wheel();
        vehicle.set_inputs(1.0, 0.0, 0.3);
        let mut chassis = RigidBody::new(1500.0);
        chassis.transform.position = Vec3::new(0.0, 0.52, 0.0);
        chassis.velocity = Vec3::new(0.0, 0.0, 8.0);
        chassis.angular_velocity = Vec3::new(0.05, 0.2, -0.03);
        chassis.linear_damping = 0.0;
        chassis.angular_damping = 0.0;
        for _ in 0..300 {
            step(&mut vehicle, &mut chassis, gyroscopic);
        }
        assert!(
            vehicle.any_wheel_in_contact(),
            "the car must be on the ground"
        );
        let q = chassis.transform.rotation.quaternion();
        (
            chassis.transform.position,
            chassis.velocity,
            chassis.angular_velocity,
            [q.i, q.j, q.k, q.w],
        )
    };
    assert_eq!(run(true), run(false));
}
