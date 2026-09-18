// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the constraints of a world: distance, ball-socket, hinge and slider
//! constraints between the bodies of a [`PyPhysicsWorld`], solved by the
//! engine's joints.
//!
//! The reference values are analytic: the length of a pendulum rod, its small
//! angle period, the conserved momenta of two jointed free bodies, the plane of
//! a hinge and the line of a slider.

use super::*;
use crate::types::{PyRigidBodyConfig, PySimConfig};
use std::f64::consts::PI;

/// Time step of the tests (s).
const DT: f64 = 1.0 / 240.0;

/// Gravity of the tests (m/s²).
const GRAVITY: f64 = 9.81;

/// A configuration whose bodies never fall asleep.
fn awake(config: PySimConfig) -> PySimConfig {
    PySimConfig {
        sleep_enabled: false,
        ..config
    }
}

/// A world with Earth gravity whose bodies never fall asleep.
fn gravity_world() -> PyPhysicsWorld {
    PyPhysicsWorld::new(awake(PySimConfig::earth_gravity()))
}

/// A world without gravity whose bodies never fall asleep.
fn space_world() -> PyPhysicsWorld {
    PyPhysicsWorld::new(awake(PySimConfig::zero_gravity()))
}

/// A dynamic body of the given mass at `position`.
fn dynamic(world: &mut PyPhysicsWorld, mass: f64, position: [f64; 3]) -> u32 {
    world.add_rigid_body(PyRigidBodyConfig::dynamic(mass, position))
}

/// A static body at `position`.
fn fixed(world: &mut PyPhysicsWorld, position: [f64; 3]) -> u32 {
    world.add_rigid_body(PyRigidBodyConfig::static_body(position))
}

fn position(world: &PyPhysicsWorld, body: u32) -> [f64; 3] {
    world.get_position(body).expect("the body exists")
}

fn velocity(world: &PyPhysicsWorld, body: u32) -> [f64; 3] {
    world.get_velocity(body).expect("the body exists")
}

fn angular_velocity(world: &PyPhysicsWorld, body: u32) -> [f64; 3] {
    world.get_angular_velocity(body).expect("the body exists")
}

fn orientation(world: &PyPhysicsWorld, body: u32) -> [f64; 4] {
    world.get_orientation(body).expect("the body exists")
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The constraint of the world with the given handle.
fn stored(world: &PyPhysicsWorld, handle: u32) -> PyConstraint {
    world
        .constraints
        .iter()
        .find(|c| c.handle == handle)
        .cloned()
        .expect("the constraint is in the world")
}

/// World position of the point `local` of `body`.
fn world_point(world: &PyPhysicsWorld, body: u32, local: [f64; 3]) -> [f64; 3] {
    let rotation = quat_from_xyzw(orientation(world, body));
    let rotated = rotation * Vec3::from(local);
    let p = position(world, body);
    [p[0] + rotated.x, p[1] + rotated.y, p[2] + rotated.z]
}

/// Distance of the two anchors of the stored constraint.
fn anchor_gap(world: &PyPhysicsWorld, handle: u32) -> f64 {
    let c = stored(world, handle);
    norm(sub(
        world_point(world, c.body_a, c.anchor_a),
        world_point(world, c.body_b, c.anchor_b),
    ))
}

/// Rotation angle of a quaternion `[x, y, z, w]` about the z axis, assuming
/// that it is a rotation about z.
fn angle_about_z(q: [f64; 4]) -> f64 {
    2.0 * q[2].atan2(q[3])
}

// ---------------------------------------------------------------------------
// Distance constraints: the pendulum
// ---------------------------------------------------------------------------

#[test]
fn test_distance_pendulum_keeps_its_length_and_stays_below_the_free_fall_speed() {
    let mut world = gravity_world();
    let pivot = fixed(&mut world, [0.0; 3]);
    let bob = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    let handle = world
        .add_constraint(PyConstraint::distance(
            0, pivot, bob, [0.0; 3], [0.0; 3], 1.0,
        ))
        .expect("both bodies exist");
    assert_eq!(handle, 0);

    // Released from the horizontal, the bob cannot be faster than a body that
    // fell the same height, sqrt(2 g L).
    let speed_limit = (2.0 * GRAVITY).sqrt();
    let mut worst_length_error = 0.0_f64;
    let mut worst_rod_speed = 0.0_f64;
    let mut fastest = 0.0_f64;
    let mut lowest = f64::INFINITY;
    for _ in 0..1000 {
        world.step(DT);
        let p = position(&world, bob);
        let v = velocity(&world, bob);
        let length = norm(p);
        worst_length_error = worst_length_error.max((length - 1.0).abs());
        // The rod removes the velocity along it.
        worst_rod_speed = worst_rod_speed.max((dot(v, p) / length).abs());
        fastest = fastest.max(norm(v));
        lowest = lowest.min(p[1]);
    }
    assert!(
        worst_length_error < 1.0e-3,
        "the rod length drifted by {worst_length_error}"
    );
    assert!(
        worst_rod_speed < 5.0e-3,
        "the bob moves along the rod at {worst_rod_speed} m/s"
    );
    assert!(
        fastest <= speed_limit,
        "the bob reached {fastest} m/s, more than sqrt(2 g L) = {speed_limit}"
    );
    assert!(
        fastest > 0.9 * speed_limit,
        "the bob only reached {fastest} m/s of {speed_limit}"
    );
    assert!(
        lowest < -0.99,
        "the bob did not swing down to the bottom: lowest y = {lowest}"
    );
}

#[test]
fn test_distance_pendulum_swings_with_the_small_angle_period() {
    let amplitude = 0.05_f64;
    let mut world = gravity_world();
    let pivot = fixed(&mut world, [0.0; 3]);
    let bob = dynamic(&mut world, 1.0, [amplitude.sin(), -amplitude.cos(), 0.0]);
    world
        .add_constraint(PyConstraint::distance(
            0, pivot, bob, [0.0; 3], [0.0; 3], 1.0,
        ))
        .expect("both bodies exist");

    // The bob passes the vertical after a quarter period; the finite amplitude
    // lengthens the period by the factor 1 + amplitude^2 / 16.
    let quarter_period =
        0.25 * 2.0 * PI * (1.0 / GRAVITY).sqrt() * (1.0 + amplitude * amplitude / 16.0);
    let mut crossing = None;
    let mut previous_x = position(&world, bob)[0];
    for step in 1..=480 {
        world.step(DT);
        let x = position(&world, bob)[0];
        if previous_x > 0.0 && x <= 0.0 {
            let fraction = previous_x / (previous_x - x);
            crossing = Some((step as f64 - 1.0 + fraction) * DT);
            break;
        }
        previous_x = x;
    }
    let crossing = crossing.expect("the bob passes the vertical within two seconds");
    assert!(
        (crossing - quarter_period).abs() < 0.01 * quarter_period,
        "quarter period {crossing} s instead of {quarter_period} s"
    );
}

#[test]
fn test_distance_constraint_between_two_free_bodies_keeps_the_anchor_distance() {
    let mut world = space_world();
    let a = dynamic(&mut world, 1.0, [0.0; 3]);
    let b = dynamic(&mut world, 3.0, [2.0, 0.0, 0.0]);
    // The anchors are offset from the centres: 0.5 m up on A, 0.5 m down on B,
    // which are sqrt(2^2 + 1^2) apart.
    let target = 5.0_f64.sqrt();
    let handle = world
        .add_constraint(PyConstraint::distance(
            0,
            a,
            b,
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            target,
        ))
        .expect("both bodies exist");
    // Pull the bodies apart and spin one of them.
    world.set_velocity(a, [-1.0, 0.0, 0.0]);
    world.set_velocity(b, [0.5, 0.0, 0.0]);
    world.set_angular_velocity(b, [0.0, 0.0, 2.0]);
    let mut worst = 0.0_f64;
    for step in 0..960 {
        world.step(DT);
        if step >= 240 {
            let c = stored(&world, handle);
            let gap = norm(sub(
                world_point(&world, c.body_a, c.anchor_a),
                world_point(&world, c.body_b, c.anchor_b),
            ));
            worst = worst.max((gap - target).abs());
        }
    }
    assert!(
        worst < 2.0e-3,
        "the anchor distance drifted by {worst} from {target}"
    );
}

// ---------------------------------------------------------------------------
// Point-to-point constraints
// ---------------------------------------------------------------------------

#[test]
fn test_point_to_point_pivot_becomes_an_anchor_in_each_body_frame() {
    let mut world = space_world();
    // Body A is turned by 90 degrees about z.
    let half_root = std::f64::consts::FRAC_1_SQRT_2;
    let a = world.add_rigid_body({
        let mut config = PyRigidBodyConfig::dynamic(1.0, [0.0; 3]);
        config.orientation = [0.0, 0.0, half_root, half_root];
        config
    });
    let b = dynamic(&mut world, 1.0, [2.0, 0.0, 0.0]);
    let handle = world
        .add_constraint(PyConstraint::point_to_point(7, a, b, [1.0, 0.0, 0.0]))
        .expect("both bodies exist");
    let c = stored(&world, handle);
    // The pivot (1, 0, 0) is at (0, -1, 0) in the frame of A and (-1, 0, 0) in
    // the frame of B.
    for (actual, expected) in c.anchor_a.iter().zip([0.0, -1.0, 0.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "anchor_a {:?}",
            c.anchor_a
        );
    }
    for (actual, expected) in c.anchor_b.iter().zip([-1.0, 0.0, 0.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "anchor_b {:?}",
            c.anchor_b
        );
    }
    // The handle given to the builder is replaced by the world's.
    assert_eq!(c.handle, handle);
    assert_eq!(handle, 0);
    // The constraint is satisfied at once.
    assert!(anchor_gap(&world, handle) < 1.0e-12);
    // B relative to A is a rotation by -90 degrees about z.
    let frame = c.frame.expect("the world stored the frames");
    for (actual, expected) in frame
        .reference_rotation
        .iter()
        .zip([0.0, 0.0, -half_root, half_root])
    {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{:?}",
            frame.reference_rotation
        );
    }
}

#[test]
fn test_point_to_point_keeps_the_offset_anchors_together_and_conserves_momentum() {
    let mut world = space_world();
    let a = dynamic(&mut world, 1.0, [0.0; 3]);
    let b = dynamic(&mut world, 2.0, [2.0, 0.0, 0.0]);
    world.set_velocity(a, [0.0, 1.0, 0.0]);
    world.set_velocity(b, [0.0, -0.5, 0.0]);
    let handle = world
        .add_constraint(PyConstraint::point_to_point(0, a, b, [1.0, 0.0, 0.0]))
        .expect("both bodies exist");

    // Fallback inertia of a body without a collider: 0.4 m (see the world's
    // `approximate_inertia`).
    let (mass_a, mass_b) = (1.0, 2.0);
    let (inertia_a, inertia_b) = (0.4 * mass_a, 0.4 * mass_b);
    let momentum = |world: &PyPhysicsWorld| {
        let (va, vb) = (velocity(world, a), velocity(world, b));
        [
            mass_a * va[0] + mass_b * vb[0],
            mass_a * va[1] + mass_b * vb[1],
            mass_a * va[2] + mass_b * vb[2],
        ]
    };
    let angular_momentum = |world: &PyPhysicsWorld| {
        let orbital_a = cross(position(world, a), velocity(world, a));
        let orbital_b = cross(position(world, b), velocity(world, b));
        let (wa, wb) = (angular_velocity(world, a), angular_velocity(world, b));
        [
            mass_a * orbital_a[0] + mass_b * orbital_b[0] + inertia_a * wa[0] + inertia_b * wb[0],
            mass_a * orbital_a[1] + mass_b * orbital_b[1] + inertia_a * wa[1] + inertia_b * wb[1],
            mass_a * orbital_a[2] + mass_b * orbital_b[2] + inertia_a * wa[2] + inertia_b * wb[2],
        ]
    };
    let momentum_before = momentum(&world);
    let angular_before = angular_momentum(&world);
    assert!(norm(angular_before) > 1.0, "L = {angular_before:?}");

    let mut worst_gap = 0.0_f64;
    for step in 0..480 {
        world.step(DT);
        if step >= 120 {
            worst_gap = worst_gap.max(anchor_gap(&world, handle));
        }
    }
    assert!(worst_gap < 1.0e-3, "the pivot came apart by {worst_gap}");
    let momentum_after = momentum(&world);
    assert!(
        norm(sub(momentum_after, momentum_before)) < 1.0e-9,
        "momentum {momentum_before:?} -> {momentum_after:?}"
    );
    let angular_after = angular_momentum(&world);
    assert!(
        norm(sub(angular_after, angular_before)) < 0.02 * norm(angular_before),
        "angular momentum {angular_before:?} -> {angular_after:?}"
    );
    // The bodies swing around the pivot: they are rotating.
    assert!(norm(angular_velocity(&world, a)) > 0.1);
    assert!(norm(angular_velocity(&world, b)) > 0.1);
}

#[test]
fn test_distance_chain_shares_its_bodies_between_constraints() {
    let mut world = gravity_world();
    let top = fixed(&mut world, [0.0; 3]);
    let first = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    let second = dynamic(&mut world, 1.0, [2.0, 0.0, 0.0]);
    let third = dynamic(&mut world, 1.0, [3.0, 0.0, 0.0]);
    let links = [
        world
            .add_constraint(PyConstraint::distance(
                0, top, first, [0.0; 3], [0.0; 3], 1.0,
            ))
            .expect("link"),
        world
            .add_constraint(PyConstraint::distance(
                0, first, second, [0.0; 3], [0.0; 3], 1.0,
            ))
            .expect("link"),
        world
            .add_constraint(PyConstraint::distance(
                0, second, third, [0.0; 3], [0.0; 3], 1.0,
            ))
            .expect("link"),
    ];
    assert_eq!(links, [0, 1, 2]);
    let ends = [(top, first), (first, second), (second, third)];
    let mut worst = 0.0_f64;
    let mut lowest = f64::INFINITY;
    for _ in 0..720 {
        world.step(DT);
        for (upper, lower) in ends {
            let link = norm(sub(position(&world, lower), position(&world, upper)));
            worst = worst.max((link - 1.0).abs());
        }
        lowest = lowest.min(position(&world, third)[1]);
    }
    // The chain falls like a whip from the horizontal; its links stay rigid.
    assert!(lowest < -2.0, "the free end only fell to y = {lowest}");
    assert!(worst < 3.0e-2, "a link changed its length by {worst}");
}

// ---------------------------------------------------------------------------
// Hinge constraints
// ---------------------------------------------------------------------------

#[test]
fn test_hinge_swings_in_the_plane_perpendicular_to_its_axis() {
    let mut world = gravity_world();
    let wall = fixed(&mut world, [0.0; 3]);
    let door = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::hinge(
            0,
            wall,
            door,
            [0.0; 3],
            [0.0, 0.0, 1.0],
        ))
        .expect("both bodies exist");

    let mut worst_out_of_plane = 0.0_f64;
    let mut worst_off_axis_spin = 0.0_f64;
    let mut worst_tilt = 0.0_f64;
    let mut worst_length_error = 0.0_f64;
    let mut worst_angle_mismatch = 0.0_f64;
    let mut fastest_spin = 0.0_f64;
    for _ in 0..1000 {
        world.step(DT);
        let p = position(&world, door);
        let w = angular_velocity(&world, door);
        let q = orientation(&world, door);
        worst_out_of_plane = worst_out_of_plane.max(p[2].abs());
        worst_off_axis_spin = worst_off_axis_spin.max(w[0].abs()).max(w[1].abs());
        worst_tilt = worst_tilt.max(q[0].abs()).max(q[1].abs());
        worst_length_error = worst_length_error.max((p[0].hypot(p[1]) - 1.0).abs());
        // A body hinged about the origin turns by the polar angle of its centre.
        let mismatch = angle_about_z(q) - p[1].atan2(p[0]);
        worst_angle_mismatch = worst_angle_mismatch.max(mismatch.sin().abs());
        fastest_spin = fastest_spin.max(w[2].abs());
    }
    assert!(worst_out_of_plane < 1.0e-9, "z = {worst_out_of_plane}");
    assert!(worst_off_axis_spin < 1.0e-9, "spin {worst_off_axis_spin}");
    assert!(worst_tilt < 1.0e-9, "tilt {worst_tilt}");
    assert!(worst_length_error < 1.0e-3, "radius {worst_length_error}");
    assert!(
        worst_angle_mismatch < 5.0e-3,
        "the body does not turn with its centre: {worst_angle_mismatch}"
    );
    // Released from the horizontal, a body of inertia 0.4 about its centre spins
    // at sqrt(2 m g L / (m L^2 + I)) = 3.74 rad/s at the bottom.
    assert!(fastest_spin > 3.3, "spin only reached {fastest_spin} rad/s");
    assert!(fastest_spin < 3.8, "spin reached {fastest_spin} rad/s");
}

#[test]
fn test_hinge_swings_with_the_period_of_a_physical_pendulum() {
    let amplitude = 0.05_f64;
    let mut world = gravity_world();
    let wall = fixed(&mut world, [0.0; 3]);
    let door = dynamic(&mut world, 1.0, [amplitude.sin(), -amplitude.cos(), 0.0]);
    world
        .add_constraint(PyConstraint::hinge(
            0,
            wall,
            door,
            [0.0; 3],
            [0.0, 0.0, 1.0],
        ))
        .expect("both bodies exist");

    // I + m L^2 = 0.4 + 1: the rotational inertia of the door about its centre
    // lengthens the period of a point mass by sqrt(1.4).
    let period = 2.0 * PI * (1.4 / GRAVITY).sqrt() * (1.0 + amplitude * amplitude / 16.0);
    let quarter_period = 0.25 * period;
    let mut crossing = None;
    let mut previous_x = position(&world, door)[0];
    for step in 1..=720 {
        world.step(DT);
        let x = position(&world, door)[0];
        if previous_x > 0.0 && x <= 0.0 {
            let fraction = previous_x / (previous_x - x);
            crossing = Some((step as f64 - 1.0 + fraction) * DT);
            break;
        }
        previous_x = x;
    }
    let crossing = crossing.expect("the door passes the vertical within three seconds");
    assert!(
        (crossing - quarter_period).abs() < 0.015 * quarter_period,
        "quarter period {crossing} s instead of {quarter_period} s"
    );
}

#[test]
fn test_hinge_axis_is_a_world_direction_even_for_a_turned_body() {
    let mut world = gravity_world();
    // The wall is turned by 90 degrees about y: its local z axis is the world x axis.
    let half_root = std::f64::consts::FRAC_1_SQRT_2;
    let wall = world.add_rigid_body({
        let mut config = PyRigidBodyConfig::static_body([0.0; 3]);
        config.orientation = [0.0, half_root, 0.0, half_root];
        config
    });
    let door = dynamic(&mut world, 1.0, [0.0, 0.0, 1.0]);
    let handle = world
        .add_constraint(PyConstraint::hinge(
            0,
            wall,
            door,
            [0.0; 3],
            [1.0, 0.0, 0.0],
        ))
        .expect("both bodies exist");
    // The wall is turned by 90 degrees about y: the world x axis is its local z axis.
    let frame = stored(&world, handle).frame.expect("frames");
    for (actual, expected) in frame.local_axis.iter().zip([0.0, 0.0, 1.0]) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{:?}",
            frame.local_axis
        );
    }
    // The door swings about the world x axis: its x coordinate stays zero.
    let mut lowest = f64::INFINITY;
    for _ in 0..600 {
        world.step(DT);
        let p = position(&world, door);
        assert!(p[0].abs() < 1.0e-9, "left the plane x = 0: {p:?}");
        assert!((p[1].hypot(p[2]) - 1.0).abs() < 1.0e-3, "radius at {p:?}");
        lowest = lowest.min(p[1]);
    }
    assert!(lowest < -0.99, "the door did not swing down: y = {lowest}");
}

// ---------------------------------------------------------------------------
// Slider constraints
// ---------------------------------------------------------------------------

#[test]
fn test_slider_moves_along_its_axis_only_and_does_not_rotate() {
    let mut world = gravity_world();
    let rail = fixed(&mut world, [0.0; 3]);
    let carriage = dynamic(&mut world, 2.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(
            PyConstraint::slider(0, rail, carriage, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                .with_stiffness(1.0),
        )
        .expect("both bodies exist");
    // Push it along the rail: 3 N s on 2 kg is 1.5 m/s.
    world.apply_impulse(carriage, [3.0, 0.0, 0.0], None);

    let mut worst_off_rail = 0.0_f64;
    let mut worst_spin = 0.0_f64;
    let mut worst_rotation = 0.0_f64;
    for _ in 0..240 {
        world.step(DT);
        let p = position(&world, carriage);
        worst_off_rail = worst_off_rail.max(p[1].abs()).max(p[2].abs());
        worst_spin = worst_spin.max(norm(angular_velocity(&world, carriage)));
        let q = orientation(&world, carriage);
        worst_rotation = worst_rotation.max(norm([q[0], q[1], q[2]]));
    }
    let p = position(&world, carriage);
    let v = velocity(&world, carriage);
    // Gravity is carried by the rail; the push is not resisted.
    assert!(worst_off_rail < 1.0e-3, "left the rail by {worst_off_rail}");
    assert!((p[0] - 2.5).abs() < 1.0e-3, "x = {} after one second", p[0]);
    assert!((v[0] - 1.5).abs() < 1.0e-9, "vx = {}", v[0]);
    assert!(v[1].abs() < 1.0e-9 && v[2].abs() < 1.0e-9, "v = {v:?}");
    assert!(worst_spin < 1.0e-12, "spin {worst_spin}");
    assert!(worst_rotation < 1.0e-12, "rotation {worst_rotation}");
}

#[test]
fn test_slider_axis_is_a_world_direction_not_a_body_direction() {
    let mut world = space_world();
    let rail = fixed(&mut world, [0.0; 3]);
    let carriage = dynamic(&mut world, 1.0, [0.0, 1.0, 1.0]);
    // A diagonal rail through the carriage.
    let direction = [0.0, 1.0, 1.0];
    world
        .add_constraint(
            PyConstraint::slider(0, rail, carriage, [0.0, 1.0, 1.0], direction).with_stiffness(1.0),
        )
        .expect("both bodies exist");
    // A push across the rail is absorbed, a push along it is not.
    world.set_velocity(carriage, [2.0, 1.0, -1.0]);
    for _ in 0..24 {
        world.step(DT);
    }
    let v = velocity(&world, carriage);
    assert!(v[0].abs() < 1.0e-9, "across the rail: {v:?}");
    assert!((v[1] + v[2]).abs() < 1.0e-9, "across the rail: {v:?}");
    // The component along the rail (0, 1, 1) / sqrt 2 is zero here too, so
    // check a velocity along it separately.
    world.set_velocity(carriage, [0.0, 1.0, 1.0]);
    let before = position(&world, carriage);
    for _ in 0..24 {
        world.step(DT);
    }
    let v = velocity(&world, carriage);
    assert!(
        (v[1] - 1.0).abs() < 1.0e-9 && (v[2] - 1.0).abs() < 1.0e-9,
        "{v:?}"
    );
    let moved = sub(position(&world, carriage), before);
    assert!(
        (moved[1] - 0.1).abs() < 1.0e-3 && (moved[2] - 0.1).abs() < 1.0e-3,
        "{moved:?}"
    );
    assert!(moved[0].abs() < 1.0e-9);
}

#[test]
fn test_slider_removes_the_angular_velocity_a_torque_creates() {
    let mut world = space_world();
    let rail = fixed(&mut world, [0.0; 3]);
    let carriage = dynamic(&mut world, 2.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(
            PyConstraint::slider(0, rail, carriage, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                .with_stiffness(1.0),
        )
        .expect("both bodies exist");
    let mut worst_spin = 0.0_f64;
    for _ in 0..100 {
        world.apply_torque(carriage, [0.0, 0.0, 1.0]);
        world.step(DT);
        worst_spin = worst_spin.max(norm(angular_velocity(&world, carriage)));
    }
    assert!(worst_spin < 1.0e-12, "spin {worst_spin}");
    // The lock acts on the velocities: the torque turns the carriage a little
    // within each step before the spin is removed (tau dt^2 / I per step,
    // 100 steps of 1 / 0.8 / 240^2 rad = 2.2e-3 rad).
    let angle = angle_about_z(orientation(&world, carriage)).abs();
    assert!(angle < 3.0e-3, "residual rotation {angle}");
}

// ---------------------------------------------------------------------------
// Stiffness, enabling, sleeping, validation
// ---------------------------------------------------------------------------

/// One step of a body moving at 2 m/s that is tied to a static body at its own
/// position; returns its velocity.
fn velocity_after_one_step(stiffness: f64, iterations: u32, enabled: bool) -> f64 {
    let mut world = PyPhysicsWorld::new(PySimConfig {
        solver_iterations: iterations,
        ..awake(PySimConfig::zero_gravity())
    });
    let post = fixed(&mut world, [0.0; 3]);
    let body = dynamic(&mut world, 1.0, [0.0; 3]);
    world.set_velocity(body, [2.0, 0.0, 0.0]);
    let mut constraint =
        PyConstraint::point_to_point(0, post, body, [0.0; 3]).with_stiffness(stiffness);
    constraint.set_enabled(enabled);
    world.add_constraint(constraint).expect("both bodies exist");
    world.step(DT);
    velocity(&world, body)[0]
}

#[test]
fn test_stiffness_is_the_relaxation_of_every_velocity_sweep() {
    // One sweep removes the fraction `stiffness` of the violating velocity, so
    // `solver_iterations` sweeps leave (1 - stiffness)^iterations of it.
    assert!(velocity_after_one_step(1.0, 1, true).abs() < 1.0e-12);
    assert!((velocity_after_one_step(0.5, 1, true) - 1.0).abs() < 1.0e-12);
    assert!((velocity_after_one_step(0.25, 1, true) - 1.5).abs() < 1.0e-12);
    assert!((velocity_after_one_step(0.5, 3, true) - 0.25).abs() < 1.0e-12);
    assert!((velocity_after_one_step(0.1, 4, true) - 2.0 * 0.9_f64.powi(4)).abs() < 1.0e-12);
}

#[test]
fn test_disabled_and_zero_stiffness_constraints_do_nothing() {
    assert!((velocity_after_one_step(1.0, 8, false) - 2.0).abs() < 1.0e-15);
    assert!((velocity_after_one_step(0.0, 8, true) - 2.0).abs() < 1.0e-15);
    // Re-enabling restores the constraint.
    let mut world = space_world();
    let post = fixed(&mut world, [0.0; 3]);
    let body = dynamic(&mut world, 1.0, [0.0; 3]);
    world.set_velocity(body, [2.0, 0.0, 0.0]);
    let mut constraint = PyConstraint::point_to_point(0, post, body, [0.0; 3]);
    constraint.set_enabled(false);
    world.add_constraint(constraint).expect("both bodies exist");
    world.step(DT);
    assert!((velocity(&world, body)[0] - 2.0).abs() < 1.0e-15);
    world.constraints[0].set_enabled(true);
    for _ in 0..10 {
        world.step(DT);
    }
    assert!(velocity(&world, body)[0].abs() < 1.0e-2);
}

#[test]
fn test_a_sleeping_body_wakes_when_its_partner_pulls_it() {
    // The default configuration lets bodies sleep.
    let mut world = PyPhysicsWorld::new(PySimConfig::zero_gravity());
    let sleeper = dynamic(&mut world, 1.0, [0.0; 3]);
    let mover = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(
            PyConstraint::distance(0, sleeper, mover, [0.0; 3], [0.0; 3], 1.0).with_stiffness(1.0),
        )
        .expect("both bodies exist");
    world.sleep_body(sleeper);
    assert!(world.is_sleeping(sleeper));

    // The mover leaves along the rod: it drags the sleeper along, and as the
    // masses are equal they share the speed.
    world.set_velocity(mover, [1.0, 0.0, 0.0]);
    world.step(DT);
    assert!(!world.is_sleeping(sleeper));
    assert!((velocity(&world, sleeper)[0] - 0.5).abs() < 1.0e-9);
    assert!((velocity(&world, mover)[0] - 0.5).abs() < 1.0e-9);
}

#[test]
fn test_waking_follows_the_whole_chain_of_sleeping_bodies() {
    let mut world = PyPhysicsWorld::new(PySimConfig::zero_gravity());
    let first = dynamic(&mut world, 1.0, [0.0; 3]);
    let second = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    let mover = dynamic(&mut world, 1.0, [2.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::point_to_point(
            0,
            first,
            second,
            [0.5, 0.0, 0.0],
        ))
        .expect("both bodies exist");
    world
        .add_constraint(PyConstraint::point_to_point(
            0,
            second,
            mover,
            [1.5, 0.0, 0.0],
        ))
        .expect("both bodies exist");
    world.sleep_body(first);
    world.sleep_body(second);
    world.set_velocity(mover, [0.0, 1.0, 0.0]);
    world.step(DT);
    // The mover wakes the second body, which in turn wakes the first one, and
    // both take part in the same step.
    assert!(!world.is_sleeping(second));
    assert!(!world.is_sleeping(first));
    assert!(
        velocity(&world, second)[1] > 0.05,
        "{:?}",
        velocity(&world, second)
    );
    assert!(
        velocity(&world, first)[1] > 0.0,
        "{:?}",
        velocity(&world, first)
    );
}

#[test]
fn test_a_kinematic_carrier_wakes_and_drags_a_sleeping_body() {
    let mut world = PyPhysicsWorld::new(PySimConfig::zero_gravity());
    let carrier = world.add_rigid_body({
        let mut config = PyRigidBodyConfig::dynamic(1.0, [0.0; 3]);
        config.is_kinematic = true;
        config
    });
    let load = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    let handle = world
        .add_constraint(
            PyConstraint::point_to_point(0, carrier, load, [0.5, 0.0, 0.0]).with_stiffness(1.0),
        )
        .expect("both bodies exist");
    world.sleep_body(load);
    // The carrier is moved from outside, 1 m/s along y.
    world.set_velocity(carrier, [0.0, 1.0, 0.0]);
    for step in 0..240 {
        world.set_position(carrier, [0.0, (step + 1) as f64 * DT, 0.0]);
        world.step(DT);
        assert!(
            !world.is_sleeping(load),
            "the load fell asleep at step {step}"
        );
    }
    // The pivot follows the carrier: the load has been carried up by about 1 m
    // (its centre is offset from the pivot, so it may also have turned).
    assert!(anchor_gap(&world, handle) < 2.0e-3);
    let carried = position(&world, load)[1];
    assert!(
        carried > 0.5 && carried < 1.5,
        "the load is at y = {carried}"
    );
    assert!((position(&world, carrier)[1] - 1.0).abs() < 1.0e-12);
}

#[test]
fn test_a_constraint_between_sleeping_or_immovable_bodies_is_not_solved() {
    let mut world = PyPhysicsWorld::new(PySimConfig::zero_gravity());
    let post = fixed(&mut world, [0.0; 3]);
    let sleeper = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::distance(
            0, post, sleeper, [0.0; 3], [0.0; 3], 1.0,
        ))
        .expect("both bodies exist");
    // A second constraint joins two static posts.
    let other_post = fixed(&mut world, [5.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::distance(
            0, post, other_post, [0.0; 3], [0.0; 3], 5.0,
        ))
        .expect("both bodies exist");
    world.sleep_body(sleeper);
    for _ in 0..10 {
        world.step(DT);
    }
    assert!(world.is_sleeping(sleeper));
    assert_eq!(position(&world, sleeper), [1.0, 0.0, 0.0]);
    assert_eq!(velocity(&world, sleeper), [0.0; 3]);
}

#[test]
fn test_a_constraint_does_not_act_on_a_body_that_reuses_a_removed_slot() {
    let mut world = space_world();
    let post = fixed(&mut world, [0.0; 3]);
    let first = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::distance(
            0, post, first, [0.0; 3], [0.0; 3], 1.0,
        ))
        .expect("both bodies exist");
    assert!(world.remove_body(first));
    // The slot of the removed body is reused by a stranger.
    let stranger = dynamic(&mut world, 1.0, [5.0, 0.0, 0.0]);
    assert_eq!(stranger, first);
    world.set_velocity(stranger, [1.0, 0.0, 0.0]);
    for _ in 0..24 {
        world.step(DT);
    }
    let p = position(&world, stranger);
    assert!((p[0] - (5.0 + 24.0 * DT)).abs() < 1.0e-12, "x = {}", p[0]);
    assert_eq!(velocity(&world, stranger), [1.0, 0.0, 0.0]);
    // The inert constraint stays registered until it is removed.
    assert_eq!(world.constraint_count(), 1);
    assert!(world.remove_constraint(0));
    assert_eq!(world.constraint_count(), 0);
}

#[test]
fn test_add_constraint_rejects_invalid_constraints() {
    let mut world = space_world();
    let a = dynamic(&mut world, 1.0, [0.0; 3]);
    let b = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);

    let unknown = world.add_constraint(PyConstraint::point_to_point(0, a, 9, [0.0; 3]));
    assert_eq!(unknown, Err(ConstraintError::UnknownBody { handle: 9 }));
    let unknown = world.add_constraint(PyConstraint::point_to_point(0, 8, b, [0.0; 3]));
    assert_eq!(unknown, Err(ConstraintError::UnknownBody { handle: 8 }));
    let same = world.add_constraint(PyConstraint::point_to_point(0, a, a, [0.0; 3]));
    assert_eq!(same, Err(ConstraintError::SameBody { handle: a }));

    let flat_hinge = world.add_constraint(PyConstraint::hinge(0, a, b, [0.0; 3], [0.0; 3]));
    assert_eq!(flat_hinge, Err(ConstraintError::ZeroAxis { kind: "hinge" }));
    let flat_slider = world.add_constraint(PyConstraint::slider(0, a, b, [0.0; 3], [0.0; 3]));
    assert_eq!(
        flat_slider,
        Err(ConstraintError::ZeroAxis { kind: "slider" })
    );

    let negative = world.add_constraint(PyConstraint::distance(0, a, b, [0.0; 3], [0.0; 3], -1.0));
    assert_eq!(
        negative,
        Err(ConstraintError::NegativeDistance { value: -1.0 })
    );
    let infinite = world.add_constraint(PyConstraint::point_to_point(
        0,
        a,
        b,
        [0.0, f64::INFINITY, 0.0],
    ));
    assert!(matches!(
        infinite,
        Err(ConstraintError::NotFinite { value, .. }) if value.is_infinite()
    ));
    let not_a_number = world.add_constraint(PyConstraint::distance(
        0,
        a,
        b,
        [f64::NAN, 0.0, 0.0],
        [0.0; 3],
        1.0,
    ));
    assert!(matches!(
        not_a_number,
        Err(ConstraintError::NotFinite { .. })
    ));

    let mut too_stiff = PyConstraint::distance(0, a, b, [0.0; 3], [0.0; 3], 1.0);
    too_stiff.stiffness = 1.5;
    assert_eq!(
        world.add_constraint(too_stiff),
        Err(ConstraintError::StiffnessOutOfRange { value: 1.5 })
    );
    let mut unknown_stiffness = PyConstraint::distance(0, a, b, [0.0; 3], [0.0; 3], 1.0);
    unknown_stiffness.stiffness = f64::NAN;
    assert!(matches!(
        world.add_constraint(unknown_stiffness),
        Err(ConstraintError::NotFinite { .. })
    ));

    // Nothing was added; valid constraints get consecutive handles.
    assert_eq!(world.constraint_count(), 0);
    let first = world.add_constraint(PyConstraint::point_to_point(0, a, b, [0.5, 0.0, 0.0]));
    let second = world.add_constraint(PyConstraint::hinge(
        0,
        a,
        b,
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ));
    let third = world.add_constraint(PyConstraint::slider(
        0,
        a,
        b,
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ));
    assert_eq!((first, second, third), (Ok(0), Ok(1), Ok(2)));
    assert_eq!(world.constraint_count(), 3);
    // The hinge axis is stored as a unit vector.
    let hinge = stored(&world, 1).frame.expect("frames");
    assert!((norm(hinge.local_axis) - 1.0).abs() < 1.0e-15);
}

#[test]
fn test_stiffness_setter_validates_and_local_anchors_replace_the_pivot() {
    let mut constraint = PyConstraint::distance(0, 0, 1, [0.0; 3], [0.0; 3], 1.0);
    assert_eq!(constraint.stiffness(), 0.5);
    assert_eq!(constraint.set_stiffness(0.75), Ok(()));
    assert_eq!(constraint.stiffness(), 0.75);
    assert_eq!(
        constraint.set_stiffness(-0.1),
        Err(ConstraintError::StiffnessOutOfRange { value: -0.1 })
    );
    assert!(matches!(
        constraint.set_stiffness(f64::NAN),
        Err(ConstraintError::NotFinite { .. })
    ));
    assert_eq!(constraint.stiffness(), 0.75);

    let pivoted = PyConstraint::point_to_point(0, 0, 1, [1.0, 2.0, 3.0]);
    assert_eq!(pivoted.pivot, Some([1.0, 2.0, 3.0]));
    let local = pivoted.with_local_anchors([0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]);
    assert_eq!(local.pivot, None);

    // Local anchors are used as given, whatever the poses of the bodies.
    let mut world = space_world();
    let a = dynamic(&mut world, 1.0, [0.0; 3]);
    let b = dynamic(&mut world, 1.0, [1.0, 0.0, 0.0]);
    let handle = world
        .add_constraint(
            PyConstraint::point_to_point(0, a, b, [9.0, 9.0, 9.0])
                .with_local_anchors([0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]),
        )
        .expect("both bodies exist");
    let c = stored(&world, handle);
    assert_eq!(c.anchor_a, [0.5, 0.0, 0.0]);
    assert_eq!(c.anchor_b, [-0.5, 0.0, 0.0]);
    assert!(anchor_gap(&world, handle) < 1.0e-15);
}

#[test]
fn test_solving_does_nothing_for_an_unusable_time_step() {
    let mut world = space_world();
    let post = fixed(&mut world, [0.0; 3]);
    let body = dynamic(&mut world, 1.0, [0.0; 3]);
    world.set_velocity(body, [2.0, 0.0, 0.0]);
    world
        .add_constraint(PyConstraint::point_to_point(0, post, body, [0.0; 3]))
        .expect("both bodies exist");
    for dt in [0.0, -DT, f64::NAN, f64::INFINITY] {
        world.resolve_constraints(dt);
        assert_eq!(velocity(&world, body), [2.0, 0.0, 0.0], "dt = {dt}");
    }
}
