// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of `WasmWorld` joints, contacts, sensors and ray casts.

use super::types_3::{WasmPhysicsEvent, WasmWorld};
use super::types_4::{WasmRigidBodyHandle, WasmSimulationConfig};
use super::vec_math::{q_rotate, v_add, v_length, v_sub};
use crate::error::Error;

fn world_anchor(world: &WasmWorld, body: WasmRigidBodyHandle, local: [f64; 3]) -> [f64; 3] {
    let p = world.get_position(body).expect("body exists");
    let q = world.get_rotation(body).expect("body exists");
    v_add(p, q_rotate(q, local))
}

#[test]
fn distance_joint_pendulum_keeps_length() {
    let mut world = WasmWorld::earth();
    let pivot = world.add_static_body(0.0, 0.0, 0.0);
    let bob = world.add_dynamic_body(1.0, 1.0, 0.0, 0.0);
    world.add_sphere_collider(bob, 0.1);
    world
        .add_distance_joint(pivot, bob, 1.0)
        .expect("valid joint");
    let mut lowest = 0.0_f64;
    let mut far_side = 0.0_f64;
    for step in 0..600 {
        world.step();
        let p = world.get_position(bob).expect("bob");
        let length = v_length(p);
        assert!(
            (length - 1.0).abs() < 1e-3,
            "step {step}: pendulum length {length}"
        );
        lowest = lowest.min(p[1]);
        far_side = far_side.min(p[0]);
    }
    assert!(
        lowest < -0.95,
        "bob must swing through the bottom: {lowest}"
    );
    assert!(far_side < -0.8, "bob must reach the far side: {far_side}");
    // The static pivot never moves.
    assert_eq!(world.get_position(pivot), Some([0.0; 3]));
}

#[test]
fn ball_joint_with_offset_anchor_keeps_anchors_together() {
    let mut world = WasmWorld::earth();
    let pivot = world.add_static_body(0.0, 2.0, 0.0);
    let bar = world.add_dynamic_body(2.0, 1.0, 2.0, 0.0);
    world.add_box_collider(bar, 1.0, 0.1, 0.1);
    world
        .add_ball_joint(pivot, bar, [0.0; 3], [-1.0, 0.0, 0.0])
        .expect("valid joint");
    let mut worst = 0.0_f64;
    for _ in 0..600 {
        world.step();
        let a = world_anchor(&world, pivot, [0.0; 3]);
        let b = world_anchor(&world, bar, [-1.0, 0.0, 0.0]);
        worst = worst.max(v_length(v_sub(a, b)));
    }
    assert!(worst < 1e-3, "anchor separation {worst}");
    // The bar swings about the pivot, so it must have rotated.
    let q = world.get_rotation(bar).expect("bar");
    assert!(q[3].abs() < 0.99, "bar must rotate, q = {q:?}");
}

#[test]
fn joints_validate_bodies_and_follow_body_removal() {
    let mut world = WasmWorld::earth();
    let a = world.add_dynamic_body(1.0, 0.0, 0.0, 0.0);
    let b = world.add_dynamic_body(1.0, 1.0, 0.0, 0.0);
    assert!(matches!(
        world.add_ball_joint(a, WasmRigidBodyHandle(99), [0.0; 3], [0.0; 3]),
        Err(Error::BodyNotFound(99))
    ));
    assert!(world.add_distance_joint(a, a, 1.0).is_err());
    assert!(world.add_distance_joint(a, b, -1.0).is_err());
    assert!(world.add_distance_joint(a, b, f64::NAN).is_err());
    assert!(
        world
            .add_ball_joint(a, b, [f64::INFINITY, 0.0, 0.0], [0.0; 3])
            .is_err()
    );
    assert_eq!(world.joint_count(), 0);
    world.add_distance_joint(a, b, 1.0).expect("valid joint");
    assert_eq!(world.joint_count(), 1);
    assert!(world.remove_body(b));
    assert_eq!(world.joint_count(), 0, "joints of a removed body go too");
}

#[test]
fn head_on_spheres_separate_and_conserve_momentum() {
    let mut world = WasmWorld::new(WasmSimulationConfig::zero_gravity());
    let a = world.add_dynamic_body(1.0, -1.0, 0.0, 0.0);
    let b = world.add_dynamic_body(1.0, 1.0, 0.0, 0.0);
    let ca = world.add_sphere_collider(a, 0.5);
    let cb = world.add_sphere_collider(b, 0.5);
    world.set_restitution(ca, 1.0);
    world.set_restitution(cb, 1.0);
    world.set_velocity(a, [2.0, 0.0, 0.0]);
    world.set_velocity(b, [-2.0, 0.0, 0.0]);
    let mut saw_contact = false;
    for _ in 0..90 {
        world.step();
        saw_contact |= !world.drain_contact_events().is_empty();
    }
    assert!(saw_contact, "the spheres must touch");
    let va = world.get_velocity(a).expect("a");
    let vb = world.get_velocity(b).expect("b");
    assert!(va[0] < -1.9 && vb[0] > 1.9, "elastic bounce: {va:?} {vb:?}");
    assert!((va[0] + vb[0]).abs() < 1e-9, "momentum conserved");
    let pa = world.get_position(a).expect("a");
    let pb = world.get_position(b).expect("b");
    assert!(pb[0] - pa[0] > 1.0, "spheres separated");
}

#[test]
fn box_comes_to_rest_on_plane() {
    let mut world = WasmWorld::earth();
    let ground = world.add_static_body(0.0, 0.0, 0.0);
    world.add_plane_collider(ground, 0.0, 1.0, 0.0, 0.0);
    let crate_body = world.add_dynamic_body(2.0, 0.0, 2.0, 0.0);
    world.add_box_collider(crate_body, 0.5, 0.5, 0.5);
    for _ in 0..300 {
        world.step();
    }
    let p = world.get_position(crate_body).expect("box");
    assert!(
        (p[1] - 0.5).abs() < 0.02,
        "box rests on its face, y = {}",
        p[1]
    );
    let v = world.get_velocity(crate_body).expect("box");
    assert!(v_length(v) < 0.05, "box at rest, v = {v:?}");
    let q = world.get_rotation(crate_body).expect("box");
    assert!(
        (q[3].abs() - 1.0).abs() < 1e-6,
        "box stays upright, q = {q:?}"
    );
    // A flat face rests on four corner contacts.
    world.wake_body(crate_body);
    world.step();
    let contacts = world.drain_contact_events();
    assert_eq!(contacts.len(), 4, "four corner contacts: {contacts:?}");
    let total: f64 = contacts.iter().map(|c| c.impulse).sum();
    let weight_impulse = 2.0 * 9.81 / 60.0;
    assert!(
        (total - weight_impulse).abs() < 0.2 * weight_impulse,
        "normal impulses carry the weight: {total} vs {weight_impulse}"
    );
}

#[test]
fn tilted_box_lands_and_settles_flat() {
    let mut world = WasmWorld::earth();
    let ground = world.add_static_body(0.0, 0.0, 0.0);
    world.add_plane_collider(ground, 0.0, 1.0, 0.0, 0.0);
    let body = world.add_dynamic_body(1.0, 0.0, 1.5, 0.0);
    world.add_box_collider(body, 0.5, 0.25, 0.5);
    // Give it a spin so it lands on an edge.
    if let Some(b) = world.bodies.get_mut(&body.0) {
        b.angular_vel = [0.0, 0.0, 2.0];
    }
    for _ in 0..600 {
        world.step();
    }
    let p = world.get_position(body).expect("box");
    assert!(
        p[1] < 0.6 && p[1] > 0.2,
        "box settled on the floor, y = {}",
        p[1]
    );
    let v = world.get_velocity(body).expect("box");
    assert!(v_length(v) < 0.05, "box at rest, v = {v:?}");
}

#[test]
fn box_stacks_on_static_box() {
    let mut world = WasmWorld::earth();
    let base = world.add_static_body(0.0, 0.0, 0.0);
    world.add_box_collider(base, 1.0, 0.5, 1.0);
    let top = world.add_dynamic_body(1.0, 0.0, 2.0, 0.0);
    world.add_box_collider(top, 0.5, 0.5, 0.5);
    for _ in 0..300 {
        world.step();
    }
    let p = world.get_position(top).expect("top");
    assert!(
        (p[1] - 1.0).abs() < 0.03,
        "box rests on the base, y = {}",
        p[1]
    );
    assert_eq!(world.get_position(base), Some([0.0; 3]));
}

#[test]
fn static_sphere_is_not_pushed() {
    let mut world = WasmWorld::earth();
    let rock = world.add_static_body(0.0, 0.0, 0.0);
    world.add_sphere_collider(rock, 1.0);
    let ball = world.add_dynamic_body(1.0, 0.0, 3.0, 0.0);
    world.add_sphere_collider(ball, 0.5);
    for _ in 0..240 {
        world.step();
    }
    assert_eq!(world.get_position(rock), Some([0.0; 3]));
    let p = world.get_position(ball).expect("ball");
    assert!(p[1] > 1.45, "ball rests on the rock, y = {}", p[1]);
}

#[test]
fn sensor_sphere_falls_through_floor_with_events() {
    let mut world = WasmWorld::earth();
    let ground = world.add_static_body(0.0, 0.0, 0.0);
    let floor = world.add_plane_collider(ground, 0.0, 1.0, 0.0, 0.0);
    let probe = world.add_dynamic_body(1.0, 0.0, 1.0, 0.0);
    let sensor = world.add_sphere_collider(probe, 0.3);
    world.set_sensor(sensor, true);
    let mut events = Vec::new();
    let mut contacts = 0;
    for _ in 0..120 {
        world.step();
        events.extend(world.drain_events());
        contacts += world.drain_contact_events().len();
    }
    let p = world.get_position(probe).expect("probe");
    assert!(p[1] < -1.0, "sensor falls through, y = {}", p[1]);
    assert_eq!(contacts, 0, "sensors produce no contact response");
    let entered = events
        .iter()
        .filter(|e| {
            matches!(e, WasmPhysicsEvent::SensorEntered { sensor: s, other: o }
                if *s == sensor && *o == floor)
        })
        .count();
    let exited = events
        .iter()
        .filter(|e| matches!(e, WasmPhysicsEvent::SensorExited { sensor: s, .. } if *s == sensor))
        .count();
    assert_eq!(entered, 1, "one enter event: {events:?}");
    assert_eq!(exited, 0, "a half-space keeps overlapping a body below it");
}

#[test]
fn sensor_exit_is_reported() {
    let mut world = WasmWorld::new(WasmSimulationConfig::zero_gravity());
    let zone_body = world.add_static_body(0.0, 0.0, 0.0);
    let zone = world.add_sphere_collider(zone_body, 1.0);
    world.set_sensor(zone, true);
    let mover = world.add_dynamic_body(1.0, -3.0, 0.0, 0.0);
    let mover_collider = world.add_sphere_collider(mover, 0.2);
    world.set_velocity(mover, [3.0, 0.0, 0.0]);
    let mut events = Vec::new();
    for _ in 0..180 {
        world.step();
        events.extend(world.drain_events());
    }
    let enter = events.iter().position(|e| {
        matches!(e, WasmPhysicsEvent::SensorEntered { sensor, other }
            if *sensor == zone && *other == mover_collider)
    });
    let exit = events.iter().position(|e| {
        matches!(e, WasmPhysicsEvent::SensorExited { sensor, other }
            if *sensor == zone && *other == mover_collider)
    });
    assert!(
        matches!((enter, exit), (Some(i), Some(j)) if i < j),
        "{events:?}"
    );
    // Passing through a sensor does not change the motion.
    let v = world.get_velocity(mover).expect("mover");
    assert!((v[0] - 3.0).abs() < 1e-12);
}

#[test]
fn bounce_reports_real_impulse_and_collision_events() {
    let mut world = WasmWorld::earth();
    let ground = world.add_static_body(0.0, 0.0, 0.0);
    world.add_plane_collider(ground, 0.0, 1.0, 0.0, 0.0);
    let ball = world.add_dynamic_body(2.0, 0.0, 2.0, 0.0);
    let collider = world.add_sphere_collider(ball, 0.5);
    world.set_restitution(collider, 0.5);
    let mut first_impulse = None;
    let mut started = 0;
    for _ in 0..120 {
        let v_before = world.get_velocity(ball).expect("ball")[1];
        world.step();
        started += world
            .drain_events()
            .iter()
            .filter(|e| matches!(e, WasmPhysicsEvent::CollisionStarted { .. }))
            .count();
        let contacts = world.drain_contact_events();
        if first_impulse.is_none() && !contacts.is_empty() {
            let v_after = world.get_velocity(ball).expect("ball")[1];
            let impulse: f64 = contacts.iter().map(|c| c.impulse).sum();
            first_impulse = Some((impulse, v_before, v_after, contacts[0].depth));
        }
    }
    let (impulse, v_before, v_after, depth) = first_impulse.expect("ball hits the floor");
    // J = m (v_after - v_in) where v_in includes this step's gravity.
    let v_in = v_before - 9.81 / 60.0;
    assert!(
        (impulse - 2.0 * (v_after - v_in)).abs() < 1e-9,
        "impulse {impulse} must match the momentum change"
    );
    assert!(
        v_after > 0.0 && (v_after + 0.5 * v_in).abs() < 1e-6,
        "e = 0.5"
    );
    assert!((impulse - depth).abs() > 1e-3, "impulse is not the depth");
    assert!(started >= 1, "collision start event emitted");
}

#[test]
fn raycast_hits_rotated_box() {
    let mut world = WasmWorld::new(WasmSimulationConfig::zero_gravity());
    let body = world.add_dynamic_body(1.0, 0.0, 0.0, 0.0);
    let collider = world.add_box_collider(body, 1.0, 1.0, 1.0);
    let hit = world.raycast([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0);
    assert!(hit.hit && hit.collider == Some(collider));
    assert!((hit.toi - 4.0).abs() < 1e-9);
    assert!((hit.normal[0] + 1.0).abs() < 1e-9);
    // Rotate by 45 degrees about Z: the corner now points at the ray.
    if let Some(b) = world.bodies.get_mut(&body.0) {
        b.angular_vel = [0.0, 0.0, std::f64::consts::FRAC_PI_4];
    }
    for _ in 0..60 {
        world.step();
    }
    let q = world.get_rotation(body).expect("box");
    assert!((2.0 * q[2].atan2(q[3]) - std::f64::consts::FRAC_PI_4).abs() < 1e-9);
    let hit = world.raycast([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0);
    assert!(hit.hit);
    assert!(
        (hit.toi - (5.0 - std::f64::consts::SQRT_2)).abs() < 1e-9,
        "toi {}",
        hit.toi
    );
    let miss = world.raycast([-5.0, 1.5, 0.0], [1.0, 0.0, 0.0], 100.0);
    assert!(!miss.hit, "ray above the rotated corner misses");
}

#[test]
fn simulation_is_deterministic() {
    let build = || {
        let mut world = WasmWorld::earth();
        let ground = world.add_static_body(0.0, 0.0, 0.0);
        world.add_plane_collider(ground, 0.0, 1.0, 0.0, 0.0);
        for i in 0..6 {
            let b = world.add_dynamic_body(1.0, 0.3 * i as f64, 1.0 + 1.1 * i as f64, 0.0);
            if i % 2 == 0 {
                world.add_sphere_collider(b, 0.5);
            } else {
                world.add_box_collider(b, 0.4, 0.4, 0.4);
            }
        }
        for _ in 0..200 {
            world.step();
        }
        (0..7)
            .map(|i| world.get_position(WasmRigidBodyHandle(i)).expect("body"))
            .collect::<Vec<_>>()
    };
    assert_eq!(build(), build());
}
