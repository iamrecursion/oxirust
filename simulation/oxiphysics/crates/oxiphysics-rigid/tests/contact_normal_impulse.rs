// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0
//! Regression tests for [`PhysicsWorld::contact_normal_impulse`]: the impulse
//! reported for each contact of the last step is the one the sequential-impulse
//! solver actually applied, so a resting body's contacts carry its weight.

use oxiphysics_core::math::Vec3;
use oxiphysics_geometry::{BoxShape, Shape, Sphere};
use oxiphysics_rigid::world::PhysicsWorld;
use oxiphysics_rigid::{Collider, RigidBody};
use std::sync::Arc;

fn resting_sphere_world(mass: f64) -> PhysicsWorld {
    let dt = 1.0 / 120.0;
    let mut world = PhysicsWorld::new([0.0, -9.81, 0.0], dt);
    let mut ground = RigidBody::new_static();
    ground.transform.position = Vec3::new(0.0, -0.5, 0.0);
    let ground_h = world.add_rigid_body(ground);
    world.add_collider(
        Collider::new(Arc::new(BoxShape::new(Vec3::new(10.0, 0.5, 10.0)))).with_restitution(0.0),
        Some(ground_h),
    );
    let shape: Arc<dyn Shape> = Arc::new(Sphere::new(0.5));
    let mut ball = RigidBody::new(mass);
    ball.transform.position = Vec3::new(0.0, 0.5, 0.0);
    ball.set_inertia_from_shape(shape.as_ref());
    ball.update_world_inertia();
    let ball_h = world.add_rigid_body(ball);
    world.add_collider(Collider::new(shape).with_restitution(0.0), Some(ball_h));
    // Keep the ball awake so every step produces a solved contact.
    world.time_before_sleep = f64::INFINITY;
    world
}

#[test]
fn resting_contact_impulse_balances_weight() {
    let mass = 2.0;
    let mut world = resting_sphere_world(mass);
    for _ in 0..240 {
        world.step();
    }
    assert!(
        !world.last_contacts.is_empty(),
        "ball must touch the ground"
    );
    let mut total = 0.0;
    for cp in &world.last_contacts {
        let j = world
            .contact_normal_impulse(cp)
            .expect("every resting contact is solved by the default solver");
        assert!(j >= 0.0, "normal impulses push, got {j}");
        total += j;
    }
    let weight_impulse = mass * 9.81 * world.dt;
    assert!(
        (total - weight_impulse).abs() < 0.05 * weight_impulse,
        "summed contact impulse {total} should equal m g dt = {weight_impulse}"
    );
}

#[test]
fn contact_from_another_step_has_no_impulse() {
    let mut world = resting_sphere_world(1.0);
    for _ in 0..10 {
        world.step();
    }
    let mut foreign = world
        .last_contacts
        .first()
        .cloned()
        .expect("ball touches the ground");
    foreign.contact_point[0] += 0.25;
    assert!(world.contact_normal_impulse(&foreign).is_none());
}
