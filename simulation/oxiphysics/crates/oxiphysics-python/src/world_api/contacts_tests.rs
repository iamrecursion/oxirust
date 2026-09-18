// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the world contact pipeline (`contacts.rs`).

use super::MixRule;
use crate::types::{PyColliderShape, PyRigidBodyConfig, PySimConfig};
use crate::world_api::PyPhysicsWorld;

fn zero_g_world() -> PyPhysicsWorld {
    PyPhysicsWorld::new(PySimConfig {
        sleep_enabled: false,
        ..PySimConfig::zero_gravity()
    })
}

fn earth_world() -> PyPhysicsWorld {
    PyPhysicsWorld::new(PySimConfig::earth_gravity())
}

fn sphere(mass: f64, pos: [f64; 3], radius: f64) -> PyRigidBodyConfig {
    PyRigidBodyConfig::dynamic(mass, pos).with_shape(PyColliderShape::sphere(radius))
}

fn ground_box(hy: f64) -> PyRigidBodyConfig {
    PyRigidBodyConfig::static_body([0.0, 0.0, 0.0])
        .with_shape(PyColliderShape::box_shape(20.0, hy, 20.0))
}

fn run(world: &mut PyPhysicsWorld, dt: f64, steps: usize) {
    for _ in 0..steps {
        world.step(dt);
    }
}

fn pos(world: &PyPhysicsWorld, h: u32) -> [f64; 3] {
    world.get_position(h).expect("live body")
}

fn vel(world: &PyPhysicsWorld, h: u32) -> [f64; 3] {
    world.get_velocity(h).expect("live body")
}

/// Assert that a body resting on a support sits at `rest` height, allowing the
/// steady-state sink of the position-level contact correction: each step
/// gravity moves the body `g dt^2` into the support and the correction removes
/// `beta (depth - slop)`, so the depth settles at `slop + g dt^2 / beta`.
fn assert_rests(world: &PyPhysicsWorld, y: f64, rest: f64, dt: f64) {
    let cfg = world.config();
    let g = cfg.gravity[1].abs();
    let sink = cfg.max_penetration + g * dt * dt / cfg.baumgarte_factor;
    assert!(
        y <= rest + 1e-6 && y >= rest - sink - 1e-3,
        "rest height {y}, expected within [{}, {rest}]",
        rest - sink - 1e-3
    );
}

// --- Normal impulse direction (F3705) ---

#[test]
fn elastic_head_on_spheres_swap_velocities() {
    let mut world = zero_g_world();
    let a = world.add_rigid_body(sphere(1.0, [0.0, 0.0, 0.0], 1.0).with_restitution(1.0));
    let b = world.add_rigid_body(sphere(1.0, [1.9, 0.0, 0.0], 1.0).with_restitution(1.0));
    world.set_velocity(a, [1.0, 0.0, 0.0]);
    world.set_velocity(b, [-1.0, 0.0, 0.0]);
    world.step(1e-3);
    let (va, vb) = (vel(&world, a), vel(&world, b));
    assert!(
        (va[0] + 1.0).abs() < 1e-9,
        "A should bounce back, got {va:?}"
    );
    assert!(
        (vb[0] - 1.0).abs() < 1e-9,
        "B should bounce back, got {vb:?}"
    );
    let contacts = world.get_contacts();
    assert_eq!(contacts.len(), 1);
    let c = &contacts[0];
    assert_eq!((c.body_a, c.body_b), (a, b));
    assert!(
        (c.normal[0] - 1.0).abs() < 1e-9,
        "normal A->B: {:?}",
        c.normal
    );
    // j = (1 + e) * |v_rel| / (1/m_a + 1/m_b) = 2 * 2 / 2.
    assert!((c.impulse - 2.0).abs() < 1e-9, "impulse {}", c.impulse);
}

#[test]
fn inelastic_head_on_spheres_stop() {
    let mut world = zero_g_world();
    let a = world.add_rigid_body(sphere(2.0, [0.0, 0.0, 0.0], 1.0).with_restitution(0.0));
    let b = world.add_rigid_body(sphere(2.0, [1.95, 0.0, 0.0], 1.0).with_restitution(0.0));
    world.set_velocity(a, [3.0, 0.0, 0.0]);
    world.set_velocity(b, [-3.0, 0.0, 0.0]);
    world.step(1e-3);
    assert!(vel(&world, a)[0].abs() < 1e-9);
    assert!(vel(&world, b)[0].abs() < 1e-9);
}

#[test]
fn separating_overlapping_pair_keeps_separating() {
    let mut world = zero_g_world();
    let a = world.add_rigid_body(sphere(1.0, [0.0, 0.0, 0.0], 1.0).with_restitution(1.0));
    let b = world.add_rigid_body(sphere(1.0, [1.5, 0.0, 0.0], 1.0).with_restitution(1.0));
    world.set_velocity(a, [-1.0, 0.0, 0.0]);
    world.set_velocity(b, [1.0, 0.0, 0.0]);
    let before = pos(&world, b)[0] - pos(&world, a)[0];
    world.step(1e-3);
    assert_eq!(vel(&world, a), [-1.0, 0.0, 0.0]);
    assert_eq!(vel(&world, b), [1.0, 0.0, 0.0]);
    let after = pos(&world, b)[0] - pos(&world, a)[0];
    assert!(
        after > before,
        "pair must keep separating: {before} -> {after}"
    );
    let contacts = world.get_contacts();
    assert_eq!(contacts.len(), 1);
    assert_eq!(contacts[0].impulse, 0.0);
}

#[test]
fn sphere_bounces_off_static_sphere() {
    let mut world = zero_g_world();
    let wall = world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, 0.0, 0.0])
            .with_shape(PyColliderShape::sphere(1.0))
            .with_restitution(1.0),
    );
    let ball = world.add_rigid_body(sphere(1.0, [1.95, 0.0, 0.0], 1.0).with_restitution(1.0));
    world.set_velocity(ball, [-2.0, 0.0, 0.0]);
    world.step(1e-3);
    assert!((vel(&world, ball)[0] - 2.0).abs() < 1e-9);
    assert_eq!(vel(&world, wall), [0.0, 0.0, 0.0]);
    assert_eq!(pos(&world, wall), [0.0, 0.0, 0.0]);
}

// --- Positional correction (F3706) ---

#[test]
fn baumgarte_correction_is_mass_weighted_for_light_bodies() {
    let mut world = zero_g_world();
    let a = world.add_rigid_body(sphere(0.01, [0.0, 0.0, 0.0], 1.0));
    let b = world.add_rigid_body(sphere(0.01, [1.9, 0.0, 0.0], 1.0));
    world.step(1.0 / 60.0);
    let cfg = world.config().clone();
    let expected = cfg.baumgarte_factor * (0.1 - cfg.max_penetration);
    let gap = pos(&world, b)[0] - pos(&world, a)[0];
    assert!(
        (gap - 1.9 - expected).abs() < 1e-12,
        "separation grew by {} instead of {expected}",
        gap - 1.9
    );
    // Equal masses share the correction equally.
    assert!((pos(&world, a)[0] + 0.5 * expected).abs() < 1e-12);
    assert!((pos(&world, b)[0] - 1.9 - 0.5 * expected).abs() < 1e-12);
}

#[test]
fn baumgarte_correction_splits_by_inverse_mass() {
    let mut world = zero_g_world();
    let heavy = world.add_rigid_body(sphere(100.0, [0.0, 0.0, 0.0], 1.0));
    let light = world.add_rigid_body(sphere(1.0, [1.9, 0.0, 0.0], 1.0));
    world.step(1.0 / 60.0);
    let cfg = world.config().clone();
    let total = cfg.baumgarte_factor * (0.1 - cfg.max_penetration);
    let moved_heavy = -pos(&world, heavy)[0];
    let moved_light = pos(&world, light)[0] - 1.9;
    assert!((moved_heavy + moved_light - total).abs() < 1e-12);
    assert!((moved_light / moved_heavy - 100.0).abs() < 1e-6);
}

#[test]
fn correction_ignores_penetration_within_slop() {
    let mut world = zero_g_world();
    let a = world.add_rigid_body(sphere(1.0, [0.0, 0.0, 0.0], 1.0));
    let slop = world.config().max_penetration;
    let b = world.add_rigid_body(sphere(1.0, [2.0 - 0.5 * slop, 0.0, 0.0], 1.0));
    world.step(1.0 / 60.0);
    assert_eq!(pos(&world, a), [0.0, 0.0, 0.0]);
    assert_eq!(pos(&world, b), [2.0 - 0.5 * slop, 0.0, 0.0]);
    assert_eq!(world.get_contacts().len(), 1);
}

// --- Shapes other than spheres (F3703) ---

#[test]
fn sphere_dropped_on_static_box_rests_on_top() {
    let mut world = earth_world();
    world.add_rigid_body(ground_box(0.5));
    let ball = world.add_rigid_body(sphere(1.0, [0.0, 3.0, 0.0], 0.5));
    run(&mut world, 1.0 / 120.0, 480);
    let p = pos(&world, ball);
    assert_rests(&world, p[1], 1.0, 1.0 / 120.0);
    assert!(p[0].abs() < 1e-6 && p[2].abs() < 1e-6);
    assert!(vel(&world, ball)[1].abs() < 0.05);
}

#[test]
fn box_dropped_on_plane_rests_flat() {
    let mut world = earth_world();
    world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, 0.0, 0.0])
            .with_shape(PyColliderShape::plane([0.0, 1.0, 0.0], 0.0)),
    );
    let crate_body = world.add_rigid_body(
        PyRigidBodyConfig::dynamic(2.0, [0.0, 2.0, 0.0])
            .with_shape(PyColliderShape::box_shape(0.5, 0.25, 0.5))
            .with_restitution(0.0),
    );
    run(&mut world, 1.0 / 120.0, 480);
    let p = pos(&world, crate_body);
    assert_rests(&world, p[1], 0.25, 1.0 / 120.0);
    let q = world.get_orientation(crate_body).expect("live body");
    assert!(
        (q[3].abs() - 1.0).abs() < 1e-6,
        "flat box must not tip: {q:?}"
    );
}

#[test]
fn sphere_lands_on_plane_owned_by_higher_slot() {
    // The plane body comes after the sphere, exercising the flipped normal.
    let mut world = earth_world();
    let ball = world.add_rigid_body(sphere(1.0, [0.0, 2.0, 0.0], 0.5).with_restitution(0.0));
    world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, 0.0, 0.0])
            .with_shape(PyColliderShape::plane([0.0, 1.0, 0.0], 0.0)),
    );
    run(&mut world, 1.0 / 120.0, 360);
    assert_rests(&world, pos(&world, ball)[1], 0.5, 1.0 / 120.0);
}

#[test]
fn lying_capsule_rests_on_box() {
    let mut world = earth_world();
    world.add_rigid_body(ground_box(0.5));
    // Rotate the capsule axis from +Y onto +X (90 degrees about Z).
    let half = std::f64::consts::FRAC_PI_4;
    let mut cfg = PyRigidBodyConfig::dynamic(1.0, [0.0, 2.0, 0.0])
        .with_shape(PyColliderShape::capsule(0.25, 0.5))
        .with_restitution(0.0);
    cfg.orientation = [0.0, 0.0, half.sin(), half.cos()];
    let capsule = world.add_rigid_body(cfg);
    run(&mut world, 1.0 / 120.0, 480);
    let p = pos(&world, capsule);
    assert_rests(&world, p[1], 0.75, 1.0 / 120.0);
    // Lying flat on two end contacts: the axis stays horizontal.
    let q = world.get_orientation(capsule).expect("live body");
    let axis = crate::world_api::quat_rotate_vec(q, [0.0, 1.0, 0.0]);
    assert!(axis[1].abs() < 1e-3, "capsule axis tilted: {axis:?}");
}

#[test]
fn falling_box_stacks_on_static_box() {
    let mut world = earth_world();
    world.add_rigid_body(ground_box(0.5));
    let top = world.add_rigid_body(
        PyRigidBodyConfig::dynamic(1.0, [0.0, 2.0, 0.0])
            .with_shape(PyColliderShape::box_shape(0.5, 0.5, 0.5))
            .with_restitution(0.0),
    );
    run(&mut world, 1.0 / 120.0, 480);
    let p = pos(&world, top);
    assert_rests(&world, p[1], 1.0, 1.0 / 120.0);
    let q = world.get_orientation(top).expect("live body");
    assert!((q[3].abs() - 1.0).abs() < 1e-6);
}

#[test]
fn static_pairs_produce_no_contacts() {
    let mut world = earth_world();
    world.add_rigid_body(ground_box(0.5));
    world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, 0.5, 0.0]).with_shape(PyColliderShape::sphere(1.0)),
    );
    world.step(1.0 / 60.0);
    assert_eq!(world.contact_count(), 0);
}

// --- Friction (F3704) ---

fn sliding_sphere(friction: f64) -> (PyPhysicsWorld, u32) {
    let mut world = PyPhysicsWorld::new(PySimConfig {
        sleep_enabled: false,
        ..PySimConfig::earth_gravity()
    });
    world.add_rigid_body(ground_box(0.5).with_friction(friction));
    let ball = world.add_rigid_body(
        sphere(1.0, [0.0, 0.999, 0.0], 0.5)
            .with_friction(friction)
            .with_restitution(0.0),
    );
    world.set_velocity(ball, [2.0, 0.0, 0.0]);
    (world, ball)
}

#[test]
fn friction_turns_sliding_into_rolling() {
    let (mut world, ball) = sliding_sphere(1.0);
    run(&mut world, 1.0 / 240.0, 240);
    let v = vel(&world, ball);
    let w = world.get_angular_velocity(ball).expect("live body");
    // A solid sphere ends up rolling at 5/7 of its initial sliding speed.
    let rolling = 2.0 * 5.0 / 7.0;
    assert!((v[0] - rolling).abs() < 0.03 * rolling, "vx {}", v[0]);
    // Rolling without slipping: v = -w_z * r.
    assert!((v[0] + w[2] * 0.5).abs() < 0.02, "v {v:?} w {w:?}");
}

#[test]
fn frictionless_sphere_keeps_sliding() {
    let (mut world, ball) = sliding_sphere(0.0);
    run(&mut world, 1.0 / 240.0, 240);
    let v = vel(&world, ball);
    let w = world.get_angular_velocity(ball).expect("live body");
    assert!((v[0] - 2.0).abs() < 1e-9, "vx {}", v[0]);
    assert!(w.iter().all(|x| x.abs() < 1e-9), "w {w:?}");
}

#[test]
fn friction_mixing_min_uses_the_smaller_coefficient() {
    let mut world = PyPhysicsWorld::new(PySimConfig {
        sleep_enabled: false,
        friction_mixing: "min".to_string(),
        ..PySimConfig::earth_gravity()
    });
    world.add_rigid_body(ground_box(0.5).with_friction(0.0));
    let ball = world.add_rigid_body(
        sphere(1.0, [0.0, 0.999, 0.0], 0.5)
            .with_friction(1.0)
            .with_restitution(0.0),
    );
    world.set_velocity(ball, [2.0, 0.0, 0.0]);
    run(&mut world, 1.0 / 240.0, 120);
    assert!((vel(&world, ball)[0] - 2.0).abs() < 1e-9);
    assert!(world.get_contacts().iter().all(|c| c.friction == 0.0));
}

#[test]
fn mix_rules_parse_and_combine() {
    assert_eq!(MixRule::parse("Average"), Some(MixRule::Average));
    assert_eq!(MixRule::parse(" min "), Some(MixRule::Min));
    assert_eq!(MixRule::parse("MAX"), Some(MixRule::Max));
    assert_eq!(MixRule::parse("multiply"), Some(MixRule::Multiply));
    assert_eq!(MixRule::parse("geometric"), None);
    assert_eq!(MixRule::Average.combine(0.2, 0.6), 0.4);
    assert_eq!(MixRule::Min.combine(0.2, 0.6), 0.2);
    assert_eq!(MixRule::Max.combine(0.2, 0.6), 0.6);
    assert!((MixRule::Multiply.combine(0.2, 0.6) - 0.12).abs() < 1e-15);
}

#[test]
fn restitution_mixing_max_is_applied() {
    let mut world = PyPhysicsWorld::new(PySimConfig {
        sleep_enabled: false,
        restitution_mixing: "max".to_string(),
        ..PySimConfig::zero_gravity()
    });
    let a = world.add_rigid_body(sphere(1.0, [0.0, 0.0, 0.0], 1.0).with_restitution(0.0));
    let b = world.add_rigid_body(sphere(1.0, [1.9, 0.0, 0.0], 1.0).with_restitution(1.0));
    world.set_velocity(a, [1.0, 0.0, 0.0]);
    world.set_velocity(b, [-1.0, 0.0, 0.0]);
    world.step(1e-3);
    assert!((vel(&world, a)[0] + 1.0).abs() < 1e-9);
    assert!((vel(&world, b)[0] - 1.0).abs() < 1e-9);
}

// --- Sleeping bodies ---

#[test]
fn struck_sleeping_body_wakes_and_moves() {
    let mut world = PyPhysicsWorld::new(PySimConfig::zero_gravity());
    let a = world.add_rigid_body(sphere(1.0, [0.0, 0.0, 0.0], 1.0).with_restitution(1.0));
    let b = world.add_rigid_body(sphere(1.0, [1.95, 0.0, 0.0], 1.0).with_restitution(1.0));
    world.sleep_body(b);
    world.set_velocity(a, [2.0, 0.0, 0.0]);
    world.step(1e-3);
    assert!(!world.is_sleeping(b));
    assert!((vel(&world, a)[0]).abs() < 1e-9);
    assert!((vel(&world, b)[0] - 2.0).abs() < 1e-9);
}

// --- Ray casts ---

#[test]
fn raycast_hits_box_and_capsule() {
    let mut world = zero_g_world();
    let boxed = world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, 0.0, 5.0])
            .with_shape(PyColliderShape::box_shape(1.0, 1.0, 1.0)),
    );
    let hit = world.raycast([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 100.0);
    let (h, t) = hit.expect("box must be hit");
    assert_eq!(h, boxed);
    assert!((t - 4.0).abs() < 1e-9, "t = {t}");

    let capsule = world.add_rigid_body(
        PyRigidBodyConfig::static_body([3.0, 0.0, 0.0])
            .with_shape(PyColliderShape::capsule(0.5, 1.0)),
    );
    let (h, t) = world
        .raycast([3.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0)
        .expect("capsule must be hit");
    assert_eq!(h, capsule);
    assert!((t - 8.5).abs() < 1e-9, "t = {t}");
}

#[test]
fn raycast_respects_body_orientation() {
    let mut world = zero_g_world();
    let half = std::f64::consts::FRAC_PI_8; // 45 degrees about Y
    let mut cfg = PyRigidBodyConfig::static_body([0.0, 0.0, 5.0])
        .with_shape(PyColliderShape::box_shape(1.0, 1.0, 1.0));
    cfg.orientation = [0.0, half.sin(), 0.0, half.cos()];
    world.add_rigid_body(cfg);
    let (_, t) = world
        .raycast([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0)
        .expect("rotated box must be hit");
    assert!((t - (5.0 - 2.0_f64.sqrt())).abs() < 1e-9, "t = {t}");
}

#[test]
fn raycast_hits_plane_and_misses_parallel() {
    let mut world = zero_g_world();
    let ground = world.add_rigid_body(
        PyRigidBodyConfig::static_body([0.0, -1.0, 0.0])
            .with_shape(PyColliderShape::plane([0.0, 1.0, 0.0], 0.0)),
    );
    let (h, t) = world
        .raycast([0.0, 4.0, 0.0], [1.0, -1.0, 0.0], 100.0)
        .expect("plane must be hit");
    assert_eq!(h, ground);
    assert!((t - 5.0 * 2.0_f64.sqrt()).abs() < 1e-9, "t = {t}");
    assert!(
        world
            .raycast([0.0, 4.0, 0.0], [1.0, 0.0, 0.0], 100.0)
            .is_none()
    );
}
