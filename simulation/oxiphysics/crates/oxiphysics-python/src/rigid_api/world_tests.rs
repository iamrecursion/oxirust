// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the engine-backed `rigid.PyWorld` and its joint projection.

use super::bodies::{PyCollider, PyRigidBody, PyShapeType};
use super::joints::{PyJoint, PyJointType};
use super::world::{PyPhysicsPipeline, PyWorld, quat_from_xyzw, quat_to_xyzw};

const DT: f64 = 1.0 / 60.0;

fn sphere_body(mass: f64, position: [f64; 3], radius: f64) -> PyRigidBody {
    let mut b = PyRigidBody::new(0, mass, position);
    b.add_collider(PyCollider::new(radius, 0.5, 0.0));
    b
}

fn ground(world: &mut PyWorld) -> u32 {
    let mut g = PyRigidBody::static_body(0, [0.0, -0.5, 0.0]);
    let mut c = PyCollider::box_collider(20.0, 0.5, 20.0);
    c.restitution = 0.0;
    g.add_collider(c);
    world.add_body(g).expect("valid ground")
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn run(world: &mut PyWorld, seconds: f64) {
    let steps = (seconds / DT).round() as usize;
    for _ in 0..steps {
        world.advance(DT, 1).expect("valid step");
    }
}

/// Rotate `v` by the `[x, y, z, w]` quaternion `q`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let uq = quat_from_xyzw(q).expect("valid quaternion");
    let r = uq * oxiphysics::core::math::Vec3::new(v[0], v[1], v[2]);
    [r.x, r.y, r.z]
}

#[test]
fn bodies_can_be_added_and_overlapping_spheres_collide() {
    let mut world = PyWorld::new([0.0, 0.0, 0.0]);
    let a = world
        .add_body(sphere_body(1.0, [0.0, 0.0, 0.0], 0.5))
        .expect("valid body");
    // Overlapping and approaching, so the velocity solver must push.
    let mut incoming = sphere_body(1.0, [0.8, 0.0, 0.0], 0.5);
    incoming.velocity = [-2.0, 0.0, 0.0];
    let b = world.add_body(incoming).expect("valid body");
    assert_eq!(world.body_count(), 2);
    assert_ne!(a, b);
    world.advance(DT, 1).expect("valid step");
    let contacts = world.get_contacts();
    assert!(!contacts.is_empty(), "overlapping spheres must touch");
    for c in &contacts {
        let pair = [c.body_a, c.body_b];
        assert!(pair.contains(&a) && pair.contains(&b), "handles map back");
        assert!(c.depth > 0.0);
        assert!(c.impulse > 0.0, "the solver pushed the spheres apart");
    }
    run(&mut world, 0.5);
    let pa = world.get_body(a).expect("body a").position;
    let pb = world.get_body(b).expect("body b").position;
    assert!(
        dist(pa, pb) > 0.95,
        "spheres separated, distance {}",
        dist(pa, pb)
    );
}

#[test]
fn sphere_rests_on_static_box() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    ground(&mut world);
    let ball = world
        .add_body(sphere_body(1.0, [0.0, 2.0, 0.0], 0.5))
        .expect("valid ball");
    run(&mut world, 2.0);
    let y = world.get_body(ball).expect("ball").position[1];
    assert!(
        (y - 0.5).abs() < 0.02,
        "ball should rest at y = 0.5, got {y}"
    );
    let support: f64 = world.get_contacts().iter().map(|c| c.impulse).sum();
    let weight = 9.81 * DT;
    assert!(
        world.sleeping_body_count() == 1 || (support - weight).abs() < 0.1 * weight,
        "resting contacts carry the weight: {support} vs {weight}"
    );
}

#[test]
fn ball_socket_pendulum_keeps_its_anchor() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let pivot = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("pivot");
    let bob = world
        .add_body(sphere_body(1.0, [1.0, 0.0, 0.0], 0.1))
        .expect("bob");
    let j = world
        .add_joint(PyJoint::new(0, pivot, bob, [0.0, 0.0, 0.0]))
        .expect("joint");
    let mut min_y: f64 = 0.0;
    for _ in 0..60 {
        world.advance(DT, 1).expect("step");
        let gap = world.joint_anchor_gap(j).expect("joint gap");
        assert!(gap < 1e-3, "anchor gap {gap}");
        min_y = min_y.min(world.get_body(bob).expect("bob").position[1]);
    }
    let gap = world.joint_anchor_gap(j).expect("joint gap");
    assert!(gap < 1e-3, "anchor gap after 1 s: {gap}");
    assert!(min_y < -0.5, "the bob must swing down, min y = {min_y}");
    let p = world.get_body(bob).expect("bob").position;
    let r = dist(p, [0.0; 3]);
    assert!((r - 1.0).abs() < 0.01, "pendulum length preserved: {r}");
}

#[test]
fn distance_joint_holds_target_distance() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let anchor = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("anchor");
    let bob = world
        .add_body(sphere_body(2.0, [0.0, -1.5, 0.0], 0.1))
        .expect("bob");
    world
        .set_body({
            let mut b = world.get_body(bob).expect("bob");
            b.velocity = [3.0, 0.0, 0.0];
            b
        })
        .expect("valid body");
    world
        .add_joint(PyJoint::distance(
            0,
            anchor,
            bob,
            [0.0, 0.0, 0.0],
            [0.0, -1.5, 0.0],
            1.5,
        ))
        .expect("joint");
    run(&mut world, 1.0);
    let p = world.get_body(bob).expect("bob").position;
    let r = dist(p, [0.0; 3]);
    assert!((r - 1.5).abs() < 2e-3, "distance {r} should stay 1.5");
    assert!(p[0].abs() > 0.1, "the bob swings sideways, x = {}", p[0]);
}

#[test]
fn hinge_keeps_axis_aligned_and_allows_rotation_about_it() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let base = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("base");
    let mut door = PyRigidBody::new(0, 1.0, [0.5, 0.0, 0.0]);
    door.add_collider(PyCollider::box_collider(0.5, 0.05, 0.3));
    let door = world.add_body(door).expect("door");
    let j = world
        .add_joint(PyJoint::hinge(
            0,
            base,
            door,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ))
        .expect("hinge");
    // Kick the door about an axis the hinge must suppress.
    world
        .set_body({
            let mut b = world.get_body(door).expect("door");
            b.angular_velocity = [2.0, 0.0, 0.0];
            b
        })
        .expect("valid body");
    let mut min_y: f64 = 0.0;
    for _ in 0..60 {
        world.advance(DT, 1).expect("step");
        let d = world.get_body(door).expect("door");
        let axis = rotate(d.orientation, [0.0, 0.0, 1.0]);
        assert!(axis[2] > 0.999, "hinge axis stays aligned: {axis:?}");
        assert!(world.joint_anchor_gap(j).expect("gap") < 1e-3);
        min_y = min_y.min(d.position[1]);
    }
    // Gravity swings the door about the hinge axis.
    assert!(min_y < -0.4, "door swung down, min y = {min_y}");
    let p = world.get_body(door).expect("door").position;
    assert!(
        (dist(p, [0.0; 3]) - 0.5).abs() < 1e-3,
        "door stays on its hinge"
    );
}

#[test]
fn slider_moves_only_along_its_axis() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let rail = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("rail");
    let mut cart = PyRigidBody::new(0, 1.0, [0.0, 0.0, 0.0]);
    cart.velocity = [1.0, 0.0, 0.5];
    cart.angular_velocity = [0.0, 3.0, 0.0];
    let cart = world.add_body(cart).expect("cart");
    let mut joint = PyJoint::new(0, rail, cart, [0.0, 0.0, 0.0]);
    joint.joint_type = PyJointType::Slider;
    joint.axis = [1.0, 0.0, 0.0];
    world.add_joint(joint).expect("slider");
    run(&mut world, 1.0);
    let c = world.get_body(cart).expect("cart");
    assert!(c.position[0] > 0.5, "cart slides along x: {:?}", c.position);
    assert!(
        c.position[1].abs() < 1e-3,
        "no drift in y: {:?}",
        c.position
    );
    assert!(
        c.position[2].abs() < 1e-3,
        "no drift in z: {:?}",
        c.position
    );
    let x_axis = rotate(c.orientation, [1.0, 0.0, 0.0]);
    assert!(x_axis[0] > 0.999, "slider locks rotation: {x_axis:?}");
}

#[test]
fn fixed_joint_keeps_relative_pose() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let wall = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("wall");
    let mut beam = PyRigidBody::new(0, 1.0, [1.0, 0.0, 0.0]);
    beam.add_collider(PyCollider::box_collider(1.0, 0.1, 0.1));
    let beam = world.add_body(beam).expect("beam");
    let mut joint = PyJoint::new(0, wall, beam, [0.0, 0.0, 0.0]);
    joint.joint_type = PyJointType::Fixed;
    let j = world.add_joint(joint).expect("fixed");
    run(&mut world, 1.0);
    let b = world.get_body(beam).expect("beam");
    assert!(dist(b.position, [1.0, 0.0, 0.0]) < 5e-3, "{:?}", b.position);
    let x_axis = rotate(b.orientation, [1.0, 0.0, 0.0]);
    assert!(x_axis[0] > 0.9999, "orientation locked: {x_axis:?}");
    assert!(world.joint_anchor_gap(j).expect("gap") < 1e-3);
}

#[test]
fn angular_velocity_rotates_orientation() {
    let mut world = PyWorld::new([0.0, 0.0, 0.0]);
    let mut spinner = sphere_body(1.0, [0.0, 0.0, 0.0], 0.5);
    spinner.angular_velocity = [0.0, std::f64::consts::FRAC_PI_2, 0.0];
    spinner.can_sleep = false;
    let h = world.add_body(spinner).expect("spinner");
    run(&mut world, 1.0);
    let b = world.get_body(h).expect("spinner");
    // A quarter turn about +y maps +x onto −z.
    let x_axis = rotate(b.orientation, [1.0, 0.0, 0.0]);
    assert!(x_axis[2] < -0.99, "x axis after a quarter turn: {x_axis:?}");
}

#[test]
fn remove_body_drops_colliders_and_joints() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let a = world
        .add_body(sphere_body(1.0, [0.0, 0.0, 0.0], 0.5))
        .expect("a");
    let b = world
        .add_body(sphere_body(1.0, [2.0, 0.0, 0.0], 0.5))
        .expect("b");
    let j = world
        .add_joint(PyJoint::new(0, a, b, [1.0, 0.0, 0.0]))
        .expect("joint");
    assert_eq!(world.collider_count(), 2);
    assert!(world.remove_body(a));
    assert!(!world.remove_body(a), "second removal reports absence");
    assert_eq!(world.body_count(), 1);
    assert_eq!(world.collider_count(), 1);
    assert_eq!(world.joint_count(), 0);
    assert!(world.get_joint(j).is_none());
    assert!(world.get_body(a).is_none());
    world.advance(DT, 2).expect("still steps");
    assert_eq!(world.engine().body_count(), 1);
}

#[test]
fn quaternion_round_trip_uses_xyzw_order() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let q = [0.0, s, 0.0, s];
    let back = quat_to_xyzw(&quat_from_xyzw(q).expect("unit quaternion"));
    for (x, y) in q.iter().zip(back.iter()) {
        assert!((x - y).abs() < 1e-12, "{q:?} vs {back:?}");
    }
    let mut world = PyWorld::new([0.0, 0.0, 0.0]);
    let mut body = sphere_body(1.0, [0.0, 0.0, 0.0], 0.5);
    body.orientation = q;
    let h = world.add_body(body).expect("body");
    let o = world.get_body(h).expect("body").orientation;
    assert!(
        (o[1] - s).abs() < 1e-12 && (o[3] - s).abs() < 1e-12,
        "{o:?}"
    );
    assert!(quat_from_xyzw([0.0; 4]).is_err());
}

#[test]
fn invalid_bodies_and_joints_are_rejected() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let mut hull = PyRigidBody::new(0, 1.0, [0.0; 3]);
    let mut c = PyCollider::new(0.5, 0.5, 0.3);
    c.shape = PyShapeType::ConvexHull;
    hull.add_collider(c);
    assert!(world.add_body(hull).is_err(), "hull without vertices");
    assert!(
        world.add_body(PyRigidBody::new(0, 0.0, [0.0; 3])).is_err(),
        "massless dynamic body"
    );
    assert_eq!(world.body_count(), 0);
    let a = world
        .add_body(PyRigidBody::new(0, 1.0, [0.0; 3]))
        .expect("a");
    assert!(world.add_joint(PyJoint::new(0, a, 99, [0.0; 3])).is_err());
    assert!(world.add_joint(PyJoint::new(0, a, a, [0.0; 3])).is_err());
    let b = world
        .add_body(PyRigidBody::new(0, 1.0, [1.0, 0.0, 0.0]))
        .expect("b");
    let mut hinge = PyJoint::hinge(0, a, b, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert!(world.add_joint(hinge.clone()).is_err(), "zero hinge axis");
    hinge.axis = [0.0, 1.0, 0.0];
    hinge.stiffness = 1.5;
    assert!(world.add_joint(hinge).is_err(), "stiffness above one");
    assert!(world.advance(0.0, 1).is_err());
    assert!(world.advance(f64::NAN, 1).is_err());
    let mut cfg = world.config.clone();
    cfg.baumgarte_factor = f64::NAN;
    world.config = cfg;
    assert!(world.advance(DT, 1).is_err(), "non-finite config rejected");
}

#[test]
fn snapshots_do_not_alias_world_state() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let h = world
        .add_body(sphere_body(1.0, [0.0, 5.0, 0.0], 0.5))
        .expect("body");
    let mut snapshot = world.bodies();
    assert_eq!(snapshot.len(), 1);
    snapshot.add(PyRigidBody::new(0, 1.0, [0.0; 3]));
    assert_eq!(world.body_count(), 1, "the snapshot is a copy");
    assert!(world.apply_impulse(h, [0.0, 10.0, 0.0]).expect("finite"));
    let v = world.get_body(h).expect("body").velocity;
    assert!(
        (v[1] - 10.0).abs() < 1e-12,
        "impulse / mass = 10 m/s: {v:?}"
    );
    assert!(!world.apply_impulse(77, [1.0, 0.0, 0.0]).expect("finite"));
}

#[test]
fn pipeline_counts_substeps_and_advances_time() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let h = world
        .add_body(sphere_body(1.0, [0.0, 10.0, 0.0], 0.5))
        .expect("body");
    let mut pipeline = PyPhysicsPipeline::new();
    pipeline.step(&mut world, 0.1, 4).expect("valid step");
    assert_eq!(pipeline.total_steps(), 4);
    assert!((world.elapsed_time() - 0.1).abs() < 1e-12);
    let v = world.get_body(h).expect("body").velocity[1];
    assert!((v + 0.981).abs() < 1e-9, "v = g t, got {v}");
    assert!(pipeline.step(&mut world, -1.0, 1).is_err());
    assert_eq!(pipeline.total_steps(), 4);
}

#[test]
fn disabled_joint_has_no_effect() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let pivot = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("pivot");
    let bob = world
        .add_body(sphere_body(1.0, [1.0, 0.0, 0.0], 0.1))
        .expect("bob");
    let j = world
        .add_joint(PyJoint::new(0, pivot, bob, [0.0, 0.0, 0.0]))
        .expect("joint");
    assert!(world.set_joint_enabled(j, false));
    run(&mut world, 1.0);
    let y = world.get_body(bob).expect("bob").position[1];
    // Free fall for one second: y = −g t² / 2 (semi-implicit Euler: within a step).
    assert!(y < -4.5, "free fall without the joint, y = {y}");
}

#[test]
fn two_link_chain_between_dynamic_bodies_stays_connected() {
    let mut world = PyWorld::new([0.0, -9.81, 0.0]);
    let pivot = world
        .add_body(PyRigidBody::static_body(0, [0.0, 0.0, 0.0]))
        .expect("pivot");
    let l1 = world
        .add_body(sphere_body(1.0, [0.75, 0.0, 0.0], 0.2))
        .expect("link 1");
    let l2 = world
        .add_body(sphere_body(0.5, [2.25, 0.0, 0.0], 0.2))
        .expect("link 2");
    let j1 = world
        .add_joint(PyJoint::new(0, pivot, l1, [0.0, 0.0, 0.0]))
        .expect("joint 1");
    let j2 = world
        .add_joint(PyJoint::new(0, l1, l2, [1.5, 0.0, 0.0]))
        .expect("joint 2");
    let mut pipeline = PyPhysicsPipeline::new();
    for _ in 0..60 {
        pipeline.step(&mut world, DT, 2).expect("step");
        for j in [j1, j2] {
            let gap = world.joint_anchor_gap(j).expect("gap");
            assert!(gap < 1e-3, "joint {j} gap {gap}");
        }
    }
    assert_eq!(pipeline.total_steps(), 120);
    let y2 = world.get_body(l2).expect("link 2").position[1];
    assert!(y2 < -0.5, "the chain swings down, y = {y2}");
}
