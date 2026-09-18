// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The implicit gyroscopic stage of [`PhysicsPipeline::step`].
//!
//! The angular velocity of a [`RigidBody`] is a world-frame quantity. A torque-free body that is
//! not isotropic cannot keep it constant: the conserved quantity is the world-frame angular
//! momentum `L = R I Rᵀ ω`. The pipeline used to integrate the orientation with the angular
//! velocity of the start of the step, so the *vector* `L` of an asymmetric body drifted by almost
//! 10 % in ten seconds even though its magnitude `|L|` is conserved to rounding. Every
//! conservation check below therefore compares the vector, never the magnitude.
//!
//! The stage runs once per step, right after the contact solve and before the integration, also
//! when continuous collision detection splits the integration into time-of-impact sub-steps.

use oxiphysics::core::math::{Mat3, Vec3};
use oxiphysics::core::{BodyHandle, PhysicsConfig, Transform};
use oxiphysics::geometry::{BoxShape, Shape, Sphere};
use oxiphysics::pipeline::PhysicsPipeline;
use oxiphysics::rigid::gyroscopic::gyroscopic_implicit_step_world;
use oxiphysics::rigid::{Collider, ColliderSet, RigidBody, RigidBodySet};
use std::sync::Arc;

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

/// Zero gravity and bodies that never fall asleep.
fn free_space_config(ccd_enabled: bool) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3::zeros(),
        time_before_sleep: 1.0e9,
        linear_sleep_threshold: 0.0,
        angular_sleep_threshold: 0.0,
        ccd_enabled,
        ..PhysicsConfig::default()
    }
}

/// Undamped unit-mass box with the given world-frame angular velocity.
fn box_body(omega: Vec3) -> RigidBody {
    let mut body = RigidBody::new(1.0);
    body.local_inertia = box_inertia();
    body.update_world_inertia();
    body.angular_velocity = omega;
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
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
    max_magnitude_drift: f64,
    max_energy_ratio: f64,
    min_energy_ratio: f64,
}

impl Conservation {
    fn new(body: &RigidBody) -> Self {
        Self {
            initial_momentum: world_angular_momentum(body),
            initial_energy: rotational_energy(body),
            max_vector_drift: 0.0,
            max_magnitude_drift: 0.0,
            max_energy_ratio: 1.0,
            min_energy_ratio: 1.0,
        }
    }

    fn record(&mut self, body: &RigidBody) {
        let momentum = world_angular_momentum(body);
        let reference = self.initial_momentum.norm();
        self.max_vector_drift = self
            .max_vector_drift
            .max((momentum - self.initial_momentum).norm() / reference);
        self.max_magnitude_drift = self
            .max_magnitude_drift
            .max((momentum.norm() - reference).abs() / reference);
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

/// A pipeline with the stage switched on or off.
fn make_pipeline(ccd_enabled: bool, implicit_gyroscopic: bool) -> PhysicsPipeline {
    let mut pipeline = PhysicsPipeline::with_config(free_space_config(ccd_enabled));
    pipeline.use_implicit_gyroscopic = implicit_gyroscopic;
    pipeline
}

// ---------------------------------------------------------------------------
// The stage in the step
// ---------------------------------------------------------------------------

#[test]
fn the_pipeline_conserves_the_angular_momentum_vector_of_an_asymmetric_body() {
    for ccd_enabled in [false, true] {
        let mut pipeline = make_pipeline(ccd_enabled, true);
        let mut bodies = RigidBodySet::new();
        let colliders = ColliderSet::new();
        let handle = bodies.insert(box_body(initial_spin()));
        let mut conservation = Conservation::new(body_of(&bodies, handle));
        let mut max_omega_change = 0.0_f64;
        for _ in 0..STEPS {
            pipeline.step(DT, &mut bodies, &colliders);
            let body = body_of(&bodies, handle);
            conservation.record(body);
            max_omega_change =
                max_omega_change.max((body.angular_velocity - initial_spin()).norm());
        }
        conservation.assert_conserved(&format!("PhysicsPipeline::step (ccd {ccd_enabled})"));
        // The world-frame angular velocity genuinely changed: the body tumbles.
        assert!(
            max_omega_change > 1e-2,
            "omega never left its initial value (max change {max_omega_change:.3e})"
        );
    }
}

#[test]
fn with_the_stage_switched_off_omega_stays_constant_and_only_the_vector_drifts() {
    // The pre-fix behaviour, still available on request: omega stays constant, |L| is conserved
    // to rounding and the vector L drifts by almost 10 %.
    for ccd_enabled in [false, true] {
        let mut pipeline = make_pipeline(ccd_enabled, false);
        let mut bodies = RigidBodySet::new();
        let colliders = ColliderSet::new();
        let handle = bodies.insert(box_body(initial_spin()));
        let mut conservation = Conservation::new(body_of(&bodies, handle));
        for _ in 0..STEPS {
            pipeline.step(DT, &mut bodies, &colliders);
            let body = body_of(&bodies, handle);
            assert_eq!(body.angular_velocity, initial_spin());
            conservation.record(body);
        }
        assert!(
            conservation.max_magnitude_drift < 1e-12,
            "|L| drifts by {:.3e}",
            conservation.max_magnitude_drift
        );
        assert!(
            conservation.max_vector_drift > 0.05,
            "L drifts by only {:.3e}",
            conservation.max_vector_drift
        );
    }
}

#[test]
fn the_stage_corrects_omega_before_the_orientation_is_integrated() {
    // One step of a body at the identity orientation: omega becomes the implicit gyroscopic step
    // of the box tensor, and the orientation is advanced with that corrected omega.
    let corrected = gyroscopic_implicit_step_world(initial_spin(), &box_inertia(), DT);
    assert!(
        (corrected - initial_spin()).norm() > 1e-4,
        "the spin must be one the stage changes"
    );
    let mut pipeline = make_pipeline(false, true);
    let mut bodies = RigidBodySet::new();
    let colliders = ColliderSet::new();
    let handle = bodies.insert(box_body(initial_spin()));
    pipeline.step(DT, &mut bodies, &colliders);
    let stepped = body_of(&bodies, handle);
    assert_eq!(stepped.angular_velocity, corrected);
    let mut reference = box_body(corrected);
    reference.integrate_velocity(DT);
    assert_eq!(stepped.transform.rotation, reference.transform.rotation);
}

/// The angular velocity of a ball with the inertia of the asymmetric box after one step on a
/// rough floor under gravity. The ball presses into the floor by 1 mm, which is inside the
/// penetration slop, so the contact solve moves nothing: it only turns friction into a change of
/// the angular velocity.
fn one_step_on_the_floor(initial_omega: Vec3, implicit_gyroscopic: bool) -> Vec3 {
    let mut config = free_space_config(false);
    config.gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut pipeline = PhysicsPipeline::with_config(config);
    pipeline.use_implicit_gyroscopic = implicit_gyroscopic;
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    let mut floor = RigidBody::new_static();
    floor.transform = Transform::from_position(Vec3::new(0.0, -0.5, 0.0));
    let floor_handle = bodies.insert(floor);
    colliders.insert(
        Collider::new(Arc::new(BoxShape::new(Vec3::new(10.0, 0.5, 10.0))))
            .with_body(floor_handle)
            .with_friction(1.0)
            .with_restitution(0.0),
    );

    let mut ball = box_body(initial_omega);
    ball.transform = Transform::from_position(Vec3::new(0.0, 0.499, 0.0));
    let ball_handle = bodies.insert(ball);
    colliders.insert(
        Collider::new(Arc::new(Sphere::new(0.5)))
            .with_body(ball_handle)
            .with_friction(1.0)
            .with_restitution(0.0),
    );

    pipeline.step(DT, &mut bodies, &colliders);
    body_of(&bodies, ball_handle).angular_velocity
}

#[test]
fn the_stage_corrects_the_angular_velocity_the_contact_solve_leaves() {
    // Friction at the contact point changes the angular velocity during the solve. The stage then
    // corrects that solved angular velocity: solving after the correction, or not correcting at
    // all, gives other results.
    let box_tensor = box_inertia();
    let omega = Vec3::new(1.0, 0.1, 0.5);
    let solved = one_step_on_the_floor(omega, false);
    assert!(
        (solved - omega).norm() > 1e-2,
        "friction must change the angular velocity: {solved:?} from {omega:?}"
    );
    let expected = gyroscopic_implicit_step_world(solved, &box_tensor, DT);
    assert!(
        (expected - solved).norm() > 1e-6,
        "the solved spin must be one the stage changes: {solved:?}"
    );
    assert_eq!(one_step_on_the_floor(omega, true), expected);

    let corrected_first = gyroscopic_implicit_step_world(omega, &box_tensor, DT);
    let wrong_order = one_step_on_the_floor(corrected_first, false);
    assert!(
        (wrong_order - expected).norm() > 1e-7,
        "the test cannot tell the stage before the solve from the stage after it"
    );
}

#[test]
fn the_stage_leaves_isotropic_bodies_bit_identical() {
    let spin = Vec3::new(0.3, -0.2, 0.5);
    for ccd_enabled in [false, true] {
        let run = |implicit_gyroscopic: bool| {
            let mut pipeline = make_pipeline(ccd_enabled, implicit_gyroscopic);
            let mut sphere = RigidBody::new(2.0);
            sphere.angular_velocity = spin;
            sphere.velocity = Vec3::new(1.0, 2.0, -3.0);
            let mut bodies = RigidBodySet::new();
            let colliders = ColliderSet::new();
            let handle = bodies.insert(sphere);
            for _ in 0..200 {
                pipeline.step(DT, &mut bodies, &colliders);
            }
            motion_state(body_of(&bodies, handle))
        };
        let with_stage = run(true);
        assert_eq!(with_stage, run(false));
        assert_eq!(with_stage.2, spin);
    }
}

#[test]
fn the_stage_is_configurable_through_the_builder() {
    assert!(PhysicsPipeline::new().use_implicit_gyroscopic);
    assert!(PhysicsPipeline::builder().finish().use_implicit_gyroscopic);
    assert!(
        !PhysicsPipeline::builder()
            .implicit_gyroscopic(false)
            .finish()
            .use_implicit_gyroscopic
    );
    assert!(
        PhysicsPipeline::builder()
            .implicit_gyroscopic(false)
            .implicit_gyroscopic(true)
            .finish()
            .use_implicit_gyroscopic
    );
}

// ---------------------------------------------------------------------------
// Continuous collision detection
// ---------------------------------------------------------------------------

/// Two thin walls 1 m apart (inner faces at x = 19.5 and x = 20.5) and a bullet that ping-pongs
/// between them at 150 m/s, so that nearly every step contains time-of-impact events and the
/// integration of the whole world is split into several sub-steps.
fn add_ping_pong_bullet(bodies: &mut RigidBodySet, colliders: &mut ColliderSet) {
    let thin_box: Arc<dyn Shape> = Arc::new(BoxShape::new(Vec3::new(0.05, 1.5, 1.5)));
    for x in [19.45, 20.55] {
        let mut wall = RigidBody::new_static();
        wall.transform = Transform::from_position(Vec3::new(x, 1.0, 0.0));
        let handle = bodies.insert(wall);
        colliders.insert(
            Collider::new(Arc::clone(&thin_box))
                .with_body(handle)
                .with_friction(0.0)
                .with_restitution(1.0),
        );
    }
    let mut bullet = RigidBody::new(1.0);
    bullet.transform = Transform::from_position(Vec3::new(19.9, 1.0, 0.0));
    bullet.velocity = Vec3::new(150.0, 0.0, 0.0);
    bullet.linear_damping = 0.0;
    bullet.angular_damping = 0.0;
    let handle = bodies.insert(bullet);
    let sphere: Arc<dyn Shape> = Arc::new(Sphere::new(0.05));
    colliders.insert(
        Collider::new(sphere)
            .with_body(handle)
            .with_friction(0.0)
            .with_restitution(1.0),
    );
}

#[test]
fn ccd_sub_steps_rotate_the_bodies_with_one_corrected_omega_per_step() {
    // The tumbling box has no collider, so the bullet's events never touch it, but every event
    // splits the integration of the whole world, the box included. The stage runs once per step
    // before the split: the box ends up exactly where it ends up without any event.
    let mut quiet = make_pipeline(true, true);
    let mut quiet_bodies = RigidBodySet::new();
    let quiet_colliders = ColliderSet::new();
    let quiet_box = quiet_bodies.insert(box_body(initial_spin()));

    let mut busy = make_pipeline(true, true);
    let mut busy_bodies = RigidBodySet::new();
    let mut busy_colliders = ColliderSet::new();
    let busy_box = busy_bodies.insert(box_body(initial_spin()));
    add_ping_pong_bullet(&mut busy_bodies, &mut busy_colliders);

    let mut conservation = Conservation::new(body_of(&busy_bodies, busy_box));
    let mut steps_with_events = 0_usize;
    for _ in 0..STEPS {
        quiet.step(DT, &mut quiet_bodies, &quiet_colliders);
        busy.step(DT, &mut busy_bodies, &busy_colliders);
        if busy.stats.last_ccd_events > 0 {
            steps_with_events += 1;
        }
        conservation.record(body_of(&busy_bodies, busy_box));
    }
    assert!(
        steps_with_events > STEPS / 2,
        "the bullet produced events in only {steps_with_events} of {STEPS} steps"
    );
    conservation.assert_conserved("PhysicsPipeline::step with CCD events");

    let quiet = body_of(&quiet_bodies, quiet_box);
    let busy = body_of(&busy_bodies, busy_box);
    // The rotations about one axis that the sub-steps compose add up to the single rotation of an
    // unsplit step, to rounding. A stage applied once more per sub-step would drift by 1e-3.
    assert!((busy.angular_velocity - quiet.angular_velocity).norm() < 1e-9);
    assert!(
        busy.transform
            .rotation
            .rotation_to(&quiet.transform.rotation)
            .angle()
            < 1e-9
    );
}
