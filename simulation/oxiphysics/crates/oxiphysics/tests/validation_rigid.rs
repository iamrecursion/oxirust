// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Phase 21.1 validation — rigid-body solver against analytical closed-form
//! reference problems.
//!
//! Tests, each anchored to a closed-form physics result:
//!
//! * `test_restitution_ladder` — a ball is dropped from `h = 1 m` under
//!   gravity onto a static body with restitution `e = 0.5`. Rebound apex
//!   heights follow a geometric series `h_n = h_0 * e^(2n)`, so the first
//!   rebound reaches `0.25 m` (the second impact's drop height) and the
//!   next one `0.0625 m`. The baseline `rigid_bounce_height_2` locks the
//!   `0.25 m` value with an 8% relative tolerance.
//!
//! * `test_angular_momentum_conservation` (and its `PhysicsWorld`
//!   variants) — a torque-free box in zero gravity with an initial spin
//!   is integrated for 10 s. The world-frame angular momentum **vector**
//!   `L = R · I_local · Rᵀ · ω` must be conserved within 1% (classical
//!   Euler result). Its magnitude alone would prove nothing: an
//!   integrator that keeps `ω` constant in the world frame conserves
//!   `‖L‖` to rounding while `L` itself turns by almost 10%.
//!
//! * `test_dzhanibekov_*` — a box spun about its intermediate principal
//!   axis with a `1e-3 rad/s` perturbation is unstable and flips over
//!   (the Dzhanibekov / tennis-racket effect). The flip time must match an
//!   independent RK4 integration of Euler's equations, and the flip does
//!   not happen at all without the gyroscopic term.
//!
//! * `test_stacked_box_equilibrium` — three unit boxes stacked along `+y`
//!   under Earth gravity are integrated for 3 s; the stack must reach a
//!   near-rest state with small penetration, small lateral drift, and
//!   small residual kinetic energy.
//!
//! The restitution, stacking and `PhysicsWorld` tests run the full
//! pipeline: contacts come from the attached collider shapes and the
//! collider restitution / friction reach the sequential-impulse solver.

#[path = "regression_harness.rs"]
mod harness;

use oxiphysics::geometry::{BoxShape, Shape, Sphere};
use oxiphysics::rigid::world::PhysicsWorld;
use oxiphysics::rigid::{
    BodyType, Collider, RigidBody, RigidBodySet, angular_momentum, integrate_bodies,
};
use oxiphysics_core::math::{Mat3, Vec3};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Test 1 — restitution ladder (geometric bounce series)
// ---------------------------------------------------------------------------

/// Drop a ball from `h = 1 m` onto a static body with `e = 0.5` and record
/// the rebound apex heights. The analytical values are
/// `h_1 = 1.0 * 0.5^2 = 0.25 m` and `h_2 = 0.25 * 0.5^2 = 0.0625 m`.
///
/// Both colliders carry `e = 0.5` (the contact uses `max(e_a, e_b)`), the
/// ball is a radius-0.5 sphere and the ground a wide box whose top face is
/// the plane `y = -0.5`, so the ball rests with its centre at `y = 0`.
#[test]
fn test_restitution_ladder() {
    // dt = 1/240 s gives sub-millimetre bounce resolution without
    // inflating the test runtime.
    let mut world = PhysicsWorld::new([0.0, -9.81, 0.0], 1.0 / 240.0);

    // Ball — mass 1, sphere collider r = 0.5. Damping disabled so kinetic
    // energy is only lost through the coefficient of restitution.
    let mut ball = RigidBody::new(1.0);
    ball.transform.position = Vec3::new(0.0, 1.0, 0.0);
    ball.linear_damping = 0.0;
    ball.angular_damping = 0.0;
    let ball_shape: Arc<dyn Shape> = Arc::new(Sphere::new(0.5));
    ball.set_inertia_from_shape(ball_shape.as_ref());
    ball.update_world_inertia();
    let ball_h = world.add_rigid_body(ball);
    world.add_collider(
        Collider::new(ball_shape).with_restitution(0.5),
        Some(ball_h),
    );

    // Static ground: top face at y = -0.5, i.e. the ball touches it when its
    // centre reaches y = 0 (1 m below the release height).
    let mut ground = RigidBody::new_static();
    ground.transform.position = Vec3::new(0.0, -1.0, 0.0);
    let ground_h = world.add_rigid_body(ground);
    world.add_collider(
        Collider::new(Arc::new(BoxShape::new(Vec3::new(10.0, 0.5, 10.0)))).with_restitution(0.5),
        Some(ground_h),
    );

    // Step the world for 4 s (960 steps): long enough for several bounces.
    // An apex is the step at which the vertical velocity turns from upward to
    // downward; apexes lower than 1 cm are contact jitter, not bounces.
    let mut prev_vy = 0.0_f64;
    let mut peaks: Vec<f64> = Vec::new();
    for _ in 0..960 {
        world.step();
        let b = world
            .get_body(ball_h)
            .expect("ball handle inserted above must exist");
        let y = b.transform.position.y;
        let vy = b.velocity.y;
        if prev_vy > 0.0 && vy <= 0.0 && y > 0.01 {
            peaks.push(y);
        }
        prev_vy = vy;
    }

    assert!(
        peaks.len() >= 2,
        "expected at least two rebound apexes, got {}: {peaks:?}",
        peaks.len()
    );
    let h1 = peaks[0];
    let h2 = peaks[1];

    let baseline = harness::load_baseline("rigid_bounce_height_2")
        .expect("baseline `rigid_bounce_height_2` must exist");
    eprintln!(
        "[test_restitution_ladder] rebound apexes = {peaks:?} (analytical 0.25, 0.0625 m; baseline tol_rel = {:.1}%)",
        baseline.tolerance_rel * 100.0
    );
    assert_close!(h1, baseline);
    let expected_h2 = h1 * 0.25;
    assert!(
        (h2 - expected_h2).abs() < 0.15 * expected_h2,
        "second rebound {h2} must be e^2 = 0.25 of the first ({expected_h2})"
    );
}

// ---------------------------------------------------------------------------
// Test 2 — angular momentum conservation (torque-free spin)
// ---------------------------------------------------------------------------

/// Principal moments of a solid box with side lengths `0.2 × 0.4 × 0.3 m`
/// and uniform unit mass.
///
/// The inertia tensor of a solid box of full extents `(sx, sy, sz)` and mass
/// `m` has diagonal components `m/12 · (sy² + sz²)`, `m/12 · (sx² + sz²)`,
/// `m/12 · (sx² + sy²)`. For `(0.2, 0.4, 0.3)` at `m = 1`:
///
/// * `Ixx = (0.16 + 0.09) / 12 = 0.020833…` (largest)
/// * `Iyy = (0.04 + 0.09) / 12 = 0.010833…` (smallest)
/// * `Izz = (0.04 + 0.16) / 12 = 0.016667…` (intermediate)
///
/// All three differ, so a torque-free box does not spin at a constant rate:
/// its world-frame angular velocity precesses.
const BOX_MOMENTS: [f64; 3] = [
    0.020_833_333_333_333_332,
    0.010_833_333_333_333_334,
    0.016_666_666_666_666_666,
];

/// Undamped unit-mass box `0.2 × 0.4 × 0.3 m` with the given world-frame
/// angular velocity.
fn tumbling_box(omega: Vec3) -> RigidBody {
    let mut body = RigidBody::new(1.0);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body.local_inertia =
        Mat3::from_diagonal(&Vec3::new(BOX_MOMENTS[0], BOX_MOMENTS[1], BOX_MOMENTS[2]));
    body.update_world_inertia();
    body.angular_velocity = omega;
    body
}

/// World-frame angular momentum `L = R · I_local · Rᵀ · ω`.
///
/// The stored angular velocity is a world-frame vector, so the inertia
/// tensor that pairs with it is the world-frame one, `R · I_local · Rᵀ`;
/// `R · I_local · ω` (which mixes frames) is not an angular momentum.
fn world_angular_momentum(body: &RigidBody) -> Vec3 {
    let rotation = body.transform.rotation.to_rotation_matrix();
    rotation.matrix() * body.local_inertia * rotation.matrix().transpose() * body.angular_velocity
}

/// Accumulates what a torque-free tumbling body does over a run: angular
/// momentum drift (vector and magnitude), rotational energy, and the
/// intermediate-axis flip (`z` is the intermediate principal axis of
/// [`BOX_MOMENTS`]).
struct TumbleTracker {
    initial_momentum: Vec3,
    initial_energy: f64,
    /// First time the body-frame `z` component of `ω` is negative.
    flip_time: Option<f64>,
    /// Smallest cosine between the body's `z` axis (in world coordinates) and
    /// `L`; `+1` for the initial spin, `-1` after a full flip.
    min_axis_cosine: f64,
    /// Largest `‖L − L₀‖ / ‖L₀‖`.
    max_vector_drift: f64,
    /// Largest `| ‖L‖ − ‖L₀‖ | / ‖L₀‖`.
    max_magnitude_drift: f64,
    min_energy_ratio: f64,
    max_energy_ratio: f64,
}

impl TumbleTracker {
    fn new(body: &RigidBody) -> Self {
        let initial_momentum = world_angular_momentum(body);
        Self {
            initial_momentum,
            initial_energy: 0.5 * initial_momentum.dot(&body.angular_velocity),
            flip_time: None,
            min_axis_cosine: 1.0,
            max_vector_drift: 0.0,
            max_magnitude_drift: 0.0,
            min_energy_ratio: 1.0,
            max_energy_ratio: 1.0,
        }
    }

    /// Record the state of `body` at simulation time `time`.
    fn record(&mut self, body: &RigidBody, time: f64) {
        let momentum = world_angular_momentum(body);
        let initial_norm = self.initial_momentum.norm();
        self.max_vector_drift = self
            .max_vector_drift
            .max((momentum - self.initial_momentum).norm() / initial_norm);
        self.max_magnitude_drift = self
            .max_magnitude_drift
            .max((momentum.norm() - initial_norm).abs() / initial_norm);
        let energy_ratio = 0.5 * momentum.dot(&body.angular_velocity) / self.initial_energy;
        self.min_energy_ratio = self.min_energy_ratio.min(energy_ratio);
        self.max_energy_ratio = self.max_energy_ratio.max(energy_ratio);
        let intermediate_axis = body.transform.rotation * Vec3::z();
        self.min_axis_cosine = self
            .min_axis_cosine
            .min(intermediate_axis.dot(&momentum) / momentum.norm());
        let omega_body = body
            .transform
            .rotation
            .inverse_transform_vector(&body.angular_velocity);
        if self.flip_time.is_none() && omega_body.z < 0.0 {
            self.flip_time = Some(time);
        }
    }
}

const TUMBLE_DT: f64 = 0.01;
const TUMBLE_STEPS: usize = 1000;

fn tumble_spin() -> Vec3 {
    Vec3::new(1.0, 0.1, 0.0)
}

/// A box with side lengths `0.2 × 0.4 × 0.3 m` and uniform unit mass is
/// given an initial angular velocity `ω = (1.0, 0.1, 0.0) rad/s` under
/// zero gravity and zero damping. Stepping only the integrator
/// (`integrate_bodies`) for 1000 × 0.01 s = 10 s must conserve the
/// world-frame angular momentum **vector** `L = R · I_local · Rᵀ · ω`
/// within 1% of its length.
///
/// Checking only `‖L‖` would be vacuous: an integrator without a
/// gyroscopic term keeps `ω` constant in the world frame, so `Rᵀ ω` stays
/// constant and `‖L‖` is conserved to rounding while the vector `L` turns
/// about `ω` and drifts by 9.5% (see
/// `test_angular_momentum_vector_drifts_without_gyroscopic_term`).
#[test]
fn test_angular_momentum_conservation() {
    let mut set = RigidBodySet::new();
    let handle = set.insert(tumbling_box(tumble_spin()));
    let mut tracker =
        TumbleTracker::new(set.get(handle).expect("handle inserted above must exist"));
    assert!(
        tracker.initial_momentum.norm() > 0.0,
        "initial angular momentum must be non-zero"
    );

    let gravity = Vec3::zeros();
    for step in 1..=TUMBLE_STEPS {
        integrate_bodies(&mut set, TUMBLE_DT, &gravity);
        let body = set.get(handle).expect("handle inserted above must exist");
        tracker.record(body, step as f64 * TUMBLE_DT);
    }

    eprintln!(
        "[test_angular_momentum_conservation] steps = {TUMBLE_STEPS}, dt = {TUMBLE_DT} s, \
         max vector drift = {:.3e}, max magnitude drift = {:.3e}, \
         rotational energy in [{:.6}, {:.6}] of its initial value",
        tracker.max_vector_drift,
        tracker.max_magnitude_drift,
        tracker.min_energy_ratio,
        tracker.max_energy_ratio,
    );
    assert!(
        tracker.max_vector_drift < 0.01,
        "angular momentum vector drifts by {:.3e} ({:.3}%), more than the 1% tolerance",
        tracker.max_vector_drift,
        tracker.max_vector_drift * 100.0,
    );
    assert!(tracker.max_magnitude_drift < 0.01);
    // The implicit gyroscopic step is slightly dissipative and never gains energy.
    assert!(tracker.max_energy_ratio <= 1.0 + 1e-12);
    assert!(tracker.min_energy_ratio > 0.99);

    // The library's own helper agrees with the formula above.
    let body = set.get(handle).expect("handle inserted above must exist");
    assert!((angular_momentum(body) - world_angular_momentum(body)).norm() < 1e-12);
}

/// The same box and spin through `PhysicsWorld::step` with the default
/// solver configuration, which has the implicit gyroscopic stage enabled.
#[test]
fn test_angular_momentum_conservation_physics_world() {
    let mut world = PhysicsWorld::new([0.0, 0.0, 0.0], TUMBLE_DT);
    assert!(world.solver_config.use_implicit_gyroscopic);
    let handle = world.add_rigid_body(tumbling_box(tumble_spin()));
    let mut tracker = TumbleTracker::new(world.get_body(handle).expect("body was just added"));
    for step in 1..=TUMBLE_STEPS {
        world.step();
        let body = world.get_body(handle).expect("body was just added");
        tracker.record(body, step as f64 * TUMBLE_DT);
    }
    assert!(
        tracker.max_vector_drift < 0.01,
        "angular momentum vector drifts by {:.3e}",
        tracker.max_vector_drift
    );
    assert!(tracker.max_energy_ratio <= 1.0 + 1e-12);
    assert!(tracker.min_energy_ratio > 0.99);
}

/// The companion of the tests above: with the gyroscopic stage switched off
/// the world-frame `ω` never changes, the magnitude of `L` is conserved to
/// rounding, and the vector `L` drifts by almost 10% — which is why the
/// vector, not the magnitude, is the quantity to assert on.
#[test]
fn test_angular_momentum_vector_drifts_without_gyroscopic_term() {
    let mut world = PhysicsWorld::new([0.0, 0.0, 0.0], TUMBLE_DT);
    world.solver_config.use_implicit_gyroscopic = false;
    let handle = world.add_rigid_body(tumbling_box(tumble_spin()));
    let mut tracker = TumbleTracker::new(world.get_body(handle).expect("body was just added"));
    for step in 1..=TUMBLE_STEPS {
        world.step();
        let body = world.get_body(handle).expect("body was just added");
        assert_eq!(body.angular_velocity, tumble_spin());
        tracker.record(body, step as f64 * TUMBLE_DT);
    }
    assert!(
        tracker.max_magnitude_drift < 1e-12,
        "|L| drifts by {:.3e}",
        tracker.max_magnitude_drift
    );
    assert!(
        tracker.max_vector_drift > 0.05,
        "L drifts by only {:.3e}",
        tracker.max_vector_drift
    );
}

// ---------------------------------------------------------------------------
// Test 2b — Dzhanibekov effect (intermediate-axis instability)
// ---------------------------------------------------------------------------

const DZHANIBEKOV_SPIN: f64 = 3.0;
const DZHANIBEKOV_PERTURBATION: f64 = 1.0e-3;
const DZHANIBEKOV_DT: f64 = 0.005;
const DZHANIBEKOV_HORIZON: f64 = 20.0;

/// Spin of `3 rad/s` about the intermediate axis (`z`) with a `1e-3 rad/s`
/// perturbation along `x`.
fn dzhanibekov_spin() -> Vec3 {
    Vec3::new(DZHANIBEKOV_PERTURBATION, 0.0, DZHANIBEKOV_SPIN)
}

/// Time at which the intermediate-axis component of the body-frame angular
/// velocity of a torque-free [`tumbling_box`] first changes sign, from an RK4
/// integration of Euler's equations `I ω̇ = (I ω) × ω` — independent of the
/// engine's integrator. `None` if there is no sign change before `horizon`.
fn euler_flip_time(omega0: Vec3, horizon: f64) -> Option<f64> {
    let inertia = Mat3::from_diagonal(&Vec3::new(BOX_MOMENTS[0], BOX_MOMENTS[1], BOX_MOMENTS[2]));
    let inverse = Mat3::from_diagonal(&Vec3::new(
        1.0 / BOX_MOMENTS[0],
        1.0 / BOX_MOMENTS[1],
        1.0 / BOX_MOMENTS[2],
    ));
    let rhs = |w: Vec3| inverse * (inertia * w).cross(&w);
    let h = 1.0e-4;
    let mut omega = omega0;
    let mut time = 0.0;
    while time < horizon {
        let k1 = rhs(omega);
        let k2 = rhs(omega + k1 * (0.5 * h));
        let k3 = rhs(omega + k2 * (0.5 * h));
        let k4 = rhs(omega + k3 * h);
        let next = omega + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
        if omega.z > 0.0 && next.z <= 0.0 {
            return Some(time + h * omega.z / (omega.z - next.z));
        }
        omega = next;
        time += h;
    }
    None
}

/// Check a Dzhanibekov run: the flip happens within the horizon at the time
/// Euler's equations give, the intermediate axis ends up reversed with
/// respect to `L`, and the integrator still conserves `L` and never gains
/// energy.
fn check_dzhanibekov_flip(tracker: &TumbleTracker) {
    let reference = euler_flip_time(dzhanibekov_spin(), 2.0 * DZHANIBEKOV_HORIZON)
        .expect("Euler's equations flip the box");
    assert!(
        reference < DZHANIBEKOV_HORIZON,
        "reference flip at {reference:.3} s is outside the {DZHANIBEKOV_HORIZON} s horizon"
    );
    let flip = tracker
        .flip_time
        .expect("the box must flip over within 20 s");
    eprintln!(
        "[test_dzhanibekov] flip at {flip:.3} s (Euler's equations: {reference:.3} s), \
         min cos(intermediate axis, L) = {:.5}, max L drift = {:.3e}, \
         rotational energy in [{:.5}, {:.6}] of its initial value",
        tracker.min_axis_cosine,
        tracker.max_vector_drift,
        tracker.min_energy_ratio,
        tracker.max_energy_ratio,
    );
    assert!(
        (flip - reference).abs() < 0.01 * reference,
        "flip at {flip:.3} s, Euler's equations give {reference:.3} s"
    );
    assert!(
        tracker.min_axis_cosine < -0.99,
        "the intermediate axis must end up anti-parallel to L, cosine {:.4}",
        tracker.min_axis_cosine
    );
    assert!(
        tracker.max_vector_drift < 0.01,
        "angular momentum vector drifts by {:.3e}",
        tracker.max_vector_drift
    );
    assert!(tracker.max_energy_ratio <= 1.0 + 1e-12);
    assert!(tracker.min_energy_ratio > 0.97);
}

/// `integrate_bodies` flips the box at the time Euler's equations give.
#[test]
fn test_dzhanibekov_flip_with_integrate_bodies() {
    let mut set = RigidBodySet::new();
    let handle = set.insert(tumbling_box(dzhanibekov_spin()));
    let mut tracker =
        TumbleTracker::new(set.get(handle).expect("handle inserted above must exist"));
    let gravity = Vec3::zeros();
    let steps = (DZHANIBEKOV_HORIZON / DZHANIBEKOV_DT).round() as usize;
    for step in 1..=steps {
        integrate_bodies(&mut set, DZHANIBEKOV_DT, &gravity);
        let body = set.get(handle).expect("handle inserted above must exist");
        tracker.record(body, step as f64 * DZHANIBEKOV_DT);
    }
    check_dzhanibekov_flip(&tracker);
}

/// `PhysicsWorld::step` with the default configuration flips the box too, and
/// the flip is a property of the gyroscopic term: with the stage disabled the
/// box keeps spinning about the (unstable) axis forever.
#[test]
fn test_dzhanibekov_flip_with_physics_world() {
    let steps = (DZHANIBEKOV_HORIZON / DZHANIBEKOV_DT).round() as usize;
    let run = |gyroscopic: bool| {
        let mut world = PhysicsWorld::new([0.0, 0.0, 0.0], DZHANIBEKOV_DT);
        world.solver_config.use_implicit_gyroscopic = gyroscopic;
        let handle = world.add_rigid_body(tumbling_box(dzhanibekov_spin()));
        let mut tracker = TumbleTracker::new(world.get_body(handle).expect("body was just added"));
        for step in 1..=steps {
            world.step();
            let body = world.get_body(handle).expect("body was just added");
            tracker.record(body, step as f64 * DZHANIBEKOV_DT);
        }
        tracker
    };
    check_dzhanibekov_flip(&run(true));

    let frozen = run(false);
    assert!(
        frozen.flip_time.is_none(),
        "without the gyroscopic term the box cannot flip"
    );
    assert!(frozen.min_axis_cosine > 0.999);
}

// ---------------------------------------------------------------------------
// Test 3 — stacked box equilibrium
// ---------------------------------------------------------------------------

/// Three unit boxes stacked along `+y` under Earth gravity must settle to
/// near rest within 3 s. The test asserts:
///
/// * total downward penetration below the expected contact plane `< 2e-3 m`
///   (i.e. the stack has not sunk into the ground);
/// * lateral drift from the spawn column `< 5e-3 m` per body;
/// * per-body translational kinetic energy `< 1e-3 J` (i.e. the stack has
///   come to rest).
///
/// Each body carries a unit `BoxShape` collider (half-extent 0.5) and the
/// ground is a wide static box whose top face is the plane `y = 0`, so the
/// box-box manifolds are full 4-point face contacts.
#[test]
fn test_stacked_box_equilibrium() {
    let mut world = PhysicsWorld::new([0.0, -9.81, 0.0], 1.0 / 100.0);

    // Three unit-mass unit boxes resting at y = 0.5, 1.5, 2.5.
    let mut handles = Vec::with_capacity(3);
    let expected_rest_y = [0.5_f64, 1.5, 2.5];
    for &y in &expected_rest_y {
        let mut b = RigidBody::new(1.0);
        b.transform.position = Vec3::new(0.0, y + 0.5, 0.0); // drop from +0.5 m above rest
        b.linear_damping = 0.0;
        b.angular_damping = 0.0;
        let shape: Arc<dyn Shape> = Arc::new(BoxShape::new(Vec3::new(0.5, 0.5, 0.5)));
        b.set_inertia_from_shape(shape.as_ref());
        b.update_world_inertia();
        let h = world.add_rigid_body(b);
        world.add_collider(Collider::new(shape), Some(h));
        handles.push(h);
    }

    // Static ground box: top face at y = 0, aligned with expected_rest_y[0] = 0.5.
    let mut ground = RigidBody::new_static();
    ground.transform.position = Vec3::new(0.0, -0.5, 0.0);
    let ground_h = world.add_rigid_body(ground);
    world.add_collider(
        Collider::new(Arc::new(BoxShape::new(Vec3::new(10.0, 0.5, 10.0)))),
        Some(ground_h),
    );

    // Step the world for 3 s (300 steps).
    for _ in 0..300 {
        world.step();
    }

    // Sanity check every body against the analytical resting configuration.
    for (i, &h) in handles.iter().enumerate() {
        let b = world
            .get_body(h)
            .expect("stacked body handle inserted above must exist");
        assert_eq!(
            b.body_type,
            BodyType::Dynamic,
            "stacked body {i} was demoted to non-dynamic type"
        );
        let pos = b.transform.position;
        let vel = b.velocity;
        let ke = 0.5 * b.mass * vel.norm_squared();

        let penetration = (expected_rest_y[i] - pos.y).max(0.0);
        let lateral = (pos.x * pos.x + pos.z * pos.z).sqrt();

        eprintln!(
            "[test_stacked_box_equilibrium] body {i}: pos = ({:.6}, {:.6}, {:.6}), \
             vel = ({:.6}, {:.6}, {:.6}), penetration = {penetration:.3e}, \
             lateral = {lateral:.3e}, KE = {ke:.3e}",
            pos.x, pos.y, pos.z, vel.x, vel.y, vel.z,
        );

        assert!(
            penetration < 2e-3,
            "body {i} penetration {penetration:.3e} m exceeds 2e-3 m tolerance \
             (expected y ~ {:.3}, got y = {:.6})",
            expected_rest_y[i],
            pos.y,
        );
        assert!(
            lateral < 5e-3,
            "body {i} lateral drift {lateral:.3e} m exceeds 5e-3 m tolerance \
             (pos = ({:.6}, {:.6}, {:.6}))",
            pos.x,
            pos.y,
            pos.z,
        );
        assert!(
            ke < 1e-3,
            "body {i} kinetic energy {ke:.3e} J exceeds 1e-3 J tolerance \
             (vel = ({:.6}, {:.6}, {:.6}))",
            vel.x,
            vel.y,
            vel.z,
        );
    }
}
