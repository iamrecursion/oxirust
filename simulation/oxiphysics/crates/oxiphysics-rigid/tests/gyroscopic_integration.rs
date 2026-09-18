// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The implicit gyroscopic stage of the rigid-body integrators.
//!
//! The angular velocity of a [`RigidBody`] is a world-frame quantity. A torque-free body that is not
//! isotropic cannot keep it constant: the conserved quantity is the world-frame angular momentum
//! `L = R I Rᵀ ω`. The stage under test advances `ω` by one backward-Euler step of Euler's equation
//! with the *full* inertia tensor between `integrate_forces` and `integrate_velocity`; it is part of
//! [`integrate_bodies`] and of `PhysicsWorld::step` / `step_small_steps` (on by default).
//!
//! Note that with `ω` held constant in the world frame the *magnitude* `|R I Rᵀ ω|` is conserved
//! to rounding (`Rᵀ ω` is constant when the body turns about `ω`): only the angular-momentum
//! *vector* tells the correct integrator from the old one, so every conservation check below
//! compares vectors.

use oxiphysics_core::math::{Mat3, Quat, Unit, Vec3};
use oxiphysics_rigid::gyroscopic::{gyroscopic_implicit_step, gyroscopic_implicit_step_world};
use oxiphysics_rigid::world::{PhysicsWorld, SolverConfig};
use oxiphysics_rigid::{BodyState, RigidBody, RigidBodySet, integrate_bodies};

/// Principal moments of a 0.2 x 0.4 x 0.3 m box of unit mass. They are all different, so a free
/// body tumbles instead of spinning at a constant rate.
const BOX_MOMENTS: [f64; 3] = [
    0.020_833_333_333_333_332,
    0.010_833_333_333_333_334,
    0.016_666_666_666_666_666,
];

fn box_inertia() -> Mat3 {
    Mat3::from_diagonal(&Vec3::new(BOX_MOMENTS[0], BOX_MOMENTS[1], BOX_MOMENTS[2]))
}

/// Undamped unit-mass box with the given world-frame angular velocity.
fn box_body(omega: Vec3) -> RigidBody {
    let mut body = RigidBody::new(1.0);
    body.local_inertia = box_inertia();
    body.update_world_inertia();
    body.angular_velocity = omega;
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

/// Relative distance `‖a − reference‖ / ‖reference‖`.
fn relative_distance(a: &Vec3, reference: &Vec3) -> f64 {
    (a - reference).norm() / reference.norm()
}

const DT: f64 = 0.01;
const STEPS: usize = 1000;

fn initial_spin() -> Vec3 {
    Vec3::new(1.0, 0.1, 0.0)
}

// ---------------------------------------------------------------------------
// integrate_bodies
// ---------------------------------------------------------------------------

#[test]
fn integrate_bodies_conserves_the_angular_momentum_vector_of_an_asymmetric_body() {
    let mut set = RigidBodySet::new();
    let handle = set.insert(box_body(initial_spin()));
    let (initial_l, initial_energy) = {
        let body = set.get(handle).expect("inserted body must exist");
        (world_angular_momentum(body), rotational_energy(body))
    };
    let gravity = Vec3::zeros();
    let mut max_drift = 0.0_f64;
    let (mut min_ratio, mut max_ratio) = (1.0_f64, 1.0_f64);
    for _ in 0..STEPS {
        integrate_bodies(&mut set, DT, &gravity);
        let body = set.get(handle).expect("inserted body must exist");
        max_drift = max_drift.max(relative_distance(&world_angular_momentum(body), &initial_l));
        let ratio = rotational_energy(body) / initial_energy;
        min_ratio = min_ratio.min(ratio);
        max_ratio = max_ratio.max(ratio);
    }
    // Backward Euler is first order: 4.7e-4 over 10 s at this step size.
    assert!(
        max_drift < 2.0e-3,
        "angular momentum vector drifted by {max_drift:.3e}"
    );
    // ... and slightly dissipative, never energy-gaining.
    assert!(
        max_ratio <= 1.0 + 1e-12,
        "rotational energy grew to {max_ratio}"
    );
    assert!(
        min_ratio > 0.999,
        "rotational energy fell to {min_ratio} of its initial value"
    );
}

#[test]
fn without_the_gyroscopic_stage_only_the_angular_momentum_vector_drifts() {
    // The pre-fix integrator: forces, then positions, with omega constant in the world frame.
    let mut body = box_body(initial_spin());
    let initial_l = world_angular_momentum(&body);
    let gravity = Vec3::zeros();
    let mut max_vector_drift = 0.0_f64;
    let mut max_magnitude_drift = 0.0_f64;
    for _ in 0..STEPS {
        body.integrate_forces(DT, &gravity);
        body.integrate_velocity(DT);
        let l = world_angular_momentum(&body);
        max_vector_drift = max_vector_drift.max(relative_distance(&l, &initial_l));
        max_magnitude_drift =
            max_magnitude_drift.max((l.norm() - initial_l.norm()).abs() / initial_l.norm());
    }
    // The magnitude is conserved to rounding (which made a magnitude test vacuous) ...
    assert!(
        max_magnitude_drift < 1e-12,
        "magnitude drift {max_magnitude_drift:.3e}"
    );
    // ... while the vector rotates by almost 10 % of its length.
    assert!(
        max_vector_drift > 0.05,
        "vector drift {max_vector_drift:.3e}"
    );
}

#[test]
fn products_of_inertia_are_honoured_by_the_gyroscopic_stage() {
    // The same physical body described twice. Body A has principal axes aligned with its frame
    // and a tilted orientation; body B has the identity orientation and the full inertia tensor
    // R0 D R0^T with products of inertia. Their world inertia tensors agree, so their world-frame
    // angular velocities must agree for all time. Reading only the diagonal of `local_inertia`
    // (as the pre-fix `PhysicsWorld` block did) gives body B the wrong dynamics.
    let tilt = Quat::from_axis_angle(&Unit::new_normalize(Vec3::new(1.0, 2.0, 3.0)), 0.7);
    let tilt_matrix = *tilt.to_rotation_matrix().matrix();
    let omega = Vec3::new(0.8, -0.3, 1.1);

    let mut a = box_body(omega);
    a.transform.rotation = tilt;
    a.update_world_inertia();
    let mut b = box_body(omega);
    b.local_inertia = tilt_matrix * box_inertia() * tilt_matrix.transpose();
    b.update_world_inertia();

    let products = b.local_inertia[(0, 1)]
        .abs()
        .max(b.local_inertia[(0, 2)].abs())
        .max(b.local_inertia[(1, 2)].abs());
    assert!(products > 1e-3, "tensor of body B must not be diagonal");
    assert!((a.world_inertia() - b.world_inertia()).norm() < 1e-15);

    let mut set = RigidBodySet::new();
    let ha = set.insert(a);
    let hb = set.insert(b);
    let initial_l = world_angular_momentum(set.get(ha).expect("body A must exist"));
    let gravity = Vec3::zeros();
    for _ in 0..300 {
        integrate_bodies(&mut set, DT, &gravity);
    }
    let a = set.get(ha).expect("body A must exist");
    let b = set.get(hb).expect("body B must exist");
    assert!(
        (a.angular_velocity - b.angular_velocity).norm() < 1e-9,
        "omega A {:?} vs omega B {:?}",
        a.angular_velocity,
        b.angular_velocity
    );
    assert!((a.world_inertia() - b.world_inertia()).norm() < 1e-9);
    for body in [a, b] {
        // This spin is close to the unstable intermediate axis, so the first-order error of the
        // backward-Euler step is larger than for the box of the other tests (2.3e-3 over 3 s).
        let drift = relative_distance(&world_angular_momentum(body), &initial_l);
        assert!(drift < 5e-3, "angular momentum drift {drift:.3e}");
    }
    // The gyroscopic stage really moved omega (it is not a trivial pass-through).
    assert!((a.angular_velocity - omega).norm() > 1e-3);
}

#[test]
fn a_merged_body_with_products_of_inertia_conserves_its_angular_momentum() {
    // `RigidBody::merge` shifts the parts' tensors to the common centre of mass with the
    // parallel-axis theorem, which produces products of inertia: a real full tensor.
    let mut a = RigidBody::new(2.0);
    a.transform.position = Vec3::new(1.0, 0.0, 0.0);
    a.angular_velocity = Vec3::new(0.0, 1.5, 0.0);
    let mut b = RigidBody::new(1.0);
    b.transform.position = Vec3::new(-0.5, 0.8, 0.3);
    b.velocity = Vec3::new(0.0, 0.0, 2.0);
    let merged = RigidBody::merge(&a, &b);
    let products = merged.local_inertia[(0, 1)]
        .abs()
        .max(merged.local_inertia[(0, 2)].abs())
        .max(merged.local_inertia[(1, 2)].abs());
    assert!(products > 0.1, "merged tensor is diagonal: {products}");

    let mut set = RigidBodySet::new();
    let initial_omega = merged.angular_velocity;
    let handle = set.insert(merged);
    let initial = world_angular_momentum(set.get(handle).expect("merged body must exist"));
    assert!(initial.norm() > 0.1);
    let gravity = Vec3::zeros();
    let mut max_drift = 0.0_f64;
    for _ in 0..500 {
        integrate_bodies(&mut set, DT, &gravity);
        let l = world_angular_momentum(set.get(handle).expect("merged body must exist"));
        max_drift = max_drift.max(relative_distance(&l, &initial));
    }
    assert!(max_drift < 2e-3, "angular momentum drift {max_drift:.3e}");
    // The body does tumble: the world-frame omega is not constant.
    let omega = set
        .get(handle)
        .expect("merged body must exist")
        .angular_velocity;
    assert!(
        (omega - initial_omega).norm() > 1e-2,
        "omega {omega:?} stayed at {initial_omega:?}"
    );
}

#[test]
fn bodies_without_a_gyroscopic_torque_keep_their_angular_velocity() {
    let gravity = Vec3::zeros();
    let mut set = RigidBodySet::new();
    // Isotropic (the default inertia of `RigidBody::new` is the mass times the identity).
    let sphere_spin = Vec3::new(0.3, -0.2, 0.5);
    let mut sphere = RigidBody::new(2.0);
    sphere.angular_velocity = sphere_spin;
    let sphere = set.insert(sphere);
    // A box spinning exactly about a principal axis.
    let axis_spin = Vec3::new(0.0, 0.0, 2.0);
    let axis = set.insert(box_body(axis_spin));
    // A box at rest.
    let rest = set.insert(box_body(Vec3::zeros()));
    for _ in 0..200 {
        integrate_bodies(&mut set, DT, &gravity);
    }
    for (handle, expected) in [(sphere, sphere_spin), (axis, axis_spin)] {
        let body = set.get(handle).expect("body must exist");
        assert!(
            (body.angular_velocity - expected).norm() <= 1e-12 * expected.norm(),
            "omega {:?} changed from {expected:?}",
            body.angular_velocity
        );
    }
    assert_eq!(
        set.get(rest).expect("body must exist").angular_velocity,
        Vec3::zeros()
    );
}

#[test]
fn static_kinematic_and_sleeping_bodies_are_not_corrected() {
    let omega = initial_spin();
    let mut fixed = RigidBody::new_static();
    fixed.local_inertia = box_inertia();
    fixed.angular_velocity = omega;
    let mut kinematic = RigidBody::new_kinematic();
    kinematic.local_inertia = box_inertia();
    kinematic.angular_velocity = omega;
    let mut asleep = box_body(omega);
    asleep.state = BodyState::Sleeping;
    for body in [&mut fixed, &mut kinematic, &mut asleep] {
        body.integrate_gyroscopic(DT);
        assert_eq!(body.angular_velocity, omega);
    }
    // An awake dynamic body is: delta omega = -dt I^-1 (omega x I omega), about 6e-4 here.
    let mut awake = box_body(omega);
    awake.integrate_gyroscopic(DT);
    let correction = (awake.angular_velocity - omega).norm();
    assert!(
        (5.0e-4..7.0e-4).contains(&correction),
        "correction {correction:.3e}"
    );
}

// ---------------------------------------------------------------------------
// The step functions
// ---------------------------------------------------------------------------

#[test]
fn principal_moment_entry_point_matches_the_world_frame_step() {
    let rotation = Quat::from_axis_angle(&Unit::new_normalize(Vec3::new(-1.0, 0.5, 2.0)), 1.1);
    let q = rotation.quaternion();
    let omega = [0.4, -1.3, 0.9];
    let dt = 0.002;
    let from_moments = gyroscopic_implicit_step(omega, [q.i, q.j, q.k, q.w], BOX_MOMENTS, dt);

    let r = *rotation.to_rotation_matrix().matrix();
    let inertia_world = r * box_inertia() * r.transpose();
    let omega_vec = Vec3::new(omega[0], omega[1], omega[2]);
    let from_tensor = gyroscopic_implicit_step_world(omega_vec, &inertia_world, dt);
    for axis in 0..3 {
        assert!(
            (from_moments[axis] - from_tensor[axis]).abs() < 1e-12,
            "axis {axis}: {} vs {}",
            from_moments[axis],
            from_tensor[axis]
        );
    }

    // To first order the correction is the explicit gyroscopic acceleration:
    // delta omega = -dt I_w^-1 (omega x I_w omega).
    let explicit = -dt
        * inertia_world
            .try_inverse()
            .expect("the box inertia tensor is regular")
        * omega_vec.cross(&(inertia_world * omega_vec));
    let actual = from_tensor - omega_vec;
    assert!(
        (actual - explicit).norm() < 0.02 * explicit.norm(),
        "correction {actual:?} vs first-order estimate {explicit:?}"
    );
}

#[test]
fn the_returned_angular_velocity_solves_the_backward_euler_equation() {
    // f(w1) = I_w (w1 - w0)/dt + w1 x (I_w w1) = 0 for a full (non-diagonal) tensor.
    let rotation = Quat::from_axis_angle(&Unit::new_normalize(Vec3::new(0.3, -1.0, 0.8)), 0.9);
    let r = *rotation.to_rotation_matrix().matrix();
    let inertia_world = r * Mat3::from_diagonal(&Vec3::new(1.0, 2.5, 4.0)) * r.transpose();
    for &(omega_scale, dt) in &[(0.5, 0.01), (2.0, 0.01), (5.0, 0.02), (20.0, 0.005)] {
        let w0 = Vec3::new(0.7, -0.2, 0.4).normalize() * omega_scale;
        let w1 = gyroscopic_implicit_step_world(w0, &inertia_world, dt);
        let residual = inertia_world * (w1 - w0) / dt + w1.cross(&(inertia_world * w1));
        let scale = inertia_world.norm() * w0.norm() * w0.norm();
        assert!(
            residual.norm() < 1e-9 * scale,
            "|w0| = {omega_scale}, dt = {dt}: residual {:.3e} (scale {scale:.3e})",
            residual.norm()
        );
        let energy = |w: &Vec3| 0.5 * w.dot(&(inertia_world * w));
        assert!(energy(&w1) <= energy(&w0) * (1.0 + 1e-12));
        assert!(w1 != w0, "the gyroscopic term must act on a tumbling body");
    }
}

#[test]
fn extreme_arguments_never_produce_non_finite_or_energy_gaining_results() {
    let rotation = Quat::from_axis_angle(&Unit::new_normalize(Vec3::new(1.0, 1.0, -0.5)), 0.4);
    let r = *rotation.to_rotation_matrix().matrix();
    let inertia_world = r * Mat3::from_diagonal(&Vec3::new(0.01, 1.0, 50.0)) * r.transpose();
    let directions = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.3, 0.9, -0.4),
        Vec3::new(-0.2, 0.1, 1.0),
    ];
    for &dt in &[1e-9, 1e-6, 1e-3, 0.1, 10.0, 1e3] {
        for &scale in &[1e-3, 1.0, 1e3] {
            for direction in &directions {
                let w0 = direction.normalize() * scale;
                let w1 = gyroscopic_implicit_step_world(w0, &inertia_world, dt);
                assert!(
                    w1.iter().all(|c| c.is_finite()),
                    "non-finite result {w1:?} for dt = {dt}, |w| = {scale}"
                );
                let energy = |w: &Vec3| 0.5 * w.dot(&(inertia_world * w));
                assert!(
                    energy(&w1) <= energy(&w0) * (1.0 + 1e-12),
                    "kinetic energy grew for dt = {dt}, |w| = {scale}"
                );
            }
        }
    }
    // Invalid arguments leave the angular velocity untouched.
    let w0 = Vec3::new(1.0, 2.0, 3.0);
    for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        assert_eq!(gyroscopic_implicit_step_world(w0, &inertia_world, dt), w0);
    }
    let not_finite = Vec3::new(f64::NAN, 0.0, 1.0);
    let unchanged = gyroscopic_implicit_step_world(not_finite, &inertia_world, DT);
    assert!(unchanged[0].is_nan() && unchanged[1] == 0.0 && unchanged[2] == 1.0);
    // A singular (zero) inertia tensor cannot be corrected.
    assert_eq!(gyroscopic_implicit_step_world(w0, &Mat3::zeros(), DT), w0);
}

// ---------------------------------------------------------------------------
// PhysicsWorld
// ---------------------------------------------------------------------------

fn free_body_world(
    solver: impl FnOnce(&mut SolverConfig),
) -> (PhysicsWorld, oxiphysics_core::BodyHandle) {
    let mut world = PhysicsWorld::new([0.0, 0.0, 0.0], DT);
    solver(&mut world.solver_config);
    let handle = world.add_rigid_body(box_body(initial_spin()));
    (world, handle)
}

fn world_body(world: &PhysicsWorld, handle: oxiphysics_core::BodyHandle) -> &RigidBody {
    world.get_body(handle).expect("inserted body must exist")
}

#[test]
fn the_implicit_gyroscopic_stage_is_on_by_default() {
    assert!(SolverConfig::default().use_implicit_gyroscopic);
    assert!(
        PhysicsWorld::new([0.0; 3], DT)
            .solver_config
            .use_implicit_gyroscopic
    );
}

#[test]
fn physics_world_step_conserves_the_angular_momentum_vector() {
    let (mut world, handle) = free_body_world(|_| {});
    let initial_l = world_angular_momentum(world_body(&world, handle));
    let mut max_drift = 0.0_f64;
    for _ in 0..STEPS {
        world.step();
        let l = world_angular_momentum(world_body(&world, handle));
        max_drift = max_drift.max(relative_distance(&l, &initial_l));
    }
    assert!(max_drift < 2.0e-3, "angular momentum drift {max_drift:.3e}");
}

#[test]
fn physics_world_small_steps_apply_the_gyroscopic_stage_on_every_sub_step() {
    // Four sub-steps of 0.01 s per 0.04 s frame: the same 1000 integration steps as above.
    let (mut world, handle) = free_body_world(|solver| solver.substeps = 4);
    world.dt = 4.0 * DT;
    let initial_l = world_angular_momentum(world_body(&world, handle));
    let mut max_drift = 0.0_f64;
    for _ in 0..STEPS / 4 {
        world.step_small_steps();
        let l = world_angular_momentum(world_body(&world, handle));
        max_drift = max_drift.max(relative_distance(&l, &initial_l));
    }
    assert!(max_drift < 2.0e-3, "angular momentum drift {max_drift:.3e}");

    let (mut stepped, stepped_handle) = free_body_world(|_| {});
    for _ in 0..STEPS {
        stepped.step();
    }
    let difference = world_body(&world, handle).angular_velocity
        - world_body(&stepped, stepped_handle).angular_velocity;
    assert!(
        difference.norm() < 1e-12,
        "small steps and plain steps disagree by {:.3e}",
        difference.norm()
    );
}

#[test]
fn physics_world_step_matches_integrate_bodies_for_a_free_body() {
    let (mut world, handle) = free_body_world(|_| {});
    let mut set = RigidBodySet::new();
    let reference = set.insert(box_body(initial_spin()));
    let gravity = Vec3::zeros();
    for _ in 0..200 {
        world.step();
        integrate_bodies(&mut set, DT, &gravity);
    }
    let from_world = world_body(&world, handle);
    let from_set = set.get(reference).expect("reference body must exist");
    assert!((from_world.angular_velocity - from_set.angular_velocity).norm() < 1e-12);
    assert!(
        from_world
            .transform
            .rotation
            .angle_to(&from_set.transform.rotation)
            < 1e-12
    );
}

#[test]
fn opting_out_restores_a_constant_world_frame_angular_velocity() {
    let (mut world, handle) = free_body_world(|solver| solver.use_implicit_gyroscopic = false);
    let initial_l = world_angular_momentum(world_body(&world, handle));
    let mut max_drift = 0.0_f64;
    for _ in 0..STEPS {
        world.step();
        let body = world_body(&world, handle);
        assert_eq!(body.angular_velocity, initial_spin());
        max_drift = max_drift.max(relative_distance(&world_angular_momentum(body), &initial_l));
    }
    assert!(max_drift > 0.05, "angular momentum drift {max_drift:.3e}");
}
