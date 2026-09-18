// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0
//! Public-API regression tests for degenerate inputs of the multibody
//! dynamics, the neural-control planners and the debris collision probability.
//!
//! These calls used to panic (short slices, an out-of-range link, a zero
//! capacity, zero candidates, an empty roadmap), silently return made-up values
//! (zero accelerations for a singular mass matrix) or ignore an argument that
//! the physics depends on (the miss distance).

use oxiphysics_rigid::multibody_dynamics::{
    Link, MultibodyChain, MultibodyError, build_planar_arm,
};
use oxiphysics_rigid::neural_control::{
    Activation, Demonstration, ImitationLearning, ModelPredictiveControl, MotionPlanning,
    NeuralController, ReplayBuffer, Transition,
};
use oxiphysics_rigid::orbital_mechanics::collision_probability_with_miss_distance;

const GRAVITY: [f64; 3] = [0.0, -9.81, 0.0];

// ── multibody dynamics ──────────────────────────────────────────────────

#[test]
fn dynamics_queries_report_bad_arguments_instead_of_panicking() {
    let chain = build_planar_arm(3, 1.0, 1.0);
    assert_eq!(
        chain.forward_dynamics(&[0.0; 2], GRAVITY),
        Err(MultibodyError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    );
    assert_eq!(
        chain.inverse_dynamics(&[], GRAVITY),
        Err(MultibodyError::DimensionMismatch {
            expected: 3,
            got: 0
        })
    );
    assert_eq!(chain.jacobian(3), Err(MultibodyError::LinkOutOfRange(3)));
    let mut stepped = chain.clone();
    assert!(stepped.step_rk4(&[0.0; 4], GRAVITY, 1e-3).is_err());
    assert_eq!(stepped.joint_angles, chain.joint_angles);
    assert!(chain.forward_dynamics(&[0.0; 3], GRAVITY).is_ok());
    assert_eq!(chain.jacobian(2).map(|j| j.len()), Ok(3));
}

#[test]
fn a_singular_mass_matrix_is_an_error_not_zero_acceleration() {
    // A point mass on the joint axis has no inertia about it.
    let chain = MultibodyChain::new(vec![Link::new(2.0, [0.0, 0.0, 0.5])], vec![None]);
    assert_eq!(
        chain.forward_dynamics(&[3.0], GRAVITY),
        Err(MultibodyError::SingularMassMatrix)
    );
    let mut stepped = chain.clone();
    assert_eq!(
        stepped.step_rk4(&[3.0], GRAVITY, 1e-3),
        Err(MultibodyError::SingularMassMatrix)
    );
}

// ── neural control ──────────────────────────────────────────────────────

#[test]
fn planners_and_buffers_survive_degenerate_configurations() {
    // Zero candidates and a zero action bound used to panic in cem_plan.
    let cost = |s: &[f64]| s.iter().map(|x| x * x).sum::<f64>();
    let no_candidates = ModelPredictiveControl::new(2, 1, &[4], 5, 0, 0.9, 1.0);
    assert_eq!(no_candidates.cem_plan(&[0.1, 0.2], cost, 3), vec![0.0]);
    let no_room = ModelPredictiveControl::new(2, 1, &[4], 5, 8, 0.9, -1.0);
    assert_eq!(no_room.cem_plan(&[0.1, 0.2], cost, 3), vec![0.0]);

    // A zero-capacity replay buffer used to divide by zero.
    let mut buffer = ReplayBuffer::new(0);
    buffer.push(Transition {
        state: vec![0.0],
        action: 0,
        reward: 1.0,
        next_state: vec![1.0],
        done: false,
    });
    assert!(buffer.is_empty());

    // An empty roadmap used to index a sample that does not exist.
    let planner = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 10, 0.05, false);
    assert!(planner.build_prm(0, 3).is_empty());
}

#[test]
fn back_propagation_is_available_and_seedable_learning_is_reproducible() {
    let mut net = NeuralController::new(3, &[4, 4], 2, 1.5);
    net.activation = Activation::Tanh;
    let gradient = net.backward(&[0.1, 0.2, 0.3], &[1.0, -1.0]);
    assert_eq!(gradient.len(), net.num_parameters());
    let cache = net.forward_cached(&[0.1, 0.2, 0.3]);
    assert_eq!(cache.output, net.forward(&[0.1, 0.2, 0.3]));
    assert_eq!(net.backward_cached(&cache, &[1.0, -1.0]), gradient);

    let train = |seed: u64| {
        let mut il = ImitationLearning::new(2, 1, &[4], 0.1).with_seed(seed);
        il.policy
            .set_parameters(&vec![0.1; il.policy.num_parameters()]);
        for k in 0..5 {
            let x = f64::from(k) * 0.2;
            il.add_demonstration(Demonstration::new(vec![x, -x], vec![0.5 * x]));
        }
        for _ in 0..3 {
            il.bc_step(2);
        }
        il.policy.get_parameters()
    };
    assert_eq!(train(5), train(5));
}

// ── debris collision probability ────────────────────────────────────────

#[test]
fn collision_probability_depends_on_the_miss_distance() {
    let (radius, sigma) = (10.0, 100.0);
    let direct = collision_probability_with_miss_distance(radius, 0.0, sigma);
    assert!(
        (direct - (1.0 - (-0.005_f64).exp())).abs() < 1e-14,
        "{direct}"
    );
    let mut previous = direct;
    for distance in [50.0, 100.0, 200.0, 400.0, 800.0] {
        let probability = collision_probability_with_miss_distance(radius, distance, sigma);
        assert!(probability < previous, "d = {distance}: {probability:e}");
        previous = probability;
    }
    assert!(collision_probability_with_miss_distance(radius, -1.0, sigma).is_nan());
}
