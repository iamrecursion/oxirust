// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the learners built on [`NeuralController`].
//!
//! Every learner used to estimate its gradient by central finite differences
//! (two forward passes and several parameter-vector copies per parameter).  They
//! now back-propagate.  The tests keep the old estimators as *oracles*: the
//! parameters after one update must agree with those of the finite-difference
//! update, and the cost of an update must grow linearly with the parameter
//! count.

use super::*;
use rand::rngs::{SmallRng, StdRng};
use rand::{RngExt, SeedableRng};
use std::time::{Duration, Instant};

/// Step of the oracle's central finite differences.
const EPS: f64 = 1e-6;

/// Largest tolerated difference between a parameter after the analytic update
/// and after the finite-difference update (the learning rates are ≤ 0.5 and the
/// gradients O(1), the oracle itself is good to about 1e-9).
const PARAM_TOL: f64 = 1e-6;

/// Deterministic weights and biases for `net`.
fn randomise(net: &mut NeuralController, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let params: Vec<f64> = (0..net.num_parameters())
        .map(|_| rng.random_range(-1.0_f64..1.0_f64))
        .collect();
    net.set_parameters(&params);
}

fn assert_parameters_agree(label: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    for (j, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < PARAM_TOL,
            "{label}: parameter {j}: {g} vs finite-difference update {e}"
        );
    }
}

/// Central finite-difference gradient of the scalar `loss(network)`.
fn finite_difference_gradient<L>(net: &NeuralController, loss: L) -> Vec<f64>
where
    L: Fn(&NeuralController) -> f64,
{
    let params = net.get_parameters();
    let mut probe = net.clone();
    let mut shifted = params.clone();
    (0..params.len())
        .map(|j| {
            shifted[j] = params[j] + EPS;
            probe.set_parameters(&shifted);
            let plus = loss(&probe);
            shifted[j] = params[j] - EPS;
            probe.set_parameters(&shifted);
            let minus = loss(&probe);
            shifted[j] = params[j];
            (plus - minus) / (2.0 * EPS)
        })
        .collect()
}

/// `params - lr · grad`.
fn sgd(params: &[f64], grad: &[f64], lr: f64) -> Vec<f64> {
    params
        .iter()
        .zip(grad.iter())
        .map(|(p, g)| p - lr * g)
        .collect()
}

fn log_probability(net: &NeuralController, state: &[f64], action: usize) -> f64 {
    (softmax(&net.forward(state))[action] + 1e-8).ln()
}

fn learner(state_dim: usize, hidden: &[usize], action_dim: usize) -> ReinforcementLearning {
    let mut rl = ReinforcementLearning::new(state_dim, hidden, action_dim, 0.9, 0.3, 0.2);
    randomise(&mut rl.actor, 31);
    randomise(&mut rl.critic, 32);
    rl
}

// ── REINFORCE ───────────────────────────────────────────────────────────

#[test]
fn reinforce_update_matches_the_finite_difference_update() {
    let mut rl = learner(3, &[6, 5], 4);
    let steps: [([f64; 3], usize, f64); 5] = [
        ([0.1, -0.4, 0.9], 2, 1.0),
        ([0.5, 0.5, -0.5], 0, 0.5),
        ([-0.8, 0.2, 0.3], 3, -1.0),
        ([0.0, 1.0, -1.0], 1, 2.0),
        ([0.7, -0.7, 0.1], 2, 0.0),
    ];
    for (state, action, reward) in &steps {
        rl.record_step(state.to_vec(), *action, -1.0, *reward);
    }

    // Oracle: the legacy estimator of `reinforce_update`.
    let returns = rl.compute_returns();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let stdev = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n)
        .sqrt()
        .max(1e-8);
    let mut grad = vec![0.0; rl.actor.num_parameters()];
    for (i, (state, action, _, _)) in rl.trajectory.iter().enumerate() {
        let advantage = (returns[i] - mean) / stdev;
        let dlp = finite_difference_gradient(&rl.actor, |net| log_probability(net, state, *action));
        for (g, d) in grad.iter_mut().zip(dlp.iter()) {
            *g -= advantage * d / n;
        }
    }
    let expected = sgd(&rl.actor.get_parameters(), &grad, rl.actor_lr);
    let critic_before = rl.critic.get_parameters();

    rl.reinforce_update();
    assert_parameters_agree("actor", &rl.actor.get_parameters(), &expected);
    assert_eq!(
        rl.critic.get_parameters(),
        critic_before,
        "REINFORCE has no critic step"
    );
    assert!(rl.trajectory.is_empty());
}

#[test]
fn reinforce_raises_the_probability_of_the_rewarded_action() {
    let mut rl = ReinforcementLearning::new(2, &[8], 3, 0.0, 0.5, 0.1);
    randomise(&mut rl.actor, 5);
    let state = [0.4, -0.6];
    let before = softmax(&rl.actor.forward(&state));
    // Action 0 earns +1, action 1 earns -1; with gamma = 0 the advantages are ±1.
    rl.record_step(state.to_vec(), 0, 0.0, 1.0);
    rl.record_step(state.to_vec(), 1, 0.0, -1.0);
    rl.reinforce_update();
    let after = softmax(&rl.actor.forward(&state));
    assert!(after[0] > before[0], "{before:?} -> {after:?}");
    assert!(after[1] < before[1], "{before:?} -> {after:?}");
}

#[test]
fn reinforce_ignores_invalid_actions_and_an_empty_trajectory() {
    let mut rl = learner(2, &[4], 3);
    let untouched = rl.actor.get_parameters();
    rl.reinforce_update();
    assert_eq!(rl.actor.get_parameters(), untouched);

    rl.record_step(vec![0.1, 0.2], 99, 0.0, 1.0);
    rl.record_step(vec![0.3, -0.2], 1, 0.0, -1.0);
    rl.reinforce_update();
    assert!(rl.actor.get_parameters().iter().all(|p| p.is_finite()));
    assert!(rl.trajectory.is_empty());
}

// ── actor-critic ────────────────────────────────────────────────────────

#[test]
fn a2c_update_matches_the_finite_difference_update() {
    for (action, reward, done) in [(1, 0.7, false), (2, -0.4, true), (0, 1.5, false)] {
        let mut rl = learner(3, &[7], 3);
        let state = [0.3, -0.2, 0.8];
        let next_state = [0.1, 0.6, -0.5];

        // Oracle: the legacy estimator of `a2c_update`.
        let value = rl.critic.forward(&state)[0];
        let next_value = if done {
            0.0
        } else {
            rl.critic.forward(&next_state)[0]
        };
        let advantage = reward + rl.gamma * next_value - value;
        let dv = finite_difference_gradient(&rl.critic, |net| net.forward(&state)[0]);
        let critic_grad: Vec<f64> = dv.iter().map(|d| -2.0 * advantage * d).collect();
        let expected_critic = sgd(&rl.critic.get_parameters(), &critic_grad, rl.critic_lr);
        let dlp = finite_difference_gradient(&rl.actor, |net| log_probability(net, &state, action));
        let actor_grad: Vec<f64> = dlp.iter().map(|d| -advantage * d).collect();
        let expected_actor = sgd(&rl.actor.get_parameters(), &actor_grad, rl.actor_lr);

        rl.a2c_update(&state, action, reward, &next_state, done);
        assert_parameters_agree("critic", &rl.critic.get_parameters(), &expected_critic);
        assert_parameters_agree("actor", &rl.actor.get_parameters(), &expected_actor);
    }
}

#[test]
fn a2c_update_with_an_invalid_action_trains_only_the_critic() {
    let mut rl = learner(2, &[4], 3);
    let actor_before = rl.actor.get_parameters();
    let critic_before = rl.critic.get_parameters();
    rl.a2c_update(&[0.2, 0.4], 99, 1.0, &[0.1, 0.3], false);
    assert_eq!(rl.actor.get_parameters(), actor_before);
    assert_ne!(rl.critic.get_parameters(), critic_before);
}

#[test]
fn a2c_critic_learns_a_constant_return() {
    // Terminal transitions with reward 0.5 teach the critic V(s) → 0.5.
    let mut rl = ReinforcementLearning::new(2, &[6], 2, 0.9, 0.0, 0.1);
    randomise(&mut rl.critic, 8);
    let state = [0.5, -0.5];
    let error = |rl: &ReinforcementLearning| (rl.critic.forward(&state)[0] - 0.5).abs();
    let before = error(&rl);
    for _ in 0..400 {
        rl.a2c_update(&state, 0, 0.5, &state, true);
    }
    assert!(
        error(&rl) < 0.02 && error(&rl) < before,
        "{before} -> {}",
        error(&rl)
    );
}

// ── behavioural cloning ─────────────────────────────────────────────────

fn imitation(seed: u64) -> ImitationLearning {
    let mut il = ImitationLearning::new(2, 2, &[6], 0.3).with_seed(seed);
    randomise(&mut il.policy, 77);
    il
}

#[test]
fn bc_step_matches_the_finite_difference_update() {
    // A single demonstration makes the draw deterministic.
    let mut il = imitation(1);
    let demo = Demonstration::new(vec![0.4, -0.9], vec![0.3, -0.6]);
    il.add_demonstration(demo.clone());
    let loss = |net: &NeuralController| {
        let pred = net.forward(&demo.state);
        0.5 * pred
            .iter()
            .zip(demo.action.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
    };
    let grad = finite_difference_gradient(&il.policy, loss);
    let expected = sgd(&il.policy.get_parameters(), &grad, il.lr);
    il.bc_step(4);
    assert_parameters_agree("policy", &il.policy.get_parameters(), &expected);
    assert_eq!(il.iter, 1);
}

#[test]
fn bc_step_averages_over_the_drawn_demonstrations() {
    // Two demonstrations, many draws: the update is the mean of the per-sample
    // gradients of the drawn demonstrations (checked against the generator).
    let mut il = imitation(123);
    il.add_demonstration(Demonstration::new(vec![0.4, -0.9], vec![0.3, -0.6]));
    il.add_demonstration(Demonstration::new(vec![-0.2, 0.7], vec![-0.5, 0.1]));
    let mut replay = SmallRng::seed_from_u64(123);
    let batch = 5;
    let drawn: Vec<usize> = (0..batch).map(|_| replay.random_range(0..2)).collect();
    let mut grad = vec![0.0; il.policy.num_parameters()];
    for &index in &drawn {
        let demo = il.dataset[index].clone();
        let loss = |net: &NeuralController| {
            0.5 * net
                .forward(&demo.state)
                .iter()
                .zip(demo.action.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f64>()
        };
        for (g, d) in grad
            .iter_mut()
            .zip(finite_difference_gradient(&il.policy, loss))
        {
            *g += d / batch as f64;
        }
    }
    let expected = sgd(&il.policy.get_parameters(), &grad, il.lr);
    il.bc_step(batch);
    assert_parameters_agree("policy", &il.policy.get_parameters(), &expected);
}

#[test]
fn bc_step_is_reproducible_for_a_seed_and_differs_between_seeds() {
    let demos = [
        ([0.4, -0.9], [0.3, -0.6]),
        ([-0.2, 0.7], [-0.5, 0.1]),
        ([0.9, 0.1], [0.6, 0.2]),
        ([-0.7, -0.3], [-0.1, -0.4]),
        ([0.0, 0.5], [0.0, 0.3]),
        ([0.3, 0.3], [0.2, 0.2]),
    ];
    let run = |seed: u64| {
        let mut il = imitation(seed);
        for (s, a) in &demos {
            il.add_demonstration(Demonstration::new(s.to_vec(), a.to_vec()));
        }
        for _ in 0..5 {
            il.bc_step(2);
        }
        il.policy.get_parameters()
    };
    assert_eq!(run(9), run(9), "same seed, same parameters");
    assert_ne!(run(9), run(10), "different seeds draw different batches");
}

#[test]
fn bc_step_reduces_the_cloning_loss() {
    let mut il = imitation(5);
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..24 {
        let s = [
            rng.random_range(-1.0_f64..1.0_f64),
            rng.random_range(-1.0_f64..1.0_f64),
        ];
        let a = [0.5 * (1.5 * s[0]).sin(), 0.4 * s[1] * s[0]];
        il.add_demonstration(Demonstration::new(s.to_vec(), a.to_vec()));
    }
    let loss = |il: &ImitationLearning| {
        il.dataset
            .iter()
            .map(|d| {
                let pred = il.policy.forward(&d.state);
                pred.iter()
                    .zip(d.action.iter())
                    .map(|(p, a)| (p - a).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / il.dataset.len() as f64
    };
    let before = loss(&il);
    for _ in 0..400 {
        il.bc_step(8);
    }
    let after = loss(&il);
    assert!(after < 0.25 * before, "loss {before} -> {after}");
}

#[test]
fn bc_step_without_demonstrations_or_samples_changes_nothing() {
    let mut il = imitation(3);
    let params = il.policy.get_parameters();
    il.bc_step(4);
    assert_eq!(il.policy.get_parameters(), params);
    assert_eq!(il.iter, 0, "an empty dataset is not a step");
    il.add_demonstration(Demonstration::new(vec![0.1, 0.2], vec![0.0, 0.0]));
    il.bc_step(0);
    assert_eq!(il.policy.get_parameters(), params);
}

#[test]
fn dagger_action_with_rng_follows_the_schedule_reproducibly() {
    let mut il = imitation(1);
    il.iter = 0;
    let mut rng = StdRng::seed_from_u64(4);
    // beta = 1: always the expert.
    assert_eq!(
        il.dagger_action_with_rng(&[0.1, 0.2], &[9.0, 9.0], &mut rng),
        vec![9.0, 9.0]
    );
    // beta = 2^-40: the policy.
    il.iter = 40;
    let policy_action = il.policy.forward(&[0.1, 0.2]);
    assert_eq!(
        il.dagger_action_with_rng(&[0.1, 0.2], &[9.0, 9.0], &mut rng),
        policy_action
    );
}

// ── dynamics model ──────────────────────────────────────────────────────

#[test]
fn train_step_matches_the_finite_difference_update() {
    let mut model = NeuralDynamicsModel::new(3, 2, &[7, 5]);
    randomise(&mut model.network, 41);
    let state = [0.8, -0.3, 0.5];
    let action = [0.4, -0.9];
    let next_state = [0.6, -0.1, 0.9];
    let lr = 0.25;

    let loss = |net: &NeuralController| {
        let mut input = state.to_vec();
        input.extend_from_slice(&action);
        let predicted = vadd(&state, &net.forward(&input));
        0.5 * predicted
            .iter()
            .zip(next_state.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
    };
    let grad = finite_difference_gradient(&model.network, loss);
    let expected = sgd(&model.network.get_parameters(), &grad, lr);
    model.train_step(&state, &action, &next_state, lr);
    assert_parameters_agree("network", &model.network.get_parameters(), &expected);
}

#[test]
fn train_step_fits_a_transition() {
    let mut model = NeuralDynamicsModel::new(2, 1, &[8]);
    randomise(&mut model.network, 6);
    let (state, action, next_state) = ([0.5, -0.5], [0.3], [0.7, -0.2]);
    let error = |m: &NeuralDynamicsModel| {
        m.predict(&state, &action)
            .iter()
            .zip(next_state.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
    };
    let before = error(&model);
    for _ in 0..500 {
        model.train_step(&state, &action, &next_state, 0.2);
    }
    assert!(
        error(&model) < 0.05 * before,
        "error {before} -> {}",
        error(&model)
    );
}

// ── cost of an update ───────────────────────────────────────────────────

/// An update costs one forward and one backward pass.  The finite-difference
/// estimator needed two forward passes and several `O(P)` copies *per parameter*:
/// for the 60 400 parameters below that is minutes, back-propagation needs
/// milliseconds.  The generous limit only has to separate `O(P)` from `O(P²)`.
#[test]
fn an_update_costs_linear_not_quadratic_time_in_the_parameter_count() {
    let limit = Duration::from_secs(10);
    let mut model = NeuralDynamicsModel::new(200, 100, &[100, 100]);
    assert_eq!(model.network.num_parameters(), 60_400);
    let state = vec![0.1; 200];
    let action = vec![0.2; 100];
    let target = vec![0.3; 200];

    let start = Instant::now();
    model.train_step(&state, &action, &target, 0.01);
    let train_time = start.elapsed();
    assert!(train_time < limit, "train_step took {train_time:?}");

    let mut il = ImitationLearning::new(200, 100, &[100, 100], 0.01).with_seed(3);
    il.add_demonstration(Demonstration::new(state.clone(), vec![0.1; 100]));
    let start = Instant::now();
    il.bc_step(2);
    let bc_time = start.elapsed();
    assert!(bc_time < limit, "bc_step took {bc_time:?}");

    let mut rl = ReinforcementLearning::new(200, &[100, 100], 8, 0.9, 0.01, 0.01);
    for step in 0..2 {
        rl.record_step(state.clone(), step, 0.0, step as f64);
    }
    let start = Instant::now();
    rl.reinforce_update();
    rl.a2c_update(&state, 3, 1.0, &state, false);
    let rl_time = start.elapsed();
    assert!(rl_time < limit, "RL updates took {rl_time:?}");
}

// ── sampling and bookkeeping ────────────────────────────────────────────

#[test]
fn sample_action_with_rng_is_reproducible_and_valid() {
    let rl = learner(3, &[5], 4);
    let state = [0.2, -0.1, 0.6];
    let draw = |seed: u64| {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..20)
            .map(|_| rl.sample_action_with_rng(&state, &mut rng))
            .collect::<Vec<_>>()
    };
    assert_eq!(draw(7), draw(7));
    let probs = softmax(&rl.actor.forward(&state));
    for (action, log_prob) in draw(8) {
        assert!(action < 4);
        assert!((log_prob - (probs[action] + 1e-8).ln()).abs() < 1e-12);
    }
    // Different seeds explore different actions.
    assert_ne!(
        draw(7).iter().map(|d| d.0).collect::<Vec<_>>(),
        draw(1234).iter().map(|d| d.0).collect::<Vec<_>>()
    );
}

#[test]
fn num_parameters_counts_the_flat_parameter_vector_without_building_it() {
    for (state_dim, hidden, action_dim) in [(3, &[][..], 2), (4, &[7][..], 1), (5, &[6, 3][..], 4)]
    {
        let net = NeuralController::new(state_dim, hidden, action_dim, 1.0);
        assert_eq!(net.num_parameters(), net.get_parameters().len());
    }
    let net = NeuralController::new(2, &[3], 1, 1.0);
    assert_eq!(net.num_parameters(), (2 * 3 + 3) + (3 + 1));
}
