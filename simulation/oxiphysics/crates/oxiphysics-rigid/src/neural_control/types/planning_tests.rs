// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the model-predictive planners ([`ModelPredictiveControl::plan`] and
//! [`ModelPredictiveControl::cem_plan`]), of the replay buffer and of the
//! sampling-based motion planner on degenerate input.
//!
//! The planners used to panic when the number of candidates was zero (an empty
//! elite slice) or the action bound was zero or negative; the replay buffer
//! divided by a zero capacity; the roadmap indexed a sample that does not exist
//! for `n_samples == 0`.  CEM also optimised the first action only and drew all
//! later actions of every candidate uniformly at random.

use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A one-dimensional integrator `s' = s + a`: a network without hidden layers
/// whose output is `K·tanh(a/K)` with `K = 1000`, i.e. `a` up to a relative
/// error of 3e-7 for `|a| ≤ 1`.
fn integrator_mpc(
    horizon: usize,
    num_candidates: usize,
    gamma: f64,
    action_bound: f64,
) -> ModelPredictiveControl {
    let mut mpc =
        ModelPredictiveControl::new(1, 1, &[], horizon, num_candidates, gamma, action_bound);
    let k = 1000.0;
    mpc.dynamics.network.action_scale = k;
    // Parameters: the weights of (state, action) and the bias of the output unit.
    mpc.dynamics.network.set_parameters(&[0.0, 1.0 / k, 0.0]);
    mpc
}

fn transition(tag: f64) -> Transition {
    Transition {
        state: vec![tag],
        action: 0,
        reward: tag,
        next_state: vec![tag + 1.0],
        done: false,
    }
}

fn state_tags(batch: &[&Transition]) -> Vec<f64> {
    batch.iter().map(|t| t.state[0]).collect()
}

// ── model-predictive control ────────────────────────────────────────────

#[test]
fn the_integrator_model_is_an_integrator() {
    let mpc = integrator_mpc(3, 10, 1.0, 1.0);
    let next = mpc.dynamics.predict(&[0.25], &[0.5]);
    assert!((next[0] - 0.75).abs() < 1e-6, "{next:?}");
}

#[test]
fn planners_without_anything_to_optimise_return_the_zero_action() {
    let cost = |s: &[f64]| s.iter().map(|x| x * x).sum::<f64>();
    let state = [0.5, -0.5];
    let cases = [
        // (horizon, candidates, bound)
        (5, 0, 1.0),
        (0, 10, 1.0),
        (5, 10, 0.0),
        (5, 10, -2.0),
        (5, 10, f64::NAN),
        (5, 10, f64::INFINITY),
        (0, 0, 0.0),
    ];
    for (horizon, candidates, bound) in cases {
        let mpc = ModelPredictiveControl::new(2, 3, &[4], horizon, candidates, 0.9, bound);
        let label = format!("horizon {horizon}, candidates {candidates}, bound {bound}");
        assert_eq!(mpc.plan(&state, cost), vec![0.0; 3], "plan: {label}");
        assert_eq!(
            mpc.cem_plan(&state, cost, 4),
            vec![0.0; 3],
            "cem_plan: {label}"
        );
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(
            mpc.plan_with_rng(&state, cost, &mut rng),
            vec![0.0; 3],
            "{label}"
        );
        assert_eq!(
            mpc.cem_plan_with_rng(&state, cost, 4, &mut rng),
            vec![0.0; 3],
            "{label}"
        );
    }
}

#[test]
fn a_planner_without_actions_returns_an_empty_action() {
    let mpc = ModelPredictiveControl::new(2, 0, &[], 3, 10, 0.9, 1.0);
    let cost = |s: &[f64]| s[0] * s[0];
    assert!(mpc.plan(&[0.3, 0.1], cost).is_empty());
    assert!(mpc.cem_plan(&[0.3, 0.1], cost, 3).is_empty());
}

#[test]
fn cem_with_zero_iterations_keeps_the_initial_zero_mean() {
    let mpc = integrator_mpc(3, 50, 1.0, 1.0);
    assert_eq!(mpc.cem_plan(&[0.0], |s| (s[0] - 2.5).powi(2), 0), vec![0.0]);
}

#[test]
fn cem_optimises_the_whole_action_sequence() {
    // s' = s + a from s = 0 with |a| ≤ 1 and the cost Σ (s − 2.5)² over three
    // steps.  The optimum is a = (1, 1, 1/2): the first two actions saturate and
    // the last one is an interior value that only a distribution over the whole
    // sequence can find (the old CEM drew actions 1 and 2 uniformly at random).
    let mpc = integrator_mpc(3, 300, 1.0, 1.0);
    let cost = |s: &[f64]| (s[0] - 2.5).powi(2);
    let mut rng = StdRng::seed_from_u64(2024);
    let sequence = mpc.cem_sequence(&[0.0], &cost, 15, &mut rng);
    assert_eq!(sequence.len(), 3);
    for (step, (got, expected)) in sequence.iter().zip([1.0, 1.0, 0.5]).enumerate() {
        assert!(
            (got - expected).abs() < 0.05,
            "action {step}: {got} vs {expected} (sequence {sequence:?})"
        );
    }
}

#[test]
fn cem_plan_returns_the_first_action_of_the_planned_sequence() {
    let mpc = integrator_mpc(3, 200, 1.0, 1.0);
    let cost = |s: &[f64]| (s[0] - 2.5).powi(2);
    let sequence = mpc.cem_sequence(&[0.0], &cost, 8, &mut StdRng::seed_from_u64(5));
    let first = mpc.cem_plan_with_rng(&[0.0], cost, 8, &mut StdRng::seed_from_u64(5));
    assert_eq!(first, sequence[..1].to_vec());
    assert!(first[0] > 0.9, "{first:?}");
}

#[test]
fn cem_plans_are_reproducible_for_a_seed_and_respect_the_bound() {
    let mpc = integrator_mpc(4, 60, 0.95, 0.8);
    let cost = |s: &[f64]| (s[0] + 1.0).powi(2);
    let plan = |seed: u64| mpc.cem_sequence(&[0.2], &cost, 6, &mut StdRng::seed_from_u64(seed));
    assert_eq!(plan(3), plan(3));
    assert_ne!(plan(3), plan(4));
    for action in plan(3) {
        assert!(action.abs() <= 0.8, "{action} exceeds the bound");
    }
}

#[test]
fn random_shooting_prefers_the_good_first_action_and_is_seedable() {
    let mpc = integrator_mpc(3, 400, 1.0, 1.0);
    let cost = |s: &[f64]| (s[0] - 2.5).powi(2);
    let plan = |seed: u64| mpc.plan_with_rng(&[0.0], cost, &mut StdRng::seed_from_u64(seed));
    assert_eq!(plan(11), plan(11));
    let first = plan(11);
    assert_eq!(first.len(), 1);
    assert!(first[0] > 0.3 && first[0] < 1.0, "{first:?}");
}

#[test]
fn nan_costs_are_never_preferred_and_do_not_panic() {
    let mpc = integrator_mpc(3, 40, 1.0, 1.0);
    // Every cost is NaN: random shooting has no best candidate.
    assert_eq!(mpc.plan(&[0.0], |_| f64::NAN), vec![0.0]);
    // CEM ranks NaN last (as infinite) and still returns a bounded action.
    let action = mpc.cem_plan(&[0.0], |_| f64::NAN, 3);
    assert_eq!(action.len(), 1);
    assert!(
        action[0].is_finite() && action[0].abs() <= 1.0,
        "{action:?}"
    );
    // Only a negative-NaN cost for part of the candidates: the finite ones win.
    let negative_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
    let cost = |s: &[f64]| {
        if s[0] < 0.0 {
            negative_nan
        } else {
            (s[0] - 2.5).powi(2)
        }
    };
    let action = mpc.cem_plan_with_rng(&[0.0], cost, 10, &mut StdRng::seed_from_u64(9));
    assert!(action[0] > 0.5, "{action:?}");
}

// ── replay buffer ───────────────────────────────────────────────────────

#[test]
fn a_zero_capacity_buffer_drops_every_transition() {
    let mut buffer = ReplayBuffer::new(0);
    for i in 0..5 {
        buffer.push(transition(f64::from(i)));
    }
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
    assert_eq!(buffer.write_pos, 0);
    assert!(buffer.sample(3).is_empty());
}

#[test]
fn a_full_buffer_overwrites_the_oldest_transition() {
    let mut buffer = ReplayBuffer::new(3);
    for i in 0..5 {
        buffer.push(transition(f64::from(i)));
    }
    // 0 and 1 were overwritten by 3 and 4.
    let tags: Vec<f64> = buffer.buffer.iter().map(|t| t.state[0]).collect();
    assert_eq!(tags, vec![3.0, 4.0, 2.0]);
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.write_pos, 2);
}

#[test]
fn a_buffer_whose_capacity_shrank_after_filling_does_not_panic() {
    let mut buffer = ReplayBuffer::new(5);
    for i in 0..5 {
        buffer.push(transition(f64::from(i)));
    }
    buffer.capacity = 3;
    buffer.write_pos = 4;
    buffer.push(transition(9.0));
    assert_eq!(buffer.len(), 5, "stored transitions are not dropped");
    assert_eq!(buffer.buffer[4].state[0], 9.0);
    assert_eq!(buffer.write_pos, 2);
    // Growing it again lets it fill up again.
    buffer.capacity = 7;
    buffer.push(transition(10.0));
    assert_eq!(buffer.len(), 6);
}

#[test]
fn sampling_is_without_replacement() {
    let mut buffer = ReplayBuffer::new(10);
    for i in 0..10 {
        buffer.push(transition(f64::from(i)));
    }
    let mut rng = StdRng::seed_from_u64(17);
    for amount in [0, 1, 4, 9, 10, 25] {
        let batch = buffer.sample_with_rng(amount, &mut rng);
        assert_eq!(batch.len(), amount.min(10), "amount {amount}");
        let mut tags = state_tags(&batch);
        tags.sort_by(f64::total_cmp);
        tags.dedup();
        assert_eq!(tags.len(), batch.len(), "duplicates in a batch of {amount}");
    }
    // Asking for everything returns a permutation of the buffer.
    let mut all = state_tags(&buffer.sample_with_rng(10, &mut rng));
    all.sort_by(f64::total_cmp);
    assert_eq!(all, (0..10).map(f64::from).collect::<Vec<_>>());
    // The unseeded entry point obeys the same contract.
    let mut tags = state_tags(&buffer.sample(6));
    tags.sort_by(f64::total_cmp);
    tags.dedup();
    assert_eq!(tags.len(), 6);
}

#[test]
fn sampling_draws_different_batches_and_is_reproducible_for_a_seed() {
    let mut buffer = ReplayBuffer::new(50);
    for i in 0..50 {
        buffer.push(transition(f64::from(i)));
    }
    let draw = |seed: u64| state_tags(&buffer.sample_with_rng(5, &mut StdRng::seed_from_u64(seed)));
    assert_eq!(draw(1), draw(1));
    assert_ne!(draw(1), draw(2));
    assert!(ReplayBuffer::new(4).sample(3).is_empty());
}

// ── motion planner ──────────────────────────────────────────────────────

fn nan_distance_node(config: Vec<f64>) -> PlannerNode {
    PlannerNode::new(config, usize::MAX, 0.0)
}

#[test]
fn nearest_never_prefers_a_node_with_nan_coordinates() {
    let planner = MotionPlanning::new(2, vec![-1.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false);
    let negative_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
    assert!(negative_nan.is_nan());
    let nodes = vec![
        nan_distance_node(vec![f64::NAN, 0.0]),
        nan_distance_node(vec![negative_nan, 0.0]),
        nan_distance_node(vec![0.5, 0.5]),
        nan_distance_node(vec![0.0, 0.0]),
    ];
    // The NaN nodes come first, so a comparison that ranks them below real
    // distances (as `total_cmp` does for a negative NaN) would pick one of them.
    assert_eq!(planner.nearest(&nodes, &[0.6, 0.6]), 2);
    assert_eq!(planner.nearest(&nodes, &[0.1, 0.0]), 3);
}

#[test]
fn nearest_survives_nan_queries_and_empty_node_lists() {
    let planner = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false);
    let nodes = vec![
        nan_distance_node(vec![0.2, 0.2]),
        nan_distance_node(vec![0.8, 0.8]),
    ];
    // Every distance is NaN: no panic, a valid index.
    assert!(planner.nearest(&nodes, &[f64::NAN, 0.0]) < nodes.len());
    assert_eq!(planner.nearest(&[], &[0.0, 0.0]), 0);
}

#[test]
fn random_configs_survive_pinned_inverted_and_missing_bounds() {
    // Joint 0 pinned, joint 1 inverted (`upper < lower` has no interior).
    let planner = MotionPlanning::new(
        3,
        vec![0.5, 2.0, -1.0],
        vec![0.5, 1.0, 1.0],
        0.1,
        10,
        0.05,
        false,
    );
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..100 {
        let q = planner.random_config_with_rng(&mut rng);
        assert_eq!(q.len(), 3);
        assert_eq!(q[0], 0.5, "pinned joint");
        assert_eq!(q[1], 2.0, "inverted bounds yield the lower bound");
        assert!((-1.0..1.0).contains(&q[2]));
    }
    // Bounds that do not cover every joint give a short, never collision-free sample.
    let short = MotionPlanning::new(3, vec![0.0, 0.0], vec![1.0, 1.0], 0.1, 50, 0.05, false);
    let q = short.random_config();
    assert_eq!(q.len(), 2);
    assert!(!short.is_collision_free(&q));
    let mut rng = StdRng::seed_from_u64(1);
    assert!(
        short
            .rrt_with_rng(&[0.1, 0.1, 0.1], &[0.9, 0.9, 0.9], &mut rng)
            .is_empty()
    );
    assert_eq!(short.build_prm(4, 2), vec![Vec::<usize>::new(); 4]);
}

#[test]
fn random_configs_are_reproducible_for_a_seed_and_stay_in_bounds() {
    let planner = MotionPlanning::new(
        3,
        vec![-1.0, 0.0, 2.0],
        vec![1.0, 5.0, 3.0],
        0.1,
        10,
        0.05,
        false,
    );
    let draw = |seed: u64| planner.random_config_with_rng(&mut StdRng::seed_from_u64(seed));
    assert_eq!(draw(8), draw(8));
    assert_ne!(draw(8), draw(9));
    for seed in 0..50 {
        let q = draw(seed);
        for (k, value) in q.iter().enumerate() {
            assert!(
                (planner.lower[k]..planner.upper[k]).contains(value),
                "joint {k}: {value}"
            );
        }
    }
}

#[test]
fn a_roadmap_without_samples_is_empty() {
    let planner = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false);
    // Before the fix `configs[0]` indexed an empty vector.
    assert!(planner.build_prm(0, 3).is_empty());
    assert!(planner.build_prm(0, 0).is_empty());
    assert!(
        planner
            .build_prm_with_rng(0, 5, &mut StdRng::seed_from_u64(1))
            .is_empty()
    );
    // One sample has nobody to connect to.
    assert_eq!(planner.build_prm(1, 3), vec![Vec::<usize>::new()]);
}

#[test]
fn an_obstacle_free_roadmap_connects_every_sample_to_its_nearest_neighbours() {
    let planner = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false);
    let (samples, k) = (25, 3);
    let adjacency = planner.build_prm_with_rng(samples, k, &mut StdRng::seed_from_u64(42));
    // The same generator reproduces the samples.
    let mut rng = StdRng::seed_from_u64(42);
    let configs: Vec<Vec<f64>> = (0..samples)
        .map(|_| planner.random_config_with_rng(&mut rng))
        .collect();
    assert_eq!(adjacency.len(), samples);
    for (i, neighbours) in adjacency.iter().enumerate() {
        let mut order: Vec<usize> = (0..samples).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            config_dist(&configs[i], &configs[a]).total_cmp(&config_dist(&configs[i], &configs[b]))
        });
        assert_eq!(neighbours, &order[..k].to_vec(), "sample {i}");
    }
    // And the roadmap is reproducible.
    assert_eq!(
        adjacency,
        planner.build_prm_with_rng(samples, k, &mut StdRng::seed_from_u64(42))
    );
}

/// Distance from `center` to the segment `a`–`b`.
fn distance_to_segment(a: &[f64], b: &[f64], center: &[f64]) -> f64 {
    let ab: Vec<f64> = b.iter().zip(a).map(|(q, p)| q - p).collect();
    let ac: Vec<f64> = center.iter().zip(a).map(|(c, p)| c - p).collect();
    let length_sq: f64 = ab.iter().map(|x| x * x).sum();
    let t = if length_sq > 0.0 {
        let along: f64 = ab.iter().zip(&ac).map(|(x, y)| x * y).sum();
        (along / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest: Vec<f64> = a.iter().zip(&ab).map(|(p, d)| p + t * d).collect();
    config_dist(&closest, center)
}

#[test]
fn roadmap_edges_never_cross_an_obstacle() {
    let radius = 0.25;
    let planner = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false)
        .with_obstacles(vec![PlanningObstacle {
            center: vec![0.5, 0.5],
            radius,
        }]);
    let (samples, k) = (60, 5);
    let adjacency = planner.build_prm_with_rng(samples, k, &mut StdRng::seed_from_u64(7));
    let mut rng = StdRng::seed_from_u64(7);
    let configs: Vec<Vec<f64>> = (0..samples)
        .map(|_| planner.random_config_with_rng(&mut rng))
        .collect();
    let mut edges = 0;
    for (i, neighbours) in adjacency.iter().enumerate() {
        assert!(neighbours.len() <= k && !neighbours.contains(&i));
        for &j in neighbours {
            edges += 1;
            // The planner samples the edge at 11 points, so it may graze the
            // obstacle by a few percent of the radius, never cut through it.
            let clearance = distance_to_segment(&configs[i], &configs[j], &[0.5, 0.5]);
            assert!(
                clearance >= 0.97 * radius,
                "edge {i}-{j} passes the obstacle at {clearance}"
            );
        }
    }
    assert!(
        edges > samples,
        "the roadmap should not be empty: {edges} edges"
    );
    // Some candidate edges were rejected: an obstacle-free roadmap is bigger.
    let free = MotionPlanning::new(2, vec![0.0; 2], vec![1.0; 2], 0.1, 100, 0.1, false);
    let free_edges: usize = free
        .build_prm_with_rng(samples, k, &mut StdRng::seed_from_u64(7))
        .iter()
        .map(Vec::len)
        .sum();
    assert!(free_edges > edges, "{free_edges} vs {edges}");
}
