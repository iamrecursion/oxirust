// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Gradient checks of the back-propagation of [`NeuralController`].
//!
//! The analytic gradient is compared with central finite differences of the
//! forward pass, with a hand-derived gradient of a tiny network, and with the
//! algebraic properties it has to satisfy.

use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const ALL_ACTIVATIONS: [Activation; 4] = [
    Activation::Tanh,
    Activation::Relu,
    Activation::Sigmoid,
    Activation::Linear,
];

/// Step of the central finite differences.
const EPS: f64 = 1e-6;

/// Largest tolerated difference between analytic and numerical gradient
/// entries; the numerical gradient itself is good to about 1e-9.
const GRAD_TOL: f64 = 1e-7;

/// A network with deterministic pseudo-random weights *and* biases.
fn network(
    input: usize,
    hidden: &[usize],
    output: usize,
    activation: Activation,
    action_scale: f64,
    seed: u64,
) -> NeuralController {
    let mut net = NeuralController::new(input, hidden, output, action_scale);
    net.activation = activation;
    let mut rng = StdRng::seed_from_u64(seed);
    let params: Vec<f64> = (0..net.num_parameters())
        .map(|_| rng.random_range(-1.0_f64..1.0_f64))
        .collect();
    net.set_parameters(&params);
    net
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Central finite-difference gradient of `θ ↦ d_out · forward(input)`.
fn finite_difference_gradient(net: &NeuralController, input: &[f64], d_out: &[f64]) -> Vec<f64> {
    let params = net.get_parameters();
    let mut probe = net.clone();
    let mut shifted = params.clone();
    (0..params.len())
        .map(|j| {
            shifted[j] = params[j] + EPS;
            probe.set_parameters(&shifted);
            let plus = dot(&probe.forward(input), d_out);
            shifted[j] = params[j] - EPS;
            probe.set_parameters(&shifted);
            let minus = dot(&probe.forward(input), d_out);
            shifted[j] = params[j];
            (plus - minus) / (2.0 * EPS)
        })
        .collect()
}

fn assert_gradients_agree(label: &str, analytic: &[f64], numerical: &[f64]) {
    assert_eq!(analytic.len(), numerical.len(), "{label}: length");
    for (j, (a, n)) in analytic.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < GRAD_TOL,
            "{label}: parameter {j}: backward {a}, finite differences {n}"
        );
    }
}

/// No hidden pre-activation within `margin` of the ReLU kink, where the finite
/// differences would straddle the non-differentiable point.
fn assert_away_from_the_relu_kink(net: &NeuralController, input: &[f64], margin: f64) {
    let cache = net.forward_cached(input);
    for layer in &cache.layers[..net.layers.len()] {
        for z in &layer.pre_activation {
            assert!(z.abs() > margin, "pre-activation {z} is too close to 0");
        }
    }
}

// ── finite-difference agreement ─────────────────────────────────────────

#[test]
fn backward_matches_central_finite_differences_for_every_activation() {
    // A random network with two hidden layers.
    let input = [0.3, -0.7, 1.1];
    let d_out = [0.8, -1.3];
    for activation in ALL_ACTIVATIONS {
        let net = network(3, &[5, 4], 2, activation, 2.5, 11);
        if activation == Activation::Relu {
            assert_away_from_the_relu_kink(&net, &input, 1e-3);
        }
        let analytic = net.backward(&input, &d_out);
        let numerical = finite_difference_gradient(&net, &input, &d_out);
        assert_gradients_agree(&format!("{activation:?}"), &analytic, &numerical);
    }
}

#[test]
fn backward_matches_finite_differences_on_many_shapes_and_seeds() {
    let shapes: [(usize, &[usize], usize); 5] = [
        (2, &[], 1),
        (4, &[6], 3),
        (3, &[7, 5], 2),
        (5, &[8, 6, 4], 4),
        (1, &[3, 3], 1),
    ];
    for (n_in, hidden, n_out) in shapes {
        for seed in 0..3_u64 {
            let mut rng = StdRng::seed_from_u64(1000 + seed);
            let input: Vec<f64> = (0..n_in).map(|_| rng.random_range(-1.5..1.5)).collect();
            let d_out: Vec<f64> = (0..n_out).map(|_| rng.random_range(-2.0..2.0)).collect();
            let net = network(n_in, hidden, n_out, Activation::Tanh, 1.7, seed);
            let analytic = net.backward(&input, &d_out);
            assert_eq!(analytic.len(), net.num_parameters());
            let numerical = finite_difference_gradient(&net, &input, &d_out);
            assert_gradients_agree(
                &format!("{n_in}-{hidden:?}-{n_out} seed {seed}"),
                &analytic,
                &numerical,
            );
        }
    }
}

#[test]
fn backward_agrees_with_finite_differences_for_relu_networks_on_many_inputs() {
    let net = network(3, &[6, 6], 2, Activation::Relu, 1.0, 5);
    let mut rng = StdRng::seed_from_u64(77);
    let mut checked = 0;
    while checked < 10 {
        let input: Vec<f64> = (0..3).map(|_| rng.random_range(-2.0..2.0)).collect();
        let cache = net.forward_cached(&input);
        let near_kink = cache.layers[..2]
            .iter()
            .flat_map(|layer| layer.pre_activation.iter())
            .any(|z| z.abs() < 1e-3);
        if near_kink {
            continue;
        }
        let d_out = [1.0, -0.5];
        let analytic = net.backward(&input, &d_out);
        let numerical = finite_difference_gradient(&net, &input, &d_out);
        assert_gradients_agree("relu", &analytic, &numerical);
        checked += 1;
    }
}

#[test]
fn backward_equals_the_hand_derived_gradient_of_a_tiny_network() {
    // x → w1 x + b1 (linear hidden unit) → s · tanh(w2 a + b2).
    let (w1, b1, w2, b2) = (0.5, 0.1, -0.8, 0.2);
    let (x, scale, d_out) = (2.0, 3.0, 1.5);
    let mut net = NeuralController::new(1, &[1], 1, scale);
    net.activation = Activation::Linear;
    net.set_parameters(&[w1, b1, w2, b2]);

    let a1 = w1 * x + b1;
    let raw = w2 * a1 + b2;
    let t = raw.tanh();
    assert!((net.forward(&[x])[0] - scale * t).abs() < 1e-15);
    let delta_out = d_out * scale * (1.0 - t * t);
    let delta_hidden = w2 * delta_out;
    let expected = [delta_hidden * x, delta_hidden, delta_out * a1, delta_out];

    let got = net.backward(&[x], &[d_out]);
    assert_eq!(got.len(), 4);
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-14, "{got:?} vs {expected:?}");
    }
}

// ── structure of the gradient ───────────────────────────────────────────

#[test]
fn backward_is_linear_in_d_out_and_zero_for_a_zero_loss_gradient() {
    let net = network(3, &[5, 4], 3, Activation::Sigmoid, 2.0, 3);
    let input = [0.2, 0.4, -0.9];
    let zero = net.backward(&input, &[0.0; 3]);
    assert_eq!(zero.len(), net.num_parameters());
    assert!(zero.iter().all(|&g| g == 0.0));

    let d1 = [1.0, 0.0, -2.0];
    let d2 = [0.5, 1.5, 0.25];
    let (a, b) = (0.7, -1.3);
    let combined: Vec<f64> = d1
        .iter()
        .zip(d2.iter())
        .map(|(x, y)| a * x + b * y)
        .collect();
    let lhs = net.backward(&input, &combined);
    let g1 = net.backward(&input, &d1);
    let g2 = net.backward(&input, &d2);
    for ((l, x), y) in lhs.iter().zip(g1.iter()).zip(g2.iter()) {
        assert!((l - (a * x + b * y)).abs() < 1e-12);
    }
}

#[test]
fn a_short_or_long_d_out_is_zero_padded_or_truncated_like_the_forward_pass() {
    let net = network(2, &[4], 2, Activation::Tanh, 1.0, 9);
    let input = [0.5, -0.25];
    let padded = net.backward(&input, &[0.8, 0.0]);
    assert_eq!(net.backward(&input, &[0.8]), padded);
    let truncated = net.backward(&input, &[0.8, 0.0]);
    assert_eq!(net.backward(&input, &[0.8, 0.0, 99.0, -5.0]), truncated);
    assert!(padded.iter().any(|&g| g != 0.0));
}

#[test]
fn the_cache_records_inputs_and_pre_activations_of_every_layer() {
    for activation in ALL_ACTIVATIONS {
        let net = network(3, &[4, 5], 2, activation, 1.5, 21);
        let input = [0.1, -0.2, 0.3];
        let cache = net.forward_cached(&input);
        assert_eq!(cache.layers.len(), 3);
        assert_eq!(cache.layers[0].input, input.to_vec());
        for l in 0..2 {
            let expected_next: Vec<f64> = cache.layers[l]
                .pre_activation
                .iter()
                .map(|&z| activation.apply(z))
                .collect();
            assert_eq!(
                cache.layers[l + 1].input,
                expected_next,
                "{activation:?} layer {l}"
            );
        }
        let layer0 = net.layers[0].forward_cached(&input);
        assert_eq!(layer0, cache.layers[0]);
        assert_eq!(layer0.pre_activation, net.layers[0].forward_linear(&input));
    }
}

#[test]
fn forward_cached_reproduces_forward_bit_for_bit() {
    for activation in ALL_ACTIVATIONS {
        for hidden in [&[][..], &[6][..], &[4, 3][..]] {
            let net = network(3, hidden, 2, activation, 2.2, 4);
            for input in [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [-3.0, 3.0, 3.0]] {
                assert_eq!(net.forward_cached(&input).output, net.forward(&input));
            }
        }
    }
}

#[test]
fn a_cache_of_a_network_with_another_depth_gives_a_zero_gradient() {
    let shallow = network(2, &[3], 1, Activation::Tanh, 1.0, 1);
    let deep = network(2, &[3, 3], 1, Activation::Tanh, 1.0, 2);
    let cache = shallow.forward_cached(&[0.1, 0.2]);
    let gradient = deep.backward_cached(&cache, &[1.0]);
    assert_eq!(gradient.len(), deep.num_parameters());
    assert!(gradient.iter().all(|&g| g == 0.0));
}

// ── activations ─────────────────────────────────────────────────────────

#[test]
fn activation_derivatives_match_central_finite_differences() {
    let points = [-3.0, -1.2, -0.4, 0.3, 0.9, 2.5];
    for activation in ALL_ACTIVATIONS {
        for z in points {
            let numerical = (activation.apply(z + EPS) - activation.apply(z - EPS)) / (2.0 * EPS);
            let analytic = activation.derivative(z, activation.apply(z));
            assert!(
                (analytic - numerical).abs() < 1e-8,
                "{activation:?} at {z}: {analytic} vs {numerical}"
            );
        }
    }
    // The sub-gradient of the ReLU kink is 0.
    assert_eq!(Activation::Relu.derivative(0.0, 0.0), 0.0);
}

// ── layer shapes ────────────────────────────────────────────────────────

#[test]
fn a_ragged_layer_keeps_the_parameter_layout_and_never_panics() {
    // Two weight rows of different length but only one bias: the forward pass
    // pairs entries up like `zip`, so only row 0 and bias 0 take part.
    let layer = DenseLayer {
        weights: vec![vec![1.0, 2.0, 3.0], vec![4.0]],
        biases: vec![0.5],
    };
    let input = [1.0, 10.0];
    let cache = layer.forward_cached(&input);
    assert_eq!(cache.pre_activation, vec![1.0 * 1.0 + 2.0 * 10.0 + 0.5]);

    let mut grad = Vec::new();
    let input_grad = layer.backward_linear(&cache, &[2.0], &mut grad);
    // Row 0 (3 weights: the third input does not exist), row 1 (unused), bias.
    assert_eq!(grad, vec![2.0 * 1.0, 2.0 * 10.0, 0.0, 0.0, 2.0]);
    // Wᵀδ for the two inputs.
    assert_eq!(input_grad, vec![2.0 * 1.0, 2.0 * 2.0]);
}

// ── the log-softmax gradient ────────────────────────────────────────────

#[test]
fn log_softmax_gradient_is_onehot_minus_probabilities() {
    let logits = [0.3, -1.2, 2.0, 0.0];
    let probs = softmax(&logits);
    for action in 0..logits.len() {
        let gradient = log_softmax_gradient(&logits, action).expect("valid action");
        for (k, g) in gradient.iter().enumerate() {
            let expected = if k == action {
                1.0 - probs[k]
            } else {
                -probs[k]
            };
            assert!((g - expected).abs() < 1e-15);
        }
        // The probabilities sum to one, so the gradient sums to zero.
        assert!(gradient.iter().sum::<f64>().abs() < 1e-14);
        // And it is the derivative of log softmax.
        for (k, g) in gradient.iter().enumerate() {
            let mut plus = logits;
            let mut minus = logits;
            plus[k] += EPS;
            minus[k] -= EPS;
            let numerical =
                (softmax(&plus)[action].ln() - softmax(&minus)[action].ln()) / (2.0 * EPS);
            assert!((g - numerical).abs() < 1e-8, "action {action}, logit {k}");
        }
    }
    assert_eq!(log_softmax_gradient(&logits, 4), None);
    assert_eq!(log_softmax_gradient(&[], 0), None);
}
