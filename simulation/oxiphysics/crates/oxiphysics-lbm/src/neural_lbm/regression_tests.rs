// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests of the physics-informed LBM and of the training of the
//! neural networks.

use super::*;
use crate::lattice::{bgk_d2q9, stream_d2q9_periodic};
use std::f64::consts::PI;

/// Deterministic pseudo-random numbers in `[-1, 1)`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Populations of the solver as one array per node.
fn nodes_of(solver: &PhysicsInformedLbm) -> Vec<[f64; 9]> {
    solver
        .lbm_grid
        .chunks_exact(9)
        .map(|chunk| {
            let mut node = [0.0; 9];
            node.copy_from_slice(chunk);
            node
        })
        .collect()
}

/// Independent reference: plain BGK collision followed by periodic streaming,
/// built from the lattice helpers of the crate.
fn reference_step(nodes: &mut [[f64; 9]], nx: usize, ny: usize, tau: f64) {
    let omega = 1.0 / tau;
    for cell in nodes.iter_mut() {
        let (rho, ux, uy) = macros_from_d2q9(cell);
        bgk_d2q9(cell, rho, ux, uy, omega);
    }
    stream_d2q9_periodic(nodes, nx, ny);
}

/// A smooth, strongly non-uniform initial condition.
fn perturbed_field(nx: usize, ny: usize) -> impl Fn(usize, usize) -> (f64, f64, f64) {
    move |x, y| {
        let (sx, cx) = (2.0 * PI * x as f64 / nx as f64).sin_cos();
        let (sy, cy) = (2.0 * PI * y as f64 / ny as f64).sin_cos();
        (1.0 + 0.05 * sx * cy, 0.03 * sy, -0.02 * cx)
    }
}

/// A network with every weight and bias equal to zero.
fn zero_network() -> NeuralLbm {
    let mut network = NeuralLbm::default_network(10, 9);
    for layer in network.layers.iter_mut() {
        for row in layer.weights.iter_mut() {
            row.iter_mut().for_each(|w| *w = 0.0);
        }
        layer.biases.iter_mut().for_each(|b| *b = 0.0);
    }
    network
}

/// The default network with its output layer scaled by `scale`.
fn scaled_network(scale: f64) -> NeuralLbm {
    let mut network = NeuralLbm::default_network(10, 9);
    if let Some(last) = network.layers.last_mut() {
        for row in last.weights.iter_mut() {
            row.iter_mut().for_each(|w| *w *= scale);
        }
    }
    network
}

fn max_difference(a: &[[f64; 9]], b: &[[f64; 9]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(p, q)| p.iter().zip(q.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// PhysicsInformedLbm
// ---------------------------------------------------------------------------

#[test]
fn constructor_validates_the_lattice_the_grid_and_the_relaxation_time() {
    assert!(PhysicsInformedLbm::new(4, 4, 9, 0.8).is_ok());
    assert!(matches!(
        PhysicsInformedLbm::new(4, 4, 7, 0.8),
        Err(Error::UnsupportedVelocitySet { .. })
    ));
    assert!(matches!(
        PhysicsInformedLbm::new(4, 4, 19, 0.8),
        Err(Error::UnsupportedVelocitySet { .. })
    ));
    assert!(matches!(
        PhysicsInformedLbm::new(0, 4, 9, 0.8),
        Err(Error::InvalidGridDimensions { nx: 0, ny: 4 })
    ));
    assert!(matches!(
        PhysicsInformedLbm::new(4, 0, 9, 0.8),
        Err(Error::InvalidGridDimensions { nx: 4, ny: 0 })
    ));
    for tau in [0.5, 0.3, -1.0, 0.0] {
        assert!(matches!(
            PhysicsInformedLbm::new(4, 4, 9, tau),
            Err(Error::UnstableRelaxationTime { .. })
        ));
    }
    let solver = PhysicsInformedLbm::new(5, 3, 9, 0.8).unwrap();
    assert_eq!(
        (solver.nx, solver.ny, solver.n_nodes, solver.q),
        (5, 3, 15, 9)
    );
    assert_eq!(solver.lbm_grid.len(), 9 * 15);
}

#[test]
fn custom_networks_must_have_the_right_shape() {
    assert!(PhysicsInformedLbm::with_network(3, 3, 9, 0.8, zero_network()).is_ok());
    // Wrong feature count.
    let narrow = NeuralLbm::default_network(4, 9);
    assert!(matches!(
        PhysicsInformedLbm::with_network(3, 3, 9, 0.8, narrow),
        Err(Error::ComponentCountMismatch { expected: 10, .. })
    ));
    // Wrong number of corrections.
    let short = NeuralLbm::default_network(10, 5);
    assert!(matches!(
        PhysicsInformedLbm::with_network(3, 3, 9, 0.8, short),
        Err(Error::ComponentCountMismatch { expected: 9, .. })
    ));
    // Declared sizes that disagree with the layers.
    let mut liar = NeuralLbm::default_network(10, 9);
    liar.layers.pop();
    assert!(PhysicsInformedLbm::with_network(3, 3, 9, 0.8, liar).is_err());
    // A network without layers cannot map 10 features to 9 values.
    assert!(
        PhysicsInformedLbm::with_network(3, 3, 9, 0.8, NeuralLbm::new(Vec::new(), 10, 9)).is_err()
    );
}

#[test]
fn a_new_solver_rests_in_equilibrium_and_is_a_fixed_point() {
    let mut solver = PhysicsInformedLbm::new(6, 5, 9, 0.9).unwrap();
    for node in nodes_of(&solver) {
        for (f, w) in node.iter().zip(D2Q9_WEIGHTS.iter()) {
            assert_eq!(f, w, "the rest state must be the equilibrium w_i");
        }
    }
    assert!((solver.total_mass() - 30.0).abs() < 1e-13);
    assert!(solver.compute_residual().unwrap() < 1e-15);
    for _ in 0..5 {
        solver.step().unwrap();
    }
    assert!(solver.compute_residual().unwrap() < 1e-15);
    assert!((solver.total_mass() - 30.0).abs() < 1e-12);
}

#[test]
fn the_untrained_solver_has_a_vanishing_correction() {
    let solver = PhysicsInformedLbm::new(4, 4, 9, 0.7).unwrap();
    let mut fi = equilibrium_d2q9(1.1, 0.04, -0.03);
    fi[3] += 0.01;
    fi[6] -= 0.01;
    let correction = solver.neural_force_correction(&fi);
    assert!(correction.iter().all(|&c| c == 0.0), "{correction:?}");
}

#[test]
fn zero_network_reproduces_bgk_with_periodic_streaming() {
    let (nx, ny, tau) = (7, 5, 0.8);
    let mut solver = PhysicsInformedLbm::with_network(nx, ny, 9, tau, zero_network()).unwrap();
    solver
        .initialize_equilibrium(perturbed_field(nx, ny))
        .unwrap();
    let mut reference = nodes_of(&solver);
    for step in 0..15 {
        solver.step().unwrap();
        reference_step(&mut reference, nx, ny, tau);
        let difference = max_difference(&nodes_of(&solver), &reference);
        assert!(difference < 1e-13, "step {step}: difference {difference}");
    }
    // The state is not trivial: the perturbation is still there.
    assert!(solver.compute_residual().unwrap() > 1e-6);
}

#[test]
fn the_default_constructed_solver_is_plain_bgk_too() {
    let (nx, ny, tau) = (6, 6, 0.65);
    let mut solver = PhysicsInformedLbm::new(nx, ny, 9, tau).unwrap();
    solver
        .initialize_equilibrium(perturbed_field(nx, ny))
        .unwrap();
    let mut reference = nodes_of(&solver);
    for _ in 0..10 {
        solver.step().unwrap();
        reference_step(&mut reference, nx, ny, tau);
    }
    assert!(max_difference(&nodes_of(&solver), &reference) < 1e-13);
}

#[test]
fn bgk_shear_wave_decays_with_the_lattice_viscosity() {
    // u_x(y) = U sin(k y): the amplitude decays as exp(-nu k^2 t) with
    // nu = cs^2 (tau - 1/2).  This exercises the equilibrium, the collision and the
    // direction convention of the streaming at once.
    let (nx, ny, tau) = (4_usize, 32_usize, 0.8);
    let amplitude0 = 0.02;
    let mut solver = PhysicsInformedLbm::new(nx, ny, 9, tau).unwrap();
    solver
        .initialize_equilibrium(|_, y| {
            (
                1.0,
                amplitude0 * (2.0 * PI * y as f64 / ny as f64).sin(),
                0.0,
            )
        })
        .unwrap();
    let steps = 200;
    for _ in 0..steps {
        solver.step().unwrap();
    }
    let mut projection = 0.0;
    for y in 0..ny {
        for x in 0..nx {
            let (_, ux, uy) = solver.macroscopic(x, y).unwrap();
            assert!(uy.abs() < 1e-12, "no cross flow expected, uy = {uy}");
            projection += ux * (2.0 * PI * y as f64 / ny as f64).sin();
        }
    }
    let amplitude = 2.0 * projection / (nx * ny) as f64;
    let nu = (tau - 0.5) / 3.0;
    let k = 2.0 * PI / ny as f64;
    let expected = amplitude0 * (-nu * k * k * steps as f64).exp();
    assert!(
        (amplitude - expected).abs() < 0.02 * expected,
        "amplitude {amplitude} vs analytic {expected}"
    );
}

#[test]
fn a_uniform_flow_is_advected_without_change() {
    let (nx, ny) = (5, 4);
    let mut solver = PhysicsInformedLbm::new(nx, ny, 9, 0.7).unwrap();
    solver
        .initialize_equilibrium(|_, _| (1.0, 0.05, -0.02))
        .unwrap();
    let momentum = solver.total_momentum();
    assert!((momentum[0] - 20.0 * 0.05).abs() < 1e-13);
    assert!((momentum[1] - 20.0 * (-0.02)).abs() < 1e-13);
    assert!(solver.compute_residual().unwrap() < 1e-15);
    for _ in 0..10 {
        solver.step().unwrap();
    }
    let (rho, ux, uy) = solver.macroscopic(2, 1).unwrap();
    assert!((rho - 1.0).abs() < 1e-13 && (ux - 0.05).abs() < 1e-13 && (uy + 0.02).abs() < 1e-13);
}

#[test]
fn the_projection_removes_mass_and_momentum_and_nothing_else() {
    let mut state = 17_u64;
    for _ in 0..20 {
        let mut c = [0.0; 9];
        c.iter_mut().for_each(|v| *v = lcg(&mut state));
        let original = c;
        project_conservative(&mut c);
        let mass: f64 = c.iter().sum();
        let mx: f64 = c
            .iter()
            .zip(D2Q9_VELOCITIES.iter())
            .map(|(v, e)| v * f64::from(e[0]))
            .sum();
        let my: f64 = c
            .iter()
            .zip(D2Q9_VELOCITIES.iter())
            .map(|(v, e)| v * f64::from(e[1]))
            .sum();
        assert!(
            mass.abs() < 1e-15 && mx.abs() < 1e-15 && my.abs() < 1e-15,
            "{mass} {mx} {my}"
        );
        // Idempotent.
        let mut again = c;
        project_conservative(&mut again);
        assert!(
            again
                .iter()
                .zip(c.iter())
                .all(|(a, b)| (a - b).abs() < 1e-15)
        );
        // The removed part lies in the span of the collision invariants.
        let removed: Vec<f64> = original.iter().zip(c.iter()).map(|(a, b)| a - b).collect();
        let mass_removed: f64 = original.iter().sum();
        assert!((removed.iter().sum::<f64>() - mass_removed).abs() < 1e-14);
    }
    // A correction that is already conservative (a non-equilibrium population)
    // is untouched.
    let f = {
        let mut f = equilibrium_d2q9(1.2, 0.03, 0.01);
        f[1] += 0.004;
        f[3] -= 0.004;
        f
    };
    let (rho, ux, uy) = macros_from_d2q9(&f);
    let feq = equilibrium_d2q9(rho, ux, uy);
    let mut neq = [0.0; 9];
    for i in 0..9 {
        neq[i] = f[i] - feq[i];
    }
    let before = neq;
    project_conservative(&mut neq);
    assert!(
        neq.iter()
            .zip(before.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15)
    );
}

#[test]
fn a_nonzero_network_changes_the_flow_but_conserves_mass_and_momentum() {
    let (nx, ny, tau) = (6, 6, 0.8);
    let mut corrected =
        PhysicsInformedLbm::with_network(nx, ny, 9, tau, scaled_network(1e-3)).unwrap();
    corrected
        .initialize_equilibrium(perturbed_field(nx, ny))
        .unwrap();
    let mut reference = nodes_of(&corrected);
    let mass0 = corrected.total_mass();
    let momentum0 = corrected.total_momentum();
    // The network really produces a (conservative) correction.
    let sample = nodes_of(&corrected)[7];
    let correction = corrected.neural_force_correction(&sample);
    assert!(correction.iter().any(|c| c.abs() > 1e-6), "{correction:?}");
    assert!(correction.iter().sum::<f64>().abs() < 1e-15);
    for _ in 0..20 {
        corrected.step().unwrap();
        reference_step(&mut reference, nx, ny, tau);
        assert!((corrected.total_mass() - mass0).abs() < 1e-11);
        let momentum = corrected.total_momentum();
        assert!((momentum[0] - momentum0[0]).abs() < 1e-11);
        assert!((momentum[1] - momentum0[1]).abs() < 1e-11);
    }
    let difference = max_difference(&nodes_of(&corrected), &reference);
    assert!(
        difference > 1e-5,
        "the correction must act on the stress, got {difference}"
    );
}

#[test]
fn the_residual_is_the_rms_change_per_step_and_decays_to_the_steady_state() {
    let (nx, ny) = (4, 16);
    let mut solver = PhysicsInformedLbm::new(nx, ny, 9, 0.8).unwrap();
    solver
        .initialize_equilibrium(|_, y| (1.0, 0.02 * (2.0 * PI * y as f64 / ny as f64).sin(), 0.0))
        .unwrap();
    let before = solver.lbm_grid.clone();
    let residual = solver.compute_residual().unwrap();
    assert_eq!(
        solver.lbm_grid, before,
        "computing the residual must not advance the state"
    );
    solver.step().unwrap();
    let rms = (solver
        .lbm_grid
        .iter()
        .zip(before.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / before.len() as f64)
        .sqrt();
    assert!((residual - rms).abs() < 1e-15, "{residual} vs {rms}");
    assert!(residual > 1e-5);
    // The shear wave decays away: the state approaches a steady (uniform) state.
    for _ in 0..1500 {
        solver.step().unwrap();
    }
    assert!(solver.compute_residual().unwrap() < 1e-3 * residual);
}

#[test]
fn inconsistent_public_state_is_reported_instead_of_panicking() {
    let mut solver = PhysicsInformedLbm::new(4, 4, 9, 0.8).unwrap();
    solver.lbm_grid.pop();
    assert!(matches!(
        solver.step(),
        Err(Error::ComponentCountMismatch { .. })
    ));
    assert!(solver.compute_residual().is_err());
    assert!(solver.macroscopic(0, 0).is_err());
    assert!(
        solver
            .initialize_equilibrium(|_, _| (1.0, 0.0, 0.0))
            .is_err()
    );
    let mut other = PhysicsInformedLbm::new(4, 4, 9, 0.8).unwrap();
    other.q = 7;
    assert!(other.step().is_err());
}

#[test]
fn macroscopic_reports_out_of_range_nodes() {
    let solver = PhysicsInformedLbm::new(3, 2, 9, 0.8).unwrap();
    let (rho, ux, uy) = solver.macroscopic(2, 1).unwrap();
    assert!((rho - 1.0).abs() < 1e-14 && ux.abs() < 1e-15 && uy.abs() < 1e-15);
    assert!(matches!(
        solver.macroscopic(3, 0),
        Err(Error::IndexOutOfBounds {
            x: 3,
            y: 0,
            nx: 3,
            ny: 2
        })
    ));
    assert!(solver.macroscopic(0, 2).is_err());
}

#[test]
fn a_network_of_the_wrong_shape_cannot_make_the_step_panic() {
    let mut solver = PhysicsInformedLbm::new(3, 3, 9, 0.8).unwrap();
    // Swapped-in networks with too few / too many outputs: missing corrections
    // count as zero, surplus ones are ignored.
    solver.neural_correction = NeuralLbm::default_network(10, 4);
    solver.step().unwrap();
    solver.neural_correction = NeuralLbm::default_network(3, 12);
    solver.step().unwrap();
    assert!((solver.total_mass() - 9.0).abs() < 1e-12);
}

#[test]
fn collision_correction_training_learns_a_viscosity_correction() {
    // The model relaxes with tau = 1.0, the reference with tau_ref = 0.7: the
    // difference is the conservative correction (1/tau - 1/tau_ref) f_neq.
    let tau = 1.0;
    let tau_ref = 0.7;
    let mut state = 4711_u64;
    let mut pre = Vec::new();
    let mut reference = Vec::new();
    for _ in 0..150 {
        let rho = 1.0 + 0.05 * lcg(&mut state);
        let (ux, uy) = (0.04 * lcg(&mut state), 0.04 * lcg(&mut state));
        let mut noise = [0.0; 9];
        noise.iter_mut().for_each(|v| *v = 0.05 * lcg(&mut state));
        project_conservative(&mut noise);
        let feq = equilibrium_d2q9(rho, ux, uy);
        let mut f = [0.0; 9];
        for i in 0..9 {
            f[i] = feq[i] + noise[i];
        }
        let mut post = [0.0; 9];
        for i in 0..9 {
            post[i] = f[i] - (f[i] - feq[i]) / tau_ref;
        }
        pre.push(f);
        reference.push(post);
    }
    let mut solver = PhysicsInformedLbm::new(4, 4, 9, tau).unwrap();
    // Error of the uncorrected BGK collision against the reference.
    let collision_error = |solver: &PhysicsInformedLbm| -> f64 {
        let mut sum = 0.0;
        for (f, post) in pre.iter().zip(reference.iter()) {
            let (rho, ux, uy) = macros_from_d2q9(f);
            let feq = equilibrium_d2q9(rho, ux, uy);
            let c = solver.neural_force_correction(f);
            for i in 0..9 {
                let model = f[i] - (f[i] - feq[i]) / tau + c[i];
                sum += (post[i] - model).powi(2);
            }
        }
        sum / (9 * pre.len()) as f64
    };
    let before = collision_error(&solver);
    assert!(
        before > 1e-6,
        "the two relaxation times must differ visibly: {before}"
    );
    let loss = solver
        .train_collision_correction(&pre, &reference, Optimizer::Adam, 0.01, 800)
        .unwrap();
    let after = collision_error(&solver);
    assert!(
        loss.is_finite() && loss < before,
        "loss {loss} vs initial error {before}"
    );
    assert!(
        after < 0.1 * before,
        "training must remove most of the error: {before} -> {after}"
    );
    // The learned correction is conservative at every sample.
    for f in &pre {
        let c = solver.neural_force_correction(f);
        assert!(c.iter().sum::<f64>().abs() < 1e-14);
    }
    // Mismatched sample counts are rejected.
    assert!(matches!(
        solver.train_collision_correction(
            &pre,
            &reference[..10],
            Optimizer::GradientDescent,
            1.0,
            1
        ),
        Err(Error::ComponentCountMismatch { .. })
    ));
}

// ---------------------------------------------------------------------------
// NeuralLbm training
// ---------------------------------------------------------------------------

fn regression_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let (x1, x2) = (-1.0 + 0.4 * i as f64, -1.0 + 0.4 * j as f64);
            inputs.push(vec![x1, x2]);
            targets.push(vec![0.5 * x1 - 0.3 * x2 + 0.1, x1 * x2]);
        }
    }
    (inputs, targets)
}

#[test]
fn training_lowers_the_loss() {
    let (inputs, targets) = regression_data();
    let mut network = NeuralLbm::default_network(2, 2);
    let initial = network.mean_squared_error(&inputs, &targets).unwrap();
    let trained = network.train(&inputs, &targets, 0.1, 800).unwrap();
    assert!(initial > 0.05, "initial loss {initial}");
    assert!(trained < 0.2 * initial, "loss {initial} -> {trained}");
    // The returned value is the error of the trained network.
    let again = network.mean_squared_error(&inputs, &targets).unwrap();
    assert!((trained - again).abs() < 1e-15);
    // Training further keeps lowering it.
    let more = network.train(&inputs, &targets, 0.1, 800).unwrap();
    assert!(more < trained, "{trained} -> {more}");
}

#[test]
fn adam_fits_small_scale_targets_that_gradient_descent_leaves_unlearned() {
    // Targets of order 1e-2 (the scale of lattice collision residuals) behind a
    // zero output layer: the hidden-layer gradients are tiny, so fixed-step
    // gradient descent only fits the output layer, while Adam's step does not
    // depend on the gradient scale.
    let inputs: Vec<Vec<f64>> = regression_data().0;
    let targets: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| vec![1.0e-2 * (x[0] - 0.5 * x[1]), 1.0e-2 * x[0] * x[1]])
        .collect();
    let mut network = NeuralLbm::default_network(2, 2);
    network.zero_output_layer();
    let initial = network.mean_squared_error(&inputs, &targets).unwrap();
    let mut by_descent = network.clone();
    let descent = by_descent.train(&inputs, &targets, 0.1, 1000).unwrap();
    let adam = network
        .train_with(Optimizer::Adam, &inputs, &targets, 1.0e-3, 1000)
        .unwrap();
    assert!(adam < 0.01 * initial, "Adam: {initial} -> {adam}");
    assert!(adam < descent, "Adam {adam} vs gradient descent {descent}");
    assert!(
        descent < initial,
        "gradient descent must still lower the loss"
    );
}

#[test]
fn the_first_adam_step_moves_every_parameter_by_the_learning_rate() {
    // After one bias-corrected step m_hat = g and v_hat = g^2, so each parameter
    // moves by lr * g / (|g| + eps) ~ lr * sign(g), independent of |g|.
    let (inputs, targets) = (
        vec![
            vec![0.6, -0.4, 0.9],
            vec![-0.7, 0.2, 0.5],
            vec![0.3, 0.8, -0.6],
        ],
        vec![vec![0.4, -0.2], vec![0.1, 0.5], vec![-0.3, 0.2]],
    );
    let lr = 1e-3;
    let reference = tiny_network(ActivationFn::Tanh);
    let mut descent = reference.clone();
    descent.train(&inputs, &targets, lr, 1).unwrap();
    let mut adam = reference.clone();
    adam.train_with(Optimizer::Adam, &inputs, &targets, lr, 1)
        .unwrap();
    let mut checked = 0;
    for ((layer, gd_layer), adam_layer) in reference
        .layers
        .iter()
        .zip(descent.layers.iter())
        .zip(adam.layers.iter())
    {
        let before = layer.weights.iter().flatten().chain(layer.biases.iter());
        let after_gd = gd_layer
            .weights
            .iter()
            .flatten()
            .chain(gd_layer.biases.iter());
        let after_adam = adam_layer
            .weights
            .iter()
            .flatten()
            .chain(adam_layer.biases.iter());
        for ((&w0, &w_gd), &w_adam) in before.zip(after_gd).zip(after_adam) {
            let gradient = (w0 - w_gd) / lr;
            if gradient.abs() > 1e-4 {
                let moved = w0 - w_adam;
                assert!(
                    (moved - lr * gradient.signum()).abs() < 1e-3 * lr,
                    "gradient {gradient}: Adam moved the parameter by {moved}"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 10,
        "only {checked} parameters had a usable gradient"
    );
}

#[test]
fn gradient_descent_through_train_with_is_train() {
    let (inputs, targets) = regression_data();
    let mut a = NeuralLbm::default_network(2, 2);
    let mut b = a.clone();
    let loss_a = a.train(&inputs, &targets, 0.05, 20).unwrap();
    let loss_b = b
        .train_with(Optimizer::GradientDescent, &inputs, &targets, 0.05, 20)
        .unwrap();
    assert_eq!(loss_a, loss_b);
    for (x, y) in a.layers.iter().zip(b.layers.iter()) {
        assert_eq!(x.weights, y.weights);
        assert_eq!(x.biases, y.biases);
    }
}

#[test]
fn training_with_zero_epochs_changes_nothing() {
    let (inputs, targets) = regression_data();
    let mut network = NeuralLbm::default_network(2, 2);
    let before = network.clone();
    let loss = network.train(&inputs, &targets, 0.1, 0).unwrap();
    assert_eq!(loss, before.mean_squared_error(&inputs, &targets).unwrap());
    for (a, b) in network.layers.iter().zip(before.layers.iter()) {
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.biases, b.biases);
    }
}

/// A small network with deterministic weights and the given hidden activation.
fn tiny_network(hidden: ActivationFn) -> NeuralLbm {
    let layer1 = NeuralLbmLayer::new(
        vec![
            vec![0.31, -0.42, 0.17],
            vec![-0.25, 0.38, 0.29],
            vec![0.44, 0.12, -0.36],
            vec![-0.15, -0.27, 0.41],
        ],
        vec![0.21, -0.18, 0.33, 0.12],
        hidden,
    );
    let layer2 = NeuralLbmLayer::new(
        vec![
            vec![0.52, -0.33, 0.21, 0.14],
            vec![-0.28, 0.46, -0.19, 0.37],
        ],
        vec![0.05, -0.07],
        ActivationFn::Linear,
    );
    NeuralLbm::new(vec![layer1, layer2], 3, 2)
}

#[test]
fn backpropagation_matches_central_finite_differences_for_every_activation() {
    let inputs = vec![
        vec![0.6, -0.4, 0.9],
        vec![-0.7, 0.2, 0.5],
        vec![0.3, 0.8, -0.6],
        vec![-0.2, -0.9, 0.1],
    ];
    let targets = vec![
        vec![0.4, -0.2],
        vec![0.1, 0.5],
        vec![-0.3, 0.2],
        vec![0.6, -0.1],
    ];
    let learning_rate = 1e-3;
    let step = 1e-6;
    for activation in [
        ActivationFn::Tanh,
        ActivationFn::Sigmoid,
        ActivationFn::Linear,
        ActivationFn::Relu,
    ] {
        let reference = tiny_network(activation.clone());
        let mut updated = reference.clone();
        updated.train(&inputs, &targets, learning_rate, 1).unwrap();
        for (l, layer) in reference.layers.iter().enumerate() {
            for (i, row) in layer.weights.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    let analytic = (weight - updated.layers[l].weights[i][j]) / learning_rate;
                    let numeric =
                        finite_difference(&reference, &inputs, &targets, step, |net, h| {
                            net.layers[l].weights[i][j] += h;
                        });
                    assert!(
                        (analytic - numeric).abs() < 1e-7 * (1.0 + numeric.abs()),
                        "{activation:?}: dW[{l}][{i}][{j}] analytic {analytic} vs numeric {numeric}"
                    );
                }
                let analytic = (layer.biases[i] - updated.layers[l].biases[i]) / learning_rate;
                let numeric = finite_difference(&reference, &inputs, &targets, step, |net, h| {
                    net.layers[l].biases[i] += h;
                });
                assert!(
                    (analytic - numeric).abs() < 1e-7 * (1.0 + numeric.abs()),
                    "{activation:?}: db[{l}][{i}] analytic {analytic} vs numeric {numeric}"
                );
            }
        }
    }
}

/// Central difference of the mean squared error along one parameter.
fn finite_difference(
    network: &NeuralLbm,
    inputs: &[Vec<f64>],
    targets: &[Vec<f64>],
    step: f64,
    perturb: impl Fn(&mut NeuralLbm, f64),
) -> f64 {
    let mut plus = network.clone();
    perturb(&mut plus, step);
    let mut minus = network.clone();
    perturb(&mut minus, -step);
    let loss_plus = plus.mean_squared_error(inputs, targets).unwrap();
    let loss_minus = minus.mean_squared_error(inputs, targets).unwrap();
    (loss_plus - loss_minus) / (2.0 * step)
}

#[test]
fn pre_activation_is_the_argument_of_the_activation() {
    let network = tiny_network(ActivationFn::Tanh);
    let x = [0.3, -0.8, 0.5];
    let layer = &network.layers[0];
    let z = layer.pre_activation(&x);
    let expected: Vec<f64> = layer
        .weights
        .iter()
        .zip(layer.biases.iter())
        .map(|(row, b)| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum::<f64>() + b)
        .collect();
    assert_eq!(z, expected);
    let activated: Vec<f64> = z.iter().map(|&v| v.tanh()).collect();
    assert_eq!(layer.forward(&x), activated);
}

#[test]
fn training_rejects_data_that_does_not_fit_the_network() {
    let (inputs, targets) = regression_data();
    let mut network = NeuralLbm::default_network(2, 2);
    // Empty set, unequal counts, wrong widths, non-finite entries.
    assert!(matches!(
        network.train(&[], &[], 0.1, 1),
        Err(Error::IncompatibleParameters { .. })
    ));
    assert!(matches!(
        network.train(&inputs, &targets[..5], 0.1, 1),
        Err(Error::ComponentCountMismatch { .. })
    ));
    let bad_inputs = vec![vec![1.0, 2.0, 3.0]];
    assert!(matches!(
        network.train(&bad_inputs, &targets[..1], 0.1, 1),
        Err(Error::ComponentCountMismatch {
            expected: 2,
            found: 3,
            ..
        })
    ));
    let bad_targets = vec![vec![1.0]];
    assert!(matches!(
        network.train(&inputs[..1], &bad_targets, 0.1, 1),
        Err(Error::ComponentCountMismatch {
            expected: 2,
            found: 1,
            ..
        })
    ));
    let nan_targets = vec![vec![0.0, f64::NAN]];
    assert!(matches!(
        network.train(&inputs[..1], &nan_targets, 0.1, 1),
        Err(Error::NonFiniteField { index: 1, .. })
    ));
    // Learning rates must be finite and non-negative.
    for rate in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            network.train(&inputs, &targets, rate, 1),
            Err(Error::ParameterOutOfBounds { .. })
        ));
    }
    // A network without layers has no parameters to train.
    let mut empty = NeuralLbm::new(Vec::new(), 2, 2);
    assert!(empty.train(&inputs, &targets, 0.1, 1).is_err());
    // Inconsistent layer chain.
    let mut broken = NeuralLbm::default_network(2, 2);
    broken.layers[1].weights[0].pop();
    assert!(broken.train(&inputs, &targets, 0.1, 1).is_err());
}

#[test]
fn diverging_training_restores_the_network() {
    let layer = NeuralLbmLayer::new(vec![vec![1.0]], vec![0.0], ActivationFn::Linear);
    let mut network = NeuralLbm::new(vec![layer], 1, 1);
    let inputs = vec![vec![1.0], vec![2.0]];
    let targets = vec![vec![0.0], vec![0.0]];
    let result = network.train(&inputs, &targets, 1.0e200, 3);
    assert!(
        matches!(result, Err(Error::NonFiniteField { .. })),
        "{result:?}"
    );
    assert_eq!(network.layers[0].weights, vec![vec![1.0]]);
    assert_eq!(network.layers[0].biases, vec![0.0]);
}

#[test]
fn zero_output_layer_silences_the_network() {
    let mut network = NeuralLbm::default_network(10, 9);
    let features = [1.0, 0.1, 0.2, 0.3, 0.1, 0.0, 0.05, 0.1, 0.2, 0.1];
    assert!(network.forward(&features).iter().any(|v| v.abs() > 1e-3));
    network.zero_output_layer();
    assert!(network.forward(&features).iter().all(|&v| v == 0.0));
    // The hidden layers keep their weights so that training has a gradient.
    assert!(
        network.layers[0]
            .weights
            .iter()
            .flatten()
            .any(|&w| w != 0.0)
    );
    // No layers, no effect.
    NeuralLbm::new(Vec::new(), 1, 1).zero_output_layer();
}
