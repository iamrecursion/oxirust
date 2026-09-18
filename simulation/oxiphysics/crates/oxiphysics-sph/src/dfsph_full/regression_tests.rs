// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

// Regression tests of the DFSPH pressure stages (`divergence_free_solve`,
// `density_solve`, `dfsph_two_stage_solve`): the divergence stiffness used to
// miss a factor ρ_i (a correction ≈ 1/ρ_i too weak), and the pure Jacobi
// iterations diverged as soon as the corrections had the right size. They live
// in their own file so that `dfsph_full.rs` stays below the 2000-line limit.

use super::*;

/// Cubic lattice of `n_side³` particles with spacing `spacing`. Every particle
/// carries the mass `rho0 · rest_spacing³` (so the lattice has the rest density
/// when `spacing == rest_spacing`), and `h = h_factor · rest_spacing`.
///
/// Densities and α factors are computed; velocities are zero.
fn lattice(
    n_side: usize,
    spacing: f64,
    rest_spacing: f64,
    h_factor: f64,
    rho0: f64,
) -> (DfsphState, Vec<Vec<usize>>) {
    let h = h_factor * rest_spacing;
    let mass = rho0 * rest_spacing.powi(3);
    let mut state = DfsphState::new(h, rho0, 1.0e-3);
    for i in 0..n_side {
        for j in 0..n_side {
            for k in 0..n_side {
                state.add_particle(
                    [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing],
                    mass,
                );
            }
        }
    }
    let neighbors = find_neighbors(&state.positions, h);
    state.compute_densities(&neighbors);
    state.compute_alphas(&neighbors);
    (state, neighbors)
}

/// Geometric centre of the lattice.
fn centre(state: &DfsphState) -> [f64; 3] {
    let n = state.positions.len() as f64;
    let mut c = [0.0; 3];
    for p in &state.positions {
        for (c_d, &p_d) in c.iter_mut().zip(p.iter()) {
            *c_d += p_d / n;
        }
    }
    c
}

/// Radial compression `v = -rate (x - c) exp(-|x - c|² / 2σ²)`; `sigma = None`
/// gives the uniform compression `v = -rate (x - c)`.
fn set_radial_compression(state: &mut DfsphState, rate: f64, sigma: Option<f64>) {
    let c = centre(state);
    for (p, v) in state.positions.iter().zip(state.velocities.iter_mut()) {
        let d = sub3(*p, c);
        let window = match sigma {
            Some(s) => (-dot3(d, d) / (2.0 * s * s)).exp(),
            None => 1.0,
        };
        *v = scale3(d, -rate * window);
    }
}

/// `max_i max(div v_i, 0)`: the quantity the divergence stage drives to zero.
fn max_compression(state: &DfsphState, neighbors: &[Vec<usize>]) -> f64 {
    (0..state.n_particles())
        .map(|i| state.velocity_divergence(i, neighbors).max(0.0))
        .fold(0.0_f64, f64::max)
}

/// `max_i max(ρ*_i − ρ0, 0) / ρ0` with `ρ*_i = ρ_i + dt Dρ_i/Dt` of the current
/// velocities: the quantity the density stage drives to zero.
fn max_density_excess(state: &DfsphState, neighbors: &[Vec<usize>]) -> f64 {
    (0..state.n_particles())
        .map(|i| {
            let rho_star = state.densities[i]
                + state.dt * state.velocity_divergence(i, neighbors) * state.densities[i];
            (rho_star - state.rho0).max(0.0) / state.rho0
        })
        .fold(0.0_f64, f64::max)
}

fn max_speed(state: &DfsphState) -> f64 {
    state
        .velocities
        .iter()
        .map(|v| norm3(*v))
        .fold(0.0_f64, f64::max)
}

fn total_momentum(state: &DfsphState) -> [f64; 3] {
    let mut p = [0.0; 3];
    for (v, &m) in state.velocities.iter().zip(state.masses.iter()) {
        p = add3(p, scale3(*v, m));
    }
    p
}

/// Largest component-wise difference of two velocity fields, relative to the
/// largest speed in `reference`.
fn max_relative_difference(reference: &[[f64; 3]], other: &[[f64; 3]]) -> f64 {
    let scale = reference
        .iter()
        .map(|v| norm3(*v))
        .fold(0.0_f64, f64::max)
        .max(1.0e-300);
    reference
        .iter()
        .zip(other.iter())
        .map(|(a, b)| norm3(sub3(*a, *b)))
        .fold(0.0_f64, f64::max)
        / scale
}

// ---------------------------------------------------------------------------
// Divergence stage: strength of the correction
// ---------------------------------------------------------------------------

/// The divergence-stage correction must not depend on the absolute density
/// scale: `div v` and α_i = ρ_i²/D_i are invariant when all masses (hence ρ)
/// are scaled, and κ_i = div v_i α_i/(dt ρ_i) ∝ 1/ρ, so Δv = -dt Σ m_j κ ∇W is
/// invariant. With the former extra `1/ρ_i` the correction scaled like `1/ρ`
/// and was 1000× too weak at ρ0 = 1000 kg/m³ compared with ρ0 = 1 kg/m³.
#[test]
fn divergence_correction_is_independent_of_density_scale() {
    let spacing = 0.05;
    let mut corrected = Vec::new();
    for rho0 in [1.0, 1000.0] {
        let (mut state, neighbors) = lattice(7, spacing, spacing, 1.2, rho0);
        set_radial_compression(&mut state, 1.0, Some(1.5 * spacing));
        let before = state.velocities.clone();
        let iters = state.divergence_free_solve(&neighbors, 1, 0.0);
        assert_eq!(iters, 1);
        let change = max_relative_difference(&before, &state.velocities);
        assert!(
            change > 0.05,
            "rho0 = {rho0}: the correction changed the velocities by only {change:e} (relative)"
        );
        corrected.push(state.velocities.clone());
    }
    let diff = max_relative_difference(&corrected[0], &corrected[1]);
    assert!(
        diff < 1.0e-9,
        "one-iteration result depends on the density scale: relative difference {diff:e}"
    );
}

/// The same invariance for the standalone two-stage solver (it had its own copy
/// of the divergence stiffness).
#[test]
fn two_stage_divergence_correction_is_independent_of_density_scale() {
    let spacing = 0.05;
    let mut corrected = Vec::new();
    for rho0 in [1.0, 1000.0] {
        let (mut state, neighbors) = lattice(7, spacing, spacing, 1.2, rho0);
        set_radial_compression(&mut state, 1.0, Some(1.5 * spacing));
        let before = state.velocities.clone();
        let mut velocities = state.velocities.clone();
        let (iters1, iters2) = dfsph_two_stage_solve(
            &state.positions,
            &mut velocities,
            &state.densities,
            &state.alphas,
            &state.masses,
            &neighbors,
            DfsphSolverConfig {
                h: state.h,
                rho0: state.rho0,
                dt: state.dt,
                max_iter_div: 1,
                div_eps: 0.0,
                max_iter_den: 0,
                den_eps: 0.0,
                relaxation: DEFAULT_RELAXATION,
            },
        );
        assert_eq!((iters1, iters2), (1, 0));
        assert!(max_relative_difference(&before, &velocities) > 0.05);
        corrected.push(velocities);
    }
    let diff = max_relative_difference(&corrected[0], &corrected[1]);
    assert!(
        diff < 1.0e-9,
        "one-iteration result depends on the density scale: relative difference {diff:e}"
    );
}

/// Exact check of the stiffness for one isolated pair (one neighbour each, so
/// `D_i = 2 m² (W')²`): a full step (`ω = 1`) over-corrects by exactly a factor
/// two and reverses the approach velocity, `ω = 1/2` removes it in one
/// iteration. This pins the absolute size of κ at ρ0 = 1000 kg/m³ (the old
/// stiffness gave a relative-velocity change of ≈ 1e-3 of the approach speed).
#[test]
fn isolated_pair_stiffness_has_the_exact_magnitude() {
    let h = 0.1;
    let separation = 0.1;
    let approach = 2.0;
    for rho0 in [1.0, 1000.0] {
        let mass = rho0 * 1.0e-3;
        for (omega, expected_relative_velocity) in [(0.5, 0.0), (1.0, -approach)] {
            let mut state = DfsphState::new(h, rho0, 1.0e-3);
            state.relaxation = omega;
            state.add_particle([0.0, 0.0, 0.0], mass);
            state.add_particle([separation, 0.0, 0.0], mass);
            state.velocities[0] = [approach, 0.0, 0.0];
            let neighbors = vec![vec![1], vec![0]];
            state.compute_densities(&neighbors);
            state.compute_alphas(&neighbors);
            assert!(state.velocity_divergence(0, &neighbors) > 0.0);

            state.divergence_free_solve(&neighbors, 1, 0.0);

            let relative = state.velocities[0][0] - state.velocities[1][0];
            assert!(
                (relative - expected_relative_velocity).abs() < 1.0e-9 * approach,
                "rho0 = {rho0}, omega = {omega}: relative velocity {relative}, expected {expected_relative_velocity}"
            );
            // Pairwise antisymmetric correction: the momentum is conserved.
            let momentum = total_momentum(&state);
            assert!((momentum[0] - mass * approach).abs() < 1.0e-12 * mass * approach);
            assert!(momentum[1].abs() < 1.0e-15 && momentum[2].abs() < 1.0e-15);
        }
    }
}

/// After a full step reversed the pair (ω = 1) the pair is expanding, i.e. no
/// longer compressed, and further iterations must leave it alone instead of
/// pulling it together again (one-sided correction).
#[test]
fn expanding_pair_is_not_pulled_back_together() {
    let mut state = DfsphState::new(0.1, 1000.0, 1.0e-3);
    state.relaxation = 1.0;
    state.add_particle([0.0, 0.0, 0.0], 1.0);
    state.add_particle([0.1, 0.0, 0.0], 1.0);
    state.velocities[0] = [2.0, 0.0, 0.0];
    let neighbors = vec![vec![1], vec![0]];
    state.compute_densities(&neighbors);
    state.compute_alphas(&neighbors);

    state.divergence_free_solve(&neighbors, 1, 0.0);
    let after_one = state.velocities.clone();
    assert!(state.velocity_divergence(0, &neighbors) < 0.0);

    let iters = state.divergence_free_solve(&neighbors, 5, 1.0e-12);
    assert_eq!(
        iters, 1,
        "no compression is left, the solve must stop at once"
    );
    assert_eq!(state.velocities, after_one);
}

// ---------------------------------------------------------------------------
// Divergence stage: convergence on lattices
// ---------------------------------------------------------------------------

/// A localised compressive velocity field on a lattice at ρ0 = 1000 kg/m³:
/// one iteration removes most of the compression, the solve converges within
/// 20 iterations, momentum is conserved and the velocities stay bounded.
#[test]
fn localized_compression_is_removed_on_a_lattice() {
    let spacing = 0.05;
    let (mut state, neighbors) = lattice(9, spacing, spacing, 1.2, 1000.0);
    set_radial_compression(&mut state, 1.0, Some(1.5 * spacing));
    let initial = max_compression(&state, &neighbors);
    let speed_before = max_speed(&state);
    assert!(initial > 1.0, "test field is not compressive: {initial}");

    // One iteration cuts the compression by far more than half.
    let mut one = state.clone();
    one.divergence_free_solve(&neighbors, 1, 0.0);
    let after_one = max_compression(&one, &neighbors);
    assert!(
        after_one < 0.5 * initial,
        "one iteration: {after_one} vs initial {initial}"
    );

    // Converges within 20 iterations to 1 % of the initial compression.
    let momentum_before = total_momentum(&state);
    let iters = state.divergence_free_solve(&neighbors, 20, 0.01 * initial);
    assert!(iters < 20, "did not converge: {iters} iterations");
    assert!(max_compression(&state, &neighbors) < 0.01 * initial);

    let momentum_after = total_momentum(&state);
    let mass: f64 = state.masses.iter().sum();
    for d in 0..3 {
        assert!(
            (momentum_after[d] - momentum_before[d]).abs() < 1.0e-9 * mass * speed_before,
            "momentum component {d} changed: {} -> {}",
            momentum_before[d],
            momentum_after[d]
        );
    }
    assert!(
        max_speed(&state) < 2.0 * speed_before,
        "velocities blew up: {} vs {speed_before}",
        max_speed(&state)
    );
}

/// Uniform compression of a finite cluster cannot be removed by one local
/// sweep (the interior particles see identical stiffnesses, so their pressure
/// gradients cancel and the correction has to propagate inwards from the free
/// surface), but the iteration must converge steadily.
#[test]
fn uniform_compression_converges_from_the_surface_inwards() {
    let spacing = 0.05;
    let (mut state, neighbors) = lattice(7, spacing, spacing, 1.2, 1000.0);
    set_radial_compression(&mut state, 1.0, None);
    let initial = max_compression(&state, &neighbors);
    let speed_before = max_speed(&state);
    assert!(
        initial > 2.5,
        "uniform compression rate 3/s expected: {initial}"
    );

    // Monotone decrease of the compressive divergence at the checked sweeps.
    let mut previous = initial;
    let mut probe = state.clone();
    for sweep in 1..=6 {
        probe.divergence_free_solve(&neighbors, 1, 0.0);
        let now = max_compression(&probe, &neighbors);
        assert!(
            now < previous,
            "sweep {sweep}: compression did not decrease ({previous} -> {now})"
        );
        previous = now;
    }

    let iters = state.divergence_free_solve(&neighbors, 20, 0.01 * initial);
    assert!(iters < 20, "did not converge: {iters} iterations");
    assert!(max_compression(&state, &neighbors) < 0.01 * initial);
    assert!(max_speed(&state) < 2.0 * speed_before);
}

/// The previous (unclamped, un-relaxed) Jacobi iteration amplified its highest
/// modes by ≈ 3 per sweep once the stiffness had the right size (relative
/// divergence ≈ 1e20 after 20 sweeps on a 7³ lattice). Run many sweeps from a
/// single particle kick and require bounded, converging results for several
/// kernel widths.
#[test]
fn pressure_iteration_does_not_diverge_for_a_particle_kick() {
    for h_factor in [1.0, 1.2, 2.0] {
        let spacing = 0.05;
        let (mut state, neighbors) = lattice(7, spacing, spacing, h_factor, 1000.0);
        let centre_index = state.n_particles() / 2;
        state.velocities[centre_index] = [1.0, 0.0, 0.0];
        let initial = max_compression(&state, &neighbors);
        assert!(initial > 0.0);

        let iters = state.divergence_free_solve(&neighbors, 200, 1.0e-6 * initial);
        assert!(
            iters < 200,
            "h_factor {h_factor}: no convergence in 200 sweeps"
        );
        assert!(max_compression(&state, &neighbors) < 1.0e-6 * initial);
        let speed = max_speed(&state);
        assert!(
            speed.is_finite() && speed < 10.0,
            "h_factor {h_factor}: speeds blew up to {speed}"
        );
    }
}

// ---------------------------------------------------------------------------
// Density stage
// ---------------------------------------------------------------------------

/// A lattice compressed to 95 % of the rest spacing: the density stage removes
/// the compression without exciting the Jacobi instability, leaves the interior
/// particle at the rest density and conserves momentum.
#[test]
fn density_stage_removes_compression_of_a_compressed_lattice() {
    let rest_spacing = 0.05;
    let (mut state, neighbors) = lattice(7, 0.95 * rest_spacing, rest_spacing, 1.2, 1000.0);
    let interior = state.n_particles() / 2;
    assert!(
        state.densities[interior] > 1.1 * state.rho0,
        "interior density {} is not compressed",
        state.densities[interior]
    );
    let initial = max_density_excess(&state, &neighbors);
    assert!(initial > 0.1, "initial compressive error {initial}");

    let iters = state.density_solve(&neighbors, 100, 1.0e-3);
    assert!(iters < 100, "density stage did not converge");
    let final_excess = max_density_excess(&state, &neighbors);
    assert!(
        final_excess < 1.0e-3,
        "final compressive error {final_excess}"
    );

    // The interior particle is driven to (not below) the rest density.
    let rho_star = state.densities[interior]
        + state.dt * state.velocity_divergence(interior, &neighbors) * state.densities[interior];
    assert!(
        (rho_star / state.rho0 - 1.0).abs() < 5.0e-3,
        "predicted interior density {rho_star}"
    );

    let momentum = total_momentum(&state);
    let mass: f64 = state.masses.iter().sum();
    let speed = max_speed(&state);
    assert!(speed.is_finite() && speed < 50.0, "speeds blew up: {speed}");
    for (d, &p) in momentum.iter().enumerate() {
        assert!(
            p.abs() < 1.0e-9 * mass * speed,
            "momentum component {d}: {p}"
        );
    }
}

/// A cluster whose kernel-sum densities all lie below the rest density (the
/// surface particles are strongly under-dense because their kernel sums are
/// truncated) has nothing to compress. A pressure cannot repair under-dense
/// particles, so the pressure stages must leave a free-falling cluster alone:
/// every particle keeps exactly the speed `g t`. (The unclamped density stage
/// multiplied the large negative surface error into a huge pull-together force
/// and blew up exponentially.)
#[test]
fn free_falling_cluster_is_not_disturbed_by_the_pressure_stages() {
    let spacing = 0.05;
    let (mut state, _) = lattice(5, spacing, spacing, 1.2, 1000.0);
    let max_density = state.densities.iter().cloned().fold(0.0_f64, f64::max);
    state.rho0 = 1.05 * max_density;
    let dt = state.dt;
    let gravity = [0.0, -9.81, 0.0];
    let mut sim = DfsphSimulation::new(state, gravity, 0.0);
    let steps = 20;
    sim.run(steps);
    let expected_speed = 9.81 * dt * steps as f64;
    for (i, v) in sim.state.velocities.iter().enumerate() {
        assert!(
            (v[1] + expected_speed).abs() < 1.0e-9,
            "particle {i}: vy = {}, expected {}",
            v[1],
            -expected_speed
        );
        assert!(
            v[0].abs() < 1.0e-9 && v[2].abs() < 1.0e-9,
            "particle {i}: {v:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Consistency of the two implementations
// ---------------------------------------------------------------------------

/// `dfsph_two_stage_solve` and the `DfsphState` solvers implement the same
/// iterations: identical velocities for the same inputs and relaxation.
#[test]
fn two_stage_solver_matches_the_state_solvers() {
    let rest_spacing = 0.05;
    let (mut state, neighbors) = lattice(6, 0.95 * rest_spacing, rest_spacing, 1.2, 1000.0);
    set_radial_compression(&mut state, 0.5, Some(2.0 * rest_spacing));
    for relaxation in [DEFAULT_RELAXATION, 0.3] {
        state.relaxation = relaxation;
        let cfg = DfsphSolverConfig {
            h: state.h,
            rho0: state.rho0,
            dt: state.dt,
            max_iter_div: 4,
            div_eps: 0.0,
            max_iter_den: 7,
            den_eps: 0.0,
            relaxation,
        };

        let mut via_state = state.clone();
        via_state.divergence_free_solve(&neighbors, cfg.max_iter_div, cfg.div_eps);
        via_state.density_solve(&neighbors, cfg.max_iter_den, cfg.den_eps);

        let mut velocities = state.velocities.clone();
        let (iters1, iters2) = dfsph_two_stage_solve(
            &state.positions,
            &mut velocities,
            &state.densities,
            &state.alphas,
            &state.masses,
            &neighbors,
            cfg,
        );
        assert_eq!((iters1, iters2), (4, 7));
        let diff = max_relative_difference(&via_state.velocities, &velocities);
        assert!(
            diff < 1.0e-9,
            "relaxation {relaxation}: solvers differ by {diff:e}"
        );
    }
}
