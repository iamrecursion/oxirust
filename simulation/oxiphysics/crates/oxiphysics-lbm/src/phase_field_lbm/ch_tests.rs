//! Physics regression tests of the Cahn-Hilliard lattice Boltzmann solver.
//!
//! The solver used to relax the order parameter towards `w_alpha phi` (plain
//! diffusion, the chemical potential was ignored), never applied a force to the
//! momentum distribution (no surface tension, no gravity) and kept a constant
//! relaxation time although the phase viscosities were given.  These tests pin
//! the chemical-potential driven equilibrium (mobility, signs, surface
//! tension), the capillary and gravity coupling and the variable-density,
//! variable-viscosity flow solver against closed-form results.

use super::ch_solver::{flow_equilibrium, flow_forcing, phase_equilibrium};
use super::*;
use std::f64::consts::{PI, SQRT_2};

fn rel_err(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(1e-300)
}

/// Well-resolved two-phase parameters: interface half-width 2.8, `tau_f = 0.8`.
fn base_params() -> PhaseFieldLbmParams {
    PhaseFieldLbmParams {
        epsilon: 4.0,
        mobility: 0.05,
        surface_tension: 0.02,
        tau_f: 0.8,
        tau_phi: 1.0,
        rho_a: 1.0,
        rho_b: 1.0,
        nu_a: 0.1,
        nu_b: 0.1,
        gx: 0.0,
        gy: 0.0,
    }
}

/// Next pseudo-random number in `[-1, 1)` of a 64-bit LCG.
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    2.0 * ((*state >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

/// Zeroth, first and second moments of a distribution on the D2Q9 lattice.
fn moments(dist: &[f64; 9]) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let mut m0 = 0.0;
    let mut m1 = [0.0; 2];
    let mut m2 = [[0.0; 2]; 2];
    for (alpha, &value) in dist.iter().enumerate() {
        let e = [EX[alpha], EY[alpha]];
        m0 += value;
        for (a, &ea) in e.iter().enumerate() {
            m1[a] += ea * value;
            for (b, &eb) in e.iter().enumerate() {
                m2[a][b] += ea * eb * value;
            }
        }
    }
    (m0, m1, m2)
}

fn periodic_distance(x: f64, centre: f64, n: f64) -> f64 {
    let d = (x - centre).abs();
    d.min(n - d)
}

/// Whether `validate` rejects the default parameters after `mutate`.
fn is_rejected(mutate: impl FnOnce(&mut PhaseFieldLbmParams)) -> bool {
    let mut p = base_params();
    mutate(&mut p);
    p.validate().is_err()
}

// ---------------------------------------------------------------------------
// Parameters and equilibria
// ---------------------------------------------------------------------------

#[test]
fn free_energy_coefficients_reproduce_the_surface_tension_and_interface_width() {
    for &(sigma, epsilon) in &[(0.001, 4.0), (0.02, 4.0), (0.05, 2.5), (0.3, 6.0)] {
        let mut p = base_params();
        p.surface_tension = sigma;
        p.epsilon = epsilon;
        let (kappa, beta) = (p.kappa(), p.beta());
        // The planar profile tanh(x/xi) of F = ∫ (β/4)(φ²-1)² + (κ/2)φ'² has
        // xi = sqrt(2 κ / β) ...
        let xi = (2.0 * kappa / beta).sqrt();
        assert!(rel_err(xi, epsilon / SQRT_2) < 1e-14, "xi = {xi}");
        assert!(rel_err(xi, p.interface_half_width()) < 1e-14);
        // ... and the interface energy (2√2/3) sqrt(κ β) per unit length, which
        // must be the surface tension (the old coefficients gave 4 σ).
        let energy = 2.0 * SQRT_2 / 3.0 * (kappa * beta).sqrt();
        assert!(rel_err(energy, sigma) < 1e-14, "energy = {energy}");
        // Direct quadrature of the energy density across the profile.
        let n = 4000;
        let h = 24.0 * xi / n as f64;
        let mut quadrature = 0.0;
        for i in 0..n {
            let x = (i as f64 + 0.5) * h - 12.0 * xi;
            let phi = (x / xi).tanh();
            let phi_x = (1.0 - phi * phi) / xi;
            let bulk = 0.25 * beta * (phi * phi - 1.0).powi(2);
            quadrature += (bulk + 0.5 * kappa * phi_x * phi_x) * h;
        }
        assert!(
            rel_err(quadrature, sigma) < 1e-6,
            "quadrature = {quadrature}"
        );
    }
}

#[test]
fn phase_equilibrium_carries_the_chemical_potential_in_its_second_moment() {
    let (phi, mu, gamma, u) = (0.3, 0.02, 0.4, [0.03, -0.02]);
    let (m0, m1, m2) = moments(&phase_equilibrium(phi, mu, gamma, u));
    assert!((m0 - phi).abs() < 1e-15);
    assert!((m1[0] - phi * u[0]).abs() < 1e-15);
    assert!((m1[1] - phi * u[1]).abs() < 1e-15);
    assert!((m2[0][0] - (gamma * mu + phi * u[0] * u[0])).abs() < 1e-15);
    assert!((m2[1][1] - (gamma * mu + phi * u[1] * u[1])).abs() < 1e-15);
    assert!((m2[0][1] - phi * u[0] * u[1]).abs() < 1e-15);
    // Without the chemical potential the diagonal would lose exactly Γ μ.
    let (_, _, m2_off) = moments(&phase_equilibrium(phi, 0.0, gamma, u));
    assert!((m2[0][0] - m2_off[0][0] - gamma * mu).abs() < 1e-15);
    assert!((m2[1][1] - m2_off[1][1] - gamma * mu).abs() < 1e-15);
    assert!((m2[0][1] - m2_off[0][1]).abs() < 1e-15);
}

#[test]
fn flow_equilibrium_and_forcing_have_the_velocity_based_moments() {
    let (p_star, u, a) = (0.07, [0.04, -0.03], [1.0e-3, -2.0e-3]);
    let (m0, m1, m2) = moments(&flow_equilibrium(p_star, u));
    assert!((m0 - p_star).abs() < 1e-15);
    assert!((m1[0] - u[0]).abs() < 1e-15 && (m1[1] - u[1]).abs() < 1e-15);
    assert!((m2[0][0] - (CS2 * p_star + u[0] * u[0])).abs() < 1e-15);
    assert!((m2[1][1] - (CS2 * p_star + u[1] * u[1])).abs() < 1e-15);
    assert!((m2[0][1] - u[0] * u[1]).abs() < 1e-15);
    // Guo forcing: no mass source, momentum source equal to the acceleration.
    let (f0, f1, _) = moments(&flow_forcing(u, a));
    assert!(f0.abs() < 1e-18, "mass source {f0}");
    assert!((f1[0] - a[0]).abs() < 1e-17 && (f1[1] - a[1]).abs() < 1e-17);
}

#[test]
fn interpolated_density_and_viscosity_are_clamped_to_the_phase_values() {
    let mut p = base_params();
    p.rho_b = 0.1;
    p.nu_b = 0.02;
    assert!(rel_err(p.rho_from_phi(1.0), 1.0) < 1e-15);
    assert!(rel_err(p.rho_from_phi(-1.0), 0.1) < 1e-15);
    assert!(rel_err(p.rho_from_phi(0.0), 0.55) < 1e-15);
    assert!(rel_err(p.nu_from_phi(0.0), 0.06) < 1e-15);
    // An overshoot of the order parameter must not extrapolate the density:
    // without the clamp rho(-1.5) = -0.125 would be negative.
    for &phi in &[-1.0, -1.2, -1.5, -40.0] {
        assert_eq!(p.rho_from_phi(phi), p.rho_b);
        assert_eq!(p.nu_from_phi(phi), p.nu_b);
    }
    for &phi in &[1.0, 1.2, 7.0] {
        assert_eq!(p.rho_from_phi(phi), p.rho_a);
        assert_eq!(p.nu_from_phi(phi), p.nu_a);
    }
}

#[test]
fn local_relaxation_time_follows_the_phase_viscosity() {
    let mut p = base_params();
    p.nu_a = 0.1;
    p.nu_b = 0.04;
    assert!(rel_err(p.tau_f_from_phi(1.0), 0.5 + 3.0 * 0.1) < 1e-14);
    assert!(rel_err(p.tau_f_from_phi(-1.0), 0.5 + 3.0 * 0.04) < 1e-14);
    assert!(rel_err(p.tau_f_from_phi(0.0), 0.5 + 3.0 * 0.07) < 1e-14);
    // The relaxation time of the order-parameter distribution and the
    // mobility give Γ = M / (τ_φ - 1/2).
    p.mobility = 0.3;
    p.tau_phi = 1.1;
    assert!(rel_err(p.gamma(), 0.5) < 1e-14);
}

#[test]
fn parameter_validation_rejects_unphysical_values() {
    assert!(base_params().validate().is_ok());
    assert!(PhaseFieldLbmParams::new_water_air().validate().is_ok());
    assert!(is_rejected(|p| p.epsilon = 0.0), "epsilon = 0");
    assert!(is_rejected(|p| p.epsilon = f64::NAN), "epsilon NaN");
    assert!(is_rejected(|p| p.mobility = -0.1), "negative mobility");
    assert!(is_rejected(|p| p.surface_tension = -1e-3), "negative sigma");
    assert!(is_rejected(|p| p.tau_phi = 0.5), "tau_phi = 1/2");
    assert!(is_rejected(|p| p.tau_phi = 0.3), "tau_phi < 1/2");
    assert!(is_rejected(|p| p.tau_phi = f64::NAN), "tau_phi NaN");
    assert!(is_rejected(|p| p.rho_a = 0.0), "rho_a = 0");
    assert!(is_rejected(|p| p.rho_b = -1.0), "negative rho_b");
    assert!(is_rejected(|p| p.nu_a = 0.0), "nu_a = 0");
    assert!(is_rejected(|p| p.nu_b = -0.01), "negative nu_b");
    assert!(is_rejected(|p| p.gx = f64::INFINITY), "infinite gx");
    assert!(is_rejected(|p| p.gy = f64::NAN), "NaN gy");
    // zero mobility and zero surface tension are legitimate limits
    assert!(!is_rejected(|p| p.mobility = 0.0));
    assert!(!is_rejected(|p| p.surface_tension = 0.0));
}

#[test]
fn invalid_input_is_reported_and_leaves_the_state_untouched() {
    let mut solver = CahnHilliardLbm::new(8, 8, base_params());
    solver.init_flat_interface();
    solver.params.tau_phi = 0.5;
    let before = solver.clone();
    assert!(matches!(
        solver.try_step(),
        Err(Error::UnstableRelaxationTime { .. })
    ));
    solver.step();
    assert_eq!(solver.g, before.g);
    assert_eq!(solver.f, before.f);
    assert_eq!(solver.phi, before.phi);
    assert_eq!(solver.ux, before.ux);
    assert_eq!(solver.pressure, before.pressure);

    // Field arrays that do not match the grid.
    let mut broken = CahnHilliardLbm::new(8, 8, base_params());
    broken.mu.pop();
    assert!(matches!(
        broken.try_step(),
        Err(Error::ComponentCountMismatch { .. })
    ));
    // Empty grid.
    let mut empty = CahnHilliardLbm::new(0, 4, base_params());
    assert!(matches!(
        empty.try_step(),
        Err(Error::InvalidGridDimensions { .. })
    ));
    // A non-finite distribution is detected after the step.
    let mut diverged = CahnHilliardLbm::new(8, 8, base_params());
    diverged.g[5][3] = f64::NAN;
    assert!(matches!(
        diverged.try_step(),
        Err(Error::NonFiniteField { .. })
    ));
}

// ---------------------------------------------------------------------------
// Flat interfaces
// ---------------------------------------------------------------------------

#[test]
fn init_flat_interface_is_periodic_and_centred() {
    let p = base_params();
    let (nx, ny) = (8, 64);
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    solver.init_flat_interface();
    let xi = p.interface_half_width();
    let row = |s: &CahnHilliardLbm, y: usize| s.phi[s.idx(3, y)];
    // tanh profile around ny/2 with the equilibrium width
    for y in 20..44 {
        let expected = ((y as f64 - 32.0) / xi).tanh();
        assert!((row(&solver, y) - expected).abs() < 1e-12, "row {y}");
    }
    // the second interface sits at the periodic boundary, no jump across it
    assert!(row(&solver, 0).abs() < 1e-15);
    let mut largest_step: f64 = 0.0;
    for y in 0..ny {
        let step = (row(&solver, (y + 1) % ny) - row(&solver, y)).abs();
        largest_step = largest_step.max(step);
    }
    assert!(
        largest_step < 1.0 / xi,
        "largest jump between rows is {largest_step}, the tanh slope is {}",
        1.0 / xi
    );
    assert!(row(&solver, 16) < -0.999 && row(&solver, 48) > 0.999);
    // equal phase densities: the density is uniform
    assert!(solver.rho.iter().all(|&r| (r - p.rho_a).abs() < 1e-15));
}

#[test]
fn flat_interface_energy_is_the_surface_tension() {
    for &(sigma, epsilon) in &[(0.02, 4.0), (0.01, 3.0), (0.05, 6.0)] {
        let mut p = base_params();
        p.surface_tension = sigma;
        p.epsilon = epsilon;
        let (nx, ny) = (8, 64);
        let mut solver = CahnHilliardLbm::new(nx, ny, p);
        solver.init_flat_interface();
        // two interfaces of length nx
        let per_length = solver.total_free_energy() / (2.0 * nx as f64);
        assert!(
            rel_err(per_length, sigma) < 0.015,
            "interface energy {per_length} for σ = {sigma}, ε = {epsilon}"
        );
    }
}

#[test]
fn flat_interface_keeps_its_tanh_profile_and_stays_at_rest() {
    let (nx, ny) = (8, 64);
    let mut solver = CahnHilliardLbm::new(nx, ny, base_params());
    solver.init_flat_interface();
    let phi0 = solver.phi.clone();
    let energy0 = solver.total_free_energy();
    let total0: f64 = solver.phi.iter().sum();
    for _ in 0..1000 {
        solver.try_step().expect("stable step");
    }
    let drift = solver
        .phi
        .iter()
        .zip(phi0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(drift < 6e-3, "profile moved by {drift}");
    assert!(solver.max_speed() < 1e-5, "u = {}", solver.max_speed());
    let total: f64 = solver.phi.iter().sum();
    assert!((total - total0).abs() < 1e-11);
    assert!(rel_err(solver.total_free_energy(), energy0) < 1e-3);
    // The interface stays centred at ny/2.
    assert!(solver.phi[solver.idx(0, 32)].abs() < 1e-6);
    // Without the chemical potential in the equilibrium the profile would
    // diffuse with D = c_s² (τ_φ − 1/2) = 1/6 and be smeared over many rows.
    assert!(solver.phi[solver.idx(0, 40)] > 0.98);
}

#[test]
fn order_parameter_total_is_conserved_to_round_off() {
    let (nx, ny) = (16, 16);
    let mut p = base_params();
    p.surface_tension = 0.002;
    p.rho_b = 0.5;
    p.gy = -1e-5;
    let mut solver = CahnHilliardLbm::new(nx, ny, p);
    let mut state = 4242u64;
    for v in solver.phi.iter_mut() {
        *v = 0.2 + 0.3 * next_unit(&mut state);
    }
    solver.init_equilibrium_distributions();
    let total0: f64 = solver.phi.iter().sum();
    for _ in 0..200 {
        solver.try_step().expect("stable step");
    }
    let total: f64 = solver.phi.iter().sum();
    assert!(
        (total - total0).abs() < 1e-11 * total0.abs(),
        "Σφ drifted from {total0} to {total}"
    );
    // the order parameter distribution carries Σ g = φ at every node
    for (g, &phi) in solver.g.iter().zip(solver.phi.iter()) {
        assert!((g.iter().sum::<f64>() - phi).abs() < 1e-14);
    }
}

// ---------------------------------------------------------------------------
// Cahn-Hilliard dynamics
// ---------------------------------------------------------------------------

/// Amplitude of the `cos(k x)` component of the order parameter.
fn cosine_amplitude(solver: &CahnHilliardLbm, k: f64) -> f64 {
    let mut sum = 0.0;
    for y in 0..solver.ny {
        for x in 0..solver.nx {
            sum += solver.phi[solver.idx(x, y)] * (k * x as f64).cos();
        }
    }
    2.0 * sum / (solver.nx * solver.ny) as f64
}

/// Measured and linear-theory growth rate of a `cos(kx)` perturbation of
/// the homogeneous state `phi0` (64 x 4 grid, mode number 4, 300 steps).
fn mode_rates(phi0: f64, mobility: f64) -> (f64, f64) {
    let (nx, ny) = (64, 4);
    let k = 2.0 * PI * 4.0 / nx as f64;
    let k2 = 2.0 * (1.0 - k.cos()); // lattice wavenumber squared
    let mut p = base_params();
    p.epsilon = 2.0;
    p.surface_tension = 0.01;
    p.mobility = mobility;
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    for y in 0..ny {
        for x in 0..nx {
            let i = solver.idx(x, y);
            solver.phi[i] = phi0 + 1e-3 * (k * x as f64).cos();
        }
    }
    solver.init_equilibrium_distributions();
    let a0 = cosine_amplitude(&solver, k);
    let steps = 300;
    for _ in 0..steps {
        solver.try_step().expect("stable step");
    }
    let measured = (cosine_amplitude(&solver, k) / a0).ln() / steps as f64;
    // linearised Cahn-Hilliard: dφ/dt = -M k² (ψ''(φ0) + κ k²) δφ
    let psi2 = p.beta() * (3.0 * phi0 * phi0 - 1.0);
    let predicted = -mobility * k2 * (psi2 + p.kappa() * k2);
    (measured, predicted)
}

#[test]
fn small_modes_grow_and_decay_with_the_cahn_hilliard_rate() {
    for &phi0 in &[0.0, 1.0] {
        let (slow, slow_theory) = mode_rates(phi0, 0.05);
        let (fast, fast_theory) = mode_rates(phi0, 0.1);
        for &(measured, predicted) in &[(slow, slow_theory), (fast, fast_theory)] {
            assert!(
                rel_err(measured, predicted) < 5e-3,
                "φ0 = {phi0}: rate {measured:e} vs {predicted:e}"
            );
            // the homogeneous mixture φ0 = 0 is unstable, the bulk phases stable
            assert_eq!(measured > 0.0, phi0 == 0.0);
        }
        // the rate is proportional to the mobility
        assert!(
            (fast / slow - 2.0).abs() < 4e-3,
            "φ0 = {phi0}: rate ratio {}",
            fast / slow
        );
    }
}

#[test]
fn order_parameter_is_advected_with_the_flow_velocity() {
    let mut p = base_params();
    p.epsilon = 2.0;
    p.surface_tension = 0.01;
    let (nx, ny) = (32, 4);
    let u0 = 0.05;
    let k = 2.0 * PI / nx as f64;
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    for y in 0..ny {
        for x in 0..nx {
            let i = solver.idx(x, y);
            solver.phi[i] = 1.0 + 1e-3 * (k * x as f64).cos();
            solver.ux[i] = u0;
        }
    }
    solver.init_equilibrium_distributions();
    let phase_and_amplitude = |s: &CahnHilliardLbm| -> (f64, f64) {
        let (mut c, mut sn) = (0.0, 0.0);
        for y in 0..s.ny {
            for x in 0..s.nx {
                let v = s.phi[s.idx(x, y)] - 1.0;
                c += v * (k * x as f64).cos();
                sn += v * (k * x as f64).sin();
            }
        }
        let n = (s.nx * s.ny) as f64;
        (sn.atan2(c), 2.0 * (c * c + sn * sn).sqrt() / n)
    };
    let (phase0, amplitude0) = phase_and_amplitude(&solver);
    let steps = 200;
    for _ in 0..steps {
        solver.try_step().expect("stable step");
    }
    let (phase1, amplitude1) = phase_and_amplitude(&solver);
    // The mode moves with u0 (the lattice dispersion error is below 1 %) ...
    let shift = phase1 - phase0;
    let expected_shift = k * u0 * steps as f64;
    assert!(
        rel_err(shift, expected_shift) < 0.015,
        "phase shift {shift} vs {expected_shift}"
    );
    // ... and decays only with the Cahn-Hilliard rate M k² (2β + κ k²).
    let k2 = 2.0 * (1.0 - k.cos());
    let decay = p.mobility * k2 * (2.0 * p.beta() + p.kappa() * k2);
    let expected_ratio = (-decay * steps as f64).exp();
    assert!(
        rel_err(amplitude1 / amplitude0, expected_ratio) < 3e-3,
        "amplitude ratio {} vs {expected_ratio}",
        amplitude1 / amplitude0
    );
    // uniform flow: the velocity is untouched by the order-parameter dynamics
    assert!(solver.ux.iter().all(|&u| (u - u0).abs() < 1e-8));
}

#[test]
fn spinodal_decomposition_separates_the_phases() {
    let n = 64;
    let mut p = base_params();
    p.epsilon = 2.0;
    p.surface_tension = 0.01;
    p.mobility = 3.0;
    p.tau_phi = 2.0;
    p.rho_b = 0.5;
    let mut sim = SpinodaDecompositionLbm::new(n, n, 0.0, 0.01, p);
    // deterministic noise of amplitude 0.01 around φ0 = 0
    let mut state = 12345u64;
    for v in sim.solver.phi.iter_mut() {
        *v = 0.01 * next_unit(&mut state);
    }
    sim.solver.init_equilibrium_distributions();
    let energy0 = sim.solver.total_free_energy();
    let total0: f64 = sim.solver.phi.iter().sum();
    assert!(sim.separation_order() < 0.01);

    sim.run(200);
    assert_eq!(sim.time_step, 200);
    assert!(
        sim.separation_order() < 0.01,
        "linear stage: the noise has not grown yet, order {}",
        sim.separation_order()
    );
    // The free energy never increases while the phases separate.
    let mut previous = sim.solver.total_free_energy();
    for _ in 0..180 {
        sim.run(10);
        let energy = sim.solver.total_free_energy();
        assert!(
            energy <= previous + 1e-9 * energy0,
            "free energy rose from {previous} to {energy} at step {}",
            sim.time_step
        );
        previous = energy;
    }
    assert_eq!(sim.time_step, 2000);
    assert!(
        sim.separation_order() > 0.5,
        "phases did not separate: order {}",
        sim.separation_order()
    );
    assert!(sim.early_stage_ended());
    assert!(previous < 0.8 * energy0, "energy {previous} vs {energy0}");
    let max_phi = sim.solver.phi.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(max_phi < 1.2, "order parameter overshoots to {max_phi}");
    let total: f64 = sim.solver.phi.iter().sum();
    assert!(
        (total - total0).abs() < 1e-9,
        "Σφ drifted by {}",
        total - total0
    );
    let (px, py) = sim.solver.total_momentum();
    assert!(px.abs() < 1e-9 && py.abs() < 1e-9, "momentum ({px}, {py})");
}

// ---------------------------------------------------------------------------
// Surface tension: Laplace pressure of a static droplet
// ---------------------------------------------------------------------------

/// Relax a droplet of radius 16 in a 64 x 64 box and return the measured
/// `Δp R / σ` (R from the area of phase A) and the final largest speed.
fn laplace_ratio(rho_b: f64) -> (f64, f64) {
    let mut p = base_params();
    p.rho_b = rho_b;
    let (nx, ny) = (64usize, 64usize);
    let (cx, cy) = (32.0, 32.0);
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    solver.init_droplet(cx, cy, 16.0);
    for _ in 0..5000 {
        solver.try_step().expect("stable step");
    }
    let (mut inside, mut n_in, mut outside, mut n_out, mut area) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for y in 0..ny {
        for x in 0..nx {
            let i = solver.idx(x, y);
            let dx = periodic_distance(x as f64, cx, nx as f64);
            let dy = periodic_distance(y as f64, cy, ny as f64);
            let r = (dx * dx + dy * dy).sqrt();
            if r < 8.0 {
                inside += solver.pressure[i];
                n_in += 1.0;
            }
            if r > 26.0 {
                outside += solver.pressure[i];
                n_out += 1.0;
            }
            area += 0.5 * (1.0 + solver.phi[i]);
        }
    }
    let radius = (area / PI).sqrt();
    let jump = inside / n_in - outside / n_out;
    (jump * radius / p.surface_tension, solver.max_speed())
}

#[test]
fn laplace_pressure_of_a_static_droplet_with_equal_densities() {
    let (ratio, speed) = laplace_ratio(1.0);
    assert!(
        (ratio - 1.0).abs() < 0.04,
        "Δp R / σ = {ratio}, the Laplace law gives 1"
    );
    assert!(speed < 1e-4, "spurious currents {speed}");
}

#[test]
fn laplace_pressure_does_not_depend_on_the_density_ratio() {
    let (ratio, speed) = laplace_ratio(0.1);
    assert!(
        (ratio - 1.0).abs() < 0.04,
        "Δp R / σ = {ratio} at density ratio 10, the Laplace law gives 1"
    );
    assert!(speed < 1e-4, "spurious currents {speed}");
}

// ---------------------------------------------------------------------------
// Flow solver: Galilean invariance, viscosity, gravity, conservation
// ---------------------------------------------------------------------------

#[test]
fn droplet_translates_with_a_uniform_flow_at_density_ratio_ten() {
    let mut p = base_params();
    p.rho_b = 0.1;
    let (nx, ny) = (64usize, 64usize);
    let u0 = 0.04;
    let mut solver = CahnHilliardLbm::new(nx, ny, p);
    solver.init_droplet(18.0, 32.0, 12.0);
    for v in solver.ux.iter_mut() {
        *v = u0;
    }
    solver.init_equilibrium_distributions();
    let centroid = |s: &CahnHilliardLbm| -> (f64, f64, f64) {
        let (mut sx, mut sy, mut count) = (0.0, 0.0, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                if s.phi[s.idx(x, y)] > 0.0 {
                    sx += x as f64;
                    sy += y as f64;
                    count += 1.0;
                }
            }
        }
        (sx / count, sy / count, count)
    };
    let (x0, y0, area0) = centroid(&solver);
    let (px0, _) = solver.total_momentum();
    let mass0 = solver.total_mass();
    let steps = 400;
    for _ in 0..steps {
        solver.try_step().expect("stable step");
    }
    let (x1, y1, area1) = centroid(&solver);
    let expected = u0 * steps as f64;
    // the lattice advection of the sharp interface lags by a few percent only
    assert!(
        (x1 - x0) > 0.95 * expected && (x1 - x0) < 1.01 * expected,
        "droplet moved {} instead of {expected}",
        x1 - x0
    );
    assert!((y1 - y0).abs() < 1e-6, "no transverse drift: {}", y1 - y0);
    assert!(
        (area1 - area0).abs() < 0.06 * area0,
        "area {area0} -> {area1}"
    );
    assert!((solver.total_mass() - mass0).abs() < 1e-9);
    // the velocity stays uniform up to the small interface error
    let worst = solver
        .ux
        .iter()
        .map(|&u| (u - u0).abs())
        .fold(0.0, f64::max);
    assert!(worst < 0.12 * u0, "velocity deviates by {worst}");
    // total momentum is exactly conserved
    let (px, py) = solver.total_momentum();
    assert!((px - px0).abs() < 1e-10 * px0.abs(), "Px {px0} -> {px}");
    assert!(py.abs() < 1e-10 * px0.abs(), "Py = {py}");
}

#[test]
fn shear_wave_decays_with_the_viscosity_of_each_phase() {
    // (order parameter, rho_b, nu_b): pure phase A, pure phase B with a density
    // and a viscosity different from A, and pure phase B with equal properties
    for &(phi_value, rho_b, nu_b) in &[(1.0, 0.25, 0.04), (-1.0, 0.25, 0.04), (-1.0, 1.0, 0.1)] {
        let mut p = base_params();
        p.rho_b = rho_b;
        p.nu_b = nu_b;
        let (nx, ny) = (4usize, 32usize);
        let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
        for v in solver.phi.iter_mut() {
            *v = phi_value;
        }
        let k = 2.0 * PI / ny as f64;
        for y in 0..ny {
            for x in 0..nx {
                let i = solver.idx(x, y);
                solver.ux[i] = 1e-3 * (k * y as f64).sin();
            }
        }
        solver.init_equilibrium_distributions();
        let amplitude = |s: &CahnHilliardLbm| -> f64 {
            let mut sum = 0.0;
            for y in 0..ny {
                for x in 0..nx {
                    sum += s.ux[s.idx(x, y)] * (k * y as f64).sin();
                }
            }
            2.0 * sum / (nx * ny) as f64
        };
        let a0 = amplitude(&solver);
        let steps = 300;
        for _ in 0..steps {
            solver.try_step().expect("stable step");
        }
        let measured = -(amplitude(&solver) / a0).ln() / steps as f64;
        let predicted = p.nu_from_phi(phi_value) * k * k;
        assert!(
            rel_err(measured, predicted) < 0.02,
            "φ = {phi_value}, ν = {}: decay {measured:e} vs ν k² = {predicted:e}",
            p.nu_from_phi(phi_value)
        );
    }
}

#[test]
fn hydrostatic_pressure_balances_gravity() {
    let mut p = base_params();
    p.rho_b = 0.25;
    p.surface_tension = 0.005;
    p.gy = -2e-5;
    let (nx, ny) = (8usize, 64usize);
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    solver.init_flat_interface();
    let rho_mean = solver.rho.iter().sum::<f64>() / solver.rho.len() as f64;
    // hydrostatic pressure profile dp/dy = (ρ - <ρ>) g integrated with the
    // trapezoidal rule over the initial density profile
    let rho_row: Vec<f64> = (0..ny).map(|y| solver.rho[solver.idx(0, y)]).collect();
    let mut expected = vec![0.0; ny];
    for y in 1..ny {
        let mid = 0.5 * (rho_row[y - 1] + rho_row[y]) - rho_mean;
        expected[y] = expected[y - 1] + mid * p.gy;
    }
    let mean_expected = expected.iter().sum::<f64>() / ny as f64;
    let range = expected.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - expected.iter().cloned().fold(f64::INFINITY, f64::min);
    for _ in 0..10_000 {
        solver.try_step().expect("stable step");
    }
    let p_row: Vec<f64> = (0..ny).map(|y| solver.pressure[solver.idx(0, y)]).collect();
    let mean_p = p_row.iter().sum::<f64>() / ny as f64;
    // the fluid rests ...
    assert!(solver.max_speed() < 1e-7, "u = {}", solver.max_speed());
    // ... the pressure gradient in the middle of each layer is (ρ - <ρ>) g ...
    for &(y, rho) in &[(48usize, p.rho_a), (16usize, p.rho_b)] {
        let slope = 0.5 * (p_row[y + 1] - p_row[y - 1]);
        let wanted = (rho - rho_mean) * p.gy;
        assert!(
            rel_err(slope, wanted) < 0.01,
            "pressure gradient {slope:e} at row {y}, hydrostatic value {wanted:e}"
        );
    }
    // ... and the whole profile follows the hydrostatic integral
    let worst = (0..ny)
        .map(|y| ((p_row[y] - mean_p) - (expected[y] - mean_expected)).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 0.06 * range,
        "profile off by {worst:e} of {range:e}"
    );
    // gravity alone does not move the centre of mass
    let (px, py) = solver.total_momentum();
    assert!(
        px.abs() < 1e-12 && py.abs() < 1e-12,
        "momentum ({px}, {py})"
    );
}

#[test]
fn layered_flow_has_the_variable_viscosity_profile_of_the_continuum_equation() {
    // Two fluid layers (ρ_A = 1, ν_A = 0.1; ρ_B = 0.25, ν_B = 0.05) sheared by
    // gravity along the layers.  The force density (ρ - <ρ>) g has a vanishing
    // mean, so a steady flow develops that obeys
    //   d/dy( ρ ν du/dy ) = -(ρ - <ρ>) g
    // with the dynamic viscosity ρ ν: the viscous stress is continuous across
    // the interface although the kinematic viscosities and densities jump.
    let mut p = base_params();
    p.rho_b = 0.25;
    p.nu_b = 0.05;
    p.surface_tension = 0.01;
    p.gx = 2e-6;
    let (nx, ny) = (4usize, 64usize);
    let mut solver = CahnHilliardLbm::new(nx, ny, p.clone());
    solver.init_flat_interface();
    let rho_row: Vec<f64> = (0..ny).map(|y| solver.rho[solver.idx(0, y)]).collect();
    let mu_row: Vec<f64> = (0..ny)
        .map(|y| rho_row[y] * p.nu_from_phi(solver.phi[solver.idx(0, y)]))
        .collect();
    let rho_mean = rho_row.iter().sum::<f64>() / ny as f64;
    // stress τ(y+1/2) = C - ∫ F dy and u(y+1) - u(y) = τ / μ_d at the half node
    let mut stress_part = vec![0.0; ny];
    let mut running = 0.0;
    for y in 0..ny {
        running += (rho_row[y] - rho_mean) * p.gx;
        stress_part[y] = -running;
    }
    let mu_half: Vec<f64> = (0..ny)
        .map(|y| 2.0 / (1.0 / mu_row[y] + 1.0 / mu_row[(y + 1) % ny]))
        .collect();
    let numerator: f64 = (0..ny).map(|y| stress_part[y] / mu_half[y]).sum();
    let denominator: f64 = (0..ny).map(|y| 1.0 / mu_half[y]).sum();
    let constant = -numerator / denominator; // periodic velocity
    let mut predicted = vec![0.0; ny];
    for y in 1..ny {
        predicted[y] = predicted[y - 1] + (constant + stress_part[y - 1]) / mu_half[y - 1];
    }
    let momentum: f64 = (0..ny).map(|y| rho_row[y] * predicted[y]).sum::<f64>();
    let mass: f64 = rho_row.iter().sum();
    for v in predicted.iter_mut() {
        *v -= momentum / mass; // zero total momentum, like the simulation
    }
    let scale = predicted.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

    for _ in 0..15_000 {
        solver.try_step().expect("stable step");
    }
    let measured: Vec<f64> = (0..ny)
        .map(|y| (0..nx).map(|x| solver.ux[solver.idx(x, y)]).sum::<f64>() / nx as f64)
        .collect();
    let worst = (0..ny)
        .map(|y| (measured[y] - predicted[y]).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 0.015 * scale,
        "velocity profile off by {worst:e} of {scale:e}"
    );
    // the flow is parallel to the layers and the momentum is conserved
    let cross_flow = solver.uy.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(cross_flow < 1e-4 * scale, "cross flow {cross_flow:e}");
    let (px, py) = solver.total_momentum();
    assert!(
        px.abs() < 1e-11 && py.abs() < 1e-11,
        "momentum ({px}, {py})"
    );
}

#[test]
fn uniform_gravity_does_not_move_a_homogeneous_fluid() {
    for &phi_value in &[1.0, -1.0] {
        let mut p = base_params();
        p.rho_b = 0.3;
        p.gx = 1e-3;
        p.gy = -2e-3;
        let mut solver = CahnHilliardLbm::new(8, 8, p);
        for v in solver.phi.iter_mut() {
            *v = phi_value;
        }
        solver.init_equilibrium_distributions();
        for _ in 0..50 {
            solver.try_step().expect("stable step");
        }
        assert!(solver.max_speed() < 1e-14, "u = {}", solver.max_speed());
        assert!(solver.force_x.iter().all(|&f| f.abs() < 1e-15));
        assert!(solver.acc_y.iter().all(|&a| a.abs() < 1e-15));
    }
}

#[test]
fn external_force_has_no_net_component_in_the_periodic_box() {
    let mut p = base_params();
    p.rho_b = 0.1;
    p.gx = 3e-5;
    p.gy = -1e-5;
    let mut solver = CahnHilliardLbm::new(32, 32, p);
    solver.init_droplet(16.0, 16.0, 7.0);
    let sum_x: f64 = solver.force_x.iter().sum();
    let sum_y: f64 = solver.force_y.iter().sum();
    assert!(
        sum_x.abs() < 1e-13 && sum_y.abs() < 1e-13,
        "ΣF = ({sum_x}, {sum_y})"
    );
    // but gravity acts on the density contrast: the droplet region is pushed
    let centre = solver.idx(16, 16);
    assert!(solver.force_x[centre] > 0.0 && solver.force_y[centre] < 0.0);
}

#[test]
fn falling_water_droplet_in_air_is_stable_and_conserves_momentum() {
    // The default water-air parameters: density ratio 10, viscosity ratio 10
    // (τ_f = 0.55 in the air) and gravity along -y.
    let p = PhaseFieldLbmParams::new_water_air();
    let (nx, ny) = (64usize, 64usize);
    let mut solver = CahnHilliardLbm::new(nx, ny, p);
    solver.init_droplet(32.0, 40.0, 12.0);
    let centroid = |s: &CahnHilliardLbm| -> (f64, f64) {
        let (mut sx, mut sy, mut count) = (0.0, 0.0, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                if s.phi[s.idx(x, y)] > 0.0 {
                    sx += x as f64;
                    sy += y as f64;
                    count += 1.0;
                }
            }
        }
        (sx / count, sy / count)
    };
    let (x0, y0) = centroid(&solver);
    let mass0 = solver.total_mass();
    for _ in 0..2500 {
        solver
            .try_step()
            .expect("the water-air parameters are stable");
    }
    let (x1, y1) = centroid(&solver);
    assert!(y0 - y1 > 5.0, "the dense droplet must fall: {y0} -> {y1}");
    assert!((x1 - x0).abs() < 0.05, "no sideways drift: {}", x1 - x0);
    let speed = solver.max_speed();
    assert!(speed > 0.01 && speed < 0.05, "speed {speed}");
    assert!((solver.total_mass() - mass0).abs() < 1e-9);
    // the air moves up as the water falls: zero total momentum
    let (px, py) = solver.total_momentum();
    assert!(px.abs() < 1e-9 && py.abs() < 1e-9, "momentum ({px}, {py})");
    assert!(solver.uy.iter().cloned().fold(0.0, f64::max) > 0.0);
}

// ---------------------------------------------------------------------------
// Capillary waves
// ---------------------------------------------------------------------------

/// Angular frequency of a standing capillary wave of wavelength `n` on the
/// interfaces of an `n x n` periodic box, from its first two zero crossings,
/// together with the sharp-interface frequency `sqrt(σ k³ / (ρ_A + ρ_B))`.
fn capillary_wave_frequency(n: usize, rho_b: f64, sigma: f64, steps: usize) -> (f64, f64) {
    let mut p = base_params();
    p.rho_b = rho_b;
    p.surface_tension = sigma;
    p.nu_a = 0.05;
    p.nu_b = 0.05;
    let mut solver = CahnHilliardLbm::new(n, n, p.clone());
    let xi = p.interface_half_width();
    let k = 2.0 * PI / n as f64;
    let ny_f = n as f64;
    // both interfaces (at ny/2 and at the periodic boundary) shifted by cos(kx)
    for y in 0..n {
        for x in 0..n {
            let shifted = (y as f64 - (k * x as f64).cos()).rem_euclid(ny_f);
            let distance = (shifted - ny_f / 2.0)
                .abs()
                .min(shifted)
                .min(ny_f - shifted);
            let signed = if shifted >= ny_f / 2.0 {
                distance
            } else {
                -distance
            };
            let i = solver.idx(x, y);
            solver.phi[i] = (signed / xi).tanh();
        }
    }
    solver.init_equilibrium_distributions();
    // displacement of the interface at ny/2: y_a + Σ (1 - φ)/2 over a window
    let displacement = |s: &CahnHilliardLbm| -> f64 {
        let (lo, hi) = (n / 4, 3 * n / 4);
        let heights: Vec<f64> = (0..n)
            .map(|x| {
                lo as f64
                    + (lo..hi)
                        .map(|y| 0.5 * (1.0 - s.phi[s.idx(x, y)]))
                        .sum::<f64>()
            })
            .collect();
        let mean = heights.iter().sum::<f64>() / n as f64;
        let projection: f64 = heights
            .iter()
            .enumerate()
            .map(|(x, h)| (h - mean) * (k * x as f64).cos())
            .sum();
        2.0 * projection / n as f64
    };
    let mut previous = displacement(&solver);
    let mut crossings = Vec::new();
    for step in 1..=steps {
        solver.try_step().expect("stable step");
        let current = displacement(&solver);
        if previous * current < 0.0 {
            let fraction = previous / (previous - current);
            crossings.push(step as f64 - 1.0 + fraction);
        }
        previous = current;
    }
    assert!(
        crossings.len() >= 2,
        "only {} zero crossings",
        crossings.len()
    );
    let omega = PI / (crossings[1] - crossings[0]);
    let omega0 = (sigma * k * k * k / (p.rho_a + p.rho_b)).sqrt();
    (omega, omega0)
}

#[test]
fn capillary_wave_frequency_follows_the_surface_tension_and_the_inertia() {
    let (omega_equal, omega0_equal) = capillary_wave_frequency(64, 1.0, 0.05, 1500);
    let (omega_light, omega0_light) = capillary_wave_frequency(64, 0.25, 0.05, 1500);
    let (omega_strong, omega0_strong) = capillary_wave_frequency(64, 1.0, 0.2, 800);
    // The frequency is that of the sharp-interface standing wave
    // ω² = σ k³ / (ρ_A + ρ_B) up to the diffuse-interface and viscous
    // reduction of the restoring force.
    for &(omega, omega0) in &[
        (omega_equal, omega0_equal),
        (omega_light, omega0_light),
        (omega_strong, omega0_strong),
    ] {
        let ratio = omega / omega0;
        assert!(ratio > 0.8 && ratio < 1.1, "ω/ω₀ = {ratio}");
    }
    // The inertia is the sum of the two densities: ω² ∝ 1/(ρ_A + ρ_B).
    let expected = (2.0_f64 / 1.25).sqrt();
    assert!(
        rel_err(omega_light / omega_equal, expected) < 0.015,
        "light/equal frequency ratio {} vs {expected}",
        omega_light / omega_equal
    );
    // The restoring force is the surface tension: ω ∝ sqrt(σ) (the viscous
    // damping of the stronger wave is relatively smaller, hence the margin).
    assert!(
        rel_err(omega_strong / omega_equal, 2.0) < 0.05,
        "σ-scaling {} vs 2",
        omega_strong / omega_equal
    );
}
