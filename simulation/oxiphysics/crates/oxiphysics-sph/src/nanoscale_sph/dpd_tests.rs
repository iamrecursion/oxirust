//! Regression tests for the dissipative-particle-dynamics solver
//! [`DissipativeParticleNano`]: the sign convention of the conservative and
//! dissipative forces (Groot–Warren), the momentum-conserving random force,
//! the full-step integrators and the fluctuation–dissipation balance.
//!
//! The deterministic tests use reduced units (`m = 1`, `r_c = 1`, `a = 25`,
//! `γ = 4.5`) and no random force; the thermostat test uses SI units.

use super::{DissipativeParticleNano, DpdParameters, K_B};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Reduced-unit parameters: cut-off radius 1, no random force (`σ = 0`).
fn reduced_params(conservative: f64, dissipative: f64) -> DpdParameters {
    // At zero temperature the fluctuation-dissipation amplitude vanishes.
    DpdParameters::new(conservative, dissipative, 1.0, 0.0)
}

/// Two unit-mass particles on the x axis, `separation` apart.
fn pair(
    params: DpdParameters,
    separation: f64,
    v_left: [f64; 3],
    v_right: [f64; 3],
) -> DissipativeParticleNano {
    let mut dpd = DissipativeParticleNano::new(
        vec![[0.0; 3], [separation, 0.0, 0.0]],
        vec![1.0; 2],
        params,
        1.0,
    );
    dpd.velocities = vec![v_left, v_right];
    dpd
}

fn total_kinetic_energy(dpd: &DissipativeParticleNano) -> f64 {
    dpd.velocities
        .iter()
        .zip(dpd.masses.iter())
        .map(|(v, m)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum()
}

/// Deterministic pseudo-random number in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

// ---------------------------------------------------------------------------
// Conservative force
// ---------------------------------------------------------------------------

#[test]
fn conservative_force_repels_with_the_groot_warren_profile() {
    for &r in &[0.1, 0.5, 0.9] {
        let mut dpd = pair(reduced_params(25.0, 4.5), r, [0.0; 3], [0.0; 3]);
        dpd.compute_dpd_forces(0.01);
        let magnitude = 25.0 * (1.0 - r);
        // Particle 0 sits at the origin, particle 1 on +x: repulsion pushes 0
        // towards -x and 1 towards +x.
        assert!((dpd.forces[0][0] + magnitude).abs() < 1e-12, "r = {r}");
        assert!((dpd.forces[1][0] - magnitude).abs() < 1e-12, "r = {r}");
        assert_eq!(dpd.forces[0][1], 0.0);
        assert_eq!(dpd.forces[0][2], 0.0);
    }
}

#[test]
fn pair_at_rest_moves_apart_after_one_full_step() {
    let dt = 0.01;
    let mut dpd = pair(reduced_params(25.0, 4.5), 0.5, [0.0; 3], [0.0; 3]);
    dpd.compute_dpd_forces(dt);
    dpd.integrate(dt);
    // A full kick: v = F dt / m = 12.5 * 0.01 in magnitude (not half of it),
    // then the drift moves each particle by v dt.
    assert!((dpd.velocities[0][0] + 12.5 * dt).abs() < 1e-14);
    assert!((dpd.velocities[1][0] - 12.5 * dt).abs() < 1e-14);
    let separation = dpd.positions[1][0] - dpd.positions[0][0];
    assert!(
        (separation - (0.5 + 25.0 * dt * dt)).abs() < 1e-14,
        "separation {separation}"
    );
    assert!(separation > 0.5);
    // Newton's third law: no net momentum.
    assert!((dpd.velocities[0][0] + dpd.velocities[1][0]).abs() < 1e-14);
}

#[test]
fn forces_vanish_beyond_the_cutoff_and_for_coincident_particles() {
    let mut far = pair(
        reduced_params(25.0, 4.5),
        1.0,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    far.compute_dpd_forces(0.01);
    assert_eq!(far.forces, vec![[0.0; 3]; 2]);
    let mut beyond = pair(
        reduced_params(25.0, 4.5),
        2.5,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    beyond.compute_dpd_forces(0.01);
    assert_eq!(beyond.forces, vec![[0.0; 3]; 2]);
    let mut coincident = pair(
        reduced_params(25.0, 4.5),
        0.0,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    coincident.compute_dpd_forces(0.01);
    assert_eq!(coincident.forces, vec![[0.0; 3]; 2]);
}

// ---------------------------------------------------------------------------
// Dissipative force
// ---------------------------------------------------------------------------

#[test]
fn dissipative_force_decelerates_an_approaching_pair() {
    let dt = 0.01;
    // No conservative force: only -γ w² (e·v) e acts.  w(0.5) = 0.5, so the
    // magnitude is γ w² |e·v_ij| = 4.5 * 0.25 * 2 = 2.25.
    let mut dpd = pair(
        reduced_params(0.0, 4.5),
        0.5,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    let kinetic_before = total_kinetic_energy(&dpd);
    dpd.compute_dpd_forces(dt);
    // The left particle moves towards +x and is slowed by a force along -x.
    assert!((dpd.forces[0][0] + 2.25).abs() < 1e-12, "{:?}", dpd.forces);
    assert!((dpd.forces[1][0] - 2.25).abs() < 1e-12, "{:?}", dpd.forces);
    dpd.integrate(dt);
    assert!((dpd.velocities[0][0] - (1.0 - 2.25 * dt)).abs() < 1e-14);
    assert!((dpd.velocities[1][0] + (1.0 - 2.25 * dt)).abs() < 1e-14);
    let kinetic_after = total_kinetic_energy(&dpd);
    assert!(
        kinetic_after < kinetic_before,
        "relative kinetic energy must fall: {kinetic_before} -> {kinetic_after}"
    );
    assert!((kinetic_after - (1.0 - 2.25 * dt).powi(2)).abs() < 1e-14);
}

#[test]
fn dissipative_force_also_slows_a_separating_pair() {
    let dt = 0.01;
    let mut dpd = pair(
        reduced_params(0.0, 4.5),
        0.5,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    dpd.compute_dpd_forces(dt);
    // The left particle moves towards -x; the force on it points to +x.
    assert!((dpd.forces[0][0] - 2.25).abs() < 1e-12, "{:?}", dpd.forces);
    assert!((dpd.forces[1][0] + 2.25).abs() < 1e-12, "{:?}", dpd.forces);
}

#[test]
fn dissipative_force_ignores_the_velocity_perpendicular_to_the_line_of_centres() {
    let mut dpd = pair(
        reduced_params(0.0, 4.5),
        0.5,
        [0.0, 3.0, 0.0],
        [0.0, -2.0, 1.0],
    );
    dpd.compute_dpd_forces(0.01);
    assert_eq!(dpd.forces, vec![[0.0; 3]; 2]);
}

#[test]
fn repeated_steps_dissipate_the_relative_kinetic_energy_monotonically() {
    let dt = 1.0e-3;
    let mut dpd = pair(
        reduced_params(0.0, 4.5),
        0.5,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    );
    dpd.compute_dpd_forces(dt);
    let mut previous = total_kinetic_energy(&dpd);
    for step in 0..40 {
        dpd.step(dt);
        let kinetic = total_kinetic_energy(&dpd);
        assert!(
            kinetic < previous,
            "step {step}: kinetic energy rose from {previous} to {kinetic}"
        );
        previous = kinetic;
    }
    // The pair never crossed (separation stays positive) and the momentum is
    // still zero.
    assert!(dpd.positions[1][0] - dpd.positions[0][0] > 0.3);
    assert!((dpd.velocities[0][0] + dpd.velocities[1][0]).abs() < 1e-12);
    assert!(previous < 0.97, "kinetic energy only fell to {previous}");
}

// ---------------------------------------------------------------------------
// Random force
// ---------------------------------------------------------------------------

/// A cluster of `n` particles in a cube of side `side` with a prescribed random
/// amplitude (the temperature-based amplitude is negligible in reduced units).
fn random_cluster(n: usize, side: f64, seed: u64) -> DissipativeParticleNano {
    let mut state = seed;
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            [
                side * lcg_unit(&mut state),
                side * lcg_unit(&mut state),
                side * lcg_unit(&mut state),
            ]
        })
        .collect();
    let mut params = reduced_params(25.0, 4.5);
    params.random_amplitude = 3.0;
    let mut dpd = DissipativeParticleNano::new(positions, vec![1.0; n], params, 1.0);
    dpd.velocities = (0..n)
        .map(|_| {
            [
                lcg_unit(&mut state) - 0.5,
                lcg_unit(&mut state) - 0.5,
                lcg_unit(&mut state) - 0.5,
            ]
        })
        .collect();
    dpd
}

#[test]
fn pair_forces_including_the_random_part_conserve_momentum() {
    let mut dpd = random_cluster(30, 1.6, 41);
    let mut rng = StdRng::seed_from_u64(7);
    dpd.compute_dpd_forces_with_rng(0.01, &mut rng);
    let mut total = [0.0_f64; 3];
    let mut scale = 0.0_f64;
    for f in &dpd.forces {
        for k in 0..3 {
            total[k] += f[k];
            scale += f[k].abs();
        }
    }
    assert!(scale > 1.0, "the cluster must interact: {scale}");
    for (k, t) in total.iter().enumerate() {
        assert!(t.abs() <= 1e-12 * scale, "net force component {k}: {t}");
    }
}

#[test]
fn seeded_random_forces_are_reproducible_and_depend_on_the_seed() {
    let compute = |seed: u64| {
        let mut dpd = random_cluster(20, 1.4, 42);
        let mut rng = StdRng::seed_from_u64(seed);
        dpd.compute_dpd_forces_with_rng(0.01, &mut rng);
        dpd.forces
    };
    let a = compute(1);
    let b = compute(1);
    let c = compute(2);
    assert_eq!(a, b);
    assert_ne!(a, c);
    // Without the random amplitude the result no longer depends on the seed.
    let deterministic = |seed: u64| {
        let mut dpd = random_cluster(20, 1.4, 42);
        dpd.dpd_params.random_amplitude = 0.0;
        let mut rng = StdRng::seed_from_u64(seed);
        dpd.compute_dpd_forces_with_rng(0.01, &mut rng);
        dpd.forces
    };
    assert_eq!(deterministic(1), deterministic(2));
}

#[test]
fn random_force_is_left_out_for_a_non_positive_time_step() {
    for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        let mut dpd = pair(reduced_params(25.0, 4.5), 0.5, [0.0; 3], [0.0; 3]);
        dpd.dpd_params.random_amplitude = 3.0;
        let mut rng = StdRng::seed_from_u64(3);
        dpd.compute_dpd_forces_with_rng(dt, &mut rng);
        assert!(
            (dpd.forces[0][0] + 12.5).abs() < 1e-12 && (dpd.forces[1][0] - 12.5).abs() < 1e-12,
            "dt = {dt}: {:?}",
            dpd.forces
        );
    }
}

// ---------------------------------------------------------------------------
// Integrators
// ---------------------------------------------------------------------------

#[test]
fn step_follows_the_exact_repulsive_pair_trajectory_to_second_order() {
    // Relative coordinate r = x1 - x0: r'' = 2 a (1 - r) / m, so
    // 1 - r(t) = (1 - r0) cos(sqrt(2 a / m) t) while r < r_c.
    let (a, r0, dt) = (25.0, 0.5, 1.0e-3);
    let mut dpd = pair(reduced_params(a, 0.0), r0, [0.0; 3], [0.0; 3]);
    dpd.compute_dpd_forces(dt);
    let omega = (2.0 * a).sqrt();
    for n in 1..=100 {
        dpd.step(dt);
        let t = n as f64 * dt;
        let exact = 1.0 - (1.0 - r0) * (omega * t).cos();
        let r = dpd.positions[1][0] - dpd.positions[0][0];
        assert!(
            (r - exact).abs() < 1e-5,
            "t = {t}: r = {r} vs exact {exact}"
        );
    }
}

#[test]
fn step_conserves_the_energy_of_a_conservative_pair() {
    // The potential energy U(r) = a r_c (1 - r/r_c)^2 / 2 of the pair turns
    // into kinetic energy: 3.125 in total once the pair has left the range.
    let (a, r0, dt) = (25.0, 0.5, 1.0e-3);
    let expected = 0.5 * a * (1.0 - r0) * (1.0 - r0);
    let mut dpd = pair(reduced_params(a, 0.0), r0, [0.0; 3], [0.0; 3]);
    dpd.compute_dpd_forces(dt);
    for _ in 0..400 {
        dpd.step(dt);
    }
    assert!(dpd.positions[1][0] - dpd.positions[0][0] > 1.0);
    let kinetic = total_kinetic_energy(&dpd);
    assert!(
        (kinetic - expected).abs() < 1e-4 * expected,
        "kinetic energy {kinetic} vs potential energy {expected}"
    );
    assert!((dpd.velocities[0][0] + dpd.velocities[1][0]).abs() < 1e-12);
}

#[test]
fn step_stores_the_forces_of_the_new_state() {
    let dt = 5.0e-3;
    let mut dpd = pair(
        reduced_params(25.0, 4.5),
        0.6,
        [0.3, 0.0, 0.0],
        [-0.2, 0.1, 0.0],
    );
    dpd.compute_dpd_forces(dt);
    dpd.step(dt);
    let stored = dpd.forces.clone();
    // Forces at the new positions with the half-kicked velocity predictor; the
    // random force is absent, so the evaluation is deterministic.
    let mut reference = dpd.clone();
    let half_back: Vec<[f64; 3]> = dpd
        .velocities
        .iter()
        .zip(dpd.forces.iter())
        .map(|(v, f)| {
            [
                v[0] - 0.5 * dt * f[0],
                v[1] - 0.5 * dt * f[1],
                v[2] - 0.5 * dt * f[2],
            ]
        })
        .collect();
    reference.velocities = half_back;
    reference.compute_dpd_forces(dt);
    for (s, r) in stored.iter().zip(reference.forces.iter()) {
        for k in 0..3 {
            assert!(
                (s[k] - r[k]).abs() < 1e-12,
                "{stored:?} vs {:?}",
                reference.forces
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fluctuation-dissipation balance
// ---------------------------------------------------------------------------

#[test]
fn thermostat_holds_the_relative_velocity_at_the_prescribed_temperature() {
    // Two particles held at a fixed separation r = r_c / 2 (w = 1/2) and
    // coupled only through the dissipative and random forces.  For the relative
    // velocity u = (v_i - v_j) . e the Euler-Maruyama update is
    //   u' = u (1 - kappa dt) + (2 sigma w / m) xi sqrt(dt),  kappa = 2 gamma w^2 / m,
    // whose stationary variance is 2 k_B T / (m (1 - kappa dt / 2)).
    let (mass, gamma, temperature, cutoff) = (1.0e-26, 1.0e-12, 298.15, 1.0e-9);
    let dt = 1.0e-16;
    let separation = 0.5 * cutoff;
    let params = DpdParameters::new(0.0, gamma, cutoff, temperature);
    let positions = vec![[0.0; 3], [separation, 0.0, 0.0]];
    let mut dpd = DissipativeParticleNano::new(positions.clone(), vec![mass; 2], params, cutoff);
    let mut rng = StdRng::seed_from_u64(20_260_626);

    let kappa_dt = 2.0 * gamma * 0.25 / mass * dt;
    assert!((1e-3..1e-2).contains(&kappa_dt), "kappa dt = {kappa_dt}");
    let expected = 2.0 * K_B * temperature / mass / (1.0 - 0.5 * kappa_dt);

    let (burn_in, samples) = (5_000_usize, 2_000_000_usize);
    let mut sum_u2 = 0.0_f64;
    let mut sum_momentum_drift = 0.0_f64;
    for n in 0..(burn_in + samples) {
        dpd.compute_dpd_forces_with_rng(dt, &mut rng);
        dpd.integrate(dt);
        dpd.positions = positions.clone();
        if n >= burn_in {
            // e_ij = (r_i - r_j) / r points from particle 1 to particle 0.
            let u = -(dpd.velocities[0][0] - dpd.velocities[1][0]);
            sum_u2 += u * u;
        }
        sum_momentum_drift = f64::max(
            sum_momentum_drift,
            (dpd.velocities[0][0] + dpd.velocities[1][0]).abs(),
        );
    }
    let measured = sum_u2 / samples as f64;
    assert!(
        (measured - expected).abs() < 0.08 * expected,
        "<u^2> = {measured} vs {expected} (equipartition 2 k_B T / m)"
    );
    // The pair force is internal: the centre-of-mass velocity stays zero.
    assert!(sum_momentum_drift < 1e-6 * expected.sqrt());
}
