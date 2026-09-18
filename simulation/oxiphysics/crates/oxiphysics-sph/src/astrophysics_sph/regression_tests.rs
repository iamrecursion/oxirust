// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

// Regression tests of `AdiabticSph`: the artificial viscosity used to heat the
// entropy (`compute_entropy_derivatives`) without ever entering the momentum
// equation (`compute_accelerations`), so shocks generated heat without
// decelerating the gas and the total energy grew; the thermal energy of
// `total_energy` was the constructor's `internal_energy`, which `step` never
// updated from the evolved entropy. They live in their own file so that
// `astrophysics_sph.rs` stays below the 2000-line limit.

use super::*;

const GAMMA: f64 = 5.0 / 3.0;

/// Deterministic pseudo-random numbers in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Two slabs of `nx × 3 × 3` particles (spacing `spacing`, unit mass density,
/// specific internal energy `u0`) that approach each other along x with
/// `±speed`; the slabs touch like neighbouring lattice planes. Gravity is off,
/// the entropy is initialised at the SPH densities.
fn colliding_slabs(
    spacing: f64,
    nx: usize,
    speed: f64,
    u0: f64,
    alpha: f64,
    beta: f64,
) -> AdiabticSph {
    let mass = spacing.powi(3);
    // The kernel has the compact support h, 2.2 spacings ≈ 44 neighbours.
    let h = 2.2 * spacing;
    let mut particles = Vec::new();
    for slab in 0..2 {
        let x0 = if slab == 0 { 0.0 } else { nx as f64 * spacing };
        let vx = if slab == 0 { speed } else { -speed };
        for i in 0..nx {
            for j in 0..3 {
                for k in 0..3 {
                    particles.push(AstroSphParticle::new(
                        particles.len(),
                        mass,
                        [
                            x0 + i as f64 * spacing,
                            j as f64 * spacing,
                            k as f64 * spacing,
                        ],
                        [vx, 0.0, 0.0],
                        u0,
                        h,
                        1.0,
                        GAMMA,
                    ));
                }
            }
        }
    }
    let mut sim = AdiabticSph::new(GAMMA, alpha, beta, 0.01, particles);
    sim.gravity.g = 0.0;
    sim.initialise_entropy();
    sim
}

/// `n` particles with random positions in the unit cube, random velocities,
/// masses and specific internal energies, one common smoothing length.
fn random_cloud(n: usize, h: f64, seed: u64) -> AdiabticSph {
    let mut state = seed;
    let mut particles = Vec::new();
    for id in 0..n {
        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for d in 0..3 {
            pos[d] = lcg_unit(&mut state);
            vel[d] = 2.0 * lcg_unit(&mut state) - 1.0;
        }
        let mass = (0.5 + lcg_unit(&mut state)) / n as f64;
        let u = 0.5 + lcg_unit(&mut state);
        particles.push(AstroSphParticle::new(id, mass, pos, vel, u, h, 1.0, GAMMA));
    }
    let mut sim = AdiabticSph::new(GAMMA, 1.0, 2.0, 0.01, particles);
    sim.gravity.g = 0.0;
    sim.initialise_entropy();
    sim
}

fn kinetic(sim: &AdiabticSph) -> f64 {
    GravitationalForce::total_kinetic_energy(&sim.particles)
}

fn thermal(sim: &AdiabticSph) -> f64 {
    sim.particles
        .iter()
        .map(|p| p.mass * p.thermal_energy_from_entropy(GAMMA))
        .sum()
}

fn total_entropy(sim: &AdiabticSph) -> f64 {
    sim.particles.iter().map(|p| p.mass * p.entropy).sum()
}

fn total_momentum(sim: &AdiabticSph) -> [f64; 3] {
    let mut momentum = [0.0; 3];
    for p in &sim.particles {
        for (component, &velocity) in momentum.iter_mut().zip(p.vel.iter()) {
            *component += p.mass * velocity;
        }
    }
    momentum
}

// ---------------------------------------------------------------------------
// Viscosity in the momentum equation
// ---------------------------------------------------------------------------

/// Two approaching particles: the artificial viscosity changes the
/// accelerations by exactly `−m_j Π_ij ∇_iW_ij` (and the opposite for j), which
/// decelerates the approach and conserves momentum.
#[test]
fn artificial_viscosity_enters_the_momentum_equation() {
    let particles = vec![
        AstroSphParticle::new(
            0,
            1.0,
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            GAMMA,
        ),
        AstroSphParticle::new(
            1,
            2.0,
            [0.4, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            GAMMA,
        ),
    ];
    let mut sim = AdiabticSph::new(GAMMA, 1.0, 2.0, 0.01, particles);
    sim.gravity.g = 0.0;
    sim.initialise_entropy();

    sim.compute_accelerations();
    let with_viscosity: Vec<[f64; 3]> = sim.particles.iter().map(|p| p.accel).collect();

    // Same state with the viscosity switched off.
    let (alpha, beta) = (sim.av.alpha, sim.av.beta);
    sim.av.alpha = 0.0;
    sim.av.beta = 0.0;
    sim.compute_accelerations();
    let without_viscosity: Vec<[f64; 3]> = sim.particles.iter().map(|p| p.accel).collect();
    sim.av.alpha = alpha;
    sim.av.beta = beta;

    let pi_ij = sim.pair_viscosity(0, 1);
    assert!(pi_ij > 0.0, "approaching pair must be viscous: {pi_ij}");
    let grad_w = sim.pair_kernel_gradient(0, 1);
    assert!(grad_w[0] != 0.0, "pair is outside the kernel support");
    for d in 0..3 {
        let extra_i = with_viscosity[0][d] - without_viscosity[0][d];
        let extra_j = with_viscosity[1][d] - without_viscosity[1][d];
        let expected_i = -sim.particles[1].mass * pi_ij * grad_w[d];
        let expected_j = sim.particles[0].mass * pi_ij * grad_w[d];
        assert!(
            (extra_i - expected_i).abs() <= 1.0e-12 * expected_i.abs().max(1.0e-300),
            "axis {d}: viscous acceleration of i {extra_i}, expected {expected_i}"
        );
        assert!(
            (extra_j - expected_j).abs() <= 1.0e-12 * expected_j.abs().max(1.0e-300),
            "axis {d}: viscous acceleration of j {extra_j}, expected {expected_j}"
        );
    }
    // Deceleration of the approach: i moves to +x and is pushed back to −x.
    let extra_i_x = with_viscosity[0][0] - without_viscosity[0][0];
    let extra_j_x = with_viscosity[1][0] - without_viscosity[1][0];
    assert!(
        extra_i_x < 0.0 && extra_j_x > 0.0,
        "{extra_i_x}, {extra_j_x}"
    );
    // Momentum: m_i a_i + m_j a_j = 0 with and without viscosity.
    for accel in [&with_viscosity, &without_viscosity] {
        let net = sim.particles[0].mass * accel[0][0] + sim.particles[1].mass * accel[1][0];
        assert!(net.abs() <= 1.0e-12 * (sim.particles[0].mass * accel[0][0]).abs());
    }
}

/// Receding particles are not viscous: no extra force and no heating.
#[test]
fn receding_pair_feels_no_viscosity_and_makes_no_heat() {
    let particles = vec![
        AstroSphParticle::new(
            0,
            1.0,
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            GAMMA,
        ),
        AstroSphParticle::new(
            1,
            1.0,
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            GAMMA,
        ),
    ];
    let mut sim = AdiabticSph::new(GAMMA, 1.0, 2.0, 0.01, particles);
    sim.gravity.g = 0.0;
    sim.initialise_entropy();
    assert_eq!(sim.pair_viscosity(0, 1), 0.0);
    sim.compute_entropy_derivatives();
    assert!(sim.particles.iter().all(|p| p.du_dt == 0.0));
}

// ---------------------------------------------------------------------------
// Energy balance
// ---------------------------------------------------------------------------

/// The rate of change of the total energy vanishes identically: the kinetic
/// energy removed by `a_i = −Σ m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇_iW_ij`
/// equals the thermal energy gained through the pressure work
/// `P_i/ρ_i² dρ_i/dt` and the entropy source `dA_i/dt`, pair term for pair
/// term (random cloud with many approaching pairs).
#[test]
fn viscous_heating_balances_the_work_of_the_viscous_force_instantaneously() {
    for seed in [1_u64, 2, 3] {
        let mut sim = random_cloud(60, 0.45, seed);
        sim.compute_accelerations();
        sim.compute_entropy_derivatives();

        let n = sim.particles.len();
        let mut kinetic_rate = 0.0;
        let mut thermal_rate = 0.0;
        let mut heating_rate = 0.0;
        let mut scale = 0.0;
        for i in 0..n {
            let pi = &sim.particles[i];
            kinetic_rate += pi.mass * (0..3).map(|d| pi.vel[d] * pi.accel[d]).sum::<f64>();
            // dρ_i/dt = Σ_j m_j v_ij·∇_iW_ij
            let mut drho_dt = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let grad_w = sim.pair_kernel_gradient(i, j);
                let pj = &sim.particles[j];
                drho_dt += pj.mass
                    * (0..3)
                        .map(|d| (pi.vel[d] - pj.vel[d]) * grad_w[d])
                        .sum::<f64>();
            }
            let adiabatic = pi.pressure / (pi.density * pi.density) * drho_dt;
            let heating = pi.density.powf(GAMMA - 1.0) / (GAMMA - 1.0) * pi.du_dt;
            thermal_rate += pi.mass * (adiabatic + heating);
            heating_rate += pi.mass * heating;
            scale += pi.mass
                * ((0..3).map(|d| (pi.vel[d] * pi.accel[d]).abs()).sum::<f64>()
                    + adiabatic.abs()
                    + heating.abs());
        }
        assert!(
            heating_rate > 0.02 * scale,
            "seed {seed}: the cloud must be viscous ({heating_rate} vs {scale})"
        );
        assert!(
            (kinetic_rate + thermal_rate).abs() < 1.0e-10 * scale,
            "seed {seed}: dE/dt = {} (scale {scale})",
            kinetic_rate + thermal_rate
        );
    }
}

/// Supersonic collision of two cold slabs (Mach ≈ 5 each): the shock
/// decelerates the slabs, turns kinetic energy into heat (entropy rises) and
/// conserves the total energy over 100 steps. The former code heated without
/// decelerating and gained more than 50 % of the total energy here.
#[test]
fn colliding_slabs_decelerate_heat_up_and_conserve_energy() {
    let mut sim = colliding_slabs(0.1, 4, 0.5, 0.01, 1.0, 2.0);
    let e0 = sim.total_energy();
    let (ke0, th0, s0) = (kinetic(&sim), thermal(&sim), total_entropy(&sim));
    assert!(
        (e0 - (ke0 + th0)).abs() < 1.0e-12 * e0,
        "no potential energy without gravity"
    );
    assert!(
        ke0 > 5.0 * th0,
        "cold supersonic setup: KE {ke0}, thermal {th0}"
    );

    let mut max_deviation = 0.0_f64;
    for _ in 0..100 {
        sim.step(0.004);
        max_deviation = max_deviation.max(((sim.total_energy() - e0) / e0).abs());
    }
    let e1 = sim.total_energy();

    assert!(
        max_deviation < 5.0e-3,
        "total energy drifted by {max_deviation:e} (relative); E0 = {e0}, E1 = {e1}"
    );
    // Kinetic energy was converted into heat.
    assert!(kinetic(&sim) < 0.6 * ke0, "KE {} vs {ke0}", kinetic(&sim));
    assert!(
        thermal(&sim) > 4.0 * th0,
        "thermal {} vs {th0}",
        thermal(&sim)
    );
    assert!(
        total_entropy(&sim) > 4.0 * s0,
        "entropy {} vs {s0}",
        total_entropy(&sim)
    );
    // The slabs were slowed down: the mean x velocity of each slab dropped.
    let half = sim.particles.len() / 2;
    let left: f64 = sim.particles[..half].iter().map(|p| p.vel[0]).sum::<f64>() / half as f64;
    let right: f64 = sim.particles[half..].iter().map(|p| p.vel[0]).sum::<f64>() / half as f64;
    assert!(
        left < 0.35 && right > -0.35,
        "mean slab velocities {left}, {right}"
    );
    // Pairwise equal and opposite forces: the (zero) total momentum is conserved.
    let momentum = total_momentum(&sim);
    let mass: f64 = sim.particles.iter().map(|p| p.mass).sum();
    for (d, &p) in momentum.iter().enumerate() {
        assert!(
            p.abs() < 1.0e-12 * mass * 0.5,
            "momentum component {d}: {p}"
        );
    }
    assert!((sim.time - 0.4).abs() < 1.0e-12);
}

/// The energy error of the viscous collision converges with the time step.
#[test]
fn energy_error_shrinks_with_the_time_step() {
    let mut deviations = Vec::new();
    for (dt, steps) in [(0.008, 50), (0.004, 100), (0.002, 200)] {
        let mut sim = colliding_slabs(0.1, 4, 0.5, 0.01, 1.0, 2.0);
        let e0 = sim.total_energy();
        for _ in 0..steps {
            sim.step(dt);
        }
        deviations.push(((sim.total_energy() - e0) / e0).abs());
    }
    assert!(
        deviations[0] > deviations[1] && deviations[1] > deviations[2],
        "{deviations:?}"
    );
    assert!(deviations[2] < 1.0e-3, "{deviations:?}");
}

/// Without viscosity the flow is adiabatic: the entropy of every particle is
/// exactly constant, and the total energy is still conserved by the
/// entropy formulation (hot gas expanding into vacuum).
#[test]
fn adiabatic_flow_keeps_entropy_constant_and_conserves_energy() {
    let mut sim = colliding_slabs(0.1, 4, 0.5, 1.0, 0.0, 0.0);
    let entropy_before: Vec<f64> = sim.particles.iter().map(|p| p.entropy).collect();
    let e0 = sim.total_energy();
    for _ in 0..100 {
        sim.step(0.004);
    }
    for (p, &a0) in sim.particles.iter().zip(entropy_before.iter()) {
        assert_eq!(p.entropy, a0, "particle {}", p.id);
    }
    assert!(((sim.total_energy() - e0) / e0).abs() < 1.0e-3);
    // The gas did expand: thermal energy became kinetic energy.
    assert!(kinetic(&sim) > 2.0 * 0.009);
}

// ---------------------------------------------------------------------------
// Internal energy bookkeeping
// ---------------------------------------------------------------------------

/// After `step` the specific internal energy follows the evolved entropy,
/// `u = A ρ^(γ−1)/(γ−1)`, and `total_energy` is kinetic + Σ m u.
#[test]
fn internal_energy_follows_the_evolved_entropy() {
    let mut sim = colliding_slabs(0.1, 4, 0.5, 0.01, 1.0, 2.0);
    let u_initial: Vec<f64> = sim.particles.iter().map(|p| p.internal_energy).collect();
    for _ in 0..30 {
        sim.step(0.004);
    }
    let mut changed = 0;
    for (p, &u0) in sim.particles.iter().zip(u_initial.iter()) {
        let expected = p.entropy * p.density.powf(GAMMA - 1.0) / (GAMMA - 1.0);
        assert!(
            (p.internal_energy - expected).abs() <= 1.0e-12 * expected,
            "particle {}: u = {}, A ρ^(γ-1)/(γ-1) = {expected}",
            p.id,
            p.internal_energy
        );
        if (p.internal_energy - u0).abs() > 0.05 * u0 {
            changed += 1;
        }
        // Pressure and sound speed belong to the same entropy.
        assert!(
            (p.pressure - (GAMMA - 1.0) * p.density * p.internal_energy).abs()
                <= 1.0e-12 * p.pressure
        );
    }
    assert!(
        changed > 0,
        "the internal energy never left its initial value"
    );
    let expected_total = kinetic(&sim) + thermal(&sim);
    assert!((sim.total_energy() - expected_total).abs() < 1.0e-12 * expected_total);
}

/// `initialise_entropy` keeps the prescribed specific internal energies and
/// makes `A`, `P` and `c_s` consistent with the SPH densities, so a step of
/// zero length changes nothing. With the entropy the constructor derived from
/// a guessed density, the same step would change the thermal energy.
#[test]
fn initialise_entropy_removes_the_artificial_first_step_energy_jump() {
    let spacing: f64 = 0.1;
    let mass = spacing.powi(3);
    let build = || {
        let mut particles = Vec::new();
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..3 {
                    particles.push(AstroSphParticle::new(
                        particles.len(),
                        mass,
                        [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing],
                        [0.0; 3],
                        2.0,
                        2.2 * spacing,
                        1.0,
                        GAMMA,
                    ));
                }
            }
        }
        let mut sim = AdiabticSph::new(GAMMA, 1.0, 2.0, 0.01, particles);
        sim.gravity.g = 0.0;
        sim
    };

    // The constructor assumed ρ = 1, the SPH density of the finite block is
    // lower (truncated kernel sums): a zero-length step changes the energy.
    let mut naive = build();
    let e_naive = naive.total_energy();
    naive.step(0.0);
    assert!(
        ((naive.total_energy() - e_naive) / e_naive).abs() > 1.0e-2,
        "{} -> {}",
        e_naive,
        naive.total_energy()
    );

    let mut sim = build();
    sim.initialise_entropy();
    for p in &sim.particles {
        assert!(
            (p.internal_energy - 2.0).abs() <= 1.0e-12,
            "u = {}",
            p.internal_energy
        );
        assert!(p.density > 0.0);
        let pressure = (GAMMA - 1.0) * p.density * 2.0;
        assert!((p.pressure - pressure).abs() <= 1.0e-12 * pressure);
    }
    let e0 = sim.total_energy();
    sim.step(0.0);
    assert!(((sim.total_energy() - e0) / e0).abs() < 1.0e-12);
}
