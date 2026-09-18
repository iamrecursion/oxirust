// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The density and force sums of [`WcSphSim`] evaluated on the rayon pool are
//! bitwise identical to the serial evaluation, for every number of threads.
//!
//! Every sum is a gather over the neighbour list of one particle, formed in list
//! order, so the particles are independent of each other and no reduction order
//! exists that a schedule could change.  The tests compare the bit patterns of
//! all densities, pressures, accelerations, velocities, positions and energy
//! histories after the pair sums and after several full steps, with the
//! continuum surface force switched on and off, boundary particles, neighbour
//! lists that are shorter than the particle set and entries that name the
//! particle itself or a particle that does not exist.

use rayon::ThreadPoolBuilder;

use super::types_sim::WcSphSim;
use crate::parallel::Execution;

/// Rest density (kg/m³).
const RHO0: f64 = 1000.0;
/// Numerical speed of sound (m/s).
const C0: f64 = 10.0;
/// Tait exponent.
const GAMMA: f64 = 7.0;
/// Dynamic viscosity (Pa·s).
const MU: f64 = 0.05;
/// Surface tension coefficient (N/m).
const SIGMA: f64 = 0.0728;
/// Gravity (m/s²).
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
/// Smoothing length (m).
const H: f64 = 0.06;
/// Edge of the cube the particles are scattered over (m).
const EXTENT: f64 = 0.5;
/// Number of particles: more than twice the particles a task takes, so the
/// public entry points use the pool as well.
const COUNT: usize = 700;
/// Thread counts the parallel evaluation is repeated with.
const THREADS: [usize; 3] = [1, 3, 4];

/// Deterministic pseudo-random number in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

/// Random cloud of [`COUNT`] particles with random masses and velocities; every
/// ninth particle is a boundary particle.
fn cloud_sim(sigma: f64, seed: u64) -> WcSphSim {
    let mut state = seed;
    let mut sim = WcSphSim::new(H, RHO0, C0, GAMMA, MU, sigma, GRAVITY);
    let mass0 = RHO0 * EXTENT.powi(3) / COUNT as f64;
    for i in 0..COUNT {
        let position = [
            EXTENT * lcg_unit(&mut state),
            EXTENT * lcg_unit(&mut state),
            EXTENT * lcg_unit(&mut state),
        ];
        let velocity = [
            lcg_unit(&mut state) - 0.5,
            lcg_unit(&mut state) - 0.5,
            lcg_unit(&mut state) - 0.5,
        ];
        let mass = mass0 * (0.8 + 0.4 * lcg_unit(&mut state));
        sim.particles
            .add_raw(position, velocity, mass, MU, i % 9 == 4);
    }
    sim
}

fn assert_same_scalars(serial: &[f64], parallel: &[f64], what: &str) {
    assert_eq!(serial.len(), parallel.len(), "{what}: length");
    for (i, (s, p)) in serial.iter().zip(parallel).enumerate() {
        assert_eq!(s.to_bits(), p.to_bits(), "{what}: element {i}: {s} vs {p}");
    }
}

fn assert_same_vectors(serial: &[[f64; 3]], parallel: &[[f64; 3]], what: &str) {
    assert_eq!(serial.len(), parallel.len(), "{what}: length");
    for (i, (s, p)) in serial.iter().zip(parallel).enumerate() {
        for k in 0..3 {
            assert_eq!(
                s[k].to_bits(),
                p[k].to_bits(),
                "{what}: element {i} component {k}: {s:?} vs {p:?}"
            );
        }
    }
}

fn assert_same_state(serial: &WcSphSim, parallel: &WcSphSim, when: &str) {
    let (s, p) = (&serial.particles, &parallel.particles);
    assert_same_vectors(&s.positions, &p.positions, &format!("{when}: positions"));
    assert_same_vectors(&s.velocities, &p.velocities, &format!("{when}: velocities"));
    assert_same_vectors(
        &s.accelerations,
        &p.accelerations,
        &format!("{when}: accelerations"),
    );
    assert_same_scalars(&s.densities, &p.densities, &format!("{when}: densities"));
    assert_same_scalars(&s.pressures, &p.pressures, &format!("{when}: pressures"));
    assert_same_scalars(
        &serial.kinetic_energy_history,
        &parallel.kinetic_energy_history,
        &format!("{when}: kinetic energy history"),
    );
    assert_same_scalars(
        &serial.potential_energy_history,
        &parallel.potential_energy_history,
        &format!("{when}: potential energy history"),
    );
    assert_eq!(
        serial.time.to_bits(),
        parallel.time.to_bits(),
        "{when}: time"
    );
}

/// Run `parallel` in pools of every thread count of [`THREADS`].
fn in_pools(mut parallel: impl FnMut() + Send) {
    for threads in THREADS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(&mut parallel);
    }
}

#[test]
fn pair_sums_are_bitwise_identical() {
    let mut base = cloud_sim(0.0, 0xD0D0_0001);
    let mut neighbors = base.neighbor_lists();
    // Entries the sums must skip: the particle itself and a particle that does not
    // exist; and lists that are missing altogether for the last particles.
    for i in (0..COUNT).step_by(11) {
        neighbors[i].push(i);
        neighbors[i].push(COUNT + 5);
    }
    neighbors.truncate(COUNT - 40);
    base.particles.density_sum(Execution::Serial, H, &neighbors);
    base.pressure_step();
    assert!(base.particles.densities.iter().all(|&d| d > 0.0));

    // Density summation from the positions.
    let mut serial = base.clone();
    serial.particles.densities.fill(0.0);
    serial
        .particles
        .density_sum(Execution::Serial, H, &neighbors);
    in_pools(|| {
        let mut parallel = base.clone();
        parallel.particles.densities.fill(0.0);
        parallel
            .particles
            .density_sum(Execution::Parallel, H, &neighbors);
        assert_same_scalars(
            &serial.particles.densities,
            &parallel.particles.densities,
            "density sum",
        );
    });

    // Pressure and viscosity accelerations, accumulated onto a gravity field.
    let accelerate = |sim: &mut WcSphSim, execution: Execution| {
        sim.particles.clear_accelerations();
        sim.particles.apply_gravity(GRAVITY);
        sim.particles
            .pressure_acceleration(execution, H, &neighbors);
        sim.particles.morris_viscosity(execution, H, MU, &neighbors);
    };
    let mut serial = base.clone();
    accelerate(&mut serial, Execution::Serial);
    assert!(
        serial
            .particles
            .accelerations
            .iter()
            .any(|a| a[0] != 0.0 || a[1] != 0.0)
    );
    in_pools(|| {
        let mut parallel = base.clone();
        accelerate(&mut parallel, Execution::Parallel);
        assert_same_vectors(
            &serial.particles.accelerations,
            &parallel.particles.accelerations,
            "pressure + viscosity acceleration",
        );
    });
}

#[test]
fn full_steps_are_bitwise_identical_with_and_without_surface_tension() {
    for (sigma, seed) in [(0.0, 0xD0D0_0002), (SIGMA, 0xD0D0_0003)] {
        let initial = cloud_sim(sigma, seed);
        let mut serial = initial.clone();
        let mut states = Vec::new();
        for _ in 0..4 {
            serial.step_using(Execution::Serial, 1.0e-4);
            states.push(serial.clone());
        }
        assert!(
            serial
                .particles
                .positions
                .iter()
                .zip(&initial.particles.positions)
                .any(|(a, b)| a != b),
            "the particles moved"
        );
        in_pools(|| {
            let mut parallel = initial.clone();
            for (step, expected) in states.iter().enumerate() {
                parallel.step_using(Execution::Parallel, 1.0e-4);
                assert_same_state(expected, &parallel, &format!("sigma {sigma}, step {step}"));
            }
        });
    }
}

#[test]
fn the_public_step_uses_the_pool_for_large_sets_with_the_same_result() {
    assert!(Execution::for_count(COUNT) == Execution::Parallel);
    let initial = cloud_sim(SIGMA, 0xD0D0_0004);
    let mut serial = initial.clone();
    let mut public = initial.clone();
    for step in 0..3 {
        serial.step_using(Execution::Serial, 1.0e-4);
        public.step(1.0e-4);
        assert_same_state(&serial, &public, &format!("step {step}"));
    }
    // The force and density entry points used on their own.
    let neighbors = serial.neighbor_lists();
    let mut expected = serial.clone();
    expected
        .particles
        .density_sum(Execution::Serial, H, &neighbors);
    expected.pressure_step();
    expected.force_step_using(Execution::Serial, &neighbors);
    let mut by_public_api = serial.clone();
    by_public_api.density_step_with_neighbors(&neighbors);
    by_public_api.pressure_step();
    by_public_api.force_step_with_neighbors(&neighbors);
    assert_same_state(&expected, &by_public_api, "density + force steps");
}
