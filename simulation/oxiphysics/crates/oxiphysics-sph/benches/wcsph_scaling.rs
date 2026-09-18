// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark: weakly-compressible SPH scaling with the particle count.
//!
//! Every pair sum of the WCSPH pipeline (density summation, symmetric pressure
//! acceleration, Morris viscosity and the continuum surface force) is evaluated
//! from neighbour lists, so one force evaluation costs `O(N * k)` for `k`
//! neighbours per particle.  The groups below run on cubic lattices of
//! 2744, 4913 and 10648 particles with the throughput set to the particle
//! count: a linear algorithm reports an (almost) constant elements-per-second
//! figure across the three sizes, whereas an all-pairs loop loses a factor of
//! about four in elements per second for every doubling of `N`.
//!
//! * `wcsph_force_evaluation` times the neighbour-list build, one complete
//!   density -> pressure -> force evaluation that shares a single set of lists
//!   (with and without the continuum surface force) and, for contrast, the
//!   density summation alone written as the all-pairs double loop.
//! * `wcsph_step` times a full [`WcSphSim::step`] from the rest lattice.
//!
//! Run with: `cargo bench -p oxiphysics-sph --bench wcsph_scaling`

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiphysics_sph::particle::cubic_spline_kernel;
use oxiphysics_sph::simulation::types_sim::WcSphSim;
use std::hint::black_box;

/// Lattice edge lengths: 14^3 = 2744, 17^3 = 4913 and 22^3 = 10648 particles.
const EDGES: [usize; 3] = [14, 17, 22];
/// Lattice spacing (m).
const DX: f64 = 0.01;
/// Smoothing length (m): 1.3 lattice spacings, i.e. about 80 neighbours.
const SMOOTHING_LENGTH: f64 = 1.3 * DX;
/// Rest density (kg/m^3).
const RHO0: f64 = 1000.0;
/// Numerical speed of sound (m/s).
const C0: f64 = 40.0;
/// Tait exponent.
const GAMMA: f64 = 7.0;
/// Dynamic viscosity (Pa s).
const MU: f64 = 1.0e-3;
/// Surface tension coefficient used by the surface-tension variant (N/m).
const SIGMA: f64 = 0.072;
/// Gravity (m/s^2).
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
/// Time step (s), below the CFL bound `0.25 h / c0`.
const DT: f64 = 5.0e-5;

/// Cubic lattice of `edge^3` fluid particles at rest density.
fn lattice_sim(edge: usize, sigma: f64) -> WcSphSim {
    let mut sim = WcSphSim::new(SMOOTHING_LENGTH, RHO0, C0, GAMMA, MU, sigma, GRAVITY);
    let mass = RHO0 * DX * DX * DX;
    for i in 0..edge {
        for j in 0..edge {
            for k in 0..edge {
                let position = [i as f64 * DX, j as f64 * DX, k as f64 * DX];
                sim.particles.add_raw(position, [0.0; 3], mass, MU, false);
            }
        }
    }
    sim
}

/// One density -> pressure -> force evaluation with a single set of neighbour
/// lists, exactly as [`WcSphSim::step`] does it before integrating.
fn evaluate_forces(sim: &mut WcSphSim) {
    let neighbors = sim.neighbor_lists();
    sim.density_step_with_neighbors(&neighbors);
    sim.pressure_step();
    sim.force_step_with_neighbors(&neighbors);
}

/// Density summation as an all-pairs double loop (the structure that the
/// neighbour-list sums replace); `O(N^2)` kernel evaluations.
fn all_pairs_density(sim: &mut WcSphSim) {
    let h = sim.h;
    let n = sim.particles.len();
    for i in 0..n {
        let pi = sim.particles.positions[i];
        let mut density = 0.0;
        for j in 0..n {
            let pj = sim.particles.positions[j];
            let dx = pi[0] - pj[0];
            let dy = pi[1] - pj[1];
            let dz = pi[2] - pj[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            density += sim.particles.masses[j] * cubic_spline_kernel(r, h);
        }
        sim.particles.densities[i] = density;
    }
}

fn bench_force_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("wcsph_force_evaluation");
    group.sample_size(10);
    for edge in EDGES {
        let n = edge * edge * edge;
        group.throughput(Throughput::Elements(n as u64));

        let mut sim = lattice_sim(edge, 0.0);
        group.bench_with_input(BenchmarkId::new("neighbour_lists", n), &n, |b, _| {
            b.iter(|| black_box(sim.neighbor_lists()))
        });
        group.bench_with_input(
            BenchmarkId::new("density_pressure_forces", n),
            &n,
            |b, _| b.iter(|| evaluate_forces(black_box(&mut sim))),
        );
        group.bench_with_input(
            BenchmarkId::new("all_pairs_density_reference", n),
            &n,
            |b, _| b.iter(|| all_pairs_density(black_box(&mut sim))),
        );

        let mut tension_sim = lattice_sim(edge, SIGMA);
        group.bench_with_input(BenchmarkId::new("with_surface_tension", n), &n, |b, _| {
            b.iter(|| evaluate_forces(black_box(&mut tension_sim)))
        });
    }
    group.finish();
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("wcsph_step");
    group.sample_size(10);
    for edge in EDGES {
        let n = edge * edge * edge;
        group.throughput(Throughput::Elements(n as u64));
        let template = lattice_sim(edge, 0.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || template.clone(),
                |mut sim| {
                    sim.step(DT);
                    sim
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_force_evaluation, bench_step);
criterion_main!(benches);
