// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark: weakly-compressible SPH on the rayon thread pool.
//!
//! With the `parallel` feature the per-particle gather loops of the WCSPH
//! pipeline (density summation, symmetric pressure acceleration, Morris
//! viscosity and the continuum surface force) are evaluated on the rayon pool.
//! Every particle still sums its own neighbours in list order, so the result of
//! a step does not depend on the number of worker threads; only the time does.
//! This benchmark installs pools of 1, 2, 4 and 8 threads (capped at the number
//! of available cores) and times the same lattice step in each of them, so the
//! reported speed-up is that of the force and density loops alone.  The serial
//! reference of the very same loops is `wcsph_scaling` built without the
//! feature.
//!
//! Run with: `cargo bench -p oxiphysics-sph --features parallel --bench wcsph_parallel`

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiphysics_sph::simulation::types_sim::WcSphSim;

/// Lattice edge length: 22^3 = 10648 particles.
const EDGE: usize = 22;
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
/// Surface tension coefficient (N/m).
const SIGMA: f64 = 0.072;
/// Gravity (m/s^2).
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];
/// Time step (s), below the CFL bound `0.25 h / c0`.
const DT: f64 = 5.0e-5;

/// Cubic lattice of `EDGE^3` fluid particles at rest density, with the given
/// surface tension coefficient.
fn lattice_sim(sigma: f64) -> WcSphSim {
    let mut sim = WcSphSim::new(SMOOTHING_LENGTH, RHO0, C0, GAMMA, MU, sigma, GRAVITY);
    let mass = RHO0 * DX * DX * DX;
    for i in 0..EDGE {
        for j in 0..EDGE {
            for k in 0..EDGE {
                let position = [i as f64 * DX, j as f64 * DX, k as f64 * DX];
                sim.particles.add_raw(position, [0.0; 3], mass, MU, false);
            }
        }
    }
    sim
}

/// Thread counts to measure: 1, 2, 4 and 8, capped at the available cores (and
/// always including the cap itself).
fn thread_counts() -> Vec<usize> {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut counts: Vec<usize> = [1, 2, 4, 8].into_iter().filter(|&n| n < cores).collect();
    counts.push(cores);
    counts
}

fn bench_parallel_step(c: &mut Criterion) {
    let particles = (EDGE * EDGE * EDGE) as u64;
    for (label, sigma) in [("wcsph_step", 0.0), ("wcsph_step_surface_tension", SIGMA)] {
        let mut group = c.benchmark_group(format!("parallel/{label}"));
        group.sample_size(10);
        group.throughput(Throughput::Elements(particles));
        let template = lattice_sim(sigma);
        for threads in thread_counts() {
            let pool = match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                Ok(pool) => pool,
                Err(error) => {
                    eprintln!("skipping {threads} threads: {error}");
                    continue;
                }
            };
            group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
                b.iter_batched(
                    || template.clone(),
                    |mut sim| {
                        pool.install(|| sim.step(DT));
                        black_box(sim)
                    },
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_parallel_step);
criterion_main!(benches);
