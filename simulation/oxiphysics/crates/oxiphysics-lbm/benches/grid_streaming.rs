// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmark: periodic pull streaming of the flat grids.
//!
//! Two kinds of groups are reported:
//!
//! * `pull_stream_*` compares [`pull_stream_periodic`] (separate destination
//!   array, constant source offsets in the interior, `rem_euclid` only in the
//!   one-cell periodic shell) with the streaming the grids used before: clone
//!   the whole population array, then wrap all three indices of every
//!   population with `rem_euclid`.
//! * `step_*` compares the production `FlatLbmGrid2D::step` /
//!   `FlatLbmGrid3D::step` (BGK collision plus the new streaming) with a
//!   reference step that performs the same collision followed by the old
//!   streaming, i.e. the end-to-end gain per lattice-node update.
//!
//! The default sizes are `256²` (D2Q9) and `64³` (D3Q19).  A `128³` D3Q19 grid
//! holds 320 MB per population array, so it is only included when the
//! environment variable `OXIPHYSICS_BENCH_LARGE` is set.
//!
//! Run with: `cargo bench -p oxiphysics-lbm --bench grid_streaming`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiphysics_lbm::grid::pull_stream::{D2Q9_VELOCITIES_3D, pull_stream_periodic};
use oxiphysics_lbm::grid::{FlatLbmGrid2D, FlatLbmGrid3D};
use oxiphysics_lbm::lattice::{CS2, D2Q9_WEIGHTS, D3Q19_VELOCITIES, D3Q19_WEIGHTS};
use std::hint::black_box;

/// Deterministic, non-uniform populations.
fn seeded(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| 0.2 + ((i as u64).wrapping_mul(2_654_435_761) % 1_000_003) as f64 * 1e-6)
        .collect()
}

/// The pre-optimisation streaming: clone, then wrap every index.
fn reference_stream<const Q: usize>(f: &mut [f64], dims: [usize; 3], velocities: &[[i32; 3]; Q]) {
    let [nx, ny, nz] = dims;
    let f_old = f.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = (z * ny * nx + y * nx + x) * Q;
                for (a, c) in velocities.iter().enumerate() {
                    let sx = (x as isize - c[0] as isize).rem_euclid(nx as isize) as usize;
                    let sy = (y as isize - c[1] as isize).rem_euclid(ny as isize) as usize;
                    let sz = (z as isize - c[2] as isize).rem_euclid(nz as isize) as usize;
                    f[dst + a] = f_old[(sz * ny * nx + sy * nx + sx) * Q + a];
                }
            }
        }
    }
}

/// BGK collision of a flat population array with relaxation frequency `omega`.
fn reference_collide<const Q: usize>(
    f: &mut [f64],
    omega: f64,
    velocities: &[[i32; 3]; Q],
    weights: &[f64; Q],
) {
    for cell in f.chunks_exact_mut(Q) {
        let mut rho = 0.0;
        let mut momentum = [0.0; 3];
        for (value, c) in cell.iter().zip(velocities.iter()) {
            rho += value;
            for (m, &component) in momentum.iter_mut().zip(c.iter()) {
                *m += value * component as f64;
            }
        }
        let u = if rho.abs() > 1e-15 {
            [momentum[0] / rho, momentum[1] / rho, momentum[2] / rho]
        } else {
            [0.0; 3]
        };
        let u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        for ((value, c), &w) in cell.iter_mut().zip(velocities.iter()).zip(weights.iter()) {
            let eu = c[0] as f64 * u[0] + c[1] as f64 * u[1] + c[2] as f64 * u[2];
            let feq = w * rho * (1.0 + eu / CS2 + eu * eu / (2.0 * CS2 * CS2) - u_sq / (2.0 * CS2));
            *value -= omega * (*value - feq);
        }
    }
}

fn large_enabled() -> bool {
    std::env::var_os("OXIPHYSICS_BENCH_LARGE").is_some()
}

fn bench_pull_stream_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("pull_stream_d2q9");
    for size in [128usize, 256] {
        let len = size * size * 9;
        group.throughput(Throughput::Elements(len as u64));
        let dims = [size, size, 1];
        let mut f = seeded(len);
        group.bench_with_input(BenchmarkId::new("clone_rem_euclid", size), &size, |b, _| {
            b.iter(|| reference_stream::<9>(black_box(&mut f), dims, &D2Q9_VELOCITIES_3D))
        });
        let src = seeded(len);
        let mut dst = vec![0.0; len];
        group.bench_with_input(
            BenchmarkId::new("pull_stream_periodic", size),
            &size,
            |b, _| {
                b.iter(|| {
                    pull_stream_periodic::<9>(
                        black_box(&src),
                        black_box(&mut dst),
                        dims,
                        &D2Q9_VELOCITIES_3D,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_pull_stream_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("pull_stream_d3q19");
    group.sample_size(10);
    let mut sizes = vec![32usize, 64];
    if large_enabled() {
        sizes.push(128);
    }
    for size in sizes {
        let len = size * size * size * 19;
        group.throughput(Throughput::Elements(len as u64));
        let dims = [size, size, size];
        let mut f = seeded(len);
        group.bench_with_input(BenchmarkId::new("clone_rem_euclid", size), &size, |b, _| {
            b.iter(|| reference_stream::<19>(black_box(&mut f), dims, &D3Q19_VELOCITIES))
        });
        let src = seeded(len);
        let mut dst = vec![0.0; len];
        group.bench_with_input(
            BenchmarkId::new("pull_stream_periodic", size),
            &size,
            |b, _| {
                b.iter(|| {
                    pull_stream_periodic::<19>(
                        black_box(&src),
                        black_box(&mut dst),
                        dims,
                        &D3Q19_VELOCITIES,
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_step_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_flat_d2q9");
    for size in [128usize, 256] {
        group.throughput(Throughput::Elements((size * size) as u64));
        let mut reference = FlatLbmGrid2D::new(size, size, 0.8);
        reference.f = seeded(size * size * 9);
        let omega = 1.0 / reference.tau;
        group.bench_with_input(
            BenchmarkId::new("collide_then_clone_stream", size),
            &size,
            |b, _| {
                b.iter(|| {
                    reference_collide::<9>(
                        black_box(&mut reference.f),
                        omega,
                        &D2Q9_VELOCITIES_3D,
                        &D2Q9_WEIGHTS,
                    );
                    reference_stream::<9>(&mut reference.f, [size, size, 1], &D2Q9_VELOCITIES_3D);
                })
            },
        );
        let mut grid = FlatLbmGrid2D::new(size, size, 0.8);
        grid.f = seeded(size * size * 9);
        group.bench_with_input(
            BenchmarkId::new("FlatLbmGrid2D_step", size),
            &size,
            |b, _| b.iter(|| black_box(&mut grid).step()),
        );
    }
    group.finish();
}

fn bench_step_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_flat_d3q19");
    group.sample_size(10);
    let mut sizes = vec![32usize, 64];
    if large_enabled() {
        sizes.push(128);
    }
    for size in sizes {
        group.throughput(Throughput::Elements((size * size * size) as u64));
        let mut reference = FlatLbmGrid3D::new(size, size, size, 0.8);
        reference.f = seeded(size * size * size * 19);
        let omega = 1.0 / reference.tau;
        group.bench_with_input(
            BenchmarkId::new("collide_then_clone_stream", size),
            &size,
            |b, _| {
                b.iter(|| {
                    reference_collide::<19>(
                        black_box(&mut reference.f),
                        omega,
                        &D3Q19_VELOCITIES,
                        &D3Q19_WEIGHTS,
                    );
                    reference_stream::<19>(&mut reference.f, [size, size, size], &D3Q19_VELOCITIES);
                })
            },
        );
        let mut grid = FlatLbmGrid3D::new(size, size, size, 0.8);
        grid.f = seeded(size * size * size * 19);
        group.bench_with_input(
            BenchmarkId::new("FlatLbmGrid3D_step", size),
            &size,
            |b, _| b.iter(|| black_box(&mut grid).step()),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pull_stream_2d,
    bench_pull_stream_3d,
    bench_step_2d,
    bench_step_3d
);
criterion_main!(benches);
