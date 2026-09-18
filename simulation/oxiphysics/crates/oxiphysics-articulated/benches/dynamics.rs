// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks of the per-step dynamics entry points (RNEA, ABA, CRBA) on
//! serial revolute chains of 7 and 36 DoF.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxiphysics_articulated::{
    aba::{AbaWorkspace, aba, aba_with_workspace},
    body::RigidBody,
    crba::compute_mass_matrix_crba,
    joint::RevoluteJoint,
    model::ArticulatedModel,
    rnea::rnea,
    spatial::SpatialInertia,
    spatial::SpatialTransform,
};

/// Serial chain of `n` revolute joints with alternating axes and 0.3 m links.
fn chain(n: usize) -> ArticulatedModel {
    let mut model = ArticulatedModel::new([0.0, 0.0, -9.81]);
    let axes = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    for i in 0..n {
        let inertia = SpatialInertia::from_com(
            1.0 + 0.1 * i as f64,
            [0.0, 0.0, 0.15],
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.01]],
        );
        let (parent, x_t) = if i == 0 {
            (None, SpatialTransform::IDENTITY)
        } else {
            (
                Some(i - 1),
                SpatialTransform::from_translation([0.0, 0.0, 0.3]),
            )
        };
        model.add_body(
            RigidBody::new(format!("link{i}"), inertia, parent, x_t),
            Box::new(RevoluteJoint::new(axes[i % 3])),
        );
    }
    model
}

/// Deterministic, non-trivial joint-space vector of length `n`.
fn state(n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|k| scale * (0.37 * k as f64 + 0.1).sin())
        .collect()
}

fn bench_dynamics(c: &mut Criterion) {
    for &n in &[7usize, 36] {
        let model = chain(n);
        let q = state(n, 0.8);
        let qd = state(n, 1.3);
        let qdd = state(n, 0.5);
        let tau = state(n, 2.0);

        let mut group = c.benchmark_group("articulated_dynamics");
        group.bench_with_input(BenchmarkId::new("rnea", n), &n, |b, _| {
            b.iter(|| black_box(rnea(&model, black_box(&q), black_box(&qd), black_box(&qdd))))
        });
        group.bench_with_input(BenchmarkId::new("aba", n), &n, |b, _| {
            b.iter(|| black_box(aba(&model, black_box(&q), black_box(&qd), black_box(&tau))))
        });
        let mut ws = AbaWorkspace::new(&model);
        let mut out = vec![0.0; n];
        group.bench_with_input(BenchmarkId::new("aba_with_workspace", n), &n, |b, _| {
            b.iter(|| {
                let res = aba_with_workspace(
                    &model,
                    &mut ws,
                    black_box(&q),
                    black_box(&qd),
                    black_box(&tau),
                    &mut out,
                );
                black_box((res.is_ok(), out[0]))
            })
        });
        group.bench_with_input(BenchmarkId::new("crba", n), &n, |b, _| {
            b.iter(|| black_box(compute_mass_matrix_crba(&model, black_box(&q))))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_dynamics);
criterion_main!(benches);
