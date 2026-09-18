//! Benchmarks comparing raw complex-arithmetic tree walks against the
//! Phase-2B lowered evaluation path.
//!
//! For each trig/exp/composite shape we build the `EmlTree` once outside the
//! measurement loop and evaluate it at 1000 precomputed points per iteration.
//! The `raw` variant calls [`EmlTree::eval_real`] (complex-valued recursive
//! walk) whereas the `lowered` variant calls [`EmlTree::eval_real_lowered`]
//! (lower + simplify + `LoweredOp` stack machine with native `f64::sin`/
//! `f64::cos`).
//!
//! `black_box` wraps each individual angle inside the hot loop so the
//! compiler cannot hoist the computation out or constant-fold across
//! iterations.

use criterion::{Criterion, criterion_group, criterion_main};
use oxieml::{Canonical, EmlTree, EvalCtx};
use std::hint::black_box;

const SAMPLES: usize = 1000;

/// Produce `SAMPLES` evenly spaced points in the closed interval `[lo, hi]`.
fn linspace(lo: f64, hi: f64) -> Vec<f64> {
    let denom = (SAMPLES - 1) as f64;
    (0..SAMPLES)
        .map(|i| lo + (hi - lo) * (i as f64) / denom)
        .collect()
}

fn bench_sin_eval(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let points = linspace(-std::f64::consts::PI, std::f64::consts::PI);

    let mut group = c.benchmark_group("sin_eval");
    group.bench_function("raw", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &angle in &points {
                let ctx = EvalCtx::new(&[black_box(angle)]);
                acc += tree.eval_real(&ctx).expect("sin raw eval failed");
            }
            acc
        });
    });
    group.bench_function("lowered", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &angle in &points {
                let ctx = EvalCtx::new(&[black_box(angle)]);
                acc += tree
                    .eval_real_lowered(&ctx)
                    .expect("sin lowered eval failed");
            }
            acc
        });
    });
    group.finish();
}

fn bench_cos_eval(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let tree = Canonical::cos(&x);
    let points = linspace(-std::f64::consts::PI, std::f64::consts::PI);

    let mut group = c.benchmark_group("cos_eval");
    group.bench_function("raw", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &angle in &points {
                let ctx = EvalCtx::new(&[black_box(angle)]);
                acc += tree.eval_real(&ctx).expect("cos raw eval failed");
            }
            acc
        });
    });
    group.bench_function("lowered", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &angle in &points {
                let ctx = EvalCtx::new(&[black_box(angle)]);
                acc += tree
                    .eval_real_lowered(&ctx)
                    .expect("cos lowered eval failed");
            }
            acc
        });
    });
    group.finish();
}

fn bench_exp_eval(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let tree = Canonical::exp(&x);
    let points = linspace(-2.0, 2.0);

    let mut group = c.benchmark_group("exp_eval");
    group.bench_function("raw", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &v in &points {
                let ctx = EvalCtx::new(&[black_box(v)]);
                acc += tree.eval_real(&ctx).expect("exp raw eval failed");
            }
            acc
        });
    });
    group.bench_function("lowered", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &v in &points {
                let ctx = EvalCtx::new(&[black_box(v)]);
                acc += tree
                    .eval_real_lowered(&ctx)
                    .expect("exp lowered eval failed");
            }
            acc
        });
    });
    group.finish();
}

fn bench_composite_eval(c: &mut Criterion) {
    let x = EmlTree::var(0);
    let inner = Canonical::exp(&x);
    let tree = Canonical::sin(&inner);
    let points = linspace(-1.0, 1.0);

    let mut group = c.benchmark_group("composite_eval");
    group.bench_function("raw", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &v in &points {
                let ctx = EvalCtx::new(&[black_box(v)]);
                acc += tree.eval_real(&ctx).expect("composite raw eval failed");
            }
            acc
        });
    });
    group.bench_function("lowered", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &v in &points {
                let ctx = EvalCtx::new(&[black_box(v)]);
                acc += tree
                    .eval_real_lowered(&ctx)
                    .expect("composite lowered eval failed");
            }
            acc
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_sin_eval,
    bench_cos_eval,
    bench_exp_eval,
    bench_composite_eval,
);
criterion_main!(benches);
