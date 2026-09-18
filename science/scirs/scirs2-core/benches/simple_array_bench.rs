//! Simple Array Operation Benchmarks
//!
//! Basic benchmarks for array operations to validate performance.

use criterion::{criterion_group, criterion_main, Criterion};
use scirs2_core::ndarray_ext::{Array1, Array2, Ix1, Ix2};
use scirs2_core::random::{arrays::random_uniform_array, seeded_rng};
use std::hint::black_box;

#[allow(dead_code)]
fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");

    group.bench_function("zeros_1000", |b| {
        b.iter(|| {
            let arr = Array1::<f64>::zeros(1000);
            black_box(arr)
        })
    });

    group.bench_function("ones_1000", |b| {
        b.iter(|| {
            let arr = Array1::<f64>::ones(1000);
            black_box(arr)
        })
    });

    group.bench_function("random_1000", |b| {
        b.iter(|| {
            let mut rng = seeded_rng(42);
            let arr = random_uniform_array(Ix1(1000), &mut rng);
            black_box(arr)
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_operations");

    let mut rng = seeded_rng(42);
    let arr1 = random_uniform_array(Ix1(1000), &mut rng);
    let arr2 = random_uniform_array(Ix1(1000), &mut rng);

    group.bench_function("add_1000", |b| {
        b.iter(|| {
            let result = &arr1 + &arr2;
            black_box(result)
        })
    });

    group.bench_function("multiply_1000", |b| {
        b.iter(|| {
            let result = &arr1 * &arr2;
            black_box(result)
        })
    });

    group.bench_function("sum_1000", |b| {
        b.iter(|| {
            let result = arr1.sum();
            black_box(result)
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn benchmatrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    let mut rng = seeded_rng(42);
    let mat1 = random_uniform_array(Ix2(100, 100), &mut rng);
    let mat2 = random_uniform_array(Ix2(100, 100), &mut rng);

    group.bench_function("transpose_100x100", |b| {
        b.iter(|| {
            let result = mat1.t().to_owned();
            black_box(result)
        })
    });

    group.bench_function("element_multiply_100x100", |b| {
        b.iter(|| {
            let result = &mat1 * &mat2;
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_array_operations,
    benchmatrix_operations
);
criterion_main!(benches);
