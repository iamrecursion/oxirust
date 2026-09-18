#![allow(clippy::unwrap_used)]

//! SIMD Correlation Benchmark
//!
//! Benchmarks the performance improvements from SIMD-accelerated correlation
//! calculations for various array sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array1;
use scirs2_stats::pearson_r;
use std::hint::black_box;

fn bench_pearson_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pearson_correlation");

    // Test various array sizes to see SIMD benefit
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096].iter() {
        let n = *size;

        // Create test data with some correlation
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 + 0.1 * (i as f64).sin()));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 2.0 * i as f64 + 0.2 * (i as f64).cos()));

        group.bench_with_input(BenchmarkId::new("f64", size), size, |bencher, _| {
            bencher
                .iter(|| pearson_r::<f64, _>(&black_box(x.view()), &black_box(y.view())).unwrap());
        });

        // f32 version
        let x_f32: Array1<f32> =
            Array1::from_iter((0..n).map(|i| (i as f32) + 0.1 * (i as f32).sin()));
        let y_f32: Array1<f32> =
            Array1::from_iter((0..n).map(|i| 2.0 * (i as f32) + 0.2 * (i as f32).cos()));

        group.bench_with_input(BenchmarkId::new("f32", size), size, |bencher, _| {
            bencher.iter(|| {
                pearson_r::<f32, _>(&black_box(x_f32.view()), &black_box(y_f32.view())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_correlation_matrix(c: &mut Criterion) {
    use scirs2_core::ndarray::Array2;
    use scirs2_stats::corrcoef;

    let mut group = c.benchmark_group("correlation_matrix");

    // Test correlation matrix computation for various numbers of variables
    for n_vars in [5, 10, 20, 50].iter() {
        let n_observations = 1000;
        let n = *n_vars;

        // Create random-like data
        let data: Array2<f64> = Array2::from_shape_fn((n_observations, n), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.2 + ((i * j) as f64).sin()
        });

        group.bench_with_input(BenchmarkId::new("pearson", n_vars), n_vars, |bencher, _| {
            bencher.iter(|| corrcoef::<f64, _>(&black_box(data.view()), "pearson").unwrap());
        });
    }

    group.finish();
}

fn bench_large_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_correlation");
    group.sample_size(10); // Reduce sample size for large arrays

    // Test with very large arrays to see maximum SIMD benefit
    for size in [8192, 16384].iter() {
        let n = *size;

        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 + 0.1 * (i as f64).sin()));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 2.0 * i as f64 + 0.2 * (i as f64).cos()));

        group.bench_with_input(BenchmarkId::new("f64", size), size, |bencher, _| {
            bencher
                .iter(|| pearson_r::<f64, _>(&black_box(x.view()), &black_box(y.view())).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pearson_correlation,
    bench_correlation_matrix,
    bench_large_correlation
);
criterion_main!(benches);
