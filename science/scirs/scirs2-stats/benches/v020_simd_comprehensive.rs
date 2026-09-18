#![allow(clippy::unwrap_used)]

//! Comprehensive v0.2.0 SIMD Performance Benchmarks
//!
//! This benchmark suite validates the 2-3x speedup target for SIMD enhancements
//! in scirs2-stats v0.2.0.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use std::hint::black_box;

fn bench_distribution_sampling(c: &mut Criterion) {
    use scirs2_stats::sampling_simd::{box_muller_simd, exponential_simd};

    let mut group = c.benchmark_group("distribution_sampling");

    for size in [100, 1000, 10000, 100000].iter() {
        let n = *size;

        // Box-Muller transform (normal distribution)
        group.bench_with_input(
            BenchmarkId::new("box_muller_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| {
                    box_muller_simd(black_box(n), black_box(0.0), black_box(1.0), Some(42)).unwrap()
                });
            },
        );

        // Exponential distribution sampling
        group.bench_with_input(
            BenchmarkId::new("exponential_simd", size),
            size,
            |bencher, _| {
                bencher.iter(|| exponential_simd(black_box(n), black_box(1.0), Some(42)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_correlation_simd(c: &mut Criterion) {
    use scirs2_stats::correlation_simd_enhanced::spearman_r_simd;
    use scirs2_stats::pearson_r_simd;

    let mut group = c.benchmark_group("correlation_simd");

    for size in [32, 128, 512, 2048, 8192].iter() {
        let n = *size;

        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 + 0.1 * (i as f64).sin()));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 2.0 * i as f64 + 0.2 * (i as f64).cos()));

        // Pearson correlation SIMD
        group.bench_with_input(
            BenchmarkId::new("pearson_simd", size),
            size,
            |bencher, _| {
                bencher
                    .iter(|| pearson_r_simd(&black_box(x.view()), &black_box(y.view())).unwrap());
            },
        );

        // Spearman correlation SIMD
        group.bench_with_input(
            BenchmarkId::new("spearman_simd", size),
            size,
            |bencher, _| {
                bencher
                    .iter(|| spearman_r_simd(&black_box(x.view()), &black_box(y.view())).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_covariance_matrix_simd(c: &mut Criterion) {
    use scirs2_stats::correlation_simd_enhanced::covariance_matrix_simd;

    let mut group = c.benchmark_group("covariance_matrix_simd");

    for n_vars in [5, 10, 20, 50].iter() {
        let n_observations = 1000;
        let n = *n_vars;

        let data: Array2<f64> = Array2::from_shape_fn((n_observations, n), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.2 + ((i * j) as f64).sin()
        });

        group.bench_with_input(BenchmarkId::new("simd", n_vars), n_vars, |bencher, _| {
            bencher.iter(|| {
                covariance_matrix_simd(&black_box(data.view()), black_box(false), black_box(1))
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_parallel_simd_correlation(c: &mut Criterion) {
    use scirs2_stats::parallel_simd_stats::corrcoef_parallel_simd;

    let mut group = c.benchmark_group("parallel_simd_correlation");
    group.sample_size(10); // Reduce for large matrices

    for n_vars in [10, 20, 50, 100].iter() {
        let n_observations = 1000;
        let n = *n_vars;

        let data: Array2<f64> = Array2::from_shape_fn((n_observations, n), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.2 + ((i * j) as f64).sin()
        });

        group.bench_with_input(BenchmarkId::new("pearson", n_vars), n_vars, |bencher, _| {
            bencher.iter(|| {
                corrcoef_parallel_simd(&black_box(data.view()), black_box("pearson")).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_parallel_simd_covariance(c: &mut Criterion) {
    use scirs2_stats::parallel_simd_stats::covariance_matrix_parallel_simd;

    let mut group = c.benchmark_group("parallel_simd_covariance");
    group.sample_size(10);

    for n_vars in [10, 20, 50, 100].iter() {
        let n_observations = 1000;
        let n = *n_vars;

        let data: Array2<f64> = Array2::from_shape_fn((n_observations, n), |(i, j)| {
            (i as f64) * 0.1 + (j as f64) * 0.2 + ((i * j) as f64).sin()
        });

        group.bench_with_input(BenchmarkId::new("cov", n_vars), n_vars, |bencher, _| {
            bencher.iter(|| {
                covariance_matrix_parallel_simd(&black_box(data.view()), black_box(1)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_bootstrap_parallel_simd(c: &mut Criterion) {
    use scirs2_stats::mean_simd;
    use scirs2_stats::parallel_simd_stats::bootstrap_parallel_simd;

    let mut group = c.benchmark_group("bootstrap_parallel_simd");

    for size in [100, 1000, 10000].iter() {
        let n = *size;
        let data: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));

        group.bench_with_input(BenchmarkId::new("mean", size), size, |bencher, _| {
            bencher.iter(|| {
                #[allow(clippy::redundant_closure)]
                bootstrap_parallel_simd(
                    &black_box(data.view()),
                    black_box(100),
                    |x| mean_simd(x),
                    Some(42),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_rolling_correlation_simd(c: &mut Criterion) {
    use scirs2_stats::correlation_simd_enhanced::rolling_correlation_simd;

    let mut group = c.benchmark_group("rolling_correlation_simd");

    for size in [100, 1000, 10000].iter() {
        let n = *size;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 + 0.1 * (i as f64).sin()));
        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 2.0 * i as f64 + 0.2 * (i as f64).cos()));

        group.bench_with_input(BenchmarkId::new("window_50", size), size, |bencher, _| {
            bencher.iter(|| {
                rolling_correlation_simd(&black_box(x.view()), &black_box(y.view()), black_box(50))
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_pairwise_distances_parallel_simd(c: &mut Criterion) {
    use scirs2_stats::parallel_simd_stats::pairwise_distances_parallel_simd;

    let mut group = c.benchmark_group("pairwise_distances_parallel_simd");
    group.sample_size(10);

    for n_points in [50, 100, 200].iter() {
        let n = *n_points;
        let data: Array2<f64> =
            Array2::from_shape_fn((n, 10), |(i, j)| (i as f64) * 0.1 + (j as f64) * 0.2);

        group.bench_with_input(
            BenchmarkId::new("euclidean", n_points),
            n_points,
            |bencher, _| {
                bencher.iter(|| {
                    pairwise_distances_parallel_simd(
                        &black_box(data.view()),
                        black_box("euclidean"),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distribution_sampling,
    bench_correlation_simd,
    bench_covariance_matrix_simd,
    bench_parallel_simd_correlation,
    bench_parallel_simd_covariance,
    bench_bootstrap_parallel_simd,
    bench_rolling_correlation_simd,
    bench_pairwise_distances_parallel_simd,
);
criterion_main!(benches);
