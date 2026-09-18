//! v0.2.0 Performance Validation Benchmark
//!
//! This benchmark validates the performance improvements and correctness
//! of v0.2.0 metrics implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::{
    classification::{accuracy_score, f1_score, precision_score, recall_score, roc_auc_score},
    clustering::{calinski_harabasz_score, davies_bouldin_score, silhouette_score},
    distance::*,
    information_theory::*,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};
use std::hint::black_box;

// ============================================================================
// Distance Metrics Benchmarks
// ============================================================================

fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Array1<f64> = Array1::linspace(0.0, 1.0, *size);
        let y: Array1<f64> = Array1::linspace(0.1, 1.1, *size);

        group.bench_with_input(BenchmarkId::new("euclidean", size), size, |b, _| {
            b.iter(|| {
                let _ = euclidean_distance(black_box(&x), black_box(&y));
            });
        });

        group.bench_with_input(BenchmarkId::new("manhattan", size), size, |b, _| {
            b.iter(|| {
                let _ = manhattan_distance(black_box(&x), black_box(&y));
            });
        });

        group.bench_with_input(BenchmarkId::new("cosine_similarity", size), size, |b, _| {
            b.iter(|| {
                let _ = cosine_similarity(black_box(&x), black_box(&y));
            });
        });

        group.bench_with_input(BenchmarkId::new("chebyshev", size), size, |b, _| {
            b.iter(|| {
                let _ = chebyshev_distance(black_box(&x), black_box(&y));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Information Theory Benchmarks
// ============================================================================

fn bench_information_theory(c: &mut Criterion) {
    let mut group = c.benchmark_group("information_theory");

    for size in [10, 100, 1_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Create probability distributions
        let p_vec: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();
        let sum: f64 = p_vec.iter().sum();
        let p: Array1<f64> = Array1::from_vec(p_vec.iter().map(|&x| x / sum).collect());

        let q_vec: Vec<f64> = (0..*size).map(|i| (*size - i) as f64).collect();
        let sum_q: f64 = q_vec.iter().sum();
        let q: Array1<f64> = Array1::from_vec(q_vec.iter().map(|&x| x / sum_q).collect());

        group.bench_with_input(BenchmarkId::new("entropy", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = entropy(black_box(&p));
            });
        });

        group.bench_with_input(BenchmarkId::new("kl_divergence", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = kl_divergence(black_box(&p), black_box(&q));
            });
        });

        group.bench_with_input(BenchmarkId::new("js_divergence", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = js_divergence(black_box(&p), black_box(&q));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Regression Metrics Benchmarks
// ============================================================================

fn bench_regression_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_metrics");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let y_true: Array1<f64> = Array1::linspace(0.0, 100.0, *size);
        let y_pred: Array1<f64> = Array1::linspace(0.1, 100.1, *size);

        group.bench_with_input(BenchmarkId::new("mse", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = mean_squared_error(black_box(&y_true), black_box(&y_pred));
            });
        });

        group.bench_with_input(BenchmarkId::new("mae", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = mean_absolute_error(black_box(&y_true), black_box(&y_pred));
            });
        });

        group.bench_with_input(BenchmarkId::new("r2", size), size, |b, _| {
            b.iter(|| {
                let _: Result<f64, _> = r2_score(black_box(&y_true), black_box(&y_pred));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Classification Metrics Benchmarks
// ============================================================================

fn bench_classification_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification_metrics");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let y_true: Array1<i32> = Array1::from_iter((0..*size).map(|i| i % 3));
        let y_pred: Array1<i32> = Array1::from_iter((0..*size).map(|i| (i + 1) % 3));

        group.bench_with_input(BenchmarkId::new("accuracy", size), size, |b, _| {
            b.iter(|| {
                let _ = accuracy_score(black_box(&y_true), black_box(&y_pred));
            });
        });

        group.bench_with_input(BenchmarkId::new("precision", size), size, |b, _| {
            b.iter(|| {
                let _ = precision_score(black_box(&y_true), black_box(&y_pred), 1);
            });
        });

        group.bench_with_input(BenchmarkId::new("recall", size), size, |b, _| {
            b.iter(|| {
                let _ = recall_score(black_box(&y_true), black_box(&y_pred), 1);
            });
        });

        group.bench_with_input(BenchmarkId::new("f1", size), size, |b, _| {
            b.iter(|| {
                let _ = f1_score(black_box(&y_true), black_box(&y_pred), 1);
            });
        });
    }

    // ROC AUC with float scores
    for size in [100, 1_000, 10_000].iter() {
        let y_true: Array1<u32> = Array1::from_iter((0..*size).map(|i| (i % 2) as u32));
        let y_score: Array1<f64> = Array1::linspace(0.0, 1.0, *size);

        group.bench_with_input(BenchmarkId::new("roc_auc", size), size, |b, _| {
            b.iter(|| {
                let _ = roc_auc_score(black_box(&y_true), black_box(&y_score));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Clustering Metrics Benchmarks
// ============================================================================

fn bench_clustering_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_metrics");

    for size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Create synthetic clustered data
        let n_features = 5;
        let mut data_vec = Vec::with_capacity(size * n_features);
        for i in 0..*size {
            let cluster = (i % 3) as f64;
            for j in 0..n_features {
                data_vec.push(cluster * 10.0 + (i as f64 * 0.1) + (j as f64 * 0.01));
            }
        }
        let data = Array2::from_shape_vec((*size, n_features), data_vec)
            .expect("Failed to create data array");
        let labels: Array1<usize> = Array1::from_iter((0..*size).map(|i| i % 3));

        group.bench_with_input(BenchmarkId::new("silhouette", size), size, |b, _| {
            b.iter(|| {
                let _ = silhouette_score(black_box(&data), black_box(&labels), "euclidean");
            });
        });

        group.bench_with_input(BenchmarkId::new("davies_bouldin", size), size, |b, _| {
            b.iter(|| {
                let _ = davies_bouldin_score(black_box(&data), black_box(&labels));
            });
        });

        group.bench_with_input(BenchmarkId::new("calinski_harabasz", size), size, |b, _| {
            b.iter(|| {
                let _ = calinski_harabasz_score(black_box(&data), black_box(&labels));
            });
        });
    }

    group.finish();
}

// ============================================================================
// SIMD Optimization Comparison
// ============================================================================

fn bench_simd_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Array1<f64> = Array1::linspace(0.0, 1.0, *size);
        let y: Array1<f64> = Array1::linspace(0.1, 1.1, *size);

        // SIMD-optimized version (standard layout)
        group.bench_with_input(BenchmarkId::new("euclidean_simd", size), size, |b, _| {
            b.iter(|| {
                let _ = euclidean_distance(black_box(&x), black_box(&y));
            });
        });

        // Create non-contiguous arrays to test fallback
        let x_nc = x.slice(s![..;2]).to_owned();
        let y_nc = y.slice(s![..;2]).to_owned();

        group.bench_with_input(
            BenchmarkId::new("euclidean_scalar", size / 2),
            &(size / 2),
            |b, _| {
                b.iter(|| {
                    let _ = euclidean_distance(black_box(&x_nc), black_box(&y_nc));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Mahalanobis Distance Benchmark
// ============================================================================

fn bench_mahalanobis_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("mahalanobis_distance");

    for size in [5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Array1<f64> = Array1::linspace(0.0, 1.0, *size);
        let y: Array1<f64> = Array1::linspace(0.1, 1.1, *size);
        let cov_inv = Array2::eye(*size);

        group.bench_with_input(BenchmarkId::new("mahalanobis", size), size, |b, _| {
            b.iter(|| {
                let _ = mahalanobis_distance(black_box(&x), black_box(&y), black_box(&cov_inv));
            });
        });
    }

    group.finish();
}

use scirs2_core::ndarray::s;

criterion_group!(
    benches,
    bench_distance_metrics,
    bench_information_theory,
    bench_regression_metrics,
    bench_classification_metrics,
    bench_clustering_metrics,
    bench_simd_optimization,
    bench_mahalanobis_distance
);

criterion_main!(benches);
