//! Performance regression benchmarks for scirs2-metrics
//!
//! This benchmark suite tracks performance of key metrics to detect regressions
//! and ensure consistent performance across releases.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::{
    anomaly::{js_divergence, kl_divergence, wasserstein_distance},
    classification::{accuracy_score, confusion_matrix, f1_score, precision_score, recall_score},
    clustering::{davies_bouldin_score, silhouette_score},
    optimization::numeric::StableMetrics,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

/// Generate synthetic classification data for benchmarking
#[allow(dead_code)]
fn generate_classification_data(_nsamples: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true: Array1<f64> = Array1::from_iter((0.._nsamples).map(|i| (i % 2) as f64));
    let y_pred: Array1<f64> = Array1::from_iter((0.._nsamples).map(|i| ((i + 1) % 2) as f64));
    (y_true, y_pred)
}

/// Generate synthetic classification data with integer labels for confusion matrix
#[allow(dead_code)]
fn generate_classification_data_int(_nsamples: usize) -> (Array1<i32>, Array1<i32>) {
    let y_true: Array1<i32> = Array1::from_iter((0.._nsamples).map(|i| (i % 2) as i32));
    let y_pred: Array1<i32> = Array1::from_iter((0.._nsamples).map(|i| ((i + 1) % 2) as i32));
    (y_true, y_pred)
}

/// Generate synthetic regression data for benchmarking
#[allow(dead_code)]
fn generate_regression_data(_nsamples: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true: Array1<f64> = Array1::from_iter((0.._nsamples).map(|i| i as f64));
    let y_pred: Array1<f64> = Array1::from_iter((0.._nsamples).map(|i| (i as f64) + 0.1));
    (y_true, y_pred)
}

/// Generate synthetic clustering data for benchmarking
#[allow(dead_code)]
fn generate_clustering_data(_n_samples: usize, nfeatures: usize) -> (Array2<f64>, Array1<usize>) {
    let mut data = Array2::zeros((_n_samples, nfeatures));
    let mut labels = Array1::zeros(_n_samples);

    for i in 0.._n_samples {
        let cluster = i % 3; // 3 clusters
        labels[i] = cluster;

        for j in 0..nfeatures {
            // Create cluster-specific offsets
            let offset = (cluster as f64) * 10.0;
            data[[i, j]] = offset + (i as f64) + (j as f64) * 0.1;
        }
    }

    (data, labels)
}

/// Generate synthetic probability distributions for benchmarking
#[allow(dead_code)]
fn generate_probability_distributions(_nsamples: usize) -> (Array1<f64>, Array1<f64>) {
    let mut p = Array1::zeros(_nsamples);
    let mut q = Array1::zeros(_nsamples);

    // Generate unnormalized weights
    let mut sum_p = 0.0;
    let mut sum_q = 0.0;

    for i in 0.._nsamples {
        p[i] = (i + 1) as f64;
        q[i] = ((i + 1) * 2) as f64;
        sum_p += p[i];
        sum_q += q[i];
    }

    // Normalize to make them proper probability distributions
    for i in 0.._nsamples {
        p[i] /= sum_p;
        q[i] /= sum_q;
    }

    (p, q)
}

/// Benchmark classification metrics performance
#[allow(dead_code)]
fn benchmark_classification_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification_metrics");

    for size in [100, 1000, 10000, 100000].iter() {
        let (y_true, y_pred) = generate_classification_data(*size);
        let (y_true_int, y_pred_int) = generate_classification_data_int(*size);

        group.bench_with_input(
            BenchmarkId::new("accuracy_score", size),
            size,
            |b, &_size| b.iter(|| accuracy_score(&y_true, &y_pred).expect("Operation failed")),
        );

        group.bench_with_input(
            BenchmarkId::new("precision_score", size),
            size,
            |b, &_size| {
                b.iter(|| precision_score(&y_true, &y_pred, 1.0).expect("Operation failed"))
            },
        );

        group.bench_with_input(BenchmarkId::new("recall_score", size), size, |b, &_size| {
            b.iter(|| recall_score(&y_true, &y_pred, 1.0).expect("Operation failed"))
        });

        group.bench_with_input(BenchmarkId::new("f1_score", size), size, |b, &_size| {
            b.iter(|| f1_score(&y_true, &y_pred, 1.0).expect("Operation failed"))
        });

        group.bench_with_input(
            BenchmarkId::new("confusion_matrix", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    confusion_matrix(&y_true_int, &y_pred_int, None).expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

/// Benchmark regression metrics performance
#[allow(dead_code)]
fn benchmark_regression_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_metrics");

    for size in [100, 1000, 10000, 100000].iter() {
        let (y_true, y_pred) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("mean_squared_error", size),
            size,
            |b, &_size| b.iter(|| mean_squared_error(&y_true, &y_pred).expect("Operation failed")),
        );

        group.bench_with_input(
            BenchmarkId::new("mean_absolute_error", size),
            size,
            |b, &_size| b.iter(|| mean_absolute_error(&y_true, &y_pred).expect("Operation failed")),
        );

        group.bench_with_input(BenchmarkId::new("r2_score", size), size, |b, &_size| {
            b.iter(|| r2_score(&y_true, &y_pred).expect("Operation failed"))
        });
    }

    group.finish();
}

/// Benchmark clustering metrics performance
#[allow(dead_code)]
fn benchmark_clustering_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_metrics");

    for size in [50, 100, 500, 1000].iter() {
        // Smaller sizes for clustering due to complexity
        let (data, labels) = generate_clustering_data(*size, 2);

        group.bench_with_input(
            BenchmarkId::new("silhouette_score", size),
            size,
            |b, &_size| {
                b.iter(|| silhouette_score(&data, &labels, "euclidean").expect("Operation failed"))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("davies_bouldin_score", size),
            size,
            |b, &_size| b.iter(|| davies_bouldin_score(&data, &labels).expect("Operation failed")),
        );
    }

    group.finish();
}

/// Benchmark anomaly detection metrics performance
#[allow(dead_code)]
fn benchmark_anomaly_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("anomaly_metrics");

    for size in [100, 1000, 10000].iter() {
        let (p, q) = generate_probability_distributions(*size);
        let (data1, data2) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("kl_divergence", size),
            size,
            |b, &_size| b.iter(|| kl_divergence(&p, &q).expect("Operation failed")),
        );

        group.bench_with_input(
            BenchmarkId::new("js_divergence", size),
            size,
            |b, &_size| b.iter(|| js_divergence(&p, &q).expect("Operation failed")),
        );

        group.bench_with_input(
            BenchmarkId::new("wasserstein_distance", size),
            size,
            |b, &_size| b.iter(|| wasserstein_distance(&data1, &data2).expect("Operation failed")),
        );
    }

    group.finish();
}

/// Benchmark stable statistical computation performance
#[allow(dead_code)]
fn benchmark_stable_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_metrics");
    let stable_metrics = StableMetrics::<f64>::new();

    for size in [100, 1000, 10000, 100000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        group.bench_with_input(BenchmarkId::new("stable_mean", size), size, |b, &_size| {
            b.iter(|| stable_metrics.stable_mean(&data).expect("Operation failed"))
        });

        group.bench_with_input(
            BenchmarkId::new("stable_variance", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    stable_metrics
                        .stable_variance(&data, 1)
                        .expect("Operation failed")
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("stable_std", size), size, |b, &_size| {
            b.iter(|| {
                stable_metrics
                    .stable_std(&data, 1)
                    .expect("Operation failed")
            })
        });
    }

    group.finish();
}

/// Benchmark high-dimensional data performance
#[allow(dead_code)]
fn benchmark_high_dimensional_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");

    // Test with various dimensions while keeping sample size reasonable
    for dims in [2, 5, 10, 20, 50].iter() {
        let (data, labels) = generate_clustering_data(500, *dims);

        group.bench_with_input(
            BenchmarkId::new("silhouette_high_dim", dims),
            dims,
            |b, &_dims| {
                b.iter(|| silhouette_score(&data, &labels, "euclidean").expect("Operation failed"))
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency for large datasets
#[allow(dead_code)]
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory usage patterns with different data sizes
    for size in [10000, 50000, 100000].iter() {
        let (y_true, y_pred) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("mse_large_data", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    // This tests memory allocation patterns

                    mean_squared_error(&y_true, &y_pred).expect("Operation failed")
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("accuracy_large_data", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let y_true_class: Array1<f64> = y_true.mapv(|x| (x as usize % 2) as f64);
                    let y_pred_class: Array1<f64> = y_pred.mapv(|x| (x as usize % 2) as f64);
                    accuracy_score(&y_true_class, &y_pred_class).expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

/// Benchmark worst-case scenarios for edge case handling
#[allow(dead_code)]
fn benchmark_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    // Test with very small numbers (potential underflow)
    let very_small_true = Array1::from_vec(vec![1e-100; 1000]);
    let very_small_pred = Array1::from_vec(vec![2e-100; 1000]);

    group.bench_function("mse_tiny_numbers", |b| {
        b.iter(|| mean_squared_error(&very_small_true, &very_small_pred).expect("Operation failed"))
    });

    // Test with very large numbers (potential overflow)
    let very_large_true = Array1::from_vec(vec![1e50; 1000]);
    let very_large_pred = Array1::from_vec(vec![2e50; 1000]);

    group.bench_function("mse_huge_numbers", |b| {
        b.iter(|| mean_squared_error(&very_large_true, &very_large_pred).expect("Operation failed"))
    });

    // Test with extreme class imbalance
    let mut imbalanced_true = vec![0.0; 10000];
    imbalanced_true[0] = 1.0; // Only one positive case
    let imbalanced_true = Array1::from_vec(imbalanced_true);
    let imbalanced_pred = Array1::zeros(10000);

    group.bench_function("accuracy_extreme_imbalance", |b| {
        b.iter(|| accuracy_score(&imbalanced_true, &imbalanced_pred).expect("Operation failed"))
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_classification_metrics,
    benchmark_regression_metrics,
    benchmark_clustering_metrics,
    benchmark_anomaly_metrics,
    benchmark_stable_metrics,
    benchmark_high_dimensional_performance,
    benchmark_memory_efficiency,
    benchmark_edge_cases
);

criterion_main!(benches);
