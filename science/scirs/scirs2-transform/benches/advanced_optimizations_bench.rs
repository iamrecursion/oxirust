//! Benchmarks for Advanced mode optimizations in scirs2-transform
//!
//! This benchmark suite tests the performance improvements from the advanced
//! optimizations including adaptive SIMD, memory pools, and cache-optimal processing.
//!
//! Run with: cargo bench --bench advanced_optimizations_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::RandomExt;
use scirs2_core::ndarray::{Array2, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::OptimizedArrayRandom;
use scirs2_transform::*;
use std::hint::black_box;

const SAMPLE_SIZES: &[usize] = &[100, 1000, 10_000];
const FEATURE_SIZES: &[usize] = &[10, 50, 100];

/// Benchmark adaptive SIMD normalization vs standard normalization
#[allow(dead_code)]
fn bench_adaptive_simd_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive SIMD Normalization");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in FEATURE_SIZES {
            let mut rng = thread_rng();
            let data = Array2::random_bulk(
                Ix2(n_samples, n_features),
                Uniform::new(-100.0, 100.0).expect("Operation failed"),
                &mut rng,
            );

            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            // Standard normalization
            group.bench_with_input(
                BenchmarkId::new("Standard", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result =
                            normalize_array(black_box(data), NormalizationMethod::ZScore, 0);
                    });
                },
            );

            #[cfg(feature = "simd")]
            {
                // Adaptive SIMD normalization
                group.bench_with_input(
                    BenchmarkId::new("Adaptive SIMD", format!("{}x{}", n_samples, n_features)),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            let _result = simd_normalize_adaptive(
                                black_box(data),
                                NormalizationMethod::ZScore,
                                0,
                            );
                        });
                    },
                );

                // Batch SIMD normalization
                group.bench_with_input(
                    BenchmarkId::new("Batch SIMD", format!("{}x{}", n_samples, n_features)),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            let _result = simd_normalize_batch(
                                black_box(data),
                                NormalizationMethod::ZScore,
                                0,
                                64, // 64MB batch size
                            );
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark enhanced standard scaler with different processing strategies
#[allow(dead_code)]
fn bench_enhanced_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("Enhanced Standard Scaler");

    for &n_samples in SAMPLE_SIZES {
        for &n_features in FEATURE_SIZES {
            let mut rng = thread_rng();
            let data = Array2::random_bulk(
                Ix2(n_samples, n_features),
                Uniform::new(-100.0, 100.0).expect("Operation failed"),
                &mut rng,
            );

            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            // Standard scaler
            group.bench_with_input(
                BenchmarkId::new("Standard", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut scaler = EnhancedStandardScaler::new(false, 512);
                        let _result = scaler.fit_transform(black_box(&data.view()));
                    });
                },
            );

            // Robust scaler
            group.bench_with_input(
                BenchmarkId::new("Robust", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut scaler = EnhancedStandardScaler::new(true, 512);
                        let _result = scaler.fit_transform(black_box(&data.view()));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark enhanced PCA with different algorithms
#[allow(dead_code)]
fn bench_enhanced_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("Enhanced PCA");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[20, 50, 100] {
            let mut rng = thread_rng();
            let data = Array2::random_bulk(
                Ix2(n_samples, n_features),
                Uniform::new(-10.0, 10.0).expect("Operation failed"),
                &mut rng,
            );
            let n_components = (n_features / 2).min(10);

            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            // Enhanced PCA
            group.bench_with_input(
                BenchmarkId::new("Enhanced", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut pca = EnhancedPCA::new(n_components, true, 512)
                            .expect("Enhanced PCA creation failed");
                        let _result = pca.fit_transform(black_box(&data.view()));
                    });
                },
            );

            // Optimized PCA (memory pool) - commented out as AdvancedFastPCA is not implemented
            // group.bench_with_input(
            //     BenchmarkId::new("AdvancedFast", format!("{}x{}", n_samples, n_features)),
            //     &data,
            //     |b, _data| {
            //         let mut advanced_pca =
            //             AdvancedFastPCA::new(n_components, n_samples, n_features);
            //         b.iter(|| {
            //             // Note: This would need the fitting implementation to be complete
            //             let _stats = advanced_pca.performance_stats();
            //         });
            //     },
            // );
        }
    }

    group.finish();
}

#[cfg(feature = "simd")]
/// Benchmark SIMD polynomial features with optimizations
#[allow(dead_code)]
fn bench_simd_polynomial_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Polynomial Features");

    for &n_samples in &[100, 500, 1000] {
        for &n_features in &[5, 10, 20] {
            let mut rng = thread_rng();
            let data = Array2::random_bulk(
                Ix2(n_samples, n_features),
                Uniform::new(-2.0, 2.0).expect("Operation failed"),
                &mut rng,
            );

            group.throughput(Throughput::Elements((n_samples * n_features) as u64));

            // Standard polynomial features
            group.bench_with_input(
                BenchmarkId::new("Standard", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let poly = PolynomialFeatures::new(2, false, false);
                        let _result = poly.transform(black_box(&data.view()));
                    });
                },
            );

            // SIMD polynomial features
            group.bench_with_input(
                BenchmarkId::new("SIMD", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let poly = SimdPolynomialFeatures::<f64>::new(2, false, false)
                            .expect("Operation failed");
                        let _result = poly.transform(black_box(data));
                    });
                },
            );

            // Optimized SIMD polynomial features
            group.bench_with_input(
                BenchmarkId::new("Optimized SIMD", format!("{}x{}", n_samples, n_features)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let _result = simd_polynomial_features_optimized(
                            black_box(data),
                            2,
                            false,
                            false,
                            256, // 256MB memory limit
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory pool performance
#[allow(dead_code)]
fn bench_memory_pool_performance(_c: &mut Criterion) {
    // Commented out as AdvancedFastMemoryPool is not implemented
    // let mut group = c.benchmark_group("Memory Pool Performance");

    // let pool = AdvancedFastMemoryPool::new(1000, 100, 4);

    // group.bench_function("Memory Pool Stats", |b| {
    //     b.iter(|| {
    //         let _stats = pool.stats();
    //     });
    // });

    // group.finish();
}

/// Comprehensive performance comparison
#[allow(dead_code)]
fn bench_comprehensive_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive Performance");

    let mut rng = thread_rng();
    let large_data = Array2::random_bulk(
        Ix2(5000, 50),
        Uniform::new(-10.0, 10.0).expect("Operation failed"),
        &mut rng,
    );

    group.throughput(Throughput::Elements((5000 * 50) as u64));

    // Complete preprocessing pipeline - standard
    group.bench_with_input(
        BenchmarkId::new("Standard Pipeline", "5000x50"),
        &large_data,
        |b, data| {
            b.iter(|| {
                // Normalization
                let normalized = normalize_array(black_box(data), NormalizationMethod::ZScore, 0)
                    .expect("Normalization failed");

                // Polynomial features (degree 2)
                let poly = PolynomialFeatures::new(2, false, false);
                let poly_features = poly
                    .transform(&normalized.view())
                    .expect("Polynomial transform failed");

                // PCA
                let mut pca = PCA::new(10, true, false);
                let _result = pca.fit_transform(&poly_features.view());
            });
        },
    );

    // Complete preprocessing pipeline - optimized
    group.bench_with_input(
        BenchmarkId::new("Optimized Pipeline", "5000x50"),
        &large_data,
        |b, data| {
            b.iter(|| {
                #[cfg(feature = "simd")]
                {
                    // Adaptive SIMD normalization
                    let normalized =
                        simd_normalize_adaptive(black_box(data), NormalizationMethod::ZScore, 0)
                            .expect("SIMD normalization failed");

                    // Optimized SIMD polynomial features
                    let poly_features = simd_polynomial_features_optimized(
                        &normalized,
                        2,
                        false,
                        false,
                        256, // 256MB memory limit
                    )
                    .expect("SIMD polynomial features failed");

                    // Enhanced PCA
                    let mut pca =
                        EnhancedPCA::new(10, true, 512).expect("Enhanced PCA creation failed");
                    let _result = pca.fit_transform(&poly_features.view());
                }

                #[cfg(not(feature = "simd"))]
                {
                    // Fallback to standard pipeline when SIMD is not available
                    let normalized =
                        normalize_array(black_box(data), NormalizationMethod::ZScore, 0)
                            .expect("Fallback normalization failed");

                    let poly = PolynomialFeatures::new(2, false, false);
                    let poly_features = poly
                        .transform(&normalized.view())
                        .expect("Fallback polynomial transform failed");

                    let mut pca = EnhancedPCA::new(10, true, 512)
                        .expect("Fallback enhanced PCA creation failed");
                    let _result = pca.fit_transform(&poly_features.view());
                }
            });
        },
    );

    group.finish();
}

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_adaptive_simd_normalization,
    bench_enhanced_standard_scaler,
    bench_enhanced_pca,
    bench_simd_polynomial_features,
    bench_memory_pool_performance,
    bench_comprehensive_performance,
);

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_adaptive_simd_normalization,
    bench_enhanced_standard_scaler,
    bench_enhanced_pca,
    bench_memory_pool_performance,
    bench_comprehensive_performance,
);

criterion_main!(benches);
