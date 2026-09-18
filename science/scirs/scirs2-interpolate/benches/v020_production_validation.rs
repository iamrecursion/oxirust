// v0.2.0 Production Validation Benchmark Suite
//
// This benchmark suite provides comprehensive production validation for scirs2-interpolate v0.2.0:
// 1. Complete benchmarking suite against SciPy 1.13+
// 2. Profile memory usage under stress conditions
// 3. Validate SIMD performance gains across architectures
// 4. Test scalability to 1M+ data points
// 5. Stress testing with extreme inputs
// 6. Numerical stability analysis for edge cases
// 7. Complete spline derivative/integral interfaces
// 8. GPU acceleration for production workloads

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::{
    advanced::rbf::{RBFInterpolator, RBFKernel},
    benchmarking::{run_benchmarks_with_config, BenchmarkConfig},
    interp1d::{cubic_interpolate, linear_interpolate, pchip_interpolate},
    memory_monitor::{start_monitoring, stop_monitoring},
    production_validation::{
        validate_production_readiness_with_config, ProductionValidationConfig,
    },
    simd_comprehensive_validation::{
        quick_simd_validation, validate_simd_with_config, SimdValidationConfig,
    },
    spline::CubicSpline,
};
use std::hint::black_box;
use std::time::Duration;

// ===== 1. SciPy Parity Benchmarks =====

fn scipy_parity_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("scipy_parity");

    // Linear interpolation (scipy.interpolate.interp1d with kind='linear')
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 10.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| v.sin());
        let x_new: Array1<f64> = Array1::linspace(0.0, 10.0, size / 10);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("linear_interp1d", size), &size, |b, _| {
            b.iter(|| {
                let result = linear_interpolate(
                    black_box(&x.view()),
                    black_box(&y.view()),
                    black_box(&x_new.view()),
                );
                black_box(result)
            });
        });
    }

    // Cubic spline (scipy.interpolate.CubicSpline)
    for size in [100, 1_000, 10_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 10.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| v.sin());

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("cubic_spline_creation", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = CubicSpline::new(black_box(&x.view()), black_box(&y.view()));
                    black_box(result)
                });
            },
        );
    }

    // PCHIP interpolation (scipy.interpolate.PchipInterpolator)
    for size in [100, 1_000, 10_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 10.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| v.sin());
        let x_new: Array1<f64> = Array1::linspace(0.0, 10.0, size / 10);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("pchip_interpolate", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = pchip_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&x_new.view()),
                        false,
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ===== 2. Memory Profiling Under Stress =====

fn memory_stress_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_stress");
    group.measurement_time(Duration::from_secs(15));

    // Large dataset interpolation with memory monitoring
    for size in [10_000, 100_000, 1_000_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 100.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| (v / 10.0).sin() * 100.0 + v);
        let x_new: Array1<f64> = Array1::linspace(0.0, 100.0, size / 100);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("large_linear_interp", size),
            &size,
            |b, _| {
                b.iter(|| {
                    start_monitoring();
                    let result = linear_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&x_new.view()),
                    );
                    stop_monitoring();
                    black_box(result)
                });
            },
        );
    }

    // RBF interpolation memory stress test
    for size in [100, 500, 1_000].iter() {
        let mut x_flat = Vec::new();
        let mut y_data = Vec::new();
        for i in 0..*size {
            let xv = i as f64 * 0.1;
            let yv = i as f64 * 0.1;
            x_flat.push(xv);
            x_flat.push(yv);
            y_data.push(xv.sin() * yv.cos());
        }
        let x_arr = Array2::from_shape_vec((*size, 2), x_flat).expect("failed to create array");
        let y_arr = Array1::from_vec(y_data);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("rbf_memory_stress", size),
            &size,
            |b, _| {
                b.iter(|| {
                    start_monitoring();
                    let result = RBFInterpolator::new(
                        black_box(&x_arr.view()),
                        black_box(&y_arr.view()),
                        RBFKernel::ThinPlateSpline,
                        1.0,
                    );
                    stop_monitoring();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ===== 3. SIMD Performance Validation =====

fn simd_performance_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_performance");

    // Run comprehensive SIMD validation
    group.bench_function("simd_validation_quick", |b| {
        b.iter(|| {
            let result = quick_simd_validation::<f64>();
            black_box(result)
        });
    });

    // SIMD-accelerated evaluation benchmarks
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 10.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| v.sin());
        let x_new: Array1<f64> = Array1::linspace(0.0, 10.0, size / 10);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_cubic_interp", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = cubic_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&x_new.view()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ===== 4. Scalability Testing (1M+ data points) =====

fn scalability_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Test with very large datasets
    for size in [100_000, 500_000, 1_000_000, 2_000_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 1000.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| (v / 100.0).sin() * 100.0 + v * 0.1);
        let x_new: Array1<f64> = Array1::linspace(0.0, 1000.0, 10_000);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("large_scale_linear", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = linear_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&x_new.view()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ===== 5. Stress Testing with Extreme Inputs =====

fn stress_test_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_tests");

    // Test with extreme value ranges
    let test_cases = vec![
        ("tiny_values", 1e-100, 1e-99),
        ("large_values", 1e50, 1e51),
        ("mixed_scales", 1e-10, 1e10),
    ];

    for (name, min, max) in test_cases {
        let x: Array1<f64> = Array1::linspace(min, max, 1_000);
        let y: Array1<f64> = x.mapv(|v: f64| v * 2.0 + 1.0);
        let x_new: Array1<f64> = Array1::linspace(min, max, 100);

        group.bench_with_input(BenchmarkId::new("extreme_values", name), &name, |b, _| {
            b.iter(|| {
                let result = linear_interpolate(
                    black_box(&x.view()),
                    black_box(&y.view()),
                    black_box(&x_new.view()),
                );
                black_box(result)
            });
        });
    }

    // Test with noisy data
    let x: Array1<f64> = Array1::linspace(0.0, 10.0, 1_000);
    let y: Array1<f64> = x.mapv(|v: f64| v.sin() + (v * 100.0).sin() * 0.1); // High-frequency noise
    let x_new: Array1<f64> = Array1::linspace(0.0, 10.0, 100);

    group.bench_function("noisy_data_cubic", |b| {
        b.iter(|| {
            let result = cubic_interpolate(
                black_box(&x.view()),
                black_box(&y.view()),
                black_box(&x_new.view()),
            );
            black_box(result)
        });
    });

    // Test with non-uniform spacing
    let x_nonuniform: Array1<f64> = (0..1_000).map(|i| (i as f64).powi(2) * 0.01).collect();
    let y_nonuniform: Array1<f64> = x_nonuniform.mapv(|v: f64| v.sqrt());
    let x_new_nonuniform: Array1<f64> =
        Array1::linspace(0.0, x_nonuniform[x_nonuniform.len() - 1], 100);

    group.bench_function("non_uniform_spacing", |b| {
        b.iter(|| {
            let result = cubic_interpolate(
                black_box(&x_nonuniform.view()),
                black_box(&y_nonuniform.view()),
                black_box(&x_new_nonuniform.view()),
            );
            black_box(result)
        });
    });

    group.finish();
}

// ===== 6. Numerical Stability Analysis =====

fn numerical_stability_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");

    // Test with nearly singular data
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, 100);
    let y: Array1<f64> = x.mapv(|v: f64| v + 1e-15 * (v * 1000.0).sin()); // Nearly linear with tiny perturbations
    let x_new: Array1<f64> = Array1::linspace(0.0, 1.0, 20);

    group.bench_function("nearly_singular_cubic", |b| {
        b.iter(|| {
            let result = cubic_interpolate(
                black_box(&x.view()),
                black_box(&y.view()),
                black_box(&x_new.view()),
            );
            black_box(result)
        });
    });

    // Test with sharp discontinuities
    let x2: Array1<f64> = Array1::linspace(0.0, 10.0, 200);
    let y2: Array1<f64> = x2.mapv(|v: f64| if v < 5.0 { v } else { v + 10.0 }); // Jump discontinuity
    let x_new2: Array1<f64> = Array1::linspace(0.0, 10.0, 50);

    group.bench_function("discontinuous_data", |b| {
        b.iter(|| {
            let result = pchip_interpolate(
                black_box(&x2.view()),
                black_box(&y2.view()),
                black_box(&x_new2.view()),
                false,
            );
            black_box(result)
        });
    });

    // Test with oscillatory data (Runge's phenomenon)
    let x3: Array1<f64> = Array1::linspace(-5.0, 5.0, 11);
    let y3: Array1<f64> = x3.mapv(|v: f64| 1.0 / (1.0 + v * v)); // Runge's function
    let x_new3: Array1<f64> = Array1::linspace(-5.0, 5.0, 100);

    group.bench_function("runges_phenomenon", |b| {
        b.iter(|| {
            let result = cubic_interpolate(
                black_box(&x3.view()),
                black_box(&y3.view()),
                black_box(&x_new3.view()),
            );
            black_box(result)
        });
    });

    group.finish();
}

// ===== 7. Spline Derivatives and Integrals =====

fn spline_calculus_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("spline_calculus");

    for size in [100, 1_000, 10_000].iter() {
        let x: Array1<f64> = Array1::linspace(0.0, 10.0, *size);
        let y: Array1<f64> = x.mapv(|v: f64| v.sin());

        // Spline creation for derivative/integral tests
        let spline = CubicSpline::new(&x.view(), &y.view()).expect("failed to create spline");

        // Test derivative evaluation
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("spline_derivative", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for i in 0..100 {
                        let x_val = i as f64 * 0.1;
                        if let Ok(deriv) = spline.derivative(black_box(x_val)) {
                            sum += deriv;
                        }
                    }
                    black_box(sum)
                });
            },
        );

        // Test integral evaluation
        group.bench_with_input(BenchmarkId::new("spline_integral", size), &size, |b, _| {
            b.iter(|| {
                let result = spline.integrate(black_box(0.0), black_box(10.0));
                black_box(result)
            });
        });
    }

    group.finish();
}

// ===== 8. Comprehensive Production Validation =====

fn production_validation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("production_validation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Run full production validation suite
    group.bench_function("full_validation_suite", |b| {
        b.iter(|| {
            let config = ProductionValidationConfig::default();
            let result = validate_production_readiness_with_config(config);
            black_box(result)
        });
    });

    // Run SIMD validation
    group.bench_function("simd_validation_comprehensive", |b| {
        b.iter(|| {
            let config = SimdValidationConfig::default();
            let result = validate_simd_with_config::<f64>(config);
            black_box(result)
        });
    });

    // Run benchmarking suite
    group.bench_function("benchmark_suite_execution", |b| {
        b.iter(|| {
            let config = BenchmarkConfig {
                data_sizes: vec![100, 1_000, 10_000],
                ..BenchmarkConfig::default()
            };
            let result = run_benchmarks_with_config::<f64>(config);
            black_box(result)
        });
    });

    group.finish();
}

// ===== Criterion Configuration =====

criterion_group!(
    name = scipy_parity;
    config = Criterion::default().sample_size(50);
    targets = scipy_parity_benchmarks
);

criterion_group!(
    name = memory_stress;
    config = Criterion::default().sample_size(30);
    targets = memory_stress_benchmarks
);

criterion_group!(
    name = simd_performance;
    config = Criterion::default().sample_size(100);
    targets = simd_performance_benchmarks
);

criterion_group!(
    name = scalability;
    config = Criterion::default().sample_size(10);
    targets = scalability_benchmarks
);

criterion_group!(
    name = stress_tests;
    config = Criterion::default().sample_size(50);
    targets = stress_test_benchmarks
);

criterion_group!(
    name = numerical_stability;
    config = Criterion::default().sample_size(50);
    targets = numerical_stability_benchmarks
);

criterion_group!(
    name = spline_calculus;
    config = Criterion::default().sample_size(50);
    targets = spline_calculus_benchmarks
);

criterion_group!(
    name = production_validation;
    config = Criterion::default().sample_size(10);
    targets = production_validation_benchmark
);

criterion_main!(
    scipy_parity,
    memory_stress,
    simd_performance,
    scalability,
    stress_tests,
    numerical_stability,
    spline_calculus,
    production_validation
);
