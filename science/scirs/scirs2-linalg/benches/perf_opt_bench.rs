use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::RandomExt;
use scirs2_core::ndarray::{Array2, ShapeBuilder};
use scirs2_core::random::prelude::*;
use scirs2_linalg::prelude::*;
use std::hint::black_box;

#[allow(dead_code)]
fn bench_matmul_optimizations(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("matmul_optimizations");
    group.sample_size(10); // Reduce sample size for large matrices

    let config_blocked = OptConfig::default()
        .with_blocksize(64)
        .with_parallel_threshold(256);

    let config_adaptive = OptConfig::default()
        .with_blocksize(64)
        .with_parallel_threshold(256)
        .with_algorithm(OptAlgorithm::Adaptive);

    for size in &sizes {
        let a = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });
        let b = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });

        // Standard ndarray matrix multiplication
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(a.dot(&b));
            });
        });

        // Our blocked matrix multiplication
        group.bench_with_input(BenchmarkId::new("blocked", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(
                    blocked_matmul(&a.view(), &b.view(), &config_blocked)
                        .expect("Operation failed"),
                );
            });
        });

        // Adaptive algorithm selection
        group.bench_with_input(BenchmarkId::new("adaptive", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(
                    blocked_matmul(&a.view(), &b.view(), &config_adaptive)
                        .expect("Operation failed"),
                );
            });
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_inplace_operations(c: &mut Criterion) {
    let sizes = [128, 256, 512, 1024, 2048];
    let mut group = c.benchmark_group("inplace_operations");

    for size in &sizes {
        let a = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });
        let b = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });

        // Standard addition (creates new array)
        group.bench_with_input(
            BenchmarkId::new("standard_add", size),
            size,
            |bench_, _data| {
                bench_.iter(|| {
                    let _result = black_box(&a + &b);
                });
            },
        );

        // In-place addition
        group.bench_with_input(
            BenchmarkId::new("inplace_add", size),
            size,
            |bench_, _data| {
                bench_.iter(|| {
                    let mut a_copy = a.clone();
                    inplace_add(&mut a_copy.view_mut(), &b.view()).expect("Operation failed");
                    black_box(&a_copy);
                });
            },
        );

        // Standard scaling (creates new array)
        group.bench_with_input(
            BenchmarkId::new("standard_scale", size),
            size,
            |bench_, _data| {
                bench_.iter(|| {
                    let _result = black_box(&a * 2.5);
                });
            },
        );

        // In-place scaling
        group.bench_with_input(
            BenchmarkId::new("inplace_scale", size),
            size,
            |bench_, _data| {
                bench_.iter(|| {
                    let mut a_copy = a.clone();
                    let _ = black_box(inplace_scale(&mut a_copy.view_mut(), 2.5));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_transpose_optimizations(c: &mut Criterion) {
    let sizes = [128, 256, 512, 1024, 2048];
    let mut group = c.benchmark_group("transpose_optimizations");

    for size in &sizes {
        let a = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });

        // Standard transpose
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(a.t().to_owned());
            });
        });

        // Optimized transpose
        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            size,
            |bench_, _data| {
                bench_.iter(|| {
                    let _result =
                        black_box(optimized_transpose(&a.view()).expect("Operation failed"));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_parallel_vs_serial(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    let mut group = c.benchmark_group("parallel_vs_serial");
    group.sample_size(10);

    for size in &sizes {
        let a = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });
        let b = Array2::from_shape_fn((*size, *size), |_| {
            let mut rng = thread_rng();
            let uniform_dist = Uniform::new(-1.0, 1.0).expect("Operation failed");
            rng.sample(uniform_dist)
        });

        // Serial execution
        let config_serial = OptConfig::default()
            .with_blocksize(64)
            .with_algorithm(OptAlgorithm::Blocked);

        group.bench_with_input(BenchmarkId::new("serial", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(
                    blocked_matmul(&a.view(), &b.view(), &config_serial).expect("Operation failed"),
                );
            });
        });

        // Parallel execution
        let config_parallel = OptConfig::default()
            .with_blocksize(64)
            .with_parallel_threshold(0) // Always use parallel
            .with_algorithm(OptAlgorithm::Blocked);

        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench_, _data| {
            bench_.iter(|| {
                let _result = black_box(
                    blocked_matmul(&a.view(), &b.view(), &config_parallel)
                        .expect("Operation failed"),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_optimizations,
    bench_inplace_operations,
    bench_transpose_optimizations,
    bench_parallel_vs_serial
);

criterion_main!(benches);
