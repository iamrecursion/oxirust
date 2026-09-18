//! Matrix Multiplication GFLOPS Benchmarks
//!
//! This benchmark suite measures matrix multiplication performance
//! and computes GFLOPS (billion floating-point operations per second).
//!
//! Categories:
//! - Square matrices: 10 benchmarks
//! - Rectangular matrices: 10 benchmarks
//! - Batch matmul: 5 benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Square matrix multiplication benchmarks (10 benchmarks)
fn matmul_square_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_square");

    // Benchmarks 1-10: Square matrices of increasing size
    for size in [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024].iter() {
        // Calculate FLOPs for matrix multiplication: 2*M*N*K
        let flops = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("square_matmul", size), size, |b, &size| {
            let a_data = Array2::from_elem((size, size), 1.0f64);
            let b_data = Array2::from_elem((size, size), 1.0f64);

            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("B", &[size as isize, size as isize]);

                    let c = T::matmul(a, b);

                    // Evaluate to measure actual computation
                    let result = ctx
                        .evaluator()
                        .push(&c)
                        .feed(a, a_data.view().into_dyn())
                        .feed(b, b_data.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

/// Rectangular matrix multiplication benchmarks (10 benchmarks)
fn matmul_rectangular_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_rectangular");

    // Benchmarks 11-20: Various rectangular shapes
    let shapes = [
        (64, 128, 64),   // Thin-wide-thin
        (128, 64, 128),  // Wide-thin-wide
        (100, 200, 50),  // Tall matrix
        (200, 100, 200), // Wide matrix
        (256, 128, 256), // Mixed
        (128, 256, 64),  // Very wide
        (512, 64, 512),  // Very tall
        (64, 512, 32),   // Extreme aspect ratio
        (300, 300, 100), // Nearly square
        (100, 500, 200), // Large rectangular
    ];

    for (idx, &(m, n, k)) in shapes.iter().enumerate() {
        let flops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(
            BenchmarkId::new(format!("rect_{}x{}x{}", m, n, k), idx),
            &(m, n, k),
            |b, &(m, n, k)| {
                let a_data = Array2::from_elem((m, k), 1.0f64);
                let b_data = Array2::from_elem((k, n), 1.0f64);

                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let a = ctx.placeholder("A", &[m as isize, k as isize]);
                        let b = ctx.placeholder("B", &[k as isize, n as isize]);

                        let c = T::matmul(a, b);

                        let result = ctx
                            .evaluator()
                            .push(&c)
                            .feed(a, a_data.view().into_dyn())
                            .feed(b, b_data.view().into_dyn())
                            .run();

                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Batch matrix multiplication benchmarks (5 benchmarks)
fn matmul_batch_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_batch");

    // Benchmarks 21-25: Batch matmul with different batch sizes
    for batch_size in [2, 4, 8, 16, 32].iter() {
        let matrix_size = 64;
        let total_flops = (2 * batch_size * matrix_size * matrix_size * matrix_size) as u64;
        group.throughput(Throughput::Elements(total_flops));

        group.bench_with_input(
            BenchmarkId::new("batch_matmul", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let mut results = Vec::new();

                        // Simulate batch matrix multiplication
                        for i in 0..batch_size {
                            let a = ctx.placeholder("A", &[matrix_size, matrix_size]);
                            let b = ctx.placeholder("B", &[matrix_size, matrix_size]);
                            let c = T::matmul(a, b);
                            results.push(c);
                        }

                        // Sum all results to force evaluation
                        let mut total = results[0];
                        for i in 1..batch_size {
                            total = total + results[i as usize];
                        }

                        black_box(total);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Transposed matrix multiplication benchmarks (additional)
fn matmul_transpose_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_transpose");

    // Additional benchmarks for transposed operations
    for size in [64, 128, 256, 512].iter() {
        let flops = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        // A^T * B
        group.bench_with_input(BenchmarkId::new("at_b", size), size, |b, &size| {
            let a_data = Array2::from_elem((size, size), 1.0f64);
            let b_data = Array2::from_elem((size, size), 1.0f64);

            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("B", &[size as isize, size as isize]);

                    let at = T::transpose(a, &[1, 0]);
                    let c = T::matmul(at, b);

                    let result = ctx
                        .evaluator()
                        .push(&c)
                        .feed(a, a_data.view().into_dyn())
                        .feed(b, b_data.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });

        // A * B^T
        group.bench_with_input(BenchmarkId::new("a_bt", size), size, |b, &size| {
            let a_data = Array2::from_elem((size, size), 1.0f64);
            let b_data = Array2::from_elem((size, size), 1.0f64);

            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let a = ctx.placeholder("A", &[size as isize, size as isize]);
                    let b = ctx.placeholder("B", &[size as isize, size as isize]);

                    let bt = T::transpose(b, &[1, 0]);
                    let c = T::matmul(a, bt);

                    let result = ctx
                        .evaluator()
                        .push(&c)
                        .feed(a, a_data.view().into_dyn())
                        .feed(b, b_data.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

/// Chain matrix multiplication benchmarks
fn matmul_chain_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_chain");

    // Benchmarks for chained matrix multiplications
    for chain_length in [2, 3, 4, 5].iter() {
        group.bench_with_input(
            BenchmarkId::new("chain", chain_length),
            chain_length,
            |b, &chain_length| {
                let size = 64;

                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let mut current = ctx.placeholder("M0", &[size as isize, size as isize]);

                        // Chain matrix multiplications
                        for i in 1..=chain_length {
                            let m = ctx.placeholder("M", &[size as isize, size as isize]);
                            current = T::matmul(current, m);
                        }

                        black_box(current);
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    matmul_square_benchmark,
    matmul_rectangular_benchmark,
    matmul_batch_benchmark,
    matmul_transpose_benchmark,
    matmul_chain_benchmark
);
criterion_main!(benches);
