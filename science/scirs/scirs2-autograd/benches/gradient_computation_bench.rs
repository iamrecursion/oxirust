//! Gradient Computation Scaling Benchmarks
//!
//! This benchmark suite measures gradient computation performance across different
//! network architectures and scales. Total: 30 benchmarks
//!
//! Categories:
//! - Simple gradients: 10 benchmarks
//! - Deep network gradients: 10 benchmarks
//! - Wide network gradients: 5 benchmarks
//! - Complex DAG gradients: 5 benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::{arr0, Array2};
use std::hint::black_box;

/// Simple gradient benchmarks (10 benchmarks)
fn simple_gradients_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_gradients");

    // Benchmark 1-5: Polynomial gradients of increasing degree
    for degree in [2, 3, 4, 5, 6].iter() {
        group.bench_with_input(
            BenchmarkId::new("polynomial_grad", degree),
            degree,
            |b, &degree| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let x = ctx.placeholder("x", &[]);
                        let mut y = x;
                        for _ in 0..degree {
                            y = y * x;
                        }
                        let grad = T::grad(&[y], &[x]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    // Benchmark 6-10: Matrix multiplication gradients with different sizes
    for size in [10, 20, 30, 40, 50].iter() {
        group.bench_with_input(BenchmarkId::new("matmul_grad", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = ctx.placeholder("x", &[size as isize, size as isize]);
                    let w = ctx.placeholder("w", &[size as isize, size as isize]);
                    let y = T::matmul(x, w);
                    let grad = T::grad(&[y], &[x, w]);
                    black_box(grad);
                });
            });
        });
    }

    group.finish();
}

/// Deep network gradient benchmarks (10 benchmarks)
fn deep_network_gradients_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_network_gradients");

    // Benchmark 11-20: Deep linear networks with increasing depth
    for depth in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("deep_linear", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let input_size = 64;
                        let hidden_size = 32;

                        let x = ctx.placeholder("x", &[1, input_size]);
                        let mut current = x;

                        // Create deep network
                        for i in 0..depth {
                            let w = ctx.placeholder(
                                "w",
                                &[if i == 0 { input_size } else { hidden_size }, hidden_size],
                            );
                            current = T::matmul(current, w);
                            current = T::relu(current);
                        }

                        // Compute gradient
                        let params: Vec<_> = (0..depth)
                            .map(|i| ctx.placeholder("w", &[hidden_size, hidden_size]))
                            .collect();
                        let grad = T::grad(&[current], &params);
                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Wide network gradient benchmarks (5 benchmarks)
fn wide_network_gradients_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide_network_gradients");

    // Benchmark 21-25: Wide networks with increasing width
    for width in [64, 128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("wide_network", width),
            width,
            |b, &width| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let input_size = 64;
                        let depth = 3;

                        let x = ctx.placeholder("x", &[1, input_size]);
                        let mut current = x;

                        // Create wide network
                        for i in 0..depth {
                            let w = ctx.placeholder(
                                "w",
                                &[if i == 0 { input_size } else { width }, width],
                            );
                            current = T::matmul(current, w);
                            current = T::relu(current);
                        }

                        // Compute gradient (use same placeholder w)
                        let params: Vec<_> = (0..depth)
                            .map(|_i| ctx.placeholder("w", &[width, width]))
                            .collect();
                        let grad = T::grad(&[current], &params);
                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Complex DAG gradient benchmarks (5 benchmarks)
fn complex_dag_gradients_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_dag_gradients");

    // Benchmark 26: Simple branching DAG
    group.bench_function("dag_simple_branch", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[10, 10]);
                let w1 = ctx.placeholder("w1", &[10, 10]);
                let w2 = ctx.placeholder("w2", &[10, 10]);

                // Branch 1
                let y1 = T::matmul(x, w1);
                // Branch 2
                let y2 = T::matmul(x, w2);
                // Merge
                let z = y1 + y2;

                let grad = T::grad(&[z], &[x, w1, w2]);
                black_box(grad);
            });
        });
    });

    // Benchmark 27: Multi-branch DAG
    group.bench_function("dag_multi_branch", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[20, 20]);
                let w1 = ctx.placeholder("w1", &[20, 20]);
                let w2 = ctx.placeholder("w2", &[20, 20]);
                let w3 = ctx.placeholder("w3", &[20, 20]);
                let w4 = ctx.placeholder("w4", &[20, 20]);

                // Multiple branches
                let y1 = T::matmul(x, w1);
                let y2 = T::matmul(x, w2);
                let y3 = T::matmul(x, w3);
                let y4 = T::matmul(x, w4);

                // Merge branches
                let z = y1 + y2 + y3 + y4;

                let grad = T::grad(&[z], &[x, w1, w2, w3, w4]);
                black_box(grad);
            });
        });
    });

    // Benchmark 28: Nested DAG
    group.bench_function("dag_nested", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[15, 15]);
                let w1 = ctx.placeholder("w1", &[15, 15]);
                let w2 = ctx.placeholder("w2", &[15, 15]);
                let w3 = ctx.placeholder("w3", &[15, 15]);

                // Nested structure
                let y1 = T::matmul(x, w1);
                let y2 = T::matmul(y1, w2);
                let y3 = T::matmul(x, w3);
                let z = y2 + y3;

                let grad = T::grad(&[z], &[x, w1, w2, w3]);
                black_box(grad);
            });
        });
    });

    // Benchmark 29: Diamond DAG
    group.bench_function("dag_diamond", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[25, 25]);
                let w1 = ctx.placeholder("w1", &[25, 25]);
                let w2 = ctx.placeholder("w2", &[25, 25]);
                let w3 = ctx.placeholder("w3", &[25, 25]);

                // Diamond pattern: x -> y1, y2 -> z
                let y1 = T::matmul(x, w1);
                let y2 = T::matmul(x, w2);
                let z1 = T::matmul(y1, w3);
                let z2 = T::matmul(y2, w3);
                let output = z1 + z2;

                let grad = T::grad(&[output], &[x, w1, w2, w3]);
                black_box(grad);
            });
        });
    });

    // Benchmark 30: Complex multi-output DAG
    group.bench_function("dag_multi_output", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[30, 30]);
                let w1 = ctx.placeholder("w1", &[30, 30]);
                let w2 = ctx.placeholder("w2", &[30, 30]);
                let w3 = ctx.placeholder("w3", &[30, 30]);

                let y1 = T::matmul(x, w1);
                let y2 = T::matmul(y1, w2);
                let y3 = T::matmul(y2, w3);

                // Multiple outputs
                let out1 = y1;
                let out2 = y2;
                let out3 = y3;

                let grad = T::grad(&[out1, out2, out3], &[x, w1, w2, w3]);
                black_box(grad);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    simple_gradients_benchmark,
    deep_network_gradients_benchmark,
    wide_network_gradients_benchmark,
    complex_dag_gradients_benchmark
);
criterion_main!(benches);
