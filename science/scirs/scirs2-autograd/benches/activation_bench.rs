//! Activation Function Throughput Benchmarks
//!
//! This benchmark suite measures throughput (elements/sec) of various
//! neural network activation functions.
//!
//! Benchmarks: 5 activations × 4 sizes = 20 benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Activation function throughput benchmarks (20 benchmarks)
fn activation_throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_throughput");

    let activations = ["relu", "sigmoid", "tanh", "softmax", "gelu"];
    let sizes = [1000, 10000, 100000, 1000000];

    for activation in activations.iter() {
        for size in sizes.iter() {
            group.throughput(Throughput::Elements(*size as u64));

            group.bench_with_input(
                BenchmarkId::new(*activation, size),
                &(*activation, *size),
                |b, &(activation, size)| {
                    let input_data = Array2::from_elem((1, size), 0.5f64);

                    b.iter(|| {
                        ag::run(|ctx: &mut ag::Context<f64>| {
                            let x = ctx.placeholder("x", &[1, size as isize]);

                            let y = match activation {
                                "relu" => T::relu(x),
                                "sigmoid" => T::sigmoid(x),
                                "tanh" => T::tanh(x),
                                "softmax" => T::softmax(x, 1),
                                "gelu" => {
                                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                                    let sqrt_2_pi =
                                        T::scalar((2.0 / std::f64::consts::PI).sqrt(), ctx);
                                    let c = T::scalar(0.044715, ctx);
                                    let half = T::scalar(0.5, ctx);
                                    let one = T::scalar(1.0, ctx);

                                    let x3 = x * x * x;
                                    let inner = sqrt_2_pi * (x + c * x3);
                                    let tanh_inner = T::tanh(inner);
                                    half * x * (one + tanh_inner)
                                }
                                _ => panic!("Unknown activation"),
                            };

                            let result = ctx
                                .evaluator()
                                .push(&y)
                                .feed(x, input_data.view().into_dyn())
                                .run();

                            black_box(result);
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

/// Activation backward pass benchmarks
fn activation_backward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_backward");

    let activations = ["relu", "sigmoid", "tanh", "softmax", "gelu"];
    let sizes = [1000, 10000, 100000, 1000000];

    for activation in activations.iter() {
        for size in sizes.iter() {
            group.throughput(Throughput::Elements(*size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("{}_grad", activation), size),
                &(*activation, *size),
                |b, &(activation, size)| {
                    let input_data = Array2::from_elem((1, size), 0.5f64);

                    b.iter(|| {
                        ag::run(|ctx: &mut ag::Context<f64>| {
                            let x = ctx.placeholder("x", &[1, size as isize]);

                            let y = match activation {
                                "relu" => T::relu(x),
                                "sigmoid" => T::sigmoid(x),
                                "tanh" => T::tanh(x),
                                "softmax" => T::softmax(x, 1),
                                "gelu" => {
                                    let sqrt_2_pi =
                                        T::scalar((2.0 / std::f64::consts::PI).sqrt(), ctx);
                                    let c = T::scalar(0.044715, ctx);
                                    let half = T::scalar(0.5, ctx);
                                    let one = T::scalar(1.0, ctx);

                                    let x3 = x * x * x;
                                    let inner = sqrt_2_pi * (x + c * x3);
                                    let tanh_inner = T::tanh(inner);
                                    half * x * (one + tanh_inner)
                                }
                                _ => panic!("Unknown activation"),
                            };

                            // Compute gradient
                            let grad = T::grad(&[y], &[x]);

                            let result = ctx
                                .evaluator()
                                .extend(&grad)
                                .feed(x, input_data.view().into_dyn())
                                .run();

                            black_box(result);
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

/// Combined forward and backward pass benchmarks
fn activation_full_pass_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_full_pass");

    let activations = ["relu", "sigmoid", "tanh", "softmax", "gelu"];
    let size = 100000;

    for activation in activations.iter() {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("forward_backward", activation),
            activation,
            |b, &activation| {
                let input_data = Array2::from_elem((1, size), 0.5f64);
                let grad_output_data = Array2::from_elem((1, size), 1.0f64);

                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let x = ctx.placeholder("x", &[1, size as isize]);

                        // Forward pass
                        let y = match activation {
                            "relu" => T::relu(x),
                            "sigmoid" => T::sigmoid(x),
                            "tanh" => T::tanh(x),
                            "softmax" => T::softmax(x, 1),
                            "gelu" => {
                                let sqrt_2_pi = T::scalar((2.0 / std::f64::consts::PI).sqrt(), ctx);
                                let c = T::scalar(0.044715, ctx);
                                let half = T::scalar(0.5, ctx);
                                let one = T::scalar(1.0, ctx);

                                let x3 = x * x * x;
                                let inner = sqrt_2_pi * (x + c * x3);
                                let tanh_inner = T::tanh(inner);
                                half * x * (one + tanh_inner)
                            }
                            _ => panic!("Unknown activation"),
                        };

                        // Backward pass
                        let grad_x = T::grad(&[y], &[x]);

                        // Evaluate both
                        let result = ctx
                            .evaluator()
                            .push(&y)
                            .extend(&grad_x)
                            .feed(x, input_data.view().into_dyn())
                            .run();

                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Comparison of different activation functions on the same input
fn activation_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_comparison");

    let size = 50000;
    group.throughput(Throughput::Elements(size as u64));

    // Benchmark all activations together
    group.bench_function("all_activations", |b| {
        let input_data = Array2::from_elem((1, size), 0.5f64);

        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[1, size as isize]);

                // Apply all activations
                let relu_out = T::relu(x);
                let sigmoid_out = T::sigmoid(x);
                let tanh_out = T::tanh(x);
                let softmax_out = T::softmax(x, 1);

                // Sum outputs to force evaluation
                let total = relu_out + sigmoid_out + tanh_out + softmax_out;

                let result = ctx
                    .evaluator()
                    .push(&total)
                    .feed(x, input_data.view().into_dyn())
                    .run();

                black_box(result);
            });
        });
    });

    group.finish();
}

/// Activation function fusion benchmarks
fn activation_fusion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_fusion");

    let size = 10000;
    group.throughput(Throughput::Elements(size as u64));

    // Linear + ReLU (common pattern)
    group.bench_function("linear_relu", |b| {
        let input_data = Array2::from_elem((1, size), 0.5f64);
        let weight_data = Array2::from_elem((size, size), 0.01f64);

        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[1, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                let linear = T::matmul(x, w);
                let output = T::relu(linear);

                let result = ctx
                    .evaluator()
                    .push(&output)
                    .feed(x, input_data.view().into_dyn())
                    .feed(w, weight_data.view().into_dyn())
                    .run();

                black_box(result);
            });
        });
    });

    // Linear + Sigmoid (common pattern)
    group.bench_function("linear_sigmoid", |b| {
        let input_data = Array2::from_elem((1, size), 0.5f64);
        let weight_data = Array2::from_elem((size, size), 0.01f64);

        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[1, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                let linear = T::matmul(x, w);
                let output = T::sigmoid(linear);

                let result = ctx
                    .evaluator()
                    .push(&output)
                    .feed(x, input_data.view().into_dyn())
                    .feed(w, weight_data.view().into_dyn())
                    .run();

                black_box(result);
            });
        });
    });

    // Linear + Tanh (common pattern)
    group.bench_function("linear_tanh", |b| {
        let input_data = Array2::from_elem((1, size), 0.5f64);
        let weight_data = Array2::from_elem((size, size), 0.01f64);

        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[1, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                let linear = T::matmul(x, w);
                let output = T::tanh(linear);

                let result = ctx
                    .evaluator()
                    .push(&output)
                    .feed(x, input_data.view().into_dyn())
                    .feed(w, weight_data.view().into_dyn())
                    .run();

                black_box(result);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    activation_throughput_benchmark,
    activation_backward_benchmark,
    activation_full_pass_benchmark,
    activation_comparison_benchmark,
    activation_fusion_benchmark
);
criterion_main!(benches);
