//! Memory Efficiency Benchmarks
//!
//! This benchmark suite measures memory-efficient gradient computation
//! techniques including gradient checkpointing.
//!
//! Categories:
//! - Gradient checkpointing overhead: 10 benchmarks
//! - Memory usage comparison: 10 benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Gradient checkpointing overhead benchmarks (10 benchmarks)
fn gradient_checkpointing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_checkpointing");

    // Benchmarks 1-5: With checkpointing (simulated by re-computation)
    for depth in [10, 20, 30, 40, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("with_checkpointing", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let layer_size = 64;
                        let checkpoint_interval = 5;

                        let x = ctx.placeholder("x", &[1, layer_size]);
                        let mut current = x;

                        // Build deep network with checkpointing
                        for i in 0..depth {
                            let w = ctx.placeholder("w", &[layer_size, layer_size]);
                            current = T::matmul(current, w);
                            current = T::relu(current);

                            // Simulate checkpointing by materializing at intervals
                            if i % checkpoint_interval == 0 {
                                // In a real implementation, this would be a checkpoint operation
                                current = current + T::scalar(0.0, ctx);
                            }
                        }

                        let loss = T::reduce_sum(current, &[0, 1], false);

                        // Compute gradients (will require re-computation for checkpointed segments)
                        let params: Vec<_> = (0..depth)
                            .map(|i| ctx.placeholder("w", &[layer_size, layer_size]))
                            .collect();
                        let grad = T::grad(&[loss], &params);

                        black_box(grad);
                    });
                });
            },
        );
    }

    // Benchmarks 6-10: Without checkpointing
    for depth in [10, 20, 30, 40, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("without_checkpointing", depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let layer_size = 64;
                        let x = ctx.placeholder("x", &[1, layer_size]);
                        let mut current = x;

                        // Build deep network without checkpointing
                        for i in 0..depth {
                            let w = ctx.placeholder("w", &[layer_size, layer_size]);
                            current = T::matmul(current, w);
                            current = T::relu(current);
                        }

                        let loss = T::reduce_sum(current, &[0, 1], false);

                        // Compute gradients (all activations stored in memory)
                        let params: Vec<_> = (0..depth)
                            .map(|i| ctx.placeholder("w", &[layer_size, layer_size]))
                            .collect();
                        let grad = T::grad(&[loss], &params);

                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Memory usage comparison benchmarks (10 benchmarks)
fn memory_usage_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_comparison");

    // Benchmark 11: Small batch vs large batch
    group.bench_function("small_batch_8", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let batch_size = 8;
                let input_size = 512;
                let hidden_size = 256;

                let x = ctx.placeholder("x", &[batch_size, input_size]);
                let w1 = ctx.placeholder("w1", &[input_size, hidden_size]);
                let w2 = ctx.placeholder("w2", &[hidden_size, hidden_size]);
                let w3 = ctx.placeholder("w3", &[hidden_size, input_size]);

                let h1 = T::relu(T::matmul(x, w1));
                let h2 = T::relu(T::matmul(h1, w2));
                let out = T::matmul(h2, w3);

                let loss = T::reduce_sum(out * out, &[0, 1], false);
                let grad = T::grad(&[loss], &[w1, w2, w3]);

                black_box(grad);
            });
        });
    });

    group.bench_function("large_batch_64", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let batch_size = 64;
                let input_size = 512;
                let hidden_size = 256;

                let x = ctx.placeholder("x", &[batch_size, input_size]);
                let w1 = ctx.placeholder("w1", &[input_size, hidden_size]);
                let w2 = ctx.placeholder("w2", &[hidden_size, hidden_size]);
                let w3 = ctx.placeholder("w3", &[hidden_size, input_size]);

                let h1 = T::relu(T::matmul(x, w1));
                let h2 = T::relu(T::matmul(h1, w2));
                let out = T::matmul(h2, w3);

                let loss = T::reduce_sum(out * out, &[0, 1], false);
                let grad = T::grad(&[loss], &[w1, w2, w3]);

                black_box(grad);
            });
        });
    });

    // Benchmark 13-14: In-place operations vs out-of-place
    group.bench_function("inplace_operations", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 1000;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);

                // Chain of operations that could be done in-place
                let y = x + T::scalar(1.0, ctx);
                let z = y * T::scalar(2.0, ctx);
                let w = T::relu(z);

                let loss = T::reduce_sum(w, &[0, 1], false);
                let grad = T::grad(&[loss], &[x]);

                black_box(grad);
            });
        });
    });

    group.bench_function("out_of_place_operations", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 1000;
                let x1 = ctx.placeholder("x1", &[size as isize, size as isize]);
                let x2 = ctx.placeholder("x2", &[size as isize, size as isize]);
                let x3 = ctx.placeholder("x3", &[size as isize, size as isize]);

                // Operations that create new tensors
                let y = x1 + x2;
                let z = y * x3;
                let w = T::relu(z);

                let loss = T::reduce_sum(w, &[0, 1], false);
                let grad = T::grad(&[loss], &[x1, x2, x3]);

                black_box(grad);
            });
        });
    });

    // Benchmark 15-16: Accumulate gradients vs separate gradients
    group.bench_function("accumulated_gradients", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 500;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                // Multiple forward passes
                let y1 = T::matmul(x, w);
                let y2 = T::matmul(y1, w);
                let y3 = T::matmul(y2, w);

                let loss = T::reduce_sum(y1 + y2 + y3, &[0, 1], false);

                // Accumulated gradient computation
                let grad = T::grad(&[loss], &[w]);

                black_box(grad);
            });
        });
    });

    group.bench_function("separate_gradients", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 500;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                // Multiple forward passes with separate gradient computation
                let y1 = T::matmul(x, w);
                let grad1 = T::grad(&[y1], &[w]);

                let y2 = T::matmul(y1, w);
                let grad2 = T::grad(&[y2], &[w]);

                let y3 = T::matmul(y2, w);
                let grad3 = T::grad(&[y3], &[w]);

                black_box((grad1, grad2, grad3));
            });
        });
    });

    // Benchmark 17-18: Shared parameters vs independent parameters
    group.bench_function("shared_parameters", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 256;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]); // Shared weight

                // Use same weight multiple times
                let y1 = T::matmul(x, w);
                let y2 = T::matmul(y1, w);
                let y3 = T::matmul(y2, w);

                let loss = T::reduce_sum(y3, &[0, 1], false);
                let grad = T::grad(&[loss], &[w]);

                black_box(grad);
            });
        });
    });

    group.bench_function("independent_parameters", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 256;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w1 = ctx.placeholder("w1", &[size as isize, size as isize]);
                let w2 = ctx.placeholder("w2", &[size as isize, size as isize]);
                let w3 = ctx.placeholder("w3", &[size as isize, size as isize]);

                // Use different weights
                let y1 = T::matmul(x, w1);
                let y2 = T::matmul(y1, w2);
                let y3 = T::matmul(y2, w3);

                let loss = T::reduce_sum(y3, &[0, 1], false);
                let grad = T::grad(&[loss], &[w1, w2, w3]);

                black_box(grad);
            });
        });
    });

    // Benchmark 19-20: Sparse vs dense gradients
    group.bench_function("sparse_gradient_simulation", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 1000;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                // Simulate sparse gradient with masking
                let mask = T::greater(x, T::scalar(0.5, ctx));
                let masked_x = x * mask;
                let y = T::matmul(masked_x, w);

                let loss = T::reduce_sum(y, &[0, 1], false);
                let grad = T::grad(&[loss], &[w]);

                black_box(grad);
            });
        });
    });

    group.bench_function("dense_gradient", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let size = 1000;
                let x = ctx.placeholder("x", &[size as isize, size as isize]);
                let w = ctx.placeholder("w", &[size as isize, size as isize]);

                // Dense gradient computation
                let y = T::matmul(x, w);

                let loss = T::reduce_sum(y, &[0, 1], false);
                let grad = T::grad(&[loss], &[w]);

                black_box(grad);
            });
        });
    });

    group.finish();
}

/// Gradient accumulation patterns benchmarks
fn gradient_accumulation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_accumulation");

    // Benchmark with different accumulation steps
    for accum_steps in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("accumulation_steps", accum_steps),
            accum_steps,
            |b, &accum_steps| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let batch_size = 16;
                        let size = 256;

                        let w = ctx.placeholder("w", &[size, size]);
                        let mut total_loss = T::scalar(0.0, ctx);

                        // Accumulate gradients over multiple steps
                        for step in 0..accum_steps {
                            let x = ctx.placeholder("x", &[batch_size, size]);
                            let y = T::matmul(x, w);
                            let loss = T::reduce_sum(y, &[0, 1], false);
                            total_loss = total_loss + loss;
                        }

                        // Compute accumulated gradient
                        let grad = T::grad(&[total_loss], &[w]);

                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    gradient_checkpointing_benchmark,
    memory_usage_comparison_benchmark,
    gradient_accumulation_benchmark
);
criterion_main!(benches);
