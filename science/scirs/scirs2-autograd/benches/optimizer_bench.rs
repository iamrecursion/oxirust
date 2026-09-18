//! Optimizer Update Speed Benchmarks
//!
//! This benchmark suite measures the performance of different optimizers
//! when updating parameters.
//!
//! Categories:
//! - Optimizer update speed: 4 optimizers × 4 sizes = 16 benchmarks
//! - Optimizer comparison: Additional benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Optimizer update speed benchmarks (16 benchmarks)
fn optimizer_update_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_update");

    let optimizers = ["sgd", "adam", "rmsprop", "adagrad"];
    let param_sizes = [100, 1000, 10000, 100000];

    for optimizer in optimizers.iter() {
        for &param_size in param_sizes.iter() {
            group.bench_with_input(
                BenchmarkId::new(*optimizer, param_size),
                &(*optimizer, param_size),
                |b, &(optimizer, param_size)| {
                    b.iter(|| {
                        ag::run(|ctx: &mut ag::Context<f64>| {
                            let params = ctx.placeholder("params", &[param_size]);
                            let grads = ctx.placeholder("grads", &[param_size]);

                            // Simulate optimizer update based on type
                            let updated_params = match optimizer {
                                "sgd" => {
                                    // SGD: params = params - lr * grads
                                    let lr = T::scalar(0.01, ctx);
                                    params - lr * grads
                                }
                                "adam" => {
                                    // Adam: Simplified version
                                    // m = beta1 * m + (1 - beta1) * grad
                                    // v = beta2 * v + (1 - beta2) * grad^2
                                    // params = params - lr * m / (sqrt(v) + eps)
                                    let lr = T::scalar(0.001, ctx);
                                    let beta1 = T::scalar(0.9, ctx);
                                    let beta2 = T::scalar(0.999, ctx);
                                    let eps = T::scalar(1e-8, ctx);
                                    let one = T::scalar(1.0, ctx);

                                    let m = ctx.placeholder("m", &[param_size]);
                                    let v = ctx.placeholder("v", &[param_size]);

                                    let m_new = beta1 * m + (one - beta1) * grads;
                                    let v_new = beta2 * v + (one - beta2) * grads * grads;
                                    let update = lr * m_new / (T::sqrt(v_new) + eps);

                                    params - update
                                }
                                "rmsprop" => {
                                    // RMSprop: params = params - lr * grad / (sqrt(v) + eps)
                                    let lr = T::scalar(0.01, ctx);
                                    let alpha = T::scalar(0.99, ctx);
                                    let eps = T::scalar(1e-8, ctx);
                                    let one = T::scalar(1.0, ctx);

                                    let v = ctx.placeholder("v", &[param_size]);
                                    let v_new = alpha * v + (one - alpha) * grads * grads;
                                    let update = lr * grads / (T::sqrt(v_new) + eps);

                                    params - update
                                }
                                "adagrad" => {
                                    // Adagrad: params = params - lr * grad / (sqrt(sum_grad^2) + eps)
                                    let lr = T::scalar(0.01, ctx);
                                    let eps = T::scalar(1e-8, ctx);

                                    let acc_grad = ctx.placeholder("acc_grad", &[param_size]);
                                    let acc_grad_new = acc_grad + grads * grads;
                                    let update = lr * grads / (T::sqrt(acc_grad_new) + eps);

                                    params - update
                                }
                                _ => panic!("Unknown optimizer"),
                            };

                            black_box(updated_params);
                        });
                    });
                },
            );
        }
    }

    group.finish();
}

/// Optimizer with momentum benchmarks
fn optimizer_momentum_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_momentum");

    let param_size = 10000;

    // SGD with momentum
    group.bench_function("sgd_momentum", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let params = ctx.placeholder("params", &[param_size]);
                let grads = ctx.placeholder("grads", &[param_size]);
                let velocity = ctx.placeholder("velocity", &[param_size]);

                let lr = T::scalar(0.01, ctx);
                let momentum = T::scalar(0.9, ctx);

                // velocity = momentum * velocity + lr * grads
                let velocity_new = momentum * velocity + lr * grads;
                // params = params - velocity_new
                let params_new = params - velocity_new;

                black_box((params_new, velocity_new));
            });
        });
    });

    // Nesterov momentum
    group.bench_function("sgd_nesterov", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let params = ctx.placeholder("params", &[param_size]);
                let grads = ctx.placeholder("grads", &[param_size]);
                let velocity = ctx.placeholder("velocity", &[param_size]);

                let lr = T::scalar(0.01, ctx);
                let momentum = T::scalar(0.9, ctx);

                // velocity_new = momentum * velocity + lr * grads
                let velocity_new = momentum * velocity + lr * grads;
                // params_new = params - (momentum * velocity_new + lr * grads)
                let params_new = params - (momentum * velocity_new + lr * grads);

                black_box((params_new, velocity_new));
            });
        });
    });

    group.finish();
}

/// Learning rate scheduling benchmarks
fn optimizer_lr_schedule_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_lr_schedule");

    let param_size = 10000;

    // Exponential decay
    group.bench_function("lr_exponential_decay", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let params = ctx.placeholder("params", &[param_size]);
                let grads = ctx.placeholder("grads", &[param_size]);
                let step = ctx.placeholder("step", &[]);

                // lr = initial_lr * decay_factor  (simplified for benchmarking)
                let initial_lr = T::scalar(0.1, ctx);
                let decay_factor = T::scalar(0.96, ctx);

                let lr = initial_lr * decay_factor;
                let params_new = params - lr * grads;

                black_box(params_new);
            });
        });
    });

    // Cosine annealing
    group.bench_function("lr_cosine_annealing", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let params = ctx.placeholder("params", &[param_size]);
                let grads = ctx.placeholder("grads", &[param_size]);
                let step = ctx.placeholder("step", &[]);

                // lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / max_steps))
                let min_lr = T::scalar(0.001, ctx);
                let max_lr = T::scalar(0.1, ctx);
                let max_steps = T::scalar(10000.0, ctx);
                let pi = T::scalar(std::f64::consts::PI, ctx);
                let half = T::scalar(0.5, ctx);
                let one = T::scalar(1.0, ctx);

                let cos_val = T::cos(pi * step / max_steps);
                let lr = min_lr + half * (max_lr - min_lr) * (one + cos_val);
                let params_new = params - lr * grads;

                black_box(params_new);
            });
        });
    });

    group.finish();
}

/// Gradient clipping benchmarks
fn optimizer_gradient_clipping_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_gradient_clipping");

    let param_size = 10000;

    // Clip by value
    group.bench_function("clip_by_value", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let grads = ctx.placeholder("grads", &[param_size]);

                // Clipped grads
                let grads_clipped = T::clip(grads, -1.0, 1.0);

                black_box(grads_clipped);
            });
        });
    });

    // Clip by global norm
    group.bench_function("clip_by_global_norm", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let grads = ctx.placeholder("grads", &[param_size]);

                let max_norm = T::scalar(1.0, ctx);

                // Compute global norm
                let grad_norm = T::sqrt(T::reduce_sum(grads * grads, &[0], false));

                // Scale factor
                let scale = T::minimum(max_norm / grad_norm, T::scalar(1.0, ctx));

                // Clipped grads
                let grads_clipped = grads * scale;

                black_box(grads_clipped);
            });
        });
    });

    group.finish();
}

/// Multi-parameter optimizer benchmarks
fn optimizer_multi_param_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_multi_param");

    // Benchmark with multiple parameter groups
    for num_params in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("adam_multi_param", num_params),
            num_params,
            |b, &num_params| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let param_size = 1000;

                        // Create multiple parameter tensors
                        let mut updated_params = Vec::new();

                        for i in 0..num_params {
                            let params = ctx.placeholder("params", &[param_size]);
                            let grads = ctx.placeholder("grads", &[param_size]);
                            let m = ctx.placeholder("m", &[param_size]);
                            let v = ctx.placeholder("v", &[param_size]);

                            // Adam update
                            let lr = T::scalar(0.001, ctx);
                            let beta1 = T::scalar(0.9, ctx);
                            let beta2 = T::scalar(0.999, ctx);
                            let eps = T::scalar(1e-8, ctx);
                            let one = T::scalar(1.0, ctx);

                            let m_new = beta1 * m + (one - beta1) * grads;
                            let v_new = beta2 * v + (one - beta2) * grads * grads;
                            let update = lr * m_new / (T::sqrt(v_new) + eps);
                            let params_new = params - update;

                            updated_params.push(params_new);
                        }

                        black_box(updated_params);
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    optimizer_update_benchmark,
    optimizer_momentum_benchmark,
    optimizer_lr_schedule_benchmark,
    optimizer_gradient_clipping_benchmark,
    optimizer_multi_param_benchmark
);
criterion_main!(benches);
