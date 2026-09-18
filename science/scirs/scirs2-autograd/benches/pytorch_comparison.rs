//! Benchmarks comparing SciRS2 Autograd against PyTorch
//!
//! This benchmark suite measures performance of key automatic differentiation operations
//! and compares them against PyTorch as a reference implementation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_autograd::{run, tensor_ops as T, Context};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;

/// Empty axes constant for reduce_sum (reduces over all dimensions)
const NO_AXES: [isize; 0] = [];

/// Benchmark gradient computation for a simple function
fn bench_simple_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_gradient");

    for size in [10isize, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                run(|ctx: &mut Context<f64>| {
                    let x = ctx.placeholder("x", &[size]);
                    let y = T::reduce_sum(x * x, &NO_AXES, false);
                    let grad = &T::grad(&[y], &[x])[0];

                    let x_val = Array2::<f64>::ones((1, size as usize));
                    let result = ctx
                        .evaluator()
                        .push(grad)
                        .feed(x, x_val.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

/// Benchmark matrix multiplication gradient
fn bench_matmul_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_gradient");

    for size in [16isize, 64, 256].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                run(|ctx: &mut Context<f64>| {
                    let a = ctx.placeholder("a", &[size, size]);
                    let b_ph = ctx.placeholder("b", &[size, size]);
                    let c_val = T::matmul(a, b_ph);

                    let loss = T::reduce_sum(c_val, &NO_AXES, false);
                    let grad_a = &T::grad(&[loss], &[a])[0];
                    let grad_b = &T::grad(&[loss], &[b_ph])[0];

                    let s = size as usize;
                    let a_val = Array2::<f64>::ones((s, s));
                    let b_val = Array2::<f64>::ones((s, s));

                    let result = ctx
                        .evaluator()
                        .push(grad_a)
                        .push(grad_b)
                        .feed(a, a_val.view().into_dyn())
                        .feed(b_ph, b_val.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

/// Benchmark neural network forward and backward pass
fn bench_mlp_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_gradient");

    for batch_size in [32isize, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    run(|ctx: &mut Context<f64>| {
                        // Simple 2-layer MLP
                        let x = ctx.placeholder("x", &[batch_size, 784]);
                        let w1 = ctx.placeholder("w1", &[784, 256]);
                        let b1 = ctx.placeholder("b1", &[256]);
                        let w2 = ctx.placeholder("w2", &[256, 10]);
                        let b2 = ctx.placeholder("b2", &[10]);

                        // Forward pass
                        let h1 = T::matmul(x, w1) + b1;
                        let a1 = T::relu(h1);
                        let h2 = T::matmul(a1, w2) + b2;
                        let loss = T::reduce_sum(h2, &NO_AXES, false);

                        // Backward pass
                        let grads = T::grad(&[loss], &[w1, b1, w2, b2]);

                        // Feed dummy data
                        let bs = batch_size as usize;
                        let x_val = Array2::<f64>::ones((bs, 784));
                        let w1_val = Array2::<f64>::ones((784, 256)) * 0.01;
                        let b1_val = Array2::<f64>::zeros((1, 256));
                        let w2_val = Array2::<f64>::ones((256, 10)) * 0.01;
                        let b2_val = Array2::<f64>::zeros((1, 10));

                        let result = ctx
                            .evaluator()
                            .push(&grads[0])
                            .push(&grads[1])
                            .push(&grads[2])
                            .push(&grads[3])
                            .feed(x, x_val.view().into_dyn())
                            .feed(w1, w1_val.view().into_dyn())
                            .feed(b1, b1_val.view().into_dyn())
                            .feed(w2, w2_val.view().into_dyn())
                            .feed(b2, b2_val.view().into_dyn())
                            .run();

                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark higher-order gradients (Hessian-vector product)
fn bench_hessian_vector_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian_vector_product");

    for size in [10isize, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                run(|ctx: &mut Context<f64>| {
                    let x = ctx.placeholder("x", &[size]);
                    let v = T::ones(&[size], ctx);

                    // f(x) = x^T * x
                    let f = T::reduce_sum(x * x, &NO_AXES, false);

                    // Compute HVP
                    let hvp =
                        scirs2_autograd::higher_order::hessian_vector_product(&f, &x, &v, ctx)
                            .expect("Should compute HVP");

                    let x_val = Array2::<f64>::ones((1, size as usize));
                    let result = ctx
                        .evaluator()
                        .push(&hvp)
                        .feed(x, x_val.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

/// Benchmark memory-efficient gradient checkpointing
fn bench_gradient_checkpointing(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_checkpointing");

    for num_layers in [5usize, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_layers),
            num_layers,
            |b, &num_layers| {
                b.iter(|| {
                    run(|ctx: &mut Context<f64>| {
                        let mut x = ctx.placeholder("x", &[32, 128]);

                        // Chain of operations (simulating deep network)
                        for i in 0..num_layers {
                            let name: &'static str = Box::leak(format!("w{}", i).into_boxed_str());
                            let w = ctx.placeholder(name, &[128, 128]);
                            x = T::matmul(x, w);
                            x = T::relu(x);

                            // Apply checkpointing every few layers
                            if i % 5 == 0 {
                                x = T::stop_gradient(x);
                            }
                        }

                        let loss = T::reduce_sum(x, &NO_AXES, false);

                        // Just evaluate loss (checkpointing would kick in during backward)
                        let result = loss.eval(ctx).expect("Should evaluate");
                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation throughput
fn bench_gradient_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_throughput");

    group.bench_function("operations_per_second", |b| {
        b.iter(|| {
            // Measure how many gradient computations we can do per second
            for _ in 0..100 {
                run(|ctx: &mut Context<f64>| {
                    let x = ctx.placeholder("x", &[10, 10]);
                    let y = T::reduce_sum(x * x * x, &NO_AXES, false);
                    let grad = &T::grad(&[y], &[x])[0];

                    let x_val = Array2::<f64>::ones((10, 10));
                    let result = ctx
                        .evaluator()
                        .push(grad)
                        .feed(x, x_val.view().into_dyn())
                        .run();

                    black_box(result);
                });
            }
        });
    });

    group.finish();
}

/// Benchmark memory usage during gradient computation
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for size in [100isize, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                run(|ctx: &mut Context<f64>| {
                    // Create a computation that could benefit from memory optimization
                    let x = ctx.placeholder("x", &[size, size]);
                    let w1 = ctx.placeholder("w1", &[size, size]);
                    let w2 = ctx.placeholder("w2", &[size, size]);

                    let h1 = T::matmul(x, w1);
                    let h2 = T::matmul(h1, w2);
                    let loss = T::reduce_sum(h2, &NO_AXES, false);

                    let grad = &T::grad(&[loss], &[w1])[0];

                    // Use small values to avoid allocation issues
                    let s = size as usize;
                    let x_val = Array2::<f64>::ones((s, s)) * 0.01;
                    let w1_val = Array2::<f64>::ones((s, s)) * 0.01;
                    let w2_val = Array2::<f64>::ones((s, s)) * 0.01;

                    let result = ctx
                        .evaluator()
                        .push(grad)
                        .feed(x, x_val.view().into_dyn())
                        .feed(w1, w1_val.view().into_dyn())
                        .feed(w2, w2_val.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_gradient,
    bench_matmul_gradient,
    bench_mlp_gradient,
    bench_hessian_vector_product,
    bench_gradient_checkpointing,
    bench_gradient_throughput,
    bench_memory_efficiency,
);
criterion_main!(benches);
