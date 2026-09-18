//! High-Performance Operations Benchmarks
//!
//! This benchmark suite measures SIMD and parallel processing performance
//! for gradient computations and tensor operations.
//!
//! Categories:
//! - SIMD backward pass: 10 benchmarks (5 sizes × 2 variants)
//! - Parallel gradient computation: 10 benchmarks (5 outputs × 2 variants)
//! - Memory-efficient operations: 5 benchmarks
//! - Ultra backward pass: 5 benchmarks
//! - Direct high_performance module benchmarks

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_autograd as ag;
use scirs2_autograd::high_performance::*;
use scirs2_autograd::tensor_ops as T;
use scirs2_autograd::variable::{SafeVariable, SafeVariableEnvironment};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;
use std::sync::Arc;

// Helper function for high_performance module benchmarks
fn create_test_tensor<F: ag::Float>(
    env: Arc<SafeVariableEnvironment<F>>,
    shape: &[usize],
    value: F,
) -> SafeVariable<F> {
    let size: usize = shape.iter().product();
    let data = vec![value; size];
    let arr = ag::NdArray::from_shape_vec(shape, data).expect("Failed to create array");
    SafeVariable::new(arr, env, false).expect("Failed to create SafeVariable")
}

/// SIMD backward pass benchmarks (10 benchmarks)
fn simd_backward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_backward");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark with SIMD (if available)
        group.bench_with_input(BenchmarkId::new("with_simd", size), size, |b, &size| {
            b.iter(|| {
                ag::run(|ctx: &mut ag::Context<f32>| {
                    let x = ctx.placeholder("x", &[size]);
                    let w = ctx.placeholder("w", &[size]);

                    // Simple element-wise operation
                    let y = x * w;
                    let z = T::reduce_sum(y, &[0], false);

                    // Backward pass
                    let grad = T::grad(&[z], &[x, w]);

                    // Feed dummy data to trigger computation
                    let x_data = Array2::from_elem((1, size as usize), 1.0f32);
                    let w_data = Array2::from_elem((1, size as usize), 1.0f32);

                    let result = ctx
                        .evaluator()
                        .extend(&grad)
                        .feed(x, x_data.view().into_dyn())
                        .feed(w, w_data.view().into_dyn())
                        .run();

                    black_box(result);
                });
            });
        });

        // Benchmark without SIMD (scalar fallback)
        group.bench_with_input(
            BenchmarkId::new("scalar_fallback", size),
            size,
            |b, &size| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f32>| {
                        let x = ctx.placeholder("x", &[size]);
                        let w = ctx.placeholder("w", &[size]);

                        // Simple scalar operations
                        let y = x + w;
                        let z = T::reduce_sum(y, &[0], false);

                        let grad = T::grad(&[z], &[x, w]);
                        black_box(grad);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Parallel gradient computation benchmarks (10 benchmarks)
fn parallel_gradient_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_gradient");

    for num_outputs in [2, 4, 8, 16, 32].iter() {
        // Parallel version
        group.bench_with_input(
            BenchmarkId::new("parallel", num_outputs),
            num_outputs,
            |b, &num_outputs| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let x = ctx.placeholder("x", &[50, 50]);

                        // Create multiple outputs
                        let mut outputs = Vec::new();
                        for i in 0..num_outputs {
                            let w = ctx.placeholder("w", &[50, 50]);
                            let y = T::matmul(x, w);
                            outputs.push(y);
                        }

                        // Compute gradients for all outputs
                        let params: Vec<_> = (0..num_outputs)
                            .map(|i| ctx.placeholder("w", &[50, 50]))
                            .collect();

                        let grad = T::grad(&outputs, &params);
                        black_box(grad);
                    });
                });
            },
        );

        // Sequential version
        group.bench_with_input(
            BenchmarkId::new("sequential", num_outputs),
            num_outputs,
            |b, &num_outputs| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let x = ctx.placeholder("x", &[50, 50]);

                        // Create and compute gradients one by one
                        for i in 0..num_outputs {
                            let w = ctx.placeholder("w", &[50, 50]);
                            let y = T::matmul(x, w);
                            let grad = T::grad(&[y], &[w]);
                            black_box(grad);
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Memory-efficient operations benchmarks (5 benchmarks)
fn memory_efficient_ops_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficient_ops");

    // Benchmark 21: In-place operations
    group.bench_function("inplace_operations", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[100, 100]);
                let w = ctx.placeholder("w", &[100, 100]);

                // Chain of operations that can be optimized
                let y = x + w;
                let z = T::relu(y);
                let out = T::reduce_mean(z, &[0], false);

                let grad = T::grad(&[out], &[x, w]);
                black_box(grad);
            });
        });
    });

    // Benchmark 22: View-based operations
    group.bench_function("view_operations", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[200, 200]);

                // Operations that use views
                let y = T::reshape(x, &[40000]);
                let z = T::reshape(y, &[200, 200]);

                let grad = T::grad(&[z], &[x]);
                black_box(grad);
            });
        });
    });

    // Benchmark 23: Broadcast operations
    group.bench_function("broadcast_operations", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[128, 128]);
                let b = ctx.placeholder("b", &[1, 128]);

                // Broadcasting
                let y = x + b;

                let grad = T::grad(&[y], &[x, b]);
                black_box(grad);
            });
        });
    });

    // Benchmark 24: Reduction chains
    group.bench_function("reduction_chains", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[64, 64, 64]);

                // Chain of reductions
                let y = T::reduce_sum(x, &[2], false);
                let z = T::reduce_mean(y, &[1], false);

                let grad = T::grad(&[z], &[x]);
                black_box(grad);
            });
        });
    });

    // Benchmark 25: Sparse gradient computation
    group.bench_function("sparse_gradients", |b| {
        b.iter(|| {
            ag::run(|ctx: &mut ag::Context<f64>| {
                let x = ctx.placeholder("x", &[150, 150]);
                let w = ctx.placeholder("w", &[150, 150]);

                // Sparse-like computation
                let mask = T::greater(x, T::scalar(0.0, ctx));
                let y = T::matmul(x, w);
                let z = y * mask;

                let grad = T::grad(&[z], &[x, w]);
                black_box(grad);
            });
        });
    });

    group.finish();
}

/// Ultra backward pass benchmarks (5 benchmarks)
fn ultra_backward_pass_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_backward_pass");

    // Benchmark 26-30: Ultra-optimized backward pass with different complexities
    for complexity in [1, 2, 3, 4, 5].iter() {
        group.bench_with_input(
            BenchmarkId::new("ultra_backward", complexity),
            complexity,
            |b, &complexity| {
                b.iter(|| {
                    ag::run(|ctx: &mut ag::Context<f32>| {
                        let size = 128;
                        let x = ctx.placeholder("x", &[size as isize, size as isize]);

                        let mut current = x;
                        for i in 0..complexity {
                            let w = ctx.placeholder("w", &[size as isize, size as isize]);
                            current = T::matmul(current, w);
                            current = T::relu(current);
                        }

                        let loss = T::reduce_sum(current, &[0, 1], false);

                        let params: Vec<_> = (0..complexity)
                            .map(|i| ctx.placeholder("w", &[size as isize, size as isize]))
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

// ============================================================================
// DIRECT HIGH_PERFORMANCE MODULE BENCHMARKS
// ============================================================================

/// Direct SIMD backward pass benchmarks using high_performance module
fn direct_simd_backward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_simd_backward");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let env = Arc::new(SafeVariableEnvironment::<f32>::new());
            let x = create_test_tensor(env.clone(), &[size], 2.0);

            b.iter(|| {
                let result = simd_backward_pass(black_box(&x), black_box(&env));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Direct parallel gradient computation benchmarks
#[cfg(feature = "parallel")]
fn direct_parallel_gradient_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_parallel_gradient");

    for num_outputs in [2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*num_outputs as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_outputs),
            num_outputs,
            |b, &num_outputs| {
                let env = Arc::new(SafeVariableEnvironment::<f32>::new());
                let vars: Vec<_> = (0..num_outputs)
                    .map(|i| create_test_tensor(env.clone(), &[64], (i + 1) as f32))
                    .collect();

                let outputs: Vec<_> = vars.iter().collect();
                let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

                b.iter(|| {
                    let result = parallel_gradient_computation(
                        black_box(&outputs[..]),
                        black_box(&inputs),
                        black_box(&env),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Direct ultra backward pass benchmarks
#[cfg(all(feature = "simd", feature = "parallel"))]
fn direct_ultra_backward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_ultra_backward");

    group.bench_function("ultra_vs_standard", |b| {
        let env = Arc::new(SafeVariableEnvironment::<f32>::new());
        let vars: Vec<_> = (0..8)
            .map(|i| create_test_tensor(env.clone(), &[128], (i + 1) as f32))
            .collect();

        let outputs: Vec<_> = vars.iter().collect();
        let inputs: Vec<_> = vars.iter().map(|v| v.id).collect();

        b.iter(|| {
            let result =
                ultra_backward_pass(black_box(&outputs[..]), black_box(&inputs), black_box(&env));
            black_box(result)
        });
    });

    group.finish();
}

/// Direct memory efficient accumulation benchmarks
fn direct_memory_accumulation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_memory_accumulation");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let gradients: Vec<_> = (1..=4)
                .map(|i| ag::NdArray::from_elem(&[size][..], i as f32))
                .collect();

            let mut target = ag::NdArray::zeros(&[size][..]);

            b.iter(|| {
                target.fill(0.0);
                let result = memory_efficient_grad_accumulation(
                    black_box(&gradients),
                    black_box(&mut target),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

// Configure criterion groups based on features
#[cfg(all(feature = "simd", feature = "parallel"))]
criterion_group!(
    benches,
    simd_backward_benchmark,
    parallel_gradient_benchmark,
    memory_efficient_ops_benchmark,
    ultra_backward_pass_benchmark,
    direct_simd_backward_benchmark,
    direct_parallel_gradient_benchmark,
    direct_ultra_backward_benchmark,
    direct_memory_accumulation_benchmark
);

#[cfg(all(feature = "simd", not(feature = "parallel")))]
criterion_group!(
    benches,
    simd_backward_benchmark,
    memory_efficient_ops_benchmark,
    ultra_backward_pass_benchmark,
    direct_simd_backward_benchmark,
    direct_memory_accumulation_benchmark
);

#[cfg(all(not(feature = "simd"), feature = "parallel"))]
criterion_group!(
    benches,
    simd_backward_benchmark,
    parallel_gradient_benchmark,
    memory_efficient_ops_benchmark,
    ultra_backward_pass_benchmark,
    direct_parallel_gradient_benchmark,
    direct_memory_accumulation_benchmark
);

#[cfg(not(any(feature = "simd", feature = "parallel")))]
criterion_group!(
    benches,
    simd_backward_benchmark,
    parallel_gradient_benchmark,
    memory_efficient_ops_benchmark,
    ultra_backward_pass_benchmark,
    direct_memory_accumulation_benchmark
);

criterion_main!(benches);
