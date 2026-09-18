// Mobile NEON SIMD Benchmarks
// Benchmark suite for ARM NEON optimizations on mobile platforms

#![cfg(any(target_arch = "aarch64", target_arch = "arm"))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::simd::neon::*;
use std::hint::black_box;

// Benchmark vector operations
fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_vector_ops");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let a = vec![1.0f32; *size];
        let b = vec![2.0f32; *size];
        let mut out = vec![0.0f32; *size];

        // Addition
        group.bench_with_input(BenchmarkId::new("add_f32", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_add_f32(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });

        // Multiplication
        group.bench_with_input(BenchmarkId::new("mul_f32", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_mul_f32(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });

        // Dot product
        group.bench_with_input(BenchmarkId::new("dot_f32", size), size, |bencher, _| {
            bencher.iter(|| {
                black_box(neon_dot_f32(black_box(&a), black_box(&b)));
            });
        });
    }

    group.finish();
}

// Benchmark matrix operations
fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_matrix_ops");

    for size in [32, 64, 128, 256].iter() {
        let m = *size;
        let n = *size;
        let k = *size;

        group.throughput(Throughput::Elements((m * n * k) as u64));

        let a = vec![1.0f32; m * k];
        let b = vec![2.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        // GEMM
        group.bench_with_input(BenchmarkId::new("gemm_f32", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_gemm_f32(
                    m,
                    n,
                    k,
                    1.0,
                    black_box(&a),
                    black_box(&b),
                    0.0,
                    black_box(&mut c),
                );
            });
        });

        // GEMV
        let x = vec![1.0f32; n];
        let mut y = vec![0.0f32; m];

        group.bench_with_input(BenchmarkId::new("gemv_f32", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_gemv_f32(
                    m,
                    n,
                    1.0,
                    black_box(&a),
                    black_box(&x),
                    0.0,
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

// Benchmark activation functions
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_activations");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x = (0..*size)
            .map(|i| (i as f32 / *size as f32) - 0.5)
            .collect::<Vec<_>>();
        let mut out = vec![0.0f32; *size];

        // ReLU
        group.bench_with_input(BenchmarkId::new("relu", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_relu_f32(black_box(&x), black_box(&mut out));
            });
        });

        // Sigmoid
        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_sigmoid_f32(black_box(&x), black_box(&mut out));
            });
        });

        // Tanh
        group.bench_with_input(BenchmarkId::new("tanh", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_tanh_f32(black_box(&x), black_box(&mut out));
            });
        });

        // GELU
        group.bench_with_input(BenchmarkId::new("gelu", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_gelu_f32(black_box(&x), black_box(&mut out));
            });
        });

        // Leaky ReLU
        group.bench_with_input(BenchmarkId::new("leaky_relu", size), size, |bencher, _| {
            bencher.iter(|| {
                neon_leaky_relu_f32(black_box(&x), 0.01, black_box(&mut out));
            });
        });
    }

    group.finish();
}

// Benchmark battery-optimized operations
fn bench_battery_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_battery_optimized");

    let size = 4096;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];

    // Performance mode
    let perf_opt = MobileOptimizer::new().with_battery_mode(BatteryMode::Performance);

    group.bench_function("dot_performance", |bencher| {
        bencher.iter(|| {
            black_box(neon_dot_battery_optimized(
                black_box(&a),
                black_box(&b),
                black_box(&perf_opt),
            ));
        });
    });

    // Balanced mode
    let balanced_opt = MobileOptimizer::new().with_battery_mode(BatteryMode::Balanced);

    group.bench_function("dot_balanced", |bencher| {
        bencher.iter(|| {
            black_box(neon_dot_battery_optimized(
                black_box(&a),
                black_box(&b),
                black_box(&balanced_opt),
            ));
        });
    });

    // Power saver mode
    let saver_opt = MobileOptimizer::new().with_battery_mode(BatteryMode::PowerSaver);

    group.bench_function("dot_powersaver", |bencher| {
        bencher.iter(|| {
            black_box(neon_dot_battery_optimized(
                black_box(&a),
                black_box(&b),
                black_box(&saver_opt),
            ));
        });
    });

    group.finish();
}

// Benchmark thermal-aware operations
fn bench_thermal_aware(c: &mut Criterion) {
    let mut group = c.benchmark_group("neon_thermal_aware");

    let m = 128;
    let n = 128;
    let k = 128;

    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Normal thermal state
    group.bench_function("gemm_thermal_normal", |bencher| {
        bencher.iter(|| {
            neon_gemm_thermal_aware(
                m,
                n,
                k,
                1.0,
                black_box(&a),
                black_box(&b),
                0.0,
                black_box(&mut c),
                ThermalState::Normal,
            );
        });
    });

    // Hot thermal state
    group.bench_function("gemm_thermal_hot", |bencher| {
        bencher.iter(|| {
            neon_gemm_thermal_aware(
                m,
                n,
                k,
                1.0,
                black_box(&a),
                black_box(&b),
                0.0,
                black_box(&mut c),
                ThermalState::Hot,
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_operations,
    bench_matrix_operations,
    bench_activations,
    bench_battery_optimized,
    bench_thermal_aware,
);

criterion_main!(benches);
