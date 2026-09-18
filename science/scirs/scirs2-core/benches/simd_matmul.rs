//! SIMD Matrix Multiplication Benchmark
//!
//! Benchmarks the performance of SIMD-accelerated matrix multiplication
//! comparing naive implementation against blocked GEMM for various matrix sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::simd_ops::{
    simd_dot_product_f32, simd_dot_product_f64, simd_matrix_multiply_f32, simd_matrix_multiply_f64,
};
use std::hint::black_box;

/// Naive matrix multiplication for baseline comparison
fn naive_matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Naive matrix multiplication for f32
fn naive_matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn bench_matmul_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f64");

    for size in [64, 128, 256, 512, 1024].iter() {
        let m = *size;
        let k = *size;
        let n = *size;

        let a = vec![1.0f64; m * k];
        let b = vec![1.0f64; k * n];

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f64; m * n];
                naive_matmul_f64(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c),
                );
                c
            });
        });

        // Benchmark SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f64; m * n];
                simd_matrix_multiply_f64(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    1.0,
                    black_box(&a),
                    black_box(&b),
                    0.0,
                    black_box(&mut c),
                );
                c
            });
        });
    }

    group.finish();
}

fn bench_matmul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    for size in [64, 128, 256, 512, 1024].iter() {
        let m = *size;
        let k = *size;
        let n = *size;

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f32; m * n];
                naive_matmul_f32(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c),
                );
                c
            });
        });

        // Benchmark SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f32; m * n];
                simd_matrix_multiply_f32(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    1.0,
                    black_box(&a),
                    black_box(&b),
                    0.0,
                    black_box(&mut c),
                );
                c
            });
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [64, 128, 256, 512, 1024, 4096].iter() {
        let n = *size;

        // f64 benchmarks
        let a_f64 = vec![1.0f64; n];
        let b_f64 = vec![2.0f64; n];

        group.bench_with_input(BenchmarkId::new("f64_naive", size), size, |bencher, _| {
            bencher.iter(|| {
                let sum: f64 = black_box(&a_f64)
                    .iter()
                    .zip(black_box(&b_f64).iter())
                    .map(|(x, y)| x * y)
                    .sum();
                sum
            });
        });

        group.bench_with_input(BenchmarkId::new("f64_simd", size), size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f64(black_box(&a_f64), black_box(&b_f64)));
        });

        // f32 benchmarks
        let a_f32 = vec![1.0f32; n];
        let b_f32 = vec![2.0f32; n];

        group.bench_with_input(BenchmarkId::new("f32_naive", size), size, |bencher, _| {
            bencher.iter(|| {
                let sum: f32 = black_box(&a_f32)
                    .iter()
                    .zip(black_box(&b_f32).iter())
                    .map(|(x, y)| x * y)
                    .sum();
                sum
            });
        });

        group.bench_with_input(BenchmarkId::new("f32_simd", size), size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f32(black_box(&a_f32), black_box(&b_f32)));
        });
    }

    group.finish();
}

fn bench_rectangular_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("rectangular_matmul");

    // Test non-square matrices
    let test_cases = vec![
        (128, 256, 128), // Tall
        (256, 128, 128), // Wide
        (128, 128, 256), // Deep result
        (512, 256, 128), // Mixed
    ];

    for (m, k, n) in test_cases {
        let label = format!("{}x{}x{}", m, k, n);
        let a = vec![1.0f64; m * k];
        let b = vec![1.0f64; k * n];

        group.bench_with_input(BenchmarkId::new("naive", &label), &label, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f64; m * n];
                naive_matmul_f64(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c),
                );
                c
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", &label), &label, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f64; m * n];
                simd_matrix_multiply_f64(
                    black_box(m),
                    black_box(k),
                    black_box(n),
                    1.0,
                    black_box(&a),
                    black_box(&b),
                    0.0,
                    black_box(&mut c),
                );
                c
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_f64,
    bench_matmul_f32,
    bench_dot_product,
    bench_rectangular_matmul
);
criterion_main!(benches);
