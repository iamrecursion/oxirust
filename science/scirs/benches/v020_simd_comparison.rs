//! SciRS2 v0.2.0 SIMD vs Non-SIMD Performance Comparison
//!
//! This benchmark suite compares SIMD-optimized implementations against
//! naive scalar implementations to quantify the performance benefits of
//! vectorization.
//!
//! Performance Targets:
//! - Dot product (f32): 8-12x speedup with AVX2, 12-18x with AVX-512
//! - Dot product (f64): 4-6x speedup with AVX2, 6-10x with AVX-512
//! - Matrix multiply: 10-30x speedup for large matrices
//! - Elementwise operations: 4-8x speedup
//!
//! This provides concrete evidence for the value of SIMD optimizations
//! in scientific computing workloads.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::hint::black_box;
use std::time::Duration;

// =============================================================================
// Naive Scalar Implementations (Baseline)
// =============================================================================

/// Naive scalar dot product for f32
fn naive_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Naive scalar dot product for f64
fn naive_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Naive scalar matrix multiply (ijk order)
fn naive_matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Naive scalar matrix multiply (ikj order - cache friendly)
fn cache_friendly_matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.fill(0.0);
    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_val * b[l * n + j];
            }
        }
    }
}

/// Naive elementwise addition
fn naive_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// Naive elementwise multiplication
fn naive_mul_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }
}

// =============================================================================
// Unrolled Scalar Implementations (Advanced Baseline)
// =============================================================================

/// Unrolled dot product for f32 (8-way unroll)
fn unrolled_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
        sum4 += a[base + 4] * b[base + 4];
        sum5 += a[base + 5] * b[base + 5];
        sum6 += a[base + 6] * b[base + 6];
        sum7 += a[base + 7] * b[base + 7];
    }

    let mut result = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    for i in (n - remainder)..n {
        result += a[i] * b[i];
    }

    result
}

/// Unrolled dot product for f64 (4-way unroll)
fn unrolled_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
    }

    let mut result = sum0 + sum1 + sum2 + sum3;

    for i in (n - remainder)..n {
        result += a[i] * b[i];
    }

    result
}

// =============================================================================
// Test Data Generation
// =============================================================================

fn generate_f32_data(size: usize, seed_offset: usize) -> Vec<f32> {
    (0..size)
        .map(|i| ((i + seed_offset) as f32 * 0.123456).sin() * 100.0)
        .collect()
}

fn generate_f64_data(size: usize, seed_offset: usize) -> Vec<f64> {
    (0..size)
        .map(|i| ((i + seed_offset) as f64 * 0.123456789).sin() * 100.0)
        .collect()
}

// =============================================================================
// Benchmark: Dot Product (f32)
// =============================================================================

fn bench_dot_product_f32(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f32;

    let mut group = c.benchmark_group("simd_vs_scalar/dot_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for &size in &sizes {
        let a = generate_f32_data(size, 0);
        let b = generate_f32_data(size, 1);

        group.throughput(Throughput::Elements(size as u64));

        // Naive scalar
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
            bencher.iter(|| naive_dot_f32(black_box(&a), black_box(&b)))
        });

        // Unrolled scalar
        group.bench_with_input(BenchmarkId::new("unrolled", size), &size, |bencher, _| {
            bencher.iter(|| unrolled_dot_f32(black_box(&a), black_box(&b)))
        });

        // SIMD optimized
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// Benchmark: Dot Product (f64)
// =============================================================================

fn bench_dot_product_f64(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f64;

    let mut group = c.benchmark_group("simd_vs_scalar/dot_f64");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 524288];

    for &size in &sizes {
        let a = generate_f64_data(size, 0);
        let b = generate_f64_data(size, 1);

        group.throughput(Throughput::Elements(size as u64));

        // Naive scalar
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
            bencher.iter(|| naive_dot_f64(black_box(&a), black_box(&b)))
        });

        // Unrolled scalar
        group.bench_with_input(BenchmarkId::new("unrolled", size), &size, |bencher, _| {
            bencher.iter(|| unrolled_dot_f64(black_box(&a), black_box(&b)))
        });

        // SIMD optimized
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f64(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// Benchmark: Matrix Multiplication
// =============================================================================

fn bench_matrix_multiply(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_matrix_multiply_f32;

    let mut group = c.benchmark_group("simd_vs_scalar/matmul_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [32, 64, 128, 256, 512];

    for &size in &sizes {
        let m = size;
        let k = size;
        let n = size;

        let a = generate_f32_data(m * k, 0);
        let b = generate_f32_data(k * n, 1);

        let flops = 2 * m * n * k;
        group.throughput(Throughput::Elements(flops as u64));

        // Naive (only for small sizes)
        if size <= 128 {
            group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
                bencher.iter(|| {
                    let mut c = vec![0.0f32; m * n];
                    naive_matmul_f32(m, k, n, &a, &b, &mut c);
                    black_box(c)
                })
            });
        }

        // Cache-friendly (for medium sizes)
        if size <= 256 {
            group.bench_with_input(
                BenchmarkId::new("cache_friendly", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut c = vec![0.0f32; m * n];
                        cache_friendly_matmul_f32(m, k, n, &a, &b, &mut c);
                        black_box(c)
                    })
                },
            );
        }

        // SIMD blocked
        group.bench_with_input(
            BenchmarkId::new("simd_blocked", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let mut c = vec![0.0f32; m * n];
                    simd_matrix_multiply_f32(m, k, n, 1.0, &a, &b, 0.0, &mut c);
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Elementwise Operations
// =============================================================================

fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar/elementwise");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [1024, 4096, 16384, 65536, 262144];

    for &size in &sizes {
        let a = generate_f32_data(size, 0);
        let b = generate_f32_data(size, 1);

        group.throughput(Throughput::Elements(size as u64));

        // Naive addition
        group.bench_with_input(BenchmarkId::new("add_naive", size), &size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f32; size];
                naive_add_f32(&a, &b, &mut c);
                black_box(c)
            })
        });

        // SIMD addition (using slice operations which may be auto-vectorized)
        group.bench_with_input(
            BenchmarkId::new("add_optimized", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                    black_box(c)
                })
            },
        );

        // Naive multiplication
        group.bench_with_input(BenchmarkId::new("mul_naive", size), &size, |bencher, _| {
            bencher.iter(|| {
                let mut c = vec![0.0f32; size];
                naive_mul_f32(&a, &b, &mut c);
                black_box(c)
            })
        });

        // SIMD multiplication
        group.bench_with_input(
            BenchmarkId::new("mul_optimized", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Cross-Cache-Level Performance
// =============================================================================

fn bench_cache_levels(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f32;

    let mut group = c.benchmark_group("simd_vs_scalar/cache_levels");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Sizes targeting different cache levels
    // L1: 32KB, L2: 256KB, L3: 8MB (typical)
    let configs = [
        (16, "L1_tiny"),         // 128 bytes
        (1024, "L1_fit"),        // 8 KB
        (8192, "L2_fit"),        // 64 KB
        (32768, "L2_exceed"),    // 256 KB
        (131072, "L3_fit"),      // 1 MB
        (524288, "L3_exceed"),   // 4 MB
        (2097152, "DRAM_bound"), // 16 MB
    ];

    for &(size, label) in &configs {
        let a = generate_f32_data(size, 0);
        let b = generate_f32_data(size, 1);

        group.throughput(Throughput::Bytes((size * 8) as u64)); // 2 arrays * 4 bytes

        // Naive
        group.bench_with_input(BenchmarkId::new("naive", label), label, |bencher, _| {
            bencher.iter(|| naive_dot_f32(black_box(&a), black_box(&b)))
        });

        // SIMD
        group.bench_with_input(BenchmarkId::new("simd", label), label, |bencher, _| {
            bencher.iter(|| simd_dot_product_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// Performance Analysis Summary
// =============================================================================

fn bench_speedup_summary(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::{simd_dot_product_f32, simd_dot_product_f64};

    let mut group = c.benchmark_group("simd_speedup_summary");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Representative size for speedup measurement
    let size = 65536;

    let a_f32 = generate_f32_data(size, 0);
    let b_f32 = generate_f32_data(size, 1);
    let a_f64 = generate_f64_data(size, 0);
    let b_f64 = generate_f64_data(size, 1);

    group.throughput(Throughput::Elements(size as u64));

    // f32 comparisons
    group.bench_function("f32_naive", |b| {
        b.iter(|| naive_dot_f32(black_box(&a_f32), black_box(&b_f32)))
    });

    group.bench_function("f32_unrolled", |b| {
        b.iter(|| unrolled_dot_f32(black_box(&a_f32), black_box(&b_f32)))
    });

    group.bench_function("f32_simd", |b| {
        b.iter(|| simd_dot_product_f32(black_box(&a_f32), black_box(&b_f32)))
    });

    // f64 comparisons
    group.bench_function("f64_naive", |b| {
        b.iter(|| naive_dot_f64(black_box(&a_f64), black_box(&b_f64)))
    });

    group.bench_function("f64_unrolled", |b| {
        b.iter(|| unrolled_dot_f64(black_box(&a_f64), black_box(&b_f64)))
    });

    group.bench_function("f64_simd", |b| {
        b.iter(|| simd_dot_product_f64(black_box(&a_f64), black_box(&b_f64)))
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    dot_product_benchmarks,
    bench_dot_product_f32,
    bench_dot_product_f64,
);

criterion_group!(matmul_benchmarks, bench_matrix_multiply,);

criterion_group!(elementwise_benchmarks, bench_elementwise_ops,);

criterion_group!(cache_benchmarks, bench_cache_levels,);

criterion_group!(summary_benchmarks, bench_speedup_summary,);

criterion_main!(
    dot_product_benchmarks,
    matmul_benchmarks,
    elementwise_benchmarks,
    cache_benchmarks,
    summary_benchmarks,
);
