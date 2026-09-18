//! v0.2.0 Performance Validation Benchmarks
//!
//! Comprehensive benchmark suite for validating v0.2.0 performance improvements:
//! - SIMD dot product operations (f32, f64)
//! - Matrix multiplication with blocked GEMM
//! - AdaptiveChunking memory efficiency
//! - Parallel processing scalability
//!
//! Performance targets:
//! - SIMD operations: 5-15x speedup over naive implementations
//! - Matrix multiply: 10-30x speedup for large matrices
//! - AdaptiveChunking: <10% overhead vs fixed chunking with better memory efficiency
//! - Parallel processing: Near-linear scaling up to 8 cores

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::hint;
use std::hint::black_box;
use std::time::Duration;

// =============================================================================
// SIMD Dot Product Benchmarks
// =============================================================================

/// Generate test data for f32 benchmarks
fn generate_f32_data(size: usize, seed_offset: usize) -> Vec<f32> {
    (0..size)
        .map(|i| ((i + seed_offset) as f32 * 0.123456).sin() * 100.0)
        .collect()
}

/// Generate test data for f64 benchmarks
fn generate_f64_data(size: usize, seed_offset: usize) -> Vec<f64> {
    (0..size)
        .map(|i| ((i + seed_offset) as f64 * 0.123456789).sin() * 100.0)
        .collect()
}

/// Naive dot product implementation for f32 baseline
fn naive_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Naive dot product implementation for f64 baseline
fn naive_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Unrolled dot product for f32 (hand-optimized baseline)
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

/// Benchmark SIMD dot product operations for f32
fn bench_simd_dot_f32(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f32;

    let mut group = c.benchmark_group("simd_dot_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Test various sizes from small to very large
    let sizes: &[usize] = &[
        16,      // Very small - cache-resident
        64,      // Small - fits in L1
        256,     // Medium - fits in L2
        1024,    // Large - L2/L3 boundary
        4096,    // Very large - L3
        16384,   // Huge - exceeds L3
        65536,   // Very huge - memory bound
        262144,  // Extreme - pure memory bandwidth
        1048576, // 1M elements - streaming test
    ];

    for &size in sizes {
        let a = generate_f32_data(size, 0);
        let b = generate_f32_data(size, 1);

        group.throughput(Throughput::Elements(size as u64));

        // Naive baseline
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
            bencher.iter(|| naive_dot_f32(black_box(&a), black_box(&b)))
        });

        // Unrolled baseline
        group.bench_with_input(BenchmarkId::new("unrolled", size), &size, |bencher, _| {
            bencher.iter(|| unrolled_dot_f32(black_box(&a), black_box(&b)))
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark SIMD dot product operations for f64
fn bench_simd_dot_f64(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f64;

    let mut group = c.benchmark_group("simd_dot_f64");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes: &[usize] = &[
        16,     // Very small
        64,     // Small
        256,    // Medium
        1024,   // Large
        4096,   // Very large
        16384,  // Huge
        65536,  // Very huge
        262144, // Extreme
        524288, // 512K elements (f64 is 2x larger than f32)
    ];

    for &size in sizes {
        let a = generate_f64_data(size, 0);
        let b = generate_f64_data(size, 1);

        group.throughput(Throughput::Elements(size as u64));

        // Naive baseline
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
            bencher.iter(|| naive_dot_f64(black_box(&a), black_box(&b)))
        });

        // SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f64(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// Matrix Multiplication Benchmarks
// =============================================================================

/// Naive matrix multiplication for baseline comparison
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

/// Cache-optimized naive matrix multiplication (ijk order changed to ikj)
fn cache_friendly_matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // Zero initialize
    c.fill(0.0);

    // ikj order for better cache behavior
    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_val * b[l * n + j];
            }
        }
    }
}

/// Benchmark matrix multiplication for various sizes
fn bench_matmul_square(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_matrix_multiply_f32;

    let mut group = c.benchmark_group("matmul_square_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    // Square matrix sizes - focus on performance-critical ranges
    let sizes: &[usize] = &[32, 64, 128, 256, 512, 768, 1024];

    for &size in sizes {
        let m = size;
        let k = size;
        let n = size;

        let a = generate_f32_data(m * k, 0);
        let b = generate_f32_data(k * n, 1);

        // FLOPS: 2*M*N*K (multiply + add per element)
        let flops = 2 * m * n * k;
        group.throughput(Throughput::Elements(flops as u64));

        // Naive baseline (only for smaller sizes to avoid timeout)
        if size <= 256 {
            group.bench_with_input(BenchmarkId::new("naive", size), &size, |bencher, _| {
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
                })
            });
        }

        // Cache-friendly baseline
        if size <= 512 {
            group.bench_with_input(
                BenchmarkId::new("cache_friendly", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut c = vec![0.0f32; m * n];
                        cache_friendly_matmul_f32(
                            black_box(m),
                            black_box(k),
                            black_box(n),
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                        );
                        c
                    })
                },
            );
        }

        // SIMD blocked implementation
        group.bench_with_input(
            BenchmarkId::new("simd_blocked", size),
            &size,
            |bencher, _| {
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
                })
            },
        );
    }

    group.finish();
}

/// Benchmark rectangular matrix multiplication
fn bench_matmul_rectangular(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_matrix_multiply_f32;

    let mut group = c.benchmark_group("matmul_rect_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Rectangular matrix configurations (M, K, N)
    let configs: &[(usize, usize, usize)] = &[
        (256, 64, 256),   // Tall A, wide result
        (64, 256, 256),   // Wide A, wide result
        (256, 256, 64),   // Square-ish A, narrow result
        (512, 128, 512),  // Tall-ish A
        (128, 512, 128),  // Wide A, narrow result
        (1024, 256, 128), // Very tall A
        (128, 256, 1024), // Wide result
        (512, 512, 64),   // Narrow result
    ];

    for &(m, k, n) in configs {
        let label = format!("{}x{}x{}", m, k, n);

        let a = generate_f32_data(m * k, 0);
        let b = generate_f32_data(k * n, 1);

        let flops = 2 * m * n * k;
        group.throughput(Throughput::Elements(flops as u64));

        // SIMD blocked implementation
        group.bench_with_input(BenchmarkId::new("simd", &label), &label, |bencher, _| {
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
            })
        });

        // Cache-friendly baseline for comparison
        if m * n * k <= 256 * 256 * 256 {
            group.bench_with_input(
                BenchmarkId::new("cache_friendly", &label),
                &label,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut c = vec![0.0f32; m * n];
                        cache_friendly_matmul_f32(
                            black_box(m),
                            black_box(k),
                            black_box(n),
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                        );
                        c
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark GEMM with alpha/beta scaling
fn bench_gemm_scaling(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_matrix_multiply_f32;

    let mut group = c.benchmark_group("gemm_scaling_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let size = 256;
    let m = size;
    let k = size;
    let n = size;

    let a = generate_f32_data(m * k, 0);
    let b = generate_f32_data(k * n, 1);

    let flops = 2 * m * n * k;
    group.throughput(Throughput::Elements(flops as u64));

    // Standard C = A * B
    group.bench_function("alpha1_beta0", |bencher| {
        bencher.iter(|| {
            let mut c = vec![0.0f32; m * n];
            simd_matrix_multiply_f32(m, k, n, 1.0, &a, &b, 0.0, &mut c);
            c
        })
    });

    // Scaled: C = 2.0 * A * B
    group.bench_function("alpha2_beta0", |bencher| {
        bencher.iter(|| {
            let mut c = vec![0.0f32; m * n];
            simd_matrix_multiply_f32(m, k, n, 2.0, &a, &b, 0.0, &mut c);
            c
        })
    });

    // Update: C = A * B + C
    group.bench_function("alpha1_beta1", |bencher| {
        bencher.iter(|| {
            let mut c = vec![1.0f32; m * n];
            simd_matrix_multiply_f32(m, k, n, 1.0, &a, &b, 1.0, &mut c);
            c
        })
    });

    // Full GEMM: C = 3.0 * A * B + 2.0 * C
    group.bench_function("alpha3_beta2", |bencher| {
        bencher.iter(|| {
            let mut c = vec![1.0f32; m * n];
            simd_matrix_multiply_f32(m, k, n, 3.0, &a, &b, 2.0, &mut c);
            c
        })
    });

    group.finish();
}

// =============================================================================
// AdaptiveChunking Memory Efficiency Benchmarks
// =============================================================================

/// Benchmark fixed vs adaptive chunking overhead
fn bench_chunking_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_overhead");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Data sizes to test
    let sizes: &[usize] = &[
        10_000,    // Small
        100_000,   // Medium
        1_000_000, // Large
    ];

    for &size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Fixed chunk processing (baseline)
        group.bench_with_input(
            BenchmarkId::new("fixed_chunks", size),
            &size,
            |bencher, _| {
                let chunk_size = 8192;
                bencher.iter(|| {
                    let mut sum = 0.0f64;
                    for chunk in data.chunks(chunk_size) {
                        for &val in chunk {
                            sum += val;
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Simulated adaptive chunking (with overhead measurement)
        group.bench_with_input(
            BenchmarkId::new("adaptive_chunks", size),
            &size,
            |bencher, _| {
                // Simulate adaptive chunk size determination
                let initial_chunk_size = 8192usize;
                let mut current_chunk_size = initial_chunk_size;

                bencher.iter(|| {
                    let mut sum = 0.0f64;
                    let mut processed = 0usize;

                    while processed < size {
                        let chunk_end = (processed + current_chunk_size).min(size);
                        let chunk = &data[processed..chunk_end];

                        // Process chunk
                        for &val in chunk {
                            sum += val;
                        }

                        // Simulate adaptive adjustment (minimal overhead)
                        let _chunk_time = std::time::Instant::now();
                        processed = chunk_end;

                        // Adjust chunk size based on "performance" (simulated)
                        if chunk.len() == current_chunk_size {
                            current_chunk_size = (current_chunk_size as f64 * 1.0) as usize;
                        }
                    }

                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory pressure monitoring overhead
fn bench_memory_monitoring_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_monitoring");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let data_size = 100_000usize;
    let data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();
    let chunk_size = 8192usize;

    group.throughput(Throughput::Elements(data_size as u64));

    // Without monitoring
    group.bench_function("no_monitoring", |bencher| {
        bencher.iter(|| {
            let mut sum = 0.0f64;
            for chunk in data.chunks(chunk_size) {
                for &val in chunk {
                    sum += val;
                }
            }
            black_box(sum)
        })
    });

    // With simulated monitoring (overhead measurement)
    group.bench_function("with_monitoring", |bencher| {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let check_counter = AtomicUsize::new(0);

        bencher.iter(|| {
            let mut sum = 0.0f64;
            for chunk in data.chunks(chunk_size) {
                // Simulate periodic memory pressure check
                let count = check_counter.fetch_add(1, Ordering::Relaxed);
                if count.is_multiple_of(10) {
                    // Simulate memory check (minimal operation)
                    hint::black_box(std::alloc::Layout::new::<[u8; 64]>());
                }

                for &val in chunk {
                    sum += val;
                }
            }
            black_box(sum)
        })
    });

    group.finish();
}

// =============================================================================
// Parallel Processing Scalability Benchmarks
// =============================================================================

/// Benchmark parallel reduction operations
#[cfg(feature = "parallel")]
fn bench_parallel_reduction(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_reduction");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    // Large data sizes for parallel processing
    let sizes: &[usize] = &[100_000, 1_000_000, 10_000_000];

    for &size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Sequential baseline
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bencher, _| {
            bencher.iter(|| {
                let sum: f64 = data.iter().sum();
                black_box(sum)
            })
        });

        // Parallel reduction
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bencher, _| {
            bencher.iter(|| {
                let sum: f64 = data.par_iter().sum();
                black_box(sum)
            })
        });

        // Parallel with chunking
        group.bench_with_input(
            BenchmarkId::new("parallel_chunks", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let sum: f64 = data
                        .par_chunks(65536)
                        .map(|chunk| chunk.iter().sum::<f64>())
                        .sum();
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel map operations
#[cfg(feature = "parallel")]
fn bench_parallel_map(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_map");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes: &[usize] = &[100_000, 1_000_000, 5_000_000];

    for &size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Sequential map
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result: Vec<f64> = data.iter().map(|x| x.sin()).collect();
                black_box(result)
            })
        });

        // Parallel map
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result: Vec<f64> = data.par_iter().map(|x| x.sin()).collect();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark parallel matrix operations
#[cfg(feature = "parallel")]
fn bench_parallel_matmul(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_matrix_multiply_f32;

    let mut group = c.benchmark_group("parallel_matmul");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    // Matrix sizes for parallel scaling tests
    let sizes: &[usize] = &[256, 512, 1024];

    for &size in sizes {
        let m = size;
        let k = size;
        let n = size;

        let a = generate_f32_data(m * k, 0);
        let b = generate_f32_data(k * n, 1);

        let flops = 2 * m * n * k;
        group.throughput(Throughput::Elements(flops as u64));

        // Single SIMD blocked matmul
        group.bench_with_input(
            BenchmarkId::new("single_blocked", size),
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
// Scalability Analysis Benchmarks
// =============================================================================

/// Benchmark to measure scalability across data sizes
fn bench_scalability_analysis(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::simd_dot_product_f32;

    let mut group = c.benchmark_group("scalability_analysis");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Powers of 2 for clean scalability analysis
    let sizes: &[usize] = &[
        1 << 8,  // 256
        1 << 10, // 1K
        1 << 12, // 4K
        1 << 14, // 16K
        1 << 16, // 64K
        1 << 18, // 256K
        1 << 20, // 1M
    ];

    for &size in sizes {
        let a = generate_f32_data(size, 0);
        let b = generate_f32_data(size, 1);

        // Throughput in bytes (2 vectors * size * 4 bytes)
        group.throughput(Throughput::Bytes((size * 8) as u64));

        group.bench_with_input(BenchmarkId::new("simd_dot", size), &size, |bencher, _| {
            bencher.iter(|| simd_dot_product_f32(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// Performance Comparison with Reference Implementations
// =============================================================================

/// Benchmark comparing with theoretical peak performance
fn bench_efficiency_analysis(c: &mut Criterion) {
    use scirs2_core::simd_ops::matmul::{simd_dot_product_f32, simd_matrix_multiply_f32};

    let mut group = c.benchmark_group("efficiency_analysis");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    // Dot product efficiency test
    let dot_size = 1 << 20; // 1M elements
    let a = generate_f32_data(dot_size, 0);
    let b = generate_f32_data(dot_size, 1);

    group.throughput(Throughput::Bytes((dot_size * 8) as u64));
    group.bench_function("dot_1m_f32", |bencher| {
        bencher.iter(|| simd_dot_product_f32(black_box(&a), black_box(&b)))
    });

    // Matrix multiply efficiency test
    let mat_size = 512;
    let mat_a = generate_f32_data(mat_size * mat_size, 0);
    let mat_b = generate_f32_data(mat_size * mat_size, 1);

    let flops = 2 * mat_size * mat_size * mat_size;
    group.throughput(Throughput::Elements(flops as u64));
    group.bench_function("matmul_512x512_f32", |bencher| {
        bencher.iter(|| {
            let mut c = vec![0.0f32; mat_size * mat_size];
            simd_matrix_multiply_f32(
                mat_size, mat_size, mat_size, 1.0, &mat_a, &mat_b, 0.0, &mut c,
            );
            black_box(c)
        })
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration and Main
// =============================================================================

// Non-parallel benchmark groups
criterion_group!(simd_benchmarks, bench_simd_dot_f32, bench_simd_dot_f64,);

criterion_group!(
    matmul_benchmarks,
    bench_matmul_square,
    bench_matmul_rectangular,
    bench_gemm_scaling,
);

criterion_group!(
    memory_benchmarks,
    bench_chunking_overhead,
    bench_memory_monitoring_overhead,
);

criterion_group!(
    analysis_benchmarks,
    bench_scalability_analysis,
    bench_efficiency_analysis,
);

// Parallel benchmark groups (feature-gated)
#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benchmarks,
    bench_parallel_reduction,
    bench_parallel_map,
    bench_parallel_matmul,
);

// Main with conditional compilation
#[cfg(feature = "parallel")]
criterion_main!(
    simd_benchmarks,
    matmul_benchmarks,
    memory_benchmarks,
    analysis_benchmarks,
    parallel_benchmarks,
);

#[cfg(not(feature = "parallel"))]
criterion_main!(
    simd_benchmarks,
    matmul_benchmarks,
    memory_benchmarks,
    analysis_benchmarks,
);
