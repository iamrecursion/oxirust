//! Simple benchmarks for tensor operations
//!
//! This benchmark suite covers basic tensor contraction operations that are
//! actually implemented in the tensor_contraction module.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_linalg::tensor_contraction::{batch_matmul, contract, einsum, hosvd, mode_n_product};
use std::hint::black_box;
use std::time::Duration;

/// Create a test tensor of given dimensions with deterministic values
#[allow(dead_code)]
fn create_test_tensor_3d(d1: usize, d2: usize, d3: usize) -> Array3<f64> {
    Array3::from_shape_fn((d1, d2, d3), |(i, j, k)| {
        ((i + j + k + 1) as f64 * 0.1).sin()
    })
}

/// Create a random-like matrix for tensor operations
#[allow(dead_code)]
fn create_testmatrix(m: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((m, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin())
}

/// Create a test vector
#[allow(dead_code)]
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Benchmark basic tensor contraction operations
#[allow(dead_code)]
fn bench_tensor_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_contraction");
    group.sample_size(20);

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);
        let matrix = create_testmatrix(size, size);

        let total_elements = size * size * size;
        group.throughput(Throughput::Elements(total_elements as u64));

        // Mode-n product (tensor-matrix multiplication)
        group.bench_with_input(
            BenchmarkId::new("mode_n_product_0", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| {
                    mode_n_product(black_box(&t.view()), black_box(&m.view()), 0)
                        .expect("Operation failed")
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mode_n_product_1", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| {
                    mode_n_product(black_box(&t.view()), black_box(&m.view()), 1)
                        .expect("Operation failed")
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mode_n_product_2", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| {
                    mode_n_product(black_box(&t.view()), black_box(&m.view()), 2)
                        .expect("Operation failed")
                })
            },
        );

        // Basic tensor contraction
        group.bench_with_input(
            BenchmarkId::new("tensor_contract", size),
            &tensor_3d,
            |b, t| {
                b.iter(|| {
                    contract(black_box(&t.view()), black_box(&t.view()), &[0], &[1])
                        .expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Einstein summation operations
#[allow(dead_code)]
fn bench_einsum_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_operations");
    group.sample_size(20);

    for &size in &[10, 20, 30] {
        let matrix_a = create_testmatrix(size, size);
        let matrix_b = create_testmatrix(size, size);
        let vector = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix multiplication: ij,jk->ik
        group.bench_with_input(
            BenchmarkId::new("einsum_matmul", size),
            &(&matrix_a, &matrix_b),
            |b, (a, matrix_b)| {
                b.iter(|| {
                    einsum(
                        "ij,jk->ik",
                        &[
                            black_box(&a.view().into_dyn()),
                            black_box(&matrix_b.view().into_dyn()),
                        ],
                    )
                    .expect("Operation failed")
                })
            },
        );

        // Matrix-vector multiplication: ij,j->i
        group.bench_with_input(
            BenchmarkId::new("einsum_matvec", size),
            &(&matrix_a, &vector),
            |b, (m, v)| {
                b.iter(|| {
                    einsum(
                        "ij,j->i",
                        &[
                            black_box(&m.view().into_dyn()),
                            black_box(&v.view().into_dyn()),
                        ],
                    )
                    .expect("Operation failed")
                })
            },
        );

        // Trace: ii->
        group.bench_with_input(BenchmarkId::new("einsum_trace", size), &matrix_a, |b, m| {
            b.iter(|| einsum("ii->", &[black_box(&m.view().into_dyn())]).expect("Operation failed"))
        });

        // Transpose: ij->ji
        group.bench_with_input(
            BenchmarkId::new("einsum_transpose", size),
            &matrix_a,
            |b, m| {
                b.iter(|| {
                    einsum("ij->ji", &[black_box(&m.view().into_dyn())]).expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Higher-Order SVD (HOSVD) operations
#[allow(dead_code)]
fn bench_hosvd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hosvd_operations");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in &[8, 12, 16] {
        // Smaller sizes for HOSVD due to computational cost
        let tensor_3d = create_test_tensor_3d(size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // Higher-Order SVD (3D tensor)
        group.bench_with_input(BenchmarkId::new("hosvd_3d", size), &tensor_3d, |b, t| {
            b.iter(|| hosvd(black_box(&t.view()), &[size, size, size]).expect("Operation failed"))
        });

        // HOSVD with reduced rank
        let reduced_rank = size / 2 + 1;
        group.bench_with_input(
            BenchmarkId::new("hosvd_3d_reduced", size),
            &tensor_3d,
            |b, t| {
                b.iter(|| {
                    hosvd(
                        black_box(&t.view()),
                        &[reduced_rank, reduced_rank, reduced_rank],
                    )
                    .expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

/// Benchmark batch tensor operations
#[allow(dead_code)]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.sample_size(20);

    for &batchsize in &[4, 8, 16] {
        for &size in &[8, 12, 16] {
            let batch_a = Array3::from_shape_fn((batchsize, size, size), |(b, i, j)| {
                ((b + i + j + 1) as f64 * 0.1).sin()
            });
            let batch_b = Array3::from_shape_fn((batchsize, size, size), |(b, i, j)| {
                ((b + i + j + 1) as f64 * 0.1).cos()
            });

            group.throughput(Throughput::Elements(
                batchsize as u64 * size as u64 * size as u64,
            ));

            // Batch matrix multiplication
            group.bench_with_input(
                BenchmarkId::new(format!("batch_matmul_{}x{}", batchsize, size), size),
                &(&batch_a, &batch_b),
                |b, (a, batch_b)| {
                    b.iter(|| {
                        batch_matmul(black_box(&a.view()), black_box(&batch_b.view()), 1)
                            .expect("Operation failed")
                    })
                },
            );
        }
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_tensor_contraction,
    bench_einsum_operations,
    bench_hosvd_operations,
    bench_batch_operations
);

criterion_main!(benches);
