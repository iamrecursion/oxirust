//! GPU vs CPU matmul benchmark.
//!
//! Compares OxiCUDA GPU matrix multiplication against the SciRS2 CPU backend
//! across square matrix sizes [64, 256, 1024, 2048].
//!
//! Run with GPU support enabled:
//!   cargo bench --features gpu
//!
//! On machines without an NVIDIA driver, the `gpu_matmul_square` group will
//! emit a skip message and run a no-op iteration so that CI remains green.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_infer::TlExecutor;
#[cfg(feature = "gpu")]
use tensorlogic_oxicuda_backend::{OxiCudaExecutor, OxiCudaTensor};
use tensorlogic_scirs_backend::Scirs2Exec;

/// Deterministic f32 fill pattern: `(i % 7) as f32 * 0.1`
#[cfg(feature = "gpu")]
fn make_f32_data(n: usize) -> Vec<f32> {
    let size = n * n;
    (0..size).map(|i| (i % 7) as f32 * 0.1).collect()
}

/// Deterministic f64 fill pattern: `(i % 7) as f64 * 0.1`
fn make_f64_data(n: usize) -> Vec<f64> {
    let size = n * n;
    (0..size).map(|i| (i % 7) as f64 * 0.1).collect()
}

// ---------------------------------------------------------------------------
// GPU benchmark group (only compiled with --features gpu)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn bench_gpu_matmul_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_matmul_square");

    for &n in &[64usize, 256, 1024, 2048] {
        // FLOPS proxy: each element of the N×N output requires N multiply-adds.
        group.throughput(Throughput::Elements((n * n * n) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // Attempt GPU initialisation outside of the hot loop.
            match OxiCudaExecutor::new() {
                Err(err) => {
                    // No NVIDIA driver available — run a no-op iteration so
                    // criterion does not error on an empty benchmark.
                    eprintln!("[gpu_matmul_square/{n}] GPU unavailable ({err}), skipping.");
                    b.iter(|| black_box(0u32));
                }
                Ok(mut executor) => {
                    let data_a = make_f32_data(n);
                    let data_b = make_f32_data(n);

                    let tensor_a = match OxiCudaTensor::new(vec![n, n], data_a) {
                        Ok(t) => t,
                        Err(err) => {
                            eprintln!("[gpu_matmul_square/{n}] Tensor A build failed: {err}");
                            b.iter(|| black_box(0u32));
                            return;
                        }
                    };
                    let tensor_b = match OxiCudaTensor::new(vec![n, n], data_b) {
                        Ok(t) => t,
                        Err(err) => {
                            eprintln!("[gpu_matmul_square/{n}] Tensor B build failed: {err}");
                            b.iter(|| black_box(0u32));
                            return;
                        }
                    };

                    b.iter(|| {
                        // Clone is cheap (host Vec<f32>) relative to the GPU
                        // dispatch; it keeps the tensors reusable across iters.
                        let a = black_box(tensor_a.clone());
                        let bm = black_box(tensor_b.clone());
                        let result = executor
                            .einsum("ij,jk->ik", &[a, bm])
                            .expect("GPU einsum failed in benchmark");
                        black_box(result)
                    });
                }
            }
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// CPU benchmark group (always compiled)
// ---------------------------------------------------------------------------

fn bench_cpu_matmul_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_matmul_square");

    for &n in &[64usize, 256, 1024, 2048] {
        // FLOPS proxy consistent with the GPU group.
        group.throughput(Throughput::Elements((n * n * n) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // Build tensors once outside the loop; clone per iteration.
            let data_a = make_f64_data(n);
            let data_b = make_f64_data(n);

            let tensor_a = scirs2_core::ndarray::ArrayD::from_shape_vec(
                scirs2_core::ndarray::IxDyn(&[n, n]),
                data_a,
            )
            .expect("cpu tensor_a shape/data consistent");

            let tensor_b = scirs2_core::ndarray::ArrayD::from_shape_vec(
                scirs2_core::ndarray::IxDyn(&[n, n]),
                data_b,
            )
            .expect("cpu tensor_b shape/data consistent");

            let mut executor = Scirs2Exec::new();

            b.iter(|| {
                let a = black_box(tensor_a.clone());
                let bm = black_box(tensor_b.clone());
                let result = executor
                    .einsum("ij,jk->ik", &[a, bm])
                    .expect("CPU einsum failed in benchmark");
                black_box(result)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// criterion_group! / criterion_main! — gated by feature
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_gpu_matmul_square, bench_cpu_matmul_square);

#[cfg(not(feature = "gpu"))]
criterion_group!(benches, bench_cpu_matmul_square);

criterion_main!(benches);
