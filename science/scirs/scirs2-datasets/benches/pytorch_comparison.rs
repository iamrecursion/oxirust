#![allow(clippy::unwrap_used)]
//! Benchmark comparing scirs2-datasets with PyTorch DataLoader
//!
//! This benchmark measures throughput, memory usage, and latency of
//! scirs2-datasets compared to PyTorch's DataLoader.
//!
//! Note: This benchmark only covers the Rust side. For fair comparison,
//! run the corresponding Python benchmarks separately.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{s, Array2};
use scirs2_datasets::error::Result;
use scirs2_datasets::generators::{make_classification, make_regression};
use scirs2_datasets::streaming::{StreamConfig, StreamingIterator};
use scirs2_datasets::utils::Dataset;
use std::hint::black_box;
use std::time::Duration;

// ============================================================================
// Dataset Generation
// ============================================================================

fn generate_classification_dataset(n_samples: usize, n_features: usize) -> Result<Dataset> {
    make_classification(n_samples, n_features, 2, 0, 5, Some(42))
}

fn generate_regression_dataset(n_samples: usize, n_features: usize) -> Result<Dataset> {
    make_regression(n_samples, n_features, n_features / 2, 0.1, Some(42))
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_sequential_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_loading");

    for size in [1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let dataset = generate_classification_dataset(size, 20).unwrap();
                black_box(dataset);
            });
        });
    }

    group.finish();
}

fn bench_batch_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_iteration");
    let batch_sizes = [32, 64, 128, 256];
    let dataset_size = 10_000;

    for batch_size in batch_sizes.iter() {
        group.throughput(Throughput::Elements(dataset_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let dataset = generate_classification_dataset(dataset_size, 20).unwrap();

                b.iter(|| {
                    let n_batches = dataset_size / batch_size;
                    for i in 0..n_batches {
                        let start = i * batch_size;
                        let end = ((i + 1) * batch_size).min(dataset_size);
                        let batch_data = dataset.data.slice(s![start..end, ..]);
                        black_box(batch_data);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_streaming_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_iteration");
    group.measurement_time(Duration::from_secs(10));

    for chunk_size in [1_000, 5_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*chunk_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    // Create a generator-based streaming iterator
                    let config = StreamConfig {
                        chunk_size,
                        buffer_size: 3,
                        num_workers: 1,
                        memory_limit_mb: None,
                        enable_compression: false,
                        enable_prefetch: false,
                        max_chunks: Some(10), // Limit for benchmark speed
                    };

                    let generator = |start: usize, chunk_size: usize, n_features: usize| {
                        let n_samples = chunk_size;
                        let data = Array2::zeros((n_samples, n_features));
                        Ok((data, None))
                    };

                    let mut stream =
                        StreamingIterator::from_generator(generator, chunk_size * 10, 20, config)
                            .unwrap();

                    while let Ok(Some(chunk)) = stream.next_chunk() {
                        black_box(chunk);
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Efficiency Benchmarks
// ============================================================================

fn bench_memory_efficient_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficient_processing");
    group.measurement_time(Duration::from_secs(15));

    // Simulate processing a large dataset in chunks
    let total_size = 100_000;
    let chunk_size = 10_000;

    group.bench_function("chunked_processing", |b| {
        b.iter(|| {
            let n_chunks = total_size / chunk_size;
            for i in 0..n_chunks {
                let chunk = generate_classification_dataset(chunk_size, 20).unwrap();
                // Simulate some processing
                let processed = &chunk.data * 2.0;
                black_box(processed);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Latency Benchmarks
// ============================================================================

fn bench_single_sample_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_sample_access");

    let dataset = generate_classification_dataset(10_000, 20).unwrap();

    group.bench_function("random_access", |b| {
        b.iter(|| {
            let idx = 5000;
            let sample = dataset.data.row(idx);
            black_box(sample);
        });
    });

    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let dataset = generate_classification_dataset(10_000, 100).unwrap();

    group.bench_function("column_extraction", |b| {
        b.iter(|| {
            let column = dataset.data.column(50);
            black_box(column);
        });
    });

    group.finish();
}

// ============================================================================
// Preprocessing Benchmarks
// ============================================================================

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    for size in [1_000, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let dataset = generate_classification_dataset(size, 20).unwrap();

            b.iter(|| {
                // Simple standardization
                let mean = dataset.data.mean_axis(scirs2_core::ndarray::Axis(0));
                let std = dataset.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
                let normalized = (&dataset.data - &mean.unwrap()) / &std;
                black_box(normalized);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_parallel_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_loading");
    group.measurement_time(Duration::from_secs(10));

    for num_workers in [1, 2, 4].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_workers),
            num_workers,
            |b, &num_workers| {
                b.iter(|| {
                    let config = StreamConfig {
                        chunk_size: 5_000,
                        buffer_size: 5,
                        num_workers,
                        memory_limit_mb: None,
                        enable_compression: false,
                        enable_prefetch: true,
                        max_chunks: Some(5),
                    };

                    let generator = |_start: usize, chunk_size: usize, n_features: usize| {
                        let data = Array2::zeros((chunk_size, n_features));
                        Ok((data, None))
                    };

                    let mut stream =
                        StreamingIterator::from_generator(generator, 25_000, 20, config).unwrap();

                    while let Ok(Some(chunk)) = stream.next_chunk() {
                        black_box(chunk);
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    throughput_benches,
    bench_sequential_loading,
    bench_batch_iteration,
    bench_streaming_iteration
);

criterion_group!(memory_benches, bench_memory_efficient_processing);

criterion_group!(
    latency_benches,
    bench_single_sample_access,
    bench_feature_extraction
);

criterion_group!(preprocessing_benches, bench_normalization);

criterion_group!(scaling_benches, bench_parallel_loading);

criterion_main!(
    throughput_benches,
    memory_benches,
    latency_benches,
    preprocessing_benches,
    scaling_benches
);
