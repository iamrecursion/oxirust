//! Media deduplication hashing benchmarks.
//!
//! Compares hash computation strategies commonly used for duplicate detection:
//!
//! - **SHA-256** — cryptographic, collision-resistant (used for exact-match dedup)
//! - **xxHash-64** — extremely fast non-cryptographic hash (used for quick equality checks)
//!
//! Data sizes simulate realistic media file chunk ingestion:
//!   1 MiB  — thumbnail / small asset
//!  10 MiB  — short audio clip / proxy frame sequence
//! 100 MiB  — SD video segment / full audio track

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sha2::{Digest, Sha256};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

/// Produce a deterministic pseudo-random byte buffer of exactly `size` bytes.
///
/// We use a simple LCG to avoid pulling in `rand` and to keep generation fast
/// while producing a non-trivial byte stream (not all-zeros / all-same).
fn make_data(size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(size);
    // LCG parameters (Numerical Recipes)
    let mut state: u64 = 0x_DEAD_BEEF_CAFE_BABE;
    while buf.len() < size {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bytes = state.to_le_bytes();
        let remaining = size - buf.len();
        buf.extend_from_slice(&bytes[..remaining.min(8)]);
    }
    buf
}

// Chunk sizes: 1 MiB, 10 MiB, 100 MiB
const SIZES: &[(usize, &str)] = &[
    (1 * 1024 * 1024, "1_MiB"),
    (10 * 1024 * 1024, "10_MiB"),
    (100 * 1024 * 1024, "100_MiB"),
];

// ---------------------------------------------------------------------------
// SHA-256 benchmarks
// ---------------------------------------------------------------------------

fn bench_sha256(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha256_hash");

    for &(size, label) in SIZES {
        let data = make_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let mut hasher = Sha256::new();
                hasher.update(black_box(data.as_slice()));
                let digest = hasher.finalize();
                black_box(digest)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// xxHash-64 benchmarks
// ---------------------------------------------------------------------------

fn bench_xxhash64(c: &mut Criterion) {
    let mut group = c.benchmark_group("xxhash64");

    for &(size, label) in SIZES {
        let data = make_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            b.iter(|| {
                let hash = xxhash_rust::xxh64::xxh64(black_box(data.as_slice()), 0);
                black_box(hash)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Comparative: SHA-256 vs xxHash-64 on same data
// ---------------------------------------------------------------------------

/// Head-to-head comparison at 10 MiB — the most realistic dedup chunk size.
fn bench_hash_comparison_10mib(c: &mut Criterion) {
    let size = 10 * 1024 * 1024;
    let data = make_data(size);
    let mut group = c.benchmark_group("hash_comparison_10_MiB");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::from_parameter("sha256"), &data, |b, data| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(data.as_slice()));
            black_box(hasher.finalize())
        });
    });

    group.bench_with_input(BenchmarkId::from_parameter("xxh64"), &data, |b, data| {
        b.iter(|| {
            let hash = xxhash_rust::xxh64::xxh64(black_box(data.as_slice()), 0);
            black_box(hash)
        });
    });

    group.finish();
}

/// Benchmark chunked SHA-256 hashing (4 KiB reads) to simulate streaming
/// file ingestion as it would happen during library scan.
fn bench_sha256_streaming_chunks(c: &mut Criterion) {
    let total_size = 10 * 1024 * 1024usize;
    let chunk_sizes: &[(usize, &str)] = &[
        (4 * 1024, "4_KiB_chunk"),
        (64 * 1024, "64_KiB_chunk"),
        (1024 * 1024, "1_MiB_chunk"),
    ];
    let data = make_data(total_size);
    let mut group = c.benchmark_group("sha256_streaming");
    group.throughput(Throughput::Bytes(total_size as u64));

    for &(chunk_size, label) in chunk_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let mut hasher = Sha256::new();
                    for chunk in black_box(data.as_slice()).chunks(chunk_size) {
                        hasher.update(chunk);
                    }
                    black_box(hasher.finalize())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    hash_benches,
    bench_sha256,
    bench_xxhash64,
    bench_hash_comparison_10mib,
    bench_sha256_streaming_chunks,
);
criterion_main!(hash_benches);
