//! I/O performance benchmarks
//!
//! This benchmark suite measures the performance of I/O operations:
//! - Memory source reading
//! - Bit reader operations
//! - Exp-Golomb decoding
//! - Sequential vs. random access
//! - Buffer management

mod helpers;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_io::{BitReader, MemorySource};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark memory source sequential reading
fn memory_source_sequential_read_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_source_sequential");

    let data_sizes = vec![
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
        ("10MB", 10 * 1024 * 1024),
    ];

    for (name, size) in data_sizes {
        let data = vec![0xAAu8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                black_box(source);
            });
        });
    }

    group.finish();
}

/// Benchmark memory source creation overhead
fn memory_source_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_source_creation");

    let data_sizes = vec![
        ("small_1KB", 1024),
        ("medium_100KB", 100 * 1024),
        ("large_10MB", 10 * 1024 * 1024),
    ];

    for (name, size) in data_sizes {
        let data = vec![0xAAu8; size];
        let bytes = Bytes::from(data);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &bytes, |b, bytes| {
            b.iter(|| {
                let source = MemorySource::new(black_box(bytes.clone()));
                black_box(source);
            });
        });
    }

    group.finish();
}

/// Benchmark bit reader single bit operations
fn bit_reader_single_bit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_reader_single_bit");

    let data_sizes = vec![("1KB", 1024), ("10KB", 10 * 1024), ("100KB", 100 * 1024)];

    for (name, size) in data_sizes {
        let data = vec![0xAAu8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let mut reader = BitReader::new(black_box(data));
                let mut count = 0;

                // Read bits one at a time
                for _ in 0..(size * 8).min(1000) {
                    if let Ok(bit) = reader.read_bit() {
                        count += u32::from(bit);
                    }
                }

                black_box(count);
            });
        });
    }

    group.finish();
}

/// Benchmark bit reader multi-bit operations
fn bit_reader_multi_bit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_reader_multi_bit");

    let bit_counts: Vec<(&str, u8)> = vec![
        ("4_bits", 4),
        ("8_bits", 8),
        ("16_bits", 16),
        ("32_bits", 32),
    ];

    for (name, bits) in bit_counts {
        let data = vec![0xAAu8; 1024];

        group.throughput(Throughput::Elements(1024 / (bits as u64 / 8)));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(data, bits),
            |b, (data, bits)| {
                b.iter(|| {
                    let mut reader = BitReader::new(black_box(data));
                    let mut sum = 0u64;

                    // Read multiple bits at a time
                    for _ in 0..(data.len() * 8 / (*bits as usize)).min(100) {
                        if let Ok(value) = reader.read_bits(*bits) {
                            sum = sum.wrapping_add(value);
                        }
                    }

                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bit reader byte alignment operations
fn bit_reader_byte_align_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_reader_byte_align");

    let patterns = vec![
        ("unaligned_1bit", 1),
        ("unaligned_3bits", 3),
        ("unaligned_5bits", 5),
        ("aligned_8bits", 8),
    ];

    for (name, initial_bits) in patterns {
        let data = vec![0xAAu8; 1024];

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(data, initial_bits),
            |b, (data, initial_bits)| {
                b.iter(|| {
                    let mut reader = BitReader::new(black_box(data));
                    let mut count = 0;

                    for _ in 0..100 {
                        // Read some bits to misalign
                        if reader.read_bits(*initial_bits).is_ok() {
                            // Align to byte boundary
                            reader.byte_align();
                            count += 1;
                        }
                    }

                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark exp-golomb unsigned decoding
fn exp_golomb_unsigned_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_golomb_unsigned");

    // Create test data with various exp-golomb encoded values
    let patterns = vec![
        ("small_values", vec![0x80u8; 100]),  // Value 0 (1 bit)
        ("medium_values", vec![0x40u8; 100]), // Value 1-2 (3 bits)
        ("large_values", vec![0x10u8; 100]),  // Value 3-6 (5 bits)
    ];

    for (name, data) in patterns {
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let mut reader = BitReader::new(black_box(data));
                let mut sum = 0u64;

                for _ in 0..10 {
                    if let Ok(value) = reader.read_exp_golomb() {
                        sum = sum.wrapping_add(value);
                    }
                }

                black_box(sum);
            });
        });
    }

    group.finish();
}

/// Benchmark exp-golomb signed decoding
fn exp_golomb_signed_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_golomb_signed");

    let patterns = vec![
        ("small_values", vec![0x80u8; 100]),
        ("medium_values", vec![0x40u8; 100]),
        ("large_values", vec![0x10u8; 100]),
    ];

    for (name, data) in patterns {
        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let mut reader = BitReader::new(black_box(data));
                let mut sum = 0i64;

                for _ in 0..10 {
                    if let Ok(value) = reader.read_signed_exp_golomb() {
                        sum = sum.wrapping_add(value);
                    }
                }

                black_box(sum);
            });
        });
    }

    group.finish();
}

/// Benchmark bit reader position tracking
fn bit_reader_position_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_reader_position");

    group.bench_function("position_queries", |b| {
        let data = vec![0xAAu8; 1024];

        b.iter(|| {
            let mut reader = BitReader::new(black_box(&data));
            let mut positions = Vec::new();

            for _ in 0..100 {
                if reader.read_bits(8).is_ok() {
                    positions.push(reader.bits_read());
                }
            }

            black_box(positions);
        });
    });

    group.finish();
}

/// Benchmark buffer copy operations
fn buffer_copy_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_copy");

    let sizes = vec![
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
    ];

    for (name, size) in sizes {
        let data = vec![0xAAu8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let bytes = Bytes::from(black_box(data.clone()));
                black_box(bytes);
            });
        });
    }

    group.finish();
}

/// Benchmark zero-copy slicing
fn zero_copy_slice_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_slice");

    let sizes = vec![
        ("small_16B", 16),
        ("medium_1KB", 1024),
        ("large_100KB", 100 * 1024),
    ];

    for (name, size) in sizes {
        let data = vec![0xAAu8; 10 * 1024 * 1024]; // Large buffer
        let bytes = Bytes::from(data);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(bytes, size),
            |b, (bytes, size)| {
                b.iter(|| {
                    let slice = black_box(bytes).slice(0..*size);
                    black_box(slice);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed read patterns (realistic workload)
fn mixed_read_pattern_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_read_pattern");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("video_header_parsing", |b| {
        // Simulate realistic video header parsing
        let data = vec![0xAAu8; 1024];

        b.iter(|| {
            let mut reader = BitReader::new(black_box(&data));
            let mut values = Vec::new();

            // Typical pattern: mix of fixed-width and variable-length reads
            for _ in 0..10 {
                // Read some fixed-width fields
                if let Ok(val) = reader.read_bits(4) {
                    values.push(val);
                }
                if let Ok(val) = reader.read_bits(8) {
                    values.push(val);
                }

                // Read some exp-golomb values
                if let Ok(val) = reader.read_exp_golomb() {
                    values.push(u64::from(val));
                }

                // Align to byte
                reader.byte_align();

                // Read a few bytes
                if let Ok(val) = reader.read_bits(16) {
                    values.push(val);
                }
            }

            black_box(values);
        });
    });

    group.finish();
}

/// Benchmark bit reader look-ahead operations
fn bit_reader_lookahead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_reader_lookahead");

    let lookahead_sizes: Vec<(&str, u8)> =
        vec![("1_bit", 1), ("4_bits", 4), ("8_bits", 8), ("16_bits", 16)];

    for (name, bits) in lookahead_sizes {
        let data = vec![0xAAu8; 1024];

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(data, bits),
            |b, (data, bits)| {
                b.iter(|| {
                    let mut reader = BitReader::new(black_box(data));
                    let mut sum = 0u64;

                    // Read bits as lookahead simulation
                    for _ in 0..100 {
                        if let Ok(value) = reader.read_bits(*bits) {
                            sum = sum.wrapping_add(value);
                        }
                    }

                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    io_benches,
    memory_source_sequential_read_benchmark,
    memory_source_creation_benchmark,
    bit_reader_single_bit_benchmark,
    bit_reader_multi_bit_benchmark,
    bit_reader_byte_align_benchmark,
    exp_golomb_unsigned_benchmark,
    exp_golomb_signed_benchmark,
    bit_reader_position_benchmark,
    buffer_copy_benchmark,
    zero_copy_slice_benchmark,
    mixed_read_pattern_benchmark,
    bit_reader_lookahead_benchmark,
);

criterion_main!(io_benches);
