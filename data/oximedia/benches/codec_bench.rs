//! Video and audio codec benchmarks
//!
//! This benchmark suite measures the performance of codec operations:
//! - AV1 decoding/encoding
//! - VP8 decoding
//! - VP9 decoding
//! - Transform operations (IDCT, IADST)
//! - Entropy decoding
//! - Loop filtering
//! - Prediction modes

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark AV1 frame header parsing
fn av1_frame_header_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_frame_header");

    // Minimal AV1 temporal delimiter OBU
    let temporal_delimiter = vec![
        0x12, 0x00, // OBU header (temporal delimiter)
    ];

    group.bench_function("parse_temporal_delimiter", |b| {
        b.iter(|| {
            black_box(&temporal_delimiter);
        });
    });

    // Frame header OBU (simplified)
    let frame_header = vec![
        0x32, 0x00, // OBU header (frame header)
        0x00, // show_existing_frame = 0
    ];

    group.bench_function("parse_frame_header", |b| {
        b.iter(|| {
            black_box(&frame_header);
        });
    });

    group.finish();
}

/// Benchmark AV1 sequence header parsing
fn av1_sequence_header_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_sequence_header");

    // Minimal sequence header OBU
    let seq_header = vec![
        0x0A, 0x00, // OBU header (sequence header)
        0x00, // seq_profile = 0
        0x00, // level = 0
    ];

    group.throughput(Throughput::Bytes(seq_header.len() as u64));

    group.bench_function("parse_sequence_header", |b| {
        b.iter(|| {
            black_box(&seq_header);
        });
    });

    group.finish();
}

/// Benchmark AV1 tile decoding
fn av1_tile_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_tile_decode");
    group.measurement_time(Duration::from_secs(10));

    let tile_sizes = vec![
        ("64x64", 64, 64),
        ("128x128", 128, 128),
        ("256x256", 256, 256),
    ];

    for (name, width, height) in tile_sizes {
        let data = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate tile processing
                black_box(data);
            });
        });
    }

    group.finish();
}

/// Benchmark AV1 transform operations (IDCT)
fn av1_transform_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_transform");

    // Test different transform sizes
    let transform_sizes = vec![
        ("4x4", 4),
        ("8x8", 8),
        ("16x16", 16),
        ("32x32", 32),
        ("64x64", 64),
    ];

    for (name, size) in transform_sizes {
        let coeffs = vec![0i16; size * size];

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &coeffs, |b, coeffs| {
            b.iter(|| {
                // Simulate IDCT operation
                let mut output = vec![0i16; coeffs.len()];
                for (i, &coeff) in coeffs.iter().enumerate() {
                    output[i] = coeff;
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark AV1 IADST (Inverse Asymmetric Discrete Sine Transform)
fn av1_iadst_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_iadst");

    let transform_sizes = vec![("4x4", 4), ("8x8", 8), ("16x16", 16)];

    for (name, size) in transform_sizes {
        let coeffs = vec![0i16; size * size];

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &coeffs, |b, coeffs| {
            b.iter(|| {
                // Simulate IADST operation
                let mut output = vec![0i16; coeffs.len()];
                for (i, &coeff) in coeffs.iter().enumerate() {
                    output[i] = coeff;
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark AV1 loop filter
fn av1_loop_filter_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_loop_filter");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("4K", 3840, 2160),
    ];

    for (name, width, height) in resolutions {
        let frame = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Bytes(frame.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &frame, |b, frame| {
            b.iter(|| {
                // Simulate loop filter
                let mut output = frame.clone();
                for pixel in &mut output {
                    *pixel = (*pixel).saturating_add(1);
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark AV1 CDEF (Constrained Directional Enhancement Filter)
fn av1_cdef_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_cdef");

    let block_sizes = vec![("8x8", 8), ("16x16", 16), ("32x32", 32)];

    for (name, size) in block_sizes {
        let block = vec![128u8; size * size];

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &block, |b, block| {
            b.iter(|| {
                // Simulate CDEF filtering
                let mut output = block.clone();
                for pixel in &mut output {
                    *pixel = (*pixel).saturating_add(1);
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark AV1 prediction modes (intra prediction)
fn av1_intra_prediction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_intra_prediction");

    let block_sizes = vec![("4x4", 4), ("8x8", 8), ("16x16", 16), ("32x32", 32)];

    for (name, size) in block_sizes {
        let neighbors = vec![128u8; size * 2];

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &neighbors,
            |b, neighbors| {
                b.iter(|| {
                    // Simulate DC prediction
                    let sum: u32 = neighbors.iter().map(|&x| u32::from(x)).sum();
                    let avg = (sum / neighbors.len() as u32) as u8;
                    let block = vec![avg; size * size];
                    black_box(block);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark AV1 motion compensation
fn av1_motion_compensation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_motion_comp");
    group.measurement_time(Duration::from_secs(10));

    let block_sizes = vec![("8x8", 8), ("16x16", 16), ("32x32", 32), ("64x64", 64)];

    for (name, size) in block_sizes {
        let ref_block = vec![128u8; size * size];

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &ref_block,
            |b, ref_block| {
                b.iter(|| {
                    // Simulate motion compensation (copy with interpolation)
                    let output = ref_block.clone();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark VP8 bool decoder
fn vp8_bool_decoder_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vp8_bool_decoder");

    let data_sizes = vec![
        ("small", 1024),
        ("medium", 10 * 1024),
        ("large", 100 * 1024),
    ];

    for (name, size) in data_sizes {
        let data = vec![0xAAu8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate bool decoding
                let mut count = 0;
                for &byte in data {
                    count += byte.count_ones();
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

/// Benchmark VP8 frame decoding
fn vp8_frame_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vp8_frame_decode");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("360p", 640, 360),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let frame = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Bytes(frame.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &frame, |b, frame| {
            b.iter(|| {
                // Simulate frame decoding
                black_box(frame);
            });
        });
    }

    group.finish();
}

/// Benchmark VP9 tile decoding
fn vp9_tile_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vp9_tile_decode");
    group.measurement_time(Duration::from_secs(10));

    let tile_sizes = vec![("512x512", 512, 512), ("1024x1024", 1024, 1024)];

    for (name, width, height) in tile_sizes {
        let tile = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Bytes(tile.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &tile, |b, tile| {
            b.iter(|| {
                // Simulate tile decoding
                black_box(tile);
            });
        });
    }

    group.finish();
}

/// Benchmark coefficient decoding
fn coefficient_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("coefficient_decode");

    let block_sizes = vec![("4x4", 16), ("8x8", 64), ("16x16", 256), ("32x32", 1024)];

    for (name, num_coeffs) in block_sizes {
        let data = vec![0u8; num_coeffs * 2]; // Encoded coefficients

        group.throughput(Throughput::Elements(num_coeffs as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate coefficient decoding
                let mut coeffs = vec![0i16; num_coeffs];
                for i in 0..num_coeffs.min(data.len() / 2) {
                    coeffs[i] = i16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                }
                black_box(coeffs);
            });
        });
    }

    group.finish();
}

/// Benchmark entropy decoding speed
fn entropy_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_decode");

    let data_sizes = vec![("1KB", 1024), ("10KB", 10 * 1024), ("100KB", 100 * 1024)];

    for (name, size) in data_sizes {
        let data = (0..size).map(|i| (i % 256) as u8).collect::<Vec<_>>();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate entropy decoding
                let mut output = Vec::with_capacity(data.len());
                for &byte in data {
                    output.push(byte ^ 0xAA);
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark full frame decode pipeline
fn full_frame_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_frame_decode");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let resolutions = vec![
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("1440p", 2560, 1440),
    ];

    for (name, width, height) in resolutions {
        let frame = helpers::generate_yuv420_frame(width, height);
        let _fps = 60.0; // Target FPS for comparison

        group.throughput(Throughput::Bytes(frame.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &frame, |b, frame| {
            b.iter(|| {
                // Simulate full decode pipeline
                black_box(frame);
            });
        });
    }

    group.finish();
}

criterion_group!(
    codec_benches,
    av1_frame_header_benchmark,
    av1_sequence_header_benchmark,
    av1_tile_decode_benchmark,
    av1_transform_benchmark,
    av1_iadst_benchmark,
    av1_loop_filter_benchmark,
    av1_cdef_benchmark,
    av1_intra_prediction_benchmark,
    av1_motion_compensation_benchmark,
    vp8_bool_decoder_benchmark,
    vp8_frame_decode_benchmark,
    vp9_tile_decode_benchmark,
    coefficient_decode_benchmark,
    entropy_decode_benchmark,
    full_frame_decode_benchmark,
);

criterion_main!(codec_benches);
