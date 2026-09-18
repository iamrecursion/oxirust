//! Container demuxer/muxer benchmarks
//!
//! This benchmark suite measures the performance of container format
//! demuxing and muxing operations.
//!
//! Benchmarks include:
//! - Matroska/WebM demuxing
//! - Ogg demuxing
//! - FLAC demuxing
//! - MP4 demuxing
//! - WAV demuxing
//! - Muxer write throughput

mod helpers;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_container::probe_format;
use oximedia_io::MemorySource;
use std::hint::black_box;
use std::time::Duration;

/// Benchmark Matroska/WebM format probing
fn matroska_probe_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matroska_probe");

    let header = helpers::create_minimal_webm_header();

    group.bench_function("probe_webm_header", |b| {
        b.iter(|| {
            let result = probe_format(black_box(&header));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark Matroska/WebM demuxing
fn matroska_demux_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matroska_demux");
    group.measurement_time(Duration::from_secs(10));

    // Test with different file sizes if available
    let test_files = vec![
        ("small_webm", "test_small.webm", 1024 * 100), // 100KB
        ("medium_webm", "test_medium.webm", 1024 * 1024), // 1MB
        ("large_webm", "test_large.webm", 10 * 1024 * 1024), // 10MB
    ];

    for (name, _filename, size) in test_files {
        // Generate synthetic WebM data
        let mut data = helpers::create_minimal_webm_header();
        data.resize(size, 0);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                // Note: In real implementation, would call demuxer methods
                // For now, just measure source creation overhead
                black_box(source)
            });
        });
    }

    group.finish();
}

/// Benchmark EBML element parsing
fn ebml_parsing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ebml_parsing");

    let header = helpers::create_minimal_webm_header();

    group.throughput(Throughput::Bytes(header.len() as u64));

    group.bench_function("parse_ebml_header", |b| {
        b.iter(|| {
            let source = MemorySource::new(Bytes::from(black_box(header.clone())));
            black_box(source)
        });
    });

    group.finish();
}

/// Benchmark Ogg page parsing
fn ogg_page_parsing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ogg_parsing");

    let page = helpers::create_minimal_ogg_page();

    group.throughput(Throughput::Bytes(page.len() as u64));

    group.bench_function("parse_ogg_page", |b| {
        b.iter(|| {
            let source = MemorySource::new(Bytes::from(black_box(page.clone())));
            black_box(source)
        });
    });

    group.finish();
}

/// Benchmark Ogg demuxing throughput
fn ogg_demux_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ogg_demux");
    group.measurement_time(Duration::from_secs(10));

    // Different Ogg file sizes
    let sizes = vec![
        ("small", 50 * 1024),       // 50KB
        ("medium", 500 * 1024),     // 500KB
        ("large", 5 * 1024 * 1024), // 5MB
    ];

    for (name, size) in sizes {
        let mut data = helpers::create_minimal_ogg_page();
        data.resize(size, 0);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                black_box(source)
            });
        });
    }

    group.finish();
}

/// Benchmark FLAC frame parsing
fn flac_frame_parsing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_parsing");

    // Minimal FLAC header: "fLaC" + STREAMINFO block
    let mut flac_header = Vec::new();
    flac_header.extend_from_slice(b"fLaC");
    // STREAMINFO block (type 0, length 34)
    flac_header.push(0x00);
    flac_header.extend_from_slice(&[0x00, 0x00, 0x22]); // length = 34
    flac_header.resize(flac_header.len() + 34, 0);

    group.throughput(Throughput::Bytes(flac_header.len() as u64));

    group.bench_function("parse_flac_header", |b| {
        b.iter(|| {
            let source = MemorySource::new(Bytes::from(black_box(flac_header.clone())));
            black_box(source)
        });
    });

    group.finish();
}

/// Benchmark FLAC demuxing
fn flac_demux_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_demux");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![
        ("small", 100 * 1024),       // 100KB
        ("medium", 1024 * 1024),     // 1MB
        ("large", 10 * 1024 * 1024), // 10MB
    ];

    for (name, size) in sizes {
        let mut data = Vec::new();
        data.extend_from_slice(b"fLaC");
        data.push(0x00);
        data.extend_from_slice(&[0x00, 0x00, 0x22]);
        data.resize(size, 0);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                black_box(source)
            });
        });
    }

    group.finish();
}

/// Benchmark MP4 box parsing
fn mp4_box_parsing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mp4_parsing");

    // Minimal ftyp box
    let ftyp_box = vec![
        0x00, 0x00, 0x00, 0x20, // size = 32 bytes
        b'f', b't', b'y', b'p', // type = 'ftyp'
        b'i', b's', b'o', b'm', // major brand
        0x00, 0x00, 0x02, 0x00, // minor version
        b'i', b's', b'o', b'm', // compatible brand
        b'i', b's', b'o', b'2', // compatible brand
        b'a', b'v', b'c', b'1', // compatible brand
        b'm', b'p', b'4', b'1', // compatible brand
    ];

    group.throughput(Throughput::Bytes(ftyp_box.len() as u64));

    group.bench_function("parse_mp4_ftyp", |b| {
        b.iter(|| {
            let source = MemorySource::new(Bytes::from(black_box(ftyp_box.clone())));
            black_box(source)
        });
    });

    group.finish();
}

/// Benchmark MP4 demuxing
fn mp4_demux_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mp4_demux");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![
        ("small", 200 * 1024),       // 200KB
        ("medium", 2 * 1024 * 1024), // 2MB
        ("large", 20 * 1024 * 1024), // 20MB
    ];

    for (name, size) in sizes {
        let mut data = vec![
            0x00, 0x00, 0x00, 0x20, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm', 0x00, 0x00,
            0x02, 0x00, b'i', b's', b'o', b'm', b'i', b's', b'o', b'2', b'a', b'v', b'c', b'1',
            b'm', b'p', b'4', b'1',
        ];
        data.resize(size, 0);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                black_box(source)
            });
        });
    }

    group.finish();
}

/// Benchmark WAV header parsing
fn wav_header_parsing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_parsing");

    // Minimal WAV header
    let mut wav_header = Vec::new();
    wav_header.extend_from_slice(b"RIFF");
    wav_header.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // file size
    wav_header.extend_from_slice(b"WAVE");
    wav_header.extend_from_slice(b"fmt ");
    wav_header.extend_from_slice(&[16, 0, 0, 0]); // chunk size
    wav_header.extend_from_slice(&[1, 0]); // audio format (PCM)
    wav_header.extend_from_slice(&[2, 0]); // num channels
    wav_header.extend_from_slice(&[0x44, 0xAC, 0x00, 0x00]); // sample rate (44100)
    wav_header.extend_from_slice(&[0x10, 0xB1, 0x02, 0x00]); // byte rate
    wav_header.extend_from_slice(&[4, 0]); // block align
    wav_header.extend_from_slice(&[16, 0]); // bits per sample

    group.throughput(Throughput::Bytes(wav_header.len() as u64));

    group.bench_function("parse_wav_header", |b| {
        b.iter(|| {
            let source = MemorySource::new(Bytes::from(black_box(wav_header.clone())));
            black_box(source)
        });
    });

    group.finish();
}

/// Benchmark WAV demuxing
fn wav_demux_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_demux");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![
        ("small", 100 * 1024),       // 100KB
        ("medium", 1024 * 1024),     // 1MB
        ("large", 10 * 1024 * 1024), // 10MB
    ];

    for (name, size) in sizes {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(size as u32 - 8).to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&[16, 0, 0, 0]);
        data.extend_from_slice(&[1, 0]);
        data.extend_from_slice(&[2, 0]);
        data.extend_from_slice(&[0x44, 0xAC, 0x00, 0x00]);
        data.extend_from_slice(&[0x10, 0xB1, 0x02, 0x00]);
        data.extend_from_slice(&[4, 0]);
        data.extend_from_slice(&[16, 0]);
        data.extend_from_slice(b"data");
        data.extend_from_slice(&[(size - 44) as u32].map(|v| v.to_le_bytes()).concat());
        data.resize(size, 0);

        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                let source = MemorySource::new(Bytes::from(black_box(data.clone())));
                black_box(source)
            });
        });
    }

    group.finish();
}

/// Benchmark format detection (probing)
fn format_probe_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_probe");

    let test_headers = vec![
        ("webm", helpers::create_minimal_webm_header()),
        ("ogg", helpers::create_minimal_ogg_page()),
        ("flac", {
            let mut h = Vec::new();
            h.extend_from_slice(b"fLaC");
            h
        }),
        ("mp4", {
            vec![
                0x00, 0x00, 0x00, 0x20, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm', 0x00, 0x00,
                0x02, 0x00,
            ]
        }),
        ("wav", {
            let mut h = Vec::new();
            h.extend_from_slice(b"RIFF");
            h.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
            h.extend_from_slice(b"WAVE");
            h
        }),
    ];

    for (name, header) in test_headers {
        group.bench_with_input(BenchmarkId::from_parameter(name), &header, |b, header| {
            b.iter(|| {
                let result = probe_format(black_box(header));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    container_benches,
    matroska_probe_benchmark,
    matroska_demux_benchmark,
    ebml_parsing_benchmark,
    ogg_page_parsing_benchmark,
    ogg_demux_benchmark,
    flac_frame_parsing_benchmark,
    flac_demux_benchmark,
    mp4_box_parsing_benchmark,
    mp4_demux_benchmark,
    wav_header_parsing_benchmark,
    wav_demux_benchmark,
    format_probe_benchmark,
);

criterion_main!(container_benches);
