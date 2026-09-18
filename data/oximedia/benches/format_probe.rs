//! Container format detection benchmarks.
//!
//! Measures the throughput of `probe_format` — the magic-byte sniffing routine
//! that identifies container formats from the first few bytes of media data.
//! All test headers are generated synthetically (no file I/O).

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_container::probe_format;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Synthetic magic-byte headers
// ---------------------------------------------------------------------------

/// 32-byte Matroska/WebM EBML header prefix.
fn matroska_header() -> Vec<u8> {
    // EBML element ID + length + EBMLVersion, EBMLReadVersion, DocType = "webm"
    let mut h = vec![
        0x1A, 0x45, 0xDF, 0xA3, // EBML element ID
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, // size (vint)
        0x42, 0x86, 0x81, 0x01, // EBMLVersion = 1
        0x42, 0xF7, 0x81, 0x01, // EBMLReadVersion = 1
        0x42, 0xF2, 0x81, 0x04, // EBMLMaxIDLength = 4
        0x42, 0xF3, 0x81, 0x08, // EBMLMaxSizeLength = 8
        0x42, 0x82, 0x84, 0x77, // DocType = "w..."
    ];
    h.extend_from_slice(b"ebm");
    h.push(0x00);
    h
}

/// 32-byte Ogg page header.
fn ogg_header() -> Vec<u8> {
    let mut h = Vec::with_capacity(32);
    h.extend_from_slice(b"OggS"); // capture pattern
    h.push(0x00); // version
    h.push(0x02); // header type: beginning of stream
    h.extend_from_slice(&[0u8; 8]); // granule position
    h.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // serial number
    h.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // page sequence
    h.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // checksum
    h.push(0x01); // segment count
    h.push(0x1E); // segment table[0] = 30 bytes payload
    h.extend_from_slice(&[0u8; 2]); // pad to 32
    h
}

/// 32-byte FLAC stream marker.
fn flac_header() -> Vec<u8> {
    let mut h = Vec::with_capacity(32);
    h.extend_from_slice(b"fLaC"); // stream marker
                                  // STREAMINFO metadata block header: type=0, last=1, length=34
    h.push(0x80); // last-metadata-block=1, block-type=0
    h.push(0x00);
    h.push(0x00);
    h.push(0x22); // length = 34 bytes
                  // Minimal STREAMINFO payload (fill with zeros)
    h.extend_from_slice(&[0u8; 24]);
    h
}

/// 32-byte WAV/RIFF header.
fn wav_header() -> Vec<u8> {
    let mut h = Vec::with_capacity(32);
    h.extend_from_slice(b"RIFF"); // chunk ID
    h.extend_from_slice(&(1024u32.to_le_bytes())); // chunk size
    h.extend_from_slice(b"WAVE"); // format
    h.extend_from_slice(b"fmt "); // sub-chunk ID
    h.extend_from_slice(&(16u32.to_le_bytes())); // sub-chunk size
    h.extend_from_slice(&(1u16.to_le_bytes())); // PCM audio format
    h.extend_from_slice(&(2u16.to_le_bytes())); // 2 channels
    h.extend_from_slice(&(48_000u32.to_le_bytes())); // sample rate
    h
}

/// 32-byte Y4M header.
fn y4m_header() -> Vec<u8> {
    let mut h = b"YUV4MPEG2 W1920 H1080 F25:1 Ip A1:1 C420\n".to_vec();
    h.resize(32, 0x20); // pad with spaces to 32 bytes
    h
}

/// 12 random bytes that will not match any known format (unknown).
fn unknown_header() -> Vec<u8> {
    // Chosen to be recognisably wrong but deterministic
    vec![
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    ]
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

fn bench_probe_known_formats(c: &mut Criterion) {
    let headers: &[(&str, Vec<u8>)] = &[
        ("matroska_webm", matroska_header()),
        ("ogg", ogg_header()),
        ("flac", flac_header()),
        ("wav_riff", wav_header()),
        ("y4m", y4m_header()),
    ];

    let mut group = c.benchmark_group("probe_format_known");

    for (name, header) in headers {
        let byte_count = header.len() as u64;
        group.throughput(Throughput::Bytes(byte_count));

        group.bench_with_input(
            BenchmarkId::new("magic_bytes", name),
            header,
            |b, header| {
                b.iter(|| {
                    // probe_format returns Ok(ProbeResult) or Err(OxiError)
                    let result = probe_format(black_box(header));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_probe_unknown_format(c: &mut Criterion) {
    let header = unknown_header();
    let byte_count = header.len() as u64;
    let mut group = c.benchmark_group("probe_format_unknown");
    group.throughput(Throughput::Bytes(byte_count));

    group.bench_with_input(
        BenchmarkId::from_parameter("random_12_bytes"),
        &header,
        |b, header| {
            b.iter(|| {
                let result = probe_format(black_box(header));
                black_box(result)
            });
        },
    );

    group.finish();
}

/// Benchmark the full format detection pipeline over all known formats
/// in sequence to simulate a realistic probing workload.
fn bench_probe_sequential_all(c: &mut Criterion) {
    let all_headers: Vec<Vec<u8>> = vec![
        matroska_header(),
        ogg_header(),
        flac_header(),
        wav_header(),
        y4m_header(),
        unknown_header(),
    ];

    let mut group = c.benchmark_group("probe_format_sequential");
    group.throughput(Throughput::Elements(all_headers.len() as u64));

    group.bench_function("6_formats", |b| {
        b.iter(|| {
            for header in black_box(&all_headers) {
                let result = probe_format(header);
                black_box(result);
            }
        });
    });

    group.finish();
}

criterion_group!(
    probe_benches,
    bench_probe_known_formats,
    bench_probe_unknown_format,
    bench_probe_sequential_all,
);
criterion_main!(probe_benches);
