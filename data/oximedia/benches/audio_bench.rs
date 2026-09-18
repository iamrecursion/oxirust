//! Audio codec and processing benchmarks
//!
//! This benchmark suite measures the performance of audio operations:
//! - FLAC decoding
//! - Opus decoding
//! - Vorbis decoding
//! - PCM operations
//! - Audio resampling
//! - Audio filters (EQ, compressor, limiter)

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark FLAC frame decoding
fn flac_frame_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_decode");
    group.measurement_time(Duration::from_secs(10));

    let sample_rates = vec![
        ("44.1kHz", 44_100),
        ("48kHz", 48_000),
        ("96kHz", 96_000),
        ("192kHz", 192_000),
    ];

    for (name, sample_rate) in sample_rates {
        let samples = helpers::generate_audio_samples(sample_rate, 1000, 440.0);

        group.throughput(Throughput::Elements(samples.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate FLAC decoding
                black_box(samples);
            });
        });
    }

    group.finish();
}

/// Benchmark FLAC Rice/Golomb decoding
fn flac_rice_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_rice_decode");

    let block_sizes = vec![("576", 576), ("1152", 1152), ("2304", 2304), ("4608", 4608)];

    for (name, block_size) in block_sizes {
        let data = vec![0u8; block_size];

        group.throughput(Throughput::Elements(block_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate Rice decoding
                let mut decoded = Vec::with_capacity(data.len());
                for &byte in data {
                    decoded.push(i32::from(byte));
                }
                black_box(decoded);
            });
        });
    }

    group.finish();
}

/// Benchmark FLAC LPC prediction
fn flac_lpc_prediction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_lpc_prediction");

    let orders = vec![("order_4", 4), ("order_8", 8), ("order_12", 12)];

    for (name, order) in orders {
        let samples = vec![0i32; 4096];
        let coeffs = vec![1i32; order];

        group.throughput(Throughput::Elements(samples.len() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples, coeffs),
            |b, (samples, coeffs)| {
                b.iter(|| {
                    // Simulate LPC prediction
                    let mut output = vec![0i32; samples.len()];
                    for i in coeffs.len()..samples.len() {
                        let mut prediction = 0i32;
                        for (j, &coeff) in coeffs.iter().enumerate() {
                            prediction =
                                prediction.wrapping_add(samples[i - j - 1].wrapping_mul(coeff));
                        }
                        output[i] = prediction;
                    }
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Opus packet decoding
fn opus_packet_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_decode");
    group.measurement_time(Duration::from_secs(10));

    let frame_sizes = vec![
        ("2.5ms", 120), // 2.5ms @ 48kHz
        ("5ms", 240),   // 5ms @ 48kHz
        ("10ms", 480),  // 10ms @ 48kHz
        ("20ms", 960),  // 20ms @ 48kHz
        ("40ms", 1920), // 40ms @ 48kHz
    ];

    for (name, frame_size) in frame_sizes {
        let samples = vec![0f32; frame_size];

        group.throughput(Throughput::Elements(frame_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate Opus decoding
                black_box(samples);
            });
        });
    }

    group.finish();
}

/// Benchmark Opus range decoder
fn opus_range_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_range_decode");

    let data_sizes = vec![("small", 100), ("medium", 500), ("large", 1000)];

    for (name, size) in data_sizes {
        let data = vec![0xAAu8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &data, |b, data| {
            b.iter(|| {
                // Simulate range decoding
                let mut count = 0u32;
                for &byte in data {
                    count = count.wrapping_add(u32::from(byte));
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

/// Benchmark Opus SILK decoder
fn opus_silk_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_silk_decode");

    let frame_sizes = vec![("10ms", 480), ("20ms", 960)];

    for (name, frame_size) in frame_sizes {
        let samples = vec![0i16; frame_size];

        group.throughput(Throughput::Elements(frame_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate SILK decoding
                black_box(samples);
            });
        });
    }

    group.finish();
}

/// Benchmark Opus CELT decoder
fn opus_celt_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_celt_decode");

    let frame_sizes = vec![("2.5ms", 120), ("5ms", 240), ("10ms", 480)];

    for (name, frame_size) in frame_sizes {
        let samples = vec![0f32; frame_size];

        group.throughput(Throughput::Elements(frame_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate CELT decoding
                black_box(samples);
            });
        });
    }

    group.finish();
}

/// Benchmark Vorbis packet decoding
fn vorbis_packet_decode_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vorbis_decode");
    group.measurement_time(Duration::from_secs(10));

    let block_sizes = vec![("small_block", 256), ("large_block", 2048)];

    for (name, block_size) in block_sizes {
        let samples = vec![0f32; block_size];

        group.throughput(Throughput::Elements(block_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate Vorbis decoding
                black_box(samples);
            });
        });
    }

    group.finish();
}

/// Benchmark Vorbis MDCT (Modified Discrete Cosine Transform)
fn vorbis_mdct_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vorbis_mdct");

    let block_sizes = vec![
        ("128", 128),
        ("256", 256),
        ("512", 512),
        ("1024", 1024),
        ("2048", 2048),
    ];

    for (name, block_size) in block_sizes {
        let samples = vec![0f32; block_size];

        group.throughput(Throughput::Elements(block_size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                // Simulate MDCT
                let mut output = vec![0f32; samples.len()];
                for (i, &sample) in samples.iter().enumerate() {
                    output[i] = sample;
                }
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark PCM format conversion (int to float)
fn pcm_int_to_float_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcm_int_to_float");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_pcm_i16(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let output: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 32768.0).collect();
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark PCM format conversion (float to int)
fn pcm_float_to_int_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcm_float_to_int");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let output: Vec<i16> = samples.iter().map(|&s| (s * 32767.0) as i16).collect();
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark audio resampling (simple linear interpolation)
fn audio_resample_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_resample");
    group.measurement_time(Duration::from_secs(10));

    let conversions = vec![
        ("44.1k_to_48k", 44_100, 48_000),
        ("48k_to_44.1k", 48_000, 44_100),
        ("48k_to_96k", 48_000, 96_000),
        ("96k_to_48k", 96_000, 48_000),
    ];

    for (name, src_rate, dst_rate) in conversions {
        let samples = helpers::generate_audio_samples(src_rate, 1000, 440.0);

        group.throughput(Throughput::Elements(samples.len() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples, src_rate, dst_rate),
            |b, (samples, src_rate, dst_rate)| {
                b.iter(|| {
                    // Simple linear interpolation resampling
                    let ratio = *dst_rate as f32 / *src_rate as f32;
                    let output_len = (samples.len() as f32 * ratio) as usize;
                    let mut output = Vec::with_capacity(output_len);

                    for i in 0..output_len {
                        let src_pos = i as f32 / ratio;
                        let src_idx = src_pos as usize;
                        let frac = src_pos - src_idx as f32;

                        if src_idx + 1 < samples.len() {
                            let sample =
                                samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
                            output.push(sample);
                        }
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark audio mixing (sum two channels)
fn audio_mix_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_mix");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples1 = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples2 = helpers::generate_audio_samples(48_000, 1000, 880.0);
        let samples1 = samples1.into_iter().take(count).collect::<Vec<_>>();
        let samples2 = samples2.into_iter().take(count).collect::<Vec<_>>();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples1, samples2),
            |b, (samples1, samples2)| {
                b.iter(|| {
                    let output: Vec<f32> = samples1
                        .iter()
                        .zip(samples2.iter())
                        .map(|(&s1, &s2)| (s1 + s2) * 0.5)
                        .collect();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark audio volume adjustment
fn audio_volume_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_volume");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();
        let gain = 0.5f32;

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples, gain),
            |b, (samples, gain)| {
                b.iter(|| {
                    let output: Vec<f32> = samples.iter().map(|&s| s * gain).collect();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark simple EQ filter (biquad)
fn audio_eq_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_eq");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        // Biquad filter coefficients (low-pass)
        let b0 = 0.1f32;
        let b1 = 0.2f32;
        let b2 = 0.1f32;
        let a1 = -0.5f32;
        let a2 = 0.1f32;

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let mut output = vec![0f32; samples.len()];
                let mut x1 = 0f32;
                let mut x2 = 0f32;
                let mut y1 = 0f32;
                let mut y2 = 0f32;

                for (i, &sample) in samples.iter().enumerate() {
                    let y = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                    output[i] = y;

                    x2 = x1;
                    x1 = sample;
                    y2 = y1;
                    y1 = y;
                }

                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark audio compressor
fn audio_compressor_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_compressor");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        let threshold = 0.5f32;
        let ratio = 4.0f32;

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let output: Vec<f32> = samples
                    .iter()
                    .map(|&s| {
                        let abs_s = s.abs();
                        if abs_s > threshold {
                            let excess = abs_s - threshold;
                            let compressed = threshold + excess / ratio;
                            compressed * s.signum()
                        } else {
                            s
                        }
                    })
                    .collect();
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark audio limiter
fn audio_limiter_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_limiter");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        let limit = 0.9f32;

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let output: Vec<f32> = samples.iter().map(|&s| s.clamp(-limit, limit)).collect();
                black_box(output);
            });
        });
    }

    group.finish();
}

criterion_group!(
    audio_benches,
    flac_frame_decode_benchmark,
    flac_rice_decode_benchmark,
    flac_lpc_prediction_benchmark,
    opus_packet_decode_benchmark,
    opus_range_decode_benchmark,
    opus_silk_decode_benchmark,
    opus_celt_decode_benchmark,
    vorbis_packet_decode_benchmark,
    vorbis_mdct_benchmark,
    pcm_int_to_float_benchmark,
    pcm_float_to_int_benchmark,
    audio_resample_benchmark,
    audio_mix_benchmark,
    audio_volume_benchmark,
    audio_eq_benchmark,
    audio_compressor_benchmark,
    audio_limiter_benchmark,
);

criterion_main!(audio_benches);
