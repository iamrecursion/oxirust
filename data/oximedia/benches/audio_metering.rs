//! Audio metering benchmarks — EBU R128 loudness and VU peak metering.
//!
//! Measures throughput of the ITU-R BS.1770-4 / EBU R128 K-weighted loudness
//! measurement pipeline at various buffer sizes (256 – 16 384 samples).

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_metering::{LoudnessMeter, MeterConfig, PeakMeter, PeakMeterType, Standard};
use std::f32::consts::PI;
use std::hint::black_box;

/// Generate interleaved stereo sine-wave samples at the given buffer length.
/// Channel 1 (left): 440 Hz, Channel 2 (right): 880 Hz.
/// Amplitude is -18 dBFS (≈ 0.126) so it falls within EBU R128 gate range.
fn generate_stereo_sine(frame_samples: usize) -> Vec<f32> {
    let sample_rate = 48_000_f32;
    let amp = 10_f32.powf(-18.0 / 20.0); // -18 dBFS
    let mut out = Vec::with_capacity(frame_samples * 2);
    for i in 0..frame_samples {
        let t = i as f32 / sample_rate;
        out.push(amp * (2.0 * PI * 440.0 * t).sin()); // L
        out.push(amp * (2.0 * PI * 880.0 * t).sin()); // R
    }
    out
}

// ---------------------------------------------------------------------------
// LoudnessMeter benchmarks
// ---------------------------------------------------------------------------

fn bench_loudness_meter_process(c: &mut Criterion) {
    let buffer_sizes: &[usize] = &[256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("loudness_meter_ebu_r128");

    for &frame_samples in buffer_sizes {
        let samples = generate_stereo_sine(frame_samples);
        // Samples per channel (each benchmark call processes `frame_samples` stereo frames)
        group.throughput(Throughput::Elements(frame_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("stereo_48kHz", frame_samples),
            &samples,
            |b, samples| {
                // Re-create meter each bench run so internal state is consistent.
                let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
                let mut meter =
                    LoudnessMeter::new(config).expect("bench setup: LoudnessMeter::new");
                b.iter(|| {
                    meter.process_f32(black_box(samples));
                    black_box(meter.metrics())
                });
            },
        );
    }

    group.finish();
}

fn bench_loudness_meter_large_buffer(c: &mut Criterion) {
    // Focus on throughput with a fixed 1-second block (48 000 stereo frames).
    let mut group = c.benchmark_group("loudness_meter_1sec_block");
    let sample_rate = 48_000_usize;
    let samples = generate_stereo_sine(sample_rate);

    // Measure bytes throughput: f32 × 2 channels × sample_rate samples
    let byte_count = (std::mem::size_of::<f32>() * 2 * sample_rate) as u64;
    group.throughput(Throughput::Bytes(byte_count));

    let standards: &[(&str, Standard)] = &[
        ("ebu_r128", Standard::EbuR128),
        ("atsc_a85", Standard::AtscA85),
        ("spotify", Standard::Spotify),
    ];

    for &(name, standard) in standards {
        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            let config = MeterConfig::new(standard, 48_000.0, 2);
            let mut meter =
                LoudnessMeter::new(config).expect("bench setup: LoudnessMeter::new standard");
            b.iter(|| {
                meter.process_f32(black_box(samples));
                black_box(meter.metrics())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// PeakMeter benchmarks
// ---------------------------------------------------------------------------

fn bench_peak_meter_vu(c: &mut Criterion) {
    let buffer_sizes: &[usize] = &[256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("peak_meter_vu");

    for &frame_samples in buffer_sizes {
        let samples = generate_stereo_sine(frame_samples);
        group.throughput(Throughput::Elements(frame_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("stereo_48kHz", frame_samples),
            &samples,
            |b, samples| {
                // Convert to f64 slice for PeakMeter (it accepts f64 interleaved)
                let f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
                let mut meter = PeakMeter::new(PeakMeterType::Vu, 48_000.0, 2, 0.3)
                    .expect("bench setup: PeakMeter::new VU");
                b.iter(|| {
                    meter.process_interleaved(black_box(&f64_samples));
                    black_box(meter.peak_dbfs())
                });
            },
        );
    }

    group.finish();
}

fn bench_peak_meter_rms(c: &mut Criterion) {
    let buffer_sizes: &[usize] = &[256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("peak_meter_rms");

    for &frame_samples in buffer_sizes {
        let samples = generate_stereo_sine(frame_samples);
        group.throughput(Throughput::Elements(frame_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("stereo_48kHz", frame_samples),
            &samples,
            |b, samples| {
                let f64_samples: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
                // RMS with 300 ms integration
                let mut meter = PeakMeter::new(PeakMeterType::Rms(0.3), 48_000.0, 2, 0.3)
                    .expect("bench setup: PeakMeter::new RMS");
                b.iter(|| {
                    meter.process_interleaved(black_box(&f64_samples));
                    black_box(meter.peak_dbfs())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    metering_benches,
    bench_loudness_meter_process,
    bench_loudness_meter_large_buffer,
    bench_peak_meter_vu,
    bench_peak_meter_rms,
);
criterion_main!(metering_benches);
