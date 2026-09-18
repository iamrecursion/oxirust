//! Video and audio filter benchmarks
//!
//! This benchmark suite measures the performance of various filters:
//! - Video filters: scaling, color conversion, deinterlacing
//! - Audio filters: resampling, EQ, compression, volume
//!
//! These benchmarks help measure the performance impact of filter operations
//! in real-time processing pipelines.

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;

// ============================================================================
// Video Scaling Benchmarks
// ============================================================================

/// Benchmark bilinear scaling
fn video_scale_bilinear_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_scale_bilinear");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let scale_operations = vec![
        ("720p_to_480p", 1280, 720, 854, 480),
        ("1080p_to_720p", 1920, 1080, 1280, 720),
        ("4K_to_1080p", 3840, 2160, 1920, 1080),
        ("480p_to_1080p", 854, 480, 1920, 1080),
        ("720p_to_4K", 1280, 720, 3840, 2160),
    ];

    for (name, src_w, src_h, dst_w, dst_h) in scale_operations {
        let src_frame = helpers::generate_yuv420_frame(src_w, src_h);

        group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(src_frame, src_w, src_h, dst_w, dst_h),
            |b, (src_frame, src_w, src_h, dst_w, dst_h)| {
                b.iter(|| {
                    // Simulate bilinear scaling
                    let mut dst = Vec::with_capacity((dst_w * dst_h * 3) / 2);

                    let x_ratio = *src_w as f32 / *dst_w as f32;
                    let y_ratio = *src_h as f32 / *dst_h as f32;

                    for y in 0..*dst_h {
                        for x in 0..*dst_w {
                            let src_x = (x as f32 * x_ratio) as usize;
                            let src_y = (y as f32 * y_ratio) as usize;
                            let idx = src_y * src_w + src_x;
                            if idx < src_frame.len() {
                                dst.push(src_frame[idx]);
                            }
                        }
                    }

                    black_box(dst);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bicubic scaling
fn video_scale_bicubic_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_scale_bicubic");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    let scale_operations = vec![
        ("720p_to_480p", 1280, 720, 854, 480),
        ("1080p_to_720p", 1920, 1080, 1280, 720),
        ("480p_to_1080p", 854, 480, 1920, 1080),
    ];

    for (name, src_w, src_h, dst_w, dst_h) in scale_operations {
        let src_frame = helpers::generate_yuv420_frame(src_w, src_h);

        group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(src_frame, src_w, src_h, dst_w, dst_h),
            |b, (src_frame, src_w, src_h, dst_w, dst_h)| {
                b.iter(|| {
                    // Simulate bicubic scaling (simplified)
                    let mut dst = Vec::with_capacity((dst_w * dst_h * 3) / 2);

                    let x_ratio = *src_w as f32 / *dst_w as f32;
                    let y_ratio = *src_h as f32 / *dst_h as f32;

                    for y in 0..*dst_h {
                        for x in 0..*dst_w {
                            let src_x = (x as f32 * x_ratio) as usize;
                            let src_y = (y as f32 * y_ratio) as usize;

                            // Sample a 4x4 neighborhood for bicubic
                            let mut samples = Vec::new();
                            for dy in 0..4 {
                                for dx in 0..4 {
                                    let sy = (src_y + dy).min(src_h - 1);
                                    let sx = (src_x + dx).min(src_w - 1);
                                    let idx = sy * src_w + sx;
                                    if idx < src_frame.len() {
                                        samples.push(src_frame[idx]);
                                    }
                                }
                            }

                            let avg = if !samples.is_empty() {
                                let sum: u32 = samples.iter().map(|&x| u32::from(x)).sum();
                                (sum / samples.len() as u32) as u8
                            } else {
                                0
                            };

                            dst.push(avg);
                        }
                    }

                    black_box(dst);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Lanczos scaling
fn video_scale_lanczos_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_scale_lanczos");
    group.measurement_time(Duration::from_secs(25));
    group.sample_size(10);

    let scale_operations = vec![
        ("720p_to_480p", 1280, 720, 854, 480),
        ("1080p_to_720p", 1920, 1080, 1280, 720),
    ];

    for (name, src_w, src_h, dst_w, dst_h) in scale_operations {
        let src_frame = helpers::generate_yuv420_frame(src_w, src_h);

        group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(src_frame, src_w, src_h, dst_w, dst_h),
            |b, (src_frame, src_w, src_h, dst_w, dst_h)| {
                b.iter(|| {
                    // Simulate Lanczos scaling (simplified)
                    let mut dst = Vec::with_capacity((dst_w * dst_h * 3) / 2);

                    let x_ratio = *src_w as f32 / *dst_w as f32;
                    let y_ratio = *src_h as f32 / *dst_h as f32;

                    for y in 0..*dst_h {
                        for x in 0..*dst_w {
                            let src_x = (x as f32 * x_ratio) as usize;
                            let src_y = (y as f32 * y_ratio) as usize;

                            // Sample a larger neighborhood for Lanczos
                            let mut sum = 0u32;
                            let mut count = 0u32;

                            for dy in 0..6 {
                                for dx in 0..6 {
                                    let sy = (src_y + dy).min(src_h - 1);
                                    let sx = (src_x + dx).min(src_w - 1);
                                    let idx = sy * src_w + sx;
                                    if idx < src_frame.len() {
                                        sum += u32::from(src_frame[idx]);
                                        count += 1;
                                    }
                                }
                            }

                            let avg = sum.checked_div(count).unwrap_or(0) as u8;
                            dst.push(avg);
                        }
                    }

                    black_box(dst);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Color Conversion Benchmarks
// ============================================================================

/// Benchmark YUV to RGB conversion
fn color_yuv_to_rgb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_yuv_to_rgb");
    group.measurement_time(Duration::from_secs(15));

    let resolutions = vec![
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("4K", 3840, 2160),
    ];

    for (name, width, height) in resolutions {
        let yuv_frame = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(yuv_frame, width, height),
            |b, (yuv_frame, width, height)| {
                b.iter(|| {
                    // Simulate YUV to RGB conversion
                    let mut rgb = Vec::with_capacity(width * height * 3);

                    let y_size = width * height;

                    for i in 0..y_size {
                        let y = i32::from(yuv_frame[i]);
                        let u_idx = y_size + (i / 2) % (y_size / 4);
                        let v_idx = y_size + y_size / 4 + (i / 2) % (y_size / 4);

                        let u = i32::from(yuv_frame[u_idx.min(yuv_frame.len() - 1)]) - 128;
                        let v = i32::from(yuv_frame[v_idx.min(yuv_frame.len() - 1)]) - 128;

                        let r = (y + ((1436 * v) >> 10)).clamp(0, 255) as u8;
                        let g = (y - ((352 * u + 731 * v) >> 10)).clamp(0, 255) as u8;
                        let b = (y + ((1814 * u) >> 10)).clamp(0, 255) as u8;

                        rgb.push(r);
                        rgb.push(g);
                        rgb.push(b);
                    }

                    black_box(rgb);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RGB to YUV conversion
fn color_rgb_to_yuv_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_rgb_to_yuv");
    group.measurement_time(Duration::from_secs(15));

    let resolutions = vec![
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
        ("4K", 3840, 2160),
    ];

    for (name, width, height) in resolutions {
        let rgb_frame = helpers::generate_rgb_frame(width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(rgb_frame, width, height),
            |b, (rgb_frame, width, height)| {
                b.iter(|| {
                    // Simulate RGB to YUV conversion
                    let y_size = width * height;
                    let mut yuv = Vec::with_capacity(y_size + y_size / 2);

                    for i in 0..*width * *height {
                        let r = i32::from(rgb_frame[i * 3]);
                        let g = i32::from(rgb_frame[i * 3 + 1]);
                        let b = i32::from(rgb_frame[i * 3 + 2]);

                        let y = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
                        yuv.push(y);
                    }

                    // Subsample U and V
                    for y in (0..*height).step_by(2) {
                        for x in (0..*width).step_by(2) {
                            let idx = (y * width + x) * 3;
                            if idx + 2 < rgb_frame.len() {
                                let r = i32::from(rgb_frame[idx]);
                                let g = i32::from(rgb_frame[idx + 1]);
                                let b = i32::from(rgb_frame[idx + 2]);

                                let u =
                                    (((-43 * r - 85 * g + 128 * b) >> 8) + 128).clamp(0, 255) as u8;
                                let v =
                                    (((128 * r - 107 * g - 21 * b) >> 8) + 128).clamp(0, 255) as u8;

                                yuv.push(u);
                                yuv.push(v);
                            }
                        }
                    }

                    black_box(yuv);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark YUV420 to YUV444 conversion
fn color_yuv420_to_yuv444_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_yuv420_to_yuv444");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let yuv420 = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(yuv420, width, height),
            |b, (yuv420, width, height)| {
                b.iter(|| {
                    // Simulate YUV420 to YUV444 upsampling
                    let y_size = width * height;
                    let mut yuv444 = Vec::with_capacity(y_size * 3);

                    // Copy Y plane
                    yuv444.extend_from_slice(&yuv420[0..y_size]);

                    // Upsample U and V planes
                    for i in 0..y_size {
                        let u_idx = y_size + (i / 2) % (y_size / 4);
                        let v_idx = y_size + y_size / 4 + (i / 2) % (y_size / 4);

                        yuv444.push(yuv420[u_idx.min(yuv420.len() - 1)]);
                        yuv444.push(yuv420[v_idx.min(yuv420.len() - 1)]);
                    }

                    black_box(yuv444);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Deinterlacing Benchmarks
// ============================================================================

/// Benchmark bob deinterlacing (line doubling)
fn deinterlace_bob_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterlace_bob");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("480i", 720, 480),
        ("576i", 720, 576),
        ("1080i", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let interlaced_frame = helpers::generate_yuv420_frame(width, height / 2);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(interlaced_frame, width, height),
            |b, (interlaced_frame, width, height)| {
                let width = *width;
                let height = *height;
                b.iter(|| {
                    // Simulate bob deinterlacing
                    let mut progressive = Vec::with_capacity(width * height);

                    for line in interlaced_frame.chunks(width) {
                        progressive.extend_from_slice(line);
                        progressive.extend_from_slice(line); // Double lines
                    }

                    black_box(progressive);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark weave deinterlacing
fn deinterlace_weave_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterlace_weave");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("480i", 720, 480),
        ("576i", 720, 576),
        ("1080i", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let field1 = helpers::generate_yuv420_frame(width, height / 2);
        let field2 = helpers::generate_yuv420_frame(width, height / 2);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(field1, field2, width),
            |b, (field1, field2, width)| {
                let width = *width;
                b.iter(|| {
                    // Simulate weave deinterlacing
                    let mut progressive = Vec::with_capacity(field1.len() + field2.len());

                    let lines1 = field1.chunks(width);
                    let lines2 = field2.chunks(width);

                    for (line1, line2) in lines1.zip(lines2) {
                        progressive.extend_from_slice(line1);
                        progressive.extend_from_slice(line2);
                    }

                    black_box(progressive);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark motion-adaptive deinterlacing
fn deinterlace_motion_adaptive_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterlace_motion_adaptive");
    group.measurement_time(Duration::from_secs(15));

    let resolutions = vec![("480i", 720, 480), ("720p", 1280, 720)];

    for (name, width, height) in resolutions {
        let field1 = helpers::generate_yuv420_frame(width, height / 2);
        let field2 = helpers::generate_yuv420_frame(width, height / 2);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(field1, field2, width),
            |b, (field1, field2, width)| {
                let width = *width;
                b.iter(|| {
                    // Simulate motion-adaptive deinterlacing
                    let mut progressive = Vec::with_capacity(field1.len() + field2.len());

                    let lines1 = field1.chunks(width);
                    let lines2 = field2.chunks(width);

                    for (line1, line2) in lines1.zip(lines2) {
                        // Calculate motion between fields
                        let motion: u32 = line1
                            .iter()
                            .zip(line2.iter())
                            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
                            .sum();

                        let avg_motion = motion / width as u32;

                        if avg_motion > 10 {
                            // High motion: use bob
                            progressive.extend_from_slice(line1);
                            progressive.extend_from_slice(line1);
                        } else {
                            // Low motion: use weave
                            progressive.extend_from_slice(line1);
                            progressive.extend_from_slice(line2);
                        }
                    }

                    black_box(progressive);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Audio Resampling Benchmarks
// ============================================================================

/// Benchmark linear audio resampling
fn audio_resample_linear_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_resample_linear");
    group.measurement_time(Duration::from_secs(10));

    let conversions = vec![
        ("44.1k_to_48k", 44_100, 48_000),
        ("48k_to_44.1k", 48_000, 44_100),
        ("48k_to_96k", 48_000, 96_000),
        ("96k_to_48k", 96_000, 48_000),
        ("44.1k_to_96k", 44_100, 96_000),
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

/// Benchmark sinc audio resampling
fn audio_resample_sinc_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_resample_sinc");
    group.measurement_time(Duration::from_secs(20));

    let conversions = vec![
        ("44.1k_to_48k", 44_100, 48_000),
        ("48k_to_96k", 48_000, 96_000),
    ];

    for (name, src_rate, dst_rate) in conversions {
        let samples = helpers::generate_audio_samples(src_rate, 1000, 440.0);

        group.throughput(Throughput::Elements(samples.len() as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples, src_rate, dst_rate),
            |b, (samples, src_rate, dst_rate)| {
                b.iter(|| {
                    // Simulate sinc resampling (windowed)
                    let ratio = *dst_rate as f32 / *src_rate as f32;
                    let output_len = (samples.len() as f32 * ratio) as usize;
                    let mut output = Vec::with_capacity(output_len);

                    let kernel_size = 32;

                    for i in 0..output_len {
                        let src_pos = i as f32 / ratio;
                        let center = src_pos as usize;

                        let mut sum = 0.0f32;
                        let mut weight_sum = 0.0f32;

                        for k in 0..kernel_size {
                            let idx = center.saturating_sub(kernel_size / 2) + k;
                            if idx < samples.len() {
                                let x = src_pos - idx as f32;
                                let weight = if x.abs() < 0.001 {
                                    1.0
                                } else {
                                    let pi_x = std::f32::consts::PI * x;
                                    pi_x.sin() / pi_x
                                };

                                sum += samples[idx] * weight;
                                weight_sum += weight;
                            }
                        }

                        output.push(if weight_sum > 0.0 {
                            sum / weight_sum
                        } else {
                            0.0
                        });
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Audio Filter Benchmarks
// ============================================================================

/// Benchmark parametric EQ filter
fn audio_eq_parametric_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_eq_parametric");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        // 3-band EQ (low, mid, high)
        let filters = vec![
            (0.1f32, 0.2f32, 0.1f32, -0.5f32, 0.1f32),    // Low
            (0.2f32, 0.4f32, 0.2f32, -0.6f32, 0.2f32),    // Mid
            (0.15f32, 0.3f32, 0.15f32, -0.4f32, 0.15f32), // High
        ];

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let mut output = samples.clone();

                // Apply each band
                for (b0, b1, b2, a1, a2) in &filters {
                    let mut x1 = 0f32;
                    let mut x2 = 0f32;
                    let mut y1 = 0f32;
                    let mut y2 = 0f32;

                    for sample in &mut output {
                        let x0 = *sample;
                        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                        *sample = y0;

                        x2 = x1;
                        x1 = x0;
                        y2 = y1;
                        y1 = y0;
                    }
                }

                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark dynamic range compression
fn audio_compressor_dynamic_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_compressor_dynamic");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        let threshold = 0.5f32;
        let ratio = 4.0f32;
        let attack = 0.001f32; // seconds
        let release = 0.1f32; // seconds

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let mut output = Vec::with_capacity(samples.len());
                let mut envelope = 0.0f32;

                let attack_coeff = (-1.0 / (attack * 48000.0)).exp();
                let release_coeff = (-1.0 / (release * 48000.0)).exp();

                for &sample in samples {
                    let abs_sample = sample.abs();

                    // Envelope follower
                    if abs_sample > envelope {
                        envelope = attack_coeff * envelope + (1.0 - attack_coeff) * abs_sample;
                    } else {
                        envelope = release_coeff * envelope + (1.0 - release_coeff) * abs_sample;
                    }

                    // Apply compression
                    let gain = if envelope > threshold {
                        let excess = envelope - threshold;
                        threshold + excess / ratio
                    } else {
                        envelope
                    };

                    let gain_factor = if envelope > 0.0 { gain / envelope } else { 1.0 };
                    output.push(sample * gain_factor);
                }

                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark audio limiter
fn audio_limiter_lookahead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_limiter_lookahead");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        let limit = 0.9f32;
        let lookahead_samples = 480; // 10ms at 48kHz

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &samples, |b, samples| {
            b.iter(|| {
                let mut output = Vec::with_capacity(samples.len());
                let mut buffer = vec![0.0f32; lookahead_samples];
                let mut write_pos = 0;

                for &sample in samples.iter() {
                    buffer[write_pos] = sample;
                    write_pos = (write_pos + 1) % lookahead_samples;

                    // Look ahead to find peak
                    let mut peak = 0.0f32;
                    for &buffered in &buffer {
                        peak = peak.max(buffered.abs());
                    }

                    // Apply limiting
                    let gain = if peak > limit { limit / peak } else { 1.0 };

                    let read_pos = write_pos;
                    output.push(buffer[read_pos] * gain);
                }

                black_box(output);
            });
        });
    }

    group.finish();
}

/// Benchmark volume automation
fn audio_volume_automation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_volume_automation");

    let sample_counts = vec![("1k", 1024), ("10k", 10_240), ("100k", 102_400)];

    for (name, count) in sample_counts {
        let samples = helpers::generate_audio_samples(48_000, 1000, 440.0);
        let samples = samples.into_iter().take(count).collect::<Vec<_>>();

        // Generate automation curve (fade in/out)
        let mut automation = Vec::with_capacity(samples.len());
        for i in 0..samples.len() {
            let t = i as f32 / samples.len() as f32;
            let gain = if t < 0.25 {
                t * 4.0 // Fade in
            } else if t > 0.75 {
                (1.0 - t) * 4.0 // Fade out
            } else {
                1.0 // Full volume
            };
            automation.push(gain);
        }

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(samples, automation),
            |b, (samples, automation)| {
                b.iter(|| {
                    let output: Vec<f32> = samples
                        .iter()
                        .zip(automation.iter())
                        .map(|(&s, &gain)| s * gain)
                        .collect();
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    filter_benches,
    // Video scaling
    video_scale_bilinear_benchmark,
    video_scale_bicubic_benchmark,
    video_scale_lanczos_benchmark,
    // Color conversion
    color_yuv_to_rgb_benchmark,
    color_rgb_to_yuv_benchmark,
    color_yuv420_to_yuv444_benchmark,
    // Deinterlacing
    deinterlace_bob_benchmark,
    deinterlace_weave_benchmark,
    deinterlace_motion_adaptive_benchmark,
    // Audio resampling
    audio_resample_linear_benchmark,
    audio_resample_sinc_benchmark,
    // Audio filters
    audio_eq_parametric_benchmark,
    audio_compressor_dynamic_benchmark,
    audio_limiter_lookahead_benchmark,
    audio_volume_automation_benchmark,
);

criterion_main!(filter_benches);
