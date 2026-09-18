//! Quality metrics benchmarks (PSNR, SSIM) at various frame sizes.
//!
//! Measures the throughput of full-reference quality assessment for
//! the most common broadcast frame sizes.

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::hint::black_box;

/// Fill a frame's luma plane with a gradient and chroma with constant mid-gray.
/// Used to produce a synthetic "reference" frame.
fn make_reference_frame(width: usize, height: usize) -> Frame {
    let mut frame =
        Frame::new(width, height, PixelFormat::Yuv420p).expect("bench setup: Frame::new reference");

    // Luma: diagonal gradient
    for y in 0..height {
        for x in 0..width {
            frame.planes[0][y * width + x] = ((x.wrapping_add(y)) % 256) as u8;
        }
    }

    // Chroma planes: constant mid-gray (neutral color)
    let uv_w = width / 2;
    let uv_h = height / 2;
    for i in 0..(uv_w * uv_h) {
        frame.planes[1][i] = 128;
        frame.planes[2][i] = 128;
    }

    frame
}

/// Produce a "distorted" frame by offsetting luma by a fixed amount.
fn make_distorted_frame(reference: &Frame) -> Frame {
    let mut distorted = reference.clone();
    for byte in &mut distorted.planes[0] {
        *byte = byte.wrapping_add(8);
    }
    distorted
}

fn bench_psnr(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(64, 64), (320, 240), (1280, 720), (1920, 1080)];
    let assessor = QualityAssessor::new();
    let mut group = c.benchmark_group("psnr");

    for &(w, h) in sizes {
        let reference = make_reference_frame(w, h);
        let distorted = make_distorted_frame(&reference);
        let pixel_count = (w * h) as u64;

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_with_input(
            BenchmarkId::new("yuv420p", format!("{w}x{h}")),
            &(&reference, &distorted),
            |b, &(ref_frame, dist_frame)| {
                b.iter(|| {
                    let score = assessor
                        .assess(
                            black_box(ref_frame),
                            black_box(dist_frame),
                            MetricType::Psnr,
                        )
                        .expect("bench: psnr assess");
                    black_box(score)
                });
            },
        );
    }

    group.finish();
}

fn bench_ssim(c: &mut Criterion) {
    let sizes: &[(usize, usize)] = &[(64, 64), (320, 240), (1280, 720), (1920, 1080)];
    let assessor = QualityAssessor::new();
    let mut group = c.benchmark_group("ssim");

    for &(w, h) in sizes {
        let reference = make_reference_frame(w, h);
        let distorted = make_distorted_frame(&reference);
        let pixel_count = (w * h) as u64;

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_with_input(
            BenchmarkId::new("yuv420p", format!("{w}x{h}")),
            &(&reference, &distorted),
            |b, &(ref_frame, dist_frame)| {
                b.iter(|| {
                    let score = assessor
                        .assess(
                            black_box(ref_frame),
                            black_box(dist_frame),
                            MetricType::Ssim,
                        )
                        .expect("bench: ssim assess");
                    black_box(score)
                });
            },
        );
    }

    group.finish();
}

fn bench_psnr_luma_only(c: &mut Criterion) {
    // Benchmark PSNR on luma-only (Gray8) frames to isolate Y-plane cost.
    let sizes: &[(usize, usize)] = &[(320, 240), (1280, 720), (1920, 1080)];
    let assessor = QualityAssessor::new();
    let mut group = c.benchmark_group("psnr_luma");

    for &(w, h) in sizes {
        let reference = {
            let mut f = Frame::new(w, h, PixelFormat::Gray8)
                .expect("bench setup: Frame::new gray reference");
            for y in 0..h {
                for x in 0..w {
                    f.planes[0][y * w + x] = ((x.wrapping_add(y)) % 256) as u8;
                }
            }
            f
        };
        let distorted = {
            let mut f = reference.clone();
            for byte in &mut f.planes[0] {
                *byte = byte.wrapping_add(16);
            }
            f
        };
        let pixel_count = (w * h) as u64;

        group.throughput(Throughput::Elements(pixel_count));
        group.bench_with_input(
            BenchmarkId::new("gray8", format!("{w}x{h}")),
            &(&reference, &distorted),
            |b, &(ref_frame, dist_frame)| {
                b.iter(|| {
                    let score = assessor
                        .assess(
                            black_box(ref_frame),
                            black_box(dist_frame),
                            MetricType::Psnr,
                        )
                        .expect("bench: psnr luma assess");
                    black_box(score)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    quality_benches,
    bench_psnr,
    bench_ssim,
    bench_psnr_luma_only
);
criterion_main!(quality_benches);
