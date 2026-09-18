//! Criterion speed benchmarks for the full-reference quality metrics.
//!
//! These benchmarks measure the *throughput* of PSNR, SSIM, and VMAF across a
//! couple of representative broadcast resolutions.  They are intentionally
//! **structural only** — there are no timing assertions; criterion itself
//! reports the wall-clock numbers, and CI merely needs the benches to *build*
//! and *run* without panicking.
//!
//! Accuracy (known-answer) coverage lives separately in `src/golden_tests.rs`
//! (20+ analytic tests), so this file deliberately does not re-check numerical
//! correctness.
//!
//! Run with `cargo bench -p oximedia-quality`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, PsnrCalculator, SsimCalculator, VmafCalculator};
use std::hint::black_box;

/// Resolutions exercised by every metric benchmark group.
///
/// Kept modest so the suite stays CI-friendly while still spanning a ~4× pixel
/// range (640×360 → 1280×720).  Both are comfortably above the 128 px floor
/// required by multi-scale metrics, should one be added later.
const RESOLUTIONS: &[(usize, usize)] = &[(640, 360), (1280, 720)];

/// Builds a deterministic YUV420p reference frame.
///
/// The Y plane is a smooth diagonal gradient (`(x + y)` wrapped into `[0,255]`)
/// so the metrics see real spatial structure rather than a degenerate flat
/// plane; the chroma planes carry a fixed offset so the chroma branch of each
/// weighted metric is genuinely exercised.
fn reference_frame(width: usize, height: usize) -> Frame {
    let mut f = Frame::new(width, height, PixelFormat::Yuv420p)
        .expect("YUV420p frame allocation must succeed in bench");

    for row in 0..height {
        for col in 0..width {
            f.planes[0][row * width + col] = ((col + row) % 256) as u8;
        }
    }

    // Chroma planes are subsampled (width/2 × height/2 for YUV420p); fill the
    // whole buffer with a constant so they are well-defined regardless of the
    // exact stride/size the allocator chose.
    f.planes[1].fill(110);
    f.planes[2].fill(140);
    f
}

/// Builds the distorted counterpart: the reference with a uniform +5 luma
/// offset (saturating), leaving chroma untouched.  This yields a finite,
/// non-degenerate per-plane MSE for every metric.
fn distorted_frame(reference: &Frame) -> Frame {
    let mut d = reference.clone();
    for p in &mut d.planes[0] {
        *p = p.saturating_add(5);
    }
    d
}

/// Returns the `(reference, distorted)` pair for a resolution.
fn frame_pair(width: usize, height: usize) -> (Frame, Frame) {
    let reference = reference_frame(width, height);
    let distorted = distorted_frame(&reference);
    (reference, distorted)
}

fn bench_psnr(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_psnr");
    let calc = PsnrCalculator::new();
    for &(w, h) in RESOLUTIONS {
        let (reference, distorted) = frame_pair(w, h);
        group.bench_with_input(
            BenchmarkId::new("psnr", format!("{w}x{h}")),
            &(w, h),
            |b, _| {
                b.iter(|| {
                    black_box(
                        calc.calculate(black_box(&reference), black_box(&distorted))
                            .expect("psnr calculate must succeed in bench"),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_ssim");
    let calc = SsimCalculator::new();
    for &(w, h) in RESOLUTIONS {
        let (reference, distorted) = frame_pair(w, h);
        group.bench_with_input(
            BenchmarkId::new("ssim", format!("{w}x{h}")),
            &(w, h),
            |b, _| {
                b.iter(|| {
                    black_box(
                        calc.calculate(black_box(&reference), black_box(&distorted))
                            .expect("ssim calculate must succeed in bench"),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_vmaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_vmaf");
    let calc = VmafCalculator::new();
    for &(w, h) in RESOLUTIONS {
        let (reference, distorted) = frame_pair(w, h);
        group.bench_with_input(
            BenchmarkId::new("vmaf", format!("{w}x{h}")),
            &(w, h),
            |b, _| {
                b.iter(|| {
                    black_box(
                        calc.calculate(black_box(&reference), black_box(&distorted))
                            .expect("vmaf calculate must succeed in bench"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_psnr, bench_ssim, bench_vmaf);
criterion_main!(benches);
