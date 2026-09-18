//! Comprehensive benchmarks comparing scirs2-vision with OpenCV
//!
//! This benchmark suite measures performance across various computer vision operations
//! to compare scirs2-vision's pure Rust implementation with OpenCV.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{DynamicImage, RgbImage};
use scirs2_vision::feature::{
    detect_and_compute_orb, detect_and_compute_surf, detect_surf, farneback_flow, tvl1_flow,
    FarnebackParams, OrbConfig, SurfConfig, TVL1Params,
};
use scirs2_vision::segmentation::{create_pascal_voc_classes, watershed, DeepLabV3Plus};
use scirs2_vision::{stereo_sgbm, SGBMParams};
use std::hint::black_box;

/// Create test image of given size
fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);

    // Create some pattern for more realistic benchmarking
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    DynamicImage::ImageRgb8(img)
}

/// Benchmark SURF feature detection
fn bench_surf_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("SURF Detection");

    for size in [256, 512, 1024].iter() {
        let img = create_test_image(*size, *size);
        let config = SurfConfig {
            hessian_threshold: 400.0,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("scirs2-vision", size), &img, |b, img| {
            b.iter(|| detect_surf(black_box(img), black_box(&config)).ok());
        });
    }

    group.finish();
}

/// Benchmark SURF descriptor computation
fn bench_surf_descriptors(c: &mut Criterion) {
    let mut group = c.benchmark_group("SURF Descriptors");

    for size in [256, 512].iter() {
        let img = create_test_image(*size, *size);
        let config = SurfConfig {
            hessian_threshold: 400.0,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("scirs2-vision", size), &img, |b, img| {
            b.iter(|| detect_and_compute_surf(black_box(img), black_box(&config)).ok());
        });
    }

    group.finish();
}

/// Benchmark ORB feature detection
fn bench_orb_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ORB Detection");

    for size in [256, 512, 1024].iter() {
        let img = create_test_image(*size, *size);
        let config = OrbConfig::default();

        group.bench_with_input(BenchmarkId::new("scirs2-vision", size), &img, |b, img| {
            b.iter(|| detect_and_compute_orb(black_box(img), black_box(&config)).ok());
        });
    }

    group.finish();
}

/// Benchmark Farneback optical flow
fn bench_farneback_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("Farneback Optical Flow");

    for size in [128, 256, 512].iter() {
        let img1 = create_test_image(*size, *size);
        let img2 = create_test_image(*size, *size);
        let params = FarnebackParams {
            levels: 3,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("scirs2-vision", size),
            &(&img1, &img2),
            |b, (img1, img2)| {
                b.iter(|| {
                    farneback_flow(black_box(img1), black_box(img2), black_box(&params)).ok()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark TVL1 optical flow
fn bench_tvl1_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("TVL1 Optical Flow");

    for size in [128, 256].iter() {
        let img1 = create_test_image(*size, *size);
        let img2 = create_test_image(*size, *size);
        let params = TVL1Params {
            scales: 3,
            warps: 3,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("scirs2-vision", size),
            &(&img1, &img2),
            |b, (img1, img2)| {
                b.iter(|| tvl1_flow(black_box(img1), black_box(img2), black_box(&params)).ok());
            },
        );
    }

    group.finish();
}

/// Benchmark watershed segmentation
fn bench_watershed(c: &mut Criterion) {
    let mut group = c.benchmark_group("Watershed Segmentation");

    for size in [256, 512].iter() {
        let img = create_test_image(*size, *size);

        group.bench_with_input(BenchmarkId::new("scirs2-vision", size), &img, |b, img| {
            b.iter(|| watershed(black_box(img), None, 8).ok());
        });
    }

    group.finish();
}

/// Benchmark semantic segmentation (DeepLabV3+)
fn bench_semantic_segmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Semantic Segmentation");

    let classes = create_pascal_voc_classes();
    let model = DeepLabV3Plus::new(21, (256, 256), classes);

    for size in [256, 512].iter() {
        let img = create_test_image(*size, *size);

        group.bench_with_input(BenchmarkId::new("DeepLabV3+", size), &img, |b, img| {
            b.iter(|| model.segment(black_box(img)).ok());
        });
    }

    group.finish();
}

/// Benchmark stereo matching (SGBM)
fn bench_stereo_sgbm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stereo SGBM");

    for size in [256, 512].iter() {
        let left = create_test_image(*size, *size);
        let right = create_test_image(*size, *size);
        let params = SGBMParams {
            num_disparities: 64,
            block_size: 5,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("scirs2-vision", size),
            &(&left, &right),
            |b, (left, right)| {
                b.iter(|| stereo_sgbm(black_box(left), black_box(right), black_box(&params)).ok());
            },
        );
    }

    group.finish();
}

/// Benchmark summary - Feature Detection Comparison
fn bench_feature_detection_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("Feature Detection Comparison");
    let img = create_test_image(512, 512);

    // SURF
    group.bench_function("SURF", |b| {
        let config = SurfConfig::default();
        b.iter(|| detect_surf(black_box(&img), black_box(&config)).ok());
    });

    // ORB
    group.bench_function("ORB", |b| {
        let config = OrbConfig::default();
        b.iter(|| detect_and_compute_orb(black_box(&img), black_box(&config)).ok());
    });

    group.finish();
}

/// Benchmark summary - Optical Flow Comparison
fn bench_optical_flow_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optical Flow Comparison");
    let img1 = create_test_image(256, 256);
    let img2 = create_test_image(256, 256);

    // Farneback
    group.bench_function("Farneback", |b| {
        let params = FarnebackParams::default();
        b.iter(|| farneback_flow(black_box(&img1), black_box(&img2), black_box(&params)).ok());
    });

    // TVL1
    group.bench_function("TVL1", |b| {
        let params = TVL1Params {
            scales: 3,
            ..Default::default()
        };
        b.iter(|| tvl1_flow(black_box(&img1), black_box(&img2), black_box(&params)).ok());
    });

    group.finish();
}

/// Overall performance metrics
fn bench_overall_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Overall Performance");

    let img_small = create_test_image(256, 256);
    let img_medium = create_test_image(512, 512);
    let img_large = create_test_image(1024, 1024);

    // Test different image sizes
    group.bench_function("Feature Detection (256x256)", |b| {
        let config = SurfConfig::default();
        b.iter(|| detect_surf(black_box(&img_small), black_box(&config)).ok());
    });

    group.bench_function("Feature Detection (512x512)", |b| {
        let config = SurfConfig::default();
        b.iter(|| detect_surf(black_box(&img_medium), black_box(&config)).ok());
    });

    group.bench_function("Feature Detection (1024x1024)", |b| {
        let config = SurfConfig::default();
        b.iter(|| detect_surf(black_box(&img_large), black_box(&config)).ok());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_surf_detection,
    bench_surf_descriptors,
    bench_orb_detection,
    bench_farneback_flow,
    bench_tvl1_flow,
    bench_watershed,
    bench_semantic_segmentation,
    bench_stereo_sgbm,
    bench_feature_detection_summary,
    bench_optical_flow_summary,
    bench_overall_performance,
);

criterion_main!(benches);
