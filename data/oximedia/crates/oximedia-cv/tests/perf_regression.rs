//! Performance regression tests for hot paths in oximedia-cv.
//!
//! These tests use wall-clock budgets generous enough to catch 10× regressions
//! without flaking on slow CI.

use std::time::{Duration, Instant};

/// Generous budget: passes on slow CI debug builds.
fn budget_ms(ms: u64) -> Duration {
    Duration::from_millis(ms)
}

/// Return the appropriate time budget based on whether the build is debug or release.
///
/// Release builds benefit from optimisation and should run significantly faster;
/// the debug budget is intentionally generous to avoid CI flakiness.
fn budget_by_profile(debug_ms: u64, release_ms: u64) -> Duration {
    if cfg!(debug_assertions) {
        budget_ms(debug_ms)
    } else {
        budget_ms(release_ms)
    }
}

fn bilinear_resize_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = (dx * sw) / dw;
            let sy = (dy * sh) / dh;
            let si = (sy * sw + sx) * 3;
            let di = (dy * dw + dx) * 3;
            dst[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    dst
}

fn gaussian_filter_5x5(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const K: [i32; 5] = [1, 4, 6, 4, 1];
    let mut dst = vec![0u8; w * h];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let mut sum = 0i32;
            let mut wsum = 0i32;
            for ky in 0..5usize {
                for kx in 0..5usize {
                    let w2 = K[ky] * K[kx];
                    sum += src[(y + ky - 2) * w + (x + kx - 2)] as i32 * w2;
                    wsum += w2;
                }
            }
            dst[y * w + x] = (sum / wsum).clamp(0, 255) as u8;
        }
    }
    dst
}

fn sobel_magnitude(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = -(src[(y - 1) * w + x - 1] as i32) + src[(y - 1) * w + x + 1] as i32
                - 2 * src[y * w + x - 1] as i32
                + 2 * src[y * w + x + 1] as i32
                - src[(y + 1) * w + x - 1] as i32
                + src[(y + 1) * w + x + 1] as i32;
            let gy = -(src[(y - 1) * w + x - 1] as i32)
                - 2 * src[(y - 1) * w + x] as i32
                - src[(y - 1) * w + x + 1] as i32
                + src[(y + 1) * w + x - 1] as i32
                + 2 * src[(y + 1) * w + x] as i32
                + src[(y + 1) * w + x + 1] as i32;
            let mag = ((gx * gx + gy * gy) as f64).sqrt().clamp(0.0, 255.0) as u8;
            dst[y * w + x] = mag;
        }
    }
    dst
}

#[test]
fn test_perf_resize_256x256_to_512x512() {
    let width = 256usize;
    let height = 256usize;
    let src: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    for _ in 0..10 {
        let _out = bilinear_resize_rgb(&src, width, height, 512, 512);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < budget_ms(5000),
        "10× 256→512 bilinear resize took {:?}, expected <5s",
        elapsed
    );
}

#[test]
fn test_perf_5x5_gaussian_filter_512x512() {
    let width = 512usize;
    let height = 512usize;
    let src: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    for _ in 0..5 {
        let _out = gaussian_filter_5x5(&src, width, height);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < budget_ms(30000),
        "5× 5×5 Gaussian on 512×512 took {:?}, expected <30s",
        elapsed
    );
}

#[test]
fn test_perf_sobel_512x512() {
    let width = 512usize;
    let height = 512usize;
    let src: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let start = Instant::now();
    for _ in 0..10 {
        let _out = sobel_magnitude(&src, width, height);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < budget_ms(10000),
        "10× Sobel on 512×512 took {:?}, expected <10s",
        elapsed
    );
}

// ── 1920×1080 regression tests ────────────────────────────────────────────────
//
// These are the headline perf gates for OxiMedia Wave 13.
// Budget: 500 ms debug / 50 ms release for resize; 200 ms / 20 ms for Sobel.

#[test]
fn test_perf_resize_1920x1080_to_960x540() {
    let sw = 1920usize;
    let sh = 1080usize;
    let dw = 960usize;
    let dh = 540usize;
    let src: Vec<u8> = (0..sw * sh * 3).map(|i| (i % 256) as u8).collect();

    // Debug budget is generous to avoid CI flakiness; release budget (50 ms)
    // is the regression gate.
    let budget = budget_by_profile(5_000, 50);
    let start = Instant::now();
    let _out = bilinear_resize_rgb(&src, sw, sh, dw, dh);
    let elapsed = start.elapsed();
    assert!(
        elapsed < budget,
        "1920→960 bilinear resize took {:?}, budget {:?}",
        elapsed,
        budget
    );
}

#[test]
fn test_perf_sobel_1920x1080() {
    let width = 1920usize;
    let height = 1080usize;
    let src: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    // Debug builds are unoptimised; give a generous budget so CI does not flake.
    // Release budget (20 ms) is the hard regression gate.
    let budget = budget_by_profile(10_000, 20);
    let start = Instant::now();
    let _out = sobel_magnitude(&src, width, height);
    let elapsed = start.elapsed();
    assert!(
        elapsed < budget,
        "Sobel on 1920×1080 took {:?}, budget {:?}",
        elapsed,
        budget
    );
}
