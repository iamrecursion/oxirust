//! Tests for ITU-T P.910 spatial/temporal/motion complexity metrics.

use oximedia_bench::runner::{analyze_sequence, compute_motion_sad, compute_si, compute_ti};
use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::{PixelFormat, Rational, Timestamp};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Create a `VideoFrame` with a single luma plane filled with `fill`.
fn make_frame(width: u32, height: u32, fill: u8) -> VideoFrame {
    let n = (width * height) as usize;
    let plane = Plane {
        data: vec![fill; n],
        stride: width as usize,
        width,
        height,
    };
    VideoFrame {
        format: PixelFormat::Yuv420p,
        width,
        height,
        planes: vec![plane],
        timestamp: Timestamp::new(0, Rational::new(1, 1000)),
        frame_type: oximedia_codec::FrameType::Key,
        color_info: oximedia_codec::ColorInfo::default(),
        corrupt: false,
    }
}

/// Create a `VideoFrame` with a pseudo-random luma plane (splitmix64-style hash per pixel).
fn make_gradient_frame(width: u32, height: u32) -> VideoFrame {
    let n = (width * height) as usize;
    let data: Vec<u8> = (0..n)
        .map(|i| {
            let mut h = i as u64 ^ 0x9e3779b97f4a7c15;
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            (h & 0xFF) as u8
        })
        .collect();
    let plane = Plane {
        data,
        stride: width as usize,
        width,
        height,
    };
    VideoFrame {
        format: PixelFormat::Yuv420p,
        width,
        height,
        planes: vec![plane],
        timestamp: Timestamp::new(0, Rational::new(1, 1000)),
        frame_type: oximedia_codec::FrameType::Key,
        color_info: oximedia_codec::ColorInfo::default(),
        corrupt: false,
    }
}

// ── SI tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_si_constant_color_is_zero() {
    // Constant-color frame → zero Sobel gradient → SI = 0.
    let luma_f: Vec<f64> = vec![128.0_f64; 64 * 64];
    let si = compute_si(&luma_f, 64, 64).expect("SI ok");
    assert!(
        si.abs() < 1e-9,
        "constant-color frame should have SI ≈ 0, got {si}"
    );
}

#[test]
fn test_si_noise_frame_is_positive() {
    // A frame with varying values should have positive SI.
    // Use splitmix64-style hash to generate genuine pseudo-random noise (not a linear ramp
    // which produces constant Sobel magnitudes and zero stddev).
    let luma_f: Vec<f64> = (0..(64_u64 * 64))
        .map(|i| {
            let mut h = i ^ 0x9e3779b97f4a7c15;
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            (h & 0xFF) as f64
        })
        .collect();
    let si = compute_si(&luma_f, 64, 64).expect("SI ok");
    assert!(si > 0.0, "noise-like frame should have SI > 0, got {si}");
}

#[test]
fn test_si_too_small_returns_error() {
    use oximedia_bench::runner::ComplexityError;
    let luma = vec![0.0_f64; 2 * 2];
    let result = compute_si(&luma, 2, 2);
    assert!(
        matches!(result, Err(ComplexityError::FrameTooSmall { .. })),
        "2×2 frame should return FrameTooSmall"
    );
}

// ── TI tests ──────────────────────────────────────────────────────────────────

#[test]
fn test_ti_identical_frames_is_zero() {
    let luma: Vec<f64> = vec![100.0_f64; 64 * 64];
    let ti = compute_ti(&luma, &luma, 64, 64).expect("TI ok");
    assert!(ti.abs() < 1e-9, "identical frames → TI ≈ 0, got {ti}");
}

#[test]
fn test_ti_translating_square_is_positive() {
    let w = 64usize;
    let h = 64usize;
    // Frame 0: white square in the top-left.
    let mut curr = vec![0.0_f64; w * h];
    let mut prev = vec![0.0_f64; w * h];
    for r in 10..20 {
        for c in 10..20 {
            prev[r * w + c] = 255.0;
        }
    }
    // Frame 1: same square moved 5 pixels right.
    for r in 10..20 {
        for c in 15..25 {
            curr[r * w + c] = 255.0;
        }
    }
    let ti = compute_ti(&curr, &prev, w, h).expect("TI ok");
    assert!(ti > 0.0, "moving square should produce TI > 0, got {ti}");
}

// ── Motion SAD tests ───────────────────────────────────────────────────────────

#[test]
fn test_motion_sad_identical_frames_is_zero() {
    let luma: Vec<f64> = (0..64 * 64).map(|i| (i % 256) as f64).collect();
    let sad = compute_motion_sad(&luma, &luma, 64, 64).expect("SAD ok");
    assert!(
        sad.abs() < 1e-9,
        "identical frames → motion SAD ≈ 0, got {sad}"
    );
}

#[test]
fn test_motion_sad_different_frames_is_positive() {
    let curr = vec![200.0_f64; 64 * 64];
    let prev = vec![100.0_f64; 64 * 64];
    let sad = compute_motion_sad(&curr, &prev, 64, 64).expect("SAD ok");
    assert!(
        sad > 0.0,
        "different frames should produce positive SAD, got {sad}"
    );
    // SAD per block should be exactly 16×16×100 = 25600
    let expected = 16.0 * 16.0 * 100.0;
    assert!(
        (sad - expected).abs() < 1e-6,
        "expected SAD = {expected}, got {sad}"
    );
}

// ── analyze_sequence integration ──────────────────────────────────────────────

#[test]
fn test_analyze_sequence_single_frame_no_ti() {
    let frame = make_gradient_frame(64, 64);
    let result = analyze_sequence(&[frame]).expect("analyze ok");
    assert!(result.si > 0.0, "gradient frame should have SI > 0");
    // Only one frame → TI and motion should be 0.
    assert!(
        result.ti.abs() < 1e-9,
        "single frame should have TI = 0, got {}",
        result.ti
    );
    assert!(
        result.motion_sad_mean.abs() < 1e-9,
        "single frame should have motion = 0"
    );
}

#[test]
fn test_analyze_sequence_constant_color() {
    let frames = vec![
        make_frame(64, 64, 128),
        make_frame(64, 64, 128),
        make_frame(64, 64, 128),
    ];
    let result = analyze_sequence(&frames).expect("analyze ok");
    assert!(
        result.si.abs() < 1e-9,
        "constant-color → SI ≈ 0, got {}",
        result.si
    );
    assert!(
        result.ti.abs() < 1e-9,
        "constant-color identical frames → TI ≈ 0, got {}",
        result.ti
    );
}

#[test]
fn test_analyze_sequence_no_frames_error() {
    use oximedia_bench::runner::ComplexityError;
    let result = analyze_sequence(&[]);
    assert!(
        matches!(result, Err(ComplexityError::NoFrames)),
        "empty slice should return NoFrames"
    );
}
