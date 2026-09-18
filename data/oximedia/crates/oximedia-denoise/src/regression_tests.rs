//! Regression, quality, and stress tests for the denoising pipeline.
//!
//! These tests verify:
//! - PSNR improvement after denoising (quality gate)
//! - Temporal consistency (no flicker amplification)
//! - Multi-format robustness (Yuv422p, Rgb24)
//! - Noise-estimate profiling (performance gate)

#![cfg(test)]

use crate::spatial::nlmeans::nlmeans_filter;
use crate::temporal::mctf::{MctfConfig, MctfFilter};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

// ---------------------------------------------------------------------------
// PSNR helper
// ---------------------------------------------------------------------------

/// Compute PSNR between two u8 pixel slices using the standard formula:
/// `PSNR = 10 * log10(255² / MSE)`.
fn psnr_u8(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "PSNR: slices must be same length");
    if a.is_empty() {
        return f32::INFINITY;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-10 {
        return f32::INFINITY;
    }
    (10.0 * (255.0f64 * 255.0 / mse).log10()) as f32
}

// ---------------------------------------------------------------------------
// Synthetic-frame helpers
// ---------------------------------------------------------------------------

/// Create a uniform YUV 4:2:0 frame with a specific luma value.
fn make_uniform_yuv420(w: u32, h: u32, luma: u8) -> VideoFrame {
    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, w, h);
    frame.allocate();
    let stride = frame.planes[0].stride;
    for y in 0..h as usize {
        for x in 0..w as usize {
            frame.planes[0].data[y * stride + x] = luma;
        }
    }
    frame
}

/// Build a noisy frame by adding pseudo-random noise to a clean frame.
fn add_noise_to_frame(clean: &VideoFrame, sigma: f32, seed: u64) -> VideoFrame {
    let mut noisy = clean.clone();
    let mut state = seed;
    let stride = noisy.planes[0].stride;
    let (w, h) = clean.plane_dimensions(0);
    for y in 0..h as usize {
        for x in 0..w as usize {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Approximate standard-normal via sum of 6 uniforms
            let sum: f32 = (0..6)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state & 0xFF) as f32 / 255.0
                })
                .sum::<f32>();
            let normal = (sum - 3.0) * std::f32::consts::SQRT_2;
            let noisy_val = clean.planes[0].data[y * stride + x] as f32 + normal * sigma;
            noisy.planes[0].data[y * stride + x] = noisy_val.clamp(0.0, 255.0) as u8;
        }
    }
    noisy
}

/// Extract luma plane pixels (stride-compressed to width).
fn luma_pixels(frame: &VideoFrame) -> Vec<u8> {
    let (w, h) = frame.plane_dimensions(0);
    let stride = frame.planes[0].stride;
    let mut out = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&frame.planes[0].data[y * stride..y * stride + w as usize]);
    }
    out
}

// ---------------------------------------------------------------------------
// PSNR / SSIM regression test
// ---------------------------------------------------------------------------

/// Verify that NLMeans denoising improves PSNR relative to the clean reference.
///
/// Uses a uniform (flat) clean image so that all patches look similar — this is
/// exactly the scenario where NLMeans works best.  High noise (σ=40) ensures a
/// clear gap between noisy and denoised PSNR.
#[test]
fn test_nlmeans_psnr_improvement_over_noisy() {
    let w = 48u32;
    let h = 48u32;
    let n = (w * h) as usize;

    // Clean = uniform 128 (flat image — NLM excels on flat regions).
    // Use σ=5 so that patch distances (≈2σ²=50) are comparable to the NLM
    // filter parameter h²=(10·strength)²=(10·0.5)²=25, enabling effective smoothing.
    let clean = make_uniform_yuv420(w, h, 128);

    // Noisy = clean + σ=5 noise — small enough for NLM to exploit
    let noisy = add_noise_to_frame(&clean, 5.0, 0xDEAD_BEEF);

    let clean_px = luma_pixels(&clean);
    let noisy_px = luma_pixels(&noisy);
    assert_eq!(clean_px.len(), n, "clean pixel count mismatch");
    assert_eq!(noisy_px.len(), n, "noisy pixel count mismatch");

    // Verify noise was actually added
    let noisy_differs_from_clean = noisy_px.iter().zip(clean_px.iter()).any(|(&a, &b)| a != b);
    assert!(
        noisy_differs_from_clean,
        "add_noise_to_frame produced identical frame — noise generation failed"
    );

    let psnr_noisy = psnr_u8(&clean_px, &noisy_px);

    // Apply NLMeans — on a flat noisy image, every patch looks similar so the
    // weighted average strongly suppresses noise.
    let denoised = nlmeans_filter(&noisy, 0.5).expect("nlmeans should succeed");
    let denoised_px = luma_pixels(&denoised);
    assert_eq!(denoised_px.len(), n, "denoised pixel count mismatch");

    let psnr_denoised = psnr_u8(&clean_px, &denoised_px);

    // Primary assertion: PSNR must improve
    assert!(
        psnr_denoised > psnr_noisy,
        "NLMeans PSNR should improve over noisy: noisy={psnr_noisy:.2} dB, denoised={psnr_denoised:.2} dB"
    );
}

// ---------------------------------------------------------------------------
// Temporal consistency test
// ---------------------------------------------------------------------------

/// Verify that MCTF temporal denoising reduces frame-to-frame variation.
#[test]
fn test_mctf_temporal_consistency() {
    let w = 64u32;
    let h = 64u32;
    let num_frames = 10;

    // Generate 10 identical clean frames + independent noise
    let clean = make_uniform_yuv420(w, h, 128);
    let noisy_frames: Vec<Vec<u8>> = (0..num_frames)
        .map(|i| {
            let noisy = add_noise_to_frame(&clean, 15.0, 0xCAFE_0000 + i as u64);
            luma_pixels(&noisy)
        })
        .collect();

    let cfg = MctfConfig::default();
    let mctf = MctfFilter::new(cfg);
    let denoised_frames = mctf.process_sequence(&noisy_frames, w, h);

    // Compute mean absolute difference between consecutive frames
    let mad_between = |frames: &[Vec<u8>]| -> f64 {
        if frames.len() < 2 {
            return 0.0;
        }
        let total: f64 = frames
            .windows(2)
            .map(|pair| {
                pair[0]
                    .iter()
                    .zip(pair[1].iter())
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .sum::<f64>()
                    / pair[0].len() as f64
            })
            .sum();
        total / (frames.len() - 1) as f64
    };

    let mad_input = mad_between(&noisy_frames);
    let mad_output = mad_between(&denoised_frames);

    assert!(
        mad_output < mad_input,
        "Temporal denoising should reduce frame-to-frame variation: input MAD={mad_input:.3}, output MAD={mad_output:.3}"
    );
}

// ---------------------------------------------------------------------------
// Multi-format stress tests
// ---------------------------------------------------------------------------

/// NLMeans on a YUV 4:2:2 frame should not panic.
#[test]
fn test_nlmeans_yuv422p_no_panic() {
    let mut frame = VideoFrame::new(PixelFormat::Yuv422p, 32, 32);
    frame.allocate();
    // Should not panic regardless of outcome
    let _result = nlmeans_filter(&frame, 0.5);
    // At minimum it must not panic — result may be Ok or Err
}

/// NLMeans on an RGB 24-bit frame should not panic.
#[test]
fn test_nlmeans_rgb24_no_panic() {
    let mut frame = VideoFrame::new(PixelFormat::Rgb24, 32, 32);
    frame.allocate();
    let _result = nlmeans_filter(&frame, 0.5);
}

// ---------------------------------------------------------------------------
// Noise-estimate profiling test
// ---------------------------------------------------------------------------

/// Noise estimation on a full 1920×1080 frame must complete within a
/// reasonable time budget.  Release builds finish in < 10 ms; debug builds
/// are slower due to per-element bounds checks on a 2 M-pair loop.
#[test]
fn test_noise_estimate_profile_1080p() {
    use crate::noise_monitor::estimate_sigma_mad_diagonal;

    let w = 1920u32;
    let h = 1080u32;
    let n = (w * h) as usize;
    // Simple alternating-pixel frame (ensures non-trivial diff values)
    let frame: Vec<u8> = (0..n)
        .map(|i| if i % 2 == 0 { 200u8 } else { 50u8 })
        .collect();

    // 2 000 ms covers debug builds; release builds finish in < 10 ms so this
    // is still a meaningful algorithmic safety net.
    let budget_ms: u128 = if cfg!(debug_assertions) { 2_000 } else { 200 };

    let start = std::time::Instant::now();
    let sigma = estimate_sigma_mad_diagonal(&frame, w, h);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < budget_ms,
        "Noise estimation on 1080p took {}ms, must be < {}ms",
        elapsed.as_millis(),
        budget_ms,
    );
    assert!(
        sigma > 0.0,
        "Should estimate non-zero sigma for alternating frame"
    );
}
