//! Video quality assessment example (PSNR and SSIM).
//!
//! Demonstrates objective video quality metrics using the `QualityAssessor` API.
//! Synthetic 64×64 YUV420p frames are constructed in-memory — one as a clean
//! reference, two with different distortion levels — and PSNR and SSIM scores
//! are computed and compared.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example quality_assessment --features quality -p oximedia
//! ```

use oximedia::prelude::*;

/// Build a YUV420p frame filled with a given luma value and neutral chroma.
///
/// Width and height must both be even (required by 4:2:0 subsampling).
fn make_frame(
    width: usize,
    height: usize,
    luma: u8,
) -> Result<QualityFrame, Box<dyn std::error::Error>> {
    let mut frame = QualityFrame::new(width, height, PixelFormat::Yuv420p)?;
    // Fill luma plane with the requested value
    for px in frame.luma_mut() {
        *px = luma;
    }
    // Neutral chroma (128 = no colour shift in YCbCr)
    if frame.chroma().is_some() {
        frame.planes[1].fill(128);
        frame.planes[2].fill(128);
    }
    Ok(frame)
}

/// Apply additive distortion to the luma plane, clamping to [0, 255].
fn distort_frame(frame: &QualityFrame, offset: i16) -> QualityFrame {
    let mut out = frame.clone();
    for px in out.luma_mut() {
        *px = (*px as i16 + offset).clamp(0, 255) as u8;
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Video Quality Assessment Example");
    println!("==========================================\n");

    const W: usize = 64;
    const H: usize = 64;
    const LUMA: u8 = 128; // mid-grey reference

    // ── 1. Create frames ──────────────────────────────────────────────────────
    let reference = make_frame(W, H, LUMA)?;
    let slightly_distorted = distort_frame(&reference, 10); // +10 luma units
    let heavily_distorted = distort_frame(&reference, 50); // +50 luma units

    println!("Frame configuration:");
    println!("  Size             : {}×{} pixels (YUV420p)", W, H);
    println!("  Reference luma   : {LUMA}");
    println!("  Slight distortion: luma +10");
    println!("  Heavy distortion : luma +50");
    println!();

    // ── 2. Create quality assessor ────────────────────────────────────────────
    let assessor = QualityAssessor::new();

    // ── 3. PSNR ───────────────────────────────────────────────────────────────
    println!("PSNR (Peak Signal-to-Noise Ratio):");
    println!("  Higher is better; >40 dB = excellent, 30–40 dB = good, <30 dB = poor\n");

    let psnr_slight = assessor.assess(&reference, &slightly_distorted, MetricType::Psnr)?;
    let psnr_heavy = assessor.assess(&reference, &heavily_distorted, MetricType::Psnr)?;

    println!(
        "  Reference vs slight distortion : {:.2} dB",
        psnr_slight.score
    );
    if let Some(y) = psnr_slight.components.get("Y") {
        println!("    Y-plane PSNR               : {:.2} dB", y);
    }

    println!(
        "  Reference vs heavy distortion  : {:.2} dB",
        psnr_heavy.score
    );
    if let Some(y) = psnr_heavy.components.get("Y") {
        println!("    Y-plane PSNR               : {:.2} dB", y);
    }
    println!();

    // ── 4. SSIM ───────────────────────────────────────────────────────────────
    println!("SSIM (Structural Similarity Index):");
    println!("  Range 0–1; 1.0 = identical, >0.95 = excellent, <0.80 = poor\n");

    let ssim_slight = assessor.assess(&reference, &slightly_distorted, MetricType::Ssim)?;
    let ssim_heavy = assessor.assess(&reference, &heavily_distorted, MetricType::Ssim)?;

    println!(
        "  Reference vs slight distortion : {:.4}",
        ssim_slight.score
    );
    println!("  Reference vs heavy distortion  : {:.4}", ssim_heavy.score);
    println!();

    // ── 5. Identical-frame sanity check ───────────────────────────────────────
    println!("Sanity check — identical frames:");
    let psnr_identical = assessor.assess(&reference, &reference, MetricType::Psnr)?;
    let ssim_identical = assessor.assess(&reference, &reference, MetricType::Ssim)?;
    println!("  PSNR : {:.2} dB (expect ≥ 100 dB)", psnr_identical.score);
    println!("  SSIM : {:.4}  (expect ≈ 1.0)", ssim_identical.score);
    println!();

    // ── 6. Summary ────────────────────────────────────────────────────────────
    println!("Summary:");
    println!("  As distortion increases, PSNR decreases and SSIM moves away from 1.0.");
    println!("  Both metrics confirm that heavy distortion (+50 luma) causes significantly");
    println!("  greater quality loss than slight distortion (+10 luma).");

    Ok(())
}
