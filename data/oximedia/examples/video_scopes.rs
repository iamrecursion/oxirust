//! Broadcast video scopes example.
//!
//! Demonstrates professional video scope analysis using synthetic YUV frame
//! data. Shows waveform, vectorscope, and histogram analysis along with
//! broadcast compliance checking (ITU-R BT.709 legal range 16–235 luma).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example video_scopes --features scopes -p oximedia
//! ```

use oximedia::prelude::*;
use oximedia_scopes::compliance::{check_compliance, ComplianceConfig};
use oximedia_scopes::signal_stats::compute_frame_stats;

const FRAME_WIDTH: u32 = 64;
const FRAME_HEIGHT: u32 = 64;

/// Generate a synthetic RGB24 gradient frame.
///
/// Each pixel is computed as a smooth ramp across both axes, giving a
/// full-range signal useful for testing all scope displays.
fn create_gradient_frame(width: u32, height: u32) -> Vec<u8> {
    let mut frame = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Horizontal red ramp
            let r = (x * 255 / (width - 1)) as u8;
            // Vertical green ramp
            let g = (y * 255 / (height - 1)) as u8;
            // Diagonal blue ramp
            let b = (((x + y) * 255) / ((width - 1) + (height - 1))) as u8;
            frame[idx] = r;
            frame[idx + 1] = g;
            frame[idx + 2] = b;
        }
    }
    frame
}

/// Generate a synthetic legal-range frame (all pixels safely within BT.709
/// broadcast limits: luma 16–235, neutral mid-gray).
fn create_legal_frame(width: u32, height: u32) -> Vec<u8> {
    // Mid-gray (128, 128, 128) maps to luma ~128 — well within legal range.
    vec![128u8; (width * height * 3) as usize]
}

/// Generate a synthetic out-of-range frame (pure black + white checker).
///
/// Pure black (luma ~16) and pure white (luma ~235+) stress the compliance
/// checker by pushing signals outside broadcast limits.
fn create_illegal_frame(width: u32, height: u32) -> Vec<u8> {
    let mut frame = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let value: u8 = if (x + y) % 2 == 0 { 0 } else { 255 };
            frame[idx] = value;
            frame[idx + 1] = value;
            frame[idx + 2] = value;
        }
    }
    frame
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Broadcast Video Scopes Example");
    println!("========================================\n");

    let config = ScopeConfig {
        width: 256,
        height: 256,
        show_graticule: true,
        show_labels: true,
        anti_alias: false, // Faster for synthetic demo
        waveform_mode: WaveformMode::Overlay,
        vectorscope_mode: VectorscopeMode::Circular,
        histogram_mode: HistogramMode::Overlay,
        vectorscope_gain: 1.0,
        highlight_gamut: false,
        gamut_colorspace: GamutColorspace::Rec709,
        resolution: None,
    };

    let scopes = VideoScopes::new(config);
    let frame = create_gradient_frame(FRAME_WIDTH, FRAME_HEIGHT);

    // ── Waveform Luma ─────────────────────────────────────────────────────────
    println!("--- Waveform Luma (Y channel) ---");
    let waveform = scopes.analyze(&frame, FRAME_WIDTH, FRAME_HEIGHT, ScopeType::WaveformLuma)?;
    println!("  Scope type : {:?}", waveform.scope_type);
    println!("  Output size: {}×{} px", waveform.width, waveform.height);
    println!("  RGBA bytes : {}", waveform.data.len());
    println!();

    // ── Vectorscope ───────────────────────────────────────────────────────────
    println!("--- Vectorscope (YUV Cb/Cr display) ---");
    let vectorscope = scopes.analyze(&frame, FRAME_WIDTH, FRAME_HEIGHT, ScopeType::Vectorscope)?;
    println!("  Scope type : {:?}", vectorscope.scope_type);
    println!(
        "  Output size: {}×{} px",
        vectorscope.width, vectorscope.height
    );
    println!("  RGBA bytes : {}", vectorscope.data.len());
    println!();

    // ── Histogram (Luma) ──────────────────────────────────────────────────────
    println!("--- Histogram (Luma) ---");
    let histogram = scopes.analyze(&frame, FRAME_WIDTH, FRAME_HEIGHT, ScopeType::HistogramLuma)?;
    println!("  Scope type : {:?}", histogram.scope_type);
    println!("  Output size: {}×{} px", histogram.width, histogram.height);
    println!("  RGBA bytes : {}", histogram.data.len());
    println!();

    // ── Signal Statistics ─────────────────────────────────────────────────────
    println!("--- Signal Statistics (gradient frame) ---");
    let stats = compute_frame_stats(&frame, FRAME_WIDTH, FRAME_HEIGHT);
    println!("  Pixels analyzed : {}", stats.total_pixels);
    println!(
        "  Luma  — mean: {:.1}  min: {:.0}  max: {:.0}  range: {:.0}",
        stats.luma.mean, stats.luma.min, stats.luma.max, stats.luma.range
    );
    println!(
        "  Red   — mean: {:.1}  min: {:.0}  max: {:.0}",
        stats.red.mean, stats.red.min, stats.red.max
    );
    println!(
        "  Green — mean: {:.1}  min: {:.0}  max: {:.0}",
        stats.green.mean, stats.green.min, stats.green.max
    );
    println!(
        "  Blue  — mean: {:.1}  min: {:.0}  max: {:.0}",
        stats.blue.mean, stats.blue.min, stats.blue.max
    );
    println!();

    // ── Broadcast Compliance (legal range 16–235 luma) ────────────────────────
    println!("--- Broadcast Compliance (ITU-R BT.709) ---");

    let compliance_cfg = ComplianceConfig::default();

    let legal_frame = create_legal_frame(FRAME_WIDTH, FRAME_HEIGHT);
    let legal_report = check_compliance(&legal_frame, FRAME_WIDTH, FRAME_HEIGHT, &compliance_cfg)?;
    println!(
        "  Legal frame  — passes: {}  violations: {}  ({:.2}%)",
        legal_report.passes_compliance,
        legal_report.violation_count,
        legal_report.violation_percent
    );

    let illegal_frame = create_illegal_frame(FRAME_WIDTH, FRAME_HEIGHT);
    let illegal_report =
        check_compliance(&illegal_frame, FRAME_WIDTH, FRAME_HEIGHT, &compliance_cfg)?;
    println!(
        "  Illegal frame — passes: {}  violations: {}  ({:.2}%)",
        illegal_report.passes_compliance,
        illegal_report.violation_count,
        illegal_report.violation_percent
    );

    let gradient_report = check_compliance(&frame, FRAME_WIDTH, FRAME_HEIGHT, &compliance_cfg)?;
    println!(
        "  Gradient frame — passes: {}  violations: {}  ({:.2}%)",
        gradient_report.passes_compliance,
        gradient_report.violation_count,
        gradient_report.violation_percent
    );

    println!("\nVideo scopes analysis completed.");
    Ok(())
}
