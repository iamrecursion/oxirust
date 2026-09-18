//! SMPTE timecode operations example.
//!
//! Demonstrates creation, display, arithmetic, and frame-rate handling of
//! SMPTE 12M timecodes using the `Timecode` and `FrameRate` APIs.  No files are
//! read or written — all data is generated in-memory.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example timecode_operations --features timecode -p oximedia
//! ```

use oximedia::prelude::*;

/// Add a number of frames to a timecode and return the result.
///
/// Clones the timecode, repeatedly increments by one frame, and returns
/// the final value.  Returns an error if any increment step fails.
fn add_frames(tc: &Timecode, count: u64) -> Result<Timecode, Box<dyn std::error::Error>> {
    let start = tc.to_frames() + count;
    Ok(Timecode::from_frames(start, frame_rate_from_info(tc))?)
}

/// Subtract frames from a timecode (clamping at frame 0).
fn subtract_frames(tc: &Timecode, count: u64) -> Result<Timecode, Box<dyn std::error::Error>> {
    let current = tc.to_frames();
    let target = current.saturating_sub(count);
    Ok(Timecode::from_frames(target, frame_rate_from_info(tc))?)
}

/// Reconstruct a `FrameRate` enum from the `FrameRateInfo` embedded in a `Timecode`.
fn frame_rate_from_info(tc: &Timecode) -> FrameRate {
    match (tc.frame_rate.fps, tc.frame_rate.drop_frame) {
        (24, false) => {
            // Distinguish 23.976 from 24 by checking fps == 24 and non-drop.
            // We store both as fps=24 non-drop; use Fps24 as the canonical round-trip.
            FrameRate::Fps24
        }
        (25, false) => FrameRate::Fps25,
        (30, true) => FrameRate::Fps2997DF,
        (30, false) => FrameRate::Fps30,
        (50, false) => FrameRate::Fps50,
        (60, false) => FrameRate::Fps60,
        _ => FrameRate::Fps25, // safe fallback
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia SMPTE Timecode Operations Example");
    println!("===========================================\n");

    // ── 1. Create timecodes at different frame rates ──────────────────────────
    println!("1. Creating timecodes at standard frame rates");
    println!("   ─────────────────────────────────────────");

    let tc_25 = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)?;
    let tc_2997 = Timecode::new(1, 0, 0, 2, FrameRate::Fps2997DF)?; // frame 0/1 are dropped at :00
    let tc_24 = Timecode::new(0, 59, 59, 23, FrameRate::Fps24)?;

    println!("   25 fps NDF        : {tc_25}");
    println!("   29.97 fps DF      : {tc_2997}");
    println!("   24 fps NDF        : {tc_24}");
    println!();

    // ── 2. Display format (separator varies by drop-frame flag) ───────────────
    println!("2. Display formatting");
    println!("   ───────────────────");
    println!("   Non-drop frame uses ':' separators  → {tc_25}");
    println!("   Drop frame uses ';' before frames   → {tc_2997}");
    println!();

    // ── 3. Frame-count arithmetic ─────────────────────────────────────────────
    println!("3. Arithmetic — adding and subtracting frames");
    println!("   ───────────────────────────────────────────");

    let tc_25_plus_100 = add_frames(&tc_25, 100)?;
    let tc_25_minus_50 = subtract_frames(&tc_25, 50)?;

    println!("   {tc_25} + 100 frames = {tc_25_plus_100}");
    println!("   {tc_25} − 50 frames  = {tc_25_minus_50}");

    // 29.97 DF arithmetic
    let tc_df_plus_300 = add_frames(&tc_2997, 300)?;
    println!("   {tc_2997} + 300 frames = {tc_df_plus_300}  (drop-frame)");
    println!();

    // ── 4. Convert to/from absolute frame counts ──────────────────────────────
    println!("4. Frame-count conversion (round-trip)");
    println!("   ─────────────────────────────────────");

    let frames_25 = tc_25.to_frames();
    let rt_25 = Timecode::from_frames(frames_25, FrameRate::Fps25)?;
    println!(
        "   {tc_25} → {frames_25} frames → {rt_25}  (round-trip OK: {})",
        tc_25 == rt_25
    );

    let frames_24 = tc_24.to_frames();
    let rt_24 = Timecode::from_frames(frames_24, FrameRate::Fps24)?;
    println!(
        "   {tc_24} → {frames_24} frames → {rt_24}  (round-trip OK: {})",
        tc_24 == rt_24
    );
    println!();

    // ── 5. Frame rate information ─────────────────────────────────────────────
    println!("5. Frame rate properties");
    println!("   ─────────────────────");
    for (label, rate) in [
        ("23.976 fps", FrameRate::Fps23976),
        ("25 fps    ", FrameRate::Fps25),
        ("29.97 DF  ", FrameRate::Fps2997DF),
        ("29.97 NDF ", FrameRate::Fps2997NDF),
        ("30 fps    ", FrameRate::Fps30),
    ] {
        let (num, den) = rate.as_rational();
        println!(
            "   {label}: {:.4} fps  rational={num}/{den}  drop={}",
            rate.as_float(),
            rate.is_drop_frame()
        );
    }
    println!();

    // ── 6. Increment / decrement ──────────────────────────────────────────────
    println!("6. Single-frame increment and decrement");
    println!("   ──────────────────────────────────────");

    let mut tc_mut = Timecode::new(0, 0, 0, 24, FrameRate::Fps25)?;
    println!("   Before increment: {tc_mut}");
    tc_mut.increment()?;
    println!("   After  increment: {tc_mut}  (frame wrapped to next second)");
    tc_mut.decrement()?;
    println!("   After  decrement: {tc_mut}  (back to original)");
    println!();

    println!("All timecode operations completed successfully.");
    Ok(())
}
