//! EBU R128 audio loudness metering example.
//!
//! Demonstrates broadcast-standard loudness measurement using the `LoudnessMeter`
//! API. A synthetic 1000 Hz sine wave at -18 dBFS is fed through the meter and
//! the resulting LUFS, LRA, and true-peak values are reported alongside an EBU
//! R128 compliance check.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example audio_metering --features metering -p oximedia
//! ```

use oximedia::prelude::*;

/// Generate a stereo interleaved sine wave at the given frequency and amplitude.
///
/// # Arguments
/// * `frequency`   - Tone frequency in Hz
/// * `amplitude`   - Linear amplitude (0.0 – 1.0)
/// * `sample_rate` - Sample rate in Hz
/// * `duration_s`  - Duration in seconds
fn generate_sine_stereo(
    frequency: f32,
    amplitude: f32,
    sample_rate: u32,
    duration_s: f32,
) -> Vec<f32> {
    let num_frames = (sample_rate as f32 * duration_s) as usize;
    let mut samples = Vec::with_capacity(num_frames * 2);
    let angular_freq = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;

    for n in 0..num_frames {
        let sample = amplitude * (angular_freq * n as f32).sin();
        samples.push(sample); // Left
        samples.push(sample); // Right
    }
    samples
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia EBU R128 Loudness Metering Example");
    println!("============================================\n");

    // ── 1. Configure meter for EBU R128 stereo at 48 kHz ─────────────────────
    let config = MeterConfig::new(Standard::EbuR128, 48_000.0, 2);
    let mut meter = LoudnessMeter::new(config)?;

    // ── 2. Generate synthetic audio: 1 kHz sine at -18 dBFS, 3 seconds ───────
    let amplitude: f32 = 10.0_f32.powf(-18.0_f32 / 20.0_f32); // ≈ 0.1259 linear
    let audio = generate_sine_stereo(1_000.0, amplitude, 48_000, 3.0);

    println!("Audio properties:");
    println!("  Frequency  : 1000 Hz");
    println!(
        "  Amplitude  : {amplitude:.4} ({:.1} dBFS)",
        20.0 * amplitude.log10() as f64
    );
    println!("  Sample rate: 48000 Hz");
    println!(
        "  Duration   : {:.1} s",
        audio.len() as f64 / (48_000.0 * 2.0)
    );
    println!("  Channels   : 2 (stereo)");
    println!();

    // ── 3. Feed samples into the meter ────────────────────────────────────────
    meter.process_f32(&audio);

    // ── 4. Retrieve loudness metrics ──────────────────────────────────────────
    let metrics: LoudnessMetrics = meter.metrics();

    println!("Loudness Metrics:");
    if metrics.integrated_lufs.is_finite() {
        println!(
            "  Integrated loudness : {:.2} LUFS",
            metrics.integrated_lufs
        );
    } else {
        println!("  Integrated loudness : (insufficient data for gated measurement)");
    }
    if metrics.momentary_lufs.is_finite() {
        println!("  Momentary loudness  : {:.2} LUFS", metrics.momentary_lufs);
    }
    if metrics.short_term_lufs.is_finite() {
        println!(
            "  Short-term loudness : {:.2} LUFS",
            metrics.short_term_lufs
        );
    }
    if metrics.true_peak_dbtp.is_finite() {
        println!("  True peak           : {:.2} dBTP", metrics.true_peak_dbtp);
    }
    if metrics.loudness_range > 0.0 {
        println!("  Loudness range (LRA): {:.2} LU", metrics.loudness_range);
    }
    println!();

    // ── 5. EBU R128 compliance check ─────────────────────────────────────────
    let compliance = meter.check_compliance();

    println!(
        "EBU R128 Compliance (target: {:.0} LUFS ± 1 LU, peak ≤ -1 dBTP):",
        compliance.target_lufs
    );
    println!("  Standard            : {}", compliance.standard_name());
    println!("  Loudness compliant  : {}", compliance.loudness_compliant);
    println!("  Peak compliant      : {}", compliance.peak_compliant);

    if compliance.integrated_lufs.is_finite() {
        println!("  Deviation from target: {:.2} LU", compliance.deviation_lu);
        println!(
            "  Recommended gain    : {:.2} dB",
            compliance.recommended_gain_db()
        );
    }

    println!();
    if compliance.is_compliant() {
        println!("Result: COMPLIANT with {}", compliance.standard_name());
    } else {
        println!(
            "Result: NOT compliant with {} (apply {:.1} dB gain to correct)",
            compliance.standard_name(),
            compliance.recommended_gain_db()
        );
    }

    println!("\nNote: Integrated loudness requires gated blocks; short pure-tone");
    println!("      signals may fall below the absolute gate (-70 LUFS) in some");
    println!("      implementations, returning -∞ for integrated LUFS.");

    Ok(())
}
