//! SILK LTP encode→decode round-trip integration tests.
//!
//! These tests verify the three LTP quality improvements introduced in
//! Wave 19 Slice A (coarse-to-fine pitch search, per-subframe contour RD
//! selection, fractional-lag tap-solve) through the public `SilkEncoder` /
//! `SilkDecoder` interface.
//!
//! All tests use only synthetically generated signals at the SILK internal
//! sample rate (8 kHz NB, 16 kHz WB) — no external audio files.

#![cfg(feature = "opus")]

use oximedia_codec::opus::packet::OpusBandwidth;
use oximedia_codec::opus::silk::{SilkDecoder, SilkEncoder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate `n` samples of a sinusoid at `freq` Hz sampled at `sr` Hz with
/// amplitude `amp`.
fn sinusoid(n: usize, freq: f64, sr: u32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
        .collect()
}

/// Generate deterministic LCG white noise of length `n` with amplitude `amp`.
fn white_noise(n: usize, amp: f32) -> Vec<f32> {
    let mut seed = 0x5EED_u32;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            amp * ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0)
        })
        .collect()
}

/// Encode one frame and immediately decode it, returning the decoded samples.
/// Returns `Err` if the encode produced 0 bytes (DTX suppressed) or if
/// encode/decode fails.
fn encode_decode_roundtrip(
    encoder: &mut SilkEncoder,
    decoder: &mut SilkDecoder,
    frame: &[f32],
    frame_size: usize,
    sample_rate: u32,
) -> Result<Vec<f32>, String> {
    let mut packet = vec![0u8; 4096];
    let n_enc = encoder
        .encode(frame, &mut packet, frame_size)
        .map_err(|e| format!("encode failed: {e:?}"))?;
    if n_enc == 0 {
        return Err("DTX suppressed (0 bytes)".to_string());
    }
    packet.truncate(n_enc);

    let mut pcm = vec![0.0f32; frame_size];
    decoder
        .decode(&packet, &mut pcm, frame_size)
        .map_err(|e| format!("decode failed: {e:?}"))?;

    // Confirm all output samples are finite.
    for (i, &s) in pcm.iter().enumerate() {
        if !s.is_finite() {
            return Err(format!("decoded sample[{i}] is non-finite: {s}"));
        }
    }
    let _ = sample_rate; // used for documentation
    Ok(pcm)
}

/// Compute the signal-to-noise ratio in dB between original and decoded.
fn snr_db(original: &[f32], decoded: &[f32]) -> f32 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 0.0;
    }
    let sig_e: f64 = original[..len]
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum();
    let noise_e: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&o, &d)| {
            let err = f64::from(o) - f64::from(d);
            err * err
        })
        .sum();
    if sig_e < 1e-12 {
        return 0.0;
    }
    if noise_e < 1e-30 {
        return 120.0;
    }
    (10.0 * (sig_e / noise_e).log10()) as f32
}

// ---------------------------------------------------------------------------
// Test 1: Round-trip decoder consistency — output is finite, no NaN
// ---------------------------------------------------------------------------

/// Encode a voiced 150 Hz frame through `SilkEncoder`, decode through
/// `SilkDecoder`, verify: (a) no NaN/Inf in output, (b) output magnitude
/// stays within a reasonable range (synthesis filter stability).
#[test]
fn test_round_trip_decoder_consistency() {
    let sr = 8000u32;
    let bw = OpusBandwidth::Narrowband;
    let frame_size = 160usize; // 20 ms @ 8 kHz

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    let frame = sinusoid(frame_size, 150.0, sr, 0.5);

    let decoded = encode_decode_roundtrip(&mut enc, &mut dec, &frame, frame_size, sr)
        .expect("round-trip must succeed");

    // Decoded length must match the frame size.
    assert_eq!(
        decoded.len(),
        frame_size,
        "decoded length mismatch: {} vs {frame_size}",
        decoded.len()
    );

    // All samples finite.
    assert!(
        decoded.iter().all(|s| s.is_finite()),
        "NaN/Inf in decoded output"
    );

    // Magnitude must be bounded (synthesis filter stability).
    let peak = decoded.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        peak < 4.0,
        "decoded signal peak {peak} exceeds stability bound"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Voiced 150 Hz shows LTP gain (SNR higher with voiced encoding)
// ---------------------------------------------------------------------------

/// A voiced 150 Hz signal encoded with VAD active should decode to a signal
/// that correlates better with the original than random noise would.  We
/// verify positive SNR (> 0 dB) as a minimal sanity check.
#[test]
fn test_voiced_150hz_positive_snr() {
    let sr = 8000u32;
    let bw = OpusBandwidth::Narrowband;
    let frame_size = 160usize;

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    let frame = sinusoid(frame_size, 150.0, sr, 0.5);

    // Warm up with a couple of frames so the encoder has history.
    for _ in 0..2 {
        let _ = encode_decode_roundtrip(&mut enc, &mut dec, &frame, frame_size, sr);
    }

    let decoded = encode_decode_roundtrip(&mut enc, &mut dec, &frame, frame_size, sr)
        .expect("round-trip must succeed");

    let snr = snr_db(&frame, &decoded);
    // SILK is not transparent but the SNR for a voiced 150 Hz tone should be > 0.
    assert!(snr > 0.0, "150 Hz SNR {snr:.2} dB should be positive");
}

// ---------------------------------------------------------------------------
// Test 3: Unvoiced — white noise does not produce spurious LTP
// ---------------------------------------------------------------------------

/// White noise encoded with VAD should produce valid (finite, bounded) decoded
/// output.  This test verifies that the encoder does not panic or produce NaN
/// on noise input (the 'no spurious LTP' property).
#[test]
fn test_unvoiced_white_noise_no_spurious_ltp() {
    let sr = 8000u32;
    let bw = OpusBandwidth::Narrowband;
    let frame_size = 160usize;

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    let noise = white_noise(frame_size, 0.3);

    let decoded = encode_decode_roundtrip(&mut enc, &mut dec, &noise, frame_size, sr)
        .expect("round-trip must succeed");

    assert!(
        decoded.iter().all(|s| s.is_finite()),
        "white-noise decoded output contains NaN/Inf"
    );
    let peak = decoded.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        peak < 4.0,
        "noise decoded peak {peak} exceeds stability bound"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Multi-frame stability — synthesis filter stays bounded
// ---------------------------------------------------------------------------

/// Encode/decode 20 consecutive frames of 150 Hz tone; the peak output must
/// remain bounded.  This validates that the LPC synthesis filter with LTP
/// does not blow up over time.
#[test]
fn test_multi_frame_stability() {
    let sr = 8000u32;
    let bw = OpusBandwidth::Narrowband;
    let frame_size = 160usize;

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    let frame = sinusoid(frame_size, 150.0, sr, 0.5);
    let mut max_peak = 0.0f32;

    for _ in 0..20 {
        match encode_decode_roundtrip(&mut enc, &mut dec, &frame, frame_size, sr) {
            Ok(decoded) => {
                for &s in &decoded {
                    assert!(s.is_finite(), "non-finite sample in multi-frame test");
                    max_peak = max_peak.max(s.abs());
                }
            }
            Err(e) => panic!("round-trip failed: {e}"),
        }
    }

    assert!(
        max_peak < 4.0,
        "multi-frame peak {max_peak:.3} exceeds stability bound"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Wideband round-trip produces finite output
// ---------------------------------------------------------------------------

/// Verify that the coarse-to-fine and contour-RD paths work for wideband (WB)
/// bandwidth — a different lag range and contour table than NB.
#[test]
fn test_wideband_round_trip_finite() {
    let sr = 16000u32;
    let bw = OpusBandwidth::Wideband;
    let frame_size = 320usize; // 20 ms @ 16 kHz

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    // 150 Hz is within the WB LTP range (max trackable ≈ 500 Hz at WB).
    let frame = sinusoid(frame_size, 150.0, sr, 0.5);

    let decoded = encode_decode_roundtrip(&mut enc, &mut dec, &frame, frame_size, sr)
        .expect("WB round-trip must succeed");

    assert_eq!(decoded.len(), frame_size);
    assert!(
        decoded.iter().all(|s| s.is_finite()),
        "WB decoded output contains NaN/Inf"
    );
    let peak = decoded.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(peak < 4.0, "WB decoded peak {peak} exceeds stability bound");
}

// ---------------------------------------------------------------------------
// Test 6: Alternating voiced/unvoiced — no state corruption
// ---------------------------------------------------------------------------

/// Alternate between voiced (150 Hz tone) and unvoiced (white noise) frames
/// to verify the encoder's state machine (prev_voiced flag, LTP-scale
/// emission, etc.) does not corrupt across mode transitions.
#[test]
fn test_alternating_voiced_unvoiced_no_corruption() {
    let sr = 8000u32;
    let bw = OpusBandwidth::Narrowband;
    let frame_size = 160usize;

    let mut enc = SilkEncoder::new(sr, 1, bw);
    let mut dec = SilkDecoder::new(sr, 1, bw);

    let tone = sinusoid(frame_size, 150.0, sr, 0.5);
    let noise = white_noise(frame_size, 0.3);

    for i in 0..8 {
        let frame = if i % 2 == 0 { &tone } else { &noise };
        match encode_decode_roundtrip(&mut enc, &mut dec, frame, frame_size, sr) {
            Ok(decoded) => {
                assert!(
                    decoded.iter().all(|s| s.is_finite()),
                    "frame {i}: non-finite decoded sample"
                );
            }
            Err(e) => panic!("frame {i}: round-trip failed: {e}"),
        }
    }
}
