//! Integration tests for WAV audio sample slicing — mono up-mix, stereo
//! passthrough, and offset clamping.

use oximedia_edit::render_source::{RenderSource, WavData};

// ─── Mono → stereo up-mix ────────────────────────────────────────────────────

#[test]
fn test_mono_wav_upmixed_to_stereo() {
    let wav = WavData {
        samples: vec![1.0_f32; 8], // 8 mono frames
        sample_rate: 48_000,
        channels: 1,
    };
    let src = RenderSource::Wav(wav);
    let out = src.sample_audio(0, 4, 2, 48_000); // 4 stereo frames → 8 samples
    assert_eq!(out.len(), 8);
    for &s in &out {
        assert!(
            (s - 1.0).abs() < 1e-6,
            "up-mixed sample must equal source: {s}"
        );
    }
}

// ─── Stereo passthrough ──────────────────────────────────────────────────────

#[test]
fn test_stereo_wav_passthrough() {
    // Interleaved stereo: L=0.3, R=0.7 repeated.
    let samples: Vec<f32> = (0..8).flat_map(|_| [0.3_f32, 0.7_f32]).collect();
    let wav = WavData {
        samples,
        sample_rate: 48_000,
        channels: 2,
    };
    let src = RenderSource::Wav(wav);
    let out = src.sample_audio(0, 4, 2, 48_000); // 4 stereo frames
    assert_eq!(out.len(), 8);
    for i in 0..4 {
        let l = out[i * 2];
        let r = out[i * 2 + 1];
        assert!((l - 0.3).abs() < 1e-5, "L channel {l}");
        assert!((r - 0.7).abs() < 1e-5, "R channel {r}");
    }
}

// ─── Offset clamping ─────────────────────────────────────────────────────────

#[test]
fn test_negative_source_pts_clamped_to_zero() {
    let wav = WavData {
        samples: vec![0.5_f32; 16],
        sample_rate: 48_000,
        channels: 1,
    };
    let src = RenderSource::Wav(wav);
    // Negative pts should be treated as 0.
    let out = src.sample_audio(-100, 4, 1, 48_000);
    assert_eq!(out.len(), 4);
    for &s in &out {
        assert!(
            s.abs() > 1e-6,
            "sample at clamped pts should not be silence: {s}"
        );
    }
}

#[test]
fn test_offset_past_end_is_silence() {
    let wav = WavData {
        samples: vec![0.5_f32; 8],
        sample_rate: 48_000,
        channels: 1,
    };
    let src = RenderSource::Wav(wav);
    let out = src.sample_audio(100_000, 4, 1, 48_000);
    assert_eq!(out.len(), 4);
    for &s in &out {
        assert!(s == 0.0, "past-EOF sample must be 0.0");
    }
}

// ─── Partial overlap ─────────────────────────────────────────────────────────

#[test]
fn test_partial_overlap_zero_pads_remaining() {
    // 4 mono frames total, request 8 samples starting from frame 2.
    // Expected: 2 real samples + 6 zeros.
    let wav = WavData {
        samples: vec![1.0_f32, 1.0, 1.0, 1.0],
        sample_rate: 48_000,
        channels: 1,
    };
    let src = RenderSource::Wav(wav);
    let out = src.sample_audio(2, 8, 1, 48_000); // offset 2, want 8 samples
    assert_eq!(out.len(), 8);
    // First 2 should be 1.0, rest 0.0.
    for i in 0..2 {
        assert!((out[i] - 1.0).abs() < 1e-6, "sample[{i}] should be 1.0");
    }
    for i in 2..8 {
        assert!(out[i] == 0.0, "sample[{i}] should be silence");
    }
}
