//! Integration tests for audio cross-fade blending via `TransitionRenderer`.

use bytes::Bytes;
use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_core::{Rational, SampleFormat, Timestamp};
use oximedia_edit::render::TransitionRenderer;
use oximedia_edit::transition::{Transition, TransitionType};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_audio_frame(n_samples: usize, value: f32) -> AudioFrame {
    let bytes: Vec<u8> = (0..n_samples).flat_map(|_| value.to_ne_bytes()).collect();
    AudioFrame {
        format: SampleFormat::F32,
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
        timestamp: Timestamp::new(0, Rational::new(1, 48_000)),
    }
}

fn crossfade() -> Transition {
    Transition::new(0, TransitionType::CrossFade, 0, 0, 1000, 0, 1)
}

// ─── CrossFade arithmetic ────────────────────────────────────────────────────

#[test]
fn test_crossfade_at_zero_preserves_frame_a() {
    let fa = make_audio_frame(64, 0.8);
    let fb = make_audio_frame(64, -0.8);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 0.0);

    if let AudioBuffer::Interleaved(bytes) = &out.samples {
        for c in bytes.chunks_exact(4) {
            let v = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            assert!((v - 0.8).abs() < 1e-5, "expected 0.8, got {v}");
        }
    } else {
        panic!("expected interleaved buffer");
    }
}

#[test]
fn test_crossfade_at_one_returns_frame_b() {
    let fa = make_audio_frame(64, 0.0);
    let fb = make_audio_frame(64, 0.6);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 1.0);

    if let AudioBuffer::Interleaved(bytes) = &out.samples {
        for c in bytes.chunks_exact(4) {
            let v = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            assert!((v - 0.6).abs() < 1e-5, "expected 0.6, got {v}");
        }
    } else {
        panic!("expected interleaved buffer");
    }
}

#[test]
fn test_crossfade_at_midpoint_equals_sum_halved() {
    let fa = make_audio_frame(64, 0.5);
    let fb = make_audio_frame(64, -0.5);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 0.5);

    if let AudioBuffer::Interleaved(bytes) = &out.samples {
        for c in bytes.chunks_exact(4) {
            let v = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            assert!(v.abs() < 1e-5, "0.5*0.5 + 0.5*(-0.5) = 0; got {v}");
        }
    } else {
        panic!("expected interleaved buffer");
    }
}

#[test]
fn test_crossfade_output_is_clamped_to_minus_one_plus_one() {
    // Saturating add: 1.0 * 0.75 + 1.0 * 0.75 = 1.5 → clamped to 1.0.
    let fa = make_audio_frame(16, 0.75);
    let fb = make_audio_frame(16, 0.75);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 1.0);

    if let AudioBuffer::Interleaved(bytes) = &out.samples {
        for c in bytes.chunks_exact(4) {
            let v = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
            assert!(v <= 1.0 && v >= -1.0, "sample {v} out of [-1, 1] range");
        }
    } else {
        panic!("expected interleaved buffer");
    }
}

#[test]
fn test_crossfade_format_mismatch_returns_frame_a() {
    let fa = make_audio_frame(16, 0.5);
    let mut fb = make_audio_frame(16, 0.5);
    fb.format = SampleFormat::S16;

    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 0.5);
    assert_eq!(out.format, SampleFormat::F32, "format mismatch → frame_a");
}

#[test]
fn test_crossfade_sample_rate_preserved_in_output() {
    let fa = make_audio_frame(32, 0.3);
    let fb = make_audio_frame(32, 0.3);
    let out = TransitionRenderer::mix_audio(&fa, &fb, &crossfade(), 0.5);
    assert_eq!(out.sample_rate, 48_000);
}
