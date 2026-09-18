//! Integration tests for `RenderSource` — source resolution, test patterns,
//! and format detection.

use oximedia_edit::render_source::{
    generate_sine, generate_smpte_bars, DecodedImageData, RenderSource, WavData,
};
use std::path::PathBuf;
use std::sync::Arc;

// ─── TestPattern ─────────────────────────────────────────────────────────────

#[test]
fn test_pattern_video_returns_correct_size() {
    let src = RenderSource::TestPattern;
    let frame = src.sample_video(0, 64, 36);
    assert_eq!(frame.len(), 64 * 36 * 4, "RGBA8 buffer size mismatch");
}

#[test]
fn test_pattern_video_is_not_all_black() {
    let src = RenderSource::TestPattern;
    let frame = src.sample_video(0, 32, 18);
    let all_zero = frame.iter().all(|&b| b == 0);
    assert!(!all_zero, "SMPTE bars must not be all black");
}

#[test]
fn test_pattern_audio_returns_correct_size() {
    let src = RenderSource::TestPattern;
    let audio = src.sample_audio(0, 1024, 2, 48_000);
    assert_eq!(audio.len(), 1024 * 2, "interleaved stereo buffer size");
}

#[test]
fn test_pattern_audio_sine_is_deterministic() {
    let src = RenderSource::TestPattern;
    let a = src.sample_audio(0, 256, 1, 48_000);
    let b = src.sample_audio(0, 256, 1, 48_000);
    assert_eq!(a, b, "same pts must produce the same sine data");
}

#[test]
fn test_pattern_audio_different_pts_differ() {
    let src = RenderSource::TestPattern;
    let a = src.sample_audio(0, 64, 1, 48_000);
    let b = src.sample_audio(1024, 64, 1, 48_000);
    // Different starting phases — they should not be identical.
    assert_ne!(a, b, "sine must advance with pts");
}

// ─── Unsupported ─────────────────────────────────────────────────────────────

#[test]
fn test_unsupported_video_is_black() {
    let src = RenderSource::Unsupported {
        path: PathBuf::from("clip.xyz"),
    };
    let frame = src.sample_video(0, 16, 8);
    assert!(frame.iter().all(|&b| b == 0), "unsupported → black frame");
}

#[test]
fn test_unsupported_audio_is_silence() {
    let src = RenderSource::Unsupported {
        path: PathBuf::from("clip.xyz"),
    };
    let audio = src.sample_audio(0, 48, 2, 48_000);
    assert!(audio.iter().all(|&s| s == 0.0), "unsupported → silence");
}

// ─── from_path for unsupported extension ─────────────────────────────────────

#[test]
fn test_from_path_unknown_extension_gives_unsupported() {
    let tmp = std::env::temp_dir().join("oximedia_test_unsupported_clip.xyz");
    std::fs::write(&tmp, b"not media").ok();

    let result = RenderSource::from_path(&tmp);
    assert!(result.is_ok(), "from_path should not error on unknown ext");

    if let Ok(src) = result {
        let audio = src.sample_audio(0, 48, 2, 48_000);
        assert!(audio.iter().all(|&s| s == 0.0), "unknown ext → silence");
    }

    std::fs::remove_file(&tmp).ok();
}

// ─── Decoded image (in-memory) ────────────────────────────────────────────────

#[test]
fn test_image_source_video_scales_to_target() {
    let img = DecodedImageData {
        pixels: vec![255u8, 0, 128, 255].repeat(4 * 4), // 4x4 solid colour
        width: 4,
        height: 4,
    };
    let src = RenderSource::Image(img);
    // Scale to 8x4.
    let frame = src.sample_video(0, 8, 4);
    assert_eq!(frame.len(), 8 * 4 * 4);
}

#[test]
fn test_wav_source_audio_offset_returns_silence_past_end() {
    let wav = WavData {
        samples: vec![0.5_f32; 32],
        sample_rate: 48_000,
        channels: 1,
    };
    let src = Arc::new(RenderSource::Wav(wav));
    // Offset past end of file.
    let audio = src.sample_audio(999_999, 64, 1, 48_000);
    assert!(
        audio.iter().all(|&s| s == 0.0),
        "past-EOF audio must be silence"
    );
}

// ─── generate_smpte_bars ─────────────────────────────────────────────────────

#[test]
fn test_smpte_bars_alpha_is_255() {
    let bars = generate_smpte_bars(8, 4);
    for chunk in bars.chunks_exact(4) {
        assert_eq!(chunk[3], 255, "alpha channel must be fully opaque");
    }
}

// ─── generate_sine amplitude ─────────────────────────────────────────────────

#[test]
fn test_sine_does_not_exceed_amplitude_bound() {
    let samples = generate_sine(0, 48_000, 1, 48_000);
    for &s in &samples {
        assert!(
            s.abs() <= 0.26_f32,
            "sine sample {s} exceeds expected amplitude"
        );
    }
}
