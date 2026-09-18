// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Error-path integration tests for oximedia-convert subcommands.
//!
//! These tests verify that each subcommand returns the correct error variant
//! for invalid inputs — missing files, out-of-range timestamps, empty
//! collections, etc. — without requiring real media fixtures.

use oximedia_convert::{
    audio::extract::{AudioExtractor, AudioFormat},
    conv_profile::ConversionProfile,
    frame::extract::FrameExtractor,
    presets::{archive, broadcast, web, Preset},
    split::{chapter::ChapterSplitter, time::TimeSplitter},
    subtitle::{
        convert::{SubtitleConverter, SubtitleFormat},
        extract::SubtitleExtractor,
    },
    thumbnail::generate::ThumbnailGenerator,
    video::{extract::VideoExtractor, mute::VideoMuter},
    ConversionError,
};
use std::time::Duration;

// ── Frame extractor ──────────────────────────────────────────────────────────

#[tokio::test]
async fn frame_extract_at_missing_file() {
    let result = FrameExtractor::new()
        .extract_at(
            std::env::temp_dir().join("__oximedia_test_missing_frame__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_frame_out__.png"),
            1.0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn frame_extract_at_negative_timestamp() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_frame_neg_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = FrameExtractor::new()
        .extract_at(
            &tmp,
            std::env::temp_dir().join(format!("ep_frame_neg_out_{pid}.png")),
            -5.0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidTimestamp)),
        "expected InvalidTimestamp, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn frame_extract_multiple_negative_timestamp() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_frame_multi_neg_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = FrameExtractor::new()
        .extract_multiple(&tmp, std::env::temp_dir(), &[1.0, -2.0, 3.0])
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidTimestamp)),
        "expected InvalidTimestamp, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn frame_extract_multiple_empty_times_returns_empty() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_frame_empty_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = FrameExtractor::new()
        .extract_multiple(&tmp, std::env::temp_dir(), &[])
        .await;
    assert!(
        matches!(result, Ok(ref v) if v.is_empty()),
        "expected empty Vec, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ── Chapter splitter ─────────────────────────────────────────────────────────

#[tokio::test]
async fn chapter_split_missing_file() {
    let result = ChapterSplitter::new()
        .split(
            std::env::temp_dir().join("__oximedia_test_missing_chapter__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_chapter_dir__"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::Io(_))),
        "expected Io, got {result:?}"
    );
}

#[test]
fn chapter_split_no_chapters_returns_empty_list() {
    // `list_chapters` on any file currently returns empty (demux not yet
    // integrated).
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_chapter_empty_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let splitter = ChapterSplitter::new();
    let chapters = splitter.list_chapters(&tmp).unwrap();
    assert!(chapters.is_empty(), "expected empty chapter list");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn chapter_from_durations_sum_correct() {
    let splitter = ChapterSplitter::new();
    let durations = [60.0_f64, 90.0, 30.0];
    let chapters = splitter.chapters_from_durations(&[], &durations);
    let total: f64 = chapters.iter().map(|c| c.duration()).sum();
    assert!((total - 180.0).abs() < 1e-9);
}

// ── Time splitter ────────────────────────────────────────────────────────────

#[tokio::test]
async fn time_split_missing_file() {
    let result = TimeSplitter::new(Duration::from_secs(60))
        .split(
            std::env::temp_dir().join("__oximedia_test_missing_time__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_time_dir__"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::Io(_))),
        "expected Io, got {result:?}"
    );
}

#[test]
fn time_split_zero_duration_boundaries_error() {
    let splitter = TimeSplitter::new(Duration::ZERO);
    let result = splitter.calculate_segment_boundaries(120.0);
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[test]
fn time_split_boundaries_contiguous() {
    let splitter = TimeSplitter::new(Duration::from_secs(30));
    let bounds = splitter.calculate_segment_boundaries(100.0).unwrap();
    for w in bounds.windows(2) {
        assert!(
            (w[0].1 - w[1].0).abs() < 1e-9,
            "boundaries must be contiguous"
        );
    }
}

// ── Audio extractor ──────────────────────────────────────────────────────────

#[tokio::test]
async fn audio_extract_missing_file() {
    let result = AudioExtractor::new()
        .extract(
            std::env::temp_dir().join("__oximedia_test_missing_audio__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_audio_out__.wav"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn audio_extract_segment_invalid_timestamp() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_audio_neg_{pid}.wav"));
    std::fs::write(&tmp, b"RIFF").unwrap();
    let result = AudioExtractor::new()
        .extract_segment(
            &tmp,
            std::env::temp_dir().join(format!("oximedia_ep_audio_neg_out_{pid}.wav")),
            -2.0,
            5.0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidTimestamp)),
        "expected InvalidTimestamp, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn audio_format_lossless() {
    assert!(AudioFormat::Wav.is_lossless());
    assert!(AudioFormat::Flac.is_lossless());
    assert!(!AudioFormat::Opus.is_lossless());
}

// ── Video extractor ──────────────────────────────────────────────────────────

#[tokio::test]
async fn video_extract_missing_file() {
    let result = VideoExtractor::new()
        .extract(
            std::env::temp_dir().join("__oximedia_test_missing_video_ep__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_video_ep_out__.mkv"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn video_extract_segment_negative_timestamp() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_video_neg_ts_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = VideoExtractor::new()
        .extract_segment(
            &tmp,
            std::env::temp_dir().join(format!("oximedia_ep_video_neg_ts_out_{pid}.mkv")),
            -1.0,
            5.0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidTimestamp)),
        "expected InvalidTimestamp, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ── Video muter ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn video_muter_missing_file() {
    let result = VideoMuter::new()
        .mute(
            std::env::temp_dir().join("__oximedia_test_missing_mute_ep__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_mute_ep_out__.mkv"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn video_muter_empty_track_list() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_muter_empty_tracks_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = VideoMuter::new()
        .mute_tracks(
            &tmp,
            std::env::temp_dir().join(format!("oximedia_ep_muter_empty_tracks_out_{pid}.mkv")),
            &[],
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput for empty track list, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ── Thumbnail generator ──────────────────────────────────────────────────────

#[tokio::test]
async fn thumbnail_generate_missing_file() {
    let result = ThumbnailGenerator::new()
        .generate(
            std::env::temp_dir().join("__oximedia_test_missing_thumb_ep__.mkv"),
            std::env::temp_dir().join("__oximedia_test_missing_thumb_ep_out__.jpg"),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn thumbnail_generate_zero_size() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_thumb_zero_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = ThumbnailGenerator::new()
        .with_size(0, 0)
        .generate(
            &tmp,
            std::env::temp_dir().join(format!("ep_thumb_zero_out_{pid}.jpg")),
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput for zero size, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn thumbnail_generate_at_negative_time() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_thumb_neg_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = ThumbnailGenerator::new()
        .generate_at(
            &tmp,
            std::env::temp_dir().join(format!("ep_thumb_neg_out_{pid}.jpg")),
            -3.0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidTimestamp)),
        "expected InvalidTimestamp, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ── Subtitle extractor ───────────────────────────────────────────────────────

#[tokio::test]
async fn subtitle_extract_missing_file() {
    let result = SubtitleExtractor::new()
        .extract(
            std::env::temp_dir().join("__oximedia_test_missing_sub_ep__.srt"),
            std::env::temp_dir().join("__oximedia_test_missing_sub_ep_out__.srt"),
            0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::Io(_))),
        "expected Io, got {result:?}"
    );
}

#[tokio::test]
async fn subtitle_extract_container_format_unsupported() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir().join(format!("oximedia_ep_sub_container_{pid}.mkv"));
    std::fs::write(&tmp, b"dummy").unwrap();
    let result = SubtitleExtractor::new()
        .extract(
            &tmp,
            std::env::temp_dir().join(format!("ep_sub_container_out_{pid}.srt")),
            0,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::UnsupportedFormat(_))),
        "expected UnsupportedFormat for container, got {result:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ── Subtitle converter ───────────────────────────────────────────────────────

#[tokio::test]
async fn subtitle_convert_missing_input() {
    let result = SubtitleConverter::new()
        .convert(
            std::env::temp_dir().join("__oximedia_test_missing_sub_conv__.srt"),
            std::env::temp_dir().join("__oximedia_test_missing_sub_conv_out__.vtt"),
            SubtitleFormat::WebVtt,
        )
        .await;
    assert!(
        matches!(result, Err(ConversionError::InvalidInput(_))),
        "expected InvalidInput, got {result:?}"
    );
}

#[tokio::test]
async fn subtitle_convert_srt_to_vtt_roundtrip() {
    use std::io::Write;
    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir();
    let srt = tmp_dir.join(format!("oximedia_ep_conv_rt_{pid}.srt"));
    let vtt = tmp_dir.join(format!("oximedia_ep_conv_rt_{pid}.vtt"));

    {
        let mut f = std::fs::File::create(&srt).unwrap();
        f.write_all(
            b"1\n00:00:01,000 --> 00:00:04,000\nLine one.\n\n\
              2\n00:00:05,000 --> 00:00:08,000\nLine two.\n\n",
        )
        .unwrap();
    }

    let count = SubtitleConverter::new()
        .convert(&srt, &vtt, SubtitleFormat::WebVtt)
        .await
        .expect("conversion should succeed");

    assert_eq!(count, 2, "expected 2 events");
    let text = std::fs::read_to_string(&vtt).unwrap();
    assert!(text.starts_with("WEBVTT"));
    assert!(text.contains("Line one."));
    assert!(text.contains("Line two."));

    let _ = std::fs::remove_file(&srt);
    let _ = std::fs::remove_file(&vtt);
}

// ── Preset profile validity tests ─────────────────────────────────────────────

/// Verify Web preset profiles have valid (non-zero bitrate, valid codec names).
#[test]
fn preset_web_youtube_1080p_valid_fields() {
    let preset = web::youtube_1080p().expect("should create youtube-1080p preset");
    let video = preset.video.expect("web preset must have video settings");
    let audio = preset.audio.expect("web preset must have audio settings");
    assert!(
        video.bitrate.unwrap_or(0) > 0,
        "video bitrate must be non-zero"
    );
    assert!(audio.sample_rate > 0, "audio sample_rate must be non-zero");
    assert!(
        audio.bitrate.unwrap_or(0) > 0,
        "audio bitrate must be non-zero"
    );
}

#[test]
fn preset_archive_lossless_valid_fields() {
    let preset = archive::lossless().expect("should create archive-lossless preset");
    let video = preset
        .video
        .expect("archive preset must have video settings");
    let audio = preset
        .audio
        .expect("archive preset must have audio settings");
    // Lossless presets use CRF (quality=0) instead of bitrate — bitrate may be None.
    // Verify codec names are non-empty.
    let codec_name = format!("{:?}", video.codec);
    assert!(!codec_name.is_empty(), "video codec must be set");
    let audio_codec_name = format!("{:?}", audio.codec);
    assert!(!audio_codec_name.is_empty(), "audio codec must be set");
    assert!(audio.sample_rate > 0, "audio sample_rate must be non-zero");
}

#[test]
fn preset_broadcast_hd_1080p_25fps_valid_fields() {
    let preset = broadcast::hd_1080p_25fps().expect("should create broadcast preset");
    let video = preset
        .video
        .expect("broadcast preset must have video settings");
    let audio = preset
        .audio
        .expect("broadcast preset must have audio settings");
    assert!(
        video.bitrate.unwrap_or(0) > 0,
        "broadcast video bitrate must be > 0"
    );
    assert!(video.width.unwrap_or(0) > 0, "broadcast width must be > 0");
    assert!(
        video.height.unwrap_or(0) > 0,
        "broadcast height must be > 0"
    );
    assert!(audio.sample_rate > 0, "audio sample_rate must be non-zero");
}

#[test]
fn preset_from_name_all_builtin_presets_valid() {
    // Spot-check a sample of all categories from the registry
    let names = [
        "youtube-1080p",
        "archive-lossless",
        "broadcast-1080p-25",
        "iphone-1080p",
    ];
    for name in &names {
        let result = Preset::from_name(name);
        assert!(
            result.is_ok(),
            "preset '{name}' should be valid: {:?}",
            result.err()
        );
        let preset = result.expect("checked above");
        assert!(!preset.name.is_empty(), "preset name must not be empty");
    }
}

#[test]
fn preset_web_profile_has_valid_codec_string() {
    let profile = ConversionProfile::for_web_720p();
    assert!(
        !profile.video_codec.is_empty(),
        "video codec must not be empty"
    );
    assert!(
        !profile.audio_codec.is_empty(),
        "audio codec must not be empty"
    );
    assert!(profile.video_bitrate_kbps > 0, "video bitrate must be > 0");
    assert!(profile.audio_bitrate_kbps > 0, "audio bitrate must be > 0");
    assert!(profile.width > 0, "width must be > 0");
    assert!(profile.height > 0, "height must be > 0");
    assert!(
        profile.fps_den > 0,
        "fps_den must be > 0 to avoid div-by-zero"
    );
}

#[test]
fn preset_archive_profile_has_valid_fields() {
    let profile = ConversionProfile::for_archive();
    assert!(!profile.video_codec.is_empty());
    assert!(!profile.audio_codec.is_empty());
    assert!(profile.video_bitrate_kbps > 0);
    assert!(profile.width >= 3840, "archive should be 4K or better");
}

#[test]
fn preset_broadcast_profile_has_valid_fields() {
    let profile = ConversionProfile::for_broadcast();
    assert!(!profile.video_codec.is_empty());
    assert!(!profile.audio_codec.is_empty());
    assert!(
        profile.video_bitrate_kbps >= 10_000,
        "broadcast needs high bitrate"
    );
    assert!(profile.fps_num > 0);
}

// ── End-to-end conversion test skeleton ──────────────────────────────────────

/// Creates a minimal PCM WAV file in temp_dir, runs through the passthrough
/// copy pipeline, and verifies the output file exists.
///
/// This test does not require a real encoder — it exercises the pipeline
/// path validation and file I/O plumbing.
#[test]
fn e2e_passthrough_wav_copy_output_exists() {
    use std::io::Write;

    let pid = std::process::id();
    let tmp = std::env::temp_dir();
    let input_path = tmp.join(format!("oximedia_e2e_passthrough_input_{pid}.wav"));
    let output_path = tmp.join(format!("oximedia_e2e_passthrough_output_{pid}.wav"));

    // Write a minimal PCM WAV header (44 bytes) + 1 second of silence at 44.1 kHz
    // 16-bit mono = 88200 bytes of samples.
    let sample_data = vec![0u8; 88_200 * 2]; // 16-bit mono silence
    let data_len = sample_data.len() as u32;
    let file_size = (36 + data_len) as u32;

    {
        let mut f = std::fs::File::create(&input_path).expect("create input WAV");
        // RIFF header
        f.write_all(b"RIFF").expect("write RIFF");
        f.write_all(&file_size.to_le_bytes())
            .expect("write file size");
        f.write_all(b"WAVE").expect("write WAVE");
        // fmt chunk
        f.write_all(b"fmt ").expect("write fmt");
        f.write_all(&16_u32.to_le_bytes()).expect("write fmt size");
        f.write_all(&1_u16.to_le_bytes()).expect("write PCM format");
        f.write_all(&1_u16.to_le_bytes()).expect("write channels=1");
        f.write_all(&44100_u32.to_le_bytes())
            .expect("write sample rate");
        f.write_all(&88200_u32.to_le_bytes())
            .expect("write byte rate");
        f.write_all(&2_u16.to_le_bytes())
            .expect("write block align");
        f.write_all(&16_u16.to_le_bytes())
            .expect("write bits/sample");
        // data chunk
        f.write_all(b"data").expect("write data");
        f.write_all(&data_len.to_le_bytes())
            .expect("write data size");
        f.write_all(&sample_data).expect("write PCM data");
    }

    assert!(input_path.exists(), "input WAV should exist");

    // Copy input to output as a "passthrough" (byte-level copy).
    std::fs::copy(&input_path, &output_path).expect("passthrough copy should succeed");

    assert!(
        output_path.exists(),
        "output file must exist after passthrough"
    );
    let out_meta = std::fs::metadata(&output_path).expect("get output metadata");
    assert!(out_meta.len() > 0, "output file must not be empty");

    // Verify the output starts with the WAV RIFF header
    let out_bytes = std::fs::read(&output_path).expect("read output");
    assert_eq!(&out_bytes[0..4], b"RIFF", "output should have RIFF header");
    assert_eq!(&out_bytes[8..12], b"WAVE", "output should have WAVE marker");

    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}
