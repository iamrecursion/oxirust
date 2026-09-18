//! End-to-end tests for real re-encoding through `oximedia transcode`.
//!
//! Before this feature, every codec-changing transcode was rejected by a
//! blanket `requires_frame_level()` gate ("Use MultiTrackExecutor …"), so
//! the flagship journey `oximedia transcode in.wav out.flac` failed. These
//! tests pin the real decode → filter → encode path end-to-end at the CLI
//! options level (the same entry the binary uses — see
//! `transcode_normalize_audio.rs` for why the library entry is exercised
//! directly).
//!
//! Output verification is content-level: FLAC output is re-decoded with
//! the spec-compliant decoder (`oximedia_transcode::flac_decode`, itself
//! verified against libFLAC/FFmpeg) and compared sample-exact; MJPEG
//! output is re-decoded with the real MJPEG decoder.

mod common;

use oximedia_cli::progress::ProgressFormat;
use oximedia_cli::transcode::{transcode, TranscodeOptions};
use oximedia_codec::traits::VideoDecoder as _;
use oximedia_transcode::flac_decode::decode_flac_to_i16;
use std::path::PathBuf;

/// `TranscodeOptions` with every field at its CLI default.
fn options(input: PathBuf, output: PathBuf) -> TranscodeOptions {
    TranscodeOptions {
        input,
        output,
        preset_name: None,
        video_codec: None,
        audio_codec: None,
        video_bitrate: None,
        audio_bitrate: None,
        scale: None,
        video_filter: None,
        audio_filter: None,
        start_time: None,
        duration: None,
        framerate: None,
        preset: "medium".to_string(),
        two_pass: false,
        crf: None,
        threads: 0,
        overwrite: true,
        map: Vec::new(),
        normalize_audio: false,
        progress_format: ProgressFormat::Plain,
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oximedia_cli_reenc_{}_{name}", std::process::id()))
}

/// Write a small Y4M (C420jpeg) with a moving gradient.
fn write_y4m_fixture(path: &PathBuf, w: usize, h: usize, frames: usize) {
    let mut buf = Vec::new();
    buf.extend_from_slice(format!("YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C420jpeg\n").as_bytes());
    for t in 0..frames {
        buf.extend_from_slice(b"FRAME\n");
        for y in 0..h {
            for x in 0..w {
                buf.push(((x * 4 + y * 2 + t * 8) % 256) as u8);
            }
        }
        for _ in 0..(h / 2) {
            for x in 0..(w / 2) {
                buf.push(((x * 8 + t * 4) % 256) as u8);
            }
        }
        for _ in 0..(h / 2) {
            for x in 0..(w / 2) {
                buf.push(((x * 8) % 256) as u8);
            }
        }
    }
    std::fs::write(path, buf).expect("write y4m fixture");
}

/// Decode a WAV fixture's PCM payload (16-bit LE) to i16 samples.
fn wav_pcm(path: &PathBuf) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("read wav");
    bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Extract the first JPEG image (SOI..EOI) embedded in a byte stream.
fn first_jpeg(data: &[u8]) -> Option<&[u8]> {
    let start = data.windows(3).position(|w| w == [0xFF, 0xD8, 0xFF])?;
    let end = data[start..].windows(2).position(|w| w == [0xFF, 0xD9])?;
    Some(&data[start..start + end + 2])
}

// ─── (a) Flagship: `transcode in.wav out.flac` round-trips ───────────────────

#[tokio::test]
async fn flagship_wav_to_flac_round_trips() {
    let (_dir, input) = common::write_wav_fixture(1_000.0, 48_000, 2, 1.0);
    let output = temp_path("flagship.flac");
    let _ = std::fs::remove_file(&output);

    transcode(options(input.clone(), output.clone()))
        .await
        .expect("`transcode in.wav out.flac` must succeed (flagship journey)");

    let src = wav_pcm(&input);
    let flac = std::fs::read(&output).expect("read flac output");
    assert!(flac.starts_with(b"fLaC"), "output must be real FLAC");
    assert_ne!(
        flac[..flac.len().min(4096)],
        std::fs::read(&input).expect("src")[..4096.min(flac.len())],
        "output must not be a byte copy of the input"
    );

    let (params, decoded) = decode_flac_to_i16(&flac).expect("re-decode FLAC output");
    assert_eq!(params.sample_rate, 48_000);
    assert_eq!(params.channels, 2);
    assert_eq!(decoded.len(), src.len(), "sample count must survive");
    assert_eq!(
        decoded, src,
        "16-bit WAV→FLAC must be lossless (not silence)"
    );
    assert!(decoded.iter().any(|&s| s.abs() > 1_000), "not silence");

    std::fs::remove_file(&output).ok();
}

// ─── (b) `-af volume` shifts the encoded level ───────────────────────────────

#[tokio::test]
async fn af_volume_gain_shifts_flac_level() {
    let (_dir, input) = common::write_wav_fixture(500.0, 48_000, 1, 0.5);
    let output = temp_path("gain.flac");
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.audio_filter = Some("volume=-6.0206dB".to_string());
    transcode(opts).await.expect("gain transcode must succeed");

    let src_peak = wav_pcm(&input)
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    let (_, decoded) = decode_flac_to_i16(&std::fs::read(&output).expect("read")).expect("decode");
    let out_peak = decoded
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    assert!(
        (out_peak - src_peak / 2).abs() < src_peak / 20,
        "-6.02 dB must halve the level: src peak {src_peak}, out peak {out_peak}"
    );

    std::fs::remove_file(&output).ok();
}

// ─── Linear `-af volume=0.5` form ────────────────────────────────────────────

#[tokio::test]
async fn af_volume_linear_ratio_works() {
    let (_dir, input) = common::write_wav_fixture(500.0, 48_000, 1, 0.25);
    let output = temp_path("gain_linear.flac");
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.audio_filter = Some("volume=0.25".to_string());
    transcode(opts).await.expect("linear gain must succeed");

    let src_peak = wav_pcm(&input)
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    let (_, decoded) = decode_flac_to_i16(&std::fs::read(&output).expect("read")).expect("decode");
    let out_peak = decoded
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    assert!(
        (out_peak - src_peak / 4).abs() < src_peak / 20,
        "volume=0.25 must quarter the level: src {src_peak}, out {out_peak}"
    );

    std::fs::remove_file(&output).ok();
}

// ─── (c) Video: Y4M → MJPEG in MKV, frame 0 re-decodes ───────────────────────

#[tokio::test]
async fn y4m_to_mjpeg_mkv_frame_zero_decodes() {
    let input = temp_path("vid.y4m");
    let output = temp_path("vid.mkv");
    write_y4m_fixture(&input, 64, 48, 6);
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.video_codec = Some("mjpeg".to_string());
    transcode(opts).await.expect("y4m→mjpeg-mkv must succeed");

    let mkv = std::fs::read(&output).expect("read mkv");
    assert!(
        mkv.windows(7).any(|w| w == b"V_MJPEG"),
        "track must be labelled V_MJPEG"
    );
    assert_eq!(
        mkv.windows(3).filter(|w| w == &[0xFF, 0xD8, 0xFF]).count(),
        6,
        "one JPEG per source frame"
    );

    let jpeg = first_jpeg(&mkv).expect("embedded JPEG");
    let mut dec = oximedia_codec::MjpegDecoder::new(0, 0);
    dec.send_packet(jpeg, 0).expect("send");
    let frame = dec.receive_frame().expect("decode").expect("frame 0");
    assert_eq!((frame.width, frame.height), (64, 48), "dimensions");
    let luma = &frame.planes[0].data;
    let (min, max) = (
        luma.iter().copied().min().unwrap_or(0),
        luma.iter().copied().max().unwrap_or(0),
    );
    assert!(
        max - min > 32,
        "frame 0 must carry the source gradient, not a blank ({min}..{max})"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

// ─── (d) `-vf scale` changes output dimensions ───────────────────────────────

#[tokio::test]
async fn vf_scale_changes_output_dims() {
    let input = temp_path("vid_scale.y4m");
    let output = temp_path("vid_scale.mkv");
    write_y4m_fixture(&input, 64, 48, 3);
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.video_codec = Some("mjpeg".to_string());
    opts.video_filter = Some("scale=32:24".to_string());
    transcode(opts)
        .await
        .expect("-vf scale transcode must succeed");

    let mkv = std::fs::read(&output).expect("read mkv");
    let jpeg = first_jpeg(&mkv).expect("embedded JPEG");
    let mut dec = oximedia_codec::MjpegDecoder::new(0, 0);
    dec.send_packet(jpeg, 0).expect("send");
    let frame = dec.receive_frame().expect("decode").expect("frame");
    assert_eq!(
        (frame.width, frame.height),
        (32, 24),
        "-vf scale=32:24 must change the encoded dimensions"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

// ─── `-r` frame-rate conversion ──────────────────────────────────────────────

#[tokio::test]
async fn framerate_flag_converts_fps() {
    let input = temp_path("vid_fps.y4m");
    let output = temp_path("vid_fps.y4m_out.y4m");
    write_y4m_fixture(&input, 32, 32, 10);
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.framerate = Some("50".to_string());
    transcode(opts)
        .await
        .expect("-r 50 must succeed on Y4M output");

    let out = std::fs::read(&output).expect("read y4m");
    let header_end = out.iter().position(|&b| b == b'\n').expect("header");
    let header = String::from_utf8_lossy(&out[..header_end]).to_string();
    assert!(header.contains("F50:1"), "header must be 50 fps: {header}");
    assert_eq!(
        out.windows(5).filter(|w| w == b"FRAME").count(),
        20,
        "25→50 fps must duplicate frames"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

// ─── (e) Unsupported codec requests fail cleanly ─────────────────────────────

#[tokio::test]
async fn unsupported_vp9_encode_fails_cleanly() {
    let input = temp_path("vid_vp9.y4m");
    write_y4m_fixture(&input, 32, 32, 2);
    let output = temp_path("vid_vp9.mkv");
    let _ = std::fs::remove_file(&output);

    let mut opts = options(input.clone(), output.clone());
    opts.video_codec = Some("vp9".to_string());
    let err = transcode(opts)
        .await
        .expect_err("vp9 encode must be rejected");
    let msg = format!("{err:#}");
    assert!(msg.contains("VP9"), "error must name the codec: {msg}");
    assert!(
        msg.contains("mjpeg"),
        "error must list supported alternatives: {msg}"
    );
    assert!(
        !msg.contains("MultiTrackExecutor") && !msg.contains("FrameDecoder"),
        "error must not leak Rust internals: {msg}"
    );
    assert!(!output.exists(), "no output file may be fabricated");

    std::fs::remove_file(&input).ok();
}

#[tokio::test]
async fn unsupported_audio_codecs_fail_cleanly() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 0.2);

    for (codec, needle) in [("vorbis", "Vorbis"), ("opus", "Opus"), ("aac", "AAC")] {
        let output = temp_path(&format!("audio_{codec}.mka"));
        let _ = std::fs::remove_file(&output);
        let mut opts = options(input.clone(), output.clone());
        opts.audio_codec = Some(codec.to_string());
        let err = transcode(opts)
            .await
            .expect_err("unsupported audio codec must be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains(needle), "error must name {codec}: {msg}");
        assert!(!output.exists(), "no fabricated output for {codec}");
    }
}

// ─── Unsupported `-vf`/`-af` filters fail loudly, not silently ───────────────

#[tokio::test]
async fn unsupported_filters_are_rejected_not_dropped() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 0.2);
    let output = temp_path("filters.flac");

    let mut opts = options(input.clone(), output.clone());
    opts.audio_filter = Some("loudnorm=I=-23".to_string());
    let msg = format!(
        "{:#}",
        transcode(opts)
            .await
            .expect_err("loudnorm must be rejected")
    );
    assert!(msg.contains("loudnorm"), "must name the filter: {msg}");

    let mut opts = options(input.clone(), output.clone());
    opts.video_filter = Some("hflip".to_string());
    let msg = format!(
        "{:#}",
        transcode(opts).await.expect_err("hflip must be rejected")
    );
    assert!(msg.contains("hflip"), "must name the filter: {msg}");
}

// ─── PCM and ALAC targets ────────────────────────────────────────────────────

#[tokio::test]
async fn wav_to_wav_pcm_is_byte_exact() {
    let (_dir, input) = common::write_wav_fixture(700.0, 44_100, 2, 0.5);
    let output = temp_path("pcm_out.wav");
    let _ = std::fs::remove_file(&output);

    transcode(options(input.clone(), output.clone()))
        .await
        .expect("wav→wav must succeed");

    assert_eq!(
        wav_pcm(&input),
        wav_pcm(&output),
        "PCM re-encode must be sample-exact"
    );
    std::fs::remove_file(&output).ok();
}

#[tokio::test]
async fn wav_to_caf_alac_has_valid_structure() {
    let (_dir, input) = common::write_wav_fixture(880.0, 44_100, 2, 0.5);
    let output = temp_path("alac_out.caf");
    let _ = std::fs::remove_file(&output);

    transcode(options(input.clone(), output.clone()))
        .await
        .expect("wav→caf (ALAC) must succeed");

    let caf = std::fs::read(&output).expect("read caf");
    assert_eq!(&caf[..4], b"caff");
    for chunk in [&b"desc"[..], b"kuki", b"pakt", b"data"] {
        assert!(
            caf.windows(4).any(|w| w == chunk),
            "CAF must contain the {} chunk",
            String::from_utf8_lossy(chunk)
        );
    }
    assert!(caf.windows(4).any(|w| w == b"alac"), "desc must say alac");

    std::fs::remove_file(&output).ok();
}
