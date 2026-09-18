//! Integration tests for the `transcode` command's `-ss` / `-t` / `--map` /
//! `--threads` wiring, plus the removal of `--resume`.
//!
//! Library-level tests call `oximedia_cli::transcode::transcode` directly
//! (the same production path a CLI invocation runs — see
//! `transcode_normalize_audio.rs` for the rationale) against real WAV and
//! Matroska fixtures synthesized into `std::env::temp_dir()`. Binary-level
//! tests use `assert_cmd` where the observable behavior is CLI-surface
//! (clap rejection of `--resume`, the `--threads` stderr warning).
//!
//! Output extension is `.mka` (Matroska audio) so both codecs auto-detect to
//! `None` and the pipeline stays on its packet-level stream-copy path.

mod common;

use assert_cmd::Command;
use oximedia_cli::progress::ProgressFormat;
use oximedia_cli::transcode::{transcode, TranscodeOptions};
use oximedia_container::{
    demux::{Demuxer, MatroskaDemuxer},
    mux::{MatroskaMuxer, MuxerConfig},
    Muxer, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::{FileSource, MemorySource};
use std::path::{Path, PathBuf};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn temp_out(tag: &str, ext: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "oximedia_cli_trim_map_{tag}_{}.{ext}",
        std::process::id()
    ));
    // Best-effort cleanup of stale files from a previous crashed run.
    let _ = std::fs::remove_file(&path);
    path
}

/// `TranscodeOptions` at CLI defaults for `input` → `output`.
fn options_for(input: PathBuf, output: PathBuf) -> TranscodeOptions {
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

/// Re-demux a Matroska output; panics on any non-EOF packet error.
async fn demux_mka(path: &Path) -> (Vec<StreamInfo>, Vec<Packet>) {
    let source = FileSource::open(path).await.expect("open transcode output");
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe output as Matroska");
    let streams = demuxer.streams().to_vec();
    let mut packets = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => packets.push(pkt),
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("packet error while reading output: {e}"),
        }
    }
    (streams, packets)
}

fn pts_bounds(packets: &[Packet]) -> (f64, f64) {
    let min = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::INFINITY, f64::min);
    let max = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Two-stream (video + audio) Matroska fixture written to a temp file.
/// Returns the audio packet count.
async fn write_two_stream_mkv(path: &Path) -> usize {
    let sink = MemorySource::new_writable(256 * 1024);
    let mut muxer = MatroskaMuxer::new(sink, MuxerConfig::new());

    let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
    video.codec_params.width = Some(320);
    video.codec_params.height = Some(240);
    muxer.add_stream(video).expect("add video stream");

    let mut audio = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48_000));
    audio.codec_params.sample_rate = Some(48_000);
    audio.codec_params.channels = Some(2);
    muxer.add_stream(audio).expect("add audio stream");

    muxer.write_header().await.expect("write header");

    let audio_count = 50usize;
    // Interleave: one 33 ms video packet per ~1.65 audio packets (20 ms).
    let mut audio_written = 0usize;
    for i in 0..30i64 {
        let vpkt = Packet::new(
            0,
            vec![0x56, i as u8, 0x00, 0x01].into(),
            Timestamp::new(i * 33, Rational::new(1, 1000)),
            PacketFlags::KEYFRAME,
        );
        muxer.write_packet(&vpkt).await.expect("write video packet");
        while audio_written < audio_count
            && (audio_written as i64 * 960) as f64 / 48_000.0 <= (i as f64 * 0.033)
        {
            let apkt = Packet::new(
                1,
                vec![0x41, audio_written as u8, 0x00, 0x02].into(),
                Timestamp::new(audio_written as i64 * 960, Rational::new(1, 48_000)),
                PacketFlags::KEYFRAME,
            );
            muxer.write_packet(&apkt).await.expect("write audio packet");
            audio_written += 1;
        }
    }
    while audio_written < audio_count {
        let apkt = Packet::new(
            1,
            vec![0x41, audio_written as u8, 0x00, 0x02].into(),
            Timestamp::new(audio_written as i64 * 960, Rational::new(1, 48_000)),
            PacketFlags::KEYFRAME,
        );
        muxer.write_packet(&apkt).await.expect("write audio packet");
        audio_written += 1;
    }
    muxer.write_trailer().await.expect("write trailer");

    let sink = muxer.into_sink();
    std::fs::write(path, sink.written_data()).expect("write MKV fixture");
    audio_count
}

// ─── -t / -ss (library-level, real pipeline) ─────────────────────────────────

/// `-t 1` on a 2 s WAV must halve the output packet count (vs. an untrimmed
/// run of the same input) and cap the maximum retained PTS below 1 s.
#[tokio::test]
async fn duration_flag_truncates_output() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 2.0);
    let out_full = temp_out("t_full", "mka");
    let out_trimmed = temp_out("t_trimmed", "mka");

    transcode(options_for(input.clone(), out_full.clone()))
        .await
        .expect("untrimmed transcode should succeed");

    let mut trimmed = options_for(input.clone(), out_trimmed.clone());
    trimmed.duration = Some("1".to_string());
    transcode(trimmed)
        .await
        .expect("-t 1 transcode should succeed");

    let (_streams, full_packets) = demux_mka(&out_full).await;
    let (streams, trimmed_packets) = demux_mka(&out_trimmed).await;
    std::fs::remove_file(&out_full).ok();
    std::fs::remove_file(&out_trimmed).ok();

    assert_eq!(streams.len(), 1, "WAV remux keeps its single audio stream");
    assert!(
        !trimmed_packets.is_empty() && trimmed_packets.len() < full_packets.len(),
        "-t 1 must reduce the packet count (full: {}, trimmed: {})",
        full_packets.len(),
        trimmed_packets.len()
    );
    let (_min, max) = pts_bounds(&trimmed_packets);
    assert!(
        max < 1.0,
        "no retained packet may start at/after the 1 s limit, got {max}"
    );
}

/// `-ss 00:00:01` (HH:MM:SS timecode form) on a 2 s WAV exercises the
/// non-seekable read-and-discard fallback: minimum retained PTS ≥ 1 s.
#[tokio::test]
async fn start_time_flag_skips_leading_audio() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 2.0);
    let output = temp_out("ss_hms", "mka");

    let mut options = options_for(input, output.clone());
    options.start_time = Some("00:00:01".to_string());
    transcode(options)
        .await
        .expect("-ss transcode should succeed");

    let (_streams, packets) = demux_mka(&output).await;
    std::fs::remove_file(&output).ok();

    assert!(!packets.is_empty(), "audio after 1 s must be retained");
    let (min, _max) = pts_bounds(&packets);
    // Matroska stores ms timecodes; allow 1 ms of rounding.
    assert!(
        min >= 0.999,
        "-ss 00:00:01 must discard packets before 1 s, got min PTS {min}"
    );
}

// ─── --map (library-level) ───────────────────────────────────────────────────

/// End-to-end `--map`: 2-stream MKV → select audio only → the output holds
/// exactly that stream and every packet is remapped to stream_index 0.
#[tokio::test]
async fn map_selects_single_stream_from_two_stream_input() {
    let input = temp_out("map_e2e_in", "mkv");
    let output = temp_out("map_e2e_out", "mka");
    let audio_count = write_two_stream_mkv(&input).await;

    let mut options = options_for(input.clone(), output.clone());
    options.map = vec!["0:a".to_string()];
    transcode(options)
        .await
        .expect("--map 0:a transcode should succeed");

    let (streams, packets) = demux_mka(&output).await;
    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();

    assert_eq!(
        streams.len(),
        1,
        "--map 0:a must leave exactly one stream in the output"
    );
    assert!(streams[0].is_audio(), "the surviving stream must be audio");
    assert_eq!(
        packets.len(),
        audio_count,
        "all original audio packets must survive the remap"
    );
    assert!(
        packets.iter().all(|p| p.stream_index == 0),
        "every packet must be remapped to the muxer-positional index 0"
    );
}

/// A `--map` selector that matches nothing must fail with an actionable
/// error and must not create the output file.
#[tokio::test]
async fn map_selector_matching_nothing_fails_cleanly() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 0.5);
    let output = temp_out("map_miss", "mka");

    let mut options = options_for(input, output.clone());
    options.map = vec!["0:v".to_string()];
    let msg = transcode(options)
        .await
        .expect_err("0:v on an audio-only WAV must fail")
        .to_string();

    assert!(
        msg.contains("matched no streams"),
        "error must state the miss, got: {msg}"
    );
    assert!(
        !output.exists(),
        "no output file may be created when --map resolution fails"
    );
}

/// An invalid selector must be rejected before any pipeline work, with the
/// accepted grammar in the message.
#[tokio::test]
async fn invalid_map_selector_is_rejected_with_grammar_help() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 0.5);
    let output = temp_out("map_bad", "mka");

    let mut options = options_for(input, output.clone());
    options.map = vec!["0:x".to_string()];
    let msg = transcode(options)
        .await
        .expect_err("invalid selector must be rejected")
        .to_string();

    assert!(
        msg.contains("valid --map selectors"),
        "error must list the accepted grammar, got: {msg}"
    );
    assert!(!output.exists(), "no output on selector parse failure");
}

/// Invalid `-ss` must fail fast, before the output file exists.
#[tokio::test]
async fn invalid_start_time_fails_before_output() {
    let (_dir, input) = common::write_wav_fixture(440.0, 48_000, 2, 0.5);
    let output = temp_out("bad_ss", "mka");

    let mut options = options_for(input, output.clone());
    options.start_time = Some("not-a-time".to_string());
    let msg = transcode(options)
        .await
        .expect_err("garbage -ss must be rejected")
        .to_string();

    assert!(msg.contains("-ss"), "error must name the flag, got: {msg}");
    assert!(!output.exists(), "no output on -ss parse failure");
}

// ─── CLI surface (binary-level) ──────────────────────────────────────────────

fn oximedia() -> Command {
    Command::cargo_bin("oximedia").expect("oximedia binary should exist")
}

/// `--resume` was removed (it never did anything); clap must now reject it.
#[test]
fn resume_flag_is_rejected_by_clap() {
    let assert = oximedia()
        .args(["transcode", "-i", "in.mka", "-o", "out.mka", "--resume"])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();
    assert!(
        stderr.contains("--resume") && stderr.contains("unexpected argument"),
        "clap must reject --resume as unknown, got:\n{stderr}"
    );
}

/// `--resume` no longer appears in the transcode help text.
#[test]
fn resume_flag_absent_from_help() {
    let out = oximedia()
        .args(["transcode", "--help"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let help = String::from_utf8_lossy(&out);
    assert!(
        !help.contains("--resume"),
        "transcode --help must not advertise --resume, got:\n{help}"
    );
    assert!(
        help.contains("--map"),
        "transcode --help must document --map, got:\n{help}"
    );
}

/// An explicit non-zero `--threads` must produce the honest stderr warning
/// and still succeed; the default (0 = auto) stays silent.
#[test]
fn threads_flag_warns_on_stderr_and_proceeds() {
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let input = tmp.join(format!("oximedia_cli_threads_warn_in_{pid}.wav"));
    let output = tmp.join(format!("oximedia_cli_threads_warn_out_{pid}.mka"));
    let _ = std::fs::remove_file(&output);
    std::fs::write(&input, common::make_sine_wav(440.0, 48_000, 2, 0.25))
        .expect("write WAV fixture");

    let assert = oximedia()
        .args([
            "transcode",
            "-i",
            input.to_str().expect("utf-8 temp path"),
            "-o",
            output.to_str().expect("utf-8 temp path"),
            "--threads",
            "8",
            "-y",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_string();

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();

    assert!(
        stderr.contains("--threads has no effect"),
        "an explicit --threads must warn honestly on stderr, got:\n{stderr}"
    );
}
