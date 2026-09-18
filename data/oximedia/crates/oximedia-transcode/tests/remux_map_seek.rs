//! Integration tests for `--map` stream selection and `-ss`/`-t` trimming
//! in the packet-level remux pipeline (`TranscodePipeline`).
//!
//! Real Matroska and WAV fixtures are synthesized into `std::env::temp_dir()`
//! with the workspace's own muxers, run through the full
//! `TranscodePipeline::execute()` path, and the outputs re-demuxed to assert
//! stream counts, packet-index remapping, and PTS windows.
//!
//! The single most important assertion here is in
//! [`map_audio_only_selects_and_remaps`]: after filtering, every surviving
//! packet must carry `stream_index == 0` in the output. The Matroska and Ogg
//! muxers route `write_packet` by the packet's position in the muxer's own
//! stream list, so without the drain-loop remap the muxer would reject every
//! packet from the originally-second stream as an invalid stream index.

use oximedia_container::{
    demux::{Demuxer, MatroskaDemuxer},
    mux::{MatroskaMuxer, MuxerConfig},
    Muxer, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::{FileSource, MemorySource};
use oximedia_transcode::{StreamMap, TranscodePipelineBuilder};
use std::path::{Path, PathBuf};

// ─── Fixture constants ───────────────────────────────────────────────────────

/// Video fixture timing: one packet every 33 ms (≈30 fps), timebase 1/1000.
const VIDEO_STEP_MS: i64 = 33;

/// Audio fixture timing: one packet every 960 ticks at 1/48000 (20 ms).
const AUDIO_STEP_TICKS: i64 = 960;

/// WAV fixture sample rate.
const WAV_SAMPLE_RATE: u32 = 48_000;

// ─── Fixture builders ────────────────────────────────────────────────────────

/// Unique temp path for this test run.
fn temp_path(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "oximedia_remux_map_seek_{tag}_{}.{ext}",
        std::process::id()
    ))
}

fn video_stream(index: usize) -> StreamInfo {
    let mut video = StreamInfo::new(index, CodecId::Vp9, Rational::new(1, 1000));
    video.codec_params.width = Some(320);
    video.codec_params.height = Some(240);
    video
}

fn audio_stream(index: usize) -> StreamInfo {
    let mut audio = StreamInfo::new(index, CodecId::Opus, Rational::new(1, 48_000));
    audio.codec_params.sample_rate = Some(48_000);
    audio.codec_params.channels = Some(2);
    audio
}

fn video_packet(stream_index: usize, i: i64) -> Packet {
    // Distinct payload per packet so byte-level accounting is meaningful.
    let data = vec![0x56u8, (i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0x01];
    Packet::new(
        stream_index,
        bytes::Bytes::from(data),
        Timestamp::new(i * VIDEO_STEP_MS, Rational::new(1, 1000)),
        PacketFlags::KEYFRAME,
    )
}

fn audio_packet(stream_index: usize, i: i64) -> Packet {
    let data = vec![0x41u8, (i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0x02];
    Packet::new(
        stream_index,
        bytes::Bytes::from(data),
        Timestamp::new(i * AUDIO_STEP_TICKS, Rational::new(1, 48_000)),
        PacketFlags::KEYFRAME,
    )
}

/// Build a Matroska file with the given streams/packets and write it to `path`.
async fn write_mkv_fixture(path: &Path, streams: Vec<StreamInfo>, packets: Vec<Packet>) {
    let sink = MemorySource::new_writable(512 * 1024);
    let mut muxer = MatroskaMuxer::new(sink, MuxerConfig::new());
    for stream in streams {
        muxer.add_stream(stream).expect("add fixture stream");
    }
    muxer.write_header().await.expect("write fixture header");
    for pkt in &packets {
        muxer.write_packet(pkt).await.expect("write fixture packet");
    }
    muxer.write_trailer().await.expect("write fixture trailer");

    let sink = muxer.into_sink();
    tokio::fs::write(path, sink.written_data())
        .await
        .expect("write fixture file");
}

/// Two-stream fixture: video (stream 0, 30 packets ≈ 1 s) + audio
/// (stream 1, 50 packets ≈ 1 s), interleaved in time order.
async fn write_two_stream_fixture(path: &Path) -> (usize, usize) {
    let video_count = 30usize;
    let audio_count = 50usize;

    let mut packets: Vec<Packet> = Vec::with_capacity(video_count + audio_count);
    packets.extend((0..video_count as i64).map(|i| video_packet(0, i)));
    packets.extend((0..audio_count as i64).map(|i| audio_packet(1, i)));
    packets.sort_by(|a, b| {
        a.timestamp
            .to_seconds()
            .partial_cmp(&b.timestamp.to_seconds())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    write_mkv_fixture(path, vec![video_stream(0), audio_stream(1)], packets).await;
    (video_count, audio_count)
}

/// Video-only fixture with `count` packets, 33 ms apart.
async fn write_video_fixture(path: &Path, count: i64) {
    let packets: Vec<Packet> = (0..count).map(|i| video_packet(0, i)).collect();
    write_mkv_fixture(path, vec![video_stream(0)], packets).await;
}

/// Minimal 16-bit stereo PCM WAV with a deterministic ramp payload.
fn make_wav_bytes(duration_secs: f64) -> Vec<u8> {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let num_samples = (f64::from(WAV_SAMPLE_RATE) * duration_secs) as u32;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = WAV_SAMPLE_RATE * u32::from(block_align);
    let data_size = num_samples * u32::from(block_align);

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_size).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&WAV_SAMPLE_RATE.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for i in 0..num_samples {
        let sample = (i % 2048) as i16;
        buf.extend_from_slice(&sample.to_le_bytes());
        buf.extend_from_slice(&sample.to_le_bytes());
    }
    buf
}

// ─── Output inspection ───────────────────────────────────────────────────────

/// Re-demux a Matroska output; panics on any non-EOF packet error, so every
/// test implicitly asserts a zero-packet-error drain.
async fn demux_mkv(path: &Path) -> (Vec<StreamInfo>, Vec<Packet>) {
    let source = FileSource::open(path).await.expect("open output");
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

fn parse_maps(selectors: &[&str]) -> Vec<StreamMap> {
    selectors
        .iter()
        .map(|s| StreamMap::parse(s).unwrap_or_else(|e| panic!("'{s}' must parse: {e}")))
        .collect()
}

fn cleanup(paths: &[&Path]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

// ─── --map tests ─────────────────────────────────────────────────────────────

/// Selecting only the audio stream of a two-stream file must (a) leave
/// exactly one stream in the output and (b) rewrite every surviving packet
/// to `stream_index == 0` — proving the positional remap. Without the remap
/// the Matroska muxer errors with "Invalid stream index: 1" on the first
/// audio packet, so a successful, full-count drain is itself the proof.
#[tokio::test]
async fn map_audio_only_selects_and_remaps() {
    let input = temp_path("map_audio_in", "mkv");
    let output = temp_path("map_audio_out", "mkv");
    let (_video_count, audio_count) = write_two_stream_fixture(&input).await;

    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .stream_map(parse_maps(&["0:a"]))
        .build()
        .expect("build pipeline");
    let result = pipeline.execute().await;
    assert!(result.is_ok(), "map 0:a transcode failed: {result:?}");

    let (streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert_eq!(
        streams.len(),
        1,
        "exactly one stream must survive --map 0:a"
    );
    assert!(streams[0].is_audio(), "the surviving stream must be audio");
    assert_eq!(
        packets.len(),
        audio_count,
        "every original audio packet must survive"
    );
    assert!(
        packets.iter().all(|p| p.stream_index == 0),
        "every packet must be remapped to stream_index 0 \
         (original audio index was 1)"
    );
}

/// `0:0` (absolute stream index) keeps only the video stream.
#[tokio::test]
async fn map_by_absolute_index_keeps_video() {
    let input = temp_path("map_index_in", "mkv");
    let output = temp_path("map_index_out", "mkv");
    let (video_count, _audio_count) = write_two_stream_fixture(&input).await;

    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .stream_map(parse_maps(&["0:0"]))
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("map 0:0 transcode");

    let (streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert_eq!(streams.len(), 1);
    assert!(streams[0].is_video());
    assert_eq!(packets.len(), video_count);
    assert!(packets.iter().all(|p| p.stream_index == 0));
}

/// A pure-negative map (`-0:a`) subtracts from the full stream set.
#[tokio::test]
async fn negative_map_drops_audio() {
    let input = temp_path("map_neg_in", "mkv");
    let output = temp_path("map_neg_out", "mkv");
    let (video_count, _audio_count) = write_two_stream_fixture(&input).await;

    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .stream_map(parse_maps(&["-0:a"]))
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("map -0:a transcode");

    let (streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert_eq!(streams.len(), 1, "audio must be excluded");
    assert!(streams[0].is_video());
    assert_eq!(packets.len(), video_count);
}

/// A positive selector matching nothing must fail with an actionable error
/// before any output file is created.
#[tokio::test]
async fn map_unmatched_positive_selector_fails() {
    let input = temp_path("map_miss_in", "mkv");
    let output = temp_path("map_miss_out", "mkv");
    write_two_stream_fixture(&input).await;

    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .stream_map(parse_maps(&["0:s"]))
        .build()
        .expect("build pipeline");
    let err = pipeline
        .execute()
        .await
        .expect_err("0:s must not match a video+audio file");
    let msg = err.to_string();
    let output_exists = output.exists();
    cleanup(&[&input, &output]);

    assert!(
        msg.contains("matched no streams"),
        "error must state the miss, got: {msg}"
    );
    assert!(
        msg.contains("valid --map selectors"),
        "error must list the accepted grammar, got: {msg}"
    );
    assert!(
        !output_exists,
        "no output file may be created when --map resolution fails"
    );
}

// ─── -t / -ss tests (Matroska, seekable path) ────────────────────────────────

/// `-t` alone must truncate the output at the stop PTS.
#[tokio::test]
async fn duration_truncates_matroska_video() {
    let input = temp_path("dur_in", "mkv");
    let output = temp_path("dur_out", "mkv");
    let total = 30i64; // ≈1 s of 33 ms packets
    write_video_fixture(&input, total).await;

    let limit_secs = 0.5f64;
    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .duration_secs(limit_secs)
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("-t transcode");

    let (streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    // Expected: exactly the packets whose PTS is below the stop point.
    let expected = (0..total)
        .filter(|i| (i * VIDEO_STEP_MS) as f64 / 1000.0 < limit_secs)
        .count();
    assert_eq!(streams.len(), 1);
    assert_eq!(
        packets.len(),
        expected,
        "-t {limit_secs}s must keep exactly the packets with PTS below it"
    );
    let max_pts = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_pts < limit_secs,
        "no retained packet may reach the -t stop point, got {max_pts}"
    );
}

/// `-ss` on a seekable Matroska input: the demuxer seek is attempted (and,
/// on this crate's own trailer-cued files, silently rewinds to the start —
/// the demuxer's header parse stops at the first cluster and never sees the
/// trailer-positioned Cues element), so the pipeline's PTS discard must
/// still deliver exact `-ss` semantics: minimum retained PTS ≥ the target,
/// within one packet of it.
#[tokio::test]
async fn start_time_seeks_matroska_video() {
    let input = temp_path("seek_in", "mkv");
    let output = temp_path("seek_out", "mkv");
    // 12 s of video spanning three ≈5 s clusters.
    let total = 364i64;
    write_video_fixture(&input, total).await;

    let start_secs = 6.0f64;
    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .start_time_secs(start_secs)
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("-ss transcode");

    let (_streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert!(
        !packets.is_empty(),
        "packets after the seek point must be retained"
    );
    // Exactly the packets with PTS ≥ 6 s survive.
    let expected = (0..total)
        .filter(|i| (i * VIDEO_STEP_MS) as f64 / 1000.0 >= start_secs)
        .count();
    assert_eq!(
        packets.len(),
        expected,
        "-ss {start_secs}s must drop exactly the packets before the target"
    );
    let min_pts = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_pts >= start_secs - 0.001,
        "-ss must not retain packets before the target, got {min_pts}"
    );
    assert!(
        min_pts <= start_secs + 0.1,
        "-ss must start within one packet of the target, got {min_pts}"
    );
}

// ─── -ss / -t on WAV (non-seekable read-and-discard fallback) ────────────────

/// WAV's demuxer supports no seeking, so `-ss` must take the
/// read-and-discard path — which is sample-accurate: every retained packet
/// PTS is ≥ the start point exactly.
#[tokio::test]
async fn wav_start_time_uses_discard_fallback() {
    let input = temp_path("wav_ss_in", "wav");
    let output = temp_path("wav_ss_out", "mka");
    std::fs::write(&input, make_wav_bytes(2.0)).expect("write WAV fixture");

    // Reference: demux the input to know its packet layout.
    let ref_source = FileSource::open(&input).await.expect("open WAV");
    let mut ref_demuxer = oximedia_container::demux::WavDemuxer::new(ref_source);
    ref_demuxer.probe().await.expect("probe WAV");
    let mut input_packets = 0usize;
    let mut expected_retained = 0usize;
    let start_secs = 1.0f64;
    loop {
        match ref_demuxer.read_packet().await {
            Ok(pkt) => {
                input_packets += 1;
                if pkt.timestamp.to_seconds() >= start_secs {
                    expected_retained += 1;
                }
            }
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("WAV reference demux error: {e}"),
        }
    }

    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .start_time_secs(start_secs)
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("WAV -ss transcode");

    let (streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert_eq!(streams.len(), 1);
    assert!(streams[0].is_audio());
    assert_eq!(
        packets.len(),
        expected_retained,
        "discard fallback must keep exactly the packets at/after {start_secs}s \
         (input had {input_packets})"
    );
    let min_pts = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::INFINITY, f64::min);
    // Matroska stores ms timecodes, so allow 1 ms of rounding.
    assert!(
        min_pts >= start_secs - 0.001,
        "discard fallback is PTS-exact; got min PTS {min_pts}"
    );
}

/// `-ss` + `-t` on WAV: the retained PTS window is [start, start + duration).
#[tokio::test]
async fn wav_start_plus_duration_window() {
    let input = temp_path("wav_window_in", "wav");
    let output = temp_path("wav_window_out", "mka");
    std::fs::write(&input, make_wav_bytes(2.0)).expect("write WAV fixture");

    let start_secs = 0.5f64;
    let duration_secs = 1.0f64;
    let mut pipeline = TranscodePipelineBuilder::new()
        .input(input.clone())
        .output(output.clone())
        .start_time_secs(start_secs)
        .duration_secs(duration_secs)
        .build()
        .expect("build pipeline");
    pipeline.execute().await.expect("WAV -ss -t transcode");

    let (_streams, packets) = demux_mkv(&output).await;
    cleanup(&[&input, &output]);

    assert!(!packets.is_empty(), "window must retain packets");
    let min_pts = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::INFINITY, f64::min);
    let max_pts = packets
        .iter()
        .map(|p| p.timestamp.to_seconds())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_pts >= start_secs - 0.001,
        "window start violated: {min_pts}"
    );
    assert!(
        max_pts < start_secs + duration_secs,
        "window end violated (FFmpeg -t measures from the seek point): {max_pts}"
    );
}
