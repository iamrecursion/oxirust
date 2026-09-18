//! Round-trip proof that Matroska/`WebM` seeking is real end-to-end on files
//! this workspace's own muxer produces.
//!
//! Historically, [`MatroskaMuxer`] wrote `Cues` (the seek index) in the
//! trailer -- after every `Cluster`, since cluster byte positions aren't
//! known any earlier in a single-pass muxer -- but never emitted a
//! `SeekHead` to point at them. [`MatroskaDemuxer`]'s header parser stops at
//! the first `Cluster` (to avoid buffering the whole file), so it could only
//! ever see a `Cues` element written *before* every `Cluster`. The practical
//! result: [`Demuxer::seek_to_time`] on a self-muxed file silently found no
//! cue points, fell back to "seek to segment start", and returned `Ok(())`
//! -- a caller asking to seek to 30s would silently get the file from 0s.
//!
//! This test builds a real, multi-cluster `WebM` file with the workspace's
//! own muxer, and proves three things about the fix:
//!
//! 1. [`test_seek_head_indexes_a_real_cues_element`] -- independently of
//!    this crate's own EBML parser, the raw bytes contain a `SeekHead` as
//!    the first child of `Segment`, and its `Cues` entry's `SeekPosition`
//!    resolves to the byte offset of an actual `Cues` element.
//! 2. [`test_seek_to_time_repositions_into_the_correct_cluster`] -- seeking
//!    with the demuxer's own `Demuxer::seek_to_time` genuinely repositions:
//!    the next packet read has the requested timestamp, not the segment's
//!    first packet's timestamp.
//! 3. [`test_full_file_still_parses_start_to_finish`] -- adding the
//!    `SeekHead` and fixing cue cluster positions did not regress ordinary,
//!    non-seeking sequential playback.

use bytes::Bytes;
use oximedia_container::{
    demux::{Demuxer, MatroskaDemuxer},
    mux::{MatroskaMuxer, MuxerConfig, OutputFormat},
    Muxer, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};
use oximedia_io::{FileSource, MemorySource};
use std::path::{Path, PathBuf};

/// Number of keyframes (and therefore clusters and cue points) in the
/// fixture. Each keyframe is 1000 ms apart with `max_cluster_duration_ms`
/// set well below that, so every single frame starts a fresh cluster --
/// this is deliberately a *multi*-cluster file, not a one-cluster edge case.
const FRAME_COUNT: i64 = 12;

/// Spacing between frames, in milliseconds. The video stream's timebase is
/// `1/1000`, so packet PTS values are already in millisecond units and map
/// 1:1 onto Matroska timecode-scale units (default scale = 1 ms).
const FRAME_STEP_MS: i64 = 1000;

/// Forces a new cluster on every frame (`FRAME_STEP_MS` > this).
const MAX_CLUSTER_DURATION_MS: u32 = 500;

// ─── Fixture ──────────────────────────────────────────────────────────────

/// Unique path for this test run under `std::env::temp_dir()`.
fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "oximedia_container_mkv_seekhead_{tag}_{}.webm",
        std::process::id()
    ))
}

fn video_stream() -> StreamInfo {
    let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
    video.codec_params.width = Some(320);
    video.codec_params.height = Some(240);
    video
}

fn video_packet(i: i64) -> Packet {
    let data = vec![0x56u8, (i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0x01];
    Packet::new(
        0,
        Bytes::from(data),
        Timestamp::new(i * FRAME_STEP_MS, Rational::new(1, 1000)),
        PacketFlags::KEYFRAME,
    )
}

/// Muxes [`FRAME_COUNT`] keyframes (each forced into its own cluster) with
/// this crate's own [`MatroskaMuxer`], and writes the result to a real file
/// at `path`.
async fn write_fixture(path: &Path) {
    let sink = MemorySource::new_writable(512 * 1024);
    let config = MuxerConfig::new().with_output_format(
        OutputFormat::new().with_max_cluster_duration_ms(MAX_CLUSTER_DURATION_MS),
    );
    let mut muxer = MatroskaMuxer::new(sink, config);
    muxer
        .add_stream(video_stream())
        .expect("add fixture video stream");
    muxer.write_header().await.expect("write fixture header");

    for i in 0..FRAME_COUNT {
        muxer
            .write_packet(&video_packet(i))
            .await
            .expect("write fixture packet");
    }
    muxer.write_trailer().await.expect("write fixture trailer");

    let sink = muxer.into_sink();
    tokio::fs::write(path, sink.written_data())
        .await
        .expect("write fixture file to temp_dir");
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

// ─── Minimal, independent EBML reader (does *not* use this crate's parser) ──
//
// These helpers deliberately reimplement just enough of the EBML VINT
// algorithm (the same one the Matroska spec, and this crate's `ebml.rs`,
// both describe) to walk the raw muxed bytes by hand. Using a from-scratch
// reader here -- rather than calling back into `oximedia_container`'s own
// EBML parser -- means the structural assertions below are checking the
// muxer's bytes independently of whether the demuxer's parser has some
// matching bug.

/// Reads an EBML element ID (class-marker bit retained), returning
/// `(value, byte_length)`.
fn read_id_vint(data: &[u8]) -> (u32, usize) {
    let first = data[0];
    let len = (first.leading_zeros() as usize) + 1;
    let mut value = u32::from(first);
    for &b in &data[1..len] {
        value = (value << 8) | u32::from(b);
    }
    (value, len)
}

/// Reads an EBML element size (class-marker bit stripped), returning
/// `(value, byte_length)`.
fn read_size_vint(data: &[u8]) -> (u64, usize) {
    let first = data[0];
    let len = (first.leading_zeros() as usize) + 1;
    let mask: u8 = if len >= 8 { 0 } else { 0xFFu8 >> len };
    let mut value = u64::from(first & mask);
    for &b in &data[1..len] {
        value = (value << 8) | u64::from(b);
    }
    (value, len)
}

/// Walks the raw muxed bytes far enough to independently prove that:
///
/// 1. The very first child of `Segment` is a `SeekHead` element.
/// 2. That `SeekHead` contains a `Seek` entry whose `SeekID` resolves to
///    `Cues` (EBML id `0x1C53BB6B`), with a non-zero `SeekPosition`.
/// 3. That `SeekPosition`, interpreted per spec as an offset relative to
///    `Segment`'s data start, really lands on the first byte of an actual
///    `Cues` element later in the file (not zero, not garbage).
///
/// Returns the set of distinct element IDs indexed by `Seek` entries, so
/// the caller can additionally check that `Info`/`Tracks` are indexed too.
fn assert_seek_head_indexes_real_cues(data: &[u8]) -> std::collections::HashSet<u32> {
    let mut pos = 0usize;

    // EBML header: `1A 45 DF A3`.
    let (ebml_id, id_len) = read_id_vint(&data[pos..]);
    assert_eq!(
        ebml_id, 0x1A45_DFA3,
        "file must start with an EBML header element"
    );
    pos += id_len;
    let (ebml_size, size_len) = read_size_vint(&data[pos..]);
    pos += size_len + ebml_size as usize;

    // Segment: `18 53 80 67`, unknown (streamed) size.
    let (seg_id, seg_id_len) = read_id_vint(&data[pos..]);
    assert_eq!(
        seg_id, 0x1853_8067,
        "EBML header must be followed by Segment"
    );
    pos += seg_id_len;
    let (_seg_size, seg_size_len) = read_size_vint(&data[pos..]);
    pos += seg_size_len;
    let segment_data_start = pos;

    // First child of Segment must be SeekHead: `11 4D 9B 74`.
    let (first_id, first_id_len) = read_id_vint(&data[pos..]);
    assert_eq!(
        first_id, 0x114D_9B74,
        "the first element written inside Segment must be SeekHead, got 0x{first_id:X}"
    );
    pos += first_id_len;
    let (seek_head_size, seek_head_size_len) = read_size_vint(&data[pos..]);
    pos += seek_head_size_len;
    let seek_head_content = &data[pos..pos + seek_head_size as usize];

    // Walk every Seek (`4D BB`) entry inside the SeekHead.
    let mut epos = 0usize;
    let mut cues_position: Option<u64> = None;
    let mut target_ids = std::collections::HashSet::new();
    while epos < seek_head_content.len() {
        let (entry_id, entry_id_len) = read_id_vint(&seek_head_content[epos..]);
        epos += entry_id_len;
        let (entry_size, entry_size_len) = read_size_vint(&seek_head_content[epos..]);
        epos += entry_size_len;
        let entry = &seek_head_content[epos..epos + entry_size as usize];
        assert_eq!(
            entry_id, 0x4DBB,
            "every SeekHead child must be a Seek entry, got 0x{entry_id:X}"
        );

        let mut cpos = 0usize;
        let mut seek_target: Option<u32> = None;
        let mut seek_position: Option<u64> = None;
        while cpos < entry.len() {
            let (cid, cid_len) = read_id_vint(&entry[cpos..]);
            cpos += cid_len;
            let (csize, csize_len) = read_size_vint(&entry[cpos..]);
            cpos += csize_len;
            let cdata = &entry[cpos..cpos + csize as usize];
            match cid {
                0x53AB => seek_target = Some(read_id_vint(cdata).0),
                0x53AC => {
                    let mut v = 0u64;
                    for &b in cdata {
                        v = (v << 8) | u64::from(b);
                    }
                    seek_position = Some(v);
                }
                _ => {}
            }
            cpos += csize as usize;
        }

        let target = seek_target.expect("Seek entry must contain a SeekID");
        let position = seek_position.expect("Seek entry must contain a SeekPosition");
        target_ids.insert(target);
        if target == 0x1C53_BB6B {
            cues_position = Some(position);
        }

        epos += entry_size as usize;
    }

    let cues_position =
        cues_position.expect("SeekHead must contain a Seek entry indexing Cues (0x1C53BB6B)");
    assert!(
        cues_position > 0,
        "Cues SeekPosition must not be zero (that would alias Segment's own start)"
    );

    // Resolve: the byte at `segment_data_start + cues_position` must really
    // be the start of a Cues element.
    let cues_offset = segment_data_start + cues_position as usize;
    let (resolved_id, _) = read_id_vint(&data[cues_offset..]);
    assert_eq!(
        resolved_id, 0x1C53_BB6B,
        "SeekHead's Cues SeekPosition must resolve to a real Cues element, got 0x{resolved_id:X} at offset {cues_offset}"
    );

    target_ids
}

// ─── Tests ────────────────────────────────────────────────────────────────

/// Structural proof: the muxer emits a real `SeekHead` (not just `Cues`
/// sitting unindexed in the trailer), and that `SeekHead` genuinely points
/// at the trailer-positioned `Cues` element -- checked with a from-scratch
/// EBML walk, independent of this crate's own parser.
#[tokio::test]
async fn test_seek_head_indexes_a_real_cues_element() {
    let path = temp_path("structure");
    write_fixture(&path).await;

    let data = tokio::fs::read(&path).await.expect("read fixture back");
    let target_ids = assert_seek_head_indexes_real_cues(&data);

    // Info and Tracks are always written, so the muxer should index them
    // too, not just Cues.
    assert!(
        target_ids.contains(&0x1549_A966),
        "SeekHead should also index Info"
    );
    assert!(
        target_ids.contains(&0x1654_AE6B),
        "SeekHead should also index Tracks"
    );
    assert_eq!(
        target_ids.len(),
        3,
        "expected exactly Info+Tracks+Cues entries"
    );

    cleanup(&path);
}

/// The core round-trip proof: `seek_to_time` on a self-muxed, multi-cluster
/// file must actually reposition to the requested timestamp's cluster, not
/// silently rewind to the segment start.
#[tokio::test]
async fn test_seek_to_time_repositions_into_the_correct_cluster() {
    let path = temp_path("seek");
    write_fixture(&path).await;

    let source = FileSource::open(&path).await.expect("open fixture");
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe fixture");

    // The whole point of the fix: cue points recovered via the SeekHead,
    // not silently dropped because they live in the trailer.
    assert_eq!(
        demuxer.cues().len(),
        FRAME_COUNT as usize,
        "all cue points must be recovered by following the SeekHead to the trailer"
    );
    assert!(
        demuxer.is_seekable(),
        "a probed file on FileSource must be seekable"
    );

    // Seek to the timestamp of frame index 6 (6.0s) -- comfortably in the
    // middle of an 11-second, 12-cluster file.
    const TARGET_FRAME: i64 = 6;
    let target_secs = (TARGET_FRAME * FRAME_STEP_MS) as f64 / 1000.0;
    demuxer
        .seek_to_time(target_secs)
        .await
        .expect("seek_to_time must succeed on a seekable, probed demuxer");

    let packet = demuxer
        .read_packet()
        .await
        .expect("read_packet after seek must succeed");

    let expected_pts = TARGET_FRAME * FRAME_STEP_MS;
    assert_ne!(
        packet.timestamp.pts, 0,
        "seek_to_time({target_secs}) must not silently land back at the segment start"
    );
    assert_eq!(
        packet.timestamp.pts, expected_pts,
        "seek_to_time({target_secs}) must land exactly on the frame at that timestamp"
    );

    // Draining the rest must yield exactly the remaining frames, in order,
    // all at/after the seek target -- proving the demuxer is positioned in
    // the right cluster, not just coincidentally returning one right packet.
    let mut pts_values = vec![packet.timestamp.pts];
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => pts_values.push(pkt.timestamp.pts),
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("unexpected error reading after seek: {e}"),
        }
    }

    let expected: Vec<i64> = (TARGET_FRAME..FRAME_COUNT)
        .map(|i| i * FRAME_STEP_MS)
        .collect();
    assert_eq!(
        pts_values, expected,
        "packets after seek_to_time({target_secs}) must be exactly frames {TARGET_FRAME}..{FRAME_COUNT}"
    );

    cleanup(&path);
}

/// Regression guard: normal, non-seeking sequential playback of the whole
/// file must be unaffected by the new SeekHead and the cue cluster-position
/// fix -- every frame, in order, with correct timestamps.
#[tokio::test]
async fn test_full_file_still_parses_start_to_finish() {
    let path = temp_path("full");
    write_fixture(&path).await;

    let source = FileSource::open(&path).await.expect("open fixture");
    let mut demuxer = MatroskaDemuxer::new(source);
    let probe = demuxer.probe().await.expect("probe fixture");
    assert!(probe.confidence > 0.9, "probe confidence must be high");

    let streams = demuxer.streams();
    assert_eq!(streams.len(), 1, "fixture has exactly one video stream");
    assert_eq!(streams[0].codec, CodecId::Vp9);

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => {
                assert!(pkt.is_keyframe(), "every fixture frame is a keyframe");
                pts_values.push(pkt.timestamp.pts);
            }
            Err(e) if e.is_eof() => break,
            Err(e) => panic!("unexpected error during full read: {e}"),
        }
    }

    let expected: Vec<i64> = (0..FRAME_COUNT).map(|i| i * FRAME_STEP_MS).collect();
    assert_eq!(
        pts_values, expected,
        "sequential read of the whole file must return every frame, in order, unaffected by \
         the new SeekHead / trailer Cues"
    );

    cleanup(&path);
}
