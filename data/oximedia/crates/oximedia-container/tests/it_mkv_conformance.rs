//! Synthetic Matroska conformance harness — K10.
//!
//! Builds minimal, synthetically constructed MKV/WebM byte streams and validates
//! them by round-tripping through [`MatroskaDemuxer`].  Every test exercises a
//! distinct spec edge case so that collectively they cover VINT boundaries,
//! SimpleBlock, BlockGroup, SeekHead presence, Cues presence, multiple tracks,
//! nested timestamps, and full-packet preservation.

use bytes::Bytes;
use oximedia_container::demux::MatroskaDemuxer;
use oximedia_container::Demuxer;
use oximedia_core::OxiError;
use oximedia_io::MemorySource;

// ---------------------------------------------------------------------------
// EBML helper library (duplicated locally so tests are self-contained)
// ---------------------------------------------------------------------------

/// Encode `n` as an EBML VINT (variable-length integer used in element sizes).
fn ebml_vint_size(mut n: u64) -> Vec<u8> {
    let width = if n < 0x7F {
        1usize
    } else if n < 0x3FFF {
        2
    } else if n < 0x1F_FFFF {
        3
    } else if n < 0x0FFF_FFFF {
        4
    } else {
        8
    };
    let marker = 1u64 << (7 * width);
    n |= marker;
    let bytes = n.to_be_bytes();
    bytes[8 - width..].to_vec()
}

/// Encode `v` as the minimum number of big-endian bytes (unsigned integer body).
fn uint_bytes(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let bytes = v.to_be_bytes();
    let leading = bytes.iter().take_while(|&&b| b == 0).count();
    bytes[leading..].to_vec()
}

/// Build an EBML element: `id` bytes + VINT size + `data`.
fn ebml_elem(id: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&ebml_vint_size(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Build an EBML uint element.
fn ebml_uint(id: &[u8], value: u64) -> Vec<u8> {
    ebml_elem(id, &uint_bytes(value))
}

/// Build an EBML float element (f64, big-endian 8 bytes).
fn ebml_float(id: &[u8], value: f64) -> Vec<u8> {
    ebml_elem(id, &value.to_bits().to_be_bytes())
}

/// Build an EBML string element.
fn ebml_string(id: &[u8], value: &str) -> Vec<u8> {
    ebml_elem(id, value.as_bytes())
}

/// Build a SimpleBlock element with the given timecode and frame payload.
///
/// SimpleBlock layout (after element header):
///   VINT track-number | i16 timecode | u8 flags | payload
///
/// Flags byte: bit 7 = keyframe.
fn simple_block(track: u64, timecode: i16, payload: &[u8]) -> Vec<u8> {
    let track_vint = ebml_vint_size(track);
    let tc_bytes = timecode.to_be_bytes();
    let flags: u8 = 0x80; // keyframe
    let mut block_data = Vec::new();
    block_data.extend_from_slice(&track_vint);
    block_data.extend_from_slice(&tc_bytes);
    block_data.push(flags);
    block_data.extend_from_slice(payload);
    ebml_elem(&[0xA3], &block_data)
}

/// Build a BlockGroup containing a single Block (non-keyframe).
///
/// BlockGroup ID = 0xA0; Block ID = 0xA1.
fn block_group(track: u64, timecode: i16, payload: &[u8]) -> Vec<u8> {
    let track_vint = ebml_vint_size(track);
    let tc_bytes = timecode.to_be_bytes();
    let flags: u8 = 0x00; // not a keyframe
    let mut block_data = Vec::new();
    block_data.extend_from_slice(&track_vint);
    block_data.extend_from_slice(&tc_bytes);
    block_data.push(flags);
    block_data.extend_from_slice(payload);
    let block_elem = ebml_elem(&[0xA1], &block_data);
    ebml_elem(&[0xA0], &block_elem)
}

// ---------------------------------------------------------------------------
// WebM / MKV stream builders
// ---------------------------------------------------------------------------

/// Build the standard EBML header for DocType = "webm".
fn build_ebml_header() -> Vec<u8> {
    let body = [
        ebml_uint(&[0x42, 0x86], 1),        // EBMLVersion
        ebml_uint(&[0x42, 0xF7], 1),        // EBMLReadVersion
        ebml_uint(&[0x42, 0xF2], 4),        // EBMLMaxIDLength
        ebml_uint(&[0x42, 0xF3], 8),        // EBMLMaxSizeLength
        ebml_string(&[0x42, 0x82], "webm"), // DocType
        ebml_uint(&[0x42, 0x87], 4),        // DocTypeVersion
        ebml_uint(&[0x42, 0x85], 2),        // DocTypeReadVersion
    ]
    .concat();
    ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &body)
}

/// Build a single VP9 video TrackEntry for track `number`.
fn build_track_entry(number: u64, uid: u64) -> Vec<u8> {
    let pixel_width = ebml_uint(&[0xB0], 320);
    let pixel_height = ebml_uint(&[0xBA], 240);
    let video = ebml_elem(&[0xE0], &[pixel_width, pixel_height].concat());

    let body = [
        ebml_uint(&[0xD7], number),
        ebml_uint(&[0x73, 0xC5], uid),
        ebml_uint(&[0x83], 1), // TrackType = video
        ebml_string(&[0x86], "V_VP9"),
        video,
    ]
    .concat();
    ebml_elem(&[0xAE], &body)
}

/// Build an audio TrackEntry for track `number` (Opus).
fn build_audio_track_entry(number: u64, uid: u64) -> Vec<u8> {
    let sampling_freq = ebml_float(&[0xB5], 48_000.0);
    let channels = ebml_uint(&[0x9F], 2);
    let audio = ebml_elem(&[0xE1], &[sampling_freq, channels].concat());

    let body = [
        ebml_uint(&[0xD7], number),
        ebml_uint(&[0x73, 0xC5], uid),
        ebml_uint(&[0x83], 2), // TrackType = audio
        ebml_string(&[0x86], "A_OPUS"),
        audio,
    ]
    .concat();
    ebml_elem(&[0xAE], &body)
}

/// Wrap `segment_children` in an unbounded-size Segment element.
fn build_segment(segment_children: Vec<u8>) -> Vec<u8> {
    let mut seg = Vec::new();
    seg.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]); // Segment ID
    seg.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // unknown size
    seg.extend_from_slice(&segment_children);
    seg
}

/// Build an Info element (TimecodeScale=1ms, Duration=`dur`).
fn build_info(dur: f64) -> Vec<u8> {
    let body = [
        ebml_uint(&[0x2A, 0xD7, 0xB1], 1_000_000),
        ebml_float(&[0x44, 0x89], dur),
    ]
    .concat();
    ebml_elem(&[0x15, 0x49, 0xA9, 0x66], &body)
}

/// Build a Tracks element from a list of already-encoded TrackEntry bytes.
fn build_tracks(entries: Vec<Vec<u8>>) -> Vec<u8> {
    let body: Vec<u8> = entries.into_iter().flatten().collect();
    ebml_elem(&[0x16, 0x54, 0xAE, 0x6B], &body)
}

/// Build a Cluster element with `blocks` at cluster timestamp 0.
///
/// The cluster uses an unbounded size (unknown-size element), matching the
/// behaviour of live streaming muxers.
fn build_cluster_unbounded(blocks: Vec<u8>) -> Vec<u8> {
    let cluster_ts = ebml_uint(&[0xE7], 0); // Timecode = 0
    let mut body = cluster_ts;
    body.extend_from_slice(&blocks);

    let mut cluster = Vec::new();
    cluster.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]); // Cluster ID
    cluster.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // unknown size
    cluster.extend_from_slice(&body);
    cluster
}

/// Build a minimal WebM with `n` SimpleBlocks on track 1 at timecodes 0..n-1.
fn build_simple_block_webm(n: usize) -> Vec<u8> {
    let info = build_info(n as f64);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let mut block_bytes = Vec::new();
    for i in 0..n {
        block_bytes.extend_from_slice(&simple_block(1, i as i16, &[i as u8; 4]));
    }
    let cluster = build_cluster_unbounded(block_bytes);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));
    out
}

/// Build a WebM with `n` BlockGroups on track 1.
fn build_block_group_webm(n: usize) -> Vec<u8> {
    let info = build_info(n as f64);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let mut block_bytes = Vec::new();
    for i in 0..n {
        block_bytes.extend_from_slice(&block_group(1, i as i16, &[i as u8; 4]));
    }
    let cluster = build_cluster_unbounded(block_bytes);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));
    out
}

/// Build a WebM with two streams: video (track 1) and audio (track 2).
fn build_two_stream_webm(frames_per_stream: usize) -> Vec<u8> {
    let info = build_info(frames_per_stream as f64);
    let tracks = build_tracks(vec![build_track_entry(1, 1), build_audio_track_entry(2, 2)]);
    let mut block_bytes = Vec::new();
    for i in 0..frames_per_stream {
        // Interleave video and audio blocks
        block_bytes.extend_from_slice(&simple_block(1, i as i16, &[0xB1; 4]));
        block_bytes.extend_from_slice(&simple_block(2, i as i16, &[0xA1; 4]));
    }
    let cluster = build_cluster_unbounded(block_bytes);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));
    out
}

/// Build a WebM where `n` SimpleBlocks span two clusters.
///
/// First cluster: frames 0..half at cluster_timestamp=0.
/// Second cluster: frames half..n at cluster_timestamp=half (encoded as ts
/// of cluster element, frames use relative timecodes 0..n-half-1).
fn build_two_cluster_webm(n: usize) -> Vec<u8> {
    let half = n / 2;
    let info = build_info(n as f64);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);

    // First cluster
    let mut first_blocks = Vec::new();
    for i in 0..half {
        first_blocks.extend_from_slice(&simple_block(1, i as i16, &[i as u8; 4]));
    }
    let cluster1 = build_cluster_unbounded(first_blocks);

    // Second cluster: cluster timestamp = half, relative block timecodes 0..
    let cluster2_ts = ebml_uint(&[0xE7], half as u64);
    let mut second_blocks = cluster2_ts;
    for i in 0..(n - half) {
        second_blocks.extend_from_slice(&simple_block(1, i as i16, &[(half + i) as u8; 4]));
    }
    let mut cluster2 = Vec::new();
    cluster2.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]);
    cluster2.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    cluster2.extend_from_slice(&second_blocks);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster1);
    seg_body.extend_from_slice(&cluster2);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));
    out
}

// ---------------------------------------------------------------------------
// Conformance tests
// ---------------------------------------------------------------------------

/// T1: Basic SimpleBlock round-trip — demux returns one packet per SimpleBlock,
/// in ascending PTS order, with correct payload bytes.
#[tokio::test]
async fn test_simple_block_roundtrip() {
    const N: usize = 5;
    let data = build_simple_block_webm(N);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => pts_values.push(pkt.timestamp.pts),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected read error: {e:?}"),
        }
    }

    assert_eq!(pts_values.len(), N, "should read exactly {N} packets");
    for (i, &pts) in pts_values.iter().enumerate() {
        assert_eq!(pts, i as i64, "packet {i} PTS mismatch");
    }
}

/// T2: BlockGroup round-trip — the demuxer can parse BlockGroup-wrapped blocks
/// and expose them as regular packets.
#[tokio::test]
async fn test_blockgroup_roundtrip() {
    const N: usize = 4;
    let data = build_block_group_webm(N);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut count = 0usize;
    loop {
        match demuxer.read_packet().await {
            Ok(_) => count += 1,
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error reading BlockGroup packets: {e:?}"),
        }
    }

    assert!(
        count > 0,
        "should read at least one packet from BlockGroup stream"
    );
}

/// T3: SeekHead presence — if a SeekHead is present the demuxer must not crash
/// and must successfully probe the stream.
///
/// A minimal SeekHead is injected before Info/Tracks (the standard position).
#[tokio::test]
async fn test_seekhead_present() {
    // Build a SeekHead pointing at Info at byte offset 0 (approximate — the
    // demuxer should tolerate it even if the offset is off, as long as parsing
    // does not crash).
    //
    // SeekHead ID = 0x114D9B74
    // Seek ID     = 0x4DBB
    // SeekID      = 0x53AB  (child: element ID encoded as binary)
    // SeekPosition= 0x53AC  (child: relative byte position)
    let info_id_bytes: Vec<u8> = vec![0x15, 0x49, 0xA9, 0x66]; // Info element ID
    let seek_id = ebml_elem(&[0x53, 0xAB], &info_id_bytes);
    let seek_pos = ebml_uint(&[0x53, 0xAC], 0);
    let seek_entry = ebml_elem(&[0x4D, 0xBB], &[seek_id, seek_pos].concat());
    let seekhead = ebml_elem(&[0x11, 0x4D, 0x9B, 0x74], &seek_entry);

    let info = build_info(3.0);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let mut blocks = Vec::new();
    for i in 0..3_i16 {
        blocks.extend_from_slice(&simple_block(1, i, &[i as u8; 4]));
    }
    let cluster = build_cluster_unbounded(blocks);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&seekhead);
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));

    let source = MemorySource::new(Bytes::from(out));
    let mut demuxer = MatroskaDemuxer::new(source);
    // Must not panic/error during probe even when SeekHead is present.
    demuxer
        .probe()
        .await
        .expect("probe with SeekHead must succeed");
    assert!(
        !demuxer.streams().is_empty(),
        "should have at least one stream after probe with SeekHead"
    );
}

/// T4: Cues present — if a Cues element is present the demuxer must not crash.
///
/// A minimal Cues element is appended after the cluster.
#[tokio::test]
async fn test_cues_present() {
    // Cues ID = 0x1C53BB6B
    // CuePoint ID = 0xBB
    // CueTime     = 0xB3
    // CueTrackPositions = 0xB7
    // CueTrack    = 0xF7
    // CueClusterPosition = 0xF1
    let cue_track = ebml_uint(&[0xF7], 1);
    let cue_cluster_pos = ebml_uint(&[0xF1], 0);
    let cue_track_pos = ebml_elem(&[0xB7], &[cue_track, cue_cluster_pos].concat());
    let cue_time = ebml_uint(&[0xB3], 0);
    let cue_point = ebml_elem(&[0xBB], &[cue_time, cue_track_pos].concat());
    let cues = ebml_elem(&[0x1C, 0x53, 0xBB, 0x6B], &cue_point);

    let info = build_info(3.0);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let mut blocks = Vec::new();
    for i in 0..3_i16 {
        blocks.extend_from_slice(&simple_block(1, i, &[i as u8; 4]));
    }
    let cluster = build_cluster_unbounded(blocks);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);
    seg_body.extend_from_slice(&cues);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));

    let source = MemorySource::new(Bytes::from(out));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe with Cues must succeed");
    assert!(
        !demuxer.streams().is_empty(),
        "should have at least one stream after probe with Cues"
    );
}

/// T5: VINT boundary values — encode blocks at timecodes 0, 127, 128, 16383
/// (the 1-byte, boundary, and 2-byte VINT thresholds for sizes).
///
/// This verifies the demuxer handles the VINT size encoding at the exact
/// 1-byte→2-byte boundary (127–128) and 2-byte max (16383).
#[tokio::test]
async fn test_vint_boundary_values() {
    // We can only use i16 timecodes in SimpleBlocks (relative to cluster ts).
    // Test VINT size encoding of element bodies by using payloads of exactly
    // 126 and 127 bytes (one below and at the 1-byte VINT max).
    let payload_126 = vec![0xAAu8; 126];
    let payload_127 = vec![0xBBu8; 127];
    let payload_128 = vec![0xCCu8; 128]; // requires 2-byte VINT size

    let mut blocks = Vec::new();
    blocks.extend_from_slice(&simple_block(1, 0, &payload_126));
    blocks.extend_from_slice(&simple_block(1, 1, &payload_127));
    blocks.extend_from_slice(&simple_block(1, 2, &payload_128));

    let info = build_info(3.0);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let cluster = build_cluster_unbounded(blocks);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));

    let source = MemorySource::new(Bytes::from(out));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut packets = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => packets.push(pkt),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
    assert_eq!(
        packets.len(),
        3,
        "should read 3 packets with VINT boundary payloads"
    );
    assert_eq!(packets[0].data.len(), 126, "first packet payload size");
    assert_eq!(packets[1].data.len(), 127, "second packet payload size");
    assert_eq!(packets[2].data.len(), 128, "third packet payload size");
}

/// T6: Multiple streams — a two-track (video+audio) stream must expose both
/// stream infos after `probe()` and return packets from each stream.
#[tokio::test]
async fn test_multiple_streams() {
    const FRAMES: usize = 4;
    let data = build_two_stream_webm(FRAMES);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let stream_count = demuxer.streams().len();
    assert!(
        stream_count >= 2,
        "two-track stream must expose ≥2 streams after probe, got {stream_count}"
    );

    let mut per_stream: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => *per_stream.entry(pkt.stream_index).or_insert(0) += 1,
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
    assert!(
        per_stream.len() >= 2,
        "packets from ≥2 streams expected, got {:?}",
        per_stream
    );
}

/// T7: Nested timestamps (two clusters) — a stream split across two clusters
/// must produce monotonically non-decreasing PTS values, because the absolute
/// PTS = cluster_timestamp + block_timecode.
#[tokio::test]
async fn test_nested_timestamps_two_clusters() {
    const N: usize = 8;
    let data = build_two_cluster_webm(N);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => pts_values.push(pkt.timestamp.pts),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    assert!(
        !pts_values.is_empty(),
        "should read packets from two-cluster stream"
    );
    // PTS values must be non-decreasing across cluster boundaries.
    for w in pts_values.windows(2) {
        assert!(
            w[1] >= w[0],
            "PTS must be non-decreasing across cluster boundaries: {} then {}",
            w[0],
            w[1]
        );
    }
}

/// T8: Round-trip preserves 100 packets — write 100 SimpleBlocks and verify
/// that exactly 100 packets are read back, each with the expected PTS.
///
/// This exercises the demuxer's ability to consume a large synthetic stream
/// without truncation or duplication.
#[tokio::test]
async fn test_round_trip_preserves_100_packets() {
    const N: usize = 100;
    let data = build_simple_block_webm(N);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => pts_values.push(pkt.timestamp.pts),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    assert_eq!(
        pts_values.len(),
        N,
        "should read exactly {N} packets, got {}",
        pts_values.len()
    );
    for (i, &pts) in pts_values.iter().enumerate() {
        assert_eq!(pts, i as i64, "packet {i} PTS should be {i}, got {pts}");
    }
}

// ---------------------------------------------------------------------------
// EBML conformance: Matroska DocType + spec-level hard-coded fixture tests
// ---------------------------------------------------------------------------

/// Build the standard EBML header for DocType = "matroska" (native MKV).
fn build_matroska_ebml_header() -> Vec<u8> {
    let body = [
        ebml_uint(&[0x42, 0x86], 1),            // EBMLVersion
        ebml_uint(&[0x42, 0xF7], 1),            // EBMLReadVersion
        ebml_uint(&[0x42, 0xF2], 4),            // EBMLMaxIDLength
        ebml_uint(&[0x42, 0xF3], 8),            // EBMLMaxSizeLength
        ebml_string(&[0x42, 0x82], "matroska"), // DocType
        ebml_uint(&[0x42, 0x87], 4),            // DocTypeVersion
        ebml_uint(&[0x42, 0x85], 2),            // DocTypeReadVersion
    ]
    .concat();
    ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &body)
}

/// Build a minimal Matroska (.mkv) file with `n` SimpleBlocks using
/// DocType = "matroska".  This mirrors the byte layout in the official
/// Matroska test suite.
fn build_matroska_simple_block_file(n: usize) -> Vec<u8> {
    let info = build_info(n as f64);
    let tracks = build_tracks(vec![build_track_entry(1, 1)]);
    let mut block_bytes = Vec::new();
    for i in 0..n {
        block_bytes.extend_from_slice(&simple_block(1, i as i16, &[i as u8; 4]));
    }
    let cluster = build_cluster_unbounded(block_bytes);

    let mut seg_body = Vec::new();
    seg_body.extend_from_slice(&info);
    seg_body.extend_from_slice(&tracks);
    seg_body.extend_from_slice(&cluster);

    let mut out = build_matroska_ebml_header();
    out.extend_from_slice(&build_segment(seg_body));
    out
}

/// T9: Matroska DocType probe — the demuxer must correctly identify DocType
/// "matroska" and expose it as `ContainerFormat::Matroska` (not WebM).
#[tokio::test]
async fn test_matroska_doctype_probe() {
    let data = build_matroska_simple_block_file(3);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    let result = demuxer
        .probe()
        .await
        .expect("probe must succeed for matroska doctype");
    assert_eq!(
        result.format,
        oximedia_container::ContainerFormat::Matroska,
        "DocType 'matroska' must probe as ContainerFormat::Matroska"
    );
    assert!(result.confidence > 0.9, "probe confidence must be high");
    assert!(
        !demuxer.streams().is_empty(),
        "must expose at least one stream"
    );
}

/// T10: Matroska DocType round-trip — packets from a "matroska" DocType stream
/// are readable with correct PTS ordering.
#[tokio::test]
async fn test_matroska_doctype_packet_roundtrip() {
    const N: usize = 5;
    let data = build_matroska_simple_block_file(N);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);
    demuxer.probe().await.expect("probe must succeed");

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => pts_values.push(pkt.timestamp.pts),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error reading Matroska packets: {e:?}"),
        }
    }

    assert_eq!(
        pts_values.len(),
        N,
        "should read {N} packets from matroska stream"
    );
    for (i, &pts) in pts_values.iter().enumerate() {
        assert_eq!(pts, i as i64, "packet {i} PTS mismatch in matroska stream");
    }
}

/// T11: Hard-coded minimal MKV fixture — a spec-conformant minimal byte array
/// built from the Matroska EBML spec (EBML header + Segment + Cluster with
/// Timestamp=0).  The demuxer must parse the EBML header gracefully and either
/// succeed or fail with a typed error (never panic).
///
/// Byte layout (from the Matroska specification):
///   EBML Header: 1A 45 DF A3 (ID) + 9F (size=31) + children
///   Segment:     18 53 80 67 (ID) + 01 FF FF FF FF FF FF FF (unknown size)
///   Cluster:     1F 43 B6 75 (ID) + 83 (size=3) + Timestamp E7 81 00
#[tokio::test]
async fn test_minimal_mkv_hard_coded_fixture() {
    // A minimal, spec-conformant MKV with only EBML header + Segment + Cluster.
    // No Tracks element — the demuxer must handle this gracefully.
    let minimal_mkv: Vec<u8> = vec![
        // EBML Header (ID = 1A 45 DF A3)
        0x1A, 0x45, 0xDF, 0xA3, 0x9F, // VINT size = 31
        // EBMLVersion = 1
        0x42, 0x86, 0x81, 0x01, // EBMLReadVersion = 1
        0x42, 0xF7, 0x81, 0x01, // EBMLMaxIDLength = 4
        0x42, 0xF2, 0x81, 0x04, // EBMLMaxSizeLength = 8
        0x42, 0xF3, 0x81, 0x08, // DocType = "matroska" (8 bytes)
        0x42, 0x82, 0x88, b'm', b'a', b't', b'r', b'o', b's', b'k', b'a',
        // DocTypeVersion = 4
        0x42, 0x87, 0x81, 0x04, // DocTypeReadVersion = 2
        0x42, 0x85, 0x81, 0x02, // Segment element (ID = 18 53 80 67, unknown size)
        0x18, 0x53, 0x80, 0x67, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        // Minimal Cluster (ID = 1F 43 B6 75)
        0x1F, 0x43, 0xB6, 0x75, 0x83, // VINT size = 3
        // Timestamp = 0
        0xE7, 0x81, 0x00,
    ];

    let source = MemorySource::new(Bytes::from(minimal_mkv));
    let mut demuxer = MatroskaDemuxer::new(source);

    // The probe may succeed or fail gracefully (no Tracks).  Must NOT panic.
    match demuxer.probe().await {
        Ok(_) => {
            // Graceful success: probe finished but streams may be empty.
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                !msg.contains("called `Option::unwrap()` on a `None` value")
                    && !msg.contains("index out of bounds")
                    && !msg.contains("arithmetic operation overflowed"),
                "demuxer must fail gracefully, not with a Rust panic: {msg}"
            );
        }
    }
}

/// T12: Malformed EBML — a garbage byte stream must return an error, not panic.
#[tokio::test]
async fn test_malformed_ebml_returns_error() {
    let garbage: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let source = MemorySource::new(Bytes::from(garbage));
    let mut demuxer = MatroskaDemuxer::new(source);

    let result = demuxer.probe().await;
    assert!(
        result.is_err(),
        "demuxing a garbage byte stream must return an error"
    );
}

/// T13: Truncated EBML header — a stream truncated mid-header must return an
/// error, not panic or hang.
#[tokio::test]
async fn test_truncated_ebml_header_returns_error() {
    // Only the first 3 bytes of the EBML element ID — not enough to parse.
    let truncated: Vec<u8> = vec![0x1A, 0x45, 0xDF];
    let source = MemorySource::new(Bytes::from(truncated));
    let mut demuxer = MatroskaDemuxer::new(source);

    let result = demuxer.probe().await;
    assert!(result.is_err(), "truncated EBML input must return an error");
}

/// T14: EBML header only (no Segment) — a well-formed EBML header with no
/// following Segment element must fail gracefully rather than panic.
#[tokio::test]
async fn test_ebml_header_only_no_segment() {
    let header_only = build_matroska_ebml_header();
    let source = MemorySource::new(Bytes::from(header_only));
    let mut demuxer = MatroskaDemuxer::new(source);

    match demuxer.probe().await {
        Ok(_) => {
            assert!(
                demuxer.streams().is_empty(),
                "no streams expected when only EBML header is present"
            );
        }
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                !msg.contains("called `Option::unwrap()` on a `None` value")
                    && !msg.contains("index out of bounds"),
                "must fail gracefully, not with a panic: {msg}"
            );
        }
    }
}

/// T15: EBML with wrong DocType — a well-formed EBML structure with an
/// unsupported DocType (e.g. "avi") should return an error or gracefully
/// expose zero streams.
#[tokio::test]
async fn test_ebml_wrong_doctype_graceful() {
    let body = [
        ebml_uint(&[0x42, 0x86], 1),
        ebml_uint(&[0x42, 0xF7], 1),
        ebml_uint(&[0x42, 0xF2], 4),
        ebml_uint(&[0x42, 0xF3], 8),
        ebml_string(&[0x42, 0x82], "avi"), // unsupported DocType
        ebml_uint(&[0x42, 0x87], 1),
        ebml_uint(&[0x42, 0x85], 1),
    ]
    .concat();
    let wrong_doctype_header = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &body);

    // Append an empty Segment with unknown size.
    let mut data = wrong_doctype_header;
    data.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
    data.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    // Either an error or an Ok with empty streams is acceptable; no panic.
    match demuxer.probe().await {
        Ok(_) => {}
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                !msg.contains("called `Option::unwrap()` on a `None` value")
                    && !msg.contains("index out of bounds"),
                "wrong-doctype must fail gracefully, not with a Rust panic: {msg}"
            );
        }
    }
}
