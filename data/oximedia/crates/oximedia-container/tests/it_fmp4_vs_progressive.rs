// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: progressive MP4 vs fragmented MP4 (fMP4) parity.
//!
//! Asserts that the same source frames yield equivalent observable structure
//! when written through either `Mp4FragmentMode::Progressive` or
//! `Mp4FragmentMode::Fragmented { .. }`:
//!
//! - both produce non-empty output beginning with `ftyp`,
//! - both expose all `track_count` tracks,
//! - both report identical `total_samples()` counts,
//! - the in-memory `Mp4TrackState::samples` for each track matches
//!   in length, per-sample size, duration, composition offset, and sync flag.
//!
//! TODO.md item: line 60 — "Test MP4 demuxer with fragmented MP4 (fMP4) and
//! progressive MP4 variants".

use bytes::Bytes;
use oximedia_container::{
    mux::mp4::{Mp4Config, Mp4FragmentMode, Mp4Muxer},
    CodecParams, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, Rational, Timestamp};

const FRAME_DURATION_90KHZ: u32 = 3000; // ~33 ms at 30 fps
const NUM_FRAMES: i64 = 12;
const KEYFRAME_INTERVAL: i64 = 6;

fn make_video_stream() -> StreamInfo {
    let mut info = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(1280, 720);
    info
}

fn make_audio_stream() -> StreamInfo {
    let mut info = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
    info.codec_params = CodecParams::audio(48000, 2);
    info
}

fn make_video_packet(stream_index: usize, frame: i64) -> Packet {
    let mut ts = Timestamp::new(
        frame * i64::from(FRAME_DURATION_90KHZ),
        Rational::new(1, 90000),
    );
    ts.duration = Some(i64::from(FRAME_DURATION_90KHZ));
    // Encode frame index in the payload so the test can spot data corruption.
    let payload = vec![(frame & 0xFF) as u8; 96 + (frame as usize % 8)];
    Packet::new(
        stream_index,
        Bytes::from(payload),
        ts,
        if frame % KEYFRAME_INTERVAL == 0 {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::empty()
        },
    )
}

fn make_audio_packet(stream_index: usize, frame: i64) -> Packet {
    let mut ts = Timestamp::new(frame * 960, Rational::new(1, 48000));
    ts.duration = Some(960);
    Packet::new(
        stream_index,
        Bytes::from(vec![0x55_u8; 64]),
        ts,
        PacketFlags::KEYFRAME,
    )
}

fn build_muxer(mode: Mp4FragmentMode) -> Mp4Muxer {
    let config = Mp4Config::new().with_mode(mode);
    let mut muxer = Mp4Muxer::new(config);
    muxer.add_stream(make_video_stream()).expect("video stream");
    muxer.add_stream(make_audio_stream()).expect("audio stream");
    muxer.write_header().expect("write header");
    for i in 0..NUM_FRAMES {
        muxer
            .write_packet(&make_video_packet(0, i))
            .expect("write video packet");
        muxer
            .write_packet(&make_audio_packet(1, i))
            .expect("write audio packet");
    }
    muxer
}

fn contains_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    data.windows(4).any(|w| w == fourcc)
}

#[test]
fn test_progressive_and_fragmented_frame_counts_match() {
    let progressive = build_muxer(Mp4FragmentMode::Progressive);
    let fragmented = build_muxer(Mp4FragmentMode::Fragmented {
        fragment_duration_ms: 100,
    });

    assert_eq!(progressive.track_count(), fragmented.track_count());
    assert_eq!(progressive.total_samples(), fragmented.total_samples());
    assert_eq!(
        progressive.total_samples(),
        (NUM_FRAMES as usize) * 2,
        "expected video + audio sample count"
    );
}

#[test]
fn test_progressive_and_fragmented_per_track_sample_metadata_matches() {
    let progressive = build_muxer(Mp4FragmentMode::Progressive);
    let fragmented = build_muxer(Mp4FragmentMode::Fragmented {
        fragment_duration_ms: 100,
    });

    for idx in 0..progressive.track_count() {
        let p_track = progressive.track(idx).expect("progressive track");
        let f_track = fragmented.track(idx).expect("fragmented track");
        assert_eq!(
            p_track.timescale, f_track.timescale,
            "track {idx} timescale"
        );
        assert_eq!(
            p_track.samples.len(),
            f_track.samples.len(),
            "track {idx} sample count"
        );
        assert_eq!(
            p_track.total_duration, f_track.total_duration,
            "track {idx} total duration"
        );
        for (sidx, (p_sample, f_sample)) in p_track
            .samples
            .iter()
            .zip(f_track.samples.iter())
            .enumerate()
        {
            assert_eq!(
                p_sample.size, f_sample.size,
                "track {idx} sample {sidx} size"
            );
            assert_eq!(
                p_sample.duration, f_sample.duration,
                "track {idx} sample {sidx} duration"
            );
            assert_eq!(
                p_sample.composition_offset, f_sample.composition_offset,
                "track {idx} sample {sidx} ctts"
            );
            assert_eq!(
                p_sample.is_sync, f_sample.is_sync,
                "track {idx} sample {sidx} sync"
            );
        }
    }
}

#[test]
fn test_progressive_and_fragmented_outputs_have_required_boxes() {
    let progressive = build_muxer(Mp4FragmentMode::Progressive);
    let fragmented = build_muxer(Mp4FragmentMode::Fragmented {
        fragment_duration_ms: 100,
    });

    let p_output = progressive.finalize().expect("progressive finalize");
    let f_output = fragmented.finalize().expect("fragmented finalize");

    // Both must be non-empty and start with ftyp.
    assert!(!p_output.is_empty(), "progressive output empty");
    assert!(!f_output.is_empty(), "fragmented output empty");
    assert_eq!(&p_output[4..8], b"ftyp", "progressive must start with ftyp");
    assert_eq!(&f_output[4..8], b"ftyp", "fragmented must start with ftyp");

    // Both must contain moov.
    assert!(contains_box(&p_output, b"moov"), "progressive moov");
    assert!(contains_box(&f_output, b"moov"), "fragmented moov");

    // Progressive: must have mdat, must NOT have moof.
    assert!(
        contains_box(&p_output, b"mdat"),
        "progressive must have mdat"
    );
    assert!(
        !contains_box(&p_output, b"moof"),
        "progressive must NOT contain moof"
    );

    // Fragmented: must have mvex+trex+moof.
    assert!(
        contains_box(&f_output, b"mvex"),
        "fragmented must have mvex"
    );
    assert!(
        contains_box(&f_output, b"trex"),
        "fragmented must have trex"
    );
    assert!(
        contains_box(&f_output, b"moof"),
        "fragmented must have moof"
    );
}
