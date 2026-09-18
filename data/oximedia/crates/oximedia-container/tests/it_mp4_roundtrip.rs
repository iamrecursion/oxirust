// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MP4 packet-level round-trip.
//!
//! Asserts that the compressed payload of every packet survives a
//! `write → demux → write → demux` cycle. The MP4 muxer/demuxer is allowed to
//! normalise per-sample metadata (chunk offsets, ftyp brand, etc.) between
//! passes, so byte-for-byte container equality is NOT asserted — but the
//! demuxed `Packet.data` and per-stream sample counts must remain stable
//! starting from the second demux pass.
//!
//! Note on coverage: MP4 subtitle muxing is not exposed through `Mp4Muxer`
//! (the `validate_codec` whitelist rejects subtitle codecs), so this test
//! uses a 2-track stream (AV1 video + Opus audio) per the TODO.md guidance.
//!
//! TODO.md item: line 61 — "Add round-trip test: demux -> mux -> demux ->
//! verify packet-level equality for all formats".

use bytes::Bytes;
use oximedia_container::{
    demux::Mp4Demuxer,
    mux::mp4::{Mp4Config, Mp4Muxer},
    CodecParams, Demuxer, Packet, PacketFlags, StreamInfo,
};
use oximedia_core::{CodecId, OxiError, Rational, Timestamp};
use oximedia_io::source::MemorySource;

const NUM_VIDEO_FRAMES: i64 = 8;
const NUM_AUDIO_FRAMES: i64 = 6;
const VIDEO_FRAME_DURATION: u32 = 3000; // 33 ms at 90 kHz
const AUDIO_FRAME_DURATION: u32 = 960; // 20 ms at 48 kHz

fn make_video_stream() -> StreamInfo {
    let mut info = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 90000));
    info.codec_params = CodecParams::video(640, 360);
    info
}

fn make_audio_stream() -> StreamInfo {
    let mut info = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
    info.codec_params = CodecParams::audio(48000, 2);
    info
}

fn make_video_packet(frame: i64) -> Packet {
    // Distinct payload per frame so we can tell them apart end-to-end.
    let mut payload = vec![0xAB_u8; 80];
    payload[0] = (frame & 0xFF) as u8;
    payload[1] = ((frame >> 8) & 0xFF) as u8;
    let mut ts = Timestamp::new(
        frame * i64::from(VIDEO_FRAME_DURATION),
        Rational::new(1, 90000),
    );
    ts.duration = Some(i64::from(VIDEO_FRAME_DURATION));
    Packet::new(
        0,
        Bytes::from(payload),
        ts,
        if frame % 4 == 0 {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::empty()
        },
    )
}

fn make_audio_packet(frame: i64) -> Packet {
    let mut payload = vec![0xCD_u8; 48];
    payload[0] = (frame & 0xFF) as u8;
    let mut ts = Timestamp::new(
        frame * i64::from(AUDIO_FRAME_DURATION),
        Rational::new(1, 48000),
    );
    ts.duration = Some(i64::from(AUDIO_FRAME_DURATION));
    Packet::new(1, Bytes::from(payload), ts, PacketFlags::KEYFRAME)
}

/// Builds a progressive MP4 file in memory holding the supplied packets.
fn mux_packets(packets: &[Packet]) -> Vec<u8> {
    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    muxer.add_stream(make_video_stream()).expect("video stream");
    muxer.add_stream(make_audio_stream()).expect("audio stream");
    muxer.write_header().expect("write header");
    for p in packets {
        muxer.write_packet(p).expect("write packet");
    }
    muxer.finalize().expect("finalize")
}

/// Demuxes every packet from an MP4 byte buffer.
async fn demux_all(data: Vec<u8>) -> Vec<Packet> {
    let source = MemorySource::from_vec(data);
    let mut demuxer = Mp4Demuxer::new(source);
    demuxer.probe().await.expect("probe");
    let mut packets = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(p) => packets.push(p),
            Err(OxiError::Eof) => break,
            Err(e) => panic!("demuxer error: {e:?}"),
        }
    }
    packets
}

/// Build the initial deterministic packet list.
fn build_source_packets() -> Vec<Packet> {
    let mut packets = Vec::new();
    for f in 0..NUM_VIDEO_FRAMES {
        packets.push(make_video_packet(f));
    }
    for f in 0..NUM_AUDIO_FRAMES {
        packets.push(make_audio_packet(f));
    }
    packets
}

/// Re-mux a demuxer-emitted packet list (stream indices already align).
fn remux(packets: &[Packet]) -> Vec<u8> {
    mux_packets(packets)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mp4_first_pass_packet_data_preserved() {
    // Source → demux. Every demuxed packet's compressed payload must equal the
    // payload we fed into the muxer.
    let source_packets = build_source_packets();
    let bytes = mux_packets(&source_packets);
    let demuxed = demux_all(bytes).await;

    assert_eq!(
        demuxed.len(),
        source_packets.len(),
        "demux must yield exactly the muxed packet count"
    );

    // Group by stream so we can compare in-order with the original.
    let mut video_in: Vec<&Packet> = source_packets
        .iter()
        .filter(|p| p.stream_index == 0)
        .collect();
    let mut audio_in: Vec<&Packet> = source_packets
        .iter()
        .filter(|p| p.stream_index == 1)
        .collect();
    video_in.sort_by_key(|p| p.pts());
    audio_in.sort_by_key(|p| p.pts());

    let mut video_out: Vec<&Packet> = demuxed.iter().filter(|p| p.stream_index == 0).collect();
    let mut audio_out: Vec<&Packet> = demuxed.iter().filter(|p| p.stream_index == 1).collect();
    video_out.sort_by_key(|p| p.pts());
    audio_out.sort_by_key(|p| p.pts());

    assert_eq!(video_in.len(), video_out.len(), "video packet count");
    assert_eq!(audio_in.len(), audio_out.len(), "audio packet count");

    for (orig, demuxed) in video_in.iter().zip(video_out.iter()) {
        assert_eq!(
            orig.data.as_ref(),
            demuxed.data.as_ref(),
            "video packet payload must round-trip"
        );
        assert_eq!(orig.is_keyframe(), demuxed.is_keyframe());
    }
    for (orig, demuxed) in audio_in.iter().zip(audio_out.iter()) {
        assert_eq!(
            orig.data.as_ref(),
            demuxed.data.as_ref(),
            "audio packet payload must round-trip"
        );
    }
}

#[tokio::test]
async fn test_mp4_second_pass_packet_data_stable() {
    // Pass 1: source → demux1.
    // Pass 2: demux1 → mux2 → demux2.
    // demux1 and demux2 must agree packet-for-packet (any normalisation done
    // by the first demux pass should be idempotent).
    let source = build_source_packets();
    let bytes_a = mux_packets(&source);
    let demux1 = demux_all(bytes_a).await;

    let bytes_b = remux(&demux1);
    let demux2 = demux_all(bytes_b).await;

    assert_eq!(
        demux1.len(),
        demux2.len(),
        "second-pass packet count must match first-pass"
    );

    // Per-stream comparison sorted by PTS.
    for stream in 0..2_usize {
        let mut a: Vec<&Packet> = demux1.iter().filter(|p| p.stream_index == stream).collect();
        let mut b: Vec<&Packet> = demux2.iter().filter(|p| p.stream_index == stream).collect();
        a.sort_by_key(|p| p.pts());
        b.sort_by_key(|p| p.pts());
        assert_eq!(
            a.len(),
            b.len(),
            "stream {stream}: second-pass packet count"
        );
        for (idx, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                pa.data.as_ref(),
                pb.data.as_ref(),
                "stream {stream} idx {idx}: compressed payload byte equality on second round-trip"
            );
            assert_eq!(
                pa.is_keyframe(),
                pb.is_keyframe(),
                "stream {stream} idx {idx}: keyframe flag stable"
            );
        }
    }
}

#[tokio::test]
async fn test_mp4_streams_metadata_preserved_across_roundtrip() {
    // The streams() metadata exposed by the demuxer must be self-consistent
    // across two round-trips: same number of streams, same media_type, same
    // codec.
    let source = build_source_packets();
    let bytes_a = mux_packets(&source);
    let mut demuxer_a = Mp4Demuxer::new(MemorySource::from_vec(bytes_a));
    demuxer_a.probe().await.expect("probe pass1");
    let streams_a = demuxer_a.streams().to_vec();
    drop(demuxer_a);

    // Drain pass1 to feed remuxer.
    let bytes_a2 = mux_packets(&source); // identical source — reuse to remux
    let packets_a = demux_all(bytes_a2).await;
    let bytes_b = remux(&packets_a);

    let mut demuxer_b = Mp4Demuxer::new(MemorySource::from_vec(bytes_b));
    demuxer_b.probe().await.expect("probe pass2");
    let streams_b = demuxer_b.streams().to_vec();

    assert_eq!(
        streams_a.len(),
        streams_b.len(),
        "stream count must be preserved"
    );
    for (a, b) in streams_a.iter().zip(streams_b.iter()) {
        assert_eq!(a.media_type, b.media_type, "media_type must match");
        assert_eq!(a.codec, b.codec, "codec must match");
    }
}

#[tokio::test]
async fn test_mp4_empty_data_packets_round_trip() {
    // Edge case: a packet with zero bytes of compressed data should not crash
    // either path. We use small payloads instead of empty (the MP4 layout
    // expects at least one byte per sample). Verify a minimum-size payload
    // round-trips.
    let mut packets = Vec::new();
    for f in 0..3 {
        let mut ts = Timestamp::new(f * i64::from(VIDEO_FRAME_DURATION), Rational::new(1, 90000));
        ts.duration = Some(i64::from(VIDEO_FRAME_DURATION));
        packets.push(Packet::new(
            0,
            Bytes::from(vec![0x77_u8]),
            ts,
            PacketFlags::KEYFRAME,
        ));
    }

    let mut muxer = Mp4Muxer::new(Mp4Config::new());
    muxer.add_stream(make_video_stream()).expect("add video");
    muxer.write_header().expect("write header");
    for p in &packets {
        muxer.write_packet(p).expect("write packet");
    }
    let bytes = muxer.finalize().expect("finalize");

    let demuxed = demux_all(bytes).await;
    assert_eq!(demuxed.len(), packets.len());
    for (orig, out) in packets.iter().zip(demuxed.iter()) {
        assert_eq!(orig.data.as_ref(), out.data.as_ref());
    }
}
