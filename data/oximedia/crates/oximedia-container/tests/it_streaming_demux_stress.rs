// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: streaming demuxer stress.
//!
//! Drives `streaming::demux::StreamingDemuxer` with a mock inner demuxer that
//! emits a large, configurable number of synthetic packets followed by
//! `OxiError::Eof`. The test asserts:
//!
//! - no panic across very-long packet sequences,
//! - `packets_read()` is monotonic and matches the expected count,
//! - `bytes_buffered()` stays at zero (the wrapper does not retain payload),
//! - `state()` transitions to `Eof` cleanly when the inner demuxer drains.
//!
//! TODO.md item: line 63 — "Add stress test for streaming/demux.rs with
//! simulated live stream input (continuous data)".

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_container::streaming::demux::{
    StreamingDemuxer, StreamingDemuxerConfig, StreamingState,
};
use oximedia_container::{ContainerFormat, Demuxer, Packet, PacketFlags, ProbeResult, StreamInfo};
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};

/// Mock demuxer that emits `target` synthetic packets, then `Eof`.
struct MockDemuxer {
    streams: Vec<StreamInfo>,
    target: usize,
    emitted: usize,
}

impl MockDemuxer {
    fn new(target: usize) -> Self {
        let mut info = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 90000));
        info.codec_params = oximedia_container::CodecParams::video(320, 240);
        Self {
            streams: vec![info],
            target,
            emitted: 0,
        }
    }
}

#[async_trait]
impl Demuxer for MockDemuxer {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        Ok(ProbeResult::new(ContainerFormat::Mp4, 0.99))
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if self.emitted >= self.target {
            return Err(OxiError::Eof);
        }
        let pts = self.emitted as i64 * 3000;
        let mut ts = Timestamp::new(pts, Rational::new(1, 90000));
        ts.duration = Some(3000);
        // Tiny per-packet payload — we want millions of packets without OOM.
        let payload = Bytes::from(vec![(self.emitted & 0xFF) as u8; 16]);
        let pkt = Packet::new(0, payload, ts, PacketFlags::KEYFRAME);
        self.emitted += 1;
        Ok(pkt)
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
}

async fn drain<D: Demuxer>(demuxer: &mut D) -> usize {
    let mut count = 0usize;
    loop {
        match demuxer.read_packet().await {
            Ok(_) => count += 1,
            Err(OxiError::Eof) => break,
            Err(e) => panic!("unexpected error from demuxer: {e:?}"),
        }
    }
    count
}

#[tokio::test]
async fn test_streaming_demux_100k_packets_completes_without_panic() {
    let inner = MockDemuxer::new(100_000);
    let mut streaming = StreamingDemuxer::new(inner);
    streaming.probe().await.expect("probe");

    let count = drain(&mut streaming).await;
    assert_eq!(count, 100_000, "all packets must be delivered");
    assert_eq!(streaming.packets_read(), 100_000);
    assert_eq!(
        streaming.state(),
        StreamingState::Eof,
        "must reach EOF cleanly"
    );
    // The wrapper does not retain payload — bytes_buffered must stay at zero.
    assert_eq!(streaming.bytes_buffered(), 0);
}

#[tokio::test]
async fn test_streaming_demux_low_latency_million_packets_bounded_memory() {
    // Low-latency mode skips buffering checks entirely; under load the only
    // accumulated state is the `packets_read` counter. 1M packets at ~16 B
    // payload each totals about 16 MB through the channel — easily within
    // CI memory budget.
    let target = 1_000_000_usize;
    let inner = MockDemuxer::new(target);
    let cfg = StreamingDemuxerConfig::new().with_low_latency(true);
    let mut streaming = StreamingDemuxer::with_config(inner, cfg);
    streaming.probe().await.expect("probe");

    let count = drain(&mut streaming).await;
    assert_eq!(count, target, "1M packets must be delivered without loss");
    assert_eq!(streaming.packets_read(), target as u64);
    assert_eq!(streaming.bytes_buffered(), 0, "no implicit retention");
    assert_eq!(streaming.state(), StreamingState::Eof);
}

#[tokio::test]
async fn test_streaming_demux_state_eof_terminal() {
    // After EOF, repeated calls should keep returning EOF and not panic.
    let inner = MockDemuxer::new(5);
    let mut streaming = StreamingDemuxer::new(inner);
    streaming.probe().await.expect("probe");

    let _ = drain(&mut streaming).await;
    assert_eq!(streaming.state(), StreamingState::Eof);

    // Re-poll past EOF — must continue to surface Eof.
    for _ in 0..32 {
        match streaming.read_packet().await {
            Err(OxiError::Eof) => {}
            Ok(_) => panic!("packets must not appear past EOF"),
            Err(e) => panic!("unexpected error past EOF: {e:?}"),
        }
    }
    assert_eq!(streaming.state(), StreamingState::Eof);
    assert_eq!(streaming.packets_read(), 5);
}
