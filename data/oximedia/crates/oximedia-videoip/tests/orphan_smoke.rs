//! Smoke tests for the 22 newly-registered orphan modules in `oximedia-videoip`.
//!
//! Each test instantiates or calls at least one public item from the module and
//! asserts a basic invariant to confirm compilation and basic correctness.

// ─────────────────────────────────────────────────────────────────────────────
// 1. adaptive_jitter_buffer
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_adaptive_jitter_buffer_insert_and_stats() {
    use oximedia_videoip::adaptive_jitter_buffer::{AdaptiveJitterBuffer, JitterBufferConfig};

    let config = JitterBufferConfig::default();
    let initial_depth = config.initial_depth_ms;
    let mut buf = AdaptiveJitterBuffer::new(config);

    buf.insert(1, 1_000, vec![0xAA; 64])
        .expect("first insert should succeed");
    buf.insert(2, 2_000, vec![0xBB; 64])
        .expect("second insert should succeed");

    let stats = buf.stats();
    assert!(
        stats.current_depth_ms >= initial_depth,
        "depth should be at least initial depth after inserts"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. bandwidth_shaping
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_bandwidth_shaping_register_stream() {
    use oximedia_videoip::bandwidth_shaping::{BandwidthShaper, StreamShapeConfig};

    // BandwidthShaper::new takes aggregate_limit_bps: u64
    let mut shaper = BandwidthShaper::new(100_000_000); // 100 Mbps ceiling

    let stream_cfg = StreamShapeConfig::default();
    shaper
        .register_stream("cam1", stream_cfg)
        .expect("stream should be registered");

    assert_eq!(shaper.stream_count(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. bbr_congestion
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_bbr_congestion_initial_state() {
    use oximedia_videoip::bbr::BbrConfig;
    use oximedia_videoip::bbr_congestion::{BbrCongestionController, CongestionState};

    let cfg = BbrConfig::default();
    let controller = BbrCongestionController::new(cfg);

    // Initially in startup / FillingPipe
    let state = controller.congestion_state();
    assert!(
        matches!(state, CongestionState::FillingPipe),
        "expected FillingPipe at startup, got {state:?}"
    );

    let stats = controller.stats();
    assert_eq!(stats.ack_count, 0, "no ACKs processed yet");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. diagnostic_overlay
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_diagnostic_overlay_render() {
    use oximedia_videoip::diagnostic_overlay::{DiagnosticOverlay, NetworkStats, OverlayConfig};

    let stats = NetworkStats {
        latency_ms: 12.5,
        packet_loss_pct: 0.1,
        bitrate_kbps: 8_000,
        jitter_ms: 1.2,
        fec_recovered: 3,
        frame_seq: 42,
    };

    // format_lines should produce non-empty output
    let lines = stats.format_lines();
    assert!(!lines.is_empty(), "overlay stats lines should be non-empty");

    let overlay = DiagnosticOverlay::new(OverlayConfig::default());
    let mut canvas = vec![0u8; 320 * 240 * 4];
    overlay.render_onto(&mut canvas, 320, 240, &stats);

    // At least one pixel should have been modified
    let non_zero = canvas.iter().any(|&b| b != 0);
    assert!(non_zero, "overlay should have modified canvas pixels");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. multiview
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_multiview_compositor_grid() {
    use oximedia_videoip::multiview::{MosaicLayout, MultiviewCompositor};

    let mut comp = MultiviewCompositor::with_layout(1920, 1080, MosaicLayout::Grid2x2)
        .expect("2×2 grid on 1920×1080 should be valid");

    comp.register_source("cam1", 960, 540);
    comp.register_source("cam2", 960, 540);

    assert_eq!(comp.source_count(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. network_sim
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_network_sim_perfect_delivery() {
    use bytes::Bytes;
    use oximedia_videoip::network_sim::{NetworkProfile, NetworkSimulator};

    let profile = NetworkProfile::perfect();
    // new() returns VideoIpResult<Self>
    let mut sim = NetworkSimulator::new(profile, 12345).expect("perfect profile should be valid");

    let payload = Bytes::from_static(b"hello videoip");
    // send() takes only data: Bytes (no second arg)
    sim.send(payload.clone());

    // Advance clock by 1 ms to flush any pending deliveries
    sim.advance_clock(1_000);
    let delivered = sim.receive();
    assert_eq!(
        delivered.len(),
        1,
        "one packet should be delivered on perfect network"
    );
    assert_eq!(delivered[0].data, payload);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. precise_timer
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_precise_timer_clock_advance() {
    use oximedia_videoip::precise_timer::{PreciseClock, PreciseSleeper, SleepStrategy};

    let clock = PreciseClock::now();
    // Create sleeper with OS sleep strategy
    let sleeper = PreciseSleeper::new(SleepStrategy::OsSleep);
    // Actually sleep a tiny amount
    std::thread::sleep(std::time::Duration::from_millis(1));
    let elapsed = clock.elapsed_ns();
    assert!(
        elapsed > 0,
        "elapsed_ns must be positive after sleep: got {elapsed}"
    );
    assert!(matches!(sleeper.strategy(), SleepStrategy::OsSleep));
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. ptp_clock
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_ptp_clock_timestamp_arithmetic() {
    use oximedia_videoip::ptp_clock::{ClockServo, OffsetEstimator, PtpTimestamp};

    // Verify diff and add_nanos are consistent
    let t1 = PtpTimestamp::new(1, 0);
    let t2 = PtpTimestamp::new(2, 500_000_000); // 2.5 s

    let diff = t2.diff_nanos(t1);
    assert_eq!(diff, 1_500_000_000, "diff should be 1.5 s in nanos");

    let t3 = t1.add_nanos(diff);
    assert_eq!(t3.seconds, 2);
    assert_eq!(t3.nanoseconds, 500_000_000);

    // Smoke test ClockServo
    let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
    let corr = servo.update(100); // 100 ns offset
    assert!(
        corr.abs() < 10_000.0,
        "first correction should be small: {corr}"
    );

    // Smoke test OffsetEstimator — new() returns Result<Self, String>
    let mut estimator = OffsetEstimator::new(0.125).expect("alpha 0.125 is valid");
    estimator.update(500_000);
    assert_eq!(estimator.sample_count(), 1);
    assert!(estimator.estimate_ns() > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. rtcp_sender_report
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_rtcp_sender_report_ntp_roundtrip() {
    use oximedia_videoip::rtcp_sender_report::{ntp_to_rtp, rtp_to_ntp, NtpTimestamp};
    use std::time::Duration;

    // Build a known NTP anchor
    let anchor_ntp = NtpTimestamp::from_unix_duration(Duration::from_secs(1_000_000));
    let anchor_rtp: u32 = 90_000; // 1 second at 90 kHz

    // Forward and back should round-trip
    let rtp_ts: u32 = 180_000; // 2 s worth of 90 kHz ticks
    let ntp =
        rtp_to_ntp(rtp_ts, anchor_rtp, anchor_ntp, 90_000).expect("rtp_to_ntp should succeed");
    let rtp_back =
        ntp_to_rtp(ntp, anchor_rtp, anchor_ntp, 90_000).expect("ntp_to_rtp should succeed");

    assert_eq!(rtp_back, rtp_ts, "RTP timestamp round-trip failed");
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. rtp_2110
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_rtp_2110_packetize_and_parse() {
    use oximedia_videoip::rtp_2110::{Rtp2110Packetizer, RtpPacket as Rtp2110Packet};

    let mut packetizer =
        Rtp2110Packetizer::new(192, 108).expect("small frame dims should be valid");

    let frame = vec![0xAB_u8; 192 * 108 * 2]; // YCbCr 4:2:2 = 2 bytes/px
    let packets = packetizer
        .packetize(&frame)
        .expect("packetize should succeed");

    assert!(!packets.is_empty(), "should produce at least one packet");

    // Round-trip one packet through serialise / parse
    let pkt = &packets[0];
    let bytes = pkt.to_bytes();
    let parsed = Rtp2110Packet::from_bytes(&bytes).expect("from_bytes should succeed");
    assert_eq!(parsed.seq_num, pkt.seq_num);
    assert_eq!(parsed.timestamp, pkt.timestamp);
    assert_eq!(parsed.line_num, pkt.line_num);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. rtp_jitter_buffer
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_rtp_jitter_buffer_insert() {
    use oximedia_videoip::rtp_jitter_buffer::{JitterBuffer, JitterBufferConfig, RtpPacket};
    use std::time::Duration;

    let cfg = JitterBufferConfig {
        capacity: 64,
        playout_delay: Duration::from_millis(20),
        clock_rate: 90_000,
    };
    let mut buf = JitterBuffer::new(cfg).expect("buffer with capacity > 0 should be created");

    let pkt = RtpPacket::new(1, 90_000, 0xDEAD_BEEF, 96, vec![1, 2, 3]);
    // insert() returns InsertOutcome (not Result)
    buf.insert(pkt);

    assert_eq!(buf.stats().inserted, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. rtsp_server
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_rtsp_server_session_creation() {
    use oximedia_videoip::rtsp_server::{RtspSession, RtspSessionState, RtspStatus};

    let session = RtspSession::new("abc123".to_owned(), "/live/cam1".to_owned(), 0);
    assert_eq!(session.state, RtspSessionState::Init);
    assert_eq!(session.url_path, "/live/cam1");
    assert!(!session.is_playing());

    // Verify reason phrases for common status codes
    assert_eq!(RtspStatus::Ok.reason_phrase(), "OK");
    assert_eq!(RtspStatus::NotFound.reason_phrase(), "Not Found");
    assert_eq!(
        RtspStatus::InternalError.reason_phrase(),
        "Internal Server Error"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. sdp_gen
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_sdp_gen_builder() {
    use oximedia_videoip::sdp_gen::Sdp2110Builder;

    let sdp = Sdp2110Builder::new()
        .session_name("test-session")
        .video_stream("239.0.0.1", 5004, "29.97")
        .audio_stream("239.0.0.2", 5006, 48_000, 2)
        .build();

    assert!(sdp.contains("m=video"), "SDP should contain video section");
    assert!(sdp.contains("m=audio"), "SDP should contain audio section");
    assert!(
        sdp.contains("test-session"),
        "SDP should contain session name"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. sdp_negotiation
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_sdp_negotiation_parse_offer() {
    use oximedia_videoip::sdp_negotiation::SdpParser;

    let offer = concat!(
        "v=0\r\n",
        "o=- 0 0 IN IP4 192.168.1.1\r\n",
        "s=Test\r\n",
        "t=0 0\r\n",
        "m=video 5004 RTP/AVP 96\r\n",
        "a=rtpmap:96 raw/90000\r\n",
    );

    let session = SdpParser::parse(offer).expect("valid SDP should parse");
    // Field is `media` not `media_sections`
    assert!(!session.media.is_empty(), "should have media sections");
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. srt_handshake
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_srt_handshake_caller_state_machine() {
    use oximedia_videoip::srt_handshake::{CallerHandshake, CallerPhase};

    // CallerHandshake::new takes: socket_id, initial_seq, mss, recv_latency_ms, snd_latency_ms
    let hs = CallerHandshake::new(0xDEAD_BEEF, 1000, 1316, 120, 120);
    assert!(
        matches!(hs.phase, CallerPhase::Idle),
        "should start in Idle phase"
    );
    // syn_cookie defaults to 0 before listener response
    assert_eq!(hs.syn_cookie, 0);
    assert_eq!(hs.socket_id, 0xDEAD_BEEF);
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. srt_transport
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_srt_transport_session_loopback() {
    use oximedia_videoip::srt_transport::{SrtSession, SrtTransportConfig};

    let cfg_a = SrtTransportConfig::with_latency(120);
    let cfg_b = SrtTransportConfig::with_latency(120);

    let mut session = SrtSession::new(cfg_a, cfg_b);
    let payload = b"hello srt".to_vec();
    session
        .send_lossless(payload.clone())
        .expect("lossless send should succeed");

    // Pop one delivered packet from the receiver
    let received = session.receiver.pop_packet();
    assert!(
        received.is_some(),
        "receiver should have at least one delivered packet"
    );
    let pkt = received.unwrap();
    assert_eq!(pkt.payload, payload, "payload should match");
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. st2110_metadata
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_st2110_metadata_params() {
    use oximedia_videoip::st2110_metadata::{Rational, St211020Params};

    let rate = Rational::new(60_000, 1_001).expect("valid rational");
    assert!((rate.to_f64() - 59.94).abs() < 0.01);

    // Use the pre-built HD 1080p constructor
    let params = St211020Params::hd_1080p59_94();
    assert_eq!(params.width, 1920);
    assert_eq!(params.height, 1080);
    params
        .validate()
        .expect("standard HD params should be valid");
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. stream_recording_mux
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_stream_recording_mux_config_builder() {
    use oximedia_videoip::stream_recording_mux::{ContainerFormat, RecordingConfig, VideoCodecId};

    let cfg = RecordingConfig::archival_mkv("/tmp/test", 1920, 1080, 29.97, 48_000, 2);
    assert_eq!(cfg.format, ContainerFormat::Mkv);
    assert!(!cfg.video_tracks.is_empty());
    assert_eq!(cfg.video_tracks[0].codec, VideoCodecId::Ffv1);
    assert_eq!(cfg.video_tracks[0].width, 1920);
    assert_eq!(cfg.video_tracks[0].height, 1080);
}

// ─────────────────────────────────────────────────────────────────────────────
// 19. stream_relay
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_stream_relay_fanout() {
    use oximedia_videoip::stream_relay::{RelayFrame, RelaySink, StreamRelay};

    let mut relay = StreamRelay::new();
    let sink = RelaySink::new(
        "sink1".to_owned(),
        32,
        oximedia_videoip::stream_relay::DropPolicy::DropOldest,
    );
    relay.add_sink(sink).expect("sink should be added");

    let frame = RelayFrame {
        source_id: "cam1".to_owned(),
        seq: 0,
        pts_us: 0,
        data: vec![0xFF; 100],
    };
    let delivered = relay
        .relay_frame(frame)
        .expect("relay_frame on running relay should succeed");
    assert_eq!(delivered, 1, "one sink should receive the frame");
}

// ─────────────────────────────────────────────────────────────────────────────
// 20. videoip_ext
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_videoip_ext_rist_roundtrip() {
    use oximedia_videoip::videoip_ext::RistPacket;

    let payload = b"smoke test";
    let encoded = RistPacket::serialize(42, 1234, payload);
    let (seq, ts, decoded) =
        RistPacket::deserialize(&encoded).expect("deserialization should succeed");

    assert_eq!(seq, 42 & 0xFFFF);
    assert_eq!(ts, 1234);
    assert_eq!(&decoded, payload);
}

// ─────────────────────────────────────────────────────────────────────────────
// 21. whip_whep
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_whip_whep_session_lifecycle() {
    use oximedia_videoip::whip_whep::{
        IceCandidate, SdpBody, SdpType, SessionRole, SessionState, WhipWhepSession,
    };

    let mut session = WhipWhepSession::new("sess-001".to_owned(), SessionRole::WhipIngester, 0);
    assert_eq!(session.state, SessionState::Offering);
    assert!(session.is_active());

    let answer = SdpBody {
        sdp: "v=0\r\n".to_owned(),
        sdp_type: SdpType::Answer,
    };
    session.set_answer(answer);
    assert_eq!(session.state, SessionState::Negotiating);

    session.add_candidate(IceCandidate {
        candidate: "candidate:1 1 udp 100 192.168.1.2 5004 typ host".to_owned(),
        sdp_mid: Some("0".to_owned()),
        sdp_mline_index: Some(0),
    });
    assert_eq!(session.pending_candidates.len(), 1);

    session.mark_connected();
    assert_eq!(session.state, SessionState::Connected);

    session.terminate();
    assert!(!session.is_active());
}

// ─────────────────────────────────────────────────────────────────────────────
// 22. ptp_clock — path delay computation (additional invariant)
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn smoke_ptp_clock_path_delay_symmetric() {
    use oximedia_videoip::ptp_clock::{PathDelay, PtpTimestamp};

    // Symmetric path: 1 ms each way
    // T1=0 (master send), T2=1ms (slave rx), T3=1.5ms (slave tx), T4=2.5ms (master rx)
    // mean delay = ((T2-T1) + (T4-T3)) / 2 = (1ms + 1ms)/2 = 1 ms
    let path_delay = PathDelay {
        t1: PtpTimestamp::new(0, 0),
        t2: PtpTimestamp::new(0, 1_000_000), // +1 ms
        t3: PtpTimestamp::new(0, 1_500_000), // +1.5 ms
        t4: PtpTimestamp::new(0, 2_500_000), // +2.5 ms
    };

    let delay = path_delay
        .mean_delay_nanos()
        .expect("delay should be computable");
    assert_eq!(delay, 1_000_000, "mean delay should be 1 ms = 1_000_000 ns");

    let offset = path_delay
        .offset_nanos()
        .expect("offset should be computable");
    assert_eq!(offset, 0, "symmetric path => zero offset");
}
