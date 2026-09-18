//! RTSP 1.0 server integration tests.
//!
//! These tests exercise the new server-side primitives added in this release:
//!
//! 1. `test_try_parse_request_options` — parse an OPTIONS request
//! 2. `test_try_parse_request_need_more` — partial bytes return NeedMore
//! 3. `test_response_encode` — `Response::encode` produces valid RTSP wire bytes
//! 4. `test_sdp_serialize_roundtrip` — `for_rtsp_stream` → serialize → parse → check
//! 5. `test_rtp_builder_parse_roundtrip` — builder produces packets parseable by `RtpPacket`
//! 6. `test_server_digest_auth` — `ServerChallenge::issue` → client builds auth → verify
//! 7. `test_server_client_integration` — full OPTIONS→DESCRIBE→SETUP→PLAY→RTP→TEARDOWN

use std::sync::Arc;
use std::time::Duration;

use oximedia_net::rtsp::{
    message::{try_parse_request, Headers, RequestParseStatus, Response},
    rtp::RtpPacketBuilder,
    server::{MountPoint, RtspServer, RtspServerConfig, ServerChallenge},
    Challenge, ClientConfig, Credentials, Method, RtpPacket, RtspClient, ServerEvent,
    SessionDescription, SetupTransport,
};
use tokio::net::TcpListener;
use tokio::time::timeout;

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: try_parse_request
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_try_parse_request_options() {
    let wire = b"OPTIONS rtsp://cam.local/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n";
    match try_parse_request(wire).expect("should not error") {
        RequestParseStatus::Complete { consumed, request } => {
            assert_eq!(consumed, wire.len());
            assert_eq!(request.method, Method::Options);
            assert_eq!(request.uri, "rtsp://cam.local/stream");
            assert_eq!(request.headers.get("cseq"), Some("1"));
            assert!(request.body.is_empty());
        }
        RequestParseStatus::NeedMore => panic!("should have parsed complete request"),
    }
}

#[test]
fn test_try_parse_request_need_more() {
    // Partial header block — no CRLFCRLF yet
    let partial = b"OPTIONS rtsp://cam/ RTSP/1.0\r\nCSeq: 1\r\n";
    assert!(
        matches!(
            try_parse_request(partial).expect("no error on partial input"),
            RequestParseStatus::NeedMore
        ),
        "partial bytes must return NeedMore"
    );

    // Empty buffer
    assert!(matches!(
        try_parse_request(b"").expect("no error on empty"),
        RequestParseStatus::NeedMore
    ));
}

#[test]
fn test_try_parse_all_methods() {
    for (method_str, expected) in &[
        ("OPTIONS", Method::Options),
        ("DESCRIBE", Method::Describe),
        ("SETUP", Method::Setup),
        ("PLAY", Method::Play),
        ("PAUSE", Method::Pause),
        ("TEARDOWN", Method::Teardown),
        ("GET_PARAMETER", Method::GetParameter),
        ("ANNOUNCE", Method::Announce),
    ] {
        let wire = format!("{method_str} rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\n\r\n");
        match try_parse_request(wire.as_bytes()).expect("parse") {
            RequestParseStatus::Complete { request, .. } => {
                assert_eq!(
                    request.method, *expected,
                    "method mismatch for {method_str}"
                );
            }
            RequestParseStatus::NeedMore => panic!("should be complete for {method_str}"),
        }
    }
}

#[test]
fn test_try_parse_request_with_body() {
    let body = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
    let raw = format!(
        "ANNOUNCE rtsp://x/y RTSP/1.0\r\nCSeq: 3\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    match try_parse_request(raw.as_bytes()).expect("parse") {
        RequestParseStatus::Complete { consumed, request } => {
            assert_eq!(consumed, raw.len());
            assert_eq!(request.method, Method::Announce);
            assert_eq!(request.body, body.as_bytes());
        }
        RequestParseStatus::NeedMore => panic!("expected complete"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Response::encode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_response_encode() {
    let resp = Response::build(200, 1);
    let wire = resp.encode();
    let text = std::str::from_utf8(&wire).expect("valid UTF-8");
    assert!(
        text.starts_with("RTSP/1.0 200 OK\r\n"),
        "response must start with status line: {text:?}"
    );
    assert!(
        text.contains("Cseq: 1\r\n"),
        "must include CSeq header: {text:?}"
    );
    assert!(
        text.ends_with("\r\n\r\n"),
        "must end with blank line: {text:?}"
    );
}

#[test]
fn test_response_encode_404() {
    let resp = Response::build(404, 7);
    let text = String::from_utf8(resp.encode()).unwrap();
    assert!(text.starts_with("RTSP/1.0 404 Not Found\r\n"));
}

#[test]
fn test_response_encode_with_sdp_body() {
    let sdp = "v=0\r\nm=video 0 RTP/AVP 96\r\n";
    let mut headers = Headers::new();
    headers.insert("CSeq", "3");
    headers.insert("Content-Type", "application/sdp");
    headers.insert("Content-Length", sdp.len().to_string());
    let resp = Response {
        status: 200,
        reason: "OK".into(),
        headers,
        body: sdp.as_bytes().to_vec(),
    };
    let wire = resp.encode();
    let text = std::str::from_utf8(&wire).unwrap();
    assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
    assert!(text.ends_with(sdp));
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: SDP serialize roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sdp_serialize_roundtrip_video_only() {
    let original =
        SessionDescription::for_rtsp_stream("127.0.0.1", 96, "H264", 90000, None, None, None, None);
    let text = original.to_string();

    // Verify the text has the right structure before parsing.
    assert!(text.starts_with("v=0\r\n"), "must start with v=0: {text:?}");
    assert!(
        text.contains("m=video"),
        "must contain video media: {text:?}"
    );
    assert!(
        text.contains("a=rtpmap:96 H264/90000"),
        "must contain rtpmap: {text:?}"
    );

    let parsed = SessionDescription::parse(&text).expect("roundtrip parse must succeed");
    let video = parsed.video().expect("must have video track");
    assert_eq!(video.formats, vec![96]);
    let rtpmap = video.primary_rtpmap().expect("must have rtpmap");
    assert_eq!(rtpmap.encoding, "H264");
    assert_eq!(rtpmap.clock_rate, 90000);
    assert!(parsed.audio().is_none(), "no audio track in video-only SDP");
}

#[test]
fn test_sdp_serialize_roundtrip_with_audio() {
    let original = SessionDescription::for_rtsp_stream(
        "10.0.0.5",
        96,
        "H264",
        90000,
        Some(97),
        Some("MPEG4-GENERIC"),
        Some(44100),
        Some(2),
    );
    let text = original.to_string();

    assert!(text.contains("m=video"));
    assert!(text.contains("m=audio"));
    assert!(text.contains("a=rtpmap:97 MPEG4-GENERIC/44100/2"));

    let parsed = SessionDescription::parse(&text).expect("roundtrip parse");
    let audio = parsed.audio().expect("must have audio track");
    let artpmap = audio.primary_rtpmap().expect("audio rtpmap");
    assert_eq!(artpmap.encoding, "MPEG4-GENERIC");
    assert_eq!(artpmap.clock_rate, 44100);
    assert_eq!(artpmap.channels, Some(2));
}

#[test]
fn test_sdp_prores_codec_name() {
    // ProRes uses a non-standard codec name per the ProRes spec.
    let sdp = SessionDescription::for_rtsp_stream(
        "127.0.0.1",
        98,
        "X-PRORES",
        90000,
        None,
        None,
        None,
        None,
    );
    let text = sdp.to_string();
    assert!(text.contains("a=rtpmap:98 X-PRORES/90000"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: RtpPacketBuilder
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rtp_builder_parse_roundtrip() {
    let ssrc = 0xDEAD_BEEF_u32;
    let payload_type = 96u8;
    let payload = b"test-rtp-payload";

    let mut builder = RtpPacketBuilder::new(ssrc, payload_type);
    builder.timestamp = 90_000;
    let raw = builder.build(payload);

    let pkt = RtpPacket::parse(&raw).expect("parse must succeed");
    assert_eq!(pkt.ssrc, ssrc, "SSRC mismatch");
    assert_eq!(pkt.payload_type, payload_type, "PT mismatch");
    assert_eq!(pkt.payload, payload, "payload mismatch");
    assert_eq!(pkt.timestamp, 90_000, "timestamp mismatch");
    assert_eq!(pkt.sequence, 0, "first packet sequence must be 0");
    assert_eq!(pkt.version, 2, "RTP version must be 2");
    assert!(!pkt.padding, "no padding");
    assert!(!pkt.extension, "no extension");
}

#[test]
fn test_rtp_builder_sequence_increments() {
    let mut builder = RtpPacketBuilder::new(1, 96);
    let raw0 = builder.build(b"a");
    let raw1 = builder.build(b"b");
    let raw2 = builder.build(b"c");

    let p0 = RtpPacket::parse(&raw0).unwrap();
    let p1 = RtpPacket::parse(&raw1).unwrap();
    let p2 = RtpPacket::parse(&raw2).unwrap();

    assert_eq!(p0.sequence, 0);
    assert_eq!(p1.sequence, 1);
    assert_eq!(p2.sequence, 2);
}

#[test]
fn test_rtp_builder_marker_bit() {
    let mut builder = RtpPacketBuilder::new(42, 96);

    // build_with_marker overrides the marker bit
    let raw_m = builder.build_with_marker(b"x", true);
    let raw_no_m = builder.build_with_marker(b"y", false);

    let pm = RtpPacket::parse(&raw_m).unwrap();
    let pn = RtpPacket::parse(&raw_no_m).unwrap();

    assert!(pm.marker, "marker bit must be set");
    assert!(!pn.marker, "marker bit must be clear");
}

#[test]
fn test_rtp_builder_sequence_wraps() {
    let mut builder = RtpPacketBuilder::new(1, 0);
    // Force sequence to 0xFFFE so the next two packets cross the boundary.
    builder.sequence = 0xFFFE;
    let raw_fffe = builder.build(b"a"); // seq 0xFFFF
    let raw_wrap = builder.build(b"b"); // seq 0x0000 (wraps)

    let p_fffe = RtpPacket::parse(&raw_fffe).unwrap();
    let p_wrap = RtpPacket::parse(&raw_wrap).unwrap();
    assert_eq!(p_fffe.sequence, 0xFFFF);
    assert_eq!(p_wrap.sequence, 0x0000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: ServerChallenge
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_server_digest_auth_issue_and_verify() {
    let server = ServerChallenge::issue("test-realm");
    let hdr = server.www_authenticate_header();
    assert!(hdr.contains("realm=\"test-realm\""));
    assert!(hdr.contains("nonce="));
    assert!(hdr.contains("algorithm=MD5"));

    // Build the client-side response using the same nonce.
    let client_challenge = Challenge::Digest {
        realm: "test-realm".into(),
        nonce: server.nonce.clone(),
        opaque: None,
        qop: None,
        algorithm: Some("MD5".into()),
    };
    let creds = Credentials {
        username: "admin".into(),
        password: "correct-horse".into(),
    };
    let auth = client_challenge.build_authorization(
        &creds,
        "DESCRIBE",
        "rtsp://server/stream",
        1,
        "cnonce",
    );

    assert!(
        server.verify(&auth, "DESCRIBE", "rtsp://server/stream", "correct-horse"),
        "verify must succeed with correct password"
    );
    assert!(
        !server.verify(&auth, "DESCRIBE", "rtsp://server/stream", "wrong-pass"),
        "verify must fail with wrong password"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6: Server integration test
// ─────────────────────────────────────────────────────────────────────────────

/// Spin up a real `RtspServer`, register a mount point, connect an `RtspClient`,
/// step through OPTIONS → DESCRIBE → SETUP → PLAY, publish 3 synthetic RTP
/// frames, receive them on the client side (via `next_event`), then TEARDOWN.
///
/// The whole test is wrapped in a 5-second `tokio::time::timeout` so a stall
/// causes a clean failure rather than a hung CI job.
#[tokio::test]
async fn test_server_client_integration() {
    timeout(Duration::from_secs(5), server_client_integration_inner())
        .await
        .expect("integration test timed out after 5 seconds");
}

async fn server_client_integration_inner() {
    // ── 1. Build the SDP for the mount point ────────────────────────────────
    let sdp_obj =
        SessionDescription::for_rtsp_stream("127.0.0.1", 96, "H264", 90000, None, None, None, None);
    let sdp_text = sdp_obj.to_string();

    // ── 2. Bind the server to port 0 (OS assigns) ───────────────────────────
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind to loopback:0");
    let port = listener.local_addr().expect("local_addr").port();

    // ── 3. Register a mount point ────────────────────────────────────────────
    let config = RtspServerConfig {
        bind_address: format!("127.0.0.1:{port}"),
        session_timeout: Duration::from_secs(60),
        max_connections: 10,
    };
    let server = RtspServer::new(config);
    let (mp, _initial_rx) = MountPoint::new("/stream".into(), sdp_text);
    let mount = server.registry().register(mp);

    // ── 4. Start the server in the background ───────────────────────────────
    let server_handle = tokio::spawn(async move {
        let _ = server.run_with_listener(listener).await;
    });

    // Give the server a tick to start its accept loop.
    tokio::task::yield_now().await;

    // ── 5. Connect the client ────────────────────────────────────────────────
    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let cfg = ClientConfig {
        io_timeout: Duration::from_secs(4),
        ..Default::default()
    };
    let mut client = RtspClient::connect_with(&url, cfg)
        .await
        .expect("client connect");

    // ── 6. OPTIONS ──────────────────────────────────────────────────────────
    let methods = client.options().await.expect("OPTIONS");
    assert!(
        methods.contains(&Method::Describe),
        "OPTIONS must advertise DESCRIBE"
    );

    // ── 7. DESCRIBE ─────────────────────────────────────────────────────────
    let sdp = client.describe().await.expect("DESCRIBE");
    let video = sdp.video().expect("video track in SDP");
    assert_eq!(
        video.primary_rtpmap().expect("rtpmap").encoding,
        "H264",
        "codec must be H264"
    );

    // ── 8. SETUP ────────────────────────────────────────────────────────────
    let control = video.control.as_deref().unwrap_or("trackID=1");
    let setup_resp = client
        .setup(control, &SetupTransport::tcp_interleaved(0))
        .await
        .expect("SETUP");
    assert!(
        !setup_resp.session.is_empty(),
        "session ID must be non-empty"
    );

    // ── 9. PLAY ─────────────────────────────────────────────────────────────
    client.play().await.expect("PLAY");

    // ── 10. Publish 3 RTP frames via the mount point ────────────────────────
    // We do this in a separate task so the publish doesn't block the client loop.
    let mount_ref = Arc::clone(&mount);
    tokio::spawn(async move {
        let mut builder = RtpPacketBuilder::new(0xCAFE_BABE, 96);
        builder.timestamp = 90_000;
        for payload in [b"frame1" as &[u8], b"frame2", b"frame3"] {
            let rtp_bytes = builder.build(payload);
            mount_ref.publish(Arc::new(rtp_bytes));
            // Small yield so the receiver task can process.
            tokio::task::yield_now().await;
        }
    });

    // ── 11. Receive the RTP frames on the client ─────────────────────────────
    // The server now forwards RTP concurrently via the split read/write halves,
    // so the published frames are delivered on the client's event stream without
    // a follow-up request. This test asserts the full control-plane flow
    // (OPTIONS→DESCRIBE→SETUP→PLAY→TEARDOWN); the dedicated
    // `test_rtp_delivered_after_play_without_further_requests` test below proves
    // the concurrent RTP egress in isolation.

    // ── 12. TEARDOWN ─────────────────────────────────────────────────────────
    client.teardown().await.expect("TEARDOWN");
    assert!(
        client.session().is_none(),
        "session must be cleared after TEARDOWN"
    );

    // Clean up the server task.
    server_handle.abort();
}

/// Prove the v2 concurrent RTP egress: after PLAY, a frame published to the
/// mount point is delivered to the client over the interleaved TCP transport
/// **without the client issuing any further request**.
///
/// Under the old single-task design the frame would only flush on the next
/// request's read cycle, so this `next_event()` would stall until the 3-second
/// inner timeout fired. With the split read/write halves the dedicated writer
/// task pushes the frame as soon as it is published.
#[tokio::test]
async fn test_rtp_delivered_after_play_without_further_requests() {
    timeout(
        Duration::from_secs(5),
        rtp_after_play_without_request_inner(),
    )
    .await
    .expect("concurrent-RTP test timed out after 5 seconds");
}

async fn rtp_after_play_without_request_inner() {
    let sdp_text =
        SessionDescription::for_rtsp_stream("127.0.0.1", 96, "H264", 90000, None, None, None, None)
            .to_string();

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind to loopback:0");
    let port = listener.local_addr().expect("local_addr").port();

    let config = RtspServerConfig {
        bind_address: format!("127.0.0.1:{port}"),
        session_timeout: Duration::from_secs(60),
        max_connections: 10,
    };
    let server = RtspServer::new(config);
    // Drop the initial receiver so the playing connection is the only subscriber
    // — this makes the published-frame count assertion below deterministic.
    let (mp, initial_rx) = MountPoint::new("/stream".into(), sdp_text);
    drop(initial_rx);
    let mount = server.registry().register(mp);

    let server_handle = tokio::spawn(async move {
        let _ = server.run_with_listener(listener).await;
    });
    tokio::task::yield_now().await;

    let url = format!("rtsp://127.0.0.1:{port}/stream");
    let cfg = ClientConfig {
        io_timeout: Duration::from_secs(4),
        ..Default::default()
    };
    let mut client = RtspClient::connect_with(&url, cfg)
        .await
        .expect("client connect");

    // Control plane: OPTIONS → DESCRIBE → SETUP → PLAY.
    let _ = client.options().await.expect("OPTIONS");
    let sdp = client.describe().await.expect("DESCRIBE");
    let video = sdp.video().expect("video track in SDP");
    let control = video.control.as_deref().unwrap_or("trackID=1");
    let _ = client
        .setup(control, &SetupTransport::tcp_interleaved(0))
        .await
        .expect("SETUP");
    client.play().await.expect("PLAY");

    // Publish ONE RTP frame AFTER play() returned. The PLAY response is sent by
    // the server only after it subscribes, so by now the connection is a live
    // subscriber — the publish must reach exactly one receiver.
    let mut builder = RtpPacketBuilder::new(0xCAFE_BABE, 96);
    builder.timestamp = 90_000;
    let payload = b"live-frame-after-play" as &[u8];
    let rtp_bytes = builder.build(payload);
    let delivered = mount.publish(Arc::new(rtp_bytes));
    assert_eq!(
        delivered, 1,
        "playing connection must be a live broadcast subscriber after PLAY"
    );

    // Receive it via next_event() WITHOUT issuing another RTSP request. A 3s
    // guard converts a regression (no concurrent egress) into a clean failure.
    let event = timeout(Duration::from_secs(3), client.next_event())
        .await
        .expect("RTP frame must arrive concurrently, with no follow-up request")
        .expect("next_event must succeed");

    match event {
        ServerEvent::Packet(pkt) => {
            assert_eq!(pkt.channel, 0, "RTP must arrive on interleaved channel 0");
            let rtp = RtpPacket::parse(&pkt.data).expect("valid RTP packet");
            assert_eq!(rtp.payload, payload, "RTP payload must round-trip");
            assert_eq!(rtp.payload_type, 96, "payload type must be preserved");
            assert_eq!(rtp.ssrc, 0xCAFE_BABE, "SSRC must be preserved");
        }
        other => panic!("expected an interleaved RTP packet, got {other:?}"),
    }

    // TEARDOWN must still succeed (and stop the writer task) after live egress.
    client.teardown().await.expect("TEARDOWN");
    assert!(
        client.session().is_none(),
        "session must be cleared after TEARDOWN"
    );

    server_handle.abort();
}

/// Verify that the server correctly returns 404 for unknown mount paths.
#[tokio::test]
async fn test_server_describe_404_for_unknown_path() {
    let result = timeout(Duration::from_secs(5), async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let config = RtspServerConfig {
            bind_address: format!("127.0.0.1:{port}"),
            session_timeout: Duration::from_secs(60),
            max_connections: 10,
        };
        let server = RtspServer::new(config);
        // Register nothing — any path should 404.

        let server_handle = tokio::spawn(async move {
            let _ = server.run_with_listener(listener).await;
        });
        tokio::task::yield_now().await;

        let url = format!("rtsp://127.0.0.1:{port}/nonexistent");
        let cfg = ClientConfig {
            io_timeout: Duration::from_secs(4),
            ..Default::default()
        };
        let mut client = RtspClient::connect_with(&url, cfg).await.unwrap();
        let describe_result = client.describe().await;

        server_handle.abort();

        // DESCRIBE should fail with 404
        assert!(
            describe_result.is_err(),
            "DESCRIBE of unregistered path must fail"
        );
        if let Err(oximedia_net::error::NetError::Http { status, .. }) = describe_result {
            assert_eq!(status, 404, "expected 404 status");
        }
    })
    .await;
    assert!(result.is_ok(), "404 test timed out");
}

/// Verify the mount-point registry operations (register, lookup, unregister, list).
#[test]
fn test_mount_point_registry_operations() {
    use oximedia_net::rtsp::server::MountPointRegistry;

    let registry = MountPointRegistry::new();

    // Empty registry.
    assert!(registry.lookup("/stream").is_none());
    assert!(registry.list_paths().is_empty());

    // Register a mount point.
    let (mp1, _rx1) = MountPoint::new("/stream".into(), "v=0\r\n".into());
    let _shared = registry.register(mp1);
    assert!(registry.lookup("/stream").is_some());
    assert_eq!(registry.list_paths(), vec!["/stream"]);

    // Register another.
    let (mp2, _rx2) = MountPoint::new("/other".into(), "v=0\r\n".into());
    registry.register(mp2);
    let mut paths = registry.list_paths();
    paths.sort();
    assert_eq!(paths, vec!["/other", "/stream"]);

    // Unregister.
    assert!(registry.unregister("/stream"));
    assert!(registry.lookup("/stream").is_none());
    assert!(
        !registry.unregister("/stream"),
        "double unregister returns false"
    );

    assert_eq!(registry.list_paths(), vec!["/other"]);
}

/// Verify that publishing to a mount point reaches subscribers.
#[tokio::test]
async fn test_mount_point_publish_subscribe() {
    let (mp, mut rx) = MountPoint::new("/test".into(), "v=0\r\n".into());

    let rtp_bytes = Arc::new(vec![0x80u8, 0x60, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA]);
    let count = mp.publish(Arc::clone(&rtp_bytes));
    assert_eq!(count, 1, "one subscriber should receive the packet");

    let received = rx.recv().await.expect("should receive published packet");
    assert_eq!(
        *received, *rtp_bytes,
        "received bytes must match published bytes"
    );
}

/// Verify that `next_sequence` wraps correctly at u16 boundary.
#[test]
fn test_rtp_builder_next_sequence_wrap() {
    let mut b = RtpPacketBuilder::new(0, 0);
    // Internal sequence starts at u16::MAX, so first call wraps to 0.
    assert_eq!(b.next_sequence(), 0);
    assert_eq!(b.next_sequence(), 1);
    // Jump near wrap boundary.
    b.sequence = 0xFFFE;
    assert_eq!(b.next_sequence(), 0xFFFF);
    assert_eq!(b.next_sequence(), 0x0000);
}
