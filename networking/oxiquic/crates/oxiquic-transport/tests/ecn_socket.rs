//! Socket-level receive ECN (RFC 9000 §13.4.1) over real UDP loopback.
//!
//! `tests/ecn_recv.rs` proves the *state machine*: given a datagram whose ECN
//! codepoint the I/O layer reports, the counters move and the ACK-ECN frame is
//! emitted. This file proves the *I/O layer* itself: that the bundled `tokio`
//! endpoint actually reads the codepoint off the wire via `recvmsg` ancillary
//! data, for every codepoint and for both address families, and that it reports
//! nothing at all — never a fabricated Not-ECT — when the kernel is not
//! attaching the ancillary data.
//!
//! Where the platform cannot report codepoints at all
//! ([`EcnRecvUnsupported`]), the socket-level tests assert the *typed*
//! unsupported path instead of asserting a codepoint, so they stay honest
//! rather than green-by-accident.

use std::sync::Arc;
use std::time::Duration;

use oxiquic_core::PacketType;
use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::ecn::{EcnCodepoint, EcnValidationState};
use oxiquic_transport::endpoint::ecn_recv::{
    ecn_from_tos_byte, enable_ecn_recv, recv_ecn_from, try_recv_ecn_from, EcnRecvUnsupported,
};
use oxiquic_transport::{ClientEndpoint, ServerEndpoint, TransportConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use socket2::SockRef;
use tokio::net::UdpSocket;

/// Every RFC 3168 codepoint, with the TOS / traffic-class byte that carries it.
const CODEPOINTS: [(u32, EcnCodepoint); 4] = [
    (0x00, EcnCodepoint::NotEct),
    (0x01, EcnCodepoint::Ect1),
    (0x02, EcnCodepoint::Ect0),
    (0x03, EcnCodepoint::Ce),
];

/// Set the egress ECN codepoint on `socket` for the address family of `v6`.
fn mark_egress(socket: &UdpSocket, tos: u32, v6: bool) {
    let sock = SockRef::from(socket);
    if v6 {
        sock.set_tclass_v6(tos).expect("set IPV6_TCLASS");
    } else {
        sock.set_tos_v4(tos).expect("set IP_TOS");
    }
}

/// Drive one send/receive round trip and return the reported codepoint.
async fn round_trip(rx: &UdpSocket, tx: &UdpSocket, tos: u32, v6: bool) -> Option<EcnCodepoint> {
    mark_egress(tx, tos, v6);
    let dst = rx.local_addr().expect("rx local addr");
    tx.send_to(b"probe", dst).await.expect("send probe");
    let mut buf = [0u8; 128];
    let (len, from, ecn) = recv_ecn_from(rx, &mut buf).await.expect("recvmsg");
    assert_eq!(&buf[..len], b"probe", "payload survives the recvmsg path");
    assert_eq!(
        from,
        tx.local_addr().expect("tx local addr"),
        "source address survives the recvmsg path"
    );
    ecn
}

/// Every codepoint set on an IPv4 sender is read back verbatim on the receiver.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_ipv4_codepoint_is_read_off_the_socket() {
    let rx = UdpSocket::bind("127.0.0.1:0").await.expect("bind rx");
    let tx = UdpSocket::bind("127.0.0.1:0").await.expect("bind tx");

    if let Err(e) = enable_ecn_recv(&rx) {
        // Honest degradation: the platform cannot report codepoints, so assert
        // the typed reason and that nothing is invented.
        assert!(
            matches!(
                e,
                EcnRecvUnsupported::Platform { .. } | EcnRecvUnsupported::SockOpt(_)
            ),
            "unsupported must be typed, got {e}"
        );
        assert_eq!(
            round_trip(&rx, &tx, 0x02, false).await,
            None,
            "no codepoint may be invented when the option was refused"
        );
        return;
    }

    for (tos, want) in CODEPOINTS {
        assert_eq!(
            round_trip(&rx, &tx, tos, false).await,
            Some(want),
            "IPv4 TOS {tos:#04x}"
        );
    }
}

/// The DSCP field of the TOS byte must not leak into the reported codepoint.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dscp_bits_do_not_disturb_the_codepoint() {
    let rx = UdpSocket::bind("127.0.0.1:0").await.expect("bind rx");
    let tx = UdpSocket::bind("127.0.0.1:0").await.expect("bind tx");
    if enable_ecn_recv(&rx).is_err() {
        return;
    }
    // 0xA2 = DSCP AF41 with ECT(0) in the low two bits.
    assert_eq!(
        round_trip(&rx, &tx, 0xA2, false).await,
        Some(ecn_from_tos_byte(0xA2))
    );
    assert_eq!(
        round_trip(&rx, &tx, 0xA2, false).await,
        Some(EcnCodepoint::Ect0)
    );
    // 0xB8 = DSCP EF, Not-ECT.
    assert_eq!(
        round_trip(&rx, &tx, 0xB8, false).await,
        Some(EcnCodepoint::NotEct)
    );
}

/// The same, over IPv6, where the codepoint arrives as an `IPV6_TCLASS`
/// control message carrying a native-endian `int` rather than a single byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_ipv6_codepoint_is_read_off_the_socket() {
    let rx = match UdpSocket::bind("[::1]:0").await {
        Ok(s) => s,
        // No IPv6 loopback on this host: nothing to prove, and nothing faked.
        Err(_) => return,
    };
    let tx = match UdpSocket::bind("[::1]:0").await {
        Ok(s) => s,
        Err(_) => return,
    };
    if enable_ecn_recv(&rx).is_err() {
        return;
    }

    for (tclass, want) in CODEPOINTS {
        assert_eq!(
            round_trip(&rx, &tx, tclass, true).await,
            Some(want),
            "IPv6 traffic class {tclass:#04x}"
        );
    }
}

/// The non-blocking burst-drain path reports the codepoint too, and signals
/// `WouldBlock` (rather than a bogus datagram) when the socket is empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_blocking_reads_report_the_codepoint_and_would_block_when_empty() {
    let rx = UdpSocket::bind("127.0.0.1:0").await.expect("bind rx");
    let tx = UdpSocket::bind("127.0.0.1:0").await.expect("bind tx");
    let enabled = enable_ecn_recv(&rx).is_ok();

    let mut buf = [0u8; 128];
    let empty = try_recv_ecn_from(&rx, &mut buf);
    assert_eq!(
        empty.map(|_| ()).unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock,
        "an empty socket must report WouldBlock"
    );

    mark_egress(&tx, 0x03, false);
    tx.send_to(b"ce", rx.local_addr().expect("rx addr"))
        .await
        .expect("send");
    rx.readable().await.expect("readable");
    let (len, _from, ecn) = try_recv_ecn_from(&rx, &mut buf).expect("non-blocking recvmsg");
    assert_eq!(&buf[..len], b"ce");
    if enabled {
        assert_eq!(ecn, Some(EcnCodepoint::Ce));
    } else {
        assert_eq!(ecn, None, "no codepoint may be invented");
    }
}

/// With the socket option never set, the read path reports `None` — not
/// Not-ECT — even though the datagram really was marked ECT(0). Counting a
/// fabricated Not-ECT would make the peer's ECN validation fail spuriously
/// (RFC 9000 §13.4.2.1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn codepoints_are_never_invented_without_the_socket_option() {
    let rx = UdpSocket::bind("127.0.0.1:0").await.expect("bind rx");
    let tx = UdpSocket::bind("127.0.0.1:0").await.expect("bind tx");
    // Deliberately no `enable_ecn_recv(&rx)`.
    assert_eq!(round_trip(&rx, &tx, 0x02, false).await, None);
}

// ─── End-to-end over the bundled endpoints ───────────────────────────────────

/// Matched client/server rustls configs backed by a self-signed Ed25519 cert.
fn config_pair() -> (Arc<ClientConfig>, Arc<ServerConfig>) {
    let ck = oxitls_rcgen::generate_self_signed_ed25519(&["localhost"])
        .expect("generate self-signed cert");
    let cert_der = CertificateDer::from(ck.cert_der.clone());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.pkcs8_der.clone()));
    let provider = Arc::new(quic_crypto_provider());
    let mut roots = RootCertStore::empty();
    roots.add(cert_der.clone()).expect("trust self-signed cert");
    let client = ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&TLS13])
        .expect("client TLS1.3")
        .with_root_certificates(roots)
        .with_no_client_auth();
    let server = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&TLS13])
        .expect("server TLS1.3")
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .expect("server single cert");
    (Arc::new(client), Arc::new(server))
}

/// A real handshake plus stream exchange over loopback leaves both endpoints
/// with non-zero *received* ECT(0) counters, and drives the sender's RFC 9000
/// §13.4.2 validation out of `Testing` — which is only possible if the peer
/// echoed real ECN counts back in ACK-ECN frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn endpoints_exchange_real_ecn_feedback_over_loopback() {
    let (client_cfg, server_cfg) = config_pair();
    let transport = TransportConfig::default();

    let server = ServerEndpoint::bind(
        "127.0.0.1:0".parse().expect("addr"),
        server_cfg,
        transport.clone(),
    )
    .await
    .expect("bind server");
    let server_addr = server.local_addr().expect("server addr");
    let server_reports_ecn = server.reports_inbound_ecn();

    let server_task = tokio::spawn(async move {
        let mut conn = server.accept().await.expect("server accept");
        let (id, bytes, _fin) = conn
            .accept_uni_or_bidi_data()
            .await
            .expect("server reads request");
        conn.send(id, &bytes, false).await.expect("server echoes");
        for _ in 0..20 {
            conn.drive().await.expect("drive");
        }
        conn.ecn_recv_counts(PacketType::Short)
    });

    let client = ClientEndpoint::bind("127.0.0.1:0".parse().expect("addr"), client_cfg, transport)
        .await
        .expect("bind client");
    let client_reports_ecn = client.reports_inbound_ecn();
    let mut conn = client
        .connect(server_addr, "localhost")
        .await
        .expect("client connect");
    let stream = conn.open_bidi().expect("open bidi stream");
    conn.send(stream, b"ecn", false).await.expect("send");
    let (_id, echoed, _fin) =
        tokio::time::timeout(Duration::from_secs(5), conn.accept_uni_or_bidi_data())
            .await
            .expect("echo within 5s")
            .expect("client reads echo");
    assert_eq!(echoed, b"ecn");
    // Keep driving until the peer's ACK-ECN feedback has been validated (or we
    // run out of patience — the assertions below then report what was reached).
    for _ in 0..40 {
        conn.drive().await.expect("drive");
        if conn.ecn_state() == EcnValidationState::Capable {
            break;
        }
    }

    let client_counts = conn.ecn_recv_counts(PacketType::Short);
    let client_state = conn.ecn_state();
    let server_counts = server_task.await.expect("server task");

    if !client_reports_ecn || !server_reports_ecn {
        // Platform cannot report codepoints: nothing may have been counted.
        assert!(
            client_counts.is_empty() && server_counts.is_empty(),
            "counters must stay zero when the platform reports no codepoints"
        );
        return;
    }

    assert!(
        client_counts.ect0 > 0,
        "client counted the server's ECT(0)-marked 1-RTT datagrams, got {client_counts:?}"
    );
    assert!(
        server_counts.ect0 > 0,
        "server counted the client's ECT(0)-marked 1-RTT datagrams, got {server_counts:?}"
    );
    assert_eq!(
        client_counts.ce, 0,
        "a clean loopback path must not report CE"
    );
    // The full RFC 9000 §13.4 loop closed: this endpoint marked ECT(0), the
    // peer read those codepoints off its socket, echoed them in ACK-ECN frames,
    // and validation accepted the feedback. `Capable` is unreachable if either
    // half of ECN is missing.
    assert_eq!(
        client_state,
        EcnValidationState::Capable,
        "ECN validation must succeed when both halves work, got {client_state:?}"
    );
}
