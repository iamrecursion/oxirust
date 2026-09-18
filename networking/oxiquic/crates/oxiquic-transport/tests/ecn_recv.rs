//! Receive-side ECN (RFC 9000 §13.4.1, RFC 9002 §7.4).
//!
//! These tests drive two in-process [`Connection`]s over a synthetic "network"
//! that stamps an IP ECN codepoint on every datagram, exactly as a
//! `recvmsg`-capable I/O layer would report it through
//! [`DatagramMeta`]. They prove the full receive half of ECN:
//!
//! 1. The codepoint of an inbound datagram is counted into the packet-number
//!    space of every packet decrypted out of it.
//! 2. Those counters are echoed to the peer in the ECN section of an ACK-ECN
//!    (0x03) frame — observed indirectly but decisively, because the peer's
//!    RFC 9000 §13.4.2 validation state machine can only reach `Capable` when
//!    it receives ECN feedback covering its ECT(0)-marked packets.
//! 3. A CE mark applied by the (simulated) network to traffic *towards* an
//!    endpoint is echoed back and reduces the sender's congestion window.
//! 4. When the I/O layer reports no codepoint (`ecn: None` — what the bundled
//!    tokio endpoint falls back to on a platform whose kernel refuses
//!    `IP_RECVTOS` / `IPV6_RECVTCLASS`), nothing is counted and plain ACKs are
//!    sent, which makes the peer's validation state machine disable ECN rather
//!    than invent feedback.
//!
//! The socket half — that the bundled endpoint really does read the codepoint
//! off the wire — is proven separately in `tests/ecn_socket.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use oxiquic_core::PacketType;
use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::connection::DatagramMeta;
use oxiquic_transport::ecn::{EcnCodepoint, EcnValidationState};
use oxiquic_transport::{Connection, TransportConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

// ─── Harness ─────────────────────────────────────────────────────────────────

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

fn addr(port: u16) -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], port))
}

const CLIENT_ADDR_PORT: u16 = 40000;
const SERVER_ADDR_PORT: u16 = 4433;

/// Extract `(dcid, scid)` from a long-header Initial datagram.
fn parse_initial_cids(datagram: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let first = *datagram.first()?;
    if first & 0x80 == 0 {
        return None;
    }
    let dcid_len = *datagram.get(5)? as usize;
    let dcid = datagram.get(6..6 + dcid_len)?.to_vec();
    let scid_len = *datagram.get(6 + dcid_len)? as usize;
    let scid = datagram
        .get(7 + dcid_len..7 + dcid_len + scid_len)?
        .to_vec();
    Some((dcid, scid))
}

fn make_pair(first_datagram_ecn: Option<EcnCodepoint>) -> (Connection, Connection) {
    let (client_cfg, server_cfg) = config_pair();
    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();

    let mut client = Connection::new_client(
        client_cfg,
        ServerName::try_from("localhost").expect("server name"),
        addr(SERVER_ADDR_PORT),
        params.clone(),
        Default::default(),
        Default::default(),
    )
    .expect("client conn");

    let now = Instant::now();
    let mut first = Vec::new();
    client
        .poll_transmit(now, &mut first)
        .expect("client initial");
    let (dcid, scid) = parse_initial_cids(&first).expect("parse initial cids");

    let mut server = Connection::new_server(
        server_cfg,
        oxiquic_core::ConnectionId::new(dcid),
        oxiquic_core::ConnectionId::new(scid),
        addr(CLIENT_ADDR_PORT),
        params,
        Default::default(),
        Default::default(),
    )
    .expect("server conn");

    let mut owned = first.clone();
    server
        .handle_datagram_with_meta(
            now,
            &mut owned,
            DatagramMeta {
                src: addr(CLIENT_ADDR_PORT),
                ecn: first_datagram_ecn,
            },
        )
        .expect("server first datagram");

    (client, server)
}

/// The ECN codepoint the synthetic network stamps on datagrams in each
/// direction. `None` models a platform that cannot report a codepoint.
#[derive(Clone, Copy)]
struct NetworkEcn {
    to_server: Option<EcnCodepoint>,
    to_client: Option<EcnCodepoint>,
}

/// Exchange all pending datagrams in both directions until quiescent, stamping
/// each datagram with the configured codepoint as the network would.
fn exchange_all(client: &mut Connection, server: &mut Connection, now: Instant, net: NetworkEcn) {
    for _ in 0..200 {
        let mut any = false;
        loop {
            let mut buf = Vec::new();
            if client.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                let meta = DatagramMeta {
                    src: addr(CLIENT_ADDR_PORT),
                    ecn: net.to_server,
                };
                server.handle_datagram_with_meta(now, &mut buf, meta).ok();
                any = true;
            } else {
                break;
            }
        }
        loop {
            let mut buf = Vec::new();
            if server.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                let meta = DatagramMeta {
                    src: addr(SERVER_ADDR_PORT),
                    ecn: net.to_client,
                };
                client.handle_datagram_with_meta(now, &mut buf, meta).ok();
                any = true;
            } else {
                break;
            }
        }
        if !any {
            break;
        }
    }
}

fn complete_handshake(client: &mut Connection, server: &mut Connection, net: NetworkEcn) {
    let now = Instant::now();
    for _ in 0..100 {
        exchange_all(client, server, now, net);
        if !client.is_handshaking() && !server.is_handshaking() {
            return;
        }
    }
    panic!("handshake did not complete");
}

const ECT0_BOTH: NetworkEcn = NetworkEcn {
    to_server: Some(EcnCodepoint::Ect0),
    to_client: Some(EcnCodepoint::Ect0),
};

const NO_ECN: NetworkEcn = NetworkEcn {
    to_server: None,
    to_client: None,
};

// ─── Tests ───────────────────────────────────────────────────────────────────

/// An ECT(0)-marked datagram increments the ECT(0) counter of the
/// packet-number space its packets decrypt into (RFC 9000 §13.4.1), and a
/// coalesced Initial + Handshake datagram increments *both* spaces.
#[test]
fn inbound_codepoints_are_counted_per_packet_number_space() {
    let (mut client, mut server) = make_pair(Some(EcnCodepoint::Ect0));
    complete_handshake(&mut client, &mut server, ECT0_BOTH);

    let initial = server.ecn_recv_counts(PacketType::Initial);
    let handshake = server.ecn_recv_counts(PacketType::Handshake);
    let application = server.ecn_recv_counts(PacketType::Short);

    assert!(
        initial.ect0 > 0,
        "the client's Initial packets were ECT(0)-marked: {initial:?}"
    );
    assert!(
        handshake.ect0 > 0,
        "the client's Handshake packets were ECT(0)-marked: {handshake:?}"
    );
    assert_eq!(initial.ce, 0, "no CE marks were applied");
    assert_eq!(initial.ect1, 0, "no ECT(1) marks were applied");
    // The Application space counts whatever 1-RTT traffic followed; the point
    // is that the three spaces are tallied independently.
    assert_eq!(
        application.ce, 0,
        "no CE marks were applied to 1-RTT traffic"
    );
}

/// A Not-ECT datagram is counted by no counter at all, so ACKs stay plain
/// (0x02) and neither side's validation state machine can advance.
#[test]
fn not_ect_datagrams_move_no_counter() {
    let (mut client, mut server) = make_pair(Some(EcnCodepoint::NotEct));
    let not_ect = NetworkEcn {
        to_server: Some(EcnCodepoint::NotEct),
        to_client: Some(EcnCodepoint::NotEct),
    };
    complete_handshake(&mut client, &mut server, not_ect);

    for pt in [
        PacketType::Initial,
        PacketType::Handshake,
        PacketType::Short,
    ] {
        let counts = server.ecn_recv_counts(pt);
        assert!(
            counts.is_empty(),
            "Not-ECT must not increment any counter for {pt:?}: {counts:?}"
        );
    }
}

/// End-to-end ECN feedback: both endpoints mark ECT(0) on egress, both count
/// the marks they receive and echo them in ACK-ECN frames, so both validation
/// state machines reach `Capable` (RFC 9000 §13.4.2.1). This is only reachable
/// if the receive-side counters really are transmitted.
#[test]
fn ecn_feedback_promotes_both_peers_to_capable() {
    let (mut client, mut server) = make_pair(Some(EcnCodepoint::Ect0));
    complete_handshake(&mut client, &mut server, ECT0_BOTH);

    let now = Instant::now();
    let stream = client.open_bidi().expect("open bidi stream");
    client
        .send_stream(stream, b"ecn probe", true)
        .expect("send stream data");
    exchange_all(&mut client, &mut server, now, ECT0_BOTH);

    assert_eq!(
        client.ecn_state(),
        EcnValidationState::Capable,
        "the client's ECT(0) packets were acknowledged with ECN counts"
    );
    assert_eq!(
        server.ecn_state(),
        EcnValidationState::Capable,
        "the server's ECT(0) packets were acknowledged with ECN counts"
    );
}

/// When the I/O layer reports no codepoint — the honest state of the bundled
/// tokio endpoint — no counter moves and only plain (0x02) ACKs are produced.
///
/// RFC 9000 §13.4.2.1 then requires the *sender* of ECT(0)-marked packets to
/// fail ECN validation, because an ACK that newly acknowledges ECT-marked
/// packets without ECN counts is indistinguishable from a path that bleached
/// the marks. The correct, safe outcome is therefore that ECN switches itself
/// off and egress falls back to Not-ECT: nothing is fabricated, and no CE
/// signal is ever invented from missing feedback.
#[test]
fn missing_platform_support_disables_ecn_instead_of_faking_it() {
    let (mut client, mut server) = make_pair(None);
    complete_handshake(&mut client, &mut server, NO_ECN);

    let now = Instant::now();
    let stream = client.open_bidi().expect("open bidi stream");
    client
        .send_stream(stream, b"no ecn", true)
        .expect("send stream data");
    exchange_all(&mut client, &mut server, now, NO_ECN);

    assert!(
        server.ecn_recv_counts(PacketType::Short).is_empty(),
        "no codepoint reported means no counter moves"
    );
    for pt in [PacketType::Initial, PacketType::Handshake] {
        assert!(
            server.ecn_recv_counts(pt).is_empty(),
            "no codepoint reported means no counter moves for {pt:?}"
        );
    }
    assert_eq!(
        client.ecn_state(),
        EcnValidationState::Failed,
        "plain ACKs for ECT(0) packets must fail validation (RFC 9000 §13.4.2.1)"
    );
    assert_eq!(
        client.ecn_desired_codepoint(),
        EcnCodepoint::NotEct,
        "a failed space must stop marking ECT(0)"
    );
}

/// A CE mark applied by the network to traffic towards the server is counted,
/// echoed to the client in an ACK-ECN, and reduces the client's congestion
/// window (RFC 9002 §7.4: a CE increase is a congestion signal).
#[test]
fn ce_mark_towards_peer_is_echoed_and_reduces_the_sender_window() {
    let (mut client, mut server) = make_pair(Some(EcnCodepoint::Ect0));
    complete_handshake(&mut client, &mut server, ECT0_BOTH);

    let now = Instant::now();
    // Establish ECN capability first so the client trusts the feedback.
    let stream = client.open_bidi().expect("open bidi stream");
    client
        .send_stream(stream, b"warm up", false)
        .expect("send warm-up data");
    exchange_all(&mut client, &mut server, now, ECT0_BOTH);
    assert_eq!(client.ecn_state(), EcnValidationState::Capable);

    let window_before = client.congestion_window();
    let ce_before = server.ecn_recv_counts(PacketType::Short).ce;

    // The network now marks the client's packets CE.
    let ce_to_server = NetworkEcn {
        to_server: Some(EcnCodepoint::Ce),
        to_client: Some(EcnCodepoint::Ect0),
    };
    client
        .send_stream(stream, b"congested", true)
        .expect("send congested data");
    exchange_all(&mut client, &mut server, now, ce_to_server);

    let ce_after = server.ecn_recv_counts(PacketType::Short).ce;
    assert!(
        ce_after > ce_before,
        "the server must count the CE marks it received ({ce_before} -> {ce_after})"
    );
    assert!(
        client.congestion_window() < window_before,
        "an echoed CE increase must reduce the sender's window \
         ({window_before} -> {})",
        client.congestion_window()
    );
    assert_eq!(
        client.ecn_state(),
        EcnValidationState::Capable,
        "CE feedback is valid feedback: ECN must stay enabled"
    );
}
