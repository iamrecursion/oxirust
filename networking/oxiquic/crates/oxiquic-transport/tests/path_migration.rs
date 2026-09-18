//! Path migration (RFC 9000 §9) tests.
//!
//! Two scenarios driven entirely in memory via the synchronous [`Connection`]
//! state machine:
//!
//! 1. `path_challenge_frame_encode_decode` — unit test confirming that
//!    `Frame::PathChallenge` / `Frame::PathResponse` round-trip through the
//!    wire encoder and decoder unchanged.
//!
//! 2. `path_challenge_response_roundtrip` — integration test: after the
//!    handshake the client initiates a PATH_CHALLENGE, both sides exchange
//!    packets, and the client's `path_validated()` becomes `true`.  Stream
//!    data continues to flow without interruption after the probe.
//!
//! 3. `migration_rotates_connection_ids_and_retires_the_old_ones` —
//!    RFC 9000 §9.5: completing a migration issues a fresh batch of connection
//!    IDs and raises `retire_prior_to` so the peer stops using the ones it
//!    carried on the old path.
//!
//! 4. `path_challenge_retransmits_on_pto_and_abandons_after_the_limit` —
//!    RFC 9000 §8.2.1 / §8.2.4: a lost PATH_CHALLENGE is replaced on the
//!    validation PTO, and validation is abandoned once the deadline passes.
//!
//! 5. `path_response_to_a_superseded_challenge_still_validates` —
//!    RFC 9000 §8.2.3: a response to *any* outstanding challenge validates the
//!    path, so a delayed probe is not wasted.
//!
//! Anti-amplification for unvalidated paths is covered in
//! `tests/path_amplification.rs`, and per-path congestion state in the
//! multipath unit tests.

use std::sync::Arc;
use std::time::{Duration, Instant};

use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::connection::multipath::PathValidation;
use oxiquic_transport::connection::DatagramMeta;
use oxiquic_transport::{Connection, TransportConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

/// Extract `(dcid, scid)` bytes from a long-header Initial datagram.
fn parse_initial_cids(datagram: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let first = *datagram.first()?;
    if first & 0x80 == 0 {
        return None;
    }
    // first(1) version(4) dcid_len(1) dcid scid_len(1) scid
    let dcid_len = *datagram.get(5)? as usize;
    let dcid = datagram.get(6..6 + dcid_len)?.to_vec();
    let scid_len = *datagram.get(6 + dcid_len)? as usize;
    let scid = datagram
        .get(7 + dcid_len..7 + dcid_len + scid_len)?
        .to_vec();
    Some((dcid, scid))
}

/// Build a matched client and server connection pair.
fn make_pair() -> (Connection, Connection) {
    let (client_cfg, server_cfg) = config_pair();
    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();

    let client = Connection::new_client(
        client_cfg,
        ServerName::try_from("localhost").expect("server name"),
        addr(4433),
        params.clone(),
        Default::default(),
        Default::default(),
    )
    .expect("client conn");

    let now = Instant::now();
    let mut first = Vec::new();
    let mut client = client;
    client
        .poll_transmit(now, &mut first)
        .expect("client initial");
    let (dcid, scid) = parse_initial_cids(&first).expect("parse initial cids");

    let server = Connection::new_server(
        server_cfg,
        oxiquic_core::ConnectionId::new(dcid),
        oxiquic_core::ConnectionId::new(scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
    )
    .expect("server conn");

    let mut server = server;
    let mut owned = first.clone();
    server
        .handle_datagram(now, &mut owned)
        .expect("server first datagram");

    (client, server)
}

/// Exchange all pending datagrams in both directions until quiescent.
fn exchange_all(client: &mut Connection, server: &mut Connection, now: Instant) {
    for _ in 0..200 {
        let mut any = false;
        loop {
            let mut buf = Vec::new();
            if client.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                server.handle_datagram(now, &mut buf).ok();
                any = true;
            } else {
                break;
            }
        }
        loop {
            let mut buf = Vec::new();
            if server.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                client.handle_datagram(now, &mut buf).ok();
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

/// Run the TLS handshake to completion.
fn complete_handshake(client: &mut Connection, server: &mut Connection) {
    let now = Instant::now();
    for _ in 0..100 {
        exchange_all(client, server, now);
        if !client.is_handshaking() && !server.is_handshaking() {
            return;
        }
    }
    panic!("handshake did not complete");
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// PATH_CHALLENGE and PATH_RESPONSE frames round-trip through the wire encoder
/// and decoder with the correct frame-type byte and data payload preserved.
#[test]
fn path_challenge_frame_encode_decode() {
    use oxiquic_transport::coding::Buf;
    use oxiquic_transport::frame::{decode_frame, Frame};

    // PATH_CHALLENGE round-trip.
    let data: [u8; 8] = [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe];
    let mut wire = Vec::new();
    Frame::PathChallenge(data).encode(&mut wire);
    assert_eq!(wire.len(), 9, "PATH_CHALLENGE wire length must be 9");
    assert_eq!(wire[0], 0x1a, "frame type byte");
    assert_eq!(&wire[1..], &data, "data bytes");
    let mut buf = Buf::new(&wire);
    match decode_frame(&mut buf).expect("decode PATH_CHALLENGE") {
        Frame::PathChallenge(d) => assert_eq!(d, data),
        other => panic!("expected PathChallenge, got {other:?}"),
    }
    assert!(buf.is_empty(), "buffer fully consumed");

    // PATH_RESPONSE round-trip.
    let resp: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    let mut wire2 = Vec::new();
    Frame::PathResponse(resp).encode(&mut wire2);
    assert_eq!(wire2.len(), 9, "PATH_RESPONSE wire length must be 9");
    assert_eq!(wire2[0], 0x1b, "frame type byte");
    assert_eq!(&wire2[1..], &resp, "data bytes");
    let mut buf2 = Buf::new(&wire2);
    match decode_frame(&mut buf2).expect("decode PATH_RESPONSE") {
        Frame::PathResponse(d) => assert_eq!(d, resp),
        other => panic!("expected PathResponse, got {other:?}"),
    }
    assert!(buf2.is_empty(), "buffer fully consumed");

    // Zero data round-trips correctly.
    let mut wire3 = Vec::new();
    Frame::PathChallenge([0u8; 8]).encode(&mut wire3);
    let mut buf3 = Buf::new(&wire3);
    match decode_frame(&mut buf3).expect("decode zero PathChallenge") {
        Frame::PathChallenge(d) => assert_eq!(d, [0u8; 8]),
        other => panic!("expected PathChallenge, got {other:?}"),
    }
}

/// After a handshake the client initiates a PATH_CHALLENGE. After one round
/// of `exchange_all`, the server has echoed back a PATH_RESPONSE and the
/// client's `path_validated()` returns `true`. Stream data also continues
/// to flow correctly before and after the probe.
#[test]
fn path_challenge_response_roundtrip() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();

    // Send some data before the path probe to confirm baseline works.
    let stream = client.open_bidi().expect("open bidi stream");
    client
        .send_stream(stream, b"before probe", false)
        .expect("send before probe");
    exchange_all(&mut client, &mut server, now);

    let (pre_data, _fin) = server.read_stream(stream).expect("read before probe");
    assert_eq!(pre_data, b"before probe", "pre-probe data delivered");

    // Client initiates a PATH_CHALLENGE; should succeed post-handshake.
    client
        .initiate_path_challenge()
        .expect("initiate_path_challenge accepted");

    // PATH_CHALLENGE arrives at server; server queues PATH_RESPONSE.
    // PATH_RESPONSE arrives at client; client sets path_validated = true.
    exchange_all(&mut client, &mut server, now);

    assert!(
        client.path_validated(),
        "path must be validated after a full PATH_CHALLENGE / PATH_RESPONSE exchange"
    );

    // Data must still flow correctly after the probe.
    client
        .send_stream(stream, b"after probe", true)
        .expect("send after probe");
    exchange_all(&mut client, &mut server, now);

    let (post_data, fin) = server.read_stream(stream).expect("read after probe");
    assert_eq!(post_data, b"after probe", "post-probe data delivered");
    assert!(fin, "stream should be finished");
}

/// A PATH_CHALLENGE from the server is echoed back by the client (RFC 9000
/// §8.2.2: every PATH_CHALLENGE must be answered with PATH_RESPONSE).
#[test]
fn server_challenge_echoed_by_client() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();

    // Server initiates the challenge this time.
    server
        .initiate_path_challenge()
        .expect("server initiate_path_challenge");

    exchange_all(&mut client, &mut server, now);

    assert!(
        server.path_validated(),
        "server path must be validated after client echoes PATH_RESPONSE"
    );
}

/// `initiate_path_challenge` returns an error before 1-RTT keys are ready.
#[test]
fn path_challenge_requires_1rtt_keys() {
    let (mut client, _server) = make_pair();
    // Client is still in the handshaking state — no 1-RTT keys yet.
    assert!(
        client.is_handshaking(),
        "client should be handshaking at this point"
    );
    let err = client
        .initiate_path_challenge()
        .expect_err("must fail before 1-RTT keys");
    // Verify the error message is meaningful.
    let msg = err.to_string();
    assert!(
        msg.contains("1-RTT") || msg.contains("connection error"),
        "error message should mention 1-RTT: {msg}"
    );
}

// ─── RFC 9000 §9.5 / §8.2 helpers ─────────────────────────────────────────────

/// The address the peer appears to move to during the migration tests.
const MIGRATED_PORT: u16 = 40001;
/// The address the client is established on (see `make_pair`).
const CLIENT_PORT: u16 = 40000;
/// The address the server is established on (see `make_pair`).
const SERVER_PORT: u16 = 4433;

fn meta(port: u16) -> DatagramMeta {
    DatagramMeta::from_peer(addr(port))
}

/// Make the client emit one datagram and hand it to the server as if it had
/// arrived from [`MIGRATED_PORT`] — exactly what a NAT rebinding looks like —
/// after registering that address as the candidate path so it gets its own
/// RFC 9000 §9.3 allowance and can afford to be probed.
fn begin_migration(client: &mut Connection, server: &mut Connection, now: Instant) {
    let stream = client.open_bidi().expect("open bidi");
    client
        .send_stream(stream, &[0xa5u8; 512], false)
        .expect("queue stream data");
    let mut moved = Vec::new();
    client
        .poll_transmit(now, &mut moved)
        .expect("client datagram");
    server.set_candidate_peer_addr(addr(MIGRATED_PORT));
    server
        .handle_datagram_with_meta(now, &mut moved, meta(MIGRATED_PORT))
        .expect("datagram from the migrated address");
}

/// Deliver everything the server has to send to the client.
fn ferry_server_to_client(server: &mut Connection, client: &mut Connection, now: Instant) {
    for _ in 0..32 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                client
                    .handle_datagram_with_meta(now, &mut out, meta(SERVER_PORT))
                    .ok();
            }
            _ => break,
        }
    }
}

/// Deliver everything the client has to send to the server, attributed to the
/// migrated address.
fn ferry_client_to_server(client: &mut Connection, server: &mut Connection, now: Instant) {
    for _ in 0..32 {
        let mut out = Vec::new();
        match client.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                server
                    .handle_datagram_with_meta(now, &mut out, meta(MIGRATED_PORT))
                    .ok();
            }
            _ => break,
        }
    }
}

/// Drain the server's send queue at `now`, returning how many datagrams were
/// addressed to the candidate (migrated) address — i.e. how many
/// PATH_CHALLENGE probes left. Everything else is discarded.
fn drain_probes(server: &mut Connection, now: Instant) -> usize {
    let mut probes = 0;
    for _ in 0..32 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(dst) if !out.is_empty() => {
                if dst == addr(MIGRATED_PORT) {
                    probes += 1;
                }
            }
            _ => break,
        }
    }
    probes
}

/// Drain the server's send queue at `now` and return the first datagram
/// addressed to the candidate address, without delivering it anywhere — the
/// probe is "in flight" and can be delivered later, or never.
fn hold_probe(server: &mut Connection, now: Instant) -> Vec<u8> {
    for _ in 0..32 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(dst) if !out.is_empty() => {
                if dst == addr(MIGRATED_PORT) {
                    return out;
                }
            }
            _ => break,
        }
    }
    Vec::new()
}

// ─── RFC 9000 §9.5: connection-ID rotation on migration ───────────────────────

/// A completed migration retires every connection ID the peer used on the old
/// path and hands it a full fresh batch (RFC 9000 §9.5): reusing the old CIDs
/// would let an observer of both paths link them to the same connection.
///
/// The peer's own `active_connection_id_limit` accounting is the check that
/// matters here — over-issuing would earn a CONNECTION_ID_LIMIT_ERROR, so the
/// test asserts both sides stay alive and the client's pool never exceeds the
/// limit it advertised.
#[test]
fn migration_rotates_connection_ids_and_retires_the_old_ones() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();
    exchange_all(&mut client, &mut server, now);

    let before = server.local_cid_seqs();
    assert!(
        before.len() > 1,
        "the server supplies spare CIDs post-handshake (got {before:?})"
    );
    assert_eq!(
        server.local_cid_retire_prior_to(),
        0,
        "nothing is retired before a migration"
    );
    let pool_size = before.len();
    let boundary = before.iter().copied().max().expect("non-empty pool") + 1;

    // The peer moves; the server probes the new address and the peer answers.
    begin_migration(&mut client, &mut server, now);
    server
        .initiate_path_challenge_at(now)
        .expect("challenge queued post-handshake");
    ferry_server_to_client(&mut server, &mut client, now);
    ferry_client_to_server(&mut client, &mut server, now);

    assert!(server.path_validated(), "the new path is validated");
    assert_eq!(
        server.peer_addr(),
        addr(MIGRATED_PORT),
        "the connection migrates to the validated address"
    );

    // §9.5: everything issued before the migration is now asked to be retired,
    // and an equally large fresh batch replaces it.
    assert_eq!(
        server.local_cid_retire_prior_to(),
        boundary,
        "retire_prior_to moves past every pre-migration CID"
    );
    let fresh: Vec<u64> = server
        .local_cid_seqs()
        .into_iter()
        .filter(|s| *s >= boundary)
        .collect();
    assert_eq!(
        fresh.len(),
        pool_size,
        "the fresh batch is as large as the pool it replaces (got {fresh:?})"
    );

    // Deliver the new CIDs and collect the client's RETIRE_CONNECTION_ID frames.
    for _ in 0..3 {
        ferry_server_to_client(&mut server, &mut client, now);
        ferry_client_to_server(&mut client, &mut server, now);
    }

    assert!(
        client.peer_close_reason().is_none(),
        "the client accepted the rotation: {:?}",
        client.peer_close_reason()
    );
    assert!(
        server.peer_close_reason().is_none(),
        "the server stayed open: {:?}",
        server.peer_close_reason()
    );
    assert_eq!(
        client.peer_cid_retire_threshold(),
        boundary,
        "the client honours the advertised retire_prior_to"
    );
    let peer_seqs = client.peer_cid_seqs();
    assert!(
        peer_seqs.iter().all(|s| *s >= boundary),
        "the client dropped every pre-migration CID (got {peer_seqs:?})"
    );
    assert!(
        peer_seqs.len() <= pool_size,
        "the peer's active_connection_id_limit is respected (got {peer_seqs:?})"
    );
    let server_seqs = server.local_cid_seqs();
    assert!(
        server_seqs.iter().all(|s| *s >= boundary),
        "RETIRE_CONNECTION_ID removes the old CIDs from the issuer too \
         (got {server_seqs:?})"
    );
}

// ─── RFC 9000 §8.2: PATH_CHALLENGE retransmission and abandonment ─────────────

/// A PATH_CHALLENGE that is never answered is retransmitted on the validation
/// PTO with fresh unpredictable data (RFC 9000 §8.2.1), and validation is
/// abandoned once three times the path PTO has elapsed (§8.2.4) — leaving the
/// connection on the address it is already established on.
#[test]
fn path_challenge_retransmits_on_pto_and_abandons_after_the_limit() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let t0 = Instant::now();
    exchange_all(&mut client, &mut server, t0);
    begin_migration(&mut client, &mut server, t0);
    server
        .initiate_path_challenge_at(t0)
        .expect("challenge queued post-handshake");

    assert_eq!(
        drain_probes(&mut server, t0),
        1,
        "exactly one PATH_CHALLENGE leaves for the candidate address"
    );
    assert_eq!(server.path_challenge_attempts(), 1);
    assert_eq!(server.outstanding_path_challenges().len(), 1);

    // No PATH_RESPONSE: the probe was lost. The §8.2.1 timer replaces it.
    let t1 = t0 + Duration::from_millis(2_000);
    server.handle_timeout(t1);
    assert_eq!(
        drain_probes(&mut server, t1),
        1,
        "a replacement PATH_CHALLENGE is sent once the validation PTO expires"
    );
    assert_eq!(server.path_challenge_attempts(), 2);
    let outstanding = server.outstanding_path_challenges();
    assert_eq!(
        outstanding.len(),
        2,
        "both challenges stay outstanding (RFC 9000 §8.2.3)"
    );
    assert_ne!(
        outstanding[0], outstanding[1],
        "RFC 9000 §8.2.1: unpredictable data in *every* PATH_CHALLENGE"
    );
    assert!(!server.path_validation_failed(), "still probing");

    // Still nothing back: give up (§8.2.4).
    let t2 = t0 + Duration::from_secs(10);
    server.handle_timeout(t2);

    assert!(
        server.path_validation_failed(),
        "validation is abandoned past three times the path PTO"
    );
    assert!(!server.path_validated());
    assert_eq!(
        server.candidate_peer_addr(),
        None,
        "the candidate address is dropped"
    );
    assert_eq!(
        server.peer_addr(),
        addr(CLIENT_PORT),
        "the connection stays on the path it already validated"
    );
    let validation = server
        .multipath_state()
        .path_by_addr(addr(MIGRATED_PORT))
        .map(|(_, p)| p.validation);
    assert_eq!(
        validation,
        Some(PathValidation::Failed),
        "the abandoned path is recorded as failed"
    );
    assert_eq!(
        drain_probes(&mut server, t2),
        0,
        "no further probes are sent after abandonment"
    );
}

/// RFC 9000 §8.2.3: a PATH_RESPONSE answering an *earlier* challenge still
/// validates the path. A challenge that was merely delayed rather than lost
/// must not be wasted just because a replacement went out in the meantime.
#[test]
fn path_response_to_a_superseded_challenge_still_validates() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let t0 = Instant::now();
    exchange_all(&mut client, &mut server, t0);
    begin_migration(&mut client, &mut server, t0);
    server
        .initiate_path_challenge_at(t0)
        .expect("challenge queued post-handshake");

    // The first probe is held in the network instead of delivered.
    let mut delayed = hold_probe(&mut server, t0);
    assert!(!delayed.is_empty(), "the first probe was emitted");
    assert_eq!(server.path_challenge_attempts(), 1);

    // The validation PTO expires and a second, different challenge goes out —
    // and is itself lost.
    let t1 = t0 + Duration::from_millis(2_000);
    server.handle_timeout(t1);
    assert_eq!(drain_probes(&mut server, t1), 1, "replacement probe sent");
    assert_eq!(server.path_challenge_attempts(), 2);

    // The delayed first probe finally arrives; the peer echoes *that* nonce.
    client
        .handle_datagram_with_meta(t1, &mut delayed, meta(SERVER_PORT))
        .expect("delayed probe decrypts");
    ferry_client_to_server(&mut client, &mut server, t1);

    assert!(
        server.path_validated(),
        "a response to any outstanding challenge validates the path"
    );
    assert_eq!(server.peer_addr(), addr(MIGRATED_PORT));
    assert!(
        !server.path_validation_failed(),
        "a validated path is not a failed one"
    );
}
