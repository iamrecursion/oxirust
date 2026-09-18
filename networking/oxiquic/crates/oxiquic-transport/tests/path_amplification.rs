//! Per-path anti-amplification for addresses adopted during migration
//! (RFC 9000 §9.3).
//!
//! The connection-level three-times limit of §8.1 is lifted once the handshake
//! proves the peer owns its address. §9.3 makes clear that this proof does not
//! transfer: when a peer appears to move to a **new** address, that address is
//! unvalidated all over again, and until a PATH_CHALLENGE / PATH_RESPONSE
//! exchange validates it the endpoint may send it at most three times what it
//! has received *from that address*. Otherwise an attacker who can inject one
//! spoofed packet from a victim's address turns an established, fully
//! handshaked connection back into a reflector.
//!
//! These tests drive the synchronous [`Connection`] directly, so no sockets or
//! timing are involved.

use std::sync::Arc;
use std::time::{Duration, Instant};

use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::connection::DatagramMeta;
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
/// The address the peer appears to move to.
const MIGRATED_ADDR_PORT: u16 = 40001;

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

fn make_pair() -> (Connection, Connection) {
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
            DatagramMeta::from_peer(addr(CLIENT_ADDR_PORT)),
        )
        .expect("server first datagram");

    (client, server)
}

/// Exchange everything both ways until quiescent, with each datagram attributed
/// to the address it really came from.
fn exchange_all(client: &mut Connection, server: &mut Connection, now: Instant) {
    for _ in 0..200 {
        let mut any = false;
        loop {
            let mut buf = Vec::new();
            if client.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                server
                    .handle_datagram_with_meta(
                        now,
                        &mut buf,
                        DatagramMeta::from_peer(addr(CLIENT_ADDR_PORT)),
                    )
                    .ok();
                any = true;
            } else {
                break;
            }
        }
        loop {
            let mut buf = Vec::new();
            if server.poll_transmit(now, &mut buf).is_some() && !buf.is_empty() {
                client
                    .handle_datagram_with_meta(
                        now,
                        &mut buf,
                        DatagramMeta::from_peer(addr(SERVER_ADDR_PORT)),
                    )
                    .ok();
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

/// Look up the registered path for `port` and return its amplification state.
fn path_amplification(
    conn: &Connection,
    port: u16,
) -> oxiquic_transport::connection::multipath::PathAmplification {
    conn.multipath_state()
        .path_by_addr(addr(port))
        .and_then(|(_, p)| p.amplification)
        .expect("migrated address must be a registered path with its own allowance")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// A migrated address is registered as its own path with a fresh, unvalidated
/// three-times allowance — the handshake's proof does not carry over.
#[test]
fn migrated_address_gets_its_own_unvalidated_allowance() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);
    assert!(
        server.address_validated(),
        "the handshake validates the original client address"
    );

    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));

    let amp = path_amplification(&server, MIGRATED_ADDR_PORT);
    assert!(
        !amp.validated,
        "a freshly adopted address starts unvalidated"
    );
    assert_eq!(amp.bytes_recv, 0, "nothing received from it yet");
    assert_eq!(amp.budget(), Some(0), "so nothing may be sent to it yet");
}

/// A new address that has sent nothing gets nothing: the PATH_CHALLENGE probe
/// is withheld until its own allowance can cover a datagram.
///
/// Crucially the *connection* must not be affected. A path-validation probe
/// aimed at an unvalidated address being budget-blocked is not the RFC 9002
/// §6.2.2.1 "cannot send at all" condition, so traffic to the established peer
/// must keep flowing and the loss-detection timer must stay armed — otherwise a
/// single small spoofed datagram from an unknown address would wedge an
/// established connection permanently.
#[test]
fn silent_new_address_gets_no_probe_and_does_not_stall_the_connection() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    // Put real data in flight so the connection genuinely has work to do.
    let now = Instant::now();
    let stream = server.open_bidi().expect("server opens a stream");
    server
        .send_stream(stream, &[0x33u8; 256], false)
        .expect("queue server data");

    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));
    server
        .initiate_path_challenge()
        .expect("challenge queued post-handshake");

    let mut out = Vec::new();
    let target = server.poll_transmit(now, &mut out);
    assert_eq!(
        target,
        Some(addr(CLIENT_ADDR_PORT)),
        "traffic must keep going to the established peer address"
    );
    assert!(!out.is_empty(), "the connection must still emit data");

    let amp = path_amplification(&server, MIGRATED_ADDR_PORT);
    assert_eq!(
        amp.bytes_sent, 0,
        "nothing may be charged to (or sent toward) the silent new address"
    );
    assert!(
        !server.amplification_blocked(),
        "an unaffordable probe must not mark the whole connection blocked"
    );
    assert!(
        server.next_timeout().is_some(),
        "the loss-detection / idle timer must stay armed with data in flight"
    );

    // Drain everything the server is willing to send: none of it may go to the
    // unvalidated address, and the challenge stays queued for later.
    for _ in 0..32 {
        let mut buf = Vec::new();
        match server.poll_transmit(now, &mut buf) {
            Some(dst) if !buf.is_empty() => assert_ne!(
                dst,
                addr(MIGRATED_ADDR_PORT),
                "no datagram may reach an address with a zero allowance"
            ),
            _ => break,
        }
    }
    assert_eq!(
        path_amplification(&server, MIGRATED_ADDR_PORT).bytes_sent,
        0,
        "the silent address must still have received nothing"
    );
}

/// Regression: a *small but non-zero* datagram from a new address gives that
/// path a budget too small for a probe packet. The connection must keep
/// running, and the challenge must be emitted later once the address has sent
/// enough — not dropped and not allowed to stall every packet-number space.
#[test]
fn undersized_new_path_budget_defers_the_probe_without_stalling() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();
    // A tiny (ACK-sized) datagram credited to the new address: 3x of it is far
    // below the smallest datagram the builder can produce.
    let stream = client.open_bidi().expect("open bidi");
    client.send_stream(stream, b"x", false).expect("queue byte");
    let mut tiny = Vec::new();
    client
        .poll_transmit(now, &mut tiny)
        .expect("client datagram");
    assert!(
        tiny.len() < 192,
        "this test needs an undersized datagram, got {} bytes",
        tiny.len()
    );

    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));
    server
        .handle_datagram_with_meta(
            now,
            &mut tiny,
            DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
        )
        .expect("tiny datagram from the migrated address");
    server
        .initiate_path_challenge()
        .expect("challenge queued post-handshake");

    // The connection keeps serving the validated path.
    let mut out = Vec::new();
    let target = server.poll_transmit(now, &mut out);
    assert_eq!(
        target,
        Some(addr(CLIENT_ADDR_PORT)),
        "an unaffordable probe must not divert traffic from the validated path"
    );
    assert!(!server.amplification_blocked());
    assert_eq!(
        path_amplification(&server, MIGRATED_ADDR_PORT).bytes_sent,
        0,
        "nothing sent to the under-funded address"
    );

    // Now the new address sends enough to fund a probe: the still-queued
    // challenge is finally emitted, to that address.
    client
        .send_stream(stream, &[0x44u8; 512], false)
        .expect("queue more data");
    let mut big = Vec::new();
    client
        .poll_transmit(now, &mut big)
        .expect("client datagram");
    server
        .handle_datagram_with_meta(
            now,
            &mut big,
            DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
        )
        .expect("larger datagram from the migrated address");

    let mut probed = false;
    for _ in 0..32 {
        let mut buf = Vec::new();
        match server.poll_transmit(now, &mut buf) {
            Some(dst) if !buf.is_empty() => {
                if dst == addr(MIGRATED_ADDR_PORT) {
                    probed = true;
                }
            }
            _ => break,
        }
    }
    assert!(
        probed,
        "the deferred PATH_CHALLENGE must be sent once the new address funds it"
    );
}

/// Once the new address has actually sent something, the probe goes **to that
/// address** (RFC 9000 §9.3.3) and the total bytes sent to it never exceed
/// three times what it sent us.
#[test]
fn probe_targets_the_new_address_and_respects_three_times_received() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();
    // The client emits a datagram; the network delivers it from the new address
    // (exactly what a NAT rebinding looks like to the server).
    let stream = client.open_bidi().expect("open bidi");
    client
        .send_stream(stream, &[0xa5u8; 512], false)
        .expect("queue stream data");
    let mut moved = Vec::new();
    client
        .poll_transmit(now, &mut moved)
        .expect("client datagram");
    let received_len = moved.len() as u64;
    assert!(received_len > 0);

    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));
    server
        .handle_datagram_with_meta(
            now,
            &mut moved,
            DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
        )
        .expect("datagram from the migrated address");

    let amp = path_amplification(&server, MIGRATED_ADDR_PORT);
    assert_eq!(
        amp.bytes_recv, received_len,
        "bytes must be credited to the path they arrived on"
    );
    assert_eq!(amp.budget(), Some(received_len * 3));

    server
        .initiate_path_challenge()
        .expect("challenge queued post-handshake");

    // Drain everything the server is willing to send while the probe is queued.
    let mut sent_to_new = 0u64;
    let mut probes = 0usize;
    for _ in 0..64 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(dst) if !out.is_empty() => {
                if dst == addr(MIGRATED_ADDR_PORT) {
                    sent_to_new += out.len() as u64;
                    probes += 1;
                }
            }
            _ => break,
        }
    }

    assert!(
        probes >= 1,
        "the PATH_CHALLENGE must be sent to the new address"
    );
    assert!(
        sent_to_new <= received_len * 3,
        "sent {sent_to_new} bytes to an unvalidated address that sent us \
         {received_len} (limit {})",
        received_len * 3
    );
    let amp = path_amplification(&server, MIGRATED_ADDR_PORT);
    assert_eq!(amp.bytes_sent, sent_to_new, "the path must be charged");
}

/// A validated address loses its per-path cap: after the PATH_RESPONSE matches,
/// the allowance is lifted and the connection may send freely again.
#[test]
fn validating_the_new_address_lifts_the_per_path_limit() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();
    // Give the migrated address a budget so the probe can leave.
    let mut moved = Vec::new();
    client.poll_transmit(now, &mut moved);
    if moved.is_empty() {
        // The client has nothing queued; make it produce a PING-bearing packet.
        let stream = client.open_bidi().expect("open bidi");
        client
            .send_stream(stream, &[0x5au8; 512], false)
            .expect("queue data");
        client
            .poll_transmit(now, &mut moved)
            .expect("client datagram");
    }
    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));
    server
        .handle_datagram_with_meta(
            now,
            &mut moved,
            DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
        )
        .expect("datagram from the migrated address");
    server
        .initiate_path_challenge()
        .expect("challenge queued post-handshake");

    // Ferry the probe to the client and its PATH_RESPONSE back.
    for _ in 0..16 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                client
                    .handle_datagram_with_meta(
                        now,
                        &mut out,
                        DatagramMeta::from_peer(addr(SERVER_ADDR_PORT)),
                    )
                    .ok();
            }
            _ => break,
        }
    }
    for _ in 0..16 {
        let mut out = Vec::new();
        match client.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                server
                    .handle_datagram_with_meta(
                        now,
                        &mut out,
                        DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
                    )
                    .ok();
            }
            _ => break,
        }
    }

    assert!(
        server.path_validated(),
        "the client echoed the PATH_CHALLENGE, so the new path is validated"
    );
    assert_eq!(
        server.peer_addr(),
        addr(MIGRATED_ADDR_PORT),
        "the connection migrates to the validated address"
    );
    let amp = path_amplification(&server, MIGRATED_ADDR_PORT);
    assert!(amp.validated, "validation must clear the per-path limit");
    assert_eq!(
        amp.budget(),
        None,
        "a validated address has no anti-amplification budget"
    );
}

/// After migrating onto a second path, packets are attributed to that path:
/// its own congestion controller carries the in-flight bytes, and the scheduler
/// keeps the connection on the migrated address instead of dragging it back.
#[test]
fn packets_after_migration_are_charged_to_the_new_paths_controller() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let now = Instant::now();
    let stream = client.open_bidi().expect("open bidi");
    client
        .send_stream(stream, &[0x11u8; 512], false)
        .expect("queue data");
    let mut moved = Vec::new();
    client
        .poll_transmit(now, &mut moved)
        .expect("client datagram");

    server.set_candidate_peer_addr(addr(MIGRATED_ADDR_PORT));
    server
        .handle_datagram_with_meta(
            now,
            &mut moved,
            DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
        )
        .expect("datagram from the migrated address");
    server
        .initiate_path_challenge()
        .expect("challenge queued post-handshake");

    // Complete the challenge/response so the new path is validated and active.
    for _ in 0..16 {
        let mut out = Vec::new();
        match server.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                client
                    .handle_datagram_with_meta(
                        now,
                        &mut out,
                        DatagramMeta::from_peer(addr(SERVER_ADDR_PORT)),
                    )
                    .ok();
            }
            _ => break,
        }
    }
    for _ in 0..16 {
        let mut out = Vec::new();
        match client.poll_transmit(now, &mut out) {
            Some(_) if !out.is_empty() => {
                server
                    .handle_datagram_with_meta(
                        now,
                        &mut out,
                        DatagramMeta::from_peer(addr(MIGRATED_ADDR_PORT)),
                    )
                    .ok();
            }
            _ => break,
        }
    }
    assert!(server.path_validated(), "path validated");

    let migrated_index = server
        .multipath_state()
        .path_by_addr(addr(MIGRATED_ADDR_PORT))
        .map(|(i, _)| i)
        .expect("migrated path registered");
    assert_ne!(
        migrated_index, 0,
        "the migrated path is not the initial one"
    );

    // Send fresh application data; it must go to the migrated address and be
    // charged to that path's congestion controller.
    let in_flight_before = server
        .multipath_state()
        .path_recovery(migrated_index)
        .expect("per-path recovery state")
        .bytes_in_flight();
    let sstream = server.open_bidi().expect("server opens a stream");
    server
        .send_stream(sstream, &[0x22u8; 256], false)
        .expect("queue server data");
    let mut out = Vec::new();
    let dst = server
        .poll_transmit(now, &mut out)
        .expect("server datagram");
    assert_eq!(
        dst,
        addr(MIGRATED_ADDR_PORT),
        "traffic must follow the migrated path"
    );
    let in_flight_after = server
        .multipath_state()
        .path_recovery(migrated_index)
        .expect("per-path recovery state")
        .bytes_in_flight();
    assert!(
        in_flight_after > in_flight_before,
        "the migrated path's own controller must carry the in-flight bytes \
         ({in_flight_before} -> {in_flight_after})"
    );
}

/// Bytes arriving from an address that is *not* a registered path never mint
/// path state — a spoofed source cannot grow the multipath table (§9.3.2).
#[test]
fn unknown_source_addresses_do_not_create_paths() {
    let (mut client, mut server) = make_pair();
    complete_handshake(&mut client, &mut server);

    let before = server.path_count();
    let now = Instant::now();
    let mut moved = Vec::new();
    let stream = client.open_bidi().expect("open bidi");
    client
        .send_stream(stream, b"spoofed", false)
        .expect("queue");
    client
        .poll_transmit(now, &mut moved)
        .expect("client datagram");
    // Deliver it claiming an address the connection knows nothing about, and
    // without any migration having been declared.
    server
        .handle_datagram_with_meta(now, &mut moved, DatagramMeta::from_peer(addr(59999)))
        .expect("datagram accepted");

    assert_eq!(
        server.path_count(),
        before,
        "an unregistered source address must not create a path"
    );
    assert!(
        server.multipath_state().path_by_addr(addr(59999)).is_none(),
        "no path state for a spoofable address"
    );
}
