//! Retry connection-ID transcript (RFC 9000 §7.3, §17.2.5).
//!
//! The Retry integrity tag is keyed only by the Original Destination Connection
//! ID, which travels in the clear in the client's first Initial packet. Any
//! attacker who can observe (or guess) that packet can therefore forge a
//! syntactically perfect Retry. What actually protects the exchange is §7.3:
//! the genuine server echoes the connection-ID transcript inside the
//! TLS-authenticated transport parameters —
//! `original_destination_connection_id`, `retry_source_connection_id` and
//! `initial_source_connection_id` — and the client checks all three.
//!
//! These tests drive the synchronous [`Connection`] directly so the forged
//! Retry can be injected byte-for-byte.

use std::sync::Arc;
use std::time::{Duration, Instant};

use oxiquic_core::ConnectionId;
use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::connection::{ConnectionState, RetryTranscript};
use oxiquic_transport::packet::encode_retry_packet;
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

fn new_client(client_cfg: Arc<ClientConfig>) -> Connection {
    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    Connection::new_client(
        client_cfg,
        ServerName::try_from("localhost").expect("server name"),
        addr(4433),
        params,
        Default::default(),
        Default::default(),
    )
    .expect("client conn")
}

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

/// Drive a client that has already processed a Retry against `server` until
/// quiescent.
fn run_to_quiescence(client: &mut Connection, server: &mut Connection) {
    let now = Instant::now();
    for _ in 0..20 {
        exchange_all(client, server, now);
    }
}

/// The complete "client sends Initial, receives a Retry, retransmits" prelude.
///
/// Returns the client (post-Retry), the DCID of its *first* Initial (the
/// ODCID), the Retry's SCID and the client's own SCID.
fn client_after_retry(
    client_cfg: Arc<ClientConfig>,
    retry_scid: &[u8],
) -> (Connection, Vec<u8>, Vec<u8>) {
    let mut client = new_client(client_cfg);
    let now = Instant::now();
    let mut first = Vec::new();
    client
        .poll_transmit(now, &mut first)
        .expect("client initial");
    let (odcid, client_scid) = parse_initial_cids(&first).expect("parse initial cids");

    // Anyone who saw the first Initial can build this packet: the integrity tag
    // is keyed by the ODCID, which is right there on the wire.
    let retry = encode_retry_packet(retry_scid, &client_scid, &odcid, b"forged-or-real-token")
        .expect("encode retry");
    let mut retry_datagram = retry.clone();
    client
        .handle_datagram(now, &mut retry_datagram)
        .expect("client processes retry");

    (client, odcid, client_scid)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// A legitimate Retry: the server rebuilds the connection with the transcript
/// recovered from the token, the client's §7.3 checks all pass, and the
/// handshake completes.
#[test]
fn genuine_retry_transcript_completes_the_handshake() {
    let (client_cfg, server_cfg) = config_pair();
    let retry_scid = vec![0x51u8; 8];
    let (mut client, odcid, client_scid) = client_after_retry(client_cfg, &retry_scid);

    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    let mut server = Connection::new_server_after_retry(
        server_cfg,
        // After a Retry the Initial keys are seeded by the Retry SCID.
        ConnectionId::new(retry_scid.clone()),
        ConnectionId::new(client_scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
        RetryTranscript {
            original_dcid: ConnectionId::new(odcid),
            retry_scid: ConnectionId::new(retry_scid),
        },
    )
    .expect("server conn after retry");

    run_to_quiescence(&mut client, &mut server);

    assert!(
        client.is_established(),
        "the client must accept a matching transcript"
    );
    assert_ne!(
        client.state(),
        ConnectionState::Closed,
        "no transport-parameter error for a genuine Retry"
    );
    assert!(server.is_established(), "the server completes too");
}

/// An **injected** Retry: the attacker forges a Retry the client accepts, but
/// the genuine server never sent one, so it omits `retry_source_connection_id`.
/// RFC 9000 §7.3 makes that fatal — the client must close the connection rather
/// than complete a handshake whose connection IDs an attacker chose.
#[test]
fn forged_retry_is_detected_via_missing_retry_source_connection_id() {
    let (client_cfg, server_cfg) = config_pair();
    let forged_scid = vec![0xf0u8; 8];
    let (mut client, _odcid, client_scid) = client_after_retry(client_cfg, &forged_scid);

    // The genuine server sees only the client's second Initial and has no idea
    // a Retry ever happened, so it builds an ordinary connection.
    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    let mut server = Connection::new_server(
        server_cfg,
        ConnectionId::new(forged_scid),
        ConnectionId::new(client_scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
    )
    .expect("server conn");

    run_to_quiescence(&mut client, &mut server);

    assert_eq!(
        client.state(),
        ConnectionState::Closed,
        "a Retry the server never sent must fail the connection"
    );
}

/// A server whose `original_destination_connection_id` does not match the DCID
/// of the client's first Initial is rejected: this is the case where an
/// attacker replaced the ODCID, so the transcript no longer describes the
/// handshake the client actually performed.
#[test]
fn mismatched_original_destination_connection_id_fails_the_connection() {
    let (client_cfg, server_cfg) = config_pair();
    let retry_scid = vec![0x51u8; 8];
    let (mut client, _odcid, client_scid) = client_after_retry(client_cfg, &retry_scid);

    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    let mut server = Connection::new_server_after_retry(
        server_cfg,
        ConnectionId::new(retry_scid.clone()),
        ConnectionId::new(client_scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
        RetryTranscript {
            // Deliberately wrong: not the DCID of the client's first Initial.
            original_dcid: ConnectionId::new(vec![0x99u8; 8]),
            retry_scid: ConnectionId::new(retry_scid),
        },
    )
    .expect("server conn after retry");

    run_to_quiescence(&mut client, &mut server);

    assert_eq!(
        client.state(),
        ConnectionState::Closed,
        "a mismatched original_destination_connection_id must fail the connection"
    );
}

/// The `retry_source_connection_id` must also *match*, not merely be present:
/// a server that reports a different Retry SCID than the one the client saw is
/// rejected.
#[test]
fn mismatched_retry_source_connection_id_fails_the_connection() {
    let (client_cfg, server_cfg) = config_pair();
    let retry_scid = vec![0x51u8; 8];
    let (mut client, odcid, client_scid) = client_after_retry(client_cfg, &retry_scid);

    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    let mut server = Connection::new_server_after_retry(
        server_cfg,
        ConnectionId::new(retry_scid),
        ConnectionId::new(client_scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
        RetryTranscript {
            original_dcid: ConnectionId::new(odcid),
            // Not the SCID of the Retry the client processed.
            retry_scid: ConnectionId::new(vec![0x77u8; 8]),
        },
    )
    .expect("server conn after retry");

    run_to_quiescence(&mut client, &mut server);

    assert_eq!(
        client.state(),
        ConnectionState::Closed,
        "a mismatched retry_source_connection_id must fail the connection"
    );
}

/// Without any Retry, the ordinary handshake must still validate the transcript
/// and succeed — the new checks must not break the common path.
#[test]
fn plain_handshake_transcript_is_accepted() {
    let (client_cfg, server_cfg) = config_pair();
    let mut client = new_client(client_cfg);

    let now = Instant::now();
    let mut first = Vec::new();
    client
        .poll_transmit(now, &mut first)
        .expect("client initial");
    let (dcid, scid) = parse_initial_cids(&first).expect("parse initial cids");

    let params = TransportConfig::default()
        .idle_timeout(Duration::from_secs(30))
        .to_transport_params();
    let mut server = Connection::new_server(
        server_cfg,
        ConnectionId::new(dcid),
        ConnectionId::new(scid),
        addr(40000),
        params,
        Default::default(),
        Default::default(),
    )
    .expect("server conn");
    let mut owned = first.clone();
    server
        .handle_datagram(now, &mut owned)
        .expect("server first datagram");

    run_to_quiescence(&mut client, &mut server);

    assert!(client.is_established());
    assert!(server.is_established());
    assert_ne!(client.state(), ConnectionState::Closed);
    assert_ne!(server.state(), ConnectionState::Closed);
}
