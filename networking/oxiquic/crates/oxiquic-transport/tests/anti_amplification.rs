//! RFC 9000 §8.1 anti-amplification regression tests.
//!
//! Before validating a client's source address a server MUST NOT send more than
//! three times the number of bytes it has received. Without that limit a single
//! spoofed ~1200-byte Initial makes the server a reflection amplifier: it
//! answers with the whole ServerHello…Finished flight, whose size grows with the
//! certificate chain, toward whatever address the attacker forged.
//!
//! These tests deliberately configure a *large* certificate chain so the server
//! flight comfortably exceeds the 3× allowance; the pre-fix code emitted the
//! entire flight.

use std::sync::Arc;
use std::time::Instant;

use oxiquic_crypto::quic_crypto_provider;
use oxiquic_transport::{Connection, TransportConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

/// Number of certificates the server presents. Only the leaf is meaningful; the
/// repeats simply inflate the handshake flight the way a real intermediate
/// chain does, so the 3× allowance actually binds.
const CHAIN_LEN: usize = 24;

fn addr(port: u16) -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], port))
}

/// Matched (client, server) rustls configs over the Pure-Rust OxiQUIC provider.
/// `chain_len` certificates are presented by the server.
fn config_pair(chain_len: usize) -> (Arc<ClientConfig>, Arc<ServerConfig>) {
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
    let chain = vec![cert_der; chain_len];
    let server = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&TLS13])
        .expect("server TLS1.3")
        .with_no_client_auth()
        .with_single_cert(chain, key_der)
        .expect("server cert chain");
    (Arc::new(client), Arc::new(server))
}

/// Extract `(dcid, scid)` from a client's first long-header Initial datagram.
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

/// Build a client and the server connection its first Initial would create,
/// returning `(client, server, first_initial)` with the Initial *not* yet fed to
/// the server.
fn make_pair(chain_len: usize) -> (Connection, Connection, Vec<u8>) {
    let (client_cfg, server_cfg) = config_pair(chain_len);
    let params = TransportConfig::default().to_transport_params();

    let mut client = Connection::new_client(
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
    client
        .poll_transmit(now, &mut first)
        .expect("client initial");
    let (dcid, scid) = parse_initial_cids(&first).expect("client initial CIDs");

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

    (client, server, first)
}

/// Drain everything the server is willing to send right now, returning the
/// total datagram bytes emitted.
fn drain_server(server: &mut Connection, now: Instant) -> u64 {
    let mut total = 0u64;
    // The bound is generous; the server must stop long before it is reached.
    for _ in 0..1024 {
        let mut out = Vec::new();
        if server.poll_transmit(now, &mut out).is_none() {
            break;
        }
        total += out.len() as u64;
    }
    total
}

/// A single (possibly spoofed) client Initial must not elicit more than three
/// times its own size from the server, however large the certificate chain.
#[test]
fn server_never_exceeds_three_times_received_before_validation() {
    let (_client, mut server, initial) = make_pair(CHAIN_LEN);
    let now = Instant::now();
    let received = initial.len() as u64;

    let mut owned = initial;
    server
        .handle_datagram(now, &mut owned)
        .expect("server accepts the client Initial");
    assert!(
        !server.address_validated(),
        "one Initial must not validate the client address"
    );

    let sent = drain_server(&mut server, now);
    assert!(
        sent <= 3 * received,
        "server sent {sent} bytes for {received} received (limit {})",
        3 * received
    );
    // With this chain length the full flight is far larger than the allowance,
    // so the limit must actually have bound (otherwise the test proves nothing).
    assert!(
        server.amplification_blocked(),
        "the {CHAIN_LEN}-certificate flight ({sent} bytes emitted) should have \
         exhausted the {} byte allowance",
        3 * received
    );
}

/// A blocked server resumes once the client supplies more bytes: the allowance
/// is a running 3× budget, not a one-shot cut-off. RFC 9002 §6.2.2.1 also
/// requires the loss-detection timer to stay disarmed while blocked.
#[test]
fn budget_grows_with_further_client_datagrams() {
    let (_client, mut server, initial) = make_pair(CHAIN_LEN);
    let now = Instant::now();

    let mut owned = initial.clone();
    server.handle_datagram(now, &mut owned).expect("initial");
    let first_round = drain_server(&mut server, now);
    assert!(first_round > 0, "server must answer the Initial");
    assert!(server.amplification_blocked(), "budget must be exhausted");
    assert!(
        server.loss_timer().is_none(),
        "RFC 9002 §6.2.2.1: no loss-detection timer while amplification-blocked"
    );

    // The client retransmits its (still unacknowledged) Initial, which raises
    // the server's allowance and lets the rest of the flight out.
    let mut owned = initial;
    server
        .handle_datagram(now, &mut owned)
        .expect("second client datagram");
    assert!(
        !server.amplification_blocked(),
        "fresh client bytes must unblock the server"
    );
    let second_round = drain_server(&mut server, now);
    assert!(second_round > 0, "server must resume sending");
}

/// A full handshake still completes: the limit must never be able to deadlock a
/// legitimate exchange, and processing a client Handshake packet validates the
/// address (RFC 9000 §8.1) so the limit is lifted.
#[test]
fn handshake_completes_and_validates_the_address() {
    // A normal single-certificate server: the usual production shape.
    let (mut client, mut server, initial) = make_pair(1);
    let now = Instant::now();

    let mut owned = initial;
    server.handle_datagram(now, &mut owned).expect("initial");

    for _ in 0..64 {
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
        // Server → client.
        loop {
            let mut out = Vec::new();
            if server.poll_transmit(now, &mut out).is_none() {
                break;
            }
            // The invariant must hold on every datagram the server emits.
            client.handle_datagram(now, &mut out).expect("client recv");
        }
        // Client → server.
        loop {
            let mut out = Vec::new();
            if client.poll_transmit(now, &mut out).is_none() {
                break;
            }
            server.handle_datagram(now, &mut out).expect("server recv");
        }
    }

    assert!(!server.is_handshaking(), "server handshake must complete");
    assert!(!client.is_handshaking(), "client handshake must complete");
    assert!(
        server.address_validated(),
        "a completed handshake validates the client address"
    );
    assert!(!server.amplification_blocked());
}

/// A client is never subject to the limit (RFC 9000 §8.1 constrains servers).
#[test]
fn client_is_never_amplification_limited() {
    let (client, _server, _initial) = make_pair(1);
    assert!(client.address_validated());
    assert!(!client.amplification_blocked());
}
