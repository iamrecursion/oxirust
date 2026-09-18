//! Minimal QUIC echo server.
//!
//! Binds a [`ServerEndpoint`] on a loopback address (default
//! `127.0.0.1:4433`, override with the first CLI argument), generates an
//! ephemeral self-signed certificate, then accepts connections in a loop and
//! echoes back whatever bytes each client sends on the first stream it opens.
//!
//! Pair with the `quic_echo_client` example (run in a second terminal, or
//! point both at a matching address):
//!
//! ```text
//! cargo run -p oxiquic --example quic_echo_server --features dangerous
//! cargo run -p oxiquic --example quic_echo_client --features dangerous
//! ```
//!
//! The client uses [`oxiquic::connect_insecure`] to skip certificate
//! verification, since this demo's certificate is a throwaway generated on
//! every run rather than one issued by a CA in the client's trust store. A
//! real deployment should use [`oxiquic::listen`] with a CA-issued
//! certificate on the server side and plain [`oxiquic::connect`] on the
//! client side.

use std::net::SocketAddr;
use std::sync::Arc;

use oxiquic::prelude::{ServerEndpoint, TransportConfig};
use oxitls_rcgen::generate_self_signed_ed25519;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::version::TLS13;
use rustls::ServerConfig;

const DEFAULT_ADDR: &str = "127.0.0.1:4433";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ADDR.to_string())
        .parse()?;

    // Ephemeral self-signed cert -- fine for this demo; a real deployment
    // should load a certificate issued by a trusted CA instead.
    let key = generate_self_signed_ed25519(&["localhost"])?;
    let cert_der = CertificateDer::from(key.cert_der.clone());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.pkcs8_der.clone()));

    let provider = Arc::new(oxiquic_crypto::quic_crypto_provider());
    let server_cfg = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&TLS13])?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)?;

    let server =
        ServerEndpoint::bind(addr, Arc::new(server_cfg), TransportConfig::default()).await?;
    let local_addr = server.local_addr()?;
    println!(
        "quic_echo_server: listening on {local_addr} (self-signed cert; run \
         quic_echo_client against this address)"
    );

    loop {
        let mut conn = server.accept().await?;
        let peer = conn.peer_addr();
        tokio::spawn(async move {
            println!("quic_echo_server: accepted connection from {peer:?}");
            loop {
                let (stream_id, data, fin) = match conn.accept_uni_or_bidi_data().await {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("quic_echo_server: connection from {peer:?} ended: {err}");
                        return;
                    }
                };
                println!(
                    "quic_echo_server: echoing {} bytes on stream {stream_id:?} back to {peer:?}",
                    data.len()
                );
                if let Err(err) = conn.send(stream_id, &data, fin).await {
                    eprintln!("quic_echo_server: send to {peer:?} failed: {err}");
                    return;
                }
                if fin {
                    return;
                }
            }
        });
    }
}
