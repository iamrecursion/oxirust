//! Minimal HTTP/3 GET round trip, entirely self-contained.
//!
//! Spins up an in-process OxiQUIC + HTTP/3 server on an ephemeral loopback
//! port with a throwaway self-signed certificate, issues a single GET
//! against it with [`oxiquic::h3_prelude::H3Client`], and prints the
//! response. Requires the `h3` feature (`transport` is a default feature):
//!
//! ```text
//! cargo run -p oxiquic --example h3_get --features h3
//! ```
//!
//! A client talking to a real server would instead use
//! [`oxiquic::connect_h3`], which validates the server's certificate against
//! the system's WebPKI trust store rather than trusting a hand-rolled
//! [`RootCertStore`] as this self-contained demo does.

use std::net::SocketAddr;
use std::sync::Arc;

use oxiquic::h3_prelude::{H3Client, H3Response, H3Server};
use oxiquic::prelude::{ClientEndpoint, ServerEndpoint, TransportConfig};
use oxitls_rcgen::generate_self_signed_ed25519;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::version::TLS13;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

/// `tokio::spawn` requires its future's output to be `Send`; `main`'s error
/// type is fixed to this bound too so `?` composes across the `.await` on
/// the spawned server task below without an extra manual conversion.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let key = generate_self_signed_ed25519(&["localhost"])?;
    let cert_der = CertificateDer::from(key.cert_der.clone());
    let priv_key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.pkcs8_der.clone()));

    let provider = Arc::new(oxiquic_crypto::quic_crypto_provider());

    // Trust only this demo's throwaway cert -- a real client uses the
    // system's WebPKI roots (see `oxiquic::connect_h3`) instead of a
    // hand-built store.
    let mut roots = RootCertStore::empty();
    roots.add(cert_der.clone())?;
    let client_cfg: Arc<ClientConfig> = Arc::new(
        ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&TLS13])?
            .with_root_certificates(roots)
            .with_no_client_auth(),
    );
    let server_cfg: Arc<ServerConfig> = Arc::new(
        ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&TLS13])?
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], priv_key_der)?,
    );

    let loopback: SocketAddr = "127.0.0.1:0".parse()?;
    let server_ep = ServerEndpoint::bind(loopback, server_cfg, TransportConfig::default()).await?;
    let server_addr = server_ep.local_addr()?;

    // Server task: accept one QUIC connection, drive it as HTTP/3, respond
    // 200 to the first request.
    let server_task: tokio::task::JoinHandle<Result<(), BoxError>> = tokio::spawn(async move {
        let quic_conn = server_ep.accept().await?;
        let mut h3_server = H3Server::new(quic_conn.into_driven()).await?;
        let Some(ctx) = h3_server.accept().await? else {
            return Err("client closed the connection before sending a request".into());
        };
        let resp = H3Response::new(200).with_body("Hello from OxiQUIC H3!");
        ctx.respond(resp).await?;
        Ok(())
    });

    // Client: connect QUIC, drive it as HTTP/3, issue one GET.
    let client_ep = ClientEndpoint::bind(loopback, client_cfg, TransportConfig::default()).await?;
    let quic_conn = client_ep.connect(server_addr, "localhost").await?;
    let mut h3_client = H3Client::new(quic_conn.into_driven()).await?;
    let resp = h3_client.get("https://localhost/").await?;

    println!("h3_get: status = {}", resp.status());
    println!("h3_get: body   = {}", resp.body_text()?);
    assert!(resp.is_success(), "expected a 2xx response");

    h3_client.close().await?;
    server_task.await??;
    Ok(())
}
