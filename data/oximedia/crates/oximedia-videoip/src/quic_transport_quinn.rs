//! Real QUIC transport using the `quinn` library (feature `quic-quinn`).
//!
//! This module is compiled only when the `quic-quinn` feature is enabled.
//! The default build keeps the existing abstract `quic_transport` model and
//! remains 100 % Pure Rust without any C dependencies.
//!
//! ## Architecture
//!
//! ```text
//! QuicTransportConfig  →  QuicTransport::bind()  →  QuicTransport (endpoint)
//!                                                          │
//!                             QuicTransport::connect() ───┘
//!                                    │
//!                              QuicConnection  (quinn connection)
//!                                    │
//!              ┌─────────────────────┘
//!              │
//!   QuicConnection::send_datagram(Bytes) → Ok(())
//!   QuicConnection::recv_datagram()       → Ok(Bytes)
//! ```
//!
//! ## TLS / certificate handling
//!
//! `quinn` requires TLS.  Every `rustls::ClientConfig`/`ServerConfig` built by
//! this module is constructed via `builder_with_provider(Arc::new(provider))`
//! with an *explicit* `CryptoProvider` — never via the bare `builder()` /
//! process-wide `CryptoProvider::install_default()` — so there is no reliance
//! on ambient global state and no install-order footgun. Self-signed
//! certificates are generated via `rcgen`.
//!
//! For loopback tests a custom `NoCertificateVerification` acceptor is installed
//! so the client skips verification of the server's self-signed cert.
//!
//! ## Crypto policy — why `ring` is still used here (not `rustls-rustcrypto`)
//!
//! `quic-quinn` is intentionally **not** in `default = []`, and its explicit
//! provider is `rustls::crypto::ring::default_provider()` (`ring` is a
//! C/assembly dependency). Two independent things were checked, both currently
//! block using the workspace's Pure-Rust `rustls-rustcrypto` provider instead:
//!
//! 1. **Compile-time**: as of quinn 0.11.11 / quinn-proto 0.11.16 (latest on
//!    crates.io), `quinn` has no vendor-neutral "bring your own `CryptoProvider`"
//!    Cargo feature — `dep:rustls` itself is only pulled in by the `rustls-ring`
//!    or `rustls-aws-lc-rs` feature, both of which compile a C/assembly crypto
//!    backend unconditionally. There is no way to get `quinn` to depend on
//!    `rustls` without also compiling one of those.
//! 2. **Runtime**: even discounting (1), `rustls-rustcrypto` 0.0.2-alpha does
//!    not implement QUIC support at all — every `Tls13CipherSuite` it defines
//!    sets `quic: None` (see its `lib.rs`), which is rustls's documented way to
//!    "opt out of QUIC support for this suite". `quinn_proto::crypto::rustls::
//!    initial_suite_from_provider()` therefore finds no usable suite and
//!    `QuicServerConfig`/`QuicClientConfig::try_from()` fail with
//!    `NoInitialCipherSuite` ("no initial cipher suite found"). This was
//!    verified directly: swapping the provider below to
//!    `rustls_rustcrypto::provider()` makes
//!    `test_quic_transport_sends_and_receives` fail with exactly that error.
//!
//! So `ring` is compiled **and** actually executed by this opt-in feature.
//! Per the COOLJAPAN Pure Rust policy, C deps are acceptable when feature-gated
//! and documented, which is the case here: `quic-quinn` stays out of
//! `default = []` and all default features remain 100 % pure Rust (verified via
//! `cargo tree -p oximedia-videoip -e no-dev`, which shows zero `ring`/`quinn`
//! nodes without `--features quic-quinn`). Revisit this once either quinn gains
//! a vendor-neutral rustls feature or `rustls-rustcrypto` implements
//! `Tls13CipherSuite::quic`.

#![cfg(feature = "quic-quinn")]

use crate::error::{VideoIpError, VideoIpResult};
use bytes::Bytes;
use std::net::SocketAddr;
use std::sync::Arc;

// Re-export for external use.
pub use quinn::{Connection, Endpoint, RecvStream, SendStream};

// ---------------------------------------------------------------------------
// Certificate generation (rcgen)
// ---------------------------------------------------------------------------

/// Generate a self-signed TLS certificate and private key for `localhost`.
///
/// Returns `(cert_der: Vec<u8>, key_der: Vec<u8>)`.
fn generate_self_signed_cert() -> VideoIpResult<(Vec<u8>, Vec<u8>)> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).map_err(|e| {
        VideoIpError::Transport(format!("rcgen self-signed cert generation failed: {e}"))
    })?;

    let cert_der = cert.cert.der().to_vec();
    let key_der = cert.signing_key.serialize_der();
    Ok((cert_der, key_der))
}

// ---------------------------------------------------------------------------
// No-op certificate verifier for loopback tests
// ---------------------------------------------------------------------------

/// A `ServerCertVerifier` that accepts any certificate.  For **testing only**.
#[derive(Debug)]
struct NoCertVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        // Accept all common signature schemes — this verifier always returns Ok.
        // Must match the provider used to build the surrounding ClientConfig
        // in `QuicTransport::connect` (see module doc: `rustls-rustcrypto`
        // cannot be used here because it has no QUIC support).
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
            .iter()
            .copied()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// QuicTransportConfig (real, extends the abstract model)
// ---------------------------------------------------------------------------

/// Configuration for a real QUIC transport endpoint.
#[derive(Debug, Clone)]
pub struct QuicTransportConfig {
    /// Local bind address.
    pub bind_addr: SocketAddr,
    /// DER-encoded X.509 server certificate.
    pub cert_der: Vec<u8>,
    /// DER-encoded PKCS#8 private key.
    pub key_der: Vec<u8>,
    /// Maximum number of concurrently open bidirectional streams.
    pub max_streams: usize,
    /// Idle connection timeout (ms).
    pub idle_timeout_ms: u64,
}

impl QuicTransportConfig {
    /// Creates a `QuicTransportConfig` with a freshly generated self-signed
    /// certificate bound to `addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if certificate generation fails.
    pub fn with_self_signed_cert(addr: SocketAddr) -> VideoIpResult<Self> {
        let (cert_der, key_der) = generate_self_signed_cert()?;
        Ok(Self {
            bind_addr: addr,
            cert_der,
            key_der,
            max_streams: 100,
            idle_timeout_ms: 30_000,
        })
    }

    /// Creates a `QuicTransportConfig` from explicit PEM-encoded cert/key.
    ///
    /// # Errors
    ///
    /// Returns an error if PEM decoding fails.
    pub fn from_pem(addr: SocketAddr, cert_pem: &[u8], key_pem: &[u8]) -> VideoIpResult<Self> {
        use rustls_pemfile::{certs, pkcs8_private_keys};
        use std::io::BufReader;

        let cert_der = certs(&mut BufReader::new(cert_pem))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| VideoIpError::Transport(format!("cert PEM parse error: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| VideoIpError::Transport("no certificate in PEM".into()))?
            .to_vec();

        let key_der = pkcs8_private_keys(&mut BufReader::new(key_pem))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| VideoIpError::Transport(format!("key PEM parse error: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| VideoIpError::Transport("no private key in PEM".into()))?
            .secret_pkcs8_der()
            .to_vec();

        Ok(Self {
            bind_addr: addr,
            cert_der,
            key_der,
            max_streams: 100,
            idle_timeout_ms: 30_000,
        })
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any field is obviously invalid.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.cert_der.is_empty() {
            return Err(VideoIpError::InvalidVideoConfig(
                "cert_der must not be empty".into(),
            ));
        }
        if self.key_der.is_empty() {
            return Err(VideoIpError::InvalidVideoConfig(
                "key_der must not be empty".into(),
            ));
        }
        if self.max_streams == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "max_streams must be > 0".into(),
            ));
        }
        if self.idle_timeout_ms == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "idle_timeout_ms must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QuicTransport — server / client endpoint
// ---------------------------------------------------------------------------

/// A QUIC transport endpoint that can both accept incoming connections and
/// initiate outgoing ones.
pub struct QuicTransport {
    endpoint: Endpoint,
}

impl QuicTransport {
    /// Creates a server-capable QUIC endpoint bound to the configured address.
    ///
    /// # Errors
    ///
    /// Returns an error if binding or TLS setup fails.
    pub fn bind(cfg: &QuicTransportConfig) -> VideoIpResult<Self> {
        cfg.validate()?;

        let cert = rustls::pki_types::CertificateDer::from(cfg.cert_der.clone());
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(cfg.key_der.clone()),
        );

        // Explicit `ring` provider (see module doc: quinn/quinn-proto require
        // a QUIC-capable CryptoProvider, and `rustls-rustcrypto` does not
        // implement QUIC support). Passed explicitly rather than relying on
        // `CryptoProvider::install_default()` global process state.
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut tls = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| VideoIpError::Transport(format!("TLS provider setup failed: {e}")))?
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .map_err(|e| VideoIpError::Transport(format!("TLS cert load failed: {e}")))?;

        tls.alpn_protocols = vec![b"oximedia-videoip".to_vec()];

        let mut server_cfg = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(tls)
                .map_err(|e| VideoIpError::Transport(format!("quinn server TLS: {e}")))?,
        ));

        let transport = Self::default_transport(cfg);
        server_cfg.transport_config(Arc::new(transport));

        let endpoint = Endpoint::server(server_cfg, cfg.bind_addr)
            .map_err(|e| VideoIpError::Transport(format!("quinn Endpoint::server: {e}")))?;

        Ok(Self { endpoint })
    }

    /// Returns the local address of this endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if `local_addr` is unavailable.
    pub fn local_addr(&self) -> VideoIpResult<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| VideoIpError::Transport(e.to_string()))
    }

    /// Connects to a remote QUIC server, using a no-op cert verifier (for tests).
    ///
    /// `server_name` is typically `"localhost"` for loopback connections.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails.
    pub async fn connect(
        &self,
        remote: SocketAddr,
        server_name: &str,
    ) -> VideoIpResult<QuicConnection> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| VideoIpError::Transport(format!("TLS provider setup failed: {e}")))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertVerification))
            .with_no_client_auth();

        tls.alpn_protocols = vec![b"oximedia-videoip".to_vec()];

        let client_cfg = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(tls)
                .map_err(|e| VideoIpError::Transport(format!("quinn client TLS: {e}")))?,
        ));

        let conn = self
            .endpoint
            .connect_with(client_cfg, remote, server_name)
            .map_err(|e| VideoIpError::Transport(format!("quinn connect: {e}")))?
            .await
            .map_err(|e| VideoIpError::Transport(format!("quinn connect await: {e}")))?;

        Ok(QuicConnection { inner: conn })
    }

    /// Accepts an incoming connection.
    ///
    /// Returns `None` when the endpoint is shut down.
    ///
    /// # Errors
    ///
    /// Returns an error if the incoming connection handshake fails.
    pub async fn accept(&self) -> VideoIpResult<Option<QuicConnection>> {
        let incoming = match self.endpoint.accept().await {
            Some(i) => i,
            None => return Ok(None),
        };

        let conn = incoming
            .await
            .map_err(|e| VideoIpError::Transport(format!("quinn accept: {e}")))?;

        Ok(Some(QuicConnection { inner: conn }))
    }

    /// Closes the endpoint.
    pub fn close(&self) {
        self.endpoint.close(quinn::VarInt::from_u32(0), b"close");
    }

    // Build a quinn TransportConfig with sensible media-streaming defaults.
    fn default_transport(cfg: &QuicTransportConfig) -> quinn::TransportConfig {
        let mut t = quinn::TransportConfig::default();
        if let Ok(var) = quinn::VarInt::from_u64(cfg.idle_timeout_ms) {
            t.max_idle_timeout(Some(quinn::IdleTimeout::from(var)));
        }
        t.max_concurrent_bidi_streams(
            quinn::VarInt::from_u64(cfg.max_streams as u64).unwrap_or(quinn::VarInt::MAX),
        );
        t
    }
}

// ---------------------------------------------------------------------------
// QuicConnection — a single peer connection
// ---------------------------------------------------------------------------

/// A QUIC connection to a single remote peer.
pub struct QuicConnection {
    inner: Connection,
}

impl QuicConnection {
    /// Sends a datagram to the peer.
    ///
    /// QUIC datagrams are unreliable (like UDP) and bypass stream ordering.
    /// Prefer datagrams for time-sensitive media packets.
    ///
    /// # Errors
    ///
    /// Returns an error if datagram support is unavailable or the send fails.
    pub fn send_datagram(&self, data: Bytes) -> VideoIpResult<()> {
        self.inner
            .send_datagram(data)
            .map_err(|e| VideoIpError::Transport(format!("send_datagram: {e}")))
    }

    /// Receives a datagram from the peer.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or an I/O error occurs.
    pub async fn recv_datagram(&self) -> VideoIpResult<Bytes> {
        self.inner
            .read_datagram()
            .await
            .map_err(|e| VideoIpError::Transport(format!("recv_datagram: {e}")))
    }

    /// Returns the remote peer's address.
    #[must_use]
    pub fn remote_addr(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    /// Closes the connection.
    pub fn close(&self) {
        self.inner.close(quinn::VarInt::from_u32(0), b"bye");
    }

    /// Opens a new outgoing bidirectional stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed or too many streams are open.
    pub async fn open_bi(&self) -> VideoIpResult<(SendStream, RecvStream)> {
        self.inner
            .open_bi()
            .await
            .map_err(|e| VideoIpError::Transport(format!("open_bi: {e}")))
    }

    /// Accepts an incoming bidirectional stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is closed.
    pub async fn accept_bi(&self) -> VideoIpResult<(SendStream, RecvStream)> {
        self.inner
            .accept_bi()
            .await
            .map_err(|e| VideoIpError::Transport(format!("accept_bi: {e}")))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn any_localhost() -> SocketAddr {
        SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)
    }

    // ── Item 1 required tests ─────────────────────────────────────────────────

    /// Verify that a QuicTransport server and client can exchange datagrams.
    #[tokio::test]
    async fn test_quic_transport_sends_and_receives() {
        // No process-wide `CryptoProvider::install_default()` needed: both
        // `QuicTransport::bind` and `QuicTransport::connect` build their
        // rustls configs via `builder_with_provider(ring::default_provider())`
        // explicitly, so there is no reliance on global default-provider state.

        // Server endpoint.
        let server_cfg =
            QuicTransportConfig::with_self_signed_cert(any_localhost()).expect("server config");
        let server = QuicTransport::bind(&server_cfg).expect("server bind");
        let server_addr = server.local_addr().expect("server addr");

        // Client endpoint (no server cert needed; NoCertVerification used).
        let client_cfg =
            QuicTransportConfig::with_self_signed_cert(any_localhost()).expect("client config");
        let client = QuicTransport::bind(&client_cfg).expect("client bind");

        // Connect client → server concurrently.
        let (server_conn_res, client_conn_res) = tokio::join!(
            async { server.accept().await },
            client.connect(server_addr, "localhost"),
        );

        let server_conn = server_conn_res
            .expect("server accept")
            .expect("server conn should be Some");
        let client_conn = client_conn_res.expect("client connect");

        // Client sends a datagram.
        let payload = Bytes::from_static(b"oximedia-quic-test");
        client_conn
            .send_datagram(payload.clone())
            .expect("send_datagram");

        // Server receives it.
        let received = server_conn.recv_datagram().await.expect("recv_datagram");
        assert_eq!(received, payload);

        client_conn.close();
        server_conn.close();
        server.close();
    }

    /// Verify that QuicTransportConfig validation catches invalid inputs.
    #[test]
    fn test_quic_transport_config_validation() {
        // Empty cert should fail.
        let bad_cert = QuicTransportConfig {
            bind_addr: any_localhost(),
            cert_der: vec![],
            key_der: vec![1, 2, 3],
            max_streams: 10,
            idle_timeout_ms: 5000,
        };
        assert!(bad_cert.validate().is_err());

        // Empty key should fail.
        let bad_key = QuicTransportConfig {
            bind_addr: any_localhost(),
            cert_der: vec![1, 2, 3],
            key_der: vec![],
            max_streams: 10,
            idle_timeout_ms: 5000,
        };
        assert!(bad_key.validate().is_err());

        // Zero streams should fail.
        let bad_streams = QuicTransportConfig {
            bind_addr: any_localhost(),
            cert_der: vec![1],
            key_der: vec![1],
            max_streams: 0,
            idle_timeout_ms: 5000,
        };
        assert!(bad_streams.validate().is_err());

        // Valid self-signed config should pass.
        let good = QuicTransportConfig::with_self_signed_cert(any_localhost())
            .expect("with_self_signed_cert");
        assert!(good.validate().is_ok());
    }
}
