//! PlayReady license-acquisition RPC layer.
//!
//! Implements the WS-Trust 1.3 SOAP/XML protocol used by Microsoft PlayReady
//! license servers.  This module provides the network transport for delivering
//! a PlayReady license challenge (SOAP envelope wrapping a Base64-encoded
//! `RequestSecurityToken`) and parsing the SOAP response that carries the
//! license.
//!
//! Two clients ship out of the box:
//!
//! * [`HyperPlainPlayReadyClient`] — `hyper` 1.x over plain TCP. Suitable for
//!   in-process mock servers and `http://` endpoints during testing.
//! * [`HyperRustlsPlayReadyClient`] — Pure-Rust TLS via `tokio-rustls` +
//!   `rustls-rustcrypto`, gated behind the `playready-network` feature.
//!
//! Both implement the [`PlayReadyLicenseClient`] trait, which is also
//! implemented by `PlayReadyClient::acquire_license`.
//!
//! # Wire Format
//!
//! ```text
//! POST /playready/license HTTP/1.1
//! Content-Type: text/xml; charset=utf-8
//! SOAPAction: "http://schemas.microsoft.com/DRM/2007/03/protocols/AcquireLicense"
//! User-Agent: oximedia-drm-playready/0.1
//!
//! <?xml version="1.0"?>
//! <soap:Envelope …>
//!   <soap:Body>
//!     <AcquireLicense>
//!       <challenge><Challenge …><LA …/></Challenge></challenge>
//!     </AcquireLicense>
//!   </soap:Body>
//! </soap:Envelope>
//! ```
//!
//! The scope of this module is **license acquisition transport and SOAP
//! framing only** — the cryptographic XMR CDM layer remains a structural
//! placeholder in `playready.rs`.

use crate::DrmError;
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::client::conn::http1 as client_http1;
use hyper::{Method, Request, Uri};
use hyper_util::rt::TokioIo;
use std::time::Duration;
use tokio::net::TcpStream;

// ─────────────────────────────────────────────────────────────────────────────
// SOAP envelope constants
// ─────────────────────────────────────────────────────────────────────────────

/// The required `SOAPAction` header value for PlayReady license acquisition.
/// The value includes the literal embedded double-quotes as required by WS-Trust.
const SOAP_ACTION: &str = r#""http://schemas.microsoft.com/DRM/2007/03/protocols/AcquireLicense""#;

const SOAP_NS_ENVELOPE: &str = "http://schemas.xmlsoap.org/soap/envelope/";
const SOAP_NS_PLAYREADY: &str = "http://schemas.microsoft.com/DRM/2007/03/protocols";

// ─────────────────────────────────────────────────────────────────────────────
// PlayReadyLicenseClient trait
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous transport for posting a PlayReady license challenge to a remote
/// license server (or in-process mock) and returning the raw SOAP response.
///
/// Implementations are responsible for:
///
/// 1. Establishing the network connection (plain TCP or TLS).
/// 2. Performing an HTTP/1.1 `POST` with `Content-Type: text/xml` and the
///    required `SOAPAction` header.
/// 3. Returning the raw SOAP response body on 2xx.
///
/// On non-2xx HTTP responses, implementations should return
/// [`DrmError::LicenseDenied`].  Network / IO / parsing failures should be
/// mapped to [`DrmError::NetworkError`].
#[async_trait]
pub trait PlayReadyLicenseClient: Send + Sync {
    /// POST `soap_body` (a complete SOAP envelope XML string) to `server_url`
    /// and return the raw response body bytes.
    async fn fetch_license(
        &self,
        server_url: &str,
        soap_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// SOAP envelope builder
// ─────────────────────────────────────────────────────────────────────────────

/// Wrap a raw PlayReady challenge payload in a WS-Trust 1.3 SOAP envelope.
///
/// The `challenge_b64` parameter should be the Base64-encoded `rmheader` bytes
/// (the output of `PlayReadyLicenseChallenge::challenge`).
pub fn build_soap_envelope(challenge_b64: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="{SOAP_NS_ENVELOPE}" xmlns:pr="{SOAP_NS_PLAYREADY}">
  <soap:Body>
    <pr:AcquireLicense>
      <pr:challenge>
        <Challenge xmlns="http://schemas.microsoft.com/DRM/2007/03/protocols/messages">
          <LA xmlns="{SOAP_NS_PLAYREADY}" Id="SignedData">
            <CUSTOMDATA>{challenge_b64}</CUSTOMDATA>
          </LA>
        </Challenge>
      </pr:challenge>
    </pr:AcquireLicense>
  </soap:Body>
</soap:Envelope>
"#
    )
}

/// Extract the `<License>` element content from a PlayReady SOAP response.
///
/// Searches for the first occurrence of `<License` and `</License>` (or
/// `<License … />`) in the XML string and returns the inner bytes as a
/// Base64-decoded payload. Returns [`DrmError::XmlError`] if no `<License>`
/// element is found.
pub fn parse_soap_response(response_body: &[u8]) -> Result<Vec<u8>, DrmError> {
    let xml = std::str::from_utf8(response_body).map_err(|e| {
        DrmError::XmlError(format!("PlayReady SOAP response is not valid UTF-8: {e}"))
    })?;

    // Locate the License element (naïve but sufficient for the transport layer).
    let start_tag = xml.find("<License").ok_or_else(|| {
        DrmError::XmlError("PlayReady SOAP response missing <License> element".to_string())
    })?;

    // Find the inner content between the opening and closing tags.
    let after_start = xml[start_tag..]
        .find('>')
        .ok_or_else(|| DrmError::XmlError("malformed <License> opening tag".to_string()))?;
    let inner_start = start_tag + after_start + 1;

    let end_tag = xml.find("</License>").ok_or_else(|| {
        DrmError::XmlError("PlayReady SOAP response missing </License> element".to_string())
    })?;

    if end_tag < inner_start {
        return Err(DrmError::XmlError(
            "malformed PlayReady SOAP response: </License> before end of opening tag".to_string(),
        ));
    }

    let b64_content = xml[inner_start..end_tag].trim();
    STANDARD
        .decode(b64_content)
        .map_err(|e| DrmError::XmlError(format!("PlayReady license Base64 decode failed: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared URL parser (mirrors widevine_rpc)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsedEndpoint {
    authority: String,
    host: String,
    path_and_query: String,
    scheme: String,
}

fn parse_endpoint(server_url: &str) -> Result<ParsedEndpoint, DrmError> {
    let uri: Uri = server_url.parse().map_err(|e| {
        DrmError::NetworkError(format!("invalid PlayReady server URL `{server_url}`: {e}"))
    })?;

    let scheme = uri
        .scheme_str()
        .ok_or_else(|| {
            DrmError::NetworkError(format!(
                "PlayReady server URL `{server_url}` is missing a scheme"
            ))
        })?
        .to_ascii_lowercase();

    let host = uri
        .host()
        .ok_or_else(|| {
            DrmError::NetworkError(format!(
                "PlayReady server URL `{server_url}` is missing a host"
            ))
        })?
        .to_owned();

    let port = uri.port_u16().unwrap_or(match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        other => {
            return Err(DrmError::NetworkError(format!(
                "unsupported scheme `{other}` in PlayReady server URL"
            )));
        }
    });

    let mut path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_owned())
        .unwrap_or_else(|| "/".to_owned());
    if path_and_query.is_empty() {
        path_and_query.push('/');
    }

    Ok(ParsedEndpoint {
        authority: format!("{host}:{port}"),
        host,
        path_and_query,
        scheme,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Common HTTP helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_soap_post_request(
    endpoint: &ParsedEndpoint,
    body: &[u8],
    extra_headers: &[(String, String)],
) -> Result<Request<Full<Bytes>>, DrmError> {
    let host_header = if (endpoint.scheme == "http" && endpoint.authority.ends_with(":80"))
        || (endpoint.scheme == "https" && endpoint.authority.ends_with(":443"))
    {
        endpoint.host.clone()
    } else {
        endpoint.authority.clone()
    };

    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(&endpoint.path_and_query)
        .header("Host", host_header)
        .header("Content-Type", "text/xml; charset=utf-8")
        .header("SOAPAction", SOAP_ACTION)
        .header("Content-Length", body.len().to_string())
        .header("User-Agent", "oximedia-drm-playready/0.1");

    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }

    builder
        .body(Full::new(Bytes::copy_from_slice(body)))
        .map_err(|e| DrmError::NetworkError(format!("failed to build PlayReady HTTP request: {e}")))
}

async fn send_and_collect<S>(
    io: TokioIo<S>,
    req: Request<Full<Bytes>>,
    timeout: Duration,
) -> Result<Vec<u8>, DrmError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let handshake = tokio::time::timeout(timeout, client_http1::handshake(io))
        .await
        .map_err(|_| DrmError::NetworkError("PlayReady HTTP handshake timed out".to_string()))?
        .map_err(|e| DrmError::NetworkError(format!("PlayReady HTTP/1.1 handshake failed: {e}")))?;

    let (mut sender, conn) = handshake;

    let conn_handle = tokio::spawn(async move {
        let _ = conn.await;
    });

    let send_fut = sender.send_request(req);
    let resp_result = tokio::time::timeout(timeout, send_fut).await;
    let resp = match resp_result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(format!(
                "failed to send PlayReady license request: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "PlayReady license request timed out before response headers".to_string(),
            ));
        }
    };

    let status = resp.status();

    let body_fut = resp.collect();
    let body_bytes = match tokio::time::timeout(timeout, body_fut).await {
        Ok(Ok(b)) => b.to_bytes(),
        Ok(Err(e)) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(format!(
                "failed to read PlayReady response body: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "PlayReady response body read timed out".to_string(),
            ));
        }
    };

    drop(conn_handle);

    if status.is_success() {
        Ok(body_bytes.to_vec())
    } else {
        let body_text = String::from_utf8_lossy(&body_bytes).into_owned();
        Err(DrmError::LicenseDenied {
            status: status.as_u16(),
            body: body_text,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HyperPlainPlayReadyClient
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-HTTP [`PlayReadyLicenseClient`] built on `hyper` 1.x.
///
/// Intended for **test mock servers** and never negotiates TLS.
/// For production traffic, use [`HyperRustlsPlayReadyClient`] (gated behind
/// the `playready-network` feature).
#[derive(Debug, Clone)]
pub struct HyperPlainPlayReadyClient {
    timeout_ms: u32,
}

impl HyperPlainPlayReadyClient {
    /// Create a plain-HTTP PlayReady client with the default 30s timeout.
    pub fn new() -> Self {
        Self { timeout_ms: 30_000 }
    }

    /// Override the per-phase timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, ms: u32) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Returns the configured per-phase timeout in milliseconds.
    pub fn timeout_ms(&self) -> u32 {
        self.timeout_ms
    }
}

impl Default for HyperPlainPlayReadyClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlayReadyLicenseClient for HyperPlainPlayReadyClient {
    async fn fetch_license(
        &self,
        server_url: &str,
        soap_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError> {
        let endpoint = parse_endpoint(server_url)?;
        if endpoint.scheme != "http" {
            return Err(DrmError::NetworkError(format!(
                "HyperPlainPlayReadyClient only supports http:// URLs, got scheme `{}`",
                endpoint.scheme
            )));
        }

        let timeout = Duration::from_millis(u64::from(self.timeout_ms));

        let tcp = tokio::time::timeout(timeout, TcpStream::connect(&endpoint.authority))
            .await
            .map_err(|_| {
                DrmError::NetworkError(format!(
                    "TCP connect to `{}` timed out after {}ms",
                    endpoint.authority, self.timeout_ms
                ))
            })?
            .map_err(|e| {
                DrmError::NetworkError(format!(
                    "TCP connect to `{}` failed: {e}",
                    endpoint.authority
                ))
            })?;

        let req = build_soap_post_request(&endpoint, soap_body, headers)?;
        let io = TokioIo::new(tcp);
        send_and_collect(io, req, timeout).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HyperRustlsPlayReadyClient — TLS transport (playready-network feature)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "playready-network")]
mod tls {
    use super::{
        build_soap_post_request, parse_endpoint, send_and_collect, Bytes, Duration, Full,
        PlayReadyLicenseClient, Request,
    };
    use crate::DrmError;
    use async_trait::async_trait;
    use hyper_util::rt::TokioIo;
    use rustls::pki_types::ServerName;
    use std::sync::Arc;
    use tokio::net::TcpStream;
    use tokio_rustls::TlsConnector;

    /// Production-grade TLS [`PlayReadyLicenseClient`] built on `tokio-rustls`
    /// with the `rustls-rustcrypto` provider (Pure-Rust TLS — no openssl,
    /// no ring).
    pub struct HyperRustlsPlayReadyClient {
        timeout_ms: u32,
        roots: Option<Arc<rustls::RootCertStore>>,
    }

    impl HyperRustlsPlayReadyClient {
        /// Create a TLS PlayReady client with the default 30s timeout.
        pub fn new() -> Self {
            Self {
                timeout_ms: 30_000,
                roots: None,
            }
        }

        /// Override the per-phase timeout in milliseconds.
        #[must_use]
        pub fn with_timeout_ms(mut self, ms: u32) -> Self {
            self.timeout_ms = ms;
            self
        }

        /// Install the trust anchors for certificate verification.
        #[must_use]
        pub fn with_root_certs(mut self, roots: Arc<rustls::RootCertStore>) -> Self {
            self.roots = Some(roots);
            self
        }

        /// Returns the configured per-phase timeout in milliseconds.
        pub fn timeout_ms(&self) -> u32 {
            self.timeout_ms
        }

        fn build_client_config(&self) -> Result<rustls::ClientConfig, DrmError> {
            let roots = self.roots.clone().ok_or_else(|| {
                DrmError::NetworkError(
                    "HyperRustlsPlayReadyClient requires explicit root certificates".to_string(),
                )
            })?;

            let provider = rustls_rustcrypto::provider();
            let config = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
                .with_safe_default_protocol_versions()
                .map_err(|e| {
                    DrmError::NetworkError(format!("rustls protocol version setup failed: {e}"))
                })?
                .with_root_certificates((*roots).clone())
                .with_no_client_auth();
            Ok(config)
        }
    }

    impl Default for HyperRustlsPlayReadyClient {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl PlayReadyLicenseClient for HyperRustlsPlayReadyClient {
        async fn fetch_license(
            &self,
            server_url: &str,
            soap_body: &[u8],
            headers: &[(String, String)],
        ) -> Result<Vec<u8>, DrmError> {
            let endpoint = parse_endpoint(server_url)?;
            if endpoint.scheme != "https" {
                return Err(DrmError::NetworkError(format!(
                    "HyperRustlsPlayReadyClient requires https:// URLs, got scheme `{}`",
                    endpoint.scheme
                )));
            }

            let timeout = Duration::from_millis(u64::from(self.timeout_ms));

            let tcp = tokio::time::timeout(timeout, TcpStream::connect(&endpoint.authority))
                .await
                .map_err(|_| {
                    DrmError::NetworkError(format!(
                        "TCP connect to `{}` timed out after {}ms",
                        endpoint.authority, self.timeout_ms
                    ))
                })?
                .map_err(|e| {
                    DrmError::NetworkError(format!(
                        "TCP connect to `{}` failed: {e}",
                        endpoint.authority
                    ))
                })?;

            let config = self.build_client_config()?;
            let connector = TlsConnector::from(Arc::new(config));

            let server_name = ServerName::try_from(endpoint.host.clone()).map_err(|e| {
                DrmError::NetworkError(format!("invalid SNI host `{}`: {e}", endpoint.host))
            })?;

            let tls = tokio::time::timeout(timeout, connector.connect(server_name, tcp))
                .await
                .map_err(|_| {
                    DrmError::NetworkError(format!(
                        "TLS handshake to `{}` timed out after {}ms",
                        endpoint.authority, self.timeout_ms
                    ))
                })?
                .map_err(|e| DrmError::NetworkError(format!("TLS handshake failed: {e}")))?;

            let req: Request<Full<Bytes>> = build_soap_post_request(&endpoint, soap_body, headers)?;
            let io = TokioIo::new(tls);
            send_and_collect(io, req, timeout).await
        }
    }
}

#[cfg(feature = "playready-network")]
pub use tls::HyperRustlsPlayReadyClient;

// ─────────────────────────────────────────────────────────────────────────────
// PlayReadyClient — top-level API
// ─────────────────────────────────────────────────────────────────────────────

use crate::playready::{PlayReadyLicense, PlayReadyLicenseChallenge};

/// High-level PlayReady license acquisition client.
///
/// Wraps a [`PlayReadyLicenseChallenge`] and a session ID, builds the SOAP
/// envelope, dispatches it via the provided [`PlayReadyLicenseClient`]
/// transport, and parses the returned license.
pub struct PlayReadyClient {
    session_id: Vec<u8>,
    challenge: PlayReadyLicenseChallenge,
}

impl PlayReadyClient {
    /// Create a new PlayReady client for the given session and challenge.
    pub fn new(session_id: Vec<u8>, challenge: PlayReadyLicenseChallenge) -> Self {
        Self {
            session_id,
            challenge,
        }
    }

    /// Returns the session identifier.
    pub fn session_id(&self) -> &[u8] {
        &self.session_id
    }

    /// Acquire a PlayReady license from the given server URL using the supplied
    /// transport client.
    ///
    /// # Flow
    ///
    /// 1. Encode the challenge payload as Base64.
    /// 2. Build a WS-Trust 1.3 SOAP envelope.
    /// 3. POST the envelope to `server_url`.
    /// 4. Parse the `<License>` element from the SOAP response.
    /// 5. Return a [`PlayReadyLicense`] containing the raw license bytes.
    pub async fn acquire_license<C>(
        &self,
        server_url: &str,
        client: &C,
        extra_headers: &[(String, String)],
    ) -> Result<PlayReadyLicense, DrmError>
    where
        C: PlayReadyLicenseClient + ?Sized,
    {
        let challenge_bytes = self.challenge.get_challenge()?;
        let challenge_b64 = STANDARD.encode(&challenge_bytes);
        let envelope = build_soap_envelope(&challenge_b64);

        let response_body = client
            .fetch_license(server_url, envelope.as_bytes(), extra_headers)
            .await?;

        let license_bytes = parse_soap_response(&response_body)?;
        Ok(PlayReadyLicense::new(license_bytes))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no network)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_soap_envelope_contains_required_elements() {
        let env = build_soap_envelope("BASE64CHALLENGE==");
        assert!(env.contains("soap:Envelope"), "must have soap:Envelope");
        assert!(env.contains("soap:Body"), "must have soap:Body");
        assert!(env.contains("AcquireLicense"), "must have AcquireLicense");
        assert!(env.contains("BASE64CHALLENGE=="), "must embed challenge");
    }

    #[test]
    fn test_soap_envelope_parse_roundtrip() {
        let challenge_data = b"hello playready";
        let b64 = STANDARD.encode(challenge_data);
        let envelope = build_soap_envelope(&b64);

        // The envelope is valid XML — verify it contains the expected namespace
        assert!(
            envelope.contains(SOAP_NS_PLAYREADY),
            "envelope must declare PlayReady namespace"
        );
        assert!(
            envelope.contains(SOAP_NS_ENVELOPE),
            "envelope must declare SOAP envelope namespace"
        );
    }

    #[test]
    fn test_parse_soap_response_ok() {
        let license_bytes = b"license-binary-data";
        let b64 = STANDARD.encode(license_bytes);
        let response = format!(
            r#"<?xml version="1.0"?><soap:Envelope><soap:Body><License>{b64}</License></soap:Body></soap:Envelope>"#
        );

        let parsed = parse_soap_response(response.as_bytes()).expect("must parse");
        assert_eq!(parsed, license_bytes.to_vec());
    }

    #[test]
    fn test_parse_soap_response_rejects_missing_license_element() {
        let response = br#"<?xml version="1.0"?><soap:Envelope><soap:Body><Error>denied</Error></soap:Body></soap:Envelope>"#;
        let err = parse_soap_response(response).expect_err("must fail without <License>");
        match err {
            DrmError::XmlError(msg) => {
                assert!(
                    msg.contains("License"),
                    "error must mention License element"
                );
            }
            other => panic!("expected XmlError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_soap_response_rejects_malformed_base64() {
        let response = br#"<soap:Body><License>NOT_VALID_BASE64_!!!!</License></soap:Body>"#;
        let err = parse_soap_response(response).expect_err("must fail on bad base64");
        match err {
            DrmError::XmlError(_) => {}
            other => panic!("expected XmlError, got {other:?}"),
        }
    }

    #[test]
    fn test_soap_action_header_value_has_embedded_quotes() {
        // The SOAPAction value for PlayReady MUST include literal double-quote
        // characters as per WS-I Basic Profile requirements.
        assert!(
            SOAP_ACTION.starts_with('"'),
            "SOAPAction must start with a literal double-quote"
        );
        assert!(
            SOAP_ACTION.ends_with('"'),
            "SOAPAction must end with a literal double-quote"
        );
    }

    #[test]
    fn test_parse_endpoint_http() {
        let p = parse_endpoint("http://license.example.com/playready").expect("parse");
        assert_eq!(p.scheme, "http");
        assert_eq!(p.host, "license.example.com");
        assert_eq!(p.authority, "license.example.com:80");
        assert_eq!(p.path_and_query, "/playready");
    }

    #[test]
    fn test_parse_endpoint_https() {
        let p = parse_endpoint("https://license.example.com/playready").expect("parse");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.authority, "license.example.com:443");
    }

    #[test]
    fn test_parse_endpoint_explicit_port() {
        let p = parse_endpoint("http://127.0.0.1:9000/pr?tok=1").expect("parse");
        assert_eq!(p.authority, "127.0.0.1:9000");
        assert_eq!(p.path_and_query, "/pr?tok=1");
    }
}
