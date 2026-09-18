//! FairPlay Streaming KSM (Key Security Module) transport layer.
//!
//! Implements the JSON-over-HTTPS protocol used by Apple FairPlay Streaming
//! Key Management Servers (KSM).  The client POSTs a JSON object containing
//! the Base64-encoded SPC (Server Playback Context) and receives a Base64-
//! encoded CKC (Content Key Context) in response.
//!
//! Two clients ship out of the box:
//!
//! * [`HyperPlainFairPlayClient`] — `hyper` 1.x over plain TCP. Suitable for
//!   in-process mock servers and `http://` endpoints during testing.
//! * [`HyperRustlsFairPlayClient`] — Pure-Rust TLS via `tokio-rustls` +
//!   `rustls-rustcrypto`, gated behind the `fairplay-network` feature.
//!
//! Both implement the [`FairPlayKeyClient`] trait, which `FairPlayClient::
//! request_key_from_server` dispatches through.
//!
//! # Wire Format
//!
//! ```text
//! POST /fps/key HTTP/1.1
//! Content-Type: application/json
//! User-Agent: oximedia-drm-fairplay/0.1
//!
//! { "asset_id": "…", "spc": "<base64 SPC>", "certificate": "<base64 cert>" }
//!
//! ← 200 OK
//! { "ckc": "<base64 CKC>" }
//! ```
//!
//! The scope of this module is **KSM HTTP transport only** — cryptographic
//! SPC/CKC binary layout remains a structural placeholder in `fairplay.rs`.

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
// FairPlayKeyClient trait
// ─────────────────────────────────────────────────────────────────────────────

/// Asynchronous transport for posting a FairPlay SPC to a Key Management
/// Server and returning the raw CKC response bytes.
///
/// Implementations are responsible for:
///
/// 1. Establishing the network connection (plain TCP or TLS).
/// 2. Performing an HTTP/1.1 `POST` with `Content-Type: application/json`.
/// 3. Returning the raw response body bytes on 2xx.
///
/// On non-2xx responses, implementations should return
/// [`DrmError::LicenseDenied`]. Network / IO / parsing failures should be
/// mapped to [`DrmError::NetworkError`].
#[async_trait]
pub trait FairPlayKeyClient: Send + Sync {
    /// POST `json_body` to `ksm_url` and return the raw response bytes.
    async fn fetch_ckc(
        &self,
        ksm_url: &str,
        json_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON request builder + CKC response parser
// ─────────────────────────────────────────────────────────────────────────────

/// Build the JSON body for a FairPlay KSM request.
///
/// The `spc_bytes` are Base64-encoded.  The optional `certificate_bytes`
/// are included when present.
pub fn build_ksm_json(
    asset_id: &str,
    spc_bytes: &[u8],
    certificate_bytes: Option<&[u8]>,
) -> Result<Vec<u8>, DrmError> {
    let mut map = serde_json::Map::new();
    map.insert(
        "asset_id".to_string(),
        serde_json::Value::String(asset_id.to_owned()),
    );
    map.insert(
        "spc".to_string(),
        serde_json::Value::String(STANDARD.encode(spc_bytes)),
    );
    if let Some(cert) = certificate_bytes {
        map.insert(
            "certificate".to_string(),
            serde_json::Value::String(STANDARD.encode(cert)),
        );
    }
    let body = serde_json::Value::Object(map);
    serde_json::to_vec(&body).map_err(DrmError::JsonError)
}

/// Parse a FairPlay KSM response body, extracting and Base64-decoding the CKC.
///
/// Expected JSON format: `{ "ckc": "<base64 CKC>" }`.
/// Returns [`DrmError::LicenseError`] if the `"ckc"` field is absent.
/// Returns [`DrmError::Base64Error`] if the CKC Base64 is malformed.
/// Returns [`DrmError::JsonError`] if the body is not valid JSON.
pub fn parse_ckc_response(response_body: &[u8]) -> Result<Vec<u8>, DrmError> {
    parse_ckc_response_inner(response_body)
}

/// Internal implementation shared by the public API and `request_key_from_server`.
fn parse_ckc_response_inner(response_body: &[u8]) -> Result<Vec<u8>, DrmError> {
    let v: serde_json::Value =
        serde_json::from_slice(response_body).map_err(DrmError::JsonError)?;

    let ckc_b64 = v.get("ckc").and_then(|c| c.as_str()).ok_or_else(|| {
        DrmError::LicenseError(
            "FairPlay KSM response is missing the required 'ckc' field".to_string(),
        )
    })?;

    STANDARD.decode(ckc_b64).map_err(DrmError::Base64Error)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared URL parser
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsedEndpoint {
    authority: String,
    host: String,
    path_and_query: String,
    scheme: String,
}

fn parse_endpoint(ksm_url: &str) -> Result<ParsedEndpoint, DrmError> {
    let uri: Uri = ksm_url.parse().map_err(|e| {
        DrmError::NetworkError(format!("invalid FairPlay KSM URL `{ksm_url}`: {e}"))
    })?;

    let scheme = uri
        .scheme_str()
        .ok_or_else(|| {
            DrmError::NetworkError(format!("FairPlay KSM URL `{ksm_url}` is missing a scheme"))
        })?
        .to_ascii_lowercase();

    let host = uri
        .host()
        .ok_or_else(|| {
            DrmError::NetworkError(format!("FairPlay KSM URL `{ksm_url}` is missing a host"))
        })?
        .to_owned();

    let port = uri.port_u16().unwrap_or(match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        other => {
            return Err(DrmError::NetworkError(format!(
                "unsupported scheme `{other}` in FairPlay KSM URL"
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

fn build_json_post_request(
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
        .header("Content-Type", "application/json")
        .header("Content-Length", body.len().to_string())
        .header("User-Agent", "oximedia-drm-fairplay/0.1");

    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }

    builder
        .body(Full::new(Bytes::copy_from_slice(body)))
        .map_err(|e| DrmError::NetworkError(format!("failed to build FairPlay HTTP request: {e}")))
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
        .map_err(|_| DrmError::NetworkError("FairPlay HTTP handshake timed out".to_string()))?
        .map_err(|e| DrmError::NetworkError(format!("FairPlay HTTP/1.1 handshake failed: {e}")))?;

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
                "failed to send FairPlay key request: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "FairPlay key request timed out before response headers".to_string(),
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
                "failed to read FairPlay CKC response body: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "FairPlay CKC response body read timed out".to_string(),
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
// HyperPlainFairPlayClient
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-HTTP [`FairPlayKeyClient`] built on `hyper` 1.x.
///
/// Intended for **test mock servers** and never negotiates TLS.
/// For production traffic use [`HyperRustlsFairPlayClient`] (requires the
/// `fairplay-network` feature).
#[derive(Debug, Clone)]
pub struct HyperPlainFairPlayClient {
    timeout_ms: u32,
}

impl HyperPlainFairPlayClient {
    /// Create a plain-HTTP FairPlay client with the default 30s timeout.
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

impl Default for HyperPlainFairPlayClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FairPlayKeyClient for HyperPlainFairPlayClient {
    async fn fetch_ckc(
        &self,
        ksm_url: &str,
        json_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError> {
        let endpoint = parse_endpoint(ksm_url)?;
        if endpoint.scheme != "http" {
            return Err(DrmError::NetworkError(format!(
                "HyperPlainFairPlayClient only supports http:// URLs, got scheme `{}`",
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

        let req = build_json_post_request(&endpoint, json_body, headers)?;
        let io = TokioIo::new(tcp);
        send_and_collect(io, req, timeout).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HyperRustlsFairPlayClient (fairplay-network feature)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "fairplay-network")]
mod tls {
    use super::{
        build_json_post_request, parse_endpoint, send_and_collect, Bytes, Duration,
        FairPlayKeyClient, Full, Request,
    };
    use crate::DrmError;
    use async_trait::async_trait;
    use hyper_util::rt::TokioIo;
    use rustls::pki_types::ServerName;
    use std::sync::Arc;
    use tokio::net::TcpStream;
    use tokio_rustls::TlsConnector;

    /// Production-grade TLS [`FairPlayKeyClient`] built on `tokio-rustls`
    /// with the `rustls-rustcrypto` provider (Pure-Rust TLS).
    pub struct HyperRustlsFairPlayClient {
        timeout_ms: u32,
        roots: Option<Arc<rustls::RootCertStore>>,
    }

    impl HyperRustlsFairPlayClient {
        /// Create a TLS FairPlay client with the default 30s timeout.
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
                    "HyperRustlsFairPlayClient requires explicit root certificates".to_string(),
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

    impl Default for HyperRustlsFairPlayClient {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl FairPlayKeyClient for HyperRustlsFairPlayClient {
        async fn fetch_ckc(
            &self,
            ksm_url: &str,
            json_body: &[u8],
            headers: &[(String, String)],
        ) -> Result<Vec<u8>, DrmError> {
            let endpoint = parse_endpoint(ksm_url)?;
            if endpoint.scheme != "https" {
                return Err(DrmError::NetworkError(format!(
                    "HyperRustlsFairPlayClient requires https:// URLs, got scheme `{}`",
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

            let req: Request<Full<Bytes>> = build_json_post_request(&endpoint, json_body, headers)?;
            let io = TokioIo::new(tls);
            send_and_collect(io, req, timeout).await
        }
    }
}

#[cfg(feature = "fairplay-network")]
pub use tls::HyperRustlsFairPlayClient;

// ─────────────────────────────────────────────────────────────────────────────
// FairPlayClient extension (request_key_from_server)
// ─────────────────────────────────────────────────────────────────────────────

use crate::fairplay::{CkcFormat, FairPlayClient, FairPlayKeyResponse};

/// Extension trait providing `request_key_from_server` for [`FairPlayClient`].
///
/// This trait is implemented by [`FairPlayClient`] and adds the async
/// network round-trip needed to send the SPC to a real KSM server.
///
/// # Usage
///
/// ```no_run
/// # use oximedia_drm::fairplay_rpc::{FairPlayClientExt, HyperPlainFairPlayClient};
/// # use oximedia_drm::fairplay::FairPlayClient;
/// # async fn example() -> oximedia_drm::Result<()> {
/// let mut client = FairPlayClient::new(b"cert-bytes".to_vec());
/// let ckc = client.request_key_from_server(
///     "http://localhost:9000/fps/key",
///     "asset-123".to_string(),
///     &HyperPlainFairPlayClient::new(),
///     &[],
/// ).await?;
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait FairPlayClientExt {
    /// Generate an SPC for `asset_id`, POST it to `ksm_url`, and process
    /// the returned CKC — all in a single call.
    ///
    /// The `client` transport determines whether the request is made over
    /// plain HTTP or TLS.  `extra_headers` are forwarded verbatim.
    ///
    /// On success the CKC is cached internally (via `process_ckc`) and
    /// the raw decoded CKC bytes are returned.
    async fn request_key_from_server<C>(
        &mut self,
        ksm_url: &str,
        asset_id: String,
        client: &C,
        extra_headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError>
    where
        C: FairPlayKeyClient + ?Sized + Sync;
}

#[async_trait]
impl FairPlayClientExt for FairPlayClient {
    async fn request_key_from_server<C>(
        &mut self,
        ksm_url: &str,
        asset_id: String,
        client: &C,
        extra_headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError>
    where
        C: FairPlayKeyClient + ?Sized + Sync,
    {
        // Generate the SPC for this asset.
        let key_request = self.request_key(asset_id.clone())?;

        // Build the JSON body.
        let json_body = build_ksm_json(
            &asset_id,
            &key_request.spc_data,
            key_request.certificate.as_deref(),
        )?;

        // POST to the KSM.
        let response_bytes = client.fetch_ckc(ksm_url, &json_body, extra_headers).await?;

        // Parse the CKC from the response.
        let ckc_bytes = parse_ckc_response_inner(&response_bytes)?;

        // Store the CKC.
        let response = FairPlayKeyResponse::new(ckc_bytes.clone(), CkcFormat::Binary);
        self.process_ckc(asset_id, response)?;

        Ok(ckc_bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no network)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod unit_tests {
    use super::*;

    // ── JSON request encoding ───────────────────────────────────────────────

    #[test]
    fn test_build_ksm_json_encodes_spc_as_base64() {
        let spc = b"some-spc-binary-data";
        let json_bytes = build_ksm_json("asset-001", spc, None).expect("build_ksm_json");
        let v: serde_json::Value = serde_json::from_slice(&json_bytes).expect("valid JSON");

        let spc_b64 = v["spc"].as_str().expect("spc field must be a string");
        let decoded = STANDARD.decode(spc_b64).expect("valid base64");
        assert_eq!(decoded, spc, "decoded SPC must match original bytes");

        let asset = v["asset_id"].as_str().expect("asset_id field");
        assert_eq!(asset, "asset-001");

        assert!(
            v.get("certificate").is_none(),
            "certificate must not appear when None"
        );
    }

    #[test]
    fn test_build_ksm_json_includes_certificate_when_present() {
        let spc = b"spc";
        let cert = b"cert-bytes";
        let json_bytes = build_ksm_json("asset-002", spc, Some(cert)).expect("build_ksm_json");
        let v: serde_json::Value = serde_json::from_slice(&json_bytes).expect("valid JSON");

        let cert_b64 = v["certificate"].as_str().expect("certificate field");
        let decoded = STANDARD.decode(cert_b64).expect("valid base64");
        assert_eq!(decoded, cert);
    }

    // ── Base64 CKC decoding ─────────────────────────────────────────────────

    #[test]
    fn test_parse_ckc_response_ok() {
        let ckc_bytes = b"binary-ckc-payload";
        let b64 = STANDARD.encode(ckc_bytes);
        let response = serde_json::to_vec(&serde_json::json!({ "ckc": b64 })).expect("json");

        let decoded = parse_ckc_response_inner(&response).expect("parse_ckc_response_inner");
        assert_eq!(decoded, ckc_bytes.to_vec());
    }

    #[test]
    fn test_parse_ckc_response_missing_field_returns_license_error() {
        let response = br#"{"other": "value"}"#;
        let err = parse_ckc_response_inner(response).expect_err("must fail without 'ckc'");
        match err {
            DrmError::LicenseError(msg) => {
                assert!(msg.contains("ckc"), "error must mention 'ckc' field");
            }
            other => panic!("expected LicenseError, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_endpoint_http() {
        let p = parse_endpoint("http://ksm.example.com/fps/key").expect("parse");
        assert_eq!(p.scheme, "http");
        assert_eq!(p.host, "ksm.example.com");
        assert_eq!(p.authority, "ksm.example.com:80");
        assert_eq!(p.path_and_query, "/fps/key");
    }

    #[test]
    fn test_parse_endpoint_https() {
        let p = parse_endpoint("https://ksm.example.com/fps/key").expect("parse");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.authority, "ksm.example.com:443");
    }
}
