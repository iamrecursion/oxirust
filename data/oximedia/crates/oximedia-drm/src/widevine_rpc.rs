//! Widevine license-acquisition RPC layer.
//!
//! This module provides the network transport used to POST a serialised
//! `WidevineLicenseRequest` body to a remote Widevine license server and
//! collect the binary response that is then fed into
//! `WidevineCdm::process_response`.
//!
//! Two clients ship out of the box:
//!
//! * [`HyperPlainLicenseClient`] — `hyper` 1.x over plain TCP. Suitable for
//!   in-process mock servers and `http://` endpoints during testing.
//! * [`HyperRustlsLicenseClient`] — Pure-Rust TLS via `tokio-rustls` 0.26 +
//!   the `rustls-rustcrypto` provider, gated behind the
//!   `widevine-network` Cargo feature for production `https://` traffic.
//!
//! Both implement the [`LicenseClient`] trait so callers (including
//! `WidevineCdm::acquire_license`) can be parameterised over the transport.
//!
//! The crypto provider for TLS is **`rustls-rustcrypto`**, keeping the
//! Pure-Rust policy intact: no openssl, no ring, no C/C++ deps.

use crate::DrmError;
use async_trait::async_trait;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::client::conn::http1 as client_http1;
use hyper::{Method, Request, Uri};
use hyper_util::rt::TokioIo;

use std::time::Duration;
use tokio::net::TcpStream;

// ──────────────────────────────────────────────────────────────────────────────
// LicenseClient trait
// ──────────────────────────────────────────────────────────────────────────────

/// Asynchronous transport for delivering a binary Widevine license request to
/// a remote (or in-process mock) server and returning the binary response body.
///
/// Implementations are responsible for:
///
/// 1. Establishing the network connection (plain TCP or TLS).
/// 2. Performing an HTTP/1.1 `POST` of the supplied `request_body` with
///    `Content-Type: application/octet-stream` and any caller-supplied
///    additional headers.
/// 3. Returning the raw response body on success.
///
/// On non-2xx HTTP responses, implementations should return
/// [`DrmError::LicenseDenied`].  Network / IO / parsing failures should be
/// mapped to [`DrmError::NetworkError`].
#[async_trait]
pub trait LicenseClient: Send + Sync {
    /// POST `request_body` to `server_url` and return the response payload.
    async fn fetch_license(
        &self,
        server_url: &str,
        request_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Parsed URL helper (no external `url` dep — covered by hyper Uri only)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsedEndpoint {
    /// `host:port` ready to feed into a TCP connect.
    authority: String,
    /// Host portion (without port). Used for the `Host` header and TLS SNI.
    host: String,
    /// Path + optional query string. At minimum `"/"`.
    path_and_query: String,
    /// Scheme, lower-cased: `"http"` or `"https"`.
    scheme: String,
}

fn parse_endpoint(server_url: &str) -> Result<ParsedEndpoint, DrmError> {
    let uri: Uri = server_url.parse().map_err(|e| {
        DrmError::NetworkError(format!("invalid license server URL `{server_url}`: {e}"))
    })?;

    let scheme = uri
        .scheme_str()
        .ok_or_else(|| {
            DrmError::NetworkError(format!(
                "license server URL `{server_url}` is missing a scheme"
            ))
        })?
        .to_ascii_lowercase();

    let host = uri
        .host()
        .ok_or_else(|| {
            DrmError::NetworkError(format!(
                "license server URL `{server_url}` is missing a host"
            ))
        })?
        .to_owned();

    let port = uri.port_u16().unwrap_or(match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        other => {
            return Err(DrmError::NetworkError(format!(
                "unsupported scheme `{other}` in license server URL"
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

// ──────────────────────────────────────────────────────────────────────────────
// Common HTTP request builder + response parser
// ──────────────────────────────────────────────────────────────────────────────

fn build_post_request(
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
        .header("Content-Type", "application/octet-stream")
        .header("Content-Length", body.len().to_string())
        .header("User-Agent", "oximedia-drm-widevine/0.1");

    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }

    builder
        .body(Full::new(Bytes::copy_from_slice(body)))
        .map_err(|e| DrmError::NetworkError(format!("failed to build HTTP request: {e}")))
}

/// Drive a hyper HTTP/1.1 client over an existing connected IO object, send
/// the supplied POST request, and return the response body on 2xx.
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
        .map_err(|_| DrmError::NetworkError("HTTP handshake timed out".to_string()))?
        .map_err(|e| DrmError::NetworkError(format!("HTTP/1.1 handshake failed: {e}")))?;

    let (mut sender, conn) = handshake;

    // Drive the connection to completion in the background.
    let conn_handle = tokio::spawn(async move {
        // Ignore connection errors — they'll surface via send_request below.
        let _ = conn.await;
    });

    let send_fut = sender.send_request(req);
    let resp_result = tokio::time::timeout(timeout, send_fut).await;
    let resp = match resp_result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(format!(
                "failed to send license request: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "license request timed out before response headers".to_string(),
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
                "failed to read license response body: {e}"
            )));
        }
        Err(_) => {
            conn_handle.abort();
            return Err(DrmError::NetworkError(
                "license response body read timed out".to_string(),
            ));
        }
    };

    // Connection task is fine to finish on its own now.
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

// ──────────────────────────────────────────────────────────────────────────────
// HyperPlainLicenseClient — plain-HTTP transport for tests / dev
// ──────────────────────────────────────────────────────────────────────────────

/// Plain-HTTP `LicenseClient` implementation built on `hyper` 1.x.
///
/// This client is intended for **test mock servers** and never negotiates TLS.
/// For production traffic, use [`HyperRustlsLicenseClient`] (gated behind the
/// `widevine-network` feature).
///
/// # Timeouts
///
/// A single timeout is applied to each of (handshake, request, body-read).
/// The default is 30 seconds.  Use [`Self::with_timeout_ms`] to override.
#[derive(Debug, Clone)]
pub struct HyperPlainLicenseClient {
    timeout_ms: u32,
}

impl HyperPlainLicenseClient {
    /// Create a plain-HTTP license client with the default 30s timeout.
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

impl Default for HyperPlainLicenseClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LicenseClient for HyperPlainLicenseClient {
    async fn fetch_license(
        &self,
        server_url: &str,
        request_body: &[u8],
        headers: &[(String, String)],
    ) -> Result<Vec<u8>, DrmError> {
        let endpoint = parse_endpoint(server_url)?;
        if endpoint.scheme != "http" {
            return Err(DrmError::NetworkError(format!(
                "HyperPlainLicenseClient only supports http:// URLs, got scheme `{}`",
                endpoint.scheme
            )));
        }

        let timeout = Duration::from_millis(u64::from(self.timeout_ms));

        let connect_fut = TcpStream::connect(&endpoint.authority);
        let tcp = tokio::time::timeout(timeout, connect_fut)
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

        let req = build_post_request(&endpoint, request_body, headers)?;
        let io = TokioIo::new(tcp);
        send_and_collect(io, req, timeout).await
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HyperRustlsLicenseClient — TLS transport using rustls-rustcrypto
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "widevine-network")]
mod tls {
    use super::{
        build_post_request, parse_endpoint, send_and_collect, Bytes, Duration, Full, LicenseClient,
        Request,
    };
    use crate::DrmError;
    use async_trait::async_trait;
    use hyper_util::rt::TokioIo;
    use rustls::pki_types::ServerName;
    use std::sync::Arc;
    use tokio::net::TcpStream;
    use tokio_rustls::TlsConnector;

    /// Production-grade TLS `LicenseClient` built on `tokio-rustls` 0.26 with
    /// the `rustls-rustcrypto` provider (Pure-Rust TLS — no openssl, no ring).
    ///
    /// Construction is **explicit about trust**: callers must supply a
    /// [`rustls::RootCertStore`] via [`Self::with_root_certs`] before
    /// invoking `fetch_license`. If no roots are configured, the request
    /// will fail with [`DrmError::NetworkError`] rather than silently
    /// trusting nothing.
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use oximedia_drm::HyperRustlsLicenseClient;
    ///
    /// let mut roots = rustls::RootCertStore::empty();
    /// // populate `roots` from your platform store / a bundle
    /// let client = HyperRustlsLicenseClient::new().with_root_certs(Arc::new(roots));
    /// ```
    pub struct HyperRustlsLicenseClient {
        timeout_ms: u32,
        roots: Option<Arc<rustls::RootCertStore>>,
    }

    impl HyperRustlsLicenseClient {
        /// Create a TLS license client with the default 30s timeout. The
        /// caller must call [`Self::with_root_certs`] before issuing a
        /// request.
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

        /// Install the trust anchors used to verify the license server's
        /// certificate chain.
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
                    "HyperRustlsLicenseClient requires explicit root certificates".to_string(),
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

    impl Default for HyperRustlsLicenseClient {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl LicenseClient for HyperRustlsLicenseClient {
        async fn fetch_license(
            &self,
            server_url: &str,
            request_body: &[u8],
            headers: &[(String, String)],
        ) -> Result<Vec<u8>, DrmError> {
            let endpoint = parse_endpoint(server_url)?;
            if endpoint.scheme != "https" {
                return Err(DrmError::NetworkError(format!(
                    "HyperRustlsLicenseClient requires https:// URLs, got scheme `{}`",
                    endpoint.scheme
                )));
            }

            let timeout = Duration::from_millis(u64::from(self.timeout_ms));

            let connect_fut = TcpStream::connect(&endpoint.authority);
            let tcp = tokio::time::timeout(timeout, connect_fut)
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

            let tls_fut = connector.connect(server_name, tcp);
            let tls = tokio::time::timeout(timeout, tls_fut)
                .await
                .map_err(|_| {
                    DrmError::NetworkError(format!(
                        "TLS handshake to `{}` timed out after {}ms",
                        endpoint.authority, self.timeout_ms
                    ))
                })?
                .map_err(|e| DrmError::NetworkError(format!("TLS handshake failed: {e}")))?;

            let req: Request<Full<Bytes>> = build_post_request(&endpoint, request_body, headers)?;
            let io = TokioIo::new(tls);
            send_and_collect(io, req, timeout).await
        }
    }
}

#[cfg(feature = "widevine-network")]
pub use tls::HyperRustlsLicenseClient;

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests for the URL parser / request builder (no network)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parses_http_with_default_port() {
        let p = parse_endpoint("http://license.example.com/widevine").expect("parse");
        assert_eq!(p.scheme, "http");
        assert_eq!(p.host, "license.example.com");
        assert_eq!(p.authority, "license.example.com:80");
        assert_eq!(p.path_and_query, "/widevine");
    }

    #[test]
    fn parses_https_with_default_port() {
        let p = parse_endpoint("https://license.example.com/get_license").expect("parse");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.authority, "license.example.com:443");
    }

    #[test]
    fn parses_explicit_port_and_query() {
        let p = parse_endpoint("http://127.0.0.1:8080/x?token=abc").expect("parse");
        assert_eq!(p.scheme, "http");
        assert_eq!(p.authority, "127.0.0.1:8080");
        assert_eq!(p.path_and_query, "/x?token=abc");
    }

    #[test]
    fn rejects_missing_scheme() {
        let err = parse_endpoint("license.example.com/widevine").unwrap_err();
        match err {
            DrmError::NetworkError(_) => {}
            other => panic!("expected NetworkError, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unsupported_scheme() {
        let err = parse_endpoint("ftp://license.example.com/widevine").unwrap_err();
        match err {
            DrmError::NetworkError(_) => {}
            other => panic!("expected NetworkError, got {other:?}"),
        }
    }

    #[test]
    fn build_post_request_sets_content_type_and_length() {
        let p = parse_endpoint("http://127.0.0.1:9000/license").expect("parse");
        let body = b"abc";
        let req = build_post_request(&p, body, &[]).expect("build");
        assert_eq!(req.method(), Method::POST);
        let headers = req.headers();
        assert_eq!(
            headers
                .get("Content-Type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "application/octet-stream"
        );
        assert_eq!(
            headers
                .get("Content-Length")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "3"
        );
    }

    #[test]
    fn build_post_request_forwards_extra_headers() {
        let p = parse_endpoint("http://127.0.0.1:9000/license").expect("parse");
        let extras = vec![
            ("X-Provider".to_string(), "alpha".to_string()),
            ("Authorization".to_string(), "Bearer xyz".to_string()),
        ];
        let req = build_post_request(&p, &[], &extras).expect("build");
        let headers = req.headers();
        assert_eq!(
            headers
                .get("X-Provider")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "alpha"
        );
        assert_eq!(
            headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "Bearer xyz"
        );
    }
}
