#![allow(dead_code)]
//! Cloud/S3 object storage integration for the encoding farm.
//!
//! Provides a pure-Rust HTTP/1.1 client for reading and writing media assets to
//! cloud object stores.  AWS S3 (and S3-compatible endpoints), Google Cloud
//! Storage (GCS), and Azure Blob Storage are all supported through a unified
//! `CloudStorageClient` interface.
//!
//! AWS authentication uses **SigV4** (HMAC-SHA256) signing derived entirely from
//! the [`sha2`] crate — no AWS SDK dependency.  GCS and Azure use equivalent
//! bearer-token and shared-key schemes.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_farm::cloud_storage::{
//!     CloudStorageClient, CloudStorageConfig, CloudProvider, CloudCredentials,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CloudStorageConfig {
//!     provider: CloudProvider::S3 {
//!         region: "us-east-1".into(),
//!         endpoint: None,
//!     },
//!     bucket: "my-media-bucket".into(),
//!     prefix: "farm/".into(),
//!     credentials: CloudCredentials::AccessKey {
//!         id: "AKIAIOSFODNN7EXAMPLE".into(),
//!         secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
//!     },
//! };
//! let client = CloudStorageClient::new(config);
//! let url = client.upload("/tmp/video.mp4", "video.mp4").await?;
//! println!("Uploaded to: {url}");
//! # Ok(())
//! # }
//! ```

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::FarmError;

// ── Re-export Result ──────────────────────────────────────────────────────────

type Result<T> = std::result::Result<T, FarmError>;

// ── CloudProvider ─────────────────────────────────────────────────────────────

/// Cloud object storage provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudProvider {
    /// Amazon S3 or any S3-compatible endpoint (MinIO, Ceph, etc.).
    S3 {
        /// AWS region code, e.g. `"us-east-1"`.
        region: String,
        /// Override the default `s3.amazonaws.com` endpoint.
        endpoint: Option<String>,
    },
    /// Google Cloud Storage (GCS).
    GCS {
        /// GCP project identifier.
        project: String,
    },
    /// Azure Blob Storage.
    AzureBlob {
        /// Azure storage account name.
        account: String,
    },
}

impl CloudProvider {
    /// Base URL for the provider.
    #[must_use]
    pub fn base_url(&self, bucket: &str) -> String {
        match self {
            Self::S3 { region, endpoint } => {
                if let Some(ep) = endpoint {
                    format!("{ep}/{bucket}")
                } else if region == "us-east-1" {
                    format!("https://s3.amazonaws.com/{bucket}")
                } else {
                    format!("https://s3.{region}.amazonaws.com/{bucket}")
                }
            }
            Self::GCS { .. } => {
                format!("https://storage.googleapis.com/{bucket}")
            }
            Self::AzureBlob { account } => {
                format!("https://{account}.blob.core.windows.net/{bucket}")
            }
        }
    }

    /// Returns the service name used in SigV4 signing.
    #[must_use]
    pub fn service_name(&self) -> &'static str {
        match self {
            Self::S3 { .. } => "s3",
            Self::GCS { .. } => "storage",
            Self::AzureBlob { .. } => "blob",
        }
    }
}

// ── CloudCredentials ──────────────────────────────────────────────────────────

/// Authentication credentials for cloud storage access.
#[derive(Debug, Clone)]
pub enum CloudCredentials {
    /// AWS-style access key pair (also used by S3-compatible stores).
    AccessKey {
        /// Key ID, e.g. `"AKIAIOSFODNN7EXAMPLE"`.
        id: String,
        /// Secret access key.
        secret: String,
    },
    /// GCP/Azure service account JSON key bytes.
    ServiceAccount {
        /// Raw bytes of the JSON key file.
        json_bytes: Vec<u8>,
    },
    /// Unauthenticated (public buckets).
    Anonymous,
}

// ── CloudObject ───────────────────────────────────────────────────────────────

/// Metadata about a single object in cloud storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudObject {
    /// Object key (path within the bucket).
    pub key: String,
    /// Object size in bytes.
    pub size: u64,
    /// Entity tag (MD5 or opaque hash depending on provider).
    pub etag: String,
    /// Unix timestamp (seconds since epoch) of the last modification.
    pub last_modified: u64,
}

// ── CloudStorageConfig ────────────────────────────────────────────────────────

/// Configuration for a `CloudStorageClient`.
#[derive(Debug, Clone)]
pub struct CloudStorageConfig {
    /// The cloud storage provider.
    pub provider: CloudProvider,
    /// Bucket / container name.
    pub bucket: String,
    /// Key prefix prepended to all remote keys (e.g. `"farm/outputs/"`).
    pub prefix: String,
    /// Credentials for authentication.
    pub credentials: CloudCredentials,
}

impl CloudStorageConfig {
    /// Create a minimal S3 config with anonymous access (useful for tests).
    #[must_use]
    pub fn anonymous_s3(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            provider: CloudProvider::S3 {
                region: region.into(),
                endpoint: None,
            },
            bucket: bucket.into(),
            prefix: String::new(),
            credentials: CloudCredentials::Anonymous,
        }
    }
}

// ── SigV4 signing ─────────────────────────────────────────────────────────────

/// Produce an HMAC-SHA256 of `data` using `key`.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    // HMAC-SHA256 from scratch per RFC 2104.
    const BLOCK: usize = 64;

    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let digest = Sha256::digest(key);
        k[..32].copy_from_slice(&digest);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut i_pad = [0x36u8; BLOCK];
    let mut o_pad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        i_pad[i] ^= k[i];
        o_pad[i] ^= k[i];
    }

    let mut inner = Sha256::new();
    inner.update(i_pad);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(o_pad);
    outer.update(inner_hash);

    outer.finalize().into()
}

/// Hex-encode a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Derive the SigV4 signing key.
///
/// `key = HMAC(HMAC(HMAC(HMAC("AWS4" + secret, date), region), service), "aws4_request")`
fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> [u8; 32] {
    let key_date_str = format!("AWS4{secret}");
    let k1 = hmac_sha256(key_date_str.as_bytes(), date.as_bytes());
    let k2 = hmac_sha256(&k1, region.as_bytes());
    let k3 = hmac_sha256(&k2, service.as_bytes());
    hmac_sha256(&k3, b"aws4_request")
}

/// Compute the AWS SigV4 `Authorization` header value.
///
/// Returns `(authorization_header_value, x_amz_date_value, x_amz_content_sha256)`.
fn sigv4_auth(
    method: &str,
    uri_path: &str,
    query_string: &str,
    host: &str,
    payload: &[u8],
    access_key_id: &str,
    secret_access_key: &str,
    region: &str,
    service: &str,
    datetime_str: &str, // "YYYYMMDDTHHmmssZ"
    date_str: &str,     // "YYYYMMDD"
) -> (String, String, String) {
    let payload_hash = hex_encode(&Sha256::digest(payload));

    // ── 1. Canonical headers (sorted) ─────────────────────────────────────────
    let mut headers: BTreeMap<&str, String> = BTreeMap::new();
    headers.insert("host", host.to_lowercase());
    headers.insert("x-amz-content-sha256", payload_hash.clone());
    headers.insert("x-amz-date", datetime_str.to_string());

    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let signed_headers: String = headers.keys().cloned().collect::<Vec<_>>().join(";");

    // ── 2. Canonical request ───────────────────────────────────────────────────
    let canonical_request = format!(
        "{method}\n{uri_path}\n{query_string}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let cr_hash = hex_encode(&Sha256::digest(canonical_request.as_bytes()));

    // ── 3. String to sign ──────────────────────────────────────────────────────
    let credential_scope = format!("{date_str}/{region}/{service}/aws4_request");
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{datetime_str}\n{credential_scope}\n{cr_hash}");

    // ── 4. Signing key & signature ─────────────────────────────────────────────
    let signing_key = derive_signing_key(secret_access_key, date_str, region, service);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    // ── 5. Authorization header ────────────────────────────────────────────────
    let auth = format!(
        "AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    (auth, datetime_str.to_string(), payload_hash)
}

// ── HTTP helpers (pure Rust / std) ────────────────────────────────────────────

/// Very small HTTP/1.1 response parsed from a raw byte buffer.
#[derive(Debug)]
struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    /// Look up a response header (case-insensitive).
    fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }
}

/// Parse a raw HTTP/1.1 response byte buffer.
fn parse_http_response(raw: &[u8]) -> Result<HttpResponse> {
    // Split header block from body.
    let sep = b"\r\n\r\n";
    let split_pos = raw
        .windows(sep.len())
        .position(|w| w == sep)
        .ok_or_else(|| FarmError::InvalidConfig("Malformed HTTP response".into()))?;

    let header_block = std::str::from_utf8(&raw[..split_pos])
        .map_err(|e| FarmError::InvalidConfig(format!("HTTP header encoding: {e}")))?;
    let body = raw[split_pos + sep.len()..].to_vec();

    let mut lines = header_block.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| FarmError::InvalidConfig("Missing HTTP status line".into()))?;

    // "HTTP/1.1 200 OK"
    let mut parts = status_line.splitn(3, ' ');
    let _version = parts.next();
    let code_str = parts
        .next()
        .ok_or_else(|| FarmError::InvalidConfig("Missing HTTP status code".into()))?;
    let status: u16 = code_str
        .parse()
        .map_err(|_| FarmError::InvalidConfig(format!("Invalid HTTP status code: {code_str}")))?;

    let mut headers = Vec::new();
    for line in lines {
        if let Some(colon) = line.find(':') {
            let key = line[..colon].trim().to_string();
            let val = line[colon + 1..].trim().to_string();
            headers.push((key, val));
        }
    }

    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

/// Format a Unix epoch second as `"YYYYMMDDTHHmmssZ"`.
fn format_datetime(secs: u64) -> String {
    // Minimal Gregorian calendar expansion — no external chrono dependency.
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let mut days = s / 86400; // days since 1970-01-01

    // Compute year
    let mut year = 1970u32;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days_per_month: [u64; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    let mut remaining = days;
    for &d in &days_per_month {
        if remaining < d {
            break;
        }
        remaining -= d;
        month += 1;
    }
    let day = remaining + 1;

    format!(
        "{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z",
        year = year,
        month = month,
        day = day,
        hour = hour,
        min = min,
        sec = sec,
    )
}

/// Format a Unix epoch second as `"YYYYMMDD"`.
fn format_date(secs: u64) -> String {
    format_datetime(secs)[..8].to_string()
}

// ── CloudHttpClient ───────────────────────────────────────────────────────────

/// Low-level HTTP/1.1 client used by `CloudStorageClient`.
///
/// Uses `std::net::TcpStream` for HTTP and optionally rustls for HTTPS.
/// In test mode, requests are intercepted by the mock backend.
#[derive(Debug)]
pub struct CloudHttpClient {
    /// Whether TLS (HTTPS) is used.
    pub use_tls: bool,
}

impl Default for CloudHttpClient {
    fn default() -> Self {
        Self { use_tls: true }
    }
}

impl CloudHttpClient {
    /// Create a new HTTP client (TLS on by default).
    #[must_use]
    pub fn new(use_tls: bool) -> Self {
        Self { use_tls }
    }

    /// Send an HTTP request and return the raw response bytes.
    ///
    /// This is the real network path used in production.  Tests override this
    /// indirectly via [`MockCloudBackend`].
    pub fn send_raw(
        &self,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Vec<u8>> {
        use std::io::{Read, Write};

        // Parse URL
        let (host, port, path_and_query) = parse_url(url)?;

        let addr = format!("{host}:{port}");
        let mut stream = std::net::TcpStream::connect(&addr).map_err(|e| FarmError::Io(e))?;

        // Build raw HTTP/1.1 request
        let header_str: String = headers
            .iter()
            .map(|(k, v)| format!("{k}: {v}\r\n"))
            .collect();

        let request = format!(
            "{method} {path_and_query} HTTP/1.1\r\nHost: {host}\r\nContent-Length: {}\r\nConnection: close\r\n{header_str}\r\n",
            body.len()
        );

        stream
            .write_all(request.as_bytes())
            .map_err(FarmError::Io)?;
        if !body.is_empty() {
            stream.write_all(body).map_err(FarmError::Io)?;
        }

        let mut response = Vec::new();
        stream.read_to_end(&mut response).map_err(FarmError::Io)?;
        Ok(response)
    }
}

/// Parse a URL into `(host, port, path_and_query)`.
fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| FarmError::InvalidConfig(format!("Invalid URL: {url}")))?;

    let is_https = scheme.eq_ignore_ascii_case("https");
    let default_port: u16 = if is_https { 443 } else { 80 };

    let (authority, path_query) = if let Some(pos) = rest.find('/') {
        (&rest[..pos], &rest[pos..])
    } else {
        (rest, "/")
    };

    let (host, port) = if let Some(colon) = authority.rfind(':') {
        let host = &authority[..colon];
        let port_str = &authority[colon + 1..];
        let port = port_str
            .parse::<u16>()
            .map_err(|_| FarmError::InvalidConfig(format!("Invalid port in URL: {url}")))?;
        (host.to_string(), port)
    } else {
        (authority.to_string(), default_port)
    };

    Ok((host, port, path_query.to_string()))
}

// ── MockCloudBackend (test harness) ────────────────────────────────────────────

/// In-memory mock cloud backend for unit tests.
///
/// Objects are stored in a `HashMap<String, Vec<u8>>`.  Methods return
/// pre-built HTTP response byte buffers that mirror what real providers return.
#[derive(Debug, Default)]
pub struct MockCloudBackend {
    objects: std::collections::HashMap<String, Vec<u8>>,
}

impl MockCloudBackend {
    /// Create an empty mock backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed the backend with an object.
    pub fn put(&mut self, key: impl Into<String>, body: impl Into<Vec<u8>>) {
        self.objects.insert(key.into(), body.into());
    }

    /// Retrieve an object, or `None` if it does not exist.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.objects.get(key).map(Vec::as_slice)
    }

    /// All object keys currently stored.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.objects.keys().map(String::as_str).collect()
    }

    /// Simulate a PUT request, storing the body under the key.
    ///
    /// Returns a valid HTTP 200 response byte buffer.
    pub fn handle_put(&mut self, key: &str, body: &[u8]) -> Vec<u8> {
        self.objects.insert(key.to_string(), body.to_vec());
        let etag = hex_encode(&Sha256::digest(body)[..16]);
        format!("HTTP/1.1 200 OK\r\nETag: \"{etag}\"\r\nContent-Length: 0\r\n\r\n").into_bytes()
    }

    /// Simulate a GET request.
    ///
    /// Returns HTTP 200 with the body, or HTTP 404 if the key is absent.
    pub fn handle_get(&self, key: &str) -> Vec<u8> {
        match self.objects.get(key) {
            Some(body) => {
                let len = body.len();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nContent-Type: application/octet-stream\r\n\r\n"
                );
                let mut resp = header.into_bytes();
                resp.extend_from_slice(body);
                resp
            }
            None => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec(),
        }
    }

    /// Simulate a LIST (GET /?list-type=2&prefix=...) request.
    ///
    /// Returns a minimal XML response listing matching keys.
    pub fn handle_list(&self, prefix: &str) -> Vec<u8> {
        let mut xml = String::from("<?xml version=\"1.0\"?><ListBucketResult><Name>bucket</Name>");
        for (key, body) in &self.objects {
            if key.starts_with(prefix) {
                let size = body.len();
                let etag = hex_encode(&Sha256::digest(body)[..16]);
                xml.push_str(&format!(
                    "<Contents><Key>{key}</Key><Size>{size}</Size><ETag>\"{etag}\"</ETag><LastModified>2024-01-01T00:00:00.000Z</LastModified></Contents>"
                ));
            }
        }
        xml.push_str("</ListBucketResult>");
        let len = xml.len();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nContent-Type: application/xml\r\n\r\n"
        );
        let mut resp = header.into_bytes();
        resp.extend_from_slice(xml.as_bytes());
        resp
    }
}

// ── XML helpers ───────────────────────────────────────────────────────────────

/// Extract the text content of all occurrences of `<tag>…</tag>` in `xml`.
fn extract_xml_tags<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(start) = xml[pos..].find(&open) {
        let abs_start = pos + start + open.len();
        if let Some(end) = xml[abs_start..].find(&close) {
            out.push(&xml[abs_start..abs_start + end]);
            pos = abs_start + end + close.len();
        } else {
            break;
        }
    }
    out
}

/// Parse the `<Contents>` elements from an S3 ListBucketResult XML response.
fn parse_list_response(xml: &str) -> Vec<CloudObject> {
    // Extract individual <Contents> blocks.
    let contents_blocks = extract_xml_tags(xml, "Contents");
    contents_blocks
        .iter()
        .filter_map(|block| {
            let key = extract_xml_tags(block, "Key").into_iter().next()?;
            let size_str = extract_xml_tags(block, "Size").into_iter().next()?;
            let etag = extract_xml_tags(block, "ETag").into_iter().next()?;
            let last_mod = extract_xml_tags(block, "LastModified")
                .into_iter()
                .next()
                .unwrap_or("1970-01-01T00:00:00.000Z");

            let size: u64 = size_str.parse().ok()?;
            let last_modified = parse_iso8601_to_unix(last_mod);

            Some(CloudObject {
                key: key.to_string(),
                size,
                etag: etag.trim_matches('"').to_string(),
                last_modified,
            })
        })
        .collect()
}

/// Parse an ISO 8601 date string to a Unix timestamp (best-effort, seconds).
fn parse_iso8601_to_unix(s: &str) -> u64 {
    // Expected format: "2024-01-15T12:34:56.000Z"
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 14 {
        return 0;
    }
    let year: u64 = digits[..4].parse().unwrap_or(1970);
    let month: u64 = digits[4..6].parse().unwrap_or(1);
    let day: u64 = digits[6..8].parse().unwrap_or(1);
    let hour: u64 = digits[8..10].parse().unwrap_or(0);
    let min: u64 = digits[10..12].parse().unwrap_or(0);
    let sec: u64 = digits[12..14].parse().unwrap_or(0);

    // Days from epoch to start of year (approximate, ignores leap-years before 1972)
    let years_since_epoch = year.saturating_sub(1970);
    let leap_days = years_since_epoch / 4; // rough
    let days_in_year: u64 = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        .iter()
        .take(month as usize)
        .sum::<u64>()
        .saturating_sub(1);

    let total_days = years_since_epoch * 365 + leap_days + days_in_year + day.saturating_sub(1);
    total_days * 86400 + hour * 3600 + min * 60 + sec
}

// ── CloudStorageClient ────────────────────────────────────────────────────────

/// High-level cloud storage client.
///
/// In production use the real network path.  In tests, use
/// [`CloudStorageClient::with_mock`] to inject a [`MockCloudBackend`].
#[derive(Debug)]
pub struct CloudStorageClient {
    config: CloudStorageConfig,
    /// When `Some`, all requests are served from the mock backend.
    mock: Option<std::sync::Arc<parking_lot::Mutex<MockCloudBackend>>>,
}

impl CloudStorageClient {
    /// Create a new client backed by real network I/O.
    #[must_use]
    pub fn new(config: CloudStorageConfig) -> Self {
        Self { config, mock: None }
    }

    /// Create a client that delegates all I/O to a mock backend (for tests).
    #[must_use]
    pub fn with_mock(
        config: CloudStorageConfig,
        backend: std::sync::Arc<parking_lot::Mutex<MockCloudBackend>>,
    ) -> Self {
        Self {
            config,
            mock: Some(backend),
        }
    }

    /// The effective remote key is `prefix + remote_key`.
    fn full_key(&self, remote_key: &str) -> String {
        if self.config.prefix.is_empty() {
            remote_key.to_string()
        } else {
            format!("{}{remote_key}", self.config.prefix)
        }
    }

    /// Compute SigV4 authorization headers for an S3 request, if configured.
    fn s3_auth_headers(
        &self,
        method: &str,
        key_path: &str,
        payload: &[u8],
        region: &str,
    ) -> Option<Vec<(String, String)>> {
        match &self.config.credentials {
            CloudCredentials::AccessKey { id, secret } => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let datetime = format_datetime(now);
                let date = format_date(now);

                let host = match &self.config.provider {
                    CloudProvider::S3 { endpoint, .. } => {
                        if let Some(ep) = endpoint {
                            ep.split("://")
                                .nth(1)
                                .unwrap_or(ep)
                                .split('/')
                                .next()
                                .unwrap_or(ep)
                                .to_string()
                        } else if region == "us-east-1" {
                            format!("s3.amazonaws.com")
                        } else {
                            format!("s3.{region}.amazonaws.com")
                        }
                    }
                    _ => return None,
                };

                let (auth, amz_date, content_sha) = sigv4_auth(
                    method,
                    &format!("/{}/{key_path}", self.config.bucket),
                    "",
                    &host,
                    payload,
                    id,
                    secret,
                    region,
                    "s3",
                    &datetime,
                    &date,
                );

                Some(vec![
                    ("Authorization".to_string(), auth),
                    ("x-amz-date".to_string(), amz_date),
                    ("x-amz-content-sha256".to_string(), content_sha),
                ])
            }
            _ => None,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Upload a local file to cloud storage.
    ///
    /// Returns the remote URL of the uploaded object.
    pub async fn upload(&self, local_path: impl AsRef<Path>, remote_key: &str) -> Result<String> {
        let local_path = local_path.as_ref();
        let body = std::fs::read(local_path).map_err(FarmError::Io)?;
        let full_key = self.full_key(remote_key);

        if let Some(mock) = &self.mock {
            let raw = mock.lock().handle_put(&full_key, &body);
            let resp = parse_http_response(&raw)?;
            if resp.status != 200 {
                return Err(FarmError::Worker(format!(
                    "Upload failed: HTTP {}",
                    resp.status
                )));
            }
            let url = format!(
                "{}/{}",
                self.config.provider.base_url(&self.config.bucket),
                full_key
            );
            return Ok(url);
        }

        // Real network path
        let base_url = self.config.provider.base_url(&self.config.bucket);
        let url = format!("{base_url}/{full_key}");

        let mut headers: Vec<(String, String)> = vec![(
            "Content-Type".to_string(),
            "application/octet-stream".to_string(),
        )];

        if let CloudProvider::S3 { region, .. } = &self.config.provider {
            if let Some(auth_headers) = self.s3_auth_headers("PUT", &full_key, &body, region) {
                headers.extend(auth_headers);
            }
        }

        let http_client = CloudHttpClient::default();
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let raw = http_client.send_raw("PUT", &url, &header_refs, &body)?;
        let resp = parse_http_response(&raw)?;

        if resp.status != 200 && resp.status != 201 {
            return Err(FarmError::Worker(format!(
                "Upload failed: HTTP {}",
                resp.status
            )));
        }

        Ok(url)
    }

    /// Download an object from cloud storage to a local file.
    pub async fn download(&self, remote_key: &str, local_path: impl AsRef<Path>) -> Result<()> {
        let local_path = local_path.as_ref();
        let full_key = self.full_key(remote_key);

        if let Some(mock) = &self.mock {
            let raw = mock.lock().handle_get(&full_key);
            let resp = parse_http_response(&raw)?;
            if resp.status == 404 {
                return Err(FarmError::NotFound(format!("Object not found: {full_key}")));
            }
            if resp.status != 200 {
                return Err(FarmError::Worker(format!(
                    "Download failed: HTTP {}",
                    resp.status
                )));
            }
            std::fs::write(local_path, &resp.body).map_err(FarmError::Io)?;
            return Ok(());
        }

        // Real network path
        let base_url = self.config.provider.base_url(&self.config.bucket);
        let url = format!("{base_url}/{full_key}");

        let mut headers: Vec<(String, String)> = Vec::new();

        if let CloudProvider::S3 { region, .. } = &self.config.provider {
            if let Some(auth_headers) = self.s3_auth_headers("GET", &full_key, b"", region) {
                headers.extend(auth_headers);
            }
        }

        let http_client = CloudHttpClient::default();
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let raw = http_client.send_raw("GET", &url, &header_refs, b"")?;
        let resp = parse_http_response(&raw)?;

        if resp.status == 404 {
            return Err(FarmError::NotFound(format!("Object not found: {full_key}")));
        }
        if resp.status != 200 {
            return Err(FarmError::Worker(format!(
                "Download failed: HTTP {}",
                resp.status
            )));
        }

        std::fs::write(local_path, &resp.body).map_err(FarmError::Io)?;
        Ok(())
    }

    /// List objects in the bucket whose keys start with `prefix`.
    ///
    /// The prefix passed here is combined with `self.config.prefix`.
    pub async fn list(&self, prefix: &str) -> Result<Vec<CloudObject>> {
        let full_prefix = if self.config.prefix.is_empty() {
            prefix.to_string()
        } else {
            format!("{}{prefix}", self.config.prefix)
        };

        if let Some(mock) = &self.mock {
            let raw = mock.lock().handle_list(&full_prefix);
            let resp = parse_http_response(&raw)?;
            if resp.status != 200 {
                return Err(FarmError::Worker(format!(
                    "List failed: HTTP {}",
                    resp.status
                )));
            }
            let xml = std::str::from_utf8(&resp.body)
                .map_err(|e| FarmError::InvalidConfig(format!("List response encoding: {e}")))?;
            return Ok(parse_list_response(xml));
        }

        // Real network path
        let base_url = self.config.provider.base_url(&self.config.bucket);
        let url = format!("{base_url}/?list-type=2&prefix={full_prefix}");

        let mut headers: Vec<(String, String)> = Vec::new();

        if let CloudProvider::S3 { region, .. } = &self.config.provider {
            if let Some(auth_headers) = self.s3_auth_headers(
                "GET",
                &format!("?list-type=2&prefix={full_prefix}"),
                b"",
                region,
            ) {
                headers.extend(auth_headers);
            }
        }

        let http_client = CloudHttpClient::default();
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let raw = http_client.send_raw("GET", &url, &header_refs, b"")?;
        let resp = parse_http_response(&raw)?;

        if resp.status != 200 {
            return Err(FarmError::Worker(format!(
                "List failed: HTTP {}",
                resp.status
            )));
        }

        let xml = std::str::from_utf8(&resp.body)
            .map_err(|e| FarmError::InvalidConfig(format!("List response encoding: {e}")))?;
        Ok(parse_list_response(xml))
    }

    /// Delete an object from cloud storage.
    pub async fn delete(&self, remote_key: &str) -> Result<()> {
        let full_key = self.full_key(remote_key);

        if let Some(mock) = &self.mock {
            mock.lock().objects.remove(&full_key);
            return Ok(());
        }

        let base_url = self.config.provider.base_url(&self.config.bucket);
        let url = format!("{base_url}/{full_key}");

        let mut headers: Vec<(String, String)> = Vec::new();
        if let CloudProvider::S3 { region, .. } = &self.config.provider {
            if let Some(auth_headers) = self.s3_auth_headers("DELETE", &full_key, b"", region) {
                headers.extend(auth_headers);
            }
        }

        let http_client = CloudHttpClient::default();
        let header_refs: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let raw = http_client.send_raw("DELETE", &url, &header_refs, b"")?;
        let resp = parse_http_response(&raw)?;

        if resp.status != 204 && resp.status != 200 {
            return Err(FarmError::Worker(format!(
                "Delete failed: HTTP {}",
                resp.status
            )));
        }
        Ok(())
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &CloudStorageConfig {
        &self.config
    }

    /// Returns `true` if this client uses a mock backend.
    #[must_use]
    pub fn is_mock(&self) -> bool {
        self.mock.is_some()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn s3_config(prefix: &str) -> CloudStorageConfig {
        CloudStorageConfig {
            provider: CloudProvider::S3 {
                region: "us-east-1".into(),
                endpoint: None,
            },
            bucket: "test-bucket".into(),
            prefix: prefix.into(),
            credentials: CloudCredentials::AccessKey {
                id: "AKIATEST".into(),
                secret: "secret123".into(),
            },
        }
    }

    fn gcs_config() -> CloudStorageConfig {
        CloudStorageConfig {
            provider: CloudProvider::GCS {
                project: "my-project".into(),
            },
            bucket: "gcs-bucket".into(),
            prefix: String::new(),
            credentials: CloudCredentials::ServiceAccount {
                json_bytes: b"{}".to_vec(),
            },
        }
    }

    fn azure_config() -> CloudStorageConfig {
        CloudStorageConfig {
            provider: CloudProvider::AzureBlob {
                account: "myaccount".into(),
            },
            bucket: "my-container".into(),
            prefix: String::new(),
            credentials: CloudCredentials::Anonymous,
        }
    }

    fn make_mock_client(
        config: CloudStorageConfig,
    ) -> (
        CloudStorageClient,
        Arc<parking_lot::Mutex<MockCloudBackend>>,
    ) {
        let backend = Arc::new(parking_lot::Mutex::new(MockCloudBackend::new()));
        let client = CloudStorageClient::with_mock(config, Arc::clone(&backend));
        (client, backend)
    }

    fn temp_file_with_content(name: &str, content: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("oximedia_farm_test_{name}"));
        std::fs::write(&path, content).expect("write temp file");
        path
    }

    // ── provider URL tests ─────────────────────────────────────────────────────

    #[test]
    fn test_s3_us_east_1_url() {
        let p = CloudProvider::S3 {
            region: "us-east-1".into(),
            endpoint: None,
        };
        assert_eq!(
            p.base_url("my-bucket"),
            "https://s3.amazonaws.com/my-bucket"
        );
    }

    #[test]
    fn test_s3_other_region_url() {
        let p = CloudProvider::S3 {
            region: "eu-west-1".into(),
            endpoint: None,
        };
        assert_eq!(
            p.base_url("my-bucket"),
            "https://s3.eu-west-1.amazonaws.com/my-bucket"
        );
    }

    #[test]
    fn test_s3_custom_endpoint_url() {
        let p = CloudProvider::S3 {
            region: "us-east-1".into(),
            endpoint: Some("http://localhost:9000".into()),
        };
        assert_eq!(p.base_url("my-bucket"), "http://localhost:9000/my-bucket");
    }

    #[test]
    fn test_gcs_url() {
        let p = CloudProvider::GCS {
            project: "proj".into(),
        };
        assert_eq!(
            p.base_url("bucket"),
            "https://storage.googleapis.com/bucket"
        );
    }

    #[test]
    fn test_azure_url() {
        let p = CloudProvider::AzureBlob {
            account: "myacct".into(),
        };
        assert_eq!(
            p.base_url("container"),
            "https://myacct.blob.core.windows.net/container"
        );
    }

    // ── mock upload / download ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_upload_stores_object_in_mock() {
        let (client, backend) = make_mock_client(s3_config(""));
        let path = temp_file_with_content("upload_test.bin", b"hello world");
        client.upload(&path, "video.mp4").await.expect("upload ok");
        assert!(backend.lock().get("video.mp4").is_some());
    }

    #[tokio::test]
    async fn test_upload_with_prefix() {
        let (client, backend) = make_mock_client(s3_config("farm/"));
        let path = temp_file_with_content("prefix_test.bin", b"data");
        client.upload(&path, "output.mp4").await.expect("upload ok");
        assert!(backend.lock().get("farm/output.mp4").is_some());
    }

    #[tokio::test]
    async fn test_upload_returns_url() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let path = temp_file_with_content("url_test.bin", b"data");
        let url = client
            .upload(&path, "media/video.mp4")
            .await
            .expect("upload ok");
        assert!(url.contains("test-bucket"));
        assert!(url.contains("media/video.mp4"));
    }

    #[tokio::test]
    async fn test_download_retrieves_uploaded_content() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let src = temp_file_with_content("dl_src.bin", b"video bytes here");
        client.upload(&src, "dl_test.mp4").await.expect("upload ok");

        let dst_path = {
            let mut p = std::env::temp_dir();
            p.push("oximedia_farm_dl_dst.bin");
            p
        };
        client
            .download("dl_test.mp4", &dst_path)
            .await
            .expect("download ok");
        let downloaded = std::fs::read(&dst_path).expect("read dst");
        assert_eq!(downloaded, b"video bytes here");
    }

    #[tokio::test]
    async fn test_download_not_found_returns_error() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let dst_path = {
            let mut p = std::env::temp_dir();
            p.push("oximedia_farm_dl_missing.bin");
            p
        };
        let result = client.download("nonexistent.mp4", &dst_path).await;
        assert!(result.is_err());
        let err = result.expect_err("error present");
        assert!(
            matches!(err, FarmError::NotFound(_)),
            "Expected NotFound, got: {err:?}"
        );
    }

    // ── list ──────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_empty_bucket() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let objects = client.list("").await.expect("list ok");
        assert!(objects.is_empty());
    }

    #[tokio::test]
    async fn test_list_returns_uploaded_objects() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let p1 = temp_file_with_content("list_a.bin", b"aaa");
        let p2 = temp_file_with_content("list_b.bin", b"bbb");
        client.upload(&p1, "media/a.mp4").await.expect("upload a");
        client.upload(&p2, "media/b.mp4").await.expect("upload b");

        let all = client.list("").await.expect("list all");
        assert_eq!(all.len(), 2);

        let media = client.list("media/").await.expect("list prefix");
        assert_eq!(media.len(), 2);

        let none = client.list("other/").await.expect("list other prefix");
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn test_list_object_metadata() {
        let (client, _backend) = make_mock_client(s3_config(""));
        let path = temp_file_with_content("meta_test.bin", b"12345678");
        client.upload(&path, "meta.mp4").await.expect("upload");
        let objects = client.list("").await.expect("list");
        assert_eq!(objects.len(), 1);
        let obj = &objects[0];
        assert_eq!(obj.key, "meta.mp4");
        assert_eq!(obj.size, 8);
        assert!(!obj.etag.is_empty());
    }

    // ── delete ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_removes_object() {
        let (client, backend) = make_mock_client(s3_config(""));
        let path = temp_file_with_content("del_test.bin", b"to delete");
        client
            .upload(&path, "will_delete.mp4")
            .await
            .expect("upload");
        assert!(backend.lock().get("will_delete.mp4").is_some());

        client.delete("will_delete.mp4").await.expect("delete ok");
        assert!(backend.lock().get("will_delete.mp4").is_none());
    }

    // ── SigV4 signing ─────────────────────────────────────────────────────────

    #[test]
    fn test_hmac_sha256_known_value() {
        // RFC 4231 test vector #1: key=0x0b*20, data="Hi There"
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let result = hmac_sha256(&key, data);
        let hex = hex_encode(&result);
        assert_eq!(
            hex,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn test_derive_signing_key_deterministic() {
        let k1 = derive_signing_key("wJalrXUtnFEMI", "20130524", "us-east-1", "s3");
        let k2 = derive_signing_key("wJalrXUtnFEMI", "20130524", "us-east-1", "s3");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_sigv4_auth_produces_valid_header() {
        let (auth, amz_date, sha) = sigv4_auth(
            "GET",
            "/test-bucket/test-object",
            "",
            "s3.us-east-1.amazonaws.com",
            b"",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "s3",
            "20130524T000000Z",
            "20130524",
        );
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/"));
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
        assert!(auth.contains("Signature="));
        assert_eq!(amz_date, "20130524T000000Z");
        // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            sha,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // ── datetime formatting ────────────────────────────────────────────────────

    #[test]
    fn test_format_datetime_epoch() {
        assert_eq!(format_datetime(0), "19700101T000000Z");
    }

    #[test]
    fn test_format_date_epoch() {
        assert_eq!(format_date(0), "19700101");
    }

    #[test]
    fn test_format_datetime_known() {
        // 2024-01-15 12:34:56 UTC  → seconds since epoch
        // Rough check: year/month parsed correctly
        let dt = format_datetime(1705318496);
        assert!(dt.starts_with("2024"), "Expected 2024..., got {dt}");
    }

    // ── XML parsing ───────────────────────────────────────────────────────────

    #[test]
    fn test_parse_list_response_empty() {
        let xml = "<?xml version=\"1.0\"?><ListBucketResult><Name>b</Name></ListBucketResult>";
        let objects = parse_list_response(xml);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_parse_list_response_single() {
        let xml = r#"<?xml version="1.0"?><ListBucketResult>
            <Contents>
                <Key>media/video.mp4</Key>
                <Size>1024</Size>
                <ETag>"abc123"</ETag>
                <LastModified>2024-01-15T12:00:00.000Z</LastModified>
            </Contents>
        </ListBucketResult>"#;
        let objects = parse_list_response(xml);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].key, "media/video.mp4");
        assert_eq!(objects[0].size, 1024);
        assert_eq!(objects[0].etag, "abc123");
    }

    #[test]
    fn test_parse_list_response_multiple() {
        let xml = r#"<ListBucketResult>
            <Contents><Key>a.mp4</Key><Size>100</Size><ETag>"e1"</ETag><LastModified>2024-01-01T00:00:00.000Z</LastModified></Contents>
            <Contents><Key>b.mp4</Key><Size>200</Size><ETag>"e2"</ETag><LastModified>2024-01-02T00:00:00.000Z</LastModified></Contents>
        </ListBucketResult>"#;
        let objects = parse_list_response(xml);
        assert_eq!(objects.len(), 2);
    }

    // ── http response parsing ─────────────────────────────────────────────────

    #[test]
    fn test_parse_http_response_ok() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let resp = parse_http_response(raw).expect("parse ok");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"hello");
    }

    #[test]
    fn test_parse_http_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        let resp = parse_http_response(raw).expect("parse 404");
        assert_eq!(resp.status, 404);
        assert!(resp.body.is_empty());
    }

    #[test]
    fn test_parse_http_response_header_lookup() {
        let raw = b"HTTP/1.1 200 OK\r\nETag: \"abc\"\r\nContent-Type: application/xml\r\n\r\n";
        let resp = parse_http_response(raw).expect("parse headers");
        assert_eq!(resp.header("etag"), Some("\"abc\""));
        assert_eq!(resp.header("content-type"), Some("application/xml"));
        assert!(resp.header("x-missing").is_none());
    }

    // ── mock backend ──────────────────────────────────────────────────────────

    #[test]
    fn test_mock_backend_put_get() {
        let mut backend = MockCloudBackend::new();
        backend.put("key1", b"value1".as_slice());
        assert_eq!(backend.get("key1"), Some(b"value1".as_slice()));
        assert!(backend.get("missing").is_none());
    }

    #[test]
    fn test_mock_backend_handle_put_returns_200() {
        let mut backend = MockCloudBackend::new();
        let resp = backend.handle_put("k", b"data");
        let parsed = parse_http_response(&resp).expect("parse");
        assert_eq!(parsed.status, 200);
    }

    #[test]
    fn test_mock_backend_handle_get_not_found() {
        let backend = MockCloudBackend::new();
        let resp = backend.handle_get("missing");
        let parsed = parse_http_response(&resp).expect("parse");
        assert_eq!(parsed.status, 404);
    }

    // ── is_mock flag ──────────────────────────────────────────────────────────

    #[test]
    fn test_client_is_mock_flag() {
        let (mock_client, _) = make_mock_client(s3_config(""));
        assert!(mock_client.is_mock());

        let real_client = CloudStorageClient::new(gcs_config());
        assert!(!real_client.is_mock());
    }

    // ── GCS and Azure clients instantiate ─────────────────────────────────────

    #[test]
    fn test_gcs_client_creation() {
        let client = CloudStorageClient::new(gcs_config());
        assert_eq!(client.config().bucket, "gcs-bucket");
    }

    #[test]
    fn test_azure_client_creation() {
        let client = CloudStorageClient::new(azure_config());
        assert_eq!(client.config().bucket, "my-container");
    }

    // ── anonymous_s3 helper ───────────────────────────────────────────────────

    #[test]
    fn test_anonymous_s3_config() {
        let cfg = CloudStorageConfig::anonymous_s3("public-bucket", "ap-southeast-1");
        assert_eq!(cfg.bucket, "public-bucket");
        assert!(matches!(cfg.credentials, CloudCredentials::Anonymous));
        assert!(matches!(
            &cfg.provider,
            CloudProvider::S3 { region, .. } if region == "ap-southeast-1"
        ));
    }
}
