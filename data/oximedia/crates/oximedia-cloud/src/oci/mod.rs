//! Oracle Cloud Infrastructure (OCI) Object Storage provider.
//!
//! This module provides a structural stub for the OCI Object Storage API.
//! The design follows the same pattern as the AWS, Azure, and GCP providers
//! already present in this crate.
//!
//! ## Structure
//!
//! - [`OciCredentials`] — OCI API key / tenancy authentication.
//! - [`OciRegion`] — well-known OCI region identifiers.
//! - [`OciObjectStorageConfig`] — configuration for an OCI namespace + bucket.
//! - [`OciObjectStorage`] — implements [`CloudStorage`] for OCI Object Storage.
//!
//! ## Authentication
//!
//! OCI uses **API key authentication**: a PEM private key is used to sign
//! every HTTP request with the OCI Signature V1 scheme (HTTP signature with
//! SHA-256 body digest). The fields on [`OciCredentials`] map directly to the
//! OCI SDK configuration file entries.
//!
//! ## Example (struct construction only — no live calls)
//!
//! ```rust
//! use oximedia_cloud::oci::{OciCredentials, OciObjectStorageConfig, OciRegion};
//!
//! let creds = OciCredentials::new(
//!     "ocid1.tenancy.oc1..aaa".to_string(),
//!     "ocid1.user.oc1..bbb".to_string(),
//!     "aa:bb:cc:dd".to_string(),
//!     "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----".to_string(),
//! );
//!
//! let config = OciObjectStorageConfig::new(
//!     creds,
//!     OciRegion::EuFrankfurt1,
//!     "my-namespace".to_string(),
//!     "my-bucket".to_string(),
//! );
//! ```

use crate::error::{CloudError, Result};
use crate::types::{
    CloudStorage, DeleteResult, ListResult, ObjectInfo, ObjectMetadata, StorageClass, StorageStats,
    UploadOptions,
};

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use chrono::Utc;
use reqwest::{Client, StatusCode};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::sha2::{Digest, Sha256};
use rsa::signature::SignatureEncoding;
use rsa::RsaPrivateKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── OCI Credentials ───────────────────────────────────────────────────────────

/// Authentication credentials for OCI Object Storage.
///
/// These map directly to entries in the OCI `~/.oci/config` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OciCredentials {
    /// Tenancy OCID (`ocid1.tenancy.oc1..xxx`).
    pub tenancy_id: String,
    /// User OCID (`ocid1.user.oc1..xxx`).
    pub user_id: String,
    /// Fingerprint of the uploaded API public key (colon-separated hex pairs).
    pub fingerprint: String,
    /// PEM-encoded RSA private key corresponding to the uploaded public key.
    pub private_key_pem: String,
    /// Optional passphrase for an encrypted private key.
    pub private_key_passphrase: Option<String>,
}

impl OciCredentials {
    /// Create new OCI API key credentials.
    #[must_use]
    pub fn new(
        tenancy_id: impl Into<String>,
        user_id: impl Into<String>,
        fingerprint: impl Into<String>,
        private_key_pem: impl Into<String>,
    ) -> Self {
        Self {
            tenancy_id: tenancy_id.into(),
            user_id: user_id.into(),
            fingerprint: fingerprint.into(),
            private_key_pem: private_key_pem.into(),
            private_key_passphrase: None,
        }
    }

    /// Set an optional passphrase for an encrypted private key.
    #[must_use]
    pub fn with_passphrase(mut self, passphrase: impl Into<String>) -> Self {
        self.private_key_passphrase = Some(passphrase.into());
        self
    }

    /// Validate that required fields are non-empty.
    pub fn validate(&self) -> Result<()> {
        if self.tenancy_id.is_empty() {
            return Err(CloudError::InvalidConfig(
                "OCI tenancy_id is empty".to_string(),
            ));
        }
        if self.user_id.is_empty() {
            return Err(CloudError::InvalidConfig(
                "OCI user_id is empty".to_string(),
            ));
        }
        if self.fingerprint.is_empty() {
            return Err(CloudError::InvalidConfig(
                "OCI fingerprint is empty".to_string(),
            ));
        }
        if self.private_key_pem.is_empty() {
            return Err(CloudError::InvalidConfig(
                "OCI private_key_pem is empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ── OCI Regions ───────────────────────────────────────────────────────────────

/// Well-known OCI region identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OciRegion {
    /// US East (Ashburn)
    UsAshburn1,
    /// US West (Phoenix)
    UsPhoenix1,
    /// US West (San Jose)
    UsSanJose1,
    /// Canada Southeast (Toronto)
    CaTorontoMontreal1,
    /// UK South (London)
    UkLondon1,
    /// EU West (Amsterdam)
    EuAmsterdam1,
    /// EU Central (Frankfurt)
    EuFrankfurt1,
    /// EU South (Milan)
    EuMilan1,
    /// AP East (Tokyo)
    ApTokyo1,
    /// AP Southeast (Sydney)
    ApSydney1,
    /// AP Southeast (Singapore)
    ApSingapore1,
    /// AP South (Mumbai)
    ApMumbai1,
    /// South Africa (Johannesburg)
    AfJohannesburg1,
    /// Middle East (Dubai)
    MeDubai1,
    /// Custom / private region.
    Custom(String),
}

impl OciRegion {
    /// Returns the OCI region identifier string (e.g. `us-ashburn-1`).
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::UsAshburn1 => "us-ashburn-1",
            Self::UsPhoenix1 => "us-phoenix-1",
            Self::UsSanJose1 => "us-sanjose-1",
            Self::CaTorontoMontreal1 => "ca-toronto-1",
            Self::UkLondon1 => "uk-london-1",
            Self::EuAmsterdam1 => "eu-amsterdam-1",
            Self::EuFrankfurt1 => "eu-frankfurt-1",
            Self::EuMilan1 => "eu-milan-1",
            Self::ApTokyo1 => "ap-tokyo-1",
            Self::ApSydney1 => "ap-sydney-1",
            Self::ApSingapore1 => "ap-singapore-1",
            Self::ApMumbai1 => "ap-mumbai-1",
            Self::AfJohannesburg1 => "af-johannesburg-1",
            Self::MeDubai1 => "me-dubai-1",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Build the Object Storage endpoint URL for this region.
    ///
    /// Format: `https://objectstorage.{region}.oraclecloud.com`
    #[must_use]
    pub fn endpoint_url(&self) -> String {
        format!("https://objectstorage.{}.oraclecloud.com", self.id())
    }
}

impl std::fmt::Display for OciRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

// ── OCI Storage class mapping ─────────────────────────────────────────────────

/// OCI Object Storage tier.
///
/// OCI uses the term *storage tier* rather than *storage class*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OciStorageTier {
    /// Standard (hot) storage — low latency, higher cost.
    Standard,
    /// Infrequent access — reduced storage cost, retrieval fee applies.
    InfrequentAccess,
    /// Archive — lowest cost, long restore time (hours).
    Archive,
}

impl OciStorageTier {
    /// Map an [`OciStorageTier`] to the generic [`StorageClass`].
    #[must_use]
    pub fn to_storage_class(self) -> StorageClass {
        match self {
            Self::Standard => StorageClass::Standard,
            Self::InfrequentAccess => StorageClass::InfrequentAccess,
            Self::Archive => StorageClass::Glacier,
        }
    }

    /// Map a generic [`StorageClass`] to the closest [`OciStorageTier`].
    #[must_use]
    pub fn from_storage_class(class: StorageClass) -> Self {
        match class {
            StorageClass::Standard
            | StorageClass::ReducedRedundancy
            | StorageClass::IntelligentTiering => Self::Standard,
            StorageClass::InfrequentAccess | StorageClass::OneZoneIA => Self::InfrequentAccess,
            StorageClass::Glacier | StorageClass::DeepArchive => Self::Archive,
        }
    }

    /// OCI API tier name string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::InfrequentAccess => "InfrequentAccess",
            Self::Archive => "Archive",
        }
    }
}

// ── OCI Object Storage Configuration ─────────────────────────────────────────

/// Configuration for an OCI Object Storage bucket.
#[derive(Debug, Clone)]
pub struct OciObjectStorageConfig {
    /// OCI API key credentials.
    pub credentials: OciCredentials,
    /// OCI region.
    pub region: OciRegion,
    /// OCI namespace (also called the *tenancy name*).
    pub namespace: String,
    /// Bucket name.
    pub bucket: String,
    /// Default storage tier for new objects.
    pub default_tier: OciStorageTier,
    /// Optional request timeout in seconds.
    pub request_timeout_secs: Option<u64>,
}

impl OciObjectStorageConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(
        credentials: OciCredentials,
        region: OciRegion,
        namespace: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            credentials,
            region,
            namespace: namespace.into(),
            bucket: bucket.into(),
            default_tier: OciStorageTier::Standard,
            request_timeout_secs: Some(30),
        }
    }

    /// Override the default storage tier.
    #[must_use]
    pub fn with_default_tier(mut self, tier: OciStorageTier) -> Self {
        self.default_tier = tier;
        self
    }

    /// Override the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = Some(secs);
        self
    }

    /// Build the base endpoint URL for this configuration.
    #[must_use]
    pub fn endpoint_url(&self) -> String {
        self.region.endpoint_url()
    }

    /// Build the Object Storage service URL for the configured namespace.
    ///
    /// Format: `https://objectstorage.{region}.oraclecloud.com/n/{namespace}/b/{bucket}/o`
    #[must_use]
    pub fn object_base_url(&self) -> String {
        format!(
            "{}/n/{}/b/{}/o",
            self.region.endpoint_url(),
            urlencoding::encode(&self.namespace),
            urlencoding::encode(&self.bucket),
        )
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        self.credentials.validate()?;
        if self.namespace.is_empty() {
            return Err(CloudError::InvalidConfig(
                "OCI namespace is empty".to_string(),
            ));
        }
        if self.bucket.is_empty() {
            return Err(CloudError::InvalidConfig("OCI bucket is empty".to_string()));
        }
        Ok(())
    }
}

// ── OCI HTTP Signature helpers ────────────────────────────────────────────────

/// Returns the RFC 2822–style date string required by OCI HTTP Signature V1.
fn oci_date_header() -> String {
    // OCI expects the same format as the HTTP `Date` header (RFC 2822 / RFC 7231).
    Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

/// Compute `base64(SHA-256(data))` as required for the `x-content-sha256` header.
fn sha256_base64(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    BASE64.encode(digest)
}

/// Load an RSA private key from PEM (PKCS#8 or PKCS#1).
fn load_rsa_private_key(pem: &str) -> Result<RsaPrivateKey> {
    if let Ok(key) = RsaPrivateKey::from_pkcs8_pem(pem) {
        return Ok(key);
    }
    RsaPrivateKey::from_pkcs1_pem(pem).map_err(|e| {
        CloudError::Authentication(format!("OCI: failed to parse RSA private key: {e}"))
    })
}

/// Sign `signing_string` with the OCI HTTP Signature scheme (RSA-PKCS1-SHA256).
///
/// Returns the base64-encoded signature bytes.
fn rsa_sign_base64(pem: &str, signing_string: &str) -> Result<String> {
    let private_key = load_rsa_private_key(pem)?;
    let signing_key = rsa::pkcs1v15::SigningKey::<Sha256>::new(private_key);
    use rsa::signature::Signer;
    let signature = signing_key.sign(signing_string.as_bytes());
    Ok(BASE64.encode(signature.to_bytes()))
}

/// Build the OCI `Authorization` header value for a GET / HEAD / DELETE request
/// (no body — only `(request-target)`, `host`, `date` are signed).
fn oci_auth_header_no_body(
    creds: &OciCredentials,
    method: &str,
    path: &str,
    host: &str,
    date: &str,
) -> Result<String> {
    let request_target = format!("{} {}", method.to_lowercase(), path);
    let signing_string = format!("(request-target): {request_target}\nhost: {host}\ndate: {date}");
    let signature = rsa_sign_base64(&creds.private_key_pem, &signing_string)?;
    let key_id = format!(
        "{}/{}/{}",
        creds.tenancy_id, creds.user_id, creds.fingerprint
    );
    Ok(format!(
        r#"Signature version="1",headers="(request-target) host date",keyId="{key_id}",algorithm="rsa-sha256",signature="{signature}""#
    ))
}

/// Build the OCI `Authorization` header value for a PUT / POST request
/// (body present — additionally signs `x-content-sha256`, `content-type`).
fn oci_auth_header_with_body(
    creds: &OciCredentials,
    method: &str,
    path: &str,
    host: &str,
    date: &str,
    content_sha256: &str,
    content_type: &str,
) -> Result<String> {
    let request_target = format!("{} {}", method.to_lowercase(), path);
    let signing_string = format!(
        "(request-target): {request_target}\nhost: {host}\ndate: {date}\nx-content-sha256: {content_sha256}\ncontent-type: {content_type}"
    );
    let signature = rsa_sign_base64(&creds.private_key_pem, &signing_string)?;
    let key_id = format!(
        "{}/{}/{}",
        creds.tenancy_id, creds.user_id, creds.fingerprint
    );
    Ok(format!(
        r#"Signature version="1",headers="(request-target) host date x-content-sha256 content-type",keyId="{key_id}",algorithm="rsa-sha256",signature="{signature}""#
    ))
}

// ── OCI API response types ────────────────────────────────────────────────────

/// One item returned by the OCI `ListObjects` API.
#[derive(Debug, Deserialize)]
struct OciObjectSummary {
    name: String,
    size: Option<u64>,
    #[serde(rename = "timeModified")]
    time_modified: Option<String>,
    etag: Option<String>,
    #[serde(rename = "storageTier")]
    storage_tier: Option<String>,
}

/// OCI `ListObjects` response envelope.
#[derive(Debug, Deserialize)]
struct OciListObjectsResponse {
    objects: Vec<OciObjectSummary>,
    #[serde(rename = "nextStartWith")]
    next_start_with: Option<String>,
}

/// OCI `HeadObject` / `GetObject` response — parsed from response headers.
struct OciObjectHeaders {
    size: u64,
    last_modified: chrono::DateTime<Utc>,
    etag: Option<String>,
    content_type: Option<String>,
    storage_tier: Option<String>,
}

impl OciObjectHeaders {
    fn from_response(resp: &reqwest::Response) -> Self {
        let headers = resp.headers();

        let size = headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0u64);

        let last_modified = headers
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| chrono::DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let etag = headers
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').to_string());

        let content_type = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let storage_tier = headers
            .get("x-oci-storage-tier")
            .or_else(|| headers.get("storage-tier"))
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        Self {
            size,
            last_modified,
            etag,
            content_type,
            storage_tier,
        }
    }
}

// ── OCI Object Storage client ─────────────────────────────────────────────────

/// OCI Object Storage client.
///
/// Implements [`CloudStorage`] using the OCI Object Storage REST API with
/// HTTP Signature V1 authentication.
pub struct OciObjectStorage {
    config: OciObjectStorageConfig,
    client: Client,
}

impl OciObjectStorage {
    /// Create a new OCI Object Storage client.
    ///
    /// # Errors
    ///
    /// Returns [`CloudError::InvalidConfig`] if the configuration is invalid.
    pub fn new(config: OciObjectStorageConfig) -> Result<Self> {
        crate::tls_provider::install_default_crypto_provider();
        config.validate()?;
        let mut builder = Client::builder();
        if let Some(secs) = config.request_timeout_secs {
            builder = builder.timeout(std::time::Duration::from_secs(secs));
        }
        let client = builder
            .build()
            .map_err(|e| CloudError::InvalidConfig(format!("OCI reqwest client: {e}")))?;
        Ok(Self { config, client })
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &OciObjectStorageConfig {
        &self.config
    }

    /// Returns the base URL for objects in the configured bucket.
    #[must_use]
    pub fn object_base_url(&self) -> String {
        self.config.object_base_url()
    }

    /// Host header value derived from the region endpoint.
    fn host(&self) -> String {
        format!("objectstorage.{}.oraclecloud.com", self.config.region.id())
    }

    /// Build the URL for a specific object key.
    fn object_url(&self, key: &str) -> String {
        format!(
            "{}/{}",
            self.config.object_base_url(),
            urlencoding::encode(key)
        )
    }

    /// Build the URL path component (after the host) for a specific object key,
    /// used for signing.
    fn object_path(&self, key: &str) -> String {
        format!(
            "/n/{}/b/{}/o/{}",
            urlencoding::encode(&self.config.namespace),
            urlencoding::encode(&self.config.bucket),
            urlencoding::encode(key),
        )
    }

    /// Map an OCI storage tier string to `StorageClass`.
    fn from_oci_tier(tier: &str) -> StorageClass {
        match tier {
            "Standard" => StorageClass::Standard,
            "InfrequentAccess" => StorageClass::InfrequentAccess,
            "Archive" => StorageClass::Glacier,
            _ => StorageClass::Standard,
        }
    }
}

#[async_trait]
impl CloudStorage for OciObjectStorage {
    /// Upload an object using HTTP PUT.
    async fn upload(&self, key: &str, data: Bytes) -> Result<()> {
        tracing::debug!("OCI upload: bucket={} key={}", self.config.bucket, key);
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let content_type = "application/octet-stream";
        let body_bytes = data.as_ref();
        let content_sha256 = sha256_base64(body_bytes);
        let auth = oci_auth_header_with_body(
            &self.config.credentials,
            "PUT",
            &path,
            &host,
            &date,
            &content_sha256,
            content_type,
        )?;

        let response = self
            .client
            .put(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("x-content-sha256", &content_sha256)
            .header("Content-Type", content_type)
            .body(data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI upload failed (HTTP {status}): {body}"
            )));
        }

        Ok(())
    }

    /// Upload an object with extended options using HTTP PUT.
    async fn upload_with_options(
        &self,
        key: &str,
        data: Bytes,
        options: UploadOptions,
    ) -> Result<()> {
        tracing::debug!(
            "OCI upload_with_options: bucket={} key={}",
            self.config.bucket,
            key
        );
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let content_type = options
            .content_type
            .as_deref()
            .unwrap_or("application/octet-stream");
        let body_bytes = data.as_ref();
        let content_sha256 = sha256_base64(body_bytes);
        let auth = oci_auth_header_with_body(
            &self.config.credentials,
            "PUT",
            &path,
            &host,
            &date,
            &content_sha256,
            content_type,
        )?;

        // Determine the storage tier
        let tier = options
            .storage_class
            .map(|sc| OciStorageTier::from_storage_class(sc).as_str())
            .unwrap_or(self.config.default_tier.as_str());

        let mut req = self
            .client
            .put(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("x-content-sha256", &content_sha256)
            .header("Content-Type", content_type)
            .header("Storage-Tier", tier);

        // User-defined metadata via `opc-meta-*` headers
        for (k, v) in &options.metadata {
            req = req.header(format!("opc-meta-{k}"), v);
        }

        if let Some(cache_control) = options.cache_control {
            req = req.header("Cache-Control", cache_control);
        }

        let response = req.body(data).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI upload_with_options failed (HTTP {status}): {body}"
            )));
        }

        Ok(())
    }

    /// Download an object using HTTP GET.
    async fn download(&self, key: &str) -> Result<Bytes> {
        tracing::debug!("OCI download: bucket={} key={}", self.config.bucket, key);
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let auth = oci_auth_header_no_body(&self.config.credentials, "GET", &path, &host, &date)?;

        let response = self
            .client
            .get(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(CloudError::NotFound(format!("OCI object not found: {key}")));
        }
        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "OCI download failed (HTTP {status})"
            )));
        }

        Ok(response.bytes().await?)
    }

    /// Download a byte-range of an object using HTTP GET with a `Range` header.
    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let auth = oci_auth_header_no_body(&self.config.credentials, "GET", &path, &host, &date)?;

        let response = self
            .client
            .get(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("Range", format!("bytes={start}-{end}"))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "OCI download_range failed (HTTP {status})"
            )));
        }

        Ok(response.bytes().await?)
    }

    /// List all objects with the given prefix by exhausting OCI's `nextStartWith` pagination.
    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        let mut all_objects = Vec::new();
        let mut next_start: Option<String> = None;

        loop {
            let result = self
                .list_paginated(prefix, next_start.clone(), 1000)
                .await?;
            all_objects.extend(result.objects);
            if result.is_truncated {
                next_start = result.continuation_token;
            } else {
                break;
            }
        }

        Ok(all_objects)
    }

    /// Paginated list using `?prefix=&limit=&start=` query parameters.
    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult> {
        // OCI list path: /n/<ns>/b/<bucket>/o?prefix=...&limit=...&start=...
        let path = format!(
            "/n/{}/b/{}/o",
            urlencoding::encode(&self.config.namespace),
            urlencoding::encode(&self.config.bucket),
        );
        let host = self.host();
        let date = oci_date_header();

        let mut query = format!("prefix={}&limit={}", urlencoding::encode(prefix), max_keys);
        if let Some(ref token) = continuation_token {
            query.push_str(&format!("&start={}", urlencoding::encode(token)));
        }

        let url = format!("{}{path}?{query}", self.config.region.endpoint_url());
        let signing_path = format!("{path}?{query}");

        let auth =
            oci_auth_header_no_body(&self.config.credentials, "GET", &signing_path, &host, &date)?;

        let response = self
            .client
            .get(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI list_paginated failed (HTTP {status}): {body}"
            )));
        }

        let list_resp: OciListObjectsResponse = response.json().await.map_err(|e| {
            CloudError::Storage(format!("OCI list_paginated: failed to parse JSON: {e}"))
        })?;

        let objects = list_resp
            .objects
            .into_iter()
            .map(|obj| {
                let last_modified = obj
                    .time_modified
                    .as_deref()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(Utc::now);

                let storage_class = obj.storage_tier.as_deref().map(Self::from_oci_tier);

                ObjectInfo {
                    key: obj.name,
                    size: obj.size.unwrap_or(0),
                    last_modified,
                    etag: obj.etag,
                    storage_class,
                    content_type: None,
                }
            })
            .collect();

        let is_truncated = list_resp.next_start_with.is_some();
        Ok(ListResult {
            objects,
            continuation_token: list_resp.next_start_with,
            is_truncated,
            common_prefixes: Vec::new(),
        })
    }

    /// Delete an object using HTTP DELETE.
    async fn delete(&self, key: &str) -> Result<()> {
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let auth =
            oci_auth_header_no_body(&self.config.credentials, "DELETE", &path, &host, &date)?;

        let response = self
            .client
            .delete(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() && status != StatusCode::NOT_FOUND {
            return Err(CloudError::Storage(format!(
                "OCI delete failed (HTTP {status})"
            )));
        }

        Ok(())
    }

    /// Delete multiple objects sequentially (OCI has no native batch delete).
    async fn delete_batch(&self, keys: &[String]) -> Result<Vec<DeleteResult>> {
        let mut results = Vec::new();
        for key in keys {
            match self.delete(key).await {
                Ok(()) => results.push(DeleteResult {
                    key: key.clone(),
                    success: true,
                    error: None,
                }),
                Err(e) => results.push(DeleteResult {
                    key: key.clone(),
                    success: false,
                    error: Some(e.to_string()),
                }),
            }
        }
        Ok(results)
    }

    /// Retrieve object metadata using HTTP HEAD.
    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let auth = oci_auth_header_no_body(&self.config.credentials, "HEAD", &path, &host, &date)?;

        let response = self
            .client
            .head(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "OCI get_metadata failed (HTTP {status})"
            )));
        }

        let h = OciObjectHeaders::from_response(&response);
        let storage_class = h.storage_tier.as_deref().map(Self::from_oci_tier);

        // Collect user metadata from `opc-meta-*` response headers
        let mut user_metadata = HashMap::new();
        for (name, value) in response.headers() {
            if let Some(meta_key) = name.as_str().strip_prefix("opc-meta-") {
                if let Ok(v) = value.to_str() {
                    user_metadata.insert(meta_key.to_string(), v.to_string());
                }
            }
        }

        let info = ObjectInfo {
            key: key.to_string(),
            size: h.size,
            last_modified: h.last_modified,
            etag: h.etag,
            storage_class,
            content_type: h.content_type.clone(),
        };

        Ok(ObjectMetadata {
            info,
            user_metadata,
            system_metadata: HashMap::new(),
            tags: HashMap::new(),
            content_encoding: None,
            content_language: None,
            cache_control: None,
            content_disposition: None,
        })
    }

    /// Update object metadata by re-uploading an empty PUT to the metadata endpoint.
    ///
    /// OCI does not have a standalone "set metadata" verb; instead we use the
    /// `UpdateObject` (PUT with `?opc-meta-*`) pattern which POSTs a zero-byte
    /// body that preserves the original object but replaces metadata.
    async fn update_metadata(&self, key: &str, metadata: HashMap<String, String>) -> Result<()> {
        // OCI does not support PATCH/metadata-only update without re-uploading.
        // We download the object, then re-upload with the new metadata.
        // For large objects this is expensive; callers should prefer fine-grained
        // metadata from the start.  A production implementation would use
        // the "Rename Object" + metadata workaround or OCI SDK bulk-set API.
        let data = self.download(key).await?;
        let options = UploadOptions {
            metadata,
            content_type: None,
            storage_class: None,
            cache_control: None,
            content_encoding: None,
            content_disposition: None,
            tags: HashMap::new(),
            encryption: None,
            acl: None,
        };
        self.upload_with_options(key, data, options).await
    }

    /// Check whether an object exists using HTTP HEAD.
    async fn exists(&self, key: &str) -> Result<bool> {
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        let auth = oci_auth_header_no_body(&self.config.credentials, "HEAD", &path, &host, &date)?;

        let response = self
            .client
            .head(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .send()
            .await?;

        let status = response.status();
        Ok(status.is_success())
    }

    /// Server-side copy using the OCI `Copy Object` API (POST to `/actions/copyObject`).
    async fn copy(&self, src: &str, dst: &str) -> Result<()> {
        // OCI copy endpoint: POST /n/<ns>/b/<bucket>/actions/copyObject
        let path = format!(
            "/n/{}/b/{}/actions/copyObject",
            urlencoding::encode(&self.config.namespace),
            urlencoding::encode(&self.config.bucket),
        );
        let host = self.host();
        let date = oci_date_header();
        let content_type = "application/json";

        let body = serde_json::json!({
            "sourceObjectName": src,
            "destinationRegion": self.config.region.id(),
            "destinationNamespace": self.config.namespace,
            "destinationBucket": self.config.bucket,
            "destinationObjectName": dst,
        });
        let body_bytes = body.to_string();
        let body_bytes_ref = body_bytes.as_bytes();
        let content_sha256 = sha256_base64(body_bytes_ref);

        let auth = oci_auth_header_with_body(
            &self.config.credentials,
            "POST",
            &path,
            &host,
            &date,
            &content_sha256,
            content_type,
        )?;

        let url = format!("{}{path}", self.config.region.endpoint_url());

        let response = self
            .client
            .post(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("x-content-sha256", &content_sha256)
            .header("Content-Type", content_type)
            .body(body_bytes)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let resp_body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI copy failed (HTTP {status}): {resp_body}"
            )));
        }

        Ok(())
    }

    /// Generate a presigned download URL using OCI Pre-Authenticated Requests (PAR).
    ///
    /// Posts to `/n/<ns>/b/<bucket>/p/` to create a PAR and returns its `accessUri`.
    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        self.create_par(key, "ObjectRead", expires_in_secs).await
    }

    /// Generate a presigned upload URL using OCI Pre-Authenticated Requests (PAR).
    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        self.create_par(key, "ObjectWrite", expires_in_secs).await
    }

    /// Change the storage tier of an object by re-uploading with a new `Storage-Tier` header.
    async fn set_storage_class(&self, key: &str, class: StorageClass) -> Result<()> {
        let tier = OciStorageTier::from_storage_class(class);
        let url = self.object_url(key);
        let path = self.object_path(key);
        let host = self.host();
        let date = oci_date_header();
        // OCI `UpdateObjectStorageTier` endpoint: POST with JSON body
        let update_path = format!(
            "/n/{}/b/{}/actions/updateObjectStorageTier",
            urlencoding::encode(&self.config.namespace),
            urlencoding::encode(&self.config.bucket),
        );
        let content_type = "application/json";
        let body = serde_json::json!({
            "objectName": key,
            "storageTier": tier.as_str(),
        });
        let body_str = body.to_string();
        let content_sha256 = sha256_base64(body_str.as_bytes());
        let auth = oci_auth_header_with_body(
            &self.config.credentials,
            "POST",
            &update_path,
            &host,
            &date,
            &content_sha256,
            content_type,
        )?;

        // Suppress unused variable warning — url is used for context
        let _ = url;
        let api_url = format!("{}{update_path}", self.config.region.endpoint_url());

        let response = self
            .client
            .post(&api_url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("x-content-sha256", &content_sha256)
            .header("Content-Type", content_type)
            .body(body_str)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let resp_body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI set_storage_class failed (HTTP {status}): {resp_body}"
            )));
        }

        // Suppress unused path variable
        let _ = path;

        Ok(())
    }

    /// Compute storage statistics for the given prefix by listing all objects.
    async fn get_stats(&self, prefix: &str) -> Result<StorageStats> {
        let objects = self.list(prefix).await?;
        let mut stats = StorageStats::default();

        for obj in objects {
            stats.total_size += obj.size;
            stats.object_count += 1;

            if let Some(class) = obj.storage_class {
                let class_name = format!("{class}");
                *stats.size_by_class.entry(class_name.clone()).or_insert(0) += obj.size;
                *stats.count_by_class.entry(class_name).or_insert(0) += 1;
            }
        }

        Ok(stats)
    }
}

/// Pre-Authenticated Request (PAR) response from OCI.
#[derive(Debug, Deserialize)]
struct OciParResponse {
    #[serde(rename = "accessUri")]
    access_uri: String,
}

impl OciObjectStorage {
    /// Create a Pre-Authenticated Request (PAR) for the given object and access type.
    ///
    /// `access_type` must be one of `"ObjectRead"`, `"ObjectWrite"`, `"ObjectReadWrite"`.
    async fn create_par(
        &self,
        key: &str,
        access_type: &str,
        expires_in_secs: u64,
    ) -> Result<String> {
        let path = format!(
            "/n/{}/b/{}/p/",
            urlencoding::encode(&self.config.namespace),
            urlencoding::encode(&self.config.bucket),
        );
        let host = self.host();
        let date = oci_date_header();
        let content_type = "application/json";

        let expiry = Utc::now() + chrono::Duration::seconds(expires_in_secs as i64);
        let expiry_str = expiry.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let body = serde_json::json!({
            "name": format!("par-{key}-{expires_in_secs}"),
            "objectName": key,
            "accessType": access_type,
            "timeExpires": expiry_str,
        });
        let body_str = body.to_string();
        let content_sha256 = sha256_base64(body_str.as_bytes());

        let auth = oci_auth_header_with_body(
            &self.config.credentials,
            "POST",
            &path,
            &host,
            &date,
            &content_sha256,
            content_type,
        )?;

        let url = format!("{}{path}", self.config.region.endpoint_url());

        let response = self
            .client
            .post(&url)
            .header("host", &host)
            .header("date", &date)
            .header("Authorization", auth)
            .header("x-content-sha256", &content_sha256)
            .header("Content-Type", content_type)
            .body(body_str)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let resp_body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "OCI PAR creation failed (HTTP {status}): {resp_body}"
            )));
        }

        let par: OciParResponse = response.json().await.map_err(|e| {
            CloudError::Storage(format!("OCI PAR: failed to parse JSON response: {e}"))
        })?;

        // OCI returns an accessUri relative to the base endpoint
        let full_url = if par.access_uri.starts_with("http") {
            par.access_uri
        } else {
            format!("{}{}", self.config.region.endpoint_url(), par.access_uri)
        };

        Ok(full_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_creds() -> OciCredentials {
        OciCredentials::new(
            "ocid1.tenancy.oc1..aaa",
            "ocid1.user.oc1..bbb",
            "aa:bb:cc:dd",
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----",
        )
    }

    // ── OciCredentials ───────────────────────────────────────────────────────

    #[test]
    fn test_oci_credentials_validate_ok() {
        assert!(make_creds().validate().is_ok());
    }

    #[test]
    fn test_oci_credentials_validate_empty_tenancy() {
        let creds = OciCredentials::new("", "user", "fp", "key");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_oci_credentials_validate_empty_user() {
        let creds = OciCredentials::new("tenancy", "", "fp", "key");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_oci_credentials_validate_empty_fingerprint() {
        let creds = OciCredentials::new("tenancy", "user", "", "key");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_oci_credentials_validate_empty_key() {
        let creds = OciCredentials::new("tenancy", "user", "fp", "");
        assert!(creds.validate().is_err());
    }

    #[test]
    fn test_oci_credentials_with_passphrase() {
        let creds = make_creds().with_passphrase("s3cr3t");
        assert_eq!(creds.private_key_passphrase, Some("s3cr3t".to_string()));
    }

    // ── OciRegion ────────────────────────────────────────────────────────────

    #[test]
    fn test_oci_region_id_strings() {
        assert_eq!(OciRegion::UsAshburn1.id(), "us-ashburn-1");
        assert_eq!(OciRegion::EuFrankfurt1.id(), "eu-frankfurt-1");
        assert_eq!(OciRegion::ApTokyo1.id(), "ap-tokyo-1");
    }

    #[test]
    fn test_oci_region_custom_id() {
        let r = OciRegion::Custom("us-gov-chicago-1".to_string());
        assert_eq!(r.id(), "us-gov-chicago-1");
    }

    #[test]
    fn test_oci_region_endpoint_url() {
        let url = OciRegion::UsAshburn1.endpoint_url();
        assert_eq!(url, "https://objectstorage.us-ashburn-1.oraclecloud.com");
    }

    #[test]
    fn test_oci_region_display() {
        assert_eq!(OciRegion::UkLondon1.to_string(), "uk-london-1");
    }

    // ── OciStorageTier ────────────────────────────────────────────────────────

    #[test]
    fn test_oci_storage_tier_to_storage_class() {
        assert_eq!(
            OciStorageTier::Standard.to_storage_class(),
            StorageClass::Standard
        );
        assert_eq!(
            OciStorageTier::InfrequentAccess.to_storage_class(),
            StorageClass::InfrequentAccess
        );
        assert_eq!(
            OciStorageTier::Archive.to_storage_class(),
            StorageClass::Glacier
        );
    }

    #[test]
    fn test_oci_storage_tier_from_storage_class() {
        assert_eq!(
            OciStorageTier::from_storage_class(StorageClass::Standard),
            OciStorageTier::Standard
        );
        assert_eq!(
            OciStorageTier::from_storage_class(StorageClass::InfrequentAccess),
            OciStorageTier::InfrequentAccess
        );
        assert_eq!(
            OciStorageTier::from_storage_class(StorageClass::DeepArchive),
            OciStorageTier::Archive
        );
    }

    #[test]
    fn test_oci_storage_tier_as_str() {
        assert_eq!(OciStorageTier::Standard.as_str(), "Standard");
        assert_eq!(
            OciStorageTier::InfrequentAccess.as_str(),
            "InfrequentAccess"
        );
        assert_eq!(OciStorageTier::Archive.as_str(), "Archive");
    }

    // ── OciObjectStorageConfig ────────────────────────────────────────────────

    #[test]
    fn test_oci_config_validate_ok() {
        let config = OciObjectStorageConfig::new(
            make_creds(),
            OciRegion::UsAshburn1,
            "my-namespace",
            "my-bucket",
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_oci_config_validate_empty_namespace() {
        let config = OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "", "bucket");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_oci_config_validate_empty_bucket() {
        let config = OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "ns", "");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_oci_config_endpoint_url() {
        let config =
            OciObjectStorageConfig::new(make_creds(), OciRegion::EuFrankfurt1, "ns", "bucket");
        assert_eq!(
            config.endpoint_url(),
            "https://objectstorage.eu-frankfurt-1.oraclecloud.com"
        );
    }

    #[test]
    fn test_oci_config_object_base_url() {
        let config =
            OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "myns", "mybucket");
        let url = config.object_base_url();
        assert!(
            url.contains("/n/myns/b/mybucket/o"),
            "unexpected url: {url}"
        );
    }

    #[test]
    fn test_oci_config_with_default_tier() {
        let config = OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "ns", "b")
            .with_default_tier(OciStorageTier::Archive);
        assert_eq!(config.default_tier, OciStorageTier::Archive);
    }

    // ── OciObjectStorage ─────────────────────────────────────────────────────

    #[test]
    fn test_oci_new_rejects_invalid_config() {
        let bad_creds = OciCredentials::new("", "user", "fp", "key");
        let config = OciObjectStorageConfig::new(bad_creds, OciRegion::UsAshburn1, "ns", "b");
        assert!(OciObjectStorage::new(config).is_err());
    }

    #[test]
    fn test_oci_new_accepts_valid_config() {
        let config =
            OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "ns", "bucket");
        assert!(OciObjectStorage::new(config).is_ok());
    }

    #[test]
    fn test_oci_object_url() {
        let config =
            OciObjectStorageConfig::new(make_creds(), OciRegion::UsAshburn1, "myns", "mybucket");
        let storage = OciObjectStorage::new(config).expect("new ok");
        let url = storage.object_url("path/to/obj.mp4");
        assert!(url.contains("/n/myns/b/mybucket/o/"), "url: {url}");
        assert!(url.ends_with("path%2Fto%2Fobj.mp4"), "url: {url}");
    }

    #[test]
    fn test_oci_host() {
        let config = OciObjectStorageConfig::new(make_creds(), OciRegion::EuFrankfurt1, "ns", "b");
        let storage = OciObjectStorage::new(config).expect("new ok");
        assert_eq!(
            storage.host(),
            "objectstorage.eu-frankfurt-1.oraclecloud.com"
        );
    }

    #[test]
    fn test_sha256_base64_known_value() {
        // SHA-256 of empty string = e3b0c44298fc1c149afbf4c8996fb924...
        let result = sha256_base64(b"");
        // base64 of the 32-byte SHA-256 of ""
        assert_eq!(result, "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=");
    }

    #[test]
    fn test_oci_date_header_format() {
        let date = oci_date_header();
        // Should look like "Mon, 04 Nov 2024 10:00:00 GMT"
        assert!(
            date.ends_with("GMT"),
            "date header should end with GMT: {date}"
        );
        assert!(date.len() > 20, "date header too short: {date}");
    }

    #[test]
    fn test_oci_auth_header_no_body_structure() {
        let creds = OciCredentials::new(
            "ocid1.tenancy.oc1..test",
            "ocid1.user.oc1..test",
            "ab:cd:ef",
            // A real RSA key is needed for signing; using a placeholder that fails gracefully.
            "INVALID_KEY",
        );
        // Signing will fail with invalid PEM, but we can test that the error is returned.
        let result = oci_auth_header_no_body(
            &creds,
            "GET",
            "/n/ns/b/bkt/o/key",
            "host.com",
            "Mon, 01 Jan 2024 00:00:00 GMT",
        );
        assert!(result.is_err(), "should fail with invalid PEM key");
    }

    #[test]
    fn test_oci_from_oci_tier_mapping() {
        assert_eq!(
            OciObjectStorage::from_oci_tier("Standard"),
            StorageClass::Standard
        );
        assert_eq!(
            OciObjectStorage::from_oci_tier("InfrequentAccess"),
            StorageClass::InfrequentAccess
        );
        assert_eq!(
            OciObjectStorage::from_oci_tier("Archive"),
            StorageClass::Glacier
        );
        assert_eq!(
            OciObjectStorage::from_oci_tier("Unknown"),
            StorageClass::Standard
        );
    }
}
