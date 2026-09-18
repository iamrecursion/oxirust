//! Azure Blob Storage implementation using the REST API with Shared Key
//! (HMAC-SHA256) authentication.
//!
//! # Why not the `azure_storage_blob` SDK crate?
//!
//! The workspace's Azure SDK track (`azure_core`/`azure_storage_blob` 1.0)
//! authenticates exclusively via Microsoft Entra ID (OAuth2 `TokenCredential`,
//! e.g. `azure_identity::DeveloperToolsCredential`/`ClientSecretCredential`).
//! It has **no** account-name/account-key "Shared Key" credential type at
//! all — that concept was dropped from the generated 1.0 client. This
//! crate's [`UnifiedConfig`] (shared across the S3/Azure/GCS backends) only
//! carries a flat `access_key`/`secret_key` pair, matching the classic
//! storage-account-name + storage-account-key model, so wiring it through
//! the new SDK would require inventing a parallel Entra ID configuration
//! surface with no equivalent in `UnifiedConfig`.
//!
//! Instead, this module talks to the [Azure Blob Service REST
//! API](https://learn.microsoft.com/en-us/rest/api/storageservices/blob-service-rest-api)
//! directly over `reqwest`, signing each request with [Shared Key Lite
//! authentication](https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key),
//! following the same approach already proven in
//! `oximedia-cloud::azure::blob::AzureBlobStorage`. This keeps the `azure`
//! feature 100% Pure Rust (`reqwest` + `hmac` + `sha2` + `base64`, no C/C++)
//! and compiling against a real, credentialed backend rather than a stub.
//!
//! Known scope limits (documented, not silently dropped): `UploadOptions`'s
//! `cache_control`/`content_encoding`/`storage_class`/`encryption`/`acl`
//! fields are not yet wired to Azure blob properties — only `content_type`
//! and `metadata` (as `x-ms-meta-*`) are honored on upload, to avoid
//! extending the Shared Key `CanonicalizedHeaders` signing surface without
//! a way to validate it against a live account. `ListOptions::delimiter`
//! (hierarchical listing) is not implemented; listing is always flat.

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ListOptions, ListResult, ObjectMetadata, Result,
    StorageError, UnifiedConfig, UploadOptions,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info};

/// Azure Blob Service REST API version used for every request in this module.
const API_VERSION: &str = "2021-06-08";

/// Minimum block size (256 KiB) for staged block-blob uploads.
const MIN_BLOCK_SIZE: usize = 256 * 1024;

/// Maximum block size (100 MiB) for staged block-blob uploads.
const MAX_BLOCK_SIZE: usize = 100 * 1024 * 1024;

/// Default block size (4 MiB) for staged block-blob uploads.
const DEFAULT_BLOCK_SIZE: usize = 4 * 1024 * 1024;

/// Maximum number of blocks per blob (Azure Blob Service limit).
const MAX_BLOCKS: usize = 50_000;

/// Uploads at or above this size (or with an unknown size) use staged
/// block-blob upload instead of a single `Put Blob` request.
const BLOCK_UPLOAD_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Process-wide guard ensuring the Pure-Rust `rustls` crypto provider is
/// installed at most once.
static INSTALL_CRYPTO_PROVIDER: Once = Once::new();

/// Installs the Pure-Rust [`rustls-rustcrypto`](https://docs.rs/rustls-rustcrypto)
/// crypto provider as the process-wide default `rustls` `CryptoProvider`.
///
/// The workspace builds `reqwest`/`rustls` with the `rustls-no-provider`
/// feature to stay 100% Pure Rust (no `aws-lc-sys`/`ring` C or assembly
/// code), which means a provider must be installed before the first TLS
/// connection is opened. Idempotent and safe to call from multiple entry
/// points (only installs once per process, guarded by [`Once`]); if another
/// crate in the same process already installed a default provider first,
/// that is not an error for us, so the result is discarded.
fn install_crypto_provider() {
    INSTALL_CRYPTO_PROVIDER.call_once(|| {
        let _ = rustls_rustcrypto::provider().install_default();
    });
}

/// Percent-encode `s` for safe use as a URL path or query-parameter value.
///
/// RFC 3986 unreserved characters (`A-Za-z0-9-_.~`) pass through unescaped;
/// every other byte (including UTF-8 continuation bytes) is escaped as
/// `%XX`. This is a small hand-rolled helper rather than a new dependency —
/// it only needs to guarantee round-trippable, unambiguous encoding, not
/// match any particular third-party crate's exact output.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

/// Extract the inner text of the first `<tag>...</tag>` occurrence in `xml`.
///
/// Returns `None` for a missing or self-closing (`<tag/>`) element.
fn extract_xml_inner(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].to_string())
}

/// Parse the Azure Blob Storage XML `List Blobs` response.
///
/// Returns `(objects, next_marker)`; `next_marker` is `Some` only when the
/// response was paginated (truncated).
fn parse_list_blobs_xml(xml: &str) -> (Vec<ObjectMetadata>, Option<String>) {
    let mut objects = Vec::new();
    let next_marker = extract_xml_inner(xml, "NextMarker").filter(|m| !m.is_empty());

    let mut remaining = xml;
    while let Some(blob_start) = remaining.find("<Blob>") {
        let after_open = &remaining[blob_start + "<Blob>".len()..];
        let blob_end = after_open.find("</Blob>").unwrap_or(after_open.len());
        let blob_xml = &after_open[..blob_end];

        let name = extract_xml_inner(blob_xml, "Name").unwrap_or_default();
        let size = extract_xml_inner(blob_xml, "Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let last_modified_str = extract_xml_inner(blob_xml, "Last-Modified").unwrap_or_default();
        let last_modified = DateTime::parse_from_rfc2822(&last_modified_str)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        let etag =
            extract_xml_inner(blob_xml, "Etag").or_else(|| extract_xml_inner(blob_xml, "ETag"));
        let content_type = extract_xml_inner(blob_xml, "Content-Type");

        if !name.is_empty() {
            objects.push(ObjectMetadata {
                key: name,
                size,
                content_type,
                last_modified,
                etag,
                metadata: HashMap::new(),
                storage_class: None,
            });
        }

        remaining = &after_open[blob_end..];
    }

    (objects, next_marker)
}

/// Azure Blob Storage access tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessTier {
    /// Frequently accessed data.
    Hot,
    /// Infrequently accessed data (stored for at least 30 days).
    Cool,
    /// Rarely accessed data (stored for at least 180 days).
    Archive,
}

impl AccessTier {
    fn as_str(&self) -> &'static str {
        match self {
            AccessTier::Hot => "Hot",
            AccessTier::Cool => "Cool",
            AccessTier::Archive => "Archive",
        }
    }

    #[allow(dead_code)]
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hot" => Some(AccessTier::Hot),
            "cool" => Some(AccessTier::Cool),
            "archive" => Some(AccessTier::Archive),
            _ => None,
        }
    }
}

/// Azure Blob Storage client, authenticated via Shared Key (HMAC-SHA256).
pub struct AzureStorage {
    client: reqwest::Client,
    account_name: String,
    container: String,
    /// Raw bytes of the base64-decoded storage account access key.
    account_key: Vec<u8>,
    _config: UnifiedConfig,
}

impl AzureStorage {
    /// Create a new Azure storage client from configuration.
    ///
    /// `config.access_key` must hold the storage account name and
    /// `config.secret_key` the base64-encoded storage account key (as shown
    /// in the Azure Portal), matching [`UnifiedConfig::azure`].
    pub async fn new(config: UnifiedConfig) -> Result<Self> {
        install_crypto_provider();

        let account_name = config
            .access_key
            .clone()
            .ok_or_else(|| StorageError::InvalidConfig("Account name required for Azure".into()))?;

        let account_key_b64 = config
            .secret_key
            .clone()
            .ok_or_else(|| StorageError::InvalidConfig("Account key required for Azure".into()))?;

        let account_key = BASE64.decode(&account_key_b64).map_err(|e| {
            StorageError::InvalidConfig(format!("Invalid Azure account key (expected base64): {e}"))
        })?;

        let client = reqwest::Client::builder().build().map_err(|e| {
            StorageError::ProviderError(format!("Failed to build Azure HTTP client: {e}"))
        })?;

        Ok(Self {
            client,
            account_name,
            container: config.bucket.clone(),
            account_key,
            _config: config,
        })
    }

    /// Base URL for the container.
    fn container_url(&self) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}",
            self.account_name, self.container
        )
    }

    /// Fully-qualified URL for a specific blob.
    fn blob_url(&self, key: &str) -> String {
        format!("{}/{}", self.container_url(), key)
    }

    /// Current time formatted as required by the `x-ms-date` header.
    fn ms_date() -> String {
        Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
    }

    /// Compute an HMAC-SHA256 signature over `string_to_sign`, base64-encoded.
    fn sign_string(&self, string_to_sign: &str) -> Result<String> {
        use hmac::{Hmac, KeyInit, Mac};
        type HmacSha256 = Hmac<sha2::Sha256>;

        let mut mac = HmacSha256::new_from_slice(&self.account_key)
            .map_err(|e| StorageError::AuthenticationError(format!("HMAC init failed: {e}")))?;
        mac.update(string_to_sign.as_bytes());
        Ok(BASE64.encode(mac.finalize().into_bytes()))
    }

    /// Build the `x-ms-*` `CanonicalizedHeaders` string (alphabetically
    /// sorted, each line `name:value\n`) for a request that also carries
    /// `extra` (already-lowercased) `x-ms-*` header name/value pairs beyond
    /// the always-present `x-ms-date` and `x-ms-version`.
    fn canonicalized_ms_headers(ms_date: &str, extra: &[(String, String)]) -> String {
        let mut all: Vec<(String, String)> = extra.to_vec();
        all.push(("x-ms-date".to_string(), ms_date.to_string()));
        all.push(("x-ms-version".to_string(), API_VERSION.to_string()));
        all.sort_by(|a, b| a.0.cmp(&b.0));

        let mut out = String::new();
        for (k, v) in &all {
            out.push_str(k);
            out.push(':');
            out.push_str(v);
            out.push('\n');
        }
        out
    }

    /// Build an `Authorization: SharedKeyLite` header value.
    ///
    /// `canonicalized_headers` must already be sorted `"header:value\n"`
    /// lines (see [`Self::canonicalized_ms_headers`]); `canonicalized_resource`
    /// must start with `"/accountname/containername[/blobname]"` and
    /// optionally contain sorted query-parameter lines.
    fn build_auth_header(
        &self,
        method: &str,
        content_type: Option<&str>,
        canonicalized_headers: &str,
        canonicalized_resource: &str,
    ) -> Result<String> {
        let content_type_str = content_type.unwrap_or("");
        let string_to_sign = format!(
            "{method}\n\n{content_type_str}\n\n{canonicalized_headers}{canonicalized_resource}",
        );
        let signature = self.sign_string(&string_to_sign)?;
        Ok(format!("SharedKeyLite {}:{}", self.account_name, signature))
    }

    /// Build the canonicalized resource string for a blob, with an optional
    /// single `name:value` query component (e.g. `"comp:metadata"`).
    fn blob_canonicalized_resource(&self, blob_name: &str, query: Option<&str>) -> String {
        let base = format!("/{}/{}/{}", self.account_name, self.container, blob_name);
        match query {
            Some(q) if !q.is_empty() => format!("{base}\n{q}"),
            _ => base,
        }
    }

    /// Build the canonicalized resource string for the container itself.
    fn container_canonicalized_resource(&self, query: Option<&str>) -> String {
        let base = format!("/{}/{}", self.account_name, self.container);
        match query {
            Some(q) if !q.is_empty() => format!("{base}\n{q}"),
            _ => base,
        }
    }

    /// Sign and send a request carrying no body and no extra `x-ms-*`
    /// headers beyond date/version (used by HEAD/DELETE/simple-GET calls).
    async fn signed_request(
        &self,
        method: reqwest::Method,
        url: &str,
        canonicalized_resource: &str,
    ) -> Result<reqwest::Response> {
        let ms_date = Self::ms_date();
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &[]);
        let auth = self.build_auth_header(
            method.as_str(),
            None,
            &canonicalized_headers,
            canonicalized_resource,
        )?;

        self.client
            .request(method, url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", ms_date)
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Azure request failed: {e}")))
    }

    /// Stage one block of a block-blob upload via `Put Block`.
    async fn put_block(&self, blob_name: &str, block_id_b64: &str, data: Vec<u8>) -> Result<()> {
        let url = format!(
            "{}?comp=block&blockid={}",
            self.blob_url(blob_name),
            percent_encode(block_id_b64)
        );
        let ms_date = Self::ms_date();
        let content_len = data.len() as u64;
        let canon_resource = self.blob_canonicalized_resource(blob_name, Some("comp:block"));
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &[]);
        let auth = self.build_auth_header("PUT", None, &canonicalized_headers, &canon_resource)?;

        let response = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", ms_date)
            .header("Content-Length", content_len)
            .body(data)
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Put Block request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to upload block for '{blob_name}' (HTTP {status}): {body}"
            )));
        }
        Ok(())
    }

    /// Commit a staged block list via `Put Block List`, optionally setting
    /// blob-level `Content-Type` and user metadata.
    async fn put_block_list(
        &self,
        blob_name: &str,
        block_ids_b64: &[String],
        content_type: Option<&str>,
        metadata: &HashMap<String, String>,
    ) -> Result<String> {
        let url = format!("{}?comp=blocklist", self.blob_url(blob_name));
        let ms_date = Self::ms_date();

        let mut body = String::from(r#"<?xml version="1.0" encoding="utf-8"?><BlockList>"#);
        for id in block_ids_b64 {
            body.push_str(&format!("<Uncommitted>{id}</Uncommitted>"));
        }
        body.push_str("</BlockList>");
        let body_bytes = body.into_bytes();
        let content_len = body_bytes.len() as u64;

        let mut extra: Vec<(String, String)> = Vec::new();
        if let Some(ct) = content_type {
            extra.push(("x-ms-blob-content-type".to_string(), ct.to_string()));
        }
        for (k, v) in metadata {
            extra.push((format!("x-ms-meta-{}", k.to_lowercase()), v.clone()));
        }

        let canon_resource = self.blob_canonicalized_resource(blob_name, Some("comp:blocklist"));
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &extra);
        let auth = self.build_auth_header("PUT", None, &canonicalized_headers, &canon_resource)?;

        let mut request = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", &ms_date)
            .header("Content-Length", content_len);
        for (k, v) in &extra {
            request = request.header(k, v);
        }

        let response = request.body(body_bytes).send().await.map_err(|e| {
            StorageError::NetworkError(format!("Put Block List request failed: {e}"))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to commit block list for '{blob_name}' (HTTP {status}): {body}"
            )));
        }

        let etag = response
            .headers()
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        Ok(etag)
    }

    /// Upload via staged blocks (`Put Block` + `Put Block List`), used for
    /// large or unknown-size streams.
    async fn upload_blocks(
        &self,
        blob_name: &str,
        stream: ByteStream,
        total_size: u64,
        options: &UploadOptions,
    ) -> Result<String> {
        info!("Starting block blob upload for: {}", blob_name);

        let block_size = calculate_block_size(total_size);
        let mut block_ids = Vec::new();
        let mut block_id: u64 = 0;
        let mut stream = stream;
        let mut buffer = Vec::new();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= block_size {
                let block_data = buffer.drain(..block_size).collect::<Vec<_>>();
                let block_id_b64 = BASE64.encode(format!("{block_id:016x}"));

                debug!("Uploading block {} for blob: {}", block_id_b64, blob_name);
                self.put_block(blob_name, &block_id_b64, block_data).await?;
                block_ids.push(block_id_b64);
                block_id += 1;

                if block_ids.len() > MAX_BLOCKS {
                    return Err(StorageError::ProviderError(
                        "Exceeded maximum number of blocks".into(),
                    ));
                }
            }
        }

        if !buffer.is_empty() {
            let block_id_b64 = BASE64.encode(format!("{block_id:016x}"));
            self.put_block(blob_name, &block_id_b64, buffer).await?;
            block_ids.push(block_id_b64);
        }

        debug!(
            "Committing {} blocks for blob: {}",
            block_ids.len(),
            blob_name
        );
        let etag = self
            .put_block_list(
                blob_name,
                &block_ids,
                options.content_type.as_deref(),
                &options.metadata,
            )
            .await?;

        info!("Block blob upload completed for: {}", blob_name);
        Ok(etag)
    }

    /// Single-shot `Put Blob` upload for small, fully-buffered content.
    async fn upload_simple(
        &self,
        blob_name: &str,
        data: Vec<u8>,
        options: &UploadOptions,
    ) -> Result<String> {
        let url = self.blob_url(blob_name);
        let ms_date = Self::ms_date();
        let content_len = data.len() as u64;
        let content_type = options
            .content_type
            .as_deref()
            .unwrap_or("application/octet-stream");

        let mut extra: Vec<(String, String)> =
            vec![("x-ms-blob-type".to_string(), "BlockBlob".to_string())];
        for (k, v) in &options.metadata {
            extra.push((format!("x-ms-meta-{}", k.to_lowercase()), v.clone()));
        }

        let canon_resource = self.blob_canonicalized_resource(blob_name, None);
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &extra);
        let auth = self.build_auth_header(
            "PUT",
            Some(content_type),
            &canonicalized_headers,
            &canon_resource,
        )?;

        let mut request = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", &ms_date)
            .header("Content-Type", content_type)
            .header("Content-Length", content_len);
        for (k, v) in &extra {
            request = request.header(k, v);
        }

        let response = request
            .body(data)
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Put Blob request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to upload blob '{blob_name}' (HTTP {status}): {body}"
            )));
        }

        let etag = response
            .headers()
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        info!("Uploaded blob: {}", blob_name);
        Ok(etag)
    }

    /// Create the container backing this client.
    pub async fn create_container(&self) -> Result<()> {
        info!("Creating container: {}", self.container);
        let url = format!("{}?restype=container", self.container_url());
        let canon_resource = self.container_canonicalized_resource(Some("restype:container"));
        let response = self
            .signed_request(reqwest::Method::PUT, &url, &canon_resource)
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(StorageError::ProviderError(format!(
                "Failed to create container (HTTP {status})"
            )));
        }
        info!("Container created: {}", self.container);
        Ok(())
    }

    /// Delete the container backing this client.
    pub async fn delete_container(&self) -> Result<()> {
        info!("Deleting container: {}", self.container);
        let url = format!("{}?restype=container", self.container_url());
        let canon_resource = self.container_canonicalized_resource(Some("restype:container"));
        let response = self
            .signed_request(reqwest::Method::DELETE, &url, &canon_resource)
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(StorageError::ProviderError(format!(
                "Failed to delete container (HTTP {status})"
            )));
        }
        info!("Container deleted: {}", self.container);
        Ok(())
    }

    /// Check whether the container backing this client exists.
    pub async fn container_exists(&self) -> Result<bool> {
        let url = format!("{}?restype=container", self.container_url());
        let canon_resource = self.container_canonicalized_resource(Some("restype:container"));
        let response = self
            .signed_request(reqwest::Method::HEAD, &url, &canon_resource)
            .await?;

        let status = response.status();
        if status.is_success() {
            Ok(true)
        } else if status.as_u16() == 404 {
            Ok(false)
        } else {
            Err(StorageError::ProviderError(format!(
                "Failed to check container existence (HTTP {status})"
            )))
        }
    }

    /// Set the access tier of a blob via `Set Blob Tier`.
    pub async fn set_blob_tier(&self, blob_name: &str, tier: AccessTier) -> Result<()> {
        info!("Setting blob tier to {} for: {}", tier.as_str(), blob_name);
        let url = format!("{}?comp=tier", self.blob_url(blob_name));
        let ms_date = Self::ms_date();
        let extra = vec![("x-ms-access-tier".to_string(), tier.as_str().to_string())];
        let canon_resource = self.blob_canonicalized_resource(blob_name, Some("comp:tier"));
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &extra);
        let auth = self.build_auth_header("PUT", None, &canonicalized_headers, &canon_resource)?;

        let response = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", ms_date)
            .header("x-ms-access-tier", tier.as_str())
            .header("Content-Length", "0")
            .send()
            .await
            .map_err(|e| {
                StorageError::NetworkError(format!("Set Blob Tier request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(StorageError::ProviderError(format!(
                "Failed to set blob tier (HTTP {status})"
            )));
        }
        info!("Blob tier set to {} for: {}", tier.as_str(), blob_name);
        Ok(())
    }

    /// Poll the copy status of a destination blob until the server-side
    /// copy completes (or fails/aborts/times out).
    async fn wait_for_copy_completion(&self, dest_key: &str) -> Result<()> {
        const MAX_POLLS: u32 = 60;
        const POLL_INTERVAL_MS: u64 = 2_000;

        for attempt in 0..MAX_POLLS {
            tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;

            let url = self.blob_url(dest_key);
            let canon_resource = self.blob_canonicalized_resource(dest_key, None);
            let response = self
                .signed_request(reqwest::Method::HEAD, &url, &canon_resource)
                .await?;

            if !response.status().is_success() {
                return Err(StorageError::ProviderError(format!(
                    "Failed to poll copy status for '{dest_key}'"
                )));
            }

            let copy_status = response
                .headers()
                .get("x-ms-copy-status")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
                .to_string();

            match copy_status.as_str() {
                "success" => return Ok(()),
                "failed" => {
                    let description = response
                        .headers()
                        .get("x-ms-copy-status-description")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("unknown error")
                        .to_string();
                    return Err(StorageError::ProviderError(format!(
                        "Azure server-side copy failed: {description}"
                    )));
                }
                "aborted" => {
                    return Err(StorageError::ProviderError(
                        "Azure server-side copy was aborted".to_string(),
                    ));
                }
                _ => {
                    debug!(
                        "Copy pending (attempt {}/{}): status={copy_status}",
                        attempt + 1,
                        MAX_POLLS
                    );
                }
            }
        }

        Err(StorageError::ProviderError(format!(
            "Azure server-side copy of '{dest_key}' did not complete within the timeout period"
        )))
    }

    /// Build a Blob SAS URL using Shared Access Signature v2 (`sv=2021-06-08`).
    fn build_blob_sas_url(
        &self,
        key: &str,
        permissions: &str,
        expires_in_secs: u64,
    ) -> Result<String> {
        let expiry = Utc::now() + chrono::Duration::seconds(expires_in_secs as i64);
        let expiry_str = expiry.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let canonicalized_resource =
            format!("/blob/{}/{}/{}", self.account_name, self.container, key);
        let string_to_sign = format!(
            "{permissions}\n\n{expiry_str}\n{canonicalized_resource}\n\nhttps\nhttps\n{API_VERSION}\nb\n\n\n\n\n\n\n\n"
        );

        let signature = self.sign_string(&string_to_sign)?;
        let encoded_sig = percent_encode(&signature);
        let url = self.blob_url(key);

        Ok(format!(
            "{url}?sv={API_VERSION}&se={expiry_str}&sp={permissions}&spr=https&sr=b&sig={encoded_sig}"
        ))
    }
}

#[async_trait]
impl CloudStorage for AzureStorage {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading stream to blob: {}", key);

        if size.map_or(true, |s| s > BLOCK_UPLOAD_THRESHOLD) {
            return self
                .upload_blocks(key, stream, size.unwrap_or(0), &options)
                .await;
        }

        let mut chunks = Vec::new();
        let mut stream = stream;
        while let Some(result) = stream.next().await {
            let chunk = result?;
            chunks.extend_from_slice(&chunk);
        }

        self.upload_simple(key, chunks, &options).await
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading file {:?} to blob: {}", file_path, key);

        let mut file = File::open(file_path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len();

        if file_size > BLOCK_UPLOAD_THRESHOLD {
            let stream: ByteStream =
                Box::pin(futures::stream::try_unfold(file, |mut file| async move {
                    let mut buffer = vec![0u8; DEFAULT_BLOCK_SIZE];
                    let n = file.read(&mut buffer).await?;
                    if n == 0 {
                        Ok(None)
                    } else {
                        buffer.truncate(n);
                        Ok(Some((Bytes::from(buffer), file)))
                    }
                }));

            self.upload_blocks(key, stream, file_size, &options).await
        } else {
            let mut contents = Vec::new();
            file.read_to_end(&mut contents).await?;
            let etag = self.upload_simple(key, contents, &options).await?;
            info!("Uploaded file to blob: {}", key);
            Ok(etag)
        }
    }

    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream> {
        debug!("Downloading stream from blob: {}", key);

        let url = self.blob_url(key);
        let ms_date = Self::ms_date();
        let canon_resource = self.blob_canonicalized_resource(key, None);

        let mut request = self.client.get(&url);
        let canonicalized_headers = if let Some((start, end)) = options.range {
            let range = format!("bytes={start}-{end}");
            request = request.header("x-ms-range", range.clone());
            Self::canonicalized_ms_headers(&ms_date, &[("x-ms-range".to_string(), range)])
        } else {
            Self::canonicalized_ms_headers(&ms_date, &[])
        };

        let auth = self.build_auth_header("GET", None, &canonicalized_headers, &canon_resource)?;
        let response = request
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", ms_date)
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Failed to download blob: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            if status.as_u16() == 404 {
                return Err(StorageError::NotFound(key.to_string()));
            }
            return Err(StorageError::ProviderError(format!(
                "Failed to download blob (HTTP {status})"
            )));
        }

        let stream = response
            .bytes_stream()
            .map(|item| item.map_err(|e| StorageError::NetworkError(format!("Stream error: {e}"))));

        Ok(Box::pin(stream))
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
    ) -> Result<()> {
        debug!("Downloading file from blob: {} to {:?}", key, file_path);

        let mut stream = self.download_stream(key, options).await?;
        let mut file = File::create(file_path).await?;

        while let Some(result) = stream.next().await {
            let chunk = result?;
            file.write_all(&chunk).await?;
        }

        file.flush().await?;
        info!("Downloaded file from blob: {} to {:?}", key, file_path);
        Ok(())
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        debug!("Getting metadata for blob: {}", key);

        let url = self.blob_url(key);
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let response = self
            .signed_request(reqwest::Method::HEAD, &url, &canon_resource)
            .await?;

        let status = response.status();
        if !status.is_success() {
            if status.as_u16() == 404 {
                return Err(StorageError::NotFound(key.to_string()));
            }
            return Err(StorageError::ProviderError(format!(
                "Failed to get blob properties (HTTP {status})"
            )));
        }

        let headers = response.headers();
        let size = headers
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let last_modified = headers
            .get("Last-Modified")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        let etag = headers
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let content_type = headers
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let mut metadata = HashMap::new();
        for (name, value) in headers.iter() {
            if let Some(meta_key) = name.as_str().strip_prefix("x-ms-meta-") {
                if let Ok(v) = value.to_str() {
                    metadata.insert(meta_key.to_string(), v.to_string());
                }
            }
        }

        Ok(ObjectMetadata {
            key: key.to_string(),
            size,
            content_type,
            last_modified,
            etag,
            metadata,
            storage_class: None,
        })
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        debug!("Deleting blob: {}", key);

        let url = self.blob_url(key);
        let canon_resource = self.blob_canonicalized_resource(key, None);
        let response = self
            .signed_request(reqwest::Method::DELETE, &url, &canon_resource)
            .await?;

        let status = response.status();
        if !status.is_success() && status.as_u16() != 404 {
            return Err(StorageError::ProviderError(format!(
                "Failed to delete blob (HTTP {status})"
            )));
        }

        info!("Deleted blob: {}", key);
        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        debug!("Deleting {} blobs", keys.len());

        let mut results = Vec::new();
        for key in keys {
            results.push(self.delete_object(key).await);
        }

        info!("Deleted {} blobs", keys.len());
        Ok(results)
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        debug!("Listing blobs with prefix: {:?}", options.prefix);

        let ms_date = Self::ms_date();
        let max_results = options.max_results.unwrap_or(1000).to_string();

        let mut query_params: Vec<(&str, String)> = vec![
            ("comp", "list".to_string()),
            ("maxresults", max_results.clone()),
            ("restype", "container".to_string()),
        ];
        if let Some(prefix) = options.prefix.as_ref().filter(|p| !p.is_empty()) {
            query_params.push(("prefix", prefix.clone()));
        }
        if let Some(marker) = options
            .continuation_token
            .as_ref()
            .filter(|m| !m.is_empty())
        {
            query_params.push(("marker", marker.clone()));
        }

        let query_string = query_params
            .iter()
            .map(|(k, v)| format!("{k}={}", percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let url = format!("{}?{}", self.container_url(), query_string);

        let mut sorted_params = query_params.clone();
        sorted_params.sort_by(|a, b| a.0.cmp(b.0));
        let mut canon_resource = format!("/{}/{}", self.account_name, self.container);
        for (k, v) in &sorted_params {
            canon_resource.push_str(&format!("\n{k}:{v}"));
        }

        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &[]);
        let auth = self.build_auth_header("GET", None, &canonicalized_headers, &canon_resource)?;

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", ms_date)
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Failed to list blobs: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to list blobs (HTTP {status}): {body}"
            )));
        }

        let xml = response.text().await.map_err(|e| {
            StorageError::NetworkError(format!("Failed to read list response: {e}"))
        })?;
        let (objects, next_marker) = parse_list_blobs_xml(&xml);
        let has_more = next_marker.is_some();

        Ok(ListResult {
            objects,
            prefixes: Vec::new(),
            next_token: next_marker,
            has_more,
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        match self.get_metadata(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        debug!("Copying blob from {} to {}", source_key, dest_key);

        let dest_url = self.blob_url(dest_key);
        let source_url = self.blob_url(source_key);
        let ms_date = Self::ms_date();
        let extra = vec![("x-ms-copy-source".to_string(), source_url.clone())];
        let canon_resource = self.blob_canonicalized_resource(dest_key, None);
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &extra);
        let auth = self.build_auth_header("PUT", None, &canonicalized_headers, &canon_resource)?;

        let response = self
            .client
            .put(&dest_url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", &ms_date)
            .header("x-ms-copy-source", &source_url)
            .header("Content-Length", "0")
            .send()
            .await
            .map_err(|e| StorageError::NetworkError(format!("Failed to copy blob: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to copy blob (HTTP {status}): {body}"
            )));
        }

        if status.as_u16() == 202 {
            self.wait_for_copy_completion(dest_key).await?;
        }

        info!("Copied blob from {} to {}", source_key, dest_key);
        Ok(())
    }

    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String> {
        debug!("Generating presigned URL for blob: {}", key);
        self.build_blob_sas_url(key, "r", expiration_secs)
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        debug!("Generating presigned upload URL for blob: {}", key);
        self.build_blob_sas_url(key, "w", expiration_secs)
    }

    async fn update_metadata(&self, key: &str, tags: HashMap<String, String>) -> Result<()> {
        debug!("Updating metadata for blob: {}", key);

        let url = format!("{}?comp=metadata", self.blob_url(key));
        let ms_date = Self::ms_date();

        let mut extra: Vec<(String, String)> = tags
            .iter()
            .map(|(k, v)| (format!("x-ms-meta-{}", k.to_lowercase()), v.clone()))
            .collect();
        extra.sort_by(|a, b| a.0.cmp(&b.0));

        let canon_resource = self.blob_canonicalized_resource(key, Some("comp:metadata"));
        let canonicalized_headers = Self::canonicalized_ms_headers(&ms_date, &extra);
        let auth = self.build_auth_header("PUT", None, &canonicalized_headers, &canon_resource)?;

        let mut request = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-version", API_VERSION)
            .header("x-ms-date", &ms_date)
            .header("Content-Length", "0");
        for (k, v) in &extra {
            request = request.header(k, v);
        }

        let response = request.send().await.map_err(|e| {
            StorageError::NetworkError(format!("Failed to update blob metadata: {e}"))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StorageError::ProviderError(format!(
                "Failed to update metadata for blob {key} (HTTP {status}): {body}"
            )));
        }

        info!("Updated metadata for blob: {}", key);
        Ok(())
    }
}

/// Calculate optimal block size so that `total_size / block_size <= MAX_BLOCKS`.
fn calculate_block_size(total_size: u64) -> usize {
    if total_size == 0 {
        return DEFAULT_BLOCK_SIZE;
    }

    let mut block_size = DEFAULT_BLOCK_SIZE;
    while total_size / block_size as u64 > MAX_BLOCKS as u64 && block_size < MAX_BLOCK_SIZE {
        block_size *= 2;
    }

    block_size.min(MAX_BLOCK_SIZE).max(MIN_BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_storage() -> AzureStorage {
        // `AzureStorage::new` is async and hits config validation only; for
        // pure-function unit tests below we construct the struct directly
        // (this module has private-field access as a child of `azure`).
        // A crypto provider must still be installed before building any
        // `reqwest::Client`, exactly as `AzureStorage::new` does.
        install_crypto_provider();
        AzureStorage {
            client: reqwest::Client::new(),
            account_name: "myaccount".to_string(),
            container: "mycontainer".to_string(),
            account_key: b"key".to_vec(),
            _config: UnifiedConfig {
                provider: crate::StorageProvider::Azure,
                bucket: "mycontainer".to_string(),
                region: None,
                endpoint: None,
                access_key: Some("myaccount".to_string()),
                secret_key: None,
                project_id: None,
                credentials_file: None,
                transfer_acceleration: false,
                path_style: false,
                max_connections: 10,
                timeout_seconds: 300,
                enable_cache: false,
                cache_dir: None,
                max_cache_size: 0,
                retry: crate::RetryConfig::default(),
                pool_config: crate::ConnectionPoolConfig::default(),
            },
        }
    }

    #[test]
    fn test_access_tier_conversion() {
        assert_eq!(AccessTier::Hot.as_str(), "Hot");
        assert_eq!(AccessTier::Cool.as_str(), "Cool");
        assert_eq!(AccessTier::Archive.as_str(), "Archive");

        assert_eq!(AccessTier::from_str("hot"), Some(AccessTier::Hot));
        assert_eq!(AccessTier::from_str("cool"), Some(AccessTier::Cool));
        assert_eq!(AccessTier::from_str("archive"), Some(AccessTier::Archive));
        assert_eq!(AccessTier::from_str("invalid"), None);
    }

    #[test]
    fn test_calculate_block_size() {
        assert_eq!(calculate_block_size(0), DEFAULT_BLOCK_SIZE);
        assert_eq!(calculate_block_size(100 * 1024 * 1024), DEFAULT_BLOCK_SIZE);

        let large_size = 1024u64 * 1024 * 1024 * 1024; // 1 TB
        let block_size = calculate_block_size(large_size);
        assert!(block_size <= MAX_BLOCK_SIZE);
        assert!(block_size >= MIN_BLOCK_SIZE);
    }

    #[test]
    fn test_percent_encode_unreserved_passthrough() {
        assert_eq!(percent_encode("abcXYZ019-_.~"), "abcXYZ019-_.~");
    }

    #[test]
    fn test_percent_encode_escapes_special_chars() {
        assert_eq!(percent_encode("a b+c/d=e"), "a%20b%2Bc%2Fd%3De");
    }

    #[test]
    fn test_sign_string_known_answer() {
        // Independently verified: HMAC-SHA256(key=b"key", msg="The quick
        // brown fox jumps over the lazy dog"), base64-encoded.
        let storage = test_storage();
        let sig = storage
            .sign_string("The quick brown fox jumps over the lazy dog")
            .expect("sign_string should succeed with a valid key");
        assert_eq!(sig, "97yD9DBThCSxMpjmqm+xQ+9NWaFJRhdZl0edvC0aPNg=");
    }

    #[test]
    fn test_sign_string_deterministic_and_sensitive_to_input() {
        let storage = test_storage();
        let sig_a = storage.sign_string("message-a").expect("sign_string");
        let sig_a2 = storage.sign_string("message-a").expect("sign_string");
        let sig_b = storage.sign_string("message-b").expect("sign_string");
        assert_eq!(sig_a, sig_a2, "signing must be deterministic");
        assert_ne!(
            sig_a, sig_b,
            "different input must yield different signature"
        );
    }

    #[test]
    fn test_canonicalized_ms_headers_sorted() {
        let extra = vec![
            ("x-ms-meta-foo".to_string(), "bar".to_string()),
            ("x-ms-blob-type".to_string(), "BlockBlob".to_string()),
        ];
        let out = AzureStorage::canonicalized_ms_headers("Mon, 01 Jan 2024 00:00:00 GMT", &extra);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 4);
        // Alphabetical: x-ms-blob-type, x-ms-date, x-ms-meta-foo, x-ms-version
        assert!(lines[0].starts_with("x-ms-blob-type:"));
        assert!(lines[1].starts_with("x-ms-date:"));
        assert!(lines[2].starts_with("x-ms-meta-foo:"));
        assert!(lines[3].starts_with("x-ms-version:"));
    }

    #[test]
    fn test_blob_canonicalized_resource_with_and_without_query() {
        let storage = test_storage();
        assert_eq!(
            storage.blob_canonicalized_resource("video.mp4", None),
            "/myaccount/mycontainer/video.mp4"
        );
        assert_eq!(
            storage.blob_canonicalized_resource("video.mp4", Some("comp:metadata")),
            "/myaccount/mycontainer/video.mp4\ncomp:metadata"
        );
    }

    #[test]
    fn test_extract_xml_inner_basic() {
        assert_eq!(
            extract_xml_inner("<Name>hello</Name>", "Name"),
            Some("hello".to_string())
        );
        assert_eq!(
            extract_xml_inner("<Content-Length>1024</Content-Length>", "Content-Length"),
            Some("1024".to_string())
        );
    }

    #[test]
    fn test_extract_xml_inner_no_match() {
        assert_eq!(extract_xml_inner("<Other>x</Other>", "Name"), None);
    }

    #[test]
    fn test_extract_xml_inner_empty_tag() {
        assert_eq!(extract_xml_inner("<NextMarker/>", "NextMarker"), None);
    }

    #[test]
    fn test_parse_list_blobs_xml_empty() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://myaccount.blob.core.windows.net/" ContainerName="mycontainer">
  <Blobs/>
  <NextMarker/>
</EnumerationResults>"#;

        let (objects, next_marker) = parse_list_blobs_xml(xml);
        assert!(objects.is_empty());
        assert!(next_marker.is_none());
    }

    #[test]
    fn test_parse_list_blobs_xml_with_blobs() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ServiceEndpoint="https://myaccount.blob.core.windows.net/" ContainerName="mycontainer">
  <Blobs>
    <Blob>
      <Name>video/2024/sample.mp4</Name>
      <Properties>
        <Last-Modified>Mon, 04 Nov 2024 10:00:00 GMT</Last-Modified>
        <Etag>0x8DC3EF1234ABCDEF</Etag>
        <Content-Length>104857600</Content-Length>
        <Content-Type>video/mp4</Content-Type>
      </Properties>
    </Blob>
    <Blob>
      <Name>video/2024/sample2.mp4</Name>
      <Properties>
        <Last-Modified>Tue, 05 Nov 2024 12:30:00 GMT</Last-Modified>
        <Etag>0x8DC3EF5678FEDCBA</Etag>
        <Content-Length>209715200</Content-Length>
        <Content-Type>video/mp4</Content-Type>
      </Properties>
    </Blob>
  </Blobs>
  <NextMarker/>
</EnumerationResults>"#;

        let (objects, next_marker) = parse_list_blobs_xml(xml);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].key, "video/2024/sample.mp4");
        assert_eq!(objects[0].size, 104_857_600);
        assert_eq!(objects[1].key, "video/2024/sample2.mp4");
        assert_eq!(objects[1].size, 209_715_200);
        assert!(next_marker.is_none());
    }

    #[test]
    fn test_parse_list_blobs_xml_with_next_marker() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults>
  <Blobs>
    <Blob>
      <Name>file1.mp4</Name>
      <Properties>
        <Content-Length>1024</Content-Length>
        <Last-Modified>Mon, 04 Nov 2024 10:00:00 GMT</Last-Modified>
        <Etag>abc123</Etag>
        <Content-Type>video/mp4</Content-Type>
      </Properties>
    </Blob>
  </Blobs>
  <NextMarker>continuation-token-xyz</NextMarker>
</EnumerationResults>"#;

        let (objects, next_marker) = parse_list_blobs_xml(xml);
        assert_eq!(objects.len(), 1);
        assert_eq!(next_marker, Some("continuation-token-xyz".to_string()));
    }
}
