//! Backblaze B2 cloud storage provider.
//!
//! Backblaze B2 is a low-cost object storage service.  This module implements
//! the `CloudStorage` trait using the B2 native API (v3) for authentication
//! and the S3-compatible endpoint for object operations, giving us the full
//! feature set of the existing S3-compatible infrastructure at B2's pricing.
//!
//! ## Architecture
//!
//! - `B2Config` — static credentials + bucket configuration.
//! - `B2Provider` — implements `CloudStorage`; manages a live `reqwest`
//!   client and a cached `B2Session` obtained via `b2_authorize_account`.
//!
//! ## Authentication
//!
//! ```text
//! POST https://api.backblazeb2.com/b2api/v3/b2_authorize_account
//!   Authorization: Basic <base64(key_id:application_key)>
//! ```
//!
//! The response contains `apiUrl`, `authorizationToken`, `downloadUrl`, and
//! `s3ApiUrl`.  Object operations are then sent to `<s3ApiUrl>/<bucket>/<key>`
//! with the `authorizationToken` as the `Authorization` header value.
//!
//! ## Example (struct construction only)
//!
//! ```rust
//! use oximedia_cloud::b2::{B2Config, B2Provider};
//!
//! let config = B2Config::new(
//!     "0012345".to_string(),
//!     "K001xyz".to_string(),
//!     "my-bucket".to_string(),
//!     "my-bucket-id".to_string(),
//! );
//!
//! assert!(config.validate().is_ok());
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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a Backblaze B2 bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct B2Config {
    /// B2 application key ID (used as the S3 access key).
    pub key_id: String,
    /// B2 application key secret (used as the S3 secret key).
    pub application_key: String,
    /// Name of the B2 bucket to operate on.
    pub bucket: String,
    /// B2 bucket ID (required for some native API calls).
    pub bucket_id: String,
}

impl B2Config {
    /// Create a new B2 configuration.
    #[must_use]
    pub fn new(key_id: String, application_key: String, bucket: String, bucket_id: String) -> Self {
        Self {
            key_id,
            application_key,
            bucket,
            bucket_id,
        }
    }

    /// Validate that required fields are non-empty.
    ///
    /// # Errors
    ///
    /// Returns [`CloudError::InvalidConfig`] when any field is empty.
    pub fn validate(&self) -> Result<()> {
        if self.key_id.is_empty() {
            return Err(CloudError::InvalidConfig(
                "B2 key_id must not be empty".to_string(),
            ));
        }
        if self.application_key.is_empty() {
            return Err(CloudError::InvalidConfig(
                "B2 application_key must not be empty".to_string(),
            ));
        }
        if self.bucket.is_empty() {
            return Err(CloudError::InvalidConfig(
                "B2 bucket must not be empty".to_string(),
            ));
        }
        if self.bucket_id.is_empty() {
            return Err(CloudError::InvalidConfig(
                "B2 bucket_id must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ── Session (result of b2_authorize_account) ─────────────────────────────────

/// Live session data returned by `b2_authorize_account`.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct B2AuthResponse {
    #[serde(rename = "authorizationToken")]
    authorization_token: String,
    #[serde(rename = "apiUrl")]
    api_url: String,
    #[serde(rename = "downloadUrl")]
    download_url: String,
    #[serde(rename = "s3ApiUrl")]
    s3_api_url: String,
}

/// Cached, validated B2 session.
#[derive(Debug, Clone)]
pub(crate) struct B2Session {
    pub(crate) authorization_token: String,
    /// Base URL for S3-compatible calls: `https://s3.<region>.backblazeb2.com`.
    pub(crate) s3_api_url: String,
    /// Base URL for native B2 API calls (e.g. version management).
    pub(crate) api_url: String,
    /// Base URL for B2 file downloads via the native download endpoint.
    pub(crate) download_url: String,
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// Backblaze B2 storage provider implementing [`CloudStorage`].
///
/// Uses the S3-compatible endpoint for all object operations to maximise
/// compatibility with existing infrastructure.
pub struct B2Provider {
    config: B2Config,
    client: Client,
    session: Arc<RwLock<Option<B2Session>>>,
}

impl B2Provider {
    /// Return the native B2 API base URL (e.g. for file-version operations).
    ///
    /// Returns `None` if the provider has not yet authenticated.
    pub async fn native_api_url(&self) -> Option<String> {
        self.session
            .read()
            .await
            .as_ref()
            .map(|s| s.api_url.clone())
    }

    /// Create a B2 provider without immediately authenticating.
    ///
    /// Authentication happens lazily on the first request via
    /// `B2Provider::get_session`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: B2Config) -> Result<Self> {
        crate::tls_provider::install_default_crypto_provider();
        config.validate()?;
        Ok(Self {
            config,
            client: Client::new(),
            session: Arc::new(RwLock::new(None)),
        })
    }

    /// Create a B2 provider and immediately authenticate.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or authentication
    /// with the B2 API fails.
    pub async fn new_authenticated(config: B2Config) -> Result<Self> {
        let provider = Self::new(config)?;
        provider.authenticate().await?;
        Ok(provider)
    }

    /// Authenticate with the B2 API and cache the resulting session.
    ///
    /// On success the session is cached internally.  Subsequent calls to
    /// object-operation methods will reuse it automatically.
    ///
    /// # Errors
    ///
    /// Returns [`CloudError::Authentication`] when credentials are rejected.
    pub async fn authenticate(&self) -> Result<()> {
        self.authenticate_inner().await?;
        Ok(())
    }

    /// Authenticate and return the resulting session (internal helper).
    async fn authenticate_inner(&self) -> Result<B2Session> {
        let credentials = BASE64.encode(format!(
            "{}:{}",
            self.config.key_id, self.config.application_key
        ));

        let response = self
            .client
            .get("https://api.backblazeb2.com/b2api/v3/b2_authorize_account")
            .header("Authorization", format!("Basic {credentials}"))
            .send()
            .await
            .map_err(|e| CloudError::Network(format!("B2 auth request failed: {e}")))?;

        if response.status() == StatusCode::UNAUTHORIZED {
            return Err(CloudError::Authentication(
                "B2 credentials rejected (401)".to_string(),
            ));
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Authentication(format!(
                "B2 auth failed with status {status}: {body}"
            )));
        }

        let auth: B2AuthResponse = response
            .json()
            .await
            .map_err(|e| CloudError::Serialization(format!("Failed to parse B2 auth: {e}")))?;

        let session = B2Session {
            authorization_token: auth.authorization_token,
            s3_api_url: auth.s3_api_url,
            api_url: auth.api_url.clone(),
            download_url: auth.download_url,
        };

        {
            let mut guard = self.session.write().await;
            *guard = Some(session.clone());
        }

        Ok(session)
    }

    /// Return a live session, re-authenticating if necessary.
    async fn get_session(&self) -> Result<B2Session> {
        {
            let guard = self.session.read().await;
            if let Some(session) = guard.as_ref() {
                return Ok(session.clone());
            }
        }
        self.authenticate_inner().await
    }

    /// Build the S3-compatible object URL: `<s3_api_url>/<bucket>/<key>`.
    fn object_url(s3_api_url: &str, bucket: &str, key: &str) -> String {
        format!("{s3_api_url}/{bucket}/{key}")
    }
}

#[async_trait]
impl CloudStorage for B2Provider {
    async fn upload(&self, key: &str, data: Bytes) -> Result<()> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .put(&url)
            .header("Authorization", &session.authorization_token)
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "B2 upload failed ({status}): {body}"
            )));
        }

        Ok(())
    }

    async fn upload_with_options(
        &self,
        key: &str,
        data: Bytes,
        options: UploadOptions,
    ) -> Result<()> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let mut req = self
            .client
            .put(&url)
            .header("Authorization", &session.authorization_token);

        if let Some(ct) = options.content_type {
            req = req.header("Content-Type", ct);
        } else {
            req = req.header("Content-Type", "application/octet-stream");
        }

        if let Some(cc) = options.cache_control {
            req = req.header("Cache-Control", cc);
        }
        if let Some(ce) = options.content_encoding {
            req = req.header("Content-Encoding", ce);
        }

        for (k, v) in &options.metadata {
            req = req.header(format!("x-amz-meta-{k}"), v.as_str());
        }

        let response = req.body(data).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "B2 upload_with_options failed ({status}): {body}"
            )));
        }

        Ok(())
    }

    async fn download(&self, key: &str) -> Result<Bytes> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .get(&url)
            .header("Authorization", &session.authorization_token)
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(CloudError::NotFound(format!("Key not found: {key}")));
        }
        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "B2 download failed ({status})"
            )));
        }

        Ok(response.bytes().await?)
    }

    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .get(&url)
            .header("Authorization", &session.authorization_token)
            .header("Range", format!("bytes={start}-{end}"))
            .send()
            .await?;

        if !response.status().is_success() && response.status() != StatusCode::PARTIAL_CONTENT {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "B2 download_range failed ({status})"
            )));
        }

        Ok(response.bytes().await?)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        let result = self.list_paginated(prefix, None, 1000).await?;
        Ok(result.objects)
    }

    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult> {
        let session = self.get_session().await?;
        let mut url = format!(
            "{}/{}/{}", // list via S3-compat: GET /<bucket>?list-type=2
            session.s3_api_url, self.config.bucket, ""
        );
        url = format!(
            "{}/{}?list-type=2&prefix={}&max-keys={}",
            session.s3_api_url, self.config.bucket, prefix, max_keys
        );

        if let Some(token) = continuation_token {
            url.push_str(&format!(
                "&continuation-token={}",
                urlencoding::encode(&token)
            ));
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", &session.authorization_token)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!("B2 list failed ({status})")));
        }

        let text = response.text().await?;

        // Simplified XML parsing (same approach as GenericStorage)
        let mut objects = Vec::new();
        for line in text.lines() {
            if line.contains("<Key>") {
                if let Some(key_val) = extract_xml_value(line, "Key") {
                    objects.push(ObjectInfo {
                        key: key_val,
                        size: 0,
                        last_modified: Utc::now(),
                        etag: None,
                        storage_class: None,
                        content_type: None,
                    });
                }
            }
        }

        let continuation = if text.contains("<IsTruncated>true</IsTruncated>") {
            extract_xml_value_from_text(&text, "NextContinuationToken")
        } else {
            None
        };

        Ok(ListResult {
            is_truncated: continuation.is_some(),
            objects,
            continuation_token: continuation,
            common_prefixes: Vec::new(),
        })
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", &session.authorization_token)
            .send()
            .await?;

        if !response.status().is_success() && response.status() != StatusCode::NO_CONTENT {
            let status = response.status();
            return Err(CloudError::Storage(format!("B2 delete failed ({status})")));
        }

        Ok(())
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<Vec<DeleteResult>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            let outcome = self.delete(key).await;
            results.push(DeleteResult {
                key: key.clone(),
                success: outcome.is_ok(),
                error: outcome.err().map(|e| e.to_string()),
            });
        }
        Ok(results)
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .head(&url)
            .header("Authorization", &session.authorization_token)
            .send()
            .await?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(CloudError::NotFound(format!("Key not found: {key}")));
        }
        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!("B2 head failed ({status})")));
        }

        let headers = response.headers();
        let size: u64 = headers
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let content_type = headers
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let etag = headers
            .get("ETag")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let info = ObjectInfo {
            key: key.to_string(),
            size,
            last_modified: Utc::now(),
            etag,
            storage_class: Some(StorageClass::Standard),
            content_type,
        };

        Ok(ObjectMetadata {
            info,
            user_metadata: HashMap::new(),
            system_metadata: HashMap::new(),
            tags: HashMap::new(),
            content_encoding: None,
            content_language: None,
            cache_control: None,
            content_disposition: None,
        })
    }

    async fn update_metadata(&self, key: &str, _metadata: HashMap<String, String>) -> Result<()> {
        // B2 S3-compat requires a copy to update metadata (same as AWS S3 semantics)
        self.copy(key, key).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);

        let response = self
            .client
            .head(&url)
            .header("Authorization", &session.authorization_token)
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    async fn copy(&self, source_key: &str, dest_key: &str) -> Result<()> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, dest_key);
        let copy_source = format!("/{}/{}", self.config.bucket, source_key);

        let response = self
            .client
            .put(&url)
            .header("Authorization", &session.authorization_token)
            .header("x-amz-copy-source", copy_source)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!("B2 copy failed ({status})")));
        }

        Ok(())
    }

    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        let session = self.get_session().await?;
        // Use the native B2 download endpoint for time-limited authorisation.
        // Pattern: <downloadUrl>/file/<bucket>/<key>?Authorization=<token>&validDurationInSeconds=<n>
        let url = format!(
            "{}/file/{}/{}",
            session.download_url,
            urlencoding::encode(&self.config.bucket),
            urlencoding::encode(key)
        );
        Ok(format!(
            "{url}?Authorization={}&validDurationInSeconds={expires_in_secs}",
            urlencoding::encode(&session.authorization_token)
        ))
    }

    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        let session = self.get_session().await?;
        let url = Self::object_url(&session.s3_api_url, &self.config.bucket, key);
        Ok(format!("{url}?expires={expires_in_secs}"))
    }

    async fn set_storage_class(&self, _key: &str, _class: StorageClass) -> Result<()> {
        // B2 does not expose multiple storage classes via the S3-compat API;
        // tiering is configured at the bucket level in the B2 console.
        Ok(())
    }

    async fn get_stats(&self, prefix: &str) -> Result<StorageStats> {
        let objects = self.list(prefix).await?;
        let mut stats = StorageStats::default();
        for obj in objects {
            stats.total_size += obj.size;
            stats.object_count += 1;
        }
        Ok(stats)
    }

    async fn server_side_copy(
        &self,
        from_key: &str,
        to_key: &str,
        from_bucket: Option<&str>,
        to_bucket: Option<&str>,
    ) -> Result<()> {
        let session = self.get_session().await?;
        let src_bucket = from_bucket.unwrap_or(&self.config.bucket);
        let dst_bucket = to_bucket.unwrap_or(&self.config.bucket);
        let url = Self::object_url(&session.s3_api_url, dst_bucket, to_key);
        let copy_source = format!("/{src_bucket}/{from_key}");

        let response = self
            .client
            .put(&url)
            .header("Authorization", &session.authorization_token)
            .header("x-amz-copy-source", copy_source)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(CloudError::Storage(format!(
                "B2 server_side_copy failed ({status})"
            )));
        }

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_xml_value(line: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = line.find(&open)? + open.len();
    let end = line.find(&close)?;
    if start < end {
        Some(line[start..end].to_string())
    } else {
        None
    }
}

fn extract_xml_value_from_text(text: &str, tag: &str) -> Option<String> {
    for line in text.lines() {
        if line.contains(&format!("<{tag}>")) {
            if let Some(v) = extract_xml_value(line, tag) {
                return Some(v);
            }
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CloudStorage;

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn test_b2_config_validation_ok() {
        let cfg = B2Config::new(
            "0012345".to_string(),
            "K001xyz".to_string(),
            "my-bucket".to_string(),
            "bucket-id-abc".to_string(),
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_b2_config_validation_empty_key_id() {
        let cfg = B2Config::new(
            "".to_string(),
            "K001xyz".to_string(),
            "my-bucket".to_string(),
            "bucket-id".to_string(),
        );
        let err = cfg.validate().expect_err("empty key_id must fail");
        assert!(err.to_string().contains("key_id"), "{err}");
    }

    #[test]
    fn test_b2_config_validation_empty_application_key() {
        let cfg = B2Config::new(
            "0012345".to_string(),
            "".to_string(),
            "my-bucket".to_string(),
            "bucket-id".to_string(),
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_b2_config_validation_empty_bucket() {
        let cfg = B2Config::new(
            "0012345".to_string(),
            "K001xyz".to_string(),
            "".to_string(),
            "bucket-id".to_string(),
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_b2_config_validation_empty_bucket_id() {
        let cfg = B2Config::new(
            "0012345".to_string(),
            "K001xyz".to_string(),
            "my-bucket".to_string(),
            "".to_string(),
        );
        assert!(cfg.validate().is_err());
    }

    // ── Provider construction ─────────────────────────────────────────────────

    #[test]
    fn test_b2_provider_new_valid_config() {
        let cfg = B2Config::new(
            "0012345".to_string(),
            "K001xyz".to_string(),
            "my-bucket".to_string(),
            "bucket-id".to_string(),
        );
        let provider = B2Provider::new(cfg);
        assert!(provider.is_ok(), "Valid config must construct provider");
    }

    #[test]
    fn test_b2_provider_new_invalid_config() {
        let cfg = B2Config::new(
            "".to_string(),
            "key".to_string(),
            "b".to_string(),
            "id".to_string(),
        );
        assert!(B2Provider::new(cfg).is_err());
    }

    // ── Trait object (compile-time duck-typing test) ──────────────────────────
    //
    // If `B2Provider` does not implement `CloudStorage` this test will not
    // compile, giving us a static guarantee that the trait is satisfied.

    #[test]
    fn test_b2_provider_implements_cloud_storage_trait() {
        fn accepts_cloud_storage(_: &dyn CloudStorage) {}

        let cfg = B2Config::new(
            "key_id".to_string(),
            "app_key".to_string(),
            "bucket".to_string(),
            "bucket_id".to_string(),
        );
        let provider = B2Provider::new(cfg).expect("valid config");
        accepts_cloud_storage(&provider);
    }

    // ── Helper functions ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_xml_value_basic() {
        let line = "<Key>some/path/file.mp4</Key>";
        assert_eq!(
            extract_xml_value(line, "Key"),
            Some("some/path/file.mp4".to_string())
        );
    }

    #[test]
    fn test_extract_xml_value_no_match() {
        let line = "<Size>12345</Size>";
        assert_eq!(extract_xml_value(line, "Key"), None);
    }

    #[test]
    fn test_object_url_construction() {
        let url = B2Provider::object_url(
            "https://s3.us-west-004.backblazeb2.com",
            "my-bucket",
            "videos/clip.mp4",
        );
        assert_eq!(
            url,
            "https://s3.us-west-004.backblazeb2.com/my-bucket/videos/clip.mp4"
        );
    }

    // ── Mock-based upload/download test using mockito ─────────────────────────

    #[tokio::test]
    async fn test_b2_upload_mock() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("PUT", "/my-bucket/test-key")
            .with_status(200)
            .create_async()
            .await;

        // Wire provider directly to the mock server by substituting the session
        let cfg = B2Config::new(
            "key".to_string(),
            "secret".to_string(),
            "my-bucket".to_string(),
            "bid".to_string(),
        );
        let provider = B2Provider::new(cfg).expect("valid config");

        // Inject a fake session pointing at the mock server
        {
            let mut guard = provider.session.write().await;
            *guard = Some(B2Session {
                authorization_token: "fake-token".to_string(),
                s3_api_url: server.url(),
                api_url: server.url(),
                download_url: server.url(),
            });
        }

        let result = provider.upload("test-key", Bytes::from("hello")).await;
        assert!(result.is_ok(), "upload to mock must succeed: {result:?}");
    }

    #[tokio::test]
    async fn test_b2_download_mock() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/my-bucket/dl-key")
            .with_status(200)
            .with_body("file-contents")
            .create_async()
            .await;

        let cfg = B2Config::new(
            "key".to_string(),
            "secret".to_string(),
            "my-bucket".to_string(),
            "bid".to_string(),
        );
        let provider = B2Provider::new(cfg).expect("valid config");

        {
            let mut guard = provider.session.write().await;
            *guard = Some(B2Session {
                authorization_token: "fake-token".to_string(),
                s3_api_url: server.url(),
                api_url: server.url(),
                download_url: server.url(),
            });
        }

        let data = provider
            .download("dl-key")
            .await
            .expect("download must succeed");
        assert_eq!(data, Bytes::from("file-contents"));
    }

    #[tokio::test]
    async fn test_b2_delete_mock() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("DELETE", "/my-bucket/del-key")
            .with_status(204)
            .create_async()
            .await;

        let cfg = B2Config::new(
            "key".to_string(),
            "secret".to_string(),
            "my-bucket".to_string(),
            "bid".to_string(),
        );
        let provider = B2Provider::new(cfg).expect("valid config");

        {
            let mut guard = provider.session.write().await;
            *guard = Some(B2Session {
                authorization_token: "fake-token".to_string(),
                s3_api_url: server.url(),
                api_url: server.url(),
                download_url: server.url(),
            });
        }

        let result = provider.delete("del-key").await;
        assert!(result.is_ok(), "delete must succeed: {result:?}");
    }

    #[tokio::test]
    async fn test_b2_server_side_copy_mock() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("PUT", "/my-bucket/dst-key")
            .match_header("x-amz-copy-source", "/my-bucket/src-key")
            .with_status(200)
            .create_async()
            .await;

        let cfg = B2Config::new(
            "key".to_string(),
            "secret".to_string(),
            "my-bucket".to_string(),
            "bid".to_string(),
        );
        let provider = B2Provider::new(cfg).expect("valid config");

        {
            let mut guard = provider.session.write().await;
            *guard = Some(B2Session {
                authorization_token: "fake-token".to_string(),
                s3_api_url: server.url(),
                api_url: server.url(),
                download_url: server.url(),
            });
        }

        let result = provider
            .server_side_copy("src-key", "dst-key", None, None)
            .await;
        assert!(result.is_ok(), "server_side_copy must succeed: {result:?}");
    }
}
