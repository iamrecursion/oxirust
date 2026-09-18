//! S3 / object-storage integration for distributed segment I/O.
//!
//! Provides the [`SegmentStore`] trait and two implementations:
//!
//! - [`S3SegmentStore`] — uploads/downloads segments via HTTP PUT/GET using
//!   the `reqwest` HTTP client (compatible with any S3-API endpoint).
//! - [`InMemorySegmentStore`] — stores segments in a `HashMap`; useful for
//!   unit tests and local development without a real S3 bucket.
//!
//! # Feature gate
//!
//! This module is gated behind the `s3` Cargo feature:
//!
//! ```toml
//! oximedia-distributed = { version = "…", features = ["s3"] }
//! ```

use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for an S3-compatible endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S3Config {
    /// S3 bucket name.
    pub bucket: String,
    /// AWS region string (e.g. `"us-east-1"`).
    pub region: String,
    /// Optional custom endpoint URL for S3-compatible services (MinIO, Wasabi,
    /// Cloudflare R2, …).  Defaults to the standard AWS endpoint when `None`.
    pub endpoint_url: Option<String>,
    /// Optional AWS access key ID.
    pub access_key_id: Option<String>,
    /// Optional AWS secret access key.
    pub secret_access_key: Option<String>,
}

impl S3Config {
    /// Build the base URL for object operations.
    ///
    /// Format: `{endpoint}/{bucket}/` or the standard AWS URL.
    #[must_use]
    pub fn base_url(&self) -> String {
        if let Some(ref endpoint) = self.endpoint_url {
            format!("{}/{}", endpoint.trim_end_matches('/'), self.bucket)
        } else {
            format!("https://{}.s3.{}.amazonaws.com", self.bucket, self.region)
        }
    }
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            bucket: "oximedia-segments".to_string(),
            region: "us-east-1".to_string(),
            endpoint_url: None,
            access_key_id: None,
            secret_access_key: None,
        }
    }
}

/// Result type used throughout this module.
pub type Result<T> = std::result::Result<T, SegmentStoreError>;

/// Errors produced by segment store operations.
#[derive(Debug, thiserror::Error)]
pub enum SegmentStoreError {
    #[error("HTTP error during segment upload/download: {0}")]
    Http(String),

    #[error("Segment not found: {key}")]
    NotFound { key: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Async trait for segment storage backends.
#[async_trait::async_trait]
pub trait SegmentStore: Send + Sync {
    /// Upload `data` and return the storage key under which it was stored.
    async fn upload_segment(&self, id: Uuid, data: Bytes) -> Result<String>;

    /// Download a segment by its storage key.
    async fn download_segment(&self, key: &str) -> Result<Bytes>;

    /// Delete a segment by its storage key.
    async fn delete_segment(&self, key: &str) -> Result<()>;

    /// Check whether a segment exists.
    async fn segment_exists(&self, key: &str) -> Result<bool>;
}

/// Derives the canonical S3 object key for a segment UUID.
///
/// Format: `segments/{uuid}` — keeps all segment objects under a common prefix
/// for lifecycle policies and bulk operations.
#[must_use]
pub fn segment_key(id: Uuid) -> String {
    format!("segments/{id}")
}

/// S3-backed segment store that uses HTTP PUT/GET to interact with any
/// S3-compatible object storage API.
pub struct S3SegmentStore {
    config: S3Config,
    client: reqwest::Client,
}

impl S3SegmentStore {
    /// Create a new S3 segment store with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentStoreError::Config`] if the HTTP client cannot be built.
    pub fn new(config: S3Config) -> Result<Self> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| SegmentStoreError::Config(e.to_string()))?;
        Ok(Self { config, client })
    }

    fn object_url(&self, key: &str) -> String {
        format!("{}/{}", self.config.base_url(), key)
    }
}

#[async_trait::async_trait]
impl SegmentStore for S3SegmentStore {
    async fn upload_segment(&self, id: Uuid, data: Bytes) -> Result<String> {
        let key = segment_key(id);
        let url = self.object_url(&key);

        let mut req = self.client.put(&url).body(data);

        if let (Some(ref aki), Some(ref sak)) = (
            self.config.access_key_id.as_ref(),
            self.config.secret_access_key.as_ref(),
        ) {
            req = req.basic_auth(aki, Some(sak));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SegmentStoreError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(SegmentStoreError::Http(format!(
                "PUT {} returned HTTP {}",
                url,
                resp.status()
            )));
        }

        Ok(key)
    }

    async fn download_segment(&self, key: &str) -> Result<Bytes> {
        let url = self.object_url(key);

        let mut req = self.client.get(&url);

        if let (Some(ref aki), Some(ref sak)) = (
            self.config.access_key_id.as_ref(),
            self.config.secret_access_key.as_ref(),
        ) {
            req = req.basic_auth(aki, Some(sak));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SegmentStoreError::Http(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(SegmentStoreError::NotFound {
                key: key.to_string(),
            });
        }

        if !resp.status().is_success() {
            return Err(SegmentStoreError::Http(format!(
                "GET {} returned HTTP {}",
                url,
                resp.status()
            )));
        }

        resp.bytes()
            .await
            .map_err(|e| SegmentStoreError::Http(e.to_string()))
    }

    async fn delete_segment(&self, key: &str) -> Result<()> {
        let url = self.object_url(key);

        let mut req = self.client.delete(&url);

        if let (Some(ref aki), Some(ref sak)) = (
            self.config.access_key_id.as_ref(),
            self.config.secret_access_key.as_ref(),
        ) {
            req = req.basic_auth(aki, Some(sak));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SegmentStoreError::Http(e.to_string()))?;

        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            return Err(SegmentStoreError::Http(format!(
                "DELETE {} returned HTTP {}",
                url,
                resp.status()
            )));
        }

        Ok(())
    }

    async fn segment_exists(&self, key: &str) -> Result<bool> {
        let url = self.object_url(key);

        let mut req = self.client.head(&url);

        if let (Some(ref aki), Some(ref sak)) = (
            self.config.access_key_id.as_ref(),
            self.config.secret_access_key.as_ref(),
        ) {
            req = req.basic_auth(aki, Some(sak));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SegmentStoreError::Http(e.to_string()))?;

        Ok(resp.status().is_success())
    }
}

/// In-memory segment store for unit tests and local development.
///
/// Thread-safe via an `Arc<RwLock<HashMap>>`.
#[derive(Debug, Clone, Default)]
pub struct InMemorySegmentStore {
    store: Arc<RwLock<HashMap<String, Bytes>>>,
}

impl InMemorySegmentStore {
    /// Create a new, empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of segments currently in the store.
    pub async fn len(&self) -> usize {
        self.store.read().await.len()
    }

    /// Returns `true` if the store contains no segments.
    pub async fn is_empty(&self) -> bool {
        self.store.read().await.is_empty()
    }
}

#[async_trait::async_trait]
impl SegmentStore for InMemorySegmentStore {
    async fn upload_segment(&self, id: Uuid, data: Bytes) -> Result<String> {
        let key = segment_key(id);
        self.store.write().await.insert(key.clone(), data);
        Ok(key)
    }

    async fn download_segment(&self, key: &str) -> Result<Bytes> {
        self.store
            .read()
            .await
            .get(key)
            .cloned()
            .ok_or_else(|| SegmentStoreError::NotFound {
                key: key.to_string(),
            })
    }

    async fn delete_segment(&self, key: &str) -> Result<()> {
        self.store.write().await.remove(key);
        Ok(())
    }

    async fn segment_exists(&self, key: &str) -> Result<bool> {
        Ok(self.store.read().await.contains_key(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_store_upload_download_roundtrip() {
        let store = InMemorySegmentStore::new();
        let id = Uuid::new_v4();
        let data = Bytes::from_static(b"hello segment data");

        let key = store
            .upload_segment(id, data.clone())
            .await
            .expect("upload should succeed");

        let downloaded = store
            .download_segment(&key)
            .await
            .expect("download should succeed");

        assert_eq!(downloaded, data);
    }

    #[tokio::test]
    async fn test_s3_store_key_format() {
        let store = InMemorySegmentStore::new();
        let id = Uuid::new_v4();
        let key = store
            .upload_segment(id, Bytes::from("test"))
            .await
            .expect("upload");

        assert!(
            key.starts_with("segments/"),
            "key should start with 'segments/': {key}"
        );
        assert!(key.contains(&id.to_string()), "key should contain UUID");
    }

    #[tokio::test]
    async fn test_in_memory_download_missing_key_fails() {
        let store = InMemorySegmentStore::new();
        let result = store.download_segment("segments/nonexistent").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SegmentStoreError::NotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_in_memory_delete_removes_segment() {
        let store = InMemorySegmentStore::new();
        let id = Uuid::new_v4();
        let key = store
            .upload_segment(id, Bytes::from("data"))
            .await
            .expect("upload");

        assert!(store.segment_exists(&key).await.unwrap());
        store.delete_segment(&key).await.expect("delete");
        assert!(!store.segment_exists(&key).await.unwrap());
    }

    #[tokio::test]
    async fn test_in_memory_store_len() {
        let store = InMemorySegmentStore::new();
        assert!(store.is_empty().await);

        store
            .upload_segment(Uuid::new_v4(), Bytes::from("a"))
            .await
            .unwrap();
        store
            .upload_segment(Uuid::new_v4(), Bytes::from("b"))
            .await
            .unwrap();

        assert_eq!(store.len().await, 2);
    }

    #[tokio::test]
    async fn test_in_memory_concurrent_access() {
        use std::sync::Arc;
        let store = Arc::new(InMemorySegmentStore::new());
        let mut handles = Vec::new();

        for i in 0_u8..8 {
            let s = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let id = Uuid::new_v4();
                let data = Bytes::copy_from_slice(&[i; 64]);
                let key = s.upload_segment(id, data.clone()).await.unwrap();
                let downloaded = s.download_segment(&key).await.unwrap();
                assert_eq!(downloaded, data);
            }));
        }

        for h in handles {
            h.await.expect("task should not panic");
        }

        assert_eq!(store.len().await, 8);
    }

    #[test]
    fn test_s3_config_base_url_default() {
        let config = S3Config::default();
        let url = config.base_url();
        assert!(url.contains("amazonaws.com"));
        assert!(url.contains("oximedia-segments"));
    }

    #[test]
    fn test_s3_config_custom_endpoint() {
        let config = S3Config {
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
            endpoint_url: Some("http://localhost:9000".to_string()),
            ..Default::default()
        };
        let url = config.base_url();
        assert!(url.starts_with("http://localhost:9000"));
        assert!(url.contains("my-bucket"));
    }

    #[test]
    fn test_segment_key_format() {
        let id = Uuid::parse_str("12345678-1234-1234-1234-123456789012").unwrap();
        let key = segment_key(id);
        assert_eq!(key, "segments/12345678-1234-1234-1234-123456789012");
    }
}
