//! Core types and traits for cloud storage

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::error::Result;

/// Core trait for cloud storage operations
#[async_trait]
pub trait CloudStorage: Send + Sync {
    /// Upload data to the storage
    async fn upload(&self, key: &str, data: Bytes) -> Result<()>;

    /// Upload with options
    async fn upload_with_options(
        &self,
        key: &str,
        data: Bytes,
        options: UploadOptions,
    ) -> Result<()>;

    /// Download data from storage
    async fn download(&self, key: &str) -> Result<Bytes>;

    /// Download a byte range
    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes>;

    /// List objects with a prefix
    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>>;

    /// List objects with pagination
    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult>;

    /// Delete an object
    async fn delete(&self, key: &str) -> Result<()>;

    /// Delete multiple objects
    async fn delete_batch(&self, keys: &[String]) -> Result<Vec<DeleteResult>>;

    /// Get object metadata
    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata>;

    /// Update object metadata
    async fn update_metadata(&self, key: &str, metadata: HashMap<String, String>) -> Result<()>;

    /// Check if object exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Copy object
    async fn copy(&self, source_key: &str, dest_key: &str) -> Result<()>;

    /// Generate a presigned URL for download
    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String>;

    /// Generate a presigned URL for upload
    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String>;

    /// Set object storage class/tier
    async fn set_storage_class(&self, key: &str, class: StorageClass) -> Result<()>;

    /// Get storage statistics
    async fn get_stats(&self, prefix: &str) -> Result<StorageStats>;

    /// Server-side copy between objects, optionally across buckets.
    ///
    /// When `from_bucket` or `to_bucket` is `None`, the provider's default
    /// bucket is used.  Providers that support native server-side copy (S3,
    /// GCS, Azure, B2) should override this with a direct API call; the
    /// default implementation falls back to a download-then-upload cycle.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be read or the destination write
    /// fails.
    async fn server_side_copy(
        &self,
        from_key: &str,
        to_key: &str,
        from_bucket: Option<&str>,
        to_bucket: Option<&str>,
    ) -> Result<()> {
        // Default fallback: download from source then upload to destination.
        // Providers with native copy support should override this.
        let _ = from_bucket; // bucket routing is provider-specific
        let _ = to_bucket;
        let data = self.download(from_key).await?;
        self.upload(to_key, data).await
    }
}

/// Information about a cloud object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectInfo {
    /// Object key/path
    pub key: String,
    /// Size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: DateTime<Utc>,
    /// ETag/checksum
    pub etag: Option<String>,
    /// Storage class
    pub storage_class: Option<StorageClass>,
    /// Content type
    pub content_type: Option<String>,
}

/// Object metadata including user-defined metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object information
    pub info: ObjectInfo,
    /// User-defined metadata
    pub user_metadata: HashMap<String, String>,
    /// System metadata
    pub system_metadata: HashMap<String, String>,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Content encoding
    pub content_encoding: Option<String>,
    /// Content language
    pub content_language: Option<String>,
    /// Cache control
    pub cache_control: Option<String>,
    /// Content disposition
    pub content_disposition: Option<String>,
}

/// Result of a list operation with pagination
#[derive(Debug, Clone)]
pub struct ListResult {
    /// Objects found
    pub objects: Vec<ObjectInfo>,
    /// Token for next page
    pub continuation_token: Option<String>,
    /// Whether there are more results
    pub is_truncated: bool,
    /// Common prefixes (directories)
    pub common_prefixes: Vec<String>,
}

/// Result of a delete operation
#[derive(Debug, Clone)]
pub struct DeleteResult {
    /// Object key
    pub key: String,
    /// Whether deletion succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Storage class/tier options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageClass {
    /// Standard/hot storage
    Standard,
    /// Infrequent access
    InfrequentAccess,
    /// Glacier/archive
    Glacier,
    /// Deep archive
    DeepArchive,
    /// Intelligent tiering
    IntelligentTiering,
    /// One zone infrequent access
    OneZoneIA,
    /// Reduced redundancy (deprecated)
    ReducedRedundancy,
}

impl fmt::Display for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageClass::Standard => write!(f, "STANDARD"),
            StorageClass::InfrequentAccess => write!(f, "STANDARD_IA"),
            StorageClass::Glacier => write!(f, "GLACIER"),
            StorageClass::DeepArchive => write!(f, "DEEP_ARCHIVE"),
            StorageClass::IntelligentTiering => write!(f, "INTELLIGENT_TIERING"),
            StorageClass::OneZoneIA => write!(f, "ONEZONE_IA"),
            StorageClass::ReducedRedundancy => write!(f, "REDUCED_REDUNDANCY"),
        }
    }
}

/// Options for upload operations
#[derive(Debug, Clone, Default)]
pub struct UploadOptions {
    /// Content type
    pub content_type: Option<String>,
    /// Content encoding
    pub content_encoding: Option<String>,
    /// Cache control
    pub cache_control: Option<String>,
    /// Content disposition
    pub content_disposition: Option<String>,
    /// User metadata
    pub metadata: HashMap<String, String>,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Storage class
    pub storage_class: Option<StorageClass>,
    /// Server-side encryption
    pub encryption: Option<String>,
    /// ACL (Access Control List)
    pub acl: Option<String>,
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total size in bytes
    pub total_size: u64,
    /// Number of objects
    pub object_count: u64,
    /// Size by storage class
    pub size_by_class: HashMap<String, u64>,
    /// Count by storage class
    pub count_by_class: HashMap<String, u64>,
}

/// Transfer progress information
#[derive(Debug, Clone)]
pub struct TransferProgress {
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Transfer rate in bytes/sec
    pub rate_bps: f64,
    /// Estimated time remaining in seconds
    pub eta_secs: Option<f64>,
}

impl TransferProgress {
    /// Calculate progress percentage
    #[must_use]
    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.bytes_transferred as f64 / self.total_bytes as f64) * 100.0
        }
    }

    /// Check if transfer is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.bytes_transferred >= self.total_bytes
    }
}

/// Lifecycle rule for automatic object management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRule {
    /// Rule ID
    pub id: String,
    /// Whether rule is enabled
    pub enabled: bool,
    /// Prefix filter
    pub prefix: Option<String>,
    /// Tags filter
    pub tags: HashMap<String, String>,
    /// Transitions
    pub transitions: Vec<Transition>,
    /// Expiration
    pub expiration: Option<Expiration>,
}

/// Storage class transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Days after object creation
    pub days: u32,
    /// Target storage class
    pub storage_class: StorageClass,
}

/// Object expiration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expiration {
    /// Days after object creation
    pub days: Option<u32>,
    /// Specific date
    pub date: Option<DateTime<Utc>>,
    /// Delete expired object delete markers
    pub expired_object_delete_marker: bool,
}

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Role ARN (AWS)
    pub role: Option<String>,
    /// Destination bucket
    pub destination_bucket: String,
    /// Destination storage class
    pub destination_storage_class: Option<StorageClass>,
    /// Replication rules
    pub rules: Vec<ReplicationRule>,
}

/// Replication rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationRule {
    /// Rule ID
    pub id: String,
    /// Whether rule is enabled
    pub enabled: bool,
    /// Prefix filter
    pub prefix: Option<String>,
    /// Priority
    pub priority: i32,
    /// Delete marker replication
    pub delete_marker_replication: bool,
}
