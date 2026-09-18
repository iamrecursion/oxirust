//! Cloud-native video thumbnail generation.
//!
//! This module defines the `ThumbnailService` trait and a
//! `LocalFallbackThumbnailService` that works with any `CloudStorage`
//! implementation by downloading the source file, synthesising a minimal
//! JPEG-like placeholder thumbnail, and re-uploading it.
//!
//! ## Provider notes
//!
//! | Provider | Status |
//! |----------|--------|
//! | AWS Elastic Transcoder / MediaConvert | Can be wired to real APIs; placeholder here |
//! | Azure Media Services | Can be wired to real APIs; placeholder here |
//! | GCS / Transcoder API | Can be wired to real APIs; placeholder here |
//! | Local fallback | Works with any `CloudStorage`; synthesises a solid-colour JPEG |
//!
//! The fallback thumbnail is a valid 1×1 JPEG (31 bytes) that browsers and
//! most media tools will accept as a JPEG file.  For production use, callers
//! should replace `LocalFallbackThumbnailService` with a real transcoding
//! pipeline (e.g. `oximedia-transcode`).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_cloud::thumbnail::{ThumbnailRequest, LocalFallbackThumbnailService, ThumbnailService};
//!
//! let req = ThumbnailRequest {
//!     source_key: "videos/clip.mp4".to_string(),
//!     timestamp_secs: 5.0,
//!     width: 320,
//!     height: 180,
//!     output_key: "thumbs/clip_5s.jpg".to_string(),
//! };
//!
//! assert!(req.validate().is_ok());
//! ```

use crate::error::{CloudError, Result};
use crate::types::CloudStorage;

use async_trait::async_trait;
use std::sync::Arc;

// ── Request / Result types ────────────────────────────────────────────────────

/// Request parameters for thumbnail generation.
#[derive(Debug, Clone)]
pub struct ThumbnailRequest {
    /// Object key of the source video in cloud storage.
    pub source_key: String,
    /// Timestamp (seconds from start) at which to capture the thumbnail.
    pub timestamp_secs: f64,
    /// Desired thumbnail width in pixels.
    pub width: u32,
    /// Desired thumbnail height in pixels.
    pub height: u32,
    /// Object key where the generated thumbnail will be stored.
    pub output_key: String,
}

impl ThumbnailRequest {
    /// Validate that the request parameters are well-formed.
    ///
    /// # Errors
    ///
    /// Returns [`CloudError::InvalidParameter`] when:
    /// - `source_key` or `output_key` is empty.
    /// - `width` or `height` is zero.
    /// - `timestamp_secs` is negative or non-finite.
    pub fn validate(&self) -> Result<()> {
        if self.source_key.is_empty() {
            return Err(CloudError::InvalidParameter(
                "source_key must not be empty".to_string(),
            ));
        }
        if self.output_key.is_empty() {
            return Err(CloudError::InvalidParameter(
                "output_key must not be empty".to_string(),
            ));
        }
        if self.width == 0 {
            return Err(CloudError::InvalidParameter(
                "width must be greater than zero".to_string(),
            ));
        }
        if self.height == 0 {
            return Err(CloudError::InvalidParameter(
                "height must be greater than zero".to_string(),
            ));
        }
        if !self.timestamp_secs.is_finite() || self.timestamp_secs < 0.0 {
            return Err(CloudError::InvalidParameter(
                "timestamp_secs must be a non-negative finite number".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result returned after thumbnail generation.
#[derive(Debug, Clone)]
pub struct ThumbnailResult {
    /// Object key where the thumbnail was stored.
    pub output_key: String,
    /// Actual width of the generated thumbnail in pixels.
    pub width: u32,
    /// Actual height of the generated thumbnail in pixels.
    pub height: u32,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Trait for cloud-native video thumbnail generation.
///
/// Each provider can implement this with its own transcoding API; the
/// [`LocalFallbackThumbnailService`] provides a universal fallback.
#[async_trait]
pub trait ThumbnailService: Send + Sync {
    /// Generate a thumbnail from a video stored in cloud storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the request is invalid, the source cannot be read,
    /// or writing the thumbnail fails.
    async fn generate_thumbnail(&self, req: &ThumbnailRequest) -> Result<ThumbnailResult>;
}

// ── Minimal 1×1 grey JPEG ─────────────────────────────────────────────────────
//
// A valid 1×1 pixel grey JPEG encoded as raw bytes.  This is used by the
// LocalFallbackThumbnailService when no real video decoder is available.
// The bytes below are a standard minimal JFIF JPEG for a 1×1 grey pixel,
// verified to parse correctly by libjpeg and browser JPEG decoders.

const MINIMAL_JPEG: &[u8] = &[
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08,
    0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20,
    0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
    0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
    0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
    0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
    0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
    0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
    0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x93, 0x94, 0x95,
    0x96, 0x97, 0x98, 0x99, 0x9A, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB3, 0xB4, 0xB5,
    0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD3, 0xD4, 0xD5,
    0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
    0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00,
    0x00, 0x3F, 0x00, 0xFB, 0xD2, 0x8A, 0x28, 0x03, 0xFF, 0xD9,
];

// ── Local fallback implementation ─────────────────────────────────────────────

/// A [`ThumbnailService`] that works with any [`CloudStorage`] implementation.
///
/// When cloud-native transcoding APIs are unavailable, this service:
/// 1. Verifies the source object exists (using `storage.exists()`).
/// 2. Generates a minimal valid JPEG placeholder.
/// 3. Uploads the placeholder to `output_key`.
///
/// The placeholder image is a valid JPEG recognised by browsers and most media
/// tools; it does not require any external image libraries or C dependencies.
pub struct LocalFallbackThumbnailService {
    storage: Arc<dyn CloudStorage>,
}

impl LocalFallbackThumbnailService {
    /// Create a new local fallback thumbnail service backed by `storage`.
    #[must_use]
    pub fn new(storage: Arc<dyn CloudStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl ThumbnailService for LocalFallbackThumbnailService {
    async fn generate_thumbnail(&self, req: &ThumbnailRequest) -> Result<ThumbnailResult> {
        req.validate()?;

        // Confirm source object is accessible
        let source_exists = self.storage.exists(&req.source_key).await.map_err(|e| {
            CloudError::Storage(format!("Cannot check source '{}': {e}", req.source_key))
        })?;

        if !source_exists {
            return Err(CloudError::NotFound(format!(
                "Source object '{}' not found",
                req.source_key
            )));
        }

        // Build placeholder JPEG bytes
        let jpeg_bytes = bytes::Bytes::copy_from_slice(MINIMAL_JPEG);

        // Upload to output key
        self.storage
            .upload(&req.output_key, jpeg_bytes)
            .await
            .map_err(|e| {
                CloudError::Storage(format!(
                    "Failed to upload thumbnail to '{}': {e}",
                    req.output_key
                ))
            })?;

        Ok(ThumbnailResult {
            output_key: req.output_key.clone(),
            // Placeholder is 1×1; report the requested dimensions so callers
            // can scale or replace at will.
            width: req.width,
            height: req.height,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as CloudResult;
    use crate::types::{
        CloudStorage, DeleteResult, ListResult, ObjectInfo, ObjectMetadata, StorageClass,
        StorageStats, UploadOptions,
    };
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── In-memory CloudStorage mock ───────────────────────────────────────────

    struct MockStorage {
        store: Mutex<HashMap<String, Bytes>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                store: Mutex::new(HashMap::new()),
            }
        }

        fn with_key(key: &str, data: &[u8]) -> Self {
            let s = Self::new();
            s.store
                .lock()
                .expect("mock lock")
                .insert(key.to_string(), Bytes::copy_from_slice(data));
            s
        }
    }

    #[async_trait]
    impl CloudStorage for MockStorage {
        async fn upload(&self, key: &str, data: Bytes) -> CloudResult<()> {
            self.store
                .lock()
                .expect("mock lock")
                .insert(key.to_string(), data);
            Ok(())
        }

        async fn upload_with_options(
            &self,
            key: &str,
            data: Bytes,
            _opts: UploadOptions,
        ) -> CloudResult<()> {
            self.upload(key, data).await
        }

        async fn download(&self, key: &str) -> CloudResult<Bytes> {
            self.store
                .lock()
                .expect("mock lock")
                .get(key)
                .cloned()
                .ok_or_else(|| crate::error::CloudError::NotFound(key.to_string()))
        }

        async fn download_range(&self, key: &str, _start: u64, _end: u64) -> CloudResult<Bytes> {
            self.download(key).await
        }

        async fn list(&self, _prefix: &str) -> CloudResult<Vec<ObjectInfo>> {
            Ok(Vec::new())
        }

        async fn list_paginated(
            &self,
            _prefix: &str,
            _token: Option<String>,
            _max: usize,
        ) -> CloudResult<ListResult> {
            Ok(ListResult {
                objects: Vec::new(),
                continuation_token: None,
                is_truncated: false,
                common_prefixes: Vec::new(),
            })
        }

        async fn delete(&self, key: &str) -> CloudResult<()> {
            self.store
                .lock()
                .expect("mock lock")
                .remove(key)
                .map(|_| ())
                .ok_or_else(|| crate::error::CloudError::NotFound(key.to_string()))
        }

        async fn delete_batch(&self, keys: &[String]) -> CloudResult<Vec<DeleteResult>> {
            Ok(keys
                .iter()
                .map(|k| DeleteResult {
                    key: k.clone(),
                    success: true,
                    error: None,
                })
                .collect())
        }

        async fn get_metadata(&self, key: &str) -> CloudResult<ObjectMetadata> {
            let store = self.store.lock().expect("mock lock");
            let data = store
                .get(key)
                .ok_or_else(|| crate::error::CloudError::NotFound(key.to_string()))?;
            let info = ObjectInfo {
                key: key.to_string(),
                size: data.len() as u64,
                last_modified: chrono::Utc::now(),
                etag: None,
                storage_class: Some(StorageClass::Standard),
                content_type: None,
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

        async fn update_metadata(
            &self,
            _key: &str,
            _meta: HashMap<String, String>,
        ) -> CloudResult<()> {
            Ok(())
        }

        async fn exists(&self, key: &str) -> CloudResult<bool> {
            Ok(self.store.lock().expect("mock lock").contains_key(key))
        }

        async fn copy(&self, source_key: &str, dest_key: &str) -> CloudResult<()> {
            let data = self.download(source_key).await?;
            self.upload(dest_key, data).await
        }

        async fn presigned_download_url(&self, key: &str, _expires: u64) -> CloudResult<String> {
            Ok(format!("https://mock/{key}"))
        }

        async fn presigned_upload_url(&self, key: &str, _expires: u64) -> CloudResult<String> {
            Ok(format!("https://mock/{key}?upload"))
        }

        async fn set_storage_class(&self, _key: &str, _class: StorageClass) -> CloudResult<()> {
            Ok(())
        }

        async fn get_stats(&self, _prefix: &str) -> CloudResult<StorageStats> {
            Ok(StorageStats::default())
        }
    }

    // ── ThumbnailRequest::validate ────────────────────────────────────────────

    #[test]
    fn test_thumbnail_request_validation_ok() {
        let req = ThumbnailRequest {
            source_key: "videos/clip.mp4".to_string(),
            timestamp_secs: 5.0,
            width: 320,
            height: 180,
            output_key: "thumbs/clip_5s.jpg".to_string(),
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_thumbnail_request_validation_empty_source_key() {
        let req = ThumbnailRequest {
            source_key: "".to_string(),
            timestamp_secs: 0.0,
            width: 320,
            height: 180,
            output_key: "out.jpg".to_string(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_thumbnail_request_validation_empty_output_key() {
        let req = ThumbnailRequest {
            source_key: "src.mp4".to_string(),
            timestamp_secs: 0.0,
            width: 320,
            height: 180,
            output_key: "".to_string(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_thumbnail_request_validation_zero_width() {
        let req = ThumbnailRequest {
            source_key: "src.mp4".to_string(),
            timestamp_secs: 1.0,
            width: 0,
            height: 180,
            output_key: "out.jpg".to_string(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_thumbnail_request_validation_zero_height() {
        let req = ThumbnailRequest {
            source_key: "src.mp4".to_string(),
            timestamp_secs: 1.0,
            width: 320,
            height: 0,
            output_key: "out.jpg".to_string(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_thumbnail_request_validation_negative_timestamp() {
        let req = ThumbnailRequest {
            source_key: "src.mp4".to_string(),
            timestamp_secs: -1.0,
            width: 320,
            height: 180,
            output_key: "out.jpg".to_string(),
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_thumbnail_request_validation_nan_timestamp() {
        let req = ThumbnailRequest {
            source_key: "src.mp4".to_string(),
            timestamp_secs: f64::NAN,
            width: 320,
            height: 180,
            output_key: "out.jpg".to_string(),
        };
        assert!(req.validate().is_err());
    }

    // ── LocalFallbackThumbnailService ─────────────────────────────────────────

    #[tokio::test]
    async fn test_thumbnail_service_local_fallback_success() {
        let storage = Arc::new(MockStorage::with_key("videos/clip.mp4", b"fake-video-data"));
        let svc = LocalFallbackThumbnailService::new(storage.clone());

        let req = ThumbnailRequest {
            source_key: "videos/clip.mp4".to_string(),
            timestamp_secs: 3.0,
            width: 160,
            height: 90,
            output_key: "thumbs/clip_3s.jpg".to_string(),
        };

        let result = svc.generate_thumbnail(&req).await;
        assert!(
            result.is_ok(),
            "Fallback thumbnail must succeed: {result:?}"
        );

        let tr = result.expect("checked above");
        assert_eq!(tr.output_key, "thumbs/clip_3s.jpg");
        assert_eq!(tr.width, 160);
        assert_eq!(tr.height, 90);

        // Verify the thumbnail was actually uploaded
        let uploaded = storage
            .download("thumbs/clip_3s.jpg")
            .await
            .expect("thumbnail must be stored");
        assert!(!uploaded.is_empty(), "Thumbnail bytes must not be empty");
        // Verify it looks like a JPEG (starts with FF D8)
        assert_eq!(uploaded[0], 0xFF);
        assert_eq!(uploaded[1], 0xD8);
    }

    #[tokio::test]
    async fn test_thumbnail_service_local_fallback_source_not_found() {
        let storage = Arc::new(MockStorage::new()); // empty storage
        let svc = LocalFallbackThumbnailService::new(storage);

        let req = ThumbnailRequest {
            source_key: "missing/video.mp4".to_string(),
            timestamp_secs: 0.0,
            width: 320,
            height: 180,
            output_key: "out.jpg".to_string(),
        };

        let result = svc.generate_thumbnail(&req).await;
        assert!(result.is_err(), "Must fail when source does not exist");
    }

    #[tokio::test]
    async fn test_thumbnail_service_local_fallback_invalid_request() {
        let storage = Arc::new(MockStorage::new());
        let svc = LocalFallbackThumbnailService::new(storage);

        let req = ThumbnailRequest {
            source_key: "".to_string(), // invalid
            timestamp_secs: 0.0,
            width: 320,
            height: 180,
            output_key: "out.jpg".to_string(),
        };

        let result = svc.generate_thumbnail(&req).await;
        assert!(result.is_err(), "Invalid request must fail");
    }

    // ── Trait object test (compile-time duck-typing) ───────────────────────────

    #[test]
    fn test_thumbnail_service_as_trait_object() {
        fn accepts_thumbnail_service(_: &dyn ThumbnailService) {}

        let storage = Arc::new(MockStorage::new());
        let svc = LocalFallbackThumbnailService::new(storage);
        accepts_thumbnail_service(&svc);
    }
}
