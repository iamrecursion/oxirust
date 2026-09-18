//! Google Cloud Storage implementation using REST API

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::sha2::{Digest, Sha256};
use rsa::signature::SignatureEncoding;
use rsa::RsaPrivateKey;
use serde::Deserialize;
use std::collections::HashMap;

use crate::error::{CloudError, Result};
use crate::types::{
    CloudStorage, DeleteResult, ListResult, ObjectInfo, ObjectMetadata, StorageClass, StorageStats,
    UploadOptions,
};

/// Google Cloud Storage backend using REST API
pub struct GcsStorage {
    client: Client,
    bucket: String,
    project_id: String,
    access_token: String,
    /// Service account email, required for V4 signed URLs.
    service_account_email: Option<String>,
    /// PEM-encoded RSA private key for the service account, required for V4 signed URLs.
    private_key_pem: Option<String>,
}

impl GcsStorage {
    /// Create a new GCS storage backend
    ///
    /// # Errors
    ///
    /// Returns an error if GCP authentication fails
    #[allow(clippy::unused_async)]
    pub async fn new(bucket: String, project_id: String) -> Result<Self> {
        crate::tls_provider::install_default_crypto_provider();
        let client = Client::new();

        // In production, would use Google Auth library to get token
        let access_token = std::env::var("GCP_ACCESS_TOKEN")
            .map_err(|_| CloudError::Authentication("GCP_ACCESS_TOKEN not set".to_string()))?;

        Ok(Self {
            client,
            bucket,
            project_id,
            access_token,
            service_account_email: None,
            private_key_pem: None,
        })
    }

    /// Create from existing access token
    #[must_use]
    pub fn from_token(bucket: String, project_id: String, access_token: String) -> Self {
        crate::tls_provider::install_default_crypto_provider();
        Self {
            client: Client::new(),
            bucket,
            project_id,
            access_token,
            service_account_email: None,
            private_key_pem: None,
        }
    }

    /// Create from service account credentials (enables V4 signed URL generation).
    #[must_use]
    pub fn from_service_account(
        bucket: String,
        project_id: String,
        access_token: String,
        service_account_email: String,
        private_key_pem: String,
    ) -> Self {
        crate::tls_provider::install_default_crypto_provider();
        Self {
            client: Client::new(),
            bucket,
            project_id,
            access_token,
            service_account_email: Some(service_account_email),
            private_key_pem: Some(private_key_pem),
        }
    }

    /// Create bucket
    ///
    /// # Errors
    ///
    /// Returns an error if bucket creation fails
    pub async fn create_bucket(&self, location: &str) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b?project={}",
            self.project_id
        );

        let request_body = serde_json::json!({
            "name": self.bucket,
            "location": location,
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to create bucket: {error_text}"
            )));
        }

        Ok(())
    }

    /// Delete bucket
    ///
    /// # Errors
    ///
    /// Returns an error if bucket deletion fails
    pub async fn delete_bucket(&self) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}",
            self.bucket
        );

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to delete bucket".to_string()));
        }

        Ok(())
    }

    /// Set object versioning
    ///
    /// # Errors
    ///
    /// Returns an error if setting versioning fails
    pub async fn enable_versioning(&self) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}",
            self.bucket
        );

        let request_body = serde_json::json!({
            "versioning": {
                "enabled": true
            }
        });

        let response = self
            .client
            .patch(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to enable versioning".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate a V4 signed URL for object download (GOOG4-RSA-SHA256).
    ///
    /// Requires the storage backend to have been created via
    /// [`GcsStorage::from_service_account`].  If service account credentials are
    /// absent, returns `CloudError::Authentication`.
    ///
    /// # Errors
    ///
    /// Returns an error if credentials are missing or if RSA signing fails.
    #[allow(clippy::unused_async)]
    pub async fn generate_signed_url_download(
        &self,
        object_name: &str,
        expires_in_secs: u64,
    ) -> Result<String> {
        let email = self.service_account_email.as_deref().ok_or_else(|| {
            CloudError::Authentication(
                "GCS V4 signed URLs require service account credentials (call from_service_account)"
                    .to_string(),
            )
        })?;
        let pem = self.private_key_pem.as_deref().ok_or_else(|| {
            CloudError::Authentication("GCS V4 signed URL: private_key_pem is not set".to_string())
        })?;

        let now = Utc::now();
        let date_str = now.format("%Y%m%d").to_string();
        let datetime_str = now.format("%Y%m%dT%H%M%SZ").to_string();

        // Credential scope
        let credential_scope = format!("{date_str}/auto/storage/goog4_request");
        let credential = format!("{email}/{credential_scope}");
        let credential_encoded = urlencoding::encode(&credential).to_string();

        // Sorted query parameters (without X-Goog-Signature)
        let signed_headers = "host";
        let query_params = format!(
            "X-Goog-Algorithm=GOOG4-RSA-SHA256\
             &X-Goog-Credential={credential_encoded}\
             &X-Goog-Date={datetime_str}\
             &X-Goog-Expires={expires_in_secs}\
             &X-Goog-SignedHeaders={signed_headers}"
        );

        // Canonical request
        let canonical_request = format!(
            "GET\n/{bucket}/{object}\n{query_params}\nhost:storage.googleapis.com\n\n{signed_headers}\nUNSIGNED-PAYLOAD",
            bucket = self.bucket,
            object = urlencoding::encode(object_name),
        );

        // String to sign
        let canonical_hash = hex::encode(Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign =
            format!("GOOG4-RSA-SHA256\n{datetime_str}\n{credential_scope}\n{canonical_hash}");

        // RSA-SHA256 sign using PKCS#1 v1.5 — deterministic, no RNG needed
        let private_key = Self::load_rsa_private_key(pem)?;
        let signing_key = rsa::pkcs1v15::SigningKey::<Sha256>::new(private_key);
        // Use sign() (deterministic) instead of try_sign_with_rng to avoid rand version conflict
        use rsa::signature::Signer;
        let signature = signing_key.sign(string_to_sign.as_bytes());
        let sig_hex = hex::encode(signature.to_bytes());

        let url = format!(
            "https://storage.googleapis.com/{}/{object}?{query_params}&X-Goog-Signature={sig_hex}",
            self.bucket,
            object = urlencoding::encode(object_name),
        );

        Ok(url)
    }

    /// Load an RSA private key from PEM (PKCS#8 or PKCS#1).
    fn load_rsa_private_key(pem: &str) -> Result<RsaPrivateKey> {
        // Try PKCS#8 first, then fall back to PKCS#1
        if let Ok(key) = RsaPrivateKey::from_pkcs8_pem(pem) {
            return Ok(key);
        }
        RsaPrivateKey::from_pkcs1_pem(pem).map_err(|e| {
            CloudError::Authentication(format!("Failed to parse RSA private key: {e}"))
        })
    }

    /// Generate resumable upload URL
    ///
    /// # Errors
    ///
    /// Returns an error if URL generation fails
    pub async fn initiate_resumable_upload(&self, object_name: &str) -> Result<String> {
        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=resumable",
            self.bucket
        );

        let request_body = serde_json::json!({
            "name": object_name,
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to initiate resumable upload".to_string(),
            ));
        }

        let upload_url = response
            .headers()
            .get("Location")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| CloudError::Storage("No upload URL in response".to_string()))?;

        Ok(upload_url.to_string())
    }

    /// Convert GCS storage class to our storage class
    fn from_gcs_storage_class(class: &str) -> StorageClass {
        match class {
            "STANDARD" => StorageClass::Standard,
            "NEARLINE" => StorageClass::InfrequentAccess,
            "COLDLINE" => StorageClass::Glacier,
            "ARCHIVE" => StorageClass::DeepArchive,
            _ => StorageClass::Standard,
        }
    }

    /// Convert our storage class to GCS storage class
    fn to_gcs_storage_class(class: StorageClass) -> &'static str {
        match class {
            StorageClass::Standard => "STANDARD",
            StorageClass::InfrequentAccess => "NEARLINE",
            StorageClass::Glacier => "COLDLINE",
            StorageClass::DeepArchive => "ARCHIVE",
            _ => "STANDARD",
        }
    }
}

#[async_trait]
impl CloudStorage for GcsStorage {
    async fn upload(&self, key: &str, data: Bytes) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}",
            self.bucket,
            urlencoding::encode(key)
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to upload object: {error_text}"
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
        let url = format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=multipart",
            self.bucket
        );

        let mut metadata = serde_json::json!({
            "name": key,
        });

        if let Some(content_type) = options.content_type {
            metadata["contentType"] = serde_json::Value::String(content_type);
        }

        if let Some(cache_control) = options.cache_control {
            metadata["cacheControl"] = serde_json::Value::String(cache_control);
        }

        if let Some(storage_class) = options.storage_class {
            metadata["storageClass"] =
                serde_json::Value::String(Self::to_gcs_storage_class(storage_class).to_string());
        }

        if !options.metadata.is_empty() {
            metadata["metadata"] = serde_json::to_value(&options.metadata)?;
        }

        // Create multipart request
        let boundary = "oximedia_boundary_123456789";
        let mut body = Vec::new();

        // Part 1: Metadata
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
        body.extend_from_slice(metadata.to_string().as_bytes());
        body.extend_from_slice(b"\r\n");

        // Part 2: Data
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n\r\n");
        body.extend_from_slice(&data);
        body.extend_from_slice(b"\r\n");

        // End boundary
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .header(
                "Content-Type",
                format!("multipart/related; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::Storage(format!(
                "Failed to upload object: {error_text}"
            )));
        }

        Ok(())
    }

    async fn download(&self, key: &str) -> Result<Bytes> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            self.bucket,
            urlencoding::encode(key)
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to download object".to_string()));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    async fn download_range(&self, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
            self.bucket,
            urlencoding::encode(key)
        );

        let range_header = format!("bytes={start}-{end}");

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .header("Range", range_header)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to download object range".to_string(),
            ));
        }

        let data = response.bytes().await?;
        Ok(data)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<ObjectInfo>> {
        let mut objects = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut url = format!(
                "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}",
                self.bucket,
                urlencoding::encode(prefix)
            );

            if let Some(token) = &page_token {
                url.push_str(&format!("&pageToken={}", urlencoding::encode(token)));
            }

            let response = self
                .client
                .get(&url)
                .bearer_auth(&self.access_token)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(CloudError::Storage("Failed to list objects".to_string()));
            }

            let list_response: GcsListResponse = response.json().await?;

            if let Some(items) = list_response.items {
                for item in items {
                    objects.push(ObjectInfo {
                        key: item.name,
                        size: item.size.parse().unwrap_or(0),
                        last_modified: DateTime::parse_from_rfc3339(&item.updated)
                            .ok()
                            .map_or_else(Utc::now, |dt| dt.with_timezone(&Utc)),
                        etag: Some(item.etag),
                        storage_class: item
                            .storage_class
                            .as_ref()
                            .map(|s| Self::from_gcs_storage_class(s)),
                        content_type: Some(item.content_type),
                    });
                }
            }

            if let Some(next_token) = list_response.next_page_token {
                page_token = Some(next_token);
            } else {
                break;
            }
        }

        Ok(objects)
    }

    async fn list_paginated(
        &self,
        prefix: &str,
        continuation_token: Option<String>,
        max_keys: usize,
    ) -> Result<ListResult> {
        let mut url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o?prefix={}&maxResults={}",
            self.bucket,
            urlencoding::encode(prefix),
            max_keys
        );

        if let Some(token) = continuation_token {
            url.push_str(&format!("&pageToken={}", urlencoding::encode(&token)));
        }

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to list objects".to_string()));
        }

        let list_response: GcsListResponse = response.json().await?;

        let mut objects = Vec::new();
        if let Some(items) = list_response.items {
            for item in items {
                objects.push(ObjectInfo {
                    key: item.name,
                    size: item.size.parse().unwrap_or(0),
                    last_modified: DateTime::parse_from_rfc3339(&item.updated)
                        .ok()
                        .map_or_else(Utc::now, |dt| dt.with_timezone(&Utc)),
                    etag: Some(item.etag),
                    storage_class: item
                        .storage_class
                        .as_ref()
                        .map(|s| Self::from_gcs_storage_class(s)),
                    content_type: Some(item.content_type),
                });
            }
        }

        let is_truncated = list_response.next_page_token.is_some();
        Ok(ListResult {
            objects,
            continuation_token: list_response.next_page_token,
            is_truncated,
            common_prefixes: list_response.prefixes.unwrap_or_default(),
        })
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(key)
        );

        let response = self
            .client
            .delete(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            return Err(CloudError::Storage("Failed to delete object".to_string()));
        }

        Ok(())
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<Vec<DeleteResult>> {
        // GCS doesn't have native batch delete, so delete one by one
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

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(key)
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to get object metadata".to_string(),
            ));
        }

        let object: GcsObject = response.json().await?;

        let info = ObjectInfo {
            key: object.name,
            size: object.size.parse().unwrap_or(0),
            last_modified: DateTime::parse_from_rfc3339(&object.updated)
                .ok()
                .map_or_else(Utc::now, |dt| dt.with_timezone(&Utc)),
            etag: Some(object.etag),
            storage_class: object
                .storage_class
                .as_ref()
                .map(|s| Self::from_gcs_storage_class(s)),
            content_type: Some(object.content_type.clone()),
        };

        Ok(ObjectMetadata {
            info,
            user_metadata: object.metadata.unwrap_or_default(),
            system_metadata: HashMap::new(),
            tags: HashMap::new(),
            content_encoding: object.content_encoding,
            content_language: object.content_language,
            cache_control: object.cache_control,
            content_disposition: object.content_disposition,
        })
    }

    async fn update_metadata(&self, key: &str, metadata: HashMap<String, String>) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(key)
        );

        let request_body = serde_json::json!({
            "metadata": metadata,
        });

        let response = self
            .client
            .patch(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to update metadata".to_string()));
        }

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(key)
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    async fn copy(&self, source_key: &str, dest_key: &str) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}/copyTo/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(source_key),
            self.bucket,
            urlencoding::encode(dest_key)
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage("Failed to copy object".to_string()));
        }

        Ok(())
    }

    async fn presigned_download_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        self.generate_signed_url_download(key, expires_in_secs)
            .await
    }

    async fn presigned_upload_url(&self, key: &str, expires_in_secs: u64) -> Result<String> {
        // Similar to download, in production would use proper signing
        Ok(format!(
            "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}&expires={}",
            self.bucket,
            urlencoding::encode(key),
            expires_in_secs
        ))
    }

    async fn set_storage_class(&self, key: &str, class: StorageClass) -> Result<()> {
        let url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
            self.bucket,
            urlencoding::encode(key)
        );

        let request_body = serde_json::json!({
            "storageClass": Self::to_gcs_storage_class(class),
        });

        let response = self
            .client
            .patch(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::Storage(
                "Failed to set storage class".to_string(),
            ));
        }

        Ok(())
    }

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

#[derive(Debug, Deserialize)]
struct GcsListResponse {
    items: Option<Vec<GcsObject>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
    prefixes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct GcsObject {
    name: String,
    size: String,
    updated: String,
    etag: String,
    #[serde(rename = "storageClass")]
    storage_class: Option<String>,
    #[serde(rename = "contentType")]
    content_type: String,
    #[serde(rename = "contentEncoding")]
    content_encoding: Option<String>,
    #[serde(rename = "contentLanguage")]
    content_language: Option<String>,
    #[serde(rename = "cacheControl")]
    cache_control: Option<String>,
    #[serde(rename = "contentDisposition")]
    content_disposition: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_class_conversion() {
        let class = StorageClass::Standard;
        let gcs_class = GcsStorage::to_gcs_storage_class(class);
        assert_eq!(gcs_class, "STANDARD");

        let converted_back = GcsStorage::from_gcs_storage_class(gcs_class);
        assert_eq!(converted_back, StorageClass::Standard);
    }

    #[test]
    fn test_gcs_storage_classes() {
        assert_eq!(
            GcsStorage::to_gcs_storage_class(StorageClass::InfrequentAccess),
            "NEARLINE"
        );
        assert_eq!(
            GcsStorage::to_gcs_storage_class(StorageClass::Glacier),
            "COLDLINE"
        );
        assert_eq!(
            GcsStorage::to_gcs_storage_class(StorageClass::DeepArchive),
            "ARCHIVE"
        );
    }
}
