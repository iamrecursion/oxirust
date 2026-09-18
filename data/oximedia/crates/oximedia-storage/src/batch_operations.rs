#![allow(dead_code)]
//! Batch operations — parallel multi-object upload and download.
//!
//! `BatchOperations` executes concurrent uploads and downloads against any
//! `CloudStorage` implementation.  The `concurrency` parameter caps the number
//! of simultaneous in-flight operations using a `tokio::sync::Semaphore`.
//! Rayon is used for CPU-bound pre/post-processing while Tokio handles I/O.

use crate::{CloudStorage, DownloadOptions, UploadOptions};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// A single upload job for batch processing.
#[derive(Debug, Clone)]
pub struct BatchUploadJob {
    /// Target object key.
    pub key: String,
    /// Raw bytes to upload.
    pub data: Vec<u8>,
    /// Upload options (content-type, metadata, etc.).
    pub options: UploadOptions,
}

impl BatchUploadJob {
    /// Convenience constructor.
    pub fn new(key: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            key: key.into(),
            data,
            options: UploadOptions::default(),
        }
    }

    /// Builder: set upload options.
    pub fn with_options(mut self, options: UploadOptions) -> Self {
        self.options = options;
        self
    }
}

/// Result for a single batch operation item.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Object key this result corresponds to.
    pub key: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Error description if the operation failed.
    pub error: Option<String>,
    /// Number of bytes transferred (upload size or downloaded size).
    pub bytes: u64,
}

impl BatchResult {
    /// Create a successful result.
    pub fn ok(key: impl Into<String>, bytes: u64) -> Self {
        Self {
            key: key.into(),
            success: true,
            error: None,
            bytes,
        }
    }

    /// Create a failed result.
    pub fn err(key: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            success: false,
            error: Some(error.into()),
            bytes: 0,
        }
    }
}

/// Summary of a completed batch operation.
#[derive(Debug, Default, Clone)]
pub struct BatchSummary {
    /// Number of successfully processed items.
    pub succeeded: usize,
    /// Number of failed items.
    pub failed: usize,
    /// Total bytes transferred.
    pub total_bytes: u64,
}

impl BatchSummary {
    /// Build a summary from a slice of results.
    pub fn from_results(results: &[BatchResult]) -> Self {
        let mut s = Self::default();
        for r in results {
            if r.success {
                s.succeeded += 1;
                s.total_bytes += r.bytes;
            } else {
                s.failed += 1;
            }
        }
        s
    }
}

/// Executes batch upload and download operations against a `CloudStorage` backend.
///
/// Concurrency is controlled by a `tokio::sync::Semaphore`; up to `concurrency`
/// I/O tasks run simultaneously.  Results are returned in the same order as the
/// input list.
pub struct BatchOperations {
    storage: Arc<dyn CloudStorage>,
}

impl BatchOperations {
    /// Wrap an existing `CloudStorage` implementation.
    pub fn new(storage: Arc<dyn CloudStorage>) -> Self {
        Self { storage }
    }

    /// Upload many objects concurrently.
    ///
    /// `concurrency` caps how many uploads run simultaneously.
    /// Results are returned in the same order as `jobs`.
    pub async fn upload_many(
        &self,
        jobs: Vec<BatchUploadJob>,
        concurrency: usize,
    ) -> Vec<BatchResult> {
        let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
        let mut handles = Vec::with_capacity(jobs.len());

        for job in jobs {
            let sem = Arc::clone(&semaphore);
            let storage = Arc::clone(&self.storage);
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.map_err(|e| e.to_string());
                let key = job.key.clone();
                let bytes_len = job.data.len() as u64;

                use bytes::Bytes;
                use futures::stream;

                let data = Bytes::from(job.data);
                let size = data.len() as u64;
                let stream = Box::pin(stream::once(async move {
                    Ok::<Bytes, crate::StorageError>(data)
                }));

                match storage
                    .upload_stream(&job.key, stream, Some(size), job.options)
                    .await
                {
                    Ok(_etag) => BatchResult::ok(key, bytes_len),
                    Err(e) => BatchResult::err(key, e.to_string()),
                }
            });
            handles.push(handle);
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(r) => results.push(r),
                Err(e) => results.push(BatchResult::err("<unknown>", e.to_string())),
            }
        }
        results
    }

    /// Download many objects concurrently.
    ///
    /// Returns a `BatchResult` per key; on success `bytes` is the downloaded size.
    /// The actual downloaded content is not returned — callers needing data should
    /// use `CloudStorage::download_stream` directly.  This method is useful for
    /// pre-warming caches or verifying availability.
    pub async fn download_many(&self, keys: Vec<String>, concurrency: usize) -> Vec<BatchResult> {
        let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
        let mut handles = Vec::with_capacity(keys.len());

        for key in keys {
            let sem = Arc::clone(&semaphore);
            let storage = Arc::clone(&self.storage);
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.map_err(|e| e.to_string());
                let key_clone = key.clone();

                use futures::StreamExt;

                let result: Result<u64, crate::StorageError> = async {
                    let mut stream = storage
                        .download_stream(&key, DownloadOptions::default())
                        .await?;

                    let mut total: u64 = 0;
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        total += chunk.len() as u64;
                    }
                    Ok(total)
                }
                .await;

                match result {
                    Ok(bytes) => BatchResult::ok(key_clone, bytes),
                    Err(e) => BatchResult::err(key_clone, e.to_string()),
                }
            });
            handles.push(handle);
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(r) => results.push(r),
                Err(e) => results.push(BatchResult::err("<unknown>", e.to_string())),
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::LocalStorage;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn make_storage() -> (Arc<dyn CloudStorage>, TempDir) {
        let dir = TempDir::new().expect("temp dir");
        let storage = LocalStorage::new(dir.path()).await.expect("local storage");
        (Arc::new(storage) as Arc<dyn CloudStorage>, dir)
    }

    // ── BatchUploadJob ─────────────────────────────────────────────────────────

    #[test]
    fn test_batch_upload_job_new() {
        let job = BatchUploadJob::new("key.mp4", vec![1, 2, 3]);
        assert_eq!(job.key, "key.mp4");
        assert_eq!(job.data.len(), 3);
        assert!(job.options.content_type.is_none());
    }

    #[test]
    fn test_batch_upload_job_with_options() {
        let opts = UploadOptions {
            content_type: Some("video/mp4".to_string()),
            ..Default::default()
        };
        let job = BatchUploadJob::new("k", vec![]).with_options(opts.clone());
        assert_eq!(job.options.content_type, opts.content_type);
    }

    // ── BatchResult ────────────────────────────────────────────────────────────

    #[test]
    fn test_batch_result_ok() {
        let r = BatchResult::ok("a.mp4", 1024);
        assert!(r.success);
        assert!(r.error.is_none());
        assert_eq!(r.bytes, 1024);
    }

    #[test]
    fn test_batch_result_err() {
        let r = BatchResult::err("b.mp4", "network timeout");
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("network timeout"));
        assert_eq!(r.bytes, 0);
    }

    // ── BatchSummary ───────────────────────────────────────────────────────────

    #[test]
    fn test_batch_summary_from_results() {
        let results = vec![
            BatchResult::ok("a", 100),
            BatchResult::ok("b", 200),
            BatchResult::err("c", "failed"),
        ];
        let s = BatchSummary::from_results(&results);
        assert_eq!(s.succeeded, 2);
        assert_eq!(s.failed, 1);
        assert_eq!(s.total_bytes, 300);
    }

    #[test]
    fn test_batch_summary_empty() {
        let s = BatchSummary::from_results(&[]);
        assert_eq!(s.succeeded, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.total_bytes, 0);
    }

    // ── upload_many / download_many via LocalStorage ──────────────────────────

    #[tokio::test]
    async fn test_upload_many_all_succeed() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let jobs = vec![
            BatchUploadJob::new("upload/a.bin", vec![0u8; 256]),
            BatchUploadJob::new("upload/b.bin", vec![1u8; 512]),
            BatchUploadJob::new("upload/c.bin", vec![2u8; 128]),
        ];
        let results = ops.upload_many(jobs, 2).await;
        assert_eq!(results.len(), 3);
        let summary = BatchSummary::from_results(&results);
        assert_eq!(summary.succeeded, 3);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.total_bytes, 256 + 512 + 128);
    }

    #[tokio::test]
    async fn test_upload_then_download_many() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(Arc::clone(&storage));

        // First upload some files
        let jobs = vec![
            BatchUploadJob::new("dl/x.bin", vec![10u8; 100]),
            BatchUploadJob::new("dl/y.bin", vec![20u8; 200]),
        ];
        let up_results = ops.upload_many(jobs, 4).await;
        assert!(up_results.iter().all(|r| r.success));

        // Then download them
        let keys = vec!["dl/x.bin".to_string(), "dl/y.bin".to_string()];
        let dl_results = ops.download_many(keys, 4).await;
        assert_eq!(dl_results.len(), 2);
        let dl_summary = BatchSummary::from_results(&dl_results);
        assert_eq!(dl_summary.succeeded, 2);
        assert_eq!(dl_summary.total_bytes, 300);
    }

    #[tokio::test]
    async fn test_download_many_missing_key_fails() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let keys = vec!["does/not/exist.mp4".to_string()];
        let results = ops.download_many(keys, 1).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].error.is_some());
    }

    #[tokio::test]
    async fn test_upload_many_concurrency_1() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let jobs: Vec<_> = (0..5)
            .map(|i| BatchUploadJob::new(format!("seq/{i}.bin"), vec![i as u8; 64]))
            .collect();
        let results = ops.upload_many(jobs, 1).await;
        assert_eq!(BatchSummary::from_results(&results).succeeded, 5);
    }

    #[tokio::test]
    async fn test_upload_many_concurrency_greater_than_jobs() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let jobs = vec![BatchUploadJob::new("solo/a.bin", vec![42u8; 32])];
        let results = ops.upload_many(jobs, 10).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
    }

    #[tokio::test]
    async fn test_download_many_empty_list() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let results = ops.download_many(vec![], 4).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_upload_many_empty_list() {
        let (storage, _dir) = make_storage().await;
        let ops = BatchOperations::new(storage);
        let results = ops.upload_many(vec![], 4).await;
        assert!(results.is_empty());
    }
}
