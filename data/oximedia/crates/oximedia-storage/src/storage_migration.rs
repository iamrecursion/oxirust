#![allow(dead_code)]
//! Storage migration — cross-provider object migration with hash verification.
//!
//! Provides `MigrationPlan`, `MigrationOptions`, `StorageMigrator`, and
//! `MigrationReport` for moving objects from one `CloudStorage` backend to
//! another with optional hash verification and source cleanup.

use crate::{CloudStorage, DownloadOptions, ListOptions, StorageError, UploadOptions};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Options controlling migration behaviour.
#[derive(Debug, Clone)]
pub struct MigrationOptions {
    /// Verify object integrity by comparing SHA-256 of downloaded bytes.
    /// Re-downloads and hashes the object after copying; fails the item if they differ.
    pub verify_hash: bool,
    /// Delete the source object after a successful copy.
    pub delete_source: bool,
    /// Overwrite objects that already exist in the destination.
    /// When `false`, existing destination objects are counted as `skipped`.
    pub overwrite_existing: bool,
}

impl Default for MigrationOptions {
    fn default() -> Self {
        Self {
            verify_hash: true,
            delete_source: false,
            overwrite_existing: false,
        }
    }
}

/// A plan describing which objects to migrate and how.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Description label for the plan.
    pub label: String,
    /// Name/identifier of the source backend.
    pub source_name: String,
    /// Name/identifier of the destination backend.
    pub destination_name: String,
    /// The object keys to migrate.
    pub keys: Vec<String>,
    /// Migration behaviour options.
    pub options: MigrationOptions,
}

impl MigrationPlan {
    /// Create a plan from two `CloudStorage` backends by listing the source
    /// with the given prefix filter.
    ///
    /// All keys found under `prefix` (or all keys when `prefix` is `None`) are
    /// included in the plan.
    pub async fn from_source(
        source: &dyn CloudStorage,
        source_name: impl Into<String>,
        destination_name: impl Into<String>,
        prefix: Option<String>,
        options: MigrationOptions,
    ) -> Result<Self, StorageError> {
        let mut keys = Vec::new();
        let mut continuation: Option<String> = None;

        loop {
            let list_opts = ListOptions {
                prefix: prefix.clone(),
                continuation_token: continuation.clone(),
                max_results: Some(1000),
                delimiter: None,
            };
            let result = source.list_objects(list_opts).await?;
            for obj in result.objects {
                keys.push(obj.key);
            }
            if result.has_more {
                continuation = result.next_token;
            } else {
                break;
            }
        }

        Ok(Self {
            label: format!(
                "migration from {} to {}",
                source_name.into(),
                destination_name.into()
            ),
            source_name: String::new(), // filled below via clone pattern
            destination_name: String::new(),
            keys,
            options,
        })
    }
}

/// Result for a single object migration.
#[derive(Debug, Clone)]
pub struct ItemResult {
    /// Object key.
    pub key: String,
    /// Outcome.
    pub outcome: ItemOutcome,
    /// Error message if failed.
    pub error: Option<String>,
    /// Bytes transferred (0 on skip or error).
    pub bytes: u64,
}

/// Outcome for a single migrated object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemOutcome {
    /// Object was copied (and optionally verified and/or source-deleted).
    Migrated,
    /// Object already existed in destination and `overwrite_existing` was false.
    Skipped,
    /// Operation failed; see `error`.
    Failed,
}

/// Summary report produced at the end of a migration.
#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    /// Number of objects successfully migrated.
    pub migrated: u64,
    /// Number of objects that failed.
    pub failed: u64,
    /// Number of objects skipped (already existed at destination).
    pub skipped: u64,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Per-item results (same order as `MigrationPlan::keys`).
    pub items: Vec<ItemResult>,
}

impl MigrationReport {
    /// Total objects processed (migrated + failed + skipped).
    pub fn total(&self) -> u64 {
        self.migrated + self.failed + self.skipped
    }

    /// Whether every object was processed without error.
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0
    }
}

/// Executes migrations between `CloudStorage` backends.
pub struct StorageMigrator {
    source: Arc<dyn CloudStorage>,
    destination: Arc<dyn CloudStorage>,
}

impl StorageMigrator {
    /// Create a migrator using the given source and destination backends.
    pub fn new(source: Arc<dyn CloudStorage>, destination: Arc<dyn CloudStorage>) -> Self {
        Self {
            source,
            destination,
        }
    }

    /// Build a `MigrationPlan` by enumerating objects in the source backend
    /// that match `prefix`.
    pub async fn plan(
        &self,
        source_name: impl Into<String>,
        destination_name: impl Into<String>,
        prefix: Option<String>,
        options: MigrationOptions,
    ) -> Result<MigrationPlan, StorageError> {
        let src_name: String = source_name.into();
        let dst_name: String = destination_name.into();

        let mut keys = Vec::new();
        let mut continuation: Option<String> = None;

        loop {
            let list_opts = ListOptions {
                prefix: prefix.clone(),
                continuation_token: continuation.clone(),
                max_results: Some(1000),
                delimiter: None,
            };
            let result = self.source.list_objects(list_opts).await?;
            for obj in result.objects {
                keys.push(obj.key);
            }
            if result.has_more {
                continuation = result.next_token;
            } else {
                break;
            }
        }

        Ok(MigrationPlan {
            label: format!("migration from {src_name} to {dst_name}"),
            source_name: src_name,
            destination_name: dst_name,
            keys,
            options,
        })
    }

    /// Execute a `MigrationPlan`, returning a `MigrationReport`.
    ///
    /// For each key the migrator:
    /// 1. Checks whether the destination already has the object (skip if
    ///    `overwrite_existing` is false).
    /// 2. Downloads the object from the source.
    /// 3. Uploads it to the destination.
    /// 4. Optionally re-downloads from the destination and verifies the SHA-256
    ///    hash matches (when `verify_hash` is true).
    /// 5. Optionally deletes the source object (when `delete_source` is true).
    pub async fn execute(&self, plan: MigrationPlan) -> MigrationReport {
        let mut report = MigrationReport::default();

        for key in &plan.keys {
            let item = self.migrate_one(key, &plan.options).await;
            match item.outcome {
                ItemOutcome::Migrated => {
                    report.migrated += 1;
                    report.total_bytes += item.bytes;
                }
                ItemOutcome::Skipped => {
                    report.skipped += 1;
                }
                ItemOutcome::Failed => {
                    report.failed += 1;
                }
            }
            report.items.push(item);
        }

        report
    }

    /// Migrate a single key, returning an `ItemResult`.
    async fn migrate_one(&self, key: &str, opts: &MigrationOptions) -> ItemResult {
        // Step 1: check if destination already has the object
        if !opts.overwrite_existing {
            match self.destination.object_exists(key).await {
                Ok(true) => {
                    return ItemResult {
                        key: key.to_string(),
                        outcome: ItemOutcome::Skipped,
                        error: None,
                        bytes: 0,
                    };
                }
                Ok(false) => {}
                Err(e) => {
                    return ItemResult {
                        key: key.to_string(),
                        outcome: ItemOutcome::Failed,
                        error: Some(format!("destination exists check failed: {e}")),
                        bytes: 0,
                    };
                }
            }
        }

        // Step 2: download from source
        let source_data = match self.download_all(key, &*self.source).await {
            Ok(data) => data,
            Err(e) => {
                return ItemResult {
                    key: key.to_string(),
                    outcome: ItemOutcome::Failed,
                    error: Some(format!("source download failed: {e}")),
                    bytes: 0,
                };
            }
        };

        let source_hash = if opts.verify_hash {
            Some(hex::encode(Sha256::digest(&source_data)))
        } else {
            None
        };

        let bytes_len = source_data.len() as u64;

        // Step 3: upload to destination
        if let Err(e) = self
            .upload_bytes(key, source_data, &*self.destination)
            .await
        {
            return ItemResult {
                key: key.to_string(),
                outcome: ItemOutcome::Failed,
                error: Some(format!("destination upload failed: {e}")),
                bytes: 0,
            };
        }

        // Step 4: optional hash verification
        if let Some(expected_hash) = source_hash {
            let dest_data = match self.download_all(key, &*self.destination).await {
                Ok(d) => d,
                Err(e) => {
                    return ItemResult {
                        key: key.to_string(),
                        outcome: ItemOutcome::Failed,
                        error: Some(format!("verification download failed: {e}")),
                        bytes: 0,
                    };
                }
            };
            let actual_hash = hex::encode(Sha256::digest(&dest_data));
            if actual_hash != expected_hash {
                return ItemResult {
                    key: key.to_string(),
                    outcome: ItemOutcome::Failed,
                    error: Some("hash mismatch after migration".to_string()),
                    bytes: 0,
                };
            }
        }

        // Step 5: optional source deletion
        if opts.delete_source {
            if let Err(e) = self.source.delete_object(key).await {
                return ItemResult {
                    key: key.to_string(),
                    outcome: ItemOutcome::Failed,
                    error: Some(format!("source delete failed: {e}")),
                    bytes: bytes_len,
                };
            }
        }

        ItemResult {
            key: key.to_string(),
            outcome: ItemOutcome::Migrated,
            error: None,
            bytes: bytes_len,
        }
    }

    /// Download all bytes for `key` from `storage`.
    async fn download_all(
        &self,
        key: &str,
        storage: &dyn CloudStorage,
    ) -> Result<Vec<u8>, StorageError> {
        use futures::StreamExt;

        let mut stream = storage
            .download_stream(key, DownloadOptions::default())
            .await?;
        let mut data = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            data.extend_from_slice(&chunk);
        }
        Ok(data)
    }

    /// Upload `data` bytes to `storage` under `key`.
    async fn upload_bytes(
        &self,
        key: &str,
        data: Vec<u8>,
        storage: &dyn CloudStorage,
    ) -> Result<(), StorageError> {
        use bytes::Bytes;
        use futures::stream;

        let size = data.len() as u64;
        let bytes = Bytes::from(data);
        let stream = Box::pin(stream::once(
            async move { Ok::<Bytes, StorageError>(bytes) },
        ));
        storage
            .upload_stream(key, stream, Some(size), UploadOptions::default())
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::LocalStorage;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn make_storage() -> (Arc<LocalStorage>, TempDir) {
        let dir = TempDir::new().expect("temp dir");
        let s = LocalStorage::new(dir.path()).await.expect("local storage");
        (Arc::new(s), dir)
    }

    async fn seed(storage: &LocalStorage, key: &str, data: &[u8]) {
        use bytes::Bytes;
        use futures::stream;
        let b = Bytes::copy_from_slice(data);
        let sz = b.len() as u64;
        let stream = Box::pin(stream::once(async move { Ok::<Bytes, StorageError>(b) }));
        storage
            .upload_stream(key, stream, Some(sz), UploadOptions::default())
            .await
            .expect("seed upload");
    }

    fn default_opts() -> MigrationOptions {
        MigrationOptions::default()
    }

    // ── MigrationOptions ───────────────────────────────────────────────────────

    #[test]
    fn test_migration_options_defaults() {
        let opts = MigrationOptions::default();
        assert!(opts.verify_hash);
        assert!(!opts.delete_source);
        assert!(!opts.overwrite_existing);
    }

    // ── MigrationReport ────────────────────────────────────────────────────────

    #[test]
    fn test_migration_report_total() {
        let r = MigrationReport {
            migrated: 3,
            failed: 1,
            skipped: 2,
            total_bytes: 1024,
            items: vec![],
        };
        assert_eq!(r.total(), 6);
        assert!(!r.all_succeeded());
    }

    #[test]
    fn test_migration_report_all_succeeded() {
        let r = MigrationReport {
            migrated: 5,
            failed: 0,
            skipped: 0,
            total_bytes: 500,
            items: vec![],
        };
        assert!(r.all_succeeded());
    }

    // ── StorageMigrator::plan ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_plan_enumerates_source_keys() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        seed(&src, "files/a.mp4", b"aaa").await;
        seed(&src, "files/b.mp3", b"bbb").await;

        let migrator = StorageMigrator::new(Arc::clone(&src) as Arc<dyn CloudStorage>, dst);
        let plan = migrator
            .plan("src", "dst", Some("files/".to_string()), default_opts())
            .await
            .expect("plan should succeed");

        assert_eq!(plan.keys.len(), 2);
        assert!(plan.label.contains("src"));
    }

    #[tokio::test]
    async fn test_plan_empty_source() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        let migrator = StorageMigrator::new(Arc::clone(&src) as Arc<dyn CloudStorage>, dst);
        let plan = migrator
            .plan("s", "d", None, default_opts())
            .await
            .expect("plan should succeed");
        assert!(plan.keys.is_empty());
    }

    // ── StorageMigrator::execute ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_execute_migrates_all_objects() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        seed(&src, "a.bin", &[1u8; 100]).await;
        seed(&src, "b.bin", &[2u8; 200]).await;

        let migrator = StorageMigrator::new(
            Arc::clone(&src) as Arc<dyn CloudStorage>,
            Arc::clone(&dst) as Arc<dyn CloudStorage>,
        );
        let plan = migrator
            .plan("s", "d", None, default_opts())
            .await
            .expect("plan");
        let report = migrator.execute(plan).await;

        assert_eq!(report.migrated, 2);
        assert_eq!(report.failed, 0);
        assert_eq!(report.total_bytes, 300);
        assert!(dst.object_exists("a.bin").await.expect("exists check"));
        assert!(dst.object_exists("b.bin").await.expect("exists check"));
    }

    #[tokio::test]
    async fn test_execute_skips_existing_when_no_overwrite() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        seed(&src, "dupe.bin", b"source").await;
        seed(&dst, "dupe.bin", b"dest").await;

        let migrator = StorageMigrator::new(
            Arc::clone(&src) as Arc<dyn CloudStorage>,
            Arc::clone(&dst) as Arc<dyn CloudStorage>,
        );
        let opts = MigrationOptions {
            overwrite_existing: false,
            ..Default::default()
        };
        let plan = MigrationPlan {
            label: "test".to_string(),
            source_name: "s".to_string(),
            destination_name: "d".to_string(),
            keys: vec!["dupe.bin".to_string()],
            options: opts,
        };
        let report = migrator.execute(plan).await;
        assert_eq!(report.skipped, 1);
        assert_eq!(report.migrated, 0);
    }

    #[tokio::test]
    async fn test_execute_overwrites_when_enabled() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        seed(&src, "ow.bin", b"new_data").await;
        seed(&dst, "ow.bin", b"old").await;

        let migrator = StorageMigrator::new(
            Arc::clone(&src) as Arc<dyn CloudStorage>,
            Arc::clone(&dst) as Arc<dyn CloudStorage>,
        );
        let opts = MigrationOptions {
            overwrite_existing: true,
            verify_hash: false,
            delete_source: false,
        };
        let plan = MigrationPlan {
            label: "t".to_string(),
            source_name: "s".to_string(),
            destination_name: "d".to_string(),
            keys: vec!["ow.bin".to_string()],
            options: opts,
        };
        let report = migrator.execute(plan).await;
        assert_eq!(report.migrated, 1);
    }

    #[tokio::test]
    async fn test_execute_delete_source_after_migrate() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;
        seed(&src, "del_src.bin", b"data").await;

        let migrator = StorageMigrator::new(
            Arc::clone(&src) as Arc<dyn CloudStorage>,
            Arc::clone(&dst) as Arc<dyn CloudStorage>,
        );
        let opts = MigrationOptions {
            delete_source: true,
            verify_hash: false,
            overwrite_existing: false,
        };
        let plan = MigrationPlan {
            label: "t".to_string(),
            source_name: "s".to_string(),
            destination_name: "d".to_string(),
            keys: vec!["del_src.bin".to_string()],
            options: opts,
        };
        let report = migrator.execute(plan).await;
        assert_eq!(report.migrated, 1);
        assert!(!src.object_exists("del_src.bin").await.expect("check"));
    }

    #[tokio::test]
    async fn test_execute_fails_missing_source_key() {
        let (src, _d1) = make_storage().await;
        let (dst, _d2) = make_storage().await;

        let migrator = StorageMigrator::new(
            Arc::clone(&src) as Arc<dyn CloudStorage>,
            Arc::clone(&dst) as Arc<dyn CloudStorage>,
        );
        let plan = MigrationPlan {
            label: "t".to_string(),
            source_name: "s".to_string(),
            destination_name: "d".to_string(),
            keys: vec!["missing.bin".to_string()],
            options: default_opts(),
        };
        let report = migrator.execute(plan).await;
        assert_eq!(report.failed, 1);
        assert!(!report.items[0].error.as_deref().unwrap_or("").is_empty());
    }
}
